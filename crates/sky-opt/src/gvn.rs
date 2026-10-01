//! Global value numbering: common-subexpression elimination across blocks
//! and regions.
//!
//! The e-graph pass merges equal expressions inside one straight-line
//! segment. This pass merges them wherever the earlier one **dominates** the
//! later one. SIR is structured, so dominance is scoping: an operation
//! dominates the operations after it in its block, and everything inside the
//! regions of those operations (the arms of an `if`, both parts of a
//! `while`). A value computed inside an `if` arm is not visible after the
//! `if`, so it does not dominate anything there; a value computed in a
//! `while`'s `before` region does not reach its `after` region either (the
//! verifier's scoping: values cross between them only as region arguments).
//! The pass walks the region tree with a table of the operations in scope
//! and drops it when a region ends, so a reused value is always visible.
//!
//! # What is numbered
//!
//! An operation is numbered when it has results and no regions, is not a
//! terminator, has no `io` effect, and is not `input` or `literal.error`
//! (which fail every time). A `call` counts when the callee exists and does
//! not print. Two operations are the same when they have the same kind,
//! attributes, operands and result types. The operands of a commutative
//! operation (`add`, `mul`, `dot`, `min`, `max`, `hypot`, ...) are ordered
//! before comparing, so `x * y` and `y * x` are one value. `if`, `while` and
//! `fused` are not numbered (they have regions), and the bodies of `fused`
//! kernels are not entered: they are scalar code with their own arguments,
//! opaque to the rewriting passes.
//!
//! # Why an operation that can fail may be reused
//!
//! Sky reports the first failure in program order. Take an operation `B`
//! identical to an earlier one `A` that dominates it, and suppose both can
//! fail (a division, a `sqrt`, an `inv`, a call).
//!
//! * `A` runs before `B` on every path that reaches `B`, with the same
//!   operand *values* (they are the same SSA values: within one activation
//!   of the enclosing region a value has one definition, and both operations
//!   are evaluated in that same activation).
//! * Operations are deterministic in their operands, so `A` and `B` either
//!   both fail, with the same message, or both succeed with the same result.
//! * If they fail, `A` fails first and the run stops there: `B` never runs,
//!   so `B`'s span would never have been reported. The error is `A`'s, at
//!   `A`'s span and with `A`'s operand labels, exactly as before the pass.
//! * If they succeed, the result of `B` is the result of `A`.
//!
//! So dropping `B` changes neither the output nor the first failure. This is
//! what forbids numbering operations with `io`: a `print` twice prints
//! twice. A call to a function that prints is the same. (The call-depth
//! limit cannot differ either: both calls are made at the same depth.)
//!
//! Nothing is moved, only removed, so the order of everything that remains
//! is untouched. Each removal is explained (`GVN-001`), except constants and
//! operations that repeat code from the same source span (a loop unrolled by
//! `specialize` has many copies of one expression, and saying so would be
//! noise).

use std::collections::HashMap;
use std::mem::Discriminant;

use sky_ir::{Effect, Function, Module, OpId, OpKind, RegionId, ValueId};

use crate::rewrite::{Related, Rewrite};
use crate::util::{op_text, value_text};

pub const GVN_RULE: &str = "GVN-001";

/// Numbers the values of every function, and reuses the earlier of two equal
/// operations. See the module documentation.
pub fn gvn(m: &mut Module) -> Vec<Related> {
    // Which callees may be numbered: they exist and do not print.
    let quiet: HashMap<String, bool> = m
        .functions
        .iter()
        .map(|f| (f.name.clone(), !f.effects.contains(Effect::Io)))
        .collect();
    let mut out = Vec::new();
    for f in &mut m.functions {
        let mut g = Numbering {
            f,
            quiet: &quiet,
            table: HashMap::new(),
            log: Vec::new(),
            reuse: Vec::new(),
            out: &mut out,
        };
        g.reuse = (0..g.f.values.len() as u32).map(ValueId).collect();
        let body = g.f.body;
        g.region(body);
    }
    out
}

/// Where operations with these operands live: the kind's discriminant and
/// the operands, in a canonical order for commutative kinds. The bucket is
/// checked for full equality (attributes, result types).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    kind: Discriminant<OpKind>,
    operands: Vec<ValueId>,
}

struct Numbering<'a> {
    f: &'a mut Function,
    quiet: &'a HashMap<String, bool>,
    /// The operations in scope, by key.
    table: HashMap<Key, Vec<OpId>>,
    /// The keys inserted, in order, so a scope's entries can be dropped.
    log: Vec<Key>,
    /// `reuse[v]`: the value that stands for `v` now (itself, or the earlier
    /// value it was merged into).
    reuse: Vec<ValueId>,
    out: &'a mut Vec<Related>,
}

impl Numbering<'_> {
    fn region(&mut self, r: RegionId) {
        for b in self.f.region(r).blocks.clone() {
            let mark = self.log.len();
            let mut kept = Vec::new();
            for id in self.f.block(b).ops.clone() {
                if self.visit(id) {
                    kept.push(id);
                }
            }
            self.f.blocks[b.index()].ops = kept;
            // The block ended: what it computed is out of scope.
            while self.log.len() > mark {
                let key = self.log.pop().expect("log is longer than mark");
                if let Some(bucket) = self.table.get_mut(&key) {
                    bucket.pop();
                }
            }
        }
    }

    /// Numbers `id`; false if it was merged into an earlier operation and
    /// must be dropped.
    fn visit(&mut self, id: OpId) -> bool {
        for i in 0..self.f.op(id).operands.len() {
            let v = self.f.op(id).operands[i];
            self.f.ops[id.index()].operands[i] = self.reuse[v.index()];
        }
        if !matches!(self.f.op(id).kind, OpKind::Fused) {
            for r in self.f.op(id).regions.clone() {
                self.region(r);
            }
        }
        if !self.numbered(id) {
            return true;
        }
        let key = self.key(id);
        let earlier = self
            .table
            .get(&key)
            .and_then(|bucket| bucket.iter().rev().find(|&&e| self.same(e, id)))
            .copied();
        match earlier {
            Some(e) => {
                self.merge(e, id);
                false
            }
            None => {
                self.table.entry(key.clone()).or_default().push(id);
                self.log.push(key);
                true
            }
        }
    }

    fn numbered(&self, id: OpId) -> bool {
        let op = self.f.op(id);
        let props = op.kind.properties();
        if op.results.is_empty()
            || !op.regions.is_empty()
            || props.terminator
            || props.effects.contains(Effect::Io)
        {
            return false;
        }
        match &op.kind {
            // These fail every time they run: nothing to reuse.
            OpKind::Input { .. } | OpKind::LiteralError { .. } => false,
            // Comparing two big constant tables costs a pass over them, and
            // every one shares a bucket: not worth it for one-off data.
            OpKind::ConstArray { .. } => false,
            OpKind::Call { callee } => self.quiet.get(callee).copied().unwrap_or(false),
            _ => true,
        }
    }

    fn key(&self, id: OpId) -> Key {
        let op = self.f.op(id);
        Key {
            kind: std::mem::discriminant(&op.kind),
            operands: canonical_operands(op),
        }
    }

    /// Whether `a` and `b` are the same operation on the same values.
    fn same(&self, a: OpId, b: OpId) -> bool {
        let (x, y) = (self.f.op(a), self.f.op(b));
        x.kind == y.kind
            && canonical_operands(x) == canonical_operands(y)
            && x.results.len() == y.results.len()
            && x.results
                .iter()
                .zip(&y.results)
                .all(|(p, q)| self.f.ty(*p) == self.f.ty(*q))
    }

    /// `later` is `earlier` again: its results become the earlier ones.
    fn merge(&mut self, earlier: OpId, later: OpId) {
        let (e, l) = (self.f.op(earlier), self.f.op(later));
        if !matches!(
            l.kind,
            OpKind::ConstFloat { .. }
                | OpKind::ConstRational(_)
                | OpKind::ConstBool(_)
                | OpKind::ConstStr(_)
                | OpKind::ConstNothing
        ) && e.span != l.span
        {
            let fallible = !l.kind.properties().effects.is_pure();
            let mut because = vec![format!(
                "the earlier `{}` has the same operands and runs first on every path to this \
                 one, so both compute the same value",
                l.kind.mnemonic()
            )];
            if fallible {
                because.push(format!(
                    "`{}` can fail, but with the same operands the earlier one has failed \
                     already if this one would, and Sky reports the first failure, so this one \
                     can never be the failure that is reported",
                    l.kind.mnemonic()
                ));
            }
            let after = e
                .results
                .iter()
                .map(|v| value_text(self.f, *v))
                .collect::<Vec<_>>()
                .join(", ");
            let related = e.span;
            self.out.push(Related {
                rewrite: Rewrite {
                    pass: "gvn",
                    rule: GVN_RULE,
                    span: l.span,
                    before: op_text(self.f, later),
                    after,
                    because,
                },
                related,
            });
        }
        let pairs: Vec<(ValueId, ValueId)> = self
            .f
            .op(later)
            .results
            .iter()
            .copied()
            .zip(self.f.op(earlier).results.iter().copied())
            .collect();
        for (from, to) in pairs {
            self.reuse[from.index()] = to;
        }
    }
}

/// The operands, ordered when the operation is commutative.
fn canonical_operands(op: &sky_ir::OpData) -> Vec<ValueId> {
    let mut v = op.operands.clone();
    let commutative = op.kind.properties().commutative
        || matches!(op.kind, OpKind::Cmp(sky_ir::CmpOp::Eq | sky_ir::CmpOp::Ne));
    if commutative && v.len() == 2 {
        v.sort();
    }
    v
}
