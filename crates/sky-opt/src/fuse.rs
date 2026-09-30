//! Kernel fusion: a tree of elementwise array operations becomes one
//! [`OpKind::Fused`] operation that computes each element once, so the
//! intermediate arrays are never built (roadmap ideas 18 and 27 to 30).
//! `docs/design/ir.md` ("Fusion") specifies the operation; this pass makes
//! it. It represents the loop and the composition, not machine code.
//!
//! # What is fused
//!
//! An operation is a *candidate* when it is `add`, `sub`, `mul`, `div`,
//! `neg` or a math function (`sqrt`, `exp`, `atan2`, ...) on arrays:
//! numbers of known type, a result of rank 1 or more, and operands laid out
//! as the checker allows (`add`/`sub`: two arrays of the result's shape;
//! `mul`: two such arrays, or one and a scalar; `div`: two such arrays, or
//! one over a scalar; `neg`: one array; a math function: as many operands as
//! it takes, each an array of the result's shape or a scalar). A candidate
//! `P` is *absorbed*
//! into its consumer `C` when
//!
//! * `P`'s result has exactly one use (so nothing else needs the
//!   temporary), which is an operand of `C`;
//! * `C` is a candidate in the same block (then `C`'s array operands all
//!   have the shape of `P`'s result).
//!
//! A *tree* is a candidate (the root) with everything absorbed into it,
//! recursively. A scalar operand is a leaf whatever computes it, and is
//! broadcast. A tree of two or more operations becomes one `fused`
//! operation at the root's position; the operations move into its body with
//! their own spans, and the `fused` keeps the root's result value (and so
//! its name and users) and the join of the spans.
//!
//! # Failure order
//!
//! The unfused chain runs one operation over all the elements, then the next,
//! so it reports the *first operation in program order* that fails at any
//! element. `fused` is defined to report that same error (see
//! [`OpKind::Fused`]): its body runs per element, and a failure only decides
//! the error if no earlier body operation fails at any element. So a tree
//! with several failing operations is fused too, and nothing about the
//! error changes. Nothing is fused where that would not hold outside the
//! tree:
//!
//! * the operations move to the root's position, so an operation that can
//!   fail is never moved past another operation that can fail, print, call,
//!   loop or touch an `Unknown` value (which can fail dynamically). If the
//!   whole tree would move past one, the tree is not fused as a whole; its
//!   subtrees are tried on their own.
//!
//! # Explanations
//!
//! One `FUSE-001` per kernel: the expression before, the kernel after, the
//! number of operations and of temporary arrays saved, the shape, which
//! operations can fail, and that the failure is unchanged. The pass runs
//! after `dce` (a dead operation would count as a use) and before
//! `cholesky`, and is idempotent: `fused` operations are not candidates, and
//! their bodies are not visited.

use std::collections::{HashMap, HashSet};

use sky_diag::Span;
use sky_ir::{
    BlockData, BlockId, Effect, EffectSet, Function, Module, NumberType, OpData, OpId, OpKind,
    RegionData, RegionId, Type, ValueDef, ValueId,
};
use sky_types::{Shape, ShapeDim, TagSet};

use crate::rewrite::Rewrite;
use crate::util::{defining_op, use_counts};

/// The rule id of a fusion.
pub const FUSE_RULE: &str = "FUSE-001";

/// Fuses elementwise trees in every function.
pub fn fuse(m: &mut Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        fuse_function(f, &mut out);
    }
    out
}

fn fuse_function(f: &mut Function, out: &mut Vec<Rewrite>) {
    // Fusing a subtree of a blocked tree can unblock the rest of it, so plan
    // again until nothing is left. Every round turns two or more candidates
    // into one operation that is not a candidate, so this ends.
    loop {
        let uses = use_counts(f);
        let mut plans: Vec<Vec<OpId>> = Vec::new();
        for block in blocks_outside_bodies(f) {
            let first = plans.len();
            plan_block(f, block, &uses, &mut plans);
            // Trees in program order (by root), whatever order they were found.
            let ops = &f.block(block).ops;
            let at = |tree: &Vec<OpId>| ops.iter().position(|o| Some(o) == tree.last());
            plans[first..].sort_by_key(at);
        }
        if plans.is_empty() {
            return;
        }
        for tree in plans {
            out.push(apply(f, &tree));
        }
    }
}

/// The blocks of the function in pre-order, except the bodies of `fused`
/// operations.
fn blocks_outside_bodies(f: &Function) -> Vec<BlockId> {
    let mut out = Vec::new();
    let mut todo: Vec<RegionId> = vec![f.body];
    while let Some(r) = todo.pop() {
        let mut nested = Vec::new();
        for &b in &f.region(r).blocks {
            out.push(b);
            for &o in &f.block(b).ops {
                let op = f.op(o);
                if op.kind != OpKind::Fused {
                    nested.extend(op.regions.iter().copied());
                }
            }
        }
        // Pre-order: the first nested region is visited next.
        todo.extend(nested.into_iter().rev());
    }
    out
}

// ----- candidates -----

fn number(f: &Function, v: ValueId) -> Option<&NumberType> {
    f.value(v).ty.as_number()
}

/// `id` is an elementwise array operation this pass can fuse.
fn is_candidate(f: &Function, id: OpId) -> bool {
    let op = f.op(id);
    if !op.regions.is_empty() {
        return false;
    }
    let [result] = op.results[..] else {
        return false;
    };
    let Some(r) = number(f, result) else {
        return false;
    };
    if r.shape.rank() == 0 {
        return false;
    }
    let mut kinds = Vec::with_capacity(2);
    for &v in &op.operands {
        let Some(n) = number(f, v) else { return false };
        // Every operand is a scalar or an array of the result's shape.
        if n.shape.rank() == 0 {
            kinds.push(false);
        } else if n.shape == r.shape {
            kinds.push(true);
        } else {
            return false;
        }
    }
    match (&op.kind, &kinds[..]) {
        (OpKind::Add | OpKind::Sub, [true, true])
        | (OpKind::Mul, [true, true] | [true, false] | [false, true])
        | (OpKind::Div, [true, true] | [true, false])
        | (OpKind::Neg, [true]) => true,
        // A function of arrays and numbers: every operand is present and,
        // the result being an array, at least one is an array.
        (OpKind::Math(f), kinds) => kinds.len() == f.arity(),
        _ => false,
    }
}

/// The effects of an operation, counting what is inside its regions. A
/// `while` may not end and a call may do anything, so they count as every
/// effect; an operation touching an `Unknown` value can fail dynamically
/// even when its kind is pure.
fn effects_of(f: &Function, id: OpId) -> EffectSet {
    let mut all = EffectSet::PURE;
    let mut todo = vec![id];
    while let Some(o) = todo.pop() {
        let op = f.op(o);
        let mut e = op.kind.properties().effects;
        if matches!(op.kind, OpKind::While) {
            e = e.with(Effect::Fail).with(Effect::Io);
        }
        let unknown = op
            .operands
            .iter()
            .chain(&op.results)
            .any(|v| *f.ty(*v) == Type::Unknown);
        if unknown {
            e = e.with(Effect::Fail);
        }
        all = all.union(e);
        for &r in &op.regions {
            for &b in &f.region(r).blocks {
                todo.extend(f.block(b).ops.iter().copied());
            }
        }
    }
    all
}

/// Finds the trees of one block.
fn plan_block(f: &Function, block: BlockId, uses: &[u32], plans: &mut Vec<Vec<OpId>>) {
    let ops = &f.block(block).ops;
    let pos: HashMap<OpId, usize> = ops.iter().enumerate().map(|(i, o)| (*o, i)).collect();
    // The operation in this block that uses each value used exactly once.
    let mut user: HashMap<ValueId, OpId> = HashMap::new();
    for &o in ops {
        for &v in &f.op(o).operands {
            if uses[v.index()] == 1 {
                user.insert(v, o);
            }
        }
    }
    let candidate: HashSet<OpId> = ops
        .iter()
        .copied()
        .filter(|o| is_candidate(f, *o))
        .collect();
    // The producer of `v` that is absorbed into `consumer`, if any.
    let absorbed = |v: ValueId, consumer: OpId| -> Option<OpId> {
        let p = defining_op(f, v)?;
        (candidate.contains(&p) && user.get(&v) == Some(&consumer)).then_some(p)
    };
    // Roots: candidates nothing absorbs.
    let mut roots: Vec<OpId> = ops
        .iter()
        .copied()
        .filter(|&o| {
            candidate.contains(&o) && {
                let v = f.op(o).results[0];
                !user.get(&v).is_some_and(|c| candidate.contains(c))
            }
        })
        .collect();
    // Effectful positions, as a prefix count.
    let mut impure_before = vec![0usize; ops.len() + 1];
    for (i, &o) in ops.iter().enumerate() {
        impure_before[i + 1] = impure_before[i] + usize::from(!effects_of(f, o).is_pure());
    }
    let is_failing = |o: OpId| f.op(o).kind.properties().effects.contains(Effect::Fail);

    while let Some(root) = roots.pop() {
        // The tree, by a walk from the root.
        let mut tree = vec![root];
        let mut i = 0;
        while i < tree.len() {
            let o = tree[i];
            i += 1;
            for &v in &f.op(o).operands {
                if let Some(p) = absorbed(v, o) {
                    tree.push(p);
                }
            }
        }
        if tree.len() < 2 {
            continue;
        }
        tree.sort_by_key(|o| pos[o]);
        // Moving the failing operations down to the root must not cross
        // another effectful operation.
        let failing: Vec<usize> = tree
            .iter()
            .filter(|o| **o != root && is_failing(**o))
            .map(|o| pos[o])
            .collect();
        let blocked = failing.first().is_some_and(|&first| {
            let last = pos[&root];
            let between = impure_before[last] - impure_before[first + 1];
            // The tree's own effectful operations strictly between.
            let own = failing.len() - 1;
            between > own
        });
        if blocked {
            // Try each absorbed child on its own.
            for &v in &f.op(root).operands {
                if let Some(p) = absorbed(v, root) {
                    roots.push(p);
                }
            }
            continue;
        }
        plans.push(tree);
    }
}

// ----- rewriting -----

/// The scalar type of one element of `t`: its dimension and representation,
/// no shape and no tags.
fn element_type(t: &Type) -> Type {
    match t {
        Type::Number(n) => Type::Number(NumberType {
            dim: n.dim,
            shape: Shape::scalar(),
            tags: TagSet::EMPTY,
            repr: n.repr,
        }),
        other => other.clone(),
    }
}

/// Replaces the tree (in block order, root last) with one `fused` operation
/// and returns its explanation.
fn apply(f: &mut Function, tree: &[OpId]) -> Rewrite {
    let root = *tree.last().expect("a tree has a root");
    let in_tree: HashSet<OpId> = tree.iter().copied().collect();
    let block = f.op(root).parent;
    let root_value = f.op(root).results[0];
    let root_type = f.ty(root_value).clone();

    // Leaves: the operands the tree does not compute, once each, in order.
    let mut leaves: Vec<ValueId> = Vec::new();
    // Where each leaf is first read, in the source: the span the operation
    // that reads it recorded (its own, if it recorded none). The body's
    // operations keep their own operand spans (their operands stay in place,
    // only the values change), which is what their errors point at.
    let mut leaf_spans: Vec<Span> = Vec::new();
    let mut seen: HashSet<ValueId> = HashSet::new();
    for &o in tree {
        for (i, &v) in f.op(o).operands.iter().enumerate() {
            let inside = defining_op(f, v).is_some_and(|p| in_tree.contains(&p));
            if !inside && seen.insert(v) {
                leaves.push(v);
                leaf_spans.push(f.op(o).operand_span(i));
            }
        }
    }
    if tree.iter().all(|&o| f.op(o).operand_spans.is_empty()) {
        leaf_spans.clear();
    }
    let rewrite = explain(f, tree, &leaves);
    let span = rewrite.span;

    // The new operation, region and block.
    let fused = OpId(f.ops.len() as u32);
    let region = RegionId(f.regions.len() as u32);
    let body = BlockId(f.blocks.len() as u32);
    f.regions.push(RegionData {
        blocks: vec![body],
        parent: Some(fused),
    });
    f.blocks.push(BlockData {
        args: Vec::new(),
        ops: Vec::new(),
        parent: Some(region),
    });
    f.ops.push(OpData {
        kind: OpKind::Fused,
        operands: leaves.clone(),
        results: vec![root_value],
        regions: vec![region],
        span,
        operand_spans: leaf_spans,
        parent: block,
    });
    f.values[root_value.index()].def = ValueDef::OpResult {
        op: fused,
        index: 0,
    };

    // One block argument per leaf.
    let mut arg_of: HashMap<ValueId, ValueId> = HashMap::new();
    for (i, &leaf) in leaves.iter().enumerate() {
        let ty = element_type(f.ty(leaf));
        let arg = ValueId(f.values.len() as u32);
        f.values.push(sky_ir::ValueData {
            ty,
            def: ValueDef::BlockArg {
                block: body,
                index: i as u32,
            },
            name: None,
        });
        f.blocks[body.index()].args.push(arg);
        arg_of.insert(leaf, arg);
    }

    // The operations move into the body, on scalars.
    for &o in tree {
        let operands: Vec<ValueId> = f
            .op(o)
            .operands
            .iter()
            .map(|v| arg_of.get(v).copied().unwrap_or(*v))
            .collect();
        f.ops[o.index()].operands = operands;
        f.ops[o.index()].parent = body;
        // The root's result is the kernel's, and stays an array.
        if o != root {
            let r = f.op(o).results[0];
            f.values[r.index()].ty = element_type(&f.values[r.index()].ty);
        }
    }
    // The root's result is the kernel's, so its body value is a new one.
    let element = ValueId(f.values.len() as u32);
    f.values.push(sky_ir::ValueData {
        ty: element_type(&root_type),
        def: ValueDef::OpResult { op: root, index: 0 },
        name: None,
    });
    f.ops[root.index()].results = vec![element];
    let yield_op = OpId(f.ops.len() as u32);
    f.ops.push(OpData {
        kind: OpKind::Yield,
        operands: vec![element],
        results: Vec::new(),
        regions: Vec::new(),
        span,
        operand_spans: Vec::new(),
        parent: body,
    });
    let mut body_ops = tree.to_vec();
    body_ops.push(yield_op);
    f.blocks[body.index()].ops = body_ops;

    // In the block, the kernel takes the root's place.
    let ops = std::mem::take(&mut f.blocks[block.index()].ops);
    f.blocks[block.index()].ops = ops
        .into_iter()
        .filter_map(|o| {
            if o == root {
                Some(fused)
            } else {
                (!in_tree.contains(&o)).then_some(o)
            }
        })
        .collect();
    rewrite
}

// ----- explanation -----

fn value_name(f: &Function, v: ValueId) -> String {
    if let Some(n) = &f.value(v).name {
        return n.clone();
    }
    match defining_op(f, v).map(|o| &f.op(o).kind) {
        Some(OpKind::ConstFloat { bits }) => f64::from_bits(*bits).to_string(),
        Some(OpKind::ConstRational(q)) => q.to_string(),
        _ => v.to_string(),
    }
}

/// The tree as an expression: `(a + b) * 2 - d`.
struct Expr<'a> {
    f: &'a Function,
    tree: &'a HashSet<OpId>,
}

/// How tight an expression binds: a sum, a product, a negation, an atom.
const SUM: u8 = 1;
const PRODUCT: u8 = 2;
const NEGATION: u8 = 3;
const ATOM: u8 = 4;

/// The deepest expression written out; deeper subtrees read `…`.
const MAX_TEXT_DEPTH: usize = 24;

impl Expr<'_> {
    fn text(&self, v: ValueId, depth: usize) -> (String, u8) {
        let Some(id) = defining_op(self.f, v).filter(|o| self.tree.contains(o)) else {
            return (value_name(self.f, v), ATOM);
        };
        if depth >= MAX_TEXT_DEPTH {
            return ("…".to_string(), ATOM);
        }
        let op = self.f.op(id);
        let sym = match op.kind {
            OpKind::Add => "+",
            OpKind::Sub => "-",
            OpKind::Mul => "*",
            OpKind::Div => "/",
            OpKind::Math(f) => {
                let args: Vec<String> = op
                    .operands
                    .iter()
                    .map(|&a| self.text(a, depth + 1).0)
                    .collect();
                return (format!("{}({})", f.name(), args.join(", ")), ATOM);
            }
            _ => {
                let (x, p) = self.text(op.operands[0], depth + 1);
                let x = if p <= NEGATION { format!("({x})") } else { x };
                return (format!("-{x}"), NEGATION);
            }
        };
        let prec = if matches!(op.kind, OpKind::Add | OpKind::Sub) {
            SUM
        } else {
            PRODUCT
        };
        let (l, lp) = self.text(op.operands[0], depth + 1);
        let (r, rp) = self.text(op.operands[1], depth + 1);
        let l = if lp < prec { format!("({l})") } else { l };
        let r = if rp <= prec { format!("({r})") } else { r };
        (format!("{l} {sym} {r}"), prec)
    }
}

fn elements(shape: &Shape) -> Option<u64> {
    shape.dims().iter().try_fold(1u64, |n, d| match d {
        ShapeDim::Known(k) => n.checked_mul(*k),
        ShapeDim::Var(_) => None,
    })
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn join_names(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [a] => a.clone(),
        [a, b] => format!("{a} and {b}"),
        [init @ .., last] => format!("{}, and {last}", init.join(", ")),
    }
}

fn explain(f: &Function, tree: &[OpId], leaves: &[ValueId]) -> Rewrite {
    let root = *tree.last().expect("a tree has a root");
    let in_tree: HashSet<OpId> = tree.iter().copied().collect();
    let expr = Expr { f, tree: &in_tree };
    let (before, _) = expr.text(f.op(root).results[0], 0);
    let names: Vec<String> = leaves.iter().map(|v| value_name(f, *v)).collect();
    let after = format!("fused({})", names.join(", "));

    let n = tree.len();
    let saved = n - 1;
    let shape = number(f, f.op(root).results[0])
        .map(|t| t.shape.clone())
        .unwrap_or_default();
    let size = match elements(&shape) {
        Some(k) => format!("of {k} elements"),
        None => format!("of shape {shape}"),
    };
    let mut because = vec![
        format!(
            "fused {} into one kernel; saves {} {size}",
            plural(n, "elementwise operation", "elementwise operations"),
            plural(saved, "temporary array", "temporary arrays"),
        ),
        format!(
            "every intermediate result is used exactly once, by the next operation, and every \
             array operand is a {shape}"
        ),
    ];
    let temporaries: Vec<String> = tree[..n - 1]
        .iter()
        .filter_map(|o| f.value(f.op(*o).results[0]).name.as_ref())
        .map(|name| format!("`{name}`"))
        .collect();
    if !temporaries.is_empty() {
        because.push(format!(
            "{} {} only used here, so {} no longer materialized",
            join_names(&temporaries),
            if temporaries.len() == 1 { "is" } else { "are" },
            if temporaries.len() == 1 {
                "it is"
            } else {
                "they are"
            },
        ));
    }
    let failing: Vec<&'static str> = tree
        .iter()
        .filter(|o| f.op(**o).kind.properties().effects.contains(Effect::Fail))
        .map(|o| f.op(*o).kind.mnemonic())
        .collect();
    let mut kinds: Vec<String> = Vec::new();
    for k in &failing {
        let k = format!("`{k}`");
        if !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    because.push(match failing.len() {
        0 => "no operation in the kernel can fail".to_string(),
        1 => format!(
            "only {} can fail, so a failure has the same message and place as before",
            kinds[0]
        ),
        k => format!(
            "{k} of the operations can fail ({}); the kernel reports the failure of the first \
             of them in program order that fails at any element, which is the error the \
             separate operations gave",
            kinds.join(", ")
        ),
    });
    let span = tree
        .iter()
        .map(|o| f.op(*o).span)
        .filter(|s| s.start != 0 || s.end != 0)
        .reduce(Span::join)
        .unwrap_or_else(|| f.op(root).span);
    Rewrite {
        pass: "fuse",
        rule: FUSE_RULE,
        span,
        before,
        after,
        because,
    }
}
