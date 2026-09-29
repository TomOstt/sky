//! Canonical numbering: the order the text format prints in and the parser
//! rebuilds in.
//!
//! The arenas of a [`Function`] are in *creation* order, which passes and
//! builders are free to vary (a `while`'s two regions may be created before
//! either is filled; a pass leaves removed operations behind). The text
//! format has one order: a pre-order walk of the function (parameters, then
//! each operation followed by the contents of its regions, in order), and
//! [`crate::parse`] allocates ids in exactly that order. So
//! `parse(print(m))` equals [`canonicalize`]`(m)`, and equals `m` itself
//! whenever `m` was built in canonical order (as the parser and the builder
//! do for straight-line and nested code).
//!
//! Value names are part of the canonical form: they are uniquified in this
//! order (`x`, `x_1`, `x_2`, ...), and a value whose name cannot be printed
//! as one token (empty, or containing anything but letters, digits and `_`,
//! or made of digits only) is unnamed.

use std::collections::{HashMap, HashSet};

use crate::builder::FunctionBuilder;
use crate::display::is_bare_name;
use crate::ir::{BlockId, Function, Module, RegionId, ValueId};
use crate::verify::MAX_REGION_DEPTH;

/// Where each reachable value and block sits in the canonical order, and
/// the final (uniquified) name of each value.
pub(crate) struct Layout {
    /// Values in canonical order.
    pub values: Vec<ValueId>,
    /// Canonical position by `ValueId::index`; `usize::MAX` if unreachable.
    pub value_pos: Vec<usize>,
    /// Canonical position by `BlockId::index`; `usize::MAX` if unreachable.
    pub block_pos: Vec<usize>,
    /// Number of blocks reached so far.
    blocks: usize,
    /// Final name by `ValueId::index`; `None` prints as its position.
    pub names: Vec<Option<String>>,
}

impl Layout {
    pub(crate) fn new(f: &Function) -> Layout {
        let mut l = Layout {
            values: Vec::new(),
            value_pos: vec![usize::MAX; f.values.len()],
            block_pos: vec![usize::MAX; f.blocks.len()],
            blocks: 0,
            names: vec![None; f.values.len()],
        };
        l.walk(f, f.body, 0);
        let mut used: HashSet<String> = HashSet::new();
        // The next suffix to try per base name, so a name reused many times
        // (`x` assigned in a long loop body) is not quadratic to uniquify.
        let mut next: HashMap<String, usize> = HashMap::new();
        for i in 0..l.values.len() {
            let v = l.values[i];
            let Some(raw) = f.values[v.index()].name.as_deref() else {
                continue;
            };
            if !is_bare_name(raw) || raw.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let mut name = raw.to_string();
            if used.contains(&name) {
                let n = next.entry(raw.to_string()).or_insert(1);
                loop {
                    name = format!("{raw}_{n}");
                    *n += 1;
                    if !used.contains(&name) {
                        break;
                    }
                }
            }
            used.insert(name.clone());
            l.names[v.index()] = Some(name);
        }
        l
    }

    fn walk(&mut self, f: &Function, r: RegionId, depth: usize) {
        if depth >= MAX_REGION_DEPTH {
            return;
        }
        let Some(region) = f.regions.get(r.index()) else {
            return;
        };
        for &b in &region.blocks {
            let Some(block) = f.blocks.get(b.index()) else {
                continue;
            };
            if self.block_pos[b.index()] != usize::MAX {
                continue;
            }
            self.block_pos[b.index()] = self.blocks;
            self.blocks += 1;
            for &v in &block.args {
                self.add_value(f, v);
            }
            for &o in &block.ops {
                let Some(op) = f.ops.get(o.index()) else {
                    continue;
                };
                for &v in &op.results {
                    self.add_value(f, v);
                }
                for &r in &op.regions {
                    self.walk(f, r, depth + 1);
                }
            }
        }
    }

    fn add_value(&mut self, f: &Function, v: ValueId) {
        if v.index() < f.values.len() && self.value_pos[v.index()] == usize::MAX {
            self.value_pos[v.index()] = self.values.len();
            self.values.push(v);
        }
    }

    /// The token after `%` for value `v`.
    pub(crate) fn value_token(&self, v: ValueId) -> String {
        match self.names.get(v.index()) {
            Some(Some(n)) => n.clone(),
            _ if self
                .value_pos
                .get(v.index())
                .is_some_and(|&p| p != usize::MAX) =>
            {
                self.value_pos[v.index()].to_string()
            }
            // Not reachable from the body: a dangling reference.
            _ => format!("?{}", v.0),
        }
    }
}

/// Rebuilds `m` in canonical order (see the module docs), dropping arena
/// entries the body does not reach. Meant for verified modules: it indexes
/// the arenas directly and panics on out-of-range ids.
pub fn canonicalize(m: &Module) -> Module {
    Module {
        edition: m.edition,
        functions: m.functions.iter().map(canonical_function).collect(),
    }
}

fn canonical_function(f: &Function) -> Function {
    let layout = Layout::new(f);
    let entry = f.entry();
    let params: Vec<_> = f
        .block(entry)
        .args
        .iter()
        .map(|&v| f.ty(v).clone())
        .collect();
    let mut b = FunctionBuilder::new(f.name.clone(), params, f.results.clone(), f.span);
    b.set_effects(f.effects);
    let mut map: Vec<Option<ValueId>> = vec![None; f.values.len()];
    for (&old, new) in f.block(entry).args.iter().zip(b.params()) {
        map[old.index()] = Some(new);
    }
    copy_block(f, entry, &mut b, &mut map, 0);
    for &old in &layout.values {
        if let (Some(name), Some(new)) = (&layout.names[old.index()], map[old.index()]) {
            b.set_name(new, name.clone());
        }
    }
    b.finish()
}

fn copy_block(
    f: &Function,
    block: BlockId,
    b: &mut FunctionBuilder,
    map: &mut [Option<ValueId>],
    depth: usize,
) {
    if depth >= MAX_REGION_DEPTH {
        return;
    }
    for &o in &f.block(block).ops {
        let op = f.op(o);
        let operands = op
            .operands
            .iter()
            .map(|v| map[v.index()].unwrap_or(*v))
            .collect();
        let tys = op.results.iter().map(|&v| f.ty(v).clone()).collect();
        let (new_op, results) = b.op(op.kind.clone(), operands, tys, op.span);
        for (&old, new) in op.results.iter().zip(results) {
            map[old.index()] = Some(new);
        }
        let here = b.insertion_block();
        for &r in &op.regions {
            let nr = b.add_region(new_op);
            for &blk in &f.region(r).blocks {
                let tys = f.block(blk).args.iter().map(|&v| f.ty(v).clone()).collect();
                let nb = b.add_block(nr, tys);
                let new_args = b.func().block(nb).args.clone();
                for (&old, new) in f.block(blk).args.iter().zip(new_args) {
                    map[old.index()] = Some(new);
                }
                b.set_insertion_block(nb);
                copy_block(f, blk, b, map, depth + 1);
            }
        }
        b.set_insertion_block(here);
    }
}
