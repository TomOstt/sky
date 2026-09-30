//! Copying operations inside one function, with fresh values.
//!
//! Loop unrolling inlines a region's operations more than once, so each copy
//! needs values of its own (SSA: one definition per value). [`Copier`] copies
//! an operation and everything nested in it, keeping kinds, types, names,
//! spans and operand spans exactly, and renames the values through a map
//! that starts with the region's block arguments bound to what they take.

use std::collections::HashMap;

use sky_ir::{
    BlockData, BlockId, Function, OpData, OpId, RegionData, RegionId, ValueData, ValueDef, ValueId,
};

/// Copies operations into a block of `f`, mapping values as it goes.
pub(crate) struct Copier<'f> {
    pub(crate) f: &'f mut Function,
    map: HashMap<ValueId, ValueId>,
    /// How many operations were copied, nested ones included.
    pub(crate) copied: usize,
}

impl<'f> Copier<'f> {
    pub(crate) fn new(f: &'f mut Function) -> Self {
        Copier {
            f,
            map: HashMap::new(),
            copied: 0,
        }
    }

    /// From now on, uses of `from` in copied operations read `to`.
    pub(crate) fn bind(&mut self, from: ValueId, to: ValueId) {
        self.map.insert(from, to);
    }

    /// What `v` is called in the copies.
    pub(crate) fn get(&self, v: ValueId) -> ValueId {
        self.map.get(&v).copied().unwrap_or(v)
    }

    /// Copies `op` (regions and all) to the end of `into`'s operation list
    /// and returns the copy. Its results are new values, bound in the map.
    pub(crate) fn copy_op(&mut self, op: OpId, into: BlockId) -> OpId {
        let old = self.f.ops[op.index()].clone();
        let new = OpId(self.f.ops.len() as u32);
        self.copied += 1;
        let operands: Vec<ValueId> = old.operands.iter().map(|v| self.get(*v)).collect();
        self.f.ops.push(OpData {
            kind: old.kind.clone(),
            operands,
            results: Vec::new(),
            regions: Vec::new(),
            span: old.span,
            operand_spans: old.operand_spans.clone(),
            parent: into,
        });
        let mut results = Vec::with_capacity(old.results.len());
        for (i, r) in old.results.iter().enumerate() {
            let fresh = self.fresh_value(
                *r,
                ValueDef::OpResult {
                    op: new,
                    index: i as u32,
                },
            );
            results.push(fresh);
        }
        self.f.ops[new.index()].results = results;
        for r in &old.regions {
            let copy = self.copy_region(*r, new);
            self.f.ops[new.index()].regions.push(copy);
        }
        self.f.blocks[into.index()].ops.push(new);
        new
    }

    fn fresh_value(&mut self, like: ValueId, def: ValueDef) -> ValueId {
        let data = &self.f.values[like.index()];
        let fresh = ValueId(self.f.values.len() as u32);
        let copy = ValueData {
            ty: data.ty.clone(),
            def,
            name: data.name.clone(),
        };
        self.f.values.push(copy);
        self.map.insert(like, fresh);
        fresh
    }

    fn copy_region(&mut self, r: RegionId, owner: OpId) -> RegionId {
        let old = self.f.regions[r.index()].clone();
        let new = RegionId(self.f.regions.len() as u32);
        self.f.regions.push(RegionData {
            blocks: Vec::new(),
            parent: Some(owner),
        });
        for b in old.blocks {
            let copy = self.copy_block(b, new);
            self.f.regions[new.index()].blocks.push(copy);
        }
        new
    }

    fn copy_block(&mut self, b: BlockId, region: RegionId) -> BlockId {
        let old = self.f.blocks[b.index()].clone();
        let new = BlockId(self.f.blocks.len() as u32);
        self.f.blocks.push(BlockData {
            args: Vec::new(),
            ops: Vec::new(),
            parent: Some(region),
        });
        for (i, a) in old.args.iter().enumerate() {
            let fresh = self.fresh_value(
                *a,
                ValueDef::BlockArg {
                    block: new,
                    index: i as u32,
                },
            );
            self.f.blocks[new.index()].args.push(fresh);
        }
        for o in old.ops {
            self.copy_op(o, new);
        }
        new
    }
}
