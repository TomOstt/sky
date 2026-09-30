//! A small function builder over the public arenas of [`sky_ir::Function`].
//!
//! `sky_ir::FunctionBuilder` fixes an operation's result types and a block's
//! arguments when they are created. Lowering cannot: an `if`'s results are
//! the variables its branches changed, and a loop's carried variables get
//! their final types only after the body is lowered. So this builder can
//! append results, block arguments and function parameters later, and insert
//! an operation before another. It creates nothing `sky_ir` cannot express:
//! the output is an ordinary [`Function`].

use sky_diag::Span;
use sky_ir::{
    BlockData, BlockId, EffectSet, Function, OpData, OpId, OpKind, RegionData, RegionId, Type,
    ValueData, ValueDef, ValueId,
};

pub(crate) struct Builder {
    f: Function,
    at: BlockId,
}

impl Builder {
    /// A function with no parameters and no results yet; the insertion
    /// point is its entry block.
    pub(crate) fn new(name: String, span: Span) -> Self {
        let f = Function {
            name,
            params: Vec::new(),
            results: Vec::new(),
            effects: EffectSet::PURE,
            body: RegionId(0),
            span,
            values: Vec::new(),
            ops: Vec::new(),
            blocks: Vec::new(),
            regions: vec![RegionData::default()],
        };
        let mut b = Builder { f, at: BlockId(0) };
        let entry = b.new_block(RegionId(0), Vec::new());
        b.at = entry;
        b
    }

    /// The function's name.
    pub(crate) fn name(&self) -> &str {
        &self.f.name
    }

    /// The insertion block.
    pub(crate) fn at(&self) -> BlockId {
        self.at
    }

    pub(crate) fn set_at(&mut self, block: BlockId) {
        self.at = block;
    }

    pub(crate) fn ty(&self, v: ValueId) -> &Type {
        &self.f.values[v.index()].ty
    }

    pub(crate) fn set_ty(&mut self, v: ValueId, ty: Type) {
        self.f.values[v.index()].ty = ty;
    }

    /// Names `v` for printing, unless it already has a name.
    pub(crate) fn name_value(&mut self, v: ValueId, name: &str) {
        let slot = &mut self.f.values[v.index()].name;
        if slot.is_none() {
            *slot = Some(name.to_string());
        }
    }

    pub(crate) fn set_results(&mut self, results: Vec<Type>) {
        self.f.results = results;
    }

    /// Appends a parameter to the entry block.
    pub(crate) fn add_param(&mut self, ty: Type) -> ValueId {
        self.f.params.push(ty.clone());
        let entry = self.f.entry();
        self.add_block_arg(entry, ty)
    }

    pub(crate) fn new_region(&mut self, op: OpId) -> RegionId {
        let r = RegionId(self.f.regions.len() as u32);
        self.f.regions.push(RegionData {
            blocks: Vec::new(),
            parent: Some(op),
        });
        self.f.ops[op.index()].regions.push(r);
        r
    }

    pub(crate) fn new_block(&mut self, region: RegionId, arg_types: Vec<Type>) -> BlockId {
        let block = BlockId(self.f.blocks.len() as u32);
        self.f.blocks.push(BlockData {
            args: Vec::new(),
            ops: Vec::new(),
            parent: Some(region),
        });
        for ty in arg_types {
            self.add_block_arg(block, ty);
        }
        self.f.regions[region.index()].blocks.push(block);
        block
    }

    pub(crate) fn add_block_arg(&mut self, block: BlockId, ty: Type) -> ValueId {
        let index = self.f.blocks[block.index()].args.len() as u32;
        let v = self.new_value(ty, ValueDef::BlockArg { block, index });
        self.f.blocks[block.index()].args.push(v);
        v
    }

    /// Appends an operation at the insertion point.
    pub(crate) fn op(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        result_types: Vec<Type>,
        span: Span,
    ) -> (OpId, Vec<ValueId>) {
        let op = self.push_op(kind, operands, span, self.at);
        let results: Vec<ValueId> = result_types
            .into_iter()
            .map(|ty| self.add_result(op, ty))
            .collect();
        self.f.blocks[self.at.index()].ops.push(op);
        (op, results)
    }

    pub(crate) fn op1(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        ty: Type,
        span: Span,
    ) -> ValueId {
        let (_, results) = self.op(kind, operands, vec![ty], span);
        results[0]
    }

    /// [`Self::op1`] that also records the operands' source spans.
    pub(crate) fn op1_spans(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        operand_spans: Vec<Span>,
        ty: Type,
        span: Span,
    ) -> ValueId {
        let (op, results) = self.op(kind, operands, vec![ty], span);
        self.set_operand_spans(op, operand_spans);
        results[0]
    }

    /// Records the source spans of `op`'s operands at their use
    /// (`OpData::operand_spans`).
    pub(crate) fn set_operand_spans(&mut self, op: OpId, spans: Vec<Span>) {
        self.f.ops[op.index()].operand_spans = spans;
    }

    /// How many operations `block` has.
    pub(crate) fn block_len(&self, block: BlockId) -> usize {
        self.f.blocks[block.index()].ops.len()
    }

    /// Moves the operations of `block` from position `from` on to right
    /// before `before` (an operation of `block` that comes earlier), in the
    /// same order: for values built after a region was lowered that the
    /// operation with the region needs before it.
    pub(crate) fn hoist_before(&mut self, block: BlockId, from: usize, before: OpId) {
        let ops = &mut self.f.blocks[block.index()].ops;
        let moved: Vec<OpId> = ops.split_off(from);
        let pos = ops.iter().rposition(|o| *o == before).unwrap_or(ops.len());
        ops.splice(pos..pos, moved);
    }

    /// Appends a result to an existing operation.
    pub(crate) fn add_result(&mut self, op: OpId, ty: Type) -> ValueId {
        let index = self.f.ops[op.index()].results.len() as u32;
        let v = self.new_value(ty, ValueDef::OpResult { op, index });
        self.f.ops[op.index()].results.push(v);
        v
    }

    /// Replaces operand `index` of `op`.
    pub(crate) fn set_operand(&mut self, op: OpId, index: usize, v: ValueId) {
        if let Some(slot) = self.f.ops[op.index()].operands.get_mut(index) {
            *slot = v;
        }
    }

    /// The arguments of `block`.
    pub(crate) fn block_args(&self, block: BlockId) -> Vec<ValueId> {
        self.f.blocks[block.index()].args.clone()
    }

    /// Finishes the function. Its effects are left pure: they depend on the
    /// callees, so `lower` infers them for the whole module at the end
    /// (`sky_ir::effects`).
    pub(crate) fn finish(self) -> Function {
        self.f
    }

    fn push_op(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        span: Span,
        parent: BlockId,
    ) -> OpId {
        let op = OpId(self.f.ops.len() as u32);
        self.f.ops.push(OpData {
            kind,
            operands,
            results: Vec::new(),
            regions: Vec::new(),
            span,
            operand_spans: Vec::new(),
            parent,
        });
        op
    }

    fn new_value(&mut self, ty: Type, def: ValueDef) -> ValueId {
        let v = ValueId(self.f.values.len() as u32);
        self.f.values.push(ValueData {
            ty,
            def,
            name: None,
        });
        v
    }
}
