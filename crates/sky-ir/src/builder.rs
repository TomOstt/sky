//! Building functions: the API lowering and tests use to create SIR.
//!
//! The builder keeps an insertion point (a block) and appends operations
//! there. It does not check types or structure; the verifier does.

use sky_diag::Span;

use crate::ir::{
    BlockData, BlockId, EffectSet, Function, OpData, OpId, RegionData, RegionId, Type, ValueData,
    ValueDef, ValueId,
};
use crate::op::OpKind;

/// Builds one [`Function`].
#[derive(Debug)]
pub struct FunctionBuilder {
    func: Function,
    at: BlockId,
}

impl FunctionBuilder {
    /// Starts `name` with parameters of `params` types. The body region and
    /// its entry block are created, the parameters are the entry block's
    /// arguments, and the insertion point is the entry block.
    pub fn new(name: impl Into<String>, params: Vec<Type>, results: Vec<Type>, span: Span) -> Self {
        let mut func = Function {
            name: name.into(),
            params: params.clone(),
            results,
            effects: EffectSet::PURE,
            body: RegionId(0),
            span,
            values: Vec::new(),
            ops: Vec::new(),
            blocks: Vec::new(),
            regions: Vec::new(),
        };
        func.regions.push(RegionData::default());
        let mut b = FunctionBuilder {
            func,
            at: BlockId(0),
        };
        let entry = b.add_block(RegionId(0), params);
        b.at = entry;
        b
    }

    /// The function so far (for inspecting types while building).
    pub fn func(&self) -> &Function {
        &self.func
    }

    /// The parameter values.
    pub fn params(&self) -> Vec<ValueId> {
        self.func.param_values().to_vec()
    }

    pub fn set_effects(&mut self, effects: EffectSet) {
        self.func.effects = effects;
    }

    /// Changes the source span of `op` (the parser learns it after the
    /// operation's regions).
    pub fn set_op_span(&mut self, op: OpId, span: Span) {
        self.func.ops[op.index()].span = span;
    }

    /// Names a value for printing (`%x`).
    pub fn set_name(&mut self, v: ValueId, name: impl Into<String>) {
        self.func.values[v.index()].name = Some(name.into());
    }

    /// The current insertion block.
    pub fn insertion_block(&self) -> BlockId {
        self.at
    }

    /// Appends operations to `block` from now on.
    pub fn set_insertion_block(&mut self, block: BlockId) {
        self.at = block;
    }

    /// Creates a new empty region owned by `op`.
    pub fn add_region(&mut self, op: OpId) -> RegionId {
        let r = RegionId(self.func.regions.len() as u32);
        self.func.regions.push(RegionData {
            blocks: Vec::new(),
            parent: Some(op),
        });
        self.func.ops[op.index()].regions.push(r);
        r
    }

    /// Creates a block at the end of `region`, with one argument per type.
    pub fn add_block(&mut self, region: RegionId, arg_types: Vec<Type>) -> BlockId {
        let block = BlockId(self.func.blocks.len() as u32);
        self.func.blocks.push(BlockData {
            args: Vec::new(),
            ops: Vec::new(),
            parent: Some(region),
        });
        for (i, ty) in arg_types.into_iter().enumerate() {
            let v = self.new_value(
                ty,
                ValueDef::BlockArg {
                    block,
                    index: i as u32,
                },
            );
            self.func.blocks[block.index()].args.push(v);
        }
        self.func.regions[region.index()].blocks.push(block);
        block
    }

    /// Appends an operation at the insertion point and returns it with its
    /// results (one per entry of `result_types`). Regions are added
    /// afterwards with [`Self::add_region`].
    pub fn op(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        result_types: Vec<Type>,
        span: Span,
    ) -> (OpId, Vec<ValueId>) {
        let op = OpId(self.func.ops.len() as u32);
        self.func.ops.push(OpData {
            kind,
            operands,
            results: Vec::new(),
            regions: Vec::new(),
            span,
            parent: self.at,
        });
        let results: Vec<ValueId> = result_types
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                self.new_value(
                    ty,
                    ValueDef::OpResult {
                        op,
                        index: i as u32,
                    },
                )
            })
            .collect();
        self.func.ops[op.index()].results = results.clone();
        self.func.blocks[self.at.index()].ops.push(op);
        (op, results)
    }

    /// Appends an operation with exactly one result and returns the result.
    pub fn op1(&mut self, kind: OpKind, operands: Vec<ValueId>, ty: Type, span: Span) -> ValueId {
        self.op(kind, operands, vec![ty], span).1[0]
    }

    /// Finishes the function.
    pub fn finish(self) -> Function {
        self.func
    }

    fn new_value(&mut self, ty: Type, def: ValueDef) -> ValueId {
        let v = ValueId(self.func.values.len() as u32);
        self.func.values.push(ValueData {
            ty,
            def,
            name: None,
        });
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_units::Dimension;

    #[test]
    fn builds_an_if_with_two_regions() {
        let s = Span::point(0);
        let len = Type::scalar(Dimension::LENGTH);
        let mut b = FunctionBuilder::new("f", vec![Type::Bool, len.clone()], vec![len.clone()], s);
        let [c, x] = b.params()[..] else {
            unreachable!()
        };
        let (iff, res) = b.op(OpKind::If, vec![c], vec![len.clone()], s);
        let outer = b.insertion_block();
        for arm in 0..2 {
            let r = b.add_region(iff);
            let blk = b.add_block(r, vec![]);
            b.set_insertion_block(blk);
            let v = if arm == 0 {
                x
            } else {
                b.op1(OpKind::Neg, vec![x], len.clone(), s)
            };
            b.op(OpKind::Yield, vec![v], vec![], s);
        }
        b.set_insertion_block(outer);
        b.op(OpKind::Return, res.clone(), vec![], s);
        let f = b.finish();
        assert_eq!(f.param_values().len(), 2);
        assert_eq!(f.op(iff).regions.len(), 2);
        assert_eq!(f.block(f.entry()).ops.len(), 2);
        assert!(matches!(
            f.value(res[0]).def,
            ValueDef::OpResult { op, index: 0 } if op == iff
        ));
    }
}
