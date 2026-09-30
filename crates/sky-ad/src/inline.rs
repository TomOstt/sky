//! Inlining of calls that pass or return arrays, before differentiation.
//!
//! The chain rule through a call is a product of the callee's partial
//! derivatives with the tangents of its arguments. For scalars that is one
//! multiplication per argument; for arrays it is a Jacobian, which SIR has
//! no way to write. But a call is only a name for its body, so a call with
//! an array operand or result is replaced by the body (with the arguments
//! substituted), and the derivative flows through it like any other code:
//! a helper `displacement(p)` inside a potential is part of the potential.
//!
//! Calls with scalar operands and results stay calls (the callee may loop,
//! which reverse mode does not handle, and forward mode does). A recursive
//! call is never inlined, and neither is one nested deeper than
//! [`MAX_INLINE_DEPTH`]: it stays a call, which differentiation reports if
//! it cannot handle.

use sky_ir::{BlockId, Function, FunctionBuilder, Module, OpData, OpKind, Type, ValueId};

use crate::AdError;

/// How many nested inlined calls (a helper of a helper of ...) are expanded.
const MAX_INLINE_DEPTH: usize = 16;

/// Whether the call `op` (to `callee`) has an array operand or result.
fn touches_arrays(f: &Function, op: &OpData) -> bool {
    let array = |v: &ValueId| matches!(f.ty(*v), Type::Number(n) if n.shape.rank() > 0);
    op.operands.iter().any(array) || op.results.iter().any(array)
}

/// Whether `f` has a call that [`inline_array_calls`] would expand.
fn wants_inlining(f: &Function, block: BlockId) -> bool {
    f.block(block).ops.iter().any(|id| {
        let op = f.op(*id);
        (matches!(op.kind, OpKind::Call { .. }) && touches_arrays(f, op))
            || op
                .regions
                .iter()
                .flat_map(|r| &f.region(*r).blocks)
                .any(|b| wants_inlining(f, *b))
    })
}

/// A copy of `f` in which every non-recursive call with an array operand or
/// result is replaced by the callee's body.
pub(crate) fn inline_array_calls(m: &Module, f: &Function) -> Result<Function, AdError> {
    if !wants_inlining(f, f.entry()) {
        return Ok(f.clone());
    }
    let mut b = FunctionBuilder::new(f.name.clone(), f.params.clone(), f.results.clone(), f.span);
    b.set_effects(f.effects);
    let mut inl = Inliner {
        module: m,
        b,
        stack: vec![f.name.clone()],
    };
    let params = inl.b.params();
    let mut map = vec![None; f.values.len()];
    for (old, new) in f.param_values().iter().zip(&params) {
        map[old.index()] = Some(*new);
        if let Some(n) = &f.value(*old).name {
            inl.b.set_name(*new, n.clone());
        }
    }
    let ret = inl.block(f, f.entry(), &mut map)?;
    let Some(ret) = ret else {
        return Err(AdError::Malformed {
            func: f.name.clone(),
            why: "the body must end in `return`".into(),
        });
    };
    inl.b.op(OpKind::Return, ret.0, vec![], ret.1);
    Ok(inl.b.finish())
}

struct Inliner<'m> {
    module: &'m Module,
    b: FunctionBuilder,
    /// The functions being expanded, outermost first: a call to one of them
    /// is recursive.
    stack: Vec<String>,
}

/// The mapped operands and span of a block's terminator.
type Terminator = (Vec<ValueId>, sky_diag::Span);

impl Inliner<'_> {
    fn get(&self, f: &Function, map: &[Option<ValueId>], v: ValueId) -> Result<ValueId, AdError> {
        map[v.index()].ok_or_else(|| AdError::Malformed {
            func: f.name.clone(),
            why: format!("{v} is used before it is defined"),
        })
    }

    /// Copies the operations of `block` of `f`. A `return` or `yield` that
    /// ends it is not copied but handed back (mapped), for the caller to
    /// place: as a `return`, a `yield`, or the results of an inlined call.
    /// `condition` is copied by the caller of a `while`, which passes its
    /// region's terminator through here as well.
    fn block(
        &mut self,
        f: &Function,
        block: BlockId,
        map: &mut Vec<Option<ValueId>>,
    ) -> Result<Option<Terminator>, AdError> {
        for &id in &f.block(block).ops {
            let op = f.op(id);
            if op.kind.properties().terminator {
                let operands = op
                    .operands
                    .iter()
                    .map(|v| self.get(f, map, *v))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Some((operands, op.span)));
            }
            if let OpKind::Call { callee } = &op.kind {
                if touches_arrays(f, op) && self.can_inline(callee) {
                    self.inline_call(f, op, callee, map)?;
                    continue;
                }
            }
            self.copy_op(f, op, map)?;
        }
        Ok(None)
    }

    fn can_inline(&self, callee: &str) -> bool {
        self.module.function(callee).is_some()
            && !self.stack.iter().any(|s| s == callee)
            && self.stack.len() < MAX_INLINE_DEPTH
    }

    fn inline_call(
        &mut self,
        f: &Function,
        op: &OpData,
        callee: &str,
        map: &mut [Option<ValueId>],
    ) -> Result<(), AdError> {
        let g = self.module.function(callee).expect("checked by can_inline");
        let mut inner = vec![None; g.values.len()];
        for (param, arg) in g.param_values().iter().zip(&op.operands) {
            inner[param.index()] = Some(self.get(f, map, *arg)?);
        }
        self.stack.push(callee.to_string());
        let ret = self.block(g, g.entry(), &mut inner);
        self.stack.pop();
        let Some((values, _)) = ret? else {
            return Err(AdError::Malformed {
                func: g.name.clone(),
                why: "the body must end in `return`".into(),
            });
        };
        if values.len() != op.results.len() {
            return Err(AdError::Malformed {
                func: g.name.clone(),
                why: "`return` must have one operand per result".into(),
            });
        }
        for (r, v) in op.results.iter().zip(values) {
            map[r.index()] = Some(v);
        }
        Ok(())
    }

    /// Copies one operation, with its regions.
    fn copy_op(
        &mut self,
        f: &Function,
        op: &OpData,
        map: &mut Vec<Option<ValueId>>,
    ) -> Result<(), AdError> {
        let operands = op
            .operands
            .iter()
            .map(|v| self.get(f, map, *v))
            .collect::<Result<Vec<_>, _>>()?;
        let tys = op.results.iter().map(|r| f.ty(*r).clone()).collect();
        let (new_op, results) = self.b.op(op.kind.clone(), operands, tys, op.span);
        for (old, new) in op.results.iter().zip(&results) {
            map[old.index()] = Some(*new);
            if let Some(n) = &f.value(*old).name {
                self.b.set_name(*new, n.clone());
            }
        }
        let outer = self.b.insertion_block();
        for &region in &op.regions {
            let r = self.b.add_region(new_op);
            let &[blk] = &f.region(region).blocks[..] else {
                return Err(AdError::Malformed {
                    func: f.name.clone(),
                    why: "a region must have exactly one block".into(),
                });
            };
            let arg_tys = f.block(blk).args.iter().map(|a| f.ty(*a).clone()).collect();
            let nb = self.b.add_block(r, arg_tys);
            let new_args = self.b.func().block(nb).args.clone();
            for (a, n) in f.block(blk).args.iter().zip(&new_args) {
                map[a.index()] = Some(*n);
            }
            self.b.set_insertion_block(nb);
            let Some((operands, span)) = self.block(f, blk, map)? else {
                return Err(AdError::Malformed {
                    func: f.name.clone(),
                    why: "a block does not end in a terminator".into(),
                });
            };
            // The terminator of a region is `yield` or `condition`.
            let kind = f
                .block(blk)
                .ops
                .last()
                .map(|t| f.op(*t).kind.clone())
                .unwrap_or(OpKind::Yield);
            self.b.op(kind, operands, vec![], span);
        }
        self.b.set_insertion_block(outer);
        Ok(())
    }
}
