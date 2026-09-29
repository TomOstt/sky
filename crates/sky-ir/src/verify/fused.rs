//! The verifier's rules for `fused`, specified in [`crate::op`]: the body is
//! a self-contained scalar computation that can only fail, and its
//! arguments and yield match the operands and the result.
//!
//! Like the rest of the verifier these rules skip what an `Unknown` type
//! hides: an operand of unknown type passes its checks, and an unknown
//! result is accepted. The body's own types must be exact.

use sky_types::Shape;

use super::FnVerifier;
use crate::ir::{BlockId, Effect, OpData, OpId, Type, ValueDef, ValueId};
use crate::op::OpKind;
use crate::print::effect_name;

impl FnVerifier<'_> {
    pub(super) fn check_fused(&mut self, o: OpId, op: &OpData) {
        let f = self.f;
        let [region] = op.regions[..] else {
            // Reported by the region count check.
            return;
        };
        let Some(&body) = f.regions.get(region.index()).and_then(|r| r.blocks.first()) else {
            return;
        };
        if op.operands.is_empty() {
            self.err_op(o, "`fused` needs at least one operand".into());
        }
        let shape = self.check_fused_operands(o, op);
        self.check_fused_args(o, op, body);
        let yielded = self.check_fused_body(o, body);
        // The result: an array of the operands' shape and the body's
        // dimension.
        match f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(r) => {
                if let Some(s) = shape.as_ref().filter(|s| **s != r.shape) {
                    self.err_op(
                        o,
                        format!(
                            "the result of `fused` should have shape {s} (the operands'), found {}",
                            r.shape
                        ),
                    );
                }
                if let Some(Type::Number(y)) = yielded.map(|v| f.ty(v)) {
                    if y.dim != r.dim {
                        self.err_op(
                            o,
                            format!(
                                "the result of `fused` should have dimension {} (the body's), \
                                 found {}",
                                y.dim, r.dim
                            ),
                        );
                    }
                }
            }
            other => self.err_op(
                o,
                format!("the result of `fused` must be a number array, found {other}"),
            ),
        }
    }

    /// The array operands share one shape, and there is at least one array
    /// (unless a type is unknown). Returns that shape.
    fn check_fused_operands(&mut self, o: OpId, op: &OpData) -> Option<Shape> {
        let f = self.f;
        let mut shape: Option<Shape> = None;
        let mut unknown = false;
        for (i, &v) in op.operands.iter().enumerate() {
            match f.ty(v) {
                Type::Number(n) if n.shape.rank() == 0 => {}
                Type::Number(n) => match &shape {
                    None => shape = Some(n.shape.clone()),
                    Some(s) if *s != n.shape => {
                        self.err_op(
                            o,
                            format!(
                                "operands of `fused` have different shapes: {s} and {} \
                                 (operand {i})",
                                n.shape
                            ),
                        );
                        return None;
                    }
                    Some(_) => {}
                },
                Type::Unknown => unknown = true,
                other => {
                    self.err_op(
                        o,
                        format!("operand {i} of `fused` must be a number, found {other}"),
                    );
                    return None;
                }
            }
        }
        if shape.is_none() && !unknown && !op.operands.is_empty() {
            self.err_op(
                o,
                "`fused` needs at least one array operand (all operands are scalars)".into(),
            );
        }
        shape
    }

    /// One block argument per operand: a scalar of the operand's dimension.
    fn check_fused_args(&mut self, o: OpId, op: &OpData, body: BlockId) {
        let f = self.f;
        let Some(block) = f.blocks.get(body.index()) else {
            return;
        };
        if block.args.len() != op.operands.len() {
            self.err_op(
                o,
                format!(
                    "the body of `fused` takes {} argument(s), one per operand; found {}",
                    op.operands.len(),
                    block.args.len()
                ),
            );
            return;
        }
        for (i, (&a, &v)) in block.args.iter().zip(&op.operands).enumerate() {
            let (Some(at), Some(vt)) = (f.values.get(a.index()), f.values.get(v.index())) else {
                continue;
            };
            match (&at.ty, &vt.ty) {
                (Type::Number(x), _) if x.shape.rank() != 0 => self.err_op(
                    o,
                    format!(
                        "argument {i} of the `fused` body must be a scalar, found {}",
                        at.ty
                    ),
                ),
                (Type::Number(x), Type::Number(y)) if x.dim != y.dim => self.err_op(
                    o,
                    format!(
                        "argument {i} of the `fused` body has dimension {}, but its operand has {}",
                        x.dim, y.dim
                    ),
                ),
                (Type::Number(_), _) => {}
                (other, _) => self.err_op(
                    o,
                    format!(
                        "argument {i} of the `fused` body must be a scalar number, found {other}"
                    ),
                ),
            }
        }
    }

    /// The body holds only scalar operations that are pure or can fail,
    /// with no regions, using only values of the body. Returns the yielded
    /// value.
    fn check_fused_body(&mut self, o: OpId, body: BlockId) -> Option<ValueId> {
        let f = self.f;
        let block = f.blocks.get(body.index())?;
        let in_body = |v: ValueId| match f.values.get(v.index()).map(|d| d.def) {
            Some(ValueDef::BlockArg { block, .. }) => block == body,
            Some(ValueDef::OpResult { op, .. }) => {
                f.ops.get(op.index()).is_some_and(|p| p.parent == body)
            }
            None => false,
        };
        let mut yielded = None;
        for &id in &block.ops {
            let Some(inner) = f.ops.get(id.index()) else {
                continue;
            };
            let name = inner.kind.mnemonic();
            if matches!(inner.kind, OpKind::Yield) {
                match inner.operands[..] {
                    [v] => yielded = Some(v),
                    _ => self.err_op(
                        id,
                        format!(
                            "the body of `fused` yields one value, found {}",
                            inner.operands.len()
                        ),
                    ),
                }
            } else if !inner.kind.properties().terminator {
                let effects = inner.kind.properties().effects;
                if let Some(e) = effects.iter().find(|e| *e != Effect::Fail) {
                    self.err_op(
                        id,
                        format!(
                            "`{name}` is not allowed in the body of `fused`: it has the `{}` \
                             effect, and a body may only be pure or fail",
                            effect_name(e)
                        ),
                    );
                }
                if !inner.regions.is_empty() {
                    self.err_op(
                        id,
                        format!("`{name}` has regions, which the body of `fused` cannot have"),
                    );
                }
                for &r in &inner.results {
                    let scalar = match f.ty(r) {
                        Type::Number(n) => n.shape.rank() == 0,
                        Type::Bool => true,
                        _ => false,
                    };
                    if !scalar {
                        self.err_op(
                            id,
                            format!(
                                "`{name}` in the body of `fused` must give a scalar number or a \
                                 Bool, found {}",
                                f.ty(r)
                            ),
                        );
                    }
                }
            }
            for &v in &inner.operands {
                if !in_body(v) {
                    self.err_op(
                        id,
                        format!(
                            "`{name}` in the body of `fused` uses {v}, which is not an argument \
                             or result of the body"
                        ),
                    );
                }
            }
        }
        if yielded.is_none() {
            self.err_op(
                o,
                "the body of `fused` must end in a `yield` of one value".into(),
            );
        }
        yielded
    }
}
