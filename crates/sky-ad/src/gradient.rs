//! The gradient by forward mode, one pass per entry: the fallback for what
//! reverse mode does not handle (a loop that carries a value the gradient
//! depends on, chiefly).
//!
//! The gradient function `d<f>/d<p>(params...) -> (value, ∂f/∂p)` has the
//! same signature as reverse mode's, so a caller cannot tell them apart. Its
//! body calls the directional derivative `d<f>/d<p>#jvp(params..., e)` (see
//! `emit::build` with `Seed::Direction`) once for every unit direction `e`
//! (a one-hot array of `p`'s shape), and gathers the `n` results, which are
//! the partial derivatives `∂f/∂pᵢ`, in an array of `p`'s shape. The
//! first call gives the value.
//!
//! The cost is `n` evaluations of `f` (with tangents) instead of reverse
//! mode's one forward and one backward sweep, and the loop over `i` is
//! unrolled, so `p`'s shape must be known at compile time and small
//! ([`MAX_PASSES`]).

use sky_ir::{Function, FunctionBuilder, Module, OpKind, SliceAxis, Type, ValueId};
use sky_types::{Shape, ShapeDim, TagSet};
use sky_units::Dimension;

use crate::emit::{number_type, shaped_type, Res};
use crate::{derivative_name_for, AdError, Kind, Task};

/// The most passes (entries of the parameter) this fallback unrolls.
pub(crate) const MAX_PASSES: u64 = 256;

pub(crate) fn build(_module: &Module, f: &Function, wrt: usize) -> Res<(Function, Vec<Task>)> {
    let name = f.name.clone();
    let params = f.param_values();
    let Some(&x) = params.get(wrt) else {
        return Err(AdError::ParamOutOfRange {
            func: name,
            wrt,
            params: params.len(),
        });
    };
    let x_ty = match f.ty(x) {
        Type::Number(n) if n.shape.rank() > 0 => n.clone(),
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: format!("parameter {wrt}"),
            })
        }
        _ => return Err(AdError::NonNumberParam { func: name, wrt }),
    };
    let [res_ty] = &f.results[..] else {
        return Err(AdError::BadResult {
            func: name,
            why: "a gradient needs a function with one result".into(),
        });
    };
    let f_ty = match res_ty {
        Type::Number(n) if n.shape.rank() == 0 => n.clone(),
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: "the result".into(),
            })
        }
        _ => {
            return Err(AdError::BadResult {
                func: name,
                why: "a gradient needs a result that is a scalar number".into(),
            })
        }
    };
    let unsupported = |why: String| AdError::Unsupported {
        func: f.name.clone(),
        op: "gradient".into(),
        why,
    };
    let mut sizes = Vec::new();
    for d in x_ty.shape.dims() {
        match d {
            ShapeDim::Known(k) => sizes.push(*k),
            ShapeDim::Var(v) => {
                return Err(unsupported(format!(
                    "one pass per entry needs the size of the parameter, and {v} is not known"
                )))
            }
        }
    }
    let total: u64 = sizes.iter().product();
    if total > MAX_PASSES {
        return Err(unsupported(format!(
            "one pass per entry would take {total} passes (at most {MAX_PASSES})"
        )));
    }
    let dim = f_ty
        .dim
        .checked_div(x_ty.dim)
        .map_err(|e| AdError::Dimension {
            func: f.name.clone(),
            why: e.to_string(),
        })?;
    let scalar = number_type(dim, f_ty.repr);
    let grad_ty = shaped_type(dim, x_ty.shape.clone(), x_ty.repr);
    let dimless = number_type(Dimension::DIMENSIONLESS, x_ty.repr);
    let unit = |shape: Shape| {
        Type::Number(sky_ir::NumberType {
            dim: Dimension::DIMENSIONLESS,
            shape,
            tags: TagSet::EMPTY,
            repr: x_ty.repr,
        })
    };

    let mut b = FunctionBuilder::new(
        derivative_name_for(f, wrt, Kind::Gradient),
        f.params.clone(),
        vec![res_ty.clone(), grad_ty.clone()],
        f.span,
    );
    b.set_effects(f.effects);
    let span = f.span;
    let args = b.params();
    let jvp = derivative_name_for(f, wrt, Kind::Jvp);
    let konst = |b: &mut FunctionBuilder, v: u64| -> ValueId {
        b.op1(
            OpKind::ConstFloat {
                bits: (v as f64).to_bits(),
            },
            vec![],
            dimless.clone(),
            span,
        )
    };
    // The identity matrices the unit directions are rows and columns of.
    let ident = |b: &mut FunctionBuilder, n: u64| -> ValueId {
        let size = konst(b, n);
        b.op1(
            OpKind::Identity,
            vec![size],
            unit(Shape::matrix(n, n)),
            span,
        )
    };
    let (id_rows, id_cols) = match sizes[..] {
        [n] => (ident(&mut b, n), None),
        [n, m] => (ident(&mut b, n), Some(ident(&mut b, m))),
        _ => return Err(unsupported("only vectors and matrices".into())),
    };

    let mut value = None;
    let mut partials = Vec::new();
    for e in 0..total {
        let direction = match (sizes[..].to_vec().as_slice(), id_cols) {
            ([n], _) => {
                let pos = konst(&mut b, e);
                b.op1(
                    OpKind::Index,
                    vec![id_rows, pos],
                    unit(Shape::vector(*n)),
                    span,
                )
            }
            ([n, m], Some(id_cols)) => {
                let (i, j) = (e / m, e % m);
                let (i0, i1) = (konst(&mut b, i), konst(&mut b, i + 1));
                let (j0, j1) = (konst(&mut b, j), konst(&mut b, j + 1));
                let full = SliceAxis::Range {
                    start: false,
                    stop: false,
                };
                let bounded = SliceAxis::Range {
                    start: true,
                    stop: true,
                };
                // Column i of the identity (n×1) times row j (1×m).
                let col = b.op1(
                    OpKind::Slice {
                        axes: vec![full, bounded],
                    },
                    vec![id_rows, i0, i1],
                    unit(Shape::matrix(*n, 1)),
                    span,
                );
                let row = b.op1(
                    OpKind::Slice {
                        axes: vec![bounded, full],
                    },
                    vec![id_cols, j0, j1],
                    unit(Shape::matrix(1, *m)),
                    span,
                );
                b.op1(
                    OpKind::MatMul,
                    vec![col, row],
                    unit(Shape::matrix(*n, *m)),
                    span,
                )
            }
            _ => unreachable!("one or two axes"),
        };
        let mut operands = args.clone();
        operands.push(direction);
        let (_, res) = b.op(
            OpKind::Call {
                callee: jvp.clone(),
            },
            operands,
            vec![res_ty.clone(), scalar.clone()],
            span,
        );
        value.get_or_insert(res[0]);
        partials.push(res[1]);
    }
    let cols = match sizes[..] {
        [_, m] => Some(m),
        _ => None,
    };
    let gradient = b.op1(OpKind::ArrayLiteral { cols }, partials, grad_ty, span);
    let value = value.expect("an array has at least one entry");
    b.op(OpKind::Return, vec![value, gradient], vec![], span);
    let needs = vec![Task {
        func: f.name.clone(),
        wrt,
        kind: Kind::Jvp,
    }];
    Ok((b.finish(), needs))
}
