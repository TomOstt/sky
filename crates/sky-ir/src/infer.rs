//! Inference helpers: the type an operation's result has, given the types of
//! its operands.
//!
//! Lowering types every value from the checker's answers, and leaves
//! [`Type::Unknown`] where the checker had none (an unannotated parameter,
//! and everything computed from one). A pass that learns more later (a
//! specialization that knows what a parameter is) has to type the values
//! again, and it must do it with the rules the verifier enforces: these are
//! them, for the operations whose result type follows from the operand types
//! alone.
//!
//! [`result_type`] answers `None` for "cannot say": an operand of unknown
//! type, an operation not listed, or operands the operation's rules reject
//! (a dimension mismatch, a shape that does not fit). A caller keeps the
//! result `Unknown` then, which the verifier always accepts, and the
//! operation stays a run-time check. The results carry no tags (a pass never
//! claims a property it did not prove) and are [`Repr::Unchosen`]; operands
//! of a chosen representation give `None`.
//!
//! [`join`] is the type of a value that is one of two: what a merge (`if`
//! arms, a loop's carried value) has.

use sky_types::{Shape, ShapeDim, TagSet};
use sky_units::Dimension;

use crate::ir::{NumberType, Repr, Type};
use crate::op::{CmpOp, OpKind};

/// A number of dimension `dim` and shape `shape`, no tags, representation
/// unchosen.
pub fn number(dim: Dimension, shape: Shape) -> Type {
    Type::Number(NumberType {
        dim,
        shape,
        tags: TagSet::EMPTY,
        repr: Repr::Unchosen,
    })
}

/// The number type of `t`, if it is one whose representation is unchosen
/// (the only ones these rules speak of).
fn plain(t: &Type) -> Option<&NumberType> {
    match t {
        Type::Number(n) if n.repr == Repr::Unchosen => Some(n),
        _ => None,
    }
}

fn scalar(n: &NumberType) -> bool {
    n.shape.rank() == 0
}

/// The type of a value that is either a `a` or a `b`: equal types stay,
/// numbers of one dimension, shape and representation keep the tags both
/// have, and anything else is [`Type::Unknown`].
pub fn join(a: &Type, b: &Type) -> Type {
    match (a, b) {
        (Type::Number(x), Type::Number(y))
            if x.dim == y.dim && x.shape == y.shape && x.repr == y.repr =>
        {
            Type::Number(NumberType {
                tags: x.tags.join(y.tags),
                ..x.clone()
            })
        }
        _ if a == b => a.clone(),
        _ => Type::Unknown,
    }
}

/// The type of the (first) result of `kind` applied to operands of types
/// `operands`, if it follows from them (see the module documentation).
pub fn result_type(kind: &OpKind, operands: &[&Type]) -> Option<Type> {
    match (kind, operands) {
        (OpKind::Add | OpKind::Sub, [a, b]) => {
            let (a, b) = (plain(a)?, plain(b)?);
            (a.dim == b.dim && a.shape == b.shape).then(|| number(a.dim, a.shape.clone()))
        }
        (OpKind::Mul, [a, b]) => {
            let (a, b) = (plain(a)?, plain(b)?);
            let dim = a.dim.checked_mul(b.dim).ok()?;
            let shape = match (scalar(a), scalar(b)) {
                (true, _) => b.shape.clone(),
                (false, true) => a.shape.clone(),
                _ => return None,
            };
            Some(number(dim, shape))
        }
        (OpKind::Div, [a, b]) => {
            let (a, b) = (plain(a)?, plain(b)?);
            if !scalar(b) {
                return None;
            }
            Some(number(a.dim.checked_div(b.dim).ok()?, a.shape.clone()))
        }
        (OpKind::Neg, [a]) => {
            let a = plain(a)?;
            Some(number(a.dim, a.shape.clone()))
        }
        (OpKind::PowConst { exponent }, [a]) => {
            let a = plain(a)?;
            Some(number(a.dim.pow_rat(*exponent).ok()?, a.shape.clone()))
        }
        (OpKind::Pow, [a, e]) => {
            let (a, e) = (plain(a)?, plain(e)?);
            (scalar(a) && scalar(e) && a.dim.is_dimensionless() && e.dim.is_dimensionless())
                .then(|| number(Dimension::DIMENSIONLESS, Shape::scalar()))
        }
        (OpKind::Math(f), args) if args.len() == f.arity() => {
            let mut dims = Vec::with_capacity(args.len());
            for a in args {
                let a = plain(a)?;
                if !scalar(a) {
                    return None;
                }
                dims.push(a.dim);
            }
            Some(number(f.result_dim(&dims).ok()?, Shape::scalar()))
        }
        (OpKind::Norm, [a]) => {
            let a = plain(a)?;
            Some(number(a.dim, Shape::scalar()))
        }
        (OpKind::Inverse, [a]) => {
            let a = plain(a)?;
            scalar(a)
                .then(|| a.dim.inverse().ok())
                .flatten()
                .map(|d| number(d, Shape::scalar()))
        }
        (OpKind::Len, [a]) => {
            let a = plain(a)?;
            (!scalar(a)).then(|| number(Dimension::DIMENSIONLESS, Shape::scalar()))
        }
        (OpKind::Cmp(c), [a, b]) => {
            let ok = match (a, b) {
                (Type::Number(x), Type::Number(y)) => {
                    x.repr == Repr::Unchosen && scalar(x) && scalar(y) && x.dim == y.dim
                }
                (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => {
                    matches!(c, CmpOp::Eq | CmpOp::Ne)
                }
                _ => false,
            };
            ok.then_some(Type::Bool)
        }
        (OpKind::Not, [Type::Bool]) => Some(Type::Bool),
        (OpKind::ExpectBool { .. }, [Type::Bool]) => Some(Type::Bool),
        (OpKind::ExpectNumber { .. }, [a @ Type::Number(_)]) => plain(a).map(|_| (*a).clone()),
        (OpKind::ArrayLiteral { cols }, entries) if !entries.is_empty() => {
            let n = entries.len() as u64;
            let shape = match cols {
                None => Shape::vector(n),
                Some(c) if *c > 0 && n % c == 0 => Shape::matrix(n / c, *c),
                Some(_) => return None,
            };
            let mut dim = None;
            for e in entries {
                let e = plain(e)?;
                if !scalar(e) || dim.is_some_and(|d| d != e.dim) {
                    return None;
                }
                dim = Some(e.dim);
            }
            Some(number(dim?, shape))
        }
        (OpKind::Index, [base, positions @ ..]) if !positions.is_empty() => {
            let a = plain(base)?;
            for p in positions {
                let p = plain(p)?;
                if !scalar(p) || !p.dim.is_dimensionless() {
                    return None;
                }
            }
            let dims = a.shape.dims();
            if positions.len() > dims.len() {
                return None;
            }
            let rest: Vec<ShapeDim> = dims[positions.len()..].to_vec();
            Some(number(a.dim, Shape::new(rest)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(dim: Dimension) -> Type {
        number(dim, Shape::scalar())
    }

    #[test]
    fn scalar_arithmetic_follows_the_dimension_rules() {
        let (l, t, one) = (
            s(Dimension::LENGTH),
            s(Dimension::TIME),
            s(Dimension::DIMENSIONLESS),
        );
        assert_eq!(result_type(&OpKind::Add, &[&l, &l]), Some(l.clone()));
        assert_eq!(result_type(&OpKind::Add, &[&l, &t]), None);
        assert_eq!(
            result_type(&OpKind::Div, &[&l, &t]),
            Some(s(Dimension::LENGTH.checked_div(Dimension::TIME).unwrap()))
        );
        assert_eq!(result_type(&OpKind::Sub, &[&one, &Type::Unknown]), None);
        assert_eq!(
            result_type(&OpKind::Cmp(CmpOp::Lt), &[&l, &l]),
            Some(Type::Bool)
        );
        assert_eq!(result_type(&OpKind::Cmp(CmpOp::Lt), &[&l, &t]), None);
        assert_eq!(
            result_type(&OpKind::Cmp(CmpOp::Lt), &[&Type::Bool, &Type::Bool]),
            None
        );
        assert_eq!(
            result_type(&OpKind::Cmp(CmpOp::Eq), &[&Type::Bool, &Type::Bool]),
            Some(Type::Bool)
        );
    }

    #[test]
    fn arrays_keep_their_shape() {
        let v = number(Dimension::LENGTH, Shape::vector(3));
        let one = s(Dimension::DIMENSIONLESS);
        assert_eq!(
            result_type(&OpKind::Index, &[&v, &one]),
            Some(s(Dimension::LENGTH))
        );
        assert_eq!(
            result_type(&OpKind::Len, &[&v]),
            Some(s(Dimension::DIMENSIONLESS))
        );
        assert_eq!(result_type(&OpKind::Len, &[&one]), None);
        assert_eq!(
            result_type(&OpKind::ArrayLiteral { cols: None }, &[&one, &one]),
            Some(number(Dimension::DIMENSIONLESS, Shape::vector(2)))
        );
    }

    #[test]
    fn join_keeps_only_what_both_have() {
        let (l, t) = (s(Dimension::LENGTH), s(Dimension::TIME));
        assert_eq!(join(&l, &l), l);
        assert_eq!(join(&l, &t), Type::Unknown);
        assert_eq!(join(&Type::Bool, &Type::Bool), Type::Bool);
        assert_eq!(join(&Type::Bool, &Type::Nothing), Type::Unknown);
    }
}
