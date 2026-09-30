//! `x^n` for a constant integer `n` as a chain of multiplications.
//!
//! A `pow.const` with an integer exponent `2 <= n <= 14` becomes the
//! multiplications the interpreter itself performs for it.
//!
//! # What the interpreter does today
//!
//! `sky_interp::arith::pow_scalar` evaluates `b ^ n` with `f64::powi` for an
//! integer exponent of size up to `POWI_MAX`. `powi` is LLVM's `powi`
//! intrinsic, which is `compiler_builtins`' `__powidf2`: right-to-left
//! binary exponentiation,
//!
//! ```text
//! mul = 1.0
//! loop { if n & 1 { mul *= a }; n >>= 1; if n == 0 { break }; a *= a }
//! ```
//!
//! (for a negative exponent, the reciprocal of the result for `|n|`).
//! Multiplying by the initial `1.0` is exact, so the first `mul *= a` is a
//! plain copy. `x^5` is `x · (x²)²`, three multiplications, with these
//! exact intermediate roundings.
//!
//! # The rewrite
//!
//! This pass emits exactly those `mul` operations, in that order and
//! association, so **every bit of the result is the same** as `powi`'s (the
//! test `powi_chains_are_bit_identical` compares them on random inputs for
//! every exponent in range). It is also **minimal**: the binary method uses
//! the fewest multiplications for every `n` up to 14 (the first exponent it
//! is not optimal for is 15, which needs 5 multiplications where the binary
//! method needs 6), so for `n <= 14` "the minimal multiplication chain" and
//! "what `powi` does" are the same chain. Exponents outside `2..=14` are left
//! alone: for `n >= 15` a shorter chain would round differently (so it needs
//! the spec's permission and is not done silently), and a negative
//! exponent is `1 / x^|n|`, which today turns an intermediate overflow into
//! `0` rather than an error, so expanding it into multiplications and a
//! division could change a result.
//!
//! Errors are the same too: a positive power that overflows anywhere in the
//! chain overflows at the end (magnitudes only grow when `|x| > 1`, and shrink
//! or underflow harmlessly otherwise), and every multiplication carries the
//! `pow`'s span, so the error is `arithmetic overflow` at the same place.
//!
//! The point is not the interpreter (one `powi` call is fast there) but what
//! the multiplications feed: they fuse into elementwise kernels
//! (`pow.const` does not), give the e-graph and common-subexpression
//! elimination shared squares, and compile to inline multiplies.

use sky_ir::{
    Function, NumberType, OpData, OpId, OpKind, Repr, Type, ValueData, ValueDef, ValueId,
};
use sky_units::Rational;

use crate::rewrite::Rewrite;
use crate::util::{in_fused_body, reachable_ops};

pub const POWER_RULE: &str = "POW-CHAIN-001";

/// The exponents expanded: `powi`'s binary chain is minimal for these.
pub const MIN_EXPONENT: i64 = 2;
pub const MAX_EXPONENT: i64 = 14;

/// The multiplications of `powi`'s chain for `n`, as `(left, right)` pairs of
/// exponents multiplied, in order: `x^n` is the last product. For `n = 5`:
/// `[(1, 1), (2, 2), (1, 4)]`, that is `x², x⁴, x·x⁴`.
pub fn chain(n: u32) -> Vec<(u32, u32)> {
    let (mut a, mut mul) = (1u32, None::<u32>);
    let mut pow = n;
    let mut steps = Vec::new();
    loop {
        if pow & 1 == 1 {
            mul = Some(match mul {
                None => a,
                Some(m) => {
                    steps.push((m, a));
                    m + a
                }
            });
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        steps.push((a, a));
        a *= 2;
    }
    steps
}

/// Applies the rewrite everywhere it holds.
pub fn powers(m: &mut sky_ir::Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        for op in reachable_ops(f) {
            if !f.block(f.op(op).parent).ops.contains(&op) || in_fused_body(f, op) {
                continue;
            }
            if let Some(rw) = expand(f, op) {
                out.push(rw);
            }
        }
    }
    out
}

fn integer_exponent(k: &OpKind) -> Option<u32> {
    let OpKind::PowConst { exponent } = k else {
        return None;
    };
    (exponent.denom() == 1)
        .then(|| exponent.numer())
        .filter(|n| (MIN_EXPONENT..=MAX_EXPONENT).contains(n))
        .and_then(|n| u32::try_from(n).ok())
}

fn expand(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    let n = integer_exponent(&op.kind)?;
    let &[x] = &op.operands[..] else {
        return None;
    };
    let &[result] = &op.results[..] else {
        return None;
    };
    // Only a scalar number: an intermediate's type is the base's dimension
    // to a power.
    let Type::Number(base) = f.ty(x).clone() else {
        return None;
    };
    if base.shape.rank() != 0 {
        return None;
    }
    let steps = chain(n);
    let operand_span = op.operand_spans.first().copied();
    let mut value_of: Vec<(u32, ValueId)> = vec![(1, x)];
    let lookup =
        |vals: &[(u32, ValueId)], e: u32| vals.iter().rev().find(|(k, _)| *k == e).map(|(_, v)| *v);
    let mut new_ops: Vec<OpId> = Vec::new();
    let last = steps.len() - 1;
    for (i, &(l, r)) in steps.iter().enumerate() {
        let (lv, rv) = (lookup(&value_of, l)?, lookup(&value_of, r)?);
        let e = l + r;
        let dim = base.dim.pow_rat(Rational::integer(i64::from(e))).ok()?;
        let value = if i == last {
            // The last multiplication defines the original result.
            result
        } else {
            let v = ValueId(f.values.len() as u32);
            f.values.push(ValueData {
                ty: Type::Number(NumberType {
                    dim,
                    shape: base.shape.clone(),
                    tags: sky_types::TagSet::EMPTY,
                    repr: Repr::Unchosen,
                }),
                def: ValueDef::OpResult {
                    op: OpId(f.ops.len() as u32),
                    index: 0,
                },
                name: None,
            });
            v
        };
        let oid = OpId(f.ops.len() as u32);
        f.ops.push(OpData {
            kind: OpKind::Mul,
            operands: vec![lv, rv],
            results: vec![value],
            regions: Vec::new(),
            span: op.span,
            operand_spans: if op.operand_spans.is_empty() {
                Vec::new()
            } else {
                vec![operand_span.unwrap_or(op.span); 2]
            },
            parent: op.parent,
        });
        if i == last {
            f.values[result.index()].def = ValueDef::OpResult { op: oid, index: 0 };
        }
        new_ops.push(oid);
        value_of.push((e, value));
    }
    // The chain goes where the `pow.const` was.
    let block = op.parent;
    let ops = std::mem::take(&mut f.blocks[block.index()].ops);
    f.blocks[block.index()].ops = ops
        .into_iter()
        .flat_map(|o| if o == id { new_ops.clone() } else { vec![o] })
        .collect();
    let xn = match &f.value(x).name {
        Some(n) => n.clone(),
        None => "x".to_string(),
    };
    let described: Vec<String> = steps
        .iter()
        .map(|&(l, r)| {
            let show = |e: u32| {
                if e == 1 {
                    xn.clone()
                } else {
                    format!("{xn}^{e}")
                }
            };
            format!("{}^{} = {} · {}", xn, l + r, show(l), show(r))
        })
        .collect();
    let product_text = steps
        .last()
        .map(|&(l, r)| {
            let show = |e: u32| {
                if e == 1 {
                    xn.clone()
                } else {
                    format!("{xn}^{e}")
                }
            };
            format!("{} · {}", show(l), show(r))
        })
        .unwrap_or_default();
    Some(Rewrite {
        pass: "powers",
        rule: POWER_RULE,
        span: op.span,
        before: format!("{xn}^{n}"),
        after: product_text,
        because: vec![
            format!(
                "the exponent {n} is a constant integer: {}",
                described.join(", ")
            ),
            format!(
                "{} multiplication{}, the fewest for {n}: no shorter chain exists for an exponent up to 14",
                steps.len(),
                if steps.len() == 1 { "" } else { "s" }
            ),
            "exact: these are the multiplications the interpreter's `powi` performs, in the same order, so every bit and the overflow error are the same".to_string(),
        ],
    })
}

#[cfg(test)]
mod chain_tests {
    use super::chain;

    #[test]
    fn chains_are_the_binary_method() {
        assert_eq!(chain(2), [(1, 1)]);
        assert_eq!(chain(3), [(1, 1), (1, 2)]);
        assert_eq!(chain(4), [(1, 1), (2, 2)]);
        assert_eq!(chain(5), [(1, 1), (2, 2), (1, 4)]);
        assert_eq!(chain(6), [(1, 1), (2, 2), (2, 4)]);
        assert_eq!(chain(7), [(1, 1), (1, 2), (2, 2), (3, 4)]);
    }

    /// The binary method's count of multiplications is optimal for every
    /// exponent the pass expands (checked against the exact minimum: the
    /// shortest addition chain lengths l(n) for n = 2..=14).
    #[test]
    fn the_binary_method_is_minimal_up_to_fourteen() {
        // l(n): 2:1 3:2 4:2 5:3 6:3 7:4 8:3 9:4 10:4 11:5 12:4 13:5 14:5
        let shortest = [1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5];
        for (i, want) in shortest.iter().enumerate() {
            let n = i as u32 + 2;
            assert_eq!(chain(n).len(), *want, "x^{n}");
        }
        // And 15 is where it stops being minimal: 6 against 5.
        assert_eq!(chain(15).len(), 6);
    }
}
