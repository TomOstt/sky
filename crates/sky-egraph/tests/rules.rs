//! One test per rule on hand-built IR, plus the protections around them:
//! effects, failures, spans and types.

mod common;

use common::*;
use sky_egraph::{rewrite_function, rule_table, RewriteConfig, RewriteContract};
use sky_interp::Value;
use sky_ir::{Function, MathFn, OpKind, Type};
use sky_types::{Tag, TagSet};
use sky_units::Dimension;

fn fast() -> RewriteConfig {
    RewriteConfig::with_contract(RewriteContract::Fast)
}

fn repro() -> RewriteConfig {
    RewriteConfig::default()
}

/// The rules that fired, in order.
fn fired(f: &mut Function, cfg: &RewriteConfig) -> Vec<&'static str> {
    rewrite_function(f, cfg).iter().map(|r| r.rule).collect()
}

/// The value a function returns for `args`, which must succeed.
fn value(f: &Function, args: &[f64]) -> f64 {
    match run(f, args).0.expect("runs").remove(0) {
        Value::Number { value, .. } => value,
        other => panic!("not a number: {other:?}"),
    }
}

/// `f(x) = build(x)`, returning one value.
fn unary(build: impl FnOnce(&mut T, sky_ir::ValueId) -> sky_ir::ValueId) -> Function {
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let r = build(&mut t, x);
    t.ret(&[r])
}

#[test]
fn every_rule_has_a_unique_id_and_a_justification() {
    let table = rule_table();
    let mut ids: Vec<_> = table.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), table.len());
    for r in &table {
        assert!(!r.justification.is_empty(), "{}", r.id);
        assert!(r.id.contains('-'), "{}", r.id);
    }
    for id in [
        "TRIG-001", "ALG-001", "ALG-002", "ALG-003", "ALG-004", "ALG-005", "ALG-010", "EXP-001",
        "EXP-002", "LA-001", "LA-002", "LA-003",
    ] {
        assert!(ids.contains(&id), "missing {id}");
    }
}

// ---- exact rules ----

#[test]
fn alg_001_mul_one() {
    for one_first in [false, true] {
        let mut f = unary(|t, x| {
            let one = t.rat(1, 1);
            if one_first {
                t.bin(OpKind::Mul, one, x)
            } else {
                t.bin(OpKind::Mul, x, one)
            }
        });
        let before = f.clone();
        let rw = rewrite_function(&mut f, &repro());
        assert_eq!(rw.len(), 1);
        assert_eq!(rw[0].rule, "ALG-001");
        assert_eq!(rw[0].pass, "egraph");
        assert_eq!(rw[0].before, "%0 * 1");
        assert_eq!(rw[0].after, "%0");
        assert_eq!(kinds(&f), ["return"]);
        for x in [0.0, -0.0, 3.5, -1e300] {
            assert!(same_outcome(&run(&before, &[x]), &run(&f, &[x])));
        }
    }
}

#[test]
fn alg_001_needs_a_dimensionless_one() {
    // x * (1 m) changes the dimension, so it is not x.
    let mut t = T::new(vec![num()], vec![dim_len()]);
    let x = t.params()[0];
    let one_m = t.op(
        OpKind::ConstRational(sky_units::Rational::ONE),
        &[],
        dim_len(),
    );
    let r = t.op(OpKind::Mul, &[x, one_m], dim_len());
    let mut f = t.ret(&[r]);
    assert!(fired(&mut f, &fast()).is_empty());
}

#[test]
fn alg_006_div_one_and_alg_007_neg_neg() {
    let mut f = unary(|t, x| {
        let one = t.rat(1, 1);
        let d = t.bin(OpKind::Div, x, one);
        let n = t.un(OpKind::Neg, d);
        t.un(OpKind::Neg, n)
    });
    let rules = fired(&mut f, &repro());
    assert!(
        rules.contains(&"ALG-006") && rules.contains(&"ALG-007"),
        "{rules:?}"
    );
    assert_eq!(kinds(&f), ["return"]);
}

/// An array's negation is `0 - x` per entry, so `-(-x)` turns `-0` into `+0`
/// and is not `x`: ALG-007 is for numbers (and for arrays proven not `-0`).
#[test]
fn alg_007_does_not_apply_to_arrays_that_may_hold_minus_zero() {
    let v = Type::Number(sky_ir::NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape: sky_types::Shape::vector(3u64),
        tags: TagSet::EMPTY,
        repr: sky_ir::Repr::Unchosen,
    });
    let mk = || {
        let mut t = T::new(vec![v.clone()], vec![v.clone()]);
        let p = t.params()[0];
        let n = t.op(OpKind::Neg, &[p], v.clone());
        let r = t.op(OpKind::Neg, &[n], v.clone());
        t.ret(&[r])
    };
    let input = vec![Value::array(
        vec![3],
        vec![-0.0, 0.0, 2.0],
        Dimension::DIMENSIONLESS,
    )];
    let run_array = |f: &Function| {
        let m = module(f);
        sky_ast::with_deep_stack(|| {
            sky_ir_interp::call(&m, "f", input.clone(), &mut Vec::new()).expect("runs")
        })
    };
    let original = mk();
    for cfg in [repro(), fast()] {
        let mut g = original.clone();
        assert!(fired(&mut g, &cfg).is_empty());
        assert_eq!(kinds(&g), ["neg", "neg", "return"]);
    }
    // `-0.0` comes back as `+0.0`, which `x` itself would not be.
    let Value::Array { data, .. } = &run_array(&original)[0] else {
        panic!("not an array")
    };
    assert_eq!(data[0].to_bits(), 0.0f64.to_bits());

    // The same program on numbers is still `x`.
    let mut s = unary(|t, x| {
        let n = t.un(OpKind::Neg, x);
        t.un(OpKind::Neg, n)
    });
    assert_eq!(fired(&mut s, &repro()), ["ALG-007"]);
    assert_eq!(value(&s, &[-0.0]).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn alg_008_sub_zero_only_for_positive_zero() {
    let mut f = unary(|t, x| {
        let z = t.rat(0, 1);
        t.bin(OpKind::Sub, x, z)
    });
    assert_eq!(fired(&mut f, &repro()), ["ALG-008"]);
    // x - (-0.0) is x + 0.0, which is not x for x = -0.0.
    let mut g = unary(|t, x| {
        let z = t.float(-0.0);
        t.bin(OpKind::Sub, x, z)
    });
    assert!(fired(&mut g, &fast()).is_empty());
}

#[test]
fn alg_009_pow_one() {
    let mut f = unary(|t, x| t.pow(x, 1, 1));
    assert_eq!(fired(&mut f, &repro()), ["ALG-009"]);
    assert_eq!(kinds(&f), ["return"]);
}

// ---- conditional rules ----

#[test]
fn alg_002_add_zero_needs_a_proof_about_negative_zero() {
    // x + 0 for an unknown x is NOT rewritten: (-0.0) + 0.0 is +0.0.
    let mut f = unary(|t, x| {
        let z = t.rat(0, 1);
        t.bin(OpKind::Add, x, z)
    });
    assert!(fired(&mut f, &fast()).is_empty());
    assert_eq!(kinds(&f), ["const.rational", "add", "return"]);

    // abs(x) + 0: abs is never -0.0, so it is exact.
    let mut g = unary(|t, x| {
        let a = t.math(MathFn::Abs, &[x]);
        let z = t.rat(0, 1);
        t.bin(OpKind::Add, a, z)
    });
    let before = g.clone();
    assert_eq!(fired(&mut g, &repro()), ["ALG-002"]);
    assert_eq!(kinds(&g), ["math.abs", "return"]);
    for x in [0.0, -0.0, -2.0] {
        assert!(same_outcome(&run(&before, &[x]), &run(&g, &[x])));
    }

    // x + (-0.0) is exact for every x.
    let mut h = unary(|t, x| {
        let z = t.float(-0.0);
        t.bin(OpKind::Add, x, z)
    });
    let before = h.clone();
    assert_eq!(fired(&mut h, &repro()), ["ALG-002"]);
    for x in [0.0, -0.0, 7.0] {
        assert!(same_outcome(&run(&before, &[x]), &run(&h, &[x])));
    }
}

#[test]
fn alg_003_sub_self_only_if_the_operand_cannot_fail() {
    // x - x with x a parameter: exactly +0.
    let mut f = unary(|t, x| t.bin(OpKind::Sub, x, x));
    let before = f.clone();
    assert_eq!(fired(&mut f, &repro()), ["ALG-003"]);
    assert_eq!(kinds(&f), ["const.float", "return"]);
    for x in [0.0, -0.0, 5.0, -1e300] {
        assert!(same_outcome(&run(&before, &[x]), &run(&f, &[x])));
    }

    // (1/x) - (1/x) fails for x = 0: dropping it would hide the failure.
    let build = |t: &mut T, x| {
        let one = t.rat(1, 1);
        let a = t.bin(OpKind::Div, one, x);
        let one2 = t.rat(1, 1);
        let b = t.bin(OpKind::Div, one2, x);
        t.bin(OpKind::Sub, a, b)
    };
    let mut g = unary(build);
    let before = g.clone();
    assert!(fired(&mut g, &repro()).is_empty());
    assert_eq!(g, before);
    assert!(run(&g, &[0.0]).0.is_err());
    // Under the fast contract the failure may go.
    let mut h = unary(build);
    assert_eq!(fired(&mut h, &fast()), ["ALG-003"]);
    assert_eq!(value(&h, &[0.0]), 0.0);
}

#[test]
fn alg_003_keeps_the_dimension() {
    let mut t = T::new(vec![dim_len()], vec![dim_len()]);
    let x = t.params()[0];
    let r = t.op(OpKind::Sub, &[x, x], dim_len());
    let mut f = t.ret(&[r]);
    assert_eq!(fired(&mut f, &repro()), ["ALG-003"]);
    let ret = f.op(*f.block(f.entry()).ops.last().unwrap()).operands[0];
    assert_eq!(f.ty(ret).as_number().unwrap().dim, Dimension::LENGTH);
}

// ---- real-only rules ----

#[test]
fn alg_004_pow_pow_is_real_only_and_keeps_dimensions() {
    let build = |t: &mut T, x| {
        let a = t.op(
            OpKind::PowConst {
                exponent: sky_units::Rational::new(2, 1),
            },
            &[x],
            Type::scalar(Dimension::LENGTH.pow_int(2).unwrap()),
        );
        t.op(
            OpKind::PowConst {
                exponent: sky_units::Rational::new(3, 1),
            },
            &[a],
            Type::scalar(Dimension::LENGTH.pow_int(6).unwrap()),
        )
    };
    let mk = || {
        let mut t = T::new(
            vec![dim_len()],
            vec![Type::scalar(Dimension::LENGTH.pow_int(6).unwrap())],
        );
        let x = t.params()[0];
        let r = build(&mut t, x);
        t.ret(&[r])
    };
    let mut f = mk();
    assert!(
        fired(&mut f, &repro()).is_empty(),
        "real-only: off by default"
    );
    let mut g = mk();
    assert_eq!(fired(&mut g, &fast()), ["ALG-004"]);
    assert_eq!(kinds(&g), ["pow.const", "return"]);
    let ret = g.op(*g.block(g.entry()).ops.last().unwrap()).operands[0];
    assert_eq!(
        g.ty(ret).as_number().unwrap().dim,
        Dimension::LENGTH.pow_int(6).unwrap()
    );
    assert!((value(&g, &[1.5]) - 1.5f64.powi(6)).abs() < 1e-9);
}

#[test]
fn alg_004_needs_an_integer_outer_exponent_or_a_positive_base() {
    // (x^2)^(1/2) is |x|, not x^1: not rewritten for an unknown-sign x.
    let mut f = unary(|t, x| {
        let a = t.pow(x, 2, 1);
        t.pow(a, 1, 2)
    });
    assert!(fired(&mut f, &fast()).is_empty());
    // With x provably positive it is fine.
    let mut g = unary(|t, x| {
        let a = t.math(MathFn::Abs, &[x]);
        let one = t.rat(1, 1);
        let p = t.bin(OpKind::Add, a, one);
        let sq = t.pow(p, 2, 1);
        t.pow(sq, 1, 2)
    });
    assert!(fired(&mut g, &fast()).contains(&"ALG-004"));
}

#[test]
fn alg_005_sqrt_of_square_is_abs() {
    let mk = || {
        let mut t = T::new(vec![dim_len()], vec![dim_len()]);
        let x = t.params()[0];
        let sq = t.op(
            OpKind::PowConst {
                exponent: sky_units::Rational::new(2, 1),
            },
            &[x],
            Type::scalar(Dimension::LENGTH.pow_int(2).unwrap()),
        );
        let r = t.op(OpKind::Math(MathFn::Sqrt), &[sq], dim_len());
        t.ret(&[r])
    };
    let mut f = mk();
    assert!(fired(&mut f, &repro()).is_empty());
    let mut g = mk();
    assert_eq!(fired(&mut g, &fast()), ["ALG-005"]);
    assert_eq!(kinds(&g), ["math.abs", "return"]);
    assert_eq!(value(&g, &[-3.0]), 3.0);
}

#[test]
fn exp_002_log_of_exp() {
    let mut f = unary(|t, x| {
        let e = t.math(MathFn::Exp, &[x]);
        t.math(MathFn::Log, &[e])
    });
    assert!(fired(&mut f, &repro()).is_empty());
    assert_eq!(kinds(&f), ["math.exp", "math.log", "return"]);
    let mut g = unary(|t, x| {
        let e = t.math(MathFn::Exp, &[x]);
        t.math(MathFn::Log, &[e])
    });
    assert_eq!(fired(&mut g, &fast()), ["EXP-002"]);
    assert_eq!(kinds(&g), ["return"]);
}

#[test]
fn exp_001_exp_of_log_needs_a_proof_of_positivity() {
    let mut f = unary(|t, x| {
        let l = t.math(MathFn::Log, &[x]);
        t.math(MathFn::Exp, &[l])
    });
    // x may be <= 0, where log fails: not rewritten even under fast.
    assert!(fired(&mut f, &fast()).is_empty());
    let mut g = unary(|t, x| {
        let a = t.math(MathFn::Abs, &[x]);
        let one = t.rat(1, 1);
        let p = t.bin(OpKind::Add, a, one);
        let l = t.math(MathFn::Log, &[p]);
        t.math(MathFn::Exp, &[l])
    });
    assert_eq!(fired(&mut g, &fast()), ["EXP-001"]);
    assert!(!kinds(&g).contains(&"math.exp"));
    assert!(!kinds(&g).contains(&"math.log"));
    assert!((value(&g, &[-2.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn trig_001_pythagorean_identity_collapses_end_to_end() {
    let mk = |dup_sq: bool| {
        unary(|t, x| {
            let s = t.math(MathFn::Sin, &[x]);
            let c = t.math(MathFn::Cos, &[x]);
            let (s2, c2) = if dup_sq {
                (t.bin(OpKind::Mul, s, s), t.bin(OpKind::Mul, c, c))
            } else {
                (t.pow(s, 2, 1), t.pow(c, 2, 1))
            };
            t.bin(OpKind::Add, s2, c2)
        })
    };
    for dup in [false, true] {
        // Off by default: identical only over the reals.
        let mut f = mk(dup);
        assert!(fired(&mut f, &repro()).is_empty());
        let mut g = mk(dup);
        let rw = rewrite_function(&mut g, &fast());
        assert_eq!(rw.len(), 1);
        assert_eq!(rw[0].rule, "TRIG-001");
        assert_eq!(rw[0].after, "1");
        assert!(rw[0].before.starts_with("sin(%0)"), "{}", rw[0].before);
        assert!(rw[0].because.contains("Pythagorean"));
        // The whole thing is one constant now.
        assert_eq!(kinds(&g), ["const.float", "return"]);
        for x in [0.0, 0.3, -7.0, 1e10] {
            assert_eq!(value(&g, &[x]), 1.0);
        }
    }
}

#[test]
fn trig_001_spans_are_joined() {
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let s = t.math(MathFn::Sin, &[x]);
    let c = t.math(MathFn::Cos, &[x]);
    let s2 = t.pow(s, 2, 1);
    let c2 = t.pow(c, 2, 1);
    let sum = t.bin(OpKind::Add, s2, c2);
    let mut f = t.ret(&[sum]);
    let add_span = f
        .op(f.block(f.entry()).ops[f.block(f.entry()).ops.len() - 2])
        .span;
    let rw = rewrite_function(&mut f, &fast());
    assert_eq!(rw[0].span, add_span);
    let konst = f.op(f.block(f.entry()).ops[0]);
    assert_eq!(konst.kind.mnemonic(), "const.float");
    assert_eq!(konst.span, add_span);
}

// ---- linear algebra ----

fn matrix_fn(
    tags: TagSet,
    build: impl FnOnce(&mut T, sky_ir::ValueId) -> sky_ir::ValueId,
) -> Function {
    let m = matrix(tags);
    let mut t = T::new(vec![m.clone()], vec![m.clone()]);
    let a = t.params()[0];
    let r = build(&mut t, a);
    t.ret(&[r])
}

#[test]
fn la_001_transpose_of_symmetric() {
    let tr = |t: &mut T, a| t.op(OpKind::Transpose, &[a], matrix(TagSet::EMPTY));
    let mut f = matrix_fn(TagSet::of(Tag::Symmetric), |t, a| {
        t.op(OpKind::Transpose, &[a], matrix(TagSet::of(Tag::Symmetric)))
    });
    let rw = rewrite_function(&mut f, &repro());
    assert_eq!(rw.len(), 1);
    assert_eq!(rw[0].rule, "LA-001");
    assert_eq!(kinds(&f), ["return"]);

    // PositiveDefinite implies Symmetric (tags are closed).
    let mut g = matrix_fn(TagSet::of(Tag::PositiveDefinite), |t, a| {
        t.op(
            OpKind::Transpose,
            &[a],
            matrix(TagSet::of(Tag::PositiveDefinite)),
        )
    });
    assert_eq!(fired(&mut g, &repro()), ["LA-001"]);

    // No tag, no rewrite.
    let mut h = matrix_fn(TagSet::EMPTY, tr);
    assert!(fired(&mut h, &fast()).is_empty());
    // Diagonal-free orthogonal is not symmetric.
    let mut o = matrix_fn(TagSet::of(Tag::Orthogonal), |t, a| {
        t.op(OpKind::Transpose, &[a], matrix(TagSet::of(Tag::Orthogonal)))
    });
    assert!(fired(&mut o, &fast()).is_empty());
}

#[test]
fn la_002_double_transpose() {
    let mut f = matrix_fn(TagSet::EMPTY, |t, a| {
        let x = t.op(OpKind::Transpose, &[a], matrix(TagSet::EMPTY));
        t.op(OpKind::Transpose, &[x], matrix(TagSet::EMPTY))
    });
    assert_eq!(fired(&mut f, &repro()), ["LA-002"]);
    assert_eq!(kinds(&f), ["return"]);
}

#[test]
fn la_003_double_inverse_is_real_only() {
    let mk = || {
        matrix_fn(TagSet::EMPTY, |t, a| {
            let x = t.op(OpKind::Inverse, &[a], matrix(TagSet::EMPTY));
            t.op(OpKind::Inverse, &[x], matrix(TagSet::EMPTY))
        })
    };
    let mut f = mk();
    assert!(fired(&mut f, &repro()).is_empty());
    assert_eq!(kinds(&f), ["inverse", "inverse", "return"]);
    let mut g = mk();
    assert_eq!(fired(&mut g, &fast()), ["LA-003"]);
    assert_eq!(kinds(&g), ["return"]);
}

// ---- effects, failures, structure ----

#[test]
fn print_stays_between_and_the_failure_stays_first() {
    // a = 1 / y   (fails for y = 0)
    // print(y)
    // b = a * 1   -> a
    let mut t = T::new(vec![num()], vec![num()]);
    let y = t.params()[0];
    let one = t.rat(1, 1);
    let a = t.bin(OpKind::Div, one, y);
    t.print(&[y]);
    let one2 = t.rat(1, 1);
    let b = t.bin(OpKind::Mul, a, one2);
    let mut f = t.ret(&[b]);
    let before = f.clone();
    assert_eq!(fired(&mut f, &repro()), ["ALG-001"]);
    assert_eq!(kinds(&f), ["const.rational", "div", "print", "return"]);
    for y in [0.0, 2.0, -4.0] {
        assert!(same_outcome(&run(&before, &[y]), &run(&f, &[y])), "y = {y}");
    }
    // The failure happens before the print, so there is no output.
    assert_eq!(run(&f, &[0.0]).1, "");
}

#[test]
fn nothing_moves_across_a_print() {
    // c = x * y (can overflow); print; d = c - c  -- the sub is in a
    // different segment from the mul, so c is an opaque value for it.
    let mut t = T::new(vec![num(), num()], vec![num()]);
    let [x, y] = t.params()[..] else {
        unreachable!()
    };
    let c = t.bin(OpKind::Mul, x, y);
    t.print(&[c]);
    let d = t.bin(OpKind::Sub, c, c);
    let mut f = t.ret(&[d]);
    let before = f.clone();
    assert_eq!(fired(&mut f, &repro()), ["ALG-003"]);
    assert_eq!(kinds(&f), ["mul", "print", "const.float", "return"]);
    for args in [[1e200, 1e200], [2.0, 3.0]] {
        assert!(same_outcome(&run(&before, &args), &run(&f, &args)));
    }
}

#[test]
fn a_dead_fallible_op_is_kept() {
    let mut t = T::new(vec![num()], vec![num()]);
    let y = t.params()[0];
    let one = t.rat(1, 1);
    let _dead = t.bin(OpKind::Div, one, y);
    let one2 = t.rat(1, 1);
    let r = t.bin(OpKind::Mul, y, one2);
    let mut f = t.ret(&[r]);
    let before = f.clone();
    assert_eq!(fired(&mut f, &repro()), ["ALG-001"]);
    assert!(kinds(&f).contains(&"div"));
    assert!(same_outcome(&run(&before, &[0.0]), &run(&f, &[0.0])));
    assert!(run(&f, &[0.0]).0.is_err());
}

#[test]
fn a_failing_op_keeps_its_own_span() {
    // (1/y) * 1: the division survives with the span it had, so the error
    // points where it did before.
    let mut t = T::new(vec![num()], vec![num()]);
    let y = t.params()[0];
    let one = t.rat(1, 1);
    let d = t.bin(OpKind::Div, one, y);
    let one2 = t.rat(1, 1);
    let m = t.bin(OpKind::Mul, d, one2);
    let mut f = t.ret(&[m]);
    let before = f.clone();
    fired(&mut f, &repro());
    let e0 = run(&before, &[0.0]).0.unwrap_err();
    let e1 = run(&f, &[0.0]).0.unwrap_err();
    assert_eq!(e0, e1);
    assert_eq!(e1.span, sky_diag::Span::new(20, 25));
}

#[test]
fn unknown_typed_values_are_not_rewritten() {
    // x * 1 with an Unknown-typed x may be a type error at run time.
    let mut t = T::new(vec![Type::Unknown], vec![Type::Unknown]);
    let x = t.params()[0];
    let one = t.rat(1, 1);
    let r = t.op(OpKind::Mul, &[x, one], Type::Unknown);
    let mut f = t.ret(&[r]);
    let before = f.clone();
    assert!(fired(&mut f, &fast()).is_empty());
    assert_eq!(f, before);
}

#[test]
fn regions_are_rewritten_and_uses_redirected() {
    // if c { x * 1 } else { x }
    let mut b = sky_ir::FunctionBuilder::new(
        "f",
        vec![Type::Bool, num()],
        vec![num()],
        sky_diag::Span::new(0, 1),
    );
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let s = sky_diag::Span::new(2, 3);
    let (iff, res) = b.op(OpKind::If, vec![c], vec![num()], s);
    let outer = b.insertion_block();
    let r = b.add_region(iff);
    let blk = b.add_block(r, vec![]);
    b.set_insertion_block(blk);
    let one = b.op1(
        OpKind::ConstRational(sky_units::Rational::ONE),
        vec![],
        num(),
        s,
    );
    let m = b.op1(OpKind::Mul, vec![x, one], num(), sky_diag::Span::new(4, 8));
    b.op(OpKind::Yield, vec![m], vec![], s);
    let r2 = b.add_region(iff);
    let blk2 = b.add_block(r2, vec![]);
    b.set_insertion_block(blk2);
    b.op(OpKind::Yield, vec![x], vec![], s);
    b.set_insertion_block(outer);
    b.op(OpKind::Return, res, vec![], s);
    let mut f = b.finish();
    assert_eq!(fired(&mut f, &repro()), ["ALG-001"]);
    let then_block = f.region(f.op(iff_op(&f)).regions[0]).blocks[0];
    let ops = &f.block(then_block).ops;
    assert_eq!(ops.len(), 1);
    assert_eq!(f.op(ops[0]).operands, vec![x]);
}

fn iff_op(f: &Function) -> sky_ir::OpId {
    f.block(f.entry()).ops[0]
}

#[test]
fn types_and_names_survive() {
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let one = t.rat(1, 1);
    let m = t.bin(OpKind::Mul, x, one);
    t.b.set_name(m, "total");
    let n = t.un(OpKind::Neg, m);
    let mut f = t.ret(&[n]);
    fired(&mut f, &repro());
    assert_eq!(kinds(&f), ["neg", "return"]);
    let neg = f.op(f.block(f.entry()).ops[0]);
    assert_eq!(neg.operands, vec![x]);
    assert!(f.ty(neg.results[0]).as_number().is_some());
}

#[test]
fn only_rules_restricts_the_set() {
    let mut f = unary(|t, x| {
        let one = t.rat(1, 1);
        let d = t.bin(OpKind::Div, x, one);
        t.bin(OpKind::Mul, d, one)
    });
    let cfg = RewriteConfig {
        only_rules: Some(vec!["ALG-006".into()]),
        ..repro()
    };
    let rules = fired(&mut f, &cfg);
    assert_eq!(rules, ["ALG-006"]);
}

// ---- facts from the exactness analysis (`sky_ir::exact`) ----

/// An expression builder over one parameter.
type Build = dyn Fn(&mut T, sky_ir::ValueId) -> sky_ir::ValueId;

const EDGE_INPUTS: [f64; 9] = [
    0.0, -0.0, 1.0, -2.5, 1e-310, 5e-324, 1e200, 1.7e308, -1.7e308,
];

fn same_on_edges(before: &Function, after: &Function) {
    for x in EDGE_INPUTS {
        assert!(
            same_outcome(&run(before, &[x]), &run(after, &[x])),
            "x = {x}"
        );
    }
}

#[test]
fn alg_003_fires_when_the_analysis_proves_the_operand_cannot_fail() {
    // Halving and dividing by two cannot overflow, whatever x is.
    let by_two = |t: &mut T, x| {
        let a = {
            let two = t.rat(2, 1);
            t.bin(OpKind::Div, x, two)
        };
        let b = {
            let two = t.rat(2, 1);
            t.bin(OpKind::Div, x, two)
        };
        t.bin(OpKind::Sub, a, b)
    };
    let times_half = |t: &mut T, x| {
        let a = {
            let h = t.rat(1, 2);
            t.bin(OpKind::Mul, x, h)
        };
        let b = {
            let h = t.rat(1, 2);
            t.bin(OpKind::Mul, x, h)
        };
        t.bin(OpKind::Sub, a, b)
    };
    // sqrt and log of a positive constant cannot fail either.
    let sqrt_two = |t: &mut T, _x| {
        let a = {
            let two = t.rat(2, 1);
            t.math(MathFn::Sqrt, &[two])
        };
        let b = {
            let two = t.rat(2, 1);
            t.math(MathFn::Sqrt, &[two])
        };
        t.bin(OpKind::Sub, a, b)
    };
    let log_three = |t: &mut T, _x| {
        let a = {
            let three = t.rat(3, 1);
            t.math(MathFn::Log, &[three])
        };
        let b = {
            let three = t.rat(3, 1);
            t.math(MathFn::Log, &[three])
        };
        t.bin(OpKind::Sub, a, b)
    };
    for build in [&by_two as &Build, &times_half, &sqrt_two, &log_three] {
        let mut f = unary(build);
        let before = f.clone();
        assert_eq!(fired(&mut f, &repro()), ["ALG-003"]);
        assert_eq!(kinds(&f), ["const.float", "return"]);
        same_on_edges(&before, &f);
    }
}

#[test]
fn alg_003_still_refuses_operands_that_can_fail() {
    // x / 0.5 doubles x, so it overflows for a large x; x * 2 too; 1 / x
    // fails at zero; sqrt(x) for a negative x; log(x) for x <= 0.
    let builds: [&Build; 5] = [
        &|t, x| {
            let mk = |t: &mut T| {
                let h = t.rat(1, 2);
                t.bin(OpKind::Div, x, h)
            };
            let (a, b) = (mk(t), mk(t));
            t.bin(OpKind::Sub, a, b)
        },
        &|t, x| {
            let mk = |t: &mut T| {
                let two = t.rat(2, 1);
                t.bin(OpKind::Mul, x, two)
            };
            let (a, b) = (mk(t), mk(t));
            t.bin(OpKind::Sub, a, b)
        },
        &|t, x| {
            let mk = |t: &mut T| {
                let one = t.rat(1, 1);
                t.bin(OpKind::Div, one, x)
            };
            let (a, b) = (mk(t), mk(t));
            t.bin(OpKind::Sub, a, b)
        },
        &|t, x| {
            let (a, b) = (t.math(MathFn::Sqrt, &[x]), t.math(MathFn::Sqrt, &[x]));
            t.bin(OpKind::Sub, a, b)
        },
        &|t, x| {
            let (a, b) = (t.math(MathFn::Log, &[x]), t.math(MathFn::Log, &[x]));
            t.bin(OpKind::Sub, a, b)
        },
    ];
    for build in builds {
        let mut f = unary(build);
        let before = f.clone();
        assert!(fired(&mut f, &repro()).is_empty());
        assert_eq!(f, before);
    }
}

#[test]
fn alg_010_x_over_x_needs_x_provably_nonzero() {
    // sqrt(2) / sqrt(2) is exactly 1.
    let mut f = unary(|t, _x| {
        let a = {
            let two = t.rat(2, 1);
            t.math(MathFn::Sqrt, &[two])
        };
        let b = {
            let two = t.rat(2, 1);
            t.math(MathFn::Sqrt, &[two])
        };
        t.bin(OpKind::Div, a, b)
    });
    let before = f.clone();
    assert_eq!(fired(&mut f, &repro()), ["ALG-010"]);
    assert_eq!(kinds(&f), ["const.float", "return"]);
    same_on_edges(&before, &f);
    assert_eq!(value(&f, &[7.0]), 1.0);

    // x / x for a parameter fails at zero, so it stays.
    let mut g = unary(|t, x| t.bin(OpKind::Div, x, x));
    let before = g.clone();
    assert!(fired(&mut g, &repro()).is_empty());
    assert_eq!(g, before);

    // x * 0.5 / x * 0.5 is zero for x = 0, so it stays too.
    let mut h = unary(|t, x| {
        let mk = |t: &mut T| {
            let half = t.rat(1, 2);
            t.bin(OpKind::Mul, x, half)
        };
        let (a, b) = (mk(t), mk(t));
        t.bin(OpKind::Div, a, b)
    });
    let before = h.clone();
    assert!(fired(&mut h, &repro()).is_empty());
    assert_eq!(h, before);
}

#[test]
fn a_nonzero_float_can_still_be_a_zero_over_the_reals() {
    // sqrt(2) * sqrt(2) - 2 is exactly 0 over the reals, and 4.4e-16 in
    // f64. Dividing by it is fine at run time, so `d / d` may be folded to
    // 1 (it is 1 in f64), and nothing may treat `d` as zero.
    let mut f = unary(|t, _x| {
        let mk = |t: &mut T| {
            let two = t.rat(2, 1);
            let s = t.math(MathFn::Sqrt, &[two]);
            let sq = t.bin(OpKind::Mul, s, s);
            let two2 = t.rat(2, 1);
            t.bin(OpKind::Sub, sq, two2)
        };
        let (a, b) = (mk(t), mk(t));
        t.bin(OpKind::Div, a, b)
    });
    let before = f.clone();
    let rules = fired(&mut f, &repro());
    assert!(rules.contains(&"ALG-010"), "{rules:?}");
    assert_eq!(value(&f, &[1.0]), 1.0);
    assert_eq!(value(&before, &[1.0]), 1.0);
}

#[test]
fn facts_are_not_used_by_the_fast_contract() {
    // Fast rewrites may change a value's last bits, which can push a razor
    // thin fact over its edge, so only Reproducible uses the facts.
    let mut f = unary(|t, _x| {
        let mk = |t: &mut T| {
            let two = t.rat(2, 1);
            t.math(MathFn::Sqrt, &[two])
        };
        let (a, b) = (mk(t), mk(t));
        t.bin(OpKind::Div, a, b)
    });
    assert!(!fired(&mut f, &fast()).contains(&"ALG-010"));
}
