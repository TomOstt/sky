//! Structural tests for each pass, on IR built with `FunctionBuilder`.

use sky_ir::{
    CmpOp, Function, FunctionBuilder, Module, NumberType, OpKind, Repr, SolveMethod, Type, ValueId,
};
use sky_types::{Shape, Tag, TagSet};
use sky_units::{Dimension, Rational};

use crate::testutil::{entry_mnemonics, finish_main, main_builder, module_of, num, sp, verify};
use crate::{cholesky, dce, fold, fold_with, FoldMode};

fn rat(b: &mut FunctionBuilder, n: i64, d: i64) -> ValueId {
    b.op1(
        OpKind::ConstRational(Rational::new(n, d)),
        vec![],
        num(),
        sp(),
    )
}

fn flt(b: &mut FunctionBuilder, x: f64) -> ValueId {
    b.op1(
        OpKind::ConstFloat { bits: x.to_bits() },
        vec![],
        num(),
        sp(),
    )
}

fn print(b: &mut FunctionBuilder, v: ValueId) {
    b.op(OpKind::Print, vec![v], vec![], sp());
}

/// The kind of the operation that defines `v`.
fn kind_of(f: &Function, v: ValueId) -> OpKind {
    match f.value(v).def {
        sky_ir::ValueDef::OpResult { op, .. } => f.op(op).kind.clone(),
        other => panic!("{v} is not an op result: {other:?}"),
    }
}

/// `(a op b)` on two rational constants, printed; returns the module and
/// the result value.
fn binary_rational(kind: OpKind, a: (i64, i64), b: (i64, i64)) -> (Module, ValueId) {
    let mut bd = main_builder();
    let x = rat(&mut bd, a.0, a.1);
    let y = rat(&mut bd, b.0, b.1);
    let r = bd.op1(kind, vec![x, y], num(), sp());
    print(&mut bd, r);
    (finish_main(bd), r)
}

// ---------------------------------------------------------------- fold

#[test]
fn rationals_fold_exactly() {
    let (mut m, r) = binary_rational(OpKind::Add, (1, 3), (1, 6));
    let rws = fold(&mut m);
    verify(&m).unwrap();
    // 1/3 + 1/6 = 1/2 exactly, and 1/2 is exact in f64, so it is bit-identical.
    assert_eq!(
        kind_of(&m.functions[0], r),
        OpKind::ConstRational(Rational::new(1, 2))
    );
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].rule, "FOLD-EXACT-001");
    assert_eq!(rws[0].pass, "fold");
    assert_eq!(rws[0].before, "1/3 + 1/6");
    assert_eq!(rws[0].after, "1/2");
}

#[test]
fn exact_result_that_rounds_differently_is_kept_float_by_default() {
    // 1/10 + 2/10 is exactly 3/10, but 0.1 + 0.2 != 0.3 in f64.
    let (mut m, r) = binary_rational(OpKind::Add, (1, 10), (2, 10));
    let rws = fold(&mut m);
    let expected = (1.0f64 / 10.0) + (2.0f64 / 10.0);
    assert_eq!(
        kind_of(&m.functions[0], r),
        OpKind::ConstFloat {
            bits: expected.to_bits()
        }
    );
    assert_eq!(rws[0].rule, "FOLD-FLOAT-001");

    let (mut m, r) = binary_rational(OpKind::Add, (1, 10), (2, 10));
    let rws = fold_with(&mut m, FoldMode::Exact);
    assert_eq!(
        kind_of(&m.functions[0], r),
        OpKind::ConstRational(Rational::new(3, 10))
    );
    assert_eq!(rws[0].rule, "FOLD-EXACT-001");
}

#[test]
fn chains_fold_in_one_run() {
    // (1 + 2) * 3
    let mut b = main_builder();
    let (one, two, three) = (rat(&mut b, 1, 1), rat(&mut b, 2, 1), rat(&mut b, 3, 1));
    let s = b.op1(OpKind::Add, vec![one, two], num(), sp());
    let p = b.op1(OpKind::Mul, vec![s, three], num(), sp());
    print(&mut b, p);
    let mut m = finish_main(b);
    let rws = fold(&mut m);
    assert_eq!(rws.len(), 2);
    assert_eq!(
        kind_of(&m.functions[0], p),
        OpKind::ConstRational(Rational::integer(9))
    );
}

#[test]
fn rational_overflow_is_not_folded_exactly() {
    // i64::MAX * 2 overflows Rational; the interpreter's f64 product is fine.
    let (mut m, r) = binary_rational(OpKind::Mul, (i64::MAX, 1), (2, 1));
    let rws = fold(&mut m);
    assert!(matches!(
        kind_of(&m.functions[0], r),
        OpKind::ConstFloat { .. }
    ));
    assert_eq!(rws[0].rule, "FOLD-FLOAT-001");
}

#[test]
fn nothing_that_fails_at_run_time_is_folded() {
    // 1/0, 0/0 as rationals and floats, and 1e308 * 10 overflow.
    let (mut m, r) = binary_rational(OpKind::Div, (1, 1), (0, 1));
    assert!(fold(&mut m).is_empty());
    assert_eq!(kind_of(&m.functions[0], r), OpKind::Div);

    let mut b = main_builder();
    let (x, z) = (flt(&mut b, 1.0), flt(&mut b, 0.0));
    let d = b.op1(OpKind::Div, vec![x, z], num(), sp());
    let big = flt(&mut b, 1e308);
    let ten = flt(&mut b, 10.0);
    let o = b.op1(OpKind::Mul, vec![big, ten], num(), sp());
    let neg8 = flt(&mut b, -8.0);
    let third = b.op1(
        OpKind::PowConst {
            exponent: Rational::new(1, 3),
        },
        vec![neg8],
        num(),
        sp(),
    );
    let zero_inv = b.op1(OpKind::Inverse, vec![z], num(), sp());
    for v in [d, o, third, zero_inv] {
        print(&mut b, v);
    }
    let mut m = finish_main(b);
    assert!(fold(&mut m).is_empty());
    let f = &m.functions[0];
    assert_eq!(kind_of(f, d), OpKind::Div);
    assert_eq!(kind_of(f, o), OpKind::Mul);
    assert!(matches!(kind_of(f, third), OpKind::PowConst { .. }));
    assert_eq!(kind_of(f, zero_inv), OpKind::Inverse);
}

#[test]
fn dimension_mismatch_is_not_folded() {
    let len = Type::scalar(Dimension::LENGTH);
    let mut b = main_builder();
    let x = b.op1(
        OpKind::ConstRational(Rational::integer(1)),
        vec![],
        len,
        sp(),
    );
    let y = rat(&mut b, 1, 1);
    let s = b.op1(OpKind::Add, vec![x, y], num(), sp());
    print(&mut b, s);
    let mut m = finish_main(b);
    assert!(fold(&mut m).is_empty());
}

#[test]
fn units_carry_through_a_fold() {
    let len = Type::scalar(Dimension::LENGTH);
    let mut b = main_builder();
    let x = b.op1(
        OpKind::ConstRational(Rational::integer(2)),
        vec![],
        len.clone(),
        sp(),
    );
    let y = b.op1(
        OpKind::ConstRational(Rational::integer(3)),
        vec![],
        len.clone(),
        sp(),
    );
    let s = b.op1(OpKind::Add, vec![x, y], len, sp());
    print(&mut b, s);
    let mut m = finish_main(b);
    let rws = fold(&mut m);
    assert_eq!(
        kind_of(&m.functions[0], s),
        OpKind::ConstRational(Rational::integer(5))
    );
    assert_eq!(rws[0].before, "2 m + 3 m");
    assert_eq!(rws[0].after, "5 m");
}

#[test]
fn negative_zero_is_not_turned_into_a_rational() {
    // -(0) is -0.0 in the interpreter; rational 0 would print differently.
    let mut b = main_builder();
    let z = rat(&mut b, 0, 1);
    let n = b.op1(OpKind::Neg, vec![z], num(), sp());
    print(&mut b, n);
    let mut m = finish_main(b);
    fold(&mut m);
    assert_eq!(
        kind_of(&m.functions[0], n),
        OpKind::ConstFloat {
            bits: (-0.0f64).to_bits()
        }
    );
}

#[test]
fn comparisons_and_not_fold() {
    let mut b = main_builder();
    let (one, two) = (rat(&mut b, 1, 1), rat(&mut b, 2, 1));
    let lt = b.op1(OpKind::Cmp(CmpOp::Lt), vec![one, two], Type::Bool, sp());
    let not = b.op1(OpKind::Not, vec![lt], Type::Bool, sp());
    let (s1, s2) = (
        b.op1(OpKind::ConstStr("a".into()), vec![], Type::Str, sp()),
        b.op1(OpKind::ConstStr("a".into()), vec![], Type::Str, sp()),
    );
    let eq = b.op1(OpKind::Cmp(CmpOp::Eq), vec![s1, s2], Type::Bool, sp());
    // Ordering strings fails at run time: must stay.
    let bad = b.op1(OpKind::Cmp(CmpOp::Lt), vec![s1, s2], Type::Bool, sp());
    for v in [not, eq, bad] {
        print(&mut b, v);
    }
    let mut m = finish_main(b);
    let rws = fold(&mut m);
    let f = &m.functions[0];
    assert_eq!(kind_of(f, lt), OpKind::ConstBool(true));
    assert_eq!(kind_of(f, not), OpKind::ConstBool(false));
    assert_eq!(kind_of(f, eq), OpKind::ConstBool(true));
    assert_eq!(kind_of(f, bad), OpKind::Cmp(CmpOp::Lt));
    let rules: Vec<_> = rws.iter().map(|r| r.rule).collect();
    assert_eq!(rules, ["FOLD-CMP-001", "FOLD-NOT-001", "FOLD-CMP-001"]);
}

/// `fn f(p) { c = <cond>; r = if c { neg p } else { p }; return r }`
fn if_program(cond: bool) -> (Module, ValueId, ValueId) {
    let mut b = FunctionBuilder::new("f", vec![num()], vec![num()], sp());
    let p = b.params()[0];
    let c = b.op1(OpKind::ConstBool(cond), vec![], Type::Bool, sp());
    let (iff, res) = b.op(OpKind::If, vec![c], vec![num()], sp());
    let outer = b.insertion_block();
    let mut inner_neg = None;
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = if arm == 0 {
            let n = b.op1(OpKind::Neg, vec![p], num(), sp());
            inner_neg = Some(n);
            n
        } else {
            b.op1(OpKind::Add, vec![p, p], num(), sp())
        };
        b.op(OpKind::Yield, vec![v], vec![], sp());
    }
    b.set_insertion_block(outer);
    b.op(OpKind::Return, res.clone(), vec![], sp());
    (module_of(b.finish()), res[0], inner_neg.unwrap())
}

#[test]
fn if_on_constant_true_inlines_the_then_branch() {
    let (mut m, _res, neg) = if_program(true);
    let rws = fold(&mut m);
    verify(&m).unwrap();
    let f = &m.functions[0];
    assert_eq!(entry_mnemonics(f), ["const.bool", "neg", "return"]);
    let ret = *f.block(f.entry()).ops.last().unwrap();
    assert_eq!(f.op(ret).operands, [neg]);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].rule, "FOLD-IF-001");
    assert!(rws[0].because[0].contains("constant true"));
}

#[test]
fn if_on_constant_false_inlines_the_else_branch() {
    let (mut m, _res, _) = if_program(false);
    fold(&mut m);
    verify(&m).unwrap();
    assert_eq!(
        entry_mnemonics(&m.functions[0]),
        ["const.bool", "add", "return"]
    );
}

#[test]
fn if_on_folded_condition_inlines_after_folding_it() {
    // c = 1 < 2 (folds to true), then the if inlines, in one run.
    let mut b = main_builder();
    let (one, two) = (rat(&mut b, 1, 1), rat(&mut b, 2, 1));
    let c = b.op1(OpKind::Cmp(CmpOp::Lt), vec![one, two], Type::Bool, sp());
    let (iff, res) = b.op(OpKind::If, vec![c], vec![num()], sp());
    let outer = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = rat(&mut b, arm + 10, 1);
        b.op(OpKind::Yield, vec![v], vec![], sp());
    }
    b.set_insertion_block(outer);
    print(&mut b, res[0]);
    let mut m = finish_main(b);
    let rws = fold(&mut m);
    verify(&m).unwrap();
    assert_eq!(rws.len(), 2);
    let after = dce(&mut m);
    verify(&m).unwrap();
    assert!(after.is_empty());
    // const 1, const 2, cmp are dead; the taken branch's 10 is printed.
    assert_eq!(
        entry_mnemonics(&m.functions[0]),
        ["const.rational", "print", "return"]
    );
}

// ----------------------------------------------------------------- dce

#[test]
fn unused_temporaries_vanish_silently() {
    let mut b = main_builder();
    let c = rat(&mut b, 1, 1);
    let n = b.op1(OpKind::Neg, vec![c], num(), sp());
    let _nn = b.op1(OpKind::Neg, vec![n], num(), sp());
    let mut m = finish_main(b);
    let rws = dce(&mut m);
    verify(&m).unwrap();
    assert!(rws.is_empty(), "lowering temporaries need no explanation");
    assert_eq!(entry_mnemonics(&m.functions[0]), ["return"]);
}

#[test]
fn removing_a_named_value_is_explained() {
    let mut b = main_builder();
    let c = rat(&mut b, 5, 1);
    b.set_name(c, "unused_limit");
    let mut m = finish_main(b);
    let rws = dce(&mut m);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].pass, "dce");
    assert_eq!(rws[0].rule, "DCE-001");
    assert_eq!(rws[0].before, "%unused_limit = const.rational 5");
    assert!(rws[0].because[0].contains("`unused_limit`"));
    assert_eq!(entry_mnemonics(&m.functions[0]), ["return"]);
}

#[test]
fn effects_are_never_removed() {
    let mut b = main_builder();
    let (x, y) = (rat(&mut b, 1, 1), rat(&mut b, 0, 1));
    // Unused, but it fails at run time.
    let _d = b.op1(OpKind::Div, vec![x, y], num(), sp());
    print(&mut b, x);
    let (_c, _r) = b.op(OpKind::Call { callee: "g".into() }, vec![], vec![], sp());
    let mut m = finish_main(b);
    assert!(dce(&mut m).is_empty());
    assert_eq!(
        entry_mnemonics(&m.functions[0]),
        [
            "const.rational",
            "const.rational",
            "div",
            "print",
            "call",
            "return"
        ]
    );
}

#[test]
fn unknown_typed_dynamic_checks_are_kept() {
    // `neg` of an Unknown value can fail in the interpreter.
    let mut b = FunctionBuilder::new("f", vec![Type::Unknown], vec![], sp());
    let p = b.params()[0];
    b.op1(OpKind::Neg, vec![p], Type::Unknown, sp());
    b.op(OpKind::Return, vec![], vec![], sp());
    let mut m = module_of(b.finish());
    dce(&mut m);
    assert_eq!(entry_mnemonics(&m.functions[0]), ["neg", "return"]);
}

/// `if c { yield <then_op>(p) } else { yield p }` with the result unused.
fn unused_if(then_kind: OpKind, named: bool) -> Module {
    let mut b = FunctionBuilder::new("f", vec![Type::Bool, num()], vec![], sp());
    let [c, p] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![num()], sp());
    if named {
        b.set_name(res[0], "r");
    }
    let outer = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = if arm == 0 {
            b.op1(then_kind.clone(), vec![p, p], num(), sp())
        } else {
            p
        };
        b.op(OpKind::Yield, vec![v], vec![], sp());
    }
    b.set_insertion_block(outer);
    b.op(OpKind::Return, vec![], vec![], sp());
    module_of(b.finish())
}

#[test]
fn unused_if_with_pure_regions_is_removed() {
    // `cmp.eq` of two numbers of one dimension is pure and cannot fail.
    let mut b = FunctionBuilder::new("f", vec![Type::Bool, num()], vec![], sp());
    let [c, p] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![Type::Bool], sp());
    b.set_name(res[0], "flag");
    let outer = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = if arm == 0 {
            b.op1(OpKind::Cmp(CmpOp::Eq), vec![p, p], Type::Bool, sp())
        } else {
            b.op1(OpKind::ConstBool(false), vec![], Type::Bool, sp())
        };
        b.op(OpKind::Yield, vec![v], vec![], sp());
    }
    b.set_insertion_block(outer);
    b.op(OpKind::Return, vec![], vec![], sp());
    let mut m = module_of(b.finish());
    let rws = dce(&mut m);
    verify(&m).unwrap();
    assert_eq!(entry_mnemonics(&m.functions[0]), ["return"]);
    assert_eq!(rws.len(), 1);
    assert!(rws[0].because[0].contains("`flag`"));
}

#[test]
fn unused_if_with_a_failing_region_is_kept() {
    let mut m = unused_if(OpKind::Add, false);
    assert!(dce(&mut m).is_empty());
    assert_eq!(entry_mnemonics(&m.functions[0]), ["if", "return"]);
    // and a kept `if` still has its ops.
    verify(&m).unwrap();
}

#[test]
fn dead_code_inside_a_kept_region_is_removed() {
    // The `if` result is used (returned); a dead constant in an arm is not.
    let mut b = FunctionBuilder::new("f", vec![Type::Bool, num()], vec![num()], sp());
    let [c, p] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![num()], sp());
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        rat(&mut b, 99, 1);
        b.op(OpKind::Yield, vec![p], vec![], sp());
    }
    b.set_insertion_block(outer);
    b.op(OpKind::Return, res, vec![], sp());
    let mut m = module_of(b.finish());
    dce(&mut m);
    verify(&m).unwrap();
    let f = &m.functions[0];
    for r in &f.op(f.block(f.entry()).ops[0]).regions {
        let blk = f.region(*r).blocks[0];
        assert_eq!(f.block(blk).ops.len(), 1, "only the yield is left");
    }
}

// ----------------------------------------------------------- cholesky

fn matrix(tags: TagSet, n: u64) -> Type {
    Type::Number(NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape: Shape::matrix(n, n),
        tags,
        repr: Repr::Unchosen,
    })
}

fn vector(n: u64) -> Type {
    Type::Number(NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape: Shape::vector(n),
        tags: TagSet::EMPTY,
        repr: Repr::Unchosen,
    })
}

struct Solve {
    m: Module,
    inv: ValueId,
    mm: ValueId,
    a: ValueId,
    b: ValueId,
}

/// `fn f(A, b) { x = A⁻¹ @ b; return x (and A⁻¹ too if `keep_inverse`) }`.
fn solve_program(a_ty: Type, b_ty: Type, result_ty: Type, keep_inverse: bool) -> Solve {
    let results = if keep_inverse {
        vec![result_ty.clone(), a_ty.clone()]
    } else {
        vec![result_ty.clone()]
    };
    let mut bd = FunctionBuilder::new("f", vec![a_ty.clone(), b_ty], results, sp());
    let [a, b] = bd.params()[..] else {
        unreachable!()
    };
    let inv = bd.op1(OpKind::Inverse, vec![a], a_ty, sp());
    let mm = bd.op1(OpKind::MatMul, vec![inv, b], result_ty, sp());
    let rets = if keep_inverse {
        vec![mm, inv]
    } else {
        vec![mm]
    };
    bd.op(OpKind::Return, rets, vec![], sp());
    Solve {
        m: module_of(bd.finish()),
        inv,
        mm,
        a,
        b,
    }
}

const PD: fn() -> TagSet = || TagSet::of(Tag::PositiveDefinite);

#[test]
fn positive_definite_inverse_times_vector_becomes_a_cholesky_solve() {
    let mut s = solve_program(matrix(PD(), 3), vector(3), vector(3), false);
    let rws = cholesky(&mut s.m);
    verify(&s.m).unwrap();
    let f = &s.m.functions[0];
    assert_eq!(
        entry_mnemonics(f),
        ["solve", "return"],
        "the inverse is gone"
    );
    let solve = f.op(match f.value(s.mm).def {
        sky_ir::ValueDef::OpResult { op, .. } => op,
        _ => unreachable!(),
    });
    assert_eq!(
        solve.kind,
        OpKind::Solve {
            method: SolveMethod::Cholesky
        }
    );
    assert_eq!(solve.operands, [s.a, s.b]);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].pass, "cholesky");
    assert_eq!(rws[0].rule, "LA-CHOLESKY-001");
    assert_eq!(rws[0].before, "A⁻¹ @ b");
    assert_eq!(rws[0].after, "solve[cholesky](A, b)");
}

/// The checker's explanation, for the same types.
fn checker_text(a: Shape, tags: TagSet, b: Shape) -> String {
    use sky_types::{cholesky_rewrite, MatmulLhs, ShapeUnifier, TensorType};
    let mut u = ShapeUnifier::new();
    let a = TensorType::tagged(&mut u, a, tags).unwrap();
    let b = TensorType::untagged(b);
    cholesky_rewrite(MatmulLhs::Inverse(&a), &b, &u)
        .unwrap()
        .explanation()
        .to_string()
}

#[test]
fn explanation_text_matches_the_checkers() {
    for (b_shape, res) in [
        (Shape::vector(3), vector(3)),
        (Shape::matrix(3, 2), {
            Type::Number(NumberType {
                dim: Dimension::DIMENSIONLESS,
                shape: Shape::matrix(3, 2),
                tags: TagSet::EMPTY,
                repr: Repr::Unchosen,
            })
        }),
    ] {
        let b_ty = Type::Number(NumberType {
            dim: Dimension::DIMENSIONLESS,
            shape: b_shape.clone(),
            tags: TagSet::EMPTY,
            repr: Repr::Unchosen,
        });
        let mut s = solve_program(matrix(PD(), 3), b_ty, res, false);
        // Through the pass manager, so this is the explanation
        // `sky check --explain` gets, not just the pass's own record.
        // (The hand-built types here have unclosed tag sets, which the
        // verifier rightly refuses; the pass does not care.)
        let report = sky_ir::PassManager::for_tests()
            .verify_each(false)
            .with(crate::CholeskyPass)
            .run(&mut s.m)
            .unwrap();
        assert_eq!(report.explanations.len(), 1);
        let x = &report.explanations[0];
        assert_eq!((x.pass, x.rule), ("cholesky", "LA-CHOLESKY-001"));
        let text = crate::cholesky_text(x).expect("a Cholesky explanation");
        assert_eq!(text, checker_text(Shape::matrix(3, 3), PD(), b_shape));
    }
}

#[test]
fn a_shared_inverse_is_kept_and_the_solve_is_still_used() {
    let mut s = solve_program(matrix(PD(), 3), vector(3), vector(3), true);
    let rws = cholesky(&mut s.m);
    verify(&s.m).unwrap();
    assert_eq!(
        entry_mnemonics(&s.m.functions[0]),
        ["inverse", "solve", "return"]
    );
    assert_eq!(rws.len(), 1);
    assert!(rws[0].because.last().unwrap().contains("other users"));
    let _ = s.inv;
}

#[test]
fn user_names_appear_in_before_and_after() {
    let mut bd = FunctionBuilder::new("f", vec![matrix(PD(), 2), vector(2)], vec![vector(2)], sp());
    let [a, b] = bd.params()[..] else {
        unreachable!()
    };
    bd.set_name(a, "K");
    bd.set_name(b, "y");
    let inv = bd.op1(OpKind::Inverse, vec![a], matrix(PD(), 2), sp());
    let mm = bd.op1(OpKind::MatMul, vec![inv, b], vector(2), sp());
    bd.op(OpKind::Return, vec![mm], vec![], sp());
    let mut m = module_of(bd.finish());
    let rws = cholesky(&mut m);
    assert_eq!(rws[0].before, "K⁻¹ @ y");
    assert_eq!(rws[0].after, "solve[cholesky](K, y)");
    // The names are in the reasons too, and they join into the checker's
    // wording for named operands.
    let x = sky_ir::Explanation {
        pass: "cholesky",
        rule: rws[0].rule,
        span: rws[0].span,
        before: rws[0].before.clone(),
        after: rws[0].after.clone(),
        because: rws[0].because.clone(),
    };
    assert_eq!(
        crate::cholesky_text(&x).unwrap(),
        "rewrite `K⁻¹ @ y` to a Cholesky solve: K is Matrix[2,2] and PositiveDefinite, \
         y is Vector[2]. Factor K = L·Lᵀ (L lower-triangular), solve L·y = y, \
         then Lᵀ·x = y. x = K⁻¹·y is Vector[2], and K⁻¹ is never formed."
    );
}

#[test]
fn no_rewrite_without_positive_definiteness() {
    for tags in [
        TagSet::EMPTY,
        TagSet::of(Tag::Symmetric),
        TagSet::of(Tag::Diagonal),
    ] {
        let mut s = solve_program(matrix(tags, 3), vector(3), vector(3), false);
        assert!(cholesky(&mut s.m).is_empty());
        assert_eq!(
            entry_mnemonics(&s.m.functions[0]),
            ["inverse", "matmul", "return"]
        );
    }
}

#[test]
fn no_rewrite_when_shapes_do_not_fit() {
    let mut s = solve_program(matrix(PD(), 3), vector(4), vector(4), false);
    assert!(cholesky(&mut s.m).is_empty());
    let mut s = solve_program(matrix(PD(), 3), vector(3), matrix(TagSet::EMPTY, 3), false);
    assert!(cholesky(&mut s.m).is_empty());
}

#[test]
fn only_an_explicit_inverse_on_the_left_is_rewritten() {
    // matmul(A, inverse(A)): the inverse is on the right. And a matmul of
    // a plain PositiveDefinite value (like `A ^ -1` lowered as a power).
    let a_ty = matrix(PD(), 3);
    let mut bd = FunctionBuilder::new("f", vec![a_ty.clone(), a_ty.clone()], vec![], sp());
    let [a, c] = bd.params()[..] else {
        unreachable!()
    };
    let inv = bd.op1(OpKind::Inverse, vec![a], a_ty.clone(), sp());
    bd.op1(OpKind::MatMul, vec![a, inv], a_ty.clone(), sp());
    bd.op1(OpKind::MatMul, vec![c, a], a_ty, sp());
    bd.op(OpKind::Return, vec![], vec![], sp());
    let mut m = module_of(bd.finish());
    assert!(cholesky(&mut m).is_empty());
}

// ---------------------------------------------------------------- show

#[test]
fn show_survives_fold_and_dce_and_keeps_its_failure() {
    let mut b = main_builder();
    let x = b.op1(
        OpKind::ConstFloat {
            bits: 25f64.to_bits(),
        },
        vec![],
        Type::scalar(Dimension::VELOCITY),
        sp(),
    );
    let shown = b.op1(
        OpKind::Show {
            unit: "km/h".into(),
        },
        vec![x],
        Type::Str,
        sp(),
    );
    print(&mut b, shown);
    // An unused conversion that overflows (1e300 m is 1e309 nm) is an error
    // the program has: neither pass may drop it as dead code.
    let big = b.op1(
        OpKind::ConstFloat {
            bits: 1e300f64.to_bits(),
        },
        vec![],
        Type::scalar(Dimension::LENGTH),
        sp(),
    );
    b.op1(
        OpKind::Show { unit: "nm".into() },
        vec![big],
        Type::Str,
        sp(),
    );
    let mut m = finish_main(b);

    let run = |m: &Module| {
        sky_ast::with_deep_stack(|| {
            let mut out = Vec::new();
            let r = sky_ir_interp::run(m, &mut out).map_err(|e| e.message.clone());
            (String::from_utf8(out).expect("utf-8"), r)
        })
    };
    let before = run(&m);
    assert_eq!(
        before,
        (
            "90 km/h\n".to_string(),
            Err("arithmetic overflow".to_string())
        )
    );
    fold(&mut m);
    dce(&mut m);
    verify(&m).unwrap();
    assert_eq!(run(&m), before);
    let shows = entry_mnemonics(&m.functions[0])
        .into_iter()
        .filter(|n| *n == "show")
        .count();
    assert_eq!(shows, 2);
}

// -------------------------------------------------------------- arrays

/// `[1, 2]` from two constants, typed as a vector.
fn literal(b: &mut FunctionBuilder) -> ValueId {
    let (x, y) = (rat(b, 1, 1), rat(b, 2, 1));
    b.op1(
        OpKind::ArrayLiteral { cols: None },
        vec![x, y],
        vector(2),
        sp(),
    )
}

#[test]
fn array_operations_are_not_folded() {
    let mut b = main_builder();
    let v = literal(&mut b);
    let i = rat(&mut b, 0, 1);
    let e = b.op1(OpKind::Index, vec![v, i], num(), sp());
    let n = b.op1(OpKind::Len, vec![v], num(), sp());
    let z = b.op1(OpKind::Zeros, vec![n], vector(2), sp());
    let sum = b.op1(OpKind::Add, vec![v, z], vector(2), sp());
    print(&mut b, e);
    print(&mut b, sum);
    let mut m = finish_main(b);
    let before = entry_mnemonics(&m.functions[0]);
    assert!(fold(&mut m).is_empty());
    assert!(fold_with(&mut m, FoldMode::Exact).is_empty());
    assert_eq!(entry_mnemonics(&m.functions[0]), before);
}

#[test]
fn unused_pure_array_literals_go_but_operations_that_can_fail_stay() {
    let mut b = main_builder();
    let v = literal(&mut b);
    let i = rat(&mut b, 5, 1);
    // Unused, but out of bounds at run time.
    b.op1(OpKind::Index, vec![v, i], num(), sp());
    b.op1(
        OpKind::Slice {
            axes: vec![sky_ir::SliceAxis::Range {
                start: true,
                stop: false,
            }],
        },
        vec![v, i],
        vector(1),
        sp(),
    );
    b.op1(OpKind::Zeros, vec![i], vector(5), sp());
    b.op1(OpKind::Identity, vec![i], matrix(TagSet::EMPTY, 5), sp());
    b.op1(OpKind::Len, vec![v], num(), sp());
    // A literal nobody reads is pure and vanishes.
    let (x, y) = (rat(&mut b, 3, 1), rat(&mut b, 4, 1));
    b.op1(
        OpKind::ArrayLiteral { cols: None },
        vec![x, y],
        vector(2),
        sp(),
    );
    let mut m = finish_main(b);
    dce(&mut m);
    verify(&m).unwrap();
    assert_eq!(
        entry_mnemonics(&m.functions[0]),
        [
            "const.rational",
            "const.rational",
            "array.literal",
            "const.rational",
            "index",
            "slice",
            "zeros",
            "identity",
            "len",
            "return"
        ]
    );
}

#[test]
fn a_solve_the_lowering_already_chose_is_left_alone() {
    let pd = matrix(TagSet::of(Tag::PositiveDefinite).closure(), 2);
    let mut b = FunctionBuilder::new("f", vec![pd, vector(2)], vec![vector(2)], sp());
    let [a, rhs] = b.params()[..] else {
        unreachable!()
    };
    let x = b.op1(
        OpKind::Solve {
            method: SolveMethod::Cholesky,
        },
        vec![a, rhs],
        vector(2),
        sp(),
    );
    b.op(OpKind::Return, vec![x], vec![], sp());
    let mut m = module_of(b.finish());
    assert!(cholesky(&mut m).is_empty());
    assert_eq!(entry_mnemonics(&m.functions[0]), ["solve", "return"]);
}
