//! The algorithm operations (`solve[lu]`, `solve[diagonal]`, `solve.right`,
//! `matmul[ta,tb,alg=...]`, `diag.mul`): they print and parse back exactly,
//! and the verifier accepts the well-typed ones and names what is wrong with
//! the others.

mod common;

use common::*;
use sky_ir::{
    parse, print, verify, Algorithm, Function, FunctionBuilder, Module, OpKind, Repr, Side,
    SolveMethod, Type,
};
use sky_types::{Shape, Tag, TagSet};
use sky_units::Dimension;

fn tagged(dim: Dimension, r: u64, c: u64, tags: TagSet) -> Type {
    num(dim, Shape::matrix(r, c), tags, Repr::Unchosen)
}

fn mat(r: u64, c: u64) -> Type {
    tagged(Dimension::DIMENSIONLESS, r, c, TagSet::EMPTY)
}

fn with(t: Tag, r: u64) -> Type {
    tagged(Dimension::DIMENSIONLESS, r, r, TagSet::of(t).closure())
}

fn vec_of(dim: Dimension, n: u64) -> Type {
    num(dim, Shape::vector(n), TagSet::EMPTY, Repr::Unchosen)
}

/// `fn f(args) -> (result) { %r = kind args; return %r }`.
fn one_op(kind: OpKind, args: Vec<Type>, result: Type) -> Module {
    let mut b = FunctionBuilder::new("f", args, vec![result.clone()], sp(0, 100));
    let ps = b.params();
    let r = b.op1(kind, ps, result, sp(3, 4));
    b.op(OpKind::Return, vec![r], vec![], sp(90, 99));
    let f: Function = b.finish();
    module(vec![f])
}

/// The same operation on one value used twice.
fn one_op_same_operand(kind: OpKind, arg: Type, result: Type) -> Module {
    let mut b = FunctionBuilder::new("f", vec![arg], vec![result.clone()], sp(0, 100));
    let x = b.params()[0];
    let r = b.op1(kind, vec![x, x], result, sp(3, 4));
    b.op(OpKind::Return, vec![r], vec![], sp(90, 99));
    module(vec![b.finish()])
}

#[track_caller]
fn accepts(m: &Module) {
    if let Err(e) = verify(m) {
        panic!("expected a valid module, got {e:#?}\n{}", print(m));
    }
}

#[track_caller]
fn rejects(m: &Module, needle: &str) {
    let e = verify(m).err().unwrap_or_default();
    assert!(
        e.iter().any(|e| e.message.contains(needle)),
        "no error containing {needle:?}; got {:#?}\n{}",
        e.iter().map(|e| e.message.clone()).collect::<Vec<_>>(),
        print(m)
    );
}

#[track_caller]
fn round_trips(m: &Module) -> String {
    let text = print(m);
    let back = parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(&back, m, "{text}");
    assert_eq!(print(&back), text, "printing is a fixed point");
    text
}

fn matmul_ex(ta: bool, tb: bool, algorithm: Algorithm) -> OpKind {
    OpKind::MatMulEx { ta, tb, algorithm }
}

// ----- text format -----

#[test]
fn solve_methods_round_trip() {
    for method in SolveMethod::ALL {
        let tags = match method {
            SolveMethod::Cholesky => Tag::PositiveDefinite,
            SolveMethod::Diagonal => Tag::Diagonal,
            SolveMethod::Lu => Tag::Symmetric,
        };
        let a = with(tags, 3);
        let x = vec_of(Dimension::DIMENSIONLESS, 3);
        let m = one_op(
            OpKind::Solve { method },
            vec![a.clone(), x.clone()],
            x.clone(),
        );
        accepts(&m);
        let text = round_trips(&m);
        assert!(
            text.contains(&format!("solve[{}]", method.mnemonic())),
            "{text}"
        );
        let m = one_op(OpKind::SolveRight { method }, vec![x.clone(), a], x);
        accepts(&m);
        let text = round_trips(&m);
        assert!(
            text.contains(&format!("solve.right[{}]", method.mnemonic())),
            "{text}"
        );
    }
}

#[test]
fn matmul_attributes_round_trip() {
    let cases = [
        (true, false, Algorithm::Auto, "matmul[ta]"),
        (false, true, Algorithm::Auto, "matmul[tb]"),
        (true, true, Algorithm::Auto, "matmul[ta,tb]"),
        (false, false, Algorithm::Gemm, "matmul[alg=gemm]"),
        (true, false, Algorithm::Gemm, "matmul[ta,alg=gemm]"),
        (true, true, Algorithm::Gemm, "matmul[ta,tb,alg=gemm]"),
    ];
    for (ta, tb, alg, text) in cases {
        let m = one_op(
            matmul_ex(ta, tb, alg),
            vec![mat(3, 3), mat(3, 3)],
            mat(3, 3),
        );
        accepts(&m);
        let printed = round_trips(&m);
        assert!(printed.contains(text), "{printed}");
    }
    let gram = one_op_same_operand(
        matmul_ex(true, false, Algorithm::Syrk),
        mat(3, 3),
        mat(3, 3),
    );
    accepts(&gram);
    assert!(round_trips(&gram).contains("matmul[ta,alg=syrk]"));
}

#[test]
fn diag_mul_round_trips() {
    let d = with(Tag::Diagonal, 3);
    let x = mat(3, 3);
    let left = one_op(
        OpKind::DiagMul { side: Side::Left },
        vec![d.clone(), x.clone()],
        x.clone(),
    );
    accepts(&left);
    assert!(round_trips(&left).contains("diag.mul[left]"));
    let right = one_op(OpKind::DiagMul { side: Side::Right }, vec![x.clone(), d], x);
    accepts(&right);
    assert!(round_trips(&right).contains("diag.mul[right]"));
}

#[test]
fn plain_matmul_still_parses_without_attributes() {
    let m = one_op(OpKind::MatMul, vec![mat(2, 3), mat(3, 4)], mat(2, 4));
    accepts(&m);
    let text = round_trips(&m);
    assert!(!text.contains("matmul["), "{text}");
}

#[test]
fn the_parser_names_bad_attributes() {
    let text = print(&one_op(
        matmul_ex(true, false, Algorithm::Auto),
        vec![mat(3, 3), mat(3, 3)],
        mat(3, 3),
    ));
    for (from, to, expect) in [
        (
            "matmul[ta]",
            "matmul[tx]",
            "expected `ta`, `tb` or `alg=NAME`",
        ),
        ("matmul[ta]", "matmul[alg=fast]", "expected `dot`"),
        ("matmul[ta]", "matmul[tb,ta]", "each once and in this order"),
        ("matmul[ta]", "matmul[ta,ta]", "each once and in this order"),
    ] {
        let bad = text.replace(from, to);
        let e = parse(&bad).expect_err(&bad);
        assert!(e.message.contains(expect), "{e}\n{bad}");
    }
    let solve = print(&one_op(
        OpKind::Solve {
            method: SolveMethod::Lu,
        },
        vec![mat(3, 3), vec_of(Dimension::DIMENSIONLESS, 3)],
        vec_of(Dimension::DIMENSIONLESS, 3),
    ));
    let bad = solve.replace("solve[lu]", "solve[qr]");
    let e = parse(&bad).expect_err(&bad);
    assert!(
        e.message
            .contains("expected `cholesky`, `lu` or `diagonal`"),
        "{e}"
    );
    let diag = print(&one_op(
        OpKind::DiagMul { side: Side::Left },
        vec![with(Tag::Diagonal, 3), mat(3, 3)],
        mat(3, 3),
    ));
    let bad = diag.replace("diag.mul[left]", "diag.mul[up]");
    let e = parse(&bad).expect_err(&bad);
    assert!(e.message.contains("expected `left` or `right`"), "{e}");
}

// ----- the verifier -----

#[test]
fn matmul_with_no_attribute_is_written_plain() {
    rejects(
        &one_op(
            matmul_ex(false, false, Algorithm::Auto),
            vec![mat(3, 3), mat(3, 3)],
            mat(3, 3),
        ),
        "without attributes",
    );
}

#[test]
fn matmul_transposes_change_the_result_shape() {
    // Aᵀ @ B: 2x3ᵀ = 3x2, times 2x4.
    accepts(&one_op(
        matmul_ex(true, false, Algorithm::Auto),
        vec![mat(2, 3), mat(2, 4)],
        mat(3, 4),
    ));
    // A @ Bᵀ: 3x2 times 4x2ᵀ = 2x4.
    accepts(&one_op(
        matmul_ex(false, true, Algorithm::Auto),
        vec![mat(3, 2), mat(4, 2)],
        mat(3, 4),
    ));
    // The untransposed shapes would not multiply.
    rejects(
        &one_op(
            matmul_ex(true, false, Algorithm::Auto),
            vec![mat(2, 3), mat(3, 4)],
            mat(3, 4),
        ),
        "cannot multiply",
    );
    // The result must be the transposed product's.
    rejects(
        &one_op(
            matmul_ex(true, false, Algorithm::Auto),
            vec![mat(2, 3), mat(2, 4)],
            mat(2, 4),
        ),
        "should have shape Matrix[3,4]",
    );
    // Dimensions multiply as for matmul.
    let (l, t) = (Dimension::LENGTH, Dimension::TIME);
    accepts(&one_op(
        matmul_ex(true, false, Algorithm::Auto),
        vec![
            tagged(l, 2, 3, TagSet::EMPTY),
            tagged(t, 2, 4, TagSet::EMPTY),
        ],
        tagged(l * t, 3, 4, TagSet::EMPTY),
    ));
    rejects(
        &one_op(
            matmul_ex(true, false, Algorithm::Auto),
            vec![
                tagged(l, 2, 3, TagSet::EMPTY),
                tagged(t, 2, 4, TagSet::EMPTY),
            ],
            tagged(l, 3, 4, TagSet::EMPTY),
        ),
        "should have dimension",
    );
}

#[test]
fn only_a_matrix_can_be_transposed() {
    let v = vec_of(Dimension::DIMENSIONLESS, 3);
    rejects(
        &one_op(
            matmul_ex(true, false, Algorithm::Auto),
            vec![v.clone(), mat(3, 4)],
            vec_of(Dimension::DIMENSIONLESS, 4),
        ),
        "only a matrix has a transpose",
    );
    rejects(
        &one_op(
            matmul_ex(false, true, Algorithm::Auto),
            vec![mat(4, 3), v],
            vec_of(Dimension::DIMENSIONLESS, 4),
        ),
        "only a matrix has a transpose",
    );
    // A matrix and a vector are fine when the flag is on the matrix.
    accepts(&one_op(
        matmul_ex(true, false, Algorithm::Auto),
        vec![mat(3, 4), vec_of(Dimension::DIMENSIONLESS, 3)],
        vec_of(Dimension::DIMENSIONLESS, 4),
    ));
}

#[test]
fn the_algorithm_class_must_fit_the_operands() {
    let v = || vec_of(Dimension::DIMENSIONLESS, 3);
    let scalar = Type::scalar(Dimension::DIMENSIONLESS);
    accepts(&one_op(
        matmul_ex(false, false, Algorithm::Dot),
        vec![v(), v()],
        scalar,
    ));
    accepts(&one_op(
        matmul_ex(false, false, Algorithm::MatVec),
        vec![mat(3, 3), v()],
        v(),
    ));
    accepts(&one_op(
        matmul_ex(false, false, Algorithm::VecMat),
        vec![v(), mat(3, 3)],
        v(),
    ));
    rejects(
        &one_op(
            matmul_ex(false, false, Algorithm::Dot),
            vec![mat(3, 3), mat(3, 3)],
            mat(3, 3),
        ),
        "does not fit operands of rank 2 and 2",
    );
    rejects(
        &one_op(
            matmul_ex(false, false, Algorithm::Gemm),
            vec![mat(3, 3), v()],
            v(),
        ),
        "does not fit operands of rank 2 and 1",
    );
    rejects(
        &one_op(
            matmul_ex(false, false, Algorithm::MatVec),
            vec![v(), mat(3, 3)],
            v(),
        ),
        "does not fit operands of rank 1 and 2",
    );
}

#[test]
fn syrk_is_one_value_times_its_own_transpose() {
    accepts(&one_op_same_operand(
        matmul_ex(true, false, Algorithm::Syrk),
        mat(3, 4),
        mat(4, 4),
    ));
    accepts(&one_op_same_operand(
        matmul_ex(false, true, Algorithm::Syrk),
        mat(3, 4),
        mat(3, 3),
    ));
    // Two different values.
    rejects(
        &one_op(
            matmul_ex(true, false, Algorithm::Syrk),
            vec![mat(3, 3), mat(3, 3)],
            mat(3, 3),
        ),
        "needs one value multiplied by its own transpose",
    );
    // The same value, but neither or both transposed.
    rejects(
        &one_op_same_operand(
            matmul_ex(false, false, Algorithm::Syrk),
            mat(3, 3),
            mat(3, 3),
        ),
        "needs one value multiplied by its own transpose",
    );
    rejects(
        &one_op_same_operand(matmul_ex(true, true, Algorithm::Syrk), mat(3, 3), mat(3, 3)),
        "needs one value multiplied by its own transpose",
    );
}

#[test]
fn solves_need_the_tag_their_method_relies_on() {
    let x = vec_of(Dimension::DIMENSIONLESS, 3);
    let solve = |method| OpKind::Solve { method };
    // Any square matrix for LU.
    accepts(&one_op(
        solve(SolveMethod::Lu),
        vec![mat(3, 3), x.clone()],
        x.clone(),
    ));
    rejects(
        &one_op(
            solve(SolveMethod::Diagonal),
            vec![mat(3, 3), x.clone()],
            x.clone(),
        ),
        "needs a Diagonal matrix",
    );
    rejects(
        &one_op(
            solve(SolveMethod::Diagonal),
            vec![with(Tag::Symmetric, 3), x.clone()],
            x.clone(),
        ),
        "needs a Diagonal matrix",
    );
    accepts(&one_op(
        solve(SolveMethod::Diagonal),
        vec![with(Tag::Diagonal, 3), x.clone()],
        x.clone(),
    ));
    rejects(
        &one_op(
            solve(SolveMethod::Cholesky),
            vec![with(Tag::Symmetric, 3), x.clone()],
            x.clone(),
        ),
        "needs a PositiveDefinite matrix",
    );
    // The same for the transposed solve.
    rejects(
        &one_op(
            OpKind::SolveRight {
                method: SolveMethod::Diagonal,
            },
            vec![x.clone(), mat(3, 3)],
            x.clone(),
        ),
        "needs a Diagonal matrix",
    );
    accepts(&one_op(
        OpKind::SolveRight {
            method: SolveMethod::Cholesky,
        },
        vec![x.clone(), with(Tag::PositiveDefinite, 3)],
        x,
    ));
}

#[test]
fn solve_shapes_and_dimensions() {
    let (l, t) = (Dimension::LENGTH, Dimension::TIME);
    let v3 = |d| vec_of(d, 3);
    let lu = OpKind::Solve {
        method: SolveMethod::Lu,
    };
    // b / A: the dimension of b over A's.
    accepts(&one_op(
        lu.clone(),
        vec![tagged(l, 3, 3, TagSet::EMPTY), v3(t)],
        v3(t * l.inverse().unwrap()),
    ));
    rejects(
        &one_op(
            lu.clone(),
            vec![tagged(l, 3, 3, TagSet::EMPTY), v3(t)],
            v3(t),
        ),
        "should have dimension",
    );
    // A matrix of right-hand sides keeps its shape.
    accepts(&one_op(lu.clone(), vec![mat(3, 3), mat(3, 5)], mat(3, 5)));
    rejects(
        &one_op(lu.clone(), vec![mat(3, 3), mat(3, 5)], mat(3, 3)),
        "should have shape Matrix[3,5]",
    );
    // A non-square A, and mismatched rows.
    rejects(
        &one_op(
            lu.clone(),
            vec![mat(3, 4), v3(Dimension::DIMENSIONLESS)],
            v3(Dimension::DIMENSIONLESS),
        ),
        "cannot invert a non-square matrix",
    );
    rejects(
        &one_op(
            lu,
            vec![mat(3, 3), vec_of(Dimension::DIMENSIONLESS, 4)],
            vec_of(Dimension::DIMENSIONLESS, 4),
        ),
        "cannot multiply",
    );
    // solve.right: x @ A⁻¹ takes x's shape; a matrix x needs n columns.
    let right = OpKind::SolveRight {
        method: SolveMethod::Lu,
    };
    accepts(&one_op(
        right.clone(),
        vec![mat(5, 3), mat(3, 3)],
        mat(5, 3),
    ));
    rejects(
        &one_op(right.clone(), vec![mat(3, 5), mat(3, 3)], mat(3, 5)),
        "cannot multiply",
    );
    rejects(
        &one_op(right, vec![mat(5, 3), mat(3, 3)], mat(3, 5)),
        "should have shape Matrix[5,3]",
    );
}

#[test]
fn diag_mul_needs_a_diagonal_operand_on_its_side() {
    let d = with(Tag::Diagonal, 3);
    let x = mat(3, 3);
    let left = |a, b| one_op(OpKind::DiagMul { side: Side::Left }, vec![a, b], mat(3, 3));
    let right = |a, b| one_op(OpKind::DiagMul { side: Side::Right }, vec![a, b], mat(3, 3));
    accepts(&left(d.clone(), x.clone()));
    accepts(&right(x.clone(), d.clone()));
    rejects(&left(x.clone(), x.clone()), "needs a Diagonal left operand");
    rejects(
        &right(x.clone(), x.clone()),
        "needs a Diagonal right operand",
    );
    // The diagonal one on the wrong side does not count.
    rejects(&left(x.clone(), d.clone()), "needs a Diagonal left operand");
    // Symmetric is not enough.
    rejects(
        &left(with(Tag::Symmetric, 3), x.clone()),
        "needs a Diagonal left operand",
    );
    // Vectors, shapes and dimensions follow matmul.
    let v = vec_of(Dimension::DIMENSIONLESS, 3);
    accepts(&one_op(
        OpKind::DiagMul { side: Side::Left },
        vec![d.clone(), v.clone()],
        v.clone(),
    ));
    accepts(&one_op(
        OpKind::DiagMul { side: Side::Right },
        vec![v.clone(), d.clone()],
        v.clone(),
    ));
    rejects(
        &one_op(
            OpKind::DiagMul { side: Side::Left },
            vec![d.clone(), mat(4, 2)],
            mat(4, 2),
        ),
        "cannot multiply",
    );
    rejects(
        &one_op(
            OpKind::DiagMul { side: Side::Left },
            vec![d.clone(), mat(3, 5)],
            mat(3, 3),
        ),
        "should have shape Matrix[3,5]",
    );
    let l = Dimension::LENGTH;
    rejects(
        &one_op(
            OpKind::DiagMul { side: Side::Left },
            vec![tagged(l, 3, 3, TagSet::of(Tag::Diagonal).closure()), x],
            mat(3, 3),
        ),
        "should have dimension",
    );
}
