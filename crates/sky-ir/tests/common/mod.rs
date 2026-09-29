//! Shared builders for the sky-ir integration tests.
#![allow(dead_code)]

use sky_diag::Span;
use sky_ir::{
    CmpOp, Edition, Effect, EffectSet, Function, FunctionBuilder, MathFn, Module, NumberType,
    OpKind, Repr, SliceAxis, SolveMethod, Type,
};
use sky_types::{Shape, ShapeDim, ShapeVar, Tag, TagSet};
use sky_units::{Dimension, Rational};

pub fn sp(a: u32, b: u32) -> Span {
    Span::new(a, b)
}

pub fn len() -> Type {
    Type::scalar(Dimension::LENGTH)
}

pub fn time() -> Type {
    Type::scalar(Dimension::TIME)
}

pub fn one() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

pub fn num(dim: Dimension, shape: Shape, tags: TagSet, repr: Repr) -> Type {
    Type::Number(NumberType {
        dim,
        shape,
        tags,
        repr,
    })
}

pub fn vector(n: impl Into<ShapeDim>) -> Type {
    num(
        Dimension::DIMENSIONLESS,
        Shape::vector(n),
        TagSet::EMPTY,
        Repr::Unchosen,
    )
}

pub fn pd_matrix(n: ShapeVar) -> Type {
    num(
        Dimension::DIMENSIONLESS,
        Shape::matrix(n.clone(), n),
        TagSet::of(Tag::PositiveDefinite).closure(),
        Repr::Unchosen,
    )
}

/// A module of `functions`, with each function's effects inferred, so the
/// verifier's effect check does not get in the way of tests about
/// something else.
pub fn module(functions: Vec<Function>) -> Module {
    let mut m = Module {
        edition: Edition::E2026,
        functions,
    };
    let _ = sky_ir::effects::infer_effects(&mut m);
    m
}

/// `fn callee(a: L) -> (L) { return a }`.
pub fn identity_fn(name: &str) -> Function {
    let mut b = FunctionBuilder::new(name, vec![len()], vec![len()], sp(0, 0));
    let a = b.params()[0];
    b.op(OpKind::Return, vec![a], vec![], sp(0, 0));
    b.finish()
}

/// A module using every `OpKind`, an `if` and a `while`, in canonical
/// (builder = parser) order, so `parse(print(m)) == m` exactly.
pub fn all_ops() -> Module {
    let n = ShapeVar::from_parts(0, "N");
    let mut b = FunctionBuilder::new(
        Module::MAIN,
        vec![pd_matrix(n.clone()), vector(n.clone()), len(), one()],
        vec![len()],
        sp(0, 200),
    );
    b.set_effects(
        EffectSet::PURE
            .with(Effect::Io)
            .with(Effect::Fail)
            .with(Effect::Call),
    );
    let [a, bv, x, k] = b.params()[..] else {
        unreachable!()
    };
    b.set_name(a, "A");
    b.set_name(bv, "b");
    b.set_name(x, "x");
    b.set_name(k, "k");

    let c1 = b.op1(
        OpKind::ConstFloat {
            bits: 1e-3f64.to_bits(),
        },
        vec![],
        len(),
        sp(10, 12),
    );
    b.set_name(c1, "eps");
    let c2 = b.op1(
        OpKind::ConstRational(Rational::new(1, 3)),
        vec![],
        one(),
        sp(0, 0),
    );
    let c3 = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp(0, 0));
    let c4 = b.op1(
        OpKind::ConstStr("hi\n\"there\" \\ \u{e9}".into()),
        vec![],
        Type::Str,
        sp(0, 0),
    );
    let c5 = b.op1(OpKind::ConstNothing, vec![], Type::Nothing, sp(0, 0));

    let sum = b.op1(OpKind::Add, vec![x, c1], len(), sp(20, 25));
    let diff = b.op1(OpKind::Sub, vec![x, c1], len(), sp(20, 25));
    let area = b.op1(
        OpKind::Mul,
        vec![sum, diff],
        num(
            Dimension::LENGTH.pow_int(2).unwrap(),
            Shape::scalar(),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(26, 30),
    );
    let ratio = b.op1(OpKind::Div, vec![area, x], len(), sp(0, 0));
    let neg = b.op1(OpKind::Neg, vec![ratio], len(), sp(0, 0));
    let sq = b.op1(
        OpKind::PowConst {
            exponent: Rational::new(2, 1),
        },
        vec![x],
        num(
            Dimension::LENGTH.pow_int(2).unwrap(),
            Shape::scalar(),
            TagSet::EMPTY,
            Repr::F64,
        ),
        sp(0, 0),
    );
    let root = b.op1(
        OpKind::PowConst {
            exponent: Rational::new(1, 2),
        },
        vec![sq],
        num(
            Dimension::LENGTH,
            Shape::scalar(),
            TagSet::EMPTY,
            Repr::Exact,
        ),
        sp(0, 0),
    );
    let p = b.op1(OpKind::Pow, vec![k, c2], one(), sp(0, 0));
    let recip = b.op1(
        OpKind::Inverse,
        vec![x],
        num(
            Dimension::LENGTH.inverse().unwrap(),
            Shape::scalar(),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let shown = b.op1(
        OpKind::Show { unit: "mm".into() },
        vec![x],
        Type::Str,
        sp(5, 9),
    );
    let a_inv = b.op1(OpKind::Inverse, vec![a], pd_matrix(n.clone()), sp(0, 0));
    let prod = b.op1(OpKind::MatMul, vec![a, bv], vector(n.clone()), sp(0, 0));
    let at = b.op1(OpKind::Transpose, vec![a], pd_matrix(n.clone()), sp(0, 0));
    let solved = b.op1(
        OpKind::Solve {
            method: SolveMethod::Cholesky,
        },
        vec![a, bv],
        vector(n.clone()),
        sp(31, 40),
    );
    // Arrays: a 2x2 literal of lengths, positions, slices and builders.
    let lit = b.op1(
        OpKind::ArrayLiteral { cols: Some(2) },
        vec![c1, c1, c1, c1],
        num(
            Dimension::LENGTH,
            Shape::matrix(2, 2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(41, 45),
    );
    let zero = b.op1(
        OpKind::ConstRational(Rational::ZERO),
        vec![],
        one(),
        sp(0, 0),
    );
    let two = b.op1(
        OpKind::ConstRational(Rational::integer(2)),
        vec![],
        one(),
        sp(0, 0),
    );
    let entry = b.op1(OpKind::Index, vec![lit, zero, zero], len(), sp(46, 48));
    let row = b.op1(
        OpKind::Index,
        vec![lit, zero],
        num(
            Dimension::LENGTH,
            Shape::vector(2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let part = b.op1(
        OpKind::Slice {
            axes: vec![
                SliceAxis::Range {
                    start: true,
                    stop: true,
                },
                SliceAxis::At,
            ],
        },
        vec![lit, zero, two, zero],
        num(
            Dimension::LENGTH,
            Shape::vector(2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let whole = b.op1(
        OpKind::Slice {
            axes: vec![
                SliceAxis::Range {
                    start: false,
                    stop: false,
                },
                SliceAxis::Range {
                    start: true,
                    stop: false,
                },
            ],
        },
        vec![lit, zero],
        num(
            Dimension::LENGTH,
            Shape::matrix(2, 2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let head = b.op1(
        OpKind::Slice {
            axes: vec![SliceAxis::Range {
                start: false,
                stop: true,
            }],
        },
        vec![row, two],
        num(
            Dimension::LENGTH,
            Shape::vector(2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let zs = b.op1(OpKind::Zeros, vec![two], vector(2), sp(0, 0));
    let zm = b.op1(
        OpKind::Zeros,
        vec![two, two],
        num(
            Dimension::DIMENSIONLESS,
            Shape::matrix(2, 2),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let eye = b.op1(
        OpKind::Identity,
        vec![two],
        num(
            Dimension::DIMENSIONLESS,
            Shape::matrix(2, 2),
            TagSet::from_tags(Tag::ALL).closure(),
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    let count = b.op1(OpKind::Len, vec![lit], one(), sp(0, 0));
    let root2 = b.op1(OpKind::Math(MathFn::Sqrt), vec![sq], len(), sp(0, 0));
    let mut last_cmp = c3;
    for op in [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ] {
        last_cmp = b.op1(OpKind::Cmp(op), vec![x, c1], Type::Bool, sp(0, 0));
    }
    let not = b.op1(OpKind::Not, vec![last_cmp], Type::Bool, sp(0, 0));

    // if not: yield x else yield -x
    let (iff, res) = b.op(OpKind::If, vec![not], vec![len()], sp(50, 90));
    let outer = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = if arm == 0 {
            x
        } else {
            b.op1(OpKind::Neg, vec![x], len(), sp(0, 0))
        };
        b.op(OpKind::Yield, vec![v], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    let picked = res[0];

    // while i < 100 m: i += eps
    let (w, wres) = b.op(
        OpKind::While,
        vec![picked, k],
        vec![len(), one()],
        sp(91, 120),
    );
    let before = b.add_region(w);
    let bb = b.add_block(before, vec![len(), one()]);
    b.set_insertion_block(bb);
    let [i, k1] = b.func().block(bb).args[..] else {
        unreachable!()
    };
    b.set_name(i, "i");
    let lim = b.op1(
        OpKind::ConstFloat {
            bits: 100f64.to_bits(),
        },
        vec![],
        len(),
        sp(0, 0),
    );
    let go = b.op1(OpKind::Cmp(CmpOp::Lt), vec![i, lim], Type::Bool, sp(0, 0));
    b.op(OpKind::Condition, vec![go, i, k1], vec![], sp(0, 0));
    let after = b.add_region(w);
    let ab = b.add_block(after, vec![len(), one()]);
    b.set_insertion_block(ab);
    let [j, k2] = b.func().block(ab).args[..] else {
        unreachable!()
    };
    let j2 = b.op1(OpKind::Add, vec![j, c1], len(), sp(0, 0));
    b.op(OpKind::Yield, vec![j2, k2], vec![], sp(0, 0));
    b.set_insertion_block(outer);

    let (_, called) = b.op(
        OpKind::Call {
            callee: "id".into(),
        },
        vec![wres[0]],
        vec![len()],
        sp(0, 0),
    );
    b.op(
        OpKind::Print,
        vec![
            called[0], c4, c5, a_inv, prod, at, solved, neg, root, p, recip, c2, c3, shown, entry,
            row, part, whole, head, zs, zm, eye, count, root2,
        ],
        vec![],
        sp(1, 9),
    );
    b.op(OpKind::Return, vec![called[0]], vec![], sp(0, 0));
    module(vec![b.finish(), identity_fn("id")])
}

/// Every kind of type, as function parameters.
pub fn all_types() -> Vec<Type> {
    let odd = Dimension::from_exponents([
        Rational::new(1, 2),
        Rational::new(-3, 2),
        Rational::integer(10),
        Rational::integer(-12),
        Rational::ONE,
        Rational::ZERO,
        Rational::new(7, 3),
    ]);
    vec![
        Type::Bool,
        Type::Str,
        Type::Nothing,
        Type::Unknown,
        one(),
        len(),
        num(
            Dimension::LENGTH / Dimension::TIME.pow_int(2).unwrap(),
            Shape::vector(3),
            TagSet::EMPTY,
            Repr::F32,
        ),
        num(
            odd,
            Shape::new(vec![
                ShapeDim::Known(2),
                ShapeDim::Known(0),
                ShapeDim::Known(u64::MAX),
            ]),
            TagSet::EMPTY,
            Repr::Exact,
        ),
        num(
            Dimension::ENERGY,
            Shape::matrix(ShapeVar::from_parts(3, "N"), ShapeVar::from_parts(3, "N")),
            TagSet::from_tags(Tag::ALL).closure(),
            Repr::F64,
        ),
        // Tags that are not closed print and parse too (the verifier rejects them).
        num(
            Dimension::DIMENSIONLESS,
            Shape::matrix(2, 2),
            TagSet::of(Tag::PositiveDefinite),
            Repr::Unchosen,
        ),
        // A variable name that needs quoting, and a big id.
        num(
            Dimension::MASS,
            Shape::vector(ShapeVar::from_parts(u64::MAX, "my var")),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
    ]
}
