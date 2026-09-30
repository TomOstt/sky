//! Tests of the symbolic domain, the analysis on hand-built functions, and
//! `dyadic`. The soundness of the ranges against the real interpreter is
//! tested in `sky-opt`, which has the interpreter.

use std::cmp::Ordering;

use sky_diag::Span;
use sky_units::{Dimension, Rational};

use super::*;
use crate::builder::FunctionBuilder;
use crate::ir::{Function, Type, ValueDef};
use crate::op::OpKind;

fn q(n: i64, d: i64) -> Rational {
    Rational::new(n, d)
}

fn r(n: i64, d: i64) -> Sym {
    Sym::rational(q(n, d))
}

fn root(n: i64) -> Sym {
    Sym::root(Rational::integer(n)).unwrap()
}

// ------------------------------------------------------------ Sym

#[test]
fn display() {
    assert_eq!(r(1, 2).to_string(), "1/2");
    assert_eq!(Sym::integer(0).to_string(), "0");
    assert_eq!(Sym::integer(-3).to_string(), "-3");
    assert_eq!(Sym::pi().to_string(), "π");
    assert_eq!(Sym::pi().mul(&r(3, 4)).unwrap().to_string(), "3π/4");
    assert_eq!(Sym::pi().mul(&r(1, 4)).unwrap().to_string(), "π/4");
    assert_eq!(Sym::pi().neg().unwrap().to_string(), "-π");
    assert_eq!(root(8).to_string(), "2√2");
    assert_eq!(r(1, 2).add(&root(3)).unwrap().to_string(), "1/2 + √3");
    assert_eq!(r(5, 1).sub(&root(2)).unwrap().to_string(), "5 - √2");
}

#[test]
fn roots_are_squarefree() {
    assert_eq!(root(4), Sym::integer(2));
    assert_eq!(root(8).terms(), &[(Atom::Sqrt(2), q(2, 1))]);
    assert_eq!(root(12).terms(), &[(Atom::Sqrt(3), q(2, 1))]);
    // √(1/2) = √2/2
    assert_eq!(
        Sym::root(q(1, 2)).unwrap().terms(),
        &[(Atom::Sqrt(2), q(1, 2))]
    );
    assert_eq!(Sym::root(q(9, 4)).unwrap(), r(3, 2));
    assert_eq!(Sym::root(q(-1, 1)), None);
    // A large non-square radicand is not factored.
    assert_eq!(Sym::root(q(1_000_003, 1)), None);
    // A large perfect square is.
    assert_eq!(
        Sym::root(q(1_000_000_000_000, 1)),
        Some(Sym::integer(1_000_000))
    );
}

#[test]
fn products_of_roots() {
    assert_eq!(root(2).mul(&root(2)).unwrap(), Sym::integer(2));
    assert_eq!(root(2).mul(&root(3)).unwrap(), root(6));
    assert_eq!(
        root(6).mul(&root(3)).unwrap(),
        root(2).mul(&r(3, 1)).unwrap()
    );
    // (1 + √2)² = 3 + 2√2
    let a = Sym::integer(1).add(&root(2)).unwrap();
    assert_eq!(
        a.powi(2).unwrap(),
        Sym::integer(3)
            .add(&root(2).mul(&r(2, 1)).unwrap())
            .unwrap()
    );
    // The conjugate product is rational.
    let b = Sym::integer(1).sub(&root(2)).unwrap();
    assert_eq!(a.mul(&b).unwrap(), Sym::integer(-1));
}

#[test]
fn division() {
    assert_eq!(Sym::integer(1).div(&r(3, 1)).unwrap(), r(1, 3));
    assert_eq!(Sym::integer(1).div(&Sym::integer(0)), None);
    // 1/√2 = √2/2
    assert_eq!(
        Sym::integer(1).div(&root(2)).unwrap(),
        Sym::root(q(1, 2)).unwrap()
    );
    // √2 / √2 = 1
    assert_eq!(root(2).div(&root(2)).unwrap(), Sym::integer(1));
    // 1/π is not representable, and neither is division by a sum.
    assert_eq!(Sym::integer(1).div(&Sym::pi()), None);
    assert_eq!(
        Sym::integer(1).div(&Sym::integer(1).add(&root(2)).unwrap()),
        None
    );
}

#[test]
fn what_is_not_representable() {
    assert_eq!(Sym::pi().mul(&Sym::pi()), None);
    assert_eq!(Sym::pi().mul(&root(2)), None);
    // π and e never share a form.
    assert_eq!(Sym::pi().add(&Sym::e()), None);
    assert_eq!(Sym::pi().sub(&Sym::pi()).unwrap(), Sym::integer(0));
    // The term cap.
    let mut s = Sym::integer(1);
    for n in [2, 3, 5] {
        s = s.add(&root(n)).unwrap();
    }
    assert_eq!(s.terms().len(), MAX_TERMS);
    assert_eq!(s.add(&root(7)), None);
    // Terms that cancel free room.
    assert_eq!(s.sub(&root(5)).unwrap().terms().len(), 3);
    assert!(Sym::integer(2).powi(16).is_some());
    assert_eq!(Sym::integer(2).powi(17), None);
}

#[test]
fn exact_cancellation_is_zero() {
    // (√2 + √3)(√2 - √3) = -1
    let a = root(2).add(&root(3)).unwrap();
    let b = root(2).sub(&root(3)).unwrap();
    assert_eq!(a.mul(&b).unwrap(), Sym::integer(-1));
    assert!(a.sub(&a).unwrap().is_zero());
    // A coefficient overflow is "not known", not a wrap.
    let big = Sym::integer(i64::MAX);
    assert_eq!(big.add(&big), None);
}

#[test]
fn signs() {
    let sign = |s: Option<Sym>| s.unwrap().sign();
    assert_eq!(sign(root(2).sub(&Sym::integer(1))), Some(Ordering::Greater));
    assert_eq!(sign(Sym::integer(1).sub(&root(2))), Some(Ordering::Less));
    assert_eq!(
        sign(Sym::pi().sub(&Sym::integer(3))),
        Some(Ordering::Greater)
    );
    assert_eq!(sign(Sym::pi().sub(&Sym::integer(4))), Some(Ordering::Less));
    assert_eq!(Sym::integer(0).sign(), Some(Ordering::Equal));
    // The enclosure cannot separate √2 from a 17-digit rational.
    let close = root(2).sub(&Sym::rational(q(
        14_142_135_623_730_951,
        10_000_000_000_000_000,
    )));
    assert_eq!(sign(close), None);
    let m = root(2).sub(&root(2).mul(&r(2, 1)).unwrap()).unwrap();
    assert_eq!(m.abs().unwrap(), root(2));
}

/// A deterministic generator: xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random expression with its value in `f64` (accurate to about 1e-12).
fn random_sym(rng: &mut Rng, depth: u32) -> Option<(Sym, f64)> {
    if depth == 0 || rng.below(4) == 0 {
        return Some(match rng.below(5) {
            0 => {
                let (n, d) = (rng.below(19) as i64 - 9, rng.below(9) as i64 + 1);
                (r(n, d), n as f64 / d as f64)
            }
            1 => {
                let n = [2i64, 3, 5, 6, 7, 8, 12][rng.below(7) as usize];
                (root(n), (n as f64).sqrt())
            }
            2 => (Sym::pi(), std::f64::consts::PI),
            3 => (Sym::e(), std::f64::consts::E),
            _ => (r(1, 2), 0.5),
        });
    }
    let (a, x) = random_sym(rng, depth - 1)?;
    let (b, y) = random_sym(rng, depth - 1)?;
    match rng.below(4) {
        0 => Some((a.add(&b)?, x + y)),
        1 => Some((a.sub(&b)?, x - y)),
        2 => Some((a.mul(&b)?, x * y)),
        _ => Some((a.div(&b)?, x / y)),
    }
}

#[test]
fn the_enclosure_contains_the_value() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0;
    for _ in 0..4000 {
        let Some((s, x)) = random_sym(&mut rng, 3) else {
            continue;
        };
        if !x.is_finite() {
            continue;
        }
        let (lo, hi) = s.enclose().expect("small coefficients");
        let tol = 1e-9 * (1.0 + x.abs());
        assert!(lo - tol <= x && x <= hi + tol, "{s}: {lo} <= {x} <= {hi}");
        assert!(lo <= hi);
        if let Some(o) = s.sign() {
            if x.abs() > 1e-9 {
                let expect = if x > 0.0 {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
                assert_eq!(o, expect, "{s} = {x}");
            }
        }
        checked += 1;
    }
    assert!(checked > 1000, "{checked}");
}

#[test]
fn sums_and_products_are_commutative_as_data() {
    let mut rng = Rng(42);
    for _ in 0..2000 {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            random_sym(&mut rng, 2),
            random_sym(&mut rng, 2),
            random_sym(&mut rng, 2),
        ) else {
            continue;
        };
        assert_eq!(a.add(&b), b.add(&a));
        assert_eq!(a.mul(&b), b.mul(&a));
        if let (Some(ab), Some(bc)) = (a.add(&b), b.add(&c)) {
            if let (Some(l), Some(rr)) = (ab.add(&c), a.add(&bc)) {
                assert_eq!(l, rr);
            }
        }
    }
}

// ------------------------------------------------------------ dyadic

#[test]
fn every_float_is_a_rational() {
    assert_eq!(dyadic(0.5), Some(q(1, 2)));
    assert_eq!(dyadic(-3.0), Some(q(-3, 1)));
    assert_eq!(dyadic(0.0), Some(Rational::ZERO));
    assert_eq!(dyadic(-0.0), Some(Rational::ZERO));
    // 0.1 is not 1/10.
    let tenth = dyadic(0.1).unwrap();
    assert_ne!(tenth, q(1, 10));
    assert_eq!(tenth.denom(), 1 << 55);
    assert_eq!(tenth.numer() as f64 / tenth.denom() as f64, 0.1);
    // Too many bits, or out of range.
    assert_eq!(dyadic(1e-300), None);
    assert_eq!(dyadic(1e300), None);
    assert_eq!(dyadic(f64::INFINITY), None);
    assert_eq!(dyadic(f64::NAN), None);
    // 2^62 fits, 2^63 does not.
    assert_eq!(dyadic(4_611_686_018_427_387_904.0), Some(q(1 << 62, 1)));
    assert_eq!(dyadic(9_223_372_036_854_775_808.0), None);
}

#[test]
fn dyadic_round_trips() {
    let mut rng = Rng(7);
    let mut hits = 0;
    for _ in 0..200_000 {
        let x = f64::from_bits(rng.next());
        if let Some(d) = dyadic(x) {
            hits += 1;
            // The division is exact when numerator and denominator are
            // exact floats, which they are below 2^53.
            if d.numer().unsigned_abs() < 1 << 53 && d.denom() < 1 << 53 {
                assert_eq!(d.numer() as f64 / d.denom() as f64, x, "{x}");
            }
        }
    }
    // Random bit patterns are rarely small dyadics; make sure some are.
    for x in [1.5, -2.25, 1024.0, 0.0625, 1e10, 123456.789] {
        assert!(dyadic(x).is_some(), "{x}");
        hits += 1;
    }
    assert!(hits > 6);
}

// ------------------------------------------------------------ analyze

fn num() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

struct B(FunctionBuilder);

impl B {
    fn new(params: usize) -> B {
        B(FunctionBuilder::new(
            "f",
            vec![num(); params],
            vec![],
            Span::point(0),
        ))
    }

    fn op(&mut self, kind: OpKind, args: &[ValueId]) -> ValueId {
        self.0.op1(kind, args.to_vec(), num(), Span::point(0))
    }

    fn flt(&mut self, x: f64) -> ValueId {
        self.op(OpKind::ConstFloat { bits: x.to_bits() }, &[])
    }

    fn rat(&mut self, n: i64, d: i64) -> ValueId {
        self.op(OpKind::ConstRational(q(n, d)), &[])
    }

    fn finish(mut self) -> Function {
        self.0.op(OpKind::Return, vec![], vec![], Span::point(0));
        self.0.finish()
    }
}

fn op_of(f: &Function, v: ValueId) -> OpId {
    match f.value(v).def {
        ValueDef::OpResult { op, .. } => op,
        _ => panic!("not an op result"),
    }
}

#[test]
fn constants_are_exact_points() {
    let mut b = B::new(0);
    let half = b.flt(0.5);
    let third = b.rat(1, 3);
    let tenth = b.flt(0.1);
    let f = b.finish();
    let a = analyze(&f);
    let k = a.known(half).unwrap();
    assert_eq!(k.ideal, Some(r(1, 2)));
    assert_eq!(k.range, Range::point(0.5));
    assert!(k.faithful());
    let k = a.known(third).unwrap();
    assert_eq!(k.ideal, Some(r(1, 3)));
    assert!(!k.faithful());
    // 0.1 as SIR means it: the binary value, not 1/10.
    let k = a.known(tenth).unwrap();
    assert_ne!(k.ideal, Some(r(1, 10)));
    assert!(k.faithful());
}

#[test]
fn one_third_plus_one_sixth_is_exactly_one_half() {
    let mut b = B::new(0);
    let (one, three, six) = (b.flt(1.0), b.flt(3.0), b.flt(6.0));
    let a = b.op(OpKind::Div, &[one, three]);
    let c = b.op(OpKind::Div, &[one, six]);
    let s = b.op(OpKind::Add, &[a, c]);
    let f = b.finish();
    let an = analyze(&f);
    assert_eq!(
        an.known(a).unwrap().note().unwrap(),
        "over the reals the result is exactly 1/3; the f64 result 0.3333333333333333 is that \
         value rounded (`sky run` rounds at every step)"
    );
    let k = an.known(s).unwrap();
    assert_eq!(k.ideal, Some(r(1, 2)));
    assert_eq!(k.range, Range::point(0.5));
    assert!(k.faithful());
    assert_eq!(
        k.note().unwrap(),
        "over the reals the result is exactly 1/2, and the f64 result is exactly that value"
    );
}

#[test]
fn tenth_plus_two_tenths_is_rounded() {
    let mut b = B::new(0);
    let (one, ten, two) = (b.flt(1.0), b.flt(10.0), b.flt(2.0));
    let a = b.op(OpKind::Div, &[one, ten]);
    let c = b.op(OpKind::Div, &[two, ten]);
    let s = b.op(OpKind::Add, &[a, c]);
    let f = b.finish();
    let an = analyze(&f);
    let k = an.known(s).unwrap();
    assert_eq!(k.ideal, Some(r(3, 10)));
    assert_eq!(k.range, Range::point(0.1 + 0.2));
    assert!(!k.faithful());
    assert!(k.note().unwrap().contains("exactly 3/10"));
}

#[test]
fn roots_and_the_float_that_disagrees() {
    // sqrt(2) * sqrt(2) - 2 is exactly zero, and 4.4e-16 in f64: the
    // analysis must not say the float is zero.
    let mut b = B::new(0);
    let two = b.flt(2.0);
    let s = b.op(OpKind::Math(MathFn::Sqrt), &[two]);
    let p = b.op(OpKind::Mul, &[s, s]);
    let d = b.op(OpKind::Sub, &[p, two]);
    let f = b.finish();
    let an = analyze(&f);
    assert_eq!(an.known(s).unwrap().ideal, Some(root(2)));
    assert_eq!(an.known(p).unwrap().ideal, Some(Sym::integer(2)));
    let k = an.known(d).unwrap();
    assert_eq!(k.ideal, Some(Sym::integer(0)));
    assert_eq!(k.range, Range::point(2f64.sqrt() * 2f64.sqrt() - 2.0));
    assert!(k.is_positive(), "the f64 is 4.4e-16, not zero");
}

#[test]
fn sqrt_and_log_of_constants_cannot_fail() {
    let mut b = B::new(1);
    let x = b.0.params()[0];
    let two = b.flt(2.0);
    let sq = b.op(OpKind::Math(MathFn::Sqrt), &[two]);
    let lg = b.op(OpKind::Math(MathFn::Log), &[two]);
    let sq_x = b.op(OpKind::Math(MathFn::Sqrt), &[x]);
    let lg_x = b.op(OpKind::Math(MathFn::Log), &[x]);
    let zero = b.flt(0.0);
    let lg0 = b.op(OpKind::Math(MathFn::Log), &[zero]);
    let neg = b.flt(-4.0);
    let sq_neg = b.op(OpKind::Math(MathFn::Sqrt), &[neg]);
    let f = b.finish();
    let an = analyze(&f);
    assert_eq!(an.proof(op_of(&f, sq)), Some(Proof::Domain));
    assert_eq!(an.proof(op_of(&f, lg)), Some(Proof::Domain));
    assert_eq!(an.proof(op_of(&f, sq_x)), None);
    assert_eq!(an.proof(op_of(&f, lg_x)), None);
    assert_eq!(an.proof(op_of(&f, lg0)), None);
    assert_eq!(an.proof(op_of(&f, sq_neg)), None);
    assert!(an.known(lg).unwrap().is_positive());
    assert!(an.known(sq).unwrap().is_positive());
    let why = an.why(&f, op_of(&f, sq)).unwrap();
    assert!(why.contains("inside the domain of `sqrt`"), "{why}");
}

#[test]
fn division_by_a_nonzero_constant() {
    let mut b = B::new(1);
    let x = b.0.params()[0];
    let (two, half, zero) = (b.flt(2.0), b.flt(0.5), b.flt(0.0));
    let by_two = b.op(OpKind::Div, &[x, two]);
    // x / 0.5 doubles x: it can overflow.
    let by_half = b.op(OpKind::Div, &[x, half]);
    let by_zero = b.op(OpKind::Div, &[x, zero]);
    let one = b.flt(1.0);
    let inv_x = b.op(OpKind::Div, &[one, x]);
    let inv_two = b.op(OpKind::Inverse, &[two]);
    let tiny = b.flt(5e-324);
    let inv_tiny = b.op(OpKind::Inverse, &[tiny]);
    let times_half = b.op(OpKind::Mul, &[x, half]);
    let times_two = b.op(OpKind::Mul, &[x, two]);
    let plus_x = b.op(OpKind::Add, &[x, x]);
    let f = b.finish();
    let an = analyze(&f);
    let p = |v| an.proof(op_of(&f, v));
    assert_eq!(p(by_two), Some(Proof::NonzeroDivisor));
    assert_eq!(p(by_half), None);
    assert_eq!(p(by_zero), None);
    assert_eq!(p(inv_x), None);
    assert_eq!(p(inv_two), Some(Proof::NonzeroDivisor));
    assert_eq!(p(inv_tiny), None, "1/5e-324 overflows");
    assert_eq!(p(times_half), Some(Proof::Bounded));
    assert_eq!(p(times_two), None);
    assert_eq!(p(plus_x), None);
    let why = an.why(&f, op_of(&f, by_two)).unwrap();
    assert!(why.starts_with("the divisor is 2 (never zero)"), "{why}");
}

#[test]
fn the_range_of_an_if_is_the_hull_of_its_arms() {
    let mut b = B::new(0);
    let cond =
        b.0.op1(OpKind::ConstBool(true), vec![], Type::Bool, Span::point(0));
    let (iff, res) = b.0.op(OpKind::If, vec![cond], vec![num()], Span::point(0));
    let outer = b.0.insertion_block();
    for x in [1.0, 4.0] {
        let reg = b.0.add_region(iff);
        let blk = b.0.add_block(reg, vec![]);
        b.0.set_insertion_block(blk);
        let v = b.flt(x);
        b.0.op(OpKind::Yield, vec![v], vec![], Span::point(0));
    }
    b.0.set_insertion_block(outer);
    let f = b.finish();
    let an = analyze(&f);
    let k = an.known(res[0]).unwrap();
    assert_eq!(k.range, Range { lo: 1.0, hi: 4.0 });
    assert!(k.is_positive());
    assert_eq!(k.ideal, None);
}

#[test]
fn unknown_types_are_not_analysed() {
    let mut b = B(FunctionBuilder::new(
        "f",
        vec![Type::Unknown],
        vec![],
        Span::point(0),
    ));
    let x = b.0.params()[0];
    let two = b.flt(2.0);
    let d =
        b.0.op1(OpKind::Div, vec![x, two], Type::Unknown, Span::point(0));
    let f = b.finish();
    let an = analyze(&f);
    assert!(an.known(x).is_none());
    assert!(an.known(d).is_none());
    assert_eq!(an.proof(op_of(&f, d)), None);
}

/// Analyses the single function of the SIR text and returns the proof of
/// the operation whose result is named `name`.
fn proof_of(text: &str, name: &str) -> (Option<Proof>, Option<Known>) {
    let m = crate::parse::parse(text).expect("parses");
    crate::verify::verify(&m).expect("valid");
    let f = &m.functions[0];
    let v = f
        .values
        .iter()
        .position(|d| d.name.as_deref() == Some(name))
        .map(|i| ValueId(i as u32))
        .expect("named value");
    let an = analyze(f);
    (an.proof(op_of(f, v)), an.known(v).cloned())
}

const SHAPES: &str = "sir edition 2026

fn @f(%v: num[1; (3)], %a: num[1; (2, 3)], %w: num[1; (N#0)], %u: unknown) -> () effects(fail) {
  %c0 = const.float 0 : num[1; ()]
  %c2 = const.float 2 : num[1; ()]
  %c3 = const.float 3 : num[1; ()]
  %c4 = const.float 4 : num[1; ()]
  %cm = const.float -1 : num[1; ()]
  %h = const.float 0.5 : num[1; ()]
  %ok = index %v, %c2 : num[1; ()]
  %past = index %v, %c3 : num[1; ()]
  %neg = index %v, %cm : num[1; ()]
  %frac = index %v, %h : num[1; ()]
  %cell = index %a, %c0, %c2 : num[1; ()]
  %cellx = index %a, %c2, %c0 : num[1; ()]
  %row = index %a, %c0 : num[1; (3)]
  %sym = index %w, %c0 : num[1; ()]
  %dyn = index %u, %c0 : unknown
  %run = slice[lo:hi] %v, %c0, %c2 : num[1; (2)]
  %empty = slice[lo:hi] %v, %c2, %c2 : num[1; (0)]
  %over = slice[lo:hi] %v, %c0, %c4 : num[1; (3)]
  %n = len %v : num[1; ()]
  %nw = len %w : num[1; ()]
  %nu = len %u : num[1; ()]
  return
}
";

#[test]
fn indices_inside_a_static_shape_are_proven_in_bounds() {
    let p = |name| proof_of(SHAPES, name).0;
    assert_eq!(p("ok"), Some(Proof::InBounds));
    assert_eq!(p("cell"), Some(Proof::InBounds));
    assert_eq!(p("row"), Some(Proof::InBounds));
    assert_eq!(p("run"), Some(Proof::InBounds));
    // Out of bounds, negative, fractional: they fail, so no proof.
    assert_eq!(p("past"), None);
    assert_eq!(p("neg"), None);
    assert_eq!(p("frac"), None);
    // The second axis has 3 entries, the first 2: index 2 on axis 0 fails.
    assert_eq!(p("cellx"), None);
    // A symbolic size, an unknown type, an empty or too long range: no proof.
    assert_eq!(p("sym"), None);
    assert_eq!(p("dyn"), None);
    assert_eq!(p("empty"), None);
    assert_eq!(p("over"), None);
}

#[test]
fn len_of_a_typed_array_cannot_fail_and_a_static_one_is_known() {
    let (proof, known) = proof_of(SHAPES, "n");
    assert_eq!(proof, Some(Proof::InBounds));
    let k = known.expect("a static length is known");
    assert_eq!(k.range.as_point(), Some(3.0));
    assert!(k.faithful());
    // A symbolic length is not known, but `len` of a typed array still succeeds.
    let (proof, known) = proof_of(SHAPES, "nw");
    assert_eq!(proof, Some(Proof::InBounds));
    assert_eq!(known.unwrap().range, Range::FINITE);
    // An unknown-typed operand may not be an array.
    assert_eq!(proof_of(SHAPES, "nu").0, None);
}
