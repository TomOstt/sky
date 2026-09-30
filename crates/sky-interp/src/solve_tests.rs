//! Tests of the algorithm variants (`crate::solve`): the solves agree with
//! the inverse to rounding, fail exactly when the inverse does, and the
//! structured products (`diag_mul`, `matmul_ex`, the Gram product) give the
//! **bits** of `matmul`.

use proptest::prelude::*;
use sky_diag::Span;
use sky_types::{Algorithm, Side, SolveMethod};
use sky_units::Dimension;

use crate::linalg;
use crate::solve::{diag_mul, lu_factor, matmul_ex, solve, solve_right};
use crate::{array, RuntimeError, Value};

fn sp(a: u32, b: u32) -> Span {
    Span::new(a, b)
}

const WHOLE: Span = Span { start: 0, end: 30 };
const LEFT: Span = Span { start: 0, end: 10 };
const RIGHT: Span = Span { start: 20, end: 30 };

fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Value {
    Value::array(vec![rows, cols], data, Dimension::DIMENSIONLESS)
}

fn vector(data: Vec<f64>) -> Value {
    Value::array(vec![data.len()], data, Dimension::DIMENSIONLESS)
}

/// The shape, the bits of every entry and the dimension: what "the same
/// value" means here (`-0.0` and `0.0` differ).
fn bits(v: &Value) -> (Vec<usize>, Vec<u64>, Dimension) {
    match v {
        Value::Array { dims, data, dim } => (
            dims.clone(),
            data.iter().map(|x| x.to_bits()).collect(),
            *dim,
        ),
        other => panic!("not an array: {other:?}"),
    }
}

#[track_caller]
fn same_bits(a: &Result<Value, RuntimeError>, b: &Result<Value, RuntimeError>) {
    match (a, b) {
        (Ok(x), Ok(y)) => assert_eq!(bits(x), bits(y)),
        (Err(x), Err(y)) => assert_eq!(x, y),
        _ => panic!("one succeeded and one failed: {a:?} vs {b:?}"),
    }
}

fn data(v: &Value) -> &[f64] {
    match v {
        Value::Array { data, .. } => data,
        other => panic!("not an array: {other:?}"),
    }
}

#[track_caller]
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{a:?} vs {b:?} (tolerance {tol})"
        );
    }
}

// ----- generators -----

/// Entries in `[-4, 4]` on a grid of quarters (exactly representable).
fn quarter() -> impl Strategy<Value = f64> {
    (-16i32..=16).prop_map(|i| f64::from(i) / 4.0)
}

/// Any finite double a program could hold, biased towards the awkward ones:
/// signed zeros, subnormals, huge values that overflow when multiplied.
fn wild() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => quarter(),
        2 => Just(-0.0),
        2 => Just(0.0),
        1 => Just(5e-324),
        1 => Just(-5e-324),
        1 => Just(1e-310),
        1 => Just(1e200),
        1 => Just(-1e200),
        1 => -1e6f64..1e6,
    ]
}

/// A diagonally dominant `n × n` matrix: well conditioned, never singular.
fn dominant(n: usize) -> impl Strategy<Value = Vec<f64>> {
    (
        proptest::collection::vec(-1.0f64..1.0, n * n),
        proptest::collection::vec(any::<bool>(), n),
    )
        .prop_map(move |(mut a, signs)| {
            for i in 0..n {
                let d = n as f64 + 1.0;
                a[i * n + i] += if signs[i] { d } else { -d };
            }
            a
        })
}

/// A symmetric positive-definite matrix: `M·Mᵀ + n·I`, exactly symmetric
/// because `linalg::matmul` multiplies and adds in the same order for `(i, j)`
/// and `(j, i)`.
fn spd(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-1.0f64..1.0, n * n).prop_map(move |m| {
        let mut mt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                mt[j * n + i] = m[i * n + j];
            }
        }
        let mut a = linalg::matmul(&m, n, n, &mt, n);
        for i in 0..n {
            a[i * n + i] += n as f64;
        }
        a
    })
}

fn rhs(n: usize, k: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0f64..10.0, n * k)
}

fn rhs_value(n: usize, k: usize, vector_rhs: bool, data: Vec<f64>) -> Value {
    if vector_rhs {
        assert_eq!(k, 1);
        vector(data)
    } else {
        mat(n, k, data)
    }
}

// ----- the solves agree with the inverse, to rounding -----

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn lu_solve_agrees_with_inverse_then_matmul(
        (n, k, vector_rhs, a, b) in (1usize..=6, 1usize..=3, any::<bool>()).prop_flat_map(
            |(n, k, v)| {
                let k = if v { 1 } else { k };
                (Just(n), Just(k), Just(v), dominant(n), rhs(n, k))
            }
        )
    ) {
        let av = mat(n, n, a);
        let bv = rhs_value(n, k, vector_rhs, b);
        let x = solve(SolveMethod::Lu, &av, &bv, WHOLE, LEFT, RIGHT).unwrap();
        let inv = array::inverse(&av, WHOLE, LEFT).unwrap();
        let y = array::matmul(&inv, &bv, WHOLE, LEFT, RIGHT).unwrap();
        prop_assert_eq!(bits(&x).0, bits(&y).0);
        close(data(&x), data(&y), 1e-10);
    }

    #[test]
    fn lu_solve_right_agrees_with_matmul_by_the_inverse(
        (n, m, vector_lhs, a, x) in (1usize..=6, 1usize..=3, any::<bool>()).prop_flat_map(
            |(n, m, v)| {
                let m = if v { 1 } else { m };
                (Just(n), Just(m), Just(v), dominant(n), rhs(m, n))
            }
        )
    ) {
        let av = mat(n, n, a);
        let xv = if vector_lhs { vector(x) } else { mat(m, n, x) };
        let z = solve_right(SolveMethod::Lu, &xv, &av, WHOLE, LEFT, RIGHT).unwrap();
        let inv = array::inverse(&av, WHOLE, RIGHT).unwrap();
        let y = array::matmul(&xv, &inv, WHOLE, LEFT, RIGHT).unwrap();
        prop_assert_eq!(bits(&z).0, bits(&y).0);
        close(data(&z), data(&y), 1e-10);
    }

    #[test]
    fn cholesky_solves_agree_with_the_inverse_on_both_sides(
        (n, m, a, b, x) in (1usize..=5, 1usize..=3).prop_flat_map(|(n, m)| {
            (Just(n), Just(m), spd(n), rhs(n, m), rhs(m, n))
        })
    ) {
        let av = mat(n, n, a);
        let inv = array::inverse(&av, WHOLE, LEFT).unwrap();
        let bv = mat(n, m, b);
        let left = solve(SolveMethod::Cholesky, &av, &bv, WHOLE, LEFT, RIGHT).unwrap();
        let want = array::matmul(&inv, &bv, WHOLE, LEFT, RIGHT).unwrap();
        close(data(&left), data(&want), 1e-10);
        let xv = mat(m, n, x);
        let right = solve_right(SolveMethod::Cholesky, &xv, &av, WHOLE, LEFT, RIGHT).unwrap();
        let want = array::matmul(&xv, &inv, WHOLE, LEFT, RIGHT).unwrap();
        prop_assert_eq!(bits(&right).0, bits(&want).0);
        close(data(&right), data(&want), 1e-10);
        // A vector on the left is a row: the same as its one-row matrix.
        let row: Vec<f64> = data(&xv)[..n].to_vec();
        let as_vector = solve_right(SolveMethod::Cholesky, &vector(row.clone()), &av, WHOLE, LEFT, RIGHT).unwrap();
        let as_matrix = solve_right(SolveMethod::Cholesky, &mat(1, n, row), &av, WHOLE, LEFT, RIGHT).unwrap();
        prop_assert_eq!(bits(&as_vector).1, bits(&as_matrix).1);
    }

    #[test]
    fn a_solve_undoes_the_product(
        (n, a, x) in (1usize..=6).prop_flat_map(|n| (Just(n), dominant(n), rhs(n, 1)))
    ) {
        // A⁻¹ (A x) = x and (x A) A⁻¹ = x, to rounding.
        let av = mat(n, n, a);
        let xv = vector(x);
        let ax = array::matmul(&av, &xv, WHOLE, LEFT, RIGHT).unwrap();
        let back = solve(SolveMethod::Lu, &av, &ax, WHOLE, LEFT, RIGHT).unwrap();
        close(data(&back), data(&xv), 1e-10);
        let xa = array::matmul(&xv, &av, WHOLE, LEFT, RIGHT).unwrap();
        let back = solve_right(SolveMethod::Lu, &xa, &av, WHOLE, LEFT, RIGHT).unwrap();
        close(data(&back), data(&xv), 1e-10);
    }
}

#[test]
fn a_solve_has_the_dimension_of_b_over_a() {
    let a = Value::array(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0], Dimension::LENGTH);
    let b = Value::array(vec![2], vec![1.0, 1.0], Dimension::TIME);
    let want = Dimension::TIME.checked_div(Dimension::LENGTH).unwrap();
    for method in [SolveMethod::Lu, SolveMethod::Diagonal] {
        let x = solve(method, &a, &b, WHOLE, LEFT, RIGHT).unwrap();
        assert_eq!(bits(&x).2, want);
        let z = solve_right(method, &b, &a, WHOLE, LEFT, RIGHT).unwrap();
        assert_eq!(bits(&z).2, want);
    }
}

#[test]
fn shapes_that_do_not_fit_are_errors_like_matmul() {
    let a = mat(3, 3, vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    let short = vector(vec![1.0, 2.0]);
    for method in [SolveMethod::Lu, SolveMethod::Diagonal] {
        let e = solve(method, &a, &short, WHOLE, LEFT, RIGHT).unwrap_err();
        assert!(e.message.contains("inner dimensions"), "{e}");
        let e = solve_right(method, &short, &a, WHOLE, LEFT, RIGHT).unwrap_err();
        assert!(e.message.contains("inner dimensions"), "{e}");
    }
    let wide = mat(2, 3, vec![1.0; 6]);
    let e = solve(SolveMethod::Lu, &wide, &short, WHOLE, LEFT, RIGHT).unwrap_err();
    assert!(e.message.contains("square"), "{e}");
}

// ----- singular exactly when the inverse is -----

fn small_int_matrix() -> impl Strategy<Value = (usize, Vec<f64>)> {
    (2usize..=4).prop_flat_map(|n| {
        (
            Just(n),
            proptest::collection::vec(-3i32..=3, n * n)
                .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<f64>>()),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    /// The LU forward phase is the Gauss-Jordan one operation for operation,
    /// so the pivots are the same numbers and the verdict is the same,
    /// including for rank-deficient and near-singular matrices.
    #[test]
    fn solve_is_singular_exactly_when_inverse_is(
        (n, a) in small_int_matrix(),
        scale in prop_oneof![Just(1.0), Just(1e-8), Just(3.0), Just(1e10)],
        row in 0usize..4,
    ) {
        // Scaling one row by a power-of-ten-ish factor makes near-singular
        // and badly scaled cases the tolerance has to judge.
        let mut a = a;
        let row = row % n;
        for j in 0..n {
            a[row * n + j] *= scale;
        }
        let singular_by_inverse = linalg::inverse(&a, n).is_err();
        prop_assert_eq!(lu_factor(&a, n).is_err(), singular_by_inverse);
        let av = mat(n, n, a);
        let b = vector(vec![1.0; n]);
        let solved = solve(SolveMethod::Lu, &av, &b, WHOLE, LEFT, RIGHT);
        let inverted = array::inverse(&av, WHOLE, LEFT);
        prop_assert_eq!(solved.is_err(), inverted.is_err());
        if let (Err(s), Err(i)) = (&solved, &inverted) {
            // The same error, word for word: message, label, notes, labels.
            prop_assert_eq!(s, i);
        }
        let right = solve_right(SolveMethod::Lu, &b, &av, WHOLE, RIGHT, LEFT);
        prop_assert_eq!(right.is_err(), inverted.is_err());
    }
}

#[test]
fn the_classic_rounding_size_singular_matrices() {
    // [[1,2,3],[4,5,6],[7,8,9]]: the elimination leaves a pivot of rounding
    // size, which both reject.
    let a = mat(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = vector(vec![1.0, 2.0, 3.0]);
    let e = solve(SolveMethod::Lu, &a, &b, WHOLE, LEFT, RIGHT).unwrap_err();
    assert_eq!(e, array::inverse(&a, WHOLE, LEFT).unwrap_err());
    assert_eq!(e.message, "the matrix is singular");
    // Exactly singular.
    let a = mat(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    let e = solve(SolveMethod::Lu, &a, &b_of(2), WHOLE, LEFT, RIGHT).unwrap_err();
    assert_eq!(e, array::inverse(&a, WHOLE, LEFT).unwrap_err());
    // Not singular: the same numbers with a real pivot.
    let a = mat(2, 2, vec![1.0, 2.0, 2.0, 5.0]);
    assert!(solve(SolveMethod::Lu, &a, &b_of(2), WHOLE, LEFT, RIGHT).is_ok());
}

fn b_of(n: usize) -> Value {
    vector(vec![1.0; n])
}

// ----- Diagonal -----

fn diagonal_matrix(d: &[f64]) -> Value {
    let n = d.len();
    let mut a = vec![0.0; n * n];
    for (i, x) in d.iter().enumerate() {
        a[i * n + i] = *x;
    }
    mat(n, n, a)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// One correctly rounded division per entry, plus the `+ 0.0` that
    /// turns `-0.0` into `0.0`, on both sides.
    #[test]
    fn a_diagonal_solve_is_the_division(
        (n, k, d, b, x) in (1usize..=5, 1usize..=3).prop_flat_map(|(n, k)| {
            (
                Just(n),
                Just(k),
                proptest::collection::vec(
                    prop_oneof![(1i32..=16).prop_map(|i| f64::from(i) / 4.0), (-16i32..=-1).prop_map(|i| f64::from(i) / 4.0)],
                    n,
                ),
                proptest::collection::vec(wild(), n * k),
                proptest::collection::vec(wild(), k * n),
            )
        })
    ) {
        let a = diagonal_matrix(&d);
        let bv = mat(n, k, b.clone());
        let left = solve(SolveMethod::Diagonal, &a, &bv, WHOLE, LEFT, RIGHT);
        let mut want = Vec::new();
        for i in 0..n {
            for j in 0..k {
                want.push((b[i * k + j] / d[i] + 0.0).to_bits());
            }
        }
        match left {
            Ok(v) => prop_assert_eq!(bits(&v).1, want),
            Err(e) => prop_assert!(e.message.contains("overflow"), "{e}"),
        }
        let xv = mat(k, n, x.clone());
        let right = solve_right(SolveMethod::Diagonal, &xv, &a, WHOLE, LEFT, RIGHT);
        let mut want = Vec::new();
        for i in 0..k {
            for j in 0..n {
                want.push((x[i * n + j] / d[j] + 0.0).to_bits());
            }
        }
        match right {
            Ok(v) => prop_assert_eq!(bits(&v).1, want),
            Err(e) => prop_assert!(e.message.contains("overflow"), "{e}"),
        }
    }
}

#[test]
fn a_diagonal_pivot_is_singular_by_the_inverses_test() {
    // 1e-30 against a largest entry of 1: below n·ε·max, so singular, as
    // the inverse says (a division would have happily returned 1e30).
    let a = diagonal_matrix(&[1.0, 1e-30]);
    let b = vector(vec![1.0, 1.0]);
    let e = solve(SolveMethod::Diagonal, &a, &b, WHOLE, LEFT, RIGHT).unwrap_err();
    assert_eq!(e, array::inverse(&a, WHOLE, LEFT).unwrap_err());
    let e = solve_right(SolveMethod::Diagonal, &b, &a, WHOLE, RIGHT, LEFT).unwrap_err();
    assert_eq!(e, array::inverse(&a, WHOLE, LEFT).unwrap_err().clone());
    // A zero on the diagonal.
    let a = diagonal_matrix(&[3.0, 0.0]);
    assert_eq!(
        solve(SolveMethod::Diagonal, &a, &b, WHOLE, LEFT, RIGHT).is_err(),
        array::inverse(&a, WHOLE, LEFT).is_err()
    );
    // And a matrix that is fine.
    let a = diagonal_matrix(&[3.0, 1e-3]);
    assert!(solve(SolveMethod::Diagonal, &a, &b, WHOLE, LEFT, RIGHT).is_ok());
    assert!(array::inverse(&a, WHOLE, LEFT).is_ok());
}

#[test]
fn a_broken_diagonal_promise_names_the_entry() {
    let a = mat(2, 2, vec![1.0, 0.0, 0.5, 2.0]);
    let b = vector(vec![1.0, 1.0]);
    let e = solve(SolveMethod::Diagonal, &a, &b, WHOLE, LEFT, RIGHT).unwrap_err();
    assert_eq!(e.message, "the matrix is not diagonal");
    assert_eq!(e.span, LEFT);
    assert!(
        e.label.as_deref().unwrap().contains("entry [1, 0]"),
        "{e:?}"
    );
    let e = solve_right(SolveMethod::Diagonal, &b, &a, WHOLE, RIGHT, LEFT).unwrap_err();
    assert_eq!(e.message, "the matrix is not diagonal");
    assert_eq!(e.span, LEFT);
    let e = diag_mul(Side::Left, &a, &b, WHOLE, LEFT, RIGHT).unwrap_err();
    assert_eq!(e.message, "the matrix is not diagonal");
    assert!(
        e.label.as_deref().unwrap().contains("entry [1, 0]"),
        "{e:?}"
    );
    // A negative zero off the diagonal is zero.
    let a = mat(2, 2, vec![1.0, -0.0, 0.0, 2.0]);
    assert!(solve(SolveMethod::Diagonal, &a, &b, WHOLE, LEFT, RIGHT).is_ok());
}

#[test]
fn a_broken_cholesky_promise_is_an_error_on_the_right_side_too() {
    let a = mat(2, 2, vec![1.0, 2.0, 0.5, 1.0]);
    let x = vector(vec![1.0, 1.0]);
    let e = solve_right(SolveMethod::Cholesky, &x, &a, WHOLE, RIGHT, LEFT).unwrap_err();
    assert_eq!(e.message, "the matrix is not symmetric");
    assert_eq!(e.span, LEFT);
    let a = mat(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
    let e = solve_right(SolveMethod::Cholesky, &x, &a, WHOLE, RIGHT, LEFT).unwrap_err();
    assert_eq!(e.message, "the matrix is not positive definite");
}

// ----- exactness: the structured products give the bits of matmul -----

fn diagonal_with_offs(d: Vec<f64>, offs: Vec<bool>) -> Value {
    let n = d.len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j {
                d[i]
            } else if offs[i * n + j] {
                -0.0
            } else {
                0.0
            };
        }
    }
    Value::array(vec![n, n], a, Dimension::LENGTH)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// `D @ X` and `X @ D` as a scaling: the bits of `matmul`, the same
    /// error when an entry overflows, negative zeros and subnormals
    /// included, for matrices and vectors on either side.
    #[test]
    fn diag_mul_is_bit_identical_to_matmul(
        (n, k, d, offs, x, y, shape) in (1usize..=4, 1usize..=4).prop_flat_map(|(n, k)| {
            (
                Just(n),
                Just(k),
                proptest::collection::vec(wild(), n),
                proptest::collection::vec(any::<bool>(), n * n),
                proptest::collection::vec(wild(), n * k),
                proptest::collection::vec(wild(), k * n),
                0u8..2,
            )
        })
    ) {
        let dv = diagonal_with_offs(d, offs);
        let xm = Value::array(vec![n, k], x.clone(), Dimension::TIME);
        let ym = Value::array(vec![k, n], y.clone(), Dimension::TIME);
        let xv = Value::array(vec![n], x[..n].to_vec(), Dimension::TIME);
        let (left, right) = if shape == 0 { (xm, ym) } else { (xv.clone(), xv) };
        let got = diag_mul(Side::Left, &dv, &left, WHOLE, LEFT, RIGHT);
        let want = array::matmul(&dv, &left, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
        let got = diag_mul(Side::Right, &dv, &right, WHOLE, RIGHT, LEFT);
        let want = array::matmul(&right, &dv, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
    }

    #[test]
    fn matmul_ex_transposes_are_transpose_then_matmul(
        (m, k, n, a, b, ta, tb) in (1usize..=4, 1usize..=4, 1usize..=4).prop_flat_map(|(m, k, n)| {
            (
                Just(m),
                Just(k),
                Just(n),
                proptest::collection::vec(wild(), m * k),
                proptest::collection::vec(wild(), k * n),
                any::<bool>(),
                any::<bool>(),
            )
        })
    ) {
        // The stored shapes are the transposed ones for a flagged operand.
        let l = if ta { Value::array(vec![k, m], transposed(&a, m, k), Dimension::LENGTH) } else { Value::array(vec![m, k], a.clone(), Dimension::LENGTH) };
        let r = if tb { Value::array(vec![n, k], transposed(&b, k, n), Dimension::TIME) } else { Value::array(vec![k, n], b.clone(), Dimension::TIME) };
        let got = matmul_ex(ta, tb, Algorithm::Auto, &l, &r, WHOLE, LEFT, RIGHT);
        let l2 = if ta { array::transpose(&l, WHOLE).unwrap() } else { l.clone() };
        let r2 = if tb { array::transpose(&r, WHOLE).unwrap() } else { r.clone() };
        let want = array::matmul(&l2, &r2, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
        // The plain product of the untransposed A and B is the same thing.
        let plain = array::matmul(
            &Value::array(vec![m, k], a, Dimension::LENGTH),
            &Value::array(vec![k, n], b, Dimension::TIME),
            WHOLE, LEFT, RIGHT,
        );
        same_bits(&got, &plain);
        // An algorithm class changes nothing.
        let class = Algorithm::Gemm;
        same_bits(&matmul_ex(ta, tb, class, &l, &r, WHOLE, LEFT, RIGHT), &want);
    }

    /// The Gram product computed as one triangle and mirrored has the bits
    /// of the full product, for `Aᵀ @ A` and `A @ Aᵀ`.
    #[test]
    fn syrk_is_bit_identical_to_the_full_product(
        (m, n, a) in (1usize..=5, 1usize..=5).prop_flat_map(|(m, n)| {
            (Just(m), Just(n), proptest::collection::vec(wild(), m * n))
        })
    ) {
        let av = Value::array(vec![m, n], a, Dimension::LENGTH);
        let at = array::transpose(&av, WHOLE).unwrap();
        // Aᵀ @ A
        let got = matmul_ex(true, false, Algorithm::Syrk, &av, &av, WHOLE, LEFT, RIGHT);
        let want = array::matmul(&at, &av, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
        // A @ Aᵀ
        let got = matmul_ex(false, true, Algorithm::Syrk, &av, &av, WHOLE, LEFT, RIGHT);
        let want = array::matmul(&av, &at, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
        // And the result is bitwise symmetric.
        if let Ok(Value::Array { dims, data, .. }) = &got {
            for i in 0..dims[0] {
                for j in 0..dims[0] {
                    prop_assert_eq!(data[i * dims[0] + j].to_bits(), data[j * dims[0] + i].to_bits());
                }
            }
        }
    }
}

fn transposed(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    for j in 0..cols {
        for i in 0..rows {
            out.push(a[i * cols + j]);
        }
    }
    out
}

#[test]
fn syrk_keeps_the_signs_of_zeros_and_subnormals() {
    let a = mat(2, 2, vec![-0.0, 5e-324, 0.0, -5e-324]);
    let at = array::transpose(&a, WHOLE).unwrap();
    for (ta, tb, l, r) in [(true, false, &at, &a), (false, true, &a, &at)] {
        let got = matmul_ex(ta, tb, Algorithm::Syrk, &a, &a, WHOLE, LEFT, RIGHT);
        let want = array::matmul(l, r, WHOLE, LEFT, RIGHT);
        same_bits(&got, &want);
    }
}

#[test]
fn syrk_overflow_is_matmuls_error() {
    let a = mat(2, 2, vec![1e200, 1.0, 1.0, 1.0]);
    let at = array::transpose(&a, WHOLE).unwrap();
    let got = matmul_ex(true, false, Algorithm::Syrk, &a, &a, WHOLE, LEFT, RIGHT);
    let want = array::matmul(&at, &a, WHOLE, LEFT, RIGHT);
    assert!(want.is_err());
    same_bits(&got, &want);
}

#[test]
fn syrk_falls_back_when_the_operands_are_not_a_matrix_and_its_transpose() {
    let a = mat(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // A different second operand: the plain product, not a wrong mirror.
    let b = mat(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
    let got = matmul_ex(true, false, Algorithm::Syrk, &a, &b, WHOLE, LEFT, RIGHT);
    let want = array::matmul(
        &array::transpose(&a, WHOLE).unwrap(),
        &b,
        WHOLE,
        LEFT,
        RIGHT,
    );
    same_bits(&got, &want);
    // Two flags, or none: nothing to mirror.
    let got = matmul_ex(true, true, Algorithm::Syrk, &a, &a, WHOLE, LEFT, RIGHT);
    let want = array::matmul(
        &array::transpose(&a, WHOLE).unwrap(),
        &array::transpose(&a, WHOLE).unwrap(),
        WHOLE,
        LEFT,
        RIGHT,
    );
    same_bits(&got, &want);
    // Operands whose shapes cannot multiply are matmul's error.
    let got = matmul_ex(false, false, Algorithm::Syrk, &a, &a, sp(0, 5), LEFT, RIGHT);
    assert!(got.unwrap_err().message.contains("inner dimensions"));
}

#[test]
fn diag_mul_defers_to_matmul_for_shapes_that_do_not_fit() {
    let d = diagonal_matrix(&[1.0, 2.0, 3.0]);
    let x = mat(2, 2, vec![1.0; 4]);
    let got = diag_mul(Side::Left, &d, &x, WHOLE, LEFT, RIGHT);
    let want = array::matmul(&d, &x, WHOLE, LEFT, RIGHT);
    same_bits(&got, &want);
    let got = diag_mul(Side::Right, &d, &x, WHOLE, LEFT, RIGHT);
    let want = array::matmul(&x, &d, WHOLE, RIGHT, LEFT);
    same_bits(&got, &want);
    assert!(got.is_err());
}

#[test]
fn diag_mul_overflow_is_matmuls_error() {
    let d = diagonal_matrix(&[1e200, 1.0]);
    let x = mat(2, 1, vec![1e200, 1.0]);
    let got = diag_mul(Side::Left, &d, &x, WHOLE, LEFT, RIGHT);
    let want = array::matmul(&d, &x, WHOLE, LEFT, RIGHT);
    assert_eq!(want.as_ref().unwrap_err().message, "arithmetic overflow");
    same_bits(&got, &want);
}
