//! The kernels equal the plain reference loops bit for bit: on random
//! shapes (awkward sizes 1, 7, 33, 65 among them), on random data that
//! includes zeros, negative zeros, huge and tiny magnitudes and values that
//! overflow to infinity, and at every thread count.

use proptest::prelude::*;
use sky_linalg::{reference, LinalgError, Lu};

/// A small deterministic generator (SplitMix64), so a failing case is
/// reproduced by its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// How wild the entries are.
#[derive(Clone, Copy, Debug)]
enum Flavor {
    /// Ordinary numbers in [-1, 1).
    Plain,
    /// Small integers (exact sums, many exact zeros).
    Ints,
    /// Mixed magnitudes from 1e-150 to 1e150, zeros and negative zeros.
    Wide,
    /// Magnitudes near the overflow threshold, so products overflow.
    Huge,
    /// Mostly exact zeros among small integers: multipliers that are zero,
    /// pivots that tie, matrices that are singular.
    Sparse,
}

fn entry(rng: &mut Rng, flavor: Flavor) -> f64 {
    let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
    match flavor {
        Flavor::Plain => 2.0 * rng.unit() - 1.0,
        Flavor::Ints => rng.below(7) as f64 - 3.0,
        Flavor::Wide => match rng.below(10) {
            0 => 0.0,
            1 => -0.0,
            2 => sign * 10f64.powi(rng.below(301) as i32 - 150),
            _ => sign * rng.unit() * 10f64.powi(rng.below(21) as i32 - 10),
        },
        Flavor::Sparse => match rng.below(10) {
            0..=6 => 0.0,
            _ => rng.below(7) as f64 - 3.0,
        },
        Flavor::Huge => match rng.below(6) {
            0 => sign * f64::MAX,
            1 => sign * 1e154 * (1.0 + rng.unit()),
            2 => sign * 1e300 * rng.unit(),
            3 => 0.0,
            _ => sign * rng.unit(),
        },
    }
}

fn matrix(rng: &mut Rng, len: usize, flavor: Flavor) -> Vec<f64> {
    (0..len).map(|_| entry(rng, flavor)).collect()
}

/// Bit equality, with any NaN equal to any NaN (the payload of a NaN made
/// by `inf - inf` is the hardware's business, and never observable in Sky,
/// where results must be finite).
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn flavors() -> impl Strategy<Value = Flavor> {
    prop_oneof![
        Just(Flavor::Plain),
        Just(Flavor::Ints),
        Just(Flavor::Wide),
        Just(Flavor::Huge),
        Just(Flavor::Sparse)
    ]
}

/// Sizes that straddle every block edge of the kernels.
fn awkward() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => prop::sample::select(vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]),
        1 => 1usize..70,
    ]
}

fn threads() -> impl Strategy<Value = usize> {
    prop::sample::select(vec![1usize, 2, 3, 8])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn gemm_equals_the_naive_loops(
        seed in any::<u64>(), m in awkward(), k in awkward(), n in awkward(),
        flavor in flavors(), t in threads(),
    ) {
        let mut rng = Rng(seed);
        let a = matrix(&mut rng, m * k, flavor);
        let b = matrix(&mut rng, k * n, flavor);
        let want = reference::matmul(&a, m, k, &b, n);
        let got = sky_linalg::gemm_with(&a, m, k, &b, n, t);
        prop_assert!(same(&got, &want), "{m}x{k} @ {k}x{n} ({flavor:?}, {t} threads)");
    }

    #[test]
    fn dot_and_sum_squares_equal_the_fixed_order(
        seed in any::<u64>(), n in awkward(), flavor in flavors(), long in any::<bool>(),
    ) {
        let n = if long { n * 61 + 5 } else { n };
        let mut rng = Rng(seed);
        let a = matrix(&mut rng, n, flavor);
        let b = matrix(&mut rng, n, flavor);
        let want = reference::dot(&a, &b);
        let got = sky_linalg::dot(&a, &b);
        prop_assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
        let want = reference::dot(&a, &a);
        let got = sky_linalg::sum_squares(&a);
        prop_assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
        // The scaled sum of squares is the same reduction of (x/s)·(x/s).
        let scale = 1.5 + rng.unit();
        let y: Vec<f64> = a.iter().map(|x| x / scale).collect();
        let want = reference::dot(&y, &y);
        let got = sky_linalg::sum_squares_scaled(&a, scale);
        prop_assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
    }
}

/// A random symmetric positive-definite matrix `B·Bᵀ + n·I`, exactly
/// symmetric by construction.
fn spd(rng: &mut Rng, n: usize, flavor: Flavor) -> Vec<f64> {
    let b = matrix(rng, n * n, flavor);
    let bt: Vec<f64> = (0..n * n).map(|x| b[(x % n) * n + x / n]).collect();
    let mut a = reference::matmul(&b, n, n, &bt, n);
    for i in 0..n {
        a[i * n + i] += n as f64;
    }
    a
}

/// A random exactly symmetric matrix (maybe not positive definite).
fn symmetric(rng: &mut Rng, n: usize, flavor: Flavor) -> Vec<f64> {
    let mut a = matrix(rng, n * n, flavor);
    for i in 0..n {
        for j in 0..i {
            a[i * n + j] = a[j * n + i];
        }
    }
    a
}

fn same_result<T: PartialEq + std::fmt::Debug>(
    got: &Result<Vec<f64>, T>,
    want: &Result<Vec<f64>, T>,
) -> bool {
    match (got, want) {
        (Ok(g), Ok(w)) => same(g, w),
        (Err(g), Err(w)) => g == w,
        _ => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn cholesky_and_its_solve_equal_the_plain_algorithm(
        seed in any::<u64>(), n in awkward(), k in awkward(),
        flavor in flavors(), t in threads(), pd in any::<bool>(),
    ) {
        let mut rng = Rng(seed);
        let a = if pd { spd(&mut rng, n, flavor) } else { symmetric(&mut rng, n, flavor) };
        let want = reference::cholesky(&a, n);
        let got = sky_linalg::cholesky_with(&a, n, t);
        prop_assert!(same_result(&got, &want), "n = {n} ({flavor:?}, {t} threads)");
        if let Ok(l) = &want {
            let b = matrix(&mut rng, n * k, flavor);
            let want = reference::cholesky_solve(l, n, &b, k);
            let got = sky_linalg::cholesky_solve_with(l, n, &b, k, t);
            prop_assert!(same(&got, &want), "solve n = {n}, k = {k}");
        }
    }

    #[test]
    fn lu_and_its_solve_equal_the_plain_algorithm(
        seed in any::<u64>(), n in awkward(), k in awkward(),
        flavor in flavors(), t in threads(), dup in any::<bool>(),
    ) {
        let mut rng = Rng(seed);
        let mut a = matrix(&mut rng, n * n, flavor);
        if dup && n > 1 {
            // Two equal rows: singular, or so close to it that it may be.
            let (r, s) = (rng.below(n), rng.below(n));
            let row: Vec<f64> = a[r * n..(r + 1) * n].to_vec();
            a[s * n..(s + 1) * n].copy_from_slice(&row);
        }
        let want = reference::lu(&a, n);
        let got = sky_linalg::lu_with(&a, n, t);
        match (&got, &want) {
            (Ok(g), Ok(w)) => {
                prop_assert!(same(&g.lu, &w.lu) && g.perm == w.perm && g.swaps == w.swaps);
                let b = matrix(&mut rng, n * k, flavor);
                let want_x = reference::lu_solve(w, &b, k);
                let got_x = g.solve_with(&b, k, t);
                prop_assert!(same(&got_x, &want_x));
            }
            (Err(g), Err(w)) => prop_assert_eq!(g, w),
            _ => prop_assert!(false, "one factored and the other did not: n = {n}"),
        }
    }

    #[test]
    fn inverse_equals_gauss_jordan(
        seed in any::<u64>(), n in awkward(), flavor in flavors(), t in threads(),
        sym in any::<bool>(), mirror in any::<bool>(),
    ) {
        let mut rng = Rng(seed);
        let a = if sym { symmetric(&mut rng, n, flavor) } else { matrix(&mut rng, n * n, flavor) };
        let want = reference::inverse(&a, n, mirror);
        let got = sky_linalg::inverse_with(&a, n, mirror, t);
        prop_assert!(same_result(&got, &want), "n = {n} ({flavor:?})");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// Large enough for the blocked (GEMM-based) inverse, on every flavor of
    /// data: exact zeros, ties, overflow, singular matrices, symmetric ones.
    #[test]
    fn the_blocked_inverse_equals_gauss_jordan(
        seed in any::<u64>(),
        n in prop::sample::select(vec![95usize, 96, 97, 127, 128, 129, 150, 200]),
        flavor in flavors(), t in threads(), sym in any::<bool>(),
        diag in any::<bool>(),
    ) {
        let mut rng = Rng(seed);
        let mut a = if sym { symmetric(&mut rng, n, flavor) } else { matrix(&mut rng, n * n, flavor) };
        if diag {
            // A strong diagonal keeps sparse and integer matrices from
            // being singular, so the elimination runs to the end.
            for i in 0..n { a[i * n + i] = if flavor_is_big(flavor) { a[i * n + i] } else { 40.0 + i as f64 }; }
        }
        let want = reference::inverse(&a, n, true);
        let got = sky_linalg::inverse_with(&a, n, true, t);
        prop_assert!(same_result(&got, &want), "n = {n} ({flavor:?}, {t} threads)");
    }
}

fn flavor_is_big(f: Flavor) -> bool {
    matches!(f, Flavor::Wide | Flavor::Huge)
}

/// Shapes big enough to use blocking across several cache blocks, more than
/// one worker, and every split (rows, columns), checked for a few sizes.
#[test]
fn big_shapes_and_every_thread_count_agree_with_the_naive_loops() {
    let shapes = [
        (130, 130, 130),
        (300, 70, 200),
        (257, 513, 129),
        (20, 3000, 100),
        (5, 2000, 700),
        (900, 40, 5),
        (1, 4000, 500),
        (1, 300, 1030),
        (1, 4000, 1100),
        (2, 9, 5000),
        (600, 1000, 1),
        (200, 300, 8),
    ];
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let mut rng = Rng(i as u64);
        let a = matrix(&mut rng, m * k, Flavor::Wide);
        let b = matrix(&mut rng, k * n, Flavor::Wide);
        let want = reference::matmul(&a, m, k, &b, n);
        for t in [1, 2, 3, 7, 16] {
            let got = sky_linalg::gemm_with(&a, m, k, &b, n, t);
            assert!(same(&got, &want), "{m}x{k} @ {k}x{n}, {t} threads");
        }
    }
}

#[test]
fn big_factorizations_and_every_thread_count_agree_with_the_plain_algorithm() {
    for (i, &n) in [97usize, 200, 331].iter().enumerate() {
        let mut rng = Rng(100 + i as u64);
        let a = spd(&mut rng, n, Flavor::Plain);
        let want = reference::cholesky(&a, n).expect("positive definite");
        let b = matrix(&mut rng, n * 40, Flavor::Plain);
        let want_x = reference::cholesky_solve(&want, n, &b, 40);
        for t in [1, 2, 5, 16] {
            let got = sky_linalg::cholesky_with(&a, n, t).expect("positive definite");
            assert!(same(&got, &want), "cholesky n = {n}, {t} threads");
            let x = sky_linalg::cholesky_solve_with(&want, n, &b, 40, t);
            assert!(same(&x, &want_x), "cholesky solve n = {n}, {t} threads");
        }
        let a = matrix(&mut rng, n * n, Flavor::Plain);
        let want_lu: Lu = reference::lu(&a, n).expect("nonsingular");
        let want_x = reference::lu_solve(&want_lu, &b, 40);
        let want_inv = reference::inverse(&a, n, true).expect("nonsingular");
        for t in [1, 2, 5, 16] {
            let got = sky_linalg::lu_with(&a, n, t).expect("nonsingular");
            assert!(
                same(&got.lu, &want_lu.lu) && got.perm == want_lu.perm,
                "lu n = {n}"
            );
            assert!(
                same(&got.solve_with(&b, 40, t), &want_x),
                "lu solve n = {n}"
            );
            let inv = sky_linalg::inverse_with(&a, n, true, t).expect("nonsingular");
            assert!(same(&inv, &want_inv), "inverse n = {n}, {t} threads");
        }
    }
}

#[test]
fn errors_are_the_same_as_the_plain_algorithm() {
    assert_eq!(
        sky_linalg::cholesky(&[1.0, 2.0, 2.0, 1.0], 2),
        Err(LinalgError::NotPositiveDefinite { row: 1 })
    );
    assert_eq!(
        sky_linalg::cholesky(&[2.0, 1.0, 1.5, 2.0], 2),
        Err(LinalgError::NotSymmetric)
    );
    assert_eq!(
        sky_linalg::inverse(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
        Err(LinalgError::Singular)
    );
    assert_eq!(
        sky_linalg::lu(&[0.0; 9], 3).map(|f| f.n),
        Err(LinalgError::Singular)
    );
}

#[test]
fn known_results() {
    // [[1, 2, 3], [4, 5, 6]] * [[7, 8], [9, 10], [11, 12]]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    assert_eq!(
        sky_linalg::gemm(&a, 2, 3, &b, 2),
        [58.0, 64.0, 139.0, 154.0]
    );
    // A dot product of 10 terms, the sum of 1..=10 (exact in any order).
    let x: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_eq!(sky_linalg::dot(&x, &[1.0; 10]), 55.0);
    // The classic Cholesky example, whose factor is [[2, 0, 0], [6, 1, 0], [-8, 5, 3]].
    let spd = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let l = sky_linalg::cholesky(&spd, 3).unwrap();
    assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
    assert_eq!(
        sky_linalg::cholesky_solve(&l, 3, &[-20.0, -43.0, 192.0], 1),
        [1.0, 2.0, 3.0]
    );
    // A system that needs a row exchange.
    let sys = [0.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let f = sky_linalg::lu(&sys, 3).unwrap();
    assert_eq!(f.perm, [1, 0, 2]);
    let x = f.solve(&[4.0, 3.0, 3.0], 1);
    let want = [1.0, 1.0, 1.0];
    let back = reference::matmul(&sys, 3, 3, &x, 1);
    for (got, w) in back.iter().zip([4.0, 3.0, 3.0]) {
        assert!((got - w).abs() < 1e-14, "{x:?} {back:?} {want:?}");
    }
    // [[4, 7], [2, 6]]⁻¹ = [[0.6, -0.7], [-0.2, 0.4]].
    let inv = sky_linalg::inverse(&[4.0, 7.0, 2.0, 6.0], 2).unwrap();
    for (got, want) in inv.iter().zip([0.6, -0.7, -0.2, 0.4]) {
        assert!((got - want).abs() < 1e-15, "{inv:?}");
    }
}
