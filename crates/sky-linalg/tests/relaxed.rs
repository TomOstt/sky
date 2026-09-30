//! The relaxed kernels: they agree with the reproducible ones to a
//! documented tolerance, they are what they claim to be (fused where fused),
//! and they reject what Sky rejects, with Sky's errors.
//!
//! # The documented tolerance
//!
//! For a well-conditioned problem (the matrices below: random entries in
//! [-1, 1), diagonally dominant or `BBᵀ + nI`):
//!
//! * products, Gram matrices and dot products: every entry within `1e-12`
//!   of `Σ|aᵢₚ·bₚⱼ|` (the componentwise scale a sum of `k ≤ 1000` terms
//!   is entitled to, `k·ε` being about `2e-13` there);
//! * Cholesky factors and solves: every entry within `1e-12` of the largest
//!   entry of the reproducible result.
//!
//! The kernels agree to a few `ε` in practice; the bound has room so that
//! a machine with a different vector width or vendor library still passes.

use sky_linalg::{
    self as sky, cholesky_relaxed_with, cholesky_solve_relaxed_with, gemm_relaxed_with, reference,
    solve_relaxed_with, solve_right_relaxed_with, syrk_relaxed_with, LinalgError,
};

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
        2.0 * ((self.next() >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

fn random(rng: &mut Rng, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.unit()).collect()
}

/// `Σ|a·b|` per entry: the scale of a product's rounding error.
fn abs_product(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let a: Vec<f64> = a.iter().map(|x| x.abs()).collect();
    let b: Vec<f64> = b.iter().map(|x| x.abs()).collect();
    reference::matmul(&a, m, k, &b, n)
}

fn assert_close_scaled(got: &[f64], want: &[f64], scale: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, ((g, w), s)) in got.iter().zip(want).zip(scale).enumerate() {
        assert!(
            (g - w).abs() <= tol * s.max(f64::MIN_POSITIVE),
            "{what}: entry {i}: {g} vs {w} (scale {s})"
        );
    }
}

fn assert_close_max(got: &[f64], want: &[f64], tol: f64, what: &str) {
    let max = want.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-300);
    let scale = vec![max; want.len()];
    assert_close_scaled(got, want, &scale, tol, what);
}

/// A diagonally dominant (so well-conditioned) `n` by `n` matrix.
fn dominant(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut a = random(rng, n * n);
    for i in 0..n {
        a[i * n + i] += n as f64 * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    a
}

/// A symmetric positive-definite `n` by `n` matrix: `B·Bᵀ + n·I`.
fn spd(rng: &mut Rng, n: usize) -> Vec<f64> {
    let b = random(rng, n * n);
    let mut a = reference::gram(&b, n, n);
    for i in 0..n {
        a[i * n + i] += n as f64;
    }
    a
}

const THREADS: [usize; 3] = [1, 2, 7];

#[test]
fn products_of_every_shape_class_agree_with_the_reference() {
    let mut rng = Rng(1);
    // Dot, matvec, vecmat, small (<= 16 columns), packed, awkward sizes,
    // and sizes over the vendor thresholds.
    let shapes = [
        (1, 1, 1),
        (1, 50, 1),
        (37, 29, 1),
        (1, 29, 37),
        (5, 5, 5),
        (33, 65, 7),
        (100, 100, 16),
        (17, 300, 17),
        (65, 33, 129),
        (128, 128, 128),
        (300, 260, 90),
        (2, 700, 1000),
    ];
    for (m, k, n) in shapes {
        let a = random(&mut rng, m * k);
        let b = random(&mut rng, k * n);
        let want = reference::matmul(&a, m, k, &b, n);
        let scale = abs_product(&a, m, k, &b, n);
        for threads in THREADS {
            let got = gemm_relaxed_with(&a, m, k, &b, n, threads);
            assert_close_scaled(&got, &want, &scale, 1e-12, &format!("gemm {m}x{k}x{n}"));
            let mut fused = vec![f64::NAN; m * n];
            sky::gemm_into_fused_with(&a, m, k, &b, n, &mut fused, threads);
            assert_close_scaled(&fused, &want, &scale, 1e-12, &format!("fused {m}x{k}x{n}"));
        }
    }
}

/// The fused product with the sum in the reference order: every entry is
/// `acc = fma(a, b, acc)` for `p = 0, 1, ...` from `+0.0`.
fn fused_reference(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn the_fused_kernels_are_fused_and_keep_each_entrys_order() {
    if !sky::fma_available() {
        return;
    }
    let mut rng = Rng(2);
    let mut differs_from_plain = false;
    for (m, k, n) in [
        (3, 40, 1),
        (1, 40, 9),
        (9, 33, 7),
        (70, 300, 90),
        (128, 512, 33),
    ] {
        let a = random(&mut rng, m * k);
        let b = random(&mut rng, k * n);
        let want = fused_reference(&a, m, k, &b, n);
        let plain = reference::matmul(&a, m, k, &b, n);
        differs_from_plain |= want != plain;
        for threads in THREADS {
            let mut got = vec![f64::NAN; m * n];
            sky::gemm_into_fused_with(&a, m, k, &b, n, &mut got, threads);
            let same = got
                .iter()
                .zip(&want)
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                same,
                "{m}x{k}x{n} on {threads} threads is not the fused reference"
            );
        }
    }
    assert!(
        differs_from_plain,
        "fusing changed nothing: is the test data too tame?"
    );
}

#[test]
fn gram_matrices_are_exactly_symmetric_and_close() {
    let mut rng = Rng(3);
    for (m, k) in [(3, 9), (16, 40), (17, 40), (60, 200), (130, 70)] {
        let b = random(&mut rng, m * k);
        let want = reference::gram(&b, m, k);
        let bt: Vec<f64> = (0..k * m).map(|i| b[(i % m) * k + i / m]).collect();
        let scale = abs_product(&b, m, k, &bt, m);
        for threads in THREADS {
            let got = syrk_relaxed_with(&b, m, k, threads);
            assert_close_scaled(&got, &want, &scale, 1e-12, &format!("syrk {m}x{k}"));
            assert!(sky::is_symmetric(&got, m), "syrk {m}x{k} is not symmetric");
        }
    }
}

#[test]
fn cholesky_agrees_and_solves_agree() {
    let mut rng = Rng(4);
    for n in [1, 5, 23, 24, 33, 64, 100, 257] {
        let a = spd(&mut rng, n);
        let want = sky::cholesky(&a, n).unwrap();
        for threads in THREADS {
            let got = cholesky_relaxed_with(&a, n, threads).unwrap();
            assert_close_max(&got, &want, 1e-12, &format!("cholesky {n}"));
            // The strict upper triangle is exactly zero, as Sky's.
            for i in 0..n {
                assert!(got[i * n + i + 1..(i + 1) * n].iter().all(|x| *x == 0.0));
            }
            for k in [1, 3, 20] {
                let b = random(&mut rng, n * k);
                let want_x = sky::cholesky_solve(&want, n, &b, k);
                let x = cholesky_solve_relaxed_with(&got, n, &b, k, threads);
                assert_close_max(&x, &want_x, 1e-12, &format!("cholesky solve {n} x {k}"));
                assert!(x.iter().all(|v| v.to_bits() != (-0.0f64).to_bits()));
            }
        }
    }
}

#[test]
fn lu_solves_agree() {
    let mut rng = Rng(5);
    for n in [1, 4, 23, 24, 40, 100, 200] {
        let a = dominant(&mut rng, n);
        for k in [1, 4, 33] {
            let b = random(&mut rng, n * k);
            let want = sky::solve(&a, n, &b, k).unwrap();
            for threads in THREADS {
                let got = solve_relaxed_with(&a, n, &b, k, threads).unwrap();
                assert_close_max(&got, &want, 1e-12, &format!("solve {n} x {k}"));
                assert!(got.iter().all(|v| v.to_bits() != (-0.0f64).to_bits()));
            }
        }
        for m in [1, 5] {
            let x = random(&mut rng, m * n);
            let want = sky::lu(&a, n).unwrap().solve_right(&x, m);
            for threads in THREADS {
                let got = solve_right_relaxed_with(&a, n, &x, m, threads).unwrap();
                assert_close_max(&got, &want, 1e-12, &format!("solve_right {m} x {n}"));
            }
        }
    }
}

/// A matrix whose last row is the sum of two others: exactly singular in
/// integer arithmetic (and every entry is a small integer, so the
/// elimination is exact until the last pivot).
fn singular_integers(n: usize) -> Vec<f64> {
    let mut rng = Rng(9);
    let mut a: Vec<f64> = (0..n * n).map(|_| (rng.next() % 9) as f64 - 4.0).collect();
    for j in 0..n {
        a[(n - 1) * n + j] = a[j] + a[n + j];
    }
    a
}

#[test]
fn singular_matrices_are_rejected_as_sky_rejects_them() {
    let (n, k) = (40, 3);
    let b = vec![1.0; n * k];
    // Exactly singular, numerically singular (a duplicated row of ordinary
    // numbers), and the classic 1..9 block inside an identity.
    let mut rng = Rng(6);
    let mut dup = random(&mut rng, n * n);
    let row0: Vec<f64> = dup[..n].to_vec();
    dup[(n - 3) * n..(n - 2) * n].copy_from_slice(&row0);
    let mut block = vec![0.0; n * n];
    for i in 0..n {
        block[i * n + i] = 1.0;
    }
    for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        .iter()
        .enumerate()
    {
        block[(i / 3) * n + i % 3] = *v;
    }
    for (name, a) in [
        ("integers", singular_integers(n)),
        ("duplicate row", dup),
        ("1..9 block", block),
        ("zero", vec![0.0; n * n]),
    ] {
        assert_eq!(
            sky::solve(&a, n, &b, k),
            Err(LinalgError::Singular),
            "{name}"
        );
        for threads in THREADS {
            assert_eq!(
                solve_relaxed_with(&a, n, &b, k, threads),
                Err(LinalgError::Singular),
                "{name}: solve"
            );
            assert_eq!(
                solve_right_relaxed_with(&a, n, &vec![1.0; 2 * n], 2, threads),
                Err(LinalgError::Singular),
                "{name}: solve_right"
            );
        }
    }
}

#[test]
fn cholesky_rejects_as_sky_rejects() {
    let mut rng = Rng(7);
    for n in [3, 30, 90] {
        // Not symmetric: by one ulp in one entry.
        let mut a = spd(&mut rng, n);
        a[1] = f64::from_bits(a[1].to_bits() + 1);
        assert_eq!(sky::cholesky(&a, n), Err(LinalgError::NotSymmetric));
        assert_eq!(
            cholesky_relaxed_with(&a, n, 2),
            Err(LinalgError::NotSymmetric)
        );
        // Not positive definite at a chosen row: a negative pivot there.
        for row in [0, n / 2, n - 1] {
            let mut a = spd(&mut rng, n);
            a[row * n + row] = -1.0;
            let want = sky::cholesky(&a, n);
            assert!(
                matches!(want, Err(LinalgError::NotPositiveDefinite { .. })),
                "{want:?}"
            );
            for threads in THREADS {
                assert_eq!(
                    cholesky_relaxed_with(&a, n, threads),
                    want,
                    "n = {n}, row {row}"
                );
            }
        }
        // Semi-definite: a zero diagonal entry with a zero row and column.
        let mut a = spd(&mut rng, n);
        let z = n / 3;
        for j in 0..n {
            a[z * n + j] = 0.0;
            a[j * n + z] = 0.0;
        }
        let want = sky::cholesky(&a, n);
        assert_eq!(want, Err(LinalgError::NotPositiveDefinite { row: z }));
        assert_eq!(cholesky_relaxed_with(&a, n, 2), want);
    }
    // The classic small ones.
    assert_eq!(
        cholesky_relaxed_with(&[1.0, 2.0, 2.0, 1.0], 2, 1),
        Err(LinalgError::NotPositiveDefinite { row: 1 })
    );
    assert_eq!(
        cholesky_relaxed_with(&[-1.0], 1, 1),
        Err(LinalgError::NotPositiveDefinite { row: 0 })
    );
}

#[test]
fn ill_conditioned_but_valid_matrices_are_accepted() {
    // A Hilbert matrix of order 8 (condition number about 1.5e10): valid for
    // both, however the answer's last digits differ.
    let n = 8;
    let h: Vec<f64> = (0..n * n)
        .map(|i| 1.0 / ((i / n + i % n + 1) as f64))
        .collect();
    let want = sky::cholesky(&h, n).unwrap();
    let got = cholesky_relaxed_with(&h, n, 1).unwrap();
    assert_close_max(&got, &want, 1e-9, "hilbert cholesky");
    let b = vec![1.0; n];
    let want = sky::solve(&h, n, &b, 1).unwrap();
    let got = solve_relaxed_with(&h, n, &b, 1, 1).unwrap();
    // Backward stable, not forward: the error is about cond * eps.
    assert_close_max(&got, &want, 1e-4, "hilbert solve");
    // The same at an order that takes the vendor path, on a matrix that is
    // valid for Sky (a diagonal spread over ten orders of magnitude).
    let n = 40;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        d[i * n + i] = 10f64.powi(-(i as i32) / 4);
    }
    let want = sky::cholesky(&d, n).unwrap();
    assert_close_max(
        &cholesky_relaxed_with(&d, n, 1).unwrap(),
        &want,
        1e-12,
        "diag",
    );
    let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let want = sky::solve(&d, n, &b, 1).unwrap();
    let got = solve_relaxed_with(&d, n, &b, 1, 1).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-12 * w.abs(), "{g} vs {w}");
    }
}

#[test]
fn a_thread_count_never_changes_a_product_but_may_change_a_long_dot() {
    let mut rng = Rng(8);
    let (m, k, n) = (90, 200, 70);
    let a = random(&mut rng, m * k);
    let b = random(&mut rng, k * n);
    let base = gemm_relaxed_with(&a, m, k, &b, n, 1);
    for threads in [2, 5, 10] {
        let got = gemm_relaxed_with(&a, m, k, &b, n, threads);
        // Products split their output: same bits (the vendor library
        // ignores the count; the fused kernels never reorder an entry).
        assert!(got
            .iter()
            .zip(&base)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    let n = (1 << 20) + 777;
    let (x, y) = (random(&mut rng, n), random(&mut rng, n));
    let scale: f64 = x.iter().zip(&y).map(|(p, q)| (p * q).abs()).sum();
    let want = sky::dot(&x, &y);
    for threads in [1, 2, 3, 8] {
        let got = sky::dot_relaxed_with(&x, &y, threads);
        assert!(
            (got - want).abs() <= 1e-12 * scale,
            "{threads}: {got} vs {want}"
        );
    }
}

#[test]
fn the_default_kernels_are_untouched_by_the_relaxed_ones() {
    // The reproducible entry points still equal the plain loops bit for bit
    // (tests/reference.rs pins this widely; this is a smoke test that
    // nothing here changed the mode of the reproducible functions).
    let mut rng = Rng(10);
    let (m, k, n) = (37, 91, 45);
    let a = random(&mut rng, m * k);
    let b = random(&mut rng, k * n);
    let want = reference::matmul(&a, m, k, &b, n);
    let got = sky::gemm_with(&a, m, k, &b, n, 3);
    assert!(got
        .iter()
        .zip(&want)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    assert_eq!(sky::numerics(), sky::Numerics::Reproducible);
}

/// What runs on a machine with no vendor library (Linux): the factorizations
/// on the fused GEMM. On macOS this is the path the vendor falls back to for
/// the small orders, run here at every order.
#[test]
fn the_fused_factorizations_agree_and_reject_like_sky() {
    let mut rng = Rng(11);
    for n in [5, 41, 100, 200] {
        let a = spd(&mut rng, n);
        let want = sky::cholesky(&a, n).unwrap();
        for threads in THREADS {
            let got = sky::cholesky_fused_with(&a, n, threads).unwrap();
            assert_close_max(&got, &want, 1e-12, &format!("fused cholesky {n}"));
        }
        let mut bad = a.clone();
        let row = n / 2;
        bad[row * n + row] = -1.0;
        assert_eq!(
            sky::cholesky_fused_with(&bad, n, 2),
            Err(LinalgError::NotPositiveDefinite { row })
        );
        let mut asym = a.clone();
        asym[n - 1] = f64::from_bits(asym[n - 1].to_bits() + 1);
        assert_eq!(
            sky::cholesky_fused_with(&asym, n, 2),
            Err(LinalgError::NotSymmetric)
        );

        let d = dominant(&mut rng, n);
        let b = random(&mut rng, n * 3);
        let want = sky::solve(&d, n, &b, 3).unwrap();
        let lu = sky::lu_fused_with(&d, n, 3).unwrap();
        assert_close_max(&lu.solve(&b, 3), &want, 1e-12, &format!("fused lu {n}"));
    }
    let n = 60;
    assert_eq!(
        sky::lu_fused_with(&singular_integers(n), n, 2).map(|_| ()),
        Err(LinalgError::Singular)
    );
}
