//! Tuning changes speed, never results: GEMM, Cholesky and LU (which is tuned
//! through the GEMM blocks and thread threshold its updates use), and the
//! products built on them, `syrk` and the inverse, give bit-identical results
//! for every candidate parameter set `sky tune` may pick, and for the extremes a stored
//! value can be clamped to, on random matrices of order 1 to 300 with the
//! awkward sizes (primes, one off a block or a tile) well represented.

use proptest::prelude::*;
use sky_linalg::tuning::{candidates, with_tuning, Blocks, Tuning, CLASSES};
use sky_tune::Params;

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

/// The kinds of entries: ordinary numbers, small integers with many exact
/// zeros (ties, zero multipliers, singular matrices), and wide magnitudes
/// with signed zeros.
#[derive(Clone, Copy, Debug)]
enum Flavor {
    Plain,
    Sparse,
    Wide,
}

fn entry(rng: &mut Rng, flavor: Flavor) -> f64 {
    let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
    match flavor {
        Flavor::Plain => 2.0 * rng.unit() - 1.0,
        Flavor::Sparse => match rng.below(10) {
            0..=5 => 0.0,
            _ => rng.below(7) as f64 - 3.0,
        },
        Flavor::Wide => match rng.below(10) {
            0 => 0.0,
            1 => -0.0,
            _ => sign * rng.unit() * 10f64.powi(rng.below(41) as i32 - 20),
        },
    }
}

fn random(rng: &mut Rng, len: usize, flavor: Flavor) -> Vec<f64> {
    (0..len).map(|_| entry(rng, flavor)).collect()
}

/// A symmetric matrix (exactly), usually positive definite (a dominant
/// diagonal), sometimes not.
fn symmetric(rng: &mut Rng, n: usize, flavor: Flavor) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = entry(rng, flavor);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    if rng.below(4) != 0 {
        for i in 0..n {
            a[i * n + i] = a[i * n + i].abs() + n as f64 * 4.0;
        }
    }
    a
}

/// A size: the awkward ones half the time, else anything in 1..=300.
fn size() -> impl Strategy<Value = usize> {
    prop_oneof![
        proptest::sample::select(vec![
            1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 39, 40, 41, 63, 64, 65, 95, 96, 97,
            127, 129, 191, 193, 255, 257, 299, 300,
        ]),
        1usize..=300,
    ]
}

fn flavor() -> impl Strategy<Value = Flavor> {
    prop_oneof![
        3 => Just(Flavor::Plain),
        1 => Just(Flavor::Sparse),
        1 => Just(Flavor::Wide),
    ]
}

/// Every parameter set to try: each candidate of each tunable (all size
/// classes) alone, and three extremes of what clamping allows (tiny blocks and
/// everything threaded; huge blocks and nothing threaded; a mixture).
fn tunings() -> Vec<Tuning> {
    let mut all = vec![Tuning::default()];
    let mut push = |t: Tuning| {
        if !all.contains(&t) {
            all.push(t);
        }
    };
    for class in 0..CLASSES.len() {
        for kernel in ["gemm", "cholesky", "threads"] {
            for p in candidates(kernel, class) {
                push(Tuning::default().with_params(kernel, &p));
            }
        }
    }
    let p = |pairs: &[(&str, i64)]| pairs.iter().fold(Params::new(), |p, (k, v)| p.with(k, *v));
    let tiny = Tuning {
        gemm: Blocks {
            mc: 16,
            kc: 16,
            nc: 64,
        },
        cholesky_nb: 8,
        par_min_madds: 1 << 14,
    };
    let huge = Tuning {
        gemm: Blocks {
            mc: 512,
            kc: 1024,
            nc: 4096,
        },
        cholesky_nb: 64,
        par_min_madds: 1 << 30,
    };
    let mixed = Tuning::default()
        .with_params("gemm", &p(&[("mc", 64), ("kc", 128), ("nc", 512)]))
        .with_params("cholesky", &p(&[("nb", 16)]))
        .with_params("threads", &p(&[("madds", 1 << 19)]));
    push(tiny);
    push(huge);
    push(mixed);
    all
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Runs `kernel` under every tuning and asserts every result equals the one
/// under the defaults.
fn check_all<R: PartialEq + std::fmt::Debug>(what: &str, kernel: impl Fn() -> R) {
    let mut all = tunings().into_iter();
    let first = all.next().unwrap();
    let want = with_tuning(first, &kernel);
    for t in all {
        let got = with_tuning(t, &kernel);
        assert_eq!(got, want, "{what} differs under {t:?}");
    }
}

/// More workers than any test machine has cores would need: the threshold
/// decides whether they are used.
const THREADS: usize = 3;

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    #[test]
    fn gemm_is_bit_identical_for_every_candidate(
        m in size(), k in size(), n in size(), flavor in flavor(), seed in any::<u64>(),
    ) {
        let mut rng = Rng(seed);
        let a = random(&mut rng, m * k, flavor);
        let b = random(&mut rng, k * n, flavor);
        check_all(&format!("gemm {m}x{k}x{n}"), || {
            bits(&sky_linalg::gemm_with(&a, m, k, &b, n, THREADS))
        });
        // The same products through the Gram-matrix kernel (lower triangle only).
        check_all(&format!("syrk {m}x{k}"), || {
            bits(&sky_linalg::syrk_with(&a, m, k, THREADS))
        });
    }

    #[test]
    fn cholesky_is_bit_identical_for_every_candidate(
        n in size(), flavor in flavor(), seed in any::<u64>(),
    ) {
        let a = symmetric(&mut Rng(seed), n, flavor);
        check_all(&format!("cholesky {n}"), || {
            sky_linalg::cholesky_with(&a, n, THREADS).map(|l| bits(&l))
        });
        // The solve after it, which the threshold also gates.
        let rhs = random(&mut Rng(seed ^ 1), n * 3, Flavor::Plain);
        if let Ok(l) = with_tuning(Tuning::default(), || sky_linalg::cholesky_with(&a, n, THREADS)) {
            check_all(&format!("cholesky_solve {n}"), || {
                bits(&sky_linalg::cholesky_solve_with(&l, n, &rhs, 3, THREADS))
            });
        }
    }

    #[test]
    fn lu_is_bit_identical_for_every_candidate(
        n in size(), flavor in flavor(), seed in any::<u64>(),
    ) {
        let a = random(&mut Rng(seed), n * n, flavor);
        check_all(&format!("lu {n}"), || {
            sky_linalg::lu_with(&a, n, THREADS).map(|f| (bits(&f.lu), f.swaps))
        });
    }

    #[test]
    fn the_inverse_is_bit_identical_for_every_candidate(
        n in 1usize..=200, flavor in flavor(), seed in any::<u64>(),
    ) {
        let a = random(&mut Rng(seed), n * n, flavor);
        check_all(&format!("inverse {n}"), || {
            sky_linalg::inverse_with(&a, n, true, THREADS).map(|x| bits(&x))
        });
    }
}

/// The tuned candidates are what `sky tune` measures, so each must be a
/// valid setting: it survives clamping unchanged.
#[test]
fn candidates_are_not_clamped() {
    for class in 0..CLASSES.len() {
        for p in candidates("gemm", class) {
            let t = Tuning::default().with_params("gemm", &p);
            assert_eq!(t.gemm.mc as i64, p.get("mc").unwrap(), "{p}");
            assert_eq!(t.gemm.kc as i64, p.get("kc").unwrap(), "{p}");
            assert_eq!(t.gemm.nc as i64, p.get("nc").unwrap(), "{p}");
        }
        for p in candidates("cholesky", class) {
            let t = Tuning::default().with_params("cholesky", &p);
            assert_eq!(t.cholesky_nb as i64, p.get("nb").unwrap(), "{p}");
        }
    }
    for p in candidates("threads", 0) {
        let t = Tuning::default().with_params("threads", &p);
        assert_eq!(t.par_min_madds as i64, p.get("madds").unwrap(), "{p}");
    }
}
