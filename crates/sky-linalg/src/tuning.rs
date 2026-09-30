//! Tuned parameters: block sizes and thresholds that change how fast a kernel
//! runs and never what it computes.
//!
//! Every kernel's accumulation order is fixed per output entry (see the crate
//! documentation), independent of the cache blocks, the panel widths and
//! whether the work is split across threads. So the values here are free to
//! be chosen by measurement: `sky tune` (see [`register`]) times a small grid
//! of candidates on this machine and stores the winners in the performance
//! database, and the kernels read them back with [`sky_tune::lookup`] once per
//! process. `tests/tuning.rs` checks bit for bit that every candidate gives
//! the same result.
//!
//! What is tuned, with today's defaults first in each grid:
//!
//! | kernel | parameters | size classes |
//! |---|---|---|
//! | `gemm` | `mc` (rows of `A` per block, 128), `kc` (the `k` extent, 256), `nc` (columns of `B`, 1024) | by the largest of `m`, `n`, `k` |
//! | `cholesky` | `nb` (columns per block, 32 up to order 128, else 64) | by the order |
//! | `threads` | `madds` (the fewest multiply-adds a kernel spreads over threads, 2^21) | one class, `all` |
//!
//!
//! LU has no block size of its own here, on purpose: its panel width is the
//! one parameter whose choice can change a bit. A panel skips an update whose
//! multiplier is zero (as the plain algorithm does) while the GEMM update
//! between panels subtracts `0 * u` regardless, and that differs when the
//! entry is `-0.0` and the product `-0.0` (`-0.0 - -0.0` is `+0.0`). Where
//! the panels end therefore decides the sign of some zeros, so it is not a
//! tunable; LU is tuned through the GEMM blocks and the thread threshold its
//! updates use.
//!
//! The size classes are [`CLASSES`]: `le64`, `le256`, `le1024` (up to that
//! size) and `gt1024`. A stored value is only a hint: it is clamped into a
//! range every kernel handles (and rounded to the microkernel's tile), so a
//! stale or edited database cannot break a kernel or change a result.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use sky_tune::{grid, Params, Registry, Tunable, Work};

use crate::micro::{MR, NR};

/// The names of the four size classes, smallest first: what `sky tune` and
/// the database call them.
pub const CLASSES: [&str; 4] = ["le64", "le256", "le1024", "gt1024"];

/// The size class of a problem whose largest dimension is `n`: an index into
/// [`CLASSES`].
pub fn class_of(n: usize) -> usize {
    match n {
        0..=64 => 0,
        65..=256 => 1,
        257..=1024 => 2,
        _ => 3,
    }
}

/// The cache blocks of the packed matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocks {
    /// Rows of `A` per cache block (a multiple of the microkernel's 4).
    pub mc: usize,
    /// The `k` extent of a cache block.
    pub kc: usize,
    /// Columns of `B` per cache block (a multiple of the microkernel's 8).
    pub nc: usize,
}

impl Blocks {
    /// The built-in blocks.
    pub const DEFAULT: Blocks = Blocks {
        mc: 128,
        kc: 256,
        nc: 1024,
    };

    fn from_params(p: &Params) -> Blocks {
        let d = Blocks::DEFAULT;
        Blocks {
            mc: p.get_clamped("mc", d.mc as i64, 16, 512) as usize / MR * MR,
            kc: p.get_clamped("kc", d.kc as i64, 16, 1024) as usize,
            nc: p.get_clamped("nc", d.nc as i64, 64, 4096) as usize / NR * NR,
        }
    }
}

/// The built-in Cholesky block: small matrices gain from the cheaper scalar
/// parts, large ones from the longer `k` of each trailing update.
pub const fn default_cholesky_nb(n: usize) -> usize {
    if n <= 128 {
        32
    } else {
        64
    }
}

/// The widest Cholesky block (the diagonal block lives in a fixed array).
pub(crate) const MAX_CHOLESKY_NB: usize = 64;

/// The built-in threshold: below this many multiply-adds a kernel stays on
/// the calling thread (waking the pool costs about 5 to 20 microseconds).
pub const DEFAULT_PAR_MIN_MADDS: usize = 1 << 21;

/// A complete set of parameters, for [`with_tuning`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// The product's cache blocks.
    pub gemm: Blocks,
    /// The Cholesky block; 0 keeps the built-in rule ([`default_cholesky_nb`]).
    pub cholesky_nb: usize,
    /// The fewest multiply-adds spread over threads.
    pub par_min_madds: usize,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            gemm: Blocks::DEFAULT,
            cholesky_nb: 0,
            par_min_madds: DEFAULT_PAR_MIN_MADDS,
        }
    }
}

impl Tuning {
    /// The defaults with the parameters of one tunable applied: `params` as
    /// tuned for `kernel` (`gemm`, `cholesky` or `threads`), clamped
    /// like a stored value. Any other kernel changes nothing.
    pub fn with_params(mut self, kernel: &str, params: &Params) -> Tuning {
        match kernel {
            "gemm" => self.gemm = Blocks::from_params(params),
            "cholesky" => self.cholesky_nb = clamp_nb(params),
            "threads" => self.par_min_madds = clamp_madds(params),
            _ => {}
        }
        self
    }
}

fn clamp_nb(p: &Params) -> usize {
    p.get_clamped("nb", 0, 8, MAX_CHOLESKY_NB as i64) as usize
}

fn clamp_madds(p: &Params) -> usize {
    p.get_clamped("madds", DEFAULT_PAR_MIN_MADDS as i64, 1 << 14, 1 << 30) as usize
}

/// What the database said, per size class.
struct Resolved {
    gemm: [Blocks; 4],
    cholesky_nb: [usize; 4],
    par_min_madds: usize,
}

fn resolved() -> &'static Resolved {
    static RESOLVED: OnceLock<Resolved> = OnceLock::new();
    RESOLVED.get_or_init(|| {
        let mut r = Resolved {
            gemm: [Blocks::DEFAULT; 4],
            cholesky_nb: [0; 4],
            par_min_madds: DEFAULT_PAR_MIN_MADDS,
        };
        for (i, class) in CLASSES.iter().enumerate() {
            if let Some(p) = sky_tune::lookup("gemm", class) {
                r.gemm[i] = Blocks::from_params(&p);
            }
            if let Some(p) = sky_tune::lookup("cholesky", class) {
                r.cholesky_nb[i] = clamp_nb(&p);
            }
        }
        if let Some(p) = sky_tune::lookup("threads", "all") {
            r.par_min_madds = clamp_madds(&p);
        }
        r
    })
}

static OVERRIDE_ON: AtomicBool = AtomicBool::new(false);
static OVERRIDE: RwLock<Option<Tuning>> = RwLock::new(None);
static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

fn overridden() -> Option<Tuning> {
    if OVERRIDE_ON.load(Ordering::Relaxed) {
        *OVERRIDE.read().unwrap_or_else(|e| e.into_inner())
    } else {
        None
    }
}

/// Runs `f` with `tuning` in force for every kernel, on every thread, instead
/// of what the database says. It is how `sky tune` times a candidate and how
/// the tests run every candidate; since tuning never changes results, other
/// threads that run kernels meanwhile only get them at a different speed.
/// Calls are serialized (do not nest them).
pub fn with_tuning<R>(tuning: Tuning, f: impl FnOnce() -> R) -> R {
    struct Restore<'a>(#[allow(dead_code)] MutexGuard<'a, ()>);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            OVERRIDE_ON.store(false, Ordering::Relaxed);
            *OVERRIDE.write().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    let _restore = Restore(OVERRIDE_LOCK.lock().unwrap_or_else(|e| e.into_inner()));
    *OVERRIDE.write().unwrap_or_else(|e| e.into_inner()) = Some(tuning);
    OVERRIDE_ON.store(true, Ordering::Relaxed);
    f()
}

/// The cache blocks for a product with these dimensions.
pub(crate) fn gemm_blocks(m: usize, n: usize, k: usize) -> Blocks {
    match overridden() {
        Some(t) => t.gemm,
        None => resolved().gemm[class_of(m.max(n).max(k))],
    }
}

/// The Cholesky block for an `n` by `n` matrix.
pub(crate) fn cholesky_nb(n: usize) -> usize {
    let nb = match overridden() {
        Some(t) => t.cholesky_nb,
        None => resolved().cholesky_nb[class_of(n)],
    };
    if nb == 0 {
        default_cholesky_nb(n)
    } else {
        nb
    }
}

/// The fewest multiply-adds a kernel spreads over threads.
pub(crate) fn par_min_madds() -> usize {
    match overridden() {
        Some(t) => t.par_min_madds,
        None => resolved().par_min_madds,
    }
}

/// A sentence for `sky check --explain` about the tuned parameters of
/// `kernel` (`gemm`, `cholesky` or `threads`) at problem size `n`, or
/// `None` when nothing was tuned on this machine for it.
pub fn explain(kernel: &str, n: usize) -> Option<String> {
    let (what, class) = match kernel {
        "gemm" => ("GEMM blocks", CLASSES[class_of(n)]),
        "cholesky" => ("Cholesky blocks", CLASSES[class_of(n)]),
        "threads" => ("the thread threshold", "all"),
        _ => return None,
    };
    let note = sky_tune::tuned_note(kernel, class)?;
    Some(format!("{what} {note}"))
}

// ---------------------------------------------------------------------------
// The tunables.

/// The problem size each size class is measured at.
const REPRESENTATIVE: [usize; 4] = [48, 192, 768, 1280];

/// The candidates of `kernel` (`gemm`, `cholesky` or `threads`) for a
/// size class (an index into [`CLASSES`]; `threads` has one class), the
/// built-in setting first. `sky tune` measures exactly these.
pub fn candidates(kernel: &str, class: usize) -> Vec<Params> {
    match kernel {
        "gemm" => {
            // `nc` only matters once a matrix is wider than a block.
            let ncs: &[i64] = match class {
                0 | 1 => &[1024],
                2 => &[1024, 512],
                _ => &[1024, 512, 2048],
            };
            grid(&[
                ("mc", &[128, 64, 256]),
                ("kc", &[256, 128, 512]),
                ("nc", ncs),
            ])
        }
        "cholesky" => {
            let default = default_cholesky_nb(REPRESENTATIVE[class.min(3)]) as i64;
            let mut nbs = vec![default];
            nbs.extend([16, 32, 48, 64].into_iter().filter(|&x| x != default));
            grid(&[("nb", &nbs)])
        }
        _ => grid(&[(
            "madds",
            &[1 << 21, 1 << 19, 1 << 20, 1 << 22, 1 << 23, 1 << 24],
        )]),
    }
}

/// A small deterministic pseudo-random matrix (SplitMix64), so every candidate
/// times the same data and the checksums can be compared.
fn matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn checksum(x: &[f64]) -> u64 {
    x.iter()
        .step_by(x.len().div_ceil(64).max(1))
        .fold(0u64, |h, v| h.rotate_left(7) ^ v.to_bits())
}

fn gemm_tunable(class: usize) -> Vec<Tunable> {
    let n = REPRESENTATIVE[class];
    let (a, b) = (matrix(n * n, 1), matrix(n * n, 2));
    let tunable = Tunable::new(
        "gemm",
        CLASSES[class],
        candidates("gemm", class),
        Work::Flops(2 * (n as u64).pow(3)),
        move |p| {
            let t = Tuning::default().with_params("gemm", p);
            with_tuning(t, || {
                let mut c = vec![0.0; n * n];
                crate::gemm_into_with(&a, n, n, &b, n, &mut c, crate::default_threads());
                checksum(&c)
            })
        },
    );
    vec![tunable.expect("valid tunable")]
}

/// A symmetric positive-definite matrix of order `n` (diagonally dominant).
fn spd(n: usize) -> Vec<f64> {
    let r = matrix(n * n, 3);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = r[i * n + j];
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    a
}

fn cholesky_tunable(class: usize) -> Vec<Tunable> {
    let n = REPRESENTATIVE[class];
    let a = spd(n);
    let tunable = Tunable::new(
        "cholesky",
        CLASSES[class],
        candidates("cholesky", class),
        Work::Flops((n as u64).pow(3) / 3),
        move |p| {
            let t = Tuning::default().with_params("cholesky", p);
            with_tuning(t, || {
                checksum(&crate::cholesky_with(&a, n, crate::default_threads()).unwrap())
            })
        },
    );
    vec![tunable.expect("valid tunable")]
}

/// The thread threshold is measured over a ladder of product sizes from
/// 2^18 to 2^24 multiply-adds, each repeated so that all cost the same
/// sequentially: a candidate wins by being right about all of them.
fn threads_tunable() -> Vec<Tunable> {
    let ladder: Vec<(usize, usize)> = [64usize, 81, 101, 128, 161, 203, 256]
        .iter()
        .map(|&n| (n, ((1usize << 24) / (n * n * n)).max(1)))
        .collect();
    let biggest = 256;
    let (a, b) = (matrix(biggest * biggest, 5), matrix(biggest * biggest, 6));
    let flops: u64 = ladder
        .iter()
        .map(|&(n, reps)| 2 * (n as u64).pow(3) * reps as u64)
        .sum();
    let tunable = Tunable::new(
        "threads",
        "all",
        candidates("threads", 0),
        Work::Flops(flops),
        move |p| {
            let t = Tuning::default().with_params("threads", p);
            with_tuning(t, || {
                let mut h = 0u64;
                for &(n, reps) in &ladder {
                    let mut c = vec![0.0; n * n];
                    for _ in 0..reps {
                        let (a, b) = (&a[..n * n], &b[..n * n]);
                        crate::gemm_into_with(a, n, n, b, n, &mut c, crate::default_threads());
                        h ^= checksum(&c);
                    }
                }
                h
            })
        },
    );
    vec![tunable.expect("valid tunable")]
}

macro_rules! per_class {
    ($($name:ident: $builder:ident $class:expr;)*) => {
        $(fn $name() -> Vec<Tunable> { $builder($class) })*
    };
}

per_class! {
    gemm_le64: gemm_tunable 0;
    gemm_le256: gemm_tunable 1;
    gemm_le1024: gemm_tunable 2;
    gemm_gt1024: gemm_tunable 3;
    cholesky_le64: cholesky_tunable 0;
    cholesky_le256: cholesky_tunable 1;
    cholesky_le1024: cholesky_tunable 2;
    cholesky_gt1024: cholesky_tunable 3;
}

/// Adds this crate's tunables to `registry` (`sky tune` calls this): the
/// product's cache blocks and the Cholesky block size, each for four size
/// classes, and the threshold above which kernels use threads. Each entry is
/// named `linalg_<kernel>_<class>` (so `sky tune linalg_gemm` tunes the
/// products and `sky tune linalg_cholesky_le64` one class of Cholesky).
pub fn register(registry: &mut Registry) {
    registry
        .add("linalg_gemm_le64", gemm_le64)
        .add("linalg_gemm_le256", gemm_le256)
        .add("linalg_gemm_le1024", gemm_le1024)
        .add("linalg_gemm_gt1024", gemm_gt1024)
        .add("linalg_cholesky_le64", cholesky_le64)
        .add("linalg_cholesky_le256", cholesky_le256)
        .add("linalg_cholesky_le1024", cholesky_le1024)
        .add("linalg_cholesky_gt1024", cholesky_gt1024)
        .add("linalg_threads", threads_tunable);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_and_clamping() {
        assert_eq!(
            [1, 64, 65, 256, 257, 1024, 1025, 1 << 20].map(class_of),
            [0, 0, 1, 1, 2, 2, 3, 3]
        );
        let wild = Params::new()
            .with("mc", 3)
            .with("kc", 1 << 30)
            .with("nc", 1001);
        let b = Blocks::from_params(&wild);
        assert_eq!((b.mc, b.kc, b.nc), (16, 1024, 1000));
        assert_eq!(Blocks::from_params(&Params::new()), Blocks::DEFAULT);
        let t = Tuning::default().with_params("cholesky", &Params::new().with("nb", 1000));
        assert_eq!(t.cholesky_nb, MAX_CHOLESKY_NB);
        let t = Tuning::default().with_params("cholesky", &Params::new().with("nb", -5));
        assert_eq!(t.cholesky_nb, 8);
        let t = Tuning::default().with_params("threads", &Params::new().with("madds", 1));
        assert_eq!(t.par_min_madds, 1 << 14);
    }

    #[test]
    fn with_tuning_is_seen_by_the_kernels() {
        let t = Tuning {
            gemm: Blocks {
                mc: 24,
                kc: 40,
                nc: 72,
            },
            cholesky_nb: 16,
            par_min_madds: 12345,
        };
        with_tuning(t, || {
            assert_eq!(gemm_blocks(300, 300, 300), t.gemm);
            assert_eq!(cholesky_nb(300), 16);
            assert_eq!(par_min_madds(), 12345);
        });
    }

    #[test]
    fn defaults_come_first_and_every_candidate_is_valid() {
        for class in 0..4 {
            let g = candidates("gemm", class);
            assert_eq!(Blocks::from_params(&g[0]), Blocks::DEFAULT);
            assert!(g.len() <= sky_tune::MAX_CANDIDATES);
            for p in &g {
                // A candidate must survive clamping unchanged.
                let b = Blocks::from_params(p);
                let got = [b.mc as i64, b.kc as i64, b.nc as i64];
                let want = ["mc", "kc", "nc"].map(|k| p.get(k).unwrap());
                assert_eq!(got, want);
            }
            let c = candidates("cholesky", class);
            let default = default_cholesky_nb(REPRESENTATIVE[class]) as i64;
            assert_eq!(c[0].get("nb"), Some(default));
        }
        assert_eq!(candidates("threads", 0)[0].get("madds"), Some(1 << 21));
    }

    #[test]
    fn the_small_tunables_agree_with_themselves() {
        // The other classes are big: the property test covers them.
        for build in [gemm_le64, cholesky_le64] {
            for t in build() {
                let first = t.run(&t.candidates[0]);
                for p in &t.candidates {
                    assert_eq!(t.run(p), first, "{} {p}", t.kernel);
                }
            }
        }
    }
}
