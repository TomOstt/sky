//! Autotuning with a local performance database (ROADMAP Phase 4, ideas
//! [16, 17]).
//!
//! A kernel has parameters that change its speed but not its answer (a block
//! size, an unroll factor). The best values depend on the machine, so Sky
//! measures them on the machine, once, when the user asks (`sky tune`), and
//! remembers the winners in a small text file keyed by
//! [`sky_hw::HardwareInfo::fingerprint`].
//!
//! # For kernel authors
//!
//! Describe what to tune, and register it:
//!
//! ```
//! use sky_tune::{grid, size_class, Tunable, Work};
//!
//! let n = 1 << 12;
//! let data = vec![1.0f64; n];
//! let tunable = Tunable::new(
//!     "sum_chunked",              // the kernel's name
//!     &size_class(n),             // the problem-size class: "2^12"
//!     grid(&[("chunk", &[64, 128, 256])]), // the first candidate is the default
//!     Work::Bytes(8 * n as u64),  // for "GB/s" in the report
//!     move |p| {
//!         let chunk = p.get_or("chunk", 64) as usize;
//!         data.chunks(chunk).map(|c| c.iter().sum::<f64>()).sum::<f64>().to_bits()
//!     },
//! )
//! .unwrap();
//! # let _ = tunable;
//! ```
//!
//! and read the result where the kernel picks its parameters:
//!
//! ```
//! use sky_tune::lookup;
//!
//! // `None` means "not tuned here": use the built-in default.
//! let chunk = lookup("sum_chunked", "2^12")
//!     .map_or(64, |p| p.get_clamped("chunk", 64, 16, 4096));
//! # let _ = chunk;
//! ```
//!
//! [`lookup`] reads the database once per process, so it is cheap to call
//! per kernel invocation; [`lookup_among`] additionally insists the stored
//! value is one of the candidates.
//!
//! # Tuning changes speed, never results
//!
//! A tunable must give the same answer for every candidate. Sky's kernels
//! are reproducible by design (their results do not depend on block sizes,
//! thread counts or SIMD width), so which candidate wins can only change how
//! fast the answer arrives. A result that did depend on the parameters would
//! not be a tunable but a different function.
//!
//! # Not a saved session
//!
//! The database is an explicit, user-initiated performance cache: written
//! only by `sky tune`, holding only machine facts and timings, never a
//! program, a value, a REPL history or any user data. Spec non-negotiable 3
//! forbids saving REPL sessions and requires caches to be data-only; this is
//! a data-only cache, and see [`db`] for how it is read defensively.

#![forbid(unsafe_code)]

pub mod db;
pub mod demo;
mod measure;
mod params;
mod registry;
mod store;

use std::path::PathBuf;
use std::sync::Mutex;

pub use db::{valid_token, Db, Record};
pub use measure::{measure, size_class, Config, Measurement, Outcome, Tunable, Work};
pub use params::{grid, Params, MAX_CANDIDATES, MAX_PARAMS, MAX_VALUE};
pub use registry::{Builder, Registry};
pub use store::{db_path, db_path_from, now, Store, TuneError};

/// Measures `tunable` (see [`measure`]) and stores the winner in the
/// database for this machine, replacing an older result. Explicit and
/// user-initiated: nothing else in Sky calls this behind the user's back.
pub fn tune(tunable: &Tunable) -> Result<Outcome, TuneError> {
    let store = Store::default_location().ok_or(TuneError::NoLocation)?;
    tune_in(&store, tunable, &Config::default())
}

/// [`tune`] with a chosen [`Config`] and [`Store`].
pub fn tune_in(store: &Store, tunable: &Tunable, config: &Config) -> Result<Outcome, TuneError> {
    let outcome = measure(tunable, config);
    store.record(&outcome)?;
    invalidate_cache();
    Ok(outcome)
}

/// What [`lookup`] has read: the database at `path`.
struct Cached {
    path: PathBuf,
    db: Db,
}

static CACHE: Mutex<Option<Cached>> = Mutex::new(None);

/// Forgets the database [`lookup`] has cached, so the next lookup rereads
/// the file. [`tune_in`] and `sky tune --reset` call it.
pub fn invalidate_cache() {
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The stored record for this machine, reading the database at most once per
/// process (and printing its warnings, once, to standard error).
fn lookup_record(kernel: &str, size_class: &str) -> Option<Record> {
    let path = db_path()?;
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.as_ref().is_none_or(|c| c.path != path) {
        let db = Db::load(&path);
        for w in &db.warnings {
            eprintln!("sky-tune: warning: {w}");
        }
        *cache = Some(Cached { path, db });
    }
    let db = &cache.as_ref()?.db;
    if db.records().is_empty() {
        return None; // no hardware probe when there is nothing to match
    }
    db.get(sky_hw::current_fingerprint(), kernel, size_class)
        .cloned()
}

/// The parameters tuned on this machine for `kernel` at `size_class`, or
/// `None` (no database, no result for this machine, or a malformed entry):
/// use the default.
///
/// The values come from a file, so treat them as hints: read them with
/// [`Params::get_clamped`] or use [`lookup_among`].
pub fn lookup(kernel: &str, size_class: &str) -> Option<Params> {
    lookup_record(kernel, size_class).map(|r| r.params)
}

/// [`lookup`], but only if the stored parameters are one of `candidates`
/// (the kernel's own list): a stale or edited entry that names something the
/// kernel does not offer is ignored.
pub fn lookup_among(kernel: &str, size_class: &str, candidates: &[Params]) -> Option<Params> {
    lookup(kernel, size_class).filter(|p| candidates.contains(p))
}

/// A note for `sky check --explain` to cite when a kernel's parameters came
/// from tuning: `tuned on this machine (2026-09-30): unroll 8, 1.41x faster
/// than the default unroll 4`. `None` when there is no result for this
/// machine.
pub fn tuned_note(kernel: &str, size_class: &str) -> Option<String> {
    let r = lookup_record(kernel, size_class)?;
    let mut note = format!(
        "tuned on this machine ({}): {}",
        format_date(r.when),
        r.params
    );
    if r.params != r.default_params {
        note.push_str(&format!(
            ", {:.2}x faster than the default {}",
            r.speedup(),
            r.default_params.short()
        ));
    } else {
        note.push_str(", the default was fastest");
    }
    Some(note)
}

/// Seconds since the Unix epoch as `YYYY-MM-DD` (UTC).
pub fn format_date(unix_secs: u64) -> String {
    // Howard Hinnant's civil-from-days.
    let days = (unix_secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn dates() {
        assert_eq!(format_date(0), "1970-01-01");
        assert_eq!(format_date(951_782_400), "2000-02-29");
        assert_eq!(format_date(1_790_000_000), "2026-09-21");
        assert_eq!(format_date(4_102_444_800), "2100-01-01");
    }

    #[test]
    fn tune_stores_and_the_store_reads_back() {
        let dir = std::env::temp_dir().join(format!("sky-tune-lib-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = Store::at(dir.join("tune.db"), "0123456789abcdef");
        assert_eq!(
            store.lookup("spin", "s"),
            None,
            "no database means defaults"
        );

        let spin = |us: u64| {
            let t = std::time::Instant::now();
            let mut x = 0u64;
            while t.elapsed() < Duration::from_micros(us) {
                x = x.wrapping_add(1);
            }
            x
        };
        let t = Tunable::new(
            "spin",
            "s",
            vec![
                Params::new().with("cost", 300),
                Params::new().with("cost", 60),
            ],
            Work::Unknown,
            move |p| spin(p.get_or("cost", 0) as u64),
        )
        .unwrap();
        let cfg = Config {
            budget: Duration::from_millis(20),
            min_repeats: 3,
            max_repeats: 6,
            min_sample: Duration::from_micros(50),
            min_gain: 0.03,
        };
        let outcome = tune_in(&store, &t, &cfg).unwrap();
        assert_eq!(outcome.winner, 1);
        assert_eq!(
            store.lookup("spin", "s"),
            Some(Params::new().with("cost", 60))
        );
        assert_eq!(store.lookup("spin", "other"), None);
        assert_eq!(store.lookup("nope", "s"), None);
        // Another machine's fingerprint sees nothing.
        let elsewhere = Store::at(dir.join("tune.db"), "fedcba9876543210");
        assert_eq!(elsewhere.lookup("spin", "s"), None);

        // The file is human-readable text.
        let text = std::fs::read_to_string(store.path()).unwrap();
        assert!(text.contains("params=cost:60 default=cost:300"), "{text}");

        assert_eq!(store.reset(), Ok(true));
        assert_eq!(store.lookup("spin", "s"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
