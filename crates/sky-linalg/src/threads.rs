//! Threads: how many, and how work is split.
//!
//! Splitting the *output* across threads never changes the order of any
//! entry's arithmetic, so results are the same for every thread count. The
//! count is, in order: what [`set_threads`] was given (the `--threads` flag
//! of `sky run`), `SKY_THREADS` (a whole number from 1 to [`MAX_THREADS`]),
//! or [`std::thread::available_parallelism`]. Work below a size threshold
//! runs on the calling thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// The most threads a count may ask for (the same bound as `--threads`).
pub const MAX_THREADS: usize = 1024;

/// A thread count from the text of `SKY_THREADS`: a whole number from 1 to
/// [`MAX_THREADS`].
fn parse_threads(text: Option<&str>) -> Option<usize> {
    text.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| (1..=MAX_THREADS).contains(n))
}

/// The count [`set_threads`] chose; 0 means "not set".
static CHOSEN: AtomicUsize = AtomicUsize::new(0);

/// Sets how many threads the kernels use from now on, above `SKY_THREADS`
/// and the machine's parallelism (`n` is clamped to `1..=MAX_THREADS`; 0
/// undoes it). It changes how fast an answer arrives, never the answer.
pub fn set_threads(n: usize) {
    CHOSEN.store(n.min(MAX_THREADS), Ordering::Relaxed);
}

/// The number of threads the kernels use by default: what [`set_threads`]
/// chose, else `SKY_THREADS` if it is a valid count, else the available
/// parallelism (at least 1).
pub fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    let chosen = CHOSEN.load(Ordering::Relaxed);
    if chosen != 0 {
        return chosen;
    }
    *N.get_or_init(|| {
        parse_threads(std::env::var("SKY_THREADS").ok().as_deref())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
    })
}

/// Below this many multiply-adds a kernel stays on the calling thread: the
/// cost of waking the pool is about 5 to 20 microseconds.
pub(crate) const PARALLEL_MIN_MADDS: usize = 1 << 21;

/// A pool with exactly `threads` workers (cached, so repeated calls share).
fn pool(threads: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    pools
        .entry(threads)
        .or_insert_with(|| {
            Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("sky-linalg-{i}"))
                    .build()
                    .expect("thread pool"),
            )
        })
        .clone()
}

/// Runs `f` on every item, in parallel on `threads` workers when there is
/// more than one item and more than one thread. Each item is independent.
pub(crate) fn for_each<T: Send, F: Fn(T) + Sync + Send>(threads: usize, items: Vec<T>, f: F) {
    if threads <= 1 || items.len() <= 1 {
        items.into_iter().for_each(f);
    } else {
        pool(threads).install(|| items.into_par_iter().for_each(f));
    }
}

/// Runs `f` on every item like [`for_each`] and collects the results in
/// order.
pub(crate) fn map_collect<T: Send, R: Send, F: Fn(T) -> R + Sync + Send>(
    threads: usize,
    items: Vec<T>,
    f: F,
) -> Vec<R> {
    if threads <= 1 || items.len() <= 1 {
        items.into_iter().map(f).collect()
    } else {
        pool(threads).install(|| items.into_par_iter().map(f).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sky_threads_is_a_positive_integer() {
        assert_eq!(parse_threads(Some("4")), Some(4));
        assert_eq!(parse_threads(Some(" 12 ")), Some(12));
        assert_eq!(parse_threads(Some("1")), Some(1));
        assert_eq!(parse_threads(Some("1024")), Some(1024));
        for bad in ["0", "-2", "many", "", "2.5", "1025", "99999"] {
            assert_eq!(parse_threads(Some(bad)), None, "{bad:?}");
        }
        assert_eq!(parse_threads(None), None);
        assert!(default_threads() >= 1);
    }

    #[test]
    fn work_runs_once_per_item_on_any_thread_count() {
        for threads in [1, 2, 5] {
            let out = map_collect(threads, (0..37).collect(), |i: usize| i * i);
            assert_eq!(out, (0..37).map(|i| i * i).collect::<Vec<_>>());
        }
    }
}
