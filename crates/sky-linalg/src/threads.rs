//! Threads: how many, and how work is split.
//!
//! Splitting the *output* across threads never changes the order of any
//! entry's arithmetic, so results are the same for every thread count. The
//! count comes from `SKY_THREADS` (a positive integer) or, without it, from
//! [`std::thread::available_parallelism`]. Work below a size threshold runs
//! on the calling thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// The number of threads the kernels use by default: `SKY_THREADS` if it is
/// a positive integer, else the available parallelism (at least 1).
pub fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SKY_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            })
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
