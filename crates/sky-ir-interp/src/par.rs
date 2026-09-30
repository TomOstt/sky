//! The parallel runtime: a shared thread pool per thread count, the plan
//! from `sky_ir::parallel`, and the deterministic elementwise loops.
//!
//! What may run in parallel and why it cannot change a result is the
//! module documentation of `sky_ir::parallel`; this file is the machinery.
//! Nothing here reads a clock, a thread id or a counter: which chunk covers
//! which elements is a function of the element count and the configuration,
//! so a run is the same on 2 threads and on 64.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use sky_ir::parallel::{Config, Plan};
use sky_ir::Module;

/// A worker's stack. Workers run interpreter code (a started call, and
/// whatever it starts in turn), so they get room like `with_deep_stack`
/// does, plus some for the calls other workers' waiting lets them pick up.
/// Only the pages a worker touches are ever committed.
const WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;

/// The pool for `threads` workers: created on first use and kept for the
/// life of the process, so a test suite or a benchmark that runs many
/// programs does not start threads for each.
fn pool_for(threads: usize) -> Option<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Option<Arc<ThreadPool>>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    pools
        .entry(threads)
        .or_insert_with(|| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(WORKER_STACK_BYTES)
                .thread_name(|i| format!("sky-par-{i}"))
                .build()
                .ok()
                .map(Arc::new)
        })
        .clone()
}

/// What one run needs to run in parallel.
pub(crate) struct ParRt {
    pub cfg: Config,
    pub plan: Plan,
    /// Started when the first parallel work asks for it: a run that never
    /// has any (most of them) never starts a thread.
    pool: OnceLock<Option<Arc<ThreadPool>>>,
}

impl ParRt {
    /// The runtime for `module` under `cfg`, or `None` when the run is
    /// sequential (one thread).
    pub fn new(module: &Module, cfg: &Config) -> Option<ParRt> {
        if !cfg.is_parallel() {
            return None;
        }
        Some(ParRt {
            cfg: cfg.clone(),
            plan: sky_ir::parallel::analyze(module, cfg),
            pool: OnceLock::new(),
        })
    }

    /// The pool, or `None` if no threads could be started, in which case
    /// everything runs sequentially.
    pub fn pool(&self) -> Option<&ThreadPool> {
        self.pool
            .get_or_init(|| pool_for(self.cfg.threads))
            .as_deref()
    }

    /// The length of the chunks `n` elements are split into.
    fn chunk_len(&self, n: usize) -> usize {
        n.div_ceil(self.cfg.chunks(n)).max(1)
    }

    /// `out[i] = f(a[i])`, and whether every entry is finite.
    pub fn map1(&self, a: &[f64], f: impl Fn(f64) -> f64 + Sync) -> Option<(Vec<f64>, bool)> {
        let pool = self.pool()?;
        let mut out = vec![0.0; a.len()];
        let len = self.chunk_len(a.len());
        let finite = pool.install(|| {
            out.par_chunks_mut(len)
                .zip(a.par_chunks(len))
                .map(|(o, x)| {
                    for (o, &x) in o.iter_mut().zip(x) {
                        *o = f(x);
                    }
                    o.iter().all(|v| v.is_finite())
                })
                .reduce(|| true, |p, q| p && q)
        });
        Some((out, finite))
    }

    /// `out[i] = f(a[i], b[i])`, and whether every entry is finite.
    pub fn map2(
        &self,
        a: &[f64],
        b: &[f64],
        f: impl Fn(f64, f64) -> f64 + Sync,
    ) -> Option<(Vec<f64>, bool)> {
        let pool = self.pool()?;
        let mut out = vec![0.0; a.len()];
        let len = self.chunk_len(a.len());
        let finite = pool.install(|| {
            out.par_chunks_mut(len)
                .zip(a.par_chunks(len))
                .zip(b.par_chunks(len))
                .map(|((o, x), y)| {
                    for ((o, &x), &y) in o.iter_mut().zip(x).zip(y) {
                        *o = f(x, y);
                    }
                    o.iter().all(|v| v.is_finite())
                })
                .reduce(|| true, |p, q| p && q)
        });
        Some((out, finite))
    }
}
