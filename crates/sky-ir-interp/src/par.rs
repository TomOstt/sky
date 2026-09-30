//! The parallel runtime: a shared thread pool per thread count, the plan
//! from `sky_ir::parallel`, and the deterministic elementwise loops.
//!
//! What may run in parallel and why it cannot change a result is the
//! module documentation of `sky_ir::parallel`; this file is the machinery.
//! Nothing here reads a clock, a thread id or a counter: which chunk covers
//! which elements is a function of the element count and the configuration,
//! so a run is the same on 2 threads and on 64. The pool and the elementwise
//! loops are `sky_jit::par`'s, shared with native code.

use std::sync::{Arc, OnceLock};

use rayon::ThreadPool;
use sky_ir::parallel::{Config, Plan};
use sky_ir::Module;

/// What one run needs to run in parallel.
pub(crate) struct ParRt {
    pub cfg: Config,
    pub plan: Plan,
    /// Started when the first parallel work asks for it: a run that never
    /// has any (most of them) never starts a thread. The pool is the one
    /// native code shares (`sky_jit::par`): one per thread count.
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
            .get_or_init(|| sky_jit::par::pool_for(self.cfg.threads))
            .as_deref()
    }
}
