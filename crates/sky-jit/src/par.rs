//! Threads for native code: the pool, and the elementwise loops that run on
//! it.
//!
//! What may run concurrently, and why that cannot change a result, is the
//! module documentation of `sky_ir::parallel`; the SIR interpreter and native
//! code share this machinery. Nothing here reads a clock, a thread id or a
//! counter: which chunk covers which elements is a function of the element
//! count and the [`Config`], so a run is the same on 2 threads and on 64.
//!
//! The chunking of a compiled `fused` kernel is [`crate::kernel::Kernel::run_par`]'s;
//! this file has the pool and the elementwise array operations that are not
//! kernels (`add`, `sub`, `mul`, `div`, `neg` on arrays).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use sky_diag::Span;
use sky_interp::{RuntimeError, Value};
use sky_ir::parallel::Config;
use sky_ir::OpKind;

/// A worker's stack. Workers run the whole call tree of a started call (in
/// the SIR interpreter, interpreter frames; in native code, native frames),
/// so they get room like `with_deep_stack` does, plus some for the work other
/// workers' waiting lets them pick up. Only the pages a worker touches are
/// ever committed.
pub const WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;

/// The pool for `threads` workers: created on first use and kept for the
/// life of the process, so a test suite or a benchmark that runs many
/// programs does not start threads for each. `None` if no threads could be
/// started, in which case everything runs sequentially.
pub fn pool_for(threads: usize) -> Option<Arc<ThreadPool>> {
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

/// The length of the chunks `n` elements are split into under `cfg`: the
/// same split for a kernel, an elementwise operation, and every engine.
pub fn chunk_len(cfg: &Config, n: usize) -> usize {
    n.div_ceil(cfg.chunks(n)).max(1)
}

/// `out[i] = f(a[i])`, and whether every entry is finite, in chunks on
/// `pool`.
pub fn map1(
    pool: &ThreadPool,
    cfg: &Config,
    a: &[f64],
    f: impl Fn(f64) -> f64 + Sync,
) -> (Vec<f64>, bool) {
    let mut out = vec![0.0; a.len()];
    let len = chunk_len(cfg, a.len());
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
    (out, finite)
}

/// `out[i] = f(a[i], b[i])`, and whether every entry is finite, in chunks on
/// `pool`.
pub fn map2(
    pool: &ThreadPool,
    cfg: &Config,
    a: &[f64],
    b: &[f64],
    f: impl Fn(f64, f64) -> f64 + Sync,
) -> (Vec<f64>, bool) {
    let mut out = vec![0.0; a.len()];
    let len = chunk_len(cfg, a.len());
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
    (out, finite)
}

/// An elementwise operation on arrays, run on the pool when the array is big
/// (`Config::array_elements`). `None` leaves the operation to the sequential
/// code (`sky_interp::arith`), which produces every error, so this only ever
/// takes the case that has no error before the entries are computed:
///
/// * `+`, `-`: two arrays of one shape and one dimension;
/// * `*`: an array and a number, or two arrays of one shape, whose
///   dimensions multiply;
/// * `/`: an array and a nonzero number, or two arrays of one shape with no
///   zero entry in the divisor, whose dimensions divide (a zero entry leaves
///   the operation to the sequential code, which reports the lowest failing
///   entry);
/// * `-x` of an array.
///
/// The entries are the same `f64` operations as the sequential loop's, so the
/// same bits. The one failure left is that an entry is not finite, one error
/// for the whole operation whichever entry (or thread) found it.
///
/// `operands` are the operation's operands; `span` is the operation's.
pub fn elementwise(
    kind: &OpKind,
    operands: &[&Value],
    span: Span,
    cfg: &Config,
) -> Option<Result<Value, RuntimeError>> {
    if !cfg.is_parallel() {
        return None;
    }
    let min = cfg.array_elements.max(2);
    if let OpKind::Neg = kind {
        let [Value::Array { dims, data, dim }] = operands else {
            return None;
        };
        if data.len() < min {
            return None;
        }
        let pool = pool_for(cfg.threads)?;
        // `0 - x`, as `sky_interp::array::negate`: a zero stays `+0`.
        let (out, _) = map1(&pool, cfg, data, |x| 0.0 - x);
        return Some(Ok(Value::array(dims.clone(), out, *dim)));
    }
    let [l, r] = operands else {
        return None;
    };
    let (dims, dim, mapped) = match (kind, l, r) {
        (
            OpKind::Add | OpKind::Sub,
            Value::Array {
                dims: d1,
                data: x,
                dim: da,
            },
            Value::Array {
                dims: d2,
                data: y,
                dim: db,
            },
        ) if d1 == d2 && da == db && x.len() >= min => {
            let f = if matches!(kind, OpKind::Add) {
                |p: f64, q: f64| p + q
            } else {
                |p: f64, q: f64| p - q
            };
            let pool = pool_for(cfg.threads)?;
            (d1, *da, map2(&pool, cfg, x, y, f))
        }
        (OpKind::Mul, Value::Number { value: k, dim: dk }, Value::Array { dims, data, dim })
        | (OpKind::Mul, Value::Array { dims, data, dim }, Value::Number { value: k, dim: dk })
            if data.len() >= min =>
        {
            let out = dim.checked_mul(*dk).ok()?;
            let k = *k;
            let pool = pool_for(cfg.threads)?;
            (dims, out, map1(&pool, cfg, data, |x| x * k))
        }
        (OpKind::Div, Value::Array { dims, data, dim }, Value::Number { value: k, dim: dk })
            if data.len() >= min && *k != 0.0 =>
        {
            let out = dim.checked_div(*dk).ok()?;
            let k = *k;
            let pool = pool_for(cfg.threads)?;
            (dims, out, map1(&pool, cfg, data, |x| x / k))
        }
        // Two arrays of one shape, entry by entry.
        (
            OpKind::Mul,
            Value::Array {
                dims: d1,
                data: x,
                dim: da,
            },
            Value::Array {
                dims: d2,
                data: y,
                dim: db,
            },
        ) if d1 == d2 && x.len() >= min => {
            let out = da.checked_mul(*db).ok()?;
            let pool = pool_for(cfg.threads)?;
            (d1, out, map2(&pool, cfg, x, y, |p, q| p * q))
        }
        (
            OpKind::Div,
            Value::Array {
                dims: d1,
                data: x,
                dim: da,
            },
            Value::Array {
                dims: d2,
                data: y,
                dim: db,
            },
        ) if d1 == d2 && x.len() >= min && y.iter().all(|&q| q != 0.0) => {
            let out = da.checked_div(*db).ok()?;
            let pool = pool_for(cfg.threads)?;
            (d1, out, map2(&pool, cfg, x, y, |p, q| p / q))
        }
        _ => return None,
    };
    let (out, finite) = mapped;
    if !finite {
        return Some(Err(sky_interp::arith::overflow(span)));
    }
    Some(Ok(Value::array(dims.clone(), out, dim)))
}
