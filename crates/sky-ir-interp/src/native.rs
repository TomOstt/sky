//! Native kernels for `fused` operations on large arrays.
//!
//! The interpreter runs a `fused` body element by element on `Value`s, which
//! costs on the order of a hundred nanoseconds per operation. For a long
//! array it hands the body to `sky_jit::kernel`, which compiles it once
//! (per operation and operand kinds, cached for the run) and runs it as
//! vector code. The result and the error are the interpreter's own: see the
//! exactness section of the `sky_jit::kernel` docs, and the differential
//! property test in this crate.
//!
//! # When
//!
//! Compiling a body takes 120 to 180 microseconds (measured on an Apple M2
//! Pro; `cargo bench -p sky-ir-interp --bench fused_threshold`), and the
//! interpreter's loop takes about 0.1 microseconds per element and
//! operation, so a single run of a two-operation body breaks even at about
//! 600 elements and a six-operation one at about 250. The threshold,
//! [`JitOptions::DEFAULT_THRESHOLD`] (512), is in that range. It applies to
//! the elements the interpreter has run for one `fused` operation in this
//! run, not to one call: an operation on 512 elements compiles at its first
//! run, and one on 64 elements (inside a loop, say) compiles at its eighth,
//! when the interpreter has spent about what compiling costs, so the total
//! is never more than about twice the better choice. The threshold, and
//! whether native code is used at all, can be set:
//!
//! * `SKY_JIT=off` (or `0`, `false`, `no`) turns it off for the process, for
//!   debugging;
//! * `SKY_JIT_MIN_ELEMENTS=<n>` sets the threshold (`0`: every `fused`);
//! * [`crate::call_with_jit`] takes the options for one run.

use std::collections::HashMap;
use std::sync::OnceLock;

use sky_interp::Value;
use sky_ir::{Function, OpData};
use sky_jit::kernel::{compile_kernel, Input, InputSig, Kernel};

use crate::IrRuntimeError;

/// Whether and when a run uses native kernels for `fused` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitOptions {
    /// Use native kernels at all.
    pub enabled: bool,
    /// Compile a `fused` once the interpreter has run this many elements of
    /// it (so at once for an array this long).
    pub min_elements: usize,
}

impl JitOptions {
    /// The element count from which compiling pays off on the machine the
    /// default was measured on; see the module docs. A run uses the count
    /// `sky tune` found best on this machine, if it has (`sky_jit::tuning`),
    /// and this otherwise.
    pub const DEFAULT_THRESHOLD: usize = sky_jit::tuning::DEFAULT_KERNEL_THRESHOLD;

    /// The options `SKY_JIT` and `SKY_JIT_MIN_ELEMENTS` ask for, read once
    /// per process.
    pub fn from_env() -> Self {
        static OPTIONS: OnceLock<JitOptions> = OnceLock::new();
        *OPTIONS.get_or_init(|| {
            Self::from_vars(
                std::env::var("SKY_JIT").ok().as_deref(),
                std::env::var("SKY_JIT_MIN_ELEMENTS").ok().as_deref(),
            )
        })
    }

    /// The options for the values of `SKY_JIT` and `SKY_JIT_MIN_ELEMENTS`.
    pub(crate) fn from_vars(jit: Option<&str>, min_elements: Option<&str>) -> Self {
        let off = jit.is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "off" | "0" | "false" | "no"
            )
        });
        JitOptions {
            enabled: !off,
            min_elements: min_elements
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(sky_jit::tuning::kernel_threshold),
        }
    }

    /// Native kernels never used: the interpreter's loop only.
    pub const OFF: JitOptions = JitOptions {
        enabled: false,
        min_elements: usize::MAX,
    };

    /// Native kernels for every `fused` (whatever its length).
    pub const ALWAYS: JitOptions = JitOptions {
        enabled: true,
        min_elements: 0,
    };
}

impl Default for JitOptions {
    fn default() -> Self {
        Self::from_env()
    }
}

/// The most operands of a `fused` that are run natively (their kinds and data
/// are gathered on the stack).
pub(crate) const MAX_INLINE: usize = 8;

/// How many `fused` operations ran natively in this process, for tests.
#[cfg(test)]
pub(crate) static NATIVE_RUNS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The kernels compiled so far in one run.
pub(crate) struct Jit {
    options: JitOptions,
    /// By the address of the `fused` operation (stable for the run, which
    /// borrows the module).
    kernels: HashMap<usize, Entry>,
    /// How many times a body was compiled (or refused), for tests.
    #[cfg(test)]
    pub(crate) compiles: usize,
}

/// What is known of one `fused` operation.
#[derive(Default)]
struct Entry {
    /// The operand kinds the kernel was compiled (or refused) for; empty
    /// before the first compile.
    sigs: Vec<InputSig>,
    /// The kernel, or `None` if the body was refused (so it is not tried
    /// again).
    kernel: Option<Kernel>,
    /// Elements the interpreter's loop has run for this operation so far.
    interpreted: usize,
}

/// What [`Jit::run`] did.
pub(crate) enum Native {
    /// The result's entries and dimension.
    Done(Vec<f64>, sky_units::Dimension),
    /// The kernel failed with the interpreter's error for these elements.
    Failed(IrRuntimeError),
    /// Not used: the caller runs the interpreter's loop.
    Declined,
}

impl Jit {
    pub(crate) fn new(options: JitOptions) -> Self {
        Jit {
            options,
            kernels: HashMap::new(),
            #[cfg(test)]
            compiles: 0,
        }
    }

    /// Whether native kernels are used at all.
    pub(crate) fn enabled(&self) -> bool {
        self.options.enabled
    }

    /// Runs `op` (a `fused` of `func`, operands `inputs`, `count` elements)
    /// natively if that is enabled, worth it (see the module docs), and the
    /// body compiles.
    #[cfg(test)]
    pub(crate) fn run<V: std::borrow::Borrow<Value>>(
        &mut self,
        func: &Function,
        op: &OpData,
        inputs: &[V],
        count: usize,
    ) -> Native {
        self.run_into(func, op, inputs, count, &mut Vec::new(), None)
    }

    /// [`Jit::run`] writing the result into `out` (cleared and resized to
    /// `count`, and taken out of it on success), so a caller that keeps
    /// buffers for the purpose allocates nothing: the interpreter takes `out`
    /// from the buffers of arrays that died.
    ///
    /// Nothing here allocates once the kernel is compiled: the operand kinds
    /// and data are gathered on the stack (a `fused` has few operands; one
    /// with more than [`MAX_INLINE`] is not run natively).
    ///
    /// With `par` (a run with threads), a kernel big enough for
    /// `Config::fused_work` runs in chunks on the pool
    /// (`sky_jit::kernel::Kernel::run_par`): the same bits and the same error.
    pub(crate) fn run_into<V: std::borrow::Borrow<Value>>(
        &mut self,
        func: &Function,
        op: &OpData,
        inputs: &[V],
        count: usize,
        out: &mut Vec<f64>,
        par: Option<&sky_ir::parallel::Config>,
    ) -> Native {
        if !self.options.enabled || inputs.len() > MAX_INLINE {
            return Native::Declined;
        }
        let n = inputs.len();
        let mut sigs = [InputSig {
            dim: sky_units::Dimension::DIMENSIONLESS,
            array: false,
        }; MAX_INLINE];
        let mut args = [Input::Scalar(0.0); MAX_INLINE];
        for (k, input) in inputs.iter().enumerate() {
            match input.borrow() {
                Value::Array { data, dim, .. } => {
                    sigs[k] = InputSig {
                        dim: *dim,
                        array: true,
                    };
                    args[k] = Input::Array(data);
                }
                Value::Number { value, dim } => {
                    sigs[k] = InputSig {
                        dim: *dim,
                        array: false,
                    };
                    args[k] = Input::Scalar(*value);
                }
                _ => return Native::Declined,
            }
        }
        let key = op as *const OpData as usize;
        let entry = self.kernels.entry(key).or_default();
        if entry.sigs != sigs[..n] {
            // Compile once the interpreter has spent about what compiling
            // costs on this operation (immediately, for one long array).
            entry.interpreted = entry.interpreted.saturating_add(count);
            if entry.interpreted < self.options.min_elements {
                return Native::Declined;
            }
            entry.kernel = compile_kernel(func, op, &sigs[..n]).ok();
            entry.sigs = sigs[..n].to_vec();
            #[cfg(test)]
            {
                self.compiles += 1;
            }
        }
        let Some(kernel) = &entry.kernel else {
            return Native::Declined;
        };
        #[cfg(test)]
        NATIVE_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        out.clear();
        out.resize(count, 0.0);
        let ran = match par {
            Some(cfg) => kernel.run_par(&args[..n], out, cfg),
            None => kernel.run(&args[..n], out),
        };
        match ran {
            Ok(()) => Native::Done(std::mem::take(out), kernel.output_dim()),
            Err(e) => Native::Failed(e.into()),
        }
    }
}
