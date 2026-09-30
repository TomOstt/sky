//! Native code for SIR through Cranelift (ROADMAP Phase 4: "speed on one
//! machine").
//!
//! This crate is the home of the native engine: SIR functions translated to
//! Cranelift IR and compiled in memory for the host CPU. The layout agreed
//! for Phase 4:
//!
//! * `module.rs`: whole-module translation (scalars, control flow, calls,
//!   errors, printing, arrays through runtime calls), `sky run
//!   --engine=native`, and `--emit=clif`.
//! * `kernel.rs`: compiled `fused` elementwise kernels, callable on their
//!   own (the SIR interpreter uses them for large arrays) and from `module`.
//! * `runtime.rs`: the `extern "C"` functions native code calls (math
//!   functions through `sky_interp::math::apply`, array operations, errors),
//!   so native results are bit-identical to the interpreters'. They are
//!   `#[no_mangle]`, so an object file can import them by name.
//! * `aot.rs`: the same translation written to an object file, for `sky
//!   build`; see [`aot`].
//!
//! The whole-module engine's entry points are [`compile`] (a SIR module to
//! machine code), [`Compiled::run`] and [`Compiled::run_or_rerun`], and
//! [`emit_clif`] (the Cranelift IR as text). It behaves exactly like the SIR
//! interpreter; `module.rs` says how.
//!
//! [`host_isa`] and [`smoke`] exist so every part starts from a toolchain
//! known to work on the host.

#![deny(unsafe_code)]

use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};

/// The target description for the CPU this process runs on, with
/// optimisation on (`opt_level = speed`).
pub fn host_isa() -> Result<OwnedTargetIsa, String> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    // Code is placed wherever the JIT's allocator puts it.
    flags.set("is_pic", "false").map_err(|e| e.to_string())?;
    // A function returns its status and all of its results (a derivative
    // function: status, value, derivative). x86-64 System V has two return
    // registers for integers and two for floats, so Cranelift refuses more
    // unless the extras go through a hidden return area, which this allows.
    // AArch64 has more return registers, which is why only x86-64 needs it.
    flags
        .set("enable_multi_ret_implicit_sret", "true")
        .map_err(|e| e.to_string())?;
    let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
    isa_builder
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())
}

pub mod aot;
mod icache;
pub mod kernel;
pub mod kernel_rt;
mod module;
mod runtime;
pub mod tuning;
/// A JIT builder for `isa` whose code and data live in **one** region of
/// `reserve` bytes, reserved up front (untouched pages cost no memory).
///
/// Compiled code reaches its own data and functions with 32-bit PC-relative
/// relocations on x86-64. cranelift-jit's default provider allocates code and
/// data separately, and on Linux they can land more than 2 GiB apart, which
/// makes finalizing panic (the differential fuzzer found it). An arena keeps
/// everything within reach; a module that outgrows it fails to compile and
/// the engine falls back, it never crashes.
pub(crate) fn jit_builder(
    isa: OwnedTargetIsa,
    reserve: usize,
) -> Result<cranelift_jit::JITBuilder, String> {
    let mut builder =
        cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let arena =
        cranelift_jit::ArenaMemoryProvider::new_with_size(reserve).map_err(|e| e.to_string())?;
    builder.memory_provider(Box::new(arena));
    Ok(builder)
}

/// The address space a whole program's native code may use.
pub(crate) const MODULE_ARENA_BYTES: usize = 256 << 20;
/// The address space one compiled `fused` kernel may use.
pub(crate) const KERNEL_ARENA_BYTES: usize = 16 << 20;

pub mod smoke;

pub use module::{compile, emit_clif, CompileError, Compiled, RunFailure, SkipWriter};
