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
//!   so native results are bit-identical to the interpreters'.
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
    let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
    isa_builder
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())
}

pub mod smoke;
