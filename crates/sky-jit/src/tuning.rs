//! The tuned threshold above which a `fused` operation gets a native kernel.
//!
//! Compiling a body costs about what the interpreter's loop spends on some
//! hundreds of elements, so a short array is cheaper to interpret and a long
//! one to compile. Where the two cross depends on the machine, and `sky tune`
//! (see `sky_ir_interp::register`) measures it. The threshold decides only
//! which of two engines with the same results runs the loop, so tuning it
//! never changes an output.

use std::sync::OnceLock;

/// The built-in threshold, measured on one machine
/// (`cargo bench -p sky-ir-interp --bench fused_threshold`).
pub const DEFAULT_KERNEL_THRESHOLD: usize = 512;

/// The kernel name `sky tune` stores the threshold under (size class
/// `cold`, parameter `min_elements`).
pub const KERNEL: &str = "fused_native";

/// The elements a `fused` operation runs interpreted before its body is
/// compiled: what `sky tune` found best on this machine, else
/// [`DEFAULT_KERNEL_THRESHOLD`]. Read once per process, clamped to a sane
/// range.
pub fn kernel_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        sky_tune::lookup(KERNEL, "cold").map_or(DEFAULT_KERNEL_THRESHOLD, |p| {
            p.get_clamped("min_elements", DEFAULT_KERNEL_THRESHOLD as i64, 0, 1 << 24) as usize
        })
    })
}
