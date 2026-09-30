//! The numerics mode: reproducible (the default) or relaxed.
//!
//! Every kernel in this crate has a **reproducible** form (the functions
//! that do not say otherwise): bit-identical results on every machine,
//! engine and thread count. The **relaxed** forms (`*_relaxed*`) trade that
//! for speed: they may fuse a multiply and an add into one rounding, sum in
//! an order that depends on the vector width or the thread count, and hand
//! the work to the platform's BLAS/LAPACK. They are separate functions, not
//! a flag on the reproducible ones, so a reproducible call can never change
//! meaning by accident.
//!
//! [`Numerics`] and [`set_numerics`] are how a *program run* chooses which
//! family the array operations of the interpreters use (`sky run
//! --numerics=relaxed` sets it once, before anything executes, as
//! `--threads` sets [`crate::set_threads`]). The kernels themselves never
//! read it; the callers that dispatch (`sky_types::linalg`) do.

use std::sync::atomic::{AtomicU8, Ordering};

/// Which family of kernels a run uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Numerics {
    /// Fixed order of every accumulation, no fused multiply-add, no vendor
    /// libraries: the same bits everywhere. The default.
    #[default]
    Reproducible,
    /// Fast: fused multiply-add, reassociated sums, threads-dependent
    /// sums, vendor BLAS/LAPACK. Results agree with the reproducible ones
    /// to rounding error, not bit for bit.
    Relaxed,
}

impl Numerics {
    /// The name the command line and `sky.toml` use.
    pub fn name(self) -> &'static str {
        match self {
            Numerics::Reproducible => "reproducible",
            Numerics::Relaxed => "relaxed",
        }
    }

    /// Parses [`Numerics::name`]'s text.
    pub fn parse(text: &str) -> Option<Numerics> {
        match text {
            "reproducible" => Some(Numerics::Reproducible),
            "relaxed" => Some(Numerics::Relaxed),
            _ => None,
        }
    }
}

static MODE: AtomicU8 = AtomicU8::new(0);

/// Chooses the numerics of the array operations that dispatch on the mode
/// (see the module docs), for the whole process from now on.
pub fn set_numerics(mode: Numerics) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

/// The numerics [`set_numerics`] chose ([`Numerics::Reproducible`] until
/// then).
pub fn numerics() -> Numerics {
    match MODE.load(Ordering::Relaxed) {
        1 => Numerics::Relaxed,
        _ => Numerics::Reproducible,
    }
}

/// Whether this machine has the fused multiply-add the relaxed kernels use:
/// always on aarch64 (NEON), and on x86-64 when the CPU reports AVX2 and
/// FMA3. Elsewhere the relaxed kernels run the plain arithmetic.
pub fn fma_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// The vendor library the relaxed kernels call on this platform, if any:
/// `Some("Accelerate")` on macOS.
pub fn vendor() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        Some("Accelerate")
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for m in [Numerics::Reproducible, Numerics::Relaxed] {
            assert_eq!(Numerics::parse(m.name()), Some(m));
        }
        assert_eq!(Numerics::parse("fast"), None);
        assert_eq!(Numerics::default(), Numerics::Reproducible);
    }
}
