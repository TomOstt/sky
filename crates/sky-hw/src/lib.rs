//! Hardware discovery for Sky (ROADMAP Phase 4, idea [20]).
//!
//! [`HardwareInfo::detect`] asks the machine what it is: the CPU model, the
//! SIMD features, the core counts, the cache sizes, the memory. It reads only
//! what `std` and the operating system offer (`is_x86_feature_detected!`,
//! `sysctl` on macOS, `/proc` and `/sys` on Linux) and uses no `unsafe` code.
//! Every fact is an `Option` or an empty list: a machine that will not say
//! something is "unknown", never an error, and Sky then uses a conservative
//! default.
//!
//! [`HardwareInfo::fingerprint`] is a stable hash of the parts that decide
//! which kernel parameters are fastest (architecture, model, features, cache
//! sizes). `sky-tune` keys its performance database by it, so results
//! measured on one machine are never applied to a different one.
//!
//! The parsers are pure functions over text (see [`parse`]), so they are
//! tested on every platform, not only the one that produces the text.

#![forbid(unsafe_code)]

mod detect;
pub mod parse;
mod report;

use std::sync::OnceLock;

pub use report::fmt_bytes;

/// What this machine can do. Every field that could not be discovered is
/// `None` (or empty), never a guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareInfo {
    /// CPU architecture (`std::env::consts::ARCH`: `aarch64`, `x86_64`, ...).
    pub arch: String,
    /// Operating system (`std::env::consts::OS`).
    pub os: String,
    /// Marketing name of the CPU (`Apple M2 Pro`), if the OS says.
    pub model: Option<String>,
    /// SIMD features the CPU supports, in a fixed order from oldest to
    /// newest (`sse2`, ..., `avx512f`, `fma`, or `neon`, `sve`, `sve2`).
    pub simd_features: Vec<String>,
    /// Physical cores.
    pub physical_cores: Option<usize>,
    /// Logical cores (hardware threads).
    pub logical_cores: Option<usize>,
    /// Performance cores, on machines with two kinds (Apple Silicon,
    /// big.LITTLE); `None` when the machine has one kind or Sky cannot tell.
    pub performance_cores: Option<usize>,
    /// Efficiency cores, on machines with two kinds.
    pub efficiency_cores: Option<usize>,
    /// How many threads this process may use at once (`std`'s
    /// `available_parallelism`, which respects affinity and cgroup limits).
    pub available_parallelism: Option<usize>,
    /// L1 data cache per core, in bytes.
    pub l1d_bytes: Option<u64>,
    /// L2 cache in bytes (on Apple Silicon, of the performance cluster).
    pub l2_bytes: Option<u64>,
    /// L3 cache in bytes.
    pub l3_bytes: Option<u64>,
    /// Cache line size in bytes.
    pub cache_line_bytes: Option<u64>,
    /// Total memory in bytes.
    pub memory_bytes: Option<u64>,
    /// Virtual-memory page size in bytes.
    pub page_bytes: Option<u64>,
}

impl HardwareInfo {
    /// Asks this machine what it can do. Runs a few short OS queries
    /// (`sysctl`, `getconf`) and reads a few files; takes milliseconds.
    /// Prefer [`current`] when the answer is needed more than once.
    pub fn detect() -> HardwareInfo {
        detect::detect()
    }

    /// A stable 16-hex-digit hash of the architecture, the model, the SIMD
    /// features and the cache sizes. Two runs on the same machine give the
    /// same fingerprint, on any Rust version (the hash is FNV-1a over a
    /// fixed text, not `std`'s unspecified hasher); machines that differ in
    /// any of those facts give different ones. Core counts and memory are
    /// left out on purpose: they change with cgroups and load, and the
    /// fastest block size does not depend on them.
    pub fn fingerprint(&self) -> String {
        format!("{:016x}", fnv1a(self.fingerprint_text().as_bytes()))
    }

    /// The text [`fingerprint`](Self::fingerprint) hashes, for tests and
    /// debugging. Feature order does not matter.
    pub fn fingerprint_text(&self) -> String {
        let mut features: Vec<&str> = self.simd_features.iter().map(String::as_str).collect();
        features.sort_unstable();
        features.dedup();
        let num = |v: Option<u64>| v.map_or_else(|| "?".to_string(), |v| v.to_string());
        format!(
            "sky-hw-fp1;arch={};model={};features={};l1d={};l2={};l3={}",
            self.arch,
            self.model.as_deref().unwrap_or("?"),
            features.join(","),
            num(self.l1d_bytes),
            num(self.l2_bytes),
            num(self.l3_bytes),
        )
    }

    /// The widest SIMD Sky can use here as `(name, bits)`, or `None` when the
    /// architecture is not known to have any (kernels then run scalar).
    /// AVX-512 counts only with `avx512f`; SVE is reported as a feature but
    /// Sky's kernels are written for fixed widths, so it stays at NEON's 128.
    pub fn simd_isa(&self) -> Option<(&'static str, u32)> {
        let has = |f: &str| self.simd_features.iter().any(|x| x == f);
        if has("avx512f") {
            Some(("AVX-512", 512))
        } else if has("avx2") {
            Some(("AVX2", 256))
        } else if has("avx") {
            Some(("AVX", 256))
        } else if has("sse2") {
            Some(("SSE2", 128))
        } else if has("neon") {
            Some(("NEON", 128))
        } else {
            None
        }
    }

    /// Lanes of `f64` in one SIMD register, or 1 when SIMD is unknown.
    pub fn f64_lanes(&self) -> usize {
        self.simd_isa().map_or(1, |(_, bits)| (bits / 64) as usize)
    }

    /// How many threads parallel kernels should use: the performance cores
    /// when the machine has two kinds (efficiency cores would only slow the
    /// slowest chunk), else the physical cores, else the logical ones, never
    /// more than this process may run at once, and at least 1.
    pub fn kernel_threads(&self) -> usize {
        let cores = self
            .performance_cores
            .filter(|&n| n > 0)
            .or(self.physical_cores)
            .or(self.logical_cores)
            .unwrap_or(1);
        let cap = self.available_parallelism.unwrap_or(usize::MAX);
        cores.min(cap).max(1)
    }

    /// Whether the machine has two kinds of cores and both counts are known.
    pub fn is_hybrid(&self) -> bool {
        matches!((self.performance_cores, self.efficiency_cores), (Some(p), Some(e)) if p > 0 && e > 0)
    }
}

/// [`HardwareInfo::detect`], run once per process.
pub fn current() -> &'static HardwareInfo {
    static CURRENT: OnceLock<HardwareInfo> = OnceLock::new();
    CURRENT.get_or_init(HardwareInfo::detect)
}

/// The fingerprint of this machine, computed once per process.
pub fn current_fingerprint() -> &'static str {
    static FP: OnceLock<String> = OnceLock::new();
    FP.get_or_init(|| current().fingerprint())
}

/// 64-bit FNV-1a: tiny, specified, and identical everywhere.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HardwareInfo {
        HardwareInfo {
            arch: "aarch64".into(),
            os: "macos".into(),
            model: Some("Apple M2 Pro".into()),
            simd_features: vec!["neon".into()],
            physical_cores: Some(10),
            logical_cores: Some(10),
            performance_cores: Some(6),
            efficiency_cores: Some(4),
            available_parallelism: Some(10),
            l1d_bytes: Some(131072),
            l2_bytes: Some(16 << 20),
            l3_bytes: None,
            cache_line_bytes: Some(128),
            memory_bytes: Some(16 << 30),
            page_bytes: Some(16384),
        }
    }

    #[test]
    fn fnv_matches_reference_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn fingerprint_is_stable_and_pinned() {
        // Pinned: changing the recipe silently would orphan every stored
        // tuning result, so it must be a deliberate, versioned change.
        let fp = sample().fingerprint();
        assert_eq!(fp.len(), 16);
        assert_eq!(fp, sample().fingerprint());
        assert_eq!(
            sample().fingerprint_text(),
            "sky-hw-fp1;arch=aarch64;model=Apple M2 Pro;features=neon;l1d=131072;l2=16777216;l3=?"
        );
    }

    #[test]
    fn fingerprint_depends_on_what_matters_only() {
        let base = sample();
        let mut other = base.clone();
        other.physical_cores = Some(2);
        other.memory_bytes = None;
        other.page_bytes = None;
        assert_eq!(base.fingerprint(), other.fingerprint());

        let mut a = base.clone();
        a.simd_features = vec!["sve".into(), "neon".into()];
        let mut b = base.clone();
        b.simd_features = vec!["neon".into(), "sve".into()];
        assert_eq!(
            a.fingerprint(),
            b.fingerprint(),
            "feature order is irrelevant"
        );
        assert_ne!(a.fingerprint(), base.fingerprint());

        let changes: [fn(&mut HardwareInfo); 5] = [
            |h| h.arch = "x86_64".into(),
            |h| h.model = None,
            |h| h.l2_bytes = Some(4 << 20),
            |h| h.l1d_bytes = None,
            |h| h.l3_bytes = Some(1),
        ];
        for change in changes {
            let mut h = base.clone();
            change(&mut h);
            assert_ne!(h.fingerprint(), base.fingerprint());
        }
    }

    #[test]
    fn simd_and_threads_follow_the_facts() {
        let mut h = sample();
        assert_eq!(h.simd_isa(), Some(("NEON", 128)));
        assert_eq!(h.f64_lanes(), 2);
        assert_eq!(h.kernel_threads(), 6);
        assert!(h.is_hybrid());

        h.simd_features = vec!["sse2".into(), "avx".into(), "avx2".into(), "fma".into()];
        assert_eq!(h.simd_isa(), Some(("AVX2", 256)));
        assert_eq!(h.f64_lanes(), 4);
        h.simd_features.push("avx512f".into());
        assert_eq!(h.f64_lanes(), 8);

        h.performance_cores = None;
        h.efficiency_cores = None;
        assert_eq!(h.kernel_threads(), 10);
        assert!(!h.is_hybrid());
        h.available_parallelism = Some(4);
        assert_eq!(h.kernel_threads(), 4, "a cgroup limit caps the threads");
        h.physical_cores = None;
        h.logical_cores = None;
        h.available_parallelism = None;
        assert_eq!(h.kernel_threads(), 1);
        h.simd_features.clear();
        assert_eq!(h.simd_isa(), None);
        assert_eq!(h.f64_lanes(), 1);
    }

    #[test]
    fn detect_returns_sane_facts_on_this_machine() {
        let h = HardwareInfo::detect();
        assert_eq!(h.arch, std::env::consts::ARCH);
        assert_eq!(h.os, std::env::consts::OS);
        assert!(h.kernel_threads() >= 1);
        assert_eq!(h.fingerprint(), HardwareInfo::detect().fingerprint());
        if let Some(p) = h.page_bytes {
            assert!(p.is_power_of_two() && p >= 1024, "page size {p}");
        }
        if let (Some(p), Some(l)) = (h.physical_cores, h.logical_cores) {
            assert!(p >= 1 && l >= p, "{p} physical, {l} logical");
        }
        for size in [h.l1d_bytes, h.l2_bytes, h.l3_bytes].into_iter().flatten() {
            assert!(size >= 1024, "cache size {size}");
        }
        if cfg!(target_arch = "aarch64") {
            assert!(h.simd_features.iter().any(|f| f == "neon"));
        }
        if cfg!(target_arch = "x86_64") {
            assert!(h.simd_features.iter().any(|f| f == "sse2"));
        }
    }
}
