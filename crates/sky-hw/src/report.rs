//! The readable text of `sky hw`: what the machine has, and what Sky will
//! use each fact for.

use crate::HardwareInfo;

/// `16777216` as `16 MiB`. Exact multiples only lose nothing; others print
/// with one decimal (`1.5 MiB`).
pub fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("TiB", 1 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    ];
    for (name, size) in UNITS {
        if bytes >= size {
            return if bytes % size == 0 {
                format!("{} {name}", bytes / size)
            } else {
                format!("{:.1} {name}", bytes as f64 / size as f64)
            };
        }
    }
    format!("{bytes} B")
}

impl HardwareInfo {
    /// The facts, one per line, as `(label, value)` with `unknown` for
    /// anything the machine would not say.
    pub fn facts(&self) -> Vec<(&'static str, String)> {
        let unknown = || "unknown".to_string();
        let bytes = |v: Option<u64>| v.map_or_else(unknown, fmt_bytes);
        let count = |v: Option<usize>| v.map_or_else(unknown, |n| n.to_string());

        let cores = match (self.physical_cores, self.logical_cores) {
            (None, None) => unknown(),
            (p, l) => {
                let mut s = format!("{} physical, {} logical", count(p), count(l));
                if let (Some(p), Some(e)) = (self.performance_cores, self.efficiency_cores) {
                    s.push_str(&format!(" ({p} performance, {e} efficiency)"));
                }
                s
            }
        };
        let simd = if self.simd_features.is_empty() {
            "none detected".to_string()
        } else {
            self.simd_features.join(" ")
        };
        vec![
            ("CPU", self.model.clone().unwrap_or_else(unknown)),
            ("architecture", format!("{} ({})", self.arch, self.os)),
            ("SIMD", simd),
            ("cores", cores),
            ("L1d cache", bytes(self.l1d_bytes)),
            ("L2 cache", bytes(self.l2_bytes)),
            ("L3 cache", bytes(self.l3_bytes)),
            ("memory", bytes(self.memory_bytes)),
            ("page size", bytes(self.page_bytes)),
            ("fingerprint", self.fingerprint()),
        ]
    }

    /// What Sky will use the facts for, one sentence per fact. Facts the
    /// machine would not give say what Sky does instead.
    pub fn uses(&self) -> Vec<String> {
        let mut out = Vec::new();

        out.push(match self.simd_isa() {
            Some((name, bits)) => format!(
                "SIMD width {bits} bits ({name}): kernels use {} f64 lanes",
                self.f64_lanes()
            ),
            None => "SIMD unknown: kernels use scalar code".to_string(),
        });

        let threads = self.kernel_threads();
        out.push(match (self.physical_cores, self.logical_cores) {
            (None, None) if self.performance_cores.is_none() => {
                "cores unknown: parallel kernels use 1 thread".to_string()
            }
            (p, l) => {
                let total = p.or(l).unwrap_or(threads);
                let mut s = format!("{total} cores");
                if self.is_hybrid() {
                    s.push_str(&format!(
                        " ({} performance)",
                        self.performance_cores.unwrap_or(0)
                    ));
                }
                s.push_str(&format!(
                    ": parallel kernels use {threads} thread{}",
                    if threads == 1 { "" } else { "s" }
                ));
                s
            }
        });

        out.push(match (self.l2_bytes, self.l3_bytes, self.l1d_bytes) {
            (Some(l2), _, _) => format!("L2 {}: matrix blocks sized to fit", fmt_bytes(l2)),
            (None, Some(l3), _) => format!("L3 {}: matrix blocks sized to fit", fmt_bytes(l3)),
            (None, None, Some(l1)) => {
                format!("L1d {}: matrix blocks sized to fit", fmt_bytes(l1))
            }
            (None, None, None) => {
                "cache sizes unknown: matrix blocks use safe defaults".to_string()
            }
        });

        out.push(format!(
            "fingerprint {}: tuning results (`sky tune`) are stored for this machine only",
            self.fingerprint()
        ));
        out
    }

    /// The whole `sky hw` text: the facts, then the uses.
    pub fn report(&self) -> String {
        let facts = self.facts();
        let width = facts.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let mut text = String::from("this machine\n");
        for (label, value) in facts {
            text.push_str(&format!("  {label:<width$}  {value}\n"));
        }
        text.push_str("\nwhat Sky uses it for\n");
        for line in self.uses() {
            text.push_str(&format!("  {line}\n"));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown() -> HardwareInfo {
        HardwareInfo {
            arch: "riscv64".into(),
            os: "linux".into(),
            model: None,
            simd_features: vec![],
            physical_cores: None,
            logical_cores: None,
            performance_cores: None,
            efficiency_cores: None,
            available_parallelism: None,
            l1d_bytes: None,
            l2_bytes: None,
            l3_bytes: None,
            cache_line_bytes: None,
            memory_bytes: None,
            page_bytes: None,
        }
    }

    #[test]
    fn sizes_read_naturally() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(131072), "128 KiB");
        assert_eq!(fmt_bytes(16 << 20), "16 MiB");
        assert_eq!(fmt_bytes(3 << 29), "1.5 GiB");
        assert_eq!(fmt_bytes(16 << 30), "16 GiB");
    }

    #[test]
    fn an_apple_machine_reads_like_the_spec() {
        let h = HardwareInfo {
            arch: "aarch64".into(),
            os: "macos".into(),
            model: Some("Apple M2 Pro".into()),
            simd_features: vec!["neon".into()],
            physical_cores: Some(10),
            logical_cores: Some(10),
            performance_cores: Some(8),
            efficiency_cores: Some(2),
            available_parallelism: Some(10),
            l2_bytes: Some(16 << 20),
            ..unknown()
        };
        let uses = h.uses();
        assert_eq!(
            uses[0],
            "SIMD width 128 bits (NEON): kernels use 2 f64 lanes"
        );
        assert_eq!(
            uses[1],
            "10 cores (8 performance): parallel kernels use 8 threads"
        );
        assert_eq!(uses[2], "L2 16 MiB: matrix blocks sized to fit");
        let report = h.report();
        assert!(report.contains("8 performance, 2 efficiency"));
        assert!(report.contains("fingerprint"));
    }

    #[test]
    fn an_unknown_machine_degrades_gracefully() {
        let h = unknown();
        let uses = h.uses();
        assert_eq!(uses[0], "SIMD unknown: kernels use scalar code");
        assert_eq!(uses[1], "cores unknown: parallel kernels use 1 thread");
        assert_eq!(
            uses[2],
            "cache sizes unknown: matrix blocks use safe defaults"
        );
        let report = h.report();
        assert!(report.contains("unknown"));
        assert!(!report.contains("None"));
    }
}
