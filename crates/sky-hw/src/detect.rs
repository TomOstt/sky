//! The OS probes: the only code in this crate that touches the machine.
//! Each probe returns `None` when the platform does not offer the fact.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::{parse, HardwareInfo};

/// The most bytes read from any file or command output. `/proc/cpuinfo` on a
/// 512-thread machine is about 1 MiB; this leaves room and still bounds a
/// pathological file.
const MAX_READ: u64 = 8 << 20;

pub(crate) fn detect() -> HardwareInfo {
    let macos = cfg!(target_os = "macos");
    let linux = cfg!(target_os = "linux");
    let cpuinfo = if linux { read("/proc/cpuinfo") } else { None };

    let available_parallelism = std::thread::available_parallelism().ok().map(|n| n.get());

    let mut simd_features = x86_features();
    if cfg!(target_arch = "aarch64") {
        // NEON (Advanced SIMD) is part of the AArch64 baseline.
        simd_features.push("neon".to_string());
        if linux {
            for f in cpuinfo
                .as_deref()
                .map(parse::cpuinfo_arm_simd)
                .unwrap_or_default()
            {
                if f != "neon" {
                    simd_features.push(f);
                }
            }
        }
    }

    let mut info = HardwareInfo {
        arch: std::env::consts::ARCH.to_string(),
        os: std::env::consts::OS.to_string(),
        model: None,
        simd_features,
        physical_cores: None,
        logical_cores: None,
        performance_cores: None,
        efficiency_cores: None,
        available_parallelism,
        l1d_bytes: None,
        l2_bytes: None,
        l3_bytes: None,
        cache_line_bytes: None,
        memory_bytes: None,
        page_bytes: None,
    };

    if macos {
        detect_macos(&mut info);
    } else if linux {
        detect_linux(&mut info, cpuinfo.as_deref());
    }

    info.model = info.model.take().and_then(|m| clean_model(&m));
    if info.page_bytes.is_none() {
        info.page_bytes = command("/usr/bin/getconf", &["PAGESIZE"])
            .or_else(|| command("/bin/getconf", &["PAGESIZE"]))
            .as_deref()
            .and_then(parse::positive);
    }
    if info.logical_cores.is_none() {
        info.logical_cores = available_parallelism;
    }
    // A physical count above the logical one is a contradiction (a
    // container's sysfs against its cgroup, say): trust neither half.
    if let (Some(p), Some(l)) = (info.physical_cores, info.logical_cores) {
        if p > l {
            info.physical_cores = None;
        }
    }
    info
}

/// SIMD features of an x86 CPU, oldest to newest. Empty on other
/// architectures.
fn x86_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut out: Vec<String> = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_x86_feature_detected!($name) {
                    out.push($name.to_string());
                })*
            };
        }
        probe!("sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx", "avx2", "avx512f", "fma");
    }
    out
}

fn detect_macos(info: &mut HardwareInfo) {
    let sysctl = |key: &str| {
        command("/usr/sbin/sysctl", &["-n", key]).or_else(|| command("/sbin/sysctl", &["-n", key]))
    };
    let number = |key: &str| sysctl(key).as_deref().and_then(parse::positive);
    let size = |key: &str| sysctl(key).as_deref().and_then(parse::cache_size);

    info.model = sysctl("machdep.cpu.brand_string");
    info.physical_cores = number("hw.physicalcpu").map(|n| n as usize);
    info.logical_cores = number("hw.logicalcpu").map(|n| n as usize);
    info.memory_bytes = number("hw.memsize");
    info.page_bytes = number("hw.pagesize");
    info.cache_line_bytes = number("hw.cachelinesize");

    // Apple Silicon has two clusters, and `hw.l2cachesize` describes one of
    // them (the efficiency cluster on the machines checked). Kernels run on
    // the performance cores, so their level (perflevel0) comes first.
    info.l1d_bytes = size("hw.perflevel0.l1dcachesize").or_else(|| size("hw.l1dcachesize"));
    info.l2_bytes = size("hw.perflevel0.l2cachesize").or_else(|| size("hw.l2cachesize"));
    info.l3_bytes = size("hw.l3cachesize");

    if let Some(perf) = number("hw.perflevel0.physicalcpu") {
        // The efficiency count is only meaningful next to a second level.
        if let Some(eff) = number("hw.perflevel1.physicalcpu") {
            info.performance_cores = Some(perf as usize);
            info.efficiency_cores = Some(eff as usize);
        }
    }
}

fn detect_linux(info: &mut HardwareInfo, cpuinfo: Option<&str>) {
    info.model = cpuinfo
        .and_then(parse::cpuinfo_model)
        .or_else(|| read("/sys/firmware/devicetree/base/model").map(clean_devicetree));
    info.memory_bytes = read("/proc/meminfo")
        .as_deref()
        .and_then(parse::meminfo_total);

    let cpu_dir = Path::new("/sys/devices/system/cpu");
    let mut cpus: Vec<u32> = Vec::new();
    if let Ok(entries) = fs::read_dir(cpu_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(n) = name.strip_prefix("cpu").and_then(|n| n.parse::<u32>().ok()) {
                cpus.push(n);
            }
        }
    }
    cpus.sort_unstable();

    // Online CPUs have a `topology` directory; offline ones do not.
    let mut cores: BTreeSet<(String, String)> = BTreeSet::new();
    let mut logical = 0usize;
    let mut capacities = Vec::new();
    for n in &cpus {
        let cpu = cpu_dir.join(format!("cpu{n}"));
        let topo = cpu.join("topology");
        let (Some(pkg), Some(core)) = (
            read(topo.join("physical_package_id")),
            read(topo.join("core_id")),
        ) else {
            continue;
        };
        logical += 1;
        cores.insert((pkg.trim().to_string(), core.trim().to_string()));
        if let Some(c) = read(cpu.join("cpu_capacity"))
            .as_deref()
            .and_then(parse::positive)
        {
            capacities.push(c);
        }
    }
    if logical > 0 {
        info.logical_cores = Some(logical);
        info.physical_cores = Some(cores.len());
    }
    if capacities.len() == logical && logical > 0 {
        if let Some((p, e)) = parse::hybrid_from_capacities(&capacities) {
            // Capacities are per logical CPU; hybrid Arm cores have no SMT,
            // so they are also core counts. With SMT the split would not be
            // trustworthy, so report it only when threads equal cores.
            if info.physical_cores == Some(logical) {
                info.performance_cores = Some(p);
                info.efficiency_cores = Some(e);
            }
        }
    }

    let mut entries = Vec::new();
    let cache_dir = cpu_dir.join("cpu0/cache");
    if let Ok(dir) = fs::read_dir(&cache_dir) {
        let mut indexes: Vec<_> = dir
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("index"))
            .map(|e| e.path())
            .collect();
        indexes.sort();
        for index in indexes {
            if let (Some(level), Some(kind), Some(size)) = (
                read(index.join("level")),
                read(index.join("type")),
                read(index.join("size")),
            ) {
                entries.push((level, kind, size));
            }
            if info.cache_line_bytes.is_none() {
                info.cache_line_bytes = read(index.join("coherency_line_size"))
                    .as_deref()
                    .and_then(parse::positive);
            }
        }
    }
    let (l1d, l2, l3) = parse::linux_caches(&entries);
    info.l1d_bytes = l1d;
    info.l2_bytes = l2;
    info.l3_bytes = l3;
}

/// A CPU model name reduced to printable ASCII (at most 96 characters), so
/// that a hostile or corrupt source cannot put terminal escapes or bidi
/// controls into what `sky hw` prints or into the fingerprint text.
/// Whitespace runs collapse to one space. `None` if nothing is left.
pub(crate) fn clean_model(text: &str) -> Option<String> {
    let kept: String = text
        .chars()
        .map(|c| if c.is_ascii_whitespace() { ' ' } else { c })
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(96)
        .collect();
    let collapsed = kept.split_whitespace().collect::<Vec<_>>().join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

/// A device-tree `model` file is NUL-terminated text.
fn clean_devicetree(text: String) -> String {
    text.trim_matches(|c: char| c == '\0' || c.is_whitespace())
        .to_string()
}

/// Reads a small text file, capped at [`MAX_READ`] bytes. `None` if it is
/// missing, unreadable or not UTF-8.
fn read(path: impl AsRef<Path>) -> Option<String> {
    use std::io::Read;
    let file = fs::File::open(path).ok()?;
    let mut text = String::new();
    file.take(MAX_READ).read_to_string(&mut text).ok()?;
    (!text.trim().is_empty()).then_some(text)
}

/// Runs `program` (an absolute path, so `PATH` cannot redirect it) with
/// `args` and returns its trimmed standard output; `None` if it is missing,
/// fails, or prints nothing. Standard input and error are closed.
fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.len() as u64 > MAX_READ {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::clean_model;

    #[test]
    fn models_are_reduced_to_printable_ascii() {
        assert_eq!(clean_model("Apple M2 Pro").as_deref(), Some("Apple M2 Pro"));
        assert_eq!(
            clean_model("Intel(R)  Xeon(R)\tCPU\n").as_deref(),
            Some("Intel(R) Xeon(R) CPU")
        );
        assert_eq!(
            clean_model("A\u{1b}[31mB\u{202e}C").as_deref(),
            Some("A[31mBC")
        );
        assert_eq!(clean_model("\u{1b}\u{202e}"), None);
        assert_eq!(clean_model(""), None);
        assert_eq!(clean_model(&"x".repeat(500)).unwrap().len(), 96);
    }
}
