//! Pure parsers for the text the operating system gives back. They take
//! `&str` and return facts, so they run (and are tested) on every platform.
//! Anything they cannot make sense of is `None`: never a panic, never a
//! guess.

/// A size such as `32K`, `1024K`, `32M` (Linux sysfs) or plain digits
/// (bytes, as `sysctl` prints). Anything else, and any overflow, is `None`.
pub fn cache_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, shift) = match text.chars().last()? {
        'K' | 'k' => (&text[..text.len() - 1], 10),
        'M' | 'm' => (&text[..text.len() - 1], 20),
        'G' | 'g' => (&text[..text.len() - 1], 30),
        _ => (text, 0),
    };
    let n: u64 = digits.trim().parse().ok()?;
    let bytes = n.checked_mul(1u64 << shift)?;
    (bytes > 0).then_some(bytes)
}

/// A positive integer (a core count, a page size).
pub fn positive(text: &str) -> Option<u64> {
    text.trim().parse::<u64>().ok().filter(|&n| n > 0)
}

/// Total memory in bytes from the text of `/proc/meminfo` (`MemTotal:
/// 16384 kB`).
pub fn meminfo_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let mut parts = line["MemTotal:".len()..].split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let unit = match parts.next() {
        Some("kB") | Some("KB") | Some("kb") => 1u64 << 10,
        Some("mB") | Some("MB") => 1u64 << 20,
        None => 1,
        Some(_) => return None,
    };
    n.checked_mul(unit).filter(|&b| b > 0)
}

/// The value of the first `key : value` line of `/proc/cpuinfo` whose key is
/// one of `keys`, tried in order (so the more specific key wins).
fn cpuinfo_value(text: &str, keys: &[&str]) -> Option<String> {
    for key in keys {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once(':') {
                let v = v.trim();
                if k.trim().eq_ignore_ascii_case(key) && !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// The CPU model name from `/proc/cpuinfo`: `model name` on x86, `Model
/// Name` or `Hardware` or `Model` on Arm and others.
pub fn cpuinfo_model(text: &str) -> Option<String> {
    cpuinfo_value(
        text,
        &["model name", "Hardware", "Model", "cpu model", "cpu"],
    )
}

/// SIMD features an Arm `/proc/cpuinfo` lists in its `Features` line, as Sky
/// names them: `neon` (`asimd`; Armv8 always has it), `sve`, `sve2`.
pub fn cpuinfo_arm_simd(text: &str) -> Vec<String> {
    let Some(features) = cpuinfo_value(text, &["Features"]) else {
        return Vec::new();
    };
    let has = |f: &str| features.split_whitespace().any(|x| x == f);
    let mut out = Vec::new();
    if has("asimd") || has("neon") {
        out.push("neon".to_string());
    }
    if has("sve") {
        out.push("sve".to_string());
    }
    if has("sve2") {
        out.push("sve2".to_string());
    }
    out
}

/// L1 data, L2 and L3 sizes from Linux cache descriptions, one
/// `(level, type, size)` triple per `index*` directory (`"1"`, `"Data"`,
/// `"32K"`). An instruction cache is never taken for L1d. The first match
/// wins (they come from cpu0, so they are the caches of one core).
pub fn linux_caches(
    entries: &[(String, String, String)],
) -> (Option<u64>, Option<u64>, Option<u64>) {
    let find = |level: &str, kinds: &[&str]| {
        entries
            .iter()
            .filter(|(l, t, _)| l.trim() == level && kinds.contains(&t.trim()))
            .find_map(|(_, _, size)| cache_size(size))
    };
    (
        find("1", &["Data", "Unified"]),
        find("2", &["Unified", "Data"]),
        find("3", &["Unified", "Data"]),
    )
}

/// Performance and efficiency core counts from the per-CPU `cpu_capacity`
/// values Linux publishes on heterogeneous Arm machines. A core with less
/// than half the biggest capacity is an efficiency core, the rest are
/// performance cores. `None` when all capacities are equal (one kind of
/// core) or there are none.
pub fn hybrid_from_capacities(capacities: &[u64]) -> Option<(usize, usize)> {
    let max = *capacities.iter().max()?;
    let efficiency = capacities.iter().filter(|&&c| c * 2 < max).count();
    (max > 0 && efficiency > 0).then_some((capacities.len() - efficiency, efficiency))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sizes() {
        assert_eq!(cache_size("32K"), Some(32 << 10));
        assert_eq!(cache_size("1024K\n"), Some(1 << 20));
        assert_eq!(cache_size("32M"), Some(32 << 20));
        assert_eq!(cache_size("1G"), Some(1 << 30));
        assert_eq!(cache_size("16777216"), Some(16 << 20));
        assert_eq!(cache_size(""), None);
        assert_eq!(cache_size("K"), None);
        assert_eq!(cache_size("0"), None);
        assert_eq!(cache_size("-4K"), None);
        assert_eq!(cache_size("lots"), None);
        assert_eq!(cache_size("99999999999999999999G"), None);
        assert_eq!(cache_size("18446744073709551615G"), None);
        assert_eq!(cache_size("é"), None);
    }

    #[test]
    fn meminfo() {
        let text = "MemTotal:       16384000 kB\nMemFree: 1 kB\n";
        assert_eq!(meminfo_total(text), Some(16384000 * 1024));
        assert_eq!(meminfo_total("MemFree: 1 kB"), None);
        assert_eq!(meminfo_total("MemTotal: x kB"), None);
        assert_eq!(meminfo_total("MemTotal: 5 furlongs"), None);
    }

    #[test]
    fn cpuinfo() {
        let x86 = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n";
        assert_eq!(
            cpuinfo_model(x86).as_deref(),
            Some("Intel(R) Xeon(R) CPU @ 2.20GHz")
        );
        let arm =
            "processor\t: 0\nFeatures\t: fp asimd evtstrm sve sve2 cpuid\nCPU part\t: 0xd40\n";
        assert_eq!(cpuinfo_model(arm), None);
        assert_eq!(cpuinfo_arm_simd(arm), ["neon", "sve", "sve2"]);
        assert_eq!(cpuinfo_arm_simd("Features\t: fp asimd\n"), ["neon"]);
        assert!(cpuinfo_arm_simd("Features\t: fp\n").is_empty());
        assert!(cpuinfo_arm_simd("").is_empty());
        // `sve` must match a whole word, not `svebf16`.
        assert_eq!(cpuinfo_arm_simd("Features\t: asimd svebf16\n"), ["neon"]);
    }

    #[test]
    fn linux_cache_levels() {
        let e = |l: &str, t: &str, s: &str| (l.to_string(), t.to_string(), s.to_string());
        let entries = [
            e("1", "Data", "48K"),
            e("1", "Instruction", "32K"),
            e("2", "Unified", "2048K"),
            e("3", "Unified", "32M"),
        ];
        assert_eq!(
            linux_caches(&entries),
            (Some(48 << 10), Some(2 << 20), Some(32 << 20))
        );
        let only_icache = [e("1", "Instruction", "32K")];
        assert_eq!(linux_caches(&only_icache), (None, None, None));
        let junk = [e("2", "Unified", "many")];
        assert_eq!(linux_caches(&junk), (None, None, None));
        assert_eq!(linux_caches(&[]), (None, None, None));
    }

    #[test]
    fn hybrid_capacities() {
        assert_eq!(hybrid_from_capacities(&[1024; 8]), None);
        assert_eq!(hybrid_from_capacities(&[]), None);
        assert_eq!(hybrid_from_capacities(&[0, 0]), None);
        assert_eq!(
            hybrid_from_capacities(&[1024, 1024, 1024, 1024, 430, 430, 430, 430]),
            Some((4, 4))
        );
        // Three tiers: only the small cores are "efficiency".
        assert_eq!(
            hybrid_from_capacities(&[1024, 700, 700, 700, 300, 300, 300, 300]),
            Some((4, 4))
        );
    }

    #[test]
    fn positive_numbers() {
        assert_eq!(positive(" 16384\n"), Some(16384));
        assert_eq!(positive("0"), None);
        assert_eq!(positive("x"), None);
    }
}
