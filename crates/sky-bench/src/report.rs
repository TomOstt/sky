//! The table for people and the JSON for tracking.

use std::fmt::Write;

use crate::bench::{Config, Measurement, ENGINES, PHASES};
use crate::stats::Stats;

/// A duration with a readable unit: `812.3 ms`, `41.2 µs`.
pub fn duration(ns: u64) -> String {
    let n = ns as f64;
    if ns >= 1_000_000_000 {
        format!("{:.3} s", n / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", n / 1e6)
    } else if ns >= 1_000 {
        format!("{:.1} µs", n / 1e3)
    } else {
        format!("{ns} ns")
    }
}

fn cell(s: &Stats) -> String {
    format!("{} ±{:.0}%", duration(s.median), s.spread_percent())
}

fn ratio(num: &Stats, den: &Stats) -> String {
    if den.median == 0 {
        return "-".to_string();
    }
    format!("{:.2}x", num.median as f64 / den.median as f64)
}

/// The two tables: run times per engine, then compile-side phase times.
pub fn table(results: &[Measurement], cfg: Config) -> String {
    let mut out = String::new();
    let mode = if cfg.quick { ", quick sizes" } else { "" };
    let _ = writeln!(
        out,
        "run time, median of {} rep{} (±range as % of median{mode}); ratios are time relative to (a)",
        cfg.reps,
        if cfg.reps == 1 { "" } else { "s" },
    );
    let mut rows = vec![vec![
        "workload".to_string(),
        "N".to_string(),
        ENGINES[0].1.to_string(),
        ENGINES[1].1.to_string(),
        ENGINES[2].1.to_string(),
        ENGINES[3].1.to_string(),
        "b/a".to_string(),
        "c/a".to_string(),
        "c/b".to_string(),
        "d/c".to_string(),
        "output".to_string(),
    ]];
    for m in results {
        let [a, b, c, d] = &m.engines[..] else {
            continue;
        };
        rows.push(vec![
            m.name.to_string(),
            m.size.to_string(),
            cell(a),
            cell(b),
            cell(c),
            cell(d),
            ratio(b, a),
            ratio(c, a),
            ratio(c, b),
            ratio(d, c),
            if m.mismatch.is_none() {
                "same"
            } else {
                "DIFFERENT"
            }
            .to_string(),
        ]);
    }
    layout(&mut out, &rows);

    let _ = writeln!(
        out,
        "\ncompile side, median (lowered -> optimized SIR lines in the last column)"
    );
    let mut head = vec!["workload".to_string()];
    head.extend(PHASES.iter().map(|p| p.to_string()));
    head.push("SIR lines".to_string());
    let mut rows = vec![head];
    for m in results {
        let mut row = vec![m.name.to_string()];
        row.extend(m.phases.iter().map(|s| duration(s.median)));
        row.push(format!("{} -> {}", m.ir_lines.0, m.ir_lines.1));
        rows.push(row);
    }
    layout(&mut out, &rows);
    out
}

/// Left-aligns the first column and right-aligns the rest.
fn layout(out: &mut String, rows: &[Vec<String>]) {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..cols)
        .map(|c| {
            rows.iter()
                .filter_map(|r| r.get(c))
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    for (i, row) in rows.iter().enumerate() {
        let line: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(c, s)| {
                if c == 0 {
                    format!("{s:<w$}", w = widths[c])
                } else {
                    format!("{s:>w$}", w = widths[c])
                }
            })
            .collect();
        let _ = writeln!(out, "{}", line.join("  ").trim_end());
        if i == 0 {
            let _ = writeln!(
                out,
                "{}",
                "-".repeat(widths.iter().sum::<usize>() + 2 * (cols - 1))
            );
        }
    }
}

/// A JSON string literal. Control characters, quotes and backslashes are
/// escaped; other Unicode is written as is (JSON is UTF-8).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_stats(s: &Stats) -> String {
    let samples: Vec<String> = s.samples.iter().map(u64::to_string).collect();
    format!(
        "{{\"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\"spread_percent\":{:.3},\"samples_ns\":[{}]}}",
        s.median,
        s.min,
        s.max,
        s.spread_percent(),
        samples.join(",")
    )
}

/// The results as one JSON document, for tracking runs over time. `rev` is
/// whatever identifies the build (CI passes the commit SHA), or `None`.
///
/// Schema version 1. Times are integer nanoseconds; `run` is keyed by engine
/// (`tree_walker`, `sir`, `sir_opt`) and `compile` by phase (`lex`, `parse`,
/// `check`, `lower`, `pipeline`, `verify`); each holds the median, min, max,
/// spread and every sample.
pub fn json(results: &[Measurement], cfg: Config, rev: Option<&str>) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"schema\":1,\"tool\":\"sky-bench\",\"rev\":{},\"reps\":{},\"warmup\":{},\"quick\":{},\"programs\":[",
        rev.map_or_else(|| "null".to_string(), json_str),
        cfg.reps,
        cfg.warmup,
        cfg.quick
    );
    for (i, m) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let run: Vec<String> = ENGINES
            .iter()
            .zip(&m.engines)
            .map(|((key, _), s)| format!("{}:{}", json_str(key), json_stats(s)))
            .collect();
        let compile: Vec<String> = PHASES
            .iter()
            .zip(&m.phases)
            .map(|(key, s)| format!("{}:{}", json_str(key), json_stats(s)))
            .collect();
        let _ = write!(
            out,
            "{{\"name\":{},\"size\":{},\"output\":{},\"outputs_match\":{},\"ir_lines\":{{\"lowered\":{},\"optimized\":{}}},\"run\":{{{}}},\"compile\":{{{}}}}}",
            json_str(m.name),
            m.size,
            json_str(&m.output),
            m.mismatch.is_none(),
            m.ir_lines.0,
            m.ir_lines.1,
            run.join(","),
            compile.join(",")
        );
    }
    out.push_str("]}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_pick_a_unit() {
        assert_eq!(duration(12), "12 ns");
        assert_eq!(duration(41_200), "41.2 µs");
        assert_eq!(duration(2_500_000), "2.50 ms");
        assert_eq!(duration(3_100_000_000), "3.100 s");
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(
            json_str("a\"b\\c\n\u{1}m/s"),
            "\"a\\\"b\\\\c\\n\\u0001m/s\""
        );
    }

    #[test]
    fn json_and_table_carry_every_workload() {
        use crate::bench::{measure, programs};
        let cfg = Config {
            reps: 1,
            warmup: 0,
            quick: true,
        };
        let p = programs()
            .into_iter()
            .find(|p| p.name == "recursion")
            .expect("recursion workload");
        let m = measure(&p, cfg).expect("measures");
        let results = [m];
        let text = table(&results, cfg);
        assert!(text.contains("recursion"), "{text}");
        assert!(text.contains("same"), "{text}");
        let doc = json(&results, cfg, Some("abc"));
        for needle in [
            "\"schema\":1",
            "\"rev\":\"abc\"",
            "\"name\":\"recursion\"",
            "\"tree_walker\":{",
            "\"sir_opt\":{",
            "\"pipeline\":{",
            "\"outputs_match\":true",
        ] {
            assert!(doc.contains(needle), "{needle} missing from {doc}");
        }
        // Braces and brackets balance (outside strings, of which there are
        // none containing them here).
        assert_eq!(doc.matches('{').count(), doc.matches('}').count());
        assert_eq!(doc.matches('[').count(), doc.matches(']').count());
    }
}
