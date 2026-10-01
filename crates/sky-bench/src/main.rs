//! `sky-bench`: measures Sky's engines on the programs in `benches/programs`.
//!
//! ```text
//! cargo run --release -p sky-bench -- [--filter X] [--json out.json]
//!                                     [--reps N] [--warmup N] [--quick] [--list]
//! cargo run --release -p sky-bench -- compare [--filter X] [--reps N] ...
//! ```
//!
//! `sky-bench par [--threads 1,2,4,8] [--reps N] [--quick]` measures automatic
//! parallelism instead (speedups over threads, and the break-even sizes the
//! default thresholds come from): see `par.rs`.
//! The first form times Sky's own engines; `compare` times Sky against C, Rust,
//! Python and BLAS/LAPACK and writes `benches/compare/RESULTS.md` (see
//! `compare/mod.rs`).
//!
//! See the "Benchmarks" section of docs/guide.md. Not shipped: nothing depends
//! on this crate, and it uses only `std` for timing.

mod bench;
mod compare;
mod native_par;
mod par;
mod report;
mod stats;

use std::process::ExitCode;

use bench::Config;

const USAGE: &str = "usage: sky-bench [--filter <substring>] [--json <file>] [--reps <n>]\n\
                     \x20                [--warmup <n>] [--quick] [--list]\n\n\
                     --filter   only workloads whose name contains the text\n\
                     --json     also write the results as JSON to the file\n\
                     --reps     timed repetitions per measurement (default 5)\n\
                     --warmup   untimed runs of each engine first (default 1)\n\
                     --quick    tiny sizes, for a smoke run\n\
                     --list     print the workload names and exit\n\n\
                     sky-bench par [--threads 1,2,4,8] [--reps <n>] [--quick] [--native]\n\
                     \x20          measure automatic parallelism (speedups, break-even sizes);\n\
                     \x20          --native: on native code instead of the SIR interpreter\n\n\
                     Run it with `cargo run --release -p sky-bench`: debug timings mean nothing.\n\n\
                     `sky-bench compare --help` for the cross-language comparison.";

struct Args {
    filter: Option<String>,
    json: Option<String>,
    cfg: Config,
    list: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut out = Args {
        filter: None,
        json: None,
        cfg: Config {
            reps: 5,
            warmup: 1,
            quick: false,
        },
        list: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        let number = |flag: &str, v: String| {
            v.parse::<usize>()
                .map_err(|_| format!("{flag} needs a whole number, found `{v}`"))
        };
        match arg.as_str() {
            "--filter" => out.filter = Some(value("--filter")?),
            "--json" => out.json = Some(value("--json")?),
            "--reps" => {
                out.cfg.reps = number("--reps", value("--reps")?)?;
                if out.cfg.reps == 0 {
                    return Err("--reps must be at least 1".to_string());
                }
            }
            "--warmup" => out.cfg.warmup = number("--warmup", value("--warmup")?)?,
            "--quick" => out.cfg.quick = true,
            "--list" => out.list = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(out)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("par") {
        return match par::parse_args(&args[1..]) {
            Ok(o) => {
                par::main(&o);
                ExitCode::SUCCESS
            }
            Err(msg) => {
                eprintln!("error: {msg}\n\n{USAGE}");
                ExitCode::from(2)
            }
        };
    }
    if args.first().map(String::as_str) == Some("compare") {
        return compare::main(&args[1..]);
    }
    let args = match parse_args(&args) {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let selected: Vec<_> = bench::programs()
        .into_iter()
        .filter(|p| {
            args.filter
                .as_ref()
                .is_none_or(|f| p.name.contains(f.as_str()))
        })
        .collect();
    if args.list {
        for p in &selected {
            println!("{}", p.name);
        }
        return ExitCode::SUCCESS;
    }
    if selected.is_empty() {
        eprintln!("error: no workload matches the filter");
        return ExitCode::from(2);
    }
    if cfg!(debug_assertions) {
        eprintln!("warning: a debug build; use `cargo run --release -p sky-bench`\n");
    }

    let mut results = Vec::new();
    let mut failed = false;
    for p in &selected {
        eprintln!("running {} ...", p.name);
        match bench::measure(p, args.cfg) {
            Ok(m) => {
                if let Some(why) = &m.mismatch {
                    eprintln!("error: {}: the engines disagree: {why}", p.name);
                    failed = true;
                }
                results.push(m);
            }
            Err(e) => {
                eprintln!("error: {e}");
                failed = true;
            }
        }
    }
    println!("{}", report::table(&results, args.cfg));
    if let Some(path) = &args.json {
        let rev = std::env::var("SKY_BENCH_REV").ok();
        let doc = report::json(&results, args.cfg, rev.as_deref());
        if let Err(e) = std::fs::write(path, doc) {
            eprintln!("error: cannot write {path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn flags_parse() {
        let a = parse(&[
            "--filter", "fib", "--json", "o.json", "--reps", "3", "--quick",
        ])
        .expect("valid");
        assert_eq!(a.filter.as_deref(), Some("fib"));
        assert_eq!(a.json.as_deref(), Some("o.json"));
        assert_eq!((a.cfg.reps, a.cfg.warmup, a.cfg.quick), (3, 1, true));
    }

    #[test]
    fn bad_flags_are_usage_errors() {
        assert!(parse(&["--reps"]).is_err());
        assert!(parse(&["--reps", "0"]).is_err());
        assert!(parse(&["--reps", "x"]).is_err());
        assert!(parse(&["--nope"]).is_err());
    }
}
