//! `sky tune`: measure the registered kernels on this machine and remember
//! the winners; `--show` lists the database; `--reset` clears it.
//!
//! Explicit and user-initiated: this is the only thing that writes the
//! performance database, and it holds timings, never programs or values (the
//! spec's "Hardware and tuning" section explains why this is not a saved
//! session).

use std::process::ExitCode;
use std::time::Duration;

use sky_diag::display_safe;
use sky_tune::{format_date, Config, Db, Registry, Store};

/// Everything `sky tune` can tune. Crates with tunable kernels add
/// themselves here, one line each, e.g. `sky_linalg::register(&mut r);`.
pub fn registry() -> Registry {
    Registry::with_demos()
}

/// What `sky tune` was asked to do.
#[derive(Debug, PartialEq, Eq)]
pub enum Args {
    /// Measure (the kernels whose name contains `filter`), with an optional
    /// per-tunable budget in milliseconds.
    Run {
        filter: Option<String>,
        budget_ms: Option<u64>,
    },
    /// List the database.
    Show,
    /// Delete the database.
    Reset,
}

/// Parses `sky tune`'s arguments. `Err` is a usage error: the message, if
/// it needs one beyond the usage text.
pub fn parse(args: &[String]) -> Result<Args, Option<String>> {
    let (mut show, mut reset, mut budget_ms, mut filter) = (false, false, None, None);
    for arg in args {
        if arg == "--show" {
            if std::mem::replace(&mut show, true) {
                return Err(None);
            }
        } else if arg == "--reset" {
            if std::mem::replace(&mut reset, true) {
                return Err(None);
            }
        } else if let Some(text) = arg.strip_prefix("--budget-ms=") {
            let ms: u64 = text
                .parse()
                .ok()
                .filter(|ms| (1..=600_000).contains(ms))
                .ok_or_else(|| {
                    Some(format!(
                        "--budget-ms needs a whole number of milliseconds from 1 to 600000, not `{}`",
                        display_safe(text)
                    ))
                })?;
            if budget_ms.replace(ms).is_some() {
                return Err(None);
            }
        } else if arg.starts_with("--") || filter.replace(arg.clone()).is_some() {
            return Err(None);
        }
    }
    match (show, reset, budget_ms, filter) {
        (true, false, None, None) => Ok(Args::Show),
        (false, true, None, None) => Ok(Args::Reset),
        (false, false, budget_ms, filter) => Ok(Args::Run { filter, budget_ms }),
        _ => Err(Some(
            "--show and --reset take no other arguments".to_string(),
        )),
    }
}

/// Runs `sky tune`. Exit status 0 on success, 1 when the database cannot be
/// written, 2 when no registered kernel matches the filter.
pub fn tune(args: Args) -> ExitCode {
    let Some(store) = Store::default_location() else {
        eprintln!("error: {}", sky_tune::TuneError::NoLocation);
        return ExitCode::FAILURE;
    };
    match args {
        Args::Show => show(&store),
        Args::Reset => reset(&store),
        Args::Run { filter, budget_ms } => run(&store, filter.as_deref(), budget_ms),
    }
}

fn shown_path(store: &Store) -> String {
    display_safe(&store.path().display().to_string()).into_owned()
}

fn run(store: &Store, filter: Option<&str>, budget_ms: Option<u64>) -> ExitCode {
    let registry = registry();
    let groups = registry.build(filter);
    if groups.iter().all(|(_, tunables)| tunables.is_empty()) {
        match filter {
            Some(f) => eprintln!(
                "error: no tunable kernel matches `{}` (available: {})",
                display_safe(f),
                registry.names().join(", ")
            ),
            None => eprintln!("error: no tunable kernels are registered"),
        }
        return ExitCode::from(2);
    }
    let mut config = Config::default();
    if let Some(ms) = budget_ms {
        config.budget = Duration::from_millis(ms);
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "note: this is a debug build; the timings describe unoptimized code (use a release build for results worth keeping)"
        );
    }
    let hw = sky_hw::current();
    println!(
        "tuning on {} (fingerprint {})",
        hw.model.as_deref().unwrap_or("an unknown CPU"),
        store.fingerprint()
    );
    let mut stored = 0;
    for tunable in groups.iter().flat_map(|(_, tunables)| tunables) {
        let outcome = match sky_tune::tune_in(store, tunable, &config) {
            Ok(outcome) => outcome,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        stored += 1;
        println!("\n{outcome}");
        for (i, m) in outcome.measurements.iter().enumerate() {
            let mark = match (i == outcome.winner, i == 0) {
                (true, true) => "  <- chosen (default)",
                (true, false) => "  <- chosen",
                (false, true) => "  (default)",
                (false, false) => "",
            };
            println!("    {:<16}{}{mark}", m.params.to_string(), outcome.rate(i));
        }
    }
    println!(
        "\nstored {stored} result{} in {}",
        if stored == 1 { "" } else { "s" },
        shown_path(store)
    );
    ExitCode::SUCCESS
}

fn show(store: &Store) -> ExitCode {
    let db = store.load();
    for w in &db.warnings {
        eprintln!("warning: {}", display_safe(w));
    }
    print!(
        "{}",
        show_text(&db, store.fingerprint(), &shown_path(store))
    );
    ExitCode::SUCCESS
}

/// The `--show` listing: this machine's results first, then other
/// machines'. (Values are validated on load, so they print as they are.)
fn show_text(db: &Db, fingerprint: &str, path: &str) -> String {
    let line = |r: &sky_tune::Record| {
        let outcome = if r.params == r.default_params {
            "the default was fastest".to_string()
        } else {
            format!(
                "{:.2}x faster than the default {}",
                r.speedup(),
                r.default_params.short()
            )
        };
        format!(
            "  {} [{}]: {}, {outcome} (tuned {})\n",
            r.kernel,
            r.size_class,
            r.params,
            format_date(r.when)
        )
    };
    let mut text = format!("performance database {path}\n");
    let (mine, others): (Vec<_>, Vec<_>) = db
        .records()
        .iter()
        .partition(|r| r.fingerprint == fingerprint);
    if db.records().is_empty() {
        text.push_str("no tuning results yet; `sky tune` measures this machine\n");
        return text;
    }
    text.push_str(&format!("\nthis machine ({fingerprint})\n"));
    if mine.is_empty() {
        text.push_str("  no results yet; run `sky tune`\n");
    }
    for r in mine {
        text.push_str(&line(r));
    }
    if !others.is_empty() {
        text.push_str("\nother machines (ignored here)\n");
        for r in others {
            text.push_str(&format!("  {}:", r.fingerprint));
            text.push_str(&line(r));
        }
    }
    text
}

fn reset(store: &Store) -> ExitCode {
    let result = store.reset();
    sky_tune::invalidate_cache();
    match result {
        Ok(true) => {
            println!("removed {}", shown_path(store));
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("no tuning results to remove ({})", shown_path(store));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", display_safe(&e.to_string()));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn argument_parsing() {
        assert_eq!(
            parse(&args(&[])),
            Ok(Args::Run {
                filter: None,
                budget_ms: None
            })
        );
        assert_eq!(parse(&args(&["--show"])), Ok(Args::Show));
        assert_eq!(parse(&args(&["--reset"])), Ok(Args::Reset));
        assert_eq!(
            parse(&args(&["dot", "--budget-ms=50"])),
            Ok(Args::Run {
                filter: Some("dot".into()),
                budget_ms: Some(50)
            })
        );
        for bad in [
            &["--show", "--reset"][..],
            &["--show", "dot"],
            &["--show", "--show"],
            &["--reset", "--budget-ms=5"],
            &["a", "b"],
            &["--nope"],
            &["--budget-ms=0"],
            &["--budget-ms=x"],
            &["--budget-ms=5", "--budget-ms=6"],
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
