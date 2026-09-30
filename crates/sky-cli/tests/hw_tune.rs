//! `sky hw` and `sky tune`. The performance database is redirected with
//! `SKY_TUNE_DB` (in the child's environment only), so no test touches the
//! user's real cache.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("hw-tune-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn sky(db: &PathBuf, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .env("SKY_TUNE_DB", db)
        .output()
        .expect("run sky")
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn hw_says_what_the_machine_has_and_what_sky_uses_it_for() {
    let dir = scratch("hw");
    let o = sky(&dir.join("unused.db"), &["hw"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    for needle in [
        "this machine",
        "architecture",
        "SIMD",
        "cores",
        "L2 cache",
        "fingerprint",
        "what Sky uses it for",
        "parallel kernels use",
        "matrix blocks",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
    assert!(
        text.contains("SIMD width") || text.contains("SIMD unknown"),
        "{text}"
    );
    // `sky hw` has no arguments, and reading the hardware writes nothing.
    assert_eq!(
        sky(&dir.join("unused.db"), &["hw", "x"]).status.code(),
        Some(2)
    );
    assert!(!dir.join("unused.db").exists());
}

#[test]
fn tune_show_reset_round_trip() {
    let dir = scratch("tune");
    let db = dir.join("cache").join("tune.db");

    let o = sky(&db, &["tune", "--show"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(out(&o).contains("no tuning results yet"), "{}", out(&o));
    assert!(!db.exists(), "--show writes nothing");

    let o = sky(&db, &["tune", "--budget-ms=20"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    assert!(text.contains("dot_blocked [2^18] unroll "), "{text}");
    assert!(text.contains("transpose_tiled [512x512] tile "), "{text}");
    assert!(text.contains("GB/s"), "{text}");
    assert!(text.contains("stored 2 results in"), "{text}");
    assert!(text.contains("<- chosen"), "{text}");

    let stored = fs::read_to_string(&db).unwrap();
    assert!(stored.contains("sky-tune-db 1"), "{stored}");
    assert_eq!(stored.matches("\nresult ").count(), 2, "{stored}");

    let o = sky(&db, &["tune", "--show"]);
    let text = out(&o);
    assert!(text.contains("this machine"), "{text}");
    assert!(text.contains("dot_blocked [2^18]"), "{text}");
    assert!(text.contains("transpose_tiled [512x512]"), "{text}");

    // A filter tunes one kernel and keeps the other's result.
    let o = sky(&db, &["tune", "--budget-ms=20", "dot"]);
    assert!(out(&o).contains("stored 1 result in"), "{}", out(&o));
    assert!(!out(&o).contains("transpose_tiled"));
    assert_eq!(
        fs::read_to_string(&db)
            .unwrap()
            .matches("\nresult ")
            .count(),
        2
    );

    let o = sky(&db, &["tune", "--reset"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(out(&o).starts_with("removed "), "{}", out(&o));
    assert!(!db.exists());
    let o = sky(&db, &["tune", "--reset"]);
    assert!(
        out(&o).contains("no tuning results to remove"),
        "{}",
        out(&o)
    );
}

#[test]
fn tune_usage_errors_exit_2() {
    let dir = scratch("usage");
    let db = dir.join("tune.db");
    for args in [
        &["tune", "--show", "--reset"][..],
        &["tune", "--bogus"],
        &["tune", "--budget-ms=0"],
        &["tune", "a", "b"],
        &["tune", "no-such-kernel"],
    ] {
        let o = sky(&db, args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", err(&o));
    }
    assert!(!db.exists());
}

#[test]
fn a_file_that_is_not_our_database_is_never_overwritten_or_deleted() {
    let dir = scratch("foreign");
    let db = dir.join("thesis.txt");
    fs::write(&db, "chapter one\n").unwrap();

    let o = sky(&db, &["tune", "--budget-ms=20"]);
    assert_eq!(o.status.code(), Some(1), "{}", out(&o));
    assert!(err(&o).contains("not overwriting"), "{}", err(&o));
    let o = sky(&db, &["tune", "--reset"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("not deleting"), "{}", err(&o));
    assert_eq!(fs::read_to_string(&db).unwrap(), "chapter one\n");
}

#[test]
fn malformed_entries_are_reported_by_show_and_ignored() {
    let dir = scratch("malformed");
    let db = dir.join("tune.db");
    fs::write(
        &db,
        "sky-tune-db 1\nresult fp=nothex kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=1 when=1\n",
    )
    .unwrap();
    let o = sky(&db, &["tune", "--show"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(err(&o).contains("bad fingerprint"), "{}", err(&o));
    assert!(out(&o).contains("no tuning results yet"), "{}", out(&o));
}
