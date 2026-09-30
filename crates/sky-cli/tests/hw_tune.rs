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

    // The demos, one at a time (the registry now holds heavier kernels too).
    let o = sky(&db, &["tune", "--budget-ms=20", "dot_blocked"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    assert!(text.contains("dot_blocked [2^18] unroll "), "{text}");
    assert!(text.contains("GB/s"), "{text}");
    assert!(text.contains("stored 1 result in"), "{text}");
    let o = sky(&db, &["tune", "--budget-ms=20", "transpose_tiled"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    assert!(text.contains("transpose_tiled [512x512] tile "), "{text}");
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

    let o = sky(&db, &["tune", "--budget-ms=20", "dot"]);
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

/// The linear algebra and interpreter tunables (the smallest size class of
/// each kernel, and the cheap thresholds: the big classes take minutes in an
/// unoptimised interpreter), through `sky tune --budget-ms`, with the
/// database in a temporary file.
#[test]
fn tune_measures_the_kernel_tunables_and_check_explain_cites_them() {
    let dir = scratch("kernels");
    let db = dir.join("tune.db");
    let names = sky_tune_names(&db);
    for needle in [
        "linalg_gemm_le64",
        "linalg_gemm_gt1024",
        "linalg_cholesky_le256",
        "linalg_threads",
        "interp_native",
        "interp_fused_work",
        "interp_array_elements",
    ] {
        assert!(
            names.contains(needle),
            "{needle} is not registered: {names}"
        );
    }

    let mut results = Vec::new();
    for name in [
        "linalg_gemm_le64",
        "linalg_cholesky_le64",
        "linalg_threads",
        "interp_native",
    ] {
        let o = sky(&db, &["tune", "--budget-ms=20", name]);
        assert_eq!(o.status.code(), Some(0), "{name}: {}", err(&o));
        let text = out(&o);
        assert!(text.contains("stored 1 result in"), "{text}");
        assert!(text.contains("<- chosen"), "{text}");
        results.push(text);
    }
    // Each candidate is listed with a rate (or milliseconds).
    assert!(results[0].contains("gemm [le64] "), "{}", results[0]);
    assert!(
        results[0].contains("kc 256, mc 128, nc 1024"),
        "{}",
        results[0]
    );
    assert!(results[0].contains("GFLOP/s"), "{}", results[0]);
    assert!(results[1].contains("cholesky [le64] nb "), "{}", results[1]);
    assert!(
        results[2].contains("threads [all] madds "),
        "{}",
        results[2]
    );
    assert!(
        results[3].contains("fused_native [cold] min_elements "),
        "{}",
        results[3]
    );

    let stored = fs::read_to_string(&db).unwrap();
    assert_eq!(stored.matches("\nresult ").count(), 4, "{stored}");
    let shown = out(&sky(&db, &["tune", "--show"]));
    for line in [
        "gemm [le64]",
        "cholesky [le64]",
        "threads [all]",
        "fused_native [cold]",
    ] {
        assert!(shown.contains(line), "{shown}");
    }

    // `sky check --explain` cites what was tuned where it picks a kernel.
    let src = dir.join("chain.sky");
    fs::write(
        &src,
        "A: Matrix[500, 500]\nB: Matrix[500, 500]\nC: Matrix[500, 500]\nv: Vector[500]\nx = A @ B @ C @ v\n",
    )
    .unwrap();
    let o = sky(&db, &["check", "--explain", src.to_str().unwrap()]);
    let text = err(&o);
    assert!(
        text.contains("GEMM blocks for sizes up to 64: tuned on this machine ("),
        "{text}"
    );
    // Only the classes that were tuned, and nothing without a database.
    assert!(!text.contains("sizes up to 256"), "{text}");
    let o = sky(
        &dir.join("none.db"),
        &["check", "--explain", src.to_str().unwrap()],
    );
    assert!(!err(&o).contains("tuned on this machine"), "{}", err(&o));
}

/// The registered names, from the error `sky tune` gives for a filter that
/// matches none of them.
fn sky_tune_names(db: &PathBuf) -> String {
    let o = sky(db, &["tune", "no-such-kernel"]);
    assert_eq!(o.status.code(), Some(2));
    err(&o)
}

/// Tuning never changes what a program prints: the same output with no
/// database, and with one holding the extreme values a hand edit could put
/// there (clamped by the kernels).
#[test]
fn a_tuned_database_changes_no_output() {
    let dir = scratch("same-output");
    let src = dir.join("prog.sky");
    fs::write(
        &src,
        "K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nf: Vector[2] = [1, 2]\nx = K⁻¹ @ f\nprint(x)\nprint(K @ K)\n",
    )
    .unwrap();
    let run = |db: &PathBuf| {
        let o = sky(db, &["run", src.to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(0), "{}", err(&o));
        out(&o)
    };
    let plain = run(&dir.join("none.db"));
    assert!(!plain.is_empty());

    let db = dir.join("tune.db");
    let fp = sky_hw::current_fingerprint();
    let row = |kernel: &str, class: &str, params: &str, default: &str| {
        format!(
            "result fp={fp} kernel={kernel} size={class} params={params} default={default} best_ns=1 default_ns=2 when=1790000000\n"
        )
    };
    let mut text = String::from("sky-tune-db 1\n");
    text += &row("gemm", "le64", "kc:3,mc:1,nc:9", "kc:256,mc:128,nc:1024");
    text += &row("cholesky", "le64", "nb:1000000", "nb:32");
    text += &row("threads", "all", "madds:1", "madds:2097152");
    text += &row("fused_native", "cold", "min_elements:0", "min_elements:512");
    text += &row("par_fused_work", "all", "work:0", "work:4096");
    text += &row("par_array_elements", "all", "elements:0", "elements:262144");
    fs::write(&db, text).unwrap();
    assert_eq!(run(&db), plain);
}
