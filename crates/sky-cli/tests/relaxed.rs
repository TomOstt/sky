//! Relaxed numerics end to end (spec "Relaxed numerics"): how a run opts in,
//! what it still guarantees, how it reports itself, and that every engine
//! respects the mode.
//!
//! Runs are subprocesses of the `sky` binary in a scratch directory with a
//! `.git`, so project discovery stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("relaxed-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// `sky <args> prog.sky` on `src`, with `sky.toml` if given.
fn sky_in(name: &str, args: &[&str], manifest: Option<&str>, src: &str) -> Output {
    let dir = scratch(name);
    fs::write(dir.join("prog.sky"), src).unwrap();
    if let Some(m) = manifest {
        fs::write(dir.join("sky.toml"), m).unwrap();
    }
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg("prog.sky")
        .env_remove("SKY_THREADS")
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn sky(name: &str, args: &[&str], src: &str) -> Output {
    sky_in(name, args, None, src)
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

const NOTE_START: &str = "note: results computed in relaxed numerics (";
const NOTE_END: &str = "): not bit-reproducible across machines\n";

/// stderr without the closing relaxed note (which a relaxed run that
/// printed something ends with).
fn without_note(stderr: &str) -> &str {
    match stderr.rfind(NOTE_START) {
        Some(i) if stderr[i..].ends_with(NOTE_END) => &stderr[..i],
        _ => stderr,
    }
}

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

/// A Sky literal for a matrix (17 significant digits round-trip a double).
fn matrix_literal(a: &[f64], n: usize, m: usize) -> String {
    let rows: Vec<String> = (0..n)
        .map(|i| {
            let row: Vec<String> = (0..m).map(|j| format!("{:.17}", a[i * m + j])).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

fn vector_literal(v: &[f64]) -> String {
    let e: Vec<String> = v.iter().map(|x| format!("{x:.17}")).collect();
    format!("[{}]", e.join(", "))
}

/// Every number printed, in order.
fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | '+')))
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|_| panic!("not a number: {t:?} in {text}"))
        })
        .collect()
}

/// `got` and `want` agree to `tol` of the largest magnitude in `want`.
fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
    let scale = want.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-300);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}: entry {i}: {g} vs {w} (tolerance {tol} of {scale})"
        );
    }
}

/// A function whose parameter is *declared* PositiveDefinite: the promise is
/// checked when it runs (a literal is proven only up to 32 by 32), by the
/// Cholesky factorization the solve is rewritten to.
fn pd_solvers(n: usize) -> String {
    format!(
        "def solve_pd(K: PositiveDefinite Matrix[{n}, {n}], b: Vector[{n}]) -> Vector[{n}]:\n    return K⁻¹ @ b\n\n\
         def solve_pd_right(K: PositiveDefinite Matrix[{n}, {n}], x: Vector[{n}]) -> Vector[{n}]:\n    return x @ K⁻¹\n\n"
    )
}

/// A program that runs every array kernel the modes differ in, on matrices
/// big enough to take the vendor path: a product, a Gram product, a dot
/// product, a norm, Cholesky solves (`A` is a 32 by 32 literal the checker
/// proves PositiveDefinite, the most it proves) and LU solves (`M` is not
/// tagged), on matrices of order `n`.
fn kernels_program(n: usize) -> String {
    let mut rng = Rng(42);
    let b: Vec<f64> = (0..n * n).map(|_| rng.unit()).collect();
    // A = C Cᵀ + 32 I: symmetric positive definite, exactly symmetric (each
    // entry is computed once and copied), well conditioned.
    let na = 32;
    let c: Vec<f64> = (0..na * na).map(|_| rng.unit()).collect();
    let mut a = vec![0.0; na * na];
    for i in 0..na {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..na {
                s += c[i * na + k] * c[j * na + k];
            }
            if i == j {
                s += na as f64;
            }
            a[i * na + j] = s;
            a[j * na + i] = s;
        }
    }
    let w: Vec<f64> = (0..na).map(|_| rng.unit()).collect();
    // M: diagonally dominant, not symmetric.
    let mut m: Vec<f64> = (0..n * n).map(|_| rng.unit()).collect();
    for i in 0..n {
        m[i * n + i] += n as f64;
    }
    let u: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    let v: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    format!(
        "A: PositiveDefinite Matrix[{na}, {na}] = {}\n\
         w: Vector[{na}] = {}\n\
         M: Matrix[{n}, {n}] = {}\n\
         B: Matrix[{n}, {n}] = {}\n\
         u: Vector[{n}] = {}\n\
         v: Vector[{n}] = {}\n\
         print(M @ B)\n\
         print(B @ Bᵀ)\n\
         print(dot(u, v))\n\
         print(norm(B))\n\
         print(A⁻¹ @ w)\n\
         print(w @ A⁻¹)\n\
         print(M⁻¹ @ v)\n\
         print(u @ M⁻¹)\n",
        matrix_literal(&a, na, na),
        vector_literal(&w),
        matrix_literal(&m, n, n),
        matrix_literal(&b, n, n),
        vector_literal(&u),
        vector_literal(&v),
    )
}

// ----- opting in -----

#[test]
fn an_unknown_mode_is_a_usage_error() {
    for cmd in ["run", "check"] {
        let o = sky("bad", &[cmd, "--numerics=fast"], "print(1)\n");
        assert_eq!(o.status.code(), Some(2), "{cmd}");
        assert!(
            err(&o).contains("unknown numerics `fast` (use `reproducible` or `relaxed`)"),
            "{}",
            err(&o)
        );
        assert!(out(&o).is_empty());
        let o = sky(
            "twice",
            &[cmd, "--numerics=relaxed", "--numerics=relaxed"],
            "1\n",
        );
        assert_eq!(o.status.code(), Some(2), "{cmd}: a flag twice");
    }
}

#[test]
fn the_default_run_is_silent_and_the_flag_names_it() {
    let src = "A = [[1.0, 2.0], [3.0, 4.0]]\nprint(A @ A)\nprint(dot([1, 2, 3], [4, 5, 6]))\n";
    let plain = sky("plain", &["run"], src);
    let named = sky("named", &["run", "--numerics=reproducible"], src);
    assert_eq!(out(&plain), "[[7, 10], [15, 22]]\n32\n");
    assert_eq!(plain.status.code(), Some(0));
    assert!(err(&plain).is_empty(), "{}", err(&plain));
    assert_eq!(out(&named), out(&plain));
    assert_eq!(err(&named), err(&plain));
}

#[test]
fn a_relaxed_run_that_printed_says_so_once_on_stderr() {
    let src = "A = [[1.0, 2.0], [3.0, 4.0]]\nprint(A @ A)\nprint(dot([1, 2, 3], [4, 5, 6]))\n";
    let o = sky("note", &["run", "--numerics=relaxed"], src);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o), "[[7, 10], [15, 22]]\n32\n");
    assert_eq!(
        err(&o),
        "note: results computed in relaxed numerics (--numerics=relaxed): not bit-reproducible across machines\n"
    );
    // Nothing printed, nothing to say.
    let o = sky("quiet", &["run", "--numerics=relaxed"], "x = 1 + 1\n");
    assert_eq!(o.status.code(), Some(0));
    assert!(out(&o).is_empty() && err(&o).is_empty(), "{}", err(&o));
}

#[test]
fn sky_toml_opts_a_project_in_and_the_flag_wins() {
    let src = "print(dot([1, 2, 3], [4, 5, 6]))\n";
    let manifest = "edition = \"2026\"\nnumerics = \"relaxed\"\n";
    let o = sky_in("manifest", &["run"], Some(manifest), src);
    assert_eq!(out(&o), "32\n");
    assert_eq!(
        err(&o),
        "note: results computed in relaxed numerics (sky.toml): not bit-reproducible across machines\n"
    );
    // The flag overrides the manifest, both ways.
    let o = sky_in(
        "override",
        &["run", "--numerics=reproducible"],
        Some(manifest),
        src,
    );
    assert_eq!((out(&o).as_str(), err(&o).as_str()), ("32\n", ""));
    let o = sky_in(
        "override2",
        &["run", "--numerics=relaxed"],
        Some("numerics = \"reproducible\"\n"),
        src,
    );
    assert!(err(&o).contains("(--numerics=relaxed)"), "{}", err(&o));
    // A bad value is a manifest error, not a silent default.
    let o = sky_in("badvalue", &["run"], Some("numerics = \"fast\"\n"), src);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("numerics `fast` is not `reproducible` or `relaxed`"),
        "{}",
        err(&o)
    );
    assert!(out(&o).is_empty());
}

// ----- what changes: results, on every engine -----

#[test]
fn every_engine_agrees_with_the_reproducible_result_to_a_tolerance() {
    let n = 36;
    let src = kernels_program(n);
    let reference = sky("ref", &["run", "--engine=ast"], &src);
    assert_eq!(reference.status.code(), Some(0), "{}", err(&reference));
    assert!(err(&reference).is_empty());
    let want = numbers(&out(&reference));
    assert!(want.len() > n * n * 2);
    // Every engine, on a few threads; the tree-walker also on one thread and
    // on many (the kernels split their output, or a long dot product, by the
    // count, and the tolerance must hold for all of them).
    let runs = [
        ("ast", "1"),
        ("ast", "8"),
        ("sir", "3"),
        ("native", "3"),
        ("auto", "3"),
    ];
    for (engine, threads) in runs {
        let flag = format!("--engine={engine}");
        let t = format!("--threads={threads}");
        let o = sky("relaxed", &["run", &flag, &t, "--numerics=relaxed"], &src);
        assert_eq!(o.status.code(), Some(0), "{engine}: {}", err(&o));
        assert_eq!(
            err(&o),
            "note: results computed in relaxed numerics (--numerics=relaxed): not bit-reproducible across machines\n"
        );
        close(
            &numbers(&out(&o)),
            &want,
            1e-11,
            &format!("{engine} on {threads} threads"),
        );
    }
    // Reproducible stays bit-identical across engines and thread counts.
    for (engine, threads) in [("sir", "1"), ("native", "5")] {
        let (flag, t) = (format!("--engine={engine}"), format!("--threads={threads}"));
        let o = sky("repro", &["run", &flag, &t], &src);
        assert_eq!(out(&o), out(&reference), "{engine}");
    }
}

// ----- what does not change: errors -----

/// A program that prints, then fails inside the array kernels.
fn failing(kind: &str, n: usize) -> String {
    let mut rng = Rng(7);
    let mut sym = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let x = if i == j { n as f64 } else { 0.1 * rng.unit() };
            sym[i * n + j] = x;
            sym[j * n + i] = x;
        }
    }
    let (prelude, decl, body) = match kind {
        "singular" => {
            // The last row is the sum of two others (small integers, so it
            // is exactly singular in the elimination's arithmetic too).
            let ints: Vec<f64> = (0..n * n)
                .map(|i| ((i * 7 + i / n * 3) % 9) as f64 - 4.0)
                .collect();
            let mut m = ints;
            for j in 0..n {
                m[(n - 1) * n + j] = m[j] + m[n + j];
            }
            (
                String::new(),
                format!("M: Matrix[{n}, {n}] = {}", matrix_literal(&m, n, n)),
                "M⁻¹ @ b",
            )
        }
        "not-positive-definite" => {
            // A negative pivot at a known row.
            let row = n / 2;
            sym[row * n + row] = -1.0;
            (
                pd_solvers(n),
                format!("M: Matrix[{n}, {n}] = {}", matrix_literal(&sym, n, n)),
                "solve_pd(M, b)",
            )
        }
        _ => {
            // Symmetric to one ulp in one entry, but declared symmetric.
            sym[1] = f64::from_bits(sym[1].to_bits() + 1);
            (
                pd_solvers(n),
                format!("M: Matrix[{n}, {n}] = {}", matrix_literal(&sym, n, n)),
                "solve_pd(M, b)",
            )
        }
    };
    let b = vector_literal(&vec![1.0; n]);
    format!("{prelude}print(1)\n{decl}\nb: Vector[{n}] = {b}\nprint(2)\nprint({body})\nprint(3)\n")
}

#[test]
fn errors_are_the_same_kind_at_the_same_place_in_both_modes() {
    for kind in ["singular", "not-positive-definite", "not-symmetric"] {
        for n in [5, 40] {
            let src = failing(kind, n);
            let want = sky("errref", &["run", "--engine=ast"], &src);
            assert_eq!(want.status.code(), Some(1), "{kind} {n}: {}", err(&want));
            assert_eq!(out(&want), "1\n2\n", "{kind}");
            // (SIR and native code do not run a function whose parameter is
            // declared with a tag yet; `auto` falls back to the tree-walker.)
            let engines: &[&str] = if kind == "singular" {
                &["ast", "sir", "native", "auto"]
            } else {
                &["ast", "auto"]
            };
            for engine in engines {
                let flag = format!("--engine={engine}");
                let o = sky("errrelaxed", &["run", &flag, "--numerics=relaxed"], &src);
                assert_eq!(o.status.code(), Some(1), "{kind} {n} {engine}");
                assert_eq!(
                    out(&o),
                    out(&want),
                    "{kind} {n} {engine}: output before the error"
                );
                assert_eq!(
                    without_note(&err(&o)),
                    err(&want),
                    "{kind} {n} {engine}: the same diagnostic"
                );
                assert!(
                    err(&o).ends_with(NOTE_END),
                    "{kind} {n} {engine}: printed, so noted"
                );
            }
        }
    }
    let src = failing("singular", 40);
    assert!(err(&sky("s", &["run"], &src)).contains("the matrix is singular"));
    let src = failing("not-positive-definite", 40);
    assert!(err(&sky("s", &["run"], &src)).contains("the factorization failed at row 20"));
    let src = failing("not-symmetric", 40);
    assert!(err(&sky("s", &["run"], &src)).contains("the matrix is not symmetric"));
}

#[test]
fn shapes_units_and_finiteness_are_still_checked() {
    // A shape error is a check error before anything runs.
    let o = sky(
        "shape",
        &["run", "--numerics=relaxed"],
        "A = [[1.0, 2.0]]\nB = [[1.0, 2.0]]\nprint(A @ B)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(out(&o).is_empty() && !err(&o).contains("note: results computed"));
    let reference = sky(
        "shape-ref",
        &["run"],
        "A = [[1.0, 2.0]]\nB = [[1.0, 2.0]]\nprint(A @ B)\n",
    );
    assert_eq!(err(&o), err(&reference));
    // A unit mismatch too.
    let o = sky(
        "units",
        &["run", "--numerics=relaxed"],
        "print(dot([1 m, 2 m], [1 s, 2 s]) + 1 kg)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    // An overflow in a relaxed product is still an error at the product.
    let src = "A = [[1e200, 1e200], [1e200, 1e200]]\nprint(1)\nprint(A @ A)\nprint(2)\n";
    let want = sky("ovf-ref", &["run"], src);
    assert_eq!(want.status.code(), Some(1));
    for engine in ["ast", "sir", "native"] {
        let flag = format!("--engine={engine}");
        let o = sky("ovf", &["run", &flag, "--numerics=relaxed"], src);
        assert_eq!(o.status.code(), Some(1), "{engine}");
        assert_eq!(out(&o), "1\n", "{engine}");
        assert_eq!(without_note(&err(&o)), err(&want), "{engine}");
    }
}

// ----- reporting -----

#[test]
fn explain_says_where_relaxed_numerics_apply_and_only_when_relaxed() {
    let src = "A = [[1.0, 2.0], [3.0, 4.0]]\nprint(A @ A)\nprint(dot([1, 2], [3, 4]))\n";
    let plain = sky("explain", &["check", "--explain"], src);
    assert!(!out(&plain).contains("numerics"), "{}", out(&plain));
    assert!(!err(&plain).contains("NUM-"), "{}", err(&plain));
    let named = sky(
        "explain-named",
        &["check", "--explain", "--numerics=reproducible"],
        src,
    );
    assert_eq!((out(&named), err(&named)), (out(&plain), err(&plain)));

    let o = sky(
        "explain-relaxed",
        &["check", "--explain", "--numerics=relaxed"],
        src,
    );
    let stdout = out(&o);
    assert!(
        stdout.contains("numerics: relaxed (--numerics=relaxed): fused multiply-add, reassociated sums and the platform's BLAS/LAPACK; results are not bit-reproducible across machines"),
        "{stdout}"
    );
    let stderr = err(&o);
    assert!(
        stderr.contains("[NUM-001] `A @ A` is a matrix product on relaxed numerics"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[NUM-001] `dot([1, 2], [3, 4])` is a dot product on relaxed numerics"),
        "{stderr}"
    );
    assert!(
        stderr.contains("to rounding error (about 1e-12 relative"),
        "{stderr}"
    );
    assert_eq!(o.status.code(), Some(0));

    // The project's setting is reported with the manifest it came from.
    let o = sky_in(
        "explain-manifest",
        &["check", "--explain"],
        Some("numerics = \"relaxed\"\n"),
        src,
    );
    assert!(
        out(&o).contains("numerics: relaxed (declared in "),
        "{}",
        out(&o)
    );
    assert!(out(&o).contains("sky.toml)"), "{}", out(&o));
}

#[test]
fn explain_names_the_scalar_multiply_adds_native_code_fuses() {
    let src = "t = 0\nfor i in range(3):\n    t = t + (i * 0.5 + 1)\nprint(t)\n";
    let o = sky(
        "explain-fma",
        &["check", "--explain", "--numerics=relaxed"],
        src,
    );
    let stderr = err(&o);
    assert!(
        stderr.contains("is one fused multiply-add on the native engine"),
        "{stderr}"
    );
    let plain = sky("explain-fma-plain", &["check", "--explain"], src);
    assert!(!err(&plain).contains("fused multiply-add"));
}

#[test]
fn emit_clif_shows_the_fused_operations_only_when_relaxed() {
    let src = "t = 0\nfor i in range(3):\n    t = t + (i * 0.5 + 1)\nprint(t)\n";
    let plain = sky("clif", &["check", "--emit=clif"], src);
    assert!(!out(&plain).contains("fma"), "{}", out(&plain));
    let relaxed = sky(
        "clif-relaxed",
        &["check", "--emit=clif", "--numerics=relaxed"],
        src,
    );
    // aarch64 always has the instruction; on x86-64 only with FMA3, which
    // Cranelift is told about only if the host has it.
    if cfg!(target_arch = "aarch64") {
        assert!(out(&relaxed).contains("fma"), "{}", out(&relaxed));
    }
}
