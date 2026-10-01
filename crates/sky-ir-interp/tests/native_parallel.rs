//! Automatic parallelism in native code (`sky_ir::parallel`, `sky-jit`):
//! big kernels in chunks and independent pure calls on the pool must print
//! what the sequential run prints, fail as it fails (the whole rendered
//! diagnostic), and give the same bytes on every thread count.
//!
//! These tests live here, not in sky-jit, because the interpreter (the
//! oracle) depends on sky-jit.

use sky_diag::SourceFile;
use sky_ir::parallel::Config;
use sky_ir::Module;

/// What a run did: stdout, and the failure as the whole rendered diagnostic.
#[derive(Debug, PartialEq)]
struct Run {
    stdout: String,
    failure: Option<String>,
}

fn lower(src: &str) -> Module {
    let mut m =
        sky_lower::lower_source(src).unwrap_or_else(|e| panic!("does not lower:\n{src}\n{e}"));
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut m)
        .expect("the pipeline runs");
    m
}

fn render(e: &sky_ir_interp::IrRuntimeError, src: &str) -> String {
    e.to_diagnostic()
        .render_to_string(&SourceFile::new("test.sky", src))
}

/// The SIR interpreter, sequential: the oracle.
fn sir(m: &Module, src: &str) -> Run {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run_with(m, &mut out, &Config::sequential());
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e, src)),
        }
    })
}

/// Native code with `cfg` (the SIR interpreter finishing the run, with the
/// same configuration, if native code deoptimizes). `help`: whether a thread
/// that joins the calls it started runs the ones no worker has started yet
/// (the default), or leaves them all to workers.
fn native(m: &Module, src: &str, cfg: &Config, help: bool) -> Run {
    sky_ast::with_deep_stack(|| {
        sky_jit::par::set_help_at_join_for_tests(help);
        let code =
            sky_jit::compile_par(m, sky_jit::Numerics::Reproducible, sky_jit::Slp::Auto, cfg)
                .unwrap_or_else(|e| panic!("{e}"));
        let mut out = Vec::new();
        let (r, _) = code.run_or_rerun(&mut out, |w| sky_ir_interp::run_with(m, &mut { w }, cfg));
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e, src)),
        }
    })
}

/// Every configuration a native run must be the same under.
fn configs() -> Vec<Config> {
    let mut all = Vec::new();
    for t in [2, 4, 8] {
        all.push(Config::eager(t));
        all.push(Config::with_threads(t));
        all.push(Config {
            chunk_min: 3,
            ..Config::eager(t)
        });
    }
    all
}

/// How much the counters of `sky_jit::par` grew: kernels split, elementwise
/// operations split, calls on workers.
#[derive(Debug, Clone, Copy)]
struct Grew {
    kernels: u64,
    elementwise: u64,
    calls: u64,
    retried: u64,
}

/// Runs `f` with the process-wide counters of `sky_jit::par` to itself (the
/// tests of this file take turns), and says how much they grew.
fn measured<T>(f: impl FnOnce() -> T) -> (T, Grew) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let before = sky_jit::par::stats();
    let r = f();
    let after = sky_jit::par::stats();
    (
        r,
        Grew {
            kernels: after.kernels_split - before.kernels_split,
            elementwise: after.elementwise_split - before.elementwise_split,
            calls: after.calls_on_workers - before.calls_on_workers,
            retried: after.calls_retried - before.calls_retried,
        },
    )
}

/// The program does the same natively on every configuration as the
/// sequential SIR interpreter does, and returns that with what ran in
/// parallel.
fn agree_measured(src: &str) -> (Run, Grew) {
    measured(|| agree(src))
}

/// The program does the same natively on every configuration as the
/// sequential SIR interpreter does, and returns that.
fn agree(src: &str) -> Run {
    let m = lower(src);
    let want = sir(&m, src);
    let seq = native(&m, src, &Config::sequential(), true);
    assert_eq!(seq, want, "sequential native code differs on:\n{src}");
    for cfg in configs() {
        for help in [false, true] {
            let got = native(&m, src, &cfg, help);
            assert_eq!(
                got, want,
                "{cfg:?} (help at join: {help}) differs on:\n{src}"
            );
        }
    }
    want
}

#[test]
fn independent_calls_run_on_workers_and_agree() {
    let src = "def work(n):\n    total = 0\n    i = 0\n    while i < n:\n        total = total + sqrt(i + 1) / (i + 1)\n        i = i + 1\n    return total\n\na = work(3000)\nb = work(4000)\nprint(a + b)\n";
    let (r, grew) = agree_measured(src);
    assert!(r.failure.is_none(), "{r:?}");
    assert!(r.stdout.starts_with("233."), "{}", r.stdout);
    assert!(grew.calls > 0, "no call ran on a worker");
}

/// A vector literal of `n` entries: `base`, and `special` at the indexes in
/// `at`.
fn vector(n: usize, base: &str, special: &str, at: &[usize]) -> String {
    let entries: Vec<&str> = (0..n)
        .map(|i| if at.contains(&i) { special } else { base })
        .collect();
    format!("[{}]", entries.join(", "))
}

#[test]
fn a_big_kernel_is_split_and_gives_the_same_bits() {
    // Entries that are all finite: no failure, a fused chain of operations.
    let n = 5000;
    let entries: Vec<String> = (0..n)
        .map(|i| format!("{}", (i % 97) as f64 * 0.37 - 11.0))
        .collect();
    let src = format!(
        "x = [{}]\ny = x * 3 - x * 0.5\nz = (x + y) * 0.25 + (x - y) * 0.125\nprint(z[0], z[{}], z[2500])\n",
        entries.join(", "),
        n - 1
    );
    let (r, grew) = agree_measured(&src);
    assert!(r.failure.is_none(), "{r:?}");
    assert!(grew.kernels > 0, "no kernel was split");
}

#[test]
fn a_kernel_failing_in_several_chunks_reports_the_sequential_error() {
    // The body (after fusion) multiplies, adds, then divides by a scalar zero.
    // The product overflows at elements in three different chunks; the division
    // fails at every element, but the `mul` is first in the body, so the
    // overflow is the error whichever chunk or thread meets what first.
    let n = 4000;
    let src = format!(
        "x = {}\nk = 1e300\nz = 0\na = x * k\nb = a + x\nc = b / z\nprint(c[0])\n",
        vector(n, "2", "3e10", &[700, 1900, 3999])
    );
    let (r, grew) = agree_measured(&src);
    let e = r.failure.expect("the program fails");
    assert!(e.contains("overflow"), "{e}");
    assert!(grew.kernels > 0, "no kernel was split");
    // Without the overflow the division is the failure, in every chunk.
    let src = format!(
        "x = {}\nz = 0\na = x * 2\nb = a + x\nc = b / z\nprint(c[0])\n",
        vector(n, "2", "3", &[700, 1900, 3999])
    );
    let e = agree(&src).failure.expect("the program fails");
    assert!(e.contains("division by zero"), "{e}");
}

#[test]
fn a_big_elementwise_operation_overflows_in_one_chunk_as_in_all() {
    let n = 6000;
    let src = format!(
        "x = {}\ny = x + x\nprint(y[0])\nw = y * 1e10\nprint(w[1])\n",
        vector(n, "1.5", "1.7e308", &[4321])
    );
    let (r, grew) = agree_measured(&src);
    let e = r.failure.expect("the addition overflows");
    assert!(e.contains("overflow"), "{e}");
    assert!(grew.elementwise > 0, "no elementwise operation was split");
}

/// Helper functions: a cheap one that fails at 1, a loop, a Bool one.
const PRELUDE: &str = "\
def f(x):
    return 1 / (x - 1)

def g(x):
    return x * 2 + 1

def heavy(n):
    total = 0
    i = 0
    while i < n:
        total = total + sqrt(i + 1) / (i + 1)
        i = i + 1
    return total

def pos(x):
    return x > 0

def scale(v):
    return v * 2

def boom(x):
    assert x < 3
    return x + 1
";

/// `body` after the prelude, which must give `want` (a substring of the
/// error, or `None` for success) on every configuration, with calls on
/// workers unless the body says `noisy` (whose callee prints).
fn case(body: &str, want: Option<&str>) -> Run {
    let src = format!("{PRELUDE}\n{body}");
    let (r, grew) = agree_measured(&src);
    match (want, &r.failure) {
        (None, None) => {}
        (Some(w), Some(e)) => assert!(e.contains(w), "wanted `{w}`, got:\n{e}\nin:\n{body}"),
        (w, e) => panic!("wanted {w:?}, got {e:?} in:\n{body}"),
    }
    if !body.contains("noisy") && !body.starts_with("z = 0\nw = sqrt") {
        assert!(grew.calls > 0, "no call ran on a worker in:\n{body}");
    }
    r
}

#[test]
fn a_call_that_fails_beats_a_later_operation_that_fails() {
    // `f(1)` divides by zero; the in-place `1 / z` would, but comes later.
    case(
        "z = 0\na = f(1)\nw = 1 / z\nb = f(2)\nprint(a, w, b)\n",
        Some("division by zero"),
    );
    // With the calls fine, the in-place failure is the error.
    let r = case(
        "z = 0\na = f(2)\nw = 1 / z\nb = f(3)\nprint(a, w, b)\n",
        Some("division by zero"),
    );
    assert_eq!(r.stdout, "");
    // A failing in-place operation after a call that fails: the call's error
    // is the one with the call's span, so the two programs differ only there.
    let first = case(
        "z = 0\na = f(1)\nw = sqrt(0 - 1)\nb = f(2)\nprint(a, w, b)\n",
        Some("division by zero"),
    );
    assert!(first.failure.is_some());
    // And the reverse: the in-place failure is first, a later call fails too.
    case(
        "z = 0\nw = sqrt(0 - 1)\na = f(1)\nb = f(2)\nprint(a, w, b)\n",
        Some("sqrt"),
    );
}

#[test]
fn the_earliest_failing_call_is_reported() {
    case(
        "a = g(1)\nb = f(1)\nc = boom(5)\nd = f(1)\nprint(a, b, c, d)\n",
        Some("division by zero"),
    );
    case(
        "a = g(1)\nc = boom(5)\nb = f(1)\nd = f(1)\nprint(a, b, c, d)\n",
        Some("assert"),
    );
}

#[test]
fn an_operation_that_reads_a_started_result_waits_for_it() {
    let r = case("t = g(1) + g(2) * g(3)\nu = t - g(4)\nprint(t, u)\n", None);
    assert_eq!(r.stdout.lines().count(), 1);
    // A deferred operation that fails, before a later call that fails.
    case(
        "t = g(1e300) * g(1e300)\nu = f(1)\nprint(t, u)\n",
        Some("overflow"),
    );
}

#[test]
fn bools_and_small_arrays_cross_to_the_workers_whole() {
    let r = case("p = pos(1)\nq = pos(0 - 1)\nprint(p, q)\n", None);
    assert_eq!(r.stdout, "true false\n");
    // A vector result comes back through memory (three words).
    let r = case(
        "a = scale([1, 2, 3])\nb = scale([4, 5, 6])\nprint(a, b)\n",
        None,
    );
    assert_eq!(r.stdout, "[2, 4, 6] [8, 10, 12]\n");
    case(
        "a = scale([1e308, 2, 3])\nb = scale([4, 5, 6])\nprint(a, b)\n",
        Some("overflow"),
    );
}

#[test]
fn calls_in_a_loop_fail_at_the_same_iteration() {
    let r = case(
        "i = 0\ns = 0\nwhile i < 6:\n    s = s + f(i + 3) + heavy(50 + i)\n    print(s)\n    i = i + 1\n",
        None,
    );
    assert_eq!(r.stdout.lines().count(), 6);
    // `f(1)` fails in the third iteration, after two lines of output.
    let r = case(
        "i = 0\ns = 0\nwhile i < 6:\n    s = s + f(3 - i) + heavy(50 + i)\n    print(s)\n    i = i + 1\n",
        Some("division by zero"),
    );
    assert_eq!(r.stdout.lines().count(), 2);
}

#[test]
fn windows_nest_inside_started_calls() {
    // `outer` has a window of its own; two calls of it are a window in
    // main, so the inner window runs on a worker, where its calls run in
    // place.
    let body = "\
def outer(x):
    a = g(x)
    b = heavy(100 + x)
    c = g(x + 1)
    return a + b + c

p = outer(1)
q = outer(2)
print(p, q)
";
    case(body, None);
    // A failure in the inner window of the later call is the error.
    case(
        &body
            .replace("c = g(x + 1)", "c = f(x)")
            .replace("outer(2)", "outer(1)"),
        Some("division by zero"),
    );
}

#[test]
fn a_window_in_a_deep_recursion_keeps_the_sequential_error() {
    // `r` recurses past the call-depth limit; each level has a window of
    // calls. The error is the recursion limit, at the same call.
    let r = case(
        "\
def r(n):
    a = g(n)
    b = g(n + 1)
    return r(n + 1) + a + b

print(r(0))
",
        Some("recursion is too deep"),
    );
    assert_eq!(r.stdout, "");
    // A recursion that stays inside the limit, windows at every level.
    let r = case(
        "\
def r(n):
    a = g(n)
    b = g(n + 1)
    if n < 300:
        return r(n + 1) + a + b
    return a + b

print(r(0))
",
        None,
    );
    assert_eq!(r.stdout.lines().count(), 1);
}

#[test]
fn a_callee_that_prints_is_never_started() {
    let r = case(
        "def noisy(x):\n    print(x)\n    return x + 1\n\na = noisy(1)\nb = heavy(200)\nc = noisy(2)\nprint(a, b, c)\n",
        None,
    );
    assert!(r.stdout.starts_with("1\n2\n"), "{}", r.stdout);
}

#[test]
fn a_call_that_runs_out_of_a_workers_stack_is_run_again_on_the_caller() {
    // The budget of a worker is zero: every started call that makes calls runs
    // out of it on its worker, and is run again on the calling thread, where
    // the run's whole stack is. Nothing can show: same output, same errors.
    let src = format!(
        "{PRELUDE}\n\
         def loopy(n):\n    total = 0\n    i = 0\n    while i < n:\n        total = total + heavy(10) + g(i)\n        i = i + 1\n    return total\n\n\
         a = loopy(5)\nb = loopy(6)\nc = boom(7)\nprint(a, b, c)\n"
    );
    let ((r, plain), grew) = measured(|| {
        let plain = agree(&src);
        sky_jit::par::set_worker_stack_budget_for_tests(Some(0));
        let r = agree(&src);
        sky_jit::par::set_worker_stack_budget_for_tests(None);
        (r, plain)
    });
    assert_eq!(r, plain);
    assert!(
        r.failure.as_ref().is_some_and(|e| e.contains("assert")),
        "{r:?}"
    );
    assert!(grew.retried > 0, "no call was run again: {grew:?}");
}
