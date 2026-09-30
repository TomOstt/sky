//! Vectorized native code (`sky-jit`'s SLP pass) against the SIR interpreter:
//! pairs of entries and pairs of scalar operations computed as two-lane
//! vectors must give the same output bits and the same failure (the first
//! failing operation in program order, with its own operands) as the
//! interpreter, with overflow in either lane, NaN-producing inputs and `-0`.
//!
//! Every program runs with vectorizing off, on in its default mode and
//! forced on everywhere (`Slp::Always`), and the three must agree with the
//! interpreter and with each other. The runtime values (`big`, `zero`, ...)
//! are made in a loop, so the optimizer cannot fold a failure away at
//! compile time.

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::Module;
use sky_jit::{compile_with, emit_clif_with, Slp};

#[derive(Debug, PartialEq)]
struct Run {
    stdout: String,
    failure: Option<String>,
}

fn render(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("test.sky", src))
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

fn sir_run(m: &Module, src: &str) -> Run {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e.to_diagnostic(), src)),
        }
    })
}

fn native_run(m: &Module, src: &str, slp: Slp) -> Run {
    sky_ast::with_deep_stack(|| {
        let code = compile_with(m, slp).unwrap_or_else(|e| panic!("{e}"));
        let mut out = Vec::new();
        let (r, deopt) = code.run_or_rerun(&mut out, |w| sky_ir_interp::run(m, &mut { w }));
        assert!(!deopt, "native code deoptimized on:\n{src}");
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e.to_diagnostic(), src)),
        }
    })
}

/// Runs `src` on the SIR interpreter and natively in every mode; all agree.
fn agree(src: &str) -> Run {
    let m = lower(src);
    let want = sir_run(&m, src);
    for slp in [Slp::Off, Slp::Auto, Slp::Always] {
        let got = native_run(&m, src, slp);
        assert_eq!(got, want, "native code ({slp:?}) differs on:\n{src}");
    }
    want
}

/// Whether forced vectorizing made `f64x2` code for `src`.
fn packs(src: &str) -> bool {
    let m = lower(src);
    emit_clif_with(&m, Slp::Always)
        .expect("emits")
        .contains("f64x2")
}

/// Runtime numbers no pass can fold: `big` = 1e100 (up to rounding), `huge`
/// = 1e200, `zero` = 0, `nzero` = -0, `one` = 1.
const PRELUDE: &str = "big = 1.0\nhuge = 1.0\nzero = 1.0\ni = 0\n\
    while i < 100:\n    big = big * 10.0\n    i += 1\n\
    i = 0\nwhile i < 200:\n    huge = huge * 10.0\n    i += 1\n\
    zero = zero - 1.0\none = zero + 2.0 - 1.0\nnzero = zero * (0.0 - 1.0)\n";

fn prog(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

#[test]
fn an_overflow_in_either_lane_is_the_interpreters() {
    for n in [2usize, 3, 4, 5] {
        for lane in 0..n {
            let entries: Vec<String> = (0..n)
                .map(|i| if i == lane { "huge" } else { "one" }.to_string())
                .collect();
            let v = format!("[{}]", entries.join(", "));
            // One operation, a chain (the second fails), and a scalar on
            // the left.
            for body in [
                format!("v = {v}\nw = v * huge\nprint(w)\n"),
                format!("v = {v}\nw = v + v\nprint(1)\nprint(w * huge)\n"),
                format!("v = {v}\nw = (v * one + v) * huge - v\nprint(w)\n"),
                format!("v = {v}\nw = huge * v + v\nprint(w)\n"),
            ] {
                let src = prog(&body);
                let r = agree(&src);
                assert!(r.failure.is_some(), "expected an error from:\n{src}");
                assert!(packs(&src), "no packed code for:\n{src}");
            }
        }
    }
}

#[test]
fn the_first_failing_operation_reports_whichever_lane_it_is_in() {
    // `a` is 1e100. A lane holding 1e209 fails the first multiplication, one
    // holding 1e110 the second, one holding 1e10 the third (and a lane
    // holding 1 none). The interpreter reports the earliest operation in
    // program order that fails at any entry, then the first entry.
    let values = ["one", "ten", "v110", "v209"];
    for n in [2usize, 3, 4] {
        let total = values.len().pow(n as u32);
        for code in 0..total {
            let mut c = code;
            let entries: Vec<&str> = (0..n)
                .map(|_| {
                    let e = values[c % values.len()];
                    c /= values.len();
                    e
                })
                .collect();
            let src = prog(&format!(
                "a = big\nten = 1e10\nv110 = 1e110\nv209 = 1e209\nv = [{}]\nw = v * a * a * a\nprint(w)\n",
                entries.join(", ")
            ));
            // The interpreter's answer, in every mode.
            let r = agree(&src);
            assert_eq!(r.failure.is_some(), entries.iter().any(|e| *e != "one"));
            if code == 0 {
                assert!(packs(&src), "no packed code for:\n{src}");
            }
        }
    }
}

#[test]
fn a_zero_divisor_and_a_quotient_that_overflows() {
    for n in [2usize, 3, 4] {
        for lane in 0..n {
            let entries: Vec<String> = (0..n)
                .map(|i| if i == lane { "zero" } else { "one" }.to_string())
                .collect();
            let v = format!("[{}]", entries.join(", "));
            for body in [
                // 0 / 0 is NaN in one lane, 1 / 0 is infinity in the others.
                format!("v = {v}\nprint(v / zero)\n"),
                format!("v = {v}\nw = v * 2.0 + v\nprint(w / zero)\n"),
                // A tiny divisor: every quotient overflows (or one does).
                format!("v = {v} * huge\nprint(v / 1e-200)\n"),
                format!("v = {v}\nw = v / 1e-300 / 1e-300\nprint(w)\n"),
                // Fine quotients.
                format!("v = {v}\nprint(v / 3.0, v / one / 7.0)\n"),
            ] {
                agree(&prog(&body));
            }
        }
    }
}

#[test]
fn negative_zero_survives_the_lanes() {
    // `zero` is +0 and `nzero` is -0; array negation is `0 - x`, so a zero
    // stays +0 there, and a product or a sum keeps the sign the scalar
    // operation would give.
    for v in [
        "[zero, nzero, one, nzero]",
        "[nzero, nzero, nzero, nzero]",
        "[nzero, zero, nzero]",
        "[zero, nzero]",
        "[one, nzero, zero, one, nzero]",
    ] {
        for body in [
            format!("v = {v}\nprint(-v)\n"),
            format!("v = {v}\nprint(v * nzero)\nprint(v * zero)\n"),
            format!("v = {v}\nprint(v + v)\nprint(v - v)\nprint(v + nzero * v)\n"),
            format!("v = {v}\nprint(v * (0.0 - one) + v)\nprint(v / (0.0 - one))\n"),
            format!("v = {v}\nw = -v * one + v * nzero - v\nprint(w)\n"),
            format!("v = {v}\nprint(dot(v, v), dot(v, -v), norm(v))\n"),
        ] {
            let src = prog(&body);
            agree(&src);
        }
    }
}

#[test]
fn a_loop_fails_at_the_iteration_and_lane_the_interpreter_stops_at() {
    for n in [2usize, 3, 4] {
        for lane in 0..n {
            let entries: Vec<String> = (0..n)
                .map(|i| if i == lane { "1e100" } else { "1.5" }.to_string())
                .collect();
            let v = format!("[{}]", entries.join(", "));
            for body in [
                // Doubling: the big lane overflows at about iteration 700.
                format!("v = {v}\nfor i in range(2000):\n    v = v * 2.0\n    if i / 200 == floor(i / 200):\n        print(i, v[0])\nprint(v)\n"),
                // A chain in the loop, the failing operation after a print.
                format!("v = {v}\nu = v * 0.5\nk = 0\nwhile k < 3000:\n    print(k)\n    u = u + v * 2.0 - u * 0.25\n    v = v * 1.5 + u\n    k += 1\n"),
                // Independent iterations (packed), ending in an overflow.
                format!("s = 0.0\nfor i in range(40):\n    t = i * 1e8\n    p = {v} * t\n    q = p * t + p * 2.0 - p\n    s = s + q[0] + q[{}]\n    print(i, s)\n", n - 1),
            ] {
                agree(&prog(&body));
            }
        }
    }
}

#[test]
fn chains_of_independent_operations_are_bit_exact() {
    // Not failing: the values must be the same bits in every mode.
    for body in [
        "s = 0.0\nu = [0.5, 0.25, 0.125, 0.0625]\nfor i in range(300):\n    t = i * 0.001\n    p = [t, t + 1.0, t + 2.0, t + 3.0]\n    q = p * 0.5 - u * t + p * 0.25 - u * 0.5 + p * t - u * 0.125\n    s = s + q[0] + q[1] + q[2] + q[3]\nprint(s)\n",
        "s = 0.0\nu = [0.1, 0.2, 0.3]\nfor i in range(300):\n    t = i * 0.37\n    p = [t, 1.0 / (1.0 + t), t * t]\n    q = (p * 0.1 + u) / 3.0 - p / 7.0\n    s = s + dot(q, q) + norm(q)\nprint(s)\n",
        "s = 0.0\nfor i in range(200):\n    t = i * 0.01\n    A = [[t, 1.0, 0.5, 0.25], [1.0, t, 0.0, 0.1], [0.3, 0.2, t, 1.0], [0.0, 0.5, 1.0, t]]\n    x = [t, 1.0, 2.0, 3.0]\n    y = A @ x * 0.5 + x\n    s = s + y[0] - y[3] + dot(y, x)\nprint(s)\n",
        // A time step (latency bound: default mode leaves it alone, forced
        // mode packs it).
        "x = [1.0, 0.0, 0.5, 0.25]\nv = [0.0, 1.0, 0.25, 0.5]\nfor i in range(2000):\n    a = x * 0.5 + v * 0.01\n    v = v - a * 0.001\n    x = x + v * 0.001\nprint(x, v)\n",
    ] {
        let src = prog(body);
        let r = agree(&src);
        assert!(r.failure.is_none(), "{src}\n{r:?}");
        assert!(packs(&src), "no packed code for:\n{src}");
    }
}

#[test]
fn pairs_of_scalar_operations_report_the_first_failing_one() {
    for body in [
        // Neither fails.
        "x = huge * 1e100\ny = big * big\nprint(x, y)\n",
        "a = 3.0 * 7.0\nb = one * 9.0\nprint(a, b)\n",
        // Only the first fails.
        "x = huge * huge\ny = big * big\nprint(1)\nprint(x, y)\n",
        // Only the second.
        "x = big * big\ny = huge * huge\nprint(1)\nprint(x, y)\n",
        // Both: the first is reported.
        "x = huge * huge\ny = huge * huge * huge\nprint(2)\nprint(x, y)\n",
        "x = huge + huge * 1e100 - 1.0\ny = huge * huge\nprint(2)\nprint(x, y)\n",
        // Sums, differences and quotients; a zero divisor in the second.
        "x = huge * 1e100 + huge * 1e100\ny = big - big\nprint(x, y)\n",
        "x = big + big\ny = big - huge * 1e100\nprint(x, y)\n",
        "x = big / one\ny = big / zero\nprint(3)\nprint(x, y)\n",
        "x = big / zero\ny = huge / 1e-200\nprint(x, y)\n",
        "x = zero / zero\ny = huge / 1e-200\nprint(x, y)\n",
        // With units: the two results have different dimensions.
        "d = 3 * big\nx = d * d\nv = 4.0\ny = v * v\nprint(1)\nprint(x, y)\n",
        "a = huge * 2 m\nb = 3 m/s\nx = a * a\ny = b * b\nprint(x, y)\n",
        "a = 2 m\nb = 3 m\nx = a + b\ny = a - b\nprint(x, y, x * y)\n",
        // -0.
        "x = nzero * one\ny = zero * nzero\nprint(x, y)\n",
        "x = nzero + nzero\ny = zero - zero\nz = nzero - zero\nprint(x, y, z)\n",
        // A pair in a loop.
        "x = 1.0\ny = 1.0\nfor i in range(3000):\n    x = x * 1.5\n    y = y * 1.25\n    if i / 500 == floor(i / 500):\n        print(i, x, y)\nprint(x, y)\n",
        "a = 0.1\nb = 0.2\nfor i in range(1000):\n    c = a * a\n    d = b * b\n    a = c + d * 0.5\n    b = d - c * 0.25\nprint(a, b)\n",
    ] {
        agree(&prog(body));
    }
}

#[test]
fn scalar_pairs_are_really_packed() {
    let src = prog("x = big * big\ny = huge * huge\nprint(1)\nprint(x, y)\n");
    assert!(packs(&src), "no packed code for:\n{src}");
    let m = lower(&src);
    let off = emit_clif_with(&m, Slp::Off).expect("emits");
    assert!(!off.contains("f64x2"), "Off must not vectorize:\n{off}");
}
