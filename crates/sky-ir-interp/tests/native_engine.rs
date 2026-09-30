//! The native engine (`sky-jit`) against the SIR interpreter, on programs written to
//! reach what native code does differently: the boundary between typed and
//! unknown values, the recursion limit, printing order, arithmetic whose
//! result depends on the exact operations. (The broad comparison, with the
//! tree-walker too, is in sky-ir-interp's `differential` and
//! `fuzz_differential` tests.) These tests live here, not in sky-jit, because
//! the interpreter depends on sky-jit (compiled kernels), so sky-jit cannot
//! have the interpreter as a dependency, not even a dev-dependency.

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::Module;
use sky_jit::{compile, emit_clif, RunFailure, SkipWriter};

/// What a run did: stdout, and the failure as the whole rendered diagnostic.
#[derive(Debug, PartialEq)]
struct Run {
    stdout: String,
    failure: Option<String>,
}

fn render(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("test.sky", src))
}

fn lower(src: &str) -> Module {
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("does not lower:\n{src}\n{e}"));
    let mut opt = ir;
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut opt)
        .expect("the pipeline runs");
    opt
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

/// Native code, with the SIR interpreter finishing the run if it
/// deoptimizes; also says whether it did.
fn native_run(m: &Module, src: &str) -> (Run, bool) {
    sky_ast::with_deep_stack(|| {
        let code = compile(m).unwrap_or_else(|e| panic!("{e}"));
        let mut out = Vec::new();
        let (r, deopt) = code.run_or_rerun(&mut out, |w| sky_ir_interp::run(m, &mut { w }));
        let run = Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e.to_diagnostic(), src)),
        };
        (run, deopt)
    })
}

/// The program gives the same output and failure natively and on SIR, and
/// native code never deoptimizes on it.
fn agree(src: &str) -> Run {
    let m = lower(src);
    let want = sir_run(&m, src);
    let (got, deopt) = native_run(&m, src);
    assert!(!deopt, "native code deoptimized on:\n{src}");
    assert_eq!(got, want, "native code differs on:\n{src}");
    want
}

#[test]
fn scalar_programs_agree() {
    let r = agree(
        "m = 2 kg\nk = 50 N/m\nx = 1 m\nv = 0 m/s\ni = 0\n\
         while i < 1000:\n    a = -k * x / m\n    v += a * 0.001 s\n    x += v * 0.001 s\n    i += 1\n\
         print(x, v, 0.5 * m * v^2 + 0.5 * k * x^2)\n",
    );
    assert!(r.failure.is_none());
}

/// Every small integer power is the multiplications `powi` makes, whatever
/// the base: bit for bit, including the overflow errors.
#[test]
fn integer_powers_are_bit_exact() {
    let bases = [
        "0.1",
        "1 / 3",
        "2",
        "-2",
        "10",
        "1e10",
        "-7.5",
        "1e-5",
        "3.7e80",
        "0",
        "1.0000001",
        "123456.789",
        "-0.3",
    ];
    for b in bases {
        for e in [
            "0", "1", "2", "3", "4", "5", "7", "8", "13", "31", "64", "-1", "-2", "-3", "-7", "-64",
        ] {
            agree(&format!("x = {b}\nprint(x^{e})\n"));
        }
        // Fractional exponents and exponents outside the `powi` range call `powf`.
        for e in ["0.5", "1/3", "2.5", "-0.5", "65", "-65", "100"] {
            agree(&format!("x = {b}\nprint(x^({e}))\n"));
        }
    }
}

#[test]
fn math_functions_are_bit_exact() {
    let inputs = [
        "0.1", "0.5", "1", "2", "10", "100", "0.001", "1e-9", "1.5", "0.99", "3.14159", "12.7",
        "-0.5", "-2", "0", "1e100", "700",
    ];
    for x in inputs {
        for f in [
            "sqrt", "cbrt", "abs", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin",
            "acos", "atan", "floor", "ceil", "round", "expm1", "log1p", "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh", "erf", "sign",
        ] {
            agree(&format!("x = {x}\nprint({f}(x))\n"));
        }
        for f in ["atan2", "hypot", "min", "max"] {
            for y in ["0.1", "-3", "0", "7.5"] {
                agree(&format!("x = {x}\nprint({f}(x, {y}))\n"));
            }
        }
    }
}

#[test]
fn division_overflow_and_domain_errors_are_the_interpreters() {
    for src in [
        "x = 4\nz = 2 - 2\nprint(1)\nprint(x / z)\nprint(2)\n",
        "x = 1e200\nprint(x * x)\n",
        "x = 1e308\nprint(x + x)\n",
        "x = 1e-200\nprint(1 / (x * x * x))\n",
        "z = 0 m\nprint(z^-2)\n",
        "z = 0 m\nprint(inv(z))\n",
        "print(sqrt(0 - 1))\n",
        "x = 0 - 8\nprint(x^(1/3))\n",
        "print(1 m + 2 s)\n",
        "def f(a: m, b: s) -> m:\n    return a + b\nprint(1)\nprint(f(1 m, 2 s))\n",
    ] {
        // Some of these the checker rejects; only those that lower are run.
        if sky_lower::lower_source(src).is_err() {
            continue;
        }
        agree(src);
    }
}

#[test]
fn printing_keeps_program_order_across_calls() {
    let r = agree(
        "def shout(n):\n    print(n)\n    return n + 1\n\
         print(shout(1) + shout(10))\nprint(\"done\", 1 < 2, \"a\" == \"a\")\n",
    );
    assert_eq!(r.stdout, "1\n10\n13\ndone true true\n");
}

#[test]
fn arrays_and_strings_go_through_the_runtime() {
    agree(
        "v = [1 m, 2 m, 3 m]\nA = [[1, 2], [3, 4]]\n\
         print(v + v * 2, v[1], A @ A, Aᵀ, A⁻¹, len(v), zeros(2), identity(2))\n\
         print(v[1:3], A[0], A[1, 0])\n\
         i = 5\nprint(v[i])\n",
    );
}

#[test]
fn contracts_report_the_call_site_of_a_requires() {
    let r = agree(
        "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\n\
         print(fall_time(20 m))\nprint(fall_time(-1 m))\n",
    );
    let e = r.failure.expect("fails");
    assert!(e.contains("called here"), "{e}");
}

#[test]
fn recursion_stops_at_the_interpreters_depth() {
    // 1000 nested calls are allowed, the 1001st is not (the entry call does
    // not count), and the error is `sky run`'s, at the call.
    // Typed (all scalars native) and untyped (all values boxed) alike.
    for def in [
        "def d(n: m) -> m:\n    if n == 0 m:\n        return 0 m\n    return d(n - 1 m) + 1 m\n",
        "def d(n):\n    if n == 0 m:\n        return 0 m\n    return d(n - 1 m) + 1 m\n",
    ] {
        deep_recursion(def);
    }
}

fn deep_recursion(def: &str) {
    let deep = |n: u32| {
        let src = format!("{def}print(d({n} m))\n");
        let m = lower(&src);
        native_run(&m, &src).0
    };
    assert_eq!(deep(999).stdout, "999 m\n");
    assert_eq!(deep(999).failure, None);
    let too_deep = deep(1000);
    let e = too_deep.failure.expect("fails");
    assert!(
        e.contains("recursion is too deep: more than 1000 nested calls"),
        "{e}"
    );
    assert!(e.contains("calling `d` here"), "{e}");
    assert_eq!(too_deep.stdout, "");
}

#[test]
fn unbounded_recursion_is_an_error_not_a_crash() {
    let src = "def f(x):\n    return f(x) + 1\nprint(f(1))\n";
    let m = lower(src);
    let (r, deopt) = native_run(&m, src);
    assert!(!deopt);
    let e = r.failure.expect("fails");
    assert!(e.contains("recursion is too deep"), "{e}");
}

// ----- values whose type is unknown -----

fn parse(text: &str) -> Module {
    sky_ir::parse(text).unwrap_or_else(|e| panic!("{e}"))
}

/// A `main` that prints, calls `f` with a number of dimension `arg`, and
/// prints the result; `f` takes an unknown value and declares a length.
fn unknown_into_typed(arg: &str) -> Module {
    parse(&format!(
        "sir edition 2026\n\
         fn @<main>() -> () effects(io, fail, call) {{\n\
           %0 = const.str \"before\" : str\n\
           print %0\n\
           %1 = const.float 3 : num[{arg}; ()]\n\
           %2 = call @f(%1) : num[L; ()]\n\
           print %2\n\
           %3 = add %2, %2 : num[L; ()]\n\
           print %3\n\
           return\n\
         }}\n\
         fn @f(%x: unknown) -> (num[L; ()]) {{\n\
           return %x\n\
         }}\n"
    ))
}

#[test]
fn an_unknown_value_that_is_what_its_declaration_says_stays_native() {
    let m = unknown_into_typed("L");
    sky_ir::verify(&m).expect("verifies");
    let out = sky_ast::with_deep_stack(|| {
        let code = compile(&m).expect("compiles");
        let mut out = Vec::new();
        code.run(&mut out).expect("runs natively");
        out
    });
    assert_eq!(String::from_utf8(out).unwrap(), "before\n3 m\n6 m\n");
}

#[test]
fn an_unknown_value_that_is_not_deoptimizes_and_sir_finishes_the_run() {
    // The value is a time where the declaration says a length. SIR goes on
    // with the time (`3 s`, and `6 s` after the addition); native code must
    // not compute with a bare number instead.
    let m = unknown_into_typed("T");
    sky_ir::verify(&m).expect("verifies");
    let want = sir_run(&m, "");
    assert_eq!(want.stdout, "before\n3 s\n6 s\n");
    sky_ast::with_deep_stack(|| {
        let code = compile(&m).expect("compiles");
        let mut out = Vec::new();
        assert_eq!(
            code.run(&mut out),
            Err(RunFailure::Deopt { written: 7 }),
            "stopped after `before`"
        );
        assert_eq!(String::from_utf8(out).unwrap(), "before\n");
    });
    let (got, deopt) = native_run(&m, "");
    assert!(deopt);
    assert_eq!(got, want);
}

#[test]
fn skip_writer_drops_exactly_the_bytes_already_written() {
    use std::io::Write;
    let mut sink = Vec::new();
    {
        let mut w = SkipWriter::new(&mut sink, 5);
        w.write_all(b"abc").unwrap();
        w.write_all(b"defgh").unwrap();
        w.write_all(b"ij").unwrap();
    }
    assert_eq!(sink, b"fghij");
}

// ----- the module as a whole -----

#[test]
fn a_module_that_does_not_verify_is_refused() {
    let mut m = lower("print(1)\n");
    m.functions[0].results.push(sky_ir::Type::Bool);
    let e = compile(&m).err().expect("refused");
    assert!(
        e.to_string().starts_with("cannot compile natively: "),
        "{e}"
    );
}

#[test]
fn clif_text_has_every_function_and_the_status_convention() {
    let m =
        lower("def f(x: m) -> m:\n    return x * 2\nfor i in range(3):\n    print(f(i * 1 m))\n");
    let text = emit_clif(&m).expect("emits");
    assert!(text.contains("sky_f0__main_"), "{text}");
    assert!(text.contains("sky_f1_f"), "{text}");
    // Every function takes (depth, stack limit, call site) and returns a
    // status first.
    assert!(text.contains("(i32, i64, i32) -> i32"), "{text}");
    assert!(text.contains("fmul"), "{text}");
}

// ----- checks that native code makes cheaply -----
//
// Native code checks finiteness as `x - x` ordered, skips checks the
// exactness analysis proves unnecessary, runs the arms of a comparison-only
// `if` on both paths, and calls the math core directly. None of it may move,
// merge or drop an error: the first failing operation in program order is
// reported, after the output before it and before any output after it.

#[test]
fn an_overflow_after_prints_reports_the_first_failing_operation() {
    for src in [
        // The first overflow is `y`; the second (`z`) never runs.
        "print(1)\nx = 1e200\nprint(x)\ny = x * x\nprint(y)\nz = y * y\nprint(z)\n",
        // Two failing operations in a row: the earlier one is reported.
        "x = 1e300\nprint(7)\na = x * 1e10\nb = a * a\nprint(a, b)\n",
        // An overflow that a later operation would hide (1 / inf is 0).
        "x = 1e300\nprint(1)\ny = 1 / (x * x)\nprint(y)\n",
        // Sums and differences.
        "x = 1.7e308\nprint(0)\ny = x + x\nprint(y)\n",
        "x = 1.7e308\nprint(0)\ny = 0 - x - x\nprint(y)\n",
        // A power and a negative power of zero.
        "x = 1e200\nprint(2)\nprint(x^2)\nprint(3)\n",
        "x = 0\nprint(2)\nprint(x^-2)\nprint(3)\n",
    ] {
        let r = agree(src);
        assert!(r.failure.is_some(), "expected an error from:\n{src}");
    }
}

#[test]
fn an_overflow_inside_a_loop_reports_the_iteration_the_interpreter_stops_at() {
    for src in [
        // Doubling overflows after about 1000 steps.
        "x = 1.0\nfor i in range(2000):\n    x = x * 2\n    if i / 250 == floor(i / 250):\n        print(i, x)\nprint(x)\n",
        // The failing operation is inside a while, after a print.
        "x = 1.0\nn = 0\nwhile n < 5000:\n    print(n)\n    x = x * 3\n    n = n + 1\n",
        // Two accumulators; the second overflows first.
        "a = 1.0\nb = 1.0\nfor i in range(400):\n    a = a + 1\n    b = b * b + 1\n    print(a)\nprint(b)\n",
        // A nested loop with a break flag around the overflow.
        "t = 1.0\nfor i in range(50):\n    for j in range(50):\n        if j > 40:\n            break\n        t = t * 1.5\n        print(t)\n",
        // A division by zero that appears mid-loop.
        "n = 5\nfor i in range(10):\n    print(100 / (n - i))\n",
    ] {
        let r = agree(src);
        assert!(r.failure.is_some(), "expected an error from:\n{src}");
        assert!(
            !r.stdout.is_empty(),
            "nothing was printed before the failure:\n{src}"
        );
    }
}

#[test]
fn comparison_only_ifs_and_break_flags_agree() {
    for src in [
        // `and`/`or` chains (short-circuit ifs) over numbers and Bools.
        "n = 0\nc = 0\nwhile n < 100 and c < 30:\n    if n / 3 == floor(n / 3) or n / 7 == floor(n / 7):\n        c = c + 1\n    n = n + 1\nprint(n, c)\n",
        "a = 3\nb = 0 - 2\nprint(a > 0 and b < 0, a > 0 and b > 0, a < 0 or b < 0, not (a < 0 or b > 0))\n",
        // `break` and `continue` flags in nested loops.
        "s = 0\nfor i in range(20):\n    if i / 2 == floor(i / 2):\n        continue\n    for j in range(20):\n        if i * j > 50:\n            break\n        s = s + i * j\nprint(s)\n",
        // A Mandelbrot escape loop, small.
        "total = 0\nfor py in range(20):\n    ci = 0 - 1.5 + py * 0.15\n    for px in range(20):\n        cr = 0 - 2.0 + px * 0.15\n        zr = 0.0\n        zi = 0.0\n        i = 0\n        while i < 60:\n            zr2 = zr * zr\n            zi2 = zi * zi\n            if zr2 + zi2 > 4.0:\n                break\n            zi = 2.0 * zr * zi + ci\n            zr = zr2 - zi2 + cr\n            i = i + 1\n        total = total + i\nprint(total)\n",
        // The failing operation sits after a taken break: it must not run.
        "for i in range(10):\n    if i > 3:\n        break\n    print(1 / (3 - i))\nprint(9)\n",
        // ... and one that is reached: 1 / 0 at i = 3.
        "for i in range(10):\n    print(1 / (3 - i))\n",
    ] {
        agree(src);
    }
}

#[test]
fn math_domain_and_overflow_edges_are_the_interpreters() {
    let inputs = [
        "0",
        "0 - 0.0",
        "1",
        "0 - 1",
        "0.5",
        "0 - 0.5",
        "2",
        "0 - 2",
        "1e-300",
        "710",
        "1000",
        "1e308",
        "0 - 1e308",
        "709.7",
        "0 - 745",
        "1e5",
    ];
    for x in inputs {
        for f in [
            "log", "log10", "log2", "log1p", "acosh", "atanh", "asin", "acos", "exp", "expm1",
            "sinh", "cosh", "tanh", "asinh", "cbrt", "erf", "sign", "round", "sqrt", "tan", "atan",
            "sin", "cos",
        ] {
            agree(&format!("x = {x}\nprint(1)\nprint({f}(x))\nprint(2)\n"));
        }
        for f in ["atan2", "hypot"] {
            for y in ["0", "0 - 1", "3", "1e308"] {
                agree(&format!(
                    "x = {x}\nprint(1)\nprint({f}(x, {y}))\nprint(2)\n"
                ));
            }
        }
    }
}

#[test]
fn operations_the_analysis_proves_cannot_fail_still_compute_the_same_bits() {
    for src in [
        "x = 1e308\nprint(x / 2, x / 3, 0.5 * x)\n",
        "x = 0 - 1e308\nprint(x / 2, x * 0.5, x / 7)\n",
        "print(sqrt(2), log(3), exp(1), 2 / 3, 10 / 4, 3^2)\n",
        "x = 1e-320\nprint(x / 2, x * 0.5)\n",
        "d = 0\nx = 5\nprint(1)\nprint(x / d)\n",
    ] {
        agree(src);
    }
}
