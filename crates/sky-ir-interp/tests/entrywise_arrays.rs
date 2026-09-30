//! Entry-by-entry arrays (`a * b`, `a / b`, math functions of arrays) on
//! every engine: the tree-walking interpreter, the SIR interpreter (lowered,
//! optimized, with the parallel and native kernel paths forced), and native
//! code. The same stdout and the same rendered diagnostic everywhere, and
//! the printed entries equal, bit for bit, a scalar-per-entry reference
//! computed here with plain `f64` arithmetic (and `sky_interp::math`'s
//! scalar core for the functions).

use std::collections::HashSet;

use sky_derive::SirDerivatives;
use sky_diag::{Diagnostic, Severity, SourceFile, Span};
use sky_interp::{run_module_with_derivatives, RuntimeEnv};
use sky_ir_interp::{JitOptions, ParConfig};
use sky_units::MathFn;

#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    error: Option<String>,
    diagnostic: Option<String>,
}

fn rendered(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("test.sky", src))
}

fn parse_checked(src: &str) -> (sky_ast::Module, HashSet<Span>, sky_types::Checked) {
    let lexed = sky_syntax::lex(src).unwrap_or_else(|_| panic!("lex error in {src:.200?}"));
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "parse errors in {src:.200?}: {errors:?}");
    let checked = sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errs.is_empty(), "check errors in {src:.200?}: {errs:?}");
    let solves = checked.cholesky_solves.clone();
    (module, solves, checked)
}

fn outcome(out: Vec<u8>, result: Result<(), Diagnostic>, src: &str) -> Outcome {
    Outcome {
        stdout: String::from_utf8(out).expect("utf-8 output"),
        error: result.as_ref().err().map(|d| d.message.clone()),
        diagnostic: result.err().map(|d| rendered(&d, src)),
    }
}

fn run_tree_walker(src: &str) -> Outcome {
    let (module, solves, checked) = parse_checked(src);
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        env.add_chain_plans(&checked.chains);
        let mut derivatives = SirDerivatives::new(&module, src);
        let r = run_module_with_derivatives(
            &module,
            src,
            &mut env,
            &mut out,
            &solves,
            &mut derivatives,
        );
        outcome(out, r.map(|_| ()).map_err(|e| e.to_diagnostic()), src)
    })
}

fn run_sir(ir: &sky_ir::Module, src: &str, cfg: &ParConfig) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run_with(ir, &mut out, cfg);
        outcome(out, r.map_err(|e| e.to_diagnostic()), src)
    })
}

/// `main` with the native kernels forced on (`JitOptions::ALWAYS`).
fn run_sir_native_kernels(ir: &sky_ir::Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::call_with_jit(
            ir,
            sky_ir::Module::MAIN,
            Vec::new(),
            &mut out,
            JitOptions::ALWAYS,
        );
        outcome(out, r.map(|_| ()).map_err(|e| e.to_diagnostic()), src)
    })
}

fn run_native(ir: &sky_ir::Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let compiled = sky_jit::compile(ir).unwrap_or_else(|e| {
            panic!("native code refused a program:\n{src:.300}\n{e}");
        });
        let mut out = Vec::new();
        let (r, deopt) = compiled.run_or_rerun(&mut out, |w| sky_ir_interp::run(ir, &mut { w }));
        assert!(!deopt, "native code deoptimized on:\n{src:.300}");
        outcome(out, r.map_err(|e| e.to_diagnostic()), src)
    })
}

fn optimized(ir: &sky_ir::Module) -> sky_ir::Module {
    let mut ir = ir.clone();
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut ir)
        .expect("the pipeline runs");
    ir
}

/// The number of `fused` operations in the module (how much fused).
fn fused_count(m: &sky_ir::Module) -> usize {
    m.functions
        .iter()
        .map(|f| {
            f.ops
                .iter()
                .filter(|o| o.kind == sky_ir::OpKind::Fused)
                .count()
        })
        .sum()
}

/// Runs `src` on every engine and configuration and demands one outcome.
fn assert_same(src: &str) -> Outcome {
    let expected = run_tree_walker(src);
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering: {e}\n{src:.300}"));
    let opt = optimized(&ir);
    let seq = ParConfig::sequential();
    let eager = ParConfig::eager(4);
    let runs = [
        ("SIR, lowered", run_sir(&ir, src, &seq)),
        ("SIR, optimized", run_sir(&opt, src, &seq)),
        ("SIR, lowered, parallel", run_sir(&ir, src, &eager)),
        ("SIR, optimized, parallel", run_sir(&opt, src, &eager)),
        (
            "SIR, optimized, native kernels",
            run_sir_native_kernels(&opt, src),
        ),
        ("native, lowered", run_native(&ir, src)),
        ("native, optimized", run_native(&opt, src)),
    ];
    for (which, got) in runs {
        assert_eq!(
            got, expected,
            "{which} differs from the tree-walker on:\n{src:.400}"
        );
    }
    expected
}

/// Like [`assert_same`], and the optimized SIR must have fused something.
fn assert_same_and_fused(src: &str) -> Outcome {
    let ir = sky_lower::lower_source(src).expect("lowers");
    assert!(
        fused_count(&optimized(&ir)) >= 1,
        "nothing fused in:\n{src:.400}"
    );
    assert_same(src)
}

// ----- references -----

/// The entries of a printed array of dimensionless numbers or numbers with a
/// unit (`[5 m, 10 m]`, `[[1, 2], [3, 4]]`), as `f64`s, in order.
fn entries(line: &str) -> Vec<f64> {
    line.replace(['[', ']'], "")
        .split(", ")
        .map(|t| {
            let n = t.split(' ').next().unwrap();
            n.parse::<f64>()
                .unwrap_or_else(|_| panic!("not a number: {t:?} in {line:.100}"))
        })
        .collect()
}

/// Bit for bit, except that a printed `-0` reads `0` (so zeros compare as
/// numbers).
fn same_entries(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = if *w == 0.0 {
            *g == 0.0
        } else {
            g.to_bits() == w.to_bits()
        };
        assert!(ok, "{what}: entry {i}: {g:e} vs {w:e}");
    }
}

fn literal(xs: &[f64], unit: &str) -> String {
    let items: Vec<String> = xs.iter().map(|x| format!("{x:e}{unit}")).collect();
    format!("[{}]", items.join(", "))
}

/// A deterministic, awkward-but-finite dataset.
fn dataset(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
            // Positive, varied in scale, never zero.
            0.01 + u * 100.0
        })
        .collect()
}

// ----- the physics chain -----

#[test]
fn the_physics_chain_over_vectors_is_bit_exact_on_every_engine() {
    for n in [3usize, 9, 600] {
        let (x, y, d) = (dataset(n, 1), dataset(n, 2), dataset(n, 3));
        let src = format!(
            "x = {}\ny = {}\nd = {}\nr = sqrt(x*x + y*y) * 0.5 + d\nprint(r)\n",
            literal(&x, ""),
            literal(&y, ""),
            literal(&d, ""),
        );
        let o = assert_same_and_fused(&src);
        assert_eq!(o.error, None);
        let want: Vec<f64> = (0..n)
            .map(|i| (x[i] * x[i] + y[i] * y[i]).sqrt() * 0.5 + d[i])
            .collect();
        same_entries(&entries(o.stdout.trim_end()), &want, &format!("n={n}"));
    }
}

#[test]
fn the_chain_with_units_keeps_its_dimensions() {
    let src = "x = [3 m, 6 m, 9 m]\ny = [4 m, 8 m, 12 m]\nt = [1 s, 2 s, 4 s]\n\
               r = sqrt(x*x + y*y)\nprint(r)\nprint(r / t)\nprint(x * y)\nprint(x / t)\n";
    let o = assert_same_and_fused(src);
    assert_eq!(
        o.stdout,
        "[5 m, 10 m, 15 m]\n[5 m/s, 5 m/s, 3.75 m/s]\n[12 m^2, 48 m^2, 108 m^2]\n[3 m/s, 3 m/s, 2.25 m/s]\n"
    );
}

#[test]
fn a_product_of_matrices_is_entry_by_entry_and_at_is_the_matrix_product() {
    let src = "A = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\n\
               print(A * B)\nprint(A @ B)\nprint(A / B)\nprint(sqrt(A * B))\n";
    let o = assert_same(src);
    assert_eq!(
        o.stdout,
        "[[5, 12], [21, 32]]\n[[19, 22], [43, 50]]\n[[0.2, 0.3333333333333333], [0.42857142857142855, 0.5]]\n\
         [[2.23606797749979, 3.4641016151377544], [4.58257569495584, 5.656854249492381]]\n"
    );
}

#[test]
fn a_number_and_an_array_mix_in_the_two_argument_functions() {
    let src = "x = [1 m, 5 m, 3 m]\nprint(min(x, 2 m), max(2 m, x), hypot(x, 4 m))\n\
               print(atan2(x, 2 m))\n";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    assert!(o.stdout.starts_with("[1 m, 2 m, 2 m] [2 m, 5 m, 3 m] "));
}

// ----- every math function -----

/// The scalar reference of `f` on the SI values `x` and `y`.
fn scalar(f: MathFn, x: f64, y: f64) -> Result<f64, String> {
    sky_interp::math::apply_f64(f, x, y, Span::new(0, 1), Span::new(0, 1))
        .map_err(|e| e.message.clone())
}

fn program_for(f: MathFn, xs: &[f64], ys: &[f64]) -> String {
    let name = f.name();
    // Functions that take any dimension, and the two-argument functions
    // (which need one dimension), are given lengths; the rest, pure numbers.
    let any = matches!(f, MathFn::Sqrt | MathFn::Cbrt | MathFn::Abs | MathFn::Sign);
    if f.arity() == 2 {
        format!(
            "x = {}\ny = {}\nprint({name}(x, y))\nprint({name}(x, 2 m))\n",
            literal(xs, " m"),
            literal(ys, " m")
        )
    } else if any {
        format!("x = {}\nprint({name}(x))\n", literal(xs, " m"))
    } else {
        format!("x = {}\nprint({name}(x))\n", literal(xs, ""))
    }
}

#[test]
fn every_math_function_is_the_scalar_function_on_every_entry() {
    // Values that are valid for most functions, then ones valid for the
    // acosh, and ones that fail almost everywhere.
    let sets: [(&[f64], &[f64]); 4] = [
        (&[0.5, 0.25, 0.75, 0.9, 0.125], &[0.5, 2.0, 0.75, 0.3, 4.0]),
        (&[1.5, 2.0, 3.0, 4.5, 10.0], &[1.5, 0.5, 3.0, 1.0, 2.0]),
        (&[-0.5, 0.5, -2.0, 0.0, 3.0], &[0.0, 0.0, 1.0, -1.0, 2.5]),
        (
            &[1e3, -1e3, 800.0, 1e-300, 0.0],
            &[-0.0, 0.0, 5e-324, 1e300, -1e300],
        ),
    ];
    let mut compared = 0;
    let mut failed = 0;
    for f in MathFn::ALL {
        for (xs, ys) in sets {
            let src = program_for(f, xs, ys);
            let o = assert_same(&src);
            // The first printed line is the array result (or the error the
            // scalar function gives at the lowest failing entry).
            let want: Result<Vec<f64>, String> = xs
                .iter()
                .zip(ys)
                .map(|(x, y)| scalar(f, *x, if f.arity() == 2 { *y } else { 0.0 }))
                .collect();
            match want {
                Ok(v) => {
                    assert_eq!(o.error, None, "{src}");
                    let first = o.stdout.lines().next().unwrap();
                    same_entries(&entries(first), &v, &format!("{f:?} {xs:?}"));
                    compared += 1;
                }
                Err(message) => {
                    assert_eq!(o.error.as_deref(), Some(message.as_str()), "{src}");
                    assert_eq!(o.stdout, "", "{src}");
                    failed += 1;
                }
            }
        }
    }
    assert!(compared >= 30, "only {compared} programs succeeded");
    assert!(failed >= 15, "only {failed} programs failed");
}

// ----- errors: the same diagnostic everywhere -----

#[test]
fn runtime_errors_are_the_scalar_errors_with_the_same_rendering() {
    for (src, message) in [
        (
            "x = [1, 2, 3]\nz = [1, 0, 1]\nprint(x / z)\n",
            "division by zero",
        ),
        ("x = [1e200, 1, 1]\nprint(x * x)\n", "arithmetic overflow"),
        (
            "x = [1e300, 1]\ny = [1e-300, 1]\nprint(x / y)\n",
            "arithmetic overflow",
        ),
        (
            "x = [4, -1, 9]\nprint(sqrt(x))\n",
            "`sqrt` of a negative number has no real value",
        ),
        (
            "x = [1, 0, 2]\nprint(log(x))\n",
            "`log` of zero has no real value",
        ),
        (
            "x = [1, 0]\ny = [1, 0]\nprint(atan2(x, y))\n",
            "`atan2` of the zero vector has no real value",
        ),
        ("x = [1, 1000]\nprint(exp(x))\n", "arithmetic overflow"),
        (
            "x = [0.5, 2]\nprint(asin(x))\n",
            "`asin` of a number outside [-1, 1] has no real value",
        ),
    ] {
        let o = assert_same(src);
        assert_eq!(o.error.as_deref(), Some(message), "{src}");
        assert_eq!(o.stdout, "", "{src}");
        let d = o.diagnostic.unwrap();
        assert!(d.contains("test.sky:"), "{d}");
    }
}

#[test]
fn shape_and_dimension_mistakes_are_check_errors() {
    for (src, message) in [
        (
            "x = [1, 2, 3]\ny = [1, 2]\nz = x * y\n",
            "cannot apply `*` to Vector[3] and Vector[2]",
        ),
        (
            "x = [1, 2, 3]\ny = [1, 2]\nz = x / y\n",
            "cannot apply `/` to Vector[3] and Vector[2]",
        ),
        (
            "x = [1, 2, 3]\nM = [[1, 2, 3], [4, 5, 6]]\nz = x * M\n",
            "cannot apply `*` to a vector and a matrix",
        ),
        (
            "x = [1 m, 2 m]\nz = exp(x)\n",
            "the argument of `exp` must be dimensionless, found length",
        ),
        (
            "x = [1 m, 2 m]\ny = [1 s, 2 s]\nz = max(x, y)\n",
            "the arguments of `max` must have the same dimension, found length and time",
        ),
        (
            "x = [1, 2, 3]\ny = [1, 2]\nz = atan2(x, y)\n",
            "cannot apply `atan2` to Vector[3] and Vector[2]",
        ),
        (
            "x = [1, 2, 3]\nz = x ^ 2\n",
            "cannot raise a vector to a power",
        ),
    ] {
        let lexed = sky_syntax::lex(src).unwrap();
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:?}");
        let checked = sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
        let errs: Vec<&str> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| d.message.as_str())
            .collect();
        assert_eq!(errs, [message], "{src}");
    }
}

// ----- values the checker cannot type -----

#[test]
fn parameters_without_annotations_take_arrays_and_numbers() {
    let src = "def f(x):\n    return sqrt(x * x)\n\
               def g(x, y):\n    return x / y + max(x, y)\n\
               print(f([3, 4]), f(5), f([[1, 2], [3, 4]]))\n\
               print(g([4, 9], [2, 3]), g(4, 2))\n";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    assert_eq!(o.stdout, "[3, 4] 5 [[1, 2], [3, 4]]\n[6, 12] 6\n");
    // The runtime errors of an untyped call are the typed ones.
    let o = assert_same("def f(x, y):\n    return x / y\nprint(f([1, 2], [1, 0]))\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
    let o = assert_same("def f(x, y):\n    return x * y\nprint(f([1, 2], [1, 2, 3]))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("cannot apply `*` to Vector[2] and Vector[3]")
    );
    let o = assert_same("def f(x, y):\n    return atan2(x, y)\nprint(f([1, 2], [1, 2, 3]))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("cannot apply `atan2` to Vector[2] and Vector[3]")
    );
}

// ----- failures inside a fused chain -----

#[test]
fn a_late_operation_failing_early_and_an_early_one_failing_late() {
    for n in [3usize, 100, 600] {
        let mut x = vec![4.0; n];
        let mut y = vec![1.0; n];
        // `sqrt(x)` (the earlier operation) fails only at the last entry,
        // `log(y)` (the later) at the first: the chain reports sqrt's.
        x[n - 1] = -1.0;
        y[0] = 0.0;
        let src = format!(
            "x = {}\ny = {}\nr = sqrt(x) + log(y)\nprint(r)\n",
            literal(&x, ""),
            literal(&y, "")
        );
        let o = assert_same_and_fused(&src);
        assert_eq!(
            o.error.as_deref(),
            Some("`sqrt` of a negative number has no real value"),
            "n={n}"
        );
        let d = o.diagnostic.unwrap();
        assert!(d.contains("sqrt(x)"), "{d}");
        // Only the later operation fails: its lowest entry decides.
        x[n - 1] = 4.0;
        y[0] = 1.0;
        y[n / 2] = -1.0;
        y[n - 1] = 0.0;
        let src = format!(
            "x = {}\ny = {}\nr = sqrt(x) + log(y)\nprint(r)\n",
            literal(&x, ""),
            literal(&y, "")
        );
        let o = assert_same_and_fused(&src);
        assert_eq!(
            o.error.as_deref(),
            Some("`log` of a negative number has no real value"),
            "n={n}"
        );
        // A divisor that is zero because an earlier operation made it so.
        let src = format!(
            "x = {}\nz = {}\nr = abs(x / (z - z))\nprint(r)\n",
            literal(&vec![1.0; n], ""),
            literal(&vec![2.0; n], "")
        );
        let o = assert_same_and_fused(&src);
        assert_eq!(o.error.as_deref(), Some("division by zero"));
    }
}

#[test]
fn a_chain_that_overflows_late_is_still_the_first_failing_operations_error() {
    let n = 600;
    let mut x = vec![2.0; n];
    x[7] = 1e200;
    let src = format!("x = {}\nr = (x * x) / (x - x)\nprint(r)\n", literal(&x, ""));
    // `x * x` overflows at entry 7; `x - x` is zero everywhere: the
    // multiplication is first in program order.
    let o = assert_same_and_fused(&src);
    assert_eq!(o.error.as_deref(), Some("arithmetic overflow"));
}

// ----- long arrays: parallel and native kernel paths, bit-exact -----

/// `[[a0], [a1], ...]`, a column.
fn column(xs: &[f64]) -> String {
    let items: Vec<String> = xs.iter().map(|x| format!("[{x:e}]")).collect();
    format!("[{}]", items.join(", "))
}

#[test]
fn long_arrays_take_the_parallel_and_native_paths_and_stay_bit_exact() {
    // 150 by 150 matrices from outer products (a column times a row), so the
    // source stays short: 22500 entries each, above every kernel threshold.
    let m = 150usize;
    let ca = [dataset(m, 11), dataset(m, 12), dataset(m, 13)];
    let cb = [dataset(m, 21), dataset(m, 22), dataset(m, 23)];
    let outer = |k: usize| -> Vec<f64> {
        (0..m * m)
            .map(|e| 0.0 + ca[k][e / m] * cb[k][e % m])
            .collect()
    };
    let (x, y, s) = (outer(0), outer(1), outer(2));
    let row = |k: usize| format!("[{}]", literal(&cb[k], ""));
    let mut src = String::new();
    for (name, k) in [("x", 0), ("y", 1), ("s", 2)] {
        src += &format!("{name} = {} @ {}\n", column(&ca[k]), row(k));
    }
    src += "r = min(sqrt(x * y) / s, hypot(x, y)) * atan2(y, x) - floor(x / s) * exp(x / 1e5)\n";
    src += "print(r[0, 0], r[75, 75], r[149, 149])\nprint(r[3])\nprint(r[149])\n";
    let o = assert_same_and_fused(&src);
    assert_eq!(
        o.error,
        None,
        "{}",
        o.diagnostic.clone().unwrap_or_default()
    );
    let want: Vec<f64> = (0..m * m)
        .map(|i| {
            let a = (x[i] * y[i]).sqrt() / s[i];
            let b = x[i].hypot(y[i]);
            a.min(b) * y[i].atan2(x[i]) - (x[i] / s[i]).floor() * (x[i] / 1e5).exp()
        })
        .collect();
    let lines: Vec<&str> = o.stdout.lines().collect();
    same_entries(
        &lines[0]
            .split(' ')
            .map(|t| t.parse::<f64>().unwrap())
            .collect::<Vec<f64>>(),
        &[want[0], want[75 * m + 75], want[m * m - 1]],
        "corners",
    );
    same_entries(&entries(lines[1]), &want[3 * m..4 * m], "row 3");
    same_entries(&entries(lines[2]), &want[149 * m..], "row 149");
}
