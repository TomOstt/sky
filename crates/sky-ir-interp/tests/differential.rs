//! Differential test: a Sky program must behave identically through
//! `sky run`'s tree-walking interpreter and through lowering to SIR plus
//! the reference interpreter in this crate. "Identically" means the same
//! stdout bytes and the same runtime error, compared as the whole rendered
//! diagnostic (message, primary span and label, secondary labels, notes,
//! help), or both succeed.
//!
//! The native side compiles the same SIR (as lowered, and after the default
//! optimization pipeline) to machine code with `sky_jit` and must agree with
//! the tree-walker in the same way; a program native code refuses to compile
//! fails the test.
//!
//! The SIR side runs `sky_lower::lower_source`, which also verifies the
//! lowered module, so every program here is checked three ways: same
//! output, same errors, and valid SIR.

use std::collections::HashSet;

use sky_derive::SirDerivatives;
use sky_diag::{Diagnostic, Severity, SourceFile, Span};
use sky_interp::{run_module_with_derivatives, RuntimeEnv};

/// What running a program did: its stdout, and the runtime error that
/// stopped it, if any: its message, and the diagnostic as `sky run` prints
/// it, rendered against the source (so a span, a label or a note that differs
/// is a difference).
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    error: Option<String>,
    diagnostic: Option<String>,
}

/// A runtime error's diagnostic, rendered as `sky run` prints it.
fn rendered(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("test.sky", src))
}

/// Lex, parse and check `src`, panicking with the diagnostics if it is not
/// a valid program: a differential test on a program `sky run` refuses
/// would compare nothing. Also returns the `A⁻¹ @ b` sites the checker
/// rewrote to Cholesky solves, which `sky run` honours.
fn parse_checked(src: &str) -> (sky_ast::Module, HashSet<Span>, sky_types::Checked) {
    let lexed = sky_syntax::lex(src).unwrap_or_else(|_| panic!("lex error in {src:?}"));
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "parse errors in {src:?}: {errors:?}");
    let checked = sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errs.is_empty(), "check errors in {src:?}: {errs:?}");
    let solves = checked.cholesky_solves.clone();
    (module, solves, checked)
}

/// The `sky run` side: exactly what `sky-cli`'s `run` does after checking.
fn run_tree_walker(src: &str) -> Outcome {
    let (module, solves, checked) = parse_checked(src);
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        env.add_chain_plans(&checked.chains);
        // A derivative runs on SIR through `sky-derive`, as in `sky run`.
        let mut derivatives = SirDerivatives::new(&module, src);
        let result = run_module_with_derivatives(
            &module,
            src,
            &mut env,
            &mut out,
            &solves,
            &mut derivatives,
        );
        Outcome {
            stdout: String::from_utf8(out).expect("utf-8 output"),
            error: result.as_ref().err().map(|e| e.message.clone()),
            diagnostic: result.err().map(|e| rendered(&e.to_diagnostic(), src)),
        }
    })
}

/// The SIR side: lower (lex, parse, check, lower, verify) and run the result
/// in this crate's interpreter. Every program given to [`assert_same`] must
/// lower: a lowering refusal fails the test.
fn run_lowered(src: &str) -> Option<Outcome> {
    let ir =
        sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering failed on:\n{src}\n{e}"));
    Some(run_ir(&ir, src))
}

/// Runs a lowered module through the reference interpreter into an
/// [`Outcome`].
fn run_ir(ir: &sky_ir::Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let result = sky_ir_interp::run(ir, &mut out);
        Outcome {
            stdout: String::from_utf8(out).expect("utf-8 output"),
            error: result.as_ref().err().map(|e| e.message.clone()),
            diagnostic: result.err().map(|e| rendered(&e.to_diagnostic(), src)),
        }
    })
}

/// What the native engine does with `ir`. Every program of this file must
/// compile: a refusal fails the test, naming the program.
fn run_native(ir: &sky_ir::Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let compiled = sky_jit::compile(ir)
            .unwrap_or_else(|e| panic!("native code refused a program:\n{src}\n{e}"));
        let mut out = Vec::new();
        // A value the checker could not type that is not what a declaration
        // says stops native code, and SIR runs the program instead (what
        // `sky run` does); nothing in this file's programs does that.
        let (result, deopt) =
            compiled.run_or_rerun(&mut out, |w| sky_ir_interp::run(ir, &mut { w }));
        assert!(!deopt, "native code deoptimized on:\n{src}");
        Outcome {
            stdout: String::from_utf8(out).expect("utf-8 output"),
            error: result.as_ref().err().map(|e| e.message.clone()),
            diagnostic: result.err().map(|e| rendered(&e.to_diagnostic(), src)),
        }
    })
}

/// `ir` after the default optimization pipeline, as `sky run` runs it.
fn optimized(ir: &sky_ir::Module) -> sky_ir::Module {
    let mut ir = ir.clone();
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut ir)
        .expect("the pipeline runs");
    ir
}

/// The two resource-limit errors. Which one trips first depends on how much
/// Rust stack each interpreter spends per call (a debug tree-walker runs out
/// of stack budget before 1000 calls; the leaner SIR interpreter reaches the
/// call limit), not on the program's meaning, so they compare as one.
fn normalized(o: Outcome) -> Outcome {
    let is_limit = |e: &str| {
        e.starts_with("recursion is too deep")
            || e == "the program nests too deeply for the interpreter"
    };
    let limit = o.error.as_deref().is_some_and(is_limit);
    Outcome {
        error: o.error.map(|e| {
            if is_limit(&e) {
                "<resource limit>".to_string()
            } else {
                e
            }
        }),
        diagnostic: if limit { None } else { o.diagnostic },
        ..o
    }
}

/// Asserts that `src` behaves the same under both interpreters. Returns the
/// (shared) outcome so a test can also pin it.
fn assert_same(src: &str) -> Outcome {
    let expected = run_tree_walker(src);
    if let Some(actual) = run_lowered(src) {
        assert_eq!(
            normalized(actual),
            normalized(expected.clone()),
            "SIR interpreter differs from `sky run` on:\n{src}"
        );
    }
    let ir = sky_lower::lower_source(src).expect("lowers");
    for (which, m) in [("lowered", ir.clone()), ("optimized", optimized(&ir))] {
        assert_eq!(
            normalized(run_native(&m, src)),
            normalized(expected.clone()),
            "native code ({which} SIR) differs from `sky run` on:\n{src}"
        );
    }
    expected
}

#[test]
fn arithmetic_and_printing() {
    let o = assert_same("print(1 + 2 * 3)\nprint(2 ^ 10, 7 / 2)\n");
    assert_eq!(o.stdout, "7\n1024 3.5\n");
    assert_eq!(o.error, None);
}

#[test]
fn units_print_in_si() {
    let o = assert_same("v = 36 km / 1 h\nprint(v)\nprint(2 m + 3 cm)\n");
    assert_eq!(o.stdout, "10 m/s\n2.03 m\n");
}

#[test]
fn while_loop_and_functions() {
    let src = "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n\
               total = 0\ni = 1\nwhile i <= 10:\n    total += i\n    i += 1\nprint(total, fact(5))\n";
    let o = assert_same(src);
    assert_eq!(o.stdout, "55 120\n");
}

#[test]
fn runtime_errors_have_the_same_messages() {
    let o = assert_same("print(1)\nprint(1 / (2 - 2))\n");
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
    let o = assert_same("print(10 ^ 400)\n");
    assert_eq!(o.error.as_deref(), Some("arithmetic overflow"));
    let o = assert_same("print((0 - 8) ^ (1 / 3))\n");
    assert_eq!(o.error.as_deref(), Some("the result is not a real number"));
}

#[test]
fn runaway_recursion_stops_cleanly() {
    let o = assert_same("def f(x):\n    return f(x)\nf(1)\n");
    // Which limit trips first (call depth or stack budget) depends on the
    // build's frame sizes; a debug `sky run` hits the stack budget first.
    // Both are clean errors, and both interpreters use the same limits.
    let e = o.error.expect("fails");
    assert!(
        e.starts_with("recursion is too deep")
            || e == "the program nests too deeply for the interpreter",
        "{e}"
    );
}

#[test]
fn math_functions_agree_including_domain_errors() {
    let o = assert_same(
        "x = 2 m * exp(-1 s / 2 s) * cos(2 * pi * 0.25 Hz * 1 s)\n\
         print(sqrt(9 m^2), hypot(3 m, 4 m), max(1 s, 2 s), round(2.5), round(-0.4))\n\
         print(atan2(1 m, 1 m), log10(1000), cbrt(27 m^3))\n",
    );
    assert_eq!(o.error, None);
    // Each failure is the same error, with the same message, in both.
    for bad in [
        "print(log(0))\n",
        "print(sqrt(0 - 1))\n",
        "print(asin(2))\n",
        "print(atan2(0 m, 0 m))\n",
        "print(exp(1000))\n",
    ] {
        let o = assert_same(bad);
        assert!(o.error.is_some(), "{bad}");
    }
}

#[test]
fn added_math_functions_and_degrees_agree_including_domain_errors() {
    let o = assert_same(
        "print(expm1(1e-10), log1p(1e-10), expm1(0.5), log1p(2))\n\
         print(sinh(1), cosh(1), tanh(0.5), asinh(2), acosh(3), atanh(0.25))\n\
         print(erf(0.5), erf(-1.5), erf(3), erf(7), erf(0))\n\
         print(sign(-3 m), sign(0 kg), sign(2 m/s), sign(0 - 0.0))\n\
         print(sin(30 deg), cos(60 deg), tan(45 deg), 180 deg)\n\
         print(asinh(1e308), acosh(1e308), tanh(1000))\n",
    );
    assert_eq!(o.error, None);
    assert!(
        o.stdout
            .starts_with("1.00000000005e-10 9.999999999500001e-11 "),
        "{}",
        o.stdout
    );
    for (bad, message) in [
        (
            "print(acosh(0.5))\n",
            "`acosh` of a number below 1 has no real value",
        ),
        (
            "print(atanh(1))\n",
            "`atanh` of a number outside (-1, 1) has no real value",
        ),
        (
            "print(atanh(0 - 2))\n",
            "`atanh` of a number outside (-1, 1) has no real value",
        ),
        ("print(log1p(0 - 1))\n", "`log1p` of -1 has no real value"),
        (
            "print(log1p(0 - 3))\n",
            "`log1p` of a number below -1 has no real value",
        ),
        ("print(expm1(1000))\n", "arithmetic overflow"),
        ("print(sinh(1000))\n", "arithmetic overflow"),
        ("print(cosh(0 - 1000))\n", "arithmetic overflow"),
    ] {
        let o = assert_same(bad);
        assert_eq!(o.error.as_deref(), Some(message), "{bad}");
    }
}

#[test]
fn derivatives_of_added_math_functions() {
    let o = assert_same(
        "def f(x: rad) -> rad:\n    return expm1(x) + log1p(x) + sinh(x) + cosh(x) + tanh(x) + erf(x)\n\
         def g(x: rad) -> rad:\n    return asinh(x) + acosh(x + 2 rad) + atanh(x / 2) + sign(x)\n\
         print(derivative(f, x)(0.5 rad))\nprint(derivative(g, x)(0.5 rad))\n",
    );
    assert_eq!(o.error, None);
    // f' = e^x + 1/(1+x) + cosh x + sinh x + sech^2 x + 2/sqrt(pi) e^(-x^2)
    // at 0.5, one number per line.
    assert_eq!(o.stdout.lines().count(), 2, "{}", o.stdout);
    let o = assert_same(
        "def root(a: rad) -> rad:\n    return acosh(a)\nprint(derivative(root, a)(1 rad))\n",
    );
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

#[test]
fn transpose_glyph_and_function_agree() {
    let o = assert_same(
        "\
A = [[1, 2, 3], [4, 5, 6]]
K = [[2 N/m, 1 N/m], [1 N/m, 3 N/m]]
print(Aᵀ, transpose(A), Aᵀᵀ)
print(Aᵀ @ A)
print(A @ Aᵀ)
print(Kᵀ, transpose(K), (A @ Aᵀ)⁻¹)
print(5 m ᵀ, transpose(2 s))
def gram(M: Matrix[R, C]):
    return Mᵀ @ M
def flip(M: Matrix[R, C]):
    return transpose(M)
print(gram(A), flip(A), gram(Aᵀ))
S = Aᵀ @ A
i = 0
while i < 3:
    S = Sᵀ + S
    i += 1
print(S)
",
    );
    assert_eq!(o.error, None);
    assert!(
        o.stdout.starts_with("[[1, 4], [2, 5], [3, 6]] [[1, 4], [2, 5], [3, 6]] [[1, 2, 3], [4, 5, 6]]\n[[17, 22, 27]"),
        "{}",
        o.stdout
    );
}

#[test]
fn transposing_a_vector_is_the_same_error_in_both() {
    // The checker rejects it, so this goes through the interpreters via a
    // function whose parameter has no declared shape.
    let o = assert_same("def flip(x):\n    return xᵀ\nprint(flip([[1, 2], [3, 4]]))\n");
    assert_eq!(o.stdout, "[[1, 3], [2, 4]]\n");
    assert_eq!(o.error, None);
    let o = assert_same("def flip(x):\n    return transpose(x)\nprint(flip([1, 2, 3]))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("cannot transpose Vector[3]: transpose needs a matrix")
    );
}

#[test]
fn for_range_forms_agree() {
    let src = "\
for i in range(3):
    print(i)
for i in range(2, 5):
    print(i)
for i in range(10, 0, -4):
    print(i)
for i in range(4, 4):
    print(i)
for i in range(0):
    print(i)
print(\"done\")
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "0\n1\n2\n2\n3\n4\n10\n6\n2\ndone\n");
    assert_eq!(o.error, None);
}

#[test]
fn for_variable_and_accumulators_after_the_loop() {
    // A loop known to run leaves its variable and everything it binds
    // defined; a `for` carries them out like a `while true`.
    let src = "\
total = 0
for i in range(1, 5):
    total += i
    last = i * i
print(total, i, last)
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "10 4 16\n");
    // A variable count may be zero: the variable is only used inside.
    let src = "\
n = 3
count = 0
for i in range(n):
    count += 1
print(count)
n = 0
for i in range(n):
    count += 100
print(count)
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "3\n3\n");
    // A variable bound before the loop survives a loop that does not run.
    let o = assert_same("i = 7\nn = 0\nfor i in range(n):\n    pass\nprint(i)\n");
    assert_eq!(o.stdout, "7\n");
}

#[test]
fn for_with_a_computed_step_and_bounds() {
    let src = "\
n = 10
step = 3
for i in range(n - 8, n + 1, step):
    print(i)
step = 0 - 4
for i in range(n, 0, step):
    print(i)
for i in range(1, n, 2 * step + 9):
    print(i)
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "2\n5\n8\n10\n6\n2\n1\n2\n3\n4\n5\n6\n7\n8\n9\n");
}

#[test]
fn for_break_and_continue_agree() {
    let src = "\
for i in range(10):
    if i == 2:
        continue
    if i == 6:
        break
    print(i)
print(i)
for i in range(3):
    for j in range(3):
        if j == i:
            continue
        if j == 2:
            break
        print(i, j)
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "0\n1\n3\n4\n5\n6\n0 1\n1 0\n2 0\n2 1\n");
}

#[test]
fn for_rebinding_the_variable_does_not_change_the_iteration() {
    let src = "for i in range(3):\n    print(i)\n    i = 100\nprint(i)\n";
    let o = assert_same(src);
    assert_eq!(o.stdout, "0\n1\n2\n100\n");
}

#[test]
fn for_inside_functions_and_with_return() {
    let src = "\
def sum_to(n):
    s = 0
    for k in range(1, n + 1):
        s += k
    return s
def first_multiple(n, m):
    for k in range(1, n):
        if k * m > 20:
            return k
    return 0
def scaled(x):
    t = 0 m
    for k in range(4):
        t += x * k
    return t
print(sum_to(10), first_multiple(100, 7), first_multiple(2, 7), scaled(2 m))
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "55 3 0 12 m\n");
}

#[test]
fn for_riemann_sum() {
    let src = "\
n = 200
dx = 1 / n
s = 0
for i in range(n):
    x = (i + 0.5) * dx
    s += x^2 * dx
print(s)
t = 0 m
for i in range(1, 11):
    t += i * 1 m
print(t)
";
    let o = assert_same(src);
    assert!(o.stdout.ends_with("55 m\n"), "{}", o.stdout);
}

#[test]
fn for_with_units_in_the_body() {
    let src = "\
v = 0 m/s
a = 2 m/s^2
dt = 0.5 s
x = 0 m
for i in range(4):
    x += v * dt
    v += a * dt
print(x, v)
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "3 m 4 m/s\n");
}

#[test]
fn a_for_over_an_array_says_lowering_needs_array_values() {
    let src = "for x in [1, 2]:\n    print(x)\n";
    let e = sky_lower::lower_source(src).expect_err("arrays do not lower yet");
    assert!(
        e.to_string()
            .contains("iteration over arrays lowers once SIR has array values"),
        "{e}"
    );
    // The tree walker still runs it.
    assert_eq!(run_tree_walker(src).stdout, "1\n2\n");
}

// ----- arrays -----

/// The spring system of `sky-cli`'s `run_repl` tests: two springs in series,
/// fixed at the left, `K u = f`. `K` is only declared Symmetric, so
/// `u = K⁻¹ f` is the general inverse.
const SPRING: &str = "\
K: Symmetric Matrix[2, 2] N/m = [[2000 N/m, -1000 N/m], [-1000 N/m, 1000 N/m]]
f: Vector[2] N = [0 N, 100 N]
u = K⁻¹ @ f
print(u)
print(u[1] / 1 mm)
";

#[test]
fn the_spring_system_prints_the_same_displacements() {
    let o = assert_same(SPRING);
    assert_eq!(o.error, None);
    assert_eq!(o.stdout, "[0.1 m, 0.2 m]\n200\n");
}

#[test]
fn a_positive_definite_stiffness_is_solved_by_cholesky_in_both() {
    // PositiveDefinite is proven from the entries, so the checker records
    // the site and `sky run` factors K. Lowering emits `solve[cholesky]`
    // for the same site, so the output is bit-identical (`assert_same`).
    let src = SPRING.replace("Symmetric", "PositiveDefinite");
    let ir = sky_lower::lower_source(&src).expect("lowers");
    let text = sky_ir::print(&ir);
    assert!(text.contains("solve[cholesky]"), "{text}");
    assert!(!text.contains("inverse"), "{text}");
    let o = assert_same(&src);
    assert_eq!(o.error, None);
    assert!(o.stdout.trim_end().contains(" m]"), "{}", o.stdout);
}

#[test]
fn only_the_sites_the_checker_rewrote_become_solves() {
    // `K` is PositiveDefinite, but `(K + K)⁻¹` has no tags (the checker
    // claims none for a sum), so it stays an inverse and a product, and the
    // spellings of the rewritten form (`K⁻¹`, `inv(K)`, `(K⁻¹)`) are solves.
    let src = "\
K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]
f = [1, 2]
a = K⁻¹ @ f
b = inv(K) @ f
c = (K + K)⁻¹ @ f
d = (K⁻¹) @ f
print(a, b, c, d)
";
    let ir = sky_lower::lower_source(src).expect("lowers");
    let text = sky_ir::print(&ir);
    assert_eq!(text.matches("solve[cholesky]").count(), 3, "{text}");
    // `(K + K)⁻¹` has no tags: a general solve (LU), no inverse formed.
    assert_eq!(text.matches("solve[lu]").count(), 1, "{text}");
    assert_eq!(text.matches("inverse").count(), 0, "{text}");
    assert_eq!(assert_same(src).error, None);
}

#[test]
fn a_cholesky_solve_with_several_right_hand_sides() {
    let src = "\
K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]
print(K⁻¹ @ [[1, 2], [3, 4]])
print(K⁻¹ @ identity(2))
print(K⁻¹ @ [1 m, 2 m])
";
    let ir = sky_lower::lower_source(src).expect("lowers");
    assert_eq!(sky_ir::print(&ir).matches("solve[cholesky]").count(), 3);
    assert_eq!(assert_same(src).error, None);
}

#[test]
fn a_function_reading_a_positive_definite_name_still_solves_by_cholesky() {
    let src = "\
K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]
def displace(f: Vector[2]):
    return K⁻¹ @ f
print(displace([1, 2]))
def solve_with(A: PositiveDefinite Matrix[2, 2], f: Vector[2]):
    return A⁻¹ @ f
print(solve_with(K, [3, 4]))
";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().count(), 2);
}

#[test]
fn literals_arithmetic_and_builtins_agree() {
    let o = assert_same(
        "\
v = [1 m, 2 m, 3 m]
w = [1, 2, 3]
A = [[1, 2], [3, 4]]
B = [[1 N, 2 N, 3 N],
     [4 N, 5 N, 6 N]]
t = [v[0], v[1] + 1 m]
print(v, w, A, B, t)
print(v + v, v - v, -v, 2 * A, A * 2, A / 4, 3 m * w)
print(A @ [1, 1], [1, 1] @ A, w @ w, A @ A)
print(zeros(3), zeros(2, 3), identity(3), len(v), len(B), len(zeros(4, 2)))
print(A⁻¹, inv(A), A⁻¹ @ A)
print(A[0], A[1, 0], A[:, 1], A[0, 1:], A[0:1, 0:2], v[1:], v[:2], v[0:3])
",
    );
    assert_eq!(o.error, None);
    assert!(o
        .stdout
        .starts_with("[1 m, 2 m, 3 m] [1, 2, 3] [[1, 2], [3, 4]]"));
}

#[test]
fn arrays_in_functions_loops_and_conditionals() {
    let o = assert_same(
        "\
def norm2(v: Vector[N]):
    return v @ v
def count(v: Vector[N]):
    return len(v)
def pair(x, y):
    return [x, y]
def first(a):
    return a[0]
def scale(A: Matrix[N, N], k):
    return A * k
i = 0
total = [0, 0]
while i < 3:
    total = total + [i, 2 * i]
    i += 1
print(total, norm2([3, 4]), count([1, 2, 3]), count(zeros(5)))
print(pair(1, 2), pair(1 m, 2 m), first(pair(5, 6)), first([[1, 2], [3, 4]]))
print(scale(identity(2), 3))
best = [1, 2] if i > 2 else [3, 4]
print(best, best[0] if i > 2 else best[1])
",
    );
    assert_eq!(o.error, None);
    assert_eq!(
        o.stdout.lines().next(),
        Some("[3, 6] 25 3 5"),
        "{}",
        o.stdout
    );
}

#[test]
fn index_and_bounds_errors_have_the_same_messages() {
    let cases = [
        (
            "v = [1, 2, 3]\ni = 3\nprint(v[i])\n",
            "index 3 is out of bounds for axis 0 with length 3",
        ),
        (
            "A = [[1, 2], [3, 4]]\ni = 2\nprint(A[0, i])\n",
            "index 2 is out of bounds for axis 1 with length 2",
        ),
        (
            "A = [[1, 2], [3, 4]]\ni = 2\nprint(A[i])\n",
            "index 2 is out of bounds for axis 0 with length 2",
        ),
        (
            "v = [1, 2, 3]\ni = 0 - 1\nprint(v[i])\n",
            "an index cannot be negative",
        ),
        (
            "v = [1, 2, 3]\ni = 1 / 2\nprint(v[i])\n",
            "an index must be an integer, found 0.5",
        ),
        (
            "def at(v, i):\n    return v[i]\nprint(at([1, 2, 3], 1 m))\n",
            "an index must be dimensionless",
        ),
        (
            "v = [1, 2, 3]\ni = 5\nprint(v[0:i])\n",
            "slice 0..5 is out of bounds for axis 0 with length 3",
        ),
        (
            "v = [1, 2, 3]\ni = 5\nprint(v[i:])\n",
            "slice 5..3 is out of bounds for axis 0 with length 3",
        ),
        (
            "v = [1, 2, 3]\ni = 2\nprint(v[i:i])\n",
            "this slice is empty",
        ),
        (
            "A = [[1, 2], [3, 4]]\ni = 3\nprint(A[0:1, 1:i])\n",
            "slice 1..3 is out of bounds for axis 1 with length 2",
        ),
        (
            "def at(v, i):\n    return v[i]\nprint(at([1, 2], 2))\n",
            "index 2 is out of bounds for axis 0 with length 2",
        ),
        (
            "def at(v, i):\n    return v[i]\nprint(at(1, 0))\n",
            "cannot index a number",
        ),
    ];
    for (src, message) in cases {
        let o = assert_same(src);
        assert_eq!(o.error.as_deref(), Some(message), "{src}");
        assert_eq!(o.stdout, "", "{src}");
    }
}

#[test]
fn output_before_an_index_error_is_kept() {
    let o = assert_same("v = [1, 2]\nprint(v)\ni = 2\nprint(v[i])\nprint(\"never\")\n");
    assert_eq!(o.stdout, "[1, 2]\n");
    assert!(o.error.is_some());
}

#[test]
fn a_singular_matrix_has_the_same_error() {
    for src in [
        "print([[1, 2], [2, 4]]⁻¹)\n",
        "A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint(inv(A))\n",
        "A = [[0, 0], [0, 0]]\nb = [1, 2]\nprint(A⁻¹ @ b)\n",
        "def inverse_of(A):\n    return A⁻¹\nprint(inverse_of([[1, 1], [1, 1]]))\n",
    ] {
        let o = assert_same(src);
        assert_eq!(o.error.as_deref(), Some("the matrix is singular"), "{src}");
    }
    // Scalars keep their own error.
    let o = assert_same("print(0⁻¹)\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

#[test]
fn other_array_runtime_errors_agree() {
    // Overflow in an elementwise operation and in a product.
    let o = assert_same("v = [1e200, 1]\nprint(v * 1e200)\n");
    assert_eq!(o.error.as_deref(), Some("arithmetic overflow"));
    let o = assert_same("A = [[1e200, 0], [0, 1e200]]\nprint(A @ A)\n");
    assert_eq!(o.error.as_deref(), Some("arithmetic overflow"));
    // Division of an array by zero.
    let o = assert_same("v = [1, 2]\nz = 0\nprint(v / z)\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
    // Comparing arrays is not defined.
    let o = assert_same("def same(a, b):\n    return a == b\nprint(same([1, 2], [1, 2]))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("cannot compare a vector and a vector")
    );
    // A function given the wrong kind of value fails dynamically, in both.
    let o = assert_same("def f(a, b):\n    return a + b\nprint(f([1, 2], 3))\n");
    assert_eq!(o.error.as_deref(), Some("cannot add a vector and a number"));
    let o = assert_same("def f(a, b):\n    return a + b\nprint(f([1, 2], [1, 2, 3]))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("cannot add Vector[2] and Vector[3]")
    );
    let o = assert_same("def n(a):\n    return len(a)\nprint(n(3))\n");
    assert_eq!(
        o.error.as_deref(),
        Some("`len` needs a vector or a matrix, found a number")
    );
    let o = assert_same("def z(n):\n    return zeros(n)\nprint(z(3))\nprint(z(0))\n");
    assert_eq!(o.stdout, "[0, 0, 0]\n");
    assert_eq!(
        o.error.as_deref(),
        Some("`zeros` needs a size, a positive integer, found 0")
    );
    let o = assert_same("def f(a, b):\n    return [a, b]\nprint(f(1 m, 2 s))\n");
    assert!(o
        .error
        .as_deref()
        .is_some_and(|e| e.starts_with("array elements must have the same dimension")));
}

#[test]
fn a_matrix_declared_without_a_value_is_an_input_nothing_supplies() {
    let o = assert_same("A: Matrix[2, 2]\nprint(A[0, 0])\n");
    assert_eq!(o.error.as_deref(), Some("`A` is declared but has no value"));
}

/// A literal too large for an `f64` fails where it is evaluated, with
/// `sky run`'s error on `sky run`'s span, in the SIR interpreter too.
#[test]
fn a_literal_too_large_fails_with_the_same_error_on_the_same_span() {
    for (src, message) in [
        ("print(1)\nx = 1e999\nprint(2)\n", "number too large"),
        ("print(1)\nx = 1e999 m\nprint(2)\n", "number too large"),
        ("print(1)\nx = 1e306 km\nprint(2)\n", "arithmetic overflow"),
        ("if 1 > 0:\n    y = 2e400\n", "number too large"),
    ] {
        let (walker, sir) = located_errors(src);
        assert_eq!(
            walker.as_ref().map(|w| w.0.as_str()),
            Some(message),
            "{src}"
        );
        assert_eq!(sir, walker, "{src}");
        let o = assert_same(src);
        assert_eq!(o.error.as_deref(), Some(message));
    }
    let o = assert_same("print(1)\nif false:\n    x = 1e999\nprint(2)\n");
    assert_eq!((o.stdout.as_str(), o.error), ("1\n2\n", None));
}

/// The declaration is where the run stops: what is printed before it stays,
/// and the error points at the declaration, as in `sky run`.
#[test]
fn an_input_declaration_fails_where_it_is_after_the_earlier_output() {
    let src = "print(1)\nA: Matrix[2, 2]\nprint(2)\n";
    let o = assert_same(src);
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("`A` is declared but has no value"));
    let (walker, sir) = located_errors(src);
    assert_eq!(sir, walker);
}

// ----- derivative(f, x)(args) -----
//
// `sky run` evaluates a derivative by lowering the function to SIR and
// running `sky_ad`'s derivative in this crate's interpreter; the lowered
// program calls that same function. The two paths must print the same bytes.

const POSITION: &str = "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\n";

#[test]
fn the_derivative_of_a_position_is_a_velocity_in_both_engines() {
    let o = assert_same(&format!(
        "{POSITION}print(x(2 s))\nprint(derivative(x, t)(2 s))\n"
    ));
    assert_eq!(o.stdout, "20 m\n20 m/s\n");
    assert_eq!(o.error, None);
}

#[test]
fn derivatives_with_several_parameters_and_module_names() {
    let o = assert_same(
        "g = 9.5 m/s^2\n\
         def fall(t: s, m0: kg) -> J:\n    return 0.5 * m0 * (g * t)^2\n\
         g = 10 m/s^2\n\
         print(derivative(fall, t)(2 s, 3 kg))\n\
         print(derivative(fall, m0)(2 s, 3 kg))\n",
    );
    // E = m g² t²/2: dE/dt = m g² t = 600 W, dE/dm = g² t²/2 = 200 J/kg
    assert_eq!(o.stdout, "600 W\n200 m^2/s^2\n");
}

#[test]
fn derivatives_through_control_flow_loops_and_recursion() {
    let o = assert_same(
        "def cube(a: m) -> m^3:\n    return a * a * a\n\
         def steps(t: s) -> m:\n    if t > 1 s:\n        return 3 m/s * t\n    return 1 m\n\
         def triple(a: m) -> m:\n    total = 0 m\n    i = 0\n    while i < 3:\n        total = total + a\n        i += 1\n    return total\n\
         print(derivative(cube, a)(2 m), derivative(steps, t)(0.5 s), derivative(steps, t)(2 s))\n\
         print(derivative(triple, a)(2 m))\n",
    );
    assert_eq!(o.stdout, "12 m^2 0 m/s 3 m/s\n3\n");
}

#[test]
fn derivatives_of_math_functions() {
    let o = assert_same(
        "def wave(t: s) -> m:\n    return 2 m * sin(2 * pi * 0.25 Hz * t)\n\
         def decay(t: s) -> m:\n    return 2 m * exp(-t / 4 s) + sqrt(t / 1 s) * 1 m\n\
         print(derivative(wave, t)(0 s))\n\
         print(derivative(decay, t)(4 s))\n",
    );
    assert_eq!(o.error, None);
    assert!(
        o.stdout.starts_with("3.141592653589793 m/s\n"),
        "{}",
        o.stdout
    );
}

#[test]
fn a_second_derivative_is_the_derivative_of_a_function_that_calls_the_first() {
    let o = assert_same(&format!(
        "{POSITION}def v(t: s) -> m/s:\n    return derivative(x, t)(t)\n\
         print(v(2 s))\nprint(derivative(v, t)(2 s))\n"
    ));
    assert_eq!(o.stdout, "20 m/s\n10 m/s^2\n");
}

#[test]
fn the_function_prints_once_per_derivative() {
    let o = assert_same(
        "def f(t: s) -> m:\n    print(\"called\")\n    return 2 m/s * t\n\
         print(derivative(f, t)(1 s))\n",
    );
    assert_eq!(o.stdout, "called\n2 m/s\n");
}

#[test]
fn a_derivative_that_fails_fails_the_same_way() {
    // The slope of sqrt at 0 is infinite: division by zero, in both.
    let o = assert_same(
        "def root(a: m^2) -> m:\n    return sqrt(a)\n\
         print(1)\nprint(derivative(root, a)(0 m^2))\n",
    );
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

/// A run-time error's message, span and rendered diagnostic, if the run
/// failed.
type Located = Option<(String, Span, String)>;

/// The error of the tree-walker and of the SIR interpreter on `src`, with
/// the span and the whole rendered diagnostic: they must be the same,
/// including where they point and every label.
fn located_errors(src: &str) -> (Located, Located) {
    let (module, solves, checked) = parse_checked(src);
    let walker = sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        env.add_chain_plans(&checked.chains);
        let mut derivatives = SirDerivatives::new(&module, src);
        run_module_with_derivatives(&module, src, &mut env, &mut out, &solves, &mut derivatives)
            .err()
            .map(|e| (e.message.clone(), e.span, rendered(&e.to_diagnostic(), src)))
    });
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering failed: {e}"));
    let sir = sky_ast::with_deep_stack(|| {
        sky_ir_interp::run(&ir, &mut Vec::new())
            .err()
            .map(|e| (e.message.clone(), e.span, rendered(&e.to_diagnostic(), src)))
    });
    // Native code fails at the same place with the same diagnostic.
    let native = sky_ast::with_deep_stack(|| {
        let compiled = sky_jit::compile(&ir)
            .unwrap_or_else(|e| panic!("native code refused a program:\n{src}\n{e}"));
        let (result, deopt) =
            compiled.run_or_rerun(&mut Vec::new(), |w| sky_ir_interp::run(&ir, &mut { w }));
        assert!(!deopt, "native code deoptimized on:\n{src}");
        result
            .err()
            .map(|e| (e.message.clone(), e.span, rendered(&e.to_diagnostic(), src)))
    });
    assert_eq!(native, sir, "native code differs from SIR on:\n{src}");
    (walker, sir)
}

/// An operand the checker cannot type (an unannotated parameter's result)
/// that is not a Bool: both interpreters stop with the same message on the
/// same span, whichever construct wanted the Bool, and whichever operand.
#[test]
fn a_non_bool_operand_of_unknown_type_fails_alike_in_every_position() {
    let def = "def f(a):\n    return a\n";
    for (body, kw) in [
        ("print(f(1) and true)", "and"),
        ("print(true and f(1))", "and"),
        ("print(false or f(1))", "or"),
        ("print(f(1) or true)", "or"),
        ("print(true and true and f(1))", "and"),
        ("print(not f(1))", "not"),
        ("if f(1):\n    print(1)", "if"),
        ("if false:\n    print(1)\nelif f(1):\n    print(2)", "if"),
        (
            "if false:\n    print(1)\nelif false:\n    print(2)\nelif f(1):\n    print(3)",
            "if",
        ),
        ("print(1 if f(1) else 2)", "if"),
        ("while f(1):\n    print(1)", "while"),
        ("k = 0\nwhile k < 3 and f(1):\n    k += 1", "and"),
    ] {
        let src = format!("{def}{body}\n");
        let (walker, sir) = located_errors(&src);
        let (msg, span, _) = walker.clone().expect("sky run fails");
        assert_eq!(
            msg,
            format!("`{kw}` needs a Bool operand, found a number"),
            "{src}"
        );
        assert_eq!(sir, walker, "the SIR interpreter differs on:\n{src}");
        // The span is the operand's.
        assert!(src[span.start as usize..span.end as usize].starts_with("f(1)"));
    }
}

/// A Bool operand that is unknown but really a Bool passes through.
#[test]
fn an_unknown_operand_that_is_a_bool_is_checked_and_passed_on() {
    let o = assert_same(
        "def f(a):\n    return a\nprint(f(true) and f(false), f(false) or f(true), not f(false))\n\
         if f(true):\n    print(1)\nk = 0\nwhile f(k < 2):\n    k += 1\nprint(k, 1 if f(false) else 2)\n",
    );
    assert_eq!(o.stdout, "false true true\n1\n2 2\n");
}

// ----- operand spans: the whole diagnostic, not only the message -----
//
// A SIR operation records the source span of each operand at its use, so
// the interpreter's error has the tree-walker's labels, secondary labels and
// notes. `assert_same` compares the whole rendered diagnostic; these say
// which parts a program's error has, so the comparison cannot pass by both
// engines dropping them.

/// Runs `src` in both engines and returns the rendered diagnostic, which
/// must be the same and must contain each of `parts`.
fn labelled(src: &str, parts: &[&str]) -> String {
    let o = assert_same(src);
    let d = o
        .diagnostic
        .unwrap_or_else(|| panic!("no error on:\n{src}"));
    for part in parts {
        assert!(d.contains(part), "{part:?} is not in\n{d}\nfor:\n{src}");
    }
    d
}

#[test]
fn a_division_by_zero_points_at_the_divisor() {
    labelled(
        "x = 4\nz = 2 - 2\nprint(x / z)\n",
        &[
            "this divides by zero",
            "this is zero",
            "= note: Sky reports",
        ],
    );
    // The divisor is the operand's own span, also in `/=`, an array scale
    // and a reciprocal.
    labelled(
        "x = 4\nz = 2 - 2\nx /= z\n",
        &["this divides by zero", "this is zero"],
    );
    labelled(
        "v = [1, 2]\nz = 0\nprint(v / z)\n",
        &["this divides by zero", "this is zero"],
    );
    labelled("x = 0\nprint(x⁻¹)\n", &["this is zero"]);
    labelled("x = 0\nprint(inv(x))\n", &["this is zero"]);
    labelled("x = 0\nprint(x ^ (0 - 1))\n", &["this is zero"]);
}

#[test]
fn an_index_error_points_at_the_index_and_the_array() {
    labelled(
        "v = [1, 2, 3]\ni = 5\nprint(v[i])\n",
        &["valid indices are 0 to 2", "this is Vector[3]"],
    );
    labelled(
        "A = [[1, 2], [3, 4]]\ni = 0\nj = 7\nprint(A[i, j])\n",
        &["valid indices are 0 to 1", "this is Matrix[2,2]"],
    );
    labelled(
        "v = [1, 2, 3]\ni = 0 - 1\nprint(v[i])\n",
        &["cannot be negative"],
    );
    labelled(
        "v = [1, 2, 3]\ni = 1 / 2\nprint(v[i])\n",
        &["must be an integer"],
    );
    labelled(
        "def f(a):\n    return a\nv = [1, 2, 3]\ni = f(1 m)\nprint(v[i])\n",
        &["must be dimensionless", "this is length"],
    );
}

#[test]
fn a_slice_error_points_at_the_whole_item() {
    let d = labelled(
        "v = [1, 2, 3, 4]\nn = 9\nprint(v[0:n])\n",
        &["slice 0..9 is out of bounds", "this is Vector[4]"],
    );
    // The underline is `0:n`, which is no operand.
    assert!(d.contains("print(v[0:n])"), "{d}");
    labelled(
        "v = [1, 2, 3, 4]\nn = 2\nprint(v[n:n])\n",
        &["this slice is empty"],
    );
    // Bounds that are missing are still part of the item.
    labelled("v = [1, 2, 3, 4]\nn = 9\nprint(v[n:])\n", &["slice 9..4"]);
    labelled(
        "A = [[1, 2], [3, 4]]\nn = 9\nprint(A[1, n:])\n",
        &["axis 1"],
    );
    labelled(
        "A = [[1, 2], [3, 4]]\nn = 9\nprint(A[n:, 0])\n",
        &["axis 0"],
    );
}

#[test]
fn a_dimension_mismatch_labels_both_operands() {
    labelled(
        "def f(a):\n    return a\na = f(1 m)\nb = f(2 s)\nprint(a + b)\n",
        &["mismatched dimensions", "length", "time"],
    );
    labelled(
        "def f(a):\n    return a\na = f(1 m)\nb = f(2 s)\nprint(a - b)\n",
        &["cannot subtract time from length"],
    );
    labelled(
        "def f(a):\n    return a\na = f(1 m)\nb = f(2 s)\na -= b\n",
        &["mismatched dimensions"],
    );
    labelled(
        "def f(a):\n    return a\na = f(1 m)\nb = f(2 s)\nprint(a < b)\n",
        &["cannot compare length and time"],
    );
    labelled(
        "def f(a):\n    return a\na = f(1 m)\nb = f(2 s)\nprint(a == b)\n",
        &["cannot compare length and time"],
    );
    labelled(
        "def f(a):\n    return a\na = f([1, 2])\nb = f([1, 2, 3])\nprint(a + b)\n",
        &["mismatched shapes", "Vector[2]", "Vector[3]"],
    );
    labelled(
        "def f(a):\n    return a\na = f([[1, 2]])\nb = f([[1, 2]])\nprint(a @ b)\n",
        &["inner dimensions 2 and 1 differ", "Matrix[1,2]"],
    );
}

#[test]
fn math_and_linear_algebra_errors_keep_their_labels() {
    labelled("x = 0 - 4\nprint(sqrt(x))\n", &["this is negative"]);
    labelled("x = 0\nprint(log(x))\n", &["this is zero"]);
    labelled("x = 2\nprint(asin(x))\n", &["this is outside [-1, 1]"]);
    labelled(
        "def f(a):\n    return a\nx = f(2 m)\nprint(exp(x))\n",
        &["this is length"],
    );
    labelled(
        "A = [[1, 2], [2, 4]]\nprint(A⁻¹)\n",
        &["the matrix is singular", "this is Matrix[2,2]"],
    );
    labelled(
        "def f(a):\n    return a\nx = f(2 m)\ne = f(3)\nprint(x ^ e)\n",
        &["the exponent must be a constant number"],
    );
    labelled(
        "x = 1e200\nprint(x * x)\n",
        &["does not fit in a 64-bit float"],
    );
}

/// A chain of array operations that `fuse` turns into one kernel keeps each
/// operation's own operand spans (`sky-cli`'s engine tests run such chains
/// through the whole pass pipeline).
#[test]
fn an_array_chain_fails_like_the_tree_walker() {
    labelled(
        "a = [1, 2, 3]\nb = [1, 2, 3]\nz = 0\nprint((a + b) / z * 2)\n",
        &["this divides by zero", "this is zero"],
    );
    labelled("a = [1e200, 2]\nprint(a * 1e200)\n", &["does not fit"]);
}

/// A value the checker cannot type (an unannotated parameter's) fails at
/// run time with the operand's span, in every position an operand is read.
#[test]
fn unknown_operands_fail_with_the_tree_walkers_diagnostic() {
    let def = "def f(a):\n    return a\n";
    for (body, message) in [
        ("print(-f(true))", "`-` needs a number, found a Bool"),
        ("print(+f(true))", "`+` needs a number, found a Bool"),
        ("print(+f(\"hi\"))", "`+` needs a number, found a string"),
        ("print(inv(f(true)))", "`inv` needs a number, found a Bool"),
        ("print(f(true)⁻¹)", "`⁻¹` needs a number, found a Bool"),
        ("print(f(true) + 1)", "cannot add a Bool and a number"),
        ("print(f(true) * 2)", "cannot multiply a Bool and a number"),
        (
            "print(sqrt(f(true)))",
            "`sqrt` needs a number, found a Bool",
        ),
        ("print(f(1)[0])", "cannot index a number"),
        (
            "print([1, 2][f(true)])",
            "an index must be a number, found a Bool",
        ),
        (
            "print([f(true), 1])",
            "an array element must be a number, found a Bool",
        ),
    ] {
        let src = format!("{def}{body}\n");
        let o = assert_same(&src);
        assert_eq!(o.error.as_deref(), Some(message), "{src}");
    }
    // A number or an array passes `+` and `inv` unchanged.
    let o = assert_same(
        "def f(a):\n    return +a\ndef g(a):\n    return inv(a)\nprint(f(2 m), f([1, 2]), g(4), g([[2, 0], [0, 4]]))\n",
    );
    assert_eq!(o.error, None);
}

/// Found by the fuzzer (seed 275714): `+x` used to lower to `x`, so for a
/// parameter the checker cannot type, given a Bool, SIR did not stop at the
/// `+` as `sky run` does.
#[test]
fn a_unary_plus_of_an_unknown_non_number_stops_at_the_plus() {
    let d = labelled(
        "def f(a):\n    return +a + a\nprint(f(true))\n",
        &["`+` needs a number, found a Bool"],
    );
    assert!(d.contains("return +a + a"), "{d}");
    labelled(
        "def f(a):\n    return +a\nprint(f(true))\n",
        &["`+` needs a number, found a Bool"],
    );
    // Only where it is evaluated: a `+` that is never reached does not fail.
    let o =
        assert_same("def f(a):\n    if false:\n        return +a\n    return 1\nprint(f(true))\n");
    assert_eq!((o.stdout.as_str(), o.error), ("1\n", None));
}

// ----- grad(f)(args) and (∂f/∂x)(args) -----
//
// `grad(f)` is one derivative per parameter, gathered in a Vector. The
// tree-walker asks `sky-derive` for each; lowering emits a call of each
// `df/dx` and an `array.literal`. Both must print the same bytes.

const POTENTIAL: &str = "k = 3 N/m\ndef U(x: m, y: m) -> J:\n    return k * (x^2 + y^2)\n";

#[test]
fn the_gradient_of_a_potential_is_a_force_in_both_engines() {
    // U = k (x² + y²): ∇U = (2 k x, 2 k y) = (6 N, 12 N) at (1 m, 2 m).
    let o = assert_same(&format!(
        "{POTENTIAL}F = grad(U)(1 m, 2 m)\nprint(F)\nprint(F[1])\nprint(grad(U)(0 m, 0 m))\n"
    ));
    assert_eq!(o.stdout, "[6 N, 12 N]\n12 N\n[0 N, 0 N]\n");
    assert_eq!(o.error, None);
}

#[test]
fn a_gradient_is_a_vector_that_arithmetic_and_units_apply_to() {
    let o = assert_same(&format!(
        "{POTENTIAL}F = -grad(U)(1 m, 2 m)\n\
         total = F + [1 N, 1 N]\n\
         print(total)\n\
         print(F[0] + F[1])\n\
         print(len(grad(U)(1 m, 1 m)))\n"
    ));
    assert_eq!(o.stdout, "[-5 N, -11 N]\n-18 N\n2\n");
}

#[test]
fn a_gradient_with_one_or_three_parameters() {
    let o = assert_same(
        "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\n\
         def r2(a: m, b: m, c: m) -> m^2:\n    return a * a + 2 * b * b + 3 * c * c\n\
         print(grad(x)(2 s))\n\
         print(grad(r2)(1 m, 1 m, 1 m))\n",
    );
    assert_eq!(o.stdout, "[20 m/s]\n[2 m, 4 m, 6 m]\n");
}

#[test]
fn a_gradient_of_parameters_in_different_units_of_one_dimension() {
    // km and m are both lengths; the derivatives are in SI: J/m = N.
    let o = assert_same(
        "def w(a: km, b: m) -> J:\n    return 2 N * a + 3 N * b\n\
         print(grad(w)(1 km, 1 m))\n",
    );
    assert_eq!(o.stdout, "[2 N, 3 N]\n");
}

#[test]
fn a_gradient_through_control_flow_loops_and_recursion() {
    let o = assert_same(
        "def wedge(a: m, b: m) -> m^2:\n    if a > b:\n        return a * a\n    return a * b\n\
         def sum3(a: m, b: m) -> m:\n    total = 0 m\n    i = 0\n    while i < 3:\n        total = total + a * (i + 1) + b\n        i += 1\n    return total\n\
         print(grad(wedge)(3 m, 1 m))\n\
         print(grad(wedge)(1 m, 3 m))\n\
         print(grad(sum3)(1 m, 1 m))\n",
    );
    assert_eq!(o.stdout, "[6 m, 0 m]\n[3 m, 1 m]\n[6, 3]\n");
}

#[test]
fn a_gradient_evaluates_its_arguments_once() {
    let o = assert_same(
        "def loud(v: m) -> m:\n    print(\"arg\")\n    return v\n\
         def f(a: m, b: m) -> m^2:\n    return a * b\n\
         print(grad(f)(loud(1 m), loud(2 m)))\n",
    );
    assert_eq!(o.stdout, "arg\narg\n[2 m, 1 m]\n");
}

#[test]
fn the_function_prints_once_per_parameter_of_a_gradient() {
    // Forward mode runs the function once per derivative.
    let o = assert_same(
        "def f(a: m, b: m) -> m^2:\n    print(\"called\")\n    return a * b\n\
         print(grad(f)(1 m, 2 m))\n",
    );
    assert_eq!(o.stdout, "called\ncalled\n[2 m, 1 m]\n");
}

#[test]
fn a_gradient_that_fails_fails_the_same_way() {
    let o = assert_same(
        "def root(a: m^2, b: m^2) -> m:\n    return sqrt(a) + sqrt(b)\n\
         print(1)\nprint(grad(root)(4 m^2, 0 m^2))\n",
    );
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

#[test]
fn partial_notation_is_the_derivative_in_both_engines() {
    let o = assert_same(&format!(
        "{POSITION}print((∂x/∂t)(2 s))\nprint(derivative(x, t)(2 s))\n\
         def v(t: s) -> m/s:\n    return (∂x/∂t)(t)\n\
         print(((∂v/∂t))(2 s))\n\
         g = 10 m/s^2\ndef fall(t: s, m0: kg) -> J:\n    return 0.5 * m0 * (g * t)^2\n\
         print((∂fall/∂m0)(2 s, 3 kg), (∂fall/∂t)(2 s, 3 kg))\n"
    ));
    assert_eq!(o.stdout, "20 m/s\n20 m/s\n10 m/s^2\n200 m^2/s^2 600 W\n");
}

// ----- gradients of arrays: grad(f)(p) and derivative(f, p)(...) -----
//
// A function of a vector parameter has a gradient with the vector's shape
// and the unit dim(f)/dim(p): the gradient of a potential of a position is a
// force. `sky-ad` computes it (in reverse mode; forward mode, one pass per
// entry, when the function loops). The tree-walker asks `sky-derive` for it
// and lowering emits a call, so both must print the same bytes, and fail
// the same way.

#[test]
fn the_force_of_a_gravitational_potential_points_to_the_origin() {
    // U = -G M m / |p|; F = -∇U has magnitude G M m / r² and points at the
    // centre. G is the predefined constant.
    let o = assert_same(
        "M = 5.972e24 kg\nmass = 1000 kg\n\
         def U(p: Vector[3] m) -> J:\n    return -G * M * mass / sqrt(p @ p)\n\
         p = [7000 km, 0 km, 0 km]\nF = -grad(U)(p)\nprint(F)\nprint(F[0])\n\
         q = [0 m, 3 m, 4 m]\nprint(grad(U)(q))\n",
    );
    assert_eq!(o.error, None);
    assert!(
        o.stdout.starts_with("[-8134.4733877551"),
        "the force at 7000 km, pointing along -x: {}",
        o.stdout
    );
}

#[test]
fn the_gradient_of_a_spring_network_uses_a_matrix() {
    // U = ½ (p - p0)ᵀ K (p - p0): ∇U = K (p - p0).
    let o = assert_same(
        "K = [[300, -100, 0], [-100, 200, -50], [0, -50, 100]] * 1 N/m\n\
         p0 = [0.5 m, 0 m, 0.25 m]\n\
         def U(p: Vector[3] m) -> J:\n    d = p - p0\n    return 0.5 * (d @ (K @ d))\n\
         p = [0.6 m, 0.1 m, 0.2 m]\nprint(U(p))\nprint(grad(U)(p))\nprint(derivative(U, p)(p))\n",
    );
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().count(), 3);
    let lines: Vec<&str> = o.stdout.lines().collect();
    assert_eq!(lines[1], lines[2], "grad(U) is derivative(U, p)");
}

#[test]
fn a_gradient_through_if_index_slice_and_transpose() {
    let o = assert_same(
        "def f(p: Vector[4] m, A: Matrix[2, 2]) -> m:\n\
         \x20   B = Aᵀ @ A\n\
         \x20   s = p[1:3]\n\
         \x20   if p[0] > 0 m:\n\
         \x20       return s[0] * B[0, 1] + p[3] * B[1, 1]\n\
         \x20   return s[1] * B[0, 0]\n\
         A = [[1, 2], [3, 4]]\n\
         print(derivative(f, p)([1 m, 2 m, 3 m, 4 m], A))\n\
         print(derivative(f, p)([-1 m, 2 m, 3 m, 4 m], A))\n\
         print(derivative(f, A)([1 m, 2 m, 3 m, 4 m], A))\n",
    );
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().count(), 3);
}

#[test]
fn a_gradient_through_a_loop_takes_a_pass_per_entry() {
    // Reverse mode has no tape for the loop, so the gradient is forward
    // mode, once per entry. It prints once per pass.
    let o = assert_same(
        "def f(p: Vector[3]) -> m:\n    print(\"called\")\n    acc = 0\n    i = 0\n\
         \x20   while i < 3:\n        acc = acc + p[i] * p[i] * (i + 1)\n        i += 1\n\
         \x20   return 1 m * acc\n\
         print(grad(f)([0.5, 1, 2]))\n",
    );
    assert_eq!(o.stdout, "called\ncalled\ncalled\n[1 m, 4 m, 12 m]\n");
}

#[test]
fn reverse_mode_runs_the_function_once() {
    let o = assert_same(
        "def f(p: Vector[3]) -> m:\n    print(\"called\")\n    return 1 m * (p @ p)\n\
         print(grad(f)([0.5, 1, 2]))\n",
    );
    assert_eq!(o.stdout, "called\n[1 m, 2 m, 4 m]\n");
}

#[test]
fn the_derivative_of_an_array_valued_function_of_a_scalar() {
    // A position vector's derivative with respect to time is a velocity.
    let o = assert_same(
        "def r(t: s) -> Vector[2] m:\n    return [5 m/s^2 * t^2, 2 m/s * t]\n\
         print(derivative(r, t)(2 s))\nprint((∂r/∂t)(1 s))\n",
    );
    assert_eq!(o.stdout, "[20 m/s, 2 m/s]\n[10 m/s, 2 m/s]\n");
}

#[test]
fn helper_functions_of_arrays_are_differentiated_through() {
    let o = assert_same(
        "def norm(v: Vector[3] m) -> m:\n    return sqrt(v @ v)\n\
         def U(p: Vector[3] m) -> J:\n    return 2 N/m * norm(p)^2\n\
         print(grad(U)([1 m, 2 m, 2 m]))\n",
    );
    assert_eq!(o.stdout, "[4 N, 8 N, 8 N]\n");
}

#[test]
fn a_gradient_that_fails_fails_alike_for_arrays() {
    let o = assert_same(
        "def root(p: Vector[2] m^2) -> m:\n    return sqrt(p[0]) + sqrt(p[1])\n\
         print(1)\nprint(grad(root)([4 m^2, 0 m^2]))\n",
    );
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

#[test]
fn a_gradient_of_a_matrix_parameter() {
    let o = assert_same(
        "def f(A: Matrix[2, 2]) -> m:\n    B = inv(A)\n    return 1 m * (B[0, 0] + A[1, 0] * A[1, 1])\n\
         print(grad(f)([[3, 1], [0.5, 2]]))\n",
    );
    assert_eq!(o.error, None);
    assert!(o.stdout.starts_with("[["), "{}", o.stdout);
}

// ----- contracts: assert, requires, ensures -----
//
// The message names the values of the names in the condition, and the span
// is the condition's; both engines build them from `sky_interp::contract`,
// and these tests hold them to it: same stdout, same message, same span.

/// Both engines fail on `src` with the same message at the same span; returns
/// the message. Their output before the failure is the same too.
fn same_failure(src: &str) -> String {
    let (walker, sir) = located_errors(src);
    assert_eq!(sir, walker, "the engines differ on:\n{src}");
    let out = assert_same(src);
    let (message, span, _) = walker.expect("the program fails");
    assert_eq!(out.error.as_deref(), Some(message.as_str()));
    assert!(
        span.start < span.end && (span.end as usize) <= src.len(),
        "{span:?}"
    );
    message
}

#[test]
fn a_true_contract_changes_nothing() {
    let o = assert_same(
        "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    ensures result > 0 s\n    return sqrt(2 * h / g_n)\n\
         E = 3 J\nassert E > 0 J, \"energy\"\nprint(fall_time(20 m), E)\n",
    );
    assert_eq!(o.stdout, "2.0196199771025523 s 3 J\n");
    assert_eq!(o.error, None);
}

#[test]
fn a_false_assertion_fails_alike() {
    let src = "print(1)\nE = -3.2 J\nassert E > 0 J\nprint(2)\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: `E > 0 J` (E = -3.2 J)"
    );
    let src = "E = -3.2 J\nassert E > 0 J, \"energy must be positive\"\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: energy must be positive (`E > 0 J`, E = -3.2 J)"
    );
    let src = "assert 1 > 2\n";
    assert_eq!(same_failure(src), "assertion failed: `1 > 2`");
}

#[test]
fn the_names_shown_are_those_of_the_condition_in_order() {
    // Names not reached by short-circuiting are still shown; a called
    // function's name is not a name; a constant is one.
    let src = "def sq(x):\n    return x * x\na = 3\nb = 4 m\nassert sq(a) + a < 1 and b > 9 m and pi > 0\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: `sq(a) + a < 1 and b > 9 m and pi > 0` (a = 3, b = 4 m, pi = 3.141592653589793)"
    );
    let src =
        "v = [1 m, 2 m]\ns = \"a\"\nflag = true\nassert v[0] > 5 m or s == \"b\" or not flag\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: `v[0] > 5 m or s == \"b\" or not flag` (v = [1 m, 2 m], s = \"a\", flag = true)"
    );
    // A condition over several lines is quoted on one.
    let src = "a = 1\nassert (a >\n    2\n    and\n    true)\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: `(a > 2 and true)` (a = 1)"
    );
}

#[test]
fn a_failed_precondition_fails_alike_and_runs_before_the_body() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    print(\"body\")\n    return sqrt(2 * h / g_n)\n\
               print(fall_time(20 m))\nx = fall_time(-1 m)\nprint(9)\n";
    assert_eq!(
        same_failure(src),
        "precondition of `fall_time` failed: `h >= 0 m` (h = -1 m)"
    );
    let o = assert_same(src);
    assert_eq!(o.stdout, "body\n2.0196199771025523 s\n");
    // Two of them, and the message of the one that fails.
    let src = "def f(x):\n    requires x > 0, \"positive\"\n    requires x < 10\n    return x\nf(3)\nf(20)\n";
    assert_eq!(
        same_failure(src),
        "precondition of `f` failed: `x < 10` (x = 20)"
    );
    let src =
        "def f(x):\n    requires x > 0, \"positive\"\n    requires x < 10\n    return x\nf(-2)\n";
    assert_eq!(
        same_failure(src),
        "precondition of `f` failed: positive (`x > 0`, x = -2)"
    );
}

#[test]
fn a_failed_postcondition_fails_alike_and_sees_the_arguments_as_passed() {
    let src = "def f(x):\n    requires x > 0\n    ensures result > x\n    x = x - 5\n    return x + 1\nf(3)\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `result > x` (result = -1, x = 3)"
    );
    // At every return, whichever it is.
    let src = "def f(x):\n    ensures result >= 0\n    if x > 0:\n        return x\n    return x\nprint(f(1))\nprint(f(-1))\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `result >= 0` (result = -1)"
    );
    let src = "def f(x: m) -> m:\n    ensures result > 0 m\n    i = 0\n    while i < 10:\n        if i == x / 1 m:\n            return x\n        i += 1\n    return 0 m - x\nprint(f(3 m))\nprint(f(-2 m))\nprint(f(20 m))\nprint(f(-1 m))\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `result > 0 m` (result = -20 m)"
    );
}

#[test]
fn a_postcondition_after_a_break_in_a_loop_and_with_no_result() {
    let src = "def f(n):\n    ensures result == 10\n    total = 0\n    for i in range(20):\n        if i == n:\n            break\n        total += 1\n    return total\nprint(f(10))\nprint(f(5))\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `result == 10` (result = 5)"
    );
    let src = "def f(x):\n    ensures x > 0\n    print(x)\nf(1)\nf(-1)\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `x > 0` (x = -1)"
    );
}

#[test]
fn contracts_read_module_names_and_result_is_an_ordinary_name_elsewhere() {
    let src = "limit = 10\ndef f(x):\n    requires x < limit\n    ensures result < limit + 1\n    return x + 5\nprint(f(3))\nf(8)\n";
    assert_eq!(
        same_failure(src),
        "postcondition of `f` failed: `result < limit + 1` (result = 13, limit = 10)"
    );
    let o = assert_same(
        "result = 3\ndef f(x):\n    ensures result == x + 1\n    return x + 1\nassert result == 3\nprint(f(2), result)\n",
    );
    assert_eq!(o.stdout, "3 3\n");
}

#[test]
fn recursion_and_nesting_check_every_level() {
    let src = "def f(n):\n    requires n >= 0\n    if n == 0:\n        return 0\n    return f(n - 1)\nprint(f(3))\nf(-1)\n";
    assert_eq!(
        same_failure(src),
        "precondition of `f` failed: `n >= 0` (n = -1)"
    );
    // A contract that calls a function with a contract.
    let src = "def pos(x):\n    requires x != 0\n    return x > 0\ndef g(x):\n    requires pos(x)\n    return x\nprint(g(2))\nprint(g(-2))\n";
    assert_eq!(
        same_failure(src),
        "precondition of `g` failed: `pos(x)` (x = -2)"
    );
    let src = "def pos(x):\n    requires x != 0\n    return x > 0\ndef g(x):\n    requires pos(x)\n    return x\ng(0)\n";
    assert_eq!(
        same_failure(src),
        "precondition of `pos` failed: `x != 0` (x = 0)"
    );
}

#[test]
fn a_contract_on_a_value_the_checker_cannot_type_is_checked_at_run_time() {
    let def = "def f(a):\n    return a\n";
    let src = format!("{def}assert f(1)\n");
    assert_eq!(
        same_failure(&src),
        "`assert` needs a Bool condition, found a number"
    );
    let src = format!("{def}def g(x):\n    requires f(x)\n    return x\ng(1)\n");
    assert_eq!(
        same_failure(&src),
        "`requires` needs a Bool condition, found a number"
    );
    let src = format!("{def}def g(x):\n    ensures f(result)\n    return x\ng(true)\ng(4)\n");
    assert_eq!(
        same_failure(&src),
        "`ensures` needs a Bool condition, found a number"
    );
    let o = assert_same(&format!(
        "{def}assert f(true)\nassert f(2) == 2\nprint(1)\n"
    ));
    assert_eq!(o.stdout, "1\n");
    let src = format!("{def}x = 5\nassert f(x > 9), \"unknown\"\n");
    assert_eq!(
        same_failure(&src),
        "assertion failed: unknown (`f(x > 9)`, x = 5)"
    );
}

#[test]
fn asserts_in_blocks_after_an_early_return_and_in_loops() {
    let src = "def f(x):\n    if x > 100:\n        return 1\n    assert x > -5, \"floor\"\n    for i in range(3):\n        assert i < x\n    return 2\nprint(f(200))\nprint(f(4))\nprint(f(2))\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: `i < x` (i = 2, x = 2)"
    );
    let src = "def f(x):\n    if x > 100:\n        return 1\n    assert x > -5, \"floor\"\n    return 2\nprint(f(200))\nprint(f(-9))\n";
    assert_eq!(
        same_failure(src),
        "assertion failed: floor (`x > -5`, x = -9)"
    );
    let src = "k = 0\nwhile k < 10:\n    k += 1\n    if k == 4:\n        continue\n    assert k != 7\nprint(k)\n";
    assert_eq!(same_failure(src), "assertion failed: `k != 7` (k = 7)");
}

#[test]
fn an_assertion_is_output_ordered_and_stops_a_failing_function_before_its_effects() {
    let src = "def f(x):\n    print(\"in\")\n    assert x > 0\n    print(\"out\")\n    return x\nprint(f(1))\nprint(f(0))\n";
    assert_eq!(same_failure(src), "assertion failed: `x > 0` (x = 0)");
    let o = assert_same(src);
    assert_eq!(o.stdout, "in\nout\n1\nin\n");
}

#[test]
fn a_derivative_of_a_function_with_a_contract_keeps_the_contract() {
    let o = assert_same(
        "def f(x: m) -> m:\n    requires x > 0 m\n    ensures result > 0 m\n    return x * x / 1 m\n\
         print(derivative(f, x)(3 m))\nprint(f(2 m))\n",
    );
    assert_eq!(o.stdout, "6\n4 m\n");
    let src = "def f(x: m) -> m:\n    requires x > 0 m, \"x must be positive\"\n    return x * x / 1 m\nprint(derivative(f, x)(-1 m))\n";
    assert_eq!(
        same_failure(src),
        "precondition of `f` failed: x must be positive (`x > 0 m`, x = -1 m)"
    );
}

// ----- `norm` and `dot`: one function for both engines -----
//
// SIR runs `sky_interp::vecfn::norm` and `dot`, the functions `sky run`
// calls, with the call's span and the arguments' spans, so results and
// diagnostics are byte-identical.

#[test]
fn norm_and_dot_of_vectors_matrices_and_scalars() {
    let o = assert_same(
        "\
v = [3 m, 4 m]
w = [1 N, 2 N]
A = [[1, 2], [2, 4]]
print(norm(v), norm(-3 m), norm(A), norm([0, 0]))
print(dot(v, w), dot([1, 2, 3], [4, 5, 6]), dot(v, v))
print(norm([1e200, 1e200]), norm([1e-200, 1e-200]))
d = dot(v, w)
print(d > 10 N*m, norm(v) == 5 m)
",
    );
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().next(), Some("5 m 3 m 5 0"), "{}", o.stdout);
}

#[test]
fn norm_and_dot_in_functions_with_symbolic_sizes() {
    let o = assert_same(
        "\
def length(v: Vector[N] m) -> m:
    return norm(v)
def work(f: Vector[N] N, d: Vector[N] m) -> J:
    return dot(f, d)
def unit(v: Vector[N] m):
    return v / norm(v)
print(length([3 m, 4 m, 0 m]), work([1 N, 2 N], [3 m, 4 m]))
print(unit([3 m, 4 m]))
",
    );
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().next(), Some("5 m 11 J"), "{}", o.stdout);
}

#[test]
fn norm_and_dot_fail_alike() {
    // Overflow: the call is the span, as in `sky run`.
    let msg = same_failure("print(norm([1.7e308, 1.7e308]))\n");
    assert_eq!(msg, "arithmetic overflow");
    let msg = same_failure("print(dot([1e200, 1e200], [1e200, 1e200]))\n");
    assert_eq!(msg, "arithmetic overflow");
    // Vectors of different lengths, known only at run time: both operands
    // are labelled with their spans.
    let src = "def f(a, b):\n    return dot(a, b)\nprint(f([1, 2, 3], [1, 2]))\n";
    let msg = same_failure(src);
    assert!(msg.contains("one length"), "{msg}");
    // Operands that the checker could not type and that are not vectors.
    let src = "def f(a):\n    return norm(a)\nprint(f(true))\n";
    let msg = same_failure(src);
    assert!(msg.starts_with("`norm` needs a vector"), "{msg}");
    let src = "def f(a, b):\n    return dot(a, b)\nprint(f([1, 2], 3))\n";
    let msg = same_failure(src);
    assert!(msg.starts_with("`dot` needs two vectors"), "{msg}");
    let src = "def f(a, b):\n    return dot(a, b)\nprint(f([[1, 2], [3, 4]], [[1, 2], [3, 4]]))\n";
    let msg = same_failure(src);
    assert!(msg.starts_with("`dot` needs two vectors"), "{msg}");
    // Output before the failure is the same too.
    let o = assert_same("print(norm([3, 4]))\nprint(norm([1.7e308, 1.7e308]))\nprint(9)\n");
    assert_eq!(o.stdout, "5\n");
}

#[test]
fn a_gradient_of_the_natural_potential_through_norm() {
    let o = assert_same(
        "def U(p: Vector[3] m) -> J:\n    return -2 J*m / norm(p)\n\
         print(grad(U)([1 m, 2 m, 2 m]))\n\
         print(-grad(U)([3 m, 0 m, 4 m]))\n",
    );
    assert_eq!(o.error, None);
    assert!(o.stdout.starts_with("[0.0740"), "{}", o.stdout);
}

#[test]
fn the_gradient_of_norm_at_the_origin_fails_alike() {
    let o = assert_same(
        "def f(p: Vector[2] m) -> m:\n    return norm(p)\n\
         print(1)\nprint(grad(f)([0 m, 0 m]))\n",
    );
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

#[test]
fn the_derivative_through_norm_and_dot_of_a_position_vector() {
    let o = assert_same(
        "def speed2(t: s) -> m^2/s^2:\n    v = [3 m/s^2 * t, 4 m/s]\n    return dot(v, v)\n\
         def speed(t: s) -> m/s:\n    return norm([3 m/s^2 * t, 4 m/s])\n\
         print(derivative(speed2, t)(2 s))\nprint(derivative(speed, t)(0 s))\n",
    );
    assert_eq!(o.error, None);
}

// ----- planned `@` chains: the order, solves instead of inverses -----
//
// The checker plans a chain (`sky_types::chain`) and both engines follow the
// plan, so they must agree bit for bit, and fail with the same diagnostic.

const CHAIN_DATA: &str = "\
A = [[1, 2], [3, 4]]
B = [[0, 1], [1, 0]]
C = [[2, 0], [0, 2]]
v = [1, 2]
";

fn ir_text(src: &str) -> String {
    sky_ir::print(&sky_lower::lower_source(src).expect("lowers"))
}

/// The number of operations named `op` (as `%x = op ...`) in the text.
fn count_ops(text: &str, op: &str) -> usize {
    text.lines()
        .filter(|l| l.contains(&format!("= {op} ")) || l.contains(&format!("= {op}[")))
        .count()
}

#[test]
fn a_reassociated_chain_agrees_and_matches_the_written_order_on_integers() {
    let src = format!("{CHAIN_DATA}print(A @ B @ C @ v)\n");
    let o = assert_same(&src);
    assert_eq!(o.error, None);
    // Integers are exact in either order, so the written order (forced by
    // parentheses) prints the same.
    let forced = format!("{CHAIN_DATA}print(((A @ B) @ C) @ v)\n");
    assert_eq!(assert_same(&forced).stdout, o.stdout);
    // The plan is on the IR: A @ (B @ (C @ v)), three products.
    let text = ir_text(&src);
    assert_eq!(count_ops(&text, "matmul"), 3, "{text}");
    let lines: Vec<&str> = text.lines().filter(|l| l.contains("= matmul ")).collect();
    // The first product multiplies C and v (the innermost of the plan).
    assert!(lines[0].contains("%C, %v"), "{text}");
}

#[test]
fn a_parenthesized_group_is_computed_as_written() {
    let src = format!("{CHAIN_DATA}print(A @ (B @ C) @ v)\nprint((A @ B) @ C @ v)\n");
    let o = assert_same(&src);
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().count(), 2);
}

#[test]
fn inverses_in_a_chain_become_solves_in_both() {
    let data = "A = [[4, 1], [1, 3]]\nB = [[2, 0], [1, 5]]\nb = [1, 2]\nM = [[1, 2], [3, 4]]\n";
    // inv(A) @ inv(B) @ b: two solves and no inverse.
    let src = format!("{data}print(inv(A) @ inv(B) @ b)\n");
    let o = assert_same(&src);
    assert_eq!(o.error, None);
    let text = ir_text(&src);
    assert_eq!(count_ops(&text, "solve"), 2, "{text}");
    assert_eq!(count_ops(&text, "inverse"), 0, "{text}");
    // x @ A⁻¹, a transposed solve.
    let src = format!("{data}print(b @ inv(A))\nprint(M @ inv(A))\n");
    assert_eq!(assert_same(&src).error, None);
    let text = ir_text(&src);
    assert_eq!(count_ops(&text, "solve.right"), 2, "{text}");
    // A⁻¹ @ B with a matrix of right-hand sides, and the inverse of a
    // product.
    let src = format!("{data}print(inv(A) @ M)\nprint(inv(A) @ inv(B))\nprint(inv(A) @ b @ b)\n");
    assert_eq!(assert_same(&src).error, None);
    let text = ir_text(&src);
    assert!(text.contains("solve[lu]"), "{text}");
    // A⁻¹ @ B⁻¹ is (B @ A)⁻¹: a product and one inverse.
    assert_eq!(count_ops(&text, "inverse"), 1, "{text}");
}

#[test]
fn a_diagonal_solve_and_an_orthogonal_transpose_agree() {
    let src = "\
D: Diagonal Matrix[3, 3] = [[2, 0, 0], [0, 4, 0], [0, 0, 8]]
b = [1, 2, 3]
M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
print(inv(D) @ b, b @ inv(D))
print(inv(D) @ M)
Q = identity(3)
print(inv(Q) @ b, Q⁻¹ @ M)
";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    assert_eq!(
        o.stdout,
        "[0.5, 0.5, 0.375] [0.5, 0.5, 0.375]\n[[0.5, 1, 1.5], [1, 1.25, 1.5], [0.875, 1, 1.25]]\n[1, 2, 3] [[1, 2, 3], [4, 5, 6], [7, 8, 10]]\n"
    );
    let text = ir_text(src);
    assert!(text.contains("solve[diagonal]"), "{text}");
    assert!(text.contains("solve.right[diagonal]"), "{text}");
    // The Orthogonal inverse is a transpose; no inverse, no solve for Q.
    assert_eq!(count_ops(&text, "inverse"), 0, "{text}");
    assert!(text.contains("= transpose "), "{text}");
}

#[test]
fn a_diagonal_parameter_is_solved_by_division_in_both() {
    // (The interpreters verify the tag as a promise; a lie cannot be told
    // through a typed call, which lowering rejects: see the `sky_interp::solve`
    // tests for the error.)
    let src = "\
def f(D: Diagonal Matrix[2, 2], b: Vector[2]):
    return inv(D) @ b
D: Diagonal Matrix[2, 2] = [[1, 0], [0, 2]]
print(f(D, [1, 2]))
";
    assert_eq!(assert_same(src).stdout, "[1, 1]\n");
    assert!(ir_text(src).contains("solve[diagonal]"));
}

#[test]
fn a_singular_matrix_in_a_chain_fails_alike_in_both() {
    for src in [
        "A = [[1, 2], [2, 4]]\nb = [1, 2]\nprint(inv(A) @ b)\n",
        "A = [[1, 2], [2, 4]]\nb = [1, 2]\nprint(b @ inv(A))\n",
        "A = [[1, 2], [2, 4]]\nB = [[1, 2], [3, 4]]\nprint(inv(A) @ inv(B))\n",
        "A = [[1, 2], [2, 4]]\nB = [[1, 2], [3, 4]]\nb = [1, 2]\nprint(inv(B) @ inv(A) @ b)\n",
        "A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nM = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\nprint(inv(A) @ M)\n",
        "D: Diagonal Matrix[2, 2] = [[1, 0], [0, 1e-30]]\nprint(inv(D) @ [1, 1])\n",
    ] {
        let (walker, sir) = located_errors(src);
        assert!(walker.is_some(), "{src}");
        assert_eq!(walker, sir, "{src}");
        assert_eq!(walker.unwrap().0, "the matrix is singular", "{src}");
    }
}

#[test]
fn the_error_is_the_inverses_error_word_for_word() {
    // Unplanned (an untyped parameter), the inverse fails; planned (typed),
    // the solve fails: the same message, label and note, on the same operand.
    let planned = "def f(A: Matrix[2, 2], b: Vector[2]):\n    return inv(A) @ b\nprint(f([[1, 2], [2, 4]], [1, 2]))\n";
    let unplanned = "def f(A, b):\n    return inv(A) @ b\nprint(f([[1, 2], [2, 4]], [1, 2]))\n";
    let (p, _) = located_errors(planned);
    let (u, _) = located_errors(unplanned);
    let (p, u) = (p.unwrap(), u.unwrap());
    assert_eq!(p.0, u.0);
    assert!(p.2.contains("this matrix has no inverse"), "{}", p.2);
    assert!(p.2.contains("this is Matrix[2,2]"), "{}", p.2);
}

#[test]
fn shapes_the_checker_cannot_see_still_agree() {
    // Unannotated parameters have no types, so nothing is planned and both
    // engines multiply as written, or fail the same way.
    let src = "def f(A, b):\n    return inv(A) @ b\nprint(f([[2, 0], [0, 4]], [1, 2]))\nprint(f([[1, 2, 3], [4, 5, 6]], [1, 2]))\n";
    let (walker, sir) = located_errors(src);
    assert_eq!(walker, sir);
    assert!(walker.is_some());
}

#[test]
fn a_chain_in_a_function_body_and_in_a_loop() {
    let src = "\
def f(A: Matrix[N, N], B: Matrix[N, N], v: Vector[N]):
    return A @ B @ v
A = [[1, 2], [3, 4]]
B = [[0, 1], [1, 0]]
print(f(A, B, [1, 2]))
print(f(B, A, [3, 4]))
C = [[2, 0], [0, 2]]
x = [1, 1]
i = 0
while i < 3:
    x = A @ B @ C @ x
    i += 1
print(x)
";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    assert_eq!(o.stdout.lines().count(), 3);
    let text = ir_text(src);
    // The function's body has the chain as A @ (B @ v).
    assert!(text.contains("fn @f"), "{text}");
}

#[test]
fn operands_are_evaluated_once_and_left_to_right_whatever_the_plan_does() {
    let src = "\
def a() -> Matrix[2, 2]:
    print(\"a\")
    return [[1, 2], [3, 4]]
def b() -> Matrix[2, 2]:
    print(\"b\")
    return [[0, 1], [1, 0]]
def c() -> Matrix[2, 2]:
    print(\"c\")
    return [[2, 0], [0, 2]]
def d() -> Vector[2]:
    print(\"d\")
    return [1, 2]
def e() -> Matrix[2, 2]:
    print(\"e\")
    return [[4, 1], [1, 3]]
print(a() @ b() @ c() @ d())
print(inv(e()) @ a() @ d())
print(d() @ inv(e()))
print(inv(e()) @ inv(a()) @ d())
";
    let o = assert_same(src);
    assert_eq!(o.error, None);
    let printed: Vec<&str> = o.stdout.lines().filter(|l| l.len() == 1).collect();
    assert_eq!(
        printed.join(""),
        "abcd".to_string() + "ead" + "de" + "ead",
        "{}",
        o.stdout
    );
}

#[test]
fn an_error_in_an_operand_stops_the_chain_before_any_product() {
    let src = "\
def f() -> Matrix[2, 2]:
    print(\"f\")
    return [[1, 2], [3, 4]]
def g() -> Vector[2]:
    print(\"g\")
    return [1 / 0, 1]
A = [[1, 2], [3, 4]]
print(A @ A @ f() @ g())
";
    let o = assert_same(src);
    assert_eq!(o.stdout, "f\ng\n");
    assert_eq!(o.error.as_deref(), Some("division by zero"));
}

// ----- reproducible array arithmetic -----

/// A matrix literal whose entries are cheap, deterministic and not exact
/// sums, so a different order of additions shows in the printed bits.
fn matrix_literal(rows: usize, cols: usize, seed: usize) -> String {
    let row = |i: usize| {
        let entries: Vec<String> = (0..cols)
            .map(|j| {
                let x = ((i * 31 + j * 17 + seed * 7) % 23) as f64 / 7.0 - 1.5;
                format!("{x}")
            })
            .collect();
        format!("[{}]", entries.join(", "))
    };
    let all: Vec<String> = (0..rows).map(row).collect();
    format!("[{}]", all.join(", "))
}

#[test]
fn long_reductions_and_blocked_products_are_the_same_bits_in_both_engines() {
    // Vectors of 9 and 33 entries take the interleaved-sums dot product;
    // the 9-by-70 product and its transpose cross the kernel's block edges.
    let v9 = "[1e16, 1, -1e16, 1, 1, 1, 1, 1, 0.5]";
    let vector = |n: usize, seed: usize| {
        let items: Vec<String> = (0..n)
            .map(|i| format!("{}", ((i * 13 + seed) % 19) as f64 / 3.0 - 2.5))
            .collect();
        format!("[{}]", items.join(", "))
    };
    let src = format!(
        "\
v = {v9}
w = [1, 1, 1, 1, 1, 1, 1, 1, 1]
print(v @ w)
a = {}
b = {}
print(a @ b)
A = {}
B = {}
C = A @ B
print(C)
print(Cᵀ @ C)
",
        vector(33, 1),
        vector(33, 5),
        matrix_literal(9, 70, 1),
        matrix_literal(70, 11, 2),
    );
    let o = assert_same(&src);
    assert_eq!(o.error, None, "{}", o.stdout);
}

#[test]
fn lu_solves_and_gram_products_of_larger_matrices_are_the_same_bits_in_both_engines() {
    // A 40 by 40 system takes the recursive LU (panels of 16 and a GEMM
    // update); `x @ A⁻¹` runs the two transposed triangular solves; the
    // Gram products of a 20 by 40 matrix take the lower-triangle kernel.
    let strong = |n: usize, seed: usize| {
        let row = |i: usize| {
            let entries: Vec<String> = (0..n)
                .map(|j| {
                    let x = ((i * 29 + j * 13 + seed * 5) % 31) as f64 / 9.0 - 1.7;
                    format!("{}", if i == j { x + 12.0 } else { x })
                })
                .collect();
            format!("[{}]", entries.join(", "))
        };
        let all: Vec<String> = (0..n).map(row).collect();
        format!("[{}]", all.join(", "))
    };
    let vector = |n: usize, seed: usize| {
        let items: Vec<String> = (0..n)
            .map(|i| format!("{}", ((i * 7 + seed) % 17) as f64 / 3.0 - 2.5))
            .collect();
        format!("[{}]", items.join(", "))
    };
    let src = format!(
        "\
S = {}
b = {}
print(S⁻¹ @ b)
print(b @ S⁻¹)
B = {}
print(B @ Bᵀ)
print(Bᵀ @ B)
",
        strong(40, 3),
        vector(40, 4),
        matrix_literal(20, 40, 5),
    );
    let o = assert_same(&src);
    assert_eq!(o.error, None, "{}", o.stdout);
    assert_eq!(o.stdout.lines().count(), 4, "{}", o.stdout);
}
