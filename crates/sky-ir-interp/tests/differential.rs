//! Differential test: a Sky program must behave identically through
//! `sky run`'s tree-walking interpreter and through lowering to SIR plus
//! the reference interpreter in this crate. "Identically" means the same
//! stdout bytes and the same runtime error message (or both succeed).
//!
//! The SIR side runs `sky_lower::lower_source`, which also verifies the
//! lowered module, so every program here is checked three ways: same
//! output, same errors, and valid SIR.

use std::collections::HashSet;

use sky_derive::SirDerivatives;
use sky_diag::{Severity, Span};
use sky_interp::{run_module_with_derivatives, RuntimeEnv};

/// What running a program did: its stdout, and the message of the runtime
/// error that stopped it, if any.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    error: Option<String>,
}

/// Lex, parse and check `src`, panicking with the diagnostics if it is not
/// a valid program: a differential test on a program `sky run` refuses
/// would compare nothing. Also returns the `A⁻¹ @ b` sites the checker
/// rewrote to Cholesky solves, which `sky run` honours.
fn parse_checked(src: &str) -> (sky_ast::Module, HashSet<Span>) {
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
    (module, checked.cholesky_solves)
}

/// The `sky run` side: exactly what `sky-cli`'s `run` does after checking.
fn run_tree_walker(src: &str) -> Outcome {
    let (module, solves) = parse_checked(src);
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
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
            error: result.err().map(|e| e.message.clone()),
        }
    })
}

/// The SIR side: lower (lex, parse, check, lower, verify) and run the result
/// in this crate's interpreter. Every program given to [`assert_same`] must
/// lower: a lowering refusal fails the test.
fn run_lowered(src: &str) -> Option<Outcome> {
    let ir =
        sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering failed on:\n{src}\n{e}"));
    Some(run_ir(&ir))
}

/// Runs a lowered module through the reference interpreter into an
/// [`Outcome`]. Used by [`run_lowered`] once lowering exists.
fn run_ir(ir: &sky_ir::Module) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let result = sky_ir_interp::run(ir, &mut out);
        Outcome {
            stdout: String::from_utf8(out).expect("utf-8 output"),
            error: result.err().map(|e| e.message.clone()),
        }
    })
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
    Outcome {
        error: o.error.map(|e| {
            if is_limit(&e) {
                "<resource limit>".to_string()
            } else {
                e
            }
        }),
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
    assert_eq!(text.matches("inverse").count(), 1, "{text}");
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

/// A run-time error's message and span, if the run failed.
type Located = Option<(String, Span)>;

/// The error of the tree-walker and of the SIR interpreter on `src`, with
/// the span: they must be the same, including where they point.
fn located_errors(src: &str) -> (Located, Located) {
    let (module, solves) = parse_checked(src);
    let walker = sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        let mut derivatives = SirDerivatives::new(&module, src);
        run_module_with_derivatives(&module, src, &mut env, &mut out, &solves, &mut derivatives)
            .err()
            .map(|e| (e.message.clone(), e.span))
    });
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering failed: {e}"));
    let sir = sky_ast::with_deep_stack(|| {
        sky_ir_interp::run(&ir, &mut Vec::new())
            .err()
            .map(|e| (e.message.clone(), e.span))
    });
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
        let (msg, span) = walker.clone().expect("sky run fails");
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
