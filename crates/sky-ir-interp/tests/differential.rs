//! Differential test: a Sky program must behave identically through
//! `sky run`'s tree-walking interpreter and through lowering to SIR plus
//! the reference interpreter in this crate. "Identically" means the same
//! stdout bytes and the same runtime error message (or both succeed).
//!
//! The SIR side runs `sky_lower::lower_source`, which also verifies the
//! lowered module, so every program here is checked three ways: same
//! output, same errors, and valid SIR.

use sky_diag::Severity;
use sky_interp::{run_module, RuntimeEnv};

/// What running a program did: its stdout, and the message of the runtime
/// error that stopped it, if any.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    error: Option<String>,
}

/// Lex, parse and check `src`, panicking with the diagnostics if it is not
/// a valid program: a differential test on a program `sky run` refuses
/// would compare nothing.
fn parse_checked(src: &str) -> sky_ast::Module {
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
    module
}

/// The `sky run` side: exactly what `sky-cli`'s `run` does after checking.
fn run_tree_walker(src: &str) -> Outcome {
    let module = parse_checked(src);
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        let result = run_module(&module, src, &mut env, &mut out);
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
