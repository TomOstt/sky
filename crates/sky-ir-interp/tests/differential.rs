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
