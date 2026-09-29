//! Differential test: a Sky program must behave identically through
//! `sky run`'s tree-walking interpreter and through lowering to SIR plus
//! the reference interpreter in this crate. "Identically" means the same
//! stdout bytes and the same runtime error message (or both succeed).
//!
//! The lowering pass is being written on another branch. Until it lands,
//! [`run_lowered`] is a marked stub, [`assert_same`] checks only that the
//! tree-walking side runs, and the tests below pin what that side does so
//! wiring up the other side is a one-line change (see `run_lowered`).

use sky_diag::Severity;
use sky_interp::{run_module, RuntimeEnv};

/// What running a program did: its stdout, and the message of the runtime
/// error that stopped it, if any.
#[derive(Debug, PartialEq)]
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

/// The SIR side: lower the checked module, run the result in this crate's
/// interpreter.
///
/// **STUB.** Lowering (`sky-lower`, another branch) does not exist here, so
/// there is nothing to compare against yet. After merging, replace the body
/// with:
///
/// ```ignore
/// let module = parse_checked(src);
/// let ir = sky_lower::lower(&module, src);
/// Some(run_ir(&ir))
/// ```
///
/// with `run_ir` below, and delete the `None`.
fn run_lowered(_src: &str) -> Option<Outcome> {
    None
}

/// Runs a lowered module through the reference interpreter into an
/// [`Outcome`]. Used by [`run_lowered`] once lowering exists.
#[allow(dead_code)] // wired up when lowering lands
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

/// Asserts that `src` behaves the same under both interpreters. Returns the
/// (shared) outcome so a test can also pin it.
fn assert_same(src: &str) -> Outcome {
    let expected = run_tree_walker(src);
    if let Some(actual) = run_lowered(src) {
        assert_eq!(
            actual, expected,
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
