//! `assert`, `requires` and `ensures` in the tree-walker (spec "Contracts"):
//! what stops the run, the message, the span, and what the contracts of a
//! function see.

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::{run_module, RuntimeEnv, RuntimeError};

fn run(src: &str) -> (String, Result<(), RuntimeError>) {
    let lexed = lex(src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "parse errors: {errors:?}");
    let mut out = Vec::new();
    let r = run_module(&module, src, &mut RuntimeEnv::with_prelude(), &mut out).map(|_| ());
    (String::from_utf8(out).expect("utf-8"), r)
}

fn fails(src: &str) -> RuntimeError {
    let (_, r) = run(src);
    r.expect_err("the run fails")
}

fn text(src: &str, e: &RuntimeError) -> String {
    src[e.span.range()].to_string()
}

#[test]
fn a_true_assertion_does_nothing() {
    let (out, r) = run("E = 3 J\nassert E > 0 J\nassert true\nprint(1)\n");
    r.expect("runs");
    assert_eq!(out, "1\n");
}

#[test]
fn a_false_assertion_stops_the_run_and_names_the_values() {
    let src = "print(1)\nE = -3.2 J\nassert E > 0 J\nprint(2)\n";
    let (out, r) = run(src);
    let e = r.expect_err("fails");
    assert_eq!(out, "1\n", "output before the failure stays");
    assert_eq!(e.message, "assertion failed: `E > 0 J` (E = -3.2 J)");
    assert_eq!(text(src, &e), "E > 0 J");
    assert_eq!(e.label.as_deref(), Some("this is false"));
}

#[test]
fn the_message_leads_when_there_is_one() {
    let src = "E = -3.2 J\nassert E > 0 J, \"energy must be positive\"\n";
    assert_eq!(
        fails(src).message,
        "assertion failed: energy must be positive (`E > 0 J`, E = -3.2 J)"
    );
    // No names to show.
    assert_eq!(
        fails("assert 1 > 2, \"never\"\n").message,
        "assertion failed: never (`1 > 2`)"
    );
    assert_eq!(fails("assert 1 > 2\n").message, "assertion failed: `1 > 2`");
}

#[test]
fn every_name_is_shown_once_in_order_and_functions_are_not_names() {
    let src = "def sq(x):\n    return x * x\na = 3\nb = 4 m\nassert sq(a) + a < 1 and b > 9 m and a > 0\n";
    assert_eq!(
        fails(src).message,
        "assertion failed: `sq(a) + a < 1 and b > 9 m and a > 0` (a = 3, b = 4 m)"
    );
    let src = "b = 4 m\nassert b > 9 m and b < 10 m\n";
    assert_eq!(
        fails(src).message,
        "assertion failed: `b > 9 m and b < 10 m` (b = 4 m)"
    );
    // Constants are names too.
    assert_eq!(
        fails("assert pi > 4\n").message,
        "assertion failed: `pi > 4` (pi = 3.141592653589793)"
    );
}

#[test]
fn values_print_as_sky_source_and_are_capped() {
    let e = fails("v = [1 m, 2 m]\ns = \"a\"\nassert v[0] > 5 m or s == \"b\"\n");
    assert_eq!(
        e.message,
        "assertion failed: `v[0] > 5 m or s == \"b\"` (v = [1 m, 2 m], s = \"a\")"
    );
    let big = (0..300)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    // A long array is named by its shape and unit, not written out.
    let e = fails(&format!("v = [{big}]\nassert v[0] > 5\n"));
    assert_eq!(e.message, "assertion failed: `v[0] > 5` (v = Vector[300])");
    let e = fails("m = 2 m * identity(5)\nassert m[0, 0] > 5 m\n");
    assert_eq!(
        e.message,
        "assertion failed: `m[0, 0] > 5 m` (m = Matrix[5, 5] m)"
    );
    let e = fails("m = 2 m * identity(4)\nassert m[0, 0] > 5 m\n");
    assert!(
        e.message.contains("m = [[2 m, 0 m, 0 m, 0 m]"),
        "{}",
        e.message
    );
}

#[test]
fn the_condition_text_is_one_line() {
    let e = fails("a = 1\nassert (a >\n    2\n    and\n    true)\n");
    assert_eq!(e.message, "assertion failed: `(a > 2 and true)` (a = 1)");
}

#[test]
fn a_condition_that_is_not_a_bool_is_an_error() {
    let e = fails("def f(x):\n    assert x\nf(3)\n");
    assert_eq!(e.message, "`assert` needs a Bool condition, found a number");
}

#[test]
fn a_precondition_is_checked_at_the_call_and_points_at_it() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\nprint(fall_time(20 m))\nx = fall_time(-1 m)\nprint(2)\n";
    let (out, r) = run(src);
    let e = r.expect_err("fails");
    assert_eq!(out, "2.0196199771025523 s\n");
    assert_eq!(
        e.message,
        "precondition of `fall_time` failed: `h >= 0 m` (h = -1 m)"
    );
    assert_eq!(text(src, &e), "h >= 0 m");
    let [(call, label)] = &e.secondary[..] else {
        panic!("one secondary label: {:?}", e.secondary)
    };
    assert_eq!(&src[call.range()], "fall_time(-1 m)");
    assert_eq!(label, "called here");
}

#[test]
fn preconditions_run_in_order_before_the_body() {
    let src = "def f(x):\n    requires x > 0, \"positive\"\n    requires x < 10\n    print(\"body\")\n    return x\nf(20)\n";
    let (out, r) = run(src);
    assert_eq!(out, "", "the body never ran");
    assert_eq!(
        r.expect_err("fails").message,
        "precondition of `f` failed: `x < 10` (x = 20)"
    );
    let (_, r) = run(
        "def f(x):\n    requires x > 0, \"positive\"\n    requires x < 10\n    return x\nf(-2)\n",
    );
    assert_eq!(
        r.expect_err("fails").message,
        "precondition of `f` failed: positive (`x > 0`, x = -2)"
    );
}

#[test]
fn a_postcondition_sees_result_and_the_arguments_as_passed() {
    let src = "def f(x):\n    requires x > 0\n    ensures result > x\n    x = x - 5\n    return x + 1\nf(3)\n";
    // `result` is -1, and `x` is the 3 that was passed, not the 0 the body
    // made of it.
    assert_eq!(
        fails(src).message,
        "postcondition of `f` failed: `result > x` (result = -1, x = 3)"
    );
}

#[test]
fn a_postcondition_holds_at_every_return() {
    let src = "def f(x):\n    ensures result >= 0\n    if x > 0:\n        return x\n    return x\nprint(f(1))\nprint(f(-1))\n";
    let (out, r) = run(src);
    assert_eq!(out, "1\n");
    assert_eq!(
        r.expect_err("fails").message,
        "postcondition of `f` failed: `result >= 0` (result = -1)"
    );
}

#[test]
fn a_postcondition_of_a_function_that_returns_nothing_sees_the_arguments() {
    let src = "def f(x):\n    ensures x > 0\n    print(x)\nf(1)\nf(-1)\n";
    let (out, r) = run(src);
    assert_eq!(out, "1\n-1\n");
    assert_eq!(
        r.expect_err("fails").message,
        "postcondition of `f` failed: `x > 0` (x = -1)"
    );
}

#[test]
fn a_failing_body_is_reported_before_its_postcondition() {
    let e = fails("def f(x):\n    ensures result > 0\n    return 1 / x\nf(0)\n");
    assert_eq!(e.message, "division by zero");
}

#[test]
fn contracts_read_module_names() {
    let src = "limit = 10\ndef f(x):\n    requires x < limit\n    ensures result < limit + 1\n    return x + 5\nprint(f(3))\nf(8)\n";
    let (out, r) = run(src);
    assert_eq!(out, "8\n");
    assert_eq!(
        r.expect_err("fails").message,
        "postcondition of `f` failed: `result < limit + 1` (result = 13, limit = 10)"
    );
}

#[test]
fn result_is_a_plain_name_outside_ensures() {
    let (out, r) = run("result = 3\ndef f(x):\n    ensures result == x + 1\n    return x + 1\nassert result == 3\nprint(f(2), result)\n");
    r.expect("runs");
    assert_eq!(out, "3 3\n");
}

#[test]
fn recursion_checks_every_level() {
    let src = "def f(n):\n    requires n >= 0\n    if n == 0:\n        return 0\n    return f(n - 1)\nf(3)\nf(-1)\n";
    assert_eq!(
        fails(src).message,
        "precondition of `f` failed: `n >= 0` (n = -1)"
    );
}

#[test]
fn the_message_is_capped_and_escaped() {
    let long = "x".repeat(500);
    let e = fails(&format!("assert false, \"{long}\"\n"));
    assert!(e.message.len() < 400, "{}", e.message.len());
    let e = fails("assert false, \"a\\nb\\0c\"\n");
    assert!(!e.message.contains('\n'), "{:?}", e.message);
    assert!(!e.message.contains('\0'), "{:?}", e.message);
}

// ----- the checker's constant contracts agree with the interpreter -----

/// A deterministic stream of choices (no new dependency for a property test).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

/// A length-valued constant expression over literals.
fn length(r: &mut Lcg, depth: usize) -> String {
    let lit = |r: &mut Lcg| {
        let n = ["0", "1", "2", "3.5", "0.1", "0.2", "100", "1e-3"][r.below(8)];
        let u = ["m", "km", "cm", "mm"][r.below(4)];
        format!("{n} {u}")
    };
    if depth == 0 {
        return lit(r);
    }
    match r.below(5) {
        0 => lit(r),
        1 => format!("({} + {})", length(r, depth - 1), length(r, depth - 1)),
        2 => format!("({} - {})", length(r, depth - 1), length(r, depth - 1)),
        3 => format!("({} * 3)", length(r, depth - 1)),
        _ => format!("-{}", lit(r)),
    }
}

/// A Bool-valued constant expression: comparisons of lengths, joined.
fn boolean(r: &mut Lcg, depth: usize) -> String {
    let cmp = |r: &mut Lcg| {
        let op = ["<", "<=", ">", ">=", "==", "!="][r.below(6)];
        format!("{} {op} {}", length(r, 2), length(r, 2))
    };
    if depth == 0 {
        return cmp(r);
    }
    match r.below(4) {
        0 => cmp(r),
        1 => format!("({} and {})", boolean(r, depth - 1), boolean(r, depth - 1)),
        2 => format!("({} or {})", boolean(r, depth - 1), boolean(r, depth - 1)),
        _ => format!("not ({})", boolean(r, depth - 1)),
    }
}

/// The checker says a constant condition always fails exactly when the
/// interpreter's assertion fails: its arithmetic is the interpreter's.
#[test]
fn constant_contracts_are_decided_as_the_interpreter_decides_them() {
    let mut r = Lcg(2026);
    let (mut warned, mut passed) = (0, 0);
    for _ in 0..600 {
        let cond = boolean(&mut r, 2);
        let src = format!("assert {cond}\n");
        let lexed = lex(&src).expect("lexes");
        let (module, errors) = parse_module(&lexed.tokens, &src);
        assert!(errors.is_empty(), "{src}: {errors:?}");
        let checked = sky_types::check_module(&module, &src);
        assert!(!checked.has_errors(), "{src}: {:?}", checked.diagnostics);
        let says_false = checked
            .diagnostics
            .iter()
            .any(|d| d.message.starts_with("this `assert` always fails"));
        let mut out = Vec::new();
        let ran = run_module(&module, &src, &mut RuntimeEnv::with_prelude(), &mut out);
        match ran {
            Ok(_) => {
                assert!(
                    !says_false,
                    "the checker warns but the assertion holds: {src}"
                );
                passed += 1;
            }
            Err(e) => {
                assert!(
                    e.message.starts_with("assertion failed"),
                    "{src}: {}",
                    e.message
                );
                // The checker may give up (`None`), but when it decides, it is right;
                // for these expressions (no division, no powers) it always decides.
                assert!(
                    says_false,
                    "the assertion fails but the checker is silent: {src}"
                );
                warned += 1;
            }
        }
    }
    assert!(warned > 50 && passed > 50, "{warned} {passed}");
}
