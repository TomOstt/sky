//! The `contracts` pass: what it discharges, what it must keep, and that
//! dropping a check never changes what a program does.
//!
//! Programs come from Sky source (`sky_lower::lower_source`), so the checks
//! are the ones a user writes; behaviour is compared with the reference
//! interpreter, before and after the whole default pipeline.

use proptest::prelude::*;
use sky_ir::Module;

use sky_ir::PassManager;

use crate::default_passes_without_specialize;

fn lower(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("lowering failed on:\n{src}\n{e}"))
}

/// The module after the default pipeline, and the rule ids it explained.
fn optimized(src: &str) -> (Module, Vec<String>) {
    let mut m = lower(src);
    // Without `specialize`, which would give each constant call a function of
    // its own (see `specialize_tests` for the two together): these tests are
    // about what `contracts` proves from the calls of a function.
    let report = default_passes_without_specialize(PassManager::new())
        .verify_each(true)
        .run(&mut m)
        .expect("the pipeline runs");
    let rules = report
        .explanations
        .iter()
        .map(|x| x.rule.to_string())
        .collect();
    (m, rules)
}

fn asserts(m: &Module) -> usize {
    sky_ir::print(m).matches("assert[").count()
}

fn contract_rules(rules: &[String]) -> Vec<&str> {
    rules
        .iter()
        .map(String::as_str)
        .filter(|r| r.starts_with("CONTRACT-"))
        .collect()
}

/// The outcome of a run: stdout and the failure, with its span.
type Outcome = (String, Option<(String, sky_diag::Span)>);

fn run(m: &Module) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        (
            String::from_utf8(out).expect("utf-8"),
            r.err().map(|e| (e.message.clone(), e.span)),
        )
    })
}

#[test]
fn a_precondition_true_at_every_call_is_dropped() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\n\
               print(fall_time(20 m), fall_time(5 m))\n";
    let (m, rules) = optimized(src);
    assert_eq!(asserts(&m), 0);
    assert_eq!(contract_rules(&rules), ["CONTRACT-002"]);
    assert_eq!(run(&m), run(&lower(src)));
}

#[test]
fn one_call_that_may_violate_keeps_the_check() {
    // The hull of the arguments (-1 m and 20 m) includes a violation.
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\n\
               print(fall_time(20 m))\nprint(fall_time(-1 m))\n";
    let (m, rules) = optimized(src);
    assert_eq!(asserts(&m), 1);
    assert!(contract_rules(&rules).is_empty(), "{rules:?}");
    let (out, err) = run(&m);
    assert_eq!(out, "2.0196199771025523 s\n");
    assert_eq!(
        err.expect("fails").0,
        "precondition of `fall_time` failed: `h >= 0 m` (h = -1 m)"
    );
}

#[test]
fn an_argument_the_analysis_knows_nothing_about_keeps_the_check() {
    // A call result, a parameter and an array element are unknown.
    for arg in ["g(u)", "u", "v[0]"] {
        let src = format!(
            "def g(a):\n    return a\ndef f(x: m):\n    requires x >= 0 m\n    return x\n\
             u = 3 m\nv = [1 m, 2 m]\nprint(f({arg}))\n"
        );
        let (m, rules) = optimized(&src);
        let dropped = contract_rules(&rules).len();
        // `u` is a constant the folder knows; the others are not.
        assert_eq!(dropped == 1, arg == "u", "{arg}:\n{}", sky_ir::print(&m));
    }
}

#[test]
fn recursion_needs_no_fixed_point_and_keeps_its_check() {
    // `f(n - 1)` passes any finite number as far as the analysis goes.
    let src = "def f(n):\n    requires n >= 0\n    if n == 0:\n        return 0\n    return f(n - 1)\nprint(f(3))\n";
    let (m, _) = optimized(src);
    assert_eq!(asserts(&m), 1);
    assert_eq!(run(&m), run(&lower(src)));
    // A recursive function whose argument does not depend on its parameter
    // gets the hull of both calls.
    let src = "def f(n: m):\n    requires n >= 0 m\n    if n > 100 m:\n        return f(1 m)\n    return n\nprint(f(3 m))\n";
    let (m, rules) = optimized(src);
    assert_eq!(asserts(&m), 0);
    assert_eq!(contract_rules(&rules), ["CONTRACT-002"]);
}

#[test]
fn a_function_nobody_calls_keeps_its_checks() {
    let src = "def f(x: m) -> m:\n    requires x >= 0 m\n    ensures result >= 0 m\n    return x\nprint(1)\n";
    let (m, _) = optimized(src);
    assert_eq!(asserts(&m), 2);
}

#[test]
fn a_postcondition_follows_from_the_arguments_through_the_body() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    ensures result > 0 s\n    return sqrt(2 * h / g_n)\n\
               print(fall_time(20 m), fall_time(5 m))\n";
    let (m, rules) = optimized(src);
    assert_eq!(asserts(&m), 0);
    assert_eq!(contract_rules(&rules), ["CONTRACT-002", "CONTRACT-003"]);
    // With a zero height the result may be zero: the check stays.
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    ensures result > 0 s\n    return sqrt(2 * h / g_n)\n\
               print(fall_time(20 m), fall_time(0 m))\n";
    let (m, rules) = optimized(src);
    assert_eq!(asserts(&m), 1);
    assert_eq!(contract_rules(&rules), ["CONTRACT-002"]);
    let (_, err) = run(&m);
    assert_eq!(
        err.expect("fails").0,
        "postcondition of `fall_time` failed: `result > 0 s` (result = 0 s)"
    );
}

#[test]
fn asserts_on_constants_are_dropped_when_true_and_kept_when_not() {
    let (m, rules) = optimized("assert 2 m > 1 m\nassert 1 > 2\nassert 3 == 3\nprint(1)\n");
    assert_eq!(asserts(&m), 1, "{}", sky_ir::print(&m));
    assert_eq!(contract_rules(&rules), ["CONTRACT-001", "CONTRACT-001"]);
    assert_eq!(run(&m).1.expect("fails").0, "assertion failed: `1 > 2`");
}

#[test]
fn and_or_and_not_chains_are_decided() {
    let (m, _) = optimized(
        "x = 3 m\nassert x > 1 m and x < 5 m and not x == 4 m\nassert x < 0 m or x > 2 m\nprint(x)\n",
    );
    assert_eq!(asserts(&m), 0, "{}", sky_ir::print(&m));
    let (m, _) =
        optimized("x = 3 m\nassert x > 1 m and x > 5 m\nassert x < 0 m or x > 4 m\nprint(x)\n");
    assert_eq!(asserts(&m), 2);
}

#[test]
fn a_condition_over_values_of_unknown_type_is_never_dropped() {
    let src = "def f(a):\n    assert a\n    return a\nprint(f(true))\n";
    let (m, _) = optimized(src);
    assert_eq!(asserts(&m), 1);
    let src = "def f(a):\n    requires a > 0\n    return a\nprint(f(3))\n";
    let (m, _) = optimized(src);
    assert_eq!(asserts(&m), 1);
}

#[test]
fn a_dropped_check_leaves_the_failures_of_its_condition_alone() {
    // The condition divides by zero: dropping the assert (true if it ran)
    // must not hide that.
    let src = "x = 0\nassert 1 / x > 0 or true\nprint(1)\n";
    let before = run(&lower(src));
    let (m, _) = optimized(src);
    assert_eq!(run(&m), before);
    assert_eq!(before.1.expect("fails").0, "division by zero");
}

#[test]
fn explanations_say_what_was_known() {
    let src = "def f(h: m) -> m:\n    requires h >= 0 m\n    return h\nprint(f(2 m), f(7 m))\n";
    let mut m = lower(src);
    let report = default_passes_without_specialize(PassManager::new())
        .run(&mut m)
        .expect("runs");
    let x = report
        .explanations
        .iter()
        .find(|x| x.rule == "CONTRACT-002")
        .expect("explained");
    assert_eq!(x.pass, "contracts");
    assert_eq!(x.before, "h >= 0 m");
    assert!(x.after.is_empty());
    let because = x.because.join("\n");
    assert!(
        because.contains("in every call of `f` in this program, `h` is between 2 and 7"),
        "{because}"
    );
    assert!(because.contains("cannot be false"), "{because}");
    assert_eq!(
        crate::contract_headline(x.rule, &x.before).as_deref(),
        Some("[CONTRACT-002] this `requires` always holds, so its check is removed: `h >= 0 m`")
    );
}

/// A comparison operator's source spelling, by index.
const OPS: [&str; 6] = ["<", "<=", ">", ">=", "==", "!="];
const VALUES: [i32; 7] = [-3, -1, 0, 1, 2, 5, 40];

fn call_list() -> impl Strategy<Value = Vec<i32>> {
    prop::collection::vec(prop::sample::select(VALUES.to_vec()), 1..4)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Whatever the contracts and the calls, dropping the checks the pass
    /// proves changes nothing: same output, same failure (message and span).
    #[test]
    fn dropping_proven_contracts_never_changes_a_run(
        pre in (0usize..6, prop::sample::select(VALUES.to_vec())),
        post in (0usize..6, prop::sample::select(VALUES.to_vec())),
        scale in prop::sample::select(vec![-2i32, 1, 3]),
        shift in prop::sample::select(VALUES.to_vec()),
        calls in call_list(),
        assert_op in 0usize..6,
        assert_at in prop::sample::select(VALUES.to_vec()),
    ) {
        let mut src = format!(
            "def f(x: m) -> m:\n    requires x {} {} m\n    ensures result {} {} m\n    return x * {scale} + {shift} m\n",
            OPS[pre.0], pre.1, OPS[post.0], post.1,
        );
        for c in &calls {
            src.push_str(&format!("print(f({c} m))\n"));
        }
        src.push_str(&format!("assert {} m {} {assert_at} m\n", calls[0], OPS[assert_op]));
        let before = lower(&src);
        let (after, _) = optimized(&src);
        prop_assert_eq!(run(&after), run(&before), "on:\n{}", src);
        prop_assert!(asserts(&after) <= asserts(&before));
    }
}
