//! Lowering contracts: `assert`, `requires` and `ensures` become
//! `assert[kind]` operations with the condition's text, the author's
//! message, the function's name and the names whose values a failure shows
//! (spec "Contracts").

mod common;

use common::*;
use sky_ir::{AssertKind, Function, OpData, OpKind, Type};

/// The assert operations of `f`, in the order of its operations (regions
/// included), with the mnemonics of the operations around them.
fn asserts(f: &Function) -> Vec<&OpData> {
    f.ops
        .iter()
        .filter(|o| matches!(o.kind, OpKind::Assert { .. }))
        .collect()
}

fn parts(o: &OpData) -> (AssertKind, &str, Option<&str>, Option<&str>, Vec<&str>) {
    let OpKind::Assert {
        kind,
        text,
        message,
        func,
        names,
    } = &o.kind
    else {
        panic!("not an assert")
    };
    (
        *kind,
        text,
        message.as_deref(),
        func.as_deref(),
        names.iter().map(String::as_str).collect(),
    )
}

/// The mnemonics of the entry block of `f`, in order.
fn entry_ops(f: &Function) -> Vec<&'static str> {
    f.block(f.entry())
        .ops
        .iter()
        .map(|o| f.op(*o).kind.mnemonic())
        .collect()
}

#[test]
fn an_assertion_is_an_op_with_its_text_message_and_names() {
    let m = lower_ok("E = 3 J\nassert E > 0 J, \"energy must be positive\"\n");
    let main = main_fn(&m);
    let [a] = asserts(main)[..] else {
        panic!("one assert")
    };
    assert_eq!(
        parts(a),
        (
            AssertKind::Assert,
            "E > 0 J",
            Some("energy must be positive"),
            None,
            vec!["E"]
        )
    );
    // The condition, then the value of each name.
    assert_eq!(a.operands.len(), 2);
    assert!(matches!(main.ty(a.operands[0]), Type::Bool));
    assert!(a.results.is_empty());
    // Its span is the condition's.
    let src = "E = 3 J\nassert E > 0 J, \"energy must be positive\"\n";
    assert_eq!(&src[a.span.range()], "E > 0 J");
}

#[test]
fn a_requires_is_the_first_thing_a_function_does() {
    let m =
        lower_ok("def f(x: m) -> m:\n    requires x > 0 m\n    y = x * 2\n    return y\nf(1 m)\n");
    let f = func(&m, "f");
    let ops = entry_ops(f);
    let at = ops.iter().position(|o| *o == "assert").expect("an assert");
    // Only the condition's own operations come before it.
    assert_eq!(&ops[..=at], ["const.float", "cmp.gt", "assert"], "{ops:?}");
    let [a] = asserts(f)[..] else { panic!() };
    assert_eq!(
        parts(a),
        (AssertKind::Requires, "x > 0 m", None, Some("f"), vec!["x"])
    );
}

#[test]
fn an_ensures_runs_just_before_the_return_with_result_and_the_arguments_as_passed() {
    let m = lower_ok(
        "def f(x: m) -> m:\n    ensures result > x\n    x = x - 5 m\n    return x + 1 m\nf(1 m)\n",
    );
    let f = func(&m, "f");
    let ops = entry_ops(f);
    let n = ops.len();
    assert_eq!(&ops[n - 3..], ["cmp.gt", "assert", "return"], "{ops:?}");
    let [a] = asserts(f)[..] else { panic!() };
    assert_eq!(
        parts(a),
        (
            AssertKind::Ensures,
            "result > x",
            None,
            Some("f"),
            vec!["result", "x"]
        )
    );
    // `result` is the value returned; `x` is the parameter, not `x - 5 m`.
    let ret = f.op(*f.block(f.entry()).ops.last().unwrap());
    assert_eq!(a.operands[1], ret.operands[0]);
    assert_eq!(a.operands[2], f.param_values()[0]);
}

#[test]
fn an_ensures_holds_at_every_return_because_there_is_one_exit() {
    let m = lower_ok(
        "def f(x):\n    ensures result >= 0\n    if x > 0:\n        return x\n    while x < 0:\n        x += 1\n        if x == -3:\n            return 0\n    return x\nf(1)\n",
    );
    let f = func(&m, "f");
    assert_eq!(asserts(f).len(), 1);
    let ops = entry_ops(f);
    let n = ops.len();
    assert_eq!(ops[n - 2], "assert");
    assert_eq!(ops[n - 1], "return");
}

#[test]
fn a_function_without_a_value_has_no_result() {
    let m = lower_ok("def f(x):\n    ensures x > 0\n    print(x)\nf(1)\n");
    let f = func(&m, "f");
    let [a] = asserts(f)[..] else { panic!() };
    assert_eq!(
        parts(a),
        (AssertKind::Ensures, "x > 0", None, Some("f"), vec!["x"])
    );
}

#[test]
fn constants_and_module_names_are_shown_but_function_names_are_not() {
    let m = lower_ok(
        "k = 2\ndef sq(x):\n    return x * x\ndef f(x):\n    requires sq(x) < k * pi\n    return x\nf(1)\n",
    );
    let f = func(&m, "f");
    let [a] = asserts(f)[..] else { panic!() };
    // `k` is a module name the function takes as a trailing parameter; `pi`
    // is a constant, a literal.
    assert_eq!(
        parts(a),
        (
            AssertKind::Requires,
            "sq(x) < k * pi",
            None,
            Some("f"),
            vec!["x", "k", "pi"]
        )
    );
}

#[test]
fn a_redefined_function_names_itself_by_its_source_name() {
    let m = lower_ok(
        "def f(x):\n    requires x > 0\n    return x\nf(1)\ndef f(x):\n    requires x > 1\n    return x\nf(2)\n",
    );
    for name in ["f", "f#2"] {
        let [a] = asserts(func(&m, name))[..] else {
            panic!()
        };
        assert_eq!(parts(a).3, Some("f"), "{name}");
    }
}

#[test]
fn a_condition_the_checker_cannot_type_is_lowered_as_it_is() {
    let m = lower_ok("def u(a):\n    return a\nassert u(true)\n");
    let main = main_fn(&m);
    let [a] = asserts(main)[..] else { panic!() };
    assert!(matches!(main.ty(a.operands[0]), Type::Unknown));
}

#[test]
fn an_and_condition_short_circuits_inside_the_operation_s_operand() {
    let m = lower_ok("x = 0\nassert x == 0 or 1 / x > 0\nassert x == 0 and 1 / x > 0\n");
    let main = main_fn(&m);
    assert_eq!(asserts(main).len(), 2);
    assert!(main.ops.iter().any(|o| matches!(o.kind, OpKind::If)));
}

#[test]
fn a_derivative_function_keeps_the_contracts() {
    let m = lower_ok(
        "def f(x: m) -> m:\n    requires x > 0 m\n    ensures result > 0 m\n    return x * x / 1 m\nprint(derivative(f, x)(3 m))\n",
    );
    let d = func(&m, "df/dx");
    let kinds: Vec<AssertKind> = asserts(d).iter().map(|o| parts(o).0).collect();
    assert_eq!(kinds, [AssertKind::Requires, AssertKind::Ensures]);
}

#[test]
fn effects_include_fail() {
    let m = lower_ok("def f(x):\n    requires x > 0\n    return x\nf(1)\n");
    assert!(func(&m, "f").effects.contains(sky_ir::Effect::Fail));
}
