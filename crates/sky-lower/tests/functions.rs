//! Functions: signatures, calls, the module-level names a function reads
//! (extra trailing parameters), recursion, nesting and effects.

mod common;

use common::*;
use sky_ir::{Effect, OpKind, Type};
use sky_units::Dimension;

#[test]
fn module_layout_main_first_then_defs_in_source_order() {
    let m = lower_ok("def a():\n    pass\ndef b():\n    pass\nx = 1\n");
    assert_eq!(names(&m), ["<main>", "a", "b"]);
    assert!(main_fn(&m).results.is_empty());
    assert_eq!(m.edition, sky_ir::Edition::E2026);
}

#[test]
fn redefinition_and_nested_defs_get_unique_names() {
    let m = lower_ok(
        "def f(x):\n    return x\na = f(1)\ndef f(x):\n    return x + 1\nb = f(1)\ndef outer(a):\n    def inner(b):\n        return b + 1 m\n    return inner(a)\n",
    );
    assert_eq!(names(&m), ["<main>", "f", "f#2", "outer", "inner"]);
    // Each call goes to the def that was current at the call.
    let calls: Vec<String> = main_fn(&m)
        .ops
        .iter()
        .filter_map(|o| match &o.kind {
            OpKind::Call { callee } => Some(callee.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, ["f", "f#2"]);
    let calls_inner = func(&m, "outer")
        .ops
        .iter()
        .any(|o| matches!(&o.kind, OpKind::Call { callee } if callee == "inner"));
    assert!(calls_inner);
}

#[test]
fn signatures_come_from_the_annotations_and_the_checker() {
    let m = lower_ok(
        "def scale(v: Vector[N], k) -> Vector[N]:\n    return v\ndef inc(x: Scalar) -> Scalar:\n    return x + 1\n",
    );
    let f = func(&m, "scale");
    assert_eq!(f.params.len(), 2);
    let v = f.params[0].as_number().expect("annotated");
    assert_eq!(v.shape.rank(), 1);
    assert_eq!(
        f.params[1],
        Type::Unknown,
        "an unannotated parameter is Unknown"
    );
    assert_eq!(f.results.len(), 1);
    assert_eq!(f.results[0].as_number().expect("number").shape.rank(), 1);
    assert_eq!(f.param_values().len(), 2);
    assert_eq!(f.value(f.param_values()[0]).name.as_deref(), Some("v"));
    let g = func(&m, "inc");
    assert_eq!(g.params, [Type::scalar(Dimension::DIMENSIONLESS)]);
    assert_eq!(g.results, [Type::scalar(Dimension::DIMENSIONLESS)]);
}

#[test]
fn a_function_without_a_value_returns_nothing_and_a_used_call_is_nothing() {
    let m = lower_ok("def shout():\n    print(\"hey\")\nc = shout()\n");
    let f = func(&m, "shout");
    assert_eq!(f.results, [Type::Nothing]);
    assert_eq!(
        top_kinds(f),
        ["const.str", "print", "const.nothing", "return"]
    );
    assert_eq!(main_fn(&m).ty(named(main_fn(&m), "c")), &Type::Nothing);
}

#[test]
fn calls_use_the_checkers_result_type_for_the_call_site() {
    let m = lower_ok(
        "def apply(A: Matrix[N, N], b: Vector[N]) -> Vector[N]:\n    return A @ b\nA3: Matrix[3, 3]\nb3: Vector[3]\nA4: Matrix[4, 4]\nb4: Vector[4]\ny3 = apply(A3, b3)\ny4 = apply(A4, b4)\n",
    );
    let f = main_fn(&m);
    let shape = |n: &str| {
        f.ty(named(f, n))
            .as_number()
            .expect("number")
            .shape
            .to_string()
    };
    // The callee's size variable is solved per call.
    assert_eq!(shape("y3"), "Vector[3]");
    assert_eq!(shape("y4"), "Vector[4]");
    // The function itself stays generic.
    let g = func(&m, "apply");
    assert_eq!(
        g.results[0].as_number().expect("number").shape.to_string(),
        "Vector[N]"
    );
}

#[test]
fn a_module_name_a_function_reads_is_an_extra_trailing_parameter() {
    let m =
        lower_ok("k = 2\ndef scaled(x):\n    return x * k\na = scaled(1)\nk = 3\nb = scaled(1)\n");
    let f = func(&m, "scaled");
    assert_eq!(f.params.len(), 2, "x, then the module name k");
    assert_eq!(f.value(f.param_values()[1]).name.as_deref(), Some("k"));
    let main = main_fn(&m);
    let calls: Vec<_> = main
        .ops
        .iter()
        .filter(|o| matches!(o.kind, OpKind::Call { .. }))
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].operands.len(), 2);
    // Each call passes the current value of k: 2, then 3 (the interpreter
    // reads the module's k when the call happens).
    let k_of = |o: &sky_ir::OpData| {
        let v = o.operands[1];
        match main.op(def_op(main, v).expect("const")).kind {
            OpKind::ConstFloat { bits } => f64::from_bits(bits),
            ref other => panic!("{other:?}"),
        }
    };
    assert_eq!(k_of(calls[0]), 2.0);
    assert_eq!(k_of(calls[1]), 3.0);
}

#[test]
fn locals_and_parameters_are_not_module_reads() {
    let m = lower_ok(
        "x = 5 m\nd = 3 m\ndef f(a):\n    x = a * 2\n    return x\ndef g(d):\n    return d\np = f(1 m)\nq = g(1 m)\n",
    );
    // `x` is assigned before it is read; `d` is a parameter.
    assert_eq!(func(&m, "f").params.len(), 1);
    assert_eq!(func(&m, "g").params.len(), 1);
}

#[test]
fn predefined_constants_are_literals_unless_the_module_rebinds_them() {
    let m = lower_ok("def energy(m):\n    return m * c^2\ne = energy(1 kg)\n");
    let f = func(&m, "energy");
    assert_eq!(f.params.len(), 1, "c is a literal, not a parameter");
    assert_eq!(count(f, "const.float"), 1);
    // Rebound at module level, it is an ordinary module name.
    let m = lower_ok("c = 2 m/s\ndef speed(t):\n    return c * t\ns = speed(1 s)\n");
    assert_eq!(func(&m, "speed").params.len(), 2);
    // ... with the value it has when the function is called, constant or not.
    let m =
        lower_ok("def speed(t):\n    return c * t\ns1 = speed(1 s)\nc = 2 m/s\ns2 = speed(1 s)\n");
    let f = func(&m, "speed");
    assert_eq!(f.params.len(), 2);
    let calls: Vec<_> = main_fn(&m)
        .ops
        .iter()
        .filter(|o| matches!(o.kind, OpKind::Call { .. }))
        .collect();
    assert_ne!(calls[0].operands[1], calls[1].operands[1]);
}

#[test]
fn module_reads_are_inherited_through_calls_and_never_from_the_callers_locals() {
    let src = "\
base = 2 m
def helper(a):
    return a + base
def outer(a):
    return helper(a) * 2
def shadow(base):
    return helper(1 m)
";
    let m = lower_ok(&(src.to_string() + "y = outer(3 m)\nz = shadow(9 m)\n"));
    // outer reads nothing itself but calls helper, so it takes base too.
    assert_eq!(func(&m, "outer").params.len(), 2);
    let outer = func(&m, "outer");
    let call = outer
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::Call { .. }))
        .expect("call");
    assert_eq!(
        call.operands,
        [outer.param_values()[0], outer.param_values()[1]]
    );
    // shadow has a parameter named base, and also needs the module's base
    // for helper: the call passes the trailing parameter, not its own.
    let shadow = func(&m, "shadow");
    assert_eq!(shadow.params.len(), 2);
    let call = shadow
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::Call { .. }))
        .expect("call");
    assert_eq!(call.operands[1], shadow.param_values()[1]);
    assert_ne!(call.operands[1], shadow.param_values()[0]);
}

#[test]
fn recursion_calls_itself() {
    let m = lower_ok(
        "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nf5 = fact(5)\n",
    );
    let f = func(&m, "fact");
    assert!(f
        .ops
        .iter()
        .any(|o| matches!(&o.kind, OpKind::Call { callee } if callee == "fact")));
    assert!(f.effects.contains(Effect::Call));
    assert!(f.effects.contains(Effect::Fail));
}

#[test]
fn function_effects_are_the_union_of_their_operations() {
    let m = lower_ok(
        "def id(x):\n    return x\ndef loud(x):\n    print(x)\n    return x\ndef div(a, b):\n    return a / b\ndef calls(x):\n    return id(x)\n",
    );
    assert!(func(&m, "id").effects.is_pure());
    assert!(func(&m, "loud").effects.contains(Effect::Io));
    assert!(func(&m, "div").effects.contains(Effect::Fail));
    assert!(func(&m, "calls").effects.contains(Effect::Call));
    assert!(!func(&m, "calls").effects.contains(Effect::Io));
}

#[test]
fn a_call_takes_the_callees_effects_through_any_depth() {
    let m = lower_ok(
        "def loud(x):\n    print(x)\n    return x\ndef mid(x):\n    return loud(x)\ndef top(x):\n    return mid(x)\na = top(1)\n",
    );
    for name in ["loud", "mid", "top"] {
        assert!(func(&m, name).effects.contains(Effect::Io), "{name}");
    }
    assert!(func(&m, "top").effects.contains(Effect::Call));
    assert!(main_fn(&m).effects.contains(Effect::Io));
}

#[test]
fn a_call_evaluates_its_arguments_in_order_before_the_module_names() {
    let m = lower_ok("k = 1\ndef f(a, b):\n    return a + b + k\nr = f(2 * k, 3 * k)\n");
    let main = main_fn(&m);
    let kinds = top_kinds(main);
    let ci = kinds.iter().position(|k| *k == "call").expect("call");
    assert_eq!(
        kinds[..ci],
        ["const.float", "const.float", "mul", "const.float", "mul"]
    );
    assert_eq!(main.op(top_op(main, "call")).operands.len(), 3);
}

#[test]
fn every_function_of_the_module_is_structurally_valid_and_ends_in_return() {
    let m = lower_ok(
        "k = 2\ndef f(n):\n    i = 0\n    while i < n:\n        if i == 3:\n            return i * k\n        i += 1\n    return 0\ndef g():\n    pass\nx = f(9)\ng()\n",
    );
    check_structure(&m);
    for f in &m.functions {
        assert_eq!(f.results.len(), usize::from(f.name != sky_ir::Module::MAIN));
    }
}
