//! Structured control flow: `if`, `while`, loop-carried variables, and the
//! flag scheme for `break`, `continue` and early `return`.

mod common;

use common::*;
use sky_ir::{BlockId, Function, OpId, OpKind, Type};
use sky_units::Dimension;

fn arg_names(f: &Function, b: BlockId) -> Vec<Option<String>> {
    f.block(b)
        .args
        .iter()
        .map(|v| f.value(*v).name.clone())
        .collect()
}

/// (before, after) of a `while`.
fn regions(f: &Function, w: OpId) -> (BlockId, BlockId) {
    let a = arms(f, w);
    (a[0], a[1])
}

fn last_op(f: &Function, b: BlockId) -> OpId {
    *f.block(b).ops.last().expect("a block has ops")
}

#[test]
fn if_with_else_merges_a_name_bound_in_both_arms() {
    let m = lower_ok("x = 3 m\nif x > 1 m:\n    y = 1 s\nelse:\n    y = 2 s\nz = y\nprint(z)\n");
    let f = main_fn(&m);
    let iff = top_op(f, "if");
    assert_eq!(f.op(iff).results.len(), 1);
    let y = f.op(iff).results[0];
    assert_eq!(f.value(y).name.as_deref(), Some("y"));
    assert_eq!(f.ty(y).as_number().expect("number").dim, Dimension::TIME);
    for b in arms(f, iff) {
        assert_eq!(f.op(last_op(f, b)).kind, OpKind::Yield);
        assert_eq!(f.op(last_op(f, b)).operands.len(), 1);
    }
    // Reading y after the `if` reads the result.
    assert_eq!(f.op(top_op(f, "print")).operands, [y]);
}

#[test]
fn a_name_bound_in_only_some_arms_is_not_merged() {
    // `y` is maybe-undefined after the `if`: nothing can read it, so the
    // `if` has no results.
    let m = lower_ok("x = 3 m\nif x > 1 m:\n    y = 1 s\n    print(y)\n");
    let f = main_fn(&m);
    assert_eq!(f.op(top_op(f, "if")).results.len(), 0);
}

#[test]
fn an_if_without_else_that_reassigns_yields_the_old_value_on_the_other_path() {
    let m = lower_ok("x = 3 m\nif x > 1 m:\n    x = 5 m\n");
    let f = main_fn(&m);
    let iff = top_op(f, "if");
    assert_eq!(f.op(iff).results.len(), 1);
    let [then_b, else_b] = arms(f, iff)[..] else {
        panic!("two arms")
    };
    assert_eq!(block_kinds(f, else_b), ["yield"]);
    // The empty else arm yields x as it was before the `if`.
    let x0 = f.op(top_ops(f, "const.float")[0]).results[0];
    assert_eq!(f.op(last_op(f, else_b)).operands, [x0]);
    assert_ne!(f.op(last_op(f, then_b)).operands, [x0]);
}

#[test]
fn elif_chains_are_flat_ifs_guarded_by_a_done_flag() {
    let m = lower_ok(
        "s = 3 m/s\nl = 10 m/s\nif s > l:\n    label = 1\nelif s == l:\n    label = 2\nelif s < l:\n    label = 3\nelse:\n    label = 4\n",
    );
    let f = main_fn(&m);
    // if, then per `elif` a condition-if and an arm-if, then `if not done`
    // for the `else`: six ifs, all siblings in the entry block.
    assert_eq!(count(f, "if"), 6);
    assert_eq!(top_ops(f, "if").len(), 6);
    assert_eq!(region_depth(f), 1, "a chain is one level, not one per arm");
    // The second condition is evaluated inside an `if done { false } else
    // { cond }`, not in the entry block.
    assert!(
        !top_kinds(f).contains(&"cmp.eq"),
        "an elif condition is not evaluated unless no earlier arm ran"
    );
    let cond_if = top_ops(f, "if")[1];
    let [taken, open] = arms(f, cond_if)[..] else {
        panic!("two arms")
    };
    assert_eq!(block_kinds(f, taken), ["const.bool", "yield"]);
    assert!(block_kinds(f, open).contains(&"cmp.eq"));
    // Each arm that another follows (and the last, before an `else`) sets
    // the flag, so it yields it besides `label`: three arm-ifs with two
    // results; the condition-ifs and the `else` have one.
    let two = f
        .ops
        .iter()
        .filter(|o| o.kind == OpKind::If && o.results.len() == 2)
        .count();
    assert_eq!(two, 3);
    // `label` is a number of one dimension throughout.
    assert_eq!(
        f.ty(named(f, "label")).as_number().expect("number").dim,
        Dimension::DIMENSIONLESS
    );
}

#[test]
fn if_arms_see_the_scope_before_the_if_not_each_other() {
    let m = lower_ok("x = 1\nif x > 0:\n    x = x + 1\nelse:\n    x = x + 10\n");
    let f = main_fn(&m);
    let iff = top_op(f, "if");
    let x0 = f.op(top_ops(f, "const.float")[0]).results[0];
    for b in arms(f, iff) {
        let add = f
            .block(b)
            .ops
            .iter()
            .find(|o| f.op(**o).kind == OpKind::Add)
            .expect("add");
        assert_eq!(f.op(*add).operands[0], x0);
    }
}

#[test]
fn while_carries_the_variables_its_body_assigns() {
    let m = lower_ok(
        "k = 3 m\ni = 0\ntotal = 0\nwhile i < 5:\n    total = total + i\n    i = i + 1\nprint(total, k)\n",
    );
    let f = main_fn(&m);
    let w = top_op(f, "while");
    // total and i are carried; k (read only) is not.
    assert_eq!(f.op(w).operands.len(), 2);
    assert_eq!(f.op(w).results.len(), 2);
    let (before, after) = regions(f, w);
    assert_eq!(f.block(before).args.len(), 2);
    assert_eq!(f.block(after).args.len(), 2);
    assert_eq!(
        arg_names(f, after),
        [Some("i".to_string()), Some("total".to_string())],
        "carried in name order: after's block arguments are the loop variables"
    );
    // `before` computes the condition and forwards the values.
    assert_eq!(
        block_kinds(f, before),
        ["const.float", "cmp.lt", "condition"]
    );
    let cond = f.op(last_op(f, before));
    assert_eq!(
        cond.operands.len(),
        3,
        "the Bool, then the forwarded values"
    );
    // `after` is the body and yields the next values.
    let y = f.op(last_op(f, after));
    assert_eq!(y.kind, OpKind::Yield);
    assert_eq!(y.operands.len(), 2);
    // No flags without break/continue/return.
    assert_eq!(count(f, "not"), 0);
    // After the loop, `print` reads the loop's results, and k is untouched.
    let pr = f.op(top_op(f, "print"));
    assert_eq!(pr.operands[0], f.op(w).results[1]);
    assert_eq!(pr.operands[1], named(f, "k"));
    assert_eq!(
        f.ty(f.op(w).results[0]),
        &Type::scalar(Dimension::DIMENSIONLESS)
    );
}

#[test]
fn the_types_of_carried_values_are_the_checkers() {
    let m = lower_ok("d = 0 m\nv = 2 m/s\nt = 1 s\nwhile d < 10 m:\n    d = d + v * t\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    assert_eq!(f.op(w).results.len(), 1);
    let (_, after) = regions(f, w);
    for v in &f.block(after).args {
        assert_eq!(f.ty(*v).as_number().expect("number").dim, Dimension::LENGTH);
    }
    assert_eq!(
        f.ty(f.op(w).results[0]).as_number().expect("number").dim,
        Dimension::LENGTH
    );
}

#[test]
fn a_name_first_assigned_in_the_loop_is_local_to_the_body() {
    let m = lower_ok("i = 0\nwhile i < 3:\n    t = i * 2\n    i = i + 1\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    assert_eq!(f.op(w).operands.len(), 1, "only i is carried");
}

#[test]
fn break_is_a_flag_carried_through_the_loop() {
    let m = lower_ok("n = 0\nwhile true:\n    n = n + 1\n    if n > 3:\n        break\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    // n and the break flag.
    assert_eq!(f.op(w).operands.len(), 2);
    let (before, after) = regions(f, w);
    // The flag starts false, before the loop.
    let init = f.op(w).operands[1];
    assert_eq!(
        f.op(def_op(f, init).expect("const")).kind,
        OpKind::ConstBool(false)
    );
    // The condition is only evaluated while the flag is false:
    // `if not brk { cond } else { false }`.
    assert_eq!(block_kinds(f, before), ["not", "if", "condition"]);
    let guard = f.op(f.block(before).ops[1]);
    let [then_b, else_b] = arms(f, f.block(before).ops[1])[..] else {
        panic!("arms")
    };
    assert_eq!(guard.results.len(), 1);
    assert_eq!(block_kinds(f, then_b), ["const.bool", "yield"]);
    assert_eq!(f.op(f.block(else_b).ops[0]).kind, OpKind::ConstBool(false));
    // The break sets the flag on its path; the loop yields it.
    assert_eq!(f.op(last_op(f, after)).operands.len(), 2);
    assert!(count(f, "const.bool") >= 3);
}

#[test]
fn continue_skips_the_rest_of_the_iteration_with_a_guard() {
    let m = lower_ok(
        "i = 0\nsteps = 0\nwhile i < 10:\n    i = i + 1\n    if i == 3:\n        continue\n    steps = steps + 1\n",
    );
    let f = main_fn(&m);
    let w = top_op(f, "while");
    // No break flag: only i and steps are carried, and the condition is not
    // guarded.
    assert_eq!(f.op(w).operands.len(), 2);
    let (before, after) = regions(f, w);
    assert_eq!(
        block_kinds(f, before),
        ["const.float", "cmp.lt", "condition"]
    );
    // The body: skip = false; ...; `if i == 3 { skip = true }`; then the
    // rest is guarded by `if not skip`.
    let kinds = block_kinds(f, after);
    assert_eq!(
        kinds[0], "const.bool",
        "the skip flag is reset each iteration"
    );
    assert!(kinds.contains(&"not"), "{kinds:?}");
    let guard = f
        .block(after)
        .ops
        .iter()
        .copied()
        .filter(|o| f.op(*o).kind == OpKind::If)
        .nth(1)
        .expect("the guard");
    // `steps + 1` is inside the guard, not at the top of the body.
    let inner = arms(f, guard)[0];
    assert!(block_kinds(f, inner).contains(&"add"));
    assert_eq!(
        kinds.iter().filter(|k| **k == "add").count(),
        1,
        "only `i + 1` is unguarded"
    );
}

#[test]
fn break_in_an_inner_loop_leaves_the_outer_loop_alone() {
    let m = lower_ok(
        "i = 0\nwhile i < 3:\n    j = 0\n    while true:\n        j = j + 1\n        if j > 2:\n            break\n    i = i + 1\n",
    );
    let f = main_fn(&m);
    let outer = top_op(f, "while");
    // The outer loop carries only i: the break belongs to the inner loop.
    assert_eq!(f.op(outer).operands.len(), 1);
    let (_, after) = regions(f, outer);
    let inner = f
        .block(after)
        .ops
        .iter()
        .copied()
        .find(|o| f.op(*o).kind == OpKind::While)
        .expect("inner loop");
    // j is first assigned in the loop, but `while true` is left only by
    // break, so it is bound afterwards: carried, with a placeholder start.
    assert_eq!(f.op(inner).operands.len(), 2, "j and the inner break flag");
}

#[test]
fn a_name_assigned_in_a_while_true_before_break_is_bound_after_it() {
    let m =
        lower_ok("while true:\n    found = 5\n    if found > 3:\n        break\nprint(found)\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    assert_eq!(f.op(w).results.len(), 2, "found and the break flag");
    // Its initial value is a placeholder built before the loop.
    let kinds = top_kinds(f);
    let wi = kinds.iter().position(|k| *k == "while").expect("while");
    assert!(kinds[..wi].contains(&"const.float"), "{kinds:?}");
    // print reads the loop's result.
    assert_eq!(f.op(top_op(f, "print")).operands[0], f.op(w).results[0]);
}

#[test]
fn a_function_with_one_final_return_uses_no_flags() {
    let m = lower_ok("def sq(x):\n    y = x * x\n    return y\n");
    let f = func(&m, "sq");
    assert_eq!(top_kinds(f), ["mul", "return"]);
    assert_eq!(count(f, "const.bool"), 0);
    let f2 = func(&lower_ok("def nothing():\n    pass\n"), "nothing").clone();
    assert_eq!(top_kinds(&f2), ["const.nothing", "return"]);
    assert_eq!(f2.results, [Type::Nothing]);
}

#[test]
fn early_return_sets_the_ret_flag_and_guards_the_rest() {
    let m = lower_ok(
        "def sign(x):\n    if x > 0 m:\n        return 1\n    if x < 0 m:\n        return 0 - 1\n    return 0\n",
    );
    let f = func(&m, "sign");
    // The body's only return is the last operation.
    assert_eq!(count(f, "return"), 1);
    assert_eq!(f.op(last_op(f, f.entry())).kind, OpKind::Return);
    // The flag and the result slot start out before any statement.
    assert_eq!(
        f.op(f.block(f.entry()).ops[0]).kind,
        OpKind::ConstBool(false)
    );
    // Each early return is followed by a guard on the flag.
    assert_eq!(count(f, "not"), 2);
    // The value returned is the slot, an `if` result.
    let ret = f.op(last_op(f, f.entry()));
    assert_eq!(ret.operands.len(), 1);
    assert!(def_op(f, ret.operands[0]).is_some());
    assert_eq!(f.results.len(), 1);
}

#[test]
fn return_inside_while_inside_if_inside_def() {
    let src = "\
def find(n):
    if n > 0:
        i = 0
        while i < 10:
            if i > n:
                return i
            i = i + 1
        return 0
    return 1
";
    let m = lower_ok(src);
    let f = func(&m, "find");
    assert_eq!(count(f, "return"), 1, "{}", dump(f));
    assert_eq!(count(f, "while"), 1);
    let w = f
        .ops
        .iter()
        .position(|o| o.kind == OpKind::While)
        .map(|i| OpId(i as u32))
        .expect("while");
    // Carried: i, the ret flag, the result slot, and the loop's break flag.
    assert_eq!(f.op(w).operands.len(), 4);
    assert_eq!(f.op(w).results.len(), 4);
    let (before, after) = regions(f, w);
    // The loop stops as soon as the return set the break flag.
    assert_eq!(block_kinds(f, before)[..2], ["not", "if"]);
    assert_eq!(f.block(before).args.len(), 4);
    assert_eq!(f.block(after).args.len(), 4);
    // Guards: after the `return i` `if` inside the loop (skip flag), after
    // the loop inside the outer `if` (ret), after the outer `if` (ret), and
    // the loop's own condition (break flag).
    assert_eq!(count(f, "not"), 4);
    // The function's result is one value, and the flags never escape.
    assert_eq!(f.results.len(), 1);
    assert_eq!(f.op(last_op(f, f.entry())).operands.len(), 1);
}

#[test]
fn return_from_nested_loops_stops_every_loop() {
    let src = "\
def f(n):
    i = 0
    while i < n:
        j = 0
        while j < n:
            if i * j > 6:
                return i
            j = j + 1
        i = i + 1
    return 0
";
    let m = lower_ok(src);
    let f = func(&m, "f");
    let whiles: Vec<OpId> = f
        .ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.kind == OpKind::While)
        .map(|(i, _)| OpId(i as u32))
        .collect();
    assert_eq!(whiles.len(), 2);
    // Both loops carry the return flag and slot and their own break flag.
    for w in whiles {
        let names: Vec<_> = arg_names(f, arms(f, w)[1]);
        assert!(names.len() >= 4, "{names:?}");
        assert!(f.op(w).operands.len() >= 4);
    }
    // The outer loop's condition is guarded too (it must stop after the
    // inner loop returned).
    let outer = *f
        .ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.kind == OpKind::While)
        .map(|(i, _)| OpId(i as u32))
        .collect::<Vec<_>>()
        .first()
        .expect("outer");
    let (before, _) = regions(f, outer);
    assert_eq!(block_kinds(f, before)[..2], ["not", "if"]);
}

#[test]
fn a_name_bound_on_the_path_that_falls_through_survives_an_arm_that_returns() {
    // `y` is bound after the `if` because the arm that does not bind it
    // returns (the checker takes it out of the merge), even though a guard
    // inside the other arm skips its assignment.
    let src = "\
def f(a, b):
    if a:
        if b:
            return 1
        y = 2
    else:
        y = 3
    return y
";
    let m = lower_ok(src);
    let f = func(&m, "f");
    assert_eq!(count(f, "return"), 1);
}

#[test]
fn a_returning_arm_takes_no_part_in_the_merge() {
    let src = "\
def clamp(v):
    if v < 0 m:
        return 0 m
    else:
        kept = v
    return kept
";
    let m = lower_ok(src);
    let f = func(&m, "clamp");
    assert_eq!(count(f, "return"), 1);
}

#[test]
fn code_after_a_statement_that_always_leaves_is_dropped() {
    let m = lower_ok("def f():\n    return 1\n    x = 2 m\n    print(x)\n");
    let f = func(&m, "f");
    // No `2 m`, no print: unreachable. The only `return` is then the last
    // live statement, so there are no flags either.
    assert_eq!(count(f, "print"), 0);
    assert_eq!(top_kinds(f), ["const.float", "return"]);
    let m = lower_ok("i = 0\nwhile i < 3:\n    i = i + 1\n    break\n    print(i)\n");
    assert_eq!(count(main_fn(&m), "print"), 0);
}

#[test]
fn while_true_with_an_early_return_needs_no_condition_ops() {
    let m = lower_ok("def forever():\n    while true:\n        return 1\n");
    let f = func(&m, "forever");
    assert_eq!(count(f, "return"), 1);
    assert_eq!(count(f, "while"), 1);
}

#[test]
fn a_function_returning_a_number_on_every_path_has_a_number_result() {
    let m = lower_ok(
        "def sign(x: Scalar):\n    if x > 0:\n        return 1\n    elif x < 0:\n        return 0 - 1\n    else:\n        return 0\n",
    );
    let f = func(&m, "sign");
    assert_eq!(f.results, [Type::scalar(Dimension::DIMENSIONLESS)]);
    // The result slot's stand-in is a number of the result type.
    assert_eq!(count(f, "return"), 1);
}
