//! `for` loops lower as a `while` over a hidden, loop-carried index. These
//! tests pin the shape; `sky-ir-interp`'s differential tests pin the
//! behaviour.

mod common;

use common::*;

#[test]
fn a_range_loop_is_a_while_whose_index_is_carried() {
    let m = lower_ok("total = 0\nfor i in range(5):\n    total += i\nprint(total)\n");
    let f = main_fn(&m);
    check_structure(&m);
    let w = top_op(f, "while");
    // The hidden index and `total` are carried, and the loop variable `i`
    // too (the loop is known to run, so `i` is defined after it).
    assert_eq!(f.op(w).operands.len(), 3);
    assert_eq!(f.op(w).results.len(), 3);
    let (before, after) = (arms(f, w)[0], arms(f, w)[1]);
    // `before` tests the index against the bound: a constant positive step
    // is a plain `<`.
    assert_eq!(block_kinds(f, before), ["cmp.lt", "condition"]);
    // `after` advances the index first, then runs the body.
    let kinds = block_kinds(f, after);
    assert_eq!(kinds.first(), Some(&"add"), "{kinds:?}");
    assert_eq!(
        kinds.iter().filter(|k| **k == "add").count(),
        2,
        "{kinds:?}"
    );
    assert_eq!(*kinds.last().expect("ops"), "yield");
    // No flags without `break`/`continue`/`return`.
    assert_eq!(count(f, "not"), 0);
}

#[test]
fn a_negative_constant_step_tests_greater_than() {
    let m = lower_ok("for i in range(9, 0, -2):\n    print(i)\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    let before = arms(f, w)[0];
    assert_eq!(block_kinds(f, before), ["cmp.gt", "condition"]);
}

#[test]
fn a_computed_step_tests_the_sign_of_the_gap_times_the_step() {
    let m = lower_ok("s = 2\nfor i in range(0, 9, s):\n    print(i)\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    let before = arms(f, w)[0];
    assert_eq!(
        block_kinds(f, before),
        ["sub", "mul", "const.float", "cmp.lt", "condition"]
    );
}

#[test]
fn only_a_loop_that_may_not_run_leaves_new_names_local() {
    // Known to run: `y` is defined after the loop, so it is carried.
    let m = lower_ok("for i in range(3):\n    y = i * 2\nprint(y)\n");
    let f = main_fn(&m);
    assert_eq!(f.op(top_op(f, "while")).operands.len(), 3, "idx, i, y");
    // A variable bound may be zero: `y` and `i` are not usable after the
    // loop, so only the hidden index is carried.
    let m = lower_ok("n = 3\nfor i in range(n):\n    y = i * 2\n    print(y)\n");
    let f = main_fn(&m);
    assert_eq!(f.op(top_op(f, "while")).operands.len(), 1, "idx");
}

#[test]
fn the_index_is_advanced_before_the_body_so_continue_still_moves_on() {
    let m = lower_ok("for i in range(6):\n    if i == 2:\n        continue\n    print(i)\n");
    let f = main_fn(&m);
    let w = top_op(f, "while");
    let after = arms(f, w)[1];
    let kinds = block_kinds(f, after);
    let add = kinds.iter().position(|k| *k == "add").expect("an add");
    let guard = kinds.iter().position(|k| *k == "if").expect("a guard");
    assert!(add < guard, "{kinds:?}");
}

#[test]
fn nested_loops_each_have_their_own_index() {
    let m = lower_ok("for i in range(2):\n    for j in range(3):\n        print(i, j)\n");
    check_structure(&m);
    let f = main_fn(&m);
    let outer = top_op(f, "while");
    let after = arms(f, outer)[1];
    assert!(block_kinds(f, after).contains(&"while"));
    // Both loop variables are named, and typed as indices even though the
    // inner one is first bound inside the outer loop.
    let text = dump(f);
    assert!(text.contains("'i") && text.contains("'j"), "{text}");
    assert!(!text.contains("?"), "{text}");
}

#[test]
fn a_loop_in_a_function_and_a_return_inside_it() {
    let m = lower_ok(
        "def f(n):\n    for k in range(1, n):\n        if k > 3:\n            return k\n    return 0\nprint(f(9))\n",
    );
    check_structure(&m);
    assert_eq!(count(func(&m, "f"), "while"), 1);
}

#[test]
fn iterating_an_array_is_a_loop_over_its_index() {
    // A vector: one `while`, the entry read by `index` at the top of the
    // body.
    let m = lower_ok("v = [1 s, 2 s, 3 s]\nfor x in v:\n    print(x)\nprint(x)\n");
    check_structure(&m);
    let main = func(&m, "<main>");
    assert_eq!(count(main, "while"), 1);
    assert!(
        count(main, "index") >= 2,
        "{}",
        sky_ir::print_function(main)
    );
    // A matrix gives rows; the length comes from the type.
    let m = lower_ok("A = [[1, 2], [3, 4], [5, 6]]\nfor r in A:\n    print(r)\n");
    check_structure(&m);
    assert_eq!(count(func(&m, "<main>"), "while"), 1);
    // A size known only as a variable is read with `len`.
    let m = lower_ok(
        "def f(w: Vector[N]) -> Scalar:\n    s = 0\n    for y in w:\n        s += y\n    return s\nprint(f([1, 2]))\n",
    );
    check_structure(&m);
    assert_eq!(count(func(&m, "f"), "len"), 1);
}

#[test]
fn iterating_a_value_of_unknown_type_is_a_clear_lowering_error() {
    // A parameter without an annotation may be a scalar or a string, which
    // the interpreter rejects at run time with its own error.
    let e = lower_err("def f(w):\n    for y in w:\n        print(y)\nf([1, 2])\n");
    assert!(
        e.contains("iteration over a value whose type is not known before the program runs"),
        "{e}"
    );
}
