//! Lowering is bounded: long chains are loops, deep nesting stays within the
//! parser's depth cap, and nothing panics on any program the front end
//! accepts or rejects.

mod common;

use common::eval::run_main;
use common::*;

/// Lowers without the recursive structure check of `lower_ok`: the IR of
/// these programs is thousands of levels deep, which the test helpers (not
/// lowering) would need a deep stack for.
fn big(src: &str) -> sky_ir::Module {
    sky_lower::lower_source(src).expect("lowers")
}

#[test]
fn a_long_elif_chain_is_not_recursion() {
    let mut src = String::from("x = 4000\nif x == 0:\n    y = 0\n");
    for i in 1..5000 {
        src.push_str(&format!("elif x == {i}:\n    y = {i}\n"));
    }
    src.push_str("else:\n    y = 0 - 1\nprint(y)\n");
    let m = big(&src);
    assert_eq!(count(main_fn(&m), "if"), 5000);
}

#[test]
fn long_chains_still_mean_the_same() {
    // Shorter, because the test evaluator recurses over the nesting.
    let mut src = String::from("x = 250\nif x == 0:\n    y = 0\n");
    for i in 1..300 {
        src.push_str(&format!("elif x == {i}:\n    y = {i}\n"));
    }
    src.push_str("else:\n    y = 0 - 1\nprint(y)\n");
    assert_eq!(run_main(&lower_ok(&src)).0, "250\n");
    let mut src = String::from("x = 1\nok = ");
    for _ in 0..300 {
        src.push_str("x > 0 and ");
    }
    src.push_str("true\nno = ");
    for _ in 0..300 {
        src.push_str("x < 0 or ");
    }
    src.push_str("false\nprint(ok, no)\n");
    assert_eq!(run_main(&lower_ok(&src)).0, "true false\n");
    let mut src = String::from("def f(x):\n");
    for i in 0..300 {
        src.push_str(&format!("    if x == {i}:\n        return {i} * 2\n"));
    }
    src.push_str("    return 0 - 1\nprint(f(299), f(5000))\n");
    assert_eq!(run_main(&lower_ok(&src)).0, "598 -1\n");
}

#[test]
fn a_long_run_of_early_returns_is_not_recursion() {
    let mut src = String::from("def f(x):\n");
    for i in 0..4000 {
        src.push_str(&format!("    if x == {i}:\n        return {i} * 2\n"));
    }
    src.push_str("    return 0 - 1\nprint(f(3999), f(5000))\n");
    let m = big(&src);
    assert_eq!(count(func(&m, "f"), "return"), 1);
    assert_eq!(
        count(func(&m, "f"), "if"),
        8000,
        "a guard after each early return"
    );
}

#[test]
fn a_long_chain_of_and_or_operands_is_not_recursion() {
    let mut src = String::from("x = 1\nok = ");
    for _ in 0..5000 {
        src.push_str("x > 0 and ");
    }
    src.push_str("true\nno = ");
    for _ in 0..5000 {
        src.push_str("x < 0 or ");
    }
    src.push_str("false\nprint(ok, no)\n");
    let m = big(&src);
    assert_eq!(count(main_fn(&m), "if"), 10_000);
}

#[test]
fn a_long_program_lowers_quickly() {
    let mut src = String::new();
    for i in 0..20_000 {
        src.push_str(&format!("v{} = {} + {}\n", i % 500, i, i % 7));
    }
    src.push_str("print(v0)\n");
    let start = std::time::Instant::now();
    let m = big(&src);
    assert!(main_fn(&m).ops.len() > 40_000);
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}

#[test]
fn nesting_at_the_parsers_depth_limit() {
    // Nested ifs with an early return in the innermost and a statement after
    // each, so every level carries flags and a guard.
    let depth = 120;
    let mut src = String::from("def f(x):\n");
    for d in 0..depth {
        src.push_str(&format!("{}if x > {d}:\n", "    ".repeat(d + 1)));
    }
    src.push_str(&format!("{}return 1\n", "    ".repeat(depth + 1)));
    for d in (0..depth).rev() {
        src.push_str(&format!("{}y{d} = {d}\n", "    ".repeat(d + 1)));
    }
    src.push_str("    return 0\nprint(f(500), f(50))\n");
    let m = lower_ok(&src);
    assert_eq!(run_main(&m).0, "1 0\n");
    assert_eq!(count(func(&m, "f"), "return"), 1);

    // Nested whiles with a break at each level.
    let depth = 60;
    let mut src = String::new();
    for d in 0..depth {
        let pad = "    ".repeat(d);
        src.push_str(&format!(
            "{pad}k{d} = 0\n{pad}while k{d} < 2:\n{pad}    k{d} += 1\n"
        ));
    }
    src.push_str(&format!("{}print(k0)\n", "    ".repeat(depth)));
    for d in (0..depth).rev() {
        src.push_str(&format!(
            "{}if k{d} == 1:\n{}    break\n",
            "    ".repeat(d + 1),
            "    ".repeat(d + 1)
        ));
    }
    let m = lower_ok(&src);
    assert_eq!(count(main_fn(&m), "while"), depth);
}

#[test]
fn deeply_parenthesized_and_nested_expressions() {
    let n = 200;
    let src = format!(
        "x = {}1{}\ny = {}\n",
        "(".repeat(n),
        ")".repeat(n),
        "-".repeat(n) + "x"
    );
    lower_ok(&src);
    let mut e = String::from("x");
    for i in 0..100 {
        e = format!("({e} if x > {i} else {i})");
    }
    lower_ok(&format!("x = 5\ny = {e}\n"));
}

/// No program of the repository's UI suites (valid or not) makes lowering
/// panic: it lowers, is rejected with a message, or never reaches lowering.
#[test]
fn no_ui_program_panics() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut n = 0;
    for dir in [
        "tests/ui/check",
        "tests/ui/types",
        "tests/ui/lexer",
        "crates/sky-types/tests/ui",
    ] {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("sky") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&p) else {
                continue;
            };
            let _ = sky_lower::lower_source(&src);
            n += 1;
        }
    }
    assert!(n > 80, "{n}");
}
