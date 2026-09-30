//! Automatic parallelism, end to end: Sky source, lowered and optimized, run
//! on 1, 2, 4 and 8 threads with every threshold at its minimum so that small
//! programs use the parallel paths. Output, error (the whole rendered
//! diagnostic) and the analysis' decisions must not depend on the thread
//! count, and when calls fail concurrently the error is the one a sequential
//! run reports.

use sky_diag::SourceFile;
use sky_ir::parallel::{self, Config};
use sky_ir::Module;
use sky_opt::default_pipeline;

/// Lowers and optimizes `src`.
fn compile(src: &str) -> Module {
    let mut ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    default_pipeline()
        .verify_each(true)
        .run(&mut ir)
        .expect("the pipeline verifies");
    ir
}

/// What a run leaves: stdout and the rendered error, if any.
type Outcome = (String, Option<String>);

fn run(ir: &Module, src: &str, cfg: &Config) -> Outcome {
    let mut out = Vec::new();
    let r = sky_ast::with_deep_stack(|| sky_ir_interp::run_with(ir, &mut out, cfg));
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.err().map(|e| {
            e.to_diagnostic()
                .render_to_string(&SourceFile::new("p.sky", src))
        }),
    )
}

/// The outcome under 1, 2, 4 and 8 threads, eager and with the default
/// thresholds; they must all be the same.
fn everywhere(src: &str) -> Outcome {
    let ir = compile(src);
    let want = run(&ir, src, &Config::sequential());
    for t in [1, 2, 4, 8] {
        for cfg in [Config::eager(t), Config::with_threads(t)] {
            assert_eq!(run(&ir, src, &cfg), want, "{cfg:?}\n{src}");
        }
    }
    want
}

fn groups_in_main(src: &str) -> usize {
    let ir = compile(src);
    parallel::analyze(&ir, &Config::eager(4))
        .function(Module::MAIN)
        .map_or(0, |f| f.groups.len())
}

const DIVIDES: &str = "def f(x):\n    return 1 / x\n";
const OVERFLOWS: &str = "def g(x):\n    return 1e300 * x * 1e300\n";
const WORKS: &str = "def h(x):\n    return x * 2 + 1\n";

#[test]
fn two_independent_pure_calls_are_a_group() {
    let src = format!("{WORKS}{DIVIDES}print(h(3) + f(4))\n");
    assert_eq!(groups_in_main(&src), 1);
    assert_eq!(everywhere(&src), ("7.25\n".to_string(), None));
}

#[test]
fn the_first_failing_call_in_program_order_is_the_error() {
    // `f` divides by zero, `g` overflows: whichever finishes first, the
    // error is that of the call that comes first in the program.
    let both =
        |first: &str, second: &str| format!("{DIVIDES}{OVERFLOWS}print({first} + {second})\n");
    let src = both("f(0)", "g(10)");
    assert_eq!(groups_in_main(&src), 1);
    let (out, err) = everywhere(&src);
    assert_eq!(out, "");
    assert!(err.expect("fails").contains("division by zero"));
    let src = both("g(10)", "f(0)");
    let (_, err) = everywhere(&src);
    assert!(err.expect("fails").contains("arithmetic overflow"));
    // Only the second fails.
    let src = both("f(2)", "g(10)");
    let (_, err) = everywhere(&src);
    assert!(err.expect("fails").contains("arithmetic overflow"));
}

#[test]
fn an_operation_between_the_calls_that_fails_is_ordered_with_them() {
    // f(a); 1 / z; g(b), for every combination of which of the three fail.
    for f_fails in [false, true] {
        for mid_fails in [false, true] {
            for g_fails in [false, true] {
                let src = format!(
                    "{DIVIDES}{OVERFLOWS}z = {}\na = f({})\nc = 1 / z\nb = g({})\nprint(a, c, b)\n",
                    if mid_fails { "0" } else { "2" },
                    if f_fails { "0" } else { "2" },
                    if g_fails { "10" } else { "1e-300" },
                );
                let (out, err) = everywhere(&src);
                assert_eq!(out.is_empty(), err.is_some(), "{src}");
                match (f_fails, mid_fails, g_fails) {
                    (true, _, _) => {
                        assert!(err.expect("fails").contains("division by zero"), "{src}")
                    }
                    (false, true, _) => {
                        assert!(err.expect("fails").contains("division by zero"), "{src}")
                    }
                    (false, false, true) => {
                        assert!(err.expect("fails").contains("arithmetic overflow"), "{src}")
                    }
                    (false, false, false) => assert!(err.is_none(), "{src}: {err:?}"),
                }
            }
        }
    }
}

#[test]
fn output_never_changes_order() {
    // A `print` between two calls keeps them apart: had the first failed, the
    // print would not have happened.
    let src = format!("{WORKS}{DIVIDES}a = h(1)\nprint(a)\nb = f(2)\nprint(a + b)\n");
    assert_eq!(groups_in_main(&src), 0);
    assert_eq!(everywhere(&src).0, "3\n3.5\n");
    // And when the second call fails, the first print still happened.
    let src = format!("{WORKS}{DIVIDES}a = h(1)\nprint(a)\nb = f(0)\nprint(a + b)\n");
    let (out, err) = everywhere(&src);
    assert_eq!(out, "3\n");
    assert!(err.expect("fails").contains("division by zero"));
}

#[test]
fn a_callee_that_prints_is_never_run_elsewhere() {
    let src = "def noisy(x):\n    print(x)\n    return x + 1\n\
               def quiet(x):\n    return x * 2\n\
               print(noisy(1) + quiet(2) + noisy(3))\n";
    assert_eq!(groups_in_main(src), 0);
    assert_eq!(everywhere(src).0, "1\n3\n10\n");
}

#[test]
fn a_recursive_callee_is_never_run_elsewhere() {
    let src = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\
               print(fib(10) + fib(12))\n";
    assert_eq!(groups_in_main(src), 0);
    assert_eq!(everywhere(src).0, "199\n");
}

#[test]
fn calls_that_use_each_others_results_are_not_grouped() {
    let src = format!("{WORKS}{DIVIDES}print(f(h(1)) + h(2))\n");
    // `f(h(1))` needs `h(1)`: only `f` and the second `h` are independent.
    assert_eq!(everywhere(&src).0, "5.333333333333333\n");
}

#[test]
fn groups_nest() {
    let src = format!(
        "{WORKS}{DIVIDES}def both(x):\n    return h(x) + f(x + 1)\n\
         print(both(1) + both(2) + both(3))\n"
    );
    let ir = compile(&src);
    let plan = parallel::analyze(&ir, &Config::eager(4));
    assert_eq!(
        plan.function("both[x:Scalar]")
            .map_or(0, |f| f.groups.len()),
        1
    );
    assert_eq!(plan.function(Module::MAIN).map_or(0, |f| f.groups.len()), 1);
    everywhere(&src);
}

#[test]
fn a_call_in_a_loop_body_is_grouped_and_deterministic() {
    let src = format!(
        "{WORKS}{DIVIDES}total = 0\ni = 1\nwhile i < 20:\n    total = total + h(i) + f(i)\n    i = i + 1\nprint(total)\n"
    );
    assert_eq!(groups_in_main(&src), 1);
    everywhere(&src);
    // The same loop that ends in a failure at the third iteration.
    let src = format!(
        "{WORKS}{DIVIDES}total = 0\ni = 5\nwhile i > -5:\n    total = total + h(i) + f(i)\n    i = i - 1\nprint(total)\n"
    );
    let (out, err) = everywhere(&src);
    assert_eq!(out, "");
    assert!(err.expect("fails").contains("division by zero"));
}

#[test]
fn the_analysis_does_not_depend_on_the_thread_count() {
    let src = format!("{WORKS}{DIVIDES}{OVERFLOWS}print(h(3) + f(4) + g(0.001))\n");
    let ir = compile(&src);
    let plans: Vec<_> = [2, 4, 8, 64]
        .iter()
        .map(|&t| {
            parallel::analyze(&ir, &Config::eager(t))
                .function(Module::MAIN)
                .map(|f| f.groups.clone())
        })
        .collect();
    assert!(plans.windows(2).all(|w| w[0] == w[1]));
    // Notes differ only in the number of threads they name.
    let notes = |t: usize| -> Vec<String> {
        parallel::explain(&ir, &Config::eager(t))
            .iter()
            .map(|n| {
                format!(
                    "{} {} {}",
                    n.rule,
                    n.headline
                        .replace(&format!("up to {t} threads"), "up to N threads"),
                    format!("{:?}", n.because).replace(&format!("{t} threads"), "N threads")
                )
            })
            .collect()
    };
    assert_eq!(notes(2), notes(8));
    assert!(!notes(2).is_empty());
}

#[test]
fn one_thread_is_sequential() {
    let src = format!("{WORKS}{DIVIDES}print(h(3) + f(4))\n");
    let ir = compile(&src);
    assert!(!parallel::analyze(&ir, &Config::sequential()).has_groups());
    let notes = parallel::explain(&ir, &Config::sequential());
    assert!(notes.iter().all(|n| n.rule != parallel::CALLS_RULE));
}
