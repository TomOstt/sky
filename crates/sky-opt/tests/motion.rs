//! `gvn` and `licm`: what they number and move, and above all what they must
//! not: the output before a failure, and the failure Sky reports first.
//!
//! Every test lowers a Sky program, runs it in the SIR interpreter, runs one
//! pass (or the default pipeline) with the verifier after it, runs it again
//! and requires the same stdout and the same rendered error. Where a program
//! is chosen because the pass must *keep* an operation in place (a print
//! before it, an operation that fails first), the test also says where the
//! operation is afterwards.

use sky_diag::SourceFile;
use sky_ir::{Function, Module, OpKind, PassManager, RegionId};
use sky_opt::{default_passes, default_passes_without_specialize, gvn, licm, Related};

/// What running a program did.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    /// The error's whole diagnostic, rendered against the program.
    error: Option<String>,
}

fn run(m: &Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let result = sky_ir_interp::run(m, &mut out);
        Outcome {
            stdout: String::from_utf8(out).expect("UTF-8 output"),
            error: result.err().map(|e| {
                e.to_diagnostic()
                    .render_to_string(&SourceFile::new("test.sky", src))
            }),
        }
    })
}

fn lower(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"))
}

/// Runs one pass on the lowered program, checks the result verifies and
/// behaves as before, and returns the program and the pass's records.
fn with_pass(src: &str, pass: fn(&mut Module) -> Vec<Related>) -> (Module, Vec<Related>, Outcome) {
    let mut m = lower(src);
    let before = run(&m, src);
    let records = pass(&mut m);
    if let Err(e) = sky_ir::verify(&m) {
        panic!("invalid IR after the pass: {e:?}\n{}", sky_ir::print(&m));
    }
    let after = run(&m, src);
    assert_eq!(
        after,
        before,
        "the pass changed the behaviour of:\n{src}\n{}",
        sky_ir::print(&m)
    );
    // A second run finds nothing more.
    let again = pass(&mut m);
    assert!(again.is_empty(), "not idempotent: {again:?}");
    (m, records, before)
}

fn gvn_of(src: &str) -> (Module, Vec<Related>, Outcome) {
    with_pass(src, gvn)
}

fn licm_of(src: &str) -> (Module, Vec<Related>, Outcome) {
    with_pass(src, licm)
}

fn function<'a>(m: &'a Module, name: &str) -> &'a Function {
    m.function(name)
        .unwrap_or_else(|| panic!("no function {name}"))
}

/// The mnemonics of the operations at the top level of `f`'s body, and of
/// those inside `while` loops (any depth), each in order.
fn placement(f: &Function) -> (Vec<&'static str>, Vec<&'static str>) {
    fn walk(
        f: &Function,
        r: RegionId,
        in_loop: bool,
        out: &mut (Vec<&'static str>, Vec<&'static str>),
    ) {
        for &b in &f.region(r).blocks {
            for &o in &f.block(b).ops {
                let op = f.op(o);
                if in_loop {
                    out.1.push(op.kind.mnemonic());
                } else {
                    out.0.push(op.kind.mnemonic());
                }
                let inner = in_loop || matches!(op.kind, OpKind::While);
                for &r in &op.regions {
                    walk(f, r, inner, out);
                }
            }
        }
    }
    let mut out = (Vec::new(), Vec::new());
    walk(f, f.body, false, &mut out);
    out
}

fn count(ops: &[&str], name: &str) -> usize {
    ops.iter().filter(|o| **o == name).count()
}

fn rules(rs: &[Related]) -> Vec<&'static str> {
    rs.iter().map(|r| r.rewrite.rule).collect()
}

/// The function `run` of a program from the tests: not specialized, so the
/// loop's bound is a parameter.
fn run_fn(m: &Module) -> &Function {
    function(m, "run")
}

// ------------------------------------------------------------------ gvn

#[test]
fn gvn_reuses_a_value_that_dominates_across_blocks() {
    let src = "\
def run(x: Scalar, y: Scalar, c: Scalar) -> Scalar:
    a = x * y
    if c > 0:
        b = x * y
        print(b)
    else:
        print(a)
    return a + x * y
print(run(2, 3, 1))
print(run(2, 3, 0))
";
    let (m, records, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "6\n12\n6\n12\n");
    // `x * y` twice more: in the arm and after the `if`.
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "mul"), 1, "{}", sky_ir::print(&m));
    let reused: Vec<_> = records
        .iter()
        .filter(|r| r.rewrite.rule == "GVN-001" && r.rewrite.before.contains("mul"))
        .collect();
    assert_eq!(reused.len(), 2, "{records:?}");
    // The explanation points at both places: the reuse, and the first
    // computation.
    for r in reused {
        assert_ne!(r.rewrite.span, r.related);
        assert!(r.related.start < r.rewrite.span.start);
    }
}

#[test]
fn gvn_does_not_reuse_a_value_from_an_arm_after_the_if() {
    let src = "\
def run(x: Scalar, y: Scalar, c: Scalar) -> Scalar:
    if c > 0:
        a = x * y
        print(a)
    b = x * y
    return b
print(run(2, 3, 1))
print(run(2, 3, 0))
";
    let (m, records, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "6\n6\n6\n");
    // The `mul` in the arm does not dominate the one after the `if`.
    assert!(records.iter().all(|r| !r.rewrite.before.contains("mul")));
    let (top, all) = placement(run_fn(&m));
    assert_eq!(count(&top, "mul") + count(&all, "mul"), 2);
}

#[test]
fn gvn_does_not_reuse_a_value_between_the_parts_of_a_loop() {
    // The `before` region's values are not visible in `after`.
    let src = "\
def run(n: Scalar, x: Scalar) -> Scalar:
    i = 0
    total = 0
    while i * x < n * x:
        total = total + i * x
        i += 1
    return total
print(run(3, 2))
";
    let (m, _, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "6\n");
    let (_, in_loop) = placement(run_fn(&m));
    // `i * x` in the condition and in the body stay two operations.
    assert_eq!(count(&in_loop, "mul"), 3, "{}", sky_ir::print(&m));
}

#[test]
fn gvn_reuses_an_outer_value_inside_a_loop() {
    let src = "\
def run(n: Scalar, x: Scalar, y: Scalar) -> Scalar:
    k = x * y
    i = 0
    total = 0
    while i < n:
        total = total + x * y
        i += 1
    return total + k
print(run(3, 2, 5))
";
    let (m, _, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "40\n");
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "mul"), 1);
    assert_eq!(count(&in_loop, "mul"), 0, "{}", sky_ir::print(&m));
}

#[test]
fn gvn_orders_the_operands_of_commutative_operations_only() {
    let src = "\
def run(x: Scalar, y: Scalar) -> Scalar:
    a = x * y
    b = y * x
    c = x + y
    d = y + x
    e = x - y
    f = y - x
    return a + b + c + d + e + f
print(run(2, 3))
";
    let (m, records, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "22\n");
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "mul"), 1);
    // `x + y` twice and the two subtractions, the chain of additions.
    assert_eq!(count(&top, "sub"), 2, "{}", sky_ir::print(&m));
    assert_eq!(records.len(), 2);
}

#[test]
fn gvn_does_not_merge_calls_that_print() {
    let src = "\
def loud(a):
    print(a)
    return a
def quiet(a):
    return a + 1
def run(x: Scalar) -> Scalar:
    return loud(x) + loud(x) + quiet(x) + quiet(x)
print(run(2))
";
    let (m, _, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "2\n2\n10\n");
    let calls = |f: &Function, name: &str| {
        f.block(f.entry())
            .ops
            .iter()
            .filter(|o| matches!(&f.op(**o).kind, OpKind::Call { callee } if callee == name))
            .count()
    };
    assert_eq!(calls(run_fn(&m), "loud"), 2);
    assert_eq!(calls(run_fn(&m), "quiet"), 1);
}

#[test]
fn gvn_keeps_the_first_failure_when_the_repeated_operation_fails() {
    // Both divisions fail; the error is the first one's, at its own line.
    let src = "\
def run(x: Scalar, y: Scalar) -> Scalar:
    a = x / y
    print(1)
    b = x / y
    return a + b
print(run(1, 0))
";
    let (m, _, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "");
    let error = outcome.error.expect("division by zero");
    assert!(error.contains("2 |     a = x / y"), "{error}");
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 1);
    // Reusing a division that succeeded is fine too.
    let (_, _, ok) = gvn_of(&src.replace("run(1, 0)", "run(1, 2)"));
    assert_eq!(ok.stdout, "1\n1\n");
}

#[test]
fn gvn_reuses_an_operation_that_fails_first_in_an_earlier_iteration() {
    // The division before the loop is the first failure; the one in the loop
    // is a copy.
    let src = "\
def run(n: Scalar, x: Scalar, y: Scalar) -> Scalar:
    a = x / y
    i = 0
    while i < n:
        print(x / y)
        i += 1
    return a
print(run(2, 1, 0))
";
    let (_, records, outcome) = gvn_of(src);
    assert_eq!(outcome.stdout, "");
    assert!(outcome.error.unwrap().contains("2 |     a = x / y"));
    assert_eq!(rules(&records), ["GVN-001"]);
}

#[test]
fn gvn_says_where_the_value_was_computed() {
    let src = "\
def run(x: Scalar, y: Scalar) -> Scalar:
    a = x * y
    print(a)
    return x * y
print(run(2, 3))
";
    let (_, records, _) = gvn_of(src);
    assert_eq!(records.len(), 1);
    let r = &records[0];
    let x = sky_ir::Explanation {
        pass: "gvn",
        rule: r.rewrite.rule,
        span: r.rewrite.span,
        before: r.rewrite.before.clone(),
        after: r.rewrite.after.clone(),
        because: r.rewrite.because.clone(),
        related: Some(r.related),
    };
    assert_eq!(
        sky_opt::motion_headline(&x, src).as_deref(),
        Some("[GVN-001] `x * y` at line 4 reuses the value computed at line 2")
    );
    // The mul can fail (overflow): the note says why reuse is still safe.
    assert!(r
        .rewrite
        .because
        .iter()
        .any(|b| b.contains("first failure")));
}

// ----------------------------------------------------------------- licm

/// A loop whose first statement is an invariant division: `1 / d`.
const DIV_FIRST: &str = "\
def run(n: Scalar, d: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        t = 1 / d
        total = total + t
        i += 1
    return total
";

#[test]
fn licm_hoists_an_infallible_operation_even_from_a_loop_that_may_not_run() {
    let src = "\
def run(n: Scalar, x: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        w = x * 0.5
        total = total + w
        i += 1
    return total
print(run(3, 4))
print(run(0, 4))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "6\n0\n");
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "mul"), 1, "{}", sky_ir::print(&m));
    assert_eq!(count(&in_loop, "mul"), 0);
    assert_eq!(rules(&records), ["LICM-001"]);
    // The proof that `x * 0.5` cannot fail is in the explanation.
    assert!(records[0]
        .rewrite
        .because
        .iter()
        .any(|b| b.contains("can fail in general, but not here")));
}

#[test]
fn licm_does_not_hoist_a_failing_operation_past_a_print() {
    let src = "\
def run(n: Scalar, d: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        print(i)
        t = 1 / d
        total = total + t
        i += 1
    return total
print(run(3, 0))
";
    let (m, records, outcome) = licm_of(src);
    // The first iteration prints 0 and then fails.
    assert_eq!(outcome.stdout, "0\n");
    assert!(outcome.error.unwrap().contains("6 |         t = 1 / d"));
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 0, "{}", sky_ir::print(&m));
    assert_eq!(count(&in_loop, "div"), 1);
    assert!(records.is_empty(), "{records:?}");
}

#[test]
fn licm_does_not_hoist_a_failing_operation_past_one_that_fails_earlier() {
    // `x / i` fails on the first iteration (i = 0) before `1 / d` (d = 0)
    // could: the reported error is the earlier one.
    let src = "\
def run(n: Scalar, d: Scalar, x: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        a = x / i
        t = 1 / d
        total = total + a + t
        i += 1
    return total
print(run(3, 0, 1))
";
    let (m, _, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "");
    assert!(outcome.error.unwrap().contains("5 |         a = x / i"));
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 0);
}

#[test]
fn licm_hoists_a_failing_operation_into_a_guard_when_the_loop_may_not_run() {
    let src = format!("{DIV_FIRST}print(run(2, 4))\nprint(run(0, 0))\nprint(run(3, 0))\n");
    let (m, records, outcome) = licm_of(&src);
    // Two trips at 1/4, a zero-trip loop that must not fail on d = 0; then
    // the error of the third call.
    assert_eq!(outcome.stdout, "0.5\n0\n");
    let error = outcome.error.expect("1 / 0");
    assert!(error.contains("5 |         t = 1 / d"), "{error}");
    assert_eq!(rules(&records), ["LICM-002"], "{}", sky_ir::print(&m));
    // The division is now in the guarded pre-header: inside an `if`, before
    // the `while`.
    let f = run_fn(&m);
    let (top, in_loop) = placement(f);
    assert_eq!(count(&in_loop, "div"), 0);
    assert_eq!(count(&top, "div"), 1);
    assert!(top.contains(&"if"));
    // The guard's copy of the loop's test is a comparison before the `if`.
    let first_if = top.iter().position(|m| *m == "if").unwrap();
    assert!(top[..first_if].contains(&"cmp.lt"));
}

#[test]
fn licm_needs_no_guard_when_the_loop_is_known_to_run() {
    // `i` starts at 0 and the bound is 3: the first test is true, and
    // nothing before the division can print or fail on the first iteration.
    let src = "\
def run(d: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < 3:
        t = 1 / d
        total = total + t
        i += 1
    return total
print(run(4))
print(run(0))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "0.75\n");
    assert!(outcome.error.is_some());
    assert_eq!(rules(&records), ["LICM-001"]);
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 1);
    assert_eq!(count(&top, "if"), 0);
    assert_eq!(count(&in_loop, "div"), 0);
}

#[test]
fn licm_does_not_assume_a_loop_that_may_not_run_runs() {
    // A zero-trip loop with a failing invariant body must not fail.
    let src = "\
def run(d: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < 0:
        total = total + 1 / d
        i += 1
    return total
print(run(0))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "0\n");
    assert_eq!(outcome.error, None);
    // The division may move only under the guard: the loop's test is
    // false, so it never runs.
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&in_loop, "div"), 0, "{}", sky_ir::print(&m));
    assert_eq!(rules(&records), ["LICM-002"]);
    assert!(top.contains(&"if"));
}

#[test]
fn licm_hoists_from_the_condition_of_the_loop() {
    // The `before` region always runs first: `1 / d` fails at the first test
    // either way, so it moves without a guard.
    let src = "\
def run(n: Scalar, d: Scalar) -> Scalar:
    i = 0
    while i < n / d:
        i += 1
    return i
print(run(3, 1))
print(run(3, 0))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "3\n");
    assert!(outcome.error.unwrap().contains("3 |     while i < n / d:"));
    assert_eq!(rules(&records), ["LICM-001"]);
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 1);
    assert_eq!(count(&in_loop, "div"), 0);
    assert_eq!(count(&top, "if"), 0);
}

#[test]
fn licm_keeps_an_operation_after_a_print_in_the_condition() {
    // `f(i)` prints on every test, so `1 / d` after it must stay.
    let src = "\
def show(a):
    print(a)
    return a
def run(n: Scalar, d: Scalar) -> Scalar:
    i = 0
    while show(i) < n + 1 / d:
        i += 1
    return i
print(run(2, 0))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "0\n");
    assert!(outcome.error.is_some());
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 0, "{}", sky_ir::print(&m));
    assert!(records.is_empty(), "{records:?}");
}

#[test]
fn licm_hoists_an_inverse_and_a_transpose_out_of_a_time_stepping_loop() {
    let src = "\
def run(n: Scalar, a: Scalar) -> Vector[2]:
    K = [[a, 1], [1, 3]]
    A = [[1, 2], [3, 4]]
    u = [1, 1]
    i = 0
    while i < n:
        Kinv = inv(K)
        At = Aᵀ
        u = Kinv @ u + At @ u * 0.01
        i += 1
    return u
print(run(3, 2))
print(run(0, 0))
print(run(3, 3))
";
    let (m, records, outcome) = licm_of(src);
    // n = 0: no failure although `a = 0` makes... (K is invertible for a=0:
    // [[0,1],[1,3]] has determinant -1; the loop simply does not run.)
    assert!(outcome.error.is_none(), "{outcome:?}");
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&in_loop, "inverse"), 0, "{}", sky_ir::print(&m));
    assert_eq!(count(&in_loop, "transpose"), 0);
    assert_eq!(count(&top, "inverse"), 1);
    assert_eq!(count(&top, "transpose"), 1);
    assert!(rules(&records).contains(&"LICM-002"), "{records:?}");
}

#[test]
fn licm_keeps_a_singular_inverse_from_failing_a_zero_trip_loop() {
    let src = "\
def run(n: Scalar) -> Scalar:
    K = [[1, 2], [2, 4]]
    i = 0
    total = 0
    while i < n:
        Kinv = inv(K)
        total = total + Kinv[0, 0]
        i += 1
    return total
print(run(0))
print(run(1))
";
    let (m, _, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "0\n");
    let error = outcome.error.expect("K is singular");
    assert!(error.contains("6 |         Kinv = inv(K)"), "{error}");
    let (_, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&in_loop, "inverse"), 0, "{}", sky_ir::print(&m));
}

#[test]
fn licm_hoists_out_of_two_nested_loops() {
    let src = "\
def run(n: Scalar, x: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        j = 0
        while j < n:
            w = x * 0.5 + 1
            total = total + w
            j += 1
        i += 1
    return total
print(run(3, 4))
";
    let (m, _, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "27\n");
    let (top, in_loop) = placement(run_fn(&m));
    // `x * 0.5` and `+ 1` are before both loops.
    assert_eq!(count(&top, "mul"), 1, "{}", sky_ir::print(&m));
    assert_eq!(count(&top, "add"), 1);
    assert_eq!(count(&in_loop, "mul"), 0);
}

#[test]
fn licm_leaves_what_the_loop_computes_and_what_is_in_an_if_arm() {
    let src = "\
def run(n: Scalar, d: Scalar, c: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        if c > 0:
            total = total + 1 / d
        total = total + i * 2
        i += 1
    return total
print(run(3, 1, 1))
print(run(3, 0, 0))
print(run(3, 0, 1))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "9\n6\n");
    assert!(outcome.error.is_some());
    let (top, _) = placement(run_fn(&m));
    assert_eq!(count(&top, "div"), 0, "{}", sky_ir::print(&m));
    // `i * 2` depends on the loop, and the division is in an `if` arm: only
    // the comparison `c > 0` (which cannot fail) moved.
    assert_eq!(records.len(), 1, "{records:?}");
    assert!(records[0].rewrite.before.contains("cmp.gt"));
}

#[test]
fn licm_never_hoists_a_call() {
    let src = "\
def one(a):
    return a + 1
def run(n: Scalar, x: Scalar) -> Scalar:
    i = 0
    total = 0
    while i < n:
        total = total + one(x)
        i += 1
    return total
print(run(2, 3))
";
    let (m, records, outcome) = licm_of(src);
    assert_eq!(outcome.stdout, "8\n");
    let (top, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&top, "call"), 0);
    assert_eq!(count(&in_loop, "call"), 1);
    assert!(records.is_empty(), "{records:?}");
}

#[test]
fn licm_says_where_the_loop_is() {
    let (_, records, _) = licm_of(&format!("{DIV_FIRST}print(run(2, 4))\n"));
    let r = &records[0];
    let x = sky_ir::Explanation {
        pass: "licm",
        rule: r.rewrite.rule,
        span: r.rewrite.span,
        before: r.rewrite.before.clone(),
        after: r.rewrite.after.clone(),
        because: r.rewrite.because.clone(),
        related: Some(r.related),
    };
    let src = format!("{DIV_FIRST}print(run(2, 4))\n");
    assert_eq!(
        sky_opt::motion_headline(&x, &src).as_deref(),
        Some(
            "[LICM-002] `1 / d` is computed once before the loop, if the loop runs at all (line 4)"
        )
    );
    assert!(r
        .rewrite
        .because
        .iter()
        .any(|b| b.contains("must not start failing")));
}

// ------------------------------------------------------------- pipeline

#[test]
fn the_pipeline_with_and_without_specialize_keeps_loops_with_hoisted_operations() {
    let src = format!("{DIV_FIRST}print(run(2, 4))\nprint(run(0, 0))\nprint(run(3, 0))\n");
    let plain = {
        let m = lower(&src);
        run(&m, &src)
    };
    for passes in [default_passes, default_passes_without_specialize] {
        let mut m = lower(&src);
        passes(PassManager::for_tests()).run(&mut m).unwrap();
        assert_eq!(run(&m, &src), plain);
    }
}

#[test]
fn a_transpose_computed_outside_the_loop_stays_out_of_the_product() {
    // `licm` moves `Aᵀ` out of the loop. `transposes` must not then fold it
    // back into `matmul[ta]` inside the loop, which would copy it on every
    // iteration in an engine that materializes the transpose. The same holds
    // for a transpose the user wrote before the loop.
    for transposed in ["Aᵀ", "At"] {
        let src = format!(
            "def run(n: Scalar, a: Scalar) -> Vector[2]:
    A = [[a, 2], [3, 4]]
    At = Aᵀ
    u = [1, 1]
    i = 0
    while i < n:
        u = {transposed} @ u * 0.01
        i += 1
    return u
print(run(3, 2))
"
        );
        let mut m = lower(&src);
        let before = run(&m, &src);
        default_passes_without_specialize(PassManager::for_tests())
            .run(&mut m)
            .unwrap();
        assert_eq!(run(&m, &src), before);
        let (top, in_loop) = placement(run_fn(&m));
        assert_eq!(
            count(&top, "transpose"),
            1,
            "{transposed}\n{}",
            sky_ir::print(&m)
        );
        assert_eq!(count(&in_loop, "transpose"), 0);
        // Still a plain `matmul` in the loop.
        assert_eq!(count(&in_loop, "matmul"), 1);
    }
    // A transpose that is computed inside the loop is still folded.
    let src = "def run(n: Scalar, a: Scalar) -> Vector[2]:
    A = [[a, 2], [3, 4]]
    u = [1, 1]
    i = 0
    while i < n:
        B = A * u[0]
        u = Bᵀ @ u * 0.01
        i += 1
    return u
print(run(3, 2))
";
    let mut m = lower(src);
    default_passes_without_specialize(PassManager::for_tests())
        .run(&mut m)
        .unwrap();
    let (_, in_loop) = placement(run_fn(&m));
    assert_eq!(count(&in_loop, "transpose"), 0, "{}", sky_ir::print(&m));
}
