//! Tests of `fuse`: what it fuses and what it leaves, the explanation, that
//! fusing changes nothing a program can observe (a differential property
//! test, with failures), and that the failure order is the unfused one.

use std::cell::Cell;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use sky_diag::Span;
use sky_ir::effects::infer_effects;
use sky_ir::{
    parse, print, verify, FunctionBuilder, Module, NumberType, OpKind, PassManager, Repr, Type,
    ValueId,
};
use sky_types::{Shape, TagSet};
use sky_units::Dimension;

use crate::rewrite::Rewrite;
use crate::{fuse, FusePass};

// ----- structural tests, on IR text -----

const HEAD: &str = "sir edition 2026\n";

/// Parses `body` as the function `@f(%a, %b, %d: Vector[3]; %s: scalar)`
/// returning `%c`, verifies it, and fuses it.
fn fused(body: &str) -> (Module, Vec<Rewrite>) {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)], %d: num[1; (3)], %s: num[1; ()], \
         %v: num[1; (3)]) -> (num[1; (3)]) effects(io, fail) {{\n{body}\n}}\n"
    );
    let mut m = parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{text}"));
    let records = fuse(&mut m);
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{}", print(&m)));
    (m, records)
}

/// The mnemonics of the operations directly in the function's entry block.
fn top_level(m: &Module) -> Vec<&'static str> {
    let f = &m.functions[0];
    f.block(f.entry())
        .ops
        .iter()
        .map(|o| f.op(*o).kind.mnemonic())
        .collect()
}

/// The mnemonics of the body of the first `fused` in the function.
fn body_of_first(m: &Module) -> Vec<&'static str> {
    let f = &m.functions[0];
    let fused = f
        .block(f.entry())
        .ops
        .iter()
        .copied()
        .find(|o| f.op(*o).kind == OpKind::Fused)
        .expect("a fused operation");
    let block = f.region(f.op(fused).regions[0]).blocks[0];
    f.block(block)
        .ops
        .iter()
        .map(|o| f.op(*o).kind.mnemonic())
        .collect()
}

#[test]
fn a_chain_becomes_one_kernel() {
    let (m, records) = fused(
        "  %5 = const.float 2 : num[1; ()] @ 20..21
  %t1 = add %a, %b : num[1; (3)] @ 12..17
  %t2 = mul %t1, %5 : num[1; (3)] @ 11..21
  %c = sub %t2, %d : num[1; (3)] @ 11..26
  return %c",
    );
    assert_eq!(top_level(&m), ["const.float", "fused", "return"]);
    assert_eq!(body_of_first(&m), ["add", "mul", "sub", "yield"]);
    assert_eq!(records.len(), 1);
    let r = &records[0];
    assert_eq!((r.pass, r.rule), ("fuse", "FUSE-001"));
    // The kernel's span is the join of the operations' spans.
    assert_eq!((r.span.start, r.span.end), (11, 26));
    let f = &m.functions[0];
    let k = f.block(f.entry()).ops[1];
    assert_eq!((f.op(k).span.start, f.op(k).span.end), (11, 26));
    // The operations keep their own spans inside.
    let body = f.block(f.region(f.op(k).regions[0]).blocks[0]);
    let spans: Vec<(u32, u32)> = body
        .ops
        .iter()
        .map(|o| (f.op(*o).span.start, f.op(*o).span.end))
        .collect();
    assert_eq!(spans, [(12, 17), (11, 21), (11, 26), (11, 26)]);
    // The result keeps its name; the operands are a, b, the scalar, d.
    assert_eq!(f.value(f.op(k).results[0]).name.as_deref(), Some("c"));
    assert_eq!(f.op(k).operands.len(), 4);
}

#[test]
fn the_explanation_reads_like_the_design_says() {
    let (_, records) = fused(
        "  %5 = const.float 2 : num[1; ()] @ 20..21
  %t1 = add %a, %b : num[1; (3)] @ 12..17
  %t2 = mul %t1, %5 : num[1; (3)] @ 11..21
  %c = sub %t2, %d : num[1; (3)] @ 11..26
  return %c",
    );
    let r = &records[0];
    assert_eq!(r.before, "(a + b) * 2 - d");
    assert_eq!(r.after, "fused(a, b, 2, d)");
    assert_eq!(
        r.because,
        [
            "fused 3 elementwise operations into one kernel; saves 2 temporary arrays of 3 \
             elements",
            "every intermediate result is used exactly once, by the next operation, and every \
             array operand is a Vector[3]",
            "`t1` and `t2` are only used here, so they are no longer materialized",
            "3 of the operations can fail (`add`, `mul`, `sub`); the kernel reports the failure \
             of the first of them in program order that fails at any element, which is the \
             error the separate operations gave",
        ]
    );
}

#[test]
fn named_temporaries_are_mentioned_and_one_failing_op_is_named() {
    let (_, records) = fused(
        "  %t = neg %a : num[1; (3)] @ 1..3
  %c = div %t, %s : num[1; (3)] @ 1..8
  return %c",
    );
    let r = &records[0];
    assert_eq!(r.before, "-a / s");
    assert_eq!(
        r.because[0],
        "fused 2 elementwise operations into one kernel; saves 1 temporary array of 3 elements"
    );
    assert_eq!(
        r.because[2],
        "`t` is only used here, so it is no longer materialized"
    );
    assert_eq!(
        r.because[3],
        "only `div` can fail, so a failure has the same message and place as before"
    );
}

#[test]
fn an_intermediate_with_two_uses_is_not_absorbed() {
    // `t` feeds two operations, so it is computed once, outside.
    let (m, records) = fused(
        "  %t = add %a, %b : num[1; (3)] @ 1..2
  %u = mul %t, %s : num[1; (3)] @ 3..4
  %c = sub %u, %t : num[1; (3)] @ 5..6
  return %c",
    );
    assert_eq!(top_level(&m), ["add", "fused", "return"]);
    assert_eq!(body_of_first(&m), ["mul", "sub", "yield"]);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].before, "t * s - t");
}

#[test]
fn a_repeated_operand_is_one_kernel_argument() {
    let (m, _) = fused(
        "  %t = add %a, %a : num[1; (3)] @ 1..2
  %c = sub %t, %a : num[1; (3)] @ 3..4
  return %c",
    );
    let f = &m.functions[0];
    let k = f.block(f.entry()).ops[0];
    assert_eq!(f.op(k).operands.len(), 1);
    assert_eq!(body_of_first(&m), ["add", "sub", "yield"]);
}

#[test]
fn single_operations_scalars_and_other_arrays_ops_are_left_alone() {
    let (m, records) = fused(
        "  %one = const.float 1 : num[1; ()] @ 1..2
  %x = add %s, %one : num[1; ()] @ 3..4
  %y = mul %x, %x : num[1; ()] @ 5..6
  %c = add %a, %b : num[1; (3)] @ 7..8
  return %c",
    );
    assert!(records.is_empty());
    assert_eq!(
        top_level(&m),
        ["const.float", "add", "mul", "add", "return"]
    );
}

#[test]
fn a_value_used_in_a_region_or_returned_is_not_absorbed() {
    // `t` is used once, but by a `print`, and `u` by an `if` arm.
    let (m, records) = fused(
        "  %t = add %a, %b : num[1; (3)] @ 1..2
  %u = mul %t, %s : num[1; (3)] @ 3..4
  print %t @ 5..6
  %yes = const.bool true : bool
  %w = if %yes : num[1; (3)] {
    ^1:
      %k = neg %u : num[1; (3)] @ 7..8
      yield %k @ 7..8
  } {
    ^2:
      yield %d @ 9..10
  } @ 7..10
  return %w",
    );
    // `t` has two uses (`mul` and `print`), so nothing here fuses, and `u`
    // is used from another block.
    assert!(records.is_empty(), "{records:?}");
    assert_eq!(
        top_level(&m),
        ["add", "mul", "print", "const.bool", "if", "return"]
    );
}

#[test]
fn fusion_works_inside_regions() {
    let (m, records) = fused(
        "  %yes = const.bool true : bool
  %w = if %yes : num[1; (3)] {
    ^1:
      %t = add %a, %b : num[1; (3)] @ 1..2
      %k = neg %t : num[1; (3)] @ 3..4
      yield %k @ 3..4
  } {
    ^2:
      yield %d @ 9..10
  } @ 1..10
  return %w",
    );
    assert_eq!(records.len(), 1);
    let text = print(&m);
    assert!(text.contains("= fused %a, %b"), "{text}");
}

#[test]
fn a_failing_operation_is_not_moved_past_another_effect() {
    // `add` can fail; `index` can fail too, and runs after it. Fusing the
    // `add` into the `mul` would run it after the `index`.
    let (m, records) = fused(
        "  %t = add %a, %b : num[1; (3)] @ 1..2
  %zero = const.float 0 : num[1; ()] @ 3..4
  %e = index %v, %zero : num[1; ()] @ 5..6
  %c = mul %t, %e : num[1; (3)] @ 7..8
  return %c",
    );
    assert!(records.is_empty());
    assert_eq!(
        top_level(&m),
        ["add", "const.float", "index", "mul", "return"]
    );
    // A `print` in between blocks it just as well.
    let (_, records) = fused(
        "  %t = sub %a, %b : num[1; (3)] @ 1..2
  print %s @ 3..4
  %c = mul %t, %s : num[1; (3)] @ 7..8
  return %c",
    );
    assert!(records.is_empty());
}

#[test]
fn a_pure_operation_may_move_past_effects_and_the_rest_is_fused() {
    // `neg` cannot fail, so it may run after the `print`.
    let (m, records) = fused(
        "  %t = neg %a : num[1; (3)] @ 1..2
  print %s @ 3..4
  %c = add %t, %b : num[1; (3)] @ 7..8
  return %c",
    );
    assert_eq!(records.len(), 1);
    assert_eq!(top_level(&m), ["print", "fused", "return"]);
    // When the whole tree is blocked, a subtree still fuses.
    let (m, records) = fused(
        "  %t1 = add %a, %b : num[1; (3)] @ 1..2
  %t2 = mul %t1, %s : num[1; (3)] @ 3..4
  %zero = const.float 0 : num[1; ()] @ 5..6
  %e = index %v, %zero : num[1; ()] @ 7..8
  %c = sub %t2, %d : num[1; (3)] @ 9..10
  return %c",
    );
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].before, "(a + b) * s");
    assert_eq!(
        top_level(&m),
        ["fused", "const.float", "index", "sub", "return"]
    );
}

#[test]
fn a_dynamically_checked_neighbour_blocks_moving_a_failure() {
    // `%q` has no type, so its `neg` can fail at run time.
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)], %q: unknown) -> (num[1; (3)]) effects(fail) {{
  %t = add %a, %b : num[1; (3)] @ 1..2
  %u = neg %q : unknown @ 3..4
  %c = neg %t : num[1; (3)] @ 5..6
  return %c
}}\n"
    );
    let mut m = parse(&text).unwrap();
    assert!(fuse(&mut m).is_empty());
}

#[test]
fn fusing_twice_changes_nothing_and_the_result_round_trips() {
    let (mut m, _) = fused(
        "  %t1 = add %a, %b : num[1; (3)] @ 1..2
  %t2 = mul %t1, %s : num[1; (3)] @ 3..4
  %c = sub %t2, %d : num[1; (3)] @ 5..6
  return %c",
    );
    let once = m.clone();
    assert!(fuse(&mut m).is_empty());
    assert_eq!(m, once);
    let text = print(&m);
    let back = parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    verify(&back).unwrap();
    assert_eq!(print(&back), text);
    assert_eq!(back, sky_ir::canonicalize(&m));
}

#[test]
fn a_matrix_chain_fuses_and_a_symbolic_shape_is_named() {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (N#1)], %b: num[1; (N#1)]) -> (num[1; (N#1)]) effects(fail) {{
  %t = add %a, %b : num[1; (N#1)] @ 1..2
  %c = neg %t : num[1; (N#1)] @ 3..4
  return %c
}}\n"
    );
    let mut m = parse(&text).unwrap();
    let records = fuse(&mut m);
    verify(&m).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].because[0],
        "fused 2 elementwise operations into one kernel; saves 1 temporary array of shape \
         Vector[N]"
    );
}

#[test]
fn the_pass_reports_through_the_manager() {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)]) -> (num[1; (3)]) effects(fail) {{
  %t = add %a, %b : num[1; (3)] @ 1..2
  %c = neg %t : num[1; (3)] @ 3..4
  return %c
}}\n"
    );
    let mut m = parse(&text).unwrap();
    let report = PassManager::for_tests().with(FusePass).run(&mut m).unwrap();
    assert_eq!(report.ran[0].0, "fuse");
    assert!(report.ran[0].1.is_yes());
    assert_eq!(report.explanations.len(), 1);
    assert_eq!(report.explanations[0].rule, "FUSE-001");
    assert_eq!(report.explanations[0].pass, "fuse");
    // Nothing left: not a change.
    let again = PassManager::for_tests().with(FusePass).run(&mut m).unwrap();
    assert!(!again.ran[0].1.is_yes());
    assert!(again.explanations.is_empty());
}

// ----- the other passes and `fused` -----

fn kernel_module(body: &str, effects: &str) -> Module {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)]) -> (num[1; (3)]) {effects} {{
  %dead = fused %a, %b : num[1; (3)] {{
    ^1(%x: num[1; ()], %y: num[1; ()]):
{body}
  }}
  return %a
}}\n"
    );
    let m = parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{text}"));
    m
}

#[test]
fn dce_removes_an_unused_kernel_that_cannot_fail_and_keeps_one_that_can() {
    let mut pure = kernel_module("      %n = neg %x : num[1; ()]\n      yield %n", "");
    let _ = crate::dce(&mut pure);
    assert_eq!(top_level(&pure), ["return"]);
    let mut failing = kernel_module(
        "      %s = add %x, %y : num[1; ()]\n      yield %s",
        "effects(fail)",
    );
    let _ = crate::dce(&mut failing);
    assert_eq!(top_level(&failing), ["fused", "return"]);
}

#[test]
fn fold_and_the_egraph_leave_kernel_bodies_alone() {
    // A body's `neg` is `0 - x`, not the scalar `-x` those passes assume:
    // `-(-x)` and `-0` must not be simplified inside a body.
    let body = "      %z = const.float 0 : num[1; ()]
      %n = neg %z : num[1; ()]
      %m = neg %x : num[1; ()]
      %o = neg %m : num[1; ()]
      %p = add %o, %n : num[1; ()]
      yield %p";
    let m = kernel_module(body, "effects(fail)");
    let mut folded = m.clone();
    assert!(crate::fold(&mut folded).is_empty());
    assert_eq!(folded, m);
    let mut rewritten = m.clone();
    let records = sky_egraph::rewrite_module(&mut rewritten, &sky_egraph::RewriteConfig::default());
    assert!(records.is_empty(), "{records:?}");
    assert_eq!(rewritten, m);
}

// ----- failure order, on a chain the naive order gets wrong -----

/// Runs `f(args)` and returns its results as `Debug` text (which tells
/// `-0.0` from `0.0`) or the error's message and span.
fn call_f(m: &Module, args: Vec<sky_ir_interp::Value>) -> Result<String, (String, Span)> {
    sky_ast::with_deep_stack(|| {
        sky_ir_interp::call(m, "f", args, &mut Vec::new())
            .map(|v| format!("{v:?}"))
            .map_err(|e| (e.message.clone(), e.span))
    })
}

fn vector(xs: &[f64]) -> sky_ir_interp::Value {
    sky_ir_interp::Value::array(vec![xs.len()], xs.to_vec(), Dimension::DIMENSIONLESS)
}

#[test]
fn the_first_failing_operation_wins_even_at_a_later_element() {
    // t1 = a * s   overflows at element 2 (1e200 * 1e200)
    // t2 = b * s   overflows at element 0
    // c  = t1 + t2
    // Unfused, `t1` runs first over all elements and fails: its span.
    // Per element, the first failure met is `t2`'s at element 0.
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)], %s: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %t1 = mul %a, %s : num[1; (3)] @ 10..11
  %t2 = mul %b, %s : num[1; (3)] @ 20..21
  %c = add %t1, %t2 : num[1; (3)] @ 30..31
  return %c
}}\n"
    );
    let before = parse(&text).unwrap();
    let mut after = before.clone();
    let records = fuse(&mut after);
    assert_eq!(records.len(), 1);
    assert!(print(&after).contains("fused"));
    let args = || {
        vec![
            vector(&[1.0, 1.0, 1e200]),
            vector(&[1e200, 1.0, 1.0]),
            sky_ir_interp::Value::scalar(1e200),
        ]
    };
    let unfused = call_f(&before, args());
    let fused = call_f(&after, args());
    assert_eq!(
        unfused,
        Err(("arithmetic overflow".to_string(), Span::new(10, 11)))
    );
    assert_eq!(fused, unfused);
    // Swap which one fails first in program order: the other span.
    let swapped = text
        .replace("10..11", "99..100")
        .replace("20..21", "10..11");
    let before = parse(&swapped).unwrap();
    let mut after = before.clone();
    let _ = fuse(&mut after);
    let unfused = call_f(&before, args());
    assert_eq!(fused_span(&unfused), Some(Span::new(99, 100)));
    assert_eq!(call_f(&after, args()), unfused);
}

fn fused_span(r: &Result<String, (String, Span)>) -> Option<Span> {
    r.as_ref().err().map(|e| e.1)
}

#[test]
fn division_by_zero_and_overflow_keep_their_own_messages() {
    // `div` by zero is decided per operation, overflow per element.
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %s: num[1; ()], %z: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %t1 = mul %a, %s : num[1; (3)] @ 10..11
  %t2 = div %t1, %z : num[1; (3)] @ 20..21
  %c = neg %t2 : num[1; (3)] @ 30..31
  return %c
}}\n"
    );
    let before = parse(&text).unwrap();
    let mut after = before.clone();
    assert_eq!(fuse(&mut after).len(), 1);
    let run = |m: &Module, s: f64, z: f64| {
        call_f(
            m,
            vec![
                vector(&[1.0, 1e200, 3.0]),
                sky_ir_interp::Value::scalar(s),
                sky_ir_interp::Value::scalar(z),
            ],
        )
    };
    for (s, z) in [(1.0, 0.0), (1e200, 0.0), (1e200, 2.0), (2.0, 2.0)] {
        assert_eq!(run(&after, s, z), run(&before, s, z), "s={s} z={z}");
    }
    assert_eq!(
        run(&before, 1e200, 0.0),
        Err(("arithmetic overflow".to_string(), Span::new(10, 11)))
    );
    assert_eq!(
        run(&before, 1.0, 0.0),
        Err(("division by zero".to_string(), Span::new(20, 21)))
    );
}

#[test]
fn negating_zero_gives_positive_zero_as_the_array_operation_does() {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %b: num[1; (3)]) -> (num[1; (3)]) effects(fail) {{
  %t = neg %a : num[1; (3)] @ 1..2
  %c = add %t, %b : num[1; (3)] @ 3..4
  return %c
}}\n"
    );
    let before = parse(&text).unwrap();
    let mut after = before.clone();
    assert_eq!(fuse(&mut after).len(), 1);
    let args = || vec![vector(&[0.0, -0.0, 1.0]), vector(&[-0.0, -0.0, -0.0])];
    let unfused = call_f(&before, args());
    assert_eq!(call_f(&after, args()), unfused);
    // `0 - 0 = +0`, and `+0 + -0 = +0`: no `-0` anywhere.
    assert!(!unfused.unwrap().contains("-0.0"));
}

// ----- a differential property test -----

const FLOATS: [f64; 12] = [
    0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 3.0, 1e200, -1e200, 1.7e308, 1e-300, 1e154,
];

#[derive(Clone, Debug)]
enum Step {
    /// A vector of 3, a vector of 1, or a 2x2 matrix, of dimension 1 or L.
    Array(u8, bool, Vec<usize>),
    Scalar(usize, bool),
    /// `add`, `sub`, `mul`, `div` or `neg` on arrays and scalars of the pool.
    Elementwise(u8, usize, usize),
    /// A scalar `add`, `sub`, `mul` or `div`: it can fail, too.
    ScalarOp(u8, usize, usize),
    /// `v[k]` on a vector: fails when `k` is out of range.
    Index(usize, u8),
    Print(usize),
}

fn program() -> impl Strategy<Value = Vec<Step>> {
    let idx = 0usize..64;
    let float_ix = 0..FLOATS.len();
    let step = prop_oneof![
        3 => (0u8..3, any::<bool>(), proptest::collection::vec(float_ix.clone(), 4))
            .prop_map(|(s, l, e)| Step::Array(s, l, e)),
        2 => (float_ix, any::<bool>()).prop_map(|(x, l)| Step::Scalar(x, l)),
        10 => (0u8..5, idx.clone(), idx.clone()).prop_map(|(o, a, b)| Step::Elementwise(o, a, b)),
        2 => (0u8..4, idx.clone(), idx.clone()).prop_map(|(o, a, b)| Step::ScalarOp(o, a, b)),
        2 => (idx.clone(), 0u8..5).prop_map(|(a, k)| Step::Index(a, k)),
        2 => idx.prop_map(Step::Print),
    ];
    proptest::collection::vec(step, 1..24)
}

#[derive(Clone, Copy)]
struct Arr {
    v: ValueId,
    shape: u8,
    dim: Dimension,
}

fn shape_of(kind: u8) -> (Shape, usize, Option<u64>) {
    match kind {
        0 => (Shape::vector(3), 3, None),
        1 => (Shape::vector(1), 1, None),
        _ => (Shape::matrix(2, 2), 4, Some(2)),
    }
}

fn ty(dim: Dimension, shape: Shape) -> Type {
    Type::Number(NumberType {
        dim,
        shape,
        tags: TagSet::EMPTY,
        repr: Repr::Unchosen,
    })
}

/// The dimensions the generator allows, so exponents stay small.
fn allowed(d: Dimension) -> bool {
    [-1, 0, 1, 2]
        .into_iter()
        .any(|k| Dimension::LENGTH.pow_int(k).is_ok_and(|x| x == d))
}

struct Gen {
    b: FunctionBuilder,
    spans: u32,
    arrays: Vec<Arr>,
    scalars: Vec<(ValueId, Dimension)>,
}

impl Gen {
    fn span(&mut self) -> Span {
        self.spans += 1;
        Span::new(self.spans, self.spans + 1)
    }

    fn op1(&mut self, kind: OpKind, args: Vec<ValueId>, t: Type) -> ValueId {
        let s = self.span();
        self.b.op1(kind, args, t, s)
    }

    fn constant(&mut self, x: f64, dim: Dimension) -> ValueId {
        self.op1(
            OpKind::ConstFloat { bits: x.to_bits() },
            vec![],
            ty(dim, Shape::scalar()),
        )
    }

    fn step(&mut self, st: &Step) {
        match st {
            Step::Array(kind, length, elems) => {
                let dim = if *length {
                    Dimension::LENGTH
                } else {
                    Dimension::DIMENSIONLESS
                };
                let (shape, n, cols) = shape_of(*kind);
                let entries: Vec<ValueId> = elems[..n]
                    .iter()
                    .map(|i| self.constant(FLOATS[*i], dim))
                    .collect();
                let v = self.op1(OpKind::ArrayLiteral { cols }, entries, ty(dim, shape));
                self.arrays.push(Arr {
                    v,
                    shape: *kind,
                    dim,
                });
            }
            Step::Scalar(x, length) => {
                let dim = if *length {
                    Dimension::LENGTH
                } else {
                    Dimension::DIMENSIONLESS
                };
                let v = self.constant(FLOATS[*x], dim);
                self.scalars.push((v, dim));
            }
            Step::Elementwise(o, i, j) => {
                if self.arrays.is_empty() {
                    return;
                }
                let a = self.arrays[i % self.arrays.len()];
                let (kind, args, dim) = match o {
                    0 | 1 => {
                        let same: Vec<Arr> = self
                            .arrays
                            .iter()
                            .filter(|x| x.shape == a.shape && x.dim == a.dim)
                            .copied()
                            .collect();
                        let b = same[j % same.len()];
                        let kind = if *o == 0 { OpKind::Add } else { OpKind::Sub };
                        (kind, vec![a.v, b.v], a.dim)
                    }
                    2 | 3 => {
                        if self.scalars.is_empty() {
                            return;
                        }
                        let (s, sd) = self.scalars[j % self.scalars.len()];
                        let (dim, kind, args) = if *o == 2 {
                            let args = if j % 2 == 0 {
                                vec![a.v, s]
                            } else {
                                vec![s, a.v]
                            };
                            (a.dim.checked_mul(sd), OpKind::Mul, args)
                        } else {
                            (a.dim.checked_div(sd), OpKind::Div, vec![a.v, s])
                        };
                        match dim {
                            Ok(d) if allowed(d) => (kind, args, d),
                            _ => return,
                        }
                    }
                    _ => (OpKind::Neg, vec![a.v], a.dim),
                };
                let (shape, ..) = shape_of(a.shape);
                let v = self.op1(kind, args, ty(dim, shape));
                self.arrays.push(Arr {
                    v,
                    shape: a.shape,
                    dim,
                });
            }
            Step::ScalarOp(o, i, j) => {
                if self.scalars.is_empty() {
                    return;
                }
                let (a, da) = self.scalars[i % self.scalars.len()];
                let (b, db) = self.scalars[j % self.scalars.len()];
                let (kind, dim) = match o {
                    0 | 1 if da == db => (if *o == 0 { OpKind::Add } else { OpKind::Sub }, da),
                    2 => match da.checked_mul(db) {
                        Ok(d) if allowed(d) => (OpKind::Mul, d),
                        _ => return,
                    },
                    3 => match da.checked_div(db) {
                        Ok(d) if allowed(d) => (OpKind::Div, d),
                        _ => return,
                    },
                    _ => return,
                };
                let v = self.op1(kind, vec![a, b], ty(dim, Shape::scalar()));
                self.scalars.push((v, dim));
            }
            Step::Index(i, k) => {
                let vectors: Vec<Arr> = self
                    .arrays
                    .iter()
                    .filter(|a| a.shape < 2)
                    .copied()
                    .collect();
                if vectors.is_empty() {
                    return;
                }
                let a = vectors[i % vectors.len()];
                let pos = self.constant(f64::from(*k), Dimension::DIMENSIONLESS);
                let v = self.op1(OpKind::Index, vec![a.v, pos], ty(a.dim, Shape::scalar()));
                self.scalars.push((v, a.dim));
            }
            Step::Print(i) => {
                let all = self.arrays.len() + self.scalars.len();
                if all == 0 {
                    return;
                }
                let k = i % all;
                let v = if k < self.arrays.len() {
                    self.arrays[k].v
                } else {
                    self.scalars[k - self.arrays.len()].0
                };
                let s = self.span();
                self.b.op(OpKind::Print, vec![v], vec![], s);
            }
        }
    }
}

/// `<main>` runs the steps, prints a few values, and returns the last array
/// and the last scalar, so the results can be compared bit for bit.
fn build(steps: &[Step]) -> Module {
    let mut g = Gen {
        b: FunctionBuilder::new(Module::MAIN, vec![], vec![], Span::point(0)),
        spans: 0,
        arrays: Vec::new(),
        scalars: Vec::new(),
    };
    for st in steps {
        g.step(st);
    }
    let mut results = Vec::new();
    let mut types = Vec::new();
    if let Some(a) = g.arrays.last().copied() {
        results.push(a.v);
        types.push(ty(a.dim, shape_of(a.shape).0));
    }
    if let Some((s, d)) = g.scalars.last().copied() {
        results.push(s);
        types.push(ty(d, Shape::scalar()));
    }
    g.b.op(OpKind::Return, results, vec![], Span::point(0));
    let mut f = g.b.finish();
    f.results = types;
    let mut m = Module {
        functions: vec![f],
        ..Module::default()
    };
    let _ = infer_effects(&mut m);
    m
}

/// Everything a run does: the output text, then the results (`Debug` tells
/// `-0.0` from `0.0`) or the error's message and span.
type Outcome = (String, Result<String, (String, Span)>);

fn outcome(m: &Module) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::call(m, Module::MAIN, Vec::new(), &mut out);
        (
            String::from_utf8(out).unwrap(),
            r.map(|v| format!("{v:?}"))
                .map_err(|e| (e.message.clone(), e.span)),
        )
    })
}

/// Fuses `steps`' program and compares the runs. Returns whether anything
/// was fused.
fn check(steps: &[Step]) -> Result<bool, TestCaseError> {
    let before = build(steps);
    verify(&before).map_err(|e| TestCaseError::fail(format!("{e:#?}\n{}", print(&before))))?;
    let mut after = before.clone();
    let records = fuse(&mut after);
    verify(&after).map_err(|e| TestCaseError::fail(format!("{e:#?}\n{}", print(&after))))?;
    prop_assert_eq!(
        outcome(&before),
        outcome(&after),
        "\nbefore:\n{}\nafter:\n{}",
        print(&before),
        print(&after)
    );
    // The result is stable: nothing more to fuse, and the text round-trips.
    let again = fuse(&mut after);
    prop_assert!(again.is_empty());
    let text = print(&after);
    let back = parse(&text).map_err(|e| TestCaseError::fail(format!("{e}\n{text}")))?;
    prop_assert_eq!(print(&back), text);
    Ok(!records.is_empty())
}

#[test]
fn fusing_random_elementwise_programs_changes_nothing_observable() {
    // A fixed seed: the coverage floors below count cases, and with a fresh
    // random seed per run they hovered at the floor (292 fused, one run).
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 1500,
            ..Config::default()
        },
        proptest::test_runner::TestRng::deterministic_rng(
            proptest::test_runner::RngAlgorithm::ChaCha,
        ),
    );
    let (fused_cases, failing_cases) = (Cell::new(0), Cell::new(0));
    runner
        .run(&program(), |steps| {
            fused_cases.set(fused_cases.get() + usize::from(check(&steps)?));
            failing_cases
                .set(failing_cases.get() + usize::from(outcome(&build(&steps)).1.is_err()));
            Ok(())
        })
        .unwrap();
    // The property is only worth something if it fused, and failed, often.
    let (fused_cases, failing_cases) = (fused_cases.get(), failing_cases.get());
    assert!(fused_cases >= 300, "only {fused_cases} programs fused");
    assert!(failing_cases >= 100, "only {failing_cases} programs failed");
}

// ----- operand spans -----

#[test]
fn a_kernel_keeps_the_operand_spans_of_its_body_and_says_where_its_leaves_were_read() {
    let text = format!(
        "{HEAD}\nfn @f(%a: num[1; (3)], %s: num[1; ()], %z: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %t1 = mul %a, %s : num[1; (3)] @ 10..20 (10..11, 14..15)
  %t2 = div %t1, %z : num[1; (3)] @ 10..30 (10..20, 24..25)
  %c = neg %t2 : num[1; (3)] @ 9..30 (10..30)
  return %c
}}\n"
    );
    let before = parse(&text).unwrap();
    let mut after = before.clone();
    assert_eq!(fuse(&mut after).len(), 1);
    verify(&after).unwrap_or_else(|e| panic!("{e:#?}\n{}", print(&after)));
    let f = &after.functions[0];
    let kernel = f
        .ops
        .iter()
        .find(|o| o.kind == OpKind::Fused && o.parent == f.entry())
        .expect("a kernel");
    // The leaves are `a`, `s` and `z`, first read at these places.
    assert_eq!(
        kernel.operand_spans,
        [Span::new(10, 11), Span::new(14, 15), Span::new(24, 25)]
    );
    let body = f.region(kernel.regions[0]).blocks[0];
    let spans: Vec<Vec<Span>> = f
        .block(body)
        .ops
        .iter()
        .map(|o| f.op(*o).operand_spans.clone())
        .collect();
    assert_eq!(
        spans,
        [
            vec![Span::new(10, 11), Span::new(14, 15)],
            vec![Span::new(10, 20), Span::new(24, 25)],
            vec![Span::new(10, 30)],
            vec![],
        ]
    );
    // So the error of the kernel is the unfused chain's, with its labels.
    let args = |z: f64| {
        vec![
            vector(&[1.0, 2.0, 3.0]),
            sky_ir_interp::Value::scalar(2.0),
            sky_ir_interp::Value::scalar(z),
        ]
    };
    let diagnostic = |m: &Module| {
        sky_ast::with_deep_stack(|| {
            sky_ir_interp::call(m, "f", args(0.0), &mut Vec::new())
                .expect_err("divides by zero")
                .to_diagnostic()
        })
    };
    assert_eq!(diagnostic(&after), diagnostic(&before));
    let d = diagnostic(&after);
    assert_eq!(d.message, "division by zero");
    assert_eq!(d.primary.span, Span::new(10, 30));
    assert_eq!(d.secondary.len(), 1);
    assert_eq!(d.secondary[0].span, Span::new(24, 25));
}

#[test]
fn a_kernel_of_operations_without_operand_spans_has_none() {
    let (m, _) = fused(
        "  %t = mul %a, %s : num[1; (3)] @ 10..11
  %c = add %t, %d : num[1; (3)] @ 12..13
  return %c",
    );
    let f = &m.functions[0];
    let kernel = f
        .ops
        .iter()
        .find(|o| o.kind == OpKind::Fused && o.parent == f.entry())
        .expect("a kernel");
    assert!(kernel.operand_spans.is_empty());
}
