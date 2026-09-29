//! Hand-built IR for every operation, control flow, calls, the error
//! paths, malformed IR, and a fuzz test that no input panics.

use sky_diag::Span;
use sky_ir::{
    BlockId, CmpOp, Function, FunctionBuilder, Module, OpId, OpKind, RegionId, SolveMethod, Type,
    ValueId,
};
use sky_units::{Dimension, Rational};

use crate::{call, run, IrRuntimeError, Value};

fn sp() -> Span {
    Span::point(0)
}

fn num() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

fn module(functions: Vec<Function>) -> Module {
    Module {
        functions,
        ..Module::default()
    }
}

/// A builder for `<main>`, which takes and returns nothing.
fn main_builder() -> FunctionBuilder {
    FunctionBuilder::new(Module::MAIN, vec![], vec![], sp())
}

/// Ends `<main>` and wraps it in a module.
fn finish_main(mut b: FunctionBuilder) -> Module {
    b.op(OpKind::Return, vec![], vec![], sp());
    module(vec![b.finish()])
}

fn constant(b: &mut FunctionBuilder, x: f64) -> ValueId {
    b.op1(
        OpKind::ConstFloat { bits: x.to_bits() },
        vec![],
        num(),
        sp(),
    )
}

/// Runs `<main>` on the deep stack every caller must use; the output text
/// and the result.
fn run_main(m: &Module) -> (Result<(), IrRuntimeError>, String) {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = run(m, &mut out);
        (r, String::from_utf8(out).expect("utf-8 output"))
    })
}

fn output(m: &Module) -> String {
    let (r, out) = run_main(m);
    r.expect("runs");
    out
}

fn error(m: &Module) -> IrRuntimeError {
    run_main(m).0.expect_err("fails")
}

fn call_fn(m: &Module, name: &str, args: Vec<Value>) -> Result<Vec<Value>, IrRuntimeError> {
    sky_ast::with_deep_stack(|| call(m, name, args, &mut Vec::new()))
}

/// Applies `kind` to `args` in a function whose parameters and result are
/// `Unknown`, so every check is the dynamic one.
fn apply(kind: OpKind, args: Vec<Value>) -> Result<Value, IrRuntimeError> {
    let unknown = vec![Type::Unknown; args.len()];
    let mut b = FunctionBuilder::new("f", unknown, vec![Type::Unknown], sp());
    let params = b.params();
    let r = b.op1(kind, params, Type::Unknown, sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    call_fn(&m, "f", args).map(|mut v| v.remove(0))
}

fn n(x: f64) -> Value {
    Value::scalar(x)
}

fn msg(e: &IrRuntimeError) -> &str {
    &e.message
}

#[test]
fn constants() {
    let mut b = main_builder();
    let len = Type::scalar(Dimension::LENGTH);
    let x = b.op1(
        OpKind::ConstFloat {
            bits: 2.5f64.to_bits(),
        },
        vec![],
        len,
        sp(),
    );
    let q = b.op1(
        OpKind::ConstRational(Rational::new(1, 4)),
        vec![],
        num(),
        sp(),
    );
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let s = b.op1(OpKind::ConstStr("a\nb".into()), vec![], Type::Str, sp());
    let z = b.op1(OpKind::ConstNothing, vec![], Type::Nothing, sp());
    b.op(OpKind::Print, vec![x, q, t, s, z], vec![], sp());
    assert_eq!(output(&finish_main(b)), "2.5 m 0.25 true a\nb none\n");
}

#[test]
fn arithmetic_ops() {
    assert_eq!(apply(OpKind::Add, vec![n(2.0), n(3.0)]), Ok(n(5.0)));
    assert_eq!(apply(OpKind::Sub, vec![n(2.0), n(3.0)]), Ok(n(-1.0)));
    assert_eq!(apply(OpKind::Mul, vec![n(2.0), n(3.0)]), Ok(n(6.0)));
    assert_eq!(apply(OpKind::Div, vec![n(3.0), n(2.0)]), Ok(n(1.5)));
    assert_eq!(apply(OpKind::Neg, vec![n(3.0)]), Ok(n(-3.0)));
    assert_eq!(apply(OpKind::Inverse, vec![n(4.0)]), Ok(n(0.25)));
    assert_eq!(apply(OpKind::Pow, vec![n(2.0), n(10.0)]), Ok(n(1024.0)));
    let square = OpKind::PowConst {
        exponent: Rational::integer(2),
    };
    assert_eq!(apply(square, vec![n(3.0)]), Ok(n(9.0)));
}

#[test]
fn dimensions_follow_the_operands() {
    let m = Value::number(3.0, Dimension::LENGTH);
    let t = Value::number(2.0, Dimension::TIME);
    let v = apply(OpKind::Div, vec![m.clone(), t.clone()]).expect("divides");
    assert_eq!(v, Value::number(1.5, Dimension::VELOCITY));
    let e = apply(OpKind::Add, vec![m.clone(), t]).expect_err("mismatched");
    assert_eq!(msg(&e), "cannot add length and time");
    let inv = apply(OpKind::Inverse, vec![m.clone()]).expect("inverts");
    assert_eq!(inv.to_string(), "0.3333333333333333 m^-1");
    // A base with a unit needs the constant exponent to type the result.
    let sq = OpKind::PowConst {
        exponent: Rational::integer(2),
    };
    assert_eq!(
        apply(sq, vec![m.clone()]).expect("squares").to_string(),
        "9 m^2"
    );
    let e = apply(OpKind::Pow, vec![m, n(2.0)]).expect_err("no constant exponent");
    assert_eq!(
        msg(&e),
        "the exponent must be a constant number when the base has a unit"
    );
}

#[test]
fn power_selection_and_errors() {
    // `powi` is exact where `powf` is not.
    assert_eq!(
        apply(OpKind::Pow, vec![n(10.0), n(-2.0)]),
        Ok(n(10f64.powi(-2)))
    );
    assert_eq!(
        apply(OpKind::Pow, vec![n(2.0), n(0.5)]),
        Ok(n(2f64.powf(0.5)))
    );
    // Beyond POWI_MAX the exponent goes through `powf`.
    assert_eq!(
        apply(OpKind::Pow, vec![n(1.0001), n(100.0)]),
        Ok(n(1.0001f64.powf(100.0)))
    );
    let e = apply(OpKind::Pow, vec![n(-8.0), n(1.0 / 3.0)]).expect_err("not real");
    assert_eq!(msg(&e), "the result is not a real number");
    let e = apply(OpKind::Pow, vec![n(0.0), n(-1.0)]).expect_err("zero");
    assert_eq!(msg(&e), "division by zero");
    let e = apply(OpKind::Pow, vec![n(10.0), n(400.0)]).expect_err("overflow");
    assert_eq!(msg(&e), "arithmetic overflow");
}

#[test]
fn comparisons_and_not() {
    let cases = [
        (CmpOp::Eq, 1.0, 1.0, true),
        (CmpOp::Ne, 1.0, 1.0, false),
        (CmpOp::Lt, 1.0, 2.0, true),
        (CmpOp::Le, 2.0, 2.0, true),
        (CmpOp::Gt, 1.0, 2.0, false),
        (CmpOp::Ge, 2.0, 2.0, true),
    ];
    for (c, a, b, want) in cases {
        assert_eq!(
            apply(OpKind::Cmp(c), vec![n(a), n(b)]),
            Ok(Value::Bool(want))
        );
    }
    let strs = vec![Value::Str("a".into()), Value::Str("a".into())];
    assert_eq!(
        apply(OpKind::Cmp(CmpOp::Eq), strs.clone()),
        Ok(Value::Bool(true))
    );
    let e = apply(OpKind::Cmp(CmpOp::Lt), strs).expect_err("strings do not order");
    assert_eq!(msg(&e), "cannot order a string and a string with `<`");
    let e = apply(
        OpKind::Cmp(CmpOp::Eq),
        vec![n(1.0), Value::number(1.0, Dimension::TIME)],
    )
    .expect_err("dimensions");
    assert_eq!(msg(&e), "cannot compare dimensionless and time");
    assert_eq!(
        apply(OpKind::Not, vec![Value::Bool(false)]),
        Ok(Value::Bool(true))
    );
    let e = apply(OpKind::Not, vec![n(1.0)]).expect_err("no truthiness");
    assert_eq!(msg(&e), "`not` needs a Bool operand, found a number");
}

#[test]
fn array_operations_are_not_supported_yet() {
    for kind in [
        OpKind::MatMul,
        OpKind::Transpose,
        OpKind::Solve {
            method: SolveMethod::Cholesky,
        },
    ] {
        let e = apply(kind, vec![n(1.0), n(2.0)]).expect_err("arrays");
        assert_eq!(msg(&e), "arrays are not supported by the interpreter yet");
    }
    // An op typed as working on an array is refused the same way.
    let vec3 = Type::Number(sky_ir::NumberType {
        shape: sky_types::Shape::vector(3u64),
        ..sky_ir::NumberType::scalar(Dimension::DIMENSIONLESS)
    });
    let mut b = FunctionBuilder::new(
        "f",
        vec![vec3.clone(), vec3.clone()],
        vec![vec3.clone()],
        sp(),
    );
    let [x, y] = b.params()[..] else {
        unreachable!()
    };
    let r = b.op1(OpKind::Add, vec![x, y], vec3, sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let e = call_fn(&module(vec![b.finish()]), "f", vec![n(1.0), n(2.0)]).expect_err("arrays");
    assert_eq!(msg(&e), "arrays are not supported by the interpreter yet");
}

#[test]
fn errors_match_sky_run_wording() {
    let e = apply(OpKind::Div, vec![n(1.0), n(0.0)]).expect_err("zero");
    assert_eq!(msg(&e), "division by zero");
    let e = apply(OpKind::Inverse, vec![n(0.0)]).expect_err("zero");
    assert_eq!(msg(&e), "division by zero");
    let e = apply(OpKind::Mul, vec![n(1e200), n(1e200)]).expect_err("overflow");
    assert_eq!(msg(&e), "arithmetic overflow");
    let e = apply(OpKind::Add, vec![n(f64::MAX), n(f64::MAX)]).expect_err("overflow");
    assert_eq!(msg(&e), "arithmetic overflow");
    let e = apply(OpKind::Add, vec![n(1.0), Value::Bool(true)]).expect_err("kinds");
    assert_eq!(msg(&e), "cannot add a number and a Bool");
    let e = apply(OpKind::Sub, vec![n(1.0), Value::Bool(true)]).expect_err("kinds");
    assert_eq!(msg(&e), "cannot subtract a Bool from a number");
    let e = apply(OpKind::Neg, vec![Value::Bool(true)]).expect_err("kinds");
    assert_eq!(msg(&e), "`-` needs a number, found a Bool");
    // The diagnostic carries the message.
    assert_eq!(
        e.to_diagnostic().message,
        "`-` needs a number, found a Bool"
    );
}

#[test]
fn error_carries_the_failing_ops_span() {
    let at = Span::new(10, 15);
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    let zero = constant(&mut b, 0.0);
    b.op1(OpKind::Div, vec![one, zero], num(), at);
    let e = error(&finish_main(b));
    assert_eq!(e.span, at);
    assert_eq!(e.to_diagnostic().primary.span, at);
}

#[test]
fn print_writes_a_line_per_op_and_units_print_in_si() {
    let mut b = main_builder();
    let v = b.op1(
        OpKind::ConstFloat {
            bits: 5.0f64.to_bits(),
        },
        vec![],
        Type::scalar(Dimension::VELOCITY),
        sp(),
    );
    b.op(OpKind::Print, vec![v], vec![], sp());
    b.op(OpKind::Print, vec![], vec![], sp());
    assert_eq!(output(&finish_main(b)), "5 m/s\n\n");
}

#[test]
fn print_failure_leaves_no_partial_line() {
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    let zero = constant(&mut b, 0.0);
    // The quotient never exists, so the print never runs.
    let q = b.op1(OpKind::Div, vec![one, zero], num(), sp());
    b.op(OpKind::Print, vec![one, q], vec![], sp());
    let (r, out) = run_main(&finish_main(b));
    assert!(r.is_err());
    assert_eq!(out, "");
}

/// A writer that always fails.
struct Broken;

impl std::io::Write for Broken {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("closed"))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn output_failure_is_an_error() {
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    b.op(OpKind::Print, vec![one], vec![], sp());
    let m = finish_main(b);
    let e = sky_ast::with_deep_stack(|| run(&m, &mut Broken)).expect_err("fails");
    assert_eq!(msg(&e), "cannot write output: closed");
}

/// Builds `if c then a else b` where both arms yield one value each, at the
/// builder's insertion point; `arms` fills an arm and returns its yield.
fn build_if(
    b: &mut FunctionBuilder,
    cond: ValueId,
    ty: Type,
    arms: [&mut dyn FnMut(&mut FunctionBuilder) -> ValueId; 2],
) -> ValueId {
    let (iff, res) = b.op(OpKind::If, vec![cond], vec![ty], sp());
    let outer = b.insertion_block();
    for arm in arms {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let v = arm(b);
        b.op(OpKind::Yield, vec![v], vec![], sp());
    }
    b.set_insertion_block(outer);
    res[0]
}

#[test]
fn if_with_results_takes_one_arm_only() {
    // f(c) = if c then 1 else 1/0: the failing arm must not run.
    let mut b = FunctionBuilder::new("f", vec![Type::Bool], vec![num()], sp());
    let c = b.params()[0];
    let r = build_if(
        &mut b,
        c,
        num(),
        [&mut |b| constant(b, 1.0), &mut |b| {
            let one = constant(b, 1.0);
            let zero = constant(b, 0.0);
            b.op1(OpKind::Div, vec![one, zero], num(), sp())
        }],
    );
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    assert_eq!(call_fn(&m, "f", vec![Value::Bool(true)]), Ok(vec![n(1.0)]));
    let e = call_fn(&m, "f", vec![Value::Bool(false)]).expect_err("else arm runs");
    assert_eq!(msg(&e), "division by zero");
    let e = call_fn(&m, "f", vec![n(1.0)]).expect_err("not a Bool");
    assert_eq!(msg(&e), "`if` needs a Bool operand, found a number");
}

#[test]
fn nested_if_with_results() {
    // classify(x) = if x < 0 then -1 else (if x == 0 then 0 else 1)
    let mut b = FunctionBuilder::new("classify", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let zero = constant(&mut b, 0.0);
    let neg = b.op1(OpKind::Cmp(CmpOp::Lt), vec![x, zero], Type::Bool, sp());
    let r = build_if(
        &mut b,
        neg,
        num(),
        [&mut |b| constant(b, -1.0), &mut |b| {
            let is0 = b.op1(OpKind::Cmp(CmpOp::Eq), vec![x, zero], Type::Bool, sp());
            build_if(
                b,
                is0,
                num(),
                [&mut |b| constant(b, 0.0), &mut |b| constant(b, 1.0)],
            )
        }],
    );
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    for (x, want) in [(-5.0, -1.0), (0.0, 0.0), (7.0, 1.0)] {
        assert_eq!(call_fn(&m, "classify", vec![n(x)]), Ok(vec![n(want)]));
    }
}

#[test]
fn if_can_yield_several_values_or_none() {
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, res) = b.op(OpKind::If, vec![t], vec![num(), Type::Str], sp());
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let x = constant(&mut b, 4.0);
        let s = b.op1(OpKind::ConstStr("s".into()), vec![], Type::Str, sp());
        b.op(OpKind::Yield, vec![x, s], vec![], sp());
    }
    b.set_insertion_block(outer);
    b.op(OpKind::Print, res, vec![], sp());
    // An `if` with no results (an `if` statement).
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![], vec![], sp());
    }
    b.set_insertion_block(outer);
    assert_eq!(output(&finish_main(b)), "4 s\n");
}

/// `sum = 0; i = 1; while i <= limit: sum += i; i += 1; return sum`.
fn sum_module(limit: f64) -> Module {
    let mut b = FunctionBuilder::new("sum", vec![], vec![num()], sp());
    let i0 = constant(&mut b, 1.0);
    let s0 = constant(&mut b, 0.0);
    let (w, res) = b.op(OpKind::While, vec![i0, s0], vec![num(), num()], sp());
    let outer = b.insertion_block();
    let before = b.add_region(w);
    let blk = b.add_block(before, vec![num(), num()]);
    b.set_insertion_block(blk);
    let [i, s] = b.func().block(blk).args[..] else {
        unreachable!()
    };
    let lim = constant(&mut b, limit);
    let c = b.op1(OpKind::Cmp(CmpOp::Le), vec![i, lim], Type::Bool, sp());
    b.op(OpKind::Condition, vec![c, i, s], vec![], sp());
    let after = b.add_region(w);
    let blk = b.add_block(after, vec![num(), num()]);
    b.set_insertion_block(blk);
    let [i, s] = b.func().block(blk).args[..] else {
        unreachable!()
    };
    let s2 = b.op1(OpKind::Add, vec![s, i], num(), sp());
    let one = constant(&mut b, 1.0);
    let i2 = b.op1(OpKind::Add, vec![i, one], num(), sp());
    b.op(OpKind::Yield, vec![i2, s2], vec![], sp());
    b.set_insertion_block(outer);
    b.op(OpKind::Return, vec![res[1]], vec![], sp());
    module(vec![b.finish()])
}

#[test]
fn while_sums_one_to_ten_with_loop_carried_values() {
    let m = sum_module(10.0);
    assert_eq!(call_fn(&m, "sum", vec![]), Ok(vec![n(55.0)]));
    // A loop that never runs forwards its initial values.
    let m = sum_module(0.0);
    assert_eq!(call_fn(&m, "sum", vec![]), Ok(vec![n(0.0)]));
}

#[test]
fn while_condition_must_be_a_bool() {
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    let (w, _) = b.op(OpKind::While, vec![], vec![], sp());
    let outer = b.insertion_block();
    let before = b.add_region(w);
    let blk = b.add_block(before, vec![]);
    b.set_insertion_block(blk);
    b.op(OpKind::Condition, vec![one], vec![], sp());
    let after = b.add_region(w);
    let blk = b.add_block(after, vec![]);
    b.set_insertion_block(blk);
    b.op(OpKind::Yield, vec![], vec![], sp());
    b.set_insertion_block(outer);
    let e = error(&finish_main(b));
    assert_eq!(msg(&e), "`while` needs a Bool operand, found a number");
}

/// `fact(n) = if n <= 1 then 1 else n * fact(n - 1)`.
fn factorial_module() -> Module {
    let mut b = FunctionBuilder::new("fact", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let one = constant(&mut b, 1.0);
    let base = b.op1(OpKind::Cmp(CmpOp::Le), vec![x, one], Type::Bool, sp());
    let r = build_if(
        &mut b,
        base,
        num(),
        [&mut |b| constant(b, 1.0), &mut |b| {
            let one = constant(b, 1.0);
            let m1 = b.op1(OpKind::Sub, vec![x, one], num(), sp());
            let f = b.op1(
                OpKind::Call {
                    callee: "fact".into(),
                },
                vec![m1],
                num(),
                sp(),
            );
            b.op1(OpKind::Mul, vec![x, f], num(), sp())
        }],
    );
    b.op(OpKind::Return, vec![r], vec![], sp());
    module(vec![b.finish()])
}

#[test]
fn recursion_through_call_computes_factorial() {
    let m = factorial_module();
    assert_eq!(call_fn(&m, "fact", vec![n(10.0)]), Ok(vec![n(3628800.0)]));
    assert_eq!(call_fn(&m, "fact", vec![n(1.0)]), Ok(vec![n(1.0)]));
    // 200! overflows f64 on the way up.
    let e = call_fn(&m, "fact", vec![n(200.0)]).expect_err("overflow");
    assert_eq!(msg(&e), "arithmetic overflow");
}

#[test]
fn call_from_main_prints_the_result() {
    let mut fact = factorial_module();
    let mut b = main_builder();
    let five = constant(&mut b, 5.0);
    let r = b.op1(
        OpKind::Call {
            callee: "fact".into(),
        },
        vec![five],
        num(),
        sp(),
    );
    b.op(OpKind::Print, vec![r], vec![], sp());
    b.op(OpKind::Return, vec![], vec![], sp());
    fact.functions.push(b.finish());
    assert_eq!(output(&fact), "120\n");
}

#[test]
fn a_function_without_a_return_value_gives_nothing() {
    let mut g = FunctionBuilder::new("g", vec![], vec![Type::Nothing], sp());
    let z = g.op1(OpKind::ConstNothing, vec![], Type::Nothing, sp());
    g.op(OpKind::Return, vec![z], vec![], sp());
    let mut b = main_builder();
    let r = b.op1(
        OpKind::Call { callee: "g".into() },
        vec![],
        Type::Nothing,
        sp(),
    );
    b.op(OpKind::Print, vec![r], vec![], sp());
    b.op(OpKind::Return, vec![], vec![], sp());
    assert_eq!(output(&module(vec![b.finish(), g.finish()])), "none\n");
}

#[test]
fn runaway_recursion_is_a_clean_error() {
    // f(x) = f(x)
    let mut b = FunctionBuilder::new("f", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let r = b.op1(OpKind::Call { callee: "f".into() }, vec![x], num(), sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    let e = call_fn(&m, "f", vec![n(1.0)]).expect_err("too deep");
    assert_eq!(
        msg(&e),
        "recursion is too deep: more than 1000 nested calls"
    );
}

#[test]
fn recursion_deep_inside_regions_still_stops_cleanly() {
    // Each level runs inside an `if` region, so the Rust stack grows faster
    // than the call count; either limit is a clean error, never an overflow.
    let mut b = FunctionBuilder::new("f", vec![Type::Bool], vec![num()], sp());
    let c = b.params()[0];
    let r = build_if(
        &mut b,
        c,
        num(),
        [
            &mut |b| b.op1(OpKind::Call { callee: "f".into() }, vec![c], num(), sp()),
            &mut |b| constant(b, 0.0),
        ],
    );
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    let e = call_fn(&m, "f", vec![Value::Bool(true)]).expect_err("too deep");
    assert!(
        msg(&e).starts_with("recursion is too deep")
            || msg(&e) == "the program nests too deeply for the interpreter",
        "{}",
        msg(&e)
    );
}

#[test]
fn call_checks_arity() {
    let m = factorial_module();
    let e = call_fn(&m, "fact", vec![]).expect_err("arity");
    assert_eq!(msg(&e), "`fact` takes 1 argument, found 0");
}

// ----- malformed IR: errors, never panics -----

fn assert_malformed(m: &Module, what: &str) {
    let e = error(m);
    assert!(
        msg(&e).starts_with("malformed IR: ") && msg(&e).contains(what),
        "expected malformed IR mentioning {what:?}, got {:?}",
        msg(&e)
    );
}

#[test]
fn malformed_missing_main_and_unknown_callee() {
    assert_malformed(&module(vec![]), "unknown function");
    let mut b = main_builder();
    b.op(
        OpKind::Call {
            callee: "nope".into(),
        },
        vec![],
        vec![],
        sp(),
    );
    assert_malformed(&finish_main(b), "unknown function `nope`");
}

#[test]
fn malformed_wrong_operand_counts() {
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    b.op1(OpKind::Add, vec![one], num(), sp());
    assert_malformed(&finish_main(b), "`add` takes 2 operands, found 1");
    let mut b = main_builder();
    b.op1(OpKind::Neg, vec![], num(), sp());
    assert_malformed(&finish_main(b), "`neg` takes 1 operands, found 0");
}

#[test]
fn malformed_undefined_and_out_of_range_values() {
    let mut b = main_builder();
    b.op(OpKind::Print, vec![ValueId(99)], vec![], sp());
    assert_malformed(&finish_main(b), "%99 is used before it is defined");
    // A value defined by a later op: the first result of a builder with no
    // parameters is `%0`.
    let mut b = main_builder();
    b.op(OpKind::Print, vec![ValueId(0)], vec![], sp());
    b.op1(OpKind::ConstNothing, vec![], Type::Nothing, sp());
    assert_malformed(&finish_main(b), "%0 is used before it is defined");
    // A block listing an operation that does not exist.
    let mut f = main_builder().finish();
    f.blocks[0].ops.push(OpId(77));
    assert_malformed(&module(vec![f]), "no operation op77");
}

#[test]
fn malformed_results_and_types() {
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    b.op(OpKind::Neg, vec![one], vec![], sp());
    assert_malformed(&finish_main(b), "`neg` has 0 results, expected 1");
    let mut b = main_builder();
    b.op1(
        OpKind::ConstFloat {
            bits: f64::NAN.to_bits(),
        },
        vec![],
        num(),
        sp(),
    );
    assert_malformed(&finish_main(b), "not finite");
    let mut b = main_builder();
    b.op1(
        OpKind::ConstFloat {
            bits: 1f64.to_bits(),
        },
        vec![],
        Type::Bool,
        sp(),
    );
    assert_malformed(&finish_main(b), "must have a number result");
    let mut b = main_builder();
    b.op(OpKind::Print, vec![], vec![num()], sp());
    assert_malformed(&finish_main(b), "`print` has 1 results, expected 0");
}

#[test]
fn malformed_terminators_and_regions() {
    // No terminator at all.
    let b = main_builder();
    assert_malformed(&module(vec![b.finish()]), "does not end in a terminator");
    // A body that yields.
    let mut b = main_builder();
    b.op(OpKind::Yield, vec![], vec![], sp());
    assert_malformed(&module(vec![b.finish()]), "must end in `return`");
    // Returning the wrong number of values.
    let mut b = main_builder();
    let one = constant(&mut b, 1.0);
    b.op(OpKind::Return, vec![one], vec![], sp());
    assert_malformed(&module(vec![b.finish()]), "returns 1 values");
    // `if` and `while` without their regions.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    b.op(OpKind::If, vec![t], vec![], sp());
    assert_malformed(&finish_main(b), "`if` takes 2 regions, found 0");
    let mut b = main_builder();
    b.op(OpKind::While, vec![], vec![], sp());
    assert_malformed(&finish_main(b), "`while` takes 2 regions, found 0");
    // A region with no block, and one with two.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    b.add_region(iff);
    b.add_region(iff);
    assert_malformed(&finish_main(b), "has no block");
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    let r = b.add_region(iff);
    b.add_block(r, vec![]);
    b.add_block(r, vec![]);
    b.add_region(iff);
    assert_malformed(&finish_main(b), "several blocks");
    // A region that is not in the arena.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    b.add_region(iff);
    b.add_region(iff);
    let mut f = {
        b.op(OpKind::Return, vec![], vec![], sp());
        b.finish()
    };
    f.ops[iff.index()].regions[0] = RegionId(50);
    assert_malformed(&module(vec![f]), "no region region50");
    // A block that is not in the arena.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    let r0 = b.add_region(iff);
    b.add_region(iff);
    let mut f = {
        b.op(OpKind::Return, vec![], vec![], sp());
        b.finish()
    };
    f.regions[r0.index()].blocks.push(BlockId(60));
    assert_malformed(&module(vec![f]), "no block ^60");
}

#[test]
fn malformed_wrong_terminator_kind_in_a_region() {
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Condition, vec![t], vec![], sp());
    }
    b.set_insertion_block(outer);
    assert_malformed(&finish_main(b), "must end in `yield`");
}

#[test]
fn malformed_block_argument_count() {
    // The `if` arm has an argument that nothing supplies.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (iff, _) = b.op(OpKind::If, vec![t], vec![], sp());
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![num()]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![], vec![], sp());
    }
    b.set_insertion_block(outer);
    assert_malformed(&finish_main(b), "takes 1 arguments, given 0");
}

#[test]
fn malformed_condition_and_forwarding() {
    // `condition` with no operands.
    let mut b = main_builder();
    let (w, _) = b.op(OpKind::While, vec![], vec![], sp());
    let outer = b.insertion_block();
    for term in [OpKind::Condition, OpKind::Yield] {
        let r = b.add_region(w);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(term, vec![], vec![], sp());
    }
    b.set_insertion_block(outer);
    assert_malformed(&finish_main(b), "`condition` needs a Bool operand");
    // Forwarding more values than the loop carries.
    let mut b = main_builder();
    let t = b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp());
    let (w, _) = b.op(OpKind::While, vec![], vec![], sp());
    let outer = b.insertion_block();
    for term in [OpKind::Condition, OpKind::Yield] {
        let r = b.add_region(w);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        let ops = if term == OpKind::Condition {
            vec![t, t]
        } else {
            vec![]
        };
        b.op(term, ops, vec![], sp());
    }
    b.set_insertion_block(outer);
    assert_malformed(&finish_main(b), "takes 0 arguments, given 1");
}

// ----- fuzzing: whatever the IR, the result is a value or an error -----

mod fuzz {
    use proptest::prelude::*;

    use super::*;
    use crate::machine::call_with_fuel;

    /// One op to build: everything about it is arbitrary, including operand
    /// ids that name no value.
    #[derive(Clone, Debug)]
    struct Spec {
        kind: u8,
        operands: Vec<u32>,
        results: Vec<u8>,
        regions: Vec<(Vec<u8>, Vec<Spec>)>,
    }

    fn ty_of(t: u8) -> Type {
        match t % 5 {
            0 => num(),
            1 => Type::Bool,
            2 => Type::Str,
            3 => Type::Nothing,
            _ => Type::Unknown,
        }
    }

    fn kind_of(k: u8, x: u32) -> OpKind {
        match k % 30 {
            0 => OpKind::ConstFloat {
                bits: if x % 7 == 0 {
                    u64::from(x)
                } else {
                    f64::from(x).to_bits()
                },
            },
            1 => OpKind::ConstRational(
                Rational::checked_new(i64::from(x % 9) - 4, i64::from(x % 5) + 1)
                    .unwrap_or(Rational::ONE),
            ),
            2 => OpKind::ConstBool(x % 2 == 0),
            3 => OpKind::ConstStr(format!("s{x}")),
            4 => OpKind::ConstNothing,
            5 => OpKind::Add,
            6 => OpKind::Sub,
            7 => OpKind::Mul,
            8 => OpKind::Div,
            9 => OpKind::Neg,
            10 => OpKind::PowConst {
                exponent: Rational::checked_new(i64::from(x % 9) - 4, i64::from(x % 3) + 1)
                    .unwrap_or(Rational::ONE),
            },
            11 => OpKind::Pow,
            12 => OpKind::Inverse,
            13 => OpKind::MatMul,
            14 => OpKind::Transpose,
            15 => OpKind::Solve {
                method: SolveMethod::Cholesky,
            },
            16 => OpKind::Cmp(
                [
                    CmpOp::Eq,
                    CmpOp::Ne,
                    CmpOp::Lt,
                    CmpOp::Le,
                    CmpOp::Gt,
                    CmpOp::Ge,
                ][x as usize % 6],
            ),
            17 => OpKind::Not,
            18 | 19 => OpKind::If,
            20 | 21 => OpKind::While,
            22 => OpKind::Yield,
            23 => OpKind::Condition,
            24 => OpKind::Return,
            25 => OpKind::Print,
            26 => OpKind::Call {
                callee: ["g", Module::MAIN, "missing"][x as usize % 3].into(),
            },
            _ => OpKind::ConstFloat {
                bits: f64::from(x % 11).to_bits(),
            },
        }
    }

    fn leaf() -> BoxedStrategy<Spec> {
        (
            any::<u8>(),
            prop::collection::vec(0u32..14, 0..4),
            prop::collection::vec(any::<u8>(), 0..3),
        )
            .prop_map(|(kind, operands, results)| Spec {
                kind,
                operands,
                results,
                regions: Vec::new(),
            })
            .boxed()
    }

    fn node(inner: BoxedStrategy<Spec>) -> BoxedStrategy<Spec> {
        let region = (
            prop::collection::vec(any::<u8>(), 0..3),
            prop::collection::vec(inner, 0..5),
        );
        (leaf(), prop::collection::vec(region, 0..4))
            .prop_map(|(mut s, regions)| {
                s.regions = regions;
                s
            })
            .boxed()
    }

    fn specs() -> impl Strategy<Value = Vec<Spec>> {
        let l1 = node(leaf());
        let l2 = node(l1);
        prop::collection::vec(l2, 0..10)
    }

    fn fill(b: &mut FunctionBuilder, specs: &[Spec]) {
        for s in specs {
            let x = s.operands.first().copied().unwrap_or(0);
            let (op, _) = b.op(
                kind_of(s.kind, x),
                s.operands.iter().map(|v| ValueId(*v)).collect(),
                s.results.iter().map(|t| ty_of(*t)).collect(),
                sp(),
            );
            let here = b.insertion_block();
            for (args, body) in &s.regions {
                let r = b.add_region(op);
                // Some regions get a block and some don't.
                if !body.is_empty() || args.len() == 2 {
                    let blk = b.add_block(r, args.iter().map(|t| ty_of(*t)).collect());
                    b.set_insertion_block(blk);
                    fill(b, body);
                }
            }
            b.set_insertion_block(here);
        }
    }

    fn build(main: &[Spec], g: &[Spec], end_main: bool) -> Module {
        let mut m = FunctionBuilder::new(Module::MAIN, vec![], vec![], sp());
        fill(&mut m, main);
        if end_main {
            m.op(OpKind::Return, vec![], vec![], sp());
        }
        let mut f = FunctionBuilder::new("g", vec![Type::Unknown], vec![Type::Unknown], sp());
        fill(&mut f, g);
        module(vec![m.finish(), f.finish()])
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        #[test]
        fn no_ir_makes_the_interpreter_panic(
            main in specs(),
            g in specs(),
            end_main in any::<bool>(),
            arg in -3i32..3,
        ) {
            let m = build(&main, &g, end_main);
            // A panic inside the deep-stack thread resumes here and fails
            // the case; anything else (value or error) passes.
            sky_ast::with_deep_stack(|| {
                let mut out = Vec::new();
                let _ = call_with_fuel(&m, Module::MAIN, vec![], &mut out, 5_000);
                let _ = call_with_fuel(&m, "g", vec![n(f64::from(arg))], &mut out, 5_000);
                let _ = call_with_fuel(&m, "g", vec![], &mut out, 5_000);
            });
        }
    }
}
