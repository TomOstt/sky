//! One test per verifier rule: a valid module passes, and the smallest
//! module that breaks the rule is rejected with a message naming it.

mod common;

use common::*;
use sky_ir::{
    verify, BlockId, CmpOp, Function, FunctionBuilder, Module, OpKind, Repr, SolveMethod, Type,
    ValueDef, ValueId, VerifyError,
};
use sky_types::{Shape, Tag, TagSet};
use sky_units::{Dimension, Rational};

fn errors(m: &Module) -> Vec<VerifyError> {
    verify(m).err().unwrap_or_default()
}

#[track_caller]
fn accepts(m: &Module) {
    if let Err(e) = verify(m) {
        panic!("expected a valid module, got {e:#?}\n{}", sky_ir::print(m));
    }
}

#[track_caller]
fn rejects(m: &Module, needle: &str) -> Vec<VerifyError> {
    let e = errors(m);
    assert!(
        e.iter().any(|e| e.message.contains(needle)),
        "no error containing {needle:?}; got {:#?}\n{}",
        e.iter().map(|e| e.message.clone()).collect::<Vec<_>>(),
        sky_ir::print(m)
    );
    e
}

fn one_fn(f: Function) -> Module {
    module(vec![f])
}

/// `fn f(x: L) -> (L)` with the builder left before its `return`.
fn start(params: Vec<Type>, results: Vec<Type>) -> FunctionBuilder {
    FunctionBuilder::new("f", params, results, sp(0, 100))
}

fn ret(b: &mut FunctionBuilder, vs: Vec<ValueId>) {
    b.op(OpKind::Return, vs, vec![], sp(90, 99));
}

/// `x + x` returned: the smallest valid function with one operation.
fn tiny() -> Function {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let s = b.op1(OpKind::Add, vec![x, x], len(), sp(1, 2));
    ret(&mut b, vec![s]);
    b.finish()
}

/// The operation that defines `v`.
fn def_op(f: &Function, v: ValueId) -> sky_ir::OpId {
    match f.values[v.index()].def {
        ValueDef::OpResult { op, .. } => op,
        _ => panic!("not an op result"),
    }
}

#[test]
fn a_valid_module_passes() {
    accepts(&one_fn(tiny()));
    accepts(&all_ops());
    accepts(&module(vec![]));
}

// ----- arena integrity -----

#[test]
fn operand_id_out_of_range() {
    let mut f = tiny();
    let add = def_op(&f, ValueId(1));
    f.ops[add.index()].operands[0] = ValueId(99);
    let e = rejects(&one_fn(f), "(%99) is out of range");
    assert_eq!(e[0].function, "f");
    assert_eq!(e[0].span, sp(1, 2));
    assert!(e[0].op.is_some());
}

#[test]
fn op_id_out_of_range() {
    let mut f = tiny();
    let entry = f.entry();
    f.blocks[entry.index()].ops.insert(0, sky_ir::OpId(77));
    rejects(&one_fn(f), "op77 is out of range");
}

#[test]
fn body_region_out_of_range() {
    let mut f = tiny();
    f.body = sky_ir::RegionId(9);
    rejects(&one_fn(f), "body region region9 is out of range");
}

#[test]
fn value_def_must_match_its_place() {
    let mut f = tiny();
    f.values[1].def = ValueDef::BlockArg {
        block: BlockId(0),
        index: 0,
    };
    let e = rejects(&one_fn(f), "its `def` says");
    assert!(e.iter().any(|e| e.value == Some(ValueId(1))));
}

#[test]
fn block_argument_def_must_match() {
    let mut f = tiny();
    f.values[0].def = ValueDef::BlockArg {
        block: BlockId(0),
        index: 5,
    };
    rejects(&one_fn(f), "its `def` says");
}

#[test]
fn op_parent_must_match() {
    let mut f = tiny();
    f.ops[0].parent = BlockId(3);
    rejects(&one_fn(f), "says its parent is ^3");
}

#[test]
fn block_parent_must_match() {
    let mut f = tiny();
    f.blocks[0].parent = None;
    rejects(&one_fn(f), "says its parent");
}

#[test]
fn region_parent_must_match() {
    let mut b = start(vec![Type::Bool, len()], vec![len()]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![x], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, res);
    let mut f = b.finish();
    accepts(&one_fn(f.clone()));
    f.regions[1].parent = None;
    rejects(&one_fn(f), "says its parent is None");
}

#[test]
fn body_region_must_not_have_a_parent() {
    let mut f = tiny();
    f.regions[0].parent = Some(sky_ir::OpId(0));
    rejects(&one_fn(f), "must not have a parent");
}

#[test]
fn an_operation_appears_once() {
    let mut f = tiny();
    let entry = f.entry();
    let first = f.blocks[entry.index()].ops[0];
    f.blocks[entry.index()].ops.insert(1, first);
    rejects(&one_fn(f), "appears in the tree more than once");
}

#[test]
fn a_region_appears_once_and_cycles_terminate() {
    // An operation whose region is the function body: a cycle. The verifier
    // must report it and stop.
    let mut f = tiny();
    let add = def_op(&f, ValueId(1));
    f.ops[add.index()].regions.push(f.body);
    rejects(&one_fn(f), "appears in the tree more than once");
}

#[test]
fn a_result_belongs_to_one_operation() {
    let mut f = tiny();
    let add = def_op(&f, ValueId(1));
    f.ops[add.index()].results.push(ValueId(0));
    rejects(&one_fn(f), "its `def` says");
}

#[test]
fn parameters_must_match_the_entry_block() {
    let mut f = tiny();
    f.params = vec![time()];
    rejects(&one_fn(f), "do not match the parameters");
}

// ----- SSA and scoping -----

#[test]
fn use_before_definition() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let n1 = b.op1(OpKind::Neg, vec![x], len(), sp(1, 2));
    let n2 = b.op1(OpKind::Neg, vec![n1], len(), sp(3, 4));
    ret(&mut b, vec![n2]);
    let mut f = b.finish();
    accepts(&one_fn(f.clone()));
    let entry = f.entry();
    f.blocks[entry.index()].ops.swap(0, 1);
    let e = rejects(&one_fn(f), "is used before its definition");
    assert_eq!(e[0].span, sp(3, 4));
}

#[test]
fn use_of_a_value_nobody_defines() {
    let mut f = tiny();
    // A value in the arena that no block or operation defines.
    f.values.push(sky_ir::ValueData {
        ty: len(),
        def: ValueDef::BlockArg {
            block: BlockId(0),
            index: 7,
        },
        name: None,
    });
    let add = def_op(&f, ValueId(1));
    f.ops[add.index()].operands[1] = ValueId(2);
    rejects(&one_fn(f), "used before its definition");
}

/// `if c { yield x } else { yield x }`, with `inner` optionally defined in
/// the `then` arm; returns the function builder positioned after the `if`.
fn if_with_inner() -> (FunctionBuilder, ValueId, ValueId) {
    let mut b = start(vec![Type::Bool, len()], vec![len()]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    let mut inner = x;
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        if arm == 0 {
            inner = b.op1(OpKind::Neg, vec![x], len(), sp(5, 6));
            b.op(OpKind::Yield, vec![inner], vec![], sp(0, 0));
        } else {
            b.op(OpKind::Yield, vec![x], vec![], sp(0, 0));
        }
    }
    b.set_insertion_block(outer);
    (b, res[0], inner)
}

#[test]
fn a_value_defined_in_a_region_is_not_visible_after_it() {
    let (mut b, _res, inner) = if_with_inner();
    ret(&mut b, vec![inner]);
    let e = rejects(&one_fn(b.finish()), "defined in a region that has ended");
    assert_eq!(e[0].value, Some(inner));
}

#[test]
fn an_if_result_is_not_visible_inside_its_own_arms() {
    let mut b = start(vec![Type::Bool, len()], vec![len()]);
    let [c, _x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![res[0]], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, res);
    rejects(&one_fn(b.finish()), "is used before its definition");
}

#[test]
fn enclosing_scopes_are_visible_in_nested_regions() {
    // `x` is a parameter and `eps` an earlier value of the outer block; both
    // are used two regions down. (`all_ops` covers loops.)
    let mut b = start(vec![Type::Bool, len()], vec![len()]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let eps = b.op1(OpKind::Neg, vec![x], len(), sp(0, 0));
    let (outer_if, res) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 0));
    let top = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(outer_if);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        if arm == 0 {
            let (inner_if, ires) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 0));
            let mid = b.insertion_block();
            for _ in 0..2 {
                let r2 = b.add_region(inner_if);
                let b2 = b.add_block(r2, vec![]);
                b.set_insertion_block(b2);
                let s = b.op1(OpKind::Add, vec![x, eps], len(), sp(0, 0));
                b.op(OpKind::Yield, vec![s], vec![], sp(0, 0));
            }
            b.set_insertion_block(mid);
            b.op(OpKind::Yield, ires, vec![], sp(0, 0));
        } else {
            b.op(OpKind::Yield, vec![eps], vec![], sp(0, 0));
        }
    }
    b.set_insertion_block(top);
    ret(&mut b, res);
    accepts(&one_fn(b.finish()));
}

#[test]
fn while_before_values_are_not_visible_in_after() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (w, res) = b.op(OpKind::While, vec![x], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    let before = b.add_region(w);
    let bb = b.add_block(before, vec![len()]);
    b.set_insertion_block(bb);
    let i = b.func().block(bb).args[0];
    let go = b.op1(OpKind::ConstBool(false), vec![], Type::Bool, sp(0, 0));
    let tmp = b.op1(OpKind::Neg, vec![i], len(), sp(7, 8));
    b.op(OpKind::Condition, vec![go, i], vec![], sp(0, 0));
    let after = b.add_region(w);
    let ab = b.add_block(after, vec![len()]);
    b.set_insertion_block(ab);
    // Uses `tmp` from the `before` region instead of its own argument.
    b.op(OpKind::Yield, vec![tmp], vec![], sp(0, 0));
    b.set_insertion_block(outer);
    ret(&mut b, res);
    rejects(&one_fn(b.finish()), "defined in a region that has ended");
}

#[test]
fn a_definition_is_unique() {
    // Two operations claiming the same result value.
    let mut f = tiny();
    let entry = f.entry();
    let add = def_op(&f, ValueId(1));
    let copy = f.ops[add.index()].clone();
    f.ops.push(copy);
    let new_op = sky_ir::OpId(f.ops.len() as u32 - 1);
    f.blocks[entry.index()].ops.insert(1, new_op);
    rejects(&one_fn(f), "its `def` says");
}

// ----- region shape and terminators -----

#[test]
fn a_block_must_end_in_a_terminator() {
    let mut b = start(vec![len()], vec![]);
    let x = b.params()[0];
    b.op1(OpKind::Neg, vec![x], len(), sp(1, 2));
    rejects(&one_fn(b.finish()), "does not end with a terminator");
}

#[test]
fn an_empty_block_is_rejected() {
    let b = start(vec![], vec![]);
    rejects(&one_fn(b.finish()), "is empty");
}

#[test]
fn a_terminator_must_be_last() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    ret(&mut b, vec![x]);
    let n = b.op1(OpKind::Neg, vec![x], len(), sp(0, 0));
    ret(&mut b, vec![n]);
    rejects(&one_fn(b.finish()), "is a terminator but is not the last");
}

#[test]
fn a_region_has_one_block() {
    let mut f = tiny();
    f.blocks.push(sky_ir::BlockData {
        args: vec![],
        ops: vec![],
        parent: Some(f.body),
    });
    f.regions[0].blocks.push(BlockId(1));
    rejects(&one_fn(f), "exactly one block in v1, found 2");
}

#[test]
fn an_empty_region_is_rejected() {
    let mut f = tiny();
    f.regions[0].blocks.clear();
    rejects(&one_fn(f), "exactly one block in v1, found 0");
}

#[test]
fn return_only_ends_the_function_body() {
    let mut b = start(vec![Type::Bool, len()], vec![len()]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    for arm in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        if arm == 0 {
            b.op(OpKind::Return, vec![x], vec![], sp(4, 5));
        } else {
            b.op(OpKind::Yield, vec![x], vec![], sp(0, 0));
        }
    }
    b.set_insertion_block(outer);
    ret(&mut b, res);
    let e = rejects(
        &one_fn(b.finish()),
        "`return` is not allowed here: this region must end in `yield`",
    );
    assert_eq!(e[0].span, sp(4, 5));
}

#[test]
fn the_body_must_end_in_return() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    b.op(OpKind::Yield, vec![x], vec![], sp(0, 0));
    rejects(
        &one_fn(b.finish()),
        "`yield` is not allowed here: this region must end in `return`",
    );
}

#[test]
fn a_while_before_region_must_end_in_condition() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (w, res) = b.op(OpKind::While, vec![x], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(w);
        let blk = b.add_block(r, vec![len()]);
        b.set_insertion_block(blk);
        let a = b.func().block(blk).args[0];
        b.op(OpKind::Yield, vec![a], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, res);
    rejects(
        &one_fn(b.finish()),
        "`yield` is not allowed here: this region must end in `condition`",
    );
}

/// `if c {yield ...} else {yield ...}` with the given arm yields; the
/// results of the `if` have the given types.
fn if_with(
    result_types: Vec<Type>,
    then_yield: impl Fn(&mut FunctionBuilder, ValueId) -> Vec<ValueId>,
    arm_args: Vec<Type>,
) -> Module {
    let mut b = start(vec![Type::Bool, len()], vec![]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, _res) = b.op(OpKind::If, vec![c], result_types, sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, arm_args.clone());
        b.set_insertion_block(blk);
        let ys = then_yield(&mut b, x);
        b.op(OpKind::Yield, ys, vec![], sp(6, 7));
    }
    b.set_insertion_block(outer);
    ret(&mut b, vec![]);
    one_fn(b.finish())
}

#[test]
fn if_rules() {
    accepts(&if_with(vec![len()], |_, x| vec![x], vec![]));
    rejects(
        &if_with(vec![len()], |_, x| vec![x, x], vec![]),
        "passes 2 value(s) but there are 1",
    );
    rejects(
        &if_with(vec![time()], |_, x| vec![x], vec![]),
        "value 0 is num[L; ()], but the result of the `if` is num[T; ()]",
    );
    rejects(
        &if_with(vec![len()], |_, x| vec![x], vec![len()]),
        "takes no arguments",
    );
}

#[test]
fn if_condition_and_region_count() {
    let mut b = start(vec![len()], vec![]);
    let x = b.params()[0];
    let (iff, _) = b.op(OpKind::If, vec![x], vec![], sp(0, 1));
    let outer = b.insertion_block();
    let r = b.add_region(iff);
    let blk = b.add_block(r, vec![]);
    b.set_insertion_block(blk);
    b.op(OpKind::Yield, vec![], vec![], sp(0, 0));
    b.set_insertion_block(outer);
    ret(&mut b, vec![]);
    let m = one_fn(b.finish());
    rejects(&m, "the condition of `if` must be Bool");
    rejects(&m, "`if` takes 2 region(s), found 1");
}

#[test]
fn if_takes_one_operand() {
    let mut b = start(vec![Type::Bool], vec![]);
    let (iff, _) = b.op(OpKind::If, vec![], vec![], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, vec![]);
    rejects(&one_fn(b.finish()), "`if` takes 1 operand(s), found 0");
}

#[test]
fn a_plain_operation_takes_no_regions() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (neg, res) = b.op(OpKind::Neg, vec![x], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    let r = b.add_region(neg);
    let blk = b.add_block(r, vec![]);
    b.set_insertion_block(blk);
    b.op(OpKind::Yield, vec![], vec![], sp(0, 0));
    b.set_insertion_block(outer);
    ret(&mut b, res);
    rejects(&one_fn(b.finish()), "`neg` takes 0 region(s), found 1");
}

/// A `while` over `(x: L)` whose regions and terminators are chosen by the
/// caller: `cond` is the first operand of `condition`, `fwd` the forwarded
/// values, `next` the values `after` yields; `before_args` and `after_args`
/// are the two blocks' argument types.
#[allow(clippy::type_complexity)]
fn while_with(
    before_args: Vec<Type>,
    after_args: Vec<Type>,
    cond: impl Fn(&mut FunctionBuilder, ValueId) -> ValueId,
    fwd: impl Fn(&mut FunctionBuilder, &[ValueId]) -> Vec<ValueId>,
    next: impl Fn(&mut FunctionBuilder, &[ValueId]) -> Vec<ValueId>,
) -> Module {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (w, res) = b.op(OpKind::While, vec![x], vec![len()], sp(0, 1));
    let outer = b.insertion_block();
    let before = b.add_region(w);
    let bb = b.add_block(before, before_args);
    b.set_insertion_block(bb);
    let bargs = b.func().block(bb).args.clone();
    let go = cond(&mut b, x);
    let mut ops = vec![go];
    ops.extend(fwd(&mut b, &bargs));
    b.op(OpKind::Condition, ops, vec![], sp(8, 9));
    let after = b.add_region(w);
    let ab = b.add_block(after, after_args);
    b.set_insertion_block(ab);
    let aargs = b.func().block(ab).args.clone();
    let ys = next(&mut b, &aargs);
    b.op(OpKind::Yield, ys, vec![], sp(10, 11));
    b.set_insertion_block(outer);
    ret(&mut b, res);
    one_fn(b.finish())
}

fn bool_const(b: &mut FunctionBuilder, _x: ValueId) -> ValueId {
    b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp(0, 0))
}

fn same(_: &mut FunctionBuilder, a: &[ValueId]) -> Vec<ValueId> {
    a.to_vec()
}

#[test]
fn while_rules() {
    accepts(&while_with(
        vec![len()],
        vec![len()],
        bool_const,
        same,
        same,
    ));
    // Region argument counts.
    rejects(
        &while_with(
            vec![],
            vec![len()],
            bool_const,
            |_, _| vec![],
            |_, a| a.to_vec(),
        ),
        "`before` block of `while` takes 1 argument(s)",
    );
    rejects(
        &while_with(vec![len()], vec![len(), len()], bool_const, same, |_, a| {
            a[..1].to_vec()
        }),
        "`after` block of `while` takes 1 argument(s)",
    );
    // Region argument types.
    rejects(
        &while_with(vec![time()], vec![time()], bool_const, same, same),
        "argument 0 of the `before` block of `while` is num[T; ()], but the carried value is num[L; ()]",
    );
    // condition: first operand a Bool.
    rejects(
        &while_with(
            vec![len()],
            vec![len()],
            |b, x| b.op1(OpKind::Neg, vec![x], len(), sp(0, 0)),
            same,
            same,
        ),
        "first operand of `condition` must be Bool",
    );
    // condition: forwarded values match the results.
    rejects(
        &while_with(
            vec![len()],
            vec![len()],
            bool_const,
            |b, a| vec![b.op1(OpKind::Neg, vec![a[0]], time(), sp(0, 0))],
            same,
        ),
        "a value forwarded by `condition`: value 0 is num[T; ()], but the result of the `while` is num[L; ()]",
    );
    rejects(
        &while_with(vec![len()], vec![len()], bool_const, |_, _| vec![], same),
        "`condition` takes a Bool and 1 forwarded value(s), found 1 operand(s)",
    );
    // yield: the next carried values.
    rejects(
        &while_with(
            vec![len()],
            vec![len()],
            bool_const,
            same,
            |b, a| vec![b.op1(OpKind::Neg, vec![a[0]], time(), sp(0, 0))],
        ),
        "`yield` of the `after` region: value 0 is num[T; ()], but the loop-carried value is num[L; ()]",
    );
    rejects(
        &while_with(vec![len()], vec![len()], bool_const, same, |_, _| vec![]),
        "`yield` of the `after` region passes 0 value(s) but there are 1",
    );
}

#[test]
fn while_result_count_matches_operands() {
    let mut b = start(vec![len()], vec![]);
    let x = b.params()[0];
    let (w, _) = b.op(OpKind::While, vec![x], vec![], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(w);
        let blk = b.add_block(r, vec![len()]);
        b.set_insertion_block(blk);
        let a = b.func().block(blk).args[0];
        b.op(OpKind::Yield, vec![a], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, vec![]);
    rejects(&one_fn(b.finish()), "`while` has 1 result(s), found 0");
}

// ----- types per operation -----

/// `fn f(params) -> (result)` computing `kind(params...)` with the given
/// result type, returning the result.
fn unary(kind: OpKind, arg: Type, result: Type) -> Module {
    binary_or_unary(kind, vec![arg], result)
}

fn binary(kind: OpKind, a: Type, b_ty: Type, result: Type) -> Module {
    binary_or_unary(kind, vec![a, b_ty], result)
}

fn binary_or_unary(kind: OpKind, args: Vec<Type>, result: Type) -> Module {
    let mut b = start(args, vec![result.clone()]);
    let ps = b.params();
    let r = b.op1(kind, ps, result, sp(3, 4));
    ret(&mut b, vec![r]);
    one_fn(b.finish())
}

fn area() -> Type {
    Type::scalar(Dimension::LENGTH.pow_int(2).unwrap())
}

fn vec3() -> Type {
    num(
        Dimension::LENGTH,
        Shape::vector(3),
        TagSet::EMPTY,
        Repr::Unchosen,
    )
}

fn mat(dim: Dimension, r: u64, c: u64) -> Type {
    num(dim, Shape::matrix(r, c), TagSet::EMPTY, Repr::Unchosen)
}

#[test]
fn arity() {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let r = b.op1(OpKind::Add, vec![x], len(), sp(3, 4));
    ret(&mut b, vec![r]);
    rejects(&one_fn(b.finish()), "`add` takes 2 operand(s), found 1");

    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (_, r) = b.op(OpKind::Neg, vec![x], vec![len(), len()], sp(3, 4));
    ret(&mut b, vec![r[0]]);
    rejects(&one_fn(b.finish()), "`neg` has 1 result(s), found 2");
}

#[test]
fn add_and_sub_need_equal_dimensions_and_shapes() {
    for kind in [OpKind::Add, OpKind::Sub] {
        accepts(&binary(kind.clone(), len(), len(), len()));
        rejects(
            &binary(kind.clone(), len(), time(), len()),
            "have different dimensions: L and T",
        );
        rejects(
            &binary(kind.clone(), vec3(), mat(Dimension::LENGTH, 3, 3), vec3()),
            "rank mismatch",
        );
        rejects(
            &binary(kind.clone(), vec3(), vec3(), time()),
            "should have dimension L, found T",
        );
        rejects(
            &binary(kind.clone(), vec3(), vec3(), len()),
            "should have shape Vector[3], found Scalar",
        );
        rejects(
            &binary(kind.clone(), len(), Type::Bool, len()),
            "operand 1 of `",
        );
        // No broadcasting between a scalar and an array.
        rejects(&binary(kind, len(), vec3(), vec3()), "rank mismatch");
    }
}

#[test]
fn shapes_that_agree_only_by_binding_a_variable_are_rejected() {
    let n = sky_types::ShapeVar::from_parts(0, "N");
    let var_vec = num(
        Dimension::LENGTH,
        Shape::vector(n),
        TagSet::EMPTY,
        Repr::Unchosen,
    );
    accepts(&binary(
        OpKind::Add,
        var_vec.clone(),
        var_vec.clone(),
        var_vec.clone(),
    ));
    rejects(
        &binary(OpKind::Add, var_vec.clone(), vec3(), var_vec),
        "shape variable is bound",
    );
}

#[test]
fn mul_and_div_combine_dimensions() {
    accepts(&binary(OpKind::Mul, len(), len(), area()));
    accepts(&binary(OpKind::Div, area(), len(), len()));
    rejects(
        &binary(OpKind::Mul, len(), len(), len()),
        "should have dimension L², found L",
    );
    rejects(
        &binary(OpKind::Div, area(), len(), area()),
        "should have dimension L, found L²",
    );
    // A scalar scales an array.
    accepts(&binary(
        OpKind::Mul,
        len(),
        vec3(),
        num(
            area().as_number().unwrap().dim,
            Shape::vector(3),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
    ));
    accepts(&binary(
        OpKind::Div,
        vec3(),
        len(),
        num(
            Dimension::DIMENSIONLESS,
            Shape::vector(3),
            TagSet::EMPTY,
            Repr::Unchosen,
        ),
    ));
    // No elementwise product of arrays; no scalar divided by an array.
    rejects(
        &binary(OpKind::Mul, vec3(), vec3(), vec3()),
        "`mul` needs a scalar operand",
    );
    rejects(
        &binary(OpKind::Div, len(), vec3(), vec3()),
        "`div` needs a scalar divisor",
    );
    rejects(
        &binary(OpKind::Mul, len(), vec3(), vec3()),
        "should have dimension L²",
    );
}

#[test]
fn dimension_overflow_is_an_error_not_a_panic() {
    let big = Type::scalar(Dimension::from_ints([i64::MAX, 0, 0, 0, 0, 0, 0]));
    rejects(
        &binary(OpKind::Mul, big.clone(), big.clone(), big),
        "dimension exponent too large",
    );
}

#[test]
fn neg_keeps_type() {
    accepts(&unary(OpKind::Neg, vec3(), vec3()));
    rejects(
        &unary(OpKind::Neg, vec3(), len()),
        "should have shape Vector[3]",
    );
    rejects(
        &unary(OpKind::Neg, len(), time()),
        "should have dimension L",
    );
}

#[test]
fn pow_const_raises_the_dimension() {
    let half = OpKind::PowConst {
        exponent: Rational::new(1, 2),
    };
    accepts(&unary(half.clone(), area(), len()));
    rejects(
        &unary(half, area(), area()),
        "should have dimension L, found L²",
    );
    let overflow = OpKind::PowConst {
        exponent: Rational::integer(i64::MAX),
    };
    rejects(
        &unary(overflow, area(), len()),
        "dimension exponent too large",
    );
}

#[test]
fn pow_needs_dimensionless_scalars() {
    accepts(&binary(OpKind::Pow, one(), one(), one()));
    rejects(
        &binary(OpKind::Pow, len(), one(), len()),
        "operand 0 of `pow` must be a dimensionless scalar",
    );
    rejects(
        &binary(OpKind::Pow, one(), len(), one()),
        "operand 1 of `pow` must be a dimensionless scalar",
    );
    rejects(
        &binary(OpKind::Pow, one(), one(), len()),
        "should have dimension 1, found L",
    );
}

#[test]
fn inverse_inverts_the_dimension_and_needs_a_square_matrix() {
    let per_len = Type::scalar(Dimension::LENGTH.inverse().unwrap());
    accepts(&unary(OpKind::Inverse, len(), per_len.clone()));
    rejects(
        &unary(OpKind::Inverse, len(), len()),
        "should have dimension L⁻¹, found L",
    );
    accepts(&unary(
        OpKind::Inverse,
        mat(Dimension::RESISTANCE, 2, 2),
        mat(Dimension::CONDUCTANCE, 2, 2),
    ));
    rejects(
        &unary(
            OpKind::Inverse,
            mat(Dimension::LENGTH, 2, 3),
            mat(Dimension::LENGTH.inverse().unwrap(), 3, 2),
        ),
        "cannot invert a non-square matrix",
    );
    rejects(
        &unary(OpKind::Inverse, vec3(), vec3()),
        "cannot invert a vector",
    );
}

#[test]
fn matmul_uses_the_checker_rules() {
    let a = mat(Dimension::LENGTH, 3, 4);
    let b = mat(Dimension::TIME, 4, 5);
    let prod = mat(Dimension::LENGTH * Dimension::TIME, 3, 5);
    accepts(&binary(OpKind::MatMul, a.clone(), b.clone(), prod.clone()));
    rejects(
        &binary(
            OpKind::MatMul,
            a.clone(),
            mat(Dimension::TIME, 5, 5),
            prod.clone(),
        ),
        "cannot multiply Matrix[3,4] by Matrix[5,5]",
    );
    rejects(
        &binary(
            OpKind::MatMul,
            a.clone(),
            b.clone(),
            mat(Dimension::LENGTH * Dimension::TIME, 5, 3),
        ),
        "should have shape Matrix[3,5]",
    );
    rejects(
        &binary(OpKind::MatMul, a, b, mat(Dimension::LENGTH, 3, 5)),
        "should have dimension L·T",
    );
    rejects(
        &binary(OpKind::MatMul, len(), vec3(), vec3()),
        "cannot multiply Scalar by Vector[3]",
    );
    // matrix @ vector, vector @ vector
    accepts(&binary(
        OpKind::MatMul,
        mat(Dimension::DIMENSIONLESS, 3, 3),
        vec3(),
        vec3(),
    ));
    accepts(&binary(OpKind::MatMul, vec3(), vec3(), area()));
}

#[test]
fn transpose_swaps_matrix_axes() {
    accepts(&unary(
        OpKind::Transpose,
        mat(Dimension::LENGTH, 2, 3),
        mat(Dimension::LENGTH, 3, 2),
    ));
    accepts(&unary(OpKind::Transpose, len(), len()));
    rejects(
        &unary(
            OpKind::Transpose,
            mat(Dimension::LENGTH, 2, 3),
            mat(Dimension::LENGTH, 2, 3),
        ),
        "should have shape Matrix[3,2]",
    );
    rejects(
        &unary(OpKind::Transpose, vec3(), vec3()),
        "cannot transpose Vector[3]",
    );
}

#[test]
fn solve_needs_a_positive_definite_matrix() {
    let n = sky_types::ShapeVar::from_parts(0, "N");
    let solve = OpKind::Solve {
        method: SolveMethod::Cholesky,
    };
    let b_vec = vector(n.clone());
    accepts(&binary(
        solve.clone(),
        pd_matrix(n.clone()),
        b_vec.clone(),
        b_vec.clone(),
    ));
    // Symmetric is not enough.
    let sym = num(
        Dimension::DIMENSIONLESS,
        Shape::matrix(n.clone(), n.clone()),
        TagSet::of(Tag::Symmetric),
        Repr::Unchosen,
    );
    rejects(
        &binary(solve.clone(), sym, b_vec.clone(), b_vec.clone()),
        "needs a PositiveDefinite matrix",
    );
    // Shapes and dimensions.
    rejects(
        &binary(solve.clone(), pd_matrix(n.clone()), vec3(), vec3()),
        "shape variable is bound",
    );
    rejects(
        &binary(solve.clone(), pd_matrix(n.clone()), b_vec, vector(3)),
        "should have shape Vector[N]",
    );
    let stiffness = num(
        Dimension::FORCE / Dimension::LENGTH,
        Shape::matrix(2, 2),
        TagSet::of(Tag::PositiveDefinite).closure(),
        Repr::Unchosen,
    );
    let force = num(
        Dimension::FORCE,
        Shape::vector(2),
        TagSet::EMPTY,
        Repr::Unchosen,
    );
    let disp = num(
        Dimension::LENGTH,
        Shape::vector(2),
        TagSet::EMPTY,
        Repr::Unchosen,
    );
    accepts(&binary(
        solve.clone(),
        stiffness.clone(),
        force.clone(),
        disp,
    ));
    rejects(
        &binary(solve, stiffness, force.clone(), force),
        "should have dimension L, found L·M·T⁻²",
    );
}

#[test]
fn comparisons_give_bool() {
    for c in [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ] {
        accepts(&binary(OpKind::Cmp(c), len(), len(), Type::Bool));
        rejects(
            &binary(OpKind::Cmp(c), len(), len(), len()),
            "the result of `cmp.",
        );
        rejects(
            &binary(OpKind::Cmp(c), len(), time(), Type::Bool),
            "have different dimensions",
        );
        rejects(
            &binary(OpKind::Cmp(c), vec3(), vec3(), Type::Bool),
            "compares scalars",
        );
    }
    accepts(&binary(
        OpKind::Cmp(CmpOp::Eq),
        Type::Bool,
        Type::Bool,
        Type::Bool,
    ));
    accepts(&binary(
        OpKind::Cmp(CmpOp::Ne),
        Type::Str,
        Type::Str,
        Type::Bool,
    ));
    rejects(
        &binary(OpKind::Cmp(CmpOp::Lt), Type::Bool, Type::Bool, Type::Bool),
        "cannot compare bool with bool",
    );
    rejects(
        &binary(OpKind::Cmp(CmpOp::Eq), len(), Type::Bool, Type::Bool),
        "cannot compare",
    );
}

#[test]
fn not_needs_a_bool() {
    accepts(&unary(OpKind::Not, Type::Bool, Type::Bool));
    rejects(
        &unary(OpKind::Not, len(), Type::Bool),
        "`not` needs a Bool operand",
    );
    rejects(&unary(OpKind::Not, Type::Bool, len()), "must be Bool");
}

#[test]
fn an_input_has_no_operands_and_one_result_of_any_type() {
    let input = |ty: Type| binary_or_unary(OpKind::Input { name: "A".into() }, vec![], ty);
    for error in sky_ir::LiteralError::ALL {
        accepts(&binary_or_unary(
            OpKind::LiteralError { error },
            vec![],
            len(),
        ));
        rejects(
            &binary_or_unary(OpKind::LiteralError { error }, vec![len()], len()),
            "takes 0 operand(s)",
        );
    }
    accepts(&input(Type::Bool));
    accepts(&input(len()));
    accepts(&input(Type::Unknown));
}

#[test]
fn expect_bool_checks_a_bool_or_unknown_operand_and_gives_a_bool() {
    use sky_ir::BoolContext;
    for context in BoolContext::ALL {
        let kind = OpKind::ExpectBool { context };
        accepts(&unary(kind.clone(), Type::Unknown, Type::Bool));
        accepts(&unary(kind.clone(), Type::Bool, Type::Bool));
        rejects(
            &unary(kind.clone(), len(), Type::Bool),
            "`expect.bool` needs a Bool or unknown operand",
        );
        rejects(&unary(kind, Type::Unknown, len()), "must be Bool");
    }
}

#[test]
fn constants_are_typed_exactly() {
    let konst = |kind: OpKind, ty: Type| {
        let mut b = start(vec![], vec![]);
        b.op1(kind, vec![], ty, sp(2, 3));
        ret(&mut b, vec![]);
        one_fn(b.finish())
    };
    accepts(&konst(OpKind::ConstBool(true), Type::Bool));
    rejects(&konst(OpKind::ConstBool(true), len()), "must be bool");
    rejects(
        &konst(OpKind::ConstStr("x".into()), Type::Bool),
        "must be str",
    );
    rejects(
        &konst(OpKind::ConstNothing, Type::Unknown),
        "must be nothing",
    );
    accepts(&konst(OpKind::ConstRational(Rational::new(1, 3)), one()));
    rejects(
        &konst(OpKind::ConstRational(Rational::new(1, 3)), Type::Bool),
        "must be a scalar number",
    );
    rejects(
        &konst(OpKind::ConstFloat { bits: 0 }, vec3()),
        "must be a scalar number",
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        rejects(
            &konst(
                OpKind::ConstFloat {
                    bits: bad.to_bits(),
                },
                one(),
            ),
            "`const.float` must be finite",
        );
    }
    accepts(&konst(
        OpKind::ConstFloat {
            bits: (-0.0f64).to_bits(),
        },
        one(),
    ));
}

// ----- calls and returns -----

fn caller(args: Vec<Type>, callee: &str, results: Vec<Type>) -> Module {
    let mut b = start(args, results.clone());
    let ps = b.params();
    let (_, rs) = b.op(
        OpKind::Call {
            callee: callee.into(),
        },
        ps,
        results,
        sp(5, 6),
    );
    ret(&mut b, rs);
    module(vec![b.finish(), identity_fn("id")])
}

#[test]
fn calls_match_the_callee() {
    accepts(&caller(vec![len()], "id", vec![len()]));
    rejects(
        &caller(vec![len()], "nope", vec![len()]),
        "undefined function `nope`",
    );
    rejects(
        &caller(vec![time()], "id", vec![len()]),
        "argument 0 of the call to `id` is num[T; ()], expected num[L; ()]",
    );
    rejects(
        &caller(vec![len(), len()], "id", vec![len()]),
        "takes 1 argument(s), the call passes 2",
    );
    rejects(
        &caller(vec![len()], "id", vec![time()]),
        "result 0 of the call to `id` is num[T; ()], but it returns num[L; ()]",
    );
    rejects(
        &caller(vec![len()], "id", vec![len(), len()]),
        "returns 1 value(s), the call has 2 result(s)",
    );
}

#[test]
fn return_matches_the_function_results() {
    let mut b = start(vec![len()], vec![len()]);
    ret(&mut b, vec![]);
    rejects(
        &one_fn(b.finish()),
        "`return` passes 0 value(s), the function returns 1",
    );

    let mut b = start(vec![len()], vec![time()]);
    let x = b.params()[0];
    ret(&mut b, vec![x]);
    rejects(
        &one_fn(b.finish()),
        "`return` value 0 is num[L; ()], the function returns num[T; ()]",
    );
}

#[test]
fn a_value_with_more_tags_can_be_used_as_one_with_fewer() {
    let n = sky_types::ShapeVar::from_parts(0, "N");
    let sym = num(
        Dimension::DIMENSIONLESS,
        Shape::matrix(n.clone(), n.clone()),
        TagSet::of(Tag::Symmetric),
        Repr::Unchosen,
    );
    // PositiveDefinite -> Symmetric: fine. Symmetric -> PositiveDefinite: no.
    let mut b = start(vec![pd_matrix(n.clone())], vec![sym.clone()]);
    let a = b.params()[0];
    ret(&mut b, vec![a]);
    accepts(&one_fn(b.finish()));
    let mut b = start(vec![sym], vec![pd_matrix(n)]);
    let a = b.params()[0];
    ret(&mut b, vec![a]);
    rejects(&one_fn(b.finish()), "`return` value 0 is");
}

#[test]
fn unknown_is_compatible_with_anything() {
    // Unknown operands: no rule that needs their type fires, and the result
    // may be a number or unknown.
    accepts(&binary(OpKind::Add, Type::Unknown, len(), Type::Unknown));
    accepts(&binary(OpKind::Mul, Type::Unknown, Type::Unknown, len()));
    accepts(&binary(
        OpKind::MatMul,
        Type::Unknown,
        vec3(),
        Type::Unknown,
    ));
    accepts(&binary(
        OpKind::Cmp(CmpOp::Lt),
        Type::Unknown,
        len(),
        Type::Bool,
    ));
    accepts(&unary(OpKind::Not, Type::Unknown, Type::Bool));
    // But an Unknown operand is still not a Bool where a number is needed,
    // and a known wrong kind is still wrong.
    rejects(
        &binary(OpKind::Add, Type::Unknown, Type::Bool, Type::Unknown),
        "operand 1 of `add` must be a number",
    );
    // Assignability: unknown goes anywhere.
    let mut b = start(vec![Type::Unknown], vec![len()]);
    let x = b.params()[0];
    ret(&mut b, vec![x]);
    accepts(&one_fn(b.finish()));
    let mut b = start(vec![len()], vec![Type::Unknown]);
    let x = b.params()[0];
    ret(&mut b, vec![x]);
    accepts(&one_fn(b.finish()));
    // `if` on an unknown condition.
    let mut b = start(vec![Type::Unknown, len()], vec![len()]);
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let (iff, res) = b.op(OpKind::If, vec![c], vec![Type::Unknown], sp(0, 1));
    let outer = b.insertion_block();
    for _ in 0..2 {
        let r = b.add_region(iff);
        let blk = b.add_block(r, vec![]);
        b.set_insertion_block(blk);
        b.op(OpKind::Yield, vec![x], vec![], sp(0, 0));
    }
    b.set_insertion_block(outer);
    ret(&mut b, res);
    accepts(&one_fn(b.finish()));
}

// ----- tags and module level -----

#[test]
fn tags_must_be_closed() {
    // PositiveDefinite without the Symmetric it implies.
    let open = num(
        Dimension::DIMENSIONLESS,
        Shape::matrix(2, 2),
        TagSet::of(Tag::PositiveDefinite),
        Repr::Unchosen,
    );
    let mut b = start(vec![open.clone()], vec![]);
    ret(&mut b, vec![]);
    rejects(&one_fn(b.finish()), "are not closed");
    // In a signature only.
    let mut f = tiny();
    f.results = vec![open];
    rejects(&one_fn(f), "are not closed");
    // Closed is fine.
    let closed = pd_matrix(sky_types::ShapeVar::from_parts(0, "N"));
    let mut b = start(vec![closed], vec![]);
    ret(&mut b, vec![]);
    accepts(&one_fn(b.finish()));
    // On an operation result.
    let mut b = start(vec![len()], vec![]);
    let x = b.params()[0];
    b.op1(
        OpKind::Neg,
        vec![x],
        num(
            Dimension::LENGTH,
            Shape::scalar(),
            TagSet::of(Tag::Diagonal),
            Repr::Unchosen,
        ),
        sp(0, 0),
    );
    ret(&mut b, vec![]);
    rejects(&one_fn(b.finish()), "are not closed");
}

#[test]
fn tags_need_a_square_matrix() {
    let sym = TagSet::of(Tag::Symmetric);
    for shape in [Shape::vector(3), Shape::scalar(), Shape::matrix(2, 3)] {
        let t = num(Dimension::DIMENSIONLESS, shape, sym, Repr::Unchosen);
        let mut b = start(vec![t], vec![]);
        ret(&mut b, vec![]);
        rejects(&one_fn(b.finish()), "is not a square matrix");
    }
}

#[test]
fn function_names_are_unique() {
    let m = module(vec![tiny(), tiny()]);
    rejects(&m, "function `f` is defined twice");
}

#[test]
fn errors_are_capped_and_have_locations() {
    // A function with hundreds of bad operations reports a bounded list.
    let mut b = start(vec![len()], vec![]);
    let x = b.params()[0];
    for i in 0..500 {
        b.op1(OpKind::Add, vec![x], len(), sp(i, i + 1));
    }
    ret(&mut b, vec![]);
    let e = errors(&one_fn(b.finish()));
    assert!(e.len() <= 100, "{}", e.len());
    assert_eq!(e[0].span, sp(0, 1));
    assert_eq!(e[0].function, "f");
    let text = e[0].to_string();
    assert!(
        text.starts_with("@f: `add` takes 2 operand(s), found 1"),
        "{text}"
    );
    let d = e[0].to_diagnostic();
    assert!(d.message.starts_with("invalid IR:"));
}

#[test]
fn deeply_nested_regions_are_an_error_not_a_stack_overflow() {
    let m = deep_ifs(sky_ir::MAX_REGION_DEPTH + 40);
    rejects(&m, "nested more than");
    accepts(&deep_ifs(sky_ir::MAX_REGION_DEPTH - 2));
}

fn deep_ifs(depth: usize) -> Module {
    let mut b = start(vec![Type::Bool], vec![]);
    let c = b.params()[0];
    let mut frames = Vec::new();
    for _ in 0..depth {
        let (iff, _) = b.op(OpKind::If, vec![c], vec![], sp(0, 1));
        let outer = b.insertion_block();
        let then = b.add_region(iff);
        let tb = b.add_block(then, vec![]);
        b.set_insertion_block(tb);
        frames.push((iff, outer));
    }
    while let Some((iff, outer)) = frames.pop() {
        b.op(OpKind::Yield, vec![], vec![], sp(0, 0));
        let els = b.add_region(iff);
        let eb = b.add_block(els, vec![]);
        b.set_insertion_block(eb);
        b.op(OpKind::Yield, vec![], vec![], sp(0, 0));
        b.set_insertion_block(outer);
    }
    ret(&mut b, vec![]);
    one_fn(b.finish())
}

/// `fn apply(A: Matrix[N, N], b: Vector[N]) -> Vector[N]` (returns `b`), and
/// a `<main>` that calls it with the given argument sizes.
fn generic_call(a_size: u64, b_size: u64, result_size: u64) -> Module {
    use sky_ir::NumberType;
    use sky_types::{ShapeDim, ShapeVar};
    let n = ShapeDim::Var(ShapeVar::from_parts(7, "N"));
    let num = |shape: Shape| {
        Type::Number(NumberType {
            dim: Dimension::DIMENSIONLESS,
            shape,
            tags: TagSet::EMPTY,
            repr: Repr::Unchosen,
        })
    };
    let mat_n = num(Shape::new(vec![n.clone(), n.clone()]));
    let vec_n = num(Shape::new(vec![n]));
    let mut g = FunctionBuilder::new("apply", vec![mat_n, vec_n.clone()], vec![vec_n], sp(0, 10));
    let b_param = g.params()[1];
    g.op(OpKind::Return, vec![b_param], vec![], sp(9, 10));
    let apply = g.finish();

    let mat = num(Shape::matrix(a_size, a_size));
    let vect = num(Shape::vector(b_size));
    let mut m = FunctionBuilder::new(Module::MAIN, vec![mat, vect], vec![], sp(20, 40));
    let [a, b] = m.params()[..] else {
        unreachable!()
    };
    m.op(
        OpKind::Call {
            callee: "apply".into(),
        },
        vec![a, b],
        vec![num(Shape::vector(result_size))],
        sp(21, 30),
    );
    m.op(OpKind::Return, vec![], vec![], sp(39, 40));
    module(vec![m.finish(), apply])
}

#[test]
fn a_call_instantiates_the_callees_size_variables() {
    // N = 3 at this call, consistently for both arguments and the result;
    // another call could use N = 4.
    accepts(&generic_call(3, 3, 3));
    accepts(&generic_call(4, 4, 4));
}

#[test]
fn a_size_variable_must_be_consistent_within_one_call() {
    rejects(&generic_call(3, 4, 3), "argument 1 of the call to `apply`");
    rejects(&generic_call(3, 3, 4), "result 0 of the call to `apply`");
}

// ----- show -----

/// `fn f(x: T) -> (str)` returning `show "unit" x`.
fn show_fn(param: Type, unit: &str, result: Type) -> Module {
    let mut b = start(vec![param], vec![Type::Str]);
    let x = b.params()[0];
    let s = b.op1(
        OpKind::Show { unit: unit.into() },
        vec![x],
        result,
        sp(1, 2),
    );
    ret(&mut b, vec![s]);
    one_fn(b.finish())
}

fn velocity() -> Type {
    num(
        Dimension::LENGTH.checked_div(Dimension::TIME).unwrap(),
        Shape::scalar(),
        TagSet::EMPTY,
        Repr::Unchosen,
    )
}

#[test]
fn show_needs_a_unit_of_the_operands_dimension_and_a_str_result() {
    accepts(&show_fn(len(), "mm", Type::Str));
    accepts(&show_fn(len(), "AU", Type::Str));
    accepts(&show_fn(velocity(), "km/h", Type::Str));
    accepts(&show_fn(velocity(), "m*s^-1", Type::Str));
    accepts(&show_fn(one(), "rad", Type::Str));
    // An unknown-typed operand is checked at run time.
    accepts(&show_fn(Type::Unknown, "km/h", Type::Str));

    rejects(
        &show_fn(len(), "s", Type::Str),
        "needs an operand of that dimension",
    );
    rejects(&show_fn(velocity(), "kg", Type::Str), "`show` of `kg`");
    rejects(
        &show_fn(len(), "mm", len()),
        "the result of `show` must be str",
    );
    rejects(&show_fn(Type::Bool, "mm", Type::Str), "must be a number");
    rejects(
        &show_fn(len(), "parsecs", Type::Str),
        "`show` needs a unit, found `parsecs`",
    );
    rejects(&show_fn(len(), "km h", Type::Str), "`show` needs a unit");
    rejects(&show_fn(len(), "", Type::Str), "`show` needs a unit");
    rejects(&show_fn(len(), "km + 1", Type::Str), "`show` needs a unit");
}

#[test]
fn show_takes_one_operand_and_gives_one_result() {
    let mut b = start(vec![len(), len()], vec![]);
    let [x, y] = b.params()[..] else {
        unreachable!()
    };
    let s = b.op1(
        OpKind::Show { unit: "mm".into() },
        vec![x, y],
        Type::Str,
        sp(1, 2),
    );
    b.op(OpKind::Print, vec![s], vec![], sp(3, 4));
    ret(&mut b, vec![]);
    rejects(&one_fn(b.finish()), "`show` takes 1 operand(s), found 2");
}

// ----- operand spans -----

/// `x + x` with the given operand spans.
fn add_spans(spans: Vec<sky_diag::Span>) -> Module {
    let mut b = start(vec![len()], vec![len()]);
    let x = b.params()[0];
    let (_, s) = b.op_with_spans(OpKind::Add, vec![x, x], spans, vec![len()], sp(1, 2));
    ret(&mut b, vec![s[0]]);
    one_fn(b.finish())
}

#[test]
fn operand_spans_are_none_or_one_per_operand() {
    accepts(&add_spans(vec![]));
    accepts(&add_spans(vec![sp(1, 2), sp(3, 4)]));
    for wrong in [vec![sp(1, 2)], vec![sp(1, 2), sp(3, 4), sp(5, 6)]] {
        let e = rejects(&add_spans(wrong), "operand span(s), but 2 operand(s)");
        assert_eq!(e[0].span, sp(1, 2), "the error points at the operation");
    }
}

#[test]
fn a_slice_also_has_a_span_per_axis_item() {
    use sky_ir::SliceAxis;
    let slice = |spans: Vec<sky_diag::Span>| {
        let mut b = start(vec![vector(4)], vec![]);
        let v = b.params()[0];
        let zero = b.op1(
            OpKind::ConstRational(Rational::ZERO),
            vec![],
            one(),
            sp(0, 0),
        );
        let axes = vec![SliceAxis::Range {
            start: true,
            stop: false,
        }];
        b.op_with_spans(
            OpKind::Slice { axes },
            vec![v, zero],
            spans,
            vec![vector(4)],
            sp(1, 9),
        );
        ret(&mut b, vec![]);
        one_fn(b.finish())
    };
    // The base, the bound, and the item.
    accepts(&slice(vec![sp(1, 2), sp(3, 4), sp(3, 5)]));
    accepts(&slice(vec![]));
    rejects(
        &slice(vec![sp(1, 2), sp(3, 4)]),
        "operand span(s), but 2 operand(s) and 1 axis item(s)",
    );
}

#[test]
fn expect_number_checks_a_number_or_unknown_operand_and_gives_it_back() {
    use sky_ir::NumberContext;
    for context in NumberContext::ALL {
        let kind = OpKind::ExpectNumber { context };
        accepts(&unary(kind.clone(), Type::Unknown, Type::Unknown));
        accepts(&unary(kind.clone(), len(), len()));
        // A pass that learned the operand's type leaves the result unknown.
        accepts(&unary(kind.clone(), len(), Type::Unknown));
        rejects(
            &unary(kind.clone(), Type::Bool, Type::Bool),
            "`expect.number` needs a number or unknown operand",
        );
        rejects(
            &unary(kind, len(), one()),
            "the result of `expect.number` must be unknown or",
        );
    }
}
