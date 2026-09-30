//! Last-use analysis (`sky_ir::lifetime`): where an operand dies, on every
//! path, and which operations can hold their result in an operand's buffer.
//!
//! The cases that matter are the ones where being wrong would corrupt a
//! value: a value used after a branch, a value read again by the next
//! iteration of a loop, the same value twice in one operation, and IR the
//! analysis cannot trust.

use sky_ir::{parse, Function, Lifetimes, Module, OpId, OpKind, ValueDef};

const HEAD: &str = "sir edition 2026\n";

fn module(body: &str) -> Module {
    let text = format!("{HEAD}{body}");
    parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// The operation whose first result is named `name`.
fn op_named(f: &Function, name: &str) -> OpId {
    let v = f
        .values
        .iter()
        .position(|v| v.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no value named {name}"));
    match f.values[v].def {
        ValueDef::OpResult { op, .. } => op,
        ValueDef::BlockArg { .. } => panic!("{name} is a block argument"),
    }
}

/// The first `return`, `yield` or `condition` with `operands` operands and
/// the given kind, in the arena (operations are in creation order).
fn terminator(f: &Function, kind: &OpKind, nth: usize) -> OpId {
    let ids: Vec<OpId> = (0..f.ops.len())
        .map(|i| OpId(i as u32))
        .filter(|&o| f.op(o).kind == *kind)
        .collect();
    ids[nth]
}

/// Which operands of `op` die, as a string of `d` (dies) and `.` (does not).
fn dies(life: &Lifetimes, f: &Function, op: OpId) -> String {
    (0..f.op(op).operands.len())
        .map(|i| if life.dies(op, i) { 'd' } else { '.' })
        .collect()
}

#[test]
fn straight_line_code_dies_at_the_last_use() {
    let m = module(
        "fn @f(%x: num[L; (3)], %y: num[L; (3)]) -> (num[L; (3)]) {
  %a = add %x, %y : num[L; (3)]
  %b = add %a, %x : num[L; (3)]
  return %b
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    assert_eq!(dies(&life, f, op_named(f, "a")), ".d");
    assert_eq!(dies(&life, f, op_named(f, "b")), "dd");
    assert_eq!(dies(&life, f, terminator(f, &OpKind::Return, 0)), "d");
    // `y` dies at the first, and `a` (a temporary) at the second, is the
    // buffer each result may take.
    assert_eq!(life.reuse(op_named(f, "a")), Some(1));
    assert_eq!(life.reuse(op_named(f, "b")), Some(0));
}

#[test]
fn a_value_twice_in_one_operation_dies_at_neither_use() {
    let m = module(
        "fn @f(%x: num[L; (3)]) -> (num[L; (3)]) {
  %a = add %x, %x : num[L; (3)]
  return %a
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    assert_eq!(dies(&life, f, op_named(f, "a")), "..");
    assert_eq!(life.reuse(op_named(f, "a")), None);
}

#[test]
fn a_value_used_after_an_if_dies_in_no_arm() {
    let m = module(
        "fn @f(%a: num[L; (3)], %b: num[L; (3)], %c: bool) -> (num[L; (3)]) {
  %r = if %c : num[L; (3)] {
    ^1:
      %t = add %a, %b : num[L; (3)]
      yield %t
  } {
    ^2:
      yield %a
  }
  %s = add %r, %b : num[L; (3)]
  return %s
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%b` is read after the `if`: it does not die in the `then` arm. `%a`
    // is not read after it: it dies in the `then` arm, and in the `else`
    // arm, where it is yielded.
    assert_eq!(dies(&life, f, op_named(f, "t")), "d.");
    assert_eq!(dies(&life, f, terminator(f, &OpKind::Yield, 1)), "d");
    // After the `if`, `%b` is at its last use.
    assert_eq!(dies(&life, f, op_named(f, "s")), "dd");
}

#[test]
fn a_value_used_in_one_arm_is_left_alone_on_the_other_path() {
    let m = module(
        "fn @f(%a: num[L; (3)], %c: bool) -> (num[L; ()]) {
  %z = const.float 0 : num[L; ()]
  %r = if %c : num[L; ()] {
    ^1:
      %n = neg %a : num[L; (3)]
      yield %z
  } {
    ^2:
      yield %z
  }
  return %r
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%a` dies in the arm that reads it; the other arm never touches it, so
    // an executor that moved it out finds it in its slot there (and the
    // interpreter drops nothing it did not read).
    assert_eq!(dies(&life, f, op_named(f, "n")), "d");
}

#[test]
fn a_value_from_outside_a_loop_never_dies_inside_it() {
    let m = module(
        "fn @f(%v0: num[L; (3)], %n: num[1; ()], %c: num[L; (3)]) -> (num[L; (3)]) {
  %zero = const.float 0 : num[1; ()]
  %one = const.float 1 : num[1; ()]
  %i_end, %v_end = while %zero, %v0 : num[1; ()], num[L; (3)] {
    ^1(%i: num[1; ()], %v: num[L; (3)]):
      %go = cmp.lt %i, %n : bool
      condition %go, %i, %v
  } {
    ^2(%j: num[1; ()], %w: num[L; (3)]):
      %j1 = add %j, %one : num[1; ()]
      %w1 = add %w, %c : num[L; (3)]
      yield %j1, %w1
  }
  return %v_end
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%c`, `%n` and `%one` are read by every iteration.
    assert_eq!(dies(&life, f, op_named(f, "w1")), "d.");
    assert_eq!(dies(&life, f, op_named(f, "j1")), "d.");
    // `%i` is read again by `condition`, which forwards it.
    assert_eq!(dies(&life, f, op_named(f, "go")), "..");
    // The loop-carried values are handed on: they die at `condition` and at
    // `yield`, which is what moves a buffer from one iteration to the next.
    assert_eq!(dies(&life, f, terminator(f, &OpKind::Condition, 0)), "ddd");
    assert_eq!(dies(&life, f, terminator(f, &OpKind::Yield, 0)), "dd");
    // The initial values die at the `while` itself: the loop does not read
    // them, only its block arguments.
    let w = op_named(f, "i_end");
    assert_eq!(dies(&life, f, w), "dd");
    // The result of `%w + %c` can take `%w`'s buffer, not `%c`'s.
    assert_eq!(life.reuse(op_named(f, "w1")), Some(0));
    assert!(life.in_loop(op_named(f, "w1")));
    assert!(!life.in_loop(op_named(f, "i_end")));
}

#[test]
fn an_initial_value_the_loop_reads_does_not_die_at_the_while() {
    let m = module(
        "fn @f(%v0: num[L; (3)], %n: num[1; ()]) -> (num[L; (3)]) {
  %zero = const.float 0 : num[1; ()]
  %one = const.float 1 : num[1; ()]
  %i_end, %v_end = while %zero, %v0 : num[1; ()], num[L; (3)] {
    ^1(%i: num[1; ()], %v: num[L; (3)]):
      %go = cmp.lt %i, %n : bool
      condition %go, %i, %v
  } {
    ^2(%j: num[1; ()], %w: num[L; (3)]):
      %j1 = add %j, %one : num[1; ()]
      %w1 = add %w, %v0 : num[L; (3)]
      yield %j1, %w1
  }
  return %v_end
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%v0` is read inside the loop, so the operand of the `while` is not its
    // last use, and neither is the read inside.
    assert_eq!(dies(&life, f, op_named(f, "i_end")), "d.");
    assert_eq!(dies(&life, f, op_named(f, "w1")), "d.");
}

#[test]
fn a_value_used_after_a_loop_does_not_die_at_the_while() {
    let m = module(
        "fn @f(%v0: num[L; (3)], %n: num[1; ()]) -> (num[L; (3)]) {
  %zero = const.float 0 : num[1; ()]
  %i_end, %v_end = while %zero, %v0 : num[1; ()], num[L; (3)] {
    ^1(%i: num[1; ()], %v: num[L; (3)]):
      %go = cmp.lt %i, %n : bool
      condition %go, %i, %v
  } {
    ^2(%j: num[1; ()], %w: num[L; (3)]):
      yield %j, %w
  }
  %s = add %v_end, %v0 : num[L; (3)]
  return %s
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    assert_eq!(dies(&life, f, op_named(f, "i_end")), "d.");
    assert_eq!(dies(&life, f, op_named(f, "s")), "dd");
}

#[test]
fn a_value_defined_in_a_loop_is_ordered_like_straight_line_code() {
    let m = module(
        "fn @f(%v0: num[L; (3)], %n: num[1; ()]) -> (num[L; (3)]) {
  %zero = const.float 0 : num[1; ()]
  %one = const.float 1 : num[1; ()]
  %i_end, %v_end = while %zero, %v0 : num[1; ()], num[L; (3)] {
    ^1(%i: num[1; ()], %v: num[L; (3)]):
      %go = cmp.lt %i, %n : bool
      condition %go, %i, %v
  } {
    ^2(%j: num[1; ()], %w: num[L; (3)]):
      %j1 = add %j, %one : num[1; ()]
      %a = add %w, %w : num[L; (3)]
      %b = add %a, %w : num[L; (3)]
      yield %j1, %b
  }
  return %v_end
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%w` is read twice by `%a`, and again by `%b`: dies only at `%b`.
    assert_eq!(dies(&life, f, op_named(f, "a")), "..");
    assert_eq!(dies(&life, f, op_named(f, "b")), "dd");
}

#[test]
fn a_call_argument_dies_if_the_caller_is_done_with_it() {
    let m = module(
        "fn @f(%a: num[L; (3)], %b: num[L; (3)]) -> (num[L; (3)]) {
  %r = call @id(%a) : num[L; (3)]
  %s = call @id(%b) : num[L; (3)]
  %t = add %r, %b : num[L; (3)]
  return %t
}
fn @id(%a: num[L; (3)]) -> (num[L; (3)]) {
  return %a
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    assert_eq!(dies(&life, f, op_named(f, "r")), "d");
    // `%b` is read again by the `add`.
    assert_eq!(dies(&life, f, op_named(f, "s")), ".");
    assert_eq!(dies(&life, f, op_named(f, "t")), "dd");
    // A callee that returns its argument: the parameter dies at `return`.
    let g = &m.functions[1];
    let lg = Lifetimes::analyze(g);
    assert_eq!(dies(&lg, g, terminator(g, &OpKind::Return, 0)), "d");
}

#[test]
fn reuse_needs_the_shape_and_an_array() {
    let m = module(
        "fn @f(%x: num[L; (3)], %y: num[L; (4)], %s: num[1; ()], %u: unknown) -> (num[L; (3)]) {
  %a = mul %x, %s : num[L; (3)]
  %b = mul %s, %s : num[L; ()]
  %c = neg %y : num[L; (4)]
  %d = add %u, %u : unknown
  %e = neg %u : unknown
  return %a
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // The array operand may hold the scaled result; the scalar never does.
    assert_eq!(life.reuse(op_named(f, "a")), Some(0));
    // Scalars have no buffer worth reusing.
    assert_eq!(life.reuse(op_named(f, "b")), None);
    assert_eq!(life.reuse(op_named(f, "c")), Some(0));
    // The static answer never relies on `unknown`, which the interpreter
    // decides at run time.
    assert_eq!(life.reuse(op_named(f, "d")), None);
    assert_eq!(life.reuse(op_named(f, "e")), None);
}

#[test]
fn a_fused_kernel_names_the_operand_that_holds_its_result() {
    let m = module(
        "fn @f(%x: num[L; (3)], %v: num[L; (3)], %dt: num[T; ()]) -> (num[L; (3)]) {
  %k = fused %dt, %v, %x : num[L; (3)] {
    ^1(%p: num[T; ()], %q: num[L; ()], %r: num[L; ()]):
      %m = mul %p, %q : num[L·T; ()]
      %n = add %r, %m : num[L·T; ()]
      yield %n
  }
  %z = fused %x, %k : num[L; (3)] {
    ^2(%p2: num[L; ()], %q2: num[L; ()]):
      %m2 = sub %p2, %q2 : num[L; ()]
      yield %m2
  }
  return %z
}
",
    );
    let f = &m.functions[0];
    let life = Lifetimes::analyze(f);
    // `%dt` is a scalar read once (it dies, but has no buffer to give); `%v`
    // is the first array that dies, and `%x` is read again by `%z`.
    let k = op_named(f, "k");
    assert_eq!(dies(&life, f, k), "dd.");
    assert_eq!(life.reuse(k), Some(1));
    // `%z`: `%x` is read here for the last time, and so is `%k`.
    assert_eq!(life.reuse(op_named(f, "z")), Some(0));
    let sites = life.reuse_sites();
    assert_eq!(sites.len(), 2);
    assert!(sites.iter().all(|s| !s.in_loop));
}

#[test]
fn malformed_functions_are_answered_conservatively_and_never_panic() {
    // An operand that is not a value of the function.
    let m = module(
        "fn @f(%x: num[L; (3)]) -> (num[L; (3)]) {
  %a = neg %x : num[L; (3)]
  return %a
}
",
    );
    let mut f = m.functions[0].clone();
    let a = op_named(&f, "a");
    f.ops[a.index()].operands[0] = sky_ir::ValueId(9999);
    let life = Lifetimes::analyze(&f);
    assert!(!life.dies(a, 0) || life.reuse(a).is_none());

    // A region that contains its own operation.
    let m = module(
        "fn @f(%c: bool, %x: num[L; (3)]) -> (num[L; (3)]) {
  %r = if %c : num[L; (3)] {
    ^1:
      yield %x
  } {
    ^2:
      yield %x
  }
  return %r
}
",
    );
    let mut f = m.functions[0].clone();
    let iff = op_named(&f, "r");
    let region = f.ops[iff.index()].regions[0];
    let block = f.regions[region.index()].blocks[0];
    f.blocks[block.index()].ops.push(iff);
    let life = Lifetimes::analyze(&f);
    // Nothing panicked; whatever was found is a sound answer or none.
    let _ = life.reuse_sites();

    // Operations that no block lists, and empty regions: no panic.
    let mut f = m.functions[0].clone();
    f.regions[f.body.index()].blocks.clear();
    let life = Lifetimes::analyze(&f);
    assert!(life.reuse_sites().is_empty());
    assert!(!life.dies(OpId(0), 0));
    assert!(!life.dies(OpId(999), 70));
}
