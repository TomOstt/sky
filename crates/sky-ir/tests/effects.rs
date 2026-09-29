//! Effect inference: the union over operations and nested regions, calls
//! taking the callee's effects, recursion by fixed point, and the
//! verifier's "declared effects include the inferred ones" rule.

mod common;

use common::*;
use sky_diag::Span;
use sky_ir::effects::{infer, infer_effects, InferEffects};
use sky_ir::{
    parse, print, verify, BlockId, Effect, EffectSet, FunctionBuilder, Module, OpId, OpKind,
    PassManager, RegionData, RegionId,
};

fn effects_of(text: &str) -> Vec<String> {
    let m = parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    infer(&m).into_iter().map(show).collect()
}

fn show(e: EffectSet) -> String {
    let names: Vec<&str> = e
        .iter()
        .map(|e| match e {
            Effect::Io => "io",
            Effect::Fail => "fail",
            Effect::Call => "call",
        })
        .collect();
    names.join(",")
}

const HEAD: &str = "sir edition 2026\n";

#[test]
fn an_operation_gives_its_own_effects() {
    let t = format!(
        "{HEAD}
fn @pure(%a: num[L; ()]) -> (num[L; ()]) {{
  %n = neg %a : num[L; ()]
  return %n
}}
fn @fails(%a: num[L; ()], %b: num[L; ()]) -> (num[L; ()]) {{
  %s = add %a, %b : num[L; ()]
  return %s
}}
fn @prints(%a: num[L; ()]) -> () {{
  print %a
  return
}}
"
    );
    assert_eq!(effects_of(&t), ["", "fail", "io"]);
}

#[test]
fn a_fused_kernel_has_the_effects_of_its_body() {
    let t = format!(
        "{HEAD}
fn @pure(%a: num[L; (3)]) -> (num[L; (3)]) {{
  %k = fused %a : num[L; (3)] {{
    ^1(%x: num[L; ()]):
      %n = neg %x : num[L; ()]
      yield %n
  }}
  return %k
}}
fn @fails(%a: num[L; (3)], %b: num[L; (3)]) -> (num[L; (3)]) {{
  %k = fused %a, %b : num[L; (3)] {{
    ^1(%x: num[L; ()], %y: num[L; ()]):
      %s = add %x, %y : num[L; ()]
      yield %s
  }}
  return %k
}}
"
    );
    assert_eq!(effects_of(&t), ["", "fail"]);
    // `@fails` declares no effects, so the verifier rejects the module.
    let m = parse(&t).unwrap();
    let errors = verify(&m).unwrap_err();
    assert!(errors[0].message.contains("miss the inferred ones: fail"));
}

#[test]
fn nested_regions_count() {
    let t = format!(
        "{HEAD}
fn @f(%c: bool, %a: num[L; ()]) -> () {{
  if %c {{
    ^1:
      print %a
      yield
  }} {{
    ^2:
      yield
  }}
  return
}}
fn @g(%c: bool, %a: num[L; ()]) -> (num[L; ()]) {{
  %r = while %a : num[L; ()] {{
    ^1(%i: num[L; ()]):
      condition %c, %i
  }} {{
    ^2(%j: num[L; ()]):
      %k = div %j, %j : num[1; ()]
      yield %j
  }}
  return %r
}}
"
    );
    // `if` and `while` have no effects of their own; the regions do.
    assert_eq!(effects_of(&t), ["io", "fail"]);
}

#[test]
fn a_call_takes_the_callees_effects_and_keeps_the_call_mark() {
    // The callers come before the callees on purpose: order must not matter.
    let t = format!(
        "{HEAD}
fn @outer(%a: num[L; ()]) -> () {{
  call @middle(%a)
  return
}}
fn @middle(%a: num[L; ()]) -> () {{
  call @show(%a)
  return
}}
fn @calls_pure(%a: num[L; ()]) -> (num[L; ()]) {{
  %r = call @id(%a) : num[L; ()]
  return %r
}}
fn @show(%a: num[L; ()]) -> () {{
  print %a
  return
}}
fn @id(%a: num[L; ()]) -> (num[L; ()]) {{
  return %a
}}
"
    );
    assert_eq!(
        effects_of(&t),
        ["io,call", "io,call", "call", "io", ""],
        "a caller of an IO function is IO, however far away, and a caller of a \
         pure function only has the call mark"
    );
}

#[test]
fn direct_recursion_reaches_a_fixed_point() {
    let t = format!(
        "{HEAD}
fn @loop_forever(%a: num[L; ()]) -> () {{
  call @loop_forever(%a)
  return
}}
fn @count(%c: bool, %a: num[L; ()]) -> () {{
  if %c {{
    ^1:
      %b = div %a, %a : num[1; ()]
      call @count(%c, %a)
      yield
  }} {{
    ^2:
      yield
  }}
  return
}}
"
    );
    assert_eq!(effects_of(&t), ["call", "fail,call"]);
}

#[test]
fn mutual_recursion_shares_effects_around_the_cycle() {
    let t = format!(
        "{HEAD}
fn @even(%a: num[L; ()]) -> () {{
  call @odd(%a)
  return
}}
fn @odd(%a: num[L; ()]) -> () {{
  call @even(%a)
  call @out(%a)
  return
}}
fn @out(%a: num[L; ()]) -> () {{
  print %a
  return
}}
fn @unrelated(%a: num[L; ()]) -> (num[L; ()]) {{
  return %a
}}
"
    );
    assert_eq!(effects_of(&t), ["io,call", "io,call", "io", ""]);
}

#[test]
fn a_call_to_a_missing_function_only_has_the_call_mark() {
    let t = format!(
        "{HEAD}
fn @f(%a: num[L; ()]) -> () {{
  call @nowhere(%a)
  return
}}
"
    );
    assert_eq!(effects_of(&t), ["call"]);
}

#[test]
fn operations_removed_from_their_block_do_not_count() {
    let s = Span::new(0, 0);
    let mut b = FunctionBuilder::new("f", vec![len()], vec![], s);
    let a = b.params()[0];
    let (print_op, _) = b.op(OpKind::Print, vec![a], vec![], s);
    b.op(OpKind::Return, vec![], vec![], s);
    let mut f = b.finish();
    let entry = f.entry();
    f.blocks[entry.index()].ops.retain(|o| *o != print_op);
    let m = module(vec![f]);
    assert_eq!(infer(&m), [EffectSet::PURE]);
}

#[test]
fn inference_is_total_on_malformed_modules() {
    let s = Span::new(0, 0);
    let mut b = FunctionBuilder::new("f", vec![len()], vec![], s);
    let a = b.params()[0];
    let (o, _) = b.op(OpKind::Print, vec![a], vec![], s);
    b.op(OpKind::Return, vec![], vec![], s);
    let mut f = b.finish();
    // A region that contains itself, an op id and a block id out of range.
    let entry = f.entry();
    f.blocks[entry.index()].ops.push(OpId(99));
    let inner = f.regions.len() as u32;
    f.regions.push(RegionData {
        blocks: vec![entry, BlockId(77)],
        parent: Some(o),
    });
    f.ops[o.index()].regions.push(RegionId(inner));
    f.ops[o.index()].regions.push(f.body);
    f.ops[o.index()].regions.push(RegionId(55));
    let mut m = Module {
        functions: vec![f],
        ..Module::default()
    };
    assert_eq!(infer(&m), [EffectSet::of(Effect::Io)]);
    assert!(infer_effects(&mut m).is_yes());
    // Out-of-range body.
    m.functions[0].body = RegionId(1234);
    assert_eq!(infer(&m), [EffectSet::PURE]);
}

// ----- the verifier rule -----

fn div_fn(declared: EffectSet) -> Module {
    let s = Span::new(0, 0);
    let mut b = FunctionBuilder::new("f", vec![len(), len()], vec![one()], s);
    let [x, y] = b.params()[..] else {
        unreachable!()
    };
    let q = b.op1(OpKind::Div, vec![x, y], one(), s);
    b.op(OpKind::Return, vec![q], vec![], s);
    b.set_effects(declared);
    Module {
        edition: sky_ir::Edition::E2026,
        functions: vec![b.finish()],
    }
}

#[test]
fn the_verifier_requires_declared_effects_to_include_the_inferred_ones() {
    let missing = div_fn(EffectSet::PURE);
    let errs = verify(&missing).unwrap_err();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        errs[0].to_string(),
        "@f: the declared effects (none) miss the inferred ones: fail"
    );
    let wrong = div_fn(EffectSet::of(Effect::Io));
    let errs = verify(&wrong).unwrap_err();
    assert!(
        errs[0].message.contains("declared effects (io) miss"),
        "{errs:?}"
    );
    verify(&div_fn(EffectSet::of(Effect::Fail))).unwrap();
    // More than inferred is allowed: a pass may have removed the cause.
    verify(&div_fn(EffectSet::of(Effect::Fail).with(Effect::Io))).unwrap();
}

#[test]
fn the_verifier_checks_the_effects_of_callees() {
    let t = format!(
        "{HEAD}
fn @show(%a: num[L; ()]) -> () effects(io) {{
  print %a
  return
}}
fn @outer(%a: num[L; ()]) -> () effects(call) {{
  call @show(%a)
  return
}}
"
    );
    let m = parse(&t).unwrap();
    let errs = verify(&m).unwrap_err();
    assert_eq!(
        errs[0].to_string(),
        "@outer: the declared effects (call) miss the inferred ones: io"
    );
}

#[test]
fn the_effects_pass_fixes_and_tightens_declarations() {
    let mut m = div_fn(EffectSet::PURE);
    // Verification is off for the first run: the input is invalid on purpose.
    let report = PassManager::for_tests()
        .verify_each(false)
        .with(InferEffects)
        .run(&mut m)
        .unwrap();
    assert_eq!(report.ran[0].0, "effects");
    assert!(report.ran[0].1.is_yes());
    assert_eq!(m.functions[0].effects, EffectSet::of(Effect::Fail));
    verify(&m).unwrap();

    // Declaring too much is tightened to what the body needs.
    m.functions[0].effects = EffectSet::of(Effect::Io).with(Effect::Fail);
    let report = PassManager::for_tests()
        .with(InferEffects)
        .run(&mut m)
        .unwrap();
    assert!(report.ran[0].1.is_yes());
    assert_eq!(m.functions[0].effects, EffectSet::of(Effect::Fail));
    // Already tight: no change, and no explanations either way.
    let report = PassManager::for_tests()
        .with(InferEffects)
        .run(&mut m)
        .unwrap();
    assert!(!report.ran[0].1.is_yes());
    assert!(report.explanations.is_empty());
}

#[test]
fn text_round_trips_with_inferred_effects() {
    let mut m = parse(&format!(
        "{HEAD}
fn @show(%a: num[L; ()]) -> () {{
  print %a
  return
}}
fn @outer(%a: num[L; ()]) -> () {{
  call @show(%a)
  return
}}
"
    ))
    .unwrap();
    let _ = infer_effects(&mut m);
    let text = print(&m);
    assert!(
        text.contains("fn @outer(%a: num[L; ()]) -> () effects(io, call) {"),
        "{text}"
    );
    assert_eq!(parse(&text).unwrap(), m);
    verify(&m).unwrap();
}
