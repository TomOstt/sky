//! `parse(print(m)) == m` on hand-built modules: every `OpKind`, nested
//! control flow, every type form, floats, strings, names and spans.

mod common;

use common::*;
use sky_ir::{
    canonicalize, parse, print, verify, CmpOp, FunctionBuilder, Module, OpKind, Repr, Type,
};
use sky_units::{BaseDim, Dimension, Rational};

fn roundtrip(m: &Module) -> Module {
    let text = print(m);
    let back = parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(print(&back), text, "printing is a fixed point");
    back
}

#[test]
fn every_op_kind_round_trips_exactly() {
    let m = all_ops();
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{}", print(&m)));
    assert_eq!(roundtrip(&m), m);
    assert_eq!(canonicalize(&m), m, "the builder order is canonical");
}

#[test]
fn all_ops_module_uses_every_mnemonic() {
    let text = print(&all_ops());
    let mnemonics = [
        "const.float",
        "const.rational",
        "const.bool",
        "const.str",
        "const.nothing",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "pow.const",
        "pow",
        "inverse",
        "matmul",
        "transpose",
        "solve",
        "cmp.eq",
        "cmp.ne",
        "cmp.lt",
        "cmp.le",
        "cmp.gt",
        "cmp.ge",
        "not",
        "show",
        "fused",
        "if",
        "while",
        "yield",
        "condition",
        "call",
        "return",
        "print",
    ];
    for name in mnemonics {
        let found = text.lines().any(|l| {
            let l = l.trim_start();
            let l = l.split_once(" = ").map_or(l, |(_, rest)| rest);
            l.split([' ', '[']).next() == Some(name)
        });
        assert!(found, "no `{name}` in\n{text}");
    }
}

#[test]
fn every_type_form_round_trips() {
    let types = all_types();
    let mut b = FunctionBuilder::new("types", types.clone(), types, sp(0, 0));
    let ps = b.params();
    b.op(OpKind::Return, ps, vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    let text = print(&m);
    assert!(text.contains("num[L^(1/2)·M^(-3/2)·T¹⁰"), "{text}");
    assert!(text.contains("\"my var\"#18446744073709551615"), "{text}");
    assert!(text.contains("bool") && text.contains("unknown") && text.contains("nothing"));
    assert_eq!(roundtrip(&m), m);
}

#[test]
fn repr_is_shown_only_when_chosen() {
    let text = print(&all_ops());
    assert!(text.contains("; f64]") && text.contains("; exact]"));
    assert!(!text.contains("unchosen"));
    let plain = format!("{}", len());
    assert_eq!(plain, "num[L; ()]");
    let f32 = format!(
        "{}",
        num(
            Dimension::TIME,
            sky_types::Shape::vector(3),
            sky_types::TagSet::EMPTY,
            Repr::F32
        )
    );
    assert_eq!(f32, "num[T; (3); f32]");
}

#[test]
fn every_dimension_round_trips() {
    // Each base dimension with several exponents, and some combinations.
    let mut dims = Vec::new();
    for b in BaseDim::ALL {
        for e in [
            Rational::ONE,
            Rational::integer(-1),
            Rational::integer(2),
            Rational::integer(-13),
            Rational::new(1, 2),
            Rational::new(-5, 3),
            Rational::integer(i64::MAX),
            Rational::integer(i64::MIN + 1),
        ] {
            dims.push(Dimension::base(b).pow_rat(e).unwrap());
        }
    }
    dims.push(Dimension::DIMENSIONLESS);
    dims.push(Dimension::INDUCTANCE);
    dims.push(Dimension::CAPACITANCE);
    let params: Vec<Type> = dims.iter().map(|&d| Type::scalar(d)).collect();
    let mut b = FunctionBuilder::new("dims", params, vec![], sp(0, 0));
    b.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    assert_eq!(roundtrip(&m), m);
}

#[test]
fn floats_print_exactly() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.5,
        0.1,
        1e300,
        -1e-300,
        5e-324,
        f64::MIN_POSITIVE,
        f64::MAX,
        std::f64::consts::PI,
        123456789.125,
    ];
    let mut bits: Vec<u64> = values.iter().map(|v: &f64| v.to_bits()).collect();
    // Not finite: printed as hex bits, including a NaN payload.
    bits.extend([
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        f64::NAN.to_bits(),
        0x7ff8_0000_dead_beef,
    ]);
    let mut b = FunctionBuilder::new("floats", vec![], vec![], sp(0, 0));
    for &x in &bits {
        b.op1(OpKind::ConstFloat { bits: x }, vec![], one(), sp(0, 0));
    }
    b.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    let text = print(&m);
    assert!(text.contains("const.float 0x7ff8"), "{text}");
    assert!(text.contains("const.float 10000") || !text.contains("10000.0"));
    assert_eq!(roundtrip(&m), m);
}

#[test]
fn constants_and_strings_round_trip() {
    let mut b = FunctionBuilder::new("consts", vec![], vec![], sp(0, 0));
    for s in [
        "",
        "plain",
        "quote \" backslash \\ tab \t newline \n cr \r",
        "unicode é 日本 Θ",
        "nul \0 bell \u{7} bidi \u{202e} zwj \u{200d}",
    ] {
        b.op1(OpKind::ConstStr(s.into()), vec![], Type::Str, sp(0, 0));
    }
    for r in [
        Rational::ZERO,
        Rational::new(-3, 4),
        Rational::integer(i64::MAX),
        Rational::new(i64::MIN + 1, 7),
    ] {
        b.op1(OpKind::ConstRational(r), vec![], one(), sp(0, 0));
    }
    b.op1(OpKind::ConstBool(false), vec![], Type::Bool, sp(0, 0));
    b.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    let text = print(&m);
    assert!(
        !text.contains('\u{202e}'),
        "display-unsafe characters are escaped"
    );
    assert!(text.contains("\\u{202e}"));
    assert_eq!(roundtrip(&m), m);
}

#[test]
fn spans_are_restored() {
    let mut b = FunctionBuilder::new("spans", vec![], vec![], sp(5, 500));
    b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp(3, 7));
    b.op1(OpKind::ConstBool(true), vec![], Type::Bool, sp(0, 0));
    b.op1(
        OpKind::ConstBool(true),
        vec![],
        Type::Bool,
        sp(u32::MAX - 1, u32::MAX),
    );
    b.op(OpKind::Return, vec![], vec![], sp(9, 9));
    let m = module(vec![b.finish()]);
    let text = print(&m);
    assert!(text.contains("@ 3..7") && text.contains("@ 5..500") && text.contains("@ 9..9"));
    assert_eq!(roundtrip(&m), m);
}

/// `depth` nested `if`s, each with the next one in its `then` arm.
fn nested_ifs(depth: usize) -> Module {
    let mut b = FunctionBuilder::new("nest", vec![Type::Bool, len()], vec![len()], sp(0, 0));
    let [c, x] = b.params()[..] else {
        unreachable!()
    };
    let mut frames = Vec::new();
    for d in 0..depth {
        let (iff, res) = b.op(OpKind::If, vec![c], vec![len()], sp(d as u32, d as u32 + 1));
        let outer = b.insertion_block();
        let then = b.add_region(iff);
        let tb = b.add_block(then, vec![]);
        b.set_insertion_block(tb);
        frames.push((iff, res[0], outer));
    }
    // Innermost `then` yields x; unwind, closing each `then` and adding `else`.
    let mut inner = x;
    while let Some((iff, res, outer)) = frames.pop() {
        b.op(OpKind::Yield, vec![inner], vec![], sp(0, 0));
        let els = b.add_region(iff);
        let eb = b.add_block(els, vec![]);
        b.set_insertion_block(eb);
        let n = b.op1(OpKind::Neg, vec![x], len(), sp(0, 0));
        b.op(OpKind::Yield, vec![n], vec![], sp(0, 0));
        b.set_insertion_block(outer);
        inner = res;
    }
    b.op(OpKind::Return, vec![inner], vec![], sp(0, 0));
    module(vec![b.finish()])
}

#[test]
fn nested_if_round_trips() {
    for depth in [1, 2, 5, 40] {
        let m = nested_ifs(depth);
        verify(&m).unwrap_or_else(|e| panic!("{e:#?}"));
        // The builder above creates each `else` region after the `then`
        // contents, which is not a pre-order creation of regions and blocks
        // interleaved with values... but it is: compare canonically.
        assert_eq!(roundtrip(&m), canonicalize(&m), "depth {depth}");
    }
}

/// A while loop whose `after` region holds another while loop.
#[test]
fn nested_while_round_trips() {
    let mut b = FunctionBuilder::new("loops", vec![len()], vec![len()], sp(0, 0));
    let x = b.params()[0];
    let c1 = b.op1(
        OpKind::ConstFloat {
            bits: 1f64.to_bits(),
        },
        vec![],
        len(),
        sp(0, 0),
    );
    let mut outer_block = b.insertion_block();
    let mut acc = x;
    let mut whiles = Vec::new();
    for _ in 0..3 {
        let (w, res) = b.op(OpKind::While, vec![acc], vec![len()], sp(0, 0));
        let before = b.add_region(w);
        let bb = b.add_block(before, vec![len()]);
        b.set_insertion_block(bb);
        let i = b.func().block(bb).args[0];
        let go = b.op1(OpKind::Cmp(CmpOp::Lt), vec![i, c1], Type::Bool, sp(0, 0));
        b.op(OpKind::Condition, vec![go, i], vec![], sp(0, 0));
        let after = b.add_region(w);
        let ab = b.add_block(after, vec![len()]);
        b.set_insertion_block(ab);
        acc = b.func().block(ab).args[0];
        whiles.push((res[0], outer_block));
        outer_block = ab;
    }
    // Innermost after-region: yield acc; then unwind, yielding each result.
    let mut v = acc;
    while let Some((res, block)) = whiles.pop() {
        b.op(OpKind::Yield, vec![v], vec![], sp(0, 0));
        b.set_insertion_block(block);
        v = res;
    }
    b.op(OpKind::Return, vec![v], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{}", print(&m)));
    assert_eq!(roundtrip(&m), canonicalize(&m));
}

#[test]
fn creation_order_does_not_matter() {
    // Both regions of a `while` are created before either is filled: the
    // arenas differ from the parser's pre-order, but the text and the
    // canonical form agree.
    let mut b = FunctionBuilder::new("f", vec![len()], vec![len()], sp(0, 0));
    let x = b.params()[0];
    let (w, res) = b.op(OpKind::While, vec![x], vec![len()], sp(0, 0));
    let outer = b.insertion_block();
    let before = b.add_region(w);
    let after = b.add_region(w);
    let bb = b.add_block(before, vec![len()]);
    let ab = b.add_block(after, vec![len()]);
    b.set_insertion_block(ab);
    let a = b.func().block(ab).args[0];
    b.op(OpKind::Yield, vec![a], vec![], sp(0, 0));
    b.set_insertion_block(bb);
    let i = b.func().block(bb).args[0];
    let go = b.op1(OpKind::ConstBool(false), vec![], Type::Bool, sp(0, 0));
    b.op(OpKind::Condition, vec![go, i], vec![], sp(0, 0));
    b.set_insertion_block(outer);
    b.op(OpKind::Return, vec![res[0]], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    verify(&m).unwrap();
    assert_ne!(parse(&print(&m)).unwrap(), m, "arena order differs");
    assert_eq!(roundtrip(&m), canonicalize(&m));
}

#[test]
fn names_are_uniquified_and_bad_names_dropped() {
    let mut b = FunctionBuilder::new("names", vec![len(), len()], vec![], sp(0, 0));
    let [p, q] = b.params()[..] else {
        unreachable!()
    };
    b.set_name(p, "x");
    b.set_name(q, "x");
    let s = b.op1(OpKind::Add, vec![p, q], len(), sp(0, 0));
    b.set_name(s, "x_1");
    let t = b.op1(OpKind::Add, vec![p, s], len(), sp(0, 0));
    b.set_name(t, "has space");
    let u = b.op1(OpKind::Add, vec![p, t], len(), sp(0, 0));
    b.set_name(u, "42");
    let w = b.op1(OpKind::Add, vec![p, u], len(), sp(0, 0));
    b.set_name(w, "θ");
    b.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let m = module(vec![b.finish()]);
    let text = print(&m);
    assert!(
        text.contains("%x: ") && text.contains("%x_1: ") && text.contains("%x_1_1 = "),
        "{text}"
    );
    assert!(text.contains("%θ = "), "{text}");
    let back = roundtrip(&m);
    assert_eq!(back, canonicalize(&m));
    let f = &back.functions[0];
    let names: Vec<_> = f.values.iter().map(|v| v.name.clone()).collect();
    assert_eq!(
        names,
        [
            Some("x".into()),
            Some("x_1".into()),
            Some("x_1_1".into()),
            None,
            None,
            Some("θ".into())
        ]
    );
}

#[test]
fn removed_operations_are_not_printed() {
    let mut b = FunctionBuilder::new("f", vec![len()], vec![len()], sp(0, 0));
    let x = b.params()[0];
    let dead = b.op1(OpKind::Neg, vec![x], len(), sp(1, 2));
    let live = b.op1(OpKind::Add, vec![x, x], len(), sp(3, 4));
    b.op(OpKind::Return, vec![live], vec![], sp(0, 0));
    let mut f = b.finish();
    // What a pass does when it deletes an operation: unlink it from its block.
    let entry = f.entry();
    let dead_op = match f.values[dead.index()].def {
        sky_ir::ValueDef::OpResult { op, .. } => op,
        _ => unreachable!(),
    };
    f.blocks[entry.index()].ops.retain(|&o| o != dead_op);
    let m = module(vec![f]);
    verify(&m).unwrap();
    let text = print(&m);
    assert!(!text.contains("neg"), "{text}");
    let back = roundtrip(&m);
    assert_eq!(back, canonicalize(&m));
    assert_eq!(back.functions[0].ops.len(), 2);
}

#[test]
fn function_names_and_modules() {
    let odd = FunctionBuilder::new("has space \"and\" quotes", vec![], vec![], sp(0, 0));
    let mut odd = odd;
    odd.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let mut main = FunctionBuilder::new(Module::MAIN, vec![], vec![], sp(0, 0));
    main.op(OpKind::Return, vec![], vec![], sp(0, 0));
    let m = module(vec![main.finish(), odd.finish()]);
    let text = print(&m);
    assert!(
        text.starts_with("sir edition 2026\n\nfn @<main>() -> () {\n"),
        "{text}"
    );
    assert_eq!(roundtrip(&m), m);
    let empty = module(vec![]);
    assert_eq!(print(&empty), "sir edition 2026\n");
    assert_eq!(roundtrip(&empty), empty);
}

#[test]
fn design_doc_example_parses() {
    let text = "\
sir edition 2026
fn @main() -> () effects(io, fail) {
  %0 = const.float 10000 : num[L; ()]
  %1 = const.rational 1/1000 : num[1; ()]
  %x = mul %0, %1 : num[L; ()]              @ 3..8
  print %x                                   @ 9..17
  return
}
";
    let m = parse(text).unwrap();
    verify(&m).unwrap();
    assert_eq!(m.functions[0].ops.len(), 5);
    assert_eq!(
        print(&m),
        "\
sir edition 2026

fn @main() -> () effects(io, fail) {
  %0 = const.float 10000 : num[L; ()]
  %1 = const.rational 1/1000 : num[1; ()]
  %x = mul %0, %1 : num[L; ()] @ 3..8
  print %x @ 9..17
  return
}
"
    );
}

// ----- operand spans -----

/// `x + x` with the given operation span and operand spans.
fn add_with_spans(span: sky_diag::Span, operand_spans: Vec<sky_diag::Span>) -> Module {
    let mut b = FunctionBuilder::new("f", vec![len()], vec![len()], sp(0, 0));
    let x = b.params()[0];
    let (_, s) = b.op_with_spans(OpKind::Add, vec![x, x], operand_spans, vec![len()], span);
    b.op(OpKind::Return, vec![s[0]], vec![], sp(0, 0));
    module(vec![b.finish()])
}

#[test]
fn operand_spans_print_after_the_span_and_only_when_known() {
    let m = add_with_spans(sp(10, 20), vec![sp(10, 11), sp(14, 20)]);
    let text = print(&m);
    assert!(
        text.contains("add %0, %0 : num[L; ()] @ 10..20 (10..11, 14..20)\n"),
        "{text}"
    );
    assert_eq!(roundtrip(&m), m);
    // Unknown spans are not printed, and read back as unknown.
    let bare = add_with_spans(sp(10, 20), vec![]);
    let text = print(&bare);
    assert!(
        text.contains("add %0, %0 : num[L; ()] @ 10..20\n"),
        "{text}"
    );
    assert_eq!(roundtrip(&bare), bare);
    assert!(roundtrip(&bare).functions[0].ops[0]
        .operand_spans
        .is_empty());
}

#[test]
fn operand_spans_survive_an_operation_span_of_zero() {
    // The operation's own span is written (`@ 0..0`) so the operand spans have
    // a place to follow it.
    let m = add_with_spans(sp(0, 0), vec![sp(3, 4), sp(0, 0)]);
    let text = print(&m);
    assert!(text.contains("@ 0..0 (3..4, 0..0)\n"), "{text}");
    assert_eq!(roundtrip(&m), m);
}

#[test]
fn canonicalize_keeps_operand_spans() {
    let m = all_ops();
    let with_spans = |m: &Module| {
        m.functions[0]
            .ops
            .iter()
            .filter(|o| !o.operand_spans.is_empty())
            .count()
    };
    assert!(with_spans(&m) >= 4);
    assert_eq!(with_spans(&canonicalize(&m)), with_spans(&m));
    assert_eq!(canonicalize(&m), m);
}

#[test]
fn a_slice_has_a_span_per_operand_and_per_axis() {
    let text = print(&all_ops());
    // `lit[lo:hi, at]` reads the array, two bounds and one position, and has
    // two items.
    assert!(
        text.contains("@ 50..59 (50..51, 52..53, 54..55, 57..58, 52..55, 57..58)\n"),
        "{text}"
    );
}

#[test]
fn a_malformed_operand_span_list_is_a_parse_error() {
    let ok = print(&add_with_spans(sp(10, 20), vec![sp(10, 11), sp(14, 20)]));
    for bad in [
        ok.replace("(10..11, 14..20)", "(10..11, 14..20"),
        ok.replace("(10..11, 14..20)", "(10..11,)"),
        ok.replace("(10..11, 14..20)", "()"),
        ok.replace("(10..11, 14..20)", "(10, 11)"),
        ok.replace("(10..11, 14..20)", "(10..11 14..20)"),
    ] {
        assert!(parse(&bad).is_err(), "{bad}");
    }
}
