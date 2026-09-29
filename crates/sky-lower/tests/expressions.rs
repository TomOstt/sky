//! Lowering of expressions: literals, operators, short-circuiting, calls to
//! the builtins, compound assignment, result types and spans.

mod common;

use common::*;
use sky_interp::{RuntimeEnv, Value};
use sky_ir::{CmpOp, Effect, OpKind, Type};
use sky_units::{Dimension, Rational};

/// What the tree-walking interpreter binds `x` to after `src`.
fn interp_x(src: &str) -> (f64, Dimension) {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let mut env = RuntimeEnv::with_prelude();
        let mut out = Vec::new();
        sky_interp::run_module(&ast, src, &mut env, &mut out).expect("runs");
        match env.get("x") {
            Some(Value::Number { value, dim }) => (*value, *dim),
            other => panic!("x is {other:?}"),
        }
    })
}

/// The `const.float` `x` is bound to, and its dimension.
fn lowered_x(src: &str) -> (f64, Dimension) {
    let m = lower_ok(src);
    let f = main_fn(&m);
    let v = named(f, "x");
    let op = def_op(f, v).expect("x is an operation result");
    let OpKind::ConstFloat { bits } = f.op(op).kind else {
        panic!("x is not a const.float\n{}", dump(f))
    };
    let dim = f.ty(v).as_number().expect("a number").dim;
    (f64::from_bits(bits), dim)
}

#[test]
fn literals_are_bit_for_bit_what_the_interpreter_computes() {
    for lit in [
        "1",
        "0.1",
        "1_000",
        "2.5e-3",
        "1e10",
        "36 km/h",
        "1_000 m",
        "2.5e-3 s",
        "9.81 m/s^2",
        "0.1 s",
        "500 nm",
        "25 meV",
        "1 GeV",
        "10 km^-1",
        "300 K",
        "100 kPa",
        "3 m^(1/2)",
        "7 kg*m/s^2",
        "1 ly",
        "0.3 pc",
        "12 min",
        "1 AU",
        "3.5 yr",
        "c",
        "h",
        "k_B",
        "G",
        "hbar",
        "ħ",
    ] {
        let src = format!("x = {lit}\n");
        let (iv, idim) = interp_x(&src);
        let (lv, ldim) = lowered_x(&src);
        assert_eq!(lv.to_bits(), iv.to_bits(), "{lit}: {lv} vs {iv}");
        assert_eq!(ldim, idim, "{lit}");
    }
}

#[test]
fn a_unit_literal_is_a_single_si_constant() {
    let m = lower_ok("x = 36 km/h\n");
    let f = main_fn(&m);
    assert_eq!(top_kinds(f), ["const.float", "return"]);
    let (v, dim) = lowered_x("x = 36 km/h\n");
    assert_eq!(v, 10.0, "converted factor by factor, so exactly 10");
    assert_eq!(dim, Dimension::VELOCITY);
}

#[test]
fn bool_and_string_literals() {
    let m = lower_ok("a = true\nb = false\ns = \"tab\\there \\\"q\\\" \\\\ \\n\"\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        ["const.bool", "const.bool", "const.str", "return"]
    );
    let ops = &f.block(f.entry()).ops;
    assert_eq!(f.op(ops[0]).kind, OpKind::ConstBool(true));
    assert_eq!(f.op(ops[1]).kind, OpKind::ConstBool(false));
    // Unescaped exactly as the interpreter does.
    let OpKind::ConstStr(s) = &f.op(ops[2]).kind else {
        panic!("const.str")
    };
    let src = "s = \"tab\\there \\\"q\\\" \\\\ \\n\"\n";
    let interp = sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, _) = sky_ast::parse_module(&lexed.tokens, src);
        let mut env = RuntimeEnv::new();
        sky_interp::run_module(&ast, src, &mut env, &mut Vec::new()).expect("runs");
        env.get("s").cloned()
    });
    assert_eq!(interp, Some(Value::Str(s.clone())));
    assert_eq!(f.ty(f.op(ops[2]).results[0]), &Type::Str);
    assert_eq!(f.ty(f.op(ops[0]).results[0]), &Type::Bool);
}

#[test]
fn constants_are_literals_typed_by_the_checker() {
    let m = lower_ok("m = 2 kg\nE = m * c^2\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        ["const.float", "const.float", "pow.const", "mul", "return"]
    );
    let e = named(f, "E");
    assert_eq!(f.ty(e).as_number().expect("number").dim, Dimension::ENERGY);
    let c = f.op(top_ops(f, "const.float")[1]);
    assert_eq!(
        c.kind,
        OpKind::ConstFloat {
            bits: 299_792_458f64.to_bits()
        }
    );
    assert_eq!(
        f.ty(c.results[0]).as_number().expect("number").dim,
        Dimension::VELOCITY
    );
}

#[test]
fn arithmetic_operators_and_their_dimensions() {
    let m = lower_ok(
        "a = 3 m\nb = 2 s\nv = a / b\nw = a * b\ns1 = a + a\ns2 = a - a\nn = -a\np = +a\nprint(p)\n",
    );
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        [
            "const.float",
            "const.float",
            "div",
            "mul",
            "add",
            "sub",
            "neg",
            "print",
            "return"
        ]
    );
    let dim = |n: &str| f.ty(named(f, n)).as_number().expect("number").dim;
    assert_eq!(dim("v"), Dimension::VELOCITY);
    assert_eq!(
        dim("w"),
        Dimension::LENGTH.checked_mul(Dimension::TIME).expect("mul")
    );
    assert_eq!(dim("s1"), Dimension::LENGTH);
    assert_eq!(dim("n"), Dimension::LENGTH);
    // `+a` is `a`: no operation, the name is bound to the same value.
    assert_eq!(f.op(top_op(f, "print")).operands, [named(f, "a")]);
}

#[test]
fn powers_with_a_constant_exponent_are_pow_const() {
    let m = lower_ok(
        "x = 3 m\nn = 2\na = x^2\nb = x^-1\nc = (4 m)^0.5\nd = x^(1/3)\ne = 2^n\nf = n^n\n",
    );
    let f = main_fn(&m);
    let exps: Vec<Rational> = f
        .ops
        .iter()
        .filter_map(|o| match &o.kind {
            OpKind::PowConst { exponent } => Some(*exponent),
            _ => None,
        })
        .collect();
    assert_eq!(
        exps,
        [
            Rational::new(2, 1),
            Rational::new(-1, 1),
            Rational::new(1, 2),
            Rational::new(1, 3)
        ]
    );
    // A computed exponent is `pow` with two operands.
    assert_eq!(count(f, "pow"), 2);
    let pow = f.ops.iter().find(|o| o.kind == OpKind::Pow).expect("pow");
    assert_eq!(pow.operands.len(), 2);
    let dim = |n: &str| f.ty(named(f, n)).as_number().expect("number").dim;
    assert_eq!(
        dim("a"),
        Dimension::LENGTH
            .checked_mul(Dimension::LENGTH)
            .expect("mul")
    );
    assert_eq!(dim("b"), Dimension::LENGTH.inverse().expect("inverse"));
    assert_eq!(
        dim("c"),
        Dimension::LENGTH.pow_rat(Rational::new(1, 2)).expect("pow")
    );
    // The constant exponent's own expression is not lowered: `x^2` has no
    // `const.float 2` of its own.
    // x, n, the base `4 m`, and the base `2` of `2^n`.
    assert_eq!(count(f, "const.float"), 4);
}

#[test]
fn inverse_and_inv_are_the_same_operation() {
    let m = lower_ok(
        "a = (2 s)⁻¹\nb = inv(3 m/s)\nK: PositiveDefinite Matrix[3, 3]\nc = K⁻¹\nd = inv(K)\n",
    );
    let f = main_fn(&m);
    assert_eq!(count(f, "inverse"), 4);
    let dim = |n: &str| f.ty(named(f, n)).as_number().expect("number").dim;
    assert_eq!(dim("a"), Dimension::FREQUENCY);
    assert_eq!(dim("b"), Dimension::VELOCITY.inverse().expect("inverse"));
    // Matrix inverses keep the checker's tags (positive definite inverts to
    // positive definite) and shape.
    let c = f.ty(named(f, "c")).as_number().expect("number").clone();
    assert_eq!(c.shape.rank(), 2);
    assert_eq!(c.tags.to_string(), "{Symmetric, PositiveDefinite}");
    assert_eq!(f.ty(named(f, "d")), f.ty(named(f, "c")));
}

#[test]
fn a_user_defined_inv_and_print_shadow_the_builtins() {
    let m = lower_ok(
        "def inv(x):\n    return x\ndef print(x):\n    return x\na = inv(2 m)\nb = print(1)\n",
    );
    let f = main_fn(&m);
    assert_eq!(count(f, "inverse"), 0);
    assert_eq!(count(f, "print"), 0);
    assert_eq!(count(f, "call"), 2);
    assert!(main_fn(&m).effects.contains(Effect::Call));
    assert!(!main_fn(&m).effects.contains(Effect::Io));
}

#[test]
fn matmul_and_the_checkers_matrix_types() {
    let m = lower_ok("A: Matrix[2, 3] m\nb: Vector[3] s\nx = A @ b\n");
    let f = main_fn(&m);
    assert_eq!(top_kinds(f), ["matmul", "return"]);
    let x = f.ty(named(f, "x")).as_number().expect("number").clone();
    assert_eq!(x.shape.to_string(), "Vector[2]");
    assert_eq!(
        x.dim,
        Dimension::LENGTH.checked_mul(Dimension::TIME).expect("mul")
    );
    // The inputs are the parameters of <main>, in order, with declared types.
    assert_eq!(f.params.len(), 2);
    assert_eq!(f.param_values().len(), 2);
    assert_eq!(f.value(f.param_values()[0]).name.as_deref(), Some("A"));
}

#[test]
fn comparisons_map_to_cmp_ops() {
    let m = lower_ok("a = 1 m\nb = 2 m\nc1 = a == b\nc2 = a != b\nc3 = a < b\nc4 = a <= b\nc5 = a > b\nc6 = a >= b\n");
    let f = main_fn(&m);
    let cmps: Vec<CmpOp> = f
        .ops
        .iter()
        .filter_map(|o| match o.kind {
            OpKind::Cmp(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(
        cmps,
        [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge
        ]
    );
    assert_eq!(f.ty(named(f, "c3")), &Type::Bool);
}

#[test]
fn not_and_bool_equality() {
    let m = lower_ok("done = false\nx = not done\ny = done == x\n");
    let f = main_fn(&m);
    assert_eq!(top_kinds(f), ["const.bool", "not", "cmp.eq", "return"]);
    assert_eq!(f.ty(named(f, "y")), &Type::Bool);
}

#[test]
fn string_comparison_is_a_bool_although_the_checker_says_unknown() {
    let m = lower_ok("s = \"a\"\nt = s == \"b\"\n");
    let f = main_fn(&m);
    assert_eq!(top_kinds(f), ["const.str", "const.str", "cmp.eq", "return"]);
    assert_eq!(f.ty(named(f, "t")), &Type::Bool);
    assert_eq!(f.ty(named(f, "s")), &Type::Str);
}

#[test]
fn and_short_circuits_with_an_if() {
    let m = lower_ok("x = 1 m\nok = x > 0 m and x < 5 m\n");
    let f = main_fn(&m);
    // The second comparison is not at the top level: it is inside the arm
    // that runs only when the first operand was true.
    assert_eq!(
        top_kinds(f),
        ["const.float", "const.float", "cmp.gt", "if", "return"]
    );
    let iff = top_op(f, "if");
    let [then_b, else_b] = arms(f, iff)[..] else {
        panic!("two arms")
    };
    assert_eq!(block_kinds(f, then_b), ["const.float", "cmp.lt", "yield"]);
    assert_eq!(block_kinds(f, else_b), ["const.bool", "yield"]);
    let else_const = f.op(f.block(else_b).ops[0]);
    assert_eq!(
        else_const.kind,
        OpKind::ConstBool(false),
        "and stops on false"
    );
    assert_eq!(f.op(iff).results.len(), 1);
    assert_eq!(f.ty(named(f, "ok")), &Type::Bool);
}

#[test]
fn or_short_circuits_the_other_way() {
    let m = lower_ok("x = 1 m\nok = x > 0 m or x < 5 m\n");
    let f = main_fn(&m);
    let iff = top_op(f, "if");
    let [then_b, else_b] = arms(f, iff)[..] else {
        panic!("two arms")
    };
    assert_eq!(block_kinds(f, then_b), ["const.bool", "yield"]);
    assert_eq!(
        f.op(f.block(then_b).ops[0]).kind,
        OpKind::ConstBool(true),
        "or stops on true"
    );
    assert_eq!(block_kinds(f, else_b), ["const.float", "cmp.lt", "yield"]);
}

#[test]
fn a_chain_of_operands_nests_and_mixed_chains_nest_by_precedence() {
    // a and b and c: two nested ifs; each later operand only in a then-arm.
    let m = lower_ok("a = true\nb = false\nc = true\nx = a and b and c\n");
    let f = main_fn(&m);
    assert_eq!(count(f, "if"), 2);
    // `a and b or c` is Or[And[a, b], c]: an if inside the or's else-less
    // structure; the whole thing has three ifs' worth of results.
    let m = lower_ok("a = true\nb = false\nc = true\nx = a and b or c\n");
    let f = main_fn(&m);
    assert_eq!(count(f, "if"), 2);
    assert!(f
        .ops
        .iter()
        .filter(|o| o.kind == OpKind::If)
        .all(|o| o.results.len() == 1));
}

#[test]
fn conditional_expression_is_an_if_with_a_result() {
    let m = lower_ok("limit = 10 m/s\nspeed = 3 m/s\ncapped = limit if speed > limit else speed\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        ["const.float", "const.float", "cmp.gt", "if", "return"]
    );
    let iff = top_op(f, "if");
    assert_eq!(f.op(iff).results.len(), 1);
    assert_eq!(
        f.ty(f.op(iff).results[0]).as_number().expect("number").dim,
        Dimension::VELOCITY
    );
    for b in arms(f, iff) {
        assert_eq!(block_kinds(f, b), ["yield"]);
    }
}

#[test]
fn only_the_chosen_arm_of_a_conditional_expression_is_evaluated() {
    let m = lower_ok("x = 2 m\ny = x if x > 1 m else x * (1 m / 0 m)\n");
    let f = main_fn(&m);
    // The failing division is inside the else arm, not at the top level.
    assert_eq!(count(f, "div"), 1);
    assert!(top_ops(f, "div").is_empty());
}

#[test]
fn print_is_an_effect_with_no_result() {
    let m = lower_ok("x = 2 m\nprint(x, \"hi\", true)\nprint()\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        [
            "const.float",
            "const.str",
            "const.bool",
            "print",
            "print",
            "return"
        ]
    );
    let prints = top_ops(f, "print");
    assert_eq!(f.op(prints[0]).operands.len(), 3);
    assert!(f.op(prints[0]).results.is_empty());
    assert_eq!(f.op(prints[1]).operands.len(), 0);
    assert!(f.effects.contains(Effect::Io));
}

#[test]
fn as_lowers_to_a_show_op_carrying_the_unit_text() {
    let src = "v = 90 km/h\nprint(v as km/h, 1 AU as km, v)\nv as m/s\n";
    let m = lower_ok(src);
    let f = main_fn(&m);
    // `90 km/h` is folded to SI at its literal, as always; `as` does not
    // convert anything at lowering time, it keeps the unit for `show`.
    assert_eq!(
        top_kinds(f),
        [
            "const.float",
            "show",
            "const.float",
            "show",
            "print",
            "show",
            "return"
        ]
    );
    let shows = top_ops(f, "show");
    let units: Vec<_> = shows
        .iter()
        .map(|&o| match &f.op(o).kind {
            OpKind::Show { unit } => unit.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(units, ["km/h", "km", "m/s"]);
    for &o in &shows {
        let op = f.op(o);
        assert_eq!(op.operands.len(), 1);
        assert_eq!(f.ty(op.results[0]), &Type::Str);
    }
    // The span is the whole `v as km/h`, for runtime errors.
    let first = f.op(shows[0]);
    assert_eq!(&src[first.span.range()], "v as km/h");
    // The operand is the SI value: 90 km/h is 25 m/s.
    let (x, dim) = {
        let v = first.operands[0];
        let op = def_op(f, v).expect("v is an operation result");
        let OpKind::ConstFloat { bits } = f.op(op).kind else {
            panic!("not a const.float\n{}", dump(f))
        };
        (
            f64::from_bits(bits),
            f.ty(v).as_number().expect("number").dim,
        )
    };
    assert_eq!((x, dim), (25.0, Dimension::VELOCITY));
    assert!(f.effects.contains(Effect::Fail));
}

#[test]
fn a_used_print_result_is_nothing() {
    let m = lower_ok("r = print(1)\nprint(r)\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        ["const.float", "print", "const.nothing", "print", "return"]
    );
    assert_eq!(f.ty(named(f, "r")), &Type::Nothing);
}

#[test]
fn compound_assignment_rebinds_the_name() {
    let m = lower_ok("x = 5 m\nx += 2 m\nx -= 1 m\nx *= 2\nx /= 4\nn = 3\nn ^= 2\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        [
            "const.float",
            "const.float",
            "add",
            "const.float",
            "sub",
            "const.float",
            "mul",
            "const.float",
            "div",
            "const.float",
            "pow.const",
            "return"
        ]
    );
    assert_eq!(
        f.ty(named(f, "x")).as_number().expect("number").dim,
        Dimension::LENGTH
    );
}

#[test]
fn compound_assignment_of_a_predefined_constant_reads_it() {
    let m = lower_ok("h += 1 J*s\n");
    let f = main_fn(&m);
    assert_eq!(
        top_kinds(f),
        ["const.float", "const.float", "add", "return"]
    );
}

#[test]
fn every_operation_has_a_span_of_the_source_it_came_from() {
    let src = "a = 3 m\nb = a + 2 m\nb += 1 m\nprint(b)\nok = b > 1 m and not false\n";
    let m = lower_ok(src);
    let f = main_fn(&m);
    for o in &f.ops {
        assert!(o.span.end as usize <= src.len());
        assert!(!o.span.is_empty(), "{}", o.kind.mnemonic());
    }
    let text = |mn: &str, i: usize| {
        let ops: Vec<_> = f
            .ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.kind.mnemonic() == mn)
            .map(|(k, _)| k)
            .collect();
        span_text(src, f, sky_ir::OpId(ops[i] as u32))
    };
    assert_eq!(text("const.float", 0), "3 m");
    assert_eq!(text("add", 0), "a + 2 m");
    // A compound assignment's operation carries the statement's span.
    assert_eq!(text("add", 1), "b += 1 m");
    assert_eq!(text("print", 0), "print(b)");
    assert_eq!(text("cmp.gt", 0), "b > 1 m");
    assert_eq!(text("not", 0), "not false");
    assert_eq!(text("if", 0), "b > 1 m and not false");
}

#[test]
fn effects_are_the_union_of_the_operations() {
    let pure = lower_ok("a = 1\nb = a < 2\nc = not b\n");
    assert!(main_fn(&pure).effects.is_pure());
    let failing = lower_ok("a = 1 m\nb = a / a\n");
    assert!(main_fn(&failing).effects.contains(Effect::Fail));
    assert!(!main_fn(&failing).effects.contains(Effect::Io));
    // Effects of operations inside regions count.
    let nested = lower_ok("a = 1 m\nif a > 0 m:\n    print(a)\n");
    assert!(main_fn(&nested).effects.contains(Effect::Io));
}
