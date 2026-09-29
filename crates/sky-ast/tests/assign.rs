//! Assignment targets, compound assignment, and slices (PR 2 targets).

mod common;
use common::{at, lex, module_errors, module_sexpr, sexpr};
use sky_ast::{parse_module, BinaryOp, CompoundOp, MAX_DEPTH};

#[test]
fn compound_op_maps_to_its_binary_op() {
    for (op, symbol, binary) in [
        (CompoundOp::Add, "+=", BinaryOp::Add),
        (CompoundOp::Sub, "-=", BinaryOp::Sub),
        (CompoundOp::Mul, "*=", BinaryOp::Mul),
        (CompoundOp::Div, "/=", BinaryOp::Div),
        (CompoundOp::Pow, "^=", BinaryOp::Pow),
        (CompoundOp::MatMul, "@=", BinaryOp::MatMul),
    ] {
        assert_eq!(op.symbol(), symbol);
        assert_eq!(op.binary(), binary);
        assert_eq!(format!("{}=", binary.symbol()), symbol);
    }
}

#[test]
fn targets_print_as_specified() {
    for (src, tree) in [
        ("x = e", "(= x e)"),
        ("A[i, j] = x", "(= (index A i j) x)"),
        ("A[0, :] = v", "(= (index A 0 (: _ _)) v)"),
        ("a.b = x", "(= (attr a b) x)"),
        ("x.true = 1", "(= (attr x true) 1)"),
        ("x += e", "(+= x e)"),
        ("A += B", "(+= A B)"),
        ("A[i, j] *= 2", "(*= (index A i j) 2)"),
        ("x ^= 2", "(^= x 2)"),
        ("P @= F", "(@= P F)"),
        ("x -= 1", "(-= x 1)"),
        ("x /= 2 s", "(/= x (unit 2 s))"),
        ("a.b[0].c = 1", "(= (attr (index (attr a b) 0) c) 1)"),
        ("x: Real = 1", "(: x Real 1)"),
    ] {
        assert_eq!(module_sexpr(src), tree, "{src:?}");
    }
}

#[test]
fn compound_assignment_is_never_desugared() {
    // `x += 1` is its own node: `x` appears once, and there is no `+`.
    assert_eq!(module_sexpr("x += 1"), "(+= x 1)");
    assert_eq!(module_sexpr("x += a + b"), "(+= x (+ a b))");
    assert_eq!(module_sexpr("x ^= 2 ^ 3"), "(^= x (^ 2 3))");
}

#[test]
fn slices_on_the_read_side() {
    for (src, tree) in [
        ("A[0, :]", "(index A 0 (: _ _))"),
        ("A[1:3]", "(index A (: 1 3))"),
        ("A[:n, 2:]", "(index A (: _ n) (: 2 _))"),
        ("A[:]", "(index A (: _ _))"),
        ("A[i+1:j-1]", "(index A (: (+ i 1) (- j 1)))"),
        ("A[1:3,]", "(index A (: 1 3))"),
        ("A[B[1:2]:]", "(index A (: (index B (: 1 2)) _))"),
        ("Matrix[3, 4]", "(index Matrix 3 4)"),
    ] {
        assert_eq!(sexpr(src), tree, "{src:?}");
    }
    assert_eq!(module_sexpr("v = A[0, :]"), "(= v (index A 0 (: _ _)))");
}

#[test]
fn slice_steps_are_errors() {
    for (src, at_text) in [("A[1:2:3]", ":"), ("A[::2]", ":"), ("A[1::]", ":")] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            "slice steps (`start:stop:step`) are not supported in Sky v0.1"
        );
        assert_eq!(at(src, errors[0].span), at_text);
        assert!(errors[0].span.start > errors[0].secondary[0].span.start);
    }
    for (src, message) in [
        ("A[]", "expected at least one index between `[` and `]`"),
        ("A[,]", "expected an expression, found `,`"),
        ("A[1,,2]", "expected an expression, found `,`"),
        (
            "A[1 2]",
            "unexpected integer literal `2` after an expression; expected `,` or `]`",
        ),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, message, "{src:?}");
    }
}

#[test]
fn invalid_targets_are_errors_on_the_target() {
    let help = "only a name (`x`), an index (`A[i]`), or an attribute (`a.b`) can be assigned to";
    for (src, message, target) in [
        (
            "f(x) = 1",
            "cannot assign to a function call `f(x)`",
            "f(x)",
        ),
        (
            "(x) = 1",
            "cannot assign to a parenthesized expression `(x)`",
            "(x)",
        ),
        ("1 = x", "cannot assign to a literal `1`", "1"),
        ("\"s\" = x", "cannot assign to a literal `\"s\"`", "\"s\""),
        ("10 m = x", "cannot assign to a unit literal `10 m`", "10 m"),
        (
            "a + b += 1",
            "cannot assign to an operator expression `a + b`",
            "a + b",
        ),
        (
            "-x = 1",
            "cannot assign to an operator expression `-x`",
            "-x",
        ),
        (
            "A⁻¹ = B",
            "cannot assign to an operator expression `A⁻¹`",
            "A⁻¹",
        ),
        (
            "f(x) *= 2",
            "cannot assign to a function call `f(x)`",
            "f(x)",
        ),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        let err = &errors[0];
        assert_eq!(err.message, message, "{src:?}");
        assert_eq!(at(src, err.span), target, "{src:?}");
        assert_eq!(err.secondary[0].message.as_deref(), Some("because of this"));
        assert!(err.help.as_deref().unwrap().ends_with(help), "{src:?}");
    }
    // #24's message for Bool literals is unchanged, for `=` and `+=` alike.
    for src in ["true = 1", "true += 1"] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}");
        assert_eq!(errors[0].message, "cannot assign to `true`");
        assert_eq!(at(src, errors[0].span), "true");
    }
}

#[test]
fn annotations_need_a_plain_name() {
    for (src, target) in [
        ("a.b: T = 1", "a.b"),
        ("A[0]: T = 1", "A[0]"),
        ("f(x): T", "f(x)"),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            "invalid annotation target: only a plain name can have a type annotation"
        );
        assert_eq!(at(src, errors[0].span), target);
    }
    let src = "x: Real += 1";
    let errors = module_errors(src);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(errors[0].message, "`+=` cannot follow a type annotation");
    assert_eq!(at(src, errors[0].span), "+=");
}

#[test]
fn chained_assignments_are_errors() {
    for (src, message, op) in [
        (
            "a = b = c",
            "chained assignment (`a = b = c`) is not supported",
            "=",
        ),
        (
            "x += y += z",
            "chained assignment (`x += y += z`) is not supported",
            "+=",
        ),
        (
            "a = b += c",
            "chained assignment (`x += y += z`) is not supported",
            "+=",
        ),
        (
            "x += y = z",
            "chained assignment (`a = b = c`) is not supported",
            "=",
        ),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, message, "{src:?}");
        let tail = &src[errors[0].span.start as usize..];
        assert!(tail.starts_with(op), "{src:?}");
        assert_eq!(at(src, errors[0].span), op);
    }
}

#[test]
fn double_star_assign_points_to_caret_assign() {
    for (src, span_text) in [
        ("x **= 2", "**="),
        ("x**=2", "**="),
        ("x = 10 m**=2", "**="),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, "`**=` is not an operator");
        assert_eq!(at(src, errors[0].span), span_text);
        assert_eq!(
            errors[0].help.as_deref(),
            Some("Sky writes powers as `^`; the compound assignment is `^=`")
        );
    }
    // Separated, it is a plain missing-operand error, as for `* *`.
    assert!(module_errors("x * *= 2")[0]
        .message
        .starts_with("expected an expression"));
}

#[test]
fn plus_then_equals_is_not_compound() {
    let src = "a+ =b";
    let errors = module_errors(src);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(errors[0].message, "expected an expression, found `=`");
}

#[test]
fn quoted_target_text_is_bounded() {
    // A 100k-character name must not make a 100k-character message.
    let name = "f".repeat(100_000);
    let src = format!("{name}(x) = 1");
    let errors = module_errors(&src);
    assert_eq!(errors.len(), 1);
    let message = &errors[0].message;
    assert!(message.len() < 200, "{} bytes", message.len());
    let quoted = format!("`{}…`", "f".repeat(64));
    assert!(message.ends_with(&quoted), "{message}");
    assert_eq!(errors[0].span.range(), 0..src.len() - 4);
    let diag = errors[0].clone().into_diagnostic();
    assert!(diag.help.iter().all(|h| h.len() < 200));

    // A long multi-line target is cut at its first line.
    let src = format!("(a +\n {name}) = 1");
    let errors = module_errors(&src);
    assert_eq!(errors.len(), 1);
    assert_eq!(
        errors[0].message,
        "cannot assign to a parenthesized expression `(a +…`"
    );

    // Other diagnostics that quote a name are bounded too.
    let src = format!("x = y {name}");
    let errors = module_errors(&src);
    assert!(errors[0].message.len() < 200);
    assert!(errors[0].help.as_deref().unwrap().len() < 300);
}

#[test]
fn an_index_or_slice_level_counts_once() {
    assert_eq!(MAX_DEPTH, 256);
    let nest = |item: &str, n: usize| format!("{}0{}", item.repeat(n), "]".repeat(n));
    for item in ["a[", "a[1:", "a[:", "a[0, "] {
        let n = MAX_DEPTH as usize;
        let ok = nest(item, n);
        let (_, errors) = parse_module(&lex(&ok), &ok);
        assert!(errors.is_empty(), "{item} x{n}: {errors:#?}");
        let too_deep = nest(item, n + 1);
        let (_, errors) = parse_module(&lex(&too_deep), &too_deep);
        assert_eq!(errors.len(), 1, "{item} x{}", n + 1);
        assert_eq!(
            errors[0].message,
            "expression is nested too deeply (the limit is 256 levels)"
        );
    }
    // Both bounds of one slice sit at the same level.
    let n = MAX_DEPTH as usize - 1;
    let deep = format!("{}0{}", "a[".repeat(n), "]".repeat(n));
    let src = format!("x = A[{deep}:{deep}]");
    let (_, errors) = parse_module(&lex(&src), &src);
    assert!(errors.is_empty(), "{errors:#?}");
    // As an assignment target, too.
    let src = format!("{} = 1", nest("a[1:", MAX_DEPTH as usize));
    let (_, errors) = parse_module(&lex(&src), &src);
    assert!(errors.is_empty(), "{errors:#?}");
}
