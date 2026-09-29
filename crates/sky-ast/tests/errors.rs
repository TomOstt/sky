//! Error messages, spans, and the depth limit.

mod common;
use common::{at, expr_errors, lex};
use sky_ast::{parse_expr, parse_module, Span, Token, TokenKind, MAX_DEPTH};

#[test]
fn empty_input_is_an_error_for_parse_expr() {
    for src in ["", "\n\n"] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].message,
            "expected an expression, found end of input"
        );
    }
    let (expr, errors) = parse_expr(&[], "");
    assert!(expr.is_none());
    assert_eq!(errors[0].span, Span::point(0));
}

#[test]
fn unclosed_paren_points_at_the_opener() {
    // The lexer rejects an unclosed `(` itself; drop the `)` to reach the parser.
    let src = "(a + b)";
    let errors = common::tokens_errors(&common::lex_prefix(src, 4), src);
    assert_eq!(errors.len(), 1);
    let e = &errors[0];
    assert_eq!(e.message, "expected `)`, found end of input");
    assert_eq!(e.span, Span::point(6));
    assert_eq!(e.secondary.len(), 1);
    assert_eq!(at(src, e.secondary[0].span), "(");
    assert_eq!(e.secondary[0].message.as_deref(), Some("unclosed `(`"));
}

#[test]
fn unclosed_call_and_index() {
    for (src, keep, want) in [
        ("f(a, b)", 5, "expected `,` or `)`, found end of input"),
        ("Matrix[3, 4]", 5, "expected `,` or `]`, found end of input"),
        (
            "f(a b)",
            usize::MAX,
            "unexpected name `b` after an expression; a unit may only follow a numeric literal",
        ),
    ] {
        let errors = common::tokens_errors(&common::lex_prefix(src, keep), src);
        assert_eq!(errors.len(), 1, "{src:?}");
        assert_eq!(errors[0].message, want);
    }
}

#[test]
fn dangling_operators() {
    for (src, want, span_text) in [
        ("a +", "expected an expression, found end of line", ""),
        ("* a", "expected an expression, found `*`", "*"),
        ("a * / b", "expected an expression, found `/`", "/"),
        ("-", "expected an expression, found end of line", ""),
        ("f(,)", "expected an expression, found `,`", ","),
        (
            "a.",
            "expected an attribute name after `.`, found end of line",
            "",
        ),
        (
            "a.3",
            "expected an attribute name after `.`, found integer literal `3`",
            "3",
        ),
        (
            "M[]",
            "expected at least one index between `[` and `]`",
            "[]",
        ),
        ("= b", "expected an expression, found `=`", "="),
        ("(,)", "expected an expression, found `,`", ","),
    ] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, want, "{src:?}");
        assert_eq!(at(src, errors[0].span), span_text, "{src:?}");
    }
}

#[test]
fn unit_after_a_non_number_is_a_clear_error() {
    let src = "x km";
    let errors = expr_errors(src);
    assert_eq!(errors.len(), 1);
    let e = &errors[0];
    assert!(
        e.message
            .contains("a unit may only follow a numeric literal"),
        "{}",
        e.message
    );
    assert_eq!(at(src, e.span), "km");
    assert_eq!(at(src, e.secondary[0].span), "x");
    assert!(e.help.as_deref().unwrap_or("").contains("`10 km`"));
}

#[test]
fn comparisons_are_non_associative() {
    let src = "a < b < c";
    let tokens = lex(src);
    let (expr, errors) = parse_expr(&tokens, src);
    // Recovered as left-assoc, but with an error.
    assert_eq!(expr.unwrap().to_sexpr(src), "(< (< a b) c)");
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("cannot be chained"));
    assert_eq!(errors[0].span, Span::new(6, 7));
    assert_eq!(errors[0].secondary[0].span, Span::new(2, 3));

    for src in ["a == b != c", "a < b + 1 <= c", "x = a >= b > c"] {
        let errors = common::module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}");
        assert!(errors[0].message.contains("cannot be chained"));
    }
}

fn repeat(prefix: TokenKind, n: usize, core: &[TokenKind]) -> Vec<Token> {
    let kinds = std::iter::repeat(prefix)
        .take(n)
        .chain(core.iter().copied());
    kinds
        .enumerate()
        .map(|(i, kind)| Token {
            kind,
            span: Span::new(i as u32, i as u32 + 1),
        })
        .collect()
}

#[test]
fn nesting_up_to_the_limit_parses() {
    let n = 200;
    let src = format!("{}x{}", "(".repeat(n), ")".repeat(n));
    let tokens = lex(&src);
    let (expr, errors) = parse_expr(&tokens, &src);
    assert!(errors.is_empty());
    assert!(expr.is_some());
    let sum = vec!["a"; 200].join(" + ");
    assert!(expr_errors(&sum).is_empty());
}

#[test]
fn deep_nesting_is_an_error_not_a_stack_overflow() {
    use TokenKind::*;
    let n = 100_000;
    let cases = [
        repeat(LParen, n, &[Ident]),
        repeat(Minus, n, &[Ident]),
        repeat(Partial, n, &[Ident]),
        // x ^ x ^ x ... (right-assoc recursion)
        (0..n)
            .map(|i| if i % 2 == 0 { Ident } else { Caret })
            .enumerate()
            .map(|(i, kind)| Token {
                kind,
                span: Span::new(i as u32, i as u32 + 1),
            })
            .collect(),
        // x + x + x ... (iterative, but the AST height is still bounded)
        (0..n)
            .map(|i| if i % 2 == 0 { Ident } else { Plus })
            .enumerate()
            .map(|(i, kind)| Token {
                kind,
                span: Span::new(i as u32, i as u32 + 1),
            })
            .collect(),
        // x⁻¹⁻¹⁻¹ ...
        repeat(Ident, 1, &vec![Inverse; n]),
        // f(f(f(...
        (0..n)
            .map(|i| if i % 2 == 0 { Ident } else { LParen })
            .enumerate()
            .map(|(i, kind)| Token {
                kind,
                span: Span::new(i as u32, i as u32 + 1),
            })
            .collect(),
    ];
    for tokens in cases {
        let src = "x".repeat(tokens.len());
        let (_, errors) = parse_expr(&tokens, &src);
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("nested too deeply")),
            "{errors:#?}"
        );
        let (_, errors) = parse_module(&tokens, &src);
        assert!(errors
            .iter()
            .any(|e| e.message.contains(&MAX_DEPTH.to_string())));
    }
}

#[test]
fn long_unit_chains_are_bounded() {
    // 1 m*m*m*... with every operator adjacent.
    let n = 10_000;
    let src = format!("1 m{}", "*m".repeat(n));
    let errors = expr_errors(&src);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("too many factors"));
}

#[test]
fn keywords_are_attribute_names_only_after_a_dot() {
    // Signed off by Eng Lead: any keyword may follow `.`.
    assert_eq!(
        common::sexpr("linalg.solve(A, b)"),
        "(call (attr linalg solve) A b)"
    );
    assert_eq!(common::sexpr("x.if"), "(attr x if)");
    assert_eq!(
        common::sexpr("a.def.for.field"),
        "(attr (attr (attr a def) for) field)"
    );
    assert_eq!(
        common::module_sexpr("y = eq.equation[0]\n"),
        "(= y (index (attr eq equation) 0))"
    );

    // A bare keyword is still not a name.
    for src in ["x = solve", "f(solve)", "x = def", "y = f(while)"] {
        let errors = common::module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert!(
            errors[0].message.contains("found keyword"),
            "{src:?}: {}",
            errors[0].message
        );
    }
    // At the start of a statement a reserved word says so instead.
    let errors = common::module_errors("solve = 1");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "`solve` is reserved for a future version"
    );
    // Only a name or keyword may follow `.`.
    let errors = expr_errors("x.(y)");
    assert_eq!(
        errors[0].message,
        "expected an attribute name after `.`, found `(`"
    );
}

fn nested(kind: &str, n: usize) -> String {
    match kind {
        "paren" => format!("{}x{}", "(".repeat(n), ")".repeat(n)),
        "call" => format!("{}x{}", "f(".repeat(n), ")".repeat(n)),
        "index" => format!("{}0{}", "a[".repeat(n), "]".repeat(n)),
        "slice" => format!("{}0{}", "a[1:".repeat(n), "]".repeat(n)),
        "power" => format!("x{}", "^x".repeat(n)),
        "prefix" => format!("{}x", "-".repeat(n)),
        "sum" => format!("x{}", " + x".repeat(n)),
        _ => unreachable!(),
    }
}

#[test]
fn each_nesting_level_counts_once() {
    assert_eq!(MAX_DEPTH, 256);
    for kind in ["paren", "call", "index", "slice", "power", "prefix", "sum"] {
        let ok = nested(kind, MAX_DEPTH as usize);
        let errors = expr_errors(&ok);
        assert!(errors.is_empty(), "{kind} x{MAX_DEPTH}: {errors:#?}");

        let too_deep = nested(kind, MAX_DEPTH as usize + 1);
        let errors = expr_errors(&too_deep);
        assert_eq!(errors.len(), 1, "{kind} x{}: {errors:#?}", MAX_DEPTH + 1);
        assert_eq!(
            errors[0].message, "expression is nested too deeply (the limit is 256 levels)",
            "{kind}"
        );
    }
}

#[test]
fn max_depth_fits_in_a_small_stack() {
    // Parse, print, and drop MAX_DEPTH levels on a 1 MiB stack, half the
    // 2 MiB default for spawned threads. Debug builds use the most stack per
    // level: about 3 KiB, so 256 levels need under 768 KiB (release needs
    // under 256 KiB).
    let handle = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            for kind in ["paren", "call", "index", "slice", "power", "prefix"] {
                for n in [MAX_DEPTH as usize, MAX_DEPTH as usize + 1, 10_000] {
                    let src = nested(kind, n);
                    let (module, _) = parse_module(&lex(&src), &src);
                    let _ = module.to_sexpr(&src);
                }
            }
        })
        .expect("spawn");
    handle.join().expect("no stack overflow");
}
