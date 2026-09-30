//! Contracts: `assert cond`, and the `requires` and `ensures` lines of a
//! function (spec "Contracts"). Trees, spans, and where each may appear.

mod common;

use common::{at, module_errors, module_sexpr};
use sky_ast::{parse_module, AssertKind, StmtKind};

fn messages(src: &str) -> Vec<String> {
    module_errors(src).into_iter().map(|e| e.message).collect()
}

#[test]
fn assert_is_a_statement_with_an_optional_message() {
    assert_eq!(
        module_sexpr("assert E > 0 J\n"),
        "(assert (> E (unit 0 J)))"
    );
    assert_eq!(
        module_sexpr("assert x >= 0, \"x is a length\"\n"),
        "(assert (>= x 0) \"x is a length\")"
    );
}

#[test]
fn the_condition_is_any_expression() {
    assert_eq!(
        module_sexpr("assert a < b and not c\n"),
        "(assert (and (< a b) (not c)))"
    );
    assert_eq!(
        module_sexpr("assert (a\n    < b)\n"),
        "(assert (paren (< a b)))"
    );
}

#[test]
fn a_function_states_its_contract_first() {
    let src = "def f(x):\n    requires x > 0\n    requires x < 9, \"too big\"\n    ensures result > x\n    return x + 1\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params x) (block (requires (> x 0)) (requires (< x 9) \"too big\") \
         (ensures (> result x)) (return (+ x 1))))"
    );
}

#[test]
fn a_function_may_have_only_a_contract() {
    assert_eq!(
        module_sexpr("def f(x):\n    requires x > 0\n"),
        "(def f (params x) (block (requires (> x 0))))"
    );
}

#[test]
fn assert_is_allowed_anywhere_a_statement_is() {
    let src = "def f(x):\n    if x > 0:\n        assert x < 9\n    assert x != 3\nwhile a:\n    assert b\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params x) (block (if (> x 0) (block (assert (< x 9)))) (assert (!= x 3))))\n\
         (while a (block (assert b)))"
    );
}

#[test]
fn spans_cover_the_keyword_through_the_message() {
    let src = "assert a < b, \"why\"\nassert c\n";
    let tokens = common::lex(src);
    let (module, errors) = parse_module(&tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    let texts: Vec<&str> = module.stmts.iter().map(|s| at(src, s.span)).collect();
    assert_eq!(texts, ["assert a < b, \"why\"", "assert c"]);
    let StmtKind::Assert {
        kind,
        cond,
        message,
    } = &module.stmts[0].kind
    else {
        panic!("not an assert");
    };
    assert_eq!(*kind, AssertKind::Assert);
    assert_eq!(at(src, cond.span), "a < b");
    assert_eq!(at(src, message.expect("a message")), "\"why\"");
}

#[test]
fn requires_and_ensures_only_start_a_function_body() {
    for word in ["requires", "ensures"] {
        // Not in a function.
        let errors = module_errors(&format!("{word} x > 0\n"));
        assert_eq!(errors.len(), 1, "{errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("`{word}` is only allowed at the start of a function body")
        );
        // Not in a block of one.
        let errors = module_errors(&format!("def f(x):\n    if x:\n        {word} x > 0\n"));
        assert_eq!(errors.len(), 1, "{errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("`{word}` is only allowed at the start of a function body")
        );
        // Not after another statement of the body.
        let errors = module_errors(&format!("def f(x):\n    y = x\n    {word} y > 0\n"));
        assert_eq!(errors.len(), 1, "{errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("`{word}` must come before the other statements of the function body")
        );
        // Nor after an `assert`, which is a statement.
        let errors = module_errors(&format!("def f(x):\n    assert x\n    {word} x > 0\n"));
        assert_eq!(errors.len(), 1, "{errors:#?}");
    }
}

#[test]
fn a_nested_function_has_its_own_head() {
    let src = "def f(x):\n    requires x > 0\n    def g(y):\n        requires y > 0\n        return y\n    return g(x)\n";
    assert!(module_errors(src).is_empty());
}

#[test]
fn errors_in_an_assert_are_precise() {
    assert_eq!(
        messages("assert x = 3\n"),
        ["`assert` takes a condition, not an assignment"]
    );
    assert_eq!(
        messages("assert x, 3\n"),
        ["expected a string message after `,`, found integer literal `3`"]
    );
    assert_eq!(
        messages("assert x, \"a\", \"b\"\n"),
        ["expected end of line, found `,`"]
    );
    assert_eq!(
        messages("assert\n"),
        ["expected an expression, found end of line"]
    );
    // A bad line does not upset the lines around it.
    let src = "assert x = 3\ny = 1\nassert y > 0\n";
    let tokens = common::lex(src);
    let (module, errors) = parse_module(&tokens, src);
    assert_eq!(errors.len(), 1);
    assert_eq!(module.stmts.len(), 2);
}

#[test]
fn the_words_are_reserved() {
    for word in ["assert", "requires", "ensures"] {
        let errors = module_errors(&format!("{word} = 1\n"));
        assert_eq!(errors.len(), 1, "{word}: {errors:#?}");
        let errors = module_errors(&format!("x = {word}\n"));
        assert_eq!(errors.len(), 1, "{word}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("expected an expression, found keyword `{word}`")
        );
    }
    // But attribute names after a dot are free, as for every keyword.
    assert_eq!(
        common::sexpr("a.assert.requires.ensures"),
        "(attr (attr (attr a assert) requires) ensures)"
    );
}

#[test]
fn result_is_an_ordinary_name_to_the_parser() {
    assert_eq!(
        module_sexpr("result = 1\nassert result > 0\n"),
        "(= result 1)\n(assert (> result 0))"
    );
}
