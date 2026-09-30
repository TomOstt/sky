//! `accuracy x within tol`: the contextual statement, and that neither word
//! is reserved.

mod common;
use common::{module_errors, module_sexpr};

#[test]
fn the_statement_parses() {
    assert_eq!(
        module_sexpr("accuracy y within 1e-12\n"),
        "(accuracy y 1e-12)"
    );
    // A unit makes it an absolute budget: the tolerance is any expression
    // (the checker accepts literals only).
    assert_eq!(
        module_sexpr("accuracy d within 2 mm\n"),
        "(accuracy d (unit 2 mm))"
    );
    assert_eq!(
        module_sexpr("if c:\n    accuracy y within 1e-3\n"),
        "(if c (block (accuracy y 1e-3)))"
    );
}

#[test]
fn neither_word_is_reserved() {
    assert_eq!(module_sexpr("accuracy = 2\n"), "(= accuracy 2)");
    assert_eq!(
        module_sexpr("within = accuracy + 1\n"),
        "(= within (+ accuracy 1))"
    );
    assert_eq!(module_sexpr("accuracy(x)\n"), "(call accuracy x)");
    // Only two names after `accuracy` start the statement, so a name may be
    // called `within` or `accuracy` in the statement too.
    assert_eq!(
        module_sexpr("accuracy within within 1e-3\n"),
        "(accuracy within 1e-3)"
    );
}

#[test]
fn a_missing_within_is_one_precise_error() {
    let errors = module_errors("accuracy x 1e-6\n");
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .message
            .contains("expected `within` after `accuracy x`"),
        "{errors:?}"
    );
    // Recovery: the next line is still parsed.
    let errors = module_errors("accuracy x 1e-6\ny = = 2\n");
    assert_eq!(errors.len(), 2, "{errors:?}");
}
