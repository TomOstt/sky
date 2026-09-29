//! ParseError converts into a sky_diag::Diagnostic and renders in the
//! normative format (spec/diagnostics.md).

mod common;
use common::lex;
use sky_ast::parse_module;
use sky_diag::{Diagnostic, Severity, SourceFile};

fn first_diagnostic(src: &str) -> Diagnostic {
    let (_, errors) = parse_module(&lex(src), src);
    errors.into_iter().next().expect("an error").into()
}

#[test]
fn conversion_keeps_everything() {
    let src = "x = (a + b\n";
    let (_, errors) = parse_module(&lex(src), src);
    let err = errors[0].clone();
    let diag = err.clone().into_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.message, err.message);
    assert_eq!(diag.primary.span, err.span);
    assert_eq!(diag.primary.message, None);
    assert_eq!(diag.secondary, err.secondary);
    assert!(diag.notes.is_empty() && diag.help.is_empty());
    assert_eq!(Diagnostic::from(err), diag);
}

#[test]
fn renders_unclosed_paren() {
    let src = "x = (a + b\n";
    let file = SourceFile::new("demo.sky", src);
    assert_eq!(
        first_diagnostic(src).render_to_string(&file),
        "demo.sky:1:11: error: expected `)`, found end of line\n  \
         |\n\
         1 | x = (a + b\n  \
         |           ^\n  \
         |     - unclosed `(`\n"
    );
}

#[test]
fn renders_juxtaposed_units_with_help() {
    let src = "f = 5 kg m\n";
    let file = SourceFile::new("demo.sky", src);
    assert_eq!(
        first_diagnostic(src).render_to_string(&file),
        "demo.sky:1:10: error: unexpected name `m` after the unit `kg`\n  \
         |\n\
         1 | f = 5 kg m\n  \
         |          ^\n  \
         |       -- unit\n  \
         = help: compound units are written with `*` and no spaces: `kg*m`\n"
    );
}
