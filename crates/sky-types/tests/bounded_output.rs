//! The checker's own output stays bounded however big the input is:
//! quoted source text is capped at `QUOTE_MAX_CHARS` characters, and at most
//! `MAX_STORED_REWRITES` rewrite explanations are kept.
//!
//! The rendered source snippet line and its `^^^` underline are drawn by
//! sky-diag's renderer from the span, which is never shortened; those are
//! not the checker's to cap, so these tests measure the text the checker
//! writes (message, labels, notes, help, explanation).

use std::fmt::Write;

use sky_diag::Diagnostic;
use sky_types::{
    check_module_with, Checked, Env, Shape, Tag, TagSet, MAX_STORED_REWRITES, QUOTE_MAX_CHARS,
};
use sky_units::Dimension;

const HUGE: usize = 100_000;
const KIB: usize = 1024;

fn check(src: &str, env: Env) -> Checked {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    check_module_with(&module, src, env)
}

/// Bytes of text the checker put into `d`.
/// `count` lines `u0 = K⁻¹ @ f`, `u1 = K⁻¹ @ f`, ...
fn rewrite_lines(count: usize) -> String {
    let mut src = String::new();
    for i in 0..count {
        writeln!(src, "u{i} = K⁻¹ @ f").unwrap();
    }
    src
}

fn checker_text_len(d: &Diagnostic) -> usize {
    d.message.len()
        + std::iter::once(&d.primary)
            .chain(&d.secondary)
            .filter_map(|l| l.message.as_ref())
            .map(String::len)
            .sum::<usize>()
        + d.notes.iter().map(String::len).sum::<usize>()
        + d.help.iter().map(String::len).sum::<usize>()
}

fn span_text(src: &str, d: &Diagnostic) -> String {
    src[d.primary.span.start as usize..d.primary.span.end as usize].to_string()
}

fn capped() -> String {
    format!("{}…", "x".repeat(QUOTE_MAX_CHARS))
}

#[test]
fn huge_annotation_is_quoted_capped_but_spanned_in_full() {
    let name = "x".repeat(HUGE);
    let src = format!("A: {name}\n");
    let c = check(&src, Env::new());
    assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
    let d = &c.diagnostics[0];
    assert_eq!(d.message, format!("unknown unit `{}`", capped()));
    assert!(checker_text_len(d) < KIB, "{}", checker_text_len(d));
    assert_eq!(span_text(&src, d), name);
}

#[test]
fn huge_undefined_name_is_quoted_capped() {
    let name = "x".repeat(HUGE);
    let src = format!("y = {name} + 1\n");
    let c = check(&src, Env::new());
    let d = &c.diagnostics[0];
    assert_eq!(
        d.message,
        format!("cannot find `{}` in this scope", capped())
    );
    assert!(checker_text_len(d) < KIB);
    assert_eq!(span_text(&src, d), name);
}

#[test]
fn huge_size_variable_is_capped_in_types() {
    // The size variable's name shows up in the types in the message.
    let n = "x".repeat(HUGE);
    let mut env = Env::new();
    env.bind_array(
        "v",
        Shape::vector(3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    let src = format!("A: Matrix[{n}, {n}] = v\n");
    let c = check(&src, env);
    assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
    let d = &c.diagnostics[0];
    assert!(
        d.message.starts_with("declared as a matrix"),
        "{}",
        d.message
    );
    assert!(checker_text_len(d) < KIB, "{}", checker_text_len(d));
    let ty = c.env.get("A").expect("A is bound").to_string();
    assert!(ty.len() < KIB, "{}", ty.len());
    assert!(ty.contains(&capped()), "{ty}");
}

#[test]
fn huge_operand_name_in_the_cholesky_explanation_is_capped() {
    let k = "x".repeat(HUGE);
    let mut env = Env::new();
    env.bind_array(
        k.clone(),
        Shape::matrix(3u64, 3u64),
        TagSet::of(Tag::PositiveDefinite),
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    env.bind_array(
        "f",
        Shape::vector(3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    let src = format!("u = {k}⁻¹ @ f\n");
    let c = check(&src, env);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    let site = &c.rewrites[0];
    let explanation = site.rewrite.explanation();
    // The name appears several times in the formulas; each is capped.
    assert!(explanation.len() < 2 * KIB, "{}", explanation.len());
    assert!(explanation.contains(&capped()), "{explanation}");
    let d = site.to_diagnostic(&src);
    assert!(checker_text_len(&d) < 2 * KIB, "{}", checker_text_len(&d));
    assert_eq!(
        &src[site.matrix.start as usize..site.matrix.end as usize],
        k,
        "the span still covers the whole operand"
    );
}

#[test]
fn rewrites_beyond_the_cap_are_counted_not_stored() {
    let mut env = Env::new();
    env.bind_array(
        "K",
        Shape::matrix(3u64, 3u64),
        TagSet::of(Tag::PositiveDefinite),
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    env.bind_array(
        "f",
        Shape::vector(3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    let lines = 1000;
    let src = rewrite_lines(lines);
    let c = check(&src, env);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(MAX_STORED_REWRITES, 100);
    assert_eq!(c.rewrites.len(), MAX_STORED_REWRITES);
    assert_eq!(c.rewrites_omitted, lines - MAX_STORED_REWRITES);
    // The first ones are kept, in source order.
    for (i, site) in c.rewrites.iter().enumerate() {
        let line = src.lines().nth(i).unwrap();
        let start = src.find(line).unwrap();
        assert_eq!(site.span.start as usize, start + line.find('K').unwrap());
    }
    // Checking went on past the cap: every name is bound.
    let last = format!("u{}", lines - 1);
    assert_eq!(c.env.get(&last).unwrap().to_string(), "Vector[3] [1]");
}

#[test]
fn the_cap_does_not_hide_errors_after_it() {
    let mut env = Env::new();
    env.bind_array(
        "K",
        Shape::matrix(3u64, 3u64),
        TagSet::of(Tag::PositiveDefinite),
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    env.bind_array(
        "f",
        Shape::vector(3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    let mut src = rewrite_lines(150);
    src.push_str("bad = 1 m + 1 s\n");
    let c = check(&src, env);
    assert_eq!(c.rewrites.len(), MAX_STORED_REWRITES);
    assert_eq!(c.rewrites_omitted, 50);
    assert_eq!(c.diagnostics.len(), 1);
    assert_eq!(c.diagnostics[0].message, "cannot add length and time");
}

#[test]
fn every_quoting_diagnostic_stays_small() {
    let n = "x".repeat(HUGE);
    let digits = "9".repeat(HUGE);
    let mut env = Env::new();
    // `n` is a matrix in the env for the array cases; the annotation cases
    // below use a different huge name so it is not a value in scope.
    env.bind_array(
        n.clone(),
        Shape::matrix(3u64, 3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .unwrap();
    let m = "y".repeat(HUGE);
    let big = "Y".repeat(HUGE);
    let cases = [
        // (source, expected message prefix)
        (format!("A: Matrix[{digits}, 3]\n"), "shape dimension `"),
        (
            format!("A: {big} Matrix[3, 3]\n"),
            "unknown matrix property `",
        ),
        (format!("A: Matrix[3, 3] {big}\n"), "unknown unit `"),
        (format!("A: Matrix[{n}, {n}]\n"), "size variable `"),
        (format!("q = 1 {m}\n"), "unknown unit `"),
        (format!("q = {n} + 1\n"), "cannot add a scalar to a matrix"),
        (
            format!("q = 1 / {n}\n"),
            "cannot divide a scalar by a matrix",
        ),
        (format!("q = {n}^-1\n"), "`^-1` does not invert a matrix"),
        (format!("q = ({m}) and true\n"), "cannot find `"),
    ];
    for (src, prefix) in &cases {
        let c = check(src, env.clone());
        assert!(
            c.diagnostics.iter().any(|d| d.message.starts_with(prefix)),
            "{prefix}: {:?}",
            c.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        for d in &c.diagnostics {
            assert!(
                checker_text_len(d) < KIB,
                "{prefix}: {} bytes in {:?}",
                checker_text_len(d),
                d.message
            );
            let (s, e) = (d.primary.span.start as usize, d.primary.span.end as usize);
            assert!(e - s < 16 || e - s >= HUGE, "{prefix}: span {s}..{e}");
        }
    }
}

#[test]
fn a_bool_operand_help_quotes_the_operand_capped() {
    let k = "x".repeat(HUGE);
    let src = format!("{k} = 3 m\nq = {k} and true\n");
    let c = check(&src, Env::new());
    assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
    let d = &c.diagnostics[0];
    assert_eq!(
        d.help,
        vec![format!("compare it instead, e.g. `{} != 0 m`", capped())]
    );
    assert!(checker_text_len(d) < KIB);
    assert_eq!(span_text(&src, d), k);
}
