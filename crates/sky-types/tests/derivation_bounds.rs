//! Derivation notes stay bounded however long the program or the chain of
//! bindings is: constant work per binding (no walk of the whole history), at
//! most three levels and three operands per note, and every quotation capped.
//! See the module docs of `provenance.rs` for the design.

use std::fmt::Write;
use std::time::{Duration, Instant};

use sky_diag::Diagnostic;
use sky_types::{check_module, Checked, QUOTE_MAX_CHARS};

/// Generous for a debug build; a quadratic checker takes minutes here.
const BUDGET: Duration = Duration::from_secs(20);
/// One diagnostic's checker-written text, in bytes: two derivation notes of
/// at most 1 + 3 + 9 quotes of 2 * `QUOTE_MAX_CHARS` bytes each, plus prose.
const PER_DIAGNOSTIC_MAX: usize = 2 * 13 * 2 * QUOTE_MAX_CHARS + 4096;

fn check(src: &str) -> Checked {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    check_module(&module, src)
}

fn text_len(d: &Diagnostic) -> usize {
    d.message.len()
        + std::iter::once(&d.primary)
            .chain(&d.secondary)
            .filter_map(|l| l.message.as_ref())
            .map(String::len)
            .sum::<usize>()
        + d.notes.iter().map(String::len).sum::<usize>()
        + d.help.iter().map(String::len).sum::<usize>()
}

#[test]
fn a_ten_thousand_line_chain_is_fast_and_its_note_is_short() {
    let mut src = String::from("d = 10 km\nt0 = 2 s\na0 = d / t0\n");
    for i in 1..10_000 {
        writeln!(src, "a{i} = a{} * d / t0", i - 1).unwrap();
    }
    src.push_str("bad = a9999 + t0\n");
    let start = Instant::now();
    let checked = check(&src);
    assert!(start.elapsed() < BUDGET, "took {:?}", start.elapsed());
    let errors: Vec<_> = checked.diagnostics.iter().collect();
    assert_eq!(errors.len(), 1, "{errors:#?}");
    let notes = &errors[0].notes;
    assert!(notes.len() >= 2, "{notes:#?}");
    assert!(text_len(errors[0]) <= PER_DIAGNOSTIC_MAX);
    // Three levels: the name, what it came from, what that came from. The
    // 9,997 bindings behind it are not walked or shown.
    let note = notes.iter().find(|n| n.contains("`a9999`")).unwrap();
    assert!(note.contains("`a9997`"), "{note}");
    assert!(!note.contains("`a9996`"), "{note}");
}

#[test]
fn ten_thousand_errors_produce_linear_output() {
    let mut src = String::from("d = 10 km\nt0 = 2 s\nv = d / t0\n");
    for i in 0..10_000 {
        writeln!(src, "e{i} = v + t0").unwrap();
    }
    let start = Instant::now();
    let checked = check(&src);
    assert!(start.elapsed() < BUDGET, "took {:?}", start.elapsed());
    assert_eq!(checked.diagnostics.len(), 10_000);
    let worst = checked.diagnostics.iter().map(text_len).max().unwrap();
    assert!(worst <= PER_DIAGNOSTIC_MAX, "{worst}");
}

#[test]
fn quoted_names_in_notes_are_capped() {
    let long = "x".repeat(100_000);
    let src = format!("d = 10 km\n{long} = d / 2 s\nbad = {long} + 1 s\n");
    let checked = check(&src);
    assert_eq!(checked.diagnostics.len(), 1);
    let d = &checked.diagnostics[0];
    assert!(text_len(d) <= PER_DIAGNOSTIC_MAX, "{}", text_len(d));
    assert!(d.notes.iter().any(|n| n.contains('…')), "{:?}", d.notes);
}

#[test]
fn a_deeply_nested_expression_does_not_overflow_the_walk() {
    // Just under the parser's nesting limit (256 levels).
    let depth = 200;
    let src = format!(
        "d = 10 km\nt = 2 s\nx = {}d{}\nbad = x + t\n",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    let checked = check(&src);
    assert_eq!(checked.diagnostics.len(), 1);
}
