//! The `manifest` fuzz check, shared by the fuzz target
//! (`fuzz/fuzz_targets/manifest.rs`) and a stable test in sky-project
//! (`crates/sky-project/tests/fuzz_manifest.rs`), which both include this
//! file with `#[path]`. It depends only on sky-project and sky-diag.
//!
//! The input is the manifest's bytes. Non-UTF-8 input is not parsed (the
//! reader refuses it first, `SourceError::NotUtf8`).
//!
//! Invariants, checked on every UTF-8 input:
//! - `parse_manifest` never panics, and neither does rendering its error.
//! - An error's span satisfies `start <= end <= len` and lies on char
//!   boundaries, and its diagnostic (and any secondary span) does too.
//! - The rendered message and `Display` contain no terminal-unsafe character.
//! - A manifest with a bidi control or an invisible character is never
//!   accepted, whatever else it says (Trojan Source, non-negotiable 4).
//! - Anything accepted is at most the size cap, and accepting is stable:
//!   the same text parses to the same edition.

use sky_diag::{is_display_unsafe, SourceFile};
use sky_project::{parse_manifest, Edition, MAX_MANIFEST_BYTES};

/// What a run produced, for tests that want more than "didn't crash".
/// The fuzz target ignores it, hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    NotUtf8,
    Accepted(Edition),
    /// The error's message.
    Rejected(String),
}

fn check_span(text: &str, start: u32, end: u32) {
    let (start, end) = (start as usize, end as usize);
    assert!(start <= end && end <= text.len(), "span out of bounds");
    assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
}

pub fn run(data: &[u8]) -> Outcome {
    let Ok(text) = std::str::from_utf8(data) else {
        return Outcome::NotUtf8;
    };
    match parse_manifest(text) {
        Ok(m) => {
            assert!(text.len() <= MAX_MANIFEST_BYTES);
            assert!(
                !text.chars().any(|c| is_bidi(c) || is_unsafe(c)),
                "accepted a manifest with a bidi or invisible character"
            );
            assert_eq!(parse_manifest(text).map(|m| m.edition), Ok(m.edition));
            Outcome::Accepted(m.edition)
        }
        Err(e) => {
            check_span(text, e.span.start, e.span.end);
            let d = e.to_diagnostic();
            check_span(text, d.primary.span.start, d.primary.span.end);
            for l in &d.secondary {
                check_span(text, l.span.start, l.span.end);
            }
            let file = SourceFile::new("sky.toml", text);
            let rendered = d.render_to_string(&file);
            assert!(rendered.starts_with("sky.toml:"));
            let shown = e.to_string();
            assert!(!shown.chars().any(is_display_unsafe), "{shown:?}");
            Outcome::Rejected(shown)
        }
    }
}

/// Bidi controls, listed here rather than reached through `sky_syntax` so the
/// harness's only dependencies are sky-project and sky-diag: this is an
/// independent copy of the set, which also cross-checks the parser's.
fn is_bidi(c: char) -> bool {
    matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200E}' | '\u{200F}' | '\u{061C}')
}

/// Invisible or control characters a manifest may not contain (tab and line
/// breaks are its whitespace).
fn is_unsafe(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && is_display_unsafe(c)
}
