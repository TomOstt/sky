//! Fuzz target for the Sky lexer.
//!
//! Invariants checked on every UTF-8 input:
//! - `lex` never panics, and neither does rendering its diagnostics.
//! - Trojan Source (spec v0.1 non-negotiable 4): whenever `find_bidi_control`
//!   finds a bidi control, `lex` rejects the source with `BidiControl`, and
//!   the error's span covers exactly that character.
//! - On success, token spans are in bounds, on char boundaries, in order,
//!   and non-overlapping, and the last token is `Eof`.
//! - Every diagnostic's primary span is in bounds and on char boundaries.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sky_syntax::source::find_bidi_control;
use sky_syntax::{lex, LexErrorKind, Span, TokenKind};

fn check_span(src: &str, span: Span) {
    let (start, end) = (span.start as usize, span.end as usize);
    assert!(start <= end && end <= src.len(), "span {span:?} out of bounds");
    assert!(src.is_char_boundary(start) && src.is_char_boundary(end));
}

fuzz_target!(|data: &[u8]| {
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    let bidi = find_bidi_control(src);
    if let Some(found) = &bidi {
        // The reported offset must point at the reported character,
        // and it must be the first bidi control in the source.
        assert!(src.is_char_boundary(found.byte_offset));
        assert_eq!(src[found.byte_offset..].chars().next(), Some(found.ch));
        assert!(find_bidi_control(&src[..found.byte_offset]).is_none());
    }

    let file = sky_diag::SourceFile::new("fuzz.sky", src);
    match lex(src) {
        Ok(lexed) => {
            assert!(bidi.is_none(), "lex accepted source with a bidi control");
            assert_eq!(lexed.tokens.last().map(|t| &t.kind), Some(&TokenKind::Eof));
            let mut prev_end = 0;
            for t in &lexed.tokens {
                check_span(src, t.span);
                assert!(prev_end <= t.span.start, "tokens overlap or are out of order");
                prev_end = t.span.end;
            }
            for d in &lexed.warnings {
                check_span(src, d.primary.span);
                let _ = d.display(&file).to_string();
            }
        }
        Err(err) => {
            if let Some(found) = &bidi {
                assert_eq!(err.kind, LexErrorKind::BidiControl);
                let span = err.diagnostic.primary.span;
                assert_eq!(span.start as usize, found.byte_offset);
                assert_eq!(span.end as usize, found.byte_offset + found.ch.len_utf8());
            }
            for d in err.diagnostics() {
                check_span(src, d.primary.span);
                let _ = d.display(&file).to_string();
            }
        }
    }
});
