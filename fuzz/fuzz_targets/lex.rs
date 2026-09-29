//! Fuzz target for the Sky lexer.
//!
//! Until `sky_syntax::lex` lands, this exercises `find_bidi_control`, the
//! lexer's first pass. Once `lex` exists, call it here as well and keep the
//! bidi assertions: any source the bidi check rejects must also be rejected
//! by `lex`, and `lex` must never panic on any UTF-8 input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sky_syntax::source::find_bidi_control;

fuzz_target!(|data: &[u8]| {
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    if let Some(found) = find_bidi_control(src) {
        // The reported offset must point at the reported character.
        assert!(src.is_char_boundary(found.byte_offset));
        assert_eq!(src[found.byte_offset..].chars().next(), Some(found.ch));
        // And it must be the first bidi control in the source.
        assert!(find_bidi_control(&src[..found.byte_offset]).is_none());
    }
    // TODO(lexer PR): let _ = sky_syntax::lex(src);
});
