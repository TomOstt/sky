//! Quoting user source text in diagnostics.
//!
//! Every piece of user source text the checker copies into a diagnostic
//! (messages, labels, notes, help, the `--explain` explanation, and shape
//! variable names shown in types) goes through [`quote`], so a huge
//! annotation or operand cannot blow up the output. Spans are never
//! shortened: they still cover the whole source text.

use std::borrow::Cow;

use sky_diag::display_safe;

/// Most characters of user source text quoted in one place.
pub const QUOTE_MAX_CHARS: usize = 64;

/// `text` as it may appear inside a diagnostic: escaped with
/// [`display_safe`], and cut after [`QUOTE_MAX_CHARS`] characters. A cut
/// text ends in `…` followed by the full length in bytes:
/// `aaaa…aaa… (100000 bytes)`.
///
/// Escaping here as well as in the renderer is harmless: `display_safe`
/// output contains no characters it would escape again.
// TODO(sky-diag): Gilfoyle plans to cap in `display_safe` itself; move this
// cap there and make this a plain call once that lands.
// TODO: cut at a grapheme-cluster boundary instead of a char boundary, so a
// base character never loses its combining marks. That needs segmentation
// data, which is not worth a new dependency yet.
pub(crate) fn quote(text: &str) -> Cow<'_, str> {
    match text.char_indices().nth(QUOTE_MAX_CHARS) {
        None => display_safe(text),
        Some((cut, _)) => Cow::Owned(format!(
            "{}… ({} bytes)",
            display_safe(&text[..cut]),
            text.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_unchanged() {
        assert_eq!(quote("K⁻¹ @ f"), "K⁻¹ @ f");
        let exactly = "x".repeat(QUOTE_MAX_CHARS);
        assert_eq!(quote(&exactly), exactly);
    }

    #[test]
    fn long_text_is_cut_with_its_byte_length() {
        let long = "x".repeat(QUOTE_MAX_CHARS + 1);
        let want = format!("{}… ({} bytes)", "x".repeat(QUOTE_MAX_CHARS), long.len());
        assert_eq!(quote(&long), want);
    }

    #[test]
    fn cuts_at_a_char_boundary() {
        // 3-byte chars: a byte cut at 64 would split one.
        let long = "∂".repeat(100);
        let q = quote(&long);
        assert_eq!(q, format!("{}… (300 bytes)", "∂".repeat(QUOTE_MAX_CHARS)));
    }

    #[test]
    fn escapes_what_the_renderer_escapes() {
        assert_eq!(quote("a\u{202E}b"), "a\\u{202E}b");
        let long = format!("{}\u{202E}", "a".repeat(QUOTE_MAX_CHARS + 5));
        assert!(!quote(&long).contains('\u{202E}'));
    }
}
