//! Escaping characters that could change how a terminal shows diagnostics.

use std::borrow::Cow;

/// Width in columns of one escape: `\u{XXXX}` (every escaped character is at
/// most U+FFFF, so there are always exactly four hex digits).
pub(crate) const ESCAPE_WIDTH: usize = 8;

/// True if `c` is escaped by [`display_safe`].
pub(crate) fn is_unsafe(c: char) -> bool {
    matches!(c,
        // C0 controls except tab (includes ESC, `\r`, and `\n`), DEL, and C1
        // controls (includes the 8-bit CSI U+009B).
        '\u{0000}'..='\u{0008}' | '\u{000A}'..='\u{001F}' | '\u{007F}'..='\u{009F}'
        // Bidi controls: ALM, LRM, RLM, LRE..RLO, LRI..PDI.
        | '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
        // Zero-width and invisible: ZWSP, ZWNJ, ZWJ, WORD JOINER, BOM/ZWNBSP.
        | '\u{200B}'..='\u{200D}' | '\u{2060}' | '\u{FEFF}')
}

/// Returns `s` with every character that could change how a terminal shows
/// the output replaced by a visible `\u{XXXX}` escape (uppercase hex, four
/// digits):
///
/// - C0 controls except tab, so ESC (no ANSI sequences), `\r`, and `\n`;
/// - DEL and the C1 controls U+0080–U+009F (including CSI U+009B);
/// - bidi controls U+202A–U+202E, U+2066–U+2069, U+200E, U+200F, U+061C;
/// - zero-width and invisible characters U+200B–U+200D, U+2060, U+FEFF.
///
/// Everything else, including `∂`, `⁻¹`, `ħ`, and tab, is unchanged. Returns
/// `Cow::Borrowed` when nothing needs escaping.
///
/// The renderer applies this to everything it prints, so callers pass raw
/// text to [`crate::Diagnostic`] and [`crate::SourceFile`]. Escaping changes
/// lengths: never escape source text before building a `SourceFile`, or spans
/// will point at the wrong place.
///
/// ```
/// use sky_diag::display_safe;
///
/// assert_eq!(display_safe("x = 1 # \u{202E} admin"), "x = 1 # \\u{202E} admin");
/// assert_eq!(display_safe("\u{1B}[31mred"), "\\u{001B}[31mred");
/// assert_eq!(display_safe("A⁻¹ @ ∂ħ\t"), "A⁻¹ @ ∂ħ\t");
/// ```
pub fn display_safe(s: &str) -> Cow<'_, str> {
    let Some(first) = s.find(is_unsafe) else {
        return Cow::Borrowed(s);
    };
    let mut out = String::with_capacity(s.len() + ESCAPE_WIDTH);
    out.push_str(&s[..first]);
    for c in s[first..].chars() {
        if is_unsafe(c) {
            out.push_str(&format!("\\u{{{:04X}}}", c as u32));
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every escaped character, listed independently of `is_unsafe`.
    fn escaped_set() -> Vec<char> {
        let mut v: Vec<char> = ('\u{0}'..='\u{1F}').filter(|&c| c != '\t').collect();
        v.extend('\u{7F}'..='\u{9F}');
        v.extend(['\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}']);
        v.extend(['\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}']);
        v.extend(['\u{200E}', '\u{200F}', '\u{061C}']);
        v.extend(['\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}']);
        v
    }

    #[test]
    fn every_listed_character_is_escaped_to_eight_columns() {
        for c in escaped_set() {
            let s = c.to_string();
            let e = display_safe(&s);
            assert_eq!(e, format!("\\u{{{:04X}}}", c as u32));
            assert_eq!(e.chars().count(), ESCAPE_WIDTH, "{:?}", c);
        }
    }

    #[test]
    fn nothing_else_in_the_bmp_and_beyond_is_escaped() {
        let set = escaped_set();
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            assert_eq!(is_unsafe(c), set.contains(&c), "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn safe_text_is_borrowed_unchanged() {
        for s in [
            "",
            "x = 10 km",
            "df = ∂f/∂x",
            "y = A⁻¹ @ b",
            "E = ħ ω",
            "\tindented",
        ] {
            assert!(matches!(display_safe(s), Cow::Borrowed(t) if t == s), "{s}");
        }
    }

    #[test]
    fn mixed_text_escapes_only_unsafe_characters() {
        assert_eq!(
            display_safe("a\u{202E}b\u{1B}[31mc\u{200D}d\r\ne\u{85}∂"),
            "a\\u{202E}b\\u{001B}[31mc\\u{200D}d\\u{000D}\\u{000A}e\\u{0085}∂"
        );
    }
}
