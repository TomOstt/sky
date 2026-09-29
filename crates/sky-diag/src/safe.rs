//! Escaping characters that could change how a terminal shows diagnostics.
//!
//! The escaped set is defined in `spec/diagnostics.md` ("Terminal safety")
//! and generated from the Unicode Character Database into
//! `display_unsafe_table.rs` by `tools/gen_display_unsafe.py`.

use std::borrow::Cow;
use std::fmt::Write;

use crate::display_unsafe_table::RANGES;

/// True if `c` is always escaped in rendered diagnostics:
/// `Default_Ignorable_Code_Point`, `General_Category=Cf`, U+2028, U+2029,
/// C0 controls except tab, DEL, and C1 controls (Unicode
/// [`crate::DISPLAY_UNSAFE_UNICODE_VERSION`]).
///
/// Tab is *not* in this set: the renderer keeps tabs in source lines and
/// escapes them in every other field (see [`display_safe`]). Callers such as
/// the lexer can share this set to decide which characters are invisible or
/// unsafe in source.
pub fn is_display_unsafe(c: char) -> bool {
    let c = c as u32;
    RANGES
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Writes the visible escape for `c`: `\u{` then at least four uppercase hex
/// digits, then `}` (`\u{202E}`, `\u{0009}`, `\u{E0041}`).
fn push_escape(out: &mut String, c: char) {
    write!(out, "\\u{{{:04X}}}", c as u32).expect("writing to a String cannot fail");
}

/// Columns the escape of `c` takes: 8 for U+0000–U+FFFF, 9 for U+10000–
/// U+FFFFF, 10 above. Always equal to the length of what `push_escape` writes.
pub(crate) fn escape_width(c: char) -> usize {
    let digits = match c as u32 {
        0..=0xFFFF => 4,
        0x1_0000..=0xF_FFFF => 5,
        _ => 6,
    };
    digits + 4 // `\u{` + `}`
}

/// Where a string is printed, which decides what happens to tabs.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Field {
    /// A source snippet line: tabs are kept, and the underline copies them.
    SourceLine,
    /// A file name, message, label, note, or help line: tabs are escaped.
    Text,
}

impl Field {
    pub(crate) fn escapes(self, c: char) -> bool {
        is_display_unsafe(c) || (c == '\t' && self == Field::Text)
    }
}

pub(crate) fn escape_field(s: &str, field: Field) -> Cow<'_, str> {
    let Some(first) = s.find(|c| field.escapes(c)) else {
        return Cow::Borrowed(s);
    };
    let mut out = String::with_capacity(s.len() + 8);
    out.push_str(&s[..first]);
    for c in s[first..].chars() {
        if field.escapes(c) {
            push_escape(&mut out, c);
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// Returns `s` with every [`is_display_unsafe`] character, and tab, replaced
/// by a visible `\u{XXXX}` escape (uppercase hex, at least four digits, so
/// `\u{0009}` for tab and `\u{E0041}` for TAG LATIN CAPITAL LETTER A). Returns
/// `Cow::Borrowed` when nothing needs escaping.
///
/// This is the escaping the renderer applies to file names, messages,
/// labels, notes, and help lines; use it for any other untrusted text printed
/// on one line (e.g. a path in a status line). Source snippet lines are
/// escaped the same way except that tabs are kept.
///
/// Escaping changes lengths: never escape source text before building a
/// [`crate::SourceFile`], or spans will point at the wrong place.
///
/// ```
/// use sky_diag::display_safe;
///
/// assert_eq!(display_safe("x = 1 # \u{202E} admin"), "x = 1 # \\u{202E} admin");
/// assert_eq!(display_safe("\u{1B}[31mred"), "\\u{001B}[31mred");
/// assert_eq!(display_safe("a\tb"), "a\\u{0009}b");
/// assert_eq!(display_safe("hi\u{E0041}"), "hi\\u{E0041}");
/// assert_eq!(display_safe("A⁻¹ @ ∂ħ"), "A⁻¹ @ ∂ħ");
/// ```
pub fn display_safe(s: &str) -> Cow<'_, str> {
    escape_field(s, Field::Text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_non_overlapping_and_merged() {
        assert!(!RANGES.is_empty());
        for &(lo, hi) in RANGES {
            assert!(lo <= hi, "{lo:X}..{hi:X}");
            assert!(hi <= 0x10FFFF);
        }
        for w in RANGES.windows(2) {
            let ((_, a_hi), (b_lo, _)) = (w[0], w[1]);
            // Strictly after, with a gap: adjacent ranges would have been merged.
            assert!(a_hi + 1 < b_lo, "{a_hi:X} vs {b_lo:X}");
        }
    }

    #[test]
    fn requested_characters_are_escaped() {
        let must: &[(u32, &str)] = &[
            (0x00AD, "SOFT HYPHEN"),
            (0x061C, "ARABIC LETTER MARK"),
            (0x115F, "HANGUL CHOSEONG FILLER"),
            (0x1160, "HANGUL JUNGSEONG FILLER"),
            (0x180E, "MONGOLIAN VOWEL SEPARATOR"),
            (0x200B, "ZERO WIDTH SPACE"),
            (0x200D, "ZERO WIDTH JOINER"),
            (0x200E, "LRM"),
            (0x2028, "LINE SEPARATOR"),
            (0x2029, "PARAGRAPH SEPARATOR"),
            (0x202E, "RLO"),
            (0x2060, "WORD JOINER"),
            (0x2061, "FUNCTION APPLICATION"),
            (0x2064, "INVISIBLE PLUS"),
            (0x2066, "LRI"),
            (0x2069, "PDI"),
            (0x3164, "HANGUL FILLER"),
            (0xFE00, "VS1"),
            (0xFE0F, "VS16"),
            (0xFEFF, "BOM"),
            (0xFFA0, "HALFWIDTH HANGUL FILLER"),
            (0xFFF9, "INTERLINEAR ANNOTATION ANCHOR"),
            (0xFFFB, "INTERLINEAR ANNOTATION TERMINATOR"),
            (0xE0001, "LANGUAGE TAG"),
            (0xE0041, "TAG LATIN CAPITAL LETTER A"),
            (0xE007F, "CANCEL TAG"),
            (0xE0100, "VS17"),
            (0xE01EF, "VS256"),
            (0x001B, "ESC"),
            (0x009B, "CSI"),
            (0x007F, "DEL"),
            (0x000A, "LF"),
            (0x000D, "CR"),
        ];
        for &(cp, name) in must {
            let c = char::from_u32(cp).unwrap();
            assert!(is_display_unsafe(c), "U+{cp:04X} {name}");
        }
    }

    #[test]
    fn ordinary_text_is_not_escaped() {
        for c in "\t aZ09_∂⁻¹ħ€😀μΔ漢".chars() {
            assert!(!is_display_unsafe(c), "{c:?}");
        }
        for s in ["", "x = 10 km", "df = ∂f/∂x", "y = A⁻¹ @ b", "E = ħ ω"] {
            assert!(matches!(display_safe(s), Cow::Borrowed(t) if t == s), "{s}");
        }
    }

    #[test]
    fn escape_width_is_the_length_of_the_escape() {
        for c in [
            '\u{0}',
            '\t',
            '\u{9B}',
            '\u{202E}',
            '\u{FFFB}',
            '\u{E0041}',
            '\u{E01EF}',
            '\u{10FFFF}',
        ] {
            let mut s = String::new();
            push_escape(&mut s, c);
            assert_eq!(s.chars().count(), escape_width(c), "{s}");
        }
        assert_eq!(escape_width('\u{202E}'), 8);
        assert_eq!(escape_width('\u{E0041}'), 9);
    }

    #[test]
    fn tabs_are_escaped_in_text_fields_and_kept_in_source_lines() {
        assert_eq!(display_safe("a\tb"), "a\\u{0009}b");
        assert_eq!(escape_field("a\tb", Field::SourceLine), "a\tb");
        assert_eq!(
            escape_field("a\t\u{200D}", Field::SourceLine),
            "a\t\\u{200D}"
        );
    }

    #[test]
    fn mixed_text_escapes_only_unsafe_characters() {
        assert_eq!(
            display_safe("a\u{202E}b\u{1B}[31mc\u{200D}d\r\ne\u{85}∂\u{AD}\u{FE0F}\u{E0041}"),
            "a\\u{202E}b\\u{001B}[31mc\\u{200D}d\\u{000D}\\u{000A}e\\u{0085}∂\\u{00AD}\\u{FE0F}\\u{E0041}"
        );
    }
}
