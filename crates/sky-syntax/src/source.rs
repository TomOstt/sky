use std::borrow::Cow;

/// Unicode bidi controls: LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI, LRM, RLM, ALM.
const BIDI_CONTROLS: &[char] = &[
    '\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}',
    '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}',
    '\u{200E}', '\u{200F}', '\u{061C}',
];

#[derive(Debug, PartialEq, Eq)]
pub struct BidiControl {
    pub byte_offset: usize,
    pub ch: char,
}

/// Returns the first bidi control character in `src`, if any.
pub fn find_bidi_control(src: &str) -> Option<BidiControl> {
    src.char_indices()
        .find(|(_, c)| BIDI_CONTROLS.contains(c))
        .map(|(byte_offset, ch)| BidiControl { byte_offset, ch })
}

fn unsafe_to_display(c: char, next: Option<char>) -> bool {
    match c {
        '\t' | '\n' => false,
        // A lone `\r` moves the cursor back so later text overwrites the
        // line; the `\r` of `\r\n` is part of a line terminator.
        '\r' => next != Some('\n'),
        _ => BIDI_CONTROLS.contains(&c) || c.is_control(),
    }
}

/// Returns `src` with characters that are unsafe to echo to a terminal
/// replaced, for printing source snippets in diagnostics: bidi controls
/// (which would reorder the printed line) and control characters other than
/// tab, `\n`, and the `\r` of `\r\n` (which could drive the terminal).
///
/// Each replacement is one `char` with the same UTF-8 length as the one it
/// replaces, so byte offsets and columns are unchanged: 3-byte characters
/// become U+FFFD `�`, 2-byte ones U+00A4 `¤`, 1-byte ones `?`.
pub fn display_safe(src: &str) -> Cow<'_, str> {
    let mut out: Option<String> = None;
    let mut iter = src.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let next = iter.peek().map(|&(_, n)| n);
        if unsafe_to_display(c, next) {
            out.get_or_insert_with(|| src[..i].to_string())
                .push(match c.len_utf8() {
                    1 => '?',
                    2 => '\u{00A4}',
                    _ => '\u{FFFD}',
                });
        } else if let Some(out) = &mut out {
            out.push(c);
        }
    }
    match out {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(src),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_source_is_accepted() {
        assert_eq!(find_bidi_control("velocity = 10 km / 30 min\n"), None);
    }

    #[test]
    fn unicode_operators_are_accepted() {
        assert_eq!(find_bidi_control("df = ∂f/∂x\nx = A⁻¹ @ b\n"), None);
    }

    #[test]
    fn rlo_is_rejected() {
        let src = "x = 1 \u{202E} # comment";
        assert_eq!(
            find_bidi_control(src),
            Some(BidiControl { byte_offset: 6, ch: '\u{202E}' })
        );
    }

    #[test]
    fn display_safe_replaces_bidi_and_controls_keeping_offsets() {
        let src = "a\u{202E}b\u{061C}c\u{1B}[31md\u{85}\te\rf\r\n";
        let safe = display_safe(src);
        assert_eq!(safe, "a\u{FFFD}b\u{00A4}c?[31md\u{00A4}\te?f\r\n");
        assert_eq!(safe.len(), src.len());
        assert_eq!(safe.chars().count(), src.chars().count());
        assert!(matches!(display_safe("x = ∂f\n"), Cow::Borrowed(_)));
    }
}
