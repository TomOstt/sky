//! Quoting user text in a message: [`display_capped`].

use std::borrow::Cow;

use crate::safe::display_safe;

/// Most characters of user text [`display_capped`] keeps.
pub const DISPLAY_CAP_CHARS: usize = 64;

/// Most bytes of *escaped* user text [`display_capped`] keeps. An escape is
/// up to 10 bytes for one character, so this bounds the output by what is
/// printed, not by what was read.
pub const DISPLAY_CAP_BYTES: usize = 128;

/// True if `c` has Unicode General_Category `M*` (a combining mark), which
/// must stay with the character before it.
///
/// There is no property table for marks in the tree, so this is derived from
/// `unicode-ident`: `XID_Continue` is `ID_Start` plus `Mn`, `Mc`, `Nd`, `Pc`
/// and a few `Other_ID_Continue` characters, so removing `XID_Start`, the
/// numbers, the connector punctuation and those few leaves `Mn` and `Mc`.
/// `Me` (enclosing marks) is not in `XID_Continue` and is listed by hand.
fn is_mark(c: char) -> bool {
    const ENCLOSING: &[(u32, u32)] = &[
        (0x0488, 0x0489),
        (0x1ABE, 0x1ABE),
        (0x20DD, 0x20E0),
        (0x20E2, 0x20E4),
        (0xA670, 0xA672),
    ];
    const NOT_MARKS: &[char] = &[
        '_', '\u{00B7}', '\u{0387}', '\u{203F}', '\u{2040}', '\u{2054}', '\u{FE33}', '\u{FE34}',
        '\u{FE4D}', '\u{FE4E}', '\u{FE4F}', '\u{FF3F}',
    ];
    let u = c as u32;
    if ENCLOSING.iter().any(|&(lo, hi)| (lo..=hi).contains(&u)) {
        return true;
    }
    unicode_ident::is_xid_continue(c)
        && !unicode_ident::is_xid_start(c)
        && !c.is_numeric()
        && !NOT_MARKS.contains(&c)
}

/// `text` as it may appear inside a diagnostic message, label, note or help:
/// its first line, escaped exactly as the renderer escapes ([`display_safe`],
/// so U+202E becomes `\u{202E}`), cut to at most [`DISPLAY_CAP_CHARS`]
/// characters and [`DISPLAY_CAP_BYTES`] bytes of escaped text, then `…` if
/// anything (more characters, or more lines) was dropped. Borrowed when
/// nothing was escaped or dropped. The result is already display-safe, so
/// the renderer leaves it unchanged. Spans are never shortened.
///
/// * Escaping happens one character at a time, before counting: a character
///   is kept only if its whole escape fits, so a cut never lands inside an
///   escape (`\u{20`) or a UTF-8 sequence. Never escape-then-truncate.
/// * A cut never separates a base character from the combining marks after
///   it: the base goes with them. (Only a text that starts with more marks
///   than fit is cut mid-run, since there is nothing to fall back to.)
/// * The cut is marked by `…` alone, with no length. The text is a name or
///   an expression the user can find at the span, and a suffix such as
///   `(100000 bytes)` reads badly inside the backticks messages put around
///   a quote.
pub fn display_capped(text: &str) -> Cow<'_, str> {
    let line = text.split(['\n', '\r']).next().unwrap_or(text);
    let mut out = String::new();
    let mut buf = [0u8; 4];
    let mut escaped = false;
    // `out.len()` before the last character that is not a mark: where a cut
    // may fall without splitting a cluster.
    let mut boundary = 0;
    let mut cut_at = None;
    for (kept, c) in line.chars().enumerate() {
        let piece = display_safe(c.encode_utf8(&mut buf));
        if kept == DISPLAY_CAP_CHARS || out.len() + piece.len() > DISPLAY_CAP_BYTES {
            cut_at = Some(c);
            break;
        }
        if !is_mark(c) {
            boundary = out.len();
        }
        escaped |= matches!(piece, Cow::Owned(_));
        out.push_str(&piece);
    }
    let more_lines = line.len() != text.len();
    match cut_at {
        None if !escaped && !more_lines => Cow::Borrowed(text),
        None if !more_lines => Cow::Owned(out),
        _ => {
            if cut_at.is_some_and(is_mark) && boundary > 0 {
                out.truncate(boundary);
            }
            out.push('…');
            Cow::Owned(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = DISPLAY_CAP_CHARS;

    #[test]
    fn caps_characters_and_bytes() {
        assert_eq!(display_capped(&"a".repeat(N)), "a".repeat(N));
        assert_eq!(display_capped(&"a".repeat(N + 1)), "a".repeat(N) + "…");
        // 64 two-byte characters are exactly 128 bytes: both caps met, no cut.
        assert_eq!(display_capped(&"é".repeat(64)), "é".repeat(64));
        assert_eq!(display_capped(&"é".repeat(65)), "é".repeat(64) + "…");
        // Byte cap first: 32 four-byte characters fit, the 33rd does not.
        assert_eq!(display_capped(&"𝑥".repeat(32)), "𝑥".repeat(32));
        assert_eq!(display_capped(&"𝑥".repeat(100_000)), "𝑥".repeat(32) + "…");
        assert_eq!(display_capped(&"漢".repeat(100_000)), "漢".repeat(42) + "…");
    }

    #[test]
    fn escapes_and_never_cuts_an_escape() {
        assert_eq!(display_capped("a\tb"), "a\\u{0009}b");
        assert_eq!(display_capped("a\u{202E}b"), "a\\u{202E}b");
        assert_eq!(
            display_capped(&"\u{202E}".repeat(16)),
            "\\u{202E}".repeat(16)
        );
        assert_eq!(
            display_capped(&"\u{202E}".repeat(100_000)),
            "\\u{202E}".repeat(16) + "…"
        );
        for pad in 0..=8 {
            for c in ['\u{202E}', '\u{E0041}', '\t', '\u{1B}'] {
                let src = "a".repeat(pad) + &c.to_string().repeat(100);
                let q = display_capped(&src);
                let esc = display_safe(&c.to_string()).into_owned();
                let body = q.strip_suffix('…').expect("cut");
                let tail = body.trim_start_matches('a');
                assert_eq!(tail, esc.repeat(tail.len() / esc.len()), "pad {pad}");
                assert!(body.len() + esc.len() > DISPLAY_CAP_BYTES, "pad {pad}");
            }
        }
    }

    #[test]
    fn first_line_only() {
        assert_eq!(display_capped("ab\ncd"), "ab…");
        assert_eq!(display_capped("ab\r\ncd"), "ab…");
        assert_eq!(display_capped("\nab"), "…");
    }

    #[test]
    fn borrows_when_unchanged() {
        for s in ["", "x", "A⁻¹ @ ∂ħ", &"a".repeat(N)] {
            assert!(
                matches!(display_capped(s), Cow::Borrowed(b) if b == s),
                "{s:?}"
            );
        }
        for s in ["a\tb", "\u{202E}", &"a".repeat(N + 1), "a\nb"] {
            assert!(matches!(display_capped(s), Cow::Owned(_)), "{s:?}");
        }
    }

    #[test]
    fn never_separates_a_base_from_its_marks() {
        let acute = '\u{0301}';
        // 63 letters, then `e` + acute: the cluster straddles the 64 cap, so
        // it goes whole.
        let src = format!("{}e{acute}x", "a".repeat(N - 1));
        assert_eq!(display_capped(&src), "a".repeat(N - 1) + "…");
        // It fits when there is room for the mark.
        let src = format!("{}e{acute}", "a".repeat(N - 2));
        assert_eq!(display_capped(&src), src);
        // Several marks, a spacing mark (Mc) and an enclosing mark (Me).
        for mark in [acute, '\u{0903}', '\u{20DD}', '\u{0489}'] {
            assert!(is_mark(mark), "{mark:?}");
            let src = format!("{}e{mark}{mark}{mark}", "a".repeat(N - 1));
            assert_eq!(display_capped(&src), "a".repeat(N - 1) + "…", "{mark:?}");
        }
        // Not marks: letters, digits, `_`, punctuation.
        for c in ['a', 'é', '7', '٣', '_', '‿', '·', 'µ', '漢'] {
            assert!(!is_mark(c), "{c:?}");
        }
        // Nothing before the base to fall back to: a hard cut.
        let src = format!("e{}", acute.to_string().repeat(200));
        assert_eq!(
            display_capped(&src),
            format!("e{}…", acute.to_string().repeat(N - 1))
        );
    }
}
