//! The `sky.toml` manifest: a tiny, strict, data-only format.
//!
//! Spec v0.1 non-negotiable 1 (nothing executes) and 3 (data-only formats)
//! apply to the manifest, and reading it is the first time Sky trusts bytes
//! from a project's root. So this is not TOML: it is the smallest subset that
//! carries one setting, parsed by hand in a single forward pass with no
//! recursion, no allocation proportional to anything but the input, and no
//! panicking index or slice (every offset comes from `char_indices` or from
//! ASCII-only byte tests).
//!
//! Grammar (see spec "Editions"):
//!
//! ```text
//! manifest = { line }
//! line     = ws [ pair ] ws [ "#" comment ] ( "\n" | "\r\n" | EOF )
//! pair     = key ws "=" ws string
//! key      = [A-Za-z0-9_-]+
//! string   = '"' { any character except '"', '\', control, or bidi } '"'
//! ```
//!
//! There are no escapes, tables, arrays, numbers, or multi-line strings:
//! anything else is an error, so a future field cannot be smuggled past an
//! older Sky as something it half understands. The only known key is
//! `edition`.

use std::fmt;

use sky_diag::{display_safe, is_display_unsafe, Diagnostic, Span};
use sky_syntax::source::find_bidi_control;
use sky_syntax::Edition;

/// The largest `sky.toml` Sky reads, in bytes (64 KiB): a manifest with one
/// key needs a few dozen. Small on purpose, so a hostile root can't make
/// every `sky check` read megabytes. Enforced on the handle
/// (`discover`) and again here, so the parser is safe to call on any input.
pub const MAX_MANIFEST_BYTES: usize = 64 << 10;

// `Span` offsets are `u32`.
const _: () = assert!(MAX_MANIFEST_BYTES < u32::MAX as usize);

/// Every key a manifest may set.
const KNOWN_KEYS: &[&str] = &["edition"];

/// What a valid `sky.toml` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifest {
    /// The declared edition; [`Edition::FIRST`] if the manifest names none.
    pub edition: Edition,
}

impl Default for Manifest {
    /// What a `sky.toml` with no keys means.
    fn default() -> Self {
        Manifest {
            edition: Edition::FIRST,
        }
    }
}

/// Why a manifest was refused. Every variant carries the span it is reported at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestError {
    pub kind: ManifestErrorKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestErrorKind {
    /// More than [`MAX_MANIFEST_BYTES`] bytes. (Discovery refuses larger
    /// files before reading; this covers other callers.)
    TooLarge { len: usize },
    /// A bidi control character (Trojan Source), anywhere, comments included.
    BidiControl { ch: char },
    /// A control, format, or otherwise invisible character, or a `\r` that
    /// does not end a line. Tab is whitespace and allowed.
    ControlChar { ch: char },
    /// A line that does not start with a key (`[table]`, `= 1`, `"x" = 1`, ...).
    ExpectedKey { found: char },
    /// A key with no `=` after it.
    ExpectedEquals,
    /// A value that is not a double-quoted string (a number, a bare word,
    /// an array, a single-quoted string, nothing).
    ExpectedString,
    /// A string with no closing quote before the end of the line.
    UnterminatedString,
    /// A backslash in a string: there are no escapes.
    Escape,
    /// Something other than a comment after a value.
    TrailingText,
    /// A key that is not `edition`.
    UnknownKey {
        key: String,
        suggestion: Option<&'static str>,
    },
    /// The same key twice.
    DuplicateKey { key: String, first: Span },
    /// `edition` is not a year (`"latest"`, `"20 26"`, `""`).
    EditionNotYear { value: String },
    /// A year newer than this Sky supports.
    EditionTooNew { year: u16 },
    /// A year no edition has ever had.
    EditionUnknown { year: u16 },
}

/// Parses the text of a `sky.toml`. Never panics, on any input.
pub fn parse_manifest(text: &str) -> Result<Manifest, ManifestError> {
    let err = |kind, start: usize, end: usize| ManifestError {
        kind,
        span: Span::new(start as u32, end as u32),
    };
    if text.len() > MAX_MANIFEST_BYTES {
        // The whole file is the span's subject; clamp it inside the limit so
        // a span is always a valid `u32` range.
        return Err(err(
            ManifestErrorKind::TooLarge { len: text.len() },
            0,
            MAX_MANIFEST_BYTES,
        ));
    }
    check_characters(text)?;

    let mut manifest = Manifest::default();
    let mut seen: Vec<(&str, Span)> = Vec::new();
    let mut base = 0;
    for raw in text.split('\n') {
        let line_start = base;
        base += raw.len() + 1;
        // `check_characters` ensured a `\r` only ever precedes `\n` (or ends
        // the text), so dropping one here drops a line terminator.
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let at = |i: usize| line_start + i;

        let mut i = skip_blanks(line, 0);
        let Some(first) = line[i..].chars().next() else {
            continue;
        };
        if first == '#' {
            continue;
        }

        // key
        let key_len = line[i..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
            .count();
        if key_len == 0 {
            return Err(err(
                ManifestErrorKind::ExpectedKey { found: first },
                at(i),
                at(i + first.len_utf8()),
            ));
        }
        let key = &line[i..i + key_len];
        let key_span = Span::new(at(i) as u32, at(i + key_len) as u32);
        // Checked before the value so `editon = 1` gets the key's did-you-mean,
        // not a complaint about the number.
        if !KNOWN_KEYS.contains(&key) {
            return Err(ManifestError {
                kind: ManifestErrorKind::UnknownKey {
                    key: key.to_string(),
                    suggestion: suggest(key),
                },
                span: key_span,
            });
        }
        if let Some(&(_, first_span)) = seen.iter().find(|(k, _)| *k == key) {
            return Err(ManifestError {
                kind: ManifestErrorKind::DuplicateKey {
                    key: key.to_string(),
                    first: first_span,
                },
                span: key_span,
            });
        }
        seen.push((key, key_span));
        i += key_len;

        // `=`
        i = skip_blanks(line, i);
        if line.as_bytes().get(i) != Some(&b'=') {
            let (s, e) = char_span(line, i);
            return Err(err(ManifestErrorKind::ExpectedEquals, at(s), at(e)));
        }
        i = skip_blanks(line, i + 1);

        // string
        if line.as_bytes().get(i) != Some(&b'"') {
            let (s, e) = char_span(line, i);
            return Err(err(ManifestErrorKind::ExpectedString, at(s), at(e)));
        }
        let open = i;
        let body = i + 1;
        let mut close = None;
        for (off, c) in line[body..].char_indices() {
            match c {
                '"' => {
                    close = Some(body + off);
                    break;
                }
                '\\' => {
                    return Err(err(
                        ManifestErrorKind::Escape,
                        at(body + off),
                        at(body + off + 1),
                    ))
                }
                _ => {}
            }
        }
        let Some(close) = close else {
            return Err(err(
                ManifestErrorKind::UnterminatedString,
                at(open),
                at(line.len()),
            ));
        };
        let value = &line[body..close];
        let value_span = Span::new(at(open) as u32, at(close + 1) as u32);

        // trailing: blanks, then a comment or the end
        let rest = skip_blanks(line, close + 1);
        if let Some(c) = line[rest..].chars().next() {
            if c != '#' {
                return Err(err(
                    ManifestErrorKind::TrailingText,
                    at(rest),
                    at(rest + c.len_utf8()),
                ));
            }
        }

        match key {
            "edition" => manifest.edition = parse_edition(value, value_span)?,
            _ => unreachable!("KNOWN_KEYS and this match are kept in step"),
        }
    }
    Ok(manifest)
}

fn parse_edition(value: &str, span: Span) -> Result<Edition, ManifestError> {
    let not_year = || ManifestError {
        kind: ManifestErrorKind::EditionNotYear {
            value: value.to_string(),
        },
        span,
    };
    // ASCII digits only: `u16::from_str` would also accept a leading `+`.
    if value.is_empty() || value.len() > 5 || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(not_year());
    }
    let Ok(year) = value.parse::<u16>() else {
        // Five digits but over 65535: a year, just an absurd one.
        return Err(ManifestError {
            kind: ManifestErrorKind::EditionTooNew { year: u16::MAX },
            span,
        });
    };
    match Edition::from_year(year) {
        Some(e) => Ok(e),
        None if year > Edition::LATEST.year() => Err(ManifestError {
            kind: ManifestErrorKind::EditionTooNew { year },
            span,
        }),
        None => Err(ManifestError {
            kind: ManifestErrorKind::EditionUnknown { year },
            span,
        }),
    }
}

/// Rejects, before any parsing, the characters no manifest may contain: bidi
/// controls, invisible or control characters, and a `\r` that is not part of
/// a line ending. Whole file, comments included: a comment is exactly where
/// a Trojan Source attack would hide.
fn check_characters(text: &str) -> Result<(), ManifestError> {
    let bidi = find_bidi_control(text);
    let bytes = text.as_bytes();
    for (i, c) in text.char_indices() {
        if let Some(b) = &bidi {
            if b.byte_offset == i {
                return Err(ManifestError {
                    kind: ManifestErrorKind::BidiControl { ch: b.ch },
                    span: Span::new(i as u32, (i + c.len_utf8()) as u32),
                });
            }
        }
        let bad = match c {
            '\t' | '\n' => false,
            '\r' => bytes.get(i + 1).is_some_and(|&n| n != b'\n'),
            _ => is_display_unsafe(c),
        };
        if bad {
            return Err(ManifestError {
                kind: ManifestErrorKind::ControlChar { ch: c },
                span: Span::new(i as u32, (i + c.len_utf8()) as u32),
            });
        }
    }
    Ok(())
}

/// Index of the first byte at or after `i` that is not a space or tab.
fn skip_blanks(line: &str, mut i: usize) -> usize {
    let b = line.as_bytes();
    while matches!(b.get(i), Some(b' ' | b'\t')) {
        i += 1;
    }
    i
}

/// The span of the character at `i`, or the empty span at the line's end.
fn char_span(line: &str, i: usize) -> (usize, usize) {
    match line.get(i..).and_then(|s| s.chars().next()) {
        Some(c) => (i, i + c.len_utf8()),
        None => (line.len(), line.len()),
    }
}

/// The known key closest to `key`, if it is close enough to be a typo.
fn suggest(key: &str) -> Option<&'static str> {
    let key: Vec<char> = key.chars().flat_map(char::to_lowercase).collect();
    KNOWN_KEYS
        .iter()
        .map(|k| {
            let want: Vec<char> = k.chars().collect();
            (edit_distance(&key, &want), *k)
        })
        .filter(|&(d, _)| d <= 2 && d <= (key.len() / 3).max(1))
        .min_by_key(|&(d, _)| d)
        .map(|(_, k)| k)
}

/// Levenshtein distance between two short strings.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let up = row[j + 1];
            row[j + 1] = (diag + usize::from(ca != cb)).min(up + 1).min(row[j] + 1);
            diag = up;
        }
    }
    row[b.len()]
}

impl ManifestError {
    /// The message, unescaped: the renderer escapes it, and so does
    /// [`fmt::Display`].
    fn message(&self) -> String {
        match &self.kind {
            ManifestErrorKind::TooLarge { len } => format!(
                "sky.toml is {len} bytes, over the limit of {MAX_MANIFEST_BYTES} bytes (64 KiB)"
            ),
            ManifestErrorKind::BidiControl { ch } => {
                format!("bidi control character U+{:04X} in sky.toml", *ch as u32)
            }
            ManifestErrorKind::ControlChar { ch } => {
                format!("control character U+{:04X} in sky.toml", *ch as u32)
            }
            ManifestErrorKind::ExpectedKey { .. } => "expected a key".into(),
            ManifestErrorKind::ExpectedEquals => "expected `=` after the key".into(),
            ManifestErrorKind::ExpectedString => "expected a double-quoted string".into(),
            ManifestErrorKind::UnterminatedString => "unterminated string".into(),
            ManifestErrorKind::Escape => "escapes are not supported in sky.toml strings".into(),
            ManifestErrorKind::TrailingText => "unexpected text after the value".into(),
            ManifestErrorKind::UnknownKey { key, .. } => format!("unknown key `{key}`"),
            ManifestErrorKind::DuplicateKey { key, .. } => format!("duplicate key `{key}`"),
            ManifestErrorKind::EditionNotYear { value } => {
                format!("edition `{value}` is not a year")
            }
            ManifestErrorKind::EditionTooNew { year } => format!(
                "edition {year} is newer than this Sky (supports {})",
                Edition::LATEST
            ),
            ManifestErrorKind::EditionUnknown { year } => {
                format!("edition {year} is not a Sky edition")
            }
        }
    }

    /// This error as an error diagnostic against the manifest's text (see
    /// `DiscoverError::source_file`).
    pub fn to_diagnostic(&self) -> Diagnostic {
        let d = Diagnostic::error(self.message(), self.span);
        let latest = Edition::LATEST;
        match &self.kind {
            ManifestErrorKind::TooLarge { .. } => d
                .with_note("Sky reads at most 64 KiB from a sky.toml")
                .with_help("sky.toml only needs `edition = \"2026\"`"),
            ManifestErrorKind::BidiControl { .. } => d
                .with_label("invisible text-direction control")
                .with_note("bidi controls can make text read differently from what it is (Trojan Source)")
                .with_help("delete the character"),
            ManifestErrorKind::ControlChar { .. } => d
                .with_label("not allowed here")
                .with_note("sky.toml may contain tabs and line breaks, but no other control or invisible characters")
                .with_help("delete the character"),
            ManifestErrorKind::ExpectedKey { found } => {
                let d = d.with_label("a line is blank, a `#` comment, or `key = \"value\"`");
                if *found == '[' {
                    d.with_note("tables are not supported: sky.toml is a flat list of `key = \"value\"` lines")
                        .with_help("the only key is `edition = \"2026\"`")
                } else {
                    d.with_help("write `edition = \"2026\"`")
                }
            }
            ManifestErrorKind::ExpectedEquals => d
                .with_label("expected `=` here")
                .with_help("write `key = \"value\"`"),
            ManifestErrorKind::ExpectedString => d
                .with_label("values are double-quoted strings")
                .with_note("numbers, booleans, arrays and single-quoted strings are not supported")
                .with_help(format!("write `edition = \"{latest}\"`")),
            ManifestErrorKind::UnterminatedString => d
                .with_label("this string is not closed")
                .with_note("strings end on the line they start")
                .with_help("add a closing `\"`"),
            ManifestErrorKind::Escape => d
                .with_label("backslash")
                .with_note("strings are taken literally, and may not contain `\\` or `\"`")
                .with_help("remove the backslash"),
            ManifestErrorKind::TrailingText => d
                .with_label("nothing but a `#` comment may follow a value")
                .with_help("put each `key = \"value\"` on its own line"),
            ManifestErrorKind::UnknownKey { suggestion, .. } => {
                let d = d
                    .with_label("not a sky.toml key")
                    .with_note("the only key is `edition`; sky.toml cannot add import paths or change how imports resolve");
                match suggestion {
                    Some(s) => d.with_help(format!("did you mean `{s}`?")),
                    None => d.with_help("remove this line"),
                }
            }
            ManifestErrorKind::DuplicateKey { first, .. } => d
                .with_label("set again here")
                .with_secondary(*first, "first set here")
                .with_help("remove one of them"),
            ManifestErrorKind::EditionNotYear { .. } => d
                .with_label("expected a year such as \"2026\"")
                .with_help(format!("write `edition = \"{latest}\"`")),
            ManifestErrorKind::EditionTooNew { .. } => d
                .with_label("newer than this Sky")
                .with_note("a project may not use an edition its Sky does not know, because it could mean something else")
                .with_help(format!("upgrade Sky, or set `edition = \"{latest}\"` if the project supports it")),
            ManifestErrorKind::EditionUnknown { .. } => d
                .with_label("no such edition")
                .with_note(format!("the first edition is {}", Edition::FIRST))
                .with_help(format!("write `edition = \"{latest}\"`")),
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&display_safe(&self.message()))
    }
}

impl std::error::Error for ManifestError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_diag::SourceFile;

    fn ok(text: &str) -> Manifest {
        parse_manifest(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
    }

    fn bad(text: &str) -> ManifestError {
        parse_manifest(text).expect_err(text)
    }

    /// The text the error's span covers.
    fn covered<'a>(text: &'a str, e: &ManifestError) -> &'a str {
        &text[e.span.range()]
    }

    #[test]
    fn empty_and_blank_manifests_mean_the_first_edition() {
        for t in [
            "",
            "\n",
            "  \t\n\n",
            "# only a comment\n",
            "   # indented\r\n",
        ] {
            assert_eq!(ok(t).edition, Edition::FIRST, "{t:?}");
        }
    }

    #[test]
    fn edition_is_read() {
        for t in [
            "edition = \"2026\"",
            "edition=\"2026\"\n",
            "  edition\t=\t\"2026\"  # the first\n",
            "# c\n\nedition = \"2026\"\r\n# d\r\n",
        ] {
            assert_eq!(ok(t).edition, Edition::E2026, "{t:?}");
        }
    }

    #[test]
    fn unknown_key_suggests_the_known_one() {
        let t = "editon = \"2026\"\n";
        let e = bad(t);
        assert_eq!(
            e.kind,
            ManifestErrorKind::UnknownKey {
                key: "editon".into(),
                suggestion: Some("edition")
            }
        );
        assert_eq!(covered(t, &e), "editon");
        let d = e.to_diagnostic();
        assert_eq!(d.message, "unknown key `editon`");
        assert_eq!(d.help, ["did you mean `edition`?"]);
        // Case counts as a typo too.
        assert!(matches!(
            bad("Edition = \"2026\"").kind,
            ManifestErrorKind::UnknownKey {
                suggestion: Some("edition"),
                ..
            }
        ));
    }

    #[test]
    fn unrelated_key_has_no_suggestion() {
        for key in ["import_paths", "name", "x"] {
            let t = format!("{key} = \"a\"");
            let e = bad(&t);
            assert!(
                matches!(
                    e.kind,
                    ManifestErrorKind::UnknownKey {
                        suggestion: None,
                        ..
                    }
                ),
                "{key}"
            );
            assert_eq!(e.to_diagnostic().help, ["remove this line"]);
        }
    }

    #[test]
    fn duplicate_key_points_at_both() {
        let t = "edition = \"2026\"\n# x\nedition = \"2026\"\n";
        let e = bad(t);
        let ManifestErrorKind::DuplicateKey { key, first } = &e.kind else {
            panic!("{e:?}")
        };
        assert_eq!(key, "edition");
        assert_eq!(&t[first.range()], "edition");
        assert_eq!(first.start, 0);
        assert_eq!(e.span.start as usize, t.rfind("edition").unwrap());
    }

    #[test]
    fn edition_too_new_says_what_is_supported() {
        let t = "edition = \"2031\"";
        let e = bad(t);
        assert_eq!(e.kind, ManifestErrorKind::EditionTooNew { year: 2031 });
        assert_eq!(covered(t, &e), "\"2031\"");
        assert_eq!(
            e.to_string(),
            "edition 2031 is newer than this Sky (supports 2026)"
        );
        // Five digits past u16 is still "too new", not a panic or "not a year".
        assert!(matches!(
            bad("edition = \"99999\"").kind,
            ManifestErrorKind::EditionTooNew { .. }
        ));
    }

    #[test]
    fn edition_older_than_the_first_is_unknown() {
        assert_eq!(
            bad("edition = \"2015\"").kind,
            ManifestErrorKind::EditionUnknown { year: 2015 }
        );
        assert_eq!(
            bad("edition = \"0\"").kind,
            ManifestErrorKind::EditionUnknown { year: 0 }
        );
    }

    #[test]
    fn edition_that_is_not_a_year() {
        for v in [
            "", "latest", "20 26", "+2026", "2026a", "-1", "٢٠٢٦", "123456",
        ] {
            let t = format!("edition = \"{v}\"");
            assert!(
                matches!(bad(&t).kind, ManifestErrorKind::EditionNotYear { .. }),
                "{v:?}"
            );
        }
    }

    #[test]
    fn value_must_be_a_double_quoted_string() {
        for (t, want) in [
            ("edition = 2026", "2"),
            ("edition = '2026'", "'"),
            ("edition = [\"2026\"]", "["),
            ("edition = true", "t"),
            ("edition =", ""),
            ("edition = # c", "#"),
        ] {
            let e = bad(t);
            assert_eq!(e.kind, ManifestErrorKind::ExpectedString, "{t}");
            assert_eq!(covered(t, &e), want, "{t}");
        }
    }

    #[test]
    fn key_needs_an_equals_sign() {
        for t in ["edition", "edition \"2026\"", "edition : \"2026\""] {
            assert_eq!(bad(t).kind, ManifestErrorKind::ExpectedEquals, "{t}");
        }
    }

    #[test]
    fn line_must_start_with_a_key() {
        for (t, ch) in [
            ("[project]\n", '['),
            ("= \"x\"", '='),
            ("\"edition\" = \"2026\"", '"'),
            ("é = \"x\"", 'é'),
            ("edition.x = \"1\"", '.'),
        ] {
            let e = bad(t);
            // `edition.x` reads the key `edition`, then fails at the dot.
            let expect = if t.starts_with("edition") {
                ManifestErrorKind::ExpectedEquals
            } else {
                ManifestErrorKind::ExpectedKey { found: ch }
            };
            assert_eq!(e.kind, expect, "{t}");
        }
        let e = bad("é = \"x\"");
        assert_eq!(covered("é = \"x\"", &e), "é");
    }

    #[test]
    fn string_errors() {
        let t = "edition = \"2026";
        let e = bad(t);
        assert_eq!(e.kind, ManifestErrorKind::UnterminatedString);
        assert_eq!(covered(t, &e), "\"2026");
        // A string never continues onto the next line.
        assert_eq!(
            bad("edition = \"2026\n\"").kind,
            ManifestErrorKind::UnterminatedString
        );
        let t = "edition = \"20\\26\"";
        let e = bad(t);
        assert_eq!(e.kind, ManifestErrorKind::Escape);
        assert_eq!(covered(t, &e), "\\");
    }

    #[test]
    fn nothing_but_a_comment_may_follow_a_value() {
        let t = "edition = \"2026\" 2";
        let e = bad(t);
        assert_eq!(e.kind, ManifestErrorKind::TrailingText);
        assert_eq!(covered(t, &e), "2");
        assert_eq!(
            bad("edition = \"2026\" \"2026\"").kind,
            ManifestErrorKind::TrailingText
        );
    }

    #[test]
    fn bidi_control_is_refused_everywhere() {
        for t in [
            "# \u{202E}\nedition = \"2026\"\n",
            "edition = \"20\u{202E}26\"\n",
            "\u{2066}edition = \"2026\"\n",
        ] {
            let e = bad(t);
            assert!(
                matches!(e.kind, ManifestErrorKind::BidiControl { .. }),
                "{t:?}"
            );
            assert_eq!(covered(t, &e).chars().count(), 1);
        }
    }

    #[test]
    fn control_and_invisible_characters_are_refused() {
        for c in [
            '\0', '\u{1b}', '\u{7f}', '\u{85}', '\u{200b}', '\u{feff}', '\u{2028}',
        ] {
            for t in [
                format!("# a{c}b\n"),
                format!("edition = \"2026{c}\"\n"),
                format!("{c}edition = \"2026\"\n"),
            ] {
                let e = bad(&t);
                assert_eq!(e.kind, ManifestErrorKind::ControlChar { ch: c }, "{t:?}");
                assert_eq!(covered(&t, &e), c.to_string());
            }
        }
    }

    #[test]
    fn a_lone_carriage_return_is_refused() {
        for t in ["a\rb", "edition = \"2026\"\r\r\n", "\r\r"] {
            assert!(
                matches!(bad(t).kind, ManifestErrorKind::ControlChar { ch: '\r' }),
                "{t:?}"
            );
        }
        // At the very end it terminates the last line.
        assert_eq!(ok("edition = \"2026\"\r").edition, Edition::E2026);
    }

    #[test]
    fn size_cap_is_enforced_by_the_parser_too() {
        let text = format!("# {}\n", "a".repeat(MAX_MANIFEST_BYTES));
        let e = bad(&text);
        assert!(matches!(e.kind, ManifestErrorKind::TooLarge { .. }));
        assert!(e.span.end as usize <= MAX_MANIFEST_BYTES);
        // Exactly the limit is fine.
        let text = format!("#{}", "a".repeat(MAX_MANIFEST_BYTES - 1));
        assert_eq!(text.len(), MAX_MANIFEST_BYTES);
        ok(&text);
    }

    #[test]
    fn errors_render_with_line_and_column() {
        let t = "# project\n\nedition = \"2031\"\n";
        let file = SourceFile::new("proj/sky.toml", t);
        let shown = bad(t).to_diagnostic().render_to_string(&file);
        assert!(
            shown.starts_with(
                "proj/sky.toml:3:11: error: edition 2031 is newer than this Sky (supports 2026)\n"
            ),
            "{shown}"
        );
        assert!(shown.contains("3 | edition = \"2031\""), "{shown}");
        assert!(shown.contains("= help:"), "{shown}");
    }

    #[test]
    fn every_diagnostic_span_is_in_bounds_and_on_char_boundaries() {
        for t in [
            "é",
            "edition = \"é\\\"",
            "edition = é",
            "edition é",
            "editión = \"1\"",
            "edition = \"2026\" é",
            "\u{202E}",
            "a\u{200b}",
            "[é]",
        ] {
            let e = bad(t);
            let (s, en) = (e.span.start as usize, e.span.end as usize);
            assert!(s <= en && en <= t.len(), "{t:?}");
            assert!(t.is_char_boundary(s) && t.is_char_boundary(en), "{t:?}");
            let _ = e.to_diagnostic().render_to_string(&SourceFile::new("m", t));
        }
    }
}
