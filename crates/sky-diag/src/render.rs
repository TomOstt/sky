//! Plain-text rendering. The format is normative in `spec/diagnostics.md`.

use std::fmt::{self, Write};

use crate::safe::{display_safe, is_unsafe, ESCAPE_WIDTH};
use crate::{Diagnostic, Label, SourceFile};

/// A span resolved against a source file, ready to draw.
struct Mark<'a> {
    /// 1-based line and column of the (clamped) span start.
    line: usize,
    col: usize,
    /// Whitespace before the marker: a tab for each tab in the source before
    /// the span start, [`ESCAPE_WIDTH`] spaces for each escaped char, and a
    /// space for every other char, so the markers line up under the
    /// displayed (escaped) line.
    indent: String,
    /// Number of marker chars: displayed width of the span on its first line
    /// (an escaped char counts [`ESCAPE_WIDTH`]), at least 1.
    width: usize,
    marker: char,
    label: Option<&'a str>,
    /// Clamped byte range, used to put secondary spans in source order.
    key: (usize, usize),
}

fn resolve<'a>(file: &SourceFile, label: &'a Label, marker: char) -> Mark<'a> {
    let text = file.text();
    let start = file.clamp_offset(label.span.start);
    let end = file.clamp_offset(label.span.end).max(start);
    let idx = file.line_index_of(start);
    let content = file.line_content(idx);
    let s = start.min(content.end);
    let e = end.min(content.end); // multi-line spans stop at the first line's end
    let before = &text[content.start..s];
    Mark {
        line: idx + 1,
        col: before.chars().count() + 1,
        indent: before.chars().fold(String::new(), |mut acc, c| {
            match c {
                '\t' => acc.push('\t'),
                c if is_unsafe(c) => acc.push_str(&" ".repeat(ESCAPE_WIDTH)),
                _ => acc.push(' '),
            }
            acc
        }),
        width: text[s..e].chars().map(display_width).sum::<usize>().max(1),
        marker,
        label: label.message.as_deref(),
        key: (start, end),
    }
}

/// Columns `c` takes in the rendered output.
fn display_width(c: char) -> usize {
    if is_unsafe(c) {
        ESCAPE_WIDTH
    } else {
        1
    }
}

impl Diagnostic {
    /// Writes this diagnostic in the plain-text format defined in
    /// `spec/diagnostics.md`. Never panics, whatever the spans are.
    pub fn render(&self, file: &SourceFile, out: &mut impl Write) -> fmt::Result {
        let primary = resolve(file, &self.primary, '^');
        let mut secondary: Vec<Mark<'_>> = self
            .secondary
            .iter()
            .map(|l| resolve(file, l, '-'))
            .collect();
        secondary.sort_by_key(|m| m.key); // stable: ties keep insertion order

        let max_line = secondary
            .iter()
            .map(|m| m.line)
            .fold(primary.line, usize::max);
        let w = max_line.to_string().len();
        let blank = " ".repeat(w);

        writeln!(
            out,
            "{}:{}:{}: {}: {}",
            display_safe(file.name()),
            primary.line,
            primary.col,
            self.severity,
            display_safe(&self.message)
        )?;
        writeln!(out, "{blank} |")?;
        let mut shown_line = None;
        for m in std::iter::once(&primary).chain(&secondary) {
            if shown_line != Some(m.line) {
                let src = display_safe(file.line_text(m.line).unwrap_or(""));
                if src.is_empty() {
                    writeln!(out, "{:>w$} |", m.line)?;
                } else {
                    writeln!(out, "{:>w$} | {src}", m.line)?;
                }
                shown_line = Some(m.line);
            }
            write!(out, "{blank} | {}", m.indent)?;
            for _ in 0..m.width {
                out.write_char(m.marker)?;
            }
            match m.label {
                Some(l) if !l.is_empty() => writeln!(out, " {}", display_safe(l))?,
                _ => writeln!(out)?,
            }
        }
        for n in &self.notes {
            writeln!(out, "{blank} = note: {}", display_safe(n))?;
        }
        for h in &self.help {
            writeln!(out, "{blank} = help: {}", display_safe(h))?;
        }
        Ok(())
    }

    /// Renders to a new `String`.
    pub fn render_to_string(&self, file: &SourceFile) -> String {
        let mut s = String::new();
        self.render(file, &mut s)
            .expect("writing to a String cannot fail");
        s
    }

    /// A [`fmt::Display`] adapter, e.g. `eprint!("{}", diag.display(&file))`
    /// to write straight to an `io::Write` such as stderr.
    pub fn display<'a>(&'a self, file: &'a SourceFile) -> impl fmt::Display + 'a {
        struct Rendered<'a>(&'a Diagnostic, &'a SourceFile);
        impl fmt::Display for Rendered<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.render(self.1, f)
            }
        }
        Rendered(self, file)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Diagnostic, SourceFile, Span};

    fn file(text: &str) -> SourceFile {
        SourceFile::new("demo.sky", text)
    }

    /// Span of the first occurrence of `needle` in `text`.
    fn find(text: &str, needle: &str) -> Span {
        let start = text.find(needle).expect("needle in text") as u32;
        Span::new(start, start + needle.len() as u32)
    }

    #[test]
    fn primary_with_label() {
        let src = "x = 5 meters + 20 kelvin\n";
        let d = Diagnostic::error("cannot add length and temperature", find(src, "20 kelvin"))
            .with_label("temperature");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:16: error: cannot add length and temperature
  |
1 | x = 5 meters + 20 kelvin
  |                ^^^^^^^^^ temperature
"
        );
    }

    #[test]
    fn primary_without_label_has_no_trailing_space() {
        let src = "y = 3\n";
        let d = Diagnostic::warning("unused binding `y`", find(src, "y"));
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:1: warning: unused binding `y`
  |
1 | y = 3
  | ^
"
        );
    }

    #[test]
    fn secondary_on_same_line_adds_underline_row_only() {
        let src = "x = 5 meters + 20 kelvin\n";
        let d = Diagnostic::error("cannot add length and temperature", find(src, "+"))
            .with_label("mismatched dimensions")
            .with_secondary(find(src, "20 kelvin"), "temperature")
            .with_secondary(find(src, "5 meters"), "length");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:14: error: cannot add length and temperature
  |
1 | x = 5 meters + 20 kelvin
  |              ^ mismatched dimensions
  |     -------- length
  |                --------- temperature
"
        );
    }

    #[test]
    fn secondaries_on_other_lines_render_in_source_order() {
        let src = "a = 1 m\nb = 2 s\nc = a + b\n";
        let d = Diagnostic::error("cannot add length and time", find(src, "a + b"))
            .with_secondary(find(src, "2 s"), "time")
            .with_secondary(find(src, "1 m"), "length");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:3:5: error: cannot add length and time
  |
3 | c = a + b
  |     ^^^^^
1 | a = 1 m
  |     --- length
2 | b = 2 s
  |     --- time
"
        );
    }

    #[test]
    fn notes_then_help() {
        let src = "v = 10 km\n";
        let d = Diagnostic::note("unit literal", find(src, "10 km"))
            .with_help("write `10 km` as `10000 m` if you prefer")
            .with_note("`km` is length with scale 1000");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:5: note: unit literal
  |
1 | v = 10 km
  |     ^^^^^
  = note: `km` is length with scale 1000
  = help: write `10 km` as `10000 m` if you prefer
"
        );
    }

    #[test]
    fn unicode_operators_before_span_count_as_one_column_each() {
        let src = "df = ∂f/∂x\ny = A⁻¹ @ b\n";
        let d = Diagnostic::error("shape mismatch", find(src, "b")).with_label("Vector[5]");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:2:11: error: shape mismatch
  |
2 | y = A⁻¹ @ b
  |           ^ Vector[5]
"
        );
        let d = Diagnostic::error("undefined `x`", find(src, "x")).with_label("here");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:10: error: undefined `x`
  |
1 | df = ∂f/∂x
  |          ^ here
"
        );
    }

    #[test]
    fn tabs_are_kept_in_the_underline_indent() {
        let src = "\tx =\t5 kelvin\n";
        let d = Diagnostic::error("bad", find(src, "kelvin")).with_label("unit");
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:1:8: error: bad\n  |\n1 | \tx =\t5 kelvin\n  | \t   \t  ^^^^^^ unit\n"
        );
    }

    #[test]
    fn crlf_is_not_shown_or_underlined() {
        let src = "a = 1\r\nb = a +\r\n";
        let d = Diagnostic::error("expected expression", Span::point(find(src, "+\r").end))
            .with_label("expected expression here");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:2:8: error: expected expression
  |
2 | b = a +
  |        ^ expected expression here
"
        );
        // A span covering just "\r\n" still gets exactly one caret.
        let d = Diagnostic::error("e", find(src, "\r\n"));
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:1:6: error: e\n  |\n1 | a = 1\n  |      ^\n"
        );
    }

    #[test]
    fn zero_width_span_gets_one_caret() {
        let src = "f(1, 2\n";
        let d = Diagnostic::error("expected `)`", Span::point(6)).with_label("here");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:7: error: expected `)`
  |
1 | f(1, 2
  |       ^ here
"
        );
    }

    #[test]
    fn eof_span_without_trailing_newline() {
        let src = "x = (1";
        let d = Diagnostic::error("unexpected end of file", Span::point(src.len() as u32));
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:1:7: error: unexpected end of file\n  |\n1 | x = (1\n  |       ^\n"
        );
    }

    #[test]
    fn eof_span_after_trailing_newline_points_at_empty_last_line() {
        let src = "x = (1\n";
        let d = Diagnostic::error("unexpected end of file", Span::point(src.len() as u32));
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:2:1: error: unexpected end of file\n  |\n2 |\n  | ^\n"
        );
    }

    #[test]
    fn empty_file() {
        let d = Diagnostic::error("empty program", Span::point(0));
        assert_eq!(
            d.render_to_string(&file("")),
            "demo.sky:1:1: error: empty program\n  |\n1 |\n  | ^\n"
        );
    }

    #[test]
    fn multi_line_span_underlines_first_line_only() {
        let src = "def f(x):\n    return x\n";
        let d = Diagnostic::error("missing return unit", find(src, "f(x):\n    return x"))
            .with_label("in this function");
        assert_eq!(
            d.render_to_string(&file(src)),
            "\
demo.sky:1:5: error: missing return unit
  |
1 | def f(x):
  |     ^^^^^ in this function
"
        );
    }

    #[test]
    fn gutter_widens_for_two_digit_line_numbers() {
        let src = "a = 1\n".repeat(9) + "bad\n";
        let d = Diagnostic::error("e", find(&src, "bad")).with_secondary(Span::new(0, 1), "first");
        assert_eq!(
            d.render_to_string(&file(&src)),
            "\
demo.sky:10:1: error: e
   |
10 | bad
   | ^^^
 1 | a = 1
   | - first
"
        );
    }

    #[test]
    fn out_of_range_spans_clamp_to_eof() {
        let src = "x = 1\n";
        let d = Diagnostic::error("e", Span::new(100, 200))
            .with_secondary(Span::new(4, u32::MAX), "runs past the end");
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:2:1: error: e\n  |\n2 |\n  | ^\n1 | x = 1\n  |     - runs past the end\n"
        );
    }

    #[test]
    fn non_char_boundary_and_inverted_spans_do_not_panic() {
        let src = "a∂b\n"; // '∂' is bytes 1..4
        let inside = Diagnostic::error("e", Span { start: 2, end: 3 });
        assert_eq!(
            inside.render_to_string(&file(src)),
            "demo.sky:1:2: error: e\n  |\n1 | a∂b\n  |  ^\n"
        );
        let inverted = Diagnostic::error("e", Span { start: 4, end: 1 });
        assert_eq!(
            inverted.render_to_string(&file(src)),
            "demo.sky:1:3: error: e\n  |\n1 | a∂b\n  |   ^\n"
        );
        // Every (start, end) pair over the text, plus out-of-range ends, renders.
        for start in 0..=8u32 {
            for end in 0..=8u32 {
                let _ = Diagnostic::error("e", Span { start, end })
                    .with_secondary(
                        Span {
                            start: end,
                            end: start,
                        },
                        "s",
                    )
                    .render_to_string(&file(src));
            }
        }
    }

    #[test]
    fn display_adapter_matches_render_to_string() {
        let src = "x = 1\n";
        let f = file(src);
        let d = Diagnostic::error("e", Span::new(0, 1)).with_label("l");
        assert_eq!(d.display(&f).to_string(), d.render_to_string(&f));
    }

    // ---- terminal safety -------------------------------------------------

    #[test]
    fn bidi_in_source_line_and_message_is_escaped_and_carets_cover_the_escape() {
        let src = "access = \"user\"\nif access != \"none\":  # \u{202E} check admin\n";
        let d = Diagnostic::error(
            "bidirectional control character \u{202E} is not allowed",
            find(src, "\u{202E}"),
        )
        .with_label("bidi control here")
        .with_secondary(find(src, "check"), "displayed out of order");
        assert_eq!(
            d.render_to_string(&file(src)),
            r#"demo.sky:2:25: error: bidirectional control character \u{202E} is not allowed
  |
2 | if access != "none":  # \u{202E} check admin
  |                         ^^^^^^^^ bidi control here
  |                                  ----- displayed out of order
"#
        );
    }

    #[test]
    fn ansi_escape_sequences_are_neutralised_everywhere() {
        let src = "label = \"\u{1B}[31mred\u{1B}[0m\"\n";
        let d = Diagnostic::warning("string contains \u{1B}[2J an escape", find(src, "red"))
            .with_label("\u{1B}[1mbold\u{1B}[0m label")
            .with_note("note with \u{1B}]0;title\u{7}")
            .with_help("help with \u{9B}31m (8-bit CSI)");
        assert_eq!(
            d.render_to_string(&file(src)),
            r#"demo.sky:1:15: warning: string contains \u{001B}[2J an escape
  |
1 | label = "\u{001B}[31mred\u{001B}[0m"
  |                      ^^^ \u{001B}[1mbold\u{001B}[0m label
  = note: note with \u{001B}]0;title\u{0007}
  = help: help with \u{009B}31m (8-bit CSI)
"#
        );
    }

    #[test]
    fn zero_width_joiner_is_visible_and_counted_in_the_underline() {
        let src = "ab\u{200D}c = 1\n";
        let d = Diagnostic::error(
            "invisible character in identifier",
            find(src, "ab\u{200D}c"),
        )
        .with_secondary(find(src, "1"), "value");
        assert_eq!(
            d.render_to_string(&file(src)),
            r#"demo.sky:1:1: error: invisible character in identifier
  |
1 | ab\u{200D}c = 1
  | ^^^^^^^^^^^
  |               - value
"#
        );
    }

    #[test]
    fn zero_width_span_at_an_escaped_char_gets_one_caret_at_the_escape() {
        let src = "x\u{FEFF}y\n";
        let d = Diagnostic::error("e", Span::point(1));
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:1:2: error: e\n  |\n1 | x\\u{FEFF}y\n  |  ^\n"
        );
    }

    #[test]
    fn control_characters_in_the_file_name_are_escaped() {
        let src = "x = 1\n";
        let f = SourceFile::new(
            "evil\u{1B}]0;pwned\u{7}\nfake.sky:9:9: error: forged.sky",
            src,
        );
        let d = Diagnostic::error("e", find(src, "1"));
        assert_eq!(
            d.render_to_string(&f),
            r#"evil\u{001B}]0;pwned\u{0007}\u{000A}fake.sky:9:9: error: forged.sky:1:5: error: e
  |
1 | x = 1
  |     ^
"#
        );
    }

    #[test]
    fn lone_carriage_return_and_newlines_in_messages_are_escaped() {
        let src = "a = 1\rb = 2\n";
        let d = Diagnostic::error("line one\nfake.sky:1:1: error: forged", find(src, "b"))
            .with_label("label\r\nsecond");
        assert_eq!(
            d.render_to_string(&file(src)),
            r#"demo.sky:1:7: error: line one\u{000A}fake.sky:1:1: error: forged
  |
1 | a = 1\u{000D}b = 2
  |              ^ label\u{000D}\u{000A}second
"#
        );
    }

    #[test]
    fn unicode_operators_and_tabs_are_not_escaped() {
        let src = "\tE = ħ ∂ A⁻¹\n";
        let d = Diagnostic::error("ok", find(src, "A⁻¹"));
        assert_eq!(
            d.render_to_string(&file(src)),
            "demo.sky:1:10: error: ok\n  |\n1 | \tE = ħ ∂ A⁻¹\n  | \t        ^^^\n"
        );
    }

    /// Deterministic xorshift64*, so failures reproduce without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn random_string(rng: &mut Rng, pool: &[char], max: usize) -> String {
        let len = rng.below(max + 1);
        (0..len).map(|_| pool[rng.below(pool.len())]).collect()
    }

    #[test]
    fn rendered_output_never_contains_an_unsafe_character() {
        // Every escaped character, plus ordinary ASCII, tab, and multi-byte text.
        let mut pool: Vec<char> = (0u32..=0x9F).filter_map(char::from_u32).collect();
        pool.extend(
            "\u{202A}\u{202B}\u{202C}\u{202D}\u{202E}\u{2066}\u{2067}\u{2068}\u{2069}".chars(),
        );
        pool.extend("\u{200E}\u{200F}\u{061C}\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}".chars());
        pool.extend("∂⁻¹ħ€😀 \t\n\n\r".chars());
        let is_escaped = |c: char| crate::safe::is_unsafe(c) && c != '\n';

        let mut rng = Rng(0x5EED_5EED_5EED_5EED);
        for case in 0..3000 {
            let src = random_string(&mut rng, &pool, 60);
            let len = src.len() as u32 + 4;
            let mut span = || {
                let a = rng.below(len as usize + 1) as u32;
                let b = rng.below(len as usize + 1) as u32;
                Span { start: a, end: b }
            };
            let (p, s1, s2) = (span(), span(), span());
            let name = random_string(&mut rng, &pool, 12);
            let mut text = |n| random_string(&mut rng, &pool, n);
            let d = Diagnostic::error(text(20), p)
                .with_label(text(10))
                .with_secondary(s1, text(10))
                .with_secondary(s2, text(10))
                .with_note(text(10))
                .with_help(text(10));
            let out = d.render_to_string(&SourceFile::new(name, src.clone()));

            if let Some(c) = out.chars().find(|&c| is_escaped(c)) {
                panic!(
                    "case {case}: U+{:04X} in output for {src:?}:\n{out}",
                    c as u32
                );
            }
            // Only the renderer's own newlines remain: after the header, every
            // line is a gutter row, a numbered source row, or a note/help row.
            for line in out.lines().skip(1) {
                let body = line.trim_start_matches(|c: char| c == ' ' || c.is_ascii_digit());
                assert!(
                    body.starts_with('|')
                        || body.starts_with("= note: ")
                        || body.starts_with("= help: "),
                    "case {case}: unexpected line {line:?} in:\n{out}"
                );
            }
        }
    }
}
