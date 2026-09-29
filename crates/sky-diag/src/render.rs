//! Plain-text rendering. The format is normative in `spec/diagnostics.md`.

use std::fmt::{self, Write};

use crate::{Diagnostic, Label, SourceFile};

/// A span resolved against a source file, ready to draw.
struct Mark<'a> {
    /// 1-based line and column of the (clamped) span start.
    line: usize,
    col: usize,
    /// Whitespace before the marker: a tab for each tab in the source before
    /// the span start, a space for every other char, so the markers line up.
    indent: String,
    /// Number of marker chars: chars of the span on its first line, at least 1.
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
        indent: before
            .chars()
            .map(|c| if c == '\t' { '\t' } else { ' ' })
            .collect(),
        width: text[s..e].chars().count().max(1),
        marker,
        label: label.message.as_deref(),
        key: (start, end),
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
            file.name(),
            primary.line,
            primary.col,
            self.severity,
            self.message
        )?;
        writeln!(out, "{blank} |")?;
        let mut shown_line = None;
        for m in std::iter::once(&primary).chain(&secondary) {
            if shown_line != Some(m.line) {
                let src = file.line_text(m.line).unwrap_or("");
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
                Some(l) if !l.is_empty() => writeln!(out, " {l}")?,
                _ => writeln!(out)?,
            }
        }
        for n in &self.notes {
            writeln!(out, "{blank} = note: {n}")?;
        }
        for h in &self.help {
            writeln!(out, "{blank} = help: {h}")?;
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
}
