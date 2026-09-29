//! A named source file with a line-start table for offset -> (line, col).

use std::ops::Range;

/// A source file's name and text, with a precomputed table of line starts.
///
/// Lines are separated by `\n`; a `\r` directly before a `\n` is part of the
/// line terminator (CRLF) and is never shown or counted as a column. A lone
/// `\r` is ordinary text. Columns count Unicode scalar values (`char`s), not
/// bytes or display cells.
#[derive(Clone, Debug)]
pub struct SourceFile {
    name: String,
    text: String,
    /// Byte offset of the start of each line. Always starts with 0; a text
    /// ending in `\n` has a final empty line starting at `text.len()`.
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Creates a source file. `name` is printed verbatim in diagnostic headers.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        SourceFile {
            name: name.into(),
            text,
            line_starts,
        }
    }

    /// The file name as given to [`SourceFile::new`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full source text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Number of lines, counting the empty line after a trailing `\n`.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Clamps `offset` to the text length, then moves it back to the nearest
    /// char boundary. Never panics.
    pub fn clamp_offset(&self, offset: u32) -> usize {
        let mut o = (offset as usize).min(self.text.len());
        while !self.text.is_char_boundary(o) {
            o -= 1;
        }
        o
    }

    /// 1-based (line, column) of a byte offset, after clamping. The column
    /// counts Unicode scalar values; an offset inside a CRLF terminator maps to
    /// the column just past the line's last visible character.
    pub fn line_col(&self, offset: u32) -> (usize, usize) {
        let o = self.clamp_offset(offset);
        let idx = self.line_index_of(o);
        let content = self.line_content(idx);
        let col = self.text[content.start..o.min(content.end)].chars().count() + 1;
        (idx + 1, col)
    }

    /// Text of 1-based line `line`, without its `\n` or `\r\n` terminator.
    pub fn line_text(&self, line: usize) -> Option<&str> {
        let idx = line.checked_sub(1)?;
        (idx < self.line_starts.len()).then(|| &self.text[self.line_content(idx)])
    }

    /// 0-based index of the line containing clamped byte offset `o`.
    pub(crate) fn line_index_of(&self, o: usize) -> usize {
        // line_starts[0] == 0 <= o, so partition_point is at least 1.
        self.line_starts.partition_point(|&s| s <= o) - 1
    }

    /// Byte range of 0-based line `idx`, excluding its terminator.
    pub(crate) fn line_content(&self, idx: usize) -> Range<usize> {
        let start = self.line_starts[idx];
        let mut end = self
            .line_starts
            .get(idx + 1)
            .map_or(self.text.len(), |&next| next - 1);
        if self.text[start..end].ends_with('\r') && end < self.text.len() {
            end -= 1;
        }
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(text: &str) -> SourceFile {
        SourceFile::new("t.sky", text)
    }

    #[test]
    fn line_col_start_middle_and_last_line() {
        let s = f("a = 1\nbb = 22\nccc");
        assert_eq!(s.line_col(0), (1, 1));
        assert_eq!(s.line_col(4), (1, 5));
        assert_eq!(s.line_col(5), (1, 6)); // the '\n' itself
        assert_eq!(s.line_col(6), (2, 1));
        assert_eq!(s.line_col(11), (2, 6));
        assert_eq!(s.line_col(14), (3, 1));
        assert_eq!(s.line_count(), 3);
    }

    #[test]
    fn eof_without_trailing_newline_is_past_last_char() {
        let s = f("x = 1\ny");
        assert_eq!(s.line_col(7), (2, 2));
    }

    #[test]
    fn eof_after_trailing_newline_is_start_of_empty_last_line() {
        let s = f("x = 1\n");
        assert_eq!(s.line_count(), 2);
        assert_eq!(s.line_col(6), (2, 1));
        assert_eq!(s.line_text(2), Some(""));
    }

    #[test]
    fn empty_file() {
        let s = f("");
        assert_eq!(s.line_count(), 1);
        assert_eq!(s.line_col(0), (1, 1));
        assert_eq!(s.line_col(10), (1, 1));
        assert_eq!(s.line_text(1), Some(""));
    }

    #[test]
    fn columns_count_scalar_values_not_bytes() {
        // '∂' is 3 bytes, '⁻' 3 bytes, '¹' 2 bytes: each one column.
        let s = f("df = ∂f/∂x\ny = A⁻¹ @ b\n");
        let line2 = "df = ∂f/∂x\n".len();
        let at = |needle: &str| (s.text().find(needle).unwrap()) as u32;
        assert_eq!(s.line_col(at("f/")), (1, 7));
        assert_eq!(s.line_col(at("x\n")), (1, 10));
        assert_eq!(s.line_col(at("@")), (2, 9));
        assert_eq!(at("y") as usize, line2);
    }

    #[test]
    fn crlf_terminators_are_not_columns() {
        let s = f("ab\r\ncd\r\n");
        assert_eq!(s.line_text(1), Some("ab"));
        assert_eq!(s.line_text(2), Some("cd"));
        assert_eq!(s.line_col(2), (1, 3)); // '\r'
        assert_eq!(s.line_col(3), (1, 3)); // '\n'
        assert_eq!(s.line_col(4), (2, 1));
        assert_eq!(s.line_count(), 3);
    }

    #[test]
    fn lone_cr_is_ordinary_text() {
        let s = f("a\rb\n");
        assert_eq!(s.line_text(1), Some("a\rb"));
        assert_eq!(s.line_col(2), (1, 3));
        let t = f("ab\r");
        assert_eq!(t.line_text(1), Some("ab\r"));
    }

    #[test]
    fn out_of_range_offsets_clamp_to_eof() {
        let s = f("x = 1");
        assert_eq!(s.clamp_offset(u32::MAX), 5);
        assert_eq!(s.line_col(u32::MAX), (1, 6));
    }

    #[test]
    fn non_char_boundary_offsets_floor_to_boundary() {
        let s = f("a∂b"); // '∂' at bytes 1..4
        assert_eq!(s.clamp_offset(2), 1);
        assert_eq!(s.clamp_offset(3), 1);
        assert_eq!(s.clamp_offset(4), 4);
        assert_eq!(s.line_col(3), (1, 2));
    }

    #[test]
    fn line_text_out_of_range_is_none() {
        let s = f("a\nb");
        assert_eq!(s.line_text(0), None);
        assert_eq!(s.line_text(3), None);
    }
}
