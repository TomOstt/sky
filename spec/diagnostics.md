# Sky diagnostic format

Status: normative for v0.1. Implemented by `crates/sky-diag`. Every tool that reports a diagnostic about Sky source (`sky check`, `sky run`, `sky repl`) prints it in this format, and the golden files `tests/ui/<name>.stderr` contain exactly this output, byte for byte. Changing anything here means updating the renderer and every golden file together.

## Layout

```text
demo.sky:1:14: error: cannot add length and temperature
  |
1 | x = 5 meters + 20 kelvin
  |              ^ mismatched dimensions
  |     -------- length
  |                --------- temperature
  = note: both operands of `+` must have the same dimension
  = help: convert one side, or fix the unit
```

A diagnostic is printed as these parts, in this order, each line ending in `\n`:

1. **Header:** `<file>:<line>:<col>: <severity>: <message>`. `<file>` is the file name exactly as the tool was given it, escaped as described in [Terminal safety](#terminal-safety). `<line>:<col>` is where the primary span starts.
2. **Blank gutter row:** the gutter padding followed by ` |`.
3. **Primary span:** the source line containing the primary span's start, written as `<line number> | <source text>` (escaped as described in [Terminal safety](#terminal-safety)), and under it an underline row written as `<padding> | <indent><markers>`. The primary span's markers are `^`. If the primary span has a label, the row ends with one space and the label.
4. **Secondary spans,** in source order (by start offset, then end offset; ties keep the order they were added). Their markers are `-`. If a secondary span is on the source line printed just before it, only its underline row is added. Otherwise its source line is printed first.
5. **Notes,** each `<padding> = note: <text>`, in the order they were added.
6. **Help lines,** each `<padding> = help: <text>`, in the order they were added, after all notes.

The gutter is as wide as the largest line number shown in the diagnostic. Line numbers are right-aligned in it, and `<padding>` is that many spaces. A source line that is empty is written as `<line number> |`, with no trailing space.

## Severities

`error`, `warning`, and `note`, printed in lowercase in the header. Severity does not change the layout or the marker characters.

## Spans, lines, and columns

- A span is a half-open range of byte offsets `start..end` into the file's UTF-8 text.
- Lines are separated by `\n`, and lines and columns are 1-based.
- **Columns count Unicode scalar values,** not bytes and not display cells. `∂`, `ħ`, and each of `⁻` and `¹` count as one column. Display width is not computed, so wide or combining characters before a span can shift the markers in a terminal.
- **Tabs** are printed as-is in the source line. In the underline row's indent, each character before the span becomes a tab if it is a tab, 8 spaces if it is escaped (see [Terminal safety](#terminal-safety)), and a space otherwise, so the markers line up under the displayed line at any tab width. A tab inside a span gets one marker.
- **CRLF:** a `\r` directly before a `\n` belongs to the line terminator. It is never printed, never counted as a column, and never underlined. A lone `\r` is not a line terminator: it counts as one column and is displayed escaped.
- **Marker count:** one marker per displayed column of the span on its line, where an escaped character is 8 columns, so a span covering an escaped character underlines its whole escape. There is always at least one, so a zero-width span, a span at end of file, and a span that covers only a line terminator each get exactly one marker, at the column where the span starts.
- **End of file:** a span at end of file points just past the last character. If the file ends in `\n`, that position is column 1 of the empty line after it.
- **Multi-line spans** are underlined on their first line only, from the span's start to the end of that line.
- All spans in one diagnostic refer to the same file.

## Invalid spans

Rendering never fails on a bad span. Spans are clamped as follows:

- an offset past the end of the text is treated as end of file;
- an offset inside a multi-byte character moves back to that character's first byte;
- a span whose `end` is before its `start` is treated as zero-width at `start`.

## Terminal safety

Diagnostics echo source text and messages that may contain characters able to change how a terminal shows the output (Trojan Source reordering, ANSI escape sequences, hidden text). Every string the renderer prints is therefore escaped: the file name, the message, the source lines, labels, notes, and help lines. The only exception is the renderer's own line breaks.

These characters are replaced by a visible `\u{XXXX}` escape, using four uppercase hex digits (for example `\u{202E}` or `\u{001B}`):

- C0 controls U+0000–U+001F except tab, including ESC, `\r`, and `\n`;
- DEL U+007F and the C1 controls U+0080–U+009F, including CSI U+009B;
- bidi controls U+202A–U+202E, U+2066–U+2069, U+200E, U+200F, and U+061C;
- zero-width and invisible characters U+200B–U+200D, U+2060, and U+FEFF.

All other characters are printed unchanged, including tab and non-ASCII such as `∂`, `⁻¹`, and `ħ`. A backslash is not escaped, so source text that literally contains `\u{202E}` looks the same as an escaped U+202E.

Escaping affects only the displayed text. Offsets and the header's `<line>:<col>` still refer to the unescaped source, counted as described above. In the underline row an escaped character is 8 columns wide (the length of `\u{XXXX}`), so the markers line up under the escaped text:

```text
demo.sky:2:25: error: bidirectional control character \u{202E} is not allowed
  |
2 | if access != "none":  # \u{202E} check admin
  |                         ^^^^^^^^ bidi control here
```

`sky_diag::display_safe` implements this escaping. Tools that print other untrusted text, such as a path in a status line, should use it too. Source text must never be escaped before rendering, because spans would then point at the wrong bytes.

## Output

- No colour or other terminal escape sequences, and no unescaped characters from the set in [Terminal safety](#terminal-safety).
- No trailing whitespace, except whitespace that is part of a source line.
