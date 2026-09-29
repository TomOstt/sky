//! Source spans, diagnostics, and Sky's plain-text diagnostic renderer.
//!
//! This crate has no dependencies on other Sky crates. [`Span`] lives here so
//! that `sky-syntax`, `sky-ast`, `sky-types`, `sky-interp`, and `sky-cli` all
//! share one span type (`sky-syntax` can re-export it).
//!
//! # Format
//!
//! [`Diagnostic::render`] produces the text below. It is the base format for
//! the `tests/ui/<name>.stderr` golden files, so any change to it is a change
//! to every golden test.
//!
//! ```text
//! demo.sky:1:14: error: cannot add length and temperature
//!   |
//! 1 | x = 5 meters + 20 kelvin
//!   |              ^ mismatched dimensions
//!   |     -------- length
//!   |                --------- temperature
//!   = note: both operands of `+` must have the same dimension
//!   = help: convert one side, or fix the unit
//! ```
//!
//! 1. Header: `file:line:col: severity: message`. `file` is
//!    [`SourceFile::name`] verbatim; `line:col` is the primary span's start.
//! 2. A blank gutter row (`  |`). The gutter is as wide as the largest line
//!    number shown, and line numbers are right-aligned.
//! 3. The primary span: its source line, then a row of `^` under it and the
//!    primary label, if any.
//! 4. Each secondary span, in source order (ties keep insertion order), drawn
//!    with `-`. A secondary span on the line just shown adds only an underline
//!    row; otherwise its source line is shown first.
//! 5. `= note:` lines, then `= help:` lines, each in insertion order.
//!
//! Rules:
//!
//! - Lines and columns are 1-based. Columns count Unicode scalar values
//!   (`char`s), so `∂` or each of `⁻` and `¹` is one column. The renderer does
//!   not compute display width: wide or combining characters before a span
//!   can shift the markers in a terminal.
//! - Tabs in the source line are printed as-is, and the underline indent uses
//!   a tab for each tab before the span, so markers line up in any tab width.
//!   A tab inside a span counts as one marker.
//! - `\r\n` is a line terminator: the `\r` is never printed or underlined. A
//!   lone `\r` is ordinary text.
//! - Zero-width spans, spans at end of file, and spans covering only a line
//!   terminator get exactly one marker. A span at end of a file that ends in
//!   `\n` points at the empty line after it.
//! - A span covering several lines is underlined on its first line only, up
//!   to the end of that line.
//! - Spans are clamped, never trusted: offsets past the end go to end of file,
//!   offsets inside a multi-byte character move back to its first byte, and a
//!   span with `end < start` renders as zero-width at `start`. Rendering never
//!   panics.
//! - No colour and no trailing whitespace (unless the source line has some).
//! - All spans in a diagnostic refer to the one [`SourceFile`] passed in.
//!
//! # Example
//!
//! ```
//! use sky_diag::{Diagnostic, SourceFile, Span};
//!
//! let file = SourceFile::new("demo.sky", "x = 5 meters + 20 kelvin\n");
//! let diag = Diagnostic::error("cannot add length and temperature", Span::new(13, 14))
//!     .with_label("mismatched dimensions")
//!     .with_secondary(Span::new(4, 12), "length");
//! assert_eq!(
//!     diag.render_to_string(&file),
//!     "demo.sky:1:14: error: cannot add length and temperature\n  \
//!      |\n1 | x = 5 meters + 20 kelvin\n  \
//!      |              ^ mismatched dimensions\n  \
//!      |     -------- length\n"
//! );
//! ```

mod diagnostic;
mod render;
mod source;
mod span;

pub use diagnostic::{Diagnostic, Label, Severity};
pub use source::SourceFile;
pub use span::Span;
