//! Source spans, diagnostics, and Sky's plain-text diagnostic renderer.
//!
//! This crate has no dependencies on other Sky crates. [`Span`] lives here so
//! that `sky-syntax`, `sky-ast`, `sky-types`, `sky-interp`, and `sky-cli` all
//! share one span type (`sky-syntax` can re-export it).
//!
//! # Format
//!
//! The rendered format is normative and defined in
//! [`spec/diagnostics.md`](https://github.com/TomOstt/sky/blob/main/spec/diagnostics.md):
//! a `file:line:col: severity: message` header, a gutter with the source
//! line, `^` / `-` underlines with labels, then `= note:` / `= help:` lines.
//! The `tests/ui/<name>.stderr` goldens contain exactly this output, so a
//! change here is a change to the spec and to every golden file.
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

mod capped;
mod diagnostic;
mod display_unsafe_table;
mod render;
mod safe;
mod source;
mod span;
pub mod stack;

pub use capped::{display_capped, DISPLAY_CAP_BYTES, DISPLAY_CAP_CHARS};
pub use diagnostic::{Diagnostic, Label, Severity};
pub use display_unsafe_table::UNICODE_VERSION as DISPLAY_UNSAFE_UNICODE_VERSION;
pub use safe::{display_safe, is_display_unsafe};
pub use source::SourceFile;
pub use span::Span;
pub use stack::{with_deep_stack, DEEP_STACK_BYTES};
