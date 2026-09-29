//! Lexer and parser for Sky.
//!
//! - [`lex`] turns source text into [`Token`]s plus warnings; see
//!   [`lexer`] for the token, span, and indentation rules.
//! - [`MAX_SOURCE_BYTES`] (16 MiB) is the largest source file the lexer and
//!   the module loader accept.
//! - [`Span`] is `sky_diag::Span`, re-exported so users of the syntax crate
//!   need not depend on `sky-diag` just to name it.
//!
//! Security rules (spec v0.1): source containing Unicode bidirectional
//! control characters is rejected before tokenizing ("Trojan Source"), and
//! confusable identifiers produce warnings (see [`confusable`]).

pub mod confusable;
mod invisible_table;
pub mod lexer;
pub mod source;
pub mod token;

pub use lexer::{lex, LexError, LexErrorKind, Lexed};
/// The largest source file Sky reads or lexes, in bytes (16 MiB).
///
/// This is the single source-size limit, shared by the lexer ([`lex`]
/// rejects larger input before any other check) and by the module loader
/// (`sky-project` reads at most this many bytes), so a file one of them
/// accepts is never refused by the other. It must stay at most `u32::MAX`,
/// because [`Span`] offsets are `u32`; the assertion below enforces that at
/// compile time for the lexer (`sky-project` has its own for the loader).
pub const MAX_SOURCE_BYTES: usize = 16 << 20; // 16 MiB
const _: () = assert!(MAX_SOURCE_BYTES <= u32::MAX as usize);
pub use sky_diag::Span;
pub use token::{Kw, Token, TokenKind};
