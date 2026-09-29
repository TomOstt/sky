//! Lexer and parser for Sky.
//!
//! - [`lex`] turns source text into [`Token`]s plus warnings; see
//!   [`lexer`] for the token, span, and indentation rules.
//! - [`Span`] is `sky_diag::Span`, re-exported so users of the syntax crate
//!   need not depend on `sky-diag` just to name it.
//!
//! Security rules (spec v0.1): source containing Unicode bidirectional
//! control characters is rejected before tokenizing ("Trojan Source"), and
//! confusable identifiers produce warnings (see [`confusable`]).

pub mod confusable;
pub mod lexer;
pub mod source;
pub mod token;

pub use lexer::{lex, LexError, LexErrorKind, Lexed};
pub use sky_diag::Span;
pub use token::{Kw, Token, TokenKind};
