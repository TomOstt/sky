//! Lexer and parser for Sky.
//!
//! Security rule (spec v0.1): source containing Unicode bidirectional
//! control characters is rejected before tokenizing ("Trojan Source").

pub mod source;
