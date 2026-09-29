//! Typed AST for Sky, with source spans, plus the Pratt expression parser.
//!
//! Entry points: [`parse_expr`] and [`parse_module`]. Both take the lexer's
//! token slice plus the source text (token text is `source[span]`), return
//! every error they found, and never panic on any input.
//!
//! Design decisions that still need Eng Lead sign-off are listed in
//! `crates/sky-ast/PARSER_DECISIONS.md`.

#![forbid(unsafe_code)]

pub mod ast;
pub mod parser;
pub mod precedence;
mod sexpr;
pub mod token;

pub use ast::{
    BinaryOp, Expr, ExprKind, Ident, Literal, Module, Stmt, StmtKind, UnaryOp, UnitExpr,
};
pub use parser::{parse_expr, parse_module, ParseError, MAX_DEPTH, MAX_UNIT_FACTORS};
pub use sky_diag::Span;
pub use token::{Kw, Token, TokenKind};
