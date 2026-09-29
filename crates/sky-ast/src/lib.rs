//! Typed AST for Sky, with source spans, plus the parser: a Pratt expression
//! parser and the statement and block layer above it (`if`, `while`, `def`).
//!
//! Entry points: [`parse_expr`] and [`parse_module`]. Both take the lexer's
//! token slice plus the source text (token text is `source[span]`), return
//! every error they found, and never panic on any input.
//! Entry points that parse, check, or interpret run inside
//! [`with_deep_stack`], so the depth limits hold on a stack of known size.
//!
//! Design decisions that still need Eng Lead sign-off are listed in
//! `crates/sky-ast/PARSER_DECISIONS.md`.

#![forbid(unsafe_code)]

pub mod ast;
pub mod parser;
pub mod precedence;
mod sexpr;
pub mod stack;

pub use ast::{
    AssignTarget, BinaryOp, Block, BoolOp, CompoundOp, DimExpr, Expr, ExprKind, Ident, IndexItem,
    Literal, Module, Param, ShapeAnn, ShapeKind, Stmt, StmtKind, TypeExpr, UnaryOp, UnitExpr,
    TAG_NAMES,
};
pub use parser::{
    parse_expr, parse_module, ParseError, MAX_DEPTH, MAX_QUOTED_BYTES, MAX_QUOTED_CHARS,
    MAX_UNIT_FACTORS,
};
pub use sky_syntax::{Kw, Span, Token, TokenKind};
pub use stack::{with_deep_stack, DEEP_STACK_BYTES};
