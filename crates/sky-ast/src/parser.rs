//! Pratt expression parser over the lexer's token slice.
//!
//! Guarantees (this is a fuzz target):
//! - Never panics on any token sequence and any source text. Tokens are read
//!   through bounds-checked accessors, token text through [`span_text`], and
//!   integer text through checked conversions.
//! - A missing `Eof` is fine; tokens after the first `Eof` are ignored.
//! - Recursion and AST height are bounded by [`MAX_DEPTH`]: exceeding it is a
//!   parse error, not a stack overflow.
//! - Every `None` returned by an internal parse function means an error has
//!   already been recorded, so callers just propagate it.
//!
//! The parser never evaluates anything; it only builds the AST.

#![warn(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use sky_diag::{Diagnostic, Label};
use sky_syntax::{Span, Token, TokenKind};

use crate::ast::{span_text, Expr, ExprKind, Ident, Literal, Module, Stmt, StmtKind, UnitExpr};
use crate::precedence::{infix_binding_power, is_postfix_start, prefix_binding_power, POSTFIX_BP};

/// Maximum nesting: recursive sub-expressions plus operators folded into one
/// operand chain. `(((x)))` uses 4; `a + b + c` uses 3.
pub const MAX_DEPTH: u32 = 256;

/// Maximum number of unit names in one unit literal (`kg*m/s^2` has 3).
pub const MAX_UNIT_FACTORS: usize = 64;

/// A syntax error. Maps one-to-one onto a [`sky_diag::Diagnostic`] with
/// severity `error` via [`ParseError::into_diagnostic`] or `From`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    /// The primary span (`^` markers).
    pub span: Span,
    /// Labeled secondary spans (`-` markers), e.g. the `(` an unclosed `)`
    /// belongs to.
    pub secondary: Vec<Label>,
    /// An optional `= help:` line, e.g. a suggested rewrite.
    pub help: Option<String>,
}

impl ParseError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        ParseError {
            message: message.into(),
            span,
            secondary: Vec::new(),
            help: None,
        }
    }

    fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.secondary.push(Label {
            span,
            message: Some(label.into()),
        });
        self
    }

    fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn into_diagnostic(self) -> Diagnostic {
        let mut diag = Diagnostic::error(self.message, self.span);
        diag.secondary = self.secondary;
        if let Some(help) = self.help {
            diag = diag.with_help(help);
        }
        diag
    }
}

impl From<ParseError> for Diagnostic {
    fn from(err: ParseError) -> Diagnostic {
        err.into_diagnostic()
    }
}

/// Parses one expression. Leading and trailing newlines are skipped; any
/// other trailing token is an error (the expression is still returned).
/// Returns `None` only if no expression could be built, and then the error
/// list is non-empty.
pub fn parse_expr(tokens: &[Token], src: &str) -> (Option<Expr>, Vec<ParseError>) {
    let mut p = Parser::new(tokens, src);
    p.skip_newlines();
    let expr = p.expr_bp(0);
    if let Some(e) = &expr {
        p.skip_newlines();
        if p.peek() != TokenKind::Eof {
            let err = p.unexpected_after(e, "end of input");
            p.errors.push(err);
        }
    }
    (expr, p.errors)
}

/// Parses top-level simple statements separated by `Newline`. On an error,
/// the rest of the line is skipped and parsing resumes on the next line.
pub fn parse_module(tokens: &[Token], src: &str) -> (Module, Vec<ParseError>) {
    let mut p = Parser::new(tokens, src);
    let mut stmts = Vec::new();
    loop {
        match p.peek() {
            TokenKind::Eof => break,
            // Blank lines, and dedents matching an indent reported below.
            TokenKind::Newline | TokenKind::Dedent => {
                p.bump();
                continue;
            }
            TokenKind::Indent => {
                let t = p.bump();
                p.errors.push(ParseError::new(
                    "unexpected indentation: indented blocks are not supported yet",
                    t.span,
                ));
                continue;
            }
            _ => {}
        }
        let start = p.pos;
        p.depth = 0;
        match p.parse_stmt() {
            Some(stmt) => stmts.push(stmt),
            None => p.recover_to_line_end(),
        }
        if p.pos == start {
            p.bump(); // Guarantee progress.
        }
    }
    let end = u32::try_from(src.len()).unwrap_or(u32::MAX);
    let module = Module {
        stmts,
        span: Span { start: 0, end },
    };
    (module, p.errors)
}

struct Parser<'t, 's> {
    /// Tokens up to (not including) the first `Eof`.
    tokens: &'t [Token],
    src: &'s str,
    pos: usize,
    depth: u32,
    eof_span: Span,
    errors: Vec<ParseError>,
}

impl<'t, 's> Parser<'t, 's> {
    fn new(tokens: &'t [Token], src: &'s str) -> Self {
        let end = tokens
            .iter()
            .position(|t| t.kind == TokenKind::Eof)
            .unwrap_or(tokens.len());
        let eof_span = match tokens.get(end) {
            Some(t) => t.span,
            None => tokens
                .last()
                .map(|t| Span::point(t.span.end))
                .unwrap_or(Span::point(0)),
        };
        Parser {
            tokens: tokens.get(..end).unwrap_or(tokens),
            src,
            pos: 0,
            depth: 0,
            eof_span,
            errors: Vec::new(),
        }
    }

    // ----- token access (all bounds-checked) -----

    fn current(&self) -> Token {
        self.nth(0).unwrap_or(Token {
            kind: TokenKind::Eof,
            span: self.eof_span,
        })
    }

    fn nth(&self, n: usize) -> Option<Token> {
        self.tokens.get(self.pos.saturating_add(n)).copied()
    }

    fn peek(&self) -> TokenKind {
        self.current().kind
    }

    fn bump(&mut self) -> Token {
        let t = self.current();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn prev_span(&self) -> Option<Span> {
        self.pos
            .checked_sub(1)
            .and_then(|i| self.tokens.get(i))
            .map(|t| t.span)
    }

    fn text(&self, span: Span) -> &'s str {
        span_text(self.src, span).unwrap_or("?")
    }

    fn skip_newlines(&mut self) {
        while self.peek() == TokenKind::Newline {
            self.bump();
        }
    }

    fn recover_to_line_end(&mut self) {
        loop {
            match self.peek() {
                TokenKind::Eof => return,
                TokenKind::Newline => {
                    self.bump();
                    return;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Counts one level of nesting; errors once the limit is passed.
    fn enter(&mut self, span: Span) -> Option<()> {
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            self.errors.push(ParseError::new(
                format!("expression is nested too deeply (the limit is {MAX_DEPTH} levels)"),
                span,
            ));
            return None;
        }
        Some(())
    }

    // ----- statements -----

    fn parse_stmt(&mut self) -> Option<Stmt> {
        let first = self.expr_bp(0)?;
        let tok = self.current();
        let stmt = match tok.kind {
            TokenKind::Colon => {
                self.bump();
                let name = self.binding_target(
                    &first,
                    tok.span,
                    "invalid annotation target: only a plain name can have a type annotation",
                )?;
                let ty = self.expr_bp(0)?;
                let value = if self.peek() == TokenKind::Assign {
                    self.bump();
                    Some(self.expr_bp(0)?)
                } else {
                    None
                };
                let end = value.as_ref().map_or(ty.span, |v| v.span);
                Stmt {
                    span: first.span.join(end),
                    kind: StmtKind::Binding { name, ty, value },
                }
            }
            TokenKind::Assign => {
                self.bump();
                let target = self.binding_target(
                    &first,
                    tok.span,
                    "invalid assignment target: only a plain name can be assigned in Sky v0.1",
                )?;
                let value = self.expr_bp(0)?;
                Stmt {
                    span: first.span.join(value.span),
                    kind: StmtKind::Assign { target, value },
                }
            }
            _ => Stmt {
                span: first.span,
                kind: StmtKind::Expr(first),
            },
        };
        self.expect_stmt_end(&stmt)?;
        Some(stmt)
    }

    fn binding_target(&mut self, target: &Expr, op: Span, message: &str) -> Option<Ident> {
        if let ExprKind::Name(id) = &target.kind {
            return Some(*id);
        }
        self.errors
            .push(ParseError::new(message, target.span).with_secondary(op, "because of this"));
        None
    }

    fn expect_stmt_end(&mut self, stmt: &Stmt) -> Option<()> {
        let tok = self.current();
        match tok.kind {
            TokenKind::Newline => {
                self.bump();
                Some(())
            }
            TokenKind::Eof => Some(()),
            TokenKind::Assign => {
                self.errors.push(ParseError::new(
                    "chained assignment (`a = b = c`) is not supported",
                    tok.span,
                ));
                None
            }
            _ => {
                let last = match &stmt.kind {
                    StmtKind::Expr(e) => e,
                    StmtKind::Assign { value, .. } => value,
                    StmtKind::Binding { ty, value, .. } => value.as_ref().unwrap_or(ty),
                };
                let err = self.unexpected_after(last, "end of line");
                self.errors.push(err);
                None
            }
        }
    }

    // ----- expressions -----

    fn expr_bp(&mut self, min_bp: u8) -> Option<Expr> {
        let saved = self.depth;
        let result = self.expr_bp_inner(min_bp);
        self.depth = saved;
        result
    }

    fn expr_bp_inner(&mut self, min_bp: u8) -> Option<Expr> {
        self.enter(self.current().span)?;
        let mut lhs = self.parse_prefix()?;
        // A comparison already folded at this level: comparisons are
        // non-associative, so a second one is an error.
        let mut prev_comparison: Option<Span> = None;
        loop {
            let tok = self.current();
            if is_postfix_start(tok.kind) {
                if POSTFIX_BP < min_bp {
                    break;
                }
                self.enter(tok.span)?;
                lhs = self.parse_postfix(lhs)?;
                continue;
            }
            let Some((op, l_bp, r_bp)) = infix_binding_power(tok.kind) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.bump();
            if op.is_comparison() {
                if let Some(prev) = prev_comparison {
                    self.errors.push(
                        ParseError::new(
                            "comparison operators cannot be chained in Sky v0.1",
                            tok.span,
                        )
                        .with_secondary(prev, "previous comparison")
                        .with_help("compare two values at a time"),
                    );
                }
                prev_comparison = Some(tok.span);
            }
            self.enter(tok.span)?;
            let rhs = self.expr_bp(r_bp)?;
            lhs = Expr {
                span: lhs.span.join(rhs.span),
                kind: ExprKind::Binary {
                    op,
                    op_span: tok.span,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Some(lhs)
    }

    fn parse_prefix(&mut self) -> Option<Expr> {
        let tok = self.current();
        if let Some((op, r_bp)) = prefix_binding_power(tok.kind) {
            self.bump();
            let operand = self.expr_bp(r_bp)?;
            return Some(Expr {
                span: tok.span.join(operand.span),
                kind: ExprKind::Unary {
                    op,
                    op_span: tok.span,
                    operand: Box::new(operand),
                },
            });
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Option<Expr> {
        let tok = self.current();
        let literal = |kind| Expr {
            kind: ExprKind::Literal(kind),
            span: tok.span,
        };
        match tok.kind {
            TokenKind::Int | TokenKind::Float => {
                self.bump();
                let value = if tok.kind == TokenKind::Int {
                    Literal::Int(tok.span)
                } else {
                    Literal::Float(tok.span)
                };
                if self.peek() == TokenKind::Ident {
                    return self.parse_unit_literal(value);
                }
                Some(literal(value))
            }
            TokenKind::Str => {
                self.bump();
                Some(literal(Literal::Str(tok.span)))
            }
            TokenKind::Ident => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::Name(Ident { span: tok.span }),
                    span: tok.span,
                })
            }
            TokenKind::LParen => {
                self.bump();
                let inner = self.expr_bp(0)?;
                let close = self.expect_closing(TokenKind::RParen, tok.span, &inner, "`)`")?;
                Some(Expr {
                    span: tok.span.join(close),
                    kind: ExprKind::Paren(Box::new(inner)),
                })
            }
            _ => {
                let message = match tok.kind {
                    TokenKind::Keyword(kw) => format!(
                        "expected an expression, found keyword `{}`; statements and blocks are not supported by the parser yet",
                        kw.as_str()
                    ),
                    _ => format!("expected an expression, found {}", self.describe(tok)),
                };
                self.errors.push(ParseError::new(message, tok.span));
                None
            }
        }
    }

    /// Called with the operator token current; `lhs` is the operand so far.
    fn parse_postfix(&mut self, lhs: Expr) -> Option<Expr> {
        let tok = self.bump();
        match tok.kind {
            TokenKind::Inverse => Some(Expr {
                span: lhs.span.join(tok.span),
                kind: ExprKind::Inverse {
                    operand: Box::new(lhs),
                    op_span: tok.span,
                },
            }),
            TokenKind::Dot => {
                let name = self.current();
                // Any keyword may be an attribute name (`linalg.solve`),
                // signed off by Eng Lead. Only here: a bare keyword is still
                // not a name.
                let is_name = matches!(name.kind, TokenKind::Ident | TokenKind::Keyword(_));
                if !is_name {
                    self.errors.push(
                        ParseError::new(
                            format!(
                                "expected an attribute name after `.`, found {}",
                                self.describe(name)
                            ),
                            name.span,
                        )
                        .with_secondary(tok.span, "attribute access"),
                    );
                    return None;
                }
                self.bump();
                Some(Expr {
                    span: lhs.span.join(name.span),
                    kind: ExprKind::Attribute {
                        base: Box::new(lhs),
                        name: Ident { span: name.span },
                    },
                })
            }
            TokenKind::LParen => {
                let (args, close) =
                    self.parse_delimited(tok.span, TokenKind::RParen, "`)`", true)?;
                Some(Expr {
                    span: lhs.span.join(close),
                    kind: ExprKind::Call {
                        callee: Box::new(lhs),
                        args,
                    },
                })
            }
            TokenKind::LBracket => {
                let (indices, close) =
                    self.parse_delimited(tok.span, TokenKind::RBracket, "`]`", false)?;
                Some(Expr {
                    span: lhs.span.join(close),
                    kind: ExprKind::Index {
                        base: Box::new(lhs),
                        indices,
                    },
                })
            }
            _ => {
                // Unreachable by construction; report instead of panicking.
                self.errors.push(ParseError::new(
                    format!("unexpected {}", self.describe(tok)),
                    tok.span,
                ));
                None
            }
        }
    }

    /// Comma-separated expressions after an opening delimiter (already
    /// consumed), up to and including `close`. A trailing comma is allowed.
    fn parse_delimited(
        &mut self,
        open: Span,
        close: TokenKind,
        close_str: &str,
        allow_empty: bool,
    ) -> Option<(Vec<Expr>, Span)> {
        let mut items = Vec::new();
        if self.peek() == close {
            let c = self.bump();
            if !allow_empty {
                self.errors.push(ParseError::new(
                    "expected at least one index between `[` and `]`",
                    open.join(c.span),
                ));
                return None;
            }
            return Some((items, c.span));
        }
        loop {
            let item = self.expr_bp(0)?;
            let next = self.peek();
            if next == TokenKind::Comma {
                items.push(item);
                self.bump();
                if self.peek() == close {
                    break;
                }
                continue;
            }
            if next == close {
                items.push(item);
                break;
            }
            let open_str = if close == TokenKind::RParen { "(" } else { "[" };
            let err = self
                .unexpected_after(&item, &format!("`,` or {close_str}"))
                .with_secondary(open, format!("unclosed `{open_str}`"));
            self.errors.push(err);
            return None;
        }
        let c = self.bump();
        Some((items, c.span))
    }

    fn expect_closing(
        &mut self,
        close: TokenKind,
        open: Span,
        last: &Expr,
        close_str: &str,
    ) -> Option<Span> {
        if self.peek() == close {
            return Some(self.bump().span);
        }
        let err = self
            .unexpected_after(last, close_str)
            .with_secondary(open, "unclosed `(`");
        self.errors.push(err);
        None
    }

    /// The error for an unexpected token right after the complete
    /// expression `prev`, where `expected` was wanted.
    fn unexpected_after(&self, prev: &Expr, expected: &str) -> ParseError {
        let tok = self.current();
        match tok.kind {
            TokenKind::Ident => {
                let name = self.text(tok.span);
                ParseError::new(
                    format!(
                        "unexpected name `{name}` after an expression; a unit may only follow a numeric literal"
                    ),
                    tok.span,
                )
                .with_secondary(prev.span, "this is not a numeric literal")
                .with_help(format!(
                    "write a unit right after a number (`10 {name}`), or multiply explicitly (`... * {name}`)"
                ))
            }
            TokenKind::Int | TokenKind::Float | TokenKind::Str => ParseError::new(
                format!(
                    "unexpected {} after an expression; expected {expected}",
                    self.describe(tok)
                ),
                tok.span,
            )
            .with_secondary(prev.span, "previous expression")
            .with_help("two expressions cannot be written side by side; is an operator missing?"),
            _ => ParseError::new(
                format!("expected {expected}, found {}", self.describe(tok)),
                tok.span,
            ),
        }
    }

    fn describe(&self, tok: Token) -> String {
        match tok.kind {
            TokenKind::Int => format!("integer literal `{}`", self.text(tok.span)),
            TokenKind::Float => format!("float literal `{}`", self.text(tok.span)),
            TokenKind::Str => "string literal".to_string(),
            TokenKind::Ident => format!("name `{}`", self.text(tok.span)),
            TokenKind::Newline => "end of line".to_string(),
            TokenKind::Indent => "indentation".to_string(),
            TokenKind::Dedent => "dedent".to_string(),
            TokenKind::Eof => "end of input".to_string(),
            TokenKind::Keyword(kw) => format!("keyword `{}`", kw.as_str()),
            other => format!("`{}`", symbol(other)),
        }
    }

    // ----- unit literals -----
    //
    // Grammar (spec/v0.1.md, "Unit literals"; `~` = adjacent, no whitespace):
    //   unit_literal := NUMBER unit
    //   unit         := factor (~("*" | "/")~ factor)*
    //   factor       := IDENT (~"^"~ exponent)?
    //   exponent     := ["-"~] INT | "("~ ["-"~] INT ~"/"~ INT ~")"

    fn parse_unit_literal(&mut self, value: Literal) -> Option<Expr> {
        let mut unit = self.parse_unit_factor()?;
        let mut factors = 1usize;
        loop {
            let op = self.current();
            let is_mul = match op.kind {
                TokenKind::Star => true,
                TokenKind::Slash => false,
                _ => break,
            };
            let Some(next) = self.nth(1) else { break };
            let before_adjacent = self.prev_span().is_some_and(|p| p.end == op.span.start);
            if !before_adjacent || op.span.end != next.span.start {
                break; // Spaced operator: ordinary arithmetic on the quantity.
            }
            self.bump();
            if next.kind != TokenKind::Ident {
                let verb = if is_mul { "multiply" } else { "divide" };
                self.errors.push(
                    ParseError::new(
                        format!(
                            "expected a unit name after `{}` in a unit literal, found {}",
                            symbol(op.kind),
                            self.describe(next)
                        ),
                        next.span,
                    )
                    .with_secondary(op.span, "no spaces around this, so it is part of the unit")
                    .with_help(format!(
                        "to {verb} the quantity, put spaces around `{}`",
                        symbol(op.kind)
                    )),
                );
                return None;
            }
            factors += 1;
            if factors > MAX_UNIT_FACTORS {
                self.errors.push(ParseError::new(
                    format!("unit has too many factors (the limit is {MAX_UNIT_FACTORS})"),
                    next.span,
                ));
                return None;
            }
            let rhs = self.parse_unit_factor()?;
            let span = unit.span().join(rhs.span());
            let (lhs, rhs) = (Box::new(unit), Box::new(rhs));
            unit = if is_mul {
                UnitExpr::Mul { lhs, rhs, span }
            } else {
                UnitExpr::Div { lhs, rhs, span }
            };
        }
        // `5 kg m`: compound units need `*`, never juxtaposition.
        let tok = self.current();
        if tok.kind == TokenKind::Ident {
            let unit_text = self.text(unit.span());
            let name = self.text(tok.span);
            self.errors.push(
                ParseError::new(
                    format!("unexpected name `{name}` after the unit `{unit_text}`"),
                    tok.span,
                )
                .with_secondary(unit.span(), "unit")
                .with_help(format!(
                    "compound units are written with `*` and no spaces: `{unit_text}*{name}`"
                )),
            );
            return None;
        }
        Some(Expr {
            span: value.span().join(unit.span()),
            kind: ExprKind::UnitLiteral { value, unit },
        })
    }

    /// `IDENT (~"^"~ exponent)?` with the `IDENT` current.
    fn parse_unit_factor(&mut self) -> Option<UnitExpr> {
        let name = self.bump();
        if name.kind != TokenKind::Ident {
            self.errors.push(ParseError::new(
                format!("expected a unit name, found {}", self.describe(name)),
                name.span,
            ));
            return None;
        }
        let base = UnitExpr::Name(Ident { span: name.span });
        let caret = self.current();
        let joined = |a: Span, b: Span| a.end == b.start;
        if caret.kind != TokenKind::Caret || !joined(name.span, caret.span) {
            return Some(base);
        }
        match self.nth(1) {
            Some(after) if joined(caret.span, after.span) => {}
            _ => return Some(base), // `m ^2`/`m^ 2`: ordinary `^` on the quantity.
        }
        self.bump(); // `^`
        let (num, den, exponent_span) = self.parse_unit_exponent(caret.span)?;
        let end = exponent_span;
        // `m^2^3`
        let next = self.current();
        if next.kind == TokenKind::Caret
            && joined(end, next.span)
            && self.nth(1).is_some_and(|t| joined(next.span, t.span))
        {
            self.errors.push(
                ParseError::new("unit exponents cannot be chained", next.span)
                    .with_help("write a single exponent, e.g. `m^6`"),
            );
            return None;
        }
        Some(UnitExpr::Pow {
            base: Box::new(base),
            num,
            den,
            exponent_span,
            span: name.span.join(end),
        })
    }

    /// Parses the exponent after an adjacent `^` (already consumed).
    /// Returns `(num, den, span)` with `den > 0`.
    fn parse_unit_exponent(&mut self, caret: Span) -> Option<(i64, u64, Span)> {
        let start = self.current();
        if start.kind == TokenKind::LParen {
            self.bump();
            let negative = self.eat_adjacent(TokenKind::Minus).is_some();
            let num_tok = self.eat_adjacent(TokenKind::Int);
            let slash = num_tok.and_then(|_| self.eat_adjacent(TokenKind::Slash));
            let den_tok = slash.and_then(|_| self.eat_adjacent(TokenKind::Int));
            let close = den_tok.and_then(|_| self.eat_adjacent(TokenKind::RParen));
            let (Some(num_tok), Some(den_tok), Some(close)) = (num_tok, den_tok, close) else {
                let at = self.current();
                self.errors.push(
                    ParseError::new(
                        format!(
                            "malformed rational unit exponent: found {}",
                            self.describe(at)
                        ),
                        at.span,
                    )
                    .with_secondary(start.span, "rational exponent starts here")
                    .with_help("write it as `m^(1/2)` or `m^(-3/2)`, with no spaces"),
                );
                return None;
            };
            let num = self.int_exponent(num_tok, negative)?;
            let den = self.uint_exponent(den_tok)?;
            if den == 0 {
                self.errors.push(ParseError::new(
                    "the denominator of a unit exponent cannot be zero",
                    den_tok.span,
                ));
                return None;
            }
            return Some((num, den, start.span.join(close.span)));
        }
        let negative = self.eat_adjacent(TokenKind::Minus);
        match self.eat_adjacent(TokenKind::Int) {
            Some(int_tok) => {
                let num = self.int_exponent(int_tok, negative.is_some())?;
                let span = negative.map_or(int_tok.span, |m| m.span.join(int_tok.span));
                Some((num, 1, span))
            }
            None => {
                let at = self.current();
                self.errors.push(
                    ParseError::new(
                        format!(
                            "a unit exponent must be an integer or a parenthesized rational, found {}",
                            self.describe(at)
                        ),
                        at.span,
                    )
                    .with_secondary(caret, "unit exponent")
                    .with_help("write e.g. `s^-2` or `m^(1/2)`, with no spaces"),
                );
                None
            }
        }
    }

    /// Consumes the current token if it has `kind` and directly follows the
    /// previous token.
    fn eat_adjacent(&mut self, kind: TokenKind) -> Option<Token> {
        let tok = self.current();
        let adjacent = self.prev_span().is_some_and(|p| p.end == tok.span.start);
        if tok.kind == kind && adjacent {
            self.bump();
            Some(tok)
        } else {
            None
        }
    }

    fn uint_exponent(&mut self, tok: Token) -> Option<u64> {
        let parsed = span_text(self.src, tok.span).and_then(parse_u64);
        if parsed.is_none() {
            self.errors.push(ParseError::new(
                format!(
                    "unit exponent `{}` is not a valid 64-bit integer",
                    self.text(tok.span)
                ),
                tok.span,
            ));
        }
        parsed
    }

    fn int_exponent(&mut self, tok: Token, negative: bool) -> Option<i64> {
        let magnitude = self.uint_exponent(tok)?;
        let value = i64::try_from(magnitude)
            .ok()
            .map(|v| if negative { -v } else { v });
        if value.is_none() {
            self.errors.push(ParseError::new(
                format!(
                    "unit exponent `{}` is not a valid 64-bit integer",
                    self.text(tok.span)
                ),
                tok.span,
            ));
        }
        value
    }
}

/// Decimal digits (with optional `_` separators) to `u64`; `None` on
/// overflow or any other character. Never panics.
fn parse_u64(text: &str) -> Option<u64> {
    let mut value: u64 = 0;
    let mut digits = 0usize;
    for c in text.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(10)?;
        value = value.checked_mul(10)?.checked_add(u64::from(d))?;
        digits += 1;
    }
    (digits > 0).then_some(value)
}

fn symbol(kind: TokenKind) -> &'static str {
    match kind {
        TokenKind::LParen => "(",
        TokenKind::RParen => ")",
        TokenKind::LBracket => "[",
        TokenKind::RBracket => "]",
        TokenKind::Comma => ",",
        TokenKind::Colon => ":",
        TokenKind::Assign => "=",
        TokenKind::Plus => "+",
        TokenKind::Minus => "-",
        TokenKind::Star => "*",
        TokenKind::Slash => "/",
        TokenKind::Caret => "^",
        TokenKind::At => "@",
        TokenKind::Inverse => "⁻¹",
        TokenKind::Partial => "∂",
        TokenKind::Nabla => "∇",
        TokenKind::Eq => "==",
        TokenKind::Ne => "!=",
        TokenKind::Lt => "<",
        TokenKind::Le => "<=",
        TokenKind::Gt => ">",
        TokenKind::Ge => ">=",
        TokenKind::Dot => ".",
        TokenKind::Arrow => "->",
        TokenKind::Keyword(kw) => kw.as_str(),
        TokenKind::Int => "integer",
        TokenKind::Float => "float",
        TokenKind::Ident => "name",
        TokenKind::Str => "string",
        TokenKind::Newline => "newline",
        TokenKind::Indent => "indent",
        TokenKind::Dedent => "dedent",
        TokenKind::Eof => "end of input",
    }
}

#[cfg(test)]
mod tests {
    use super::parse_u64;

    #[test]
    fn parse_u64_is_checked() {
        assert_eq!(parse_u64("0"), Some(0));
        assert_eq!(parse_u64("1_000"), Some(1000));
        assert_eq!(parse_u64("18446744073709551615"), Some(u64::MAX));
        assert_eq!(parse_u64("18446744073709551616"), None);
        assert_eq!(parse_u64("99999999999999999999"), None);
        assert_eq!(parse_u64(""), None);
        assert_eq!(parse_u64("_"), None);
        assert_eq!(parse_u64("1e3"), None);
        assert_eq!(parse_u64("٣"), None); // non-ASCII digit
    }
}
