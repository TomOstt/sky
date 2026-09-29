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

use crate::ast::{
    span_text, BinaryOp, Expr, ExprKind, Ident, Literal, Module, Stmt, StmtKind, UnaryOp, UnitExpr,
};
use crate::precedence::{infix_binding_power, is_postfix_start, prefix_binding_power, POSTFIX_BP};

/// Maximum nesting depth. Each level counts once: a `(`...`)` group, a prefix
/// operator, a call, an index, an attribute, a `⁻¹`, and each infix operator
/// folded into an operand. `x` uses 0, `(((x)))` 3, `f(g(x))` 2, `a^b^c` 2,
/// and `a + b + c` 2. The count bounds both recursion and AST height (which
/// keeps recursive `Drop` and printing safe).
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
            return self.fail_too_deep(span);
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

    /// Depth is counted where nesting happens (group, prefix, and each
    /// folded operator), not on entry here, so a level never counts twice.
    ///
    /// This and the other functions on the recursive path (`parse_operand`,
    /// `parse_paren`, `parse_delimited`) keep few locals: token checks and
    /// node building live in non-inlined helpers, so a nesting level costs
    /// little stack even in debug builds.
    fn expr_bp_inner(&mut self, min_bp: u8) -> Option<Expr> {
        let mut lhs = self.parse_operand()?;
        // A comparison already folded at this level: comparisons are
        // non-associative, so a second one is an error.
        let mut prev_comparison: Option<Span> = None;
        loop {
            lhs = match self.next_step(min_bp, &mut prev_comparison)? {
                Step::Done => return Some(lhs),
                Step::Infix { op, op_span, r_bp } => {
                    self.parse_infix_rhs(lhs, op, op_span, r_bp)?
                }
                Step::Inverse(op_span) => inverse(lhs, op_span),
                Step::Attribute(name) => attribute(lhs, name),
                Step::Call(open) => self.parse_call(lhs, open)?,
                Step::Index(open) => self.parse_index(lhs, open)?,
            };
        }
    }

    fn parse_infix_rhs(
        &mut self,
        lhs: Expr,
        op: BinaryOp,
        op_span: Span,
        r_bp: u8,
    ) -> Option<Expr> {
        let rhs = self.expr_bp(r_bp)?;
        Some(binary(op, op_span, lhs, rhs))
    }

    fn parse_call(&mut self, callee: Expr, open: Span) -> Option<Expr> {
        let (args, close) = self.parse_delimited(open, TokenKind::RParen)?;
        Some(call(callee, args, close))
    }

    fn parse_index(&mut self, base: Expr, open: Span) -> Option<Expr> {
        let (indices, close) = self.parse_delimited(open, TokenKind::RBracket)?;
        Some(index(base, indices, close))
    }

    /// Looks at the token after a complete operand and decides what, if
    /// anything, extends it. Consumes the operator (and an attribute name)
    /// and counts the nesting level.
    #[inline(never)]
    fn next_step(&mut self, min_bp: u8, prev_comparison: &mut Option<Span>) -> Option<Step> {
        let tok = self.current();
        if is_postfix_start(tok.kind) {
            if POSTFIX_BP < min_bp {
                return Some(Step::Done);
            }
            self.enter(tok.span)?;
            self.bump();
            return match tok.kind {
                TokenKind::Inverse => Some(Step::Inverse(tok.span)),
                TokenKind::LParen => Some(Step::Call(tok.span)),
                TokenKind::LBracket => Some(Step::Index(tok.span)),
                TokenKind::Dot => {
                    let name = self.current();
                    // Any keyword may be an attribute name (`linalg.solve`),
                    // signed off by Eng Lead. Only here: a bare keyword is
                    // still not a name.
                    if !matches!(name.kind, TokenKind::Ident | TokenKind::Keyword(_)) {
                        return self.fail_attribute_name(name, tok.span);
                    }
                    self.bump();
                    Some(Step::Attribute(Ident { span: name.span }))
                }
                // Unreachable by construction; report instead of panicking.
                _ => self.fail_expected_expression(tok),
            };
        }
        let Some((op, l_bp, r_bp)) = infix_binding_power(tok.kind) else {
            return Some(Step::Done);
        };
        if l_bp < min_bp {
            return Some(Step::Done);
        }
        self.bump();
        // `x**2`: two adjacent `*` tokens.
        let next = self.current();
        if op == BinaryOp::Mul && next.kind == TokenKind::Star && tok.span.end == next.span.start {
            return self.fail_double_star(tok.span.join(next.span));
        }
        if op.is_comparison() {
            if let Some(prev) = *prev_comparison {
                self.report_chained_comparison(tok.span, prev);
            }
            *prev_comparison = Some(tok.span);
        }
        self.enter(tok.span)?;
        Some(Step::Infix {
            op,
            op_span: tok.span,
            r_bp,
        })
    }

    /// A prefix operator applied to an operand, a group, or an atom.
    fn parse_operand(&mut self) -> Option<Expr> {
        let tok = self.current();
        if tok.kind == TokenKind::LParen {
            return self.parse_paren();
        }
        let Some((op, r_bp)) = prefix_binding_power(tok.kind) else {
            return self.parse_atom();
        };
        self.bump();
        self.enter(tok.span)?;
        let operand = self.expr_bp(r_bp)?;
        Some(unary(op, tok.span, operand))
    }

    /// A literal, unit literal, or name. Never recurses into `expr_bp`.
    #[inline(never)]
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
            _ => self.fail_expected_expression(tok),
        }
    }

    /// `( expr )` with the `(` current.
    fn parse_paren(&mut self) -> Option<Expr> {
        let open = self.bump().span;
        self.enter(open)?;
        let inner = self.expr_bp(0)?;
        if self.peek() != TokenKind::RParen {
            return self.fail_unclosed(&inner, open, "(", "`)`");
        }
        let close = self.bump().span;
        Some(paren(open, inner, close))
    }

    /// Comma-separated expressions after an opening delimiter (already
    /// consumed), up to and including `close`. A trailing comma is allowed.
    /// Calls (`close` = `)`) may be empty; indexing (`]`) may not.
    fn parse_delimited(&mut self, open: Span, close: TokenKind) -> Option<(Vec<Expr>, Span)> {
        let mut items = Vec::new();
        if self.peek() == close {
            let c = self.bump();
            if close == TokenKind::RBracket {
                return self.fail_empty_index(open.join(c.span));
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
            let (open_str, expected) = if close == TokenKind::RParen {
                ("(", "`,` or `)`")
            } else {
                ("[", "`,` or `]`")
            };
            return self.fail_unclosed(&item, open, open_str, expected);
        }
        let c = self.bump();
        Some((items, c.span))
    }

    // ----- error paths -----
    //
    // Kept out of line (`#[cold]`, `#[inline(never)]`) so their temporaries
    // don't enlarge the frames of the recursive functions above: that keeps
    // `MAX_DEPTH` levels well within a small stack, even in debug builds.
    // Each pushes one error and returns `None`.

    #[cold]
    #[inline(never)]
    fn fail_too_deep<T>(&mut self, span: Span) -> Option<T> {
        self.errors.push(ParseError::new(
            format!("expression is nested too deeply (the limit is {MAX_DEPTH} levels)"),
            span,
        ));
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_double_star<T>(&mut self, span: Span) -> Option<T> {
        self.errors.push(
            ParseError::new("`**` is not an operator", span).with_help("Sky writes powers as `^`"),
        );
        None
    }

    /// Recoverable: records the error and lets parsing continue.
    #[cold]
    #[inline(never)]
    fn report_chained_comparison(&mut self, at: Span, prev: Span) {
        self.errors.push(
            ParseError::new("comparison operators cannot be chained in Sky v0.1", at)
                .with_secondary(prev, "previous comparison")
                .with_help("compare two values at a time"),
        );
    }

    #[cold]
    #[inline(never)]
    fn fail_expected_expression<T>(&mut self, tok: Token) -> Option<T> {
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

    #[cold]
    #[inline(never)]
    fn fail_attribute_name<T>(&mut self, name: Token, dot: Span) -> Option<T> {
        self.errors.push(
            ParseError::new(
                format!(
                    "expected an attribute name after `.`, found {}",
                    self.describe(name)
                ),
                name.span,
            )
            .with_secondary(dot, "attribute access"),
        );
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_empty_index<T>(&mut self, span: Span) -> Option<T> {
        self.errors.push(ParseError::new(
            "expected at least one index between `[` and `]`",
            span,
        ));
        None
    }

    /// `last` is complete but the next token is not `close` (or `,` inside
    /// a call or index).
    #[cold]
    #[inline(never)]
    fn fail_unclosed<T>(
        &mut self,
        last: &Expr,
        open: Span,
        open_str: &str,
        expected: &str,
    ) -> Option<T> {
        let err = self
            .unexpected_after(last, expected)
            .with_secondary(open, format!("unclosed `{open_str}`"));
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
        self.reject_inverse_after_unit(&value, &unit, &unit)?;
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
            if is_mul && next.kind == TokenKind::Star {
                return self.fail_double_star(op.span.join(next.span));
            }
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
            let factor = (*rhs).clone();
            unit = if is_mul {
                UnitExpr::Mul { lhs, rhs, span }
            } else {
                UnitExpr::Div { lhs, rhs, span }
            };
            self.reject_inverse_after_unit(&value, &unit, &factor)?;
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

    /// `⁻¹` right after a unit (`10 km⁻¹`, `9.8 m/s⁻¹`) is an error: it would
    /// invert the whole quantity (0.1/km), while readers expect 10/km, and
    /// the dimension is the same either way, so the checker can't catch it.
    /// `unit` is the unit so far and `factor` its last factor.
    fn reject_inverse_after_unit(
        &mut self,
        value: &Literal,
        unit: &UnitExpr,
        factor: &UnitExpr,
    ) -> Option<()> {
        let tok = self.current();
        if tok.kind != TokenKind::Inverse {
            return Some(());
        }
        let number = self.text(value.span());
        let unit_text = self.text(unit.span());
        let help = match factor {
            UnitExpr::Name(name) if matches!(unit, UnitExpr::Name(_)) => format!(
                "write `{number} {unit_text}^-1` for per-{}, or `({number} {unit_text})⁻¹`",
                self.text(name.span)
            ),
            UnitExpr::Name(name) => format!(
                "write `{number} {unit_text}^-1` to invert `{}`, or `({number} {unit_text})⁻¹`",
                self.text(name.span)
            ),
            _ => format!(
                "put a single exponent after `^` (e.g. `s^-2`), or write `({number} {unit_text})⁻¹`"
            ),
        };
        self.errors.push(
            ParseError::new("`⁻¹` cannot follow a unit", tok.span)
                .with_secondary(factor.span(), "unit")
                .with_help(help),
        );
        None
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
        // `0 - magnitude` reaches i64::MIN, which `-(magnitude as i64)` can't.
        let value = if negative {
            0i64.checked_sub_unsigned(magnitude)
        } else {
            i64::try_from(magnitude).ok()
        };
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

/// What extends an operand; see `Parser::next_step`.
enum Step {
    Done,
    Infix {
        op: BinaryOp,
        op_span: Span,
        r_bp: u8,
    },
    Inverse(Span),
    Attribute(Ident),
    /// Holds the span of the opening `(` or `[`, already consumed.
    Call(Span),
    Index(Span),
}

// Node builders, out of line so the recursive parser frames stay small.

#[inline(never)]
fn binary(op: BinaryOp, op_span: Span, lhs: Expr, rhs: Expr) -> Expr {
    Expr {
        span: lhs.span.join(rhs.span),
        kind: ExprKind::Binary {
            op,
            op_span,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    }
}

#[inline(never)]
fn unary(op: UnaryOp, op_span: Span, operand: Expr) -> Expr {
    Expr {
        span: op_span.join(operand.span),
        kind: ExprKind::Unary {
            op,
            op_span,
            operand: Box::new(operand),
        },
    }
}

#[inline(never)]
fn paren(open: Span, inner: Expr, close: Span) -> Expr {
    Expr {
        span: open.join(close),
        kind: ExprKind::Paren(Box::new(inner)),
    }
}

#[inline(never)]
fn inverse(operand: Expr, op_span: Span) -> Expr {
    Expr {
        span: operand.span.join(op_span),
        kind: ExprKind::Inverse {
            operand: Box::new(operand),
            op_span,
        },
    }
}

#[inline(never)]
fn attribute(base: Expr, name: Ident) -> Expr {
    Expr {
        span: base.span.join(name.span),
        kind: ExprKind::Attribute {
            base: Box::new(base),
            name,
        },
    }
}

#[inline(never)]
fn call(callee: Expr, args: Vec<Expr>, close: Span) -> Expr {
    Expr {
        span: callee.span.join(close),
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args,
        },
    }
}

#[inline(never)]
fn index(base: Expr, indices: Vec<Expr>, close: Span) -> Expr {
    Expr {
        span: base.span.join(close),
        kind: ExprKind::Index {
            base: Box::new(base),
            indices,
        },
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
