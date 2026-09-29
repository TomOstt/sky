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

use std::borrow::Cow;

use sky_diag::{display_capped, Diagnostic, Label};
use sky_syntax::{Kw, Span, Token, TokenKind};

use crate::ast::{
    span_text, AssignTarget, BinaryOp, BoolOp, CompoundOp, DimExpr, Expr, ExprKind, Ident,
    IndexItem, Literal, Module, ShapeAnn, ShapeKind, Stmt, StmtKind, TypeExpr, UnaryOp, UnitExpr,
};
use crate::precedence::{
    bool_binding_power, infix_binding_power, is_postfix_start, prefix_binding_power, NOT_BP,
    POSTFIX_BP,
};

/// Maximum nesting depth. Each level counts once: a `(`...`)` group, a prefix
/// operator (including `not`), a call, an index (its slices included: the
/// bounds of `a[i:j]` sit one level below `a`), an attribute, a `⁻¹`, each
/// infix operator folded into an operand, and each `and`/`or` run (however
/// long: a run is one flat node). `x` uses 0, `(((x)))` 3, `f(g(x))` 2,
/// `a^b^c` 2, `a + b + c` 2, `not not x` 2, and `a and b and c` 1. The count
/// bounds both recursion and AST height (which keeps recursive `Drop` and
/// printing safe).
pub const MAX_DEPTH: u32 = 256;

/// Maximum number of unit names in one unit literal (`kg*m/s^2` has 3).
pub const MAX_UNIT_FACTORS: usize = 64;

/// Source text quoted in a diagnostic is cut to at most this many
/// characters, or [`MAX_QUOTED_BYTES`] bytes after escaping, whichever comes
/// first (then `…`), so a huge name or expression can't make a huge message.
pub const MAX_QUOTED_CHARS: usize = sky_diag::DISPLAY_CAP_CHARS;

/// The byte cap on quoted source text, counted after escaping (see
/// [`MAX_QUOTED_CHARS`]).
pub const MAX_QUOTED_BYTES: usize = sky_diag::DISPLAY_CAP_BYTES;

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
            let err = p.unexpected_after(e.span, "end of input");
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

    /// Source text for a diagnostic: see [`display_capped`].
    fn text(&self, span: Span) -> Cow<'s, str> {
        display_capped(span_text(self.src, span).unwrap_or("?"))
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
        // The statement's own `=` or compound operator, for chain errors.
        let mut first_op = None;
        let stmt = match (tok.kind, compound_op(tok.kind)) {
            (TokenKind::Colon, _) => {
                self.bump();
                let name = self.binding_target(
                    &first,
                    tok.span,
                    "annotate",
                    "invalid annotation target: only a plain name can have a type annotation",
                )?;
                let ty = self.parse_type()?;
                let value = if self.peek() == TokenKind::Assign {
                    first_op = Some(self.current());
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
            (TokenKind::Assign, _) => {
                first_op = Some(tok);
                self.bump();
                let start = first.span;
                let target = self.assign_target(first, tok.span)?;
                let value = self.expr_bp(0)?;
                Stmt {
                    span: start.join(value.span),
                    kind: StmtKind::Assign { target, value },
                }
            }
            (_, Some(op)) => {
                first_op = Some(tok);
                self.bump();
                let start = first.span;
                let target = self.assign_target(first, tok.span)?;
                let value = self.expr_bp(0)?;
                Stmt {
                    span: start.join(value.span),
                    kind: StmtKind::CompoundAssign {
                        target,
                        op,
                        op_span: tok.span,
                        value,
                    },
                }
            }
            _ => Stmt {
                span: first.span,
                kind: StmtKind::Expr(first),
            },
        };
        self.expect_stmt_end(&stmt, first_op)?;
        Some(stmt)
    }

    /// The target of `=` or of a compound assignment (`op` is its span): a
    /// name, an index, or an attribute. Anything else is an error on the
    /// target's span.
    fn assign_target(&mut self, target: Expr, op: Span) -> Option<AssignTarget> {
        let Expr { kind, span } = target;
        match kind {
            ExprKind::Name(id) => Some(AssignTarget::Name(id)),
            ExprKind::Index { base, items } => Some(AssignTarget::Index {
                bracket_span: self.bracket_span(base.span, span),
                base,
                items,
                span,
            }),
            ExprKind::Attribute { base, name } => {
                Some(AssignTarget::Attribute { base, name, span })
            }
            other => self.fail_assign_target(&other, span, op),
        }
    }

    /// `[` through `]` of an index expression spanning `whole`, whose base
    /// ends at `base.end`: the `[` is the first token after the base.
    fn bracket_span(&self, base: Span, whole: Span) -> Span {
        let i = self.tokens.partition_point(|t| t.span.start < base.end);
        let open = match self.tokens.get(i) {
            Some(t) if t.kind == TokenKind::LBracket => t.span.start,
            _ => base.end,
        };
        Span {
            start: open.min(whole.end),
            end: whole.end,
        }
    }

    /// `verb` names the action for a Bool literal target ("cannot annotate
    /// `true`"); every other non-name target gets `message`.
    fn binding_target(
        &mut self,
        target: &Expr,
        op: Span,
        verb: &str,
        message: &str,
    ) -> Option<Ident> {
        let err = match &target.kind {
            ExprKind::Name(id) => return Some(*id),
            ExprKind::Literal(Literal::Bool { value, span }) => {
                ParseError::new(format!("cannot {verb} `{value}`"), *span)
                    .with_secondary(op, "because of this")
                    .with_help("`true` and `false` are Bool literals, not names")
            }
            _ => ParseError::new(message, target.span).with_secondary(op, "because of this"),
        };
        self.errors.push(err);
        None
    }

    /// `first_op` is the statement's own `=` or compound operator, if any.
    fn expect_stmt_end(&mut self, stmt: &Stmt, first_op: Option<Token>) -> Option<()> {
        let tok = self.current();
        match tok.kind {
            TokenKind::Newline => {
                self.bump();
                Some(())
            }
            TokenKind::Eof => Some(()),
            kind if kind == TokenKind::Assign || compound_op(kind).is_some() => {
                self.fail_assign_after(stmt, first_op, tok)
            }
            _ => {
                let last = match &stmt.kind {
                    StmtKind::Expr(e) => e.span,
                    StmtKind::Assign { value, .. } | StmtKind::CompoundAssign { value, .. } => {
                        value.span
                    }
                    StmtKind::Binding { ty, value, .. } => {
                        value.as_ref().map_or(ty.span, |v| v.span)
                    }
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
        let mut lhs = self.parse_operand(min_bp)?;
        // A comparison already folded at this level: comparisons are
        // non-associative, so a second one is an error.
        let mut prev_comparison: Option<Span> = None;
        loop {
            lhs = match self.next_step(min_bp, &mut prev_comparison)? {
                Step::Done => return Some(lhs),
                Step::Infix { op, op_span, r_bp } => {
                    self.parse_infix_rhs(lhs, op, op_span, r_bp)?
                }
                Step::Bool { op, op_span, r_bp } => {
                    // A comparison can't follow a run at this level (the
                    // run's operands took it), so start afresh.
                    prev_comparison = None;
                    self.parse_bool_run(lhs, op, op_span, r_bp)?
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

    /// `lhs op x op y ...` for one `and`/`or`, with the first `op` consumed.
    /// A loop, not a recursion per operand: the run becomes one flat node,
    /// so a 10k-operand chain uses one nesting level.
    fn parse_bool_run(&mut self, lhs: Expr, op: BoolOp, first_op: Span, r_bp: u8) -> Option<Expr> {
        let mut operands = vec![lhs];
        let mut op_spans = vec![first_op];
        loop {
            operands.push(self.expr_bp(r_bp)?);
            let tok = self.current();
            match bool_binding_power(tok.kind) {
                Some((next, _, _)) if next == op => {
                    self.bump();
                    op_spans.push(tok.span);
                }
                _ => break,
            }
        }
        Some(bool_op(op, operands, op_spans))
    }

    fn parse_call(&mut self, callee: Expr, open: Span) -> Option<Expr> {
        let (args, close) = self.parse_delimited(open, TokenKind::RParen)?;
        Some(call(callee, args, close))
    }

    fn parse_index(&mut self, base: Expr, open: Span) -> Option<Expr> {
        let (items, close) = self.parse_index_items(open)?;
        Some(index(base, items, close))
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
        if let Some((op, l_bp, r_bp)) = bool_binding_power(tok.kind) {
            if l_bp < min_bp {
                return Some(Step::Done);
            }
            self.bump();
            self.enter(tok.span)?;
            return Some(Step::Bool {
                op,
                op_span: tok.span,
                r_bp,
            });
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
        if op == BinaryOp::Mul && tok.span.end == next.span.start {
            match next.kind {
                TokenKind::Star => return self.fail_double_star(tok.span.join(next.span)),
                TokenKind::StarAssign => {
                    return self.fail_double_star_assign(tok.span.join(next.span))
                }
                _ => {}
            }
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
    /// `min_bp` is what the enclosing operator requires: `not` binds more
    /// loosely than every operator except `and`/`or`, so `-not x` and
    /// `a < not b` are errors.
    fn parse_operand(&mut self, min_bp: u8) -> Option<Expr> {
        let tok = self.current();
        if tok.kind == TokenKind::LParen {
            return self.parse_paren();
        }
        let Some((op, r_bp)) = prefix_binding_power(tok.kind) else {
            return self.parse_atom();
        };
        if op == UnaryOp::Not && min_bp > NOT_BP {
            return self.fail_not_operand();
        }
        self.bump();
        self.enter(tok.span)?;
        let operand = self.expr_bp(r_bp)?;
        Some(unary(op, tok.span, operand))
    }

    /// A literal (including `true`/`false`), unit literal, or name. Never
    /// recurses into `expr_bp`.
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
            // No unit check here: a unit may only follow a number, so
            // `true m` is "unexpected name after an expression".
            TokenKind::Keyword(kw @ (Kw::True | Kw::False)) => {
                self.bump();
                Some(literal(Literal::Bool {
                    value: kw == Kw::True,
                    span: tok.span,
                }))
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
            return self.fail_unclosed(inner.span, open, "(", "`)`");
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
            return self.fail_unclosed(item.span, open, open_str, expected);
        }
        let c = self.bump();
        Some((items, c.span))
    }

    /// The items of `base[...]` after the `[` (already consumed), up to and
    /// including `]`: expressions or slices, comma-separated, at least one,
    /// trailing comma allowed. Slice bounds are parsed at the index's own
    /// nesting level, so a slice adds no level.
    ///
    /// On the recursive path, so items are built and pushed by out-of-line
    /// helpers: no `IndexItem` temporary lives in this frame.
    fn parse_index_items(&mut self, open: Span) -> Option<(Vec<IndexItem>, Span)> {
        let mut items = Vec::new();
        if self.peek() == TokenKind::RBracket {
            let c = self.bump();
            return self.fail_empty_index(open.join(c.span));
        }
        loop {
            // One item: a `start` expression, then for a slice its `:` and an
            // optional `stop`. There is a single `expr_bp` call site, so the
            // frame holds one expression slot however the item is spelled.
            let mut colon: Option<Span> = None;
            let mut has_start = false;
            loop {
                if !matches!(
                    self.peek(),
                    TokenKind::Comma | TokenKind::RBracket | TokenKind::Colon
                ) {
                    let bound = self.expr_bp(0)?;
                    push_index_bound(&mut items, bound, colon.is_some());
                    has_start = colon.is_none();
                } else if colon.is_none() && self.peek() != TokenKind::Colon {
                    return self.fail_expected_expression(self.current());
                }
                if colon.is_some() || self.peek() != TokenKind::Colon {
                    break;
                }
                colon = Some(self.begin_slice(&mut items, has_start));
            }
            if let Some(colon) = colon {
                if self.peek() == TokenKind::Colon {
                    return self.fail_slice_step(colon);
                }
            }
            match self.peek() {
                TokenKind::Comma => {
                    self.bump();
                    if self.peek() == TokenKind::RBracket {
                        break;
                    }
                }
                TokenKind::RBracket => break,
                _ => return self.fail_index_unclosed(&items, open),
            }
        }
        let c = self.bump();
        Some((items, c.span))
    }

    /// Consumes a slice's `:` (current) and turns the item being parsed
    /// into a slice: the expression just pushed becomes its `start` if
    /// `has_start`. Returns the `:` span.
    #[inline(never)]
    fn begin_slice(&mut self, items: &mut Vec<IndexItem>, has_start: bool) -> Span {
        let colon = self.bump().span;
        let start = if has_start {
            match items.pop() {
                Some(IndexItem::Expr(e)) => Some(e),
                Some(other) => {
                    items.push(other);
                    None
                }
                None => None,
            }
        } else {
            None
        };
        let span = start.as_ref().map_or(colon, |e| e.span.join(colon));
        items.push(IndexItem::Slice {
            start,
            stop: None,
            colon_span: colon,
            span,
        });
        colon
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

    /// `x **= 2`: `*` directly followed by `*=`.
    #[cold]
    #[inline(never)]
    fn fail_double_star_assign<T>(&mut self, span: Span) -> Option<T> {
        self.errors.push(
            ParseError::new("`**=` is not an operator", span)
                .with_help("Sky writes powers as `^`; the compound assignment is `^=`"),
        );
        None
    }

    /// The token after an index item is not `,` or `]`.
    #[cold]
    #[inline(never)]
    fn fail_index_unclosed<T>(&mut self, items: &[IndexItem], open: Span) -> Option<T> {
        let last = items.last().map_or(open, IndexItem::span);
        self.fail_unclosed(last, open, "[", "`,` or `]`")
    }

    /// A second `:` in a slice (current), after the slice's first `:`.
    #[cold]
    #[inline(never)]
    fn fail_slice_step<T>(&mut self, first_colon: Span) -> Option<T> {
        let tok = self.current();
        self.errors.push(
            ParseError::new(
                "slice steps (`start:stop:step`) are not supported in Sky v0.1",
                tok.span,
            )
            .with_secondary(first_colon, "slice"),
        );
        None
    }

    /// An assignment target that is not a name, index, or attribute.
    #[cold]
    #[inline(never)]
    fn fail_assign_target<T>(&mut self, kind: &ExprKind, span: Span, op: Span) -> Option<T> {
        const HELP: &str =
            "only a name (`x`), an index (`A[i]`), or an attribute (`a.b`) can be assigned to";
        let err = match kind {
            ExprKind::Literal(Literal::Bool { value, span }) => {
                ParseError::new(format!("cannot assign to `{value}`"), *span)
                    .with_secondary(op, "because of this")
                    .with_help("`true` and `false` are Bool literals, not names")
            }
            _ => {
                let what = match kind {
                    ExprKind::Literal(_) => "a literal",
                    ExprKind::UnitLiteral { .. } => "a unit literal",
                    ExprKind::Call { .. } => "a function call",
                    ExprKind::Paren(_) => "a parenthesized expression",
                    _ => "an operator expression",
                };
                let help = if matches!(kind, ExprKind::Paren(_)) {
                    format!("remove the parentheses; {HELP}")
                } else {
                    HELP.to_string()
                };
                ParseError::new(
                    format!("cannot assign to {what} `{}`", self.text(span)),
                    span,
                )
                .with_secondary(op, "because of this")
                .with_help(help)
            }
        };
        self.errors.push(err);
        None
    }

    /// `=` or a compound assignment operator (`tok`, current) after a
    /// complete statement whose own operator is `first_op`.
    #[cold]
    #[inline(never)]
    fn fail_assign_after(
        &mut self,
        stmt: &Stmt,
        first_op: Option<Token>,
        tok: Token,
    ) -> Option<()> {
        let op = symbol(tok.kind);
        let err = match (&stmt.kind, first_op) {
            // `x: T += 1`: an annotation without a value.
            (
                StmtKind::Binding {
                    name, value: None, ..
                },
                _,
            ) => ParseError::new(format!("`{op}` cannot follow a type annotation"), tok.span)
                .with_help(format!(
                    "annotate with `{name}: T = ...`, then write `{name} {op} ...` on its own line",
                    name = self.text(name.span)
                )),
            (_, Some(first)) => ParseError::new(
                format!(
                    "chained assignment (`{}` then `{op}`) is not supported",
                    symbol(first.kind)
                ),
                tok.span,
            )
            .with_secondary(first.span, "first assignment"),
            // Unreachable: only an assignment or binding can be followed
            // by an assignment operator here.
            (_, None) => ParseError::new(format!("unexpected `{op}`"), tok.span),
        };
        self.errors.push(err);
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

    /// `not` (current) where a tighter operator needs its operand, as in
    /// `-not x`. Parses the `not` expression anyway, to show it in the help.
    #[cold]
    #[inline(never)]
    fn fail_not_operand<T>(&mut self) -> Option<T> {
        let before = self.prev_span();
        let not = self.bump();
        self.enter(not.span)?;
        let operand = self.expr_bp(NOT_BP);
        let before_text = match before {
            Some(span) => format!("`{}`", self.text(span)),
            None => "the operator before it".to_string(),
        };
        let mut err = ParseError::new(
            format!("`not` binds more loosely than {before_text}, so it needs parentheses here"),
            not.span,
        );
        if let Some(span) = before {
            err = err.with_secondary(span, "this operator binds more tightly than `not`");
        }
        let written = operand
            .as_ref()
            .map(|e| not.span.join(e.span))
            .filter(|span| span.len() <= 40)
            .and_then(|span| span_text(self.src, span))
            .filter(|text| !text.contains('\n'));
        let help = match written {
            Some(text) => format!("write `({text})`"),
            None => "put the `not` expression in parentheses".to_string(),
        };
        self.errors.push(err.with_help(help));
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_expected_expression<T>(&mut self, tok: Token) -> Option<T> {
        let message = match tok.kind {
            TokenKind::Keyword(kw @ (Kw::And | Kw::Or)) => format!(
                "expected an expression, found keyword `{}`",
                kw.as_str()
            ),
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
        last: Span,
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
    /// expression spanning `prev`, where `expected` was wanted.
    fn unexpected_after(&self, prev: Span, expected: &str) -> ParseError {
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
                .with_secondary(prev, "this is not a numeric literal")
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
            .with_secondary(prev, "previous expression")
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
        let unit = self.parse_unit(UnitCtx::Literal(&value))?;
        Some(Expr {
            span: value.span().join(unit.span()),
            kind: ExprKind::UnitLiteral { value, unit },
        })
    }

    /// The unit itself, with the `IDENT` of its first factor current. Shared
    /// by unit literals (`10 km/h`) and type annotations (`v: Vector[3] m/s`)
    /// so both accept exactly the same spellings; `ctx` only picks the
    /// wording of the errors, which name the number in a literal and have no
    /// quantity to mention in an annotation.
    fn parse_unit(&mut self, ctx: UnitCtx<'_>) -> Option<UnitExpr> {
        let mut unit = self.parse_unit_factor()?;
        self.reject_inverse_after_unit(ctx, &unit)?;
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
                // In a literal a spaced operator is ordinary arithmetic on
                // the quantity. A type has no quantity, so `m / s` can only
                // be a unit written with spaces.
                if ctx.is_annotation() && next.kind == TokenKind::Ident {
                    return self.fail_spaced_unit_op(&unit, op, next);
                }
                break;
            }
            self.bump();
            if is_mul && next.kind == TokenKind::Star {
                return self.fail_double_star(op.span.join(next.span));
            }
            if is_mul && next.kind == TokenKind::StarAssign {
                return self.fail_double_star_assign(op.span.join(next.span));
            }
            if next.kind != TokenKind::Ident {
                let verb = if is_mul { "multiply" } else { "divide" };
                let (what, help) = match ctx {
                    UnitCtx::Literal(_) => (
                        "unit literal",
                        format!(
                            "to {verb} the quantity, put spaces around `{}`",
                            symbol(op.kind)
                        ),
                    ),
                    UnitCtx::Annotation => (
                        "unit",
                        format!(
                            "write a unit name after `{}`, e.g. `m/s` or `kg*m`",
                            symbol(op.kind)
                        ),
                    ),
                };
                self.errors.push(
                    ParseError::new(
                        format!(
                            "expected a unit name after `{}` in a {what}, found {}",
                            symbol(op.kind),
                            self.describe(next)
                        ),
                        next.span,
                    )
                    .with_secondary(op.span, "no spaces around this, so it is part of the unit")
                    .with_help(help),
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
            self.reject_inverse_after_unit(ctx, &unit)?;
        }
        // `5 kg m`: compound units need `*`, never juxtaposition.
        let tok = self.current();
        if tok.kind == TokenKind::Ident {
            let unit_text = self.text(unit.span());
            let name = self.text(tok.span);
            let mut err = ParseError::new(
                format!("unexpected name `{name}` after the unit `{unit_text}`"),
                tok.span,
            )
            .with_secondary(unit.span(), "unit");
            // `x: symmetric Matrix[3, 3]`: a lowercase tag reads as a unit.
            let shape_word = span_text(self.src, tok.span).and_then(ShapeKind::from_name);
            err = match (ctx, shape_word) {
                (UnitCtx::Annotation, Some(kind)) => err.with_help(format!(
                    "matrix properties are capitalized and go before the shape: `PositiveDefinite {}[…]`",
                    kind.name()
                )),
                _ => err.with_help(format!(
                    "compound units are written with `*` and no spaces: `{unit_text}*{name}`"
                )),
            };
            self.errors.push(err);
            return None;
        }
        self.reject_spaced_caret_after_unit(ctx, &unit)?;
        Some(unit)
    }

    // ----- type annotations -----
    //
    // Grammar (spec/v0.1.md, "Type annotations"; `unit` is the unit-literal
    // grammar above, so its `*`, `/`, `^` touch their operands):
    //   type  := tag* shape [unit] | tag* unit | "Scalar"
    //   tag   := NAME
    //   shape := ("Vector" | "Matrix" | "Tensor") "[" dim ("," dim)* [","] "]"
    //   dim   := INT | NAME
    //
    // A name starts a unit unless it is a shape word or a tag. A tag is a
    // capitalized name followed by another name (`PositiveDefinite Matrix`),
    // which the checker validates: the parser has no tag table, so
    // `Matrix[3,3] PositiveDefinite` (a tag after the shape) parses as a
    // unit and the checker explains it. A lone name is always a unit
    // (`x: N` is newtons), never a size variable: those only appear in
    // `[...]`. The whole type is parsed in a loop with no recursion, so it
    // adds no nesting depth.

    /// Parses the type after `:` in `name: type [= value]`. This is the one
    /// place type annotations are parsed, so `def` parameters and
    /// `-> return` types can call it too. On error it has recorded a
    /// diagnostic and returns `None`, leaving the parser mid-line for the
    /// caller's usual recovery.
    fn parse_type(&mut self) -> Option<TypeExpr> {
        let start = self.current().span;
        let mut tags = Vec::new();
        loop {
            let tok = self.current();
            if tok.kind != TokenKind::Ident {
                return self.fail_not_a_type(tok);
            }
            let word = span_text(self.src, tok.span).unwrap_or("");
            let next = self.nth(1).map(|t| t.kind);
            if let Some(kind) = ShapeKind::from_name(word) {
                return self.parse_shape_type(tags, kind, start);
            }
            if next == Some(TokenKind::LBracket) {
                return self.fail_unknown_shape(tok);
            }
            if word.chars().next().is_some_and(char::is_uppercase) && next == Some(TokenKind::Ident)
            {
                tags.push(Ident { span: tok.span });
                self.bump();
                continue;
            }
            let unit = self.parse_unit(UnitCtx::Annotation)?;
            return Some(TypeExpr {
                span: start.join(unit.span()),
                tags,
                shape: None,
                unit: Some(unit),
            });
        }
    }

    /// The rest of a type once its shape word (`Matrix`, ...) is current.
    fn parse_shape_type(
        &mut self,
        tags: Vec<Ident>,
        kind: ShapeKind,
        start: Span,
    ) -> Option<TypeExpr> {
        let word = self.bump();
        let mut dims = Vec::new();
        let mut end = word.span;
        if kind == ShapeKind::Scalar {
            let tok = self.current();
            if tok.kind == TokenKind::LBracket {
                return self.fail_scalar_dims(tok.span);
            }
        } else {
            let open = self.current();
            if open.kind != TokenKind::LBracket {
                return self.fail_shape_needs_dims(kind, word.span);
            }
            self.bump();
            end = self.parse_shape_dims(open.span, &mut dims)?;
        }
        let shape = ShapeAnn {
            kind,
            kind_span: word.span,
            dims,
            span: word.span.join(end),
        };
        let mut span = start.join(shape.span);
        let mut unit = None;
        if self.peek() == TokenKind::Ident {
            if kind == ShapeKind::Scalar {
                return self.fail_scalar_unit(shape.span);
            }
            let u = self.parse_unit(UnitCtx::Annotation)?;
            span = span.join(u.span());
            unit = Some(u);
        }
        Some(TypeExpr {
            tags,
            shape: Some(shape),
            unit,
            span,
        })
    }

    /// `dim ("," dim)* [","] "]"` after a shape's `[` (`open`). Returns the
    /// span of the `]`.
    fn parse_shape_dims(&mut self, open: Span, dims: &mut Vec<DimExpr>) -> Option<Span> {
        loop {
            let tok = self.current();
            let dim = match tok.kind {
                TokenKind::Ident => DimExpr::Name(Ident { span: tok.span }),
                TokenKind::Int => DimExpr::Int(tok.span),
                TokenKind::RBracket if dims.is_empty() => {
                    return self.fail_dim(tok, open, "a shape needs at least one dimension");
                }
                TokenKind::Colon => {
                    return self.fail_dim(tok, open, "a shape dimension cannot be a slice");
                }
                _ => {
                    let msg = format!(
                        "a shape dimension must be a non-negative integer or a name, found {}",
                        self.describe(tok)
                    );
                    return self.fail_dim(tok, open, &msg);
                }
            };
            self.bump();
            dims.push(dim);
            let next = self.current();
            match next.kind {
                TokenKind::Comma => {
                    self.bump();
                    if self.peek() == TokenKind::RBracket {
                        return Some(self.bump().span);
                    }
                }
                TokenKind::RBracket => return Some(self.bump().span),
                TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Caret => return self.fail_shape_arithmetic(dim.span(), next),
                _ => {
                    let err = ParseError::new(
                        format!(
                            "expected `,` or `]` after a shape dimension, found {}",
                            self.describe(next)
                        ),
                        next.span,
                    )
                    .with_secondary(open, "in this shape")
                    .with_help("separate dimensions with commas: `Matrix[3, 3]`");
                    self.errors.push(err);
                    return None;
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn fail_dim<T>(&mut self, tok: Token, open: Span, message: &str) -> Option<T> {
        self.errors.push(
            ParseError::new(message, tok.span)
                .with_secondary(open, "in this shape")
                .with_help("write dimensions as integers or size-variable names: `Matrix[N, 3]`"),
        );
        None
    }

    /// `Matrix[N+1, N]`: the operator's operand, if any, is part of the span.
    #[cold]
    #[inline(never)]
    fn fail_shape_arithmetic<T>(&mut self, dim: Span, op: Token) -> Option<T> {
        let mut span = dim.join(op.span);
        if let Some(rhs) = self.nth(1) {
            if matches!(rhs.kind, TokenKind::Ident | TokenKind::Int) {
                span = span.join(rhs.span);
            }
        }
        let text = self.text(span);
        self.errors.push(
            ParseError::new(
                format!("shape arithmetic like `{text}` is not supported in v0.1"),
                span,
            )
            .with_help(
                "v0.1 shapes are equality constraints only: use an integer or a size-variable name",
            ),
        );
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_shape_needs_dims<T>(&mut self, kind: ShapeKind, word: Span) -> Option<T> {
        let name = kind.name();
        let example = match kind {
            ShapeKind::Vector => "Vector[3]",
            ShapeKind::Matrix => "Matrix[3, 3]",
            _ => "Tensor[2, 3, 4]",
        };
        self.errors.push(
            ParseError::new(format!("`{name}` needs dimensions in brackets"), word)
                .with_help(format!("write it like `{example}`")),
        );
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_scalar_dims<T>(&mut self, open: Span) -> Option<T> {
        self.errors.push(
            ParseError::new("`Scalar` takes no dimensions", open).with_help(
                "a scalar is just `Scalar`; for an array write `Vector[n]` or `Matrix[m, n]`",
            ),
        );
        None
    }

    /// `x: Scalar m`: a unit alone already means a scalar of that unit.
    #[cold]
    #[inline(never)]
    fn fail_scalar_unit<T>(&mut self, shape: Span) -> Option<T> {
        let tok = self.current();
        let name = self.text(tok.span);
        self.errors.push(
            ParseError::new("a unit cannot follow `Scalar`", tok.span)
                .with_secondary(shape, "`Scalar` is dimensionless")
                .with_help(format!(
                    "a scalar with a unit is written as the unit alone: `{name}`"
                )),
        );
        None
    }

    /// `List[int]`, `Foo[3]`: only three words take dimensions.
    #[cold]
    #[inline(never)]
    fn fail_unknown_shape<T>(&mut self, tok: Token) -> Option<T> {
        let name = self.text(tok.span);
        self.errors.push(
            ParseError::new(format!("unknown shape `{name}`"), tok.span).with_help(
                "shapes are `Vector[n]`, `Matrix[m, n]`, and `Tensor[d1, d2, ...]`; a bare unit like `m/s` is a scalar",
            ),
        );
        None
    }

    /// The start of a type is not a name: a value where a type belongs
    /// (`x: 5 m`), or nothing (`x:`).
    #[cold]
    #[inline(never)]
    fn fail_not_a_type<T>(&mut self, tok: Token) -> Option<T> {
        let is_value = matches!(
            tok.kind,
            TokenKind::Int
                | TokenKind::Float
                | TokenKind::Str
                | TokenKind::Keyword(Kw::True | Kw::False)
        );
        let err = if is_value {
            let unit = match (tok.kind, self.nth(1)) {
                (TokenKind::Int | TokenKind::Float, Some(u)) if u.kind == TokenKind::Ident => {
                    Some(self.text(u.span))
                }
                _ => None,
            };
            let help = match unit {
                Some(u) => format!(
                    "declare a unit by itself (`{u}`), or give the name a value with `=`"
                ),
                None => "a type is a unit like `m/s` or a shape like `Matrix[3, 3]`; give the name a value with `=`".to_string(),
            };
            ParseError::new("an annotation is a type, not a value", tok.span).with_help(help)
        } else {
            ParseError::new(
                format!("expected a type, found {}", self.describe(tok)),
                tok.span,
            )
            .with_help("a type is a unit like `m/s`, a shape like `Matrix[3, 3]`, or both: `Vector[3] m/s`")
        };
        self.errors.push(err);
        None
    }

    /// `x: m / s` (or `m /s`): the operator of a unit has no spaces around it.
    #[cold]
    #[inline(never)]
    fn fail_spaced_unit_op<T>(&mut self, unit: &UnitExpr, op: Token, next: Token) -> Option<T> {
        let sym = symbol(op.kind);
        let unit_text = self.text(unit.span());
        let name = self.text(next.span);
        self.errors.push(
            ParseError::new(
                format!("a unit's `{sym}` cannot have spaces around it"),
                op.span,
            )
            .with_secondary(unit.span(), "unit so far")
            .with_help(format!(
                "write the unit without spaces: `{unit_text}{sym}{name}`"
            )),
        );
        None
    }

    /// A `^` right after a unit that doesn't touch both sides (`10 m ^2`,
    /// `10 m^ 2`, `10 m ^ 2`) is an error. The spacing is the only thing
    /// deciding between a unit power (`10 m^2`, 10 m²) and a power of the
    /// quantity (`(10 m)^2`, 100 m²), readers can't see which was meant, and
    /// the dimension is the same either way, so the checker can't catch it.
    /// A `^` that touches both sides was already read as part of the unit.
    fn reject_spaced_caret_after_unit(&mut self, ctx: UnitCtx<'_>, unit: &UnitExpr) -> Option<()> {
        let tok = self.current();
        if tok.kind != TokenKind::Caret {
            return Some(());
        }
        let unit_text = self.text(unit.span());
        let UnitCtx::Literal(value) = ctx else {
            // A type has no quantity to raise, so there is no ambiguity, only
            // a spelling to fix.
            self.errors.push(
                ParseError::new("a unit's `^` cannot have spaces around it", tok.span)
                    .with_secondary(unit.span(), "unit")
                    .with_help(format!(
                        "write the exponent with no spaces: `{unit_text}^2`, `{unit_text}^-1`, or `{unit_text}^(1/2)`"
                    )),
            );
            return None;
        };
        let number = self.text(value.span());
        // The exponent as written, when it is a plain (possibly negative) integer.
        let exponent = match (self.nth(1), self.nth(2)) {
            (Some(t), _) if t.kind == TokenKind::Int => Some(self.text(t.span).to_string()),
            (Some(m), Some(t)) if m.kind == TokenKind::Minus && t.kind == TokenKind::Int => {
                Some(format!("-{}", self.text(t.span)))
            }
            _ => None,
        };
        let help = match exponent {
            Some(e) if matches!(unit, UnitExpr::Name(_)) => format!(
                "write `{number} {unit_text}^{e}` for a power of the unit, or `({number} {unit_text})^{e}` to raise the whole quantity"
            ),
            Some(e) => format!(
                "write the exponent with no spaces for a power of a unit, or `({number} {unit_text})^{e}` to raise the whole quantity"
            ),
            None => format!(
                "to raise the whole quantity, write `({number} {unit_text})^…`"
            ),
        };
        self.errors.push(
            ParseError::new("ambiguous `^` after a unit", tok.span)
                .with_secondary(unit.span(), "unit")
                .with_help(help),
        );
        None
    }

    /// `⁻¹` right after a unit (`10 km⁻¹`, `9.8 m/s⁻¹`) is an error: it would
    /// invert the whole quantity (0.1/km), while readers expect 10/km, and
    /// the dimension is the same either way, so the checker can't catch it.
    /// `unit` is the unit so far; its last factor is the right-hand side of
    /// the last `*` or `/`, or the whole unit if there is none.
    fn reject_inverse_after_unit(&mut self, ctx: UnitCtx<'_>, unit: &UnitExpr) -> Option<()> {
        let tok = self.current();
        if tok.kind != TokenKind::Inverse {
            return Some(());
        }
        let factor = match unit {
            UnitExpr::Mul { rhs, .. } | UnitExpr::Div { rhs, .. } => &**rhs,
            _ => unit,
        };
        let UnitCtx::Literal(value) = ctx else {
            // A type has no quantity to invert: the only fix is `^-1`.
            let name = self.text(factor.span());
            self.errors.push(
                ParseError::new("`⁻¹` cannot follow a unit", tok.span)
                    .with_secondary(factor.span(), "unit")
                    .with_help(format!(
                        "write a negative exponent instead: `{name}^-1` (no spaces)"
                    )),
            );
            return None;
        };
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

/// Where a unit is being parsed: after a number (`10 km`), or in a type
/// annotation (`m/s`). See `Parser::parse_unit`.
#[derive(Clone, Copy)]
enum UnitCtx<'a> {
    Literal(&'a Literal),
    Annotation,
}

impl UnitCtx<'_> {
    fn is_annotation(self) -> bool {
        matches!(self, UnitCtx::Annotation)
    }
}

/// What extends an operand; see `Parser::next_step`.
enum Step {
    Done,
    /// The first `and`/`or` of a run, already consumed.
    Bool {
        op: BoolOp,
        op_span: Span,
        r_bp: u8,
    },
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
fn bool_op(op: BoolOp, operands: Vec<Expr>, op_spans: Vec<Span>) -> Expr {
    // A run always has two or more operands; fall back to the first
    // keyword's span rather than panicking.
    let span = match (operands.first(), operands.last()) {
        (Some(a), Some(b)) => a.span.join(b.span),
        _ => op_spans.first().copied().unwrap_or(Span::point(0)),
    };
    Expr {
        span,
        kind: ExprKind::BoolOp {
            op,
            operands,
            op_spans,
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
fn index(base: Expr, items: Vec<IndexItem>, close: Span) -> Expr {
    Expr {
        span: base.span.join(close),
        kind: ExprKind::Index {
            base: Box::new(base),
            items,
        },
    }
}

/// Pushes an index item's expression: a new `IndexItem::Expr`, or, if
/// `is_stop`, the stop of the slice pushed last by `begin_slice`.
#[inline(never)]
fn push_index_bound(items: &mut Vec<IndexItem>, e: Expr, is_stop: bool) {
    if is_stop {
        if let Some(IndexItem::Slice { stop, span, .. }) = items.last_mut() {
            *span = span.join(e.span);
            *stop = Some(e);
            return;
        }
    }
    items.push(IndexItem::Expr(e));
}

/// The compound assignment a token spells, if any.
fn compound_op(kind: TokenKind) -> Option<CompoundOp> {
    Some(match kind {
        TokenKind::PlusAssign => CompoundOp::Add,
        TokenKind::MinusAssign => CompoundOp::Sub,
        TokenKind::StarAssign => CompoundOp::Mul,
        TokenKind::SlashAssign => CompoundOp::Div,
        TokenKind::CaretAssign => CompoundOp::Pow,
        TokenKind::AtAssign => CompoundOp::MatMul,
        _ => return None,
    })
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
        TokenKind::PlusAssign => "+=",
        TokenKind::MinusAssign => "-=",
        TokenKind::StarAssign => "*=",
        TokenKind::SlashAssign => "/=",
        TokenKind::CaretAssign => "^=",
        TokenKind::AtAssign => "@=",
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
