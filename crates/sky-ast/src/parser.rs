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
    span_text, AssertKind, AssignTarget, BinaryOp, Block, BoolOp, CompoundOp, DimExpr, Expr,
    ExprKind, Ident, IndexItem, Literal, Module, Param, ShapeAnn, ShapeKind, Stmt, StmtKind,
    TypeExpr, UnaryOp, UnitExpr, TAG_NAMES,
};
use crate::precedence::{
    bool_binding_power, infix_binding_power, is_postfix_start, prefix_binding_power, AS_BP,
    CONDITIONAL_BP, CONDITIONAL_COND_BP, NOT_BP, PLUS_MINUS_BP, POSTFIX_BP,
};

/// Maximum nesting depth. Each level counts once: a `(`...`)` group, a prefix
/// operator (including `not`), a call, an index (its slices included: the
/// bounds of `a[i:j]` sit one level below `a`), an attribute, a `⁻¹`, each
/// infix operator folded into an operand, and each `and`/`or` run (however
/// long: a run is one flat node), each `a if c else b`, and each block (the
/// body of an `if`, `elif`, `else`, `while`, `for`, or `def`), so blocks and the
/// expressions inside them share one budget. `x` uses 0, `(((x)))` 3, `f(g(x))` 2,
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

/// Parses a file's statements: simple statements and the block statements
/// `if`/`elif`/`else`, `while`, `for`, and `def`, whose bodies are `Indent` ... `Dedent`
/// runs of statements. Every error is recorded and parsing resumes at the
/// next line (or, for a bad block header, after the header's block), so one
/// mistake doesn't desynchronize the block structure: see
/// `Parser::recover_header`.
pub fn parse_module(tokens: &[Token], src: &str) -> (Module, Vec<ParseError>) {
    let mut p = Parser::new(tokens, src);
    let stmts = p.parse_stmts(false);
    p.check_contract_placement(&stmts, false);
    let end = u32::try_from(src.len()).unwrap_or(u32::MAX);
    let module = Module {
        stmts,
        span: Span { start: 0, end },
    };
    (module, p.errors)
}

#[path = "parse_ode.rs"]
mod ode;
pub(crate) use ode::{equation_sexpr, solve_sexpr};

struct Parser<'t, 's> {
    /// True while parsing the header or options of a `solve`, where `to`,
    /// `with` and `within` end an expression (see `parse_ode.rs`).
    solve_words: bool,
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
            solve_words: false,
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

    /// The statements up to the end of the enclosing block (its `Dedent`,
    /// consumed) or of the input. Runs at the current nesting depth: a
    /// statement never inherits depth from the one before it.
    fn parse_stmts(&mut self, in_block: bool) -> Vec<Stmt> {
        let depth = self.depth;
        let mut stmts = Vec::new();
        loop {
            self.depth = depth;
            match self.peek() {
                TokenKind::Eof => break,
                // Blank lines. Dedents belong to the block that ends here; a
                // stray one at the top level (only in hand-made tokens) is
                // ignored.
                TokenKind::Newline => {
                    self.bump();
                    continue;
                }
                TokenKind::Dedent => {
                    self.bump();
                    if in_block {
                        break;
                    }
                    continue;
                }
                TokenKind::Indent => {
                    self.stray_indent(&mut stmts);
                    continue;
                }
                _ => {}
            }
            let start = self.pos;
            if let Some(stmt) = self.parse_item() {
                stmts.push(stmt);
            }
            if self.pos == start {
                self.bump(); // Guarantee progress.
            }
        }
        self.depth = depth;
        stmts
    }

    /// An `Indent` where a statement should start: nothing before it opens a
    /// block. Reports it, then parses the indented lines as a block anyway
    /// and splices their statements in, so the `Dedent` that ends them
    /// doesn't end the block we are in.
    #[cold]
    #[inline(never)]
    fn stray_indent(&mut self, stmts: &mut Vec<Stmt>) {
        let t = self.bump();
        self.errors
            .push(ParseError::new("unexpected indentation", t.span).with_help(
                "only the body of an `if`, `while`, `for`, or `def` is indented; \
                 otherwise align this line with the one before it",
            ));
        if let Some(block) = self.parse_block_body(t.span) {
            stmts.extend(block.stmts);
        }
    }

    /// One statement. `None` means an error was recorded and the position is
    /// at a line boundary (the start of the next line, or of the block's
    /// `Dedent`), so the caller just carries on.
    fn parse_item(&mut self) -> Option<Stmt> {
        let tok = self.current();
        if self.at_solve() {
            return self.parse_solve();
        }
        let TokenKind::Keyword(kw) = tok.kind else {
            return self.parse_simple_item();
        };
        match kw {
            Kw::If => self.parse_if(),
            Kw::While => self.parse_while(),
            Kw::For => self.parse_for(),
            Kw::Def => self.parse_def(),
            Kw::Equation => self.parse_equation(),
            Kw::Solve => self.reject_bare_solve(),
            Kw::Import | Kw::Field | Kw::Else | Kw::Elif => self.reject_statement(tok, kw),
            _ => self.parse_simple_item(),
        }
    }

    /// A statement that ends at its line: expression, assignment, binding,
    /// `return`, `pass`, `break`, `continue`.
    fn parse_simple_item(&mut self) -> Option<Stmt> {
        let stmt = self.parse_stmt();
        if stmt.is_none() {
            self.recover_to_line_end();
        }
        stmt
    }

    /// Skips the rest of the line and, if the line opens a block (an
    /// `Indent` follows), the whole block. For statements that are rejected
    /// as a whole (`for`, `import`, ...): their body is not parsed.
    fn recover_stmt(&mut self) {
        self.recover_to_line_end();
        if self.peek() == TokenKind::Indent {
            self.bump();
            self.skip_block_rest();
        }
    }

    /// Recovery from a bad block header (`if x = 1:`, `def f(:`): skips the
    /// rest of the line, then parses the body if one follows, dropping it.
    /// The body's own errors are still reported, and its `Dedent` is
    /// consumed here, so the enclosing block stays in sync.
    fn recover_header(&mut self, header: Span) {
        self.recover_to_line_end();
        if self.peek() == TokenKind::Indent {
            self.bump();
            let _ = self.parse_block_body(header);
        }
    }

    /// Skips to the `Dedent` matching an `Indent` already consumed, and
    /// past it. Iterative, so it is safe at any nesting.
    fn skip_block_rest(&mut self) {
        let mut level = 1u32;
        loop {
            match self.peek() {
                TokenKind::Eof => return,
                TokenKind::Indent => level = level.saturating_add(1),
                TokenKind::Dedent => {
                    level -= 1;
                    if level == 0 {
                        self.bump();
                        return;
                    }
                }
                _ => {}
            }
            self.bump();
        }
    }

    /// `import`, the reserved words, and an `else`/`elif` with no `if`
    /// before it, where a statement starts.
    #[cold]
    #[inline(never)]
    fn reject_statement(&mut self, tok: Token, kw: Kw) -> Option<Stmt> {
        let word = kw.as_str();
        let err = match kw {
            Kw::Import => ParseError::new("`import` is not supported yet", tok.span)
                .with_help("there is no module loader yet; keep the code in one file for now"),
            Kw::Field => ParseError::new(
                format!("`{word}` is reserved for a future version"),
                tok.span,
            )
            .with_help("reserved words cannot be used as names; pick another name"),
            _ => ParseError::new(format!("`{word}` without a matching `if`"), tok.span).with_help(
                "an `elif` or `else` must directly follow the block of an `if` or `elif`, \
                 at the same indentation",
            ),
        };
        self.errors.push(err);
        self.bump();
        self.recover_stmt();
        None
    }

    /// `if cond: ... (elif cond: ...)* (else: ...)?`, with `if` current. A
    /// clause with a bad header is reported and dropped, and the clauses
    /// after it are still parsed, so one bad `elif` doesn't turn the `else`
    /// after it into a second error.
    fn parse_if(&mut self) -> Option<Stmt> {
        let start = self.current().span;
        let mut clause = self.bump();
        let mut branches: Vec<(Expr, Block)> = Vec::new();
        let mut else_block = None;
        let mut end = clause.span;
        loop {
            if clause.kind == TokenKind::Keyword(Kw::Else) {
                if let Some(block) = self.parse_body(clause.span, "`else`") {
                    end = block.span;
                    else_block = Some(block);
                }
                break;
            }
            let what = if clause.kind == TokenKind::Keyword(Kw::If) {
                "`if`"
            } else {
                "`elif`"
            };
            match self.expr_bp(0) {
                Some(cond) => {
                    if let Some(block) = self.parse_body(clause.span, what) {
                        end = block.span;
                        branches.push((cond, block));
                    }
                }
                None => self.recover_header(clause.span),
            }
            match self.peek() {
                TokenKind::Keyword(Kw::Elif) => clause = self.bump(),
                TokenKind::Keyword(Kw::Else) => {
                    clause = self.bump();
                    if let Some(t) = self.else_if(clause) {
                        clause = t;
                    }
                }
                _ => break,
            }
        }
        if branches.is_empty() {
            return None;
        }
        Some(Stmt {
            span: start.join(end),
            kind: StmtKind::If {
                branches,
                else_block,
            },
        })
    }

    /// `else if`, which Python spells `elif`: reports it, consumes the `if`,
    /// and returns a token to continue as an `elif` clause.
    #[cold]
    #[inline(never)]
    fn else_if(&mut self, else_tok: Token) -> Option<Token> {
        if self.peek() != TokenKind::Keyword(Kw::If) {
            return None;
        }
        let if_tok = self.bump();
        let span = else_tok.span.join(if_tok.span);
        self.errors
            .push(ParseError::new("`else if` is not valid", span).with_help("write `elif`"));
        Some(Token {
            kind: TokenKind::Keyword(Kw::Elif),
            span,
        })
    }

    /// `while cond: body`, with `while` current.
    fn parse_while(&mut self) -> Option<Stmt> {
        let kw = self.bump();
        let Some(cond) = self.expr_bp(0) else {
            self.recover_header(kw.span);
            return None;
        };
        let body = self.parse_body(kw.span, "`while`")?;
        Some(Stmt {
            span: kw.span.join(body.span),
            kind: StmtKind::While { cond, body },
        })
    }

    /// `for name in iter: body`, with `for` current. Only a plain name can
    /// be the loop variable (no unpacking); `iter` is any expression.
    fn parse_for(&mut self) -> Option<Stmt> {
        let kw = self.bump();
        let Some((var, iter)) = self.parse_for_header(kw.span) else {
            self.recover_header(kw.span);
            return None;
        };
        let body = self.parse_body(kw.span, "`for`")?;
        Some(Stmt {
            span: kw.span.join(body.span),
            kind: StmtKind::For { var, iter, body },
        })
    }

    /// `name in iter` up to (not including) the `:`.
    fn parse_for_header(&mut self, kw: Span) -> Option<(Ident, Expr)> {
        let name_tok = self.current();
        if name_tok.kind != TokenKind::Ident {
            let message = format!(
                "expected a loop variable name after `for`, found {}",
                self.describe(name_tok)
            );
            let help = if name_tok.kind == TokenKind::LParen {
                "a loop binds one name; Sky has no tuples to unpack"
            } else {
                "write `for name in ...`"
            };
            self.errors.push(
                ParseError::new(message, name_tok.span)
                    .with_secondary(kw, "`for` starts here")
                    .with_help(help),
            );
            return None;
        }
        self.bump();
        let in_tok = self.current();
        if in_tok.kind != TokenKind::Keyword(Kw::In) {
            let message = format!(
                "expected `in` after the loop variable, found {}",
                self.describe(in_tok)
            );
            let mut err = ParseError::new(message, in_tok.span)
                .with_secondary(name_tok.span, "the loop variable");
            err = if in_tok.kind == TokenKind::Comma {
                err.with_help(
                    "a loop binds one name; loop over a matrix's rows with `for row in A:`",
                )
            } else {
                err.with_help("write `for name in range(n):` or `for x in v:`")
            };
            self.errors.push(err);
            return None;
        }
        self.bump();
        let iter = self.expr_bp(0)?;
        Some((
            Ident {
                span: name_tok.span,
            },
            iter,
        ))
    }

    /// `def name(params) (-> ret)?: body`, with `def` current.
    fn parse_def(&mut self) -> Option<Stmt> {
        let kw = self.bump();
        let Some((name, params, ret)) = self.parse_signature(kw.span) else {
            self.recover_header(kw.span);
            return None;
        };
        let body = self.parse_body(kw.span, "`def`")?;
        Some(Stmt {
            span: kw.span.join(body.span),
            kind: StmtKind::Def {
                name,
                params,
                ret,
                body,
            },
        })
    }

    /// `name(param, ...) (-> ret)?` up to (not including) the `:`.
    fn parse_signature(&mut self, kw: Span) -> Option<(Ident, Vec<Param>, Option<TypeExpr>)> {
        let name_tok = self.current();
        if name_tok.kind != TokenKind::Ident {
            let message = format!(
                "expected a function name after `def`, found {}",
                self.describe(name_tok)
            );
            return self.fail_def(message, name_tok.span, kw, "`def` starts here");
        }
        self.bump();
        let open = self.current();
        if open.kind != TokenKind::LParen {
            let message = format!(
                "expected `(` after the function name, found {}",
                self.describe(open)
            );
            return self.fail_def(message, open.span, name_tok.span, "function name");
        }
        self.bump();
        let mut params: Vec<Param> = Vec::new();
        while self.peek() != TokenKind::RParen {
            let param = self.parse_param(open.span)?;
            params.push(param);
            match self.peek() {
                TokenKind::Comma => {
                    self.bump();
                }
                TokenKind::RParen => {}
                _ => {
                    let last = params.last().map_or(open.span, |p| p.span);
                    let message = format!(
                        "expected `,` or `)` after the parameter, found {}",
                        self.describe(self.current())
                    );
                    let at = self.current().span;
                    return self.fail_def(message, at, last, "this parameter");
                }
            }
        }
        self.bump(); // `)`
        let ret = if self.peek() == TokenKind::Arrow {
            self.bump();
            Some(self.parse_type()?)
        } else {
            None
        };
        Some((
            Ident {
                span: name_tok.span,
            },
            params,
            ret,
        ))
    }

    /// `name` or `name: type`.
    fn parse_param(&mut self, open: Span) -> Option<Param> {
        let tok = self.current();
        if tok.kind != TokenKind::Ident {
            let message = format!("expected a parameter name, found {}", self.describe(tok));
            return self.fail_def(message, tok.span, open, "parameters start here");
        }
        self.bump();
        let mut span = tok.span;
        let mut ty = None;
        if self.peek() == TokenKind::Colon {
            self.bump();
            let t = self.parse_type()?;
            span = span.join(t.span);
            ty = Some(t);
        }
        if self.peek() == TokenKind::Assign {
            return self.fail_default_value(span);
        }
        Some(Param {
            name: Ident { span: tok.span },
            ty,
            span,
        })
    }

    /// The `:` NEWLINE INDENT statements DEDENT after a block header.
    /// `header` is the keyword's span and `what` names it (`` `if` ``).
    /// `None` means an error was recorded and the position is at a line
    /// boundary; see [`Self::parse_item`].
    fn parse_body(&mut self, header: Span, what: &str) -> Option<Block> {
        let block = self.parse_body_inner(header, what)?;
        self.check_contract_placement(&block.stmts, what == "`def`");
        Some(block)
    }

    /// `requires` and `ensures` state a function's contract, so they are
    /// allowed only as the first statements of a `def` body (spec
    /// "Contracts"). `stmts` are the statements of one block; `in_def` says
    /// it is a `def` body. Nested blocks were checked when they were parsed.
    fn check_contract_placement(&mut self, stmts: &[Stmt], in_def: bool) {
        let mut head = in_def;
        for s in stmts {
            match &s.kind {
                StmtKind::Assert {
                    kind: kind @ (AssertKind::Requires | AssertKind::Ensures),
                    ..
                } if !head => {
                    let word = kind.keyword();
                    let err = if in_def {
                        ParseError::new(
                            format!("`{word}` must come before the other statements of the function body"),
                            s.span,
                        )
                        .with_help("write the contract lines first, right after the `def` line")
                    } else {
                        ParseError::new(
                            format!("`{word}` is only allowed at the start of a function body"),
                            s.span,
                        )
                        .with_help(
                            "state a function's contract with `requires` and `ensures` right after its `def` line; use `assert` to check a condition here",
                        )
                    };
                    self.errors.push(err);
                }
                StmtKind::Assert {
                    kind: AssertKind::Requires | AssertKind::Ensures,
                    ..
                } => {}
                _ => head = false,
            }
        }
    }

    fn parse_body_inner(&mut self, header: Span, what: &str) -> Option<Block> {
        let colon = self.current();
        if colon.kind != TokenKind::Colon {
            return self.fail_expected_colon(colon, header, what);
        }
        self.bump();
        match self.peek() {
            TokenKind::Newline => {
                self.bump();
                if self.peek() == TokenKind::Indent {
                    self.bump();
                    return self.parse_block_body(header);
                }
                self.fail_empty_block(colon.span, what)
            }
            // A hand-made stream can end or dedent without a `Newline`.
            TokenKind::Eof | TokenKind::Dedent => self.fail_empty_block(colon.span, what),
            _ => self.parse_inline_body(colon.span, what),
        }
    }

    /// The statements of a block, with its `Indent` already consumed, up to
    /// and including its `Dedent`. Counts one nesting level.
    fn parse_block_body(&mut self, header: Span) -> Option<Block> {
        let saved = self.depth;
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            self.depth = saved;
            self.skip_block_rest();
            return self.fail_block_too_deep(header);
        }
        let at = self.prev_span().unwrap_or(header);
        let stmts = self.parse_stmts(true);
        self.depth = saved;
        Some(block(stmts, at))
    }

    /// A body on the header's own line (`if x: y = 1`), which Sky does not
    /// accept (spec, "Blocks and control flow"). Reports it, but parses the
    /// statement as the body anyway, so what follows is not misread.
    #[cold]
    #[inline(never)]
    fn parse_inline_body(&mut self, colon: Span, what: &str) -> Option<Block> {
        let first = self.current();
        self.errors.push(
            ParseError::new(
                format!("the body of {what} must be on its own indented line"),
                first.span,
            )
            .with_secondary(colon, "the block starts after this `:`")
            .with_help("put the body on its own indented line"),
        );
        // A compound statement here would recurse without counting depth.
        let compound = matches!(
            first.kind,
            TokenKind::Keyword(
                Kw::If | Kw::While | Kw::Def | Kw::For | Kw::Import | Kw::Else | Kw::Elif
            )
        );
        if compound {
            self.recover_stmt();
            return None;
        }
        let stmt = self.parse_simple_item()?;
        let span = stmt.span;
        Some(Block {
            stmts: vec![stmt],
            span,
        })
    }

    // ----- statements (simple) -----

    fn parse_stmt(&mut self) -> Option<Stmt> {
        let first = match self.peek() {
            TokenKind::Keyword(Kw::Return) => return self.parse_return(),
            TokenKind::Keyword(kw @ (Kw::Assert | Kw::Requires | Kw::Ensures)) => {
                return self.parse_contract(kw)
            }
            TokenKind::Keyword(kw @ (Kw::Pass | Kw::Break | Kw::Continue)) => {
                return self.parse_bare_keyword(kw)
            }
            _ => self.expr_bp(0)?,
        };
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

    /// `return` or `return value`, with `return` current.
    fn parse_return(&mut self) -> Option<Stmt> {
        let kw = self.bump();
        let value = match self.peek() {
            TokenKind::Newline | TokenKind::Eof | TokenKind::Dedent => None,
            _ => Some(self.expr_bp(0)?),
        };
        let end = value.as_ref().map_or(kw.span, |v| v.span);
        let stmt = Stmt {
            span: kw.span.join(end),
            kind: StmtKind::Return(value),
        };
        self.expect_stmt_end(&stmt, None)?;
        Some(stmt)
    }

    /// `assert cond`, `requires cond` or `ensures cond`, each with an
    /// optional `, "message"`, with the keyword current.
    fn parse_contract(&mut self, kw: Kw) -> Option<Stmt> {
        let kind = match kw {
            Kw::Requires => AssertKind::Requires,
            Kw::Ensures => AssertKind::Ensures,
            _ => AssertKind::Assert,
        };
        let tok = self.bump();
        let cond = self.expr_bp(0)?;
        let mut message = None;
        match self.peek() {
            TokenKind::Comma => {
                self.bump();
                let m = self.current();
                if m.kind != TokenKind::Str {
                    let found = self.describe(m);
                    self.errors.push(
                        ParseError::new(
                            format!("expected a string message after `,`, found {found}"),
                            m.span,
                        )
                        .with_secondary(tok.span, format!("`{}` starts here", kind.keyword()))
                        .with_help(format!(
                            "write `{} condition, \"what went wrong\"`",
                            kind.keyword()
                        )),
                    );
                    return None;
                }
                self.bump();
                message = Some(m.span);
            }
            TokenKind::Assign => {
                let eq = self.current();
                self.errors.push(
                    ParseError::new(
                        format!("`{}` takes a condition, not an assignment", kind.keyword()),
                        eq.span,
                    )
                    .with_secondary(cond.span, "this is the condition")
                    .with_help("compare with `==` (or `<=`, `>=`, ...)"),
                );
                return None;
            }
            _ => {}
        }
        let end = message.unwrap_or(cond.span);
        let stmt = Stmt {
            span: tok.span.join(end),
            kind: StmtKind::Assert {
                kind,
                cond,
                message,
            },
        };
        self.expect_stmt_end(&stmt, None)?;
        Some(stmt)
    }

    /// `pass`, `break`, or `continue`, with the keyword current.
    fn parse_bare_keyword(&mut self, kw: Kw) -> Option<Stmt> {
        let tok = self.bump();
        let kind = match kw {
            Kw::Pass => StmtKind::Pass,
            Kw::Break => StmtKind::Break,
            _ => StmtKind::Continue,
        };
        let stmt = Stmt {
            span: tok.span,
            kind,
        };
        self.expect_stmt_end(&stmt, None)?;
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
            _ if matches!(
                stmt.kind,
                StmtKind::Pass | StmtKind::Break | StmtKind::Continue
            ) =>
            {
                let word = self.text(stmt.span);
                self.errors.push(
                    ParseError::new(
                        format!(
                            "expected end of line after `{word}`, found {}",
                            self.describe(tok)
                        ),
                        tok.span,
                    )
                    .with_secondary(stmt.span, "this statement takes no value"),
                );
                None
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
                    StmtKind::Assert { cond, message, .. } => message.unwrap_or(cond.span),
                    // Simple statements only; block statements never get here.
                    _ => stmt.span,
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
                Step::Conditional(if_span) => {
                    prev_comparison = None;
                    self.parse_conditional(lhs, if_span)?
                }
                // `as` ends the expression: no chaining, nothing after it.
                Step::As(as_span) => return self.parse_as(lhs, as_span),
                Step::Glyph { transpose, op_span } => glyph(lhs, op_span, transpose),
                Step::Attribute(name) => attribute(lhs, name),
                Step::Call(open) => self.parse_call(lhs, open)?,
                Step::Index(open) => self.parse_index(lhs, open)?,
            };
        }
    }

    /// `lhs op rhs` with `op` already consumed and counted. `op` is `None`
    /// for `±`, which builds an `ExprKind::PlusMinus` instead of a binary
    /// node. It shares this path (rather than having its own `Step`) so the
    /// recursive frame of `expr_bp_inner` does not grow.
    fn parse_infix_rhs(
        &mut self,
        lhs: Expr,
        op: Option<BinaryOp>,
        op_span: Span,
        r_bp: u8,
    ) -> Option<Expr> {
        let rhs = self.expr_bp(r_bp)?;
        Some(match op {
            Some(op) => binary(op, op_span, lhs, rhs),
            None => crate::plus_minus::plus_minus(lhs, rhs, op_span),
        })
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

    /// `then if cond else else_` with `if` (`if_span`) already consumed and
    /// counted. The condition is an `or` expression and the `else` arm may
    /// be another conditional, so the operator is right-associative.
    fn parse_conditional(&mut self, then: Expr, if_span: Span) -> Option<Expr> {
        let cond = self.expr_bp(CONDITIONAL_COND_BP)?;
        if self.peek() != TokenKind::Keyword(Kw::Else) {
            return self.fail_missing_else(&then, &cond, if_span);
        }
        self.bump();
        let else_ = self.expr_bp(CONDITIONAL_BP)?;
        Some(conditional(then, cond, else_))
    }

    /// `value as unit` with `as` (`as_span`) already consumed and counted.
    /// The unit is read by [`Parser::parse_unit`], the same sub-grammar as
    /// a unit literal, so `km/h`, `kg*m^2/s^2` and `MeV` are spelled alike
    /// in both places. Nothing may follow it in the same expression.
    #[inline(never)]
    fn parse_as(&mut self, value: Expr, as_span: Span) -> Option<Expr> {
        let first = self.current();
        if first.kind != TokenKind::Ident {
            return self.fail_as_unit(first, as_span);
        }
        let unit = self.parse_unit(UnitCtx::Display)?;
        self.reject_after_as(&value, &unit, as_span)?;
        Some(as_expr(value, unit, as_span))
    }

    /// `as` must be the last thing in its expression: it shows a value, it
    /// does not make a number, so an operator, a second `as`, or an `if`
    /// after the unit is a mistake with a clear fix.
    fn reject_after_as(&mut self, value: &Expr, unit: &UnitExpr, as_span: Span) -> Option<()> {
        let tok = self.current();
        let is_operator = infix_binding_power(tok.kind).is_some()
            || bool_binding_power(tok.kind).is_some()
            || is_postfix_start(tok.kind);
        let (message, help) = match tok.kind {
            TokenKind::Keyword(Kw::As) => (
                "a value can be shown in only one unit".to_string(),
                "remove the second `as`".to_string(),
            ),
            TokenKind::Keyword(Kw::If) => (
                "`as` must come last in an expression".to_string(),
                "to show a conditional in a unit, write `(a if c else b) as km` or `a if c else b as km`"
                    .to_string(),
            ),
            _ if is_operator => (
                format!("`{}` cannot follow `as`", self.text(tok.span)),
                format!(
                    "`as` shows a value in `{}`; it does not produce a number to compute with. \
                     Put the arithmetic before it: `(… {} …) as {}`",
                    self.text(unit.span()),
                    self.text(tok.span),
                    self.text(unit.span())
                ),
            ),
            _ => return Some(()),
        };
        self.errors.push(
            ParseError::new(message, tok.span)
                .with_secondary(as_span.join(unit.span()), "shown in this unit")
                .with_secondary(value.span, "the value")
                .with_help(help),
        );
        None
    }

    /// `x as` followed by something that is not a unit name.
    #[cold]
    #[inline(never)]
    fn fail_as_unit<T>(&mut self, found: Token, as_span: Span) -> Option<T> {
        self.errors.push(
            ParseError::new(
                format!("expected a unit after `as`, found {}", self.describe(found)),
                found.span,
            )
            .with_secondary(as_span, "`as` shows a value in a unit")
            .with_help("write the unit to show the value in, e.g. `v as km/h`"),
        );
        None
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
                TokenKind::Inverse => Some(Step::Glyph {
                    transpose: false,
                    op_span: tok.span,
                }),
                TokenKind::Transpose => Some(Step::Glyph {
                    transpose: true,
                    op_span: tok.span,
                }),
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
        if tok.kind == TokenKind::Keyword(Kw::As) {
            if AS_BP < min_bp {
                return Some(Step::Done);
            }
            self.bump();
            self.enter(tok.span)?;
            return Some(Step::As(tok.span));
        }
        if tok.kind == TokenKind::Keyword(Kw::If) {
            if CONDITIONAL_BP < min_bp {
                return Some(Step::Done);
            }
            self.bump();
            self.enter(tok.span)?;
            return Some(Step::Conditional(tok.span));
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
        if tok.kind == TokenKind::PlusMinus {
            if PLUS_MINUS_BP.0 < min_bp {
                return Some(Step::Done);
            }
            self.bump();
            self.enter(tok.span)?;
            return Some(Step::Infix {
                op: None,
                op_span: tok.span,
                r_bp: PLUS_MINUS_BP.1,
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
            op: Some(op),
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
        if tok.kind == TokenKind::LBracket {
            return self.parse_array();
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
                if self.peek() == TokenKind::Ident && !self.stops_unit() {
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

    /// `[ expr, ... ]` with the `[` current: an array literal (a vector, or
    /// a matrix when every element is itself a literal row). At least one
    /// element: `[]` has no element to give it a dimension. A trailing comma
    /// is allowed. Rows nest like any expression, so `[[1, 2], [3, 4]]` is an
    /// array of two arrays; the checker decides what shape that makes.
    fn parse_array(&mut self) -> Option<Expr> {
        let open = self.bump().span;
        self.enter(open)?;
        if self.peek() == TokenKind::RBracket {
            let close = self.bump().span;
            return self.fail_empty_array(open.join(close));
        }
        let mut elems = Vec::new();
        loop {
            elems.push(self.expr_bp(0)?);
            match self.peek() {
                TokenKind::Comma => {
                    self.bump();
                    if self.peek() == TokenKind::RBracket {
                        break;
                    }
                }
                TokenKind::RBracket => break,
                _ => {
                    let last = elems.last().map_or(open, |e| e.span);
                    return self.fail_unclosed(last, open, "[", "`,` or `]`");
                }
            }
        }
        let close = self.bump().span;
        Some(array(open, elems, close))
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
    fn fail_block_too_deep<T>(&mut self, header: Span) -> Option<T> {
        self.errors.push(
            ParseError::new(
                format!("blocks are nested too deeply (the limit is {MAX_DEPTH} levels)"),
                header,
            )
            .with_help("split the inner blocks into a function"),
        );
        None
    }

    /// The token after a block header is not `:`. Recovers past the header
    /// and its block.
    #[cold]
    #[inline(never)]
    fn fail_expected_colon<T>(&mut self, found: Token, header: Span, what: &str) -> Option<T> {
        let mut err = ParseError::new(
            format!("expected `:` after {what}, found {}", self.describe(found)),
            found.span,
        )
        .with_secondary(header, format!("{what} starts here"));
        err = match found.kind {
            TokenKind::Assign => err.with_help("`=` assigns; to compare, write `==`"),
            _ => err.with_help("end the header line with `:`"),
        };
        self.errors.push(err);
        self.recover_header(header);
        None
    }

    /// `:` and a newline, but no indented line after it. Nothing to skip:
    /// the position is already on the next statement.
    #[cold]
    #[inline(never)]
    fn fail_empty_block<T>(&mut self, colon: Span, what: &str) -> Option<T> {
        self.errors.push(
            ParseError::new(format!("expected an indented block after {what}"), colon)
                .with_help("use `pass` for a block that does nothing"),
        );
        None
    }

    /// A malformed `def` header; `at` is the culprit and `context` the
    /// span it should be read against, labelled `label`.
    #[cold]
    #[inline(never)]
    fn fail_def<T>(&mut self, message: String, at: Span, context: Span, label: &str) -> Option<T> {
        self.errors
            .push(ParseError::new(message, at).with_secondary(context, label));
        None
    }

    #[cold]
    #[inline(never)]
    fn fail_default_value<T>(&mut self, param: Span) -> Option<T> {
        let eq = self.current();
        self.errors.push(
            ParseError::new("default parameter values are not supported yet", eq.span)
                .with_secondary(param, "this parameter")
                .with_help("pass the value at every call"),
        );
        None
    }

    /// `a if c` with no `else` after the condition.
    #[cold]
    #[inline(never)]
    fn fail_missing_else<T>(&mut self, then: &Expr, cond: &Expr, if_span: Span) -> Option<T> {
        let tok = self.current();
        let then_text = self.text(then.span);
        let cond_text = self.text(cond.span);
        self.errors.push(
            ParseError::new(
                format!(
                    "expected `else` in the conditional expression, found {}",
                    self.describe(tok)
                ),
                tok.span,
            )
            .with_secondary(if_span, "this `if` needs an `else`")
            .with_help(format!(
                "a conditional expression has two arms: `{then_text} if {cond_text} else ...`"
            )),
        );
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
            TokenKind::Keyword(kw @ (Kw::And | Kw::Or)) => {
                format!("expected an expression, found keyword `{}`", kw.as_str())
            }
            TokenKind::Keyword(kw) => {
                format!("expected an expression, found keyword `{}`", kw.as_str())
            }
            _ => format!("expected an expression, found {}", self.describe(tok)),
        };
        let mut err = ParseError::new(message, tok.span);
        if let TokenKind::Keyword(kw) = tok.kind {
            match kw {
                Kw::If => {
                    err = err.with_help(
                        "for a value that depends on a condition, write `a if cond else b`",
                    );
                }
                Kw::And | Kw::Or | Kw::Not | Kw::True | Kw::False => {}
                Kw::Equation | Kw::Solve => {
                    err = err.with_help(
                        "`equation` and `solve` start statements: `sol = solve orbit from 0 s to 1 yr with ...`",
                    );
                }
                Kw::Field => {
                    err = err.with_help("this word is reserved for a future version");
                }
                Kw::In => err = err.with_help(
                    "`in` only appears in `for` loops (`for x in v:`); Sky has no membership test",
                ),
                Kw::As => {
                    err = err.with_help(
                        "`as` follows a value: write `v as km/h` (it is also a reserved word, so it cannot name a variable)",
                    );
                }
                _ => {
                    err = err.with_help(format!(
                        "`{}` starts a statement, so it cannot be part of an expression",
                        kw.as_str()
                    ));
                }
            }
        }
        self.errors.push(err);
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
    fn fail_empty_array<T>(&mut self, span: Span) -> Option<T> {
        self.errors.push(
            ParseError::new("an array literal needs at least one element", span).with_help(
                "the elements fix an array's dimension and `[]` has none; \
                     for zeros write `zeros(n)`",
            ),
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
                    UnitCtx::Annotation | UnitCtx::Display => (
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
        if tok.kind == TokenKind::Ident && !self.stops_unit() {
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
                (UnitCtx::Annotation | UnitCtx::Display, Some(kind)) => err.with_help(format!(
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
        // Capitalized words are tags only when they lead up to a shape word
        // (`PositiveDefinite Matrix[N, N]`). Otherwise the first word starts
        // a unit, so `x: N m` gets the unit help (`N*m`) instead of being
        // read as a matrix property called `N`. Decided once, up front, so
        // a long line stays linear.
        let tagged = self.tags_lead_to_shape();
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
            if tagged && next == Some(TokenKind::Ident) {
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

    /// Whether the type starting at the current token begins with matrix
    /// properties: a run of capitalized names ending in a shape word
    /// (`Symmetric Diagonal Matrix`), or a known property word
    /// ([`TAG_NAMES`]) followed by another name (`Symmetric m`, which the
    /// checker then reports as a property without a shape).
    fn tags_lead_to_shape(&self) -> bool {
        if let (Some(first), Some(second)) = (self.nth(0), self.nth(1)) {
            let word = span_text(self.src, first.span).unwrap_or("");
            if first.kind == TokenKind::Ident
                && second.kind == TokenKind::Ident
                && TAG_NAMES.contains(&word)
            {
                return true;
            }
        }
        let mut i = 0;
        while let Some(tok) = self.nth(i) {
            if tok.kind != TokenKind::Ident {
                return false;
            }
            let word = span_text(self.src, tok.span).unwrap_or("");
            if ShapeKind::from_name(word).is_some() {
                return i > 0;
            }
            if !word.chars().next().is_some_and(char::is_uppercase) {
                return false;
            }
            i += 1;
        }
        false
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

/// Where a unit is being parsed: after a number (`10 km`), in a type
/// annotation (`m/s`), or after `as` (`v as m/s`). See `Parser::parse_unit`.
#[derive(Clone, Copy)]
enum UnitCtx<'a> {
    Literal(&'a Literal),
    Annotation,
    /// After `as`. Like an annotation there is no quantity next to the
    /// unit, so a spaced `m / s` can only be a unit written with spaces.
    Display,
}

impl UnitCtx<'_> {
    /// True where no quantity precedes the unit (a type, or `as`), so the
    /// errors have no number to mention.
    fn is_annotation(self) -> bool {
        matches!(self, UnitCtx::Annotation | UnitCtx::Display)
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
    /// A binary operator, or `±` (`op: None`).
    Infix {
        op: Option<BinaryOp>,
        op_span: Span,
        r_bp: u8,
    },
    /// The `if` of `a if c else b`, already consumed.
    Conditional(Span),
    /// The `as` of `v as unit`, already consumed.
    As(Span),
    /// The postfix glyph `⁻¹` or `ᵀ`, already consumed. One variant and one
    /// builder for both, so the recursive parser frame stays small.
    Glyph {
        transpose: bool,
        op_span: Span,
    },
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
fn as_expr(value: Expr, unit: UnitExpr, as_span: Span) -> Expr {
    Expr {
        span: value.span.join(unit.span()),
        kind: ExprKind::As {
            value: Box::new(value),
            unit: Box::new(unit),
            as_span,
        },
    }
}

#[inline(never)]
fn conditional(then: Expr, cond: Expr, else_: Expr) -> Expr {
    Expr {
        span: then.span.join(else_.span),
        kind: ExprKind::IfElse {
            then: Box::new(then),
            cond: Box::new(cond),
            else_: Box::new(else_),
        },
    }
}

/// A block from its statements. `at` (the `Indent`, or the block's
/// header) is the span of a block whose statements all failed to parse.
#[inline(never)]
fn block(stmts: Vec<Stmt>, at: Span) -> Block {
    let span = match (stmts.first(), stmts.last()) {
        (Some(a), Some(b)) => a.span.join(b.span),
        _ => at,
    };
    Block { stmts, span }
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
fn glyph(operand: Expr, op_span: Span, transpose: bool) -> Expr {
    let (operand, op_span) = (Box::new(operand), op_span);
    Expr {
        span: operand.span.join(op_span),
        kind: if transpose {
            ExprKind::Transpose { operand, op_span }
        } else {
            ExprKind::Inverse { operand, op_span }
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
fn array(open: Span, elems: Vec<Expr>, close: Span) -> Expr {
    Expr {
        span: open.join(close),
        kind: ExprKind::Array(elems),
    }
}

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
        TokenKind::PlusMinus => "±",
        TokenKind::Star => "*",
        TokenKind::Slash => "/",
        TokenKind::Caret => "^",
        TokenKind::At => "@",
        TokenKind::Inverse => "⁻¹",
        TokenKind::Transpose => "ᵀ",
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
        TokenKind::Prime => "'",
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
