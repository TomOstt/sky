//! Parsing `equation` and `solve` (spec "Differential equations"), and their
//! S-expression form.
//!
//! ```text
//! equation := "equation" NAME "(" NAME ":" type ")" ":" NEWLINE INDENT item+ DEDENT
//! item     := NAME ":" type                  -- a state variable
//!           | NAME "'" "=" expr              -- its derivative
//!           | NAME "=" expr                  -- a local name
//! solve    := NAME "=" "solve" NAME "from" expr "to" expr
//!             ("with" NAME "=" expr ("," NAME "=" expr)*)?
//!             (":" NEWLINE INDENT option+ DEDENT)?
//! option   := "tolerance" expr
//!           | "conserve" NAME "=" expr ("within" expr)?
//! ```
//!
//! `from`, `to`, `with`, `within`, `tolerance` and `conserve` are ordinary
//! names that mean something only here, so `to = 3` still works elsewhere.
//! Inside a `solve` header a number is never followed by a unit called `to`,
//! `with` or `within` (see `Parser::stops_unit`).
//!
//! Like the rest of the parser: never panics, and every `None` means an
//! error was recorded.

use sky_syntax::{Kw, Span, TokenKind};

use crate::ast::{
    span_text, Equation, EquationItem, Ident, Initial, Solve, Stmt, StmtKind, TypeExpr,
};
use crate::ast::{Expr, SolveOption};

use super::{ParseError, Parser, MAX_DEPTH};

/// The words of a `solve` header that end an expression.
const SOLVE_WORDS: [&str; 3] = ["to", "with", "within"];

impl Parser<'_, '_> {
    /// Whether the current name is a `solve` word that ends the expression
    /// before it, so a number before it is not a unit literal.
    pub(super) fn stops_unit(&self) -> bool {
        self.solve_words
            && span_text(self.src, self.current().span).is_some_and(|w| SOLVE_WORDS.contains(&w))
    }

    /// `name = solve ...` at the start of a statement.
    pub(super) fn at_solve(&self) -> bool {
        self.peek() == TokenKind::Ident
            && self.nth(1).is_some_and(|t| t.kind == TokenKind::Assign)
            && self
                .nth(2)
                .is_some_and(|t| t.kind == TokenKind::Keyword(Kw::Solve))
            && self.nth(3).is_some_and(|t| t.kind == TokenKind::Ident)
    }

    fn word_is(&self, word: &str) -> bool {
        self.peek() == TokenKind::Ident && span_text(self.src, self.current().span) == Some(word)
    }

    // ----- equation -----

    /// `equation name(t: type): body`, with `equation` current.
    pub(super) fn parse_equation(&mut self) -> Option<Stmt> {
        let kw = self.bump();
        let Some((name, time, time_ty)) = self.parse_equation_header(kw.span) else {
            self.recover_header(kw.span);
            return None;
        };
        let colon = self.current();
        if colon.kind != TokenKind::Colon {
            return self.fail_expected_colon(colon, kw.span, "the `equation` header");
        }
        self.bump();
        if self.peek() != TokenKind::Newline {
            self.errors.push(
                ParseError::new(
                    "the body of an `equation` must be on its own indented line",
                    self.current().span,
                )
                .with_secondary(colon.span, "the block starts after this `:`")
                .with_help("put each state variable and equation on its own indented line"),
            );
            self.recover_to_line_end();
            return None;
        }
        self.bump();
        if self.peek() != TokenKind::Indent {
            return self.fail_empty_block(colon.span, "`equation`");
        }
        self.bump();
        let items = self.parse_equation_items(kw.span)?;
        let end = items.last().map_or(colon.span, EquationItem::span);
        Some(Stmt {
            span: kw.span.join(end),
            kind: StmtKind::Equation(Box::new(Equation {
                name,
                time,
                time_ty,
                items,
            })),
        })
    }

    /// `name(t: type)` up to (not including) the `:`.
    fn parse_equation_header(&mut self, kw: Span) -> Option<(Ident, Ident, TypeExpr)> {
        let name_tok = self.current();
        if name_tok.kind != TokenKind::Ident {
            let message = format!(
                "expected an equation name after `equation`, found {}",
                self.describe(name_tok)
            );
            self.errors.push(
                ParseError::new(message, name_tok.span)
                    .with_secondary(kw, "`equation` starts here")
                    .with_help("write `equation orbit(t: s):`"),
            );
            return None;
        }
        self.bump();
        let open = self.current();
        if open.kind != TokenKind::LParen {
            let message = format!(
                "expected `(` after the equation name, found {}",
                self.describe(open)
            );
            self.errors.push(
                ParseError::new(message, open.span)
                    .with_secondary(name_tok.span, "equation name")
                    .with_help(
                        "name the independent variable and its unit: `equation orbit(t: s):`",
                    ),
            );
            return None;
        }
        self.bump();
        let time_tok = self.current();
        if time_tok.kind != TokenKind::Ident {
            let message = format!(
                "expected the name of the independent variable, found {}",
                self.describe(time_tok)
            );
            self.errors.push(
                ParseError::new(message, time_tok.span)
                    .with_secondary(open.span, "after this `(`")
                    .with_help(
                        "write `(t: s)`: the variable the equations differentiate by, and its unit",
                    ),
            );
            return None;
        }
        self.bump();
        let colon = self.current();
        if colon.kind != TokenKind::Colon {
            let message = format!(
                "expected `:` and a unit after `{}`, found {}",
                self.text(time_tok.span),
                self.describe(colon)
            );
            self.errors.push(
                ParseError::new(message, colon.span)
                    .with_secondary(time_tok.span, "the independent variable")
                    .with_help("its unit is part of the header: `(t: s)`"),
            );
            return None;
        }
        self.bump();
        let ty = self.parse_type()?;
        let close = self.current();
        if close.kind != TokenKind::RParen {
            let message = format!(
                "expected `)` after the independent variable, found {}",
                self.describe(close)
            );
            self.errors.push(
                ParseError::new(message, close.span)
                    .with_secondary(open.span, "unclosed `(`")
                    .with_help("an equation has one independent variable; use names from outside for parameters"),
            );
            return None;
        }
        self.bump();
        Some((
            Ident {
                span: name_tok.span,
            },
            Ident {
                span: time_tok.span,
            },
            ty,
        ))
    }

    /// The items of an equation's block, its `Indent` already consumed, up
    /// to and including the `Dedent`. Counts one nesting level.
    fn parse_equation_items(&mut self, header: Span) -> Option<Vec<EquationItem>> {
        let saved = self.depth;
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            self.depth = saved;
            self.skip_block_rest();
            return self.fail_block_too_deep(header);
        }
        let mut items = Vec::new();
        let errors_before = self.errors.len();
        loop {
            self.depth = saved.saturating_add(1);
            match self.peek() {
                TokenKind::Eof => break,
                TokenKind::Newline => {
                    self.bump();
                    continue;
                }
                TokenKind::Dedent => {
                    self.bump();
                    break;
                }
                TokenKind::Indent => {
                    let t = self.bump();
                    self.errors.push(
                        ParseError::new("unexpected indentation", t.span)
                            .with_help("the lines of an equation are aligned with each other"),
                    );
                    self.skip_block_rest();
                    continue;
                }
                _ => {}
            }
            let start = self.pos;
            match self.parse_equation_item() {
                Some(item) => items.push(item),
                None => self.recover_to_line_end(),
            }
            if self.pos == start {
                self.bump(); // Guarantee progress.
            }
        }
        self.depth = saved;
        if items.is_empty() && self.errors.len() == errors_before {
            self.errors.push(
                ParseError::new("an `equation` needs at least one line", header).with_help(
                    "declare a state variable and its derivative: `x: m` and `x' = ...`",
                ),
            );
            return None;
        }
        Some(items)
    }

    /// One line of an equation's body.
    fn parse_equation_item(&mut self) -> Option<EquationItem> {
        let tok = self.current();
        if tok.kind != TokenKind::Ident {
            let message = format!(
                "expected a state variable (`x: m`), an equation (`x' = ...`) or a name (`a = ...`), found {}",
                self.describe(tok)
            );
            self.errors.push(ParseError::new(message, tok.span));
            return None;
        }
        let name = Ident { span: tok.span };
        let next = self.nth(1).map_or(TokenKind::Eof, |t| t.kind);
        match next {
            TokenKind::Prime => {
                self.bump();
                let prime = self.bump();
                if self.peek() == TokenKind::Prime {
                    let second = self.current();
                    self.errors.push(
                        ParseError::new("second derivatives are not supported", second.span)
                            .with_secondary(prime.span, "the first derivative")
                            .with_help(
                                "write the velocity as a variable of its own: `v: m/s`, `x' = v`, `v' = ...`",
                            ),
                    );
                    return None;
                }
                let eq = self.current();
                if eq.kind != TokenKind::Assign {
                    let message = format!(
                        "expected `=` after `{}'`, found {}",
                        self.text(name.span),
                        self.describe(eq)
                    );
                    self.errors.push(
                        ParseError::new(message, eq.span)
                            .with_secondary(prime.span, "the derivative")
                            .with_help(format!("write `{}' = ...`", self.text(name.span))),
                    );
                    return None;
                }
                self.bump();
                let value = self.expr_bp(0)?;
                let span = name.span.join(value.span);
                self.expect_item_end(value.span)?;
                Some(EquationItem::Derivative {
                    name,
                    prime: prime.span,
                    value,
                    span,
                })
            }
            TokenKind::Colon => {
                self.bump();
                self.bump();
                let ty = self.parse_type()?;
                let span = name.span.join(ty.span);
                if self.peek() == TokenKind::Assign {
                    self.errors.push(
                        ParseError::new(
                            "a state variable is declared without a value",
                            self.current().span,
                        )
                        .with_secondary(ty.span, "the type")
                        .with_help("the initial value goes in `solve`: `... with x = 1 m`"),
                    );
                    return None;
                }
                self.expect_item_end(ty.span)?;
                Some(EquationItem::State { name, ty, span })
            }
            TokenKind::Assign => {
                self.bump();
                self.bump();
                let value = self.expr_bp(0)?;
                let span = name.span.join(value.span);
                self.expect_item_end(value.span)?;
                Some(EquationItem::Local { name, value, span })
            }
            _ => {
                let at = self.nth(1).map_or(tok.span, |t| t.span);
                let found = self
                    .nth(1)
                    .map_or("end of input".to_string(), |t| self.describe(t));
                self.errors.push(
                    ParseError::new(
                        format!(
                            "expected `:`, `'` or `=` after `{}`, found {found}",
                            self.text(name.span)
                        ),
                        at,
                    )
                    .with_secondary(tok.span, "this name")
                    .with_help(
                        "declare it (`x: m`), give its derivative (`x' = ...`), or define it (`a = ...`)",
                    ),
                );
                None
            }
        }
    }

    /// The end of a line inside an equation or `solve` block.
    fn expect_item_end(&mut self, last: Span) -> Option<()> {
        match self.peek() {
            TokenKind::Newline => {
                self.bump();
                Some(())
            }
            TokenKind::Eof | TokenKind::Dedent => Some(()),
            _ => {
                let err = self.unexpected_after(last, "end of line");
                self.errors.push(err);
                None
            }
        }
    }

    // ----- solve -----

    /// A `solve` that is not the right side of `name = ...`.
    pub(super) fn reject_bare_solve(&mut self) -> Option<Stmt> {
        let tok = self.bump();
        let err = if self.peek() == TokenKind::Assign {
            ParseError::new("`solve` is a keyword, so it cannot be a name", tok.span)
                .with_help("pick another name")
        } else {
            ParseError::new("`solve` gives a solution, which needs a name", tok.span)
                .with_help("write `sol = solve orbit from 0 s to 1 yr with x = ...`")
        };
        self.errors.push(err);
        self.recover_stmt();
        None
    }

    /// `name = solve eq from a to b (with x = v, ...)? (: options)?`, with
    /// the name current.
    pub(super) fn parse_solve(&mut self) -> Option<Stmt> {
        let target = Ident {
            span: self.bump().span,
        };
        self.bump(); // `=`
        let keyword = self.bump().span; // `solve`
        self.solve_words = true;
        let header = self.parse_solve_header(keyword);
        self.solve_words = false;
        let Some((equation, from, to, initial)) = header else {
            self.recover_header(keyword);
            return None;
        };
        let mut end = initial.last().map_or(to.span, |i: &Initial| i.value.span);
        let mut options = Vec::new();
        match self.peek() {
            TokenKind::Colon => {
                let colon = self.bump();
                if self.peek() != TokenKind::Newline {
                    self.errors.push(
                        ParseError::new(
                            "the options of a `solve` must be on their own indented lines",
                            self.current().span,
                        )
                        .with_secondary(colon.span, "the block starts after this `:`")
                        .with_help("write `tolerance 1e-9` on its own line"),
                    );
                    self.recover_to_line_end();
                    return None;
                }
                self.bump();
                if self.peek() != TokenKind::Indent {
                    return self.fail_empty_block(colon.span, "`solve`");
                }
                self.bump();
                options = self.parse_solve_options(keyword)?;
                if let Some(last) = options.last() {
                    end = option_end(last);
                }
            }
            TokenKind::Newline => {
                self.bump();
            }
            TokenKind::Eof | TokenKind::Dedent => {}
            _ => {
                let err = self.unexpected_after(end, "end of line or `:`");
                self.errors.push(err);
                self.recover_to_line_end();
                return None;
            }
        }
        Some(Stmt {
            span: target.span.join(end),
            kind: StmtKind::Solve(Box::new(Solve {
                target,
                keyword,
                equation,
                from,
                to,
                initial,
                options,
            })),
        })
    }

    /// `eq from a to b (with x = v, ...)?`, with the words after `solve`.
    fn parse_solve_header(&mut self, keyword: Span) -> Option<(Ident, Expr, Expr, Vec<Initial>)> {
        let eq_tok = self.current();
        if eq_tok.kind != TokenKind::Ident {
            let message = format!(
                "expected the name of an equation after `solve`, found {}",
                self.describe(eq_tok)
            );
            self.errors.push(
                ParseError::new(message, eq_tok.span)
                    .with_secondary(keyword, "`solve` starts here")
                    .with_help("write `solve orbit from 0 s to 1 yr with ...`"),
            );
            return None;
        }
        self.bump();
        let equation = Ident { span: eq_tok.span };
        self.expect_word("from", eq_tok.span, "the time span starts with `from`")?;
        let from = self.expr_bp(0)?;
        self.expect_word("to", from.span, "the time span ends with `to`")?;
        let to = self.expr_bp(0)?;
        let mut initial = Vec::new();
        if self.word_is("with") {
            let with = self.bump();
            loop {
                let name_tok = self.current();
                if name_tok.kind != TokenKind::Ident {
                    let message = format!(
                        "expected a state variable name, found {}",
                        self.describe(name_tok)
                    );
                    self.errors.push(
                        ParseError::new(message, name_tok.span)
                            .with_secondary(with.span, "initial values follow `with`")
                            .with_help("write `with x = 1 m, v = 0 m/s`"),
                    );
                    return None;
                }
                self.bump();
                if self.peek() != TokenKind::Assign {
                    let message = format!(
                        "expected `=` after `{}`, found {}",
                        self.text(name_tok.span),
                        self.describe(self.current())
                    );
                    self.errors.push(
                        ParseError::new(message, self.current().span)
                            .with_secondary(name_tok.span, "the state variable")
                            .with_help(
                                "give each state variable its initial value: `with x = 1 m`",
                            ),
                    );
                    return None;
                }
                self.bump();
                let value = self.expr_bp(0)?;
                initial.push(Initial {
                    name: Ident {
                        span: name_tok.span,
                    },
                    value,
                });
                if self.peek() == TokenKind::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        Some((equation, from, to, initial))
    }

    /// The name `word` (a word of the `solve` grammar, not a keyword).
    fn expect_word(&mut self, word: &str, prev: Span, help: &str) -> Option<()> {
        if self.word_is(word) {
            self.bump();
            return Some(());
        }
        let found = self.current();
        self.errors.push(
            ParseError::new(
                format!("expected `{word}`, found {}", self.describe(found)),
                found.span,
            )
            .with_secondary(prev, "after this")
            .with_help(format!("{help}: `solve orbit from 0 s to 1 yr`")),
        );
        None
    }

    /// The option lines of a `solve` block, its `Indent` consumed, up to and
    /// including the `Dedent`. Counts one nesting level.
    fn parse_solve_options(&mut self, header: Span) -> Option<Vec<SolveOption>> {
        let saved = self.depth;
        self.depth = self.depth.saturating_add(1);
        if self.depth > MAX_DEPTH {
            self.depth = saved;
            self.skip_block_rest();
            return self.fail_block_too_deep(header);
        }
        let mut options = Vec::new();
        loop {
            self.depth = saved.saturating_add(1);
            match self.peek() {
                TokenKind::Eof => break,
                TokenKind::Newline => {
                    self.bump();
                    continue;
                }
                TokenKind::Dedent => {
                    self.bump();
                    break;
                }
                TokenKind::Indent => {
                    let t = self.bump();
                    self.errors.push(
                        ParseError::new("unexpected indentation", t.span)
                            .with_help("the options of a `solve` are aligned with each other"),
                    );
                    self.skip_block_rest();
                    continue;
                }
                _ => {}
            }
            let start = self.pos;
            self.solve_words = true;
            let option = self.parse_solve_option();
            self.solve_words = false;
            match option {
                Some(o) => options.push(o),
                None => self.recover_to_line_end(),
            }
            if self.pos == start {
                self.bump();
            }
        }
        self.depth = saved;
        Some(options)
    }

    fn parse_solve_option(&mut self) -> Option<SolveOption> {
        let tok = self.current();
        let word = if tok.kind == TokenKind::Ident {
            span_text(self.src, tok.span)
        } else {
            None
        };
        match word {
            Some("tolerance") => {
                self.bump();
                let value = self.expr_bp(0)?;
                self.expect_item_end(value.span)?;
                Some(SolveOption::Tolerance {
                    keyword: tok.span,
                    value,
                })
            }
            Some("conserve") => {
                self.bump();
                let name_tok = self.current();
                if name_tok.kind != TokenKind::Ident {
                    let message = format!(
                        "expected a name for the conserved quantity, found {}",
                        self.describe(name_tok)
                    );
                    self.errors.push(
                        ParseError::new(message, name_tok.span)
                            .with_secondary(tok.span, "`conserve` starts here")
                            .with_help("write `conserve energy = 0.5 * m * v^2 + ...`"),
                    );
                    return None;
                }
                self.bump();
                if self.peek() != TokenKind::Assign {
                    let message = format!(
                        "expected `=` after `{}`, found {}",
                        self.text(name_tok.span),
                        self.describe(self.current())
                    );
                    self.errors.push(
                        ParseError::new(message, self.current().span)
                            .with_secondary(name_tok.span, "the conserved quantity")
                            .with_help(
                                "write `conserve energy = <expression in the state variables>`",
                            ),
                    );
                    return None;
                }
                self.bump();
                let value = self.expr_bp(0)?;
                let within = if self.word_is("within") {
                    self.bump();
                    Some(self.expr_bp(0)?)
                } else {
                    None
                };
                self.expect_item_end(within.as_ref().map_or(value.span, |w| w.span))?;
                Some(SolveOption::Conserve {
                    keyword: tok.span,
                    name: Ident {
                        span: name_tok.span,
                    },
                    value,
                    within,
                })
            }
            _ => {
                let message = match word {
                    Some(w) => format!("unknown `solve` option `{w}`"),
                    None => format!("expected a `solve` option, found {}", self.describe(tok)),
                };
                self.errors
                    .push(ParseError::new(message, tok.span).with_help(
                        "the options are `tolerance 1e-9` and `conserve energy = <expression>`",
                    ));
                None
            }
        }
    }
}

fn option_end(o: &SolveOption) -> Span {
    match o {
        SolveOption::Tolerance { value, .. } => value.span,
        SolveOption::Conserve { value, within, .. } => {
            within.as_ref().map_or(value.span, |w| w.span)
        }
    }
}

// ----- S-expressions -----

fn text(src: &str, span: Span) -> &str {
    span_text(src, span).unwrap_or("<?>")
}

/// `(equation orbit (time t <type>) (: r <type>) (d r <expr>) (= a <expr>))`.
pub(crate) fn equation_sexpr(eq: &Equation, src: &str) -> String {
    let mut out = format!(
        "(equation {} (time {} {})",
        text(src, eq.name.span),
        text(src, eq.time.span),
        eq.time_ty.to_sexpr(src)
    );
    for item in &eq.items {
        out.push(' ');
        match item {
            EquationItem::State { name, ty, .. } => {
                out.push_str(&format!(
                    "(: {} {})",
                    text(src, name.span),
                    ty.to_sexpr(src)
                ));
            }
            EquationItem::Derivative { name, value, .. } => {
                out.push_str(&format!(
                    "(d {} {})",
                    text(src, name.span),
                    value.to_sexpr(src)
                ));
            }
            EquationItem::Local { name, value, .. } => {
                out.push_str(&format!(
                    "(= {} {})",
                    text(src, name.span),
                    value.to_sexpr(src)
                ));
            }
        }
    }
    out.push(')');
    out
}

/// `(solve sol orbit from to (with (r <expr>) ...) (tolerance <expr>) (conserve e <expr> [within]))`.
pub(crate) fn solve_sexpr(s: &Solve, src: &str) -> String {
    let mut out = format!(
        "(solve {} {} {} {}",
        text(src, s.target.span),
        text(src, s.equation.span),
        s.from.to_sexpr(src),
        s.to.to_sexpr(src)
    );
    if !s.initial.is_empty() {
        out.push_str(" (with");
        for i in &s.initial {
            out.push_str(&format!(
                " ({} {})",
                text(src, i.name.span),
                i.value.to_sexpr(src)
            ));
        }
        out.push(')');
    }
    for o in &s.options {
        out.push(' ');
        match o {
            SolveOption::Tolerance { value, .. } => {
                out.push_str(&format!("(tolerance {})", value.to_sexpr(src)));
            }
            SolveOption::Conserve {
                name,
                value,
                within,
                ..
            } => {
                out.push_str(&format!(
                    "(conserve {} {}",
                    text(src, name.span),
                    value.to_sexpr(src)
                ));
                if let Some(w) = within {
                    out.push_str(&format!(" (within {})", w.to_sexpr(src)));
                }
                out.push(')');
            }
        }
    }
    out.push(')');
    out
}
