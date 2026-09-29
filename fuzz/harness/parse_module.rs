//! The `parse_module` fuzz check, shared by the fuzz target
//! (`fuzz/fuzz_targets/parse_module.rs`) and a stable test in sky-ast
//! (`crates/sky-ast/tests/fuzz_parse_module.rs`), which both include this
//! file with `#[path]`. It depends only on sky-ast, sky-syntax, and sky-diag.
//!
//! The first input byte picks the mode:
//! - even: the rest is source text (UTF-8, lossy). It is lexed with the real
//!   lexer and, if the lexer accepts it, the tokens are parsed.
//! - odd: the rest encodes a source text and an arbitrary token sequence
//!   (see [`decode_tokens`]), fed straight to the parser: any kinds, any
//!   order, overlapping spans, spans off char boundaries, `Eof` anywhere or
//!   nowhere. If bit 1 of the first byte is also set, the spans are wild
//!   (inverted, past the end of the source, near `u32::MAX`).
//!
//! Invariants, checked on every input, on a thread with a
//! [`sky_ast::DEEP_STACK_BYTES`] stack ([`run`] uses [`with_deep_stack`];
//! the fuzz target uses one persistent worker thread of the same size):
//! - Lexing, parsing, printing, dropping, and rendering diagnostics never
//!   panic and never overflow the stack.
//! - Except with wild spans: every span in the AST (walked iteratively) and
//!   in every diagnostic satisfies `start <= end <= source.len()`, and, for
//!   lexed source, lies on char boundaries. The module span is
//!   `0..source.len()`.

use sky_ast::{
    parse_module, with_deep_stack, AssignTarget, Block, Expr, ExprKind, IndexItem, Kw, Module,
    Span, Stmt, StmtKind, Token, TokenKind, TypeExpr, UnitExpr,
};
use sky_diag::{Diagnostic, SourceFile};

/// What a run produced, for tests that want more than "didn't crash".
/// The fuzz target ignores it, hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct Outcome {
    /// `None` if the lexer rejected the source (lexed mode only).
    pub stmts: Option<usize>,
    /// Parse error messages, in order.
    pub errors: Vec<String>,
}

/// Runs one fuzz input inside [`with_deep_stack`]. This is what the stable
/// test calls; the fuzz target runs [`check`] on its own worker thread with
/// the same stack size instead, hence `allow(dead_code)`.
#[allow(dead_code)]
pub fn run(data: &[u8]) -> Outcome {
    with_deep_stack(|| check(data))
}

/// Checks one fuzz input on the current thread, which must have a
/// [`sky_ast::DEEP_STACK_BYTES`] stack.
pub fn check(data: &[u8]) -> Outcome {
    let Some((&mode, rest)) = data.split_first() else {
        return Outcome::default();
    };
    if mode & 1 == 0 {
        check_lexed(&String::from_utf8_lossy(rest))
    } else {
        let wild = mode & 2 != 0;
        let (src, tokens) = decode_tokens(rest, wild);
        let spans = if wild { Spans::Wild } else { Spans::Bounded };
        check_tokens(&src, &tokens, spans)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Spans {
    /// From the lexer: in bounds and on char boundaries.
    Lexed,
    /// Hand-made: in bounds, maybe off char boundaries.
    Bounded,
    /// Hand-made and unchecked; only "no panic" applies.
    Wild,
}

fn check_lexed(src: &str) -> Outcome {
    let file = SourceFile::new("fuzz.sky", src);
    match sky_syntax::lex(src) {
        Ok(lexed) => {
            for d in &lexed.warnings {
                check_diagnostic(src, &file, d, Spans::Lexed);
            }
            check_tokens(src, &lexed.tokens, Spans::Lexed)
        }
        Err(err) => {
            for d in err.diagnostics() {
                check_diagnostic(src, &file, d, Spans::Lexed);
            }
            Outcome::default()
        }
    }
}

fn check_tokens(src: &str, tokens: &[Token], spans: Spans) -> Outcome {
    let (module, errors) = parse_module(tokens, src);
    let _ = module.to_sexpr(src);
    if spans != Spans::Wild {
        let len = u32::try_from(src.len()).unwrap_or(u32::MAX);
        assert_eq!(module.span, Span { start: 0, end: len }, "module span");
        walk_module(&module, |span| check_span(src, span, spans));
    }
    let file = SourceFile::new("fuzz.sky", src);
    let messages = errors.iter().map(|e| e.message.clone()).collect();
    for err in errors {
        let d = err.into_diagnostic();
        if spans != Spans::Wild {
            check_diagnostic(src, &file, &d, spans);
        }
    }
    Outcome {
        stmts: Some(module.stmts.len()),
        errors: messages,
    }
}

fn check_diagnostic(src: &str, file: &SourceFile, d: &Diagnostic, spans: Spans) {
    check_span(src, d.primary.span, spans);
    for label in &d.secondary {
        check_span(src, label.span, spans);
    }
    let _ = d.display(file).to_string();
}

fn check_span(src: &str, span: Span, spans: Spans) {
    let (start, end) = (span.start as usize, span.end as usize);
    assert!(
        start <= end && end <= src.len(),
        "span {span:?} out of bounds (source is {} bytes)",
        src.len()
    );
    if spans == Spans::Lexed {
        assert!(
            src.is_char_boundary(start) && src.is_char_boundary(end),
            "span {span:?} not on char boundaries"
        );
    }
}

/// Calls `f` on every span in `module`. Iterative (an explicit stack), so
/// the walk itself can't overflow however deep the tree is.
fn walk_module(module: &Module, mut f: impl FnMut(Span)) {
    enum Node<'a> {
        Expr(&'a Expr),
        Unit(&'a UnitExpr),
        Item(&'a IndexItem),
        Target(&'a AssignTarget),
    }
    let mut stack: Vec<Node<'_>> = Vec::new();
    // Statements are walked with their own worklist too (blocks nest as
    // deep as MAX_DEPTH, and this walk must not be what overflows).
    let mut stmts: Vec<&Stmt> = module.stmts.iter().rev().collect();
    fn push_type<'a>(ty: &'a TypeExpr, stack: &mut Vec<Node<'a>>, f: &mut dyn FnMut(Span)) {
        f(ty.span);
        ty.tags.iter().for_each(|t| f(t.span));
        if let Some(shape) = &ty.shape {
            f(shape.span);
            f(shape.kind_span);
            shape.dims.iter().for_each(|d| f(d.span()));
        }
        if let Some(u) = &ty.unit {
            stack.push(Node::Unit(u));
        }
    }
    fn push_block<'a>(stmts: &mut Vec<&'a Stmt>, block: &'a Block, f: &mut dyn FnMut(Span)) {
        f(block.span);
        stmts.extend(block.stmts.iter().rev());
    }
    while let Some(stmt) = stmts.pop() {
        f(stmt.span);
        match &stmt.kind {
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (cond, block) in branches.iter().rev() {
                    stack.push(Node::Expr(cond));
                    push_block(&mut stmts, block, &mut f);
                }
                if let Some(block) = else_block {
                    push_block(&mut stmts, block, &mut f);
                }
            }
            StmtKind::While { cond, body } => {
                stack.push(Node::Expr(cond));
                push_block(&mut stmts, body, &mut f);
            }
            StmtKind::Def {
                name,
                params,
                ret,
                body,
            } => {
                f(name.span);
                for p in params {
                    f(p.span);
                    f(p.name.span);
                    if let Some(ty) = &p.ty {
                        push_type(ty, &mut stack, &mut f);
                    }
                }
                if let Some(ty) = ret {
                    push_type(ty, &mut stack, &mut f);
                }
                push_block(&mut stmts, body, &mut f);
            }
            StmtKind::Return(value) => stack.extend(value.iter().map(Node::Expr)),
            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Expr(e) => stack.push(Node::Expr(e)),
            StmtKind::Assign { target, value } => {
                stack.push(Node::Target(target));
                stack.push(Node::Expr(value));
            }
            StmtKind::CompoundAssign {
                target,
                op_span,
                value,
                ..
            } => {
                f(*op_span);
                stack.push(Node::Target(target));
                stack.push(Node::Expr(value));
            }
            StmtKind::Binding { name, ty, value } => {
                f(name.span);
                push_type(ty, &mut stack, &mut f);
                if let Some(v) = value {
                    stack.push(Node::Expr(v));
                }
            }
        }
    }
    while let Some(node) = stack.pop() {
        match node {
            Node::Expr(e) => {
                f(e.span);
                match &e.kind {
                    ExprKind::Literal(lit) => f(lit.span()),
                    ExprKind::Name(id) => f(id.span),
                    ExprKind::UnitLiteral { value, unit } => {
                        f(value.span());
                        stack.push(Node::Unit(unit));
                    }
                    ExprKind::Unary {
                        op_span, operand, ..
                    } => {
                        f(*op_span);
                        stack.push(Node::Expr(operand));
                    }
                    ExprKind::Binary {
                        op_span, lhs, rhs, ..
                    } => {
                        f(*op_span);
                        stack.push(Node::Expr(lhs));
                        stack.push(Node::Expr(rhs));
                    }
                    ExprKind::BoolOp {
                        operands, op_spans, ..
                    } => {
                        op_spans.iter().copied().for_each(&mut f);
                        stack.extend(operands.iter().map(Node::Expr));
                    }
                    ExprKind::Inverse { operand, op_span } => {
                        f(*op_span);
                        stack.push(Node::Expr(operand));
                    }
                    ExprKind::Array(elems) => stack.extend(elems.iter().map(Node::Expr)),
                    ExprKind::Call { callee, args } => {
                        stack.push(Node::Expr(callee));
                        stack.extend(args.iter().map(Node::Expr));
                    }
                    ExprKind::Index { base, items } => {
                        stack.push(Node::Expr(base));
                        stack.extend(items.iter().map(Node::Item));
                    }
                    ExprKind::Attribute { base, name } => {
                        f(name.span);
                        stack.push(Node::Expr(base));
                    }
                    ExprKind::Paren(inner) => stack.push(Node::Expr(inner)),
                    ExprKind::IfElse { then, cond, else_ } => {
                        stack.extend([then, cond, else_].map(|e| Node::Expr(e)));
                    }
                    ExprKind::As {
                        value,
                        unit,
                        as_span,
                    } => {
                        f(*as_span);
                        stack.push(Node::Expr(value));
                        stack.push(Node::Unit(unit));
                    }
                }
            }
            Node::Item(item) => {
                f(item.span());
                match item {
                    IndexItem::Expr(e) => stack.push(Node::Expr(e)),
                    IndexItem::Slice {
                        start,
                        stop,
                        colon_span,
                        ..
                    } => {
                        f(*colon_span);
                        stack.extend(start.iter().chain(stop.iter()).map(Node::Expr));
                    }
                }
            }
            Node::Target(target) => {
                f(target.span());
                match target {
                    AssignTarget::Name(id) => f(id.span),
                    AssignTarget::Index {
                        base,
                        items,
                        bracket_span,
                        ..
                    } => {
                        f(*bracket_span);
                        stack.push(Node::Expr(base));
                        stack.extend(items.iter().map(Node::Item));
                    }
                    AssignTarget::Attribute { base, name, .. } => {
                        f(name.span);
                        stack.push(Node::Expr(base));
                    }
                }
            }
            Node::Unit(u) => {
                f(u.span());
                match u {
                    UnitExpr::Name(id) => f(id.span),
                    UnitExpr::Mul { lhs, rhs, .. } | UnitExpr::Div { lhs, rhs, .. } => {
                        stack.push(Node::Unit(lhs));
                        stack.push(Node::Unit(rhs));
                    }
                    UnitExpr::Pow {
                        base,
                        exponent_span,
                        ..
                    } => {
                        f(*exponent_span);
                        stack.push(Node::Unit(base));
                    }
                }
            }
        }
    }
}

/// Every token kind, so a decoded byte can name any of them.
const KINDS: &[TokenKind] = {
    use TokenKind::*;
    &[
        Int,
        Float,
        Ident,
        Str,
        Newline,
        Indent,
        Dedent,
        Eof,
        LParen,
        RParen,
        LBracket,
        RBracket,
        Comma,
        Colon,
        Assign,
        Dot,
        Arrow,
        Plus,
        Minus,
        Star,
        Slash,
        Caret,
        At,
        Inverse,
        Partial,
        Nabla,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        PlusAssign,
        MinusAssign,
        StarAssign,
        SlashAssign,
        CaretAssign,
        AtAssign,
        Keyword(Kw::Def),
        Keyword(Kw::Return),
        Keyword(Kw::If),
        Keyword(Kw::Elif),
        Keyword(Kw::Else),
        Keyword(Kw::While),
        Keyword(Kw::For),
        Keyword(Kw::In),
        Keyword(Kw::As),
        Keyword(Kw::Import),
        Keyword(Kw::Pass),
        Keyword(Kw::Break),
        Keyword(Kw::Continue),
        Keyword(Kw::Equation),
        Keyword(Kw::Solve),
        Keyword(Kw::Field),
        Keyword(Kw::And),
        Keyword(Kw::Or),
        Keyword(Kw::Not),
        Keyword(Kw::True),
        Keyword(Kw::False),
    ]
};

/// Decodes `[len: u16 LE][len source bytes][token]*`, where each token is 4
/// bytes `[kind, start lo, start hi, width]`. The source is UTF-8-lossy, so
/// its length can differ from `len`. Nothing forces an `Eof` or any order.
///
/// Bounded spans: `start` is taken modulo `source.len() + 1` and the end is
/// `start + width`, clamped to the source, so `start <= end <= source.len()`
/// but spans can overlap, repeat, run backwards, and split characters.
///
/// Wild spans: `start` is used as is and `width` is a signed offset, so spans
/// can be inverted or past the end; a kind byte of `0xf0` or more puts the
/// span near `u32::MAX`.
fn decode_tokens(data: &[u8], wild: bool) -> (String, Vec<Token>) {
    let (len, rest) = match data {
        [lo, hi, rest @ ..] => (usize::from(u16::from_le_bytes([*lo, *hi])), rest),
        _ => (0, data),
    };
    let (src_bytes, token_bytes) = rest.split_at(len.min(rest.len()));
    let src = String::from_utf8_lossy(src_bytes).into_owned();
    let n = u32::try_from(src.len()).unwrap_or(u32::MAX);
    let tokens = token_bytes
        .chunks_exact(4)
        .map(|c| {
            let kind = KINDS[usize::from(c[0]) % KINDS.len()];
            let raw = u32::from(u16::from_le_bytes([c[1], c[2]]));
            let span = if wild {
                let start = if c[0] >= 0xf0 { u32::MAX - raw } else { raw };
                let offset = i64::from(c[3] as i8);
                let end = (i64::from(start) + offset).clamp(0, i64::from(u32::MAX));
                Span {
                    start,
                    end: end as u32,
                }
            } else {
                let start = raw % n.saturating_add(1);
                let end = start.saturating_add(u32::from(c[3])).min(n);
                Span { start, end }
            };
            Token { kind, span }
        })
        .collect();
    (src, tokens)
}
