//! The SIR text format, parser half. [`crate::print`] shows an example.
//!
//! # Grammar
//!
//! Whitespace separates tokens; `//` starts a comment to the end of the
//! line. Operations are one per line: an operation ends at the end of its
//! line (or at the `}` closing its region), which is how `print` followed by
//! a line starting `%x = ...` is unambiguous.
//!
//! ```text
//! module   := "sir" "edition" YEAR function*
//! function := "fn" "@" NAME "(" [param ("," param)*] ")" "->" "(" [type ("," type)*] ")"
//!             ["effects" "(" effect ("," effect)* ")"] [span] "{" op* block* "}"
//! param    := "%" NAME ":" type
//! block    := "^" LABEL ["(" param ("," param)* ")"] ":" op*
//! op       := [values "="] MNEMONIC attr [operands] [":" type ("," type)*] region* [span]
//! attr     := "" | FLOAT | RATIONAL | "true" | "false" | STRING       (constants)
//!           | "[" RATIONAL "]"                                        (pow.const)
//!           | "[" "cholesky" "]"                                      (solve)
//!           | "[" INT "]"                                             (array.literal of a matrix: the row length)
//!           | "[" item ("," item)* "]"                                (slice)
//!           | "@" NAME "(" [operands] ")"                             (call; no operand list)
//! item     := "at" | ":" | "lo" ":" | ":" "hi" | "lo" ":" "hi"      (a position, or a range with the bounds it has)
//! region   := "{" block* "}"          (the function body's first block is implicit)
//! operands := value ("," value)*      values := value ("," value)*     value := "%" NAME
//! span     := "@" START ".." END      (byte offsets; omitted means 0..0)
//! type     := "num" "[" dim ";" shape [";" tags] [";" repr] "]"
//!           | "bool" | "str" | "nothing" | "unknown"
//! dim      := "1" | factor ("·" factor)*      factor := SYMBOL [exponent]   (as `Dimension` prints)
//! shape    := "(" [size ("," size)*] ")"      size := INT | NAME "#" ID | STRING "#" ID
//! tags     := "{" [TAG ("," TAG)*] "}"        repr := "exact" | "f32" | "f64" | "unchosen"
//! ```
//!
//! Names are letters, digits and `_`; function names may also use `<`, `>`
//! and `.`, or be a string (`@"<main>"`). A value named with digits only
//! (`%3`) is unnamed. A value must be defined before its use in the text,
//! but *scope* is not the parser's business: a use outside the region that
//! defines the value parses, and the verifier rejects it.
//!
//! # Guarantees
//!
//! Hand-written, one pass, never panics on any input. Input larger than
//! [`sky_syntax::MAX_SOURCE_BYTES`] is refused up front. The only recursion
//! is through nested regions, limited to [`MAX_REGION_DEPTH`] levels; every
//! number is parsed with checked arithmetic. The result is a `Module` in
//! canonical order (see [`crate::canon`]), built through
//! [`FunctionBuilder`]; it is not verified: run [`crate::verify`].

use std::collections::HashMap;
use std::fmt;

use sky_diag::{Diagnostic, Span};
use sky_syntax::MAX_SOURCE_BYTES;
use sky_types::{Shape, ShapeDim, ShapeVar, Tag, TagSet};
use sky_units::{BaseDim, Dimension, Rational};

use crate::builder::FunctionBuilder;
use crate::ir::{
    Edition, Effect, EffectSet, Module, NumberType, OpId, RegionId, Repr, Type, ValueId,
};
use crate::op::{BoolContext, CmpOp, MathFn, OpKind, SliceAxis, SolveMethod};
use crate::verify::MAX_REGION_DEPTH;

/// A syntax error and where it is (byte offsets into the text).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    /// An error diagnostic over the text that was parsed.
    pub fn to_diagnostic(&self) -> Diagnostic {
        Diagnostic::error(self.message.clone(), self.span)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (at bytes {}..{})",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}

type Res<T> = Result<T, ParseError>;

/// Parses SIR text. See the module docs for the grammar.
pub fn parse(src: &str) -> Result<Module, ParseError> {
    // Recursive over nested regions: see `verify`.
    sky_diag::with_deep_stack(|| parse_inner(src))
}

fn parse_inner(src: &str) -> Result<Module, ParseError> {
    if src.len() > MAX_SOURCE_BYTES {
        return Err(ParseError {
            message: format!("the text is larger than {MAX_SOURCE_BYTES} bytes"),
            span: Span::point(0),
        });
    }
    Parser { src, pos: 0 }.module()
}

/// The function being built and the names its values have so far.
struct FnState {
    b: FunctionBuilder,
    env: HashMap<String, ValueId>,
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn superscript_digit(c: char) -> Option<u32> {
    Some(match c {
        '⁰' => 0,
        '¹' => 1,
        '²' => 2,
        '³' => 3,
        '⁴' => 4,
        '⁵' => 5,
        '⁶' => 6,
        '⁷' => 7,
        '⁸' => 8,
        '⁹' => 9,
        _ => return None,
    })
}

impl<'a> Parser<'a> {
    // ----- cursor -----

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn err_at<T>(&self, start: usize, end: usize, message: impl Into<String>) -> Res<T> {
        // Keep the span inside the text, whatever the caller computed.
        let mut end = end.min(self.src.len());
        while !self.src.is_char_boundary(end) {
            end += 1;
        }
        let mut start = start.min(end);
        while !self.src.is_char_boundary(start) {
            start -= 1;
        }
        Err(ParseError {
            message: message.into(),
            span: Span {
                start: start as u32,
                end: end as u32,
            },
        })
    }

    /// An error at the next token: the next character after any spaces,
    /// or a zero-width span at the end of the line or the text.
    fn err<T>(&self, message: impl Into<String>) -> Res<T> {
        let rest = &self.src[self.pos..];
        let start = self.pos + rest.len() - rest.trim_start_matches([' ', '\t', '\r']).len();
        let end = match self.src[start..].chars().next() {
            None | Some('\n') => start,
            Some(c) => start + c.len_utf8(),
        };
        self.err_at(start, end, message)
    }

    /// Skips spaces, tabs, carriage returns and comments, but not newlines.
    fn skip_inline(&mut self) {
        loop {
            match self.peek() {
                Some(' ' | '\t' | '\r') => {
                    self.pos += 1;
                }
                Some('/') if self.src[self.pos..].starts_with("//") => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.pos += c.len_utf8();
                    }
                }
                _ => return,
            }
        }
    }

    /// Skips all whitespace, newlines and comments.
    fn skip_ws(&mut self) {
        loop {
            self.skip_inline();
            if self.peek() == Some('\n') {
                self.pos += 1;
            } else {
                return;
            }
        }
    }

    fn expect(&mut self, c: char) -> Res<()> {
        self.skip_inline();
        if self.eat(c) {
            Ok(())
        } else {
            self.err(format!("expected `{c}`"))
        }
    }

    /// True if `c` comes next, after any whitespace (position unchanged).
    fn next_is_after_ws(&mut self, c: char) -> bool {
        let save = self.pos;
        self.skip_ws();
        let found = self.peek() == Some(c);
        self.pos = save;
        found
    }

    /// A run of characters satisfying `pred`, possibly empty.
    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> &'a str {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if !pred(c) {
                break;
            }
            self.pos += c.len_utf8();
        }
        &self.src[start..self.pos]
    }

    fn word(&mut self) -> &'a str {
        self.take_while(|c| c.is_alphanumeric() || c == '_' || c == '.')
    }

    fn keyword(&mut self, kw: &str) -> Res<()> {
        self.skip_inline();
        let start = self.pos;
        if self.word() == kw {
            Ok(())
        } else {
            self.err_at(start, self.pos.max(start), format!("expected `{kw}`"))
        }
    }

    // ----- numbers and strings -----

    fn uint(&mut self) -> Res<u64> {
        let start = self.pos;
        let digits = self.take_while(|c| c.is_ascii_digit());
        if digits.is_empty() {
            return self.err("expected a number");
        }
        digits
            .parse::<u64>()
            .or_else(|_| self.err_at(start, self.pos, "number too large"))
    }

    fn int(&mut self) -> Res<i64> {
        let start = self.pos;
        self.eat('-');
        let digits = self.take_while(|c| c.is_ascii_digit());
        if digits.is_empty() {
            return self.err("expected a number");
        }
        self.src[start..self.pos]
            .parse::<i64>()
            .or_else(|_| self.err_at(start, self.pos, "number too large"))
    }

    fn rational(&mut self) -> Res<Rational> {
        let start = self.pos;
        let num = self.int()?;
        let den = if self.eat('/') { self.int()? } else { 1 };
        Rational::checked_new(num, den).or_else(|e| self.err_at(start, self.pos, e.to_string()))
    }

    fn float_bits(&mut self) -> Res<u64> {
        let start = self.pos;
        let text =
            self.take_while(|c| c.is_ascii_hexdigit() || matches!(c, 'x' | 'X' | '.' | '+' | '-'));
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            return u64::from_str_radix(hex, 16)
                .or_else(|_| self.err_at(start, self.pos, "expected up to 16 hex digits"));
        }
        if text.is_empty()
            || !text
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
        {
            return self.err_at(
                start,
                self.pos.max(start + 1),
                "expected a decimal or `0x` float",
            );
        }
        match text.parse::<f64>() {
            Ok(x) => Ok(x.to_bits()),
            Err(_) => self.err_at(start, self.pos, "malformed float"),
        }
    }

    fn string(&mut self) -> Res<String> {
        self.skip_inline();
        let start = self.pos;
        if !self.eat('"') {
            return self.err("expected a string");
        }
        let mut out = String::new();
        loop {
            let at = self.pos;
            match self.bump() {
                None | Some('\n') => return self.err_at(start, at, "unterminated string"),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some('"') => out.push('"'),
                    Some('u') => {
                        if !self.eat('{') {
                            return self.err("expected `{` after `\\u`");
                        }
                        let hex = self.take_while(|c| c.is_ascii_hexdigit());
                        if !self.eat('}') {
                            return self.err("expected `}`");
                        }
                        match u32::from_str_radix(hex, 16).ok().and_then(char::from_u32) {
                            Some(c) => out.push(c),
                            None => return self.err_at(at, self.pos, "invalid unicode escape"),
                        }
                    }
                    _ => return self.err_at(at, self.pos, "unknown escape"),
                },
                Some(c) => out.push(c),
            }
        }
    }

    fn span(&mut self) -> Res<Span> {
        let start = self.pos;
        let a = self.uint()?;
        if !self.src[self.pos..].starts_with("..") {
            return self.err("expected `..` in a span");
        }
        self.pos += 2;
        let b = self.uint()?;
        match (u32::try_from(a), u32::try_from(b)) {
            (Ok(start), Ok(end)) => Ok(Span { start, end }),
            _ => self.err_at(start, self.pos, "span offset too large"),
        }
    }

    /// An optional `@ start..end`; absent means `0..0`.
    fn opt_span(&mut self) -> Res<Span> {
        self.skip_inline();
        if self.eat('@') {
            self.skip_inline();
            self.span()
        } else {
            Ok(Span { start: 0, end: 0 })
        }
    }

    // ----- module and functions -----

    fn module(&mut self) -> Res<Module> {
        self.skip_ws();
        self.keyword("sir")?;
        self.keyword("edition")?;
        self.skip_inline();
        let start = self.pos;
        let year = self.uint()?;
        let Some(edition) = u16::try_from(year).ok().and_then(Edition::from_year) else {
            return self.err_at(start, self.pos, format!("unknown edition {year}"));
        };
        let mut m = Module {
            edition,
            functions: Vec::new(),
        };
        loop {
            self.skip_ws();
            if self.peek().is_none() {
                return Ok(m);
            }
            let f = self.function()?;
            m.functions.push(f);
        }
    }

    fn fn_name(&mut self) -> Res<String> {
        if self.peek() == Some('"') {
            return self.string();
        }
        let name = self.take_while(|c| is_name_char(c) || matches!(c, '<' | '>' | '.'));
        if name.is_empty() {
            return self.err("expected a function name");
        }
        Ok(name.to_string())
    }

    fn function(&mut self) -> Res<crate::ir::Function> {
        self.keyword("fn")?;
        self.expect('@')?;
        let name = self.fn_name()?;
        self.expect('(')?;
        let mut params: Vec<(String, Type, usize)> = Vec::new();
        self.skip_inline();
        if !self.eat(')') {
            loop {
                self.skip_inline();
                let at = self.pos;
                let tok = self.value_token()?;
                self.expect(':')?;
                let ty = self.ty()?;
                params.push((tok, ty, at));
                self.skip_inline();
                if self.eat(',') {
                    continue;
                }
                self.expect(')')?;
                break;
            }
        }
        self.expect('-')?;
        self.expect('>')?;
        self.expect('(')?;
        let mut results = Vec::new();
        self.skip_inline();
        if !self.eat(')') {
            loop {
                results.push(self.ty()?);
                self.skip_inline();
                if self.eat(',') {
                    continue;
                }
                self.expect(')')?;
                break;
            }
        }
        let mut effects = EffectSet::PURE;
        self.skip_inline();
        let save = self.pos;
        if self.word() == "effects" {
            self.expect('(')?;
            loop {
                self.skip_inline();
                let at = self.pos;
                let e = match self.word() {
                    "io" => Effect::Io,
                    "fail" => Effect::Fail,
                    "call" => Effect::Call,
                    _ => {
                        return self.err_at(
                            at,
                            self.pos.max(at + 1),
                            "expected `io`, `fail` or `call`",
                        )
                    }
                };
                effects = effects.with(e);
                self.skip_inline();
                if self.eat(',') {
                    continue;
                }
                self.expect(')')?;
                break;
            }
        } else {
            self.pos = save;
        }
        let span = self.opt_span()?;
        self.expect('{')?;

        let types = params.iter().map(|(_, t, _)| t.clone()).collect();
        let mut b = FunctionBuilder::new(name, types, results, span);
        b.set_effects(effects);
        let mut fs = FnState {
            b,
            env: HashMap::new(),
        };
        let ids = fs.b.params();
        for ((tok, _, at), id) in params.iter().zip(ids) {
            self.define(&mut fs, tok, id, *at)?;
        }
        // The entry block is implicit; more blocks (an error the verifier
        // reports) start with `^`.
        self.ops(&mut fs, 0)?;
        loop {
            self.skip_ws();
            if self.eat('}') {
                break;
            }
            if self.peek() == Some('^') {
                self.block_header(&mut fs, RegionId(0))?;
                self.ops(&mut fs, 0)?;
            } else {
                return self.err("expected an operation, a `^` block, or `}`");
            }
        }
        Ok(fs.b.finish())
    }

    // ----- values -----

    /// The name after a `%`.
    fn value_token(&mut self) -> Res<String> {
        self.skip_inline();
        if !self.eat('%') {
            return self.err("expected a value like `%x`");
        }
        let tok = self.take_while(is_name_char);
        if tok.is_empty() {
            return self.err("expected a value name after `%`");
        }
        Ok(tok.to_string())
    }

    fn define(&self, fs: &mut FnState, tok: &str, id: ValueId, at: usize) -> Res<()> {
        if fs.env.insert(tok.to_string(), id).is_some() {
            return self.err_at(at, at + 1 + tok.len(), format!("`%{tok}` is defined twice"));
        }
        if !tok.chars().all(|c| c.is_ascii_digit()) {
            fs.b.set_name(id, tok);
        }
        Ok(())
    }

    fn use_value(&mut self, fs: &FnState) -> Res<ValueId> {
        self.skip_inline();
        let at = self.pos;
        let tok = self.value_token()?;
        match fs.env.get(&tok) {
            Some(&v) => Ok(v),
            None => self.err_at(
                at,
                self.pos,
                format!("`%{tok}` is used before it is defined"),
            ),
        }
    }

    /// Zero or more comma-separated values on this line.
    fn operands(&mut self, fs: &FnState) -> Res<Vec<ValueId>> {
        let mut out = Vec::new();
        self.skip_inline();
        if self.peek() != Some('%') {
            return Ok(out);
        }
        loop {
            out.push(self.use_value(fs)?);
            self.skip_inline();
            if !self.eat(',') {
                return Ok(out);
            }
        }
    }

    // ----- blocks and operations -----

    /// `^label(%a: T, ...):` in `region`; the new block becomes the
    /// insertion point.
    fn block_header(&mut self, fs: &mut FnState, region: RegionId) -> Res<()> {
        self.expect('^')?;
        self.take_while(is_name_char);
        let mut args: Vec<(String, Type, usize)> = Vec::new();
        self.skip_inline();
        if self.eat('(') {
            self.skip_inline();
            if !self.eat(')') {
                loop {
                    self.skip_inline();
                    let at = self.pos;
                    let tok = self.value_token()?;
                    self.expect(':')?;
                    let ty = self.ty()?;
                    args.push((tok, ty, at));
                    self.skip_inline();
                    if self.eat(',') {
                        continue;
                    }
                    self.expect(')')?;
                    break;
                }
            }
        }
        self.expect(':')?;
        let types = args.iter().map(|(_, t, _)| t.clone()).collect();
        let block = fs.b.add_block(region, types);
        fs.b.set_insertion_block(block);
        let ids = fs.b.func().block(block).args.clone();
        for ((tok, _, at), id) in args.iter().zip(ids) {
            self.define(fs, tok, id, *at)?;
        }
        Ok(())
    }

    /// Operations up to a `}`, a `^` or the end of the text.
    fn ops(&mut self, fs: &mut FnState, depth: usize) -> Res<()> {
        loop {
            self.skip_ws();
            match self.peek() {
                None | Some('}' | '^') => return Ok(()),
                _ => self.op(fs, depth)?,
            }
        }
    }

    fn region(&mut self, fs: &mut FnState, op: OpId, depth: usize) -> Res<()> {
        self.skip_ws();
        if depth >= MAX_REGION_DEPTH {
            return self.err(format!(
                "regions are nested too deeply (the limit is {MAX_REGION_DEPTH} levels)"
            ));
        }
        self.expect('{')?;
        let r = fs.b.add_region(op);
        loop {
            self.skip_ws();
            if self.eat('}') {
                return Ok(());
            }
            if self.peek() != Some('^') {
                return self.err("expected a `^` block or `}`");
            }
            self.block_header(fs, r)?;
            self.ops(fs, depth)?;
        }
    }

    fn op(&mut self, fs: &mut FnState, depth: usize) -> Res<()> {
        // Result names.
        let mut results: Vec<(String, usize)> = Vec::new();
        if self.peek() == Some('%') {
            loop {
                self.skip_inline();
                let at = self.pos;
                results.push((self.value_token()?, at));
                self.skip_inline();
                if self.eat(',') {
                    continue;
                }
                self.expect('=')?;
                break;
            }
        }
        self.skip_inline();
        let word_at = self.pos;
        let word = self.word();
        let mut operands: Option<Vec<ValueId>> = None;
        let kind = match word {
            "const.float" => {
                self.skip_inline();
                OpKind::ConstFloat {
                    bits: self.float_bits()?,
                }
            }
            "const.rational" => {
                self.skip_inline();
                OpKind::ConstRational(self.rational()?)
            }
            "const.bool" => {
                self.skip_inline();
                let at = self.pos;
                match self.word() {
                    "true" => OpKind::ConstBool(true),
                    "false" => OpKind::ConstBool(false),
                    _ => {
                        return self.err_at(at, self.pos.max(at + 1), "expected `true` or `false`")
                    }
                }
            }
            "const.str" => OpKind::ConstStr(self.string()?),
            "const.nothing" => OpKind::ConstNothing,
            "show" => {
                self.skip_inline();
                OpKind::Show {
                    unit: self.string()?,
                }
            }
            "add" => OpKind::Add,
            "sub" => OpKind::Sub,
            "mul" => OpKind::Mul,
            "div" => OpKind::Div,
            "neg" => OpKind::Neg,
            "pow" => OpKind::Pow,
            "pow.const" => {
                self.expect('[')?;
                self.skip_inline();
                let exponent = self.rational()?;
                self.expect(']')?;
                OpKind::PowConst { exponent }
            }
            "inverse" => OpKind::Inverse,
            "matmul" => OpKind::MatMul,
            "transpose" => OpKind::Transpose,
            "solve" => {
                self.expect('[')?;
                self.skip_inline();
                let at = self.pos;
                let method = match self.word() {
                    "cholesky" => SolveMethod::Cholesky,
                    _ => return self.err_at(at, self.pos.max(at + 1), "expected `cholesky`"),
                };
                self.expect(']')?;
                OpKind::Solve { method }
            }
            "array.literal" => {
                self.skip_inline();
                let cols = if self.peek() == Some('[') {
                    self.expect('[')?;
                    self.skip_inline();
                    let c = self.uint()?;
                    self.expect(']')?;
                    Some(c)
                } else {
                    None
                };
                OpKind::ArrayLiteral { cols }
            }
            "index" => OpKind::Index,
            "slice" => {
                self.expect('[')?;
                let mut axes = Vec::new();
                loop {
                    axes.push(self.slice_axis()?);
                    self.skip_inline();
                    if !self.eat(',') {
                        break;
                    }
                }
                self.expect(']')?;
                OpKind::Slice { axes }
            }
            "zeros" => OpKind::Zeros,
            "identity" => OpKind::Identity,
            "len" => OpKind::Len,
            "cmp.eq" => OpKind::Cmp(CmpOp::Eq),
            "cmp.ne" => OpKind::Cmp(CmpOp::Ne),
            "cmp.lt" => OpKind::Cmp(CmpOp::Lt),
            "cmp.le" => OpKind::Cmp(CmpOp::Le),
            "cmp.gt" => OpKind::Cmp(CmpOp::Gt),
            "cmp.ge" => OpKind::Cmp(CmpOp::Ge),
            "not" => OpKind::Not,
            "expect.bool" => {
                self.expect('[')?;
                self.skip_inline();
                let at = self.pos;
                let word = self.word();
                let Some(context) = BoolContext::ALL.into_iter().find(|c| c.keyword() == word)
                else {
                    return self.err_at(
                        at,
                        self.pos.max(at + 1),
                        "expected `and`, `or`, `not`, `if` or `while`",
                    );
                };
                self.expect(']')?;
                OpKind::ExpectBool { context }
            }
            "if" => OpKind::If,
            "while" => OpKind::While,
            "yield" => OpKind::Yield,
            "condition" => OpKind::Condition,
            "return" => OpKind::Return,
            "print" => OpKind::Print,
            "call" => {
                self.expect('@')?;
                let callee = self.fn_name()?;
                self.expect('(')?;
                operands = Some(self.operands(fs)?);
                self.expect(')')?;
                OpKind::Call { callee }
            }
            "" => return self.err("expected an operation"),
            other => match MathFn::ALL.into_iter().find(|f| f.mnemonic() == other) {
                Some(f) => OpKind::Math(f),
                None => {
                    return self.err_at(word_at, self.pos, format!("unknown operation `{other}`"));
                }
            },
        };
        let operands = match operands {
            Some(o) => o,
            None => self.operands(fs)?,
        };

        // Result types.
        self.skip_inline();
        let colon_at = self.pos;
        let mut tys = Vec::new();
        if self.eat(':') {
            loop {
                tys.push(self.ty()?);
                self.skip_inline();
                if !self.eat(',') {
                    break;
                }
            }
        }
        if tys.len() != results.len() {
            let msg = if tys.is_empty() {
                format!(
                    "`{}` has {} result(s) and needs `: types`",
                    word,
                    results.len()
                )
            } else {
                format!(
                    "{} result type(s) for {} result value(s)",
                    tys.len(),
                    results.len()
                )
            };
            return self.err_at(colon_at, self.pos, msg);
        }

        // The operation exists before its regions are read, so the regions'
        // contents are its children.
        let span_placeholder = Span { start: 0, end: 0 };
        let here = fs.b.insertion_block();
        let (op, ids) = fs.b.op(kind, operands, tys, span_placeholder);
        for ((tok, at), id) in results.iter().zip(ids) {
            self.define(fs, tok, id, *at)?;
        }
        while self.next_is_after_ws('{') {
            self.region(fs, op, depth + 1)?;
        }
        fs.b.set_insertion_block(here);
        let span = self.opt_span()?;
        fs.b.set_op_span(op, span);

        self.skip_inline();
        match self.peek() {
            None | Some('\n' | '}') => Ok(()),
            Some(_) => self.err("expected the end of the line"),
        }
    }

    /// One item of a `slice`'s `[...]`: `at`, `:`, `lo:`, `:hi` or `lo:hi`.
    fn slice_axis(&mut self) -> Res<SliceAxis> {
        self.skip_inline();
        let start = self.pos;
        let first = self.word();
        if first == "at" {
            return Ok(SliceAxis::At);
        }
        let has_start = match first {
            "" => false,
            "lo" => true,
            _ => {
                return self.err_at(
                    start,
                    self.pos,
                    "expected `at`, `:`, `lo:`, `:hi` or `lo:hi`",
                )
            }
        };
        self.expect(':')?;
        let at = self.pos;
        let has_stop = match self.word() {
            "" => false,
            "hi" => true,
            _ => return self.err_at(at, self.pos, "expected `hi` or the end of the item"),
        };
        Ok(SliceAxis::Range {
            start: has_start,
            stop: has_stop,
        })
    }

    // ----- types -----

    fn ty(&mut self) -> Res<Type> {
        self.skip_inline();
        let start = self.pos;
        match self.word() {
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            "nothing" => Ok(Type::Nothing),
            "unknown" => Ok(Type::Unknown),
            "num" => self.number_type().map(Type::Number),
            _ => self.err_at(start, self.pos.max(start + 1), "expected a type"),
        }
    }

    fn number_type(&mut self) -> Res<NumberType> {
        self.expect('[')?;
        let dim = self.dimension()?;
        self.expect(';')?;
        let shape = self.shape()?;
        let mut tags: Option<TagSet> = None;
        let mut repr: Option<Repr> = None;
        loop {
            self.skip_inline();
            if self.eat(']') {
                break;
            }
            if !self.eat(';') {
                return self.err("expected `;` or `]`");
            }
            self.skip_inline();
            let at = self.pos;
            if self.peek() == Some('{') {
                if tags.replace(self.tags()?).is_some() {
                    return self.err_at(at, self.pos, "tags given twice");
                }
            } else {
                let r = match self.word() {
                    "exact" => Repr::Exact,
                    "f32" => Repr::F32,
                    "f64" => Repr::F64,
                    "unchosen" => Repr::Unchosen,
                    _ => {
                        return self.err_at(
                            at,
                            self.pos.max(at + 1),
                            "expected tags `{..}` or a representation (`exact`, `f32`, `f64`)",
                        )
                    }
                };
                if repr.replace(r).is_some() {
                    return self.err_at(at, self.pos, "representation given twice");
                }
            }
        }
        Ok(NumberType {
            dim,
            shape,
            tags: tags.unwrap_or(TagSet::EMPTY),
            repr: repr.unwrap_or_default(),
        })
    }

    fn tags(&mut self) -> Res<TagSet> {
        self.expect('{')?;
        let mut set = TagSet::EMPTY;
        self.skip_inline();
        if self.eat('}') {
            return Ok(set);
        }
        loop {
            self.skip_inline();
            let at = self.pos;
            let name = self.take_while(is_name_char);
            match Tag::from_name(name) {
                Some(t) => set = set.with(t),
                None => return self.err_at(at, self.pos.max(at + 1), "expected a tag name"),
            }
            self.skip_inline();
            if self.eat(',') {
                continue;
            }
            self.expect('}')?;
            return Ok(set);
        }
    }

    fn shape(&mut self) -> Res<Shape> {
        self.expect('(')?;
        let mut dims = Vec::new();
        self.skip_inline();
        if self.eat(')') {
            return Ok(Shape::scalar());
        }
        loop {
            self.skip_inline();
            dims.push(self.size()?);
            self.skip_inline();
            if self.eat(',') {
                continue;
            }
            self.expect(')')?;
            return Ok(Shape::new(dims));
        }
    }

    fn size(&mut self) -> Res<ShapeDim> {
        let start = self.pos;
        let name = if self.peek() == Some('"') {
            self.string()?
        } else {
            self.take_while(is_name_char).to_string()
        };
        if self.eat('#') {
            let id = self.uint()?;
            return Ok(ShapeDim::Var(ShapeVar::from_parts(id, name)));
        }
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_digit()) {
            return match name.parse::<u64>() {
                Ok(n) => Ok(ShapeDim::Known(n)),
                Err(_) => self.err_at(start, self.pos, "size too large"),
            };
        }
        self.err_at(
            start,
            self.pos.max(start + 1),
            "expected a size (`3`) or a shape variable (`N#0`)",
        )
    }

    fn dimension(&mut self) -> Res<Dimension> {
        self.skip_inline();
        if self.peek() == Some('1') {
            self.pos += 1;
            return Ok(Dimension::DIMENSIONLESS);
        }
        let mut exps = [Rational::ZERO; 7];
        let mut seen = [false; 7];
        loop {
            let at = self.pos;
            let Some(c) = self.bump() else {
                return self.err("expected a dimension");
            };
            let Some(i) = BaseDim::ALL
                .iter()
                .position(|b| b.symbol().chars().eq(std::iter::once(c)))
            else {
                return self.err_at(
                    at,
                    self.pos,
                    "expected a dimension symbol (L, M, T, I, Θ, N, J) or `1`",
                );
            };
            if std::mem::replace(&mut seen[i], true) {
                return self.err_at(at, self.pos, "a dimension symbol appears twice");
            }
            exps[i] = self.exponent()?;
            if !self.eat('·') {
                return Ok(Dimension::from_exponents(exps));
            }
        }
    }

    /// `⁻²`, `^(1/2)`, or nothing (exponent 1).
    fn exponent(&mut self) -> Res<Rational> {
        let start = self.pos;
        if self.eat('^') {
            self.expect('(')?;
            self.skip_inline();
            let r = self.rational()?;
            self.expect(')')?;
            return Ok(r);
        }
        let neg = self.eat('⁻');
        let mut n: i64 = 0;
        let mut any = false;
        while let Some(d) = self.peek().and_then(superscript_digit) {
            self.bump();
            any = true;
            n = match n.checked_mul(10).and_then(|n| n.checked_add(i64::from(d))) {
                Some(n) => n,
                None => return self.err_at(start, self.pos, "exponent too large"),
            };
        }
        match (any, neg) {
            (true, neg) => Ok(Rational::integer(if neg { -n } else { n })),
            (false, false) => Ok(Rational::ONE),
            (false, true) => self.err_at(start, self.pos, "expected exponent digits after `⁻`"),
        }
    }
}
