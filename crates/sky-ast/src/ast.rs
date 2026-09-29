//! AST node types. Every node carries a [`Span`] covering its full source
//! range, including delimiters (a call's span ends at its `)`).
//!
//! Nodes store spans, not text: text is recovered with [`span_text`] from the
//! source the tokens came from. This keeps nodes small and `Copy`-friendly
//! and leaves room for a lossless CST later (the CST will own trivia; the AST
//! stays a typed view with spans).

use sky_syntax::Span;

/// `source[span]`, or `None` if the span is out of bounds, inverted, or not on
/// a `char` boundary. Never panics.
pub fn span_text(source: &str, span: Span) -> Option<&str> {
    source.get(span.range())
}

/// A name, e.g. a variable, function, attribute, or unit name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub span: Span,
}

impl Ident {
    pub fn text<'s>(&self, source: &'s str) -> Option<&'s str> {
        span_text(source, self.span)
    }
}

/// A literal token. The value is the token text; it is not decoded here
/// (number parsing and string escapes belong to later phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(Span),
    Float(Span),
    Str(Span),
}

impl Literal {
    pub fn span(&self) -> Span {
        match *self {
            Literal::Int(s) | Literal::Float(s) | Literal::Str(s) => s,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Literal::Int(_) | Literal::Float(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `^`
    Pow,
    /// `@`
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Pow => "^",
            BinaryOp::MatMul => "@",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
        }
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `+x`
    Pos,
    /// `∂x` (a plain prefix node; no derivative-form semantics)
    Partial,
    /// `∇x` (a plain prefix node)
    Nabla,
}

impl UnaryOp {
    /// Name used by the S-expression printer.
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Pos => "pos",
            UnaryOp::Partial => "partial",
            UnaryOp::Nabla => "nabla",
        }
    }
}

/// The unit part of a unit literal, e.g. `m/s^2` in `9.8 m/s^2`.
///
/// `*` and `/` are left-associative; `^` binds tighter and applies to a
/// single unit name. Compound units always use `*`, never juxtaposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitExpr {
    Name(Ident),
    Mul {
        lhs: Box<UnitExpr>,
        rhs: Box<UnitExpr>,
        span: Span,
    },
    Div {
        lhs: Box<UnitExpr>,
        rhs: Box<UnitExpr>,
        span: Span,
    },
    /// `base^num` or `base^(num/den)`, e.g. `s^-2` or `m^(1/2)`.
    /// `den > 0` always (a zero denominator is a parse error); an integer
    /// exponent has `den == 1`. The fraction is stored as written, not
    /// reduced (`sky_units::Rational` normalizes).
    Pow {
        base: Box<UnitExpr>,
        num: i64,
        den: u64,
        /// Span of the exponent as written, e.g. `-2` in `s^-2` or `(1/2)`
        /// in `m^(1/2)`.
        exponent_span: Span,
        span: Span,
    },
}

impl UnitExpr {
    pub fn span(&self) -> Span {
        match self {
            UnitExpr::Name(id) => id.span,
            UnitExpr::Mul { span, .. }
            | UnitExpr::Div { span, .. }
            | UnitExpr::Pow { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprKind {
    Literal(Literal),
    Name(Ident),
    /// A numeric literal directly followed by a unit, e.g. `10 km`.
    /// `value` is always `Literal::Int` or `Literal::Float`.
    UnitLiteral {
        value: Literal,
        unit: UnitExpr,
    },
    Unary {
        op: UnaryOp,
        op_span: Span,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        op_span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Postfix `⁻¹`. Its own node (not `x ^ -1`) because the type checker
    /// rewrites `A⁻¹ @ b` with a positive-definite `A` into a Cholesky solve.
    Inverse {
        operand: Box<Expr>,
        op_span: Span,
    },
    /// `callee(args...)`
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// `base[indices...]`, also used for generic arguments like `Matrix[3, 4]`.
    Index {
        base: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// `base.name`
    Attribute {
        base: Box<Expr>,
        name: Ident,
    },
    /// `( inner )`, kept as a node so spans and a future CST stay faithful.
    Paren(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    pub kind: StmtKind,
    /// Covers the statement without its terminating newline.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StmtKind {
    Expr(Expr),
    /// `target = value`
    Assign {
        target: Ident,
        value: Expr,
    },
    /// `name: ty` or `name: ty = value`
    Binding {
        name: Ident,
        ty: Expr,
        value: Option<Expr>,
    },
}

/// A file's top-level statements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub stmts: Vec<Stmt>,
    /// `0..source.len()` (saturating at `u32::MAX`).
    pub span: Span,
}
