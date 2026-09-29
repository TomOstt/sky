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

/// A literal token. Numbers and strings keep only their span: the value is
/// the token text, decoded by later phases (number parsing, string escapes).
/// A Bool has nothing to decode, so it carries its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(Span),
    Float(Span),
    Str(Span),
    /// `true` or `false`, the only Bool literals.
    Bool {
        value: bool,
        span: Span,
    },
}

impl Literal {
    pub fn span(&self) -> Span {
        match *self {
            Literal::Int(s) | Literal::Float(s) | Literal::Str(s) => s,
            Literal::Bool { span, .. } => span,
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

/// `and` / `or`: short-circuiting, so a separate node from [`BinaryOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolOp {
    And,
    Or,
}

impl BoolOp {
    /// The keyword, also used by the S-expression printer.
    pub fn keyword(self) -> &'static str {
        match self {
            BoolOp::And => "and",
            BoolOp::Or => "or",
        }
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
    /// `not x`. It binds more loosely than comparisons (`not a < b` is
    /// `not (a < b)`) and more tightly than `and`/`or`.
    Not,
}

impl UnaryOp {
    /// Name used by the S-expression printer.
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Pos => "pos",
            UnaryOp::Partial => "partial",
            UnaryOp::Nabla => "nabla",
            UnaryOp::Not => "not",
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

/// A type annotation, e.g. `PositiveDefinite Matrix[N, N] kg`: read as "a
/// positive-definite N by N matrix of kilograms".
///
/// ```text
/// type := tag* shape [unit] | tag* unit | "Scalar"
/// ```
///
/// A lone name is always a unit (`x: N` is newtons, `x: K` kelvin), never a
/// size variable: size variables only appear inside `[...]`. The parser does
/// not know which capitalized names are tags; the checker validates them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeExpr {
    /// Matrix-property names before the shape, in source order.
    pub tags: Vec<Ident>,
    /// `Vector[..]`, `Matrix[..]`, `Tensor[..]`, or `Scalar`. `None` for a
    /// bare unit (`x: m/s`), which means a scalar of that unit.
    pub shape: Option<ShapeAnn>,
    /// The element unit, after the shape.
    pub unit: Option<UnitExpr>,
    /// From the first tag or shape word through the last token of the type.
    pub span: Span,
}

/// Which shape word an annotation starts with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    Scalar,
    Vector,
    Matrix,
    Tensor,
}

impl ShapeKind {
    /// The source spelling, also used by the S-expression printer.
    pub fn name(self) -> &'static str {
        match self {
            ShapeKind::Scalar => "Scalar",
            ShapeKind::Vector => "Vector",
            ShapeKind::Matrix => "Matrix",
            ShapeKind::Tensor => "Tensor",
        }
    }

    /// The shape word spelled `word`, if it is one.
    pub fn from_name(word: &str) -> Option<ShapeKind> {
        match word {
            "Scalar" => Some(ShapeKind::Scalar),
            "Vector" => Some(ShapeKind::Vector),
            "Matrix" => Some(ShapeKind::Matrix),
            "Tensor" => Some(ShapeKind::Tensor),
            _ => None,
        }
    }
}

/// The matrix-property words of v0.1 (spec "Matrix tags"). The parser uses
/// them only to read `Symmetric m` as a misplaced property rather than a
/// unit called `Symmetric`; the checker owns their meaning
/// (`sky_types::Tag`, which a test keeps in sync with this list).
pub const TAG_NAMES: [&str; 4] = ["Symmetric", "PositiveDefinite", "Diagonal", "Orthogonal"];

/// The shape part of a type: `Matrix[N, 3]`, or a bare `Scalar`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeAnn {
    pub kind: ShapeKind,
    /// The shape word alone (`Matrix`).
    pub kind_span: Span,
    /// Empty for `Scalar`; otherwise at least one dimension.
    pub dims: Vec<DimExpr>,
    /// The shape word through its `]`.
    pub span: Span,
}

/// One dimension in `[...]` of a shape: an integer or a size-variable name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimExpr {
    /// A non-negative integer literal (the checker reads its value, so a
    /// huge one is a checker error, not a parse error).
    Int(Span),
    /// A size variable.
    Name(Ident),
}

impl DimExpr {
    pub fn span(&self) -> Span {
        match self {
            DimExpr::Int(s) => *s,
            DimExpr::Name(id) => id.span,
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
    /// `a and b and c` or `a or b`: a run of one short-circuiting operator
    /// at one level, as a single node. Evaluation is left to right and
    /// stops at the first operand that decides the result, which is the
    /// same as left-associative `((a and b) and c)`. Mixed runs nest:
    /// `a and b or c` is `Or[And[a, b], c]`.
    ///
    /// Flat rather than binary so a long chain doesn't make the tree (and
    /// every recursive pass over it) as deep as the chain is long.
    BoolOp {
        op: BoolOp,
        /// Two or more operands, in source order.
        operands: Vec<Expr>,
        /// The `and`/`or` keywords: `operands.len() - 1` spans.
        op_spans: Vec<Span>,
    },
    /// Postfix `⁻¹`. Its own node (not `x ^ -1`) because the type checker
    /// rewrites `A⁻¹ @ b` with a positive-definite `A` into a Cholesky solve.
    Inverse {
        operand: Box<Expr>,
        op_span: Span,
    },
    /// `[a, b, c]` or `[[a, b], [c, d]]`: an array literal, always at least
    /// one element. The parser does not decide the shape: a matrix is an
    /// `Array` whose elements are all `Array`s (its rows), and the checker
    /// reports mixed or deeper nesting. The span includes both brackets.
    Array(Vec<Expr>),
    /// `callee(args...)`
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// `base[items...]`, e.g. `A[i, j]`, `A[0, :]`, `A[1:3]`. Always at
    /// least one item. (Type annotations do not use this: see [`TypeExpr`].)
    Index {
        base: Box<Expr>,
        items: Vec<IndexItem>,
    },
    /// `base.name`
    Attribute {
        base: Box<Expr>,
        name: Ident,
    },
    /// `( inner )`, kept as a node so spans and a future CST stay faithful.
    Paren(Box<Expr>),
    /// `then if cond else else_`, Python's conditional expression. It binds
    /// more loosely than `or` (level 0 of the precedence table), so
    /// `a or b if c else d` is `(a or b) if c else d`, and it nests to the
    /// right in the `else` arm: `a if c else b if d else e` is
    /// `a if c else (b if d else e)`. `cond` must be a Bool of shape `()`
    /// (no truthiness); only the chosen arm is evaluated.
    IfElse {
        then: Box<Expr>,
        cond: Box<Expr>,
        else_: Box<Expr>,
    },
    /// `value as unit`, e.g. `v as km/h`: a display value, the number (or
    /// array) `value` written in `unit` instead of SI. It is the loosest
    /// construct of all (looser than `a if c else b`), so it covers a whole
    /// expression: `a + b as km` is `(a + b) as km`. It does not chain and
    /// only appears where an expression ends (a statement, a call
    /// argument, inside parentheses); the checker allows the result only
    /// as an argument of `print` or as an expression statement.
    As {
        value: Box<Expr>,
        /// Parsed with the unit-literal grammar (`parse_unit`). Boxed to
        /// keep `Expr` (and every recursive parser frame) small.
        unit: Box<UnitExpr>,
        /// The `as` keyword.
        as_span: Span,
    },
}

/// One item inside `[...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexItem {
    /// `i`
    Expr(Expr),
    /// `start:stop`, either side optional; `:` alone is the whole axis.
    /// There is no step (`a:b:c` is a parse error in v0.1).
    Slice {
        start: Option<Expr>,
        stop: Option<Expr>,
        colon_span: Span,
        span: Span,
    },
}

impl IndexItem {
    pub fn span(&self) -> Span {
        match self {
            IndexItem::Expr(e) => e.span,
            IndexItem::Slice { span, .. } => *span,
        }
    }

    /// The expression, or `None` for a slice. For readers of generic
    /// arguments (`Matrix[3, 4]`), where a slice is an error.
    pub fn as_expr(&self) -> Option<&Expr> {
        match self {
            IndexItem::Expr(e) => Some(e),
            IndexItem::Slice { .. } => None,
        }
    }
}

/// The left side of `=` or of a compound assignment. Only these forms parse
/// as targets; anything else (`f(x) = 1`, `(x) = 1`, `1 = x`) is a parse
/// error at the target's span.
///
/// The parser accepts index and attribute targets with exact spans; in v0.1
/// the checker rejects writes through them (arrays are immutable, and
/// nothing yet has assignable attributes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignTarget {
    /// `x`
    Name(Ident),
    /// `base[items...]`, e.g. `A[i, j]` or `A[0, :]`.
    Index {
        base: Box<Expr>,
        items: Vec<IndexItem>,
        /// `[` through `]`.
        bracket_span: Span,
        span: Span,
    },
    /// `base.name`
    Attribute {
        base: Box<Expr>,
        name: Ident,
        span: Span,
    },
}

impl AssignTarget {
    pub fn span(&self) -> Span {
        match self {
            AssignTarget::Name(id) => id.span,
            AssignTarget::Index { span, .. } | AssignTarget::Attribute { span, .. } => *span,
        }
    }
}

/// The operator of a compound assignment: the [`BinaryOp`]s that have one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompoundOp {
    /// `+=`
    Add,
    /// `-=`
    Sub,
    /// `*=`
    Mul,
    /// `/=`
    Div,
    /// `^=`
    Pow,
    /// `@=`
    MatMul,
}

impl CompoundOp {
    pub fn symbol(self) -> &'static str {
        match self {
            CompoundOp::Add => "+=",
            CompoundOp::Sub => "-=",
            CompoundOp::Mul => "*=",
            CompoundOp::Div => "/=",
            CompoundOp::Pow => "^=",
            CompoundOp::MatMul => "@=",
        }
    }

    /// The operator it applies, for the checker and interpreter.
    pub fn binary(self) -> BinaryOp {
        match self {
            CompoundOp::Add => BinaryOp::Add,
            CompoundOp::Sub => BinaryOp::Sub,
            CompoundOp::Mul => BinaryOp::Mul,
            CompoundOp::Div => BinaryOp::Div,
            CompoundOp::Pow => BinaryOp::Pow,
            CompoundOp::MatMul => BinaryOp::MatMul,
        }
    }
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
        target: AssignTarget,
        value: Expr,
    },
    /// `name: ty` or `name: ty = value`. Only a plain name can be annotated
    /// (`a.b: T = e` and `A[i]: T = e` are parse errors).
    Binding {
        name: Ident,
        ty: TypeExpr,
        value: Option<Expr>,
    },
    /// `target op= value`, e.g. `x += 1`. Its own node, not desugared to
    /// `x = x + 1`: the target is evaluated once and errors point at `+=`.
    CompoundAssign {
        target: AssignTarget,
        op: CompoundOp,
        /// The whole `+=` token.
        op_span: Span,
        value: Expr,
    },
    /// `if c: ... elif c2: ... else: ...`. Always at least one branch (the
    /// `if`, then each `elif` in order); `else_block` is the `else`. The
    /// statement's span runs from `if` to the end of its last block.
    If {
        branches: Vec<(Expr, Block)>,
        else_block: Option<Block>,
    },
    /// `while cond: body`. Loops share the enclosing scope, like `if`.
    While {
        cond: Expr,
        body: Block,
    },
    /// `def name(params) -> ret: body`. Parameter and return annotations
    /// are types, read like a binding's (`Vector[3] m/s`).
    Def {
        name: Ident,
        params: Vec<Param>,
        /// The `T` of `-> T`.
        ret: Option<TypeExpr>,
        body: Block,
    },
    /// `return` or `return value`.
    Return(Option<Expr>),
    Pass,
    Break,
    Continue,
}

/// An indented run of statements after a `:`. Never empty in a tree the
/// parser returned without errors (an empty block is a parse error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// From the start of the first statement to the end of the last (the
    /// zero-width `Indent`/`Dedent` tokens and newlines are not included).
    pub span: Span,
}

/// One parameter of a `def`: `name` or `name: ty`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: Ident,
    pub ty: Option<TypeExpr>,
    /// The name through the end of the annotation.
    pub span: Span,
}

/// A file's top-level statements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub stmts: Vec<Stmt>,
    /// `0..source.len()` (saturating at `u32::MAX`).
    pub span: Span,
}
