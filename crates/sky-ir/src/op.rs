//! The operations of SIR v1 (level L1) and their properties.
//!
//! Each variant documents its operands, results, regions and meaning. The
//! lowering, the verifier, the text format and the reference interpreter
//! all follow these descriptions, so this file is the specification of SIR
//! semantics until the spec gains a chapter for it.
//!
//! Numbers follow the interpreter's rules (spec "Running programs"): SI
//! values in `f64`, always finite. An operation whose result would be
//! infinite or NaN (overflow, division by zero, a non-real power) fails at
//! run time, which is why most arithmetic has [`Effect::Fail`].

use sky_units::Rational;

use crate::ir::{Effect, EffectSet};

/// A comparison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub fn mnemonic(self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        }
    }
}

/// How `solve` solves `A x = b`. Chosen by a strategy pass, never written
/// by users in v1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SolveMethod {
    /// `A` is PositiveDefinite: factor `A = L·Lᵀ`, then two triangular
    /// solves. `A⁻¹` is never formed.
    Cholesky,
}

impl SolveMethod {
    pub fn mnemonic(self) -> &'static str {
        match self {
            SolveMethod::Cholesky => "cholesky",
        }
    }
}

/// One item of a `slice`'s subscript: what the source wrote for one axis,
/// in `A[i, lo:hi, :]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SliceAxis {
    /// A position `i`: the axis is dropped from the result. One operand.
    At,
    /// A range `lo:hi`: the axis stays, with the entries `lo <= i < hi`.
    /// `start` says whether the range has a lower bound (else it starts at
    /// 0) and `stop` whether it has an upper bound (else it runs to the end
    /// of the axis). One operand per bound present, `lo` first.
    Range { start: bool, stop: bool },
}

impl SliceAxis {
    /// How many operands this item takes.
    pub fn operands(self) -> usize {
        match self {
            SliceAxis::At => 1,
            SliceAxis::Range { start, stop } => usize::from(start) + usize::from(stop),
        }
    }

    /// The item as the text format writes it: `at`, `:`, `lo:`, `:hi`, `lo:hi`.
    pub fn text(self) -> &'static str {
        match self {
            SliceAxis::At => "at",
            SliceAxis::Range {
                start: false,
                stop: false,
            } => ":",
            SliceAxis::Range {
                start: true,
                stop: false,
            } => "lo:",
            SliceAxis::Range {
                start: false,
                stop: true,
            } => ":hi",
            SliceAxis::Range {
                start: true,
                stop: true,
            } => "lo:hi",
        }
    }
}

/// The math functions (`sqrt`, `exp`, `sin`, ...): the one list, with each
/// function's arity, dimension rule and failure behaviour, lives in
/// sky-units and is shared by the checker, both interpreters and SIR.
pub use sky_units::MathFn;

/// An operation kind. "Operands" and "results" below are the `operands`
/// and `results` of the [`crate::OpData`].
#[derive(Clone, PartialEq, Debug)]
pub enum OpKind {
    // ----- constants: no operands, one result -----
    /// A number in SI, stored as `f64` bits (so the kind is `Eq`-friendly
    /// and prints exactly). The result's type carries the dimension.
    ConstFloat {
        bits: u64,
    },
    /// An exact rational in SI. Lowering may use it for literals that are
    /// exact; passes keep it exact as long as they can (roadmap idea 9).
    /// When a number is finally needed it is `numer as f64 / denom as f64`
    /// (one correctly rounded division, as the interpreter computes `1/3`).
    ConstRational(Rational),
    ConstBool(bool),
    ConstStr(String),
    /// The value of `print(...)` or a call to a function without a return
    /// value, when a program uses it. Result type `Nothing`.
    ConstNothing,

    // ----- arithmetic on numbers -----
    /// `a + b`, `a - b`, `a * b`, `a / b`: two operands, one result.
    /// Operands of `add`/`sub` have equal dimensions and shapes (the
    /// checker guarantees it). Elementwise on arrays of equal shape; a
    /// scalar operand of `mul`/`div` scales an array.
    Add,
    Sub,
    Mul,
    Div,
    /// `-a`: one operand, one result.
    Neg,
    /// `a ^ e` with a constant exponent: one operand (`a`), one result. The
    /// result's dimension is `a`'s raised to `exponent`.
    PowConst {
        exponent: Rational,
    },
    /// `a ^ e` with a computed exponent: two operands (`a`, `e`), both
    /// dimensionless scalars (the checker requires a constant exponent for
    /// a base with a unit), one result.
    Pow,
    /// `a⁻¹` / `inv(a)`: one operand, one result. A scalar's reciprocal
    /// (dimension inverted) or a square matrix's inverse.
    Inverse,
    /// A mathematical function: `arity()` operands (see [`MathFn`] for the
    /// operands, dimension rule and failures of each), one result.
    Math(MathFn),
    /// `A @ B`: two operands, one result.
    MatMul,
    /// `Aᵀ`: one operand, one result. Pure: it can't fail. A scalar is its
    /// own transpose and a matrix `M×N` becomes `N×M`; a vector or a
    /// higher-rank tensor has no transpose.
    Transpose,
    /// `A⁻¹ @ b` computed without forming `A⁻¹`: two operands (`A`, `b`),
    /// one result. Produced by the Cholesky rewrite pass.
    Solve {
        method: SolveMethod,
    },

    // ----- display -----
    /// `v as unit`: one operand (a number, in SI), one result of type
    /// `Str`: the text of the operand written in `unit`, formatted as
    /// `print` would format `v as unit` (`90 km/h`). `unit` is the unit's
    /// source text, in the grammar a number takes (`km/h`, `kg*m^2/s^2`).
    /// The result is only ever an operand of `print`.
    ///
    /// This is an operation of its own, and not a `div` by a factor
    /// followed by a `print` of the number, because a unit conversion is
    /// not one division (`km/h` applies its factors one at a time, and the
    /// shown number is the shortest one that reads back as the stored
    /// value) and the unit belongs in the output. All of that is
    /// `sky_interp::show_unit_text`, the function `sky run` calls too. It
    /// fails on overflow, so it is not pure.
    Show {
        unit: String,
    },

    // ----- arrays -----
    //
    // Arrays are dense, row-major, and never empty: a vector has one axis
    // and a matrix two (v0.1 has no higher ranks), every entry is a finite
    // number, and all entries share the array's dimension. A number's type
    // says whether it is an array: `shape` is `()` for a scalar. The
    // arithmetic operations above (`add`, `sub`, `mul`, `div`, `neg`,
    // `inverse`, `matmul`, `transpose`, `solve`) already work on arrays;
    // these operations make arrays and take them apart.
    /// `[a, b, c]` and `[[a, b], [c, d]]`: `N >= 1` operands, one result.
    /// The operands are scalars of one dimension, in row-major order.
    /// `cols` says how they are arranged: `None` is a vector `(N)`, and
    /// `Some(c)` a matrix whose rows have `c` entries, `(N/c, c)` (so `c`
    /// divides `N`; `[[1, 2, 3]]` is a matrix of one row, not a vector).
    /// The result has the operands' dimension, and its type restates the
    /// shape, so every dimension of it is a number. The arrangement is an
    /// attribute as well because the result's type is `Unknown` when an
    /// entry's type is (`[x, 1]` for an unannotated parameter `x`): the
    /// shape is then only in `cols`. Pure.
    ///
    /// The result's tags are whatever the checker proved about the literal
    /// (a declaration such as `P: PositiveDefinite Matrix[2, 2] = [[4, 1],
    /// [1, 3]]` is proven from the entries); the verifier only checks that
    /// they are closed and sit on a square matrix.
    ArrayLiteral {
        cols: Option<u64>,
    },
    /// `A[i]`, `A[i, j]`: an operand `A` (a vector or matrix) followed by
    /// `k` positions, one per leading axis (`1 <= k <=` the rank), one
    /// result. Each position is a dimensionless scalar that must be a
    /// non-negative integer less than its axis's length. The result has
    /// `A`'s dimension and the trailing `rank - k` axes: a scalar for
    /// `v[i]` and `A[i, j]`, a row for `A[i]`. No tags. Effect `Fail`: a
    /// position that is not a valid index, or is out of bounds, fails at
    /// run time (a constant position is checked by the checker before).
    Index,
    /// `A[i, lo:hi, :]`: an operand `A`, then the operands of each item in
    /// order (see [`SliceAxis`]), one result. `axes` has at most one item
    /// per axis (trailing axes without an item stay whole); a position drops
    /// its axis and a range keeps it, with length `hi - lo`. Bounds are
    /// dimensionless non-negative integers, `lo <= hi <=` the axis length,
    /// and a range is never empty (`lo < hi`). Same result rules and effect
    /// as [`OpKind::Index`]; lowering emits `index` when no item is a
    /// range.
    Slice {
        axes: Vec<SliceAxis>,
    },
    /// `zeros(n)`, `zeros(m, n)`: one or two operands (the sizes,
    /// dimensionless positive integers), one result: a dimensionless array
    /// of zeros of that shape (a vector for one size, a matrix for two).
    /// When the sizes are known the result's shape says which. Effect
    /// `Fail`: a size that is not a positive integer, or an array of more
    /// than `MAX_ARRAY_ELEMENTS` entries, fails at run time.
    Zeros,
    /// `identity(n)`: one operand (the size, a dimensionless positive
    /// integer), one result: the dimensionless `n × n` identity matrix.
    /// Same failures as [`OpKind::Zeros`].
    Identity,
    /// `len(x)`: one operand (a vector or matrix), one result: the length of
    /// its first axis, a dimensionless scalar. Lowering writes a constant
    /// instead when the first axis's length is known from the operand's
    /// type (evaluating the operand still happens, for its failures and
    /// effects); this operation covers a length that is only known at run
    /// time (a function's `Vector[N]` parameter) or not at all (an operand
    /// of unknown type). Effect `Fail`: only an operand that is not an
    /// array fails, which a typed operand never is.
    Len,

    // ----- comparison and logic -----
    /// Two operands (scalars of equal dimension, two Bools, or two
    /// strings for `eq`/`ne`), one Bool result.
    Cmp(CmpOp),
    /// One Bool operand, one Bool result. (`and`/`or` short-circuit, so they
    /// lower to `if`, not to operations of their own.)
    Not,

    // ----- structured control flow -----
    /// `if`: one Bool operand (the condition) and two regions, `then` and
    /// `else`. Each region's block takes no arguments and ends in `yield`
    /// with one value per result of the `if`; the `if`'s results are the
    /// values the taken region yields.
    If,
    /// `while`, in the style of MLIR's `scf.while`. Operands are the
    /// initial loop-carried values; results are their final values. Two
    /// regions, `before` and `after`, whose blocks take one argument per
    /// loop-carried value:
    /// * `before` computes the condition and ends in `condition`, whose
    ///   first operand is the Bool and whose other operands are forwarded;
    /// * if the Bool is true, `after` runs with the forwarded values as its
    ///   arguments and ends in `yield` with the next loop-carried values,
    ///   which start `before` again;
    /// * if it is false, the loop ends and its results are the forwarded
    ///   values.
    ///
    /// Sky's `break` and `continue` lower to Bool flags carried through the
    /// loop, so the region structure stays single-entry, single-exit.
    While,
    /// Terminator of an `if` region or a `while` `after` region: its
    /// operands are the region's results.
    Yield,
    /// Terminator of a `while` `before` region: a Bool, then the values
    /// forwarded to `after` or to the loop's results.
    Condition,

    // ----- functions -----
    /// A call to the module function named `callee`: operands are the
    /// arguments, one per parameter; results are exactly the callee's
    /// declared results (none for a function without a return value; a
    /// program that uses such a call's value materialises `const.nothing`).
    Call {
        callee: String,
    },
    /// Terminator of a function body's entry block: operands are the
    /// function's results. Early `return`s in Sky lower to flags, so this
    /// is always the body's last operation, and a `return` anywhere else
    /// (inside an `if` or `while` region) is malformed.
    Return,

    // ----- effects -----
    /// `print(a, b, ...)`: writes the operands separated by spaces and a
    /// newline, as the interpreter does. No results.
    Print,
}

/// What passes may assume about an operation kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OpProperties {
    pub effects: EffectSet,
    /// Ends a block: `yield`, `condition`, `return`.
    pub terminator: bool,
    /// `op(a, b) == op(b, a)`, exactly (in floating point too).
    pub commutative: bool,
    /// `op(op(a, b), c) == op(a, op(b, c))` over the reals. Not exact in
    /// floating point, so a pass may reassociate only when the program's
    /// numerical contract allows it (never, in v1).
    pub associative_real: bool,
}

impl OpKind {
    /// The name in the text format: `add`, `const.float`, `cmp.lt`, ...
    /// Kinds with attributes print them after the name (see the printer).
    pub fn mnemonic(&self) -> &'static str {
        match self {
            OpKind::ConstFloat { .. } => "const.float",
            OpKind::ConstRational(_) => "const.rational",
            OpKind::ConstBool(_) => "const.bool",
            OpKind::ConstStr(_) => "const.str",
            OpKind::ConstNothing => "const.nothing",
            OpKind::Add => "add",
            OpKind::Sub => "sub",
            OpKind::Mul => "mul",
            OpKind::Div => "div",
            OpKind::Math(f) => f.mnemonic(),
            OpKind::Neg => "neg",
            OpKind::PowConst { .. } => "pow.const",
            OpKind::Pow => "pow",
            OpKind::Inverse => "inverse",
            OpKind::MatMul => "matmul",
            OpKind::Transpose => "transpose",
            OpKind::Solve { .. } => "solve",
            OpKind::Show { .. } => "show",

            OpKind::ArrayLiteral { .. } => "array.literal",
            OpKind::Index => "index",
            OpKind::Slice { .. } => "slice",
            OpKind::Zeros => "zeros",
            OpKind::Identity => "identity",
            OpKind::Len => "len",
            OpKind::Cmp(c) => match c {
                CmpOp::Eq => "cmp.eq",
                CmpOp::Ne => "cmp.ne",
                CmpOp::Lt => "cmp.lt",
                CmpOp::Le => "cmp.le",
                CmpOp::Gt => "cmp.gt",
                CmpOp::Ge => "cmp.ge",
            },
            OpKind::Not => "not",
            OpKind::If => "if",
            OpKind::While => "while",
            OpKind::Yield => "yield",
            OpKind::Condition => "condition",
            OpKind::Call { .. } => "call",
            OpKind::Return => "return",
            OpKind::Print => "print",
        }
    }

    pub fn properties(&self) -> OpProperties {
        let fail = EffectSet::of(Effect::Fail);
        let (effects, terminator, commutative, associative_real) = match self {
            OpKind::ConstFloat { .. }
            | OpKind::ConstRational(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstStr(_)
            | OpKind::ConstNothing
            | OpKind::Neg
            | OpKind::Transpose
            | OpKind::ArrayLiteral { .. }
            | OpKind::Cmp(_)
            | OpKind::Not => (EffectSet::PURE, false, false, false),
            // Overflow to infinity fails (numbers are always finite).
            OpKind::Add | OpKind::Mul => (fail, false, true, true),
            OpKind::Sub | OpKind::Div | OpKind::PowConst { .. } | OpKind::Pow => {
                (fail, false, false, false)
            }
            OpKind::Math(f) => {
                let e = if f.can_fail() { fail } else { EffectSet::PURE };
                (e, false, f.commutative(), false)
            }
            // Converting to a tiny unit can overflow.
            OpKind::Show { .. } => (fail, false, false, false),
            // A singular matrix or a zero scalar fails.
            OpKind::Inverse | OpKind::Solve { .. } => (fail, false, false, false),
            OpKind::MatMul => (fail, false, false, true),
            // A bad or out-of-bounds index, or a bad size, fails.
            OpKind::Index
            | OpKind::Slice { .. }
            | OpKind::Zeros
            | OpKind::Identity
            | OpKind::Len => (fail, false, false, false),
            // Control flow has the effects of its regions; the verifier and
            // passes compute those from the contents.
            OpKind::If | OpKind::While => (EffectSet::PURE, false, false, false),
            OpKind::Yield | OpKind::Condition | OpKind::Return => {
                (EffectSet::PURE, true, false, false)
            }
            OpKind::Call { .. } => (EffectSet::of(Effect::Call), false, false, false),
            OpKind::Print => (EffectSet::of(Effect::Io), false, false, false),
        };
        OpProperties {
            effects,
            terminator,
            commutative,
            associative_real,
        }
    }
}
