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

/// A mathematical function of `OpKind::Math`. The names are the ones the
/// checker and the tree-walking interpreter use for the same functions.
///
/// Every function takes finite numbers and gives a finite number or fails
/// at run time; it never gives NaN or infinity (the finite rule of
/// `sky_interp::arith`). "Fail" below is the run-time error, so the
/// operation's effect is [`Effect::Fail`] exactly for the functions that
/// can fail (see [`MathFn::can_fail`]).
///
/// | function | operands | result dimension | fails when |
/// |---|---|---|---|
/// | `sqrt` | 1, any dimension | halves every exponent | operand < 0 |
/// | `cbrt` | 1, any dimension | divides every exponent by 3 | never |
/// | `abs` | 1, any dimension | same | never |
/// | `exp` | 1, dimensionless | dimensionless | result overflows |
/// | `log`, `log10`, `log2` | 1, dimensionless | dimensionless | operand <= 0 |
/// | `sin`, `cos`, `tan`, `atan` | 1, dimensionless | dimensionless | never |
/// | `asin`, `acos` | 1, dimensionless | dimensionless | operand outside [-1, 1] |
/// | `atan2(y, x)` | 2, equal dimensions | dimensionless | never (`atan2(0, 0)` is 0) |
/// | `hypot(x, y)` | 2, equal dimensions | same | result overflows |
/// | `min`, `max` | 2, equal dimensions | same | never |
/// | `floor`, `ceil`, `round` | 1, any dimension | same | never |
///
/// `round` rounds halves away from zero. `tan` of a finite number is always
/// finite (no `f64` is exactly a multiple of pi/2). Operands are scalars in
/// v1, as for the other arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MathFn {
    Sqrt,
    Cbrt,
    Abs,
    Exp,
    Log,
    Log10,
    Log2,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Hypot,
    Min,
    Max,
    Floor,
    Ceil,
    Round,
}

impl MathFn {
    pub const ALL: [MathFn; 20] = [
        MathFn::Sqrt,
        MathFn::Cbrt,
        MathFn::Abs,
        MathFn::Exp,
        MathFn::Log,
        MathFn::Log10,
        MathFn::Log2,
        MathFn::Sin,
        MathFn::Cos,
        MathFn::Tan,
        MathFn::Asin,
        MathFn::Acos,
        MathFn::Atan,
        MathFn::Atan2,
        MathFn::Hypot,
        MathFn::Min,
        MathFn::Max,
        MathFn::Floor,
        MathFn::Ceil,
        MathFn::Round,
    ];

    /// The Sky name (`sqrt`).
    pub fn name(self) -> &'static str {
        match self {
            MathFn::Sqrt => "sqrt",
            MathFn::Cbrt => "cbrt",
            MathFn::Abs => "abs",
            MathFn::Exp => "exp",
            MathFn::Log => "log",
            MathFn::Log10 => "log10",
            MathFn::Log2 => "log2",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
            MathFn::Tan => "tan",
            MathFn::Asin => "asin",
            MathFn::Acos => "acos",
            MathFn::Atan => "atan",
            MathFn::Atan2 => "atan2",
            MathFn::Hypot => "hypot",
            MathFn::Min => "min",
            MathFn::Max => "max",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Round => "round",
        }
    }

    /// The text-format name: `math.sqrt`.
    pub fn mnemonic(self) -> &'static str {
        match self {
            MathFn::Sqrt => "math.sqrt",
            MathFn::Cbrt => "math.cbrt",
            MathFn::Abs => "math.abs",
            MathFn::Exp => "math.exp",
            MathFn::Log => "math.log",
            MathFn::Log10 => "math.log10",
            MathFn::Log2 => "math.log2",
            MathFn::Sin => "math.sin",
            MathFn::Cos => "math.cos",
            MathFn::Tan => "math.tan",
            MathFn::Asin => "math.asin",
            MathFn::Acos => "math.acos",
            MathFn::Atan => "math.atan",
            MathFn::Atan2 => "math.atan2",
            MathFn::Hypot => "math.hypot",
            MathFn::Min => "math.min",
            MathFn::Max => "math.max",
            MathFn::Floor => "math.floor",
            MathFn::Ceil => "math.ceil",
            MathFn::Round => "math.round",
        }
    }

    /// The function with this Sky name.
    pub fn from_name(name: &str) -> Option<MathFn> {
        MathFn::ALL.into_iter().find(|f| f.name() == name)
    }

    /// How many operands: 2 for `atan2`, `hypot`, `min`, `max`, else 1.
    pub fn arity(self) -> usize {
        match self {
            MathFn::Atan2 | MathFn::Hypot | MathFn::Min | MathFn::Max => 2,
            _ => 1,
        }
    }

    /// Whether the function can fail at run time (domain error or
    /// overflow), which is what makes its op [`Effect::Fail`].
    pub fn can_fail(self) -> bool {
        matches!(
            self,
            MathFn::Sqrt
                | MathFn::Exp
                | MathFn::Log
                | MathFn::Log10
                | MathFn::Log2
                | MathFn::Asin
                | MathFn::Acos
                | MathFn::Hypot
        )
    }

    /// `f(a, b) == f(b, a)` exactly. `min` and `max` are not claimed
    /// commutative: for `-0.0` and `0.0` the two orders could pick
    /// different zeros.
    pub fn commutative(self) -> bool {
        matches!(self, MathFn::Hypot)
    }
}

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
    /// `Aᵀ`: one operand, one result. Pure: it can't fail. (Running it
    /// needs arrays at run time, which v1 interpreters don't have yet.)
    Transpose,
    /// `A⁻¹ @ b` computed without forming `A⁻¹`: two operands (`A`, `b`),
    /// one result. Produced by the Cholesky rewrite pass.
    Solve {
        method: SolveMethod,
    },

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
            // A singular matrix or a zero scalar fails.
            OpKind::Inverse | OpKind::Solve { .. } => (fail, false, false, false),
            OpKind::MatMul => (fail, false, false, true),
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
