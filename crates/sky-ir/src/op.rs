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

pub use sky_types::{Algorithm, Side, SolveMethod};
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

/// The source construct a Bool was demanded by, for [`OpKind::ExpectBool`]:
/// the wording of the run-time error ("`or` needs a Bool operand, found a
/// number") names it, exactly as `sky run` does. `elif` and the conditional
/// expression `a if c else b` are `if`, as they are in `sky run`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoolContext {
    And,
    Or,
    Not,
    If,
    While,
}

impl BoolContext {
    pub const ALL: [BoolContext; 5] = [
        BoolContext::And,
        BoolContext::Or,
        BoolContext::Not,
        BoolContext::If,
        BoolContext::While,
    ];

    /// The keyword the error message uses, and the text format's attribute.
    pub fn keyword(self) -> &'static str {
        match self {
            BoolContext::And => "and",
            BoolContext::Or => "or",
            BoolContext::Not => "not",
            BoolContext::If => "if",
            BoolContext::While => "while",
        }
    }
}

/// The source construct a number was demanded by, for
/// [`OpKind::ExpectNumber`]: the wording of the run-time error ("`+` needs a
/// number, found a Bool") names it, exactly as `sky run` does.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NumberContext {
    /// The operand of a unary `+`.
    Pos,
    /// The argument of `inv(...)` (the `x⁻¹` spelling is worded by
    /// `inverse` itself).
    Inv,
}

impl NumberContext {
    pub const ALL: [NumberContext; 2] = [NumberContext::Pos, NumberContext::Inv];

    /// The text the error message uses.
    pub fn keyword(self) -> &'static str {
        match self {
            NumberContext::Pos => "+",
            NumberContext::Inv => "inv",
        }
    }

    /// The text format's attribute.
    pub fn text(self) -> &'static str {
        match self {
            NumberContext::Pos => "pos",
            NumberContext::Inv => "inv",
        }
    }
}

/// Which check of a `range` an [`OpKind::ExpectRange`] makes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RangePart {
    /// A bound or step: a dimensionless whole number of magnitude at most
    /// 2^53.
    Bound,
    /// A step: not zero.
    Step,
}

impl RangePart {
    pub const ALL: [RangePart; 2] = [RangePart::Bound, RangePart::Step];

    /// The text format's attribute.
    pub fn text(self) -> &'static str {
        match self {
            RangePart::Bound => "bound",
            RangePart::Step => "step",
        }
    }
}

/// Why a literal has no value, for [`OpKind::LiteralError`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LiteralError {
    /// The number does not fit in an `f64` ("number too large").
    TooLarge,
    /// The number fits, but its value in SI (the unit's factors applied)
    /// does not: "arithmetic overflow".
    Overflow,
}

impl LiteralError {
    pub const ALL: [LiteralError; 2] = [LiteralError::TooLarge, LiteralError::Overflow];

    /// The text format's attribute.
    pub fn text(self) -> &'static str {
        match self {
            LiteralError::TooLarge => "too_large",
            LiteralError::Overflow => "overflow",
        }
    }
}

/// Which contract an [`OpKind::Assert`] checks: `assert`, or the `requires`
/// and `ensures` of a function (the AST's own enum).
pub use sky_ast::AssertKind;

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
    /// `literal.error[too_large]`, `literal.error[overflow]`: no operands,
    /// one result, typed as the literal it stands for. A number literal
    /// (`1e999`) or a unit literal whose SI value (`1e300 ms/nm^2/μg`) is
    /// not a finite `f64` has no value, and `sky run` reports that only if the
    /// literal is evaluated; a `const.float` cannot hold it, so lowering
    /// writes this instead, and it fails when it runs with `sky run`'s error
    /// ("number too large", "arithmetic overflow") at the operation's span,
    /// which lowering sets to the span `sky run` reports. Effect `Fail`.
    LiteralError {
        error: LiteralError,
    },
    /// `input "A"`: no operands, one result, of the type the program
    /// declared for the name (a value-less declaration such as
    /// `A: PositiveDefinite Matrix[N, N]`; the checker takes its shape, tags
    /// and unit on trust). Nothing in v0.1 supplies an input, so running it
    /// always fails with "`A` is declared but has no value", at the
    /// declaration: the program runs up to there and stops, as `sky run`
    /// does. The result's type is what later operations are typed against.
    /// Effect `Fail`, so no pass removes or moves it past output.
    Input {
        name: String,
    },

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
    /// `norm(x)`: one operand (a number, vector or matrix), one result: the
    /// Euclidean length of a vector (the Frobenius norm of a matrix, the
    /// absolute value of a scalar), a scalar of the operand's dimension. The
    /// evaluation is `sky_interp::vecfn::norm`, the function `sky run` calls
    /// too, scaled for extreme entries so that a vector of `1e200`s has a
    /// norm. Effect `Fail`: a norm can overflow the finite range, and an
    /// operand that is not a number fails with the interpreter's error at
    /// the operand's span.
    Norm,
    /// `dot(a, b)`: two operands (vectors of one length), one result: their
    /// inner product, a scalar in the product of the operands' dimensions.
    /// Evaluated by `sky_interp::vecfn::dot`. Effect `Fail`: the sum can
    /// overflow, and operands that are not two vectors of one length fail
    /// with the interpreter's error (the length mismatch labels both
    /// operands' spans). It takes no matrices: that is `matmul`.
    Dot,
    /// `Aᵀ`: one operand, one result. Pure: it can't fail. A scalar is its
    /// own transpose and a matrix `M×N` becomes `N×M`; a vector or a
    /// higher-rank tensor has no transpose.
    Transpose,
    /// `A⁻¹ @ b` computed without forming `A⁻¹`: two operands (`A`, `b`),
    /// one result of `b`'s shape (`b` is a vector or a matrix with as many
    /// rows as `A`) and the dimension of `b` over `A`'s. Produced by the
    /// algorithm-choosing rewrites (Cholesky for a PositiveDefinite `A`, LU
    /// for any square `A`, division for a Diagonal one).
    Solve {
        method: SolveMethod,
    },
    /// `X @ A⁻¹` computed without forming `A⁻¹` (a transposed solve): two
    /// operands (`X`, `A`), one result of `X`'s shape (`X` is a vector or a
    /// matrix with as many columns as `A`) and the dimension of `X` over
    /// `A`'s. A vector `x` is a row: the result is the vector `y` with
    /// `y @ A = x`. The methods are those of [`OpKind::Solve`], applied to
    /// `Aᵀ` without forming it.
    SolveRight {
        method: SolveMethod,
    },
    /// `matmul` with attributes: two operands, one result. `ta` and `tb`
    /// say that the matrix operand is transposed **without materializing the
    /// transpose**: `matmul[ta] A, B` is `matmul(transpose(A), B)`, and the
    /// result type is that product's. A flag may only be set on an operand
    /// that is a matrix (a vector has no transpose). `algorithm` is the
    /// algorithm class (see [`Algorithm`]); `Auto` leaves it to the engine.
    ///
    /// **This is the contract a fast kernel implements**: the reference
    /// interpreter materializes the transpose with the same function as
    /// `transpose` and multiplies with the same function as `matmul`, so a
    /// kernel that reads `A` with swapped strides and accumulates in the
    /// same order gives the same bits. Plain `matmul` is the case with no
    /// attribute; a pass emits `matmul[...]` only when it has one to record.
    MatMulEx {
        ta: bool,
        tb: bool,
        algorithm: Algorithm,
    },
    /// `D @ X` or `X @ D` for a Diagonal `D`, computed as a scaling of
    /// rows or columns: two operands (`D`, `X` for [`Side::Left`], `X`, `D`
    /// for [`Side::Right`]), one result of `X`'s shape (`X` a vector or a
    /// matrix that fits). **Exactly the bits of `matmul`**: entry `(i, j)`
    /// is `0.0 + dᵢ·xᵢⱼ` (or `0.0 + xᵢⱼ·dⱼ`), because every other term of
    /// the sum is a product with an exact zero, which adds `±0` to a sum
    /// that started at `+0.0`. O(n²) instead of O(n³). The operand's
    /// Diagonal tag is a promise the interpreter verifies (an off-diagonal
    /// entry that is not zero is a run-time error).
    DiagMul {
        side: Side,
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

    // ----- fusion -----
    /// A fused elementwise kernel: a tree of elementwise array operations
    /// (`add`, `sub`, `mul`, `div`, `neg`) computed one element at a time,
    /// so the intermediate arrays are never built. It is what the `fuse`
    /// pass makes; the loop over the elements is implied, and no machine
    /// code is chosen (a backend emits one kernel per `fused`).
    ///
    /// * **Operands**: `k >= 1` values, each an array or a scalar. All the
    ///   arrays have one shape, and at least one operand is an array. A
    ///   scalar operand is broadcast: every element sees the same value (as
    ///   in the scaling `A * s`).
    /// * **Result**: one array of that shape; its dimension is the dimension
    ///   of the value the body yields.
    /// * **Region**: one, with one block whose arguments are the per-element
    ///   values, one per operand in order, each a scalar of the operand's
    ///   dimension (a scalar operand's argument has the operand's own
    ///   type). The block holds scalar operations and ends in a `yield` of
    ///   one value: the result's element. The body is self-contained: its
    ///   operations use only its own arguments and results, and none may
    ///   have a region, print, or call. Every operation in it is pure or can
    ///   fail, and the verifier enforces all of that.
    /// * **Meaning**: element `i` of the result is what the body computes
    ///   when its arguments are element `i` of each array operand (and the
    ///   scalar operands as they are). Each body operation is evaluated
    ///   exactly as the operation of the same kind is on numbers, except
    ///   `neg`, which is `0 - x` as on arrays (a zero stays `+0`, where a
    ///   scalar `neg` of `0` gives `-0`), so that a fused chain is
    ///   bit-identical to the array operations it replaced.
    /// * **Failure order**: the unfused chain computes one operation at a
    ///   time over all the elements, so the failure it reports is that of
    ///   the *first operation in program order* that fails at any element.
    ///   `fused` has the same rule, whatever the order a backend visits
    ///   elements in: if several body operations fail at some elements, the
    ///   error is the one of the earliest failing operation *in body order*
    ///   (the error's message and span belong to the operation, not to the
    ///   element). A backend may stop as soon as it knows no earlier
    ///   operation can fail, and must not stop before.
    /// * **Effects**: those of the body: `Fail` if any operation in it can
    ///   fail, else none (like `if` and `while`, the contents decide, so
    ///   [`OpKind::properties`] says pure and the passes look inside).
    Fused,

    /// `assume a`: one operand, one result: the operand's value, unchanged,
    /// typed with the tags the callee's parameter promises. A tag on a
    /// parameter (`K: PositiveDefinite Matrix[3, 3]`) is a promise the
    /// checker takes on trust at a call, and that `sky run` checks only where
    /// it is used (the Cholesky solve fails with "the matrix is not positive
    /// definite" if it does not hold), so an argument the checker could not
    /// prove tagged (a computed matrix) is passed on as it is; this operation
    /// only changes its type, so that the call verifies. It checks nothing
    /// and cannot fail: the operations that rely on a tag check it
    /// themselves, with the tree-walker's messages. The result has the
    /// operand's dimension, shape and representation and any tags at all
    /// (closed, on a square matrix). Pure.
    Assume,

    // ----- comparison and logic -----
    /// Two operands (scalars of equal dimension, two Bools, or two
    /// strings for `eq`/`ne`), one Bool result.
    Cmp(CmpOp),
    /// One Bool operand, one Bool result. (`and`/`or` short-circuit, so they
    /// lower to `if`, not to operations of their own.)
    Not,

    /// `expect.bool[kw] a`: one operand, one Bool result. The run-time check
    /// of an operand the checker could not type (its type is `unknown`, from
    /// an unannotated parameter) where the source needs a Bool: an operand of
    /// `and`, `or` or `not`, or the condition of an `if`, `elif`, `while` or
    /// conditional expression (`kw` says which; see [`BoolContext`]). It
    /// gives the operand back if it is a Bool and otherwise fails with
    /// "`kw` needs a Bool operand, found ..." at the operation's span, which
    /// lowering sets to the operand's, as `sky run` reports it. A typed Bool
    /// needs no check, so lowering emits this only for `unknown` operands.
    /// Effect `Fail`.
    ExpectBool {
        context: BoolContext,
    },
    /// `expect.number[ctx] a`: one operand, one result: the operand's type,
    /// or `unknown` (as lowering leaves it, when a pass learns the operand's
    /// type later). The run-time check of an operand the checker could not type
    /// (`unknown`, from an unannotated parameter) where the source needs a
    /// number or an array: the operand of a unary `+`, which lowers to
    /// nothing else (`pos`), and the argument of `inv(...)` (`inv`), whose
    /// wording differs from `x⁻¹`'s. It gives the operand back if it is a
    /// number or an array and otherwise fails with "`+` needs a number,
    /// found ..." (`ctx` says which; see [`NumberContext`]) at the
    /// operation's span, which lowering sets to the operand's, as `sky run`
    /// reports it. (A separate operation from `expect.bool`: it is not a
    /// Bool, and its result stays `unknown`.) Effect `Fail`.
    ExpectNumber {
        context: NumberContext,
    },

    /// `expect.range[bound] a`, `expect.range[step] a`: one operand, one
    /// result. The run-time checks of a `range` whose bound or step the
    /// checker could not read as a constant (`for i in range(n)`); the
    /// interpreter makes them when the loop starts, and so does SIR.
    ///
    /// * `bound` gives the operand back as a dimensionless scalar if it is
    ///   a dimensionless whole number of magnitude at most 2^53, and
    ///   otherwise fails with `sky run`'s error ("`range` needs an
    ///   integer, found 2.5", "`range` needs a dimensionless integer,
    ///   found length", "the bound 1e300 of `range` is too large") at the
    ///   operation's span, which lowering sets to the argument's. A typed
    ///   number can still fail; an operand that is not a number fails the
    ///   same way, which only an operand of unknown type can be.
    /// * `step` gives the (already checked) step back and fails with "the
    ///   step of `range` cannot be zero" if it is zero, at the operation's
    ///   span, which lowering sets to the step argument's.
    ///
    /// Lowering makes the `bound` check right after each argument is
    /// evaluated, left to right, and the `step` check after the last, as
    /// the interpreter does. Arguments that are literals are not checked
    /// here: the checker did. Effect `Fail`.
    ExpectRange {
        part: RangePart,
    },

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

    // ----- contracts -----
    /// `assert[kind] "text" ... %cond, %v0, ...`: a contract (spec
    /// "Contracts"). No results. Operand 0 is the condition, a Bool (or
    /// `Unknown`, which is checked at run time: anything but a Bool fails
    /// with "`assert` needs a Bool condition, found ..."); the other
    /// operands are the values of `names`, one each, in order, which only a
    /// failure message uses.
    ///
    /// If the condition is true nothing happens. If it is false the run
    /// stops with the message of `sky_interp::contract::failure_message`,
    /// built from the attributes and the values: `kind` says which contract
    /// (`assert`, `requires`, `ensures`); `text` is the condition as
    /// written, white space collapsed; `message` is the author's string, if
    /// any; `func` is the source name of the function a `requires` or
    /// `ensures` belongs to (present exactly for those two); `names` are the
    /// names the condition reads, whose values the message shows. The
    /// operation's span is the condition's, which is where `sky run` points.
    ///
    /// A `requires` is the first thing a function does; an `ensures` runs
    /// just before its `return`, on the arguments as they were passed (SIR
    /// values are immutable, so lowering only has to name them) and the
    /// returned value. Effect `Fail`: an assertion is behaviour, so no pass
    /// moves it past output or removes it, except the `contracts` pass, when
    /// it proves the condition true (`sky_opt::contracts`).
    Assert {
        kind: AssertKind,
        text: String,
        message: Option<String>,
        func: Option<String>,
        names: Vec<String>,
    },

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
            OpKind::LiteralError { .. } => "literal.error",
            OpKind::Input { .. } => "input",
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
            OpKind::Norm => "norm",
            OpKind::Dot => "dot",
            OpKind::Solve { .. } => "solve",
            OpKind::SolveRight { .. } => "solve.right",
            OpKind::MatMulEx { .. } => "matmul",
            OpKind::DiagMul { .. } => "diag.mul",
            OpKind::Show { .. } => "show",

            OpKind::ArrayLiteral { .. } => "array.literal",
            OpKind::Index => "index",
            OpKind::Slice { .. } => "slice",
            OpKind::Zeros => "zeros",
            OpKind::Identity => "identity",
            OpKind::Len => "len",
            OpKind::Fused => "fused",
            OpKind::Assume => "assume",
            OpKind::Cmp(c) => match c {
                CmpOp::Eq => "cmp.eq",
                CmpOp::Ne => "cmp.ne",
                CmpOp::Lt => "cmp.lt",
                CmpOp::Le => "cmp.le",
                CmpOp::Gt => "cmp.gt",
                CmpOp::Ge => "cmp.ge",
            },
            OpKind::Not => "not",
            OpKind::ExpectBool { .. } => "expect.bool",
            OpKind::ExpectNumber { .. } => "expect.number",
            OpKind::ExpectRange { .. } => "expect.range",
            OpKind::If => "if",
            OpKind::While => "while",
            OpKind::Yield => "yield",
            OpKind::Condition => "condition",
            OpKind::Call { .. } => "call",
            OpKind::Return => "return",
            OpKind::Print => "print",
            OpKind::Assert { .. } => "assert",
        }
    }

    /// The bracketed attributes of the array algorithm operations, as the
    /// text format writes them after the mnemonic (`[lu]`, `[ta,alg=syrk]`,
    /// `[left]`), or `None` for a kind without any.
    pub fn algorithm_attrs(&self) -> Option<String> {
        match self {
            OpKind::Solve { method } | OpKind::SolveRight { method } => {
                Some(format!("[{}]", method.mnemonic()))
            }
            OpKind::DiagMul { side } => Some(format!("[{}]", side.text())),
            OpKind::MatMulEx { ta, tb, algorithm } => {
                let mut parts = Vec::new();
                if *ta {
                    parts.push("ta".to_string());
                }
                if *tb {
                    parts.push("tb".to_string());
                }
                if *algorithm != Algorithm::Auto {
                    parts.push(format!("alg={}", algorithm.mnemonic()));
                }
                Some(format!("[{}]", parts.join(",")))
            }
            _ => None,
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
            | OpKind::Assume
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
            // Fail always: the literal has no value, nothing supplies an input.
            OpKind::Input { .. } | OpKind::LiteralError { .. } => (fail, false, false, false),
            // Converting to a tiny unit can overflow.
            OpKind::Show { .. } => (fail, false, false, false),
            // A value that is not a Bool fails.
            OpKind::ExpectBool { .. }
            | OpKind::ExpectNumber { .. }
            | OpKind::ExpectRange { .. } => (fail, false, false, false),
            // A false condition fails.
            OpKind::Assert { .. } => (fail, false, false, false),
            // A singular matrix or a zero scalar fails.
            OpKind::Inverse | OpKind::Solve { .. } | OpKind::SolveRight { .. } => {
                (fail, false, false, false)
            }
            OpKind::MatMul => (fail, false, false, true),
            // The sum of squares (or products) can overflow; `dot` is
            // commutative exactly, as each product is and the sum runs in
            // the same order.
            OpKind::Norm => (fail, false, false, false),
            OpKind::Dot => (fail, false, true, false),
            // Overflow, and a broken Diagonal promise, fail.
            OpKind::MatMulEx { .. } | OpKind::DiagMul { .. } => (fail, false, false, false),
            // A bad or out-of-bounds index, or a bad size, fails.
            OpKind::Index
            | OpKind::Slice { .. }
            | OpKind::Zeros
            | OpKind::Identity
            | OpKind::Len => (fail, false, false, false),
            // Control flow and `fused` have the effects of their regions;
            // the verifier and passes compute those from the contents.
            OpKind::If | OpKind::While | OpKind::Fused => (EffectSet::PURE, false, false, false),
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
