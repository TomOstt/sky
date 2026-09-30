//! The names of the algorithm variants: how a solve is computed, which side
//! of a product a structured matrix is on, and the algorithm class of a
//! matrix product.
//!
//! They live here, below both the checker's chain planner ([`crate::chain`])
//! and the interpreters, so every layer speaks about the same choice; SIR
//! re-exports them (`sky_ir::SolveMethod`, `sky_ir::Side`,
//! `sky_ir::Algorithm`) as attributes of `solve`, `diag.mul` and
//! `matmul[...]`.

/// How `solve` solves `A x = b`. Chosen by the algorithm-choosing rewrites
/// ([`crate::chain`], `sky-opt`), never written by users in v1.
///
/// None of the methods forms `A⁻¹`. Each reports a singular `A` (or a broken
/// tag promise) as a run-time error, and none is bit-identical to
/// `inverse` followed by `matmul`: the rounding differs, and the spec's
/// "Algorithm choice" section says that is allowed (the solve is at least as
/// accurate).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SolveMethod {
    /// `A` is PositiveDefinite: factor `A = L·Lᵀ`, then two triangular
    /// solves. `A⁻¹` is never formed.
    Cholesky,
    /// Any square `A`: Gaussian elimination with partial pivoting
    /// (`A = P·L·U`), then two triangular solves. The elimination is the
    /// forward phase of the Gauss-Jordan `inverse`, operation for
    /// operation, so a matrix is singular for `solve[lu]` exactly when it is
    /// for `inverse`.
    Lu,
    /// `A` is Diagonal: `xᵢ = bᵢ / aᵢᵢ`, one correctly rounded division per
    /// entry (the inverse would round twice, `bᵢ · (1/aᵢᵢ)`). Singular under
    /// the same test as `inverse` (a pivot no larger than `n·ε·max|aᵢⱼ|`).
    Diagonal,
}

impl SolveMethod {
    pub const ALL: [SolveMethod; 3] = [
        SolveMethod::Cholesky,
        SolveMethod::Lu,
        SolveMethod::Diagonal,
    ];

    pub fn mnemonic(self) -> &'static str {
        match self {
            SolveMethod::Cholesky => "cholesky",
            SolveMethod::Lu => "lu",
            SolveMethod::Diagonal => "diagonal",
        }
    }
}

/// Which side of a product the structured operand of `diag.mul` is on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    /// `D @ X`: the diagonal matrix is the left operand.
    Left,
    /// `X @ D`: the diagonal matrix is the right operand.
    Right,
}

impl Side {
    pub fn text(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

/// The algorithm a `matmul` is computed with, recorded on the operation
/// (`matmul[alg=...]`).
///
/// **Every algorithm computes exactly the reference result**: entry `(i, j)`
/// is the sum of `a[i,p]·b[p,j]` accumulated in index order `p = 0, 1, ...`
/// starting from `+0.0`, one rounding per multiply and per add, and the
/// same run-time errors. That is the contract that lets a pass pick one
/// from the shapes without changing a single printed digit (a kernel that
/// blocks or unrolls `i` and `j`, or uses SIMD lanes across `j`, keeps it;
/// one that reorders the `p` sum does not, and is not a variant of this
/// enum but a rewrite that needs the spec's permission).
///
/// `Auto` leaves the choice to the engine. The other names classify the
/// operand shapes, so a kernel or a tuning table can plug in its own code
/// per class; the verifier checks that the class fits the operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Algorithm {
    /// The engine chooses (what plain `matmul` means).
    #[default]
    Auto,
    /// `vector @ vector`: a dot product.
    Dot,
    /// `matrix @ vector`.
    MatVec,
    /// `vector @ matrix`.
    VecMat,
    /// `matrix @ matrix`.
    Gemm,
    /// A Gram product, `Aᵀ @ A` or `A @ Aᵀ` (both operands are one value,
    /// one of them transposed). The result is bitwise symmetric (`aₚᵢ·aₚⱼ`
    /// and `aₚⱼ·aₚᵢ` are the same product), so one triangle is computed and
    /// mirrored: half the multiplications, the same bits.
    Syrk,
}

impl Algorithm {
    pub const ALL: [Algorithm; 6] = [
        Algorithm::Auto,
        Algorithm::Dot,
        Algorithm::MatVec,
        Algorithm::VecMat,
        Algorithm::Gemm,
        Algorithm::Syrk,
    ];

    /// The name in the text format: `matmul[alg=syrk]`.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Algorithm::Auto => "auto",
            Algorithm::Dot => "dot",
            Algorithm::MatVec => "matvec",
            Algorithm::VecMat => "vecmat",
            Algorithm::Gemm => "gemm",
            Algorithm::Syrk => "syrk",
        }
    }
}
