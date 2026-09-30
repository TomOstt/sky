//! Fast, reproducible dense linear algebra for Sky.
//!
//! Every engine that runs a Sky program (the tree-walking interpreter, the
//! SIR interpreter, and any native one) calls these kernels, and they give
//! the same bits on every machine, thread count and engine. That is the
//! rule this crate exists to keep while being fast.
//!
//! # The reproducibility rule
//!
//! Floating-point addition is not associative, so "fast" must never mean
//! "a different order of additions". Each kernel has a fixed order of
//! operations, given by the definition (below) and independent of hardware:
//!
//! * **Matrix products** ([`gemm`], and `matrix @ vector`, `vector @
//!   matrix`): every output entry is `c = 0; for k in 0..K { c = c +
//!   a[i,k]*b[k,j] }` in that order, with a separate multiply and add and
//!   **no fused multiply-add**. The kernels get their speed by vectorizing
//!   across `i` and `j` (each lane is a different entry), register-blocking
//!   a tile of accumulators, cache-blocking (a `k` block continues from the
//!   partial sum in `C`, so the chain is unbroken), packing, and splitting
//!   the output across threads (which never changes an entry's order).
//! * **Dot products and sums of squares** ([`dot`], [`sum_squares`]): one
//!   sum can't be vectorized in sequential order, so the order is fixed
//!   differently: fewer than 8 terms are added left to right; otherwise
//!   term `i` goes to partial sum `i mod 8` (each starting at `+0.0`, in
//!   index order) and the eight are combined by the tree
//!   `((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`. See [`dot`].
//! * **Cholesky, LU, triangular solves, the inverse**: defined by the
//!   plain algorithms (see [`mod@reference`]), whose per-entry chains the
//!   blocked right-looking versions reproduce exactly: a block update is
//!   `c = c - l*u` for its `k` in ascending order, applied to the value the
//!   earlier blocks left. The inverse is Gauss-Jordan elimination with
//!   partial pivoting, unchanged in its arithmetic (blocked the same way).
//! * **Threads**: [`default_threads`] (what [`set_threads`] chose, the
//!   `SKY_THREADS` environment variable, else the available parallelism) workers, only above a size
//!   threshold. Work is split by output, deterministically.
//!
//! The [`mod@reference`] module holds the plain loops; `tests/reference.rs`
//! checks every kernel against them bit for bit on random shapes.
//!
//! # The kernels
//!
//! * **GEMM** ([`gemm`]): the Goto/BLIS structure. Cache blocks over `j`
//!   (1024 columns), `k` (256) and `i` (128 rows); the block of `B` is packed
//!   once into slivers of 8 columns (shared by all workers) and each worker
//!   packs its own rows of `A` into slivers of 4 rows; a 4 by 8 tile of
//!   accumulators lives in registers while 256 rank-one updates stream
//!   through it. Edge tiles run on zero-padded panels; a tile the caller
//!   does not need (above the diagonal of a symmetric update) is skipped.
//!   Products with at most 16 columns, a single column (`matrix @ vector`)
//!   or a single row (`vector @ matrix`) take unpacked kernels of their own.
//! * **Cholesky** ([`cholesky`]): blocked right-looking, 32 or 64 columns
//!   at a time. The diagonal block and the panel solve are vector
//!   operations along columns (on a transposed copy), and the trailing
//!   update `A₂₂ -= L₂₁·L₂₁ᵀ` is the GEMM kernel restricted to the lower
//!   triangle. Small matrices run the plain loops.
//! * **LU** ([`lu`]): Sky's LU is the forward phase of the Gauss-Jordan
//!   inverse, operation for operation. Recursive on the columns (left
//!   half, triangular solve for the block row of `U`, GEMM update, right
//!   half), with narrow panels factored on a transposed copy. [`Lu::solve`]
//!   and [`Lu::solve_right`] (`x @ A⁻¹`) are triangular solves.
//! * **Gram products** ([`syrk`]): the lower triangle of `b · bᵀ` by the
//!   GEMM kernel, mirrored.
//! * **Triangular solves** (behind [`cholesky_solve`] and [`Lu::solve`]):
//!   the dot-product form, with a register block of 16 right-hand sides per
//!   pass (a lane per right-hand side) and blocks of columns as the unit of
//!   parallel work.
//! * **The inverse** ([`inverse`]): Gauss-Jordan on `[A | I]`, as Sky has
//!   always defined it. Small matrices run it on whole rows with every row
//!   update a vector operation; from order 96 up it is blocked: a block of
//!   columns is eliminated on a transposed strip, recording each row's
//!   multipliers, and the block is applied to the rest of `[A | I]` by a
//!   triangular solve (the pivot rows) and one GEMM (all other rows), which
//!   is the same chain of roundings step by step. It falls back to the
//!   plain loops if an infinity or NaN appears.
//!
//! The structure of the GEMM follows Goto and van de Geijn, "Anatomy of
//! high-performance matrix multiplication" (2008) and the BLIS design of
//! Van Zee and van de Geijn (2015). `matrixmultiply` and `faer` were studied
//! for their designs and are used only as benchmark baselines
//! (`benches/blas.rs`); no code from them is copied, and their
//! FMA-based accumulation could not give Sky's results anyway.
//!
//! # SIMD
//!
//! The hot loops are written so LLVM vectorizes them (NEON on aarch64 is
//! part of the base ISA); on x86_64 they are also compiled with AVX2
//! enabled (never FMA) and chosen at run time with
//! `is_x86_feature_detected!`. The only `unsafe` in the crate is calling
//! those `target_feature` functions after the check.
//!
//! # Cost of not fusing
//!
//! Hardware FMA does a multiply and an add in one instruction and one
//! rounding, which is twice the peak flop rate of the same silicon
//! (`vfmadd`, `fmla`) and a different result. Sky's products round the
//! product first, so on the same core these kernels top out at half the
//! peak of an FMA library. (Apple's Accelerate also uses the AMX
//! coprocessor, which no portable code reaches.) `benches/blas.rs` measures
//! all of this.

#![deny(unsafe_code)]
// Kernels take many operands (pointers, strides, extents), and their lane
// loops are written with indices because that is how the arithmetic reads.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

mod chol;
mod dot;
mod gemm;
mod inverse;
mod lu;
mod micro;
pub mod reference;
mod threads;
mod tri;

pub use chol::{cholesky, cholesky_solve, cholesky_solve_with, cholesky_with, is_symmetric};
pub use dot::{dot, sum_squares, sum_squares_scaled, LANES};
pub use gemm::{gemm, gemm_into_with, gemm_with, syrk, syrk_with};
pub use inverse::{inverse, inverse_with};
pub use lu::{lu, lu_with, solve, Lu};
pub use threads::{default_threads, set_threads, MAX_THREADS};

/// Why a factorization or inverse failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinalgError {
    /// No pivot large enough: the matrix has no inverse (or is so close to
    /// singular that its inverse would be noise).
    Singular,
    /// Cholesky needs `aᵢⱼ == aⱼᵢ` exactly.
    NotSymmetric,
    /// Cholesky found a pivot that is not positive, at this row.
    NotPositiveDefinite { row: usize },
}
