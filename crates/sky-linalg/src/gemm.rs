//! Matrix products: the packed, blocked kernel and its small-size relatives.
//!
//! Every output entry is accumulated in the naive order,
//!
//! ```text
//! c = 0;  for k in 0..K { c = c + a[i,k] * b[k,j] }
//! ```
//!
//! with a separate multiply and add (never a fused one). Speed comes from
//! everything that leaves that order alone: vectorizing across `i` and `j`
//! (every lane is a different entry), register-blocking a `MR` by `NR` tile
//! of accumulators, cache-blocking over `j`, `k` and `i` (a `k` block starts
//! from the partial sum the previous block left in `C`, so each entry's
//! chain of roundings is unbroken), packing panels of `A` and `B` so the
//! microkernel reads contiguous memory, and splitting the output across
//! threads.
//!
//! The same driver serves the factorizations, which need `C = C - A·B`
//! (the entry's chain continues with `c = c - a*b`), on strided views and
//! sometimes only on the lower triangle of `C`.

use std::cell::RefCell;

use crate::micro::{tile, MR, NR};
use crate::threads::{self, par_min_madds};
use crate::tuning::{gemm_blocks, Blocks};

/// A read-only matrix with arbitrary strides: entry `(i, j)` is
/// `data[i * rs + j * cs]`.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    pub data: &'a [f64],
    pub rs: usize,
    pub cs: usize,
}

impl<'a> View<'a> {
    pub fn row_major(data: &'a [f64], ld: usize) -> Self {
        View {
            data,
            rs: ld,
            cs: 1,
        }
    }

    pub fn transposed(self) -> Self {
        View {
            data: self.data,
            rs: self.cs,
            cs: self.rs,
        }
    }

    #[inline(always)]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.rs + j * self.cs]
    }

    /// The view starting at `(i, j)`. Must not be called on an empty block.
    pub fn offset(self, i: usize, j: usize) -> Self {
        View {
            data: &self.data[i * self.rs + j * self.cs..],
            rs: self.rs,
            cs: self.cs,
        }
    }
}

fn round_up(x: usize, to: usize) -> usize {
    x.div_ceil(to) * to
}

thread_local! {
    static A_PACK: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static B_PACK: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Takes a thread's scratch buffer (leaving an empty one, so a re-entrant
/// call just allocates), sized to at least `len`.
fn take_scratch(cell: &'static std::thread::LocalKey<RefCell<Vec<f64>>>, len: usize) -> Vec<f64> {
    let mut v = cell.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if v.len() < len {
        v.resize(len, 0.0);
    }
    v
}

fn give_scratch(cell: &'static std::thread::LocalKey<RefCell<Vec<f64>>>, v: Vec<f64>) {
    cell.with(|c| *c.borrow_mut() = v);
}

/// Packs the `kc` by `mc` block of `a` at `(ic, pc)` into slivers of `MR`
/// rows: `dst[sliver][p][r]`, padding missing rows with zeros.
fn pack_a(a: View, ic: usize, pc: usize, mc: usize, kc: usize, dst: &mut [f64]) {
    for (s, sliver) in dst[..round_up(mc, MR) * kc]
        .chunks_exact_mut(MR * kc)
        .enumerate()
    {
        let r0 = s * MR;
        let rows = MR.min(mc - r0);
        for (p, out) in sliver.chunks_exact_mut(MR).enumerate() {
            for (r, o) in out.iter_mut().enumerate() {
                *o = if r < rows {
                    a.at(ic + r0 + r, pc + p)
                } else {
                    0.0
                };
            }
        }
    }
}

/// Packs the `kc` by `nc` block of `b` at `(pc, jc)` into slivers of `NR`
/// columns: `dst[sliver][p][j]`, padding missing columns with zeros.
fn pack_b_sliver(b: View, pc: usize, jc: usize, nc: usize, kc: usize, s: usize, out: &mut [f64]) {
    let c0 = s * NR;
    let cols = NR.min(nc - c0);
    for (p, row) in out.chunks_exact_mut(NR).enumerate() {
        for (j, o) in row.iter_mut().enumerate() {
            *o = if j < cols {
                b.at(pc + p, jc + c0 + j)
            } else {
                0.0
            };
        }
    }
    debug_assert_eq!(out.len(), kc * NR);
}

/// What one call computes: `C = (zero ? 0 : C) ± A·B`.
#[derive(Clone, Copy)]
pub(crate) struct Op {
    /// Subtract the products (`c = c - a*b`) instead of adding them.
    pub sub: bool,
    /// Start every entry from `+0.0` instead of the value in `C`.
    pub zero: bool,
    /// Only entries with `col <= row + shift` are wanted: tiles wholly above
    /// that diagonal are skipped, and a tile the diagonal cuts is computed
    /// in full. `isize::MAX / 2` means all of `C`.
    pub shift: isize,
    /// Fuse each multiply into its add (relaxed numerics): only when
    /// [`crate::fma_available`].
    pub fma: bool,
}

impl Op {
    pub const ALL: isize = isize::MAX / 2;

    pub fn set() -> Op {
        Op {
            sub: false,
            zero: true,
            shift: Op::ALL,
            fma: false,
        }
    }

    pub fn subtract(shift: isize) -> Op {
        Op {
            sub: true,
            zero: false,
            shift,
            fma: false,
        }
    }

    /// The same operation with fused multiply-adds if `fma` (which must be
    /// false unless [`crate::fma_available`]).
    pub fn fused(self, fma: bool) -> Op {
        Op { fma, ..self }
    }
}

/// The rows of one chunk of `C`: a slice that starts at its first row's
/// first column and runs through its last row's last column.
struct Chunk<'c> {
    row0: usize,
    rows: usize,
    c: &'c mut [f64],
}

/// Splits the `m` by `n` block of `C` (row stride `ldc`) into chunks of
/// `rows_per` rows, each a disjoint slice.
fn split_rows(c: &mut [f64], ldc: usize, m: usize, n: usize, rows_per: usize) -> Vec<Chunk<'_>> {
    let mut chunks = Vec::with_capacity(m.div_ceil(rows_per));
    let mut rest = c;
    let mut row0 = 0;
    while row0 < m {
        let rows = rows_per.min(m - row0);
        let last = rows == m - row0;
        let len = if last {
            (rows - 1) * ldc + n
        } else {
            rows * ldc
        };
        let (head, tail) = rest.split_at_mut(len);
        chunks.push(Chunk {
            row0,
            rows,
            c: head,
        });
        rest = tail;
        row0 += rows;
    }
    chunks
}

/// The work of one chunk against one packed panel of `B`.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    a: View,
    bp: &[f64],
    chunk: &mut Chunk<'_>,
    ldc: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
    op: Op,
    mc_max: usize,
) {
    let load = !(op.zero && pc == 0);
    let mut ap = take_scratch(&A_PACK, round_up(mc_max, MR) * kc);
    let mut ic = 0;
    while ic < chunk.rows {
        let mc = mc_max.min(chunk.rows - ic);
        let grow = chunk.row0 + ic;
        // Skip a block that lies wholly above the diagonal.
        if (jc as isize) > (grow + mc - 1) as isize + op.shift {
            ic += mc;
            continue;
        }
        pack_a(a, grow, pc, mc, kc, &mut ap);
        for jr in (0..nc).step_by(NR) {
            let nr = NR.min(nc - jr);
            let b_sliver = &bp[(jr / NR) * kc * NR..][..kc * NR];
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                if ((jc + jr) as isize) > (grow + ir + mr - 1) as isize + op.shift {
                    continue;
                }
                let a_sliver = &ap[(ir / MR) * kc * MR..][..kc * MR];
                let c = &mut chunk.c[(ic + ir) * ldc + jc + jr..];
                tile(op.sub, op.fma, kc, a_sliver, b_sliver, c, ldc, mr, nr, load);
            }
        }
        ic += mc;
    }
    give_scratch(&A_PACK, ap);
}

/// `C = (zero ? 0 : C) ± A·B` for `C` of `m` by `n` (row stride `ldc`) and
/// `k` terms per entry, on up to `threads` workers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_view(
    a: View,
    b: View,
    c: &mut [f64],
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    op: Op,
    threads: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if op.zero {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        return;
    }
    let Blocks {
        mc: mc_max,
        kc: kc_max,
        nc: nc_max,
    } = gemm_blocks(m, n, k);
    let big = m * n * k >= par_min_madds();
    let threads = if big { threads } else { 1 };
    if threads > 1 && m < 4 * MR * 2 && n >= 64 {
        return gemm_columns(a, b, c, ldc, m, n, k, op, threads);
    }
    let rows_per = if threads > 1 {
        round_up(m.div_ceil(threads * 2), MR).max(2 * MR)
    } else {
        m
    };
    let mut chunks = split_rows(c, ldc, m, n, rows_per);
    let mut bp = take_scratch(&B_PACK, kc_max.min(k) * round_up(nc_max.min(n), NR));
    for jc in (0..n).step_by(nc_max) {
        let nc = nc_max.min(n - jc);
        for pc in (0..k).step_by(kc_max) {
            let kc = kc_max.min(k - pc);
            let slivers = nc.div_ceil(NR);
            let panel = &mut bp[..slivers * kc * NR];
            if threads > 1 {
                threads::for_each(
                    threads,
                    panel.chunks_exact_mut(kc * NR).enumerate().collect(),
                    |(s, out)| pack_b_sliver(b, pc, jc, nc, kc, s, out),
                );
            } else {
                for (s, out) in panel.chunks_exact_mut(kc * NR).enumerate() {
                    pack_b_sliver(b, pc, jc, nc, kc, s, out);
                }
            }
            let panel: &[f64] = panel;
            threads::for_each(threads, chunks.iter_mut().collect(), |chunk| {
                run_chunk(a, panel, chunk, ldc, jc, nc, pc, kc, op, mc_max)
            });
        }
    }
    give_scratch(&B_PACK, bp);
}

/// The wide, short case (few rows, many columns): each worker takes a range
/// of columns, computes it in a private buffer, and the results are copied
/// back. The rows of `A` are packed by every worker, which is cheap because
/// there are few of them.
#[allow(clippy::too_many_arguments)]
fn gemm_columns(
    a: View,
    b: View,
    c: &mut [f64],
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    op: Op,
    threads: usize,
) {
    let width = round_up(n.div_ceil(threads * 2), NR).max(NR * 4);
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(width)
        .map(|j0| (j0, width.min(n - j0)))
        .collect();
    let c_ro: &[f64] = c;
    let tiles = threads::map_collect(threads, ranges.clone(), |(j0, w)| {
        let mut local = vec![0.0; m * w];
        if !op.zero {
            for i in 0..m {
                local[i * w..(i + 1) * w].copy_from_slice(&c_ro[i * ldc + j0..i * ldc + j0 + w]);
            }
        }
        let local_op = Op {
            shift: op.shift - j0 as isize,
            ..op
        };
        gemm_view(a, b.offset(0, j0), &mut local, w, m, w, k, local_op, 1);
        local
    });
    for ((j0, w), local) in ranges.into_iter().zip(tiles) {
        for i in 0..m {
            c[i * ldc + j0..i * ldc + j0 + w].copy_from_slice(&local[i * w..(i + 1) * w]);
        }
    }
}

// ---------------------------------------------------------------------------
// The public products on contiguous row-major matrices.

// The kernels below (`small`, `matvec_rows`, `vecmat_*`) are `inline(always)`
// so that the instance with `FMA` is compiled *inside* the `target_feature`
// wrapper that enables FMA3 on x86-64: `mul_add` is a `vfmadd` instruction
// there and a call of libm's `fma` anywhere else.

/// `acc + a*b`: with `FMA`, one fused rounding; without, the multiply and
/// the add rounded separately (what Rust's `acc += a * b` is).
#[inline(always)]
fn madd<const FMA: bool>(acc: f64, a: f64, b: f64) -> f64 {
    if FMA {
        a.mul_add(b, acc)
    } else {
        acc + a * b
    }
}

/// Makes `$name(fma, args...)` call `$inner::<false>` (or `$inner::<true>`
/// when `fma`), and on x86-64 run the fused instance with FMA3 enabled, so
/// `mul_add` is one instruction there and never a library call. The caller
/// passes `fma = true` only when [`crate::fma_available`].
macro_rules! fused_leaf {
    ($name:ident, $x86:ident, $inner:ident, ($($arg:ident : $ty:ty),*)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        #[allow(unsafe_code)]
        unsafe fn $x86($($arg: $ty),*) {
            $inner::<true>($($arg),*)
        }

        #[allow(clippy::too_many_arguments)]
        fn $name(fma: bool, $($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if fma && crate::fma_available() {
                // SAFETY: `fma_available` just confirmed AVX2 and FMA3; the
                // function is safe code compiled with them enabled.
                #[allow(unsafe_code)]
                unsafe {
                    return $x86($($arg),*);
                }
            }
            #[cfg(target_arch = "aarch64")]
            if fma {
                return $inner::<true>($($arg),*);
            }
            let _ = fma;
            $inner::<false>($($arg),*)
        }
    };
}

/// `c = a * b` for `a` (m by k) and `b` (k by n), all row-major, using up to
/// `threads` workers; every entry of `c` is overwritten. Each entry is
/// accumulated in the naive order with no fused multiply-add.
pub fn gemm_into_with(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    c: &mut [f64],
    threads: usize,
) {
    gemm_into_impl(a, m, k, b, n, c, threads, false)
}

/// [`gemm_into_with`] with **relaxed numerics**: every product is fused into
/// its sum (one rounding for `acc + a*b`) where the machine has the
/// instruction ([`crate::fma_available`]). The order of each entry's sum is
/// still `k` ascending, so the result differs from the reproducible one
/// only by the rounding the fusion saves, and does not depend on the thread
/// count. (The vendor library is [`crate::gemm_relaxed_with`]'s job.)
pub fn gemm_into_fused_with(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    c: &mut [f64],
    threads: usize,
) {
    gemm_into_impl(a, m, k, b, n, c, threads, crate::fma_available())
}

#[allow(clippy::too_many_arguments)]
fn gemm_into_impl(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    c: &mut [f64],
    threads: usize,
    fma: bool,
) {
    assert_eq!(
        (a.len(), b.len(), c.len()),
        (m * k, k * n, m * n),
        "gemm: shape mismatch"
    );
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }
    let madds = m * n * k;
    let threads = if madds >= par_min_madds() { threads } else { 1 };
    if n == 1 {
        matvec(fma, a, m, k, b, c, threads);
    } else if m == 1 {
        vecmat(fma, a, k, b, n, c, threads);
    } else if n <= SMALL_MAX_COLS {
        small_rows(fma, a, m, k, b, n, c, threads);
    } else {
        gemm_view(
            View::row_major(a, k),
            View::row_major(b, n),
            c,
            n,
            m,
            n,
            k,
            Op::set().fused(fma),
            threads,
        );
    }
}

/// `a * b` for `a` (m by k) and `b` (k by n), both row-major, as an m by n
/// row-major matrix (see [`gemm_into_with`]).
pub fn gemm_with(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, threads: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    gemm_into_with(a, m, k, b, n, &mut c, threads);
    c
}

/// [`gemm_with`] on the default number of threads.
pub fn gemm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    gemm_with(a, m, k, b, n, threads::default_threads())
}

/// The Gram matrix `b · bᵀ` of `b` (m by k, row-major) as an m by m
/// row-major matrix, on up to `threads` workers. Entry `(i, j)` is
/// `Σ_p b[i,p]·b[j,p]` accumulated in `p` order (no fused multiply-add), which
/// is what [`gemm_with`] gives for `b` times its transpose. The entries
/// `(i, j)` and `(j, i)` are the same products in the same order (IEEE
/// multiplication commutes), so only the lower triangle is computed and
/// the upper mirrored: half the multiplications.
pub fn syrk_with(b: &[f64], m: usize, k: usize, threads: usize) -> Vec<f64> {
    syrk_impl(b, m, k, threads, false)
}

/// [`syrk_with`] with fused multiply-adds (relaxed numerics; see
/// [`gemm_into_fused_with`]). The result is exactly symmetric, as always.
pub fn syrk_fused_with(b: &[f64], m: usize, k: usize, threads: usize) -> Vec<f64> {
    syrk_impl(b, m, k, threads, crate::fma_available())
}

fn syrk_impl(b: &[f64], m: usize, k: usize, threads: usize, fma: bool) -> Vec<f64> {
    assert_eq!(b.len(), m * k, "syrk: shape mismatch");
    let mut c = vec![0.0; m * m];
    if m == 0 || k == 0 {
        return c;
    }
    if m <= SMALL_MAX_COLS {
        // Small: the plain loops (upper triangle, mirrored).
        for i in 0..m {
            for j in i..m {
                let mut s = 0.0;
                for p in 0..k {
                    s += b[i * k + p] * b[j * k + p];
                }
                c[i * m + j] = s;
                c[j * m + i] = s;
            }
        }
        return c;
    }
    let threads = if m * m * k / 2 >= par_min_madds() {
        threads
    } else {
        1
    };
    let op = Op {
        sub: false,
        zero: true,
        shift: 0,
        fma,
    };
    // B(p, j) = b[j, p]: the transpose, read through strides.
    gemm_view(
        View::row_major(b, k),
        View::row_major(b, k).transposed(),
        &mut c,
        m,
        m,
        m,
        k,
        op,
        threads,
    );
    for i in 0..m {
        for j in i + 1..m {
            c[i * m + j] = c[j * m + i];
        }
    }
    c
}

/// [`syrk_with`] on the default number of threads.
pub fn syrk(b: &[f64], m: usize, k: usize) -> Vec<f64> {
    syrk_with(b, m, k, threads::default_threads())
}

/// The widest product the small kernels handle without packing.
pub(crate) const SMALL_MAX_COLS: usize = 16;

/// `RR` rows of [`small`], starting at row `i`, with NEON intrinsics: each
/// row of `C` is `N / 2` two-lane registers updated by `vfmaq_n_f64` (fused,
/// the entry's products added in `p` order as always), and an odd last column
/// is a scalar `mul_add`. LLVM's own vectorization of the generic loop leaves
/// `fmla` scalar at some widths, which costs half the speed of the plain
/// kernel.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn small_neon_rows<const N: usize, const RR: usize>(
    a: &[f64],
    k: usize,
    b: &[f64],
    c: &mut [f64],
    i: usize,
) {
    use std::arch::aarch64::*;
    const { assert!(N <= 16) };
    let (pairs, odd) = (N / 2, N % 2 == 1);
    // SAFETY: NEON is part of the aarch64 base ISA. Every vector load is of
    // two elements at an even offset below `N - 1` of a slice of exactly `N`
    // elements (`brow`, `row`), so it is in bounds; `a` is indexed safely.
    unsafe {
        let mut acc = [[vdupq_n_f64(0.0); 8]; RR];
        let mut last = [0.0_f64; RR];
        for p in 0..k {
            let brow = &b[p * N..p * N + N];
            let mut bv = [vdupq_n_f64(0.0); 8];
            for (v, x) in bv.iter_mut().enumerate().take(pairs) {
                *x = vld1q_f64(brow.as_ptr().add(2 * v));
            }
            for r in 0..RR {
                let av = a[(i + r) * k + p];
                for v in 0..pairs {
                    acc[r][v] = vfmaq_n_f64(acc[r][v], bv[v], av);
                }
                if odd {
                    last[r] = av.mul_add(brow[N - 1], last[r]);
                }
            }
        }
        for r in 0..RR {
            let row = &mut c[(i + r) * N..(i + r + 1) * N];
            for v in 0..pairs {
                vst1q_f64(row.as_mut_ptr().add(2 * v), acc[r][v]);
            }
            if odd {
                row[N - 1] = last[r];
            }
        }
    }
}

/// A product with at most `SMALL_MAX_COLS` columns, without packing: the `R`
/// rows in flight give the adds independent chains, and the `N`-wide
/// accumulators vectorize across columns.
#[inline(always)]
fn small<const N: usize, const R: usize, const FMA: bool>(
    a: &[f64],
    k: usize,
    b: &[f64],
    c: &mut [f64],
) {
    let m = c.len() / N;
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    if FMA {
        while i + R <= m {
            small_neon_rows::<N, R>(a, k, b, c, i);
            i += R;
        }
        while i < m {
            small_neon_rows::<N, 1>(a, k, b, c, i);
            i += 1;
        }
        return;
    }
    while i + R <= m {
        let mut acc = [[0.0_f64; N]; R];
        for p in 0..k {
            let brow = &b[p * N..p * N + N];
            for (r, acc_r) in acc.iter_mut().enumerate() {
                let av = a[(i + r) * k + p];
                for j in 0..N {
                    acc_r[j] = madd::<FMA>(acc_r[j], av, brow[j]);
                }
            }
        }
        for (r, acc_r) in acc.iter().enumerate() {
            c[(i + r) * N..(i + r + 1) * N].copy_from_slice(acc_r);
        }
        i += R;
    }
    while i < m {
        let mut acc = [0.0_f64; N];
        for p in 0..k {
            let av = a[i * k + p];
            let brow = &b[p * N..p * N + N];
            for j in 0..N {
                acc[j] = madd::<FMA>(acc[j], av, brow[j]);
            }
        }
        c[i * N..(i + 1) * N].copy_from_slice(&acc);
        i += 1;
    }
}

#[inline(always)]
fn small_dispatch<const FMA: bool>(a: &[f64], k: usize, b: &[f64], n: usize, c: &mut [f64]) {
    match n {
        1 => small::<1, 4, FMA>(a, k, b, c),
        2 => small::<2, 4, FMA>(a, k, b, c),
        3 => small::<3, 4, FMA>(a, k, b, c),
        4 => small::<4, 4, FMA>(a, k, b, c),
        5 => small::<5, 4, FMA>(a, k, b, c),
        6 => small::<6, 4, FMA>(a, k, b, c),
        7 => small::<7, 4, FMA>(a, k, b, c),
        8 => small::<8, 4, FMA>(a, k, b, c),
        9 => small::<9, 2, FMA>(a, k, b, c),
        10 => small::<10, 2, FMA>(a, k, b, c),
        11 => small::<11, 2, FMA>(a, k, b, c),
        12 => small::<12, 2, FMA>(a, k, b, c),
        13 => small::<13, 2, FMA>(a, k, b, c),
        14 => small::<14, 2, FMA>(a, k, b, c),
        15 => small::<15, 2, FMA>(a, k, b, c),
        16 => small::<16, 2, FMA>(a, k, b, c),
        _ => unreachable!("small products have at most sixteen columns"),
    }
}

fused_leaf!(
    small_leaf,
    small_fma3,
    small_dispatch,
    (a: &[f64], k: usize, b: &[f64], n: usize, c: &mut [f64])
);

#[allow(clippy::too_many_arguments)]
fn small_rows(
    fma: bool,
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    c: &mut [f64],
    threads: usize,
) {
    if threads <= 1 || m < 64 {
        return small_leaf(fma, a, k, b, n, c);
    }
    let rows_per = round_up(m.div_ceil(threads * 2), 4);
    let items: Vec<(&[f64], &mut [f64])> = a
        .chunks(rows_per * k)
        .zip(c.chunks_mut(rows_per * n))
        .collect();
    threads::for_each(threads, items, |(a_rows, c_rows)| {
        small_leaf(fma, a_rows, k, b, n, c_rows)
    });
}

/// `y = A·x` for `A` (m by k): eight rows at a time, so eight independent
/// chains, each entry summed in `k` order.
#[inline(always)]
fn matvec_rows<const FMA: bool>(a: &[f64], k: usize, x: &[f64], y: &mut [f64]) {
    const R: usize = 8;
    let m = y.len();
    let mut i = 0;
    while i + R <= m {
        let mut acc = [0.0_f64; R];
        let rows: [&[f64]; R] = std::array::from_fn(|r| &a[(i + r) * k..(i + r + 1) * k]);
        for (p, &xp) in x.iter().enumerate() {
            for r in 0..R {
                acc[r] = madd::<FMA>(acc[r], rows[r][p], xp);
            }
        }
        y[i..i + R].copy_from_slice(&acc);
        i += R;
    }
    while i < m {
        let row = &a[i * k..(i + 1) * k];
        let mut s = 0.0;
        for (av, xp) in row.iter().zip(x) {
            s = madd::<FMA>(s, *av, *xp);
        }
        y[i] = s;
        i += 1;
    }
}

fused_leaf!(
    matvec_leaf,
    matvec_fma3,
    matvec_rows,
    (a: &[f64], k: usize, x: &[f64], y: &mut [f64])
);

fn matvec(fma: bool, a: &[f64], m: usize, k: usize, x: &[f64], y: &mut [f64], threads: usize) {
    if threads <= 1 || m < 64 {
        return matvec_leaf(fma, a, k, x, y);
    }
    let rows_per = round_up(m.div_ceil(threads * 2), 8);
    let items: Vec<(&[f64], &mut [f64])> =
        a.chunks(rows_per * k).zip(y.chunks_mut(rows_per)).collect();
    threads::for_each(threads, items, |(a_rows, y_rows)| {
        matvec_leaf(fma, a_rows, k, x, y_rows)
    });
}

/// `y[j0..j0+W] = Σ_p x[p] · B[p, j0..j0+W]`, `p` ascending, for a block of
/// `W` columns held in registers.
#[inline(always)]
fn vecmat_block<const W: usize, const FMA: bool>(
    x: &[f64],
    b: &[f64],
    n: usize,
    j0: usize,
    y: &mut [f64],
) {
    let mut acc = [0.0_f64; W];
    for (p, &xp) in x.iter().enumerate() {
        let row = &b[p * n + j0..p * n + j0 + W];
        for j in 0..W {
            acc[j] = madd::<FMA>(acc[j], xp, row[j]);
        }
    }
    y.copy_from_slice(&acc);
}

/// Columns per pass when `B` is too big for the cache.
const VECMAT_STREAM_COLS: usize = 512;

/// `y = x·B` for `B` (k by n): the columns in blocks, `k` order per entry.
#[inline(always)]
fn vecmat_cols<const FMA: bool>(x: &[f64], b: &[f64], n: usize, j_start: usize, y: &mut [f64]) {
    let total = y.len();
    let mut j = 0;
    if x.len() * n >= 1 << 18 {
        // A big `B` is streamed: a long run of each row per pass (the
        // hardware prefetcher follows it) with the block of `y` in L1.
        while j + VECMAT_STREAM_COLS <= total {
            let block = &mut y[j..j + VECMAT_STREAM_COLS];
            block.fill(0.0);
            // Four rows per sweep: four streams for the prefetcher, and the
            // block of `y` touched a quarter as often. Each entry still adds
            // its products in `p` order.
            let row = |p: usize| &b[p * n + j_start + j..][..VECMAT_STREAM_COLS];
            let mut p = 0;
            while p + 4 <= x.len() {
                let (r0, r1, r2, r3) = (row(p), row(p + 1), row(p + 2), row(p + 3));
                let (x0, x1, x2, x3) = (x[p], x[p + 1], x[p + 2], x[p + 3]);
                for c in 0..VECMAT_STREAM_COLS {
                    let mut a = block[c];
                    a = madd::<FMA>(a, x0, r0[c]);
                    a = madd::<FMA>(a, x1, r1[c]);
                    a = madd::<FMA>(a, x2, r2[c]);
                    a = madd::<FMA>(a, x3, r3[c]);
                    block[c] = a;
                }
                p += 4;
            }
            while p < x.len() {
                let r = row(p);
                for (acc, &v) in block.iter_mut().zip(r) {
                    *acc = madd::<FMA>(*acc, x[p], v);
                }
                p += 1;
            }
            j += VECMAT_STREAM_COLS;
        }
    }
    while j + 16 <= total {
        vecmat_block::<16, FMA>(x, b, n, j_start + j, &mut y[j..j + 16]);
        j += 16;
    }
    while j + 4 <= total {
        vecmat_block::<4, FMA>(x, b, n, j_start + j, &mut y[j..j + 4]);
        j += 4;
    }
    while j < total {
        vecmat_block::<1, FMA>(x, b, n, j_start + j, &mut y[j..j + 1]);
        j += 1;
    }
}

fused_leaf!(
    vecmat_leaf,
    vecmat_fma3,
    vecmat_cols,
    (x: &[f64], b: &[f64], n: usize, j_start: usize, y: &mut [f64])
);

fn vecmat(fma: bool, x: &[f64], k: usize, b: &[f64], n: usize, y: &mut [f64], threads: usize) {
    debug_assert_eq!(x.len(), k);
    if threads <= 1 || n < 128 {
        return vecmat_leaf(fma, x, b, n, 0, y);
    }
    let width = round_up(n.div_ceil(threads * 2), 16);
    let items: Vec<(usize, &mut [f64])> = y
        .chunks_mut(width)
        .enumerate()
        .map(|(i, ch)| (i * width, ch))
        .collect();
    threads::for_each(threads, items, |(j0, y_cols)| {
        vecmat_leaf(fma, x, b, n, j0, y_cols)
    });
}
