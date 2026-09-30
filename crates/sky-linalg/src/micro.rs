//! The microkernel: one `MR` by `NR` tile of `C` held in registers while
//! `kc` rank-one updates stream through it.
//!
//! **Reproducible tiles.** Each lane of the tile is a different output
//! entry, and each entry evolves as `acc = acc + a*b` (or `acc - a*b`) with
//! the multiply and the add rounded separately: Rust never contracts them
//! into an FMA, and the x86 path enables AVX2 without FMA. So the tile
//! computes exactly what the naive loop computes, whatever the vector width.
//!
//! **Fused tiles** ([`Numerics::Relaxed`](crate::Numerics)). The same tile
//! with a fused multiply-add per update (`acc = fma(a, b, acc)`, one
//! rounding instead of two): NEON `vfmaq_laneq_f64` on aarch64, and FMA3 on
//! x86-64 when the CPU has it. Each entry still sums its products in `p`
//! order, so a fused tile differs from the reproducible one only by the
//! roundings the fusion removes (never by a reordering), and it is still
//! independent of block sizes and thread counts. `tile` with `fma = true`
//! must only be called when [`crate::fma_available`] says so.

/// Rows of `C` in a tile.
pub(crate) const MR: usize = 4;
/// Columns of `C` in a tile.
pub(crate) const NR: usize = 8;

/// The accumulators of a tile.
type Acc = [[f64; NR]; MR];

/// The `kc` rank-one updates of a full tile, every multiply and add rounded
/// separately.
#[inline(always)]
fn update<const SUB: bool>(kc: usize, ap: &[f64], bp: &[f64], acc: &mut Acc) {
    let ap = &ap[..kc * MR];
    let bp = &bp[..kc * NR];
    for (a, b) in ap.chunks_exact(MR).zip(bp.chunks_exact(NR)) {
        for i in 0..MR {
            let ai = a[i];
            for j in 0..NR {
                if SUB {
                    acc[i][j] -= ai * b[j];
                } else {
                    acc[i][j] += ai * b[j];
                }
            }
        }
    }
}

/// [`update`] with a fused multiply-add per step, in plain Rust: LLVM
/// lowers `mul_add` to the hardware instruction only where the function is
/// compiled with FMA enabled (the x86 `target_feature` wrapper below), so
/// this must only run there. (On aarch64 the NEON version is used.)
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn update_fused<const SUB: bool>(kc: usize, ap: &[f64], bp: &[f64], acc: &mut Acc) {
    let ap = &ap[..kc * MR];
    let bp = &bp[..kc * NR];
    for (a, b) in ap.chunks_exact(MR).zip(bp.chunks_exact(NR)) {
        for i in 0..MR {
            let ai = if SUB { -a[i] } else { a[i] };
            for j in 0..NR {
                acc[i][j] = ai.mul_add(b[j], acc[i][j]);
            }
        }
    }
}

/// The fused update with NEON intrinsics: the 4 by 8 tile is sixteen
/// two-lane registers, and each step is two loads of `A`, four of `B` and
/// sixteen `fmla` by lane.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn update_fused<const SUB: bool>(kc: usize, ap: &[f64], bp: &[f64], acc: &mut Acc) {
    use std::arch::aarch64::*;
    let ap = &ap[..kc * MR];
    let bp = &bp[..kc * NR];
    // SAFETY: NEON is part of the aarch64 base ISA. Every access is inside a
    // slice whose length was just checked: `ap` has `kc * MR` elements and a
    // step `p < kc` reads `MR` of them from `p * MR`; `bp` has `kc * NR` and
    // a step reads `NR` from `p * NR`; each row of `acc` has `NR` elements
    // and is read and written in `NR / 2` two-lane pieces.
    unsafe {
        let mut c = [[vdupq_n_f64(0.0); NR / 2]; MR];
        for i in 0..MR {
            for v in 0..NR / 2 {
                c[i][v] = vld1q_f64(acc[i].as_ptr().add(2 * v));
            }
        }
        for p in 0..kc {
            let a01 = vld1q_f64(ap.as_ptr().add(p * MR));
            let a23 = vld1q_f64(ap.as_ptr().add(p * MR + 2));
            for v in 0..NR / 2 {
                let b = vld1q_f64(bp.as_ptr().add(p * NR + 2 * v));
                if SUB {
                    c[0][v] = vfmsq_laneq_f64::<0>(c[0][v], b, a01);
                    c[1][v] = vfmsq_laneq_f64::<1>(c[1][v], b, a01);
                    c[2][v] = vfmsq_laneq_f64::<0>(c[2][v], b, a23);
                    c[3][v] = vfmsq_laneq_f64::<1>(c[3][v], b, a23);
                } else {
                    c[0][v] = vfmaq_laneq_f64::<0>(c[0][v], b, a01);
                    c[1][v] = vfmaq_laneq_f64::<1>(c[1][v], b, a01);
                    c[2][v] = vfmaq_laneq_f64::<0>(c[2][v], b, a23);
                    c[3][v] = vfmaq_laneq_f64::<1>(c[3][v], b, a23);
                }
            }
        }
        for i in 0..MR {
            for v in 0..NR / 2 {
                vst1q_f64(acc[i].as_mut_ptr().add(2 * v), c[i][v]);
            }
        }
    }
}

/// Load (or zero) the tile, update it, store the valid `mr` by `nr` part.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_body<const SUB: bool, const FUSED: bool>(
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    let mut acc = [[0.0_f64; NR]; MR];
    if mr == MR && nr == NR {
        if load {
            for (i, row) in acc.iter_mut().enumerate() {
                row.copy_from_slice(&c[i * ldc..i * ldc + NR]);
            }
        }
        if FUSED {
            update_fused::<SUB>(kc, ap, bp, &mut acc);
        } else {
            update::<SUB>(kc, ap, bp, &mut acc);
        }
        for (i, row) in acc.iter().enumerate() {
            c[i * ldc..i * ldc + NR].copy_from_slice(row);
        }
    } else {
        if load {
            for (i, row) in acc.iter_mut().enumerate().take(mr) {
                row[..nr].copy_from_slice(&c[i * ldc..i * ldc + nr]);
            }
        }
        if FUSED {
            update_fused::<SUB>(kc, ap, bp, &mut acc);
        } else {
            update::<SUB>(kc, ap, bp, &mut acc);
        }
        for (i, row) in acc.iter().enumerate().take(mr) {
            c[i * ldc..i * ldc + nr].copy_from_slice(&row[..nr]);
        }
    }
}

fn tile_generic<const SUB: bool>(
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    tile_body::<SUB, false>(kc, ap, bp, c, ldc, mr, nr, load)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn tile_avx2<const SUB: bool>(
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    tile_body::<SUB, false>(kc, ap, bp, c, ldc, mr, nr, load)
}

/// The fused tile on x86-64: the same safe code as the plain one with
/// `mul_add`, compiled with AVX2 and FMA3 so that it is `vfmadd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn tile_fma3<const SUB: bool>(
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    tile_body::<SUB, true>(kc, ap, bp, c, ldc, mr, nr, load)
}

/// The fused tile on aarch64.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn tile_neon<const SUB: bool>(
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    tile_body::<SUB, true>(kc, ap, bp, c, ldc, mr, nr, load)
}

/// `C[..mr, ..nr] = (load ? C : 0) ± Σ_p ap[p, ..] ⊗ bp[p, ..]` for one tile,
/// where `ap` is a packed sliver of `MR` rows and `bp` one of `NR` columns.
/// With `fma` the products are fused into the sum (relaxed numerics); a
/// machine without a fused kernel here runs the plain one instead.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tile(
    sub: bool,
    fma: bool,
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    #[cfg(target_arch = "aarch64")]
    if fma {
        return if sub {
            tile_neon::<true>(kc, ap, bp, c, ldc, mr, nr, load)
        } else {
            tile_neon::<false>(kc, ap, bp, c, ldc, mr, nr, load)
        };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if fma && crate::fma_available() {
            // SAFETY: `fma_available` just confirmed AVX2 and FMA3 at run
            // time, and `tile_fma3` is safe code compiled with them enabled.
            #[allow(unsafe_code)]
            unsafe {
                if sub {
                    tile_fma3::<true>(kc, ap, bp, c, ldc, mr, nr, load)
                } else {
                    tile_fma3::<false>(kc, ap, bp, c, ldc, mr, nr, load)
                }
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the `avx2` feature was just detected at run time, and
            // `tile_avx2` is the same safe code as `tile_generic`, compiled
            // with AVX2 enabled.
            #[allow(unsafe_code)]
            unsafe {
                if sub {
                    tile_avx2::<true>(kc, ap, bp, c, ldc, mr, nr, load)
                } else {
                    tile_avx2::<false>(kc, ap, bp, c, ldc, mr, nr, load)
                }
            }
            return;
        }
    }
    let _ = fma;
    if sub {
        tile_generic::<true>(kc, ap, bp, c, ldc, mr, nr, load)
    } else {
        tile_generic::<false>(kc, ap, bp, c, ldc, mr, nr, load)
    }
}
