//! The microkernel: one `MR` by `NR` tile of `C` held in registers while
//! `kc` rank-one updates stream through it.
//!
//! Each lane of the tile is a different output entry, and each entry
//! evolves as `acc = acc + a*b` (or `acc - a*b`) with the multiply and the
//! add rounded separately: Rust never contracts them into an FMA, and the
//! x86 path enables AVX2 without FMA. So the tile computes exactly what
//! the naive loop computes, whatever the vector width.

/// Rows of `C` in a tile.
pub(crate) const MR: usize = 4;
/// Columns of `C` in a tile.
pub(crate) const NR: usize = 8;

/// The `kc` rank-one updates of a full tile.
#[inline(always)]
fn update<const SUB: bool>(kc: usize, ap: &[f64], bp: &[f64], acc: &mut [[f64; NR]; MR]) {
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

/// Load (or zero) the tile, update it, store the valid `mr` by `nr` part.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_body<const SUB: bool>(
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
        update::<SUB>(kc, ap, bp, &mut acc);
        for (i, row) in acc.iter().enumerate() {
            c[i * ldc..i * ldc + NR].copy_from_slice(row);
        }
    } else {
        if load {
            for (i, row) in acc.iter_mut().enumerate().take(mr) {
                row[..nr].copy_from_slice(&c[i * ldc..i * ldc + nr]);
            }
        }
        update::<SUB>(kc, ap, bp, &mut acc);
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
    tile_body::<SUB>(kc, ap, bp, c, ldc, mr, nr, load)
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
    tile_body::<SUB>(kc, ap, bp, c, ldc, mr, nr, load)
}

/// `C[..mr, ..nr] = (load ? C : 0) ± Σ_p ap[p, ..] ⊗ bp[p, ..]` for one tile,
/// where `ap` is a packed sliver of `MR` rows and `bp` one of `NR` columns.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tile(
    sub: bool,
    kc: usize,
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
    load: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
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
    if sub {
        tile_generic::<true>(kc, ap, bp, c, ldc, mr, nr, load)
    } else {
        tile_generic::<false>(kc, ap, bp, c, ldc, mr, nr, load)
    }
}
