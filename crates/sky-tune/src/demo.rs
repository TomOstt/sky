//! Two demo tunables, so that `sky tune` does something real before any
//! production kernel is registered, and so the tuner is testable end to end.
//! They are illustrations, not Sky's kernels: the dot product's unroll factor
//! changes the order of additions (the demo's inputs are small integers, so
//! every order gives the same exact sum), which the real, reproducible
//! kernels never do.

use std::cell::RefCell;

use crate::measure::{size_class, Tunable, Work};
use crate::params::Params;
use crate::registry::Registry;

/// Adds the demos to `registry`.
pub fn register(registry: &mut Registry) {
    registry.add("dot_blocked", dot_tunables);
    registry.add("transpose_tiled", transpose_tunables);
}

/// The unroll factors tried for the dot product; 4 is the default.
pub const UNROLLS: [i64; 5] = [4, 1, 2, 8, 16];
/// The tile sizes tried for the transpose; 16 is the default.
pub const TILES: [i64; 5] = [16, 4, 8, 32, 64];

/// Dot product with `U` independent accumulators: more accumulators hide the
/// latency of the additions, up to the point the registers run out.
pub fn dot_unrolled<const U: usize>(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; U];
    let (ca, cb) = (a.chunks_exact(U), b.chunks_exact(U));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..U {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f64 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// [`dot_unrolled`] with the factor chosen at run time (1, 2, 4, 8 or 16;
/// anything else falls back to 4).
pub fn dot(a: &[f64], b: &[f64], unroll: i64) -> f64 {
    match unroll {
        1 => dot_unrolled::<1>(a, b),
        2 => dot_unrolled::<2>(a, b),
        8 => dot_unrolled::<8>(a, b),
        16 => dot_unrolled::<16>(a, b),
        _ => dot_unrolled::<4>(a, b),
    }
}

/// Transposes the `n x n` row-major matrix `src` into `dst` in `tile x tile`
/// blocks, which keeps both the rows read and the columns written in cache.
pub fn transpose_tiled(src: &[f64], dst: &mut [f64], n: usize, tile: usize) {
    let tile = tile.max(1);
    for bi in (0..n).step_by(tile) {
        for bj in (0..n).step_by(tile) {
            for i in bi..(bi + tile).min(n) {
                for j in bj..(bj + tile).min(n) {
                    dst[j * n + i] = src[i * n + j];
                }
            }
        }
    }
}

fn dot_tunables() -> Vec<Tunable> {
    let n: usize = 1 << 18;
    let a: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let space: Vec<Params> = UNROLLS
        .iter()
        .map(|&u| Params::new().with("unroll", u))
        .collect();
    let tunable = Tunable::new(
        "dot_blocked",
        &size_class(n),
        space,
        Work::Bytes(16 * n as u64),
        move |p| dot(&a, &b, p.get_or("unroll", 4)).to_bits(),
    );
    tunable.into_iter().collect()
}

fn transpose_tunables() -> Vec<Tunable> {
    let n: usize = 512;
    let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let dst = RefCell::new(vec![0.0f64; n * n]);
    let space: Vec<Params> = TILES
        .iter()
        .map(|&t| Params::new().with("tile", t))
        .collect();
    let tunable = Tunable::new(
        "transpose_tiled",
        &format!("{n}x{n}"),
        space,
        Work::Bytes(16 * (n * n) as u64),
        move |p| {
            let mut dst = dst.borrow_mut();
            transpose_tiled(&src, &mut dst, n, p.get_or("tile", 16).max(1) as usize);
            dst[n * n - 2].to_bits()
        },
    );
    tunable.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_unroll_gives_the_same_sum() {
        let n = 1000 + 7; // not a multiple of any factor: exercises the remainder
        let a: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let expect: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        for u in [1, 2, 4, 8, 16, 3] {
            assert_eq!(dot(&a, &b, u), expect, "unroll {u}");
        }
        assert_eq!(dot(&[], &[], 8), 0.0);
    }

    #[test]
    fn every_tile_transposes_correctly() {
        for n in [1usize, 5, 33] {
            let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
            for tile in [1usize, 4, 8, 16, 64] {
                let mut dst = vec![-1.0; n * n];
                transpose_tiled(&src, &mut dst, n, tile);
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(dst[j * n + i], src[i * n + j], "n={n} tile={tile}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_demos_give_the_same_answer_for_every_candidate() {
        for tunable in dot_tunables().into_iter().chain(transpose_tunables()) {
            assert_eq!(tunable.candidates.len(), 5);
            assert!(matches!(tunable.work, Work::Bytes(_)));
            let answers: Vec<u64> = tunable.candidates.iter().map(|p| tunable.run(p)).collect();
            assert!(answers.windows(2).all(|w| w[0] == w[1]), "{answers:?}");
        }
    }
}
