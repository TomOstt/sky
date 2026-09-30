//! Triangular solves with many right-hand sides.
//!
//! The solve is in dot-product form: entry `x[i]` starts from its right-hand
//! side, subtracts `m[i,j]·x[j]` for the `j` it depends on **in ascending
//! `j` order**, then divides by the diagonal. That chain is the definition
//! of the result; the kernel only decides how to run it fast. The right-hand
//! sides are independent, so each block of columns is one register-resident
//! accumulator array (a lane per column), and blocks of columns are the unit
//! of parallel work.

use crate::gemm::View;
use crate::threads;

/// Columns of the right-hand side per register block.
const W: usize = 16;

/// Solves for one block of `w` columns held in `buf` (`n` rows of `w`).
fn solve_block<const WIDTH: usize>(m: View, n: usize, unit: bool, lower: bool, buf: &mut [f64]) {
    let step = |i: usize, js: std::ops::Range<usize>, buf: &mut [f64]| {
        let mut acc = [0.0_f64; WIDTH];
        acc.copy_from_slice(&buf[i * WIDTH..(i + 1) * WIDTH]);
        for j in js {
            let l = m.data[i * m.rs + j * m.cs];
            let xj = &buf[j * WIDTH..(j + 1) * WIDTH];
            for c in 0..WIDTH {
                acc[c] -= l * xj[c];
            }
        }
        if !unit {
            let d = m.data[i * m.rs + i * m.cs];
            for a in &mut acc {
                *a /= d;
            }
        }
        buf[i * WIDTH..(i + 1) * WIDTH].copy_from_slice(&acc);
    };
    if lower {
        for i in 0..n {
            step(i, 0..i, buf);
        }
    } else {
        for i in (0..n).rev() {
            step(i, i + 1..n, buf);
        }
    }
}

/// Solves a block of `w` columns held in `buf` (`n` rows of `w`).
fn solve_dispatch(m: View, n: usize, unit: bool, lower: bool, w: usize, buf: &mut [f64]) {
    match w {
        16 => solve_block::<16>(m, n, unit, lower, buf),
        8 => solve_block::<8>(m, n, unit, lower, buf),
        4 => solve_block::<4>(m, n, unit, lower, buf),
        2 => solve_block::<2>(m, n, unit, lower, buf),
        1 => solve_block::<1>(m, n, unit, lower, buf),
        _ => unreachable!("column blocks are 16, 8, 4, 2 or 1 wide"),
    }
}

/// Solves `T·X = B` in place, where `X` starts as `B` (`n` by `k`,
/// row-major) and `T` is the lower (`lower`) or upper triangle of `m`,
/// with an implicit unit diagonal if `unit`. Only entries `m(i, j)` with
/// `j < i` (lower) or `j > i` (upper), and the diagonal unless `unit`, are
/// read. Work below `par_min_madds` multiply-adds stays on this thread.
pub(crate) fn solve_in_place(
    m: View,
    n: usize,
    unit: bool,
    lower: bool,
    x: &mut [f64],
    k: usize,
    threads: usize,
    par_min_madds: usize,
) {
    debug_assert_eq!(x.len(), n * k);
    if n == 0 || k == 0 {
        return;
    }
    if matches!(k, 1 | 2 | 4 | 8 | 16) {
        // `x` already is one block of columns.
        return solve_dispatch(m, n, unit, lower, k, x);
    }
    let mut tasks = Vec::new();
    let mut c0 = 0;
    while c0 < k {
        let mut w = W;
        while w > k - c0 {
            w /= 2;
        }
        tasks.push((c0, w));
        c0 += w;
    }
    let threads = if n * n / 2 * k >= par_min_madds {
        threads
    } else {
        1
    };
    if threads <= 1 || tasks.len() <= 1 {
        // One buffer, reused: no allocation per block of columns.
        let mut buf = vec![0.0; n * W];
        for (c0, w) in tasks {
            let buf = &mut buf[..n * w];
            gather(x, n, k, c0, w, buf);
            solve_dispatch(m, n, unit, lower, w, buf);
            scatter(buf, n, k, c0, w, x);
        }
        return;
    }
    let xr: &[f64] = x;
    let done = threads::map_collect(threads, tasks.clone(), |(c0, w)| {
        let mut buf = vec![0.0; n * w];
        gather(xr, n, k, c0, w, &mut buf);
        solve_dispatch(m, n, unit, lower, w, &mut buf);
        buf
    });
    for ((c0, w), buf) in tasks.into_iter().zip(done) {
        scatter(&buf, n, k, c0, w, x);
    }
}

/// Copies columns `c0..c0+w` of `x` (`n` by `k`) into `buf` (`n` by `w`).
fn gather(x: &[f64], n: usize, k: usize, c0: usize, w: usize, buf: &mut [f64]) {
    for i in 0..n {
        buf[i * w..(i + 1) * w].copy_from_slice(&x[i * k + c0..i * k + c0 + w]);
    }
}

/// The reverse of [`gather`].
fn scatter(buf: &[f64], n: usize, k: usize, c0: usize, w: usize, x: &mut [f64]) {
    for i in 0..n {
        x[i * k + c0..i * k + c0 + w].copy_from_slice(&buf[i * w..(i + 1) * w]);
    }
}
