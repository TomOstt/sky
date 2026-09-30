// Hand-written baselines for the fused-kernel benchmark: plain loops the
// way a person tuning by hand writes them (iterators without bounds checks,
// the compiler left to vectorize), compiled by the benchmark with
// `rustc -C opt-level=3 -C target-cpu=native`. No fused multiply-add (Rust
// never contracts), so the bits are Sky's.
//
// Output, one line per (kernel, variant, n): `rust <kernel> <variant> <n> <ns per element>`.
// `plain` is only the arithmetic; `checked` also verifies, as Sky must, that
// the values a failure could be hidden in (the yield, and a divisor) are finite.

use std::hint::black_box;
use std::time::Instant;

fn finite(r: f64) -> bool {
    r.abs() <= f64::MAX
}

#[inline(never)]
fn axpy(out: &mut [f64], x: &[f64], y: &[f64], a: f64) {
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        *o = a * *x + *y;
    }
}

#[inline(never)]
fn axpy_checked(out: &mut [f64], x: &[f64], y: &[f64], a: f64) -> bool {
    let mut ok = true;
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        let r = a * *x + *y;
        ok &= finite(r);
        *o = r;
    }
    ok
}

#[inline(never)]
fn hypot(out: &mut [f64], x: &[f64], y: &[f64]) {
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        *o = (*x * *x + *y * *y).sqrt() + 0.0;
    }
}

#[inline(never)]
fn hypot_checked(out: &mut [f64], x: &[f64], y: &[f64]) -> bool {
    let mut ok = true;
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        let r = (*x * *x + *y * *y).sqrt() + 0.0;
        ok &= finite(r);
        *o = r;
    }
    ok
}

#[inline(never)]
fn chain6(out: &mut [f64], x: &[f64], y: &[f64], s: f64) {
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        let t2 = *x * s + *y;
        let t4 = *x * *x + 1.5;
        *o = t2 / t4 - *y;
    }
}

#[inline(never)]
fn chain6_checked(out: &mut [f64], x: &[f64], y: &[f64], s: f64) -> bool {
    let mut ok = true;
    for ((o, x), y) in out.iter_mut().zip(x).zip(y) {
        let t2 = *x * s + *y;
        let t4 = *x * *x + 1.5;
        ok &= finite(t4);
        let r = t2 / t4 - *y;
        ok &= finite(r);
        *o = r;
    }
    ok
}

fn time(n: usize, mut f: impl FnMut()) -> f64 {
    let reps = (20_000_000 / n).max(1);
    f();
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / (reps * n) as f64);
    }
    best * 1e9
}

fn main() {
    for n in [1_000usize, 100_000, 10_000_000] {
        let x: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001 + 0.5).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 777) as f64 * 0.002 - 0.7).collect();
        let mut out = vec![0.0; n];
        let s = black_box(2.5);
        let report = |kernel: &str, variant: &str, ns: f64| {
            println!("rust {kernel} {variant} {n} {ns:.4}");
        };
        let ns = time(n, || axpy(&mut out, black_box(&x), black_box(&y), s));
        report("axpy", "plain", ns);
        let ns = time(n, || {
            black_box(axpy_checked(&mut out, black_box(&x), black_box(&y), s));
        });
        report("axpy", "checked", ns);
        let ns = time(n, || hypot(&mut out, black_box(&x), black_box(&y)));
        report("hypot", "plain", ns);
        let ns = time(n, || {
            black_box(hypot_checked(&mut out, black_box(&x), black_box(&y)));
        });
        report("hypot", "checked", ns);
        let ns = time(n, || chain6(&mut out, black_box(&x), black_box(&y), s));
        report("chain6", "plain", ns);
        let ns = time(n, || {
            black_box(chain6_checked(&mut out, black_box(&x), black_box(&y), s));
        });
        report("chain6", "checked", ns);
        black_box(&out);
    }
}
