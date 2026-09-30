// Cholesky solve K x = b, K = c1 c1' + c2 c2' + n I (n x n, dense, symmetric
// positive definite), R times with a different load each time.
// usage: cholesky N R

// Deterministic input data shared with every other language: multiples of 1/64.
fn val(seed: i64, i: i64) -> f64 {
    if seed < 0 {
        return 1.0;
    }
    ((i * (2 * seed + 5) + 7 * seed + 1) % 64) as f64 / 64.0
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let n = args.first().copied().unwrap_or(256);
    let reps = args.get(1).copied().unwrap_or(1);
    let v = |seed: i64, i: usize| val(seed, i as i64);
    let mut k = vec![0.0f64; n * n];
    let mut f = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            let diag = if i == j { n as f64 } else { 0.0 };
            k[i * n + j] = v(0, i) * v(0, j) + v(1, i) * v(1, j) + diag;
        }
        f[i] = v(2, i);
    }
    let mut l = vec![0.0f64; n * n];
    let mut y = vec![0.0f64; n];
    let mut x = vec![0.0f64; n];
    let mut acc = 0.0;
    for r in 0..reps {
        l.copy_from_slice(&k);
        // K = L L' (lower triangle, row-oriented).
        for j in 0..n {
            let mut s = l[j * n + j];
            for t in 0..j {
                s -= l[j * n + t] * l[j * n + t];
            }
            let d = s.sqrt();
            l[j * n + j] = d;
            for i in j + 1..n {
                let mut t = l[i * n + j];
                let (ri, rj) = (&l[i * n..i * n + j], &l[j * n..j * n + j]);
                for (a, b) in ri.iter().zip(rj) {
                    t -= a * b;
                }
                l[i * n + j] = t / d;
            }
        }
        // L y = b, then L' x = y.
        for i in 0..n {
            let mut t = (1.0 + r as f64) * f[i];
            for q in 0..i {
                t -= l[i * n + q] * y[q];
            }
            y[i] = t / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut t = y[i];
            for q in i + 1..n {
                t -= l[q * n + i] * x[q];
            }
            x[i] = t / l[i * n + i];
        }
        let sum: f64 = x.iter().sum();
        acc = acc + x[0] + x[n - 1] + sum;
    }
    println!("{:e} {:e} {:e}", acc, x[0], x[n - 1]);
}
