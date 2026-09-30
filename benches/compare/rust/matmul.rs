// Dense matmul C = A B (n x n), repeated R times, each repetition rescaling the
// product so the next depends on it. usage: matmul N R

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
    let mut a = vec![0.0f64; n * n];
    let mut b = vec![0.0f64; n * n];
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let eye = if i == j { 1.0 } else { 0.0 };
            a[i * n + j] = val(0, i as i64) * val(1, j as i64) + eye;
            b[i * n + j] = val(2, i as i64) * val(3, j as i64) + eye;
        }
    }
    let mut s = 1.0;
    for _ in 0..reps {
        c.fill(0.0);
        for i in 0..n {
            let crow = &mut c[i * n..(i + 1) * n];
            for k in 0..n {
                let aik = a[i * n + k];
                let brow = &b[k * n..(k + 1) * n];
                for (cj, bj) in crow.iter_mut().zip(brow) {
                    *cj += aik * bj;
                }
            }
        }
        s = c.iter().sum::<f64>();
        for (ai, ci) in a.iter_mut().zip(&c) {
            *ai = ci / s;
        }
    }
    println!("{:e} {:e} {:e}", s, a[0], a[n * n - 1]);
}
