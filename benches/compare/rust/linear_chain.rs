// The linear elementwise chain x = (x + y) * 0.25 + (x - y) * 0.125 over M x K
// elements, R passes (each feeds the next). usage: linear_chain M K R

// Deterministic input data shared with every other language: multiples of 1/64.
fn val(seed: i64, i: i64) -> f64 {
    if seed < 0 {
        return 1.0;
    }
    ((i * (2 * seed + 5) + 7 * seed + 1) % 64) as f64 / 64.0
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let m = args.first().copied().unwrap_or(1000);
    let k = args.get(1).copied().unwrap_or(1000);
    let reps = args.get(2).copied().unwrap_or(1);
    let mut x = vec![0.0f64; m * k];
    let mut y = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            x[i * k + j] = val(0, i as i64) * val(1, j as i64);
            y[i * k + j] = val(2, i as i64) * val(3, j as i64);
        }
    }
    for _ in 0..reps {
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi = (*xi + yi) * 0.25 + (*xi - yi) * 0.125;
        }
    }
    let sum: f64 = x.iter().sum();
    println!("{:e} {:e} {:e}", sum, x[0], x[m * k - 1]);
}
