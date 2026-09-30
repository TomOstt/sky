// Mandelbrot escape counts on a W x W grid over [-2, 1] x [-1.5, 1.5].
// usage: mandelbrot W MAXIT
fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let w = args.first().copied().unwrap_or(300);
    let maxit = args.get(1).copied().unwrap_or(500);
    let mut total: u64 = 0;
    for py in 0..w {
        let ci = -1.5 + py as f64 * (3.0 / w as f64);
        for px in 0..w {
            let cr = -2.0 + px as f64 * (3.0 / w as f64);
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut i = 0;
            while i < maxit {
                let zr2 = zr * zr;
                let zi2 = zi * zi;
                if zr2 + zi2 > 4.0 {
                    break;
                }
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                i += 1;
            }
            total += i;
        }
    }
    println!("{total}");
}
