// Classic RK4 for a damped pendulum, y = [angle, angular velocity], h = 0.001.
// usage: rk4 STEPS
type V = [f64; 2];

fn f(y: V) -> V {
    [y[1], -y[0].sin() - 0.1 * y[1]]
}

fn axpy(y: V, a: f64, k: V) -> V {
    [y[0] + a * k[0], y[1] + a * k[1]]
}

fn main() {
    let steps: u64 = std::env::args().nth(1).map_or(200_000, |a| a.parse().unwrap());
    let mut y: V = [1.0, 0.0];
    let h = 0.001;
    for _ in 0..steps {
        let k1 = f(y);
        let k2 = f(axpy(y, h / 2.0, k1));
        let k3 = f(axpy(y, h / 2.0, k2));
        let k4 = f(axpy(y, h, k3));
        for j in 0..2 {
            y[j] += (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
    }
    println!("{:e} {:e}", y[0], y[1]);
}
