// The classic N-body benchmark: Sun and the four gas giants, step 0.01.
// usage: nbody STEPS
const PI: f64 = std::f64::consts::PI;
const SOLAR_MASS: f64 = 4.0 * PI * PI;
const DPY: f64 = 365.24;

#[derive(Clone, Copy)]
struct Body {
    x: [f64; 3],
    v: [f64; 3],
    mass: f64,
}

fn bodies() -> [Body; 5] {
    let b = |x: [f64; 3], v: [f64; 3], m: f64| Body {
        x,
        v: [v[0] * DPY, v[1] * DPY, v[2] * DPY],
        mass: m * SOLAR_MASS,
    };
    let mut bs = [
        Body { x: [0.0; 3], v: [0.0; 3], mass: SOLAR_MASS },
        b(
            [4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01],
            [1.66007664274403694e-03, 7.69901118419740425e-03, -6.90460016972063023e-05],
            9.54791938424326609e-04,
        ),
        b(
            [8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01],
            [-2.76742510726862411e-03, 4.99852801234917238e-03, 2.30417297573763929e-05],
            2.85885980666130812e-04,
        ),
        b(
            [1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01],
            [2.96460137564761618e-03, 2.37847173959480950e-03, -2.96589568540237556e-05],
            4.36624404335156298e-05,
        ),
        b(
            [1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01],
            [2.68067772490389322e-03, 1.62824170038242295e-03, -9.51592254519715870e-05],
            5.15138902046611451e-05,
        ),
    ];
    let mut p = [0.0; 3];
    for b in &bs {
        for k in 0..3 {
            p[k] += b.v[k] * b.mass;
        }
    }
    for k in 0..3 {
        bs[0].v[k] = -p[k] / SOLAR_MASS;
    }
    bs
}

fn advance(bs: &mut [Body; 5], dt: f64) {
    for i in 0..5 {
        for j in i + 1..5 {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = bs[i].x[k] - bs[j].x[k];
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let mag = dt / (r2 * r2.sqrt());
            let (mi, mj) = (bs[i].mass, bs[j].mass);
            for k in 0..3 {
                bs[i].v[k] -= d[k] * mj * mag;
                bs[j].v[k] += d[k] * mi * mag;
            }
        }
    }
    for b in bs.iter_mut() {
        for k in 0..3 {
            b.x[k] += dt * b.v[k];
        }
    }
}

fn energy(bs: &[Body; 5]) -> f64 {
    let mut e = 0.0;
    for i in 0..5 {
        let v = bs[i].v;
        e += 0.5 * bs[i].mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for j in i + 1..5 {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = bs[i].x[k] - bs[j].x[k];
            }
            e -= bs[i].mass * bs[j].mass / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
    }
    e
}

fn main() {
    let steps: u64 = std::env::args().nth(1).map_or(100_000, |a| a.parse().unwrap());
    let mut bs = bodies();
    let e0 = energy(&bs);
    for _ in 0..steps {
        advance(&mut bs, 0.01);
    }
    println!("{:e} {:e}", e0, energy(&bs));
}
