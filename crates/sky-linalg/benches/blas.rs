//! Sky's kernels against hand-tuned libraries: GFLOP/s of `gemm`, Cholesky
//! (`potrf`) and LU solve (`gesv`) at n = 4, 16, 64, 256, 512, 1024, on one
//! thread and on all of them.
//!
//! Baselines, each only when present:
//! * Apple Accelerate (macOS) or OpenBLAS (Linux): `benches/blas_shim.c`
//!   compiled at run time with `/usr/bin/clang -O3 -framework Accelerate`
//!   (or `cc ... -lopenblas`) and run as a subprocess;
//! * `faer` and `matrixmultiply` (dev-dependencies).
//!
//! Run with `cargo bench -p sky-linalg` (a few minutes). Set
//! `BENCH_SIZES=64,256` to choose sizes and `BENCH_THREADS=1,10` to choose
//! thread counts. The baselines fuse multiply and add; Sky's kernels must
//! not (see the crate docs), so on the same silicon Sky's ceiling is about
//! half of theirs. Apple's Accelerate additionally runs on the AMX matrix
//! coprocessor, which no portable code can use.
//!
//! The `naive` column is the plain reference loops (`sky_linalg::reference`)
//! that define the results the fast kernels must reproduce.

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::Instant;

use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::{Accum, Mat, Par, Side};
use sky_linalg as sky;

const SIZES: [usize; 6] = [4, 16, 64, 256, 512, 1024];

/// A small deterministic generator, uniform in [-0.5, 0.5).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Runs each contender in rotation and returns the best (minimum) seconds
/// per call of each. Alternating them under the same conditions, and taking
/// the minimum of many short samples, keeps the ratios honest on a machine
/// that is doing other work.
fn race(contenders: &mut [Box<dyn FnMut() + '_>]) -> Vec<f64> {
    // Calibrate: repetitions per sample so that a sample takes about 2 ms.
    let reps: Vec<usize> = contenders
        .iter_mut()
        .map(|f| {
            f();
            let mut reps = 1usize;
            loop {
                let t = Instant::now();
                for _ in 0..reps {
                    f();
                }
                if t.elapsed().as_secs_f64() >= 0.002 || reps >= 1 << 24 {
                    return reps;
                }
                reps *= 2;
            }
        })
        .collect();
    let mut best = vec![f64::MAX; contenders.len()];
    let start = Instant::now();
    let mut rounds = 0;
    while rounds < 5 || (rounds < 400 && start.elapsed().as_secs_f64() < 1.5) {
        for (i, f) in contenders.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..reps[i] {
                f();
            }
            best[i] = best[i].min(t.elapsed().as_secs_f64() / reps[i] as f64);
        }
        rounds += 1;
    }
    best
}

fn random(rng: &mut Rng, len: usize) -> Vec<f64> {
    (0..len).map(|_| 2.0 * rng.next()).collect()
}

fn spd(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let x = rng.next();
            a[i * n + j] = x;
            a[j * n + i] = x;
        }
        a[i * n + i] += n as f64;
    }
    a
}

/// The vendor library through the C shim: `(op, n) -> GFLOP/s`.
fn vendor(sizes: &[usize], threads: usize) -> (String, HashMap<(String, usize), f64>) {
    let dir = std::env::temp_dir().join("sky-linalg-bench");
    let _ = std::fs::create_dir_all(&dir);
    let exe = dir.join("blas_shim");
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/blas_shim.c");
    let mut cc = if cfg!(target_os = "macos") {
        let mut c = Command::new("/usr/bin/clang");
        c.args([
            "-O3",
            "-Wno-deprecated-declarations",
            "-framework",
            "Accelerate",
        ]);
        c
    } else {
        let mut c = Command::new("cc");
        c.args(["-O3"]);
        c
    };
    cc.arg(src).arg("-o").arg(&exe);
    if !cfg!(target_os = "macos") {
        cc.args(["-lopenblas", "-lm"]);
    }
    let mut out = HashMap::new();
    let ok = cc
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return ("none".into(), out);
    }
    let t = threads.to_string();
    let run = Command::new(&exe)
        .args(sizes.iter().map(usize::to_string))
        .env("VECLIB_MAXIMUM_THREADS", &t)
        .env("OPENBLAS_NUM_THREADS", &t)
        .output();
    let mut name = "none".to_string();
    if let Ok(o) = run {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if let [lib, op, n, g] = f[..] {
                name = lib.to_string();
                out.insert((op.to_string(), n.parse().unwrap()), g.parse().unwrap());
            }
        }
    }
    (name, out)
}

fn matrixmultiply_gemm(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    assert!(a.len() >= n * n && b.len() >= n * n && c.len() >= n * n);
    let s = n as isize;
    // SAFETY: `a`, `b` and `c` each hold at least `n * n` elements (checked
    // above), the strides describe row-major n by n matrices, and `c` does
    // not alias `a` or `b` (it is a `&mut`).
    #[allow(unsafe_code)]
    unsafe {
        matrixmultiply::dgemm(
            n,
            n,
            n,
            1.0,
            a.as_ptr(),
            s,
            1,
            b.as_ptr(),
            s,
            1,
            0.0,
            c.as_mut_ptr(),
            s,
            1,
        );
    }
}

/// One row of the table: the op's GFLOP/s for each contender, `NaN` where a
/// library has no such routine.
struct Row {
    sky: f64,
    naive: f64,
    faer: f64,
    mm: f64,
}

fn measure(op: &str, n: usize, threads: usize, par: Par) -> Row {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1 ^ n as u64);
    let a = random(&mut rng, n * n);
    let b = random(&mut rng, n * n);
    let s = spd(&mut rng, n);
    let rhs = random(&mut rng, n);
    let fa = Mat::from_fn(n, n, |i, j| a[i * n + j]);
    let fb = Mat::from_fn(n, n, |i, j| b[i * n + j]);
    let fs = Mat::from_fn(n, n, |i, j| s[i * n + j]);
    let frhs = Mat::from_fn(n, 1, |i, _| rhs[i]);
    let mut c = vec![0.0; n * n];
    let mut fc = Mat::<f64>::zeros(n, n);
    let mut c2 = vec![0.0; n * n];
    let with_naive = n <= 256;
    let mut names = vec!["sky"];
    let mut contenders: Vec<Box<dyn FnMut() + '_>> = Vec::new();
    let flops = match op {
        "gemm" => {
            contenders.push(Box::new(|| {
                sky::gemm_into_with(
                    std::hint::black_box(&a),
                    n,
                    n,
                    std::hint::black_box(&b),
                    n,
                    &mut c,
                    threads,
                )
            }));
            if with_naive {
                names.push("naive");
                contenders.push(Box::new(|| {
                    std::hint::black_box(sky::reference::matmul(
                        std::hint::black_box(&a),
                        n,
                        n,
                        std::hint::black_box(&b),
                        n,
                    ));
                }));
            }
            names.push("faer");
            contenders.push(Box::new(|| {
                faer::linalg::matmul::matmul(
                    fc.as_mut(),
                    Accum::Replace,
                    fa.as_ref(),
                    fb.as_ref(),
                    1.0,
                    par,
                )
            }));
            names.push("mm");
            contenders.push(Box::new(|| {
                matrixmultiply_gemm(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                    &mut c2,
                    n,
                )
            }));
            2.0 * (n as f64).powi(3)
        }
        "potrf" => {
            contenders.push(Box::new(|| {
                std::hint::black_box(
                    sky::cholesky_with(std::hint::black_box(&s), n, threads).unwrap(),
                );
            }));
            if with_naive {
                names.push("naive");
                contenders.push(Box::new(|| {
                    std::hint::black_box(
                        sky::reference::cholesky(std::hint::black_box(&s), n).unwrap(),
                    );
                }));
            }
            names.push("faer");
            contenders.push(Box::new(|| {
                std::hint::black_box(fs.llt(Side::Lower).unwrap());
            }));
            (n as f64).powi(3) / 3.0
        }
        _ => {
            contenders.push(Box::new(|| {
                let f = sky::lu_with(std::hint::black_box(&a), n, threads).unwrap();
                std::hint::black_box(f.solve_with(&rhs, 1, threads));
            }));
            if with_naive {
                names.push("naive");
                contenders.push(Box::new(|| {
                    let f = sky::reference::lu(std::hint::black_box(&a), n).unwrap();
                    std::hint::black_box(sky::reference::lu_solve(&f, &rhs, 1));
                }));
            }
            names.push("faer");
            contenders.push(Box::new(|| {
                let lu = PartialPivLu::new(fa.as_ref());
                let mut x = frhs.clone();
                lu.solve_in_place(x.as_mut());
                std::hint::black_box(x);
            }));
            2.0 / 3.0 * (n as f64).powi(3) + 2.0 * (n as f64).powi(2)
        }
    };
    let secs = race(&mut contenders);
    let get = |name: &str| {
        names
            .iter()
            .position(|x| *x == name)
            .map_or(f64::NAN, |i| flops / secs[i] / 1e9)
    };
    Row {
        sky: get("sky"),
        naive: get("naive"),
        faer: get("faer"),
        mm: get("mm"),
    }
}

fn worker(threads: usize, sizes: &[usize]) {
    let par = if threads > 1 {
        Par::rayon(threads)
    } else {
        Par::Seq
    };
    faer::set_global_parallelism(par);
    let (vname, vend) = vendor(sizes, threads);
    println!("\n== {threads} thread(s) ==   GFLOP/s (higher is better); ratio = Sky / baseline");
    println!(
        "{:<6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>10}  {:>10} {:>8}",
        "op", "n", "sky", "naive", "faer", "matmul", vname, "vs faer/mm", "vs best"
    );
    let cell = |x: f64| {
        if x.is_nan() {
            "-".to_string()
        } else {
            format!("{x:.2}")
        }
    };
    let best = |xs: &[f64]| {
        xs.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f64::NAN, f64::max)
    };
    for op in ["gemm", "potrf", "gesv"] {
        for &n in sizes {
            let r = measure(op, n, threads, par);
            let v = vend.get(&(op.to_string(), n)).copied().unwrap_or(f64::NAN);
            println!(
                "{:<6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>10}  {:>10} {:>8}",
                op,
                n,
                cell(r.sky),
                cell(r.naive),
                cell(r.faer),
                cell(r.mm),
                cell(v),
                cell(r.sky / best(&[r.faer, r.mm])),
                cell(r.sky / best(&[r.faer, r.mm, v]))
            );
        }
    }
}

fn parse_list(var: &str) -> Option<Vec<usize>> {
    std::env::var(var)
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes = parse_list("BENCH_SIZES").unwrap_or_else(|| SIZES.to_vec());
    if let Some(p) = args.iter().position(|a| a == "--worker") {
        // One measurement process per thread count, so every library's
        // thread pool is configured before its first use.
        worker(args[p + 1].parse().unwrap(), &sizes);
        return;
    }
    let all = std::thread::available_parallelism().map_or(1, usize::from);
    let counts = parse_list("BENCH_THREADS").unwrap_or_else(|| vec![1, all]);
    println!("cores available: {all}");
    let size_list = sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    for t in counts {
        let s = t.to_string();
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--worker")
            .arg(&s)
            .env("SKY_THREADS", &s)
            .env("RAYON_NUM_THREADS", &s)
            .env("MATMUL_NUM_THREADS", &s)
            .env("BENCH_SIZES", &size_list)
            .status()
            .expect("worker");
        assert!(status.success());
    }
}
