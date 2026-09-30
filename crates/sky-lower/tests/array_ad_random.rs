//! Random programs over vectors, differentiated three ways: the gradient by
//! reverse mode, the same function forced through the forward-mode
//! fallback (a loop carrying the result rules reverse mode out), and a
//! central finite difference. They must agree.
//!
//! The programs are straight-line code over `Vector[3]` and scalars (sums,
//! scaling, products with constant matrices, dot products, indexing,
//! literals, a few scalar functions, and conditional expressions), built by
//! a small deterministic generator, so a failure prints the program that
//! failed and reruns identically.

mod common;

use common::lower_ok;
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

/// xorshift64*: enough randomness for choosing operations.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A small number with a fraction, like `1.5` or `-0.25`.
    fn small(&mut self) -> String {
        let n = self.below(9) as i64 - 4;
        let f = [0, 25, 5, 75][self.below(4)];
        format!("{n}.{f:02}")
    }
}

fn matrix_literal(rng: &mut Rng) -> String {
    let rows: Vec<String> = (0..3)
        .map(|_| {
            let e: Vec<String> = (0..3).map(|_| rng.small()).collect();
            format!("[{}]", e.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

/// The body of `f(p: Vector[3]) -> m`: returns the source and the scalar
/// it computes.
fn program(rng: &mut Rng, steps: usize) -> String {
    let mut lines = Vec::new();
    // Vectors and scalars defined so far.
    let mut vecs = vec!["p".to_string()];
    let mut scas = vec!["p[0]".to_string()];
    for k in 0..steps {
        let a = vecs[rng.below(vecs.len())].clone();
        let b = vecs[rng.below(vecs.len())].clone();
        let s = scas[rng.below(scas.len())].clone();
        let t = scas[rng.below(scas.len())].clone();
        let name_v = format!("v{k}");
        let name_s = format!("s{k}");
        match rng.below(14) {
            0 => {
                lines.push(format!("{name_v} = {a} + {b}"));
                vecs.push(name_v);
            }
            1 => {
                lines.push(format!("{name_v} = {a} - {b}"));
                vecs.push(name_v);
            }
            2 => {
                lines.push(format!("{name_v} = {a} * {s}"));
                vecs.push(name_v);
            }
            3 => {
                lines.push(format!("{name_v} = {a} / (1 + {s} * {s})"));
                vecs.push(name_v);
            }
            4 => {
                lines.push(format!("{name_v} = {} @ {a}", matrix_literal(rng)));
                vecs.push(name_v);
            }
            5 => {
                lines.push(format!("{name_v} = {a} @ {}", matrix_literal(rng)));
                vecs.push(name_v);
            }
            6 => {
                lines.push(format!("{name_s} = {a} @ {b}"));
                scas.push(name_s);
            }
            7 => {
                lines.push(format!("{name_s} = {a}[{}]", rng.below(3)));
                scas.push(name_s);
            }
            8 => {
                lines.push(format!("{name_s} = {s} * {t}"));
                scas.push(name_s);
            }
            9 => {
                lines.push(format!("{name_s} = sqrt(1 + {s} * {s})"));
                scas.push(name_s);
            }
            10 => {
                lines.push(format!("{name_s} = sin({s}) + {t}"));
                scas.push(name_s);
            }
            11 => {
                lines.push(format!("{name_v} = [{s}, {t}, {}[{}]]", a, rng.below(3)));
                vecs.push(name_v);
            }
            12 => {
                lines.push(format!("{name_v} = {a} if {s} > {t} else {b} * 2"));
                vecs.push(name_v);
            }
            _ => {
                let m = matrix_literal(rng);
                lines.push(format!("{name_v} = ({m}ᵀ @ {m}) @ {a} / 4"));
                vecs.push(name_v);
            }
        }
    }
    // The result: a dot product of the last vectors and a scalar.
    let last = vecs.last().unwrap().clone();
    let first = vecs[vecs.len() / 2].clone();
    let sc = scas.last().unwrap().clone();
    lines.push(format!("result = {last} @ {first} + {sc}"));
    lines.join("\n    ")
}

fn source(body: &str, looped: bool) -> String {
    if looped {
        // A loop that carries the result: reverse mode has no tape for it,
        // so the gradient comes from the forward fallback.
        format!(
            "def f(p: Vector[3]) -> m:\n    {body}\n    acc = 0\n    i = 0\n    while i < 1:\n        acc = acc + result\n        i += 1\n    return 1 m * acc\n"
        )
    } else {
        format!("def f(p: Vector[3]) -> m:\n    {body}\n    return 1 m * result\n")
    }
}

fn run(m: &sky_ir::Module, name: &str, p: &[f64]) -> Result<Vec<Value>, String> {
    let args = vec![Value::array(vec![3], p.to_vec(), Dimension::DIMENSIONLESS)];
    sky_ast::with_deep_stack(|| call(m, name, args, &mut Vec::new())).map_err(|e| e.message.clone())
}

/// The gradient by AD and by central differences at `p`; `None` when `f`
/// itself fails at `p` (an overflow, say), where there is nothing to
/// compare. Everything else must work: a program that does not lower, a
/// derivative that is not built or is ill-typed, or a derivative that fails
/// where `f` does not, is a bug.
fn gradient(src: &str, p: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut m = lower_ok(src);
    let name = sky_ad::differentiate_into(&mut m, "f", 0)
        .unwrap_or_else(|e| panic!("cannot differentiate: {e}\n{src}"));
    if let Err(errors) = sky_ir::verify(&m) {
        panic!("ill-typed derivative: {errors:?}\n{src}");
    }
    let value = |q: &[f64]| -> Option<f64> {
        match run(&m, "f", q).ok()?.first()? {
            Value::Number { value, .. } => Some(*value),
            other => panic!("not a number: {other:?}"),
        }
    };
    let f_at = value(p)?;
    let out = run(&m, &name, p).unwrap_or_else(|e| panic!("the derivative fails: {e}\n{src}"));
    assert_eq!(
        out[0],
        Value::number(f_at, Dimension::LENGTH),
        "the primal value\n{src}"
    );
    let Value::Array { data, .. } = &out[1] else {
        panic!("the gradient is not an array: {:?}", out[1]);
    };
    let mut fd = Vec::new();
    for i in 0..3 {
        let h = 1e-6;
        let (mut lo, mut hi) = (p.to_vec(), p.to_vec());
        lo[i] -= h;
        hi[i] += h;
        fd.push((value(&hi)? - value(&lo)?) / (2.0 * h));
    }
    Some((data.clone(), fd))
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= tol * y.abs().max(x.abs()).max(1.0))
}

#[test]
fn reverse_forward_and_finite_differences_agree_on_random_programs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    for case in 0..300 {
        let steps = 2 + rng.below(7);
        let body = program(&mut rng, steps);
        let p: Vec<f64> = (0..3).map(|_| 0.3 + rng.below(100) as f64 / 60.0).collect();
        let reverse_src = source(&body, false);
        let forward_src = source(&body, true);
        let (Some((rev, fd)), Some((fwd, _))) =
            (gradient(&reverse_src, &p), gradient(&forward_src, &p))
        else {
            continue;
        };
        assert!(
            close(&rev, &fwd, 1e-9),
            "case {case}: reverse {rev:?} vs forward {fwd:?} at {p:?}\n{reverse_src}"
        );
        assert!(
            close(&rev, &fd, 1e-4),
            "case {case}: reverse {rev:?} vs finite differences {fd:?} at {p:?}\n{reverse_src}"
        );
        checked += 1;
    }
    assert!(checked >= 250, "only {checked} of 300 programs ran");
}
