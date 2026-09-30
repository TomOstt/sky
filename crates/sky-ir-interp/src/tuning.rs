//! What `sky tune` measures for the SIR interpreter: the element count from
//! which a `fused` operation is compiled to a native kernel, and the sizes
//! from which `fused` kernels and elementwise array operations are split
//! across threads.
//!
//! All three thresholds choose between ways of computing the same thing
//! (the interpreter's loop or a native kernel; one thread or several, with
//! every output entry the same `f64` operations), so tuning them changes
//! speed and never an output. The tests below run each candidate and compare
//! the answers.
//!
//! Each is measured over a ladder of sizes, each size repeated so that all
//! cost about the same to the interpreter, so a candidate wins by being
//! right across the ladder rather than at the largest size.
//!
//! | kernel | parameter | default | what it decides |
//! |---|---|---|---|
//! | `fused_native` (`cold`) | `min_elements` | 512 | compile a `fused` body once the interpreter has run this many elements |
//! | `par_fused_work` (`all`) | `work` | 4096 | split a `fused` kernel of at least this many element-operations |
//! | `par_array_elements` (`all`) | `elements` | 262144 | split an elementwise array operation of at least this many elements |

use sky_interp::Value;
use sky_ir::parallel::{Config, ARRAY_ELEMENTS_KERNEL, FUSED_WORK_KERNEL};
use sky_ir::parse;
use sky_ir::Module;
use sky_tune::{grid, Registry, Tunable, Work};
use sky_units::Dimension;

use crate::{call_with_options, JitOptions};

/// A module with a function `@f` running a four-operation `fused` body over
/// `n` elements.
fn fused_module(n: usize) -> Module {
    let text = format!(
        "sir edition 2026\n\nfn @f(%a: num[1; ({n})], %b: num[1; ({n})], %k: num[1; ()]) -> (num[1; ({n})]) effects(fail) {{
  %r = fused %a, %b, %k : num[1; ({n})] {{
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
      %t = add %x, %y : num[1; ()]
      %u = mul %t, %s : num[1; ()]
      %w = sub %u, %x : num[1; ()]
      %v = add %w, %y : num[1; ()]
      yield %v
  }}
  return %r
}}
"
    );
    parse(&text).expect("the fused kernel parses")
}

/// A module with a function `@f` adding two arrays of `n` elements.
fn add_module(n: usize) -> Module {
    let text = format!(
        "sir edition 2026\n\nfn @f(%a: num[1; ({n})], %b: num[1; ({n})]) -> (num[1; ({n})]) effects(fail) {{
  %r = add %a, %b : num[1; ({n})]
  return %r
}}
"
    );
    parse(&text).expect("the add parses")
}

fn array(n: usize, scale: f64) -> Value {
    let data = (0..n).map(|i| (i % 97) as f64 * scale + 0.5).collect();
    Value::array(vec![n], data, Dimension::DIMENSIONLESS)
}

/// A digest of the results, so the timed call has an answer to keep.
fn digest(values: &[Value]) -> u64 {
    values.iter().fold(0u64, |h, v| match v {
        Value::Array { data, .. } => data
            .iter()
            .step_by(data.len().div_ceil(32).max(1))
            .fold(h, |h, x| h.rotate_left(5) ^ x.to_bits()),
        _ => h,
    })
}

/// One size of a ladder: the module, its arguments and how often it runs.
struct Rung {
    module: Module,
    args: Vec<Value>,
    reps: usize,
}

fn run(rungs: &[Rung], cfg: &Config, jit: JitOptions) -> u64 {
    let mut h = 0;
    for rung in rungs {
        for _ in 0..rung.reps {
            let out = call_with_options(
                &rung.module,
                "f",
                rung.args.clone(),
                &mut std::io::sink(),
                cfg,
                jit,
            )
            .expect("the kernel runs");
            h ^= digest(&out);
        }
    }
    h
}

fn fused_rungs(sizes: &[usize], budget: usize) -> Vec<Rung> {
    sizes
        .iter()
        .map(|&n| Rung {
            module: fused_module(n),
            args: vec![array(n, 0.01), array(n, 0.02), Value::scalar(1.5)],
            reps: (budget / n).max(1),
        })
        .collect()
}

/// The elements from which the interpreter compiles a `fused`: each call is
/// a fresh run, so it pays the compile (a run's first long `fused`).
fn native_tunables() -> Vec<Tunable> {
    let sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let rungs = fused_rungs(&sizes, 1);
    let candidates = grid(&[("min_elements", &[512, 128, 256, 1024, 2048, 4096, 8192])]);
    let cfg = Config::sequential();
    let tunable = Tunable::new(
        sky_jit::tuning::KERNEL,
        "cold",
        candidates,
        Work::Unknown,
        move |p| {
            let jit = JitOptions {
                enabled: true,
                min_elements: p.get_clamped("min_elements", 512, 0, 1 << 24) as usize,
            };
            run(&rungs, &cfg, jit)
        },
    );
    vec![tunable.expect("valid tunable")]
}

/// The work from which a `fused` kernel is split, interpreted (native
/// kernels off, which is when the split is what runs a large kernel).
fn fused_work_tunables() -> Vec<Tunable> {
    let sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    let rungs = fused_rungs(&sizes, 65536);
    let candidates = grid(&[("work", &[4096, 1024, 2048, 8192, 16384, 65536, 262_144])]);
    let threads = Config::default_threads();
    let tunable = Tunable::new(
        FUSED_WORK_KERNEL,
        "all",
        candidates,
        Work::Unknown,
        move |p| {
            let cfg = Config {
                fused_work: p.get_clamped("work", 4096, 0, 1 << 30) as usize,
                ..Config::with_threads(threads)
            };
            run(&rungs, &cfg, JitOptions::OFF)
        },
    );
    vec![tunable.expect("valid tunable")]
}

/// The elements from which an elementwise operation on arrays is split.
fn array_elements_tunables() -> Vec<Tunable> {
    let sizes = [16384, 32768, 65536, 131_072, 262_144, 524_288, 1_048_576];
    let rungs: Vec<Rung> = sizes
        .iter()
        .map(|&n| Rung {
            module: add_module(n),
            args: vec![array(n, 0.01), array(n, 0.02)],
            reps: ((1 << 20) / n).max(1),
        })
        .collect();
    let candidates = grid(&[(
        "elements",
        &[
            262_144, 32768, 65536, 131_072, 524_288, 1_048_576, 4_194_304,
        ],
    )]);
    let threads = Config::default_threads();
    let tunable = Tunable::new(
        ARRAY_ELEMENTS_KERNEL,
        "all",
        candidates,
        Work::Unknown,
        move |p| {
            let cfg = Config {
                array_elements: p.get_clamped("elements", 262_144, 2, 1 << 30) as usize,
                ..Config::with_threads(threads)
            };
            run(&rungs, &cfg, JitOptions::OFF)
        },
    );
    vec![tunable.expect("valid tunable")]
}

/// Adds this crate's tunables to `registry` (`sky tune` calls this): the
/// native-kernel threshold (`interp_native`), the `fused` split threshold
/// (`interp_fused_work`) and the array-operation split threshold
/// (`interp_array_elements`).
pub fn register(registry: &mut Registry) {
    registry
        .add("interp_native", native_tunables)
        .add("interp_fused_work", fused_work_tunables)
        .add("interp_array_elements", array_elements_tunables);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_candidate_gives_the_same_answers() {
        for build in [
            native_tunables,
            fused_work_tunables,
            array_elements_tunables,
        ] {
            for t in build() {
                let first = t.run(&t.candidates[0]);
                for p in &t.candidates {
                    assert_eq!(t.run(p), first, "{} {p}", t.kernel);
                }
            }
        }
    }

    #[test]
    fn defaults_come_first() {
        let first = |t: Vec<Tunable>, name: &str| t[0].candidates[0].get(name).unwrap();
        let d = Config::with_threads(1);
        assert_eq!(
            first(native_tunables(), "min_elements"),
            sky_jit::tuning::DEFAULT_KERNEL_THRESHOLD as i64
        );
        assert_eq!(first(fused_work_tunables(), "work"), d.fused_work as i64);
        assert_eq!(
            first(array_elements_tunables(), "elements"),
            d.array_elements as i64
        );
    }
}
