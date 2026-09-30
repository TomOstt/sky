//! Allocations per loop iteration, on the tree-walker and on the SIR
//! interpreter (raw and optimized), measured with a counting global
//! allocator.
//!
//! This is the check behind the lifetime work (`sky_ir::lifetime`): arrays
//! are immutable values, so a loop like `v = v + dt * a` used to build a new
//! array per operation per iteration. When the operand dies at the
//! operation, its buffer now holds the result, and a steady-state loop
//! allocates nothing.
//!
//! How it is measured: a program is run for `SHORT` iterations and for `LONG`
//! ones, and the difference in allocations, divided by the difference in
//! iterations, is the cost of one iteration. The setup (arrays built before
//! the loop, output, frames) cancels. Run with `--nocapture` for the table.
//!
//! The counter is per thread, so tests running in parallel do not disturb one
//! another, and everything is measured inside `with_deep_stack`, where the
//! program runs.
//!
//! # `unsafe`
//!
//! A `GlobalAlloc` cannot be written without `unsafe`. The impl below only
//! forwards to `System` and bumps a counter; it lives in this test binary
//! (no shipped crate has it), which is the only place in the repository with
//! an `unsafe` block.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sky_derive::SirDerivatives;
use sky_interp::{run_module_with_derivatives, RuntimeEnv};

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

struct Counting;

// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the counter is a thread-local `Cell`
// that needs no allocation to update.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by `f` on this thread.
fn count<T>(f: impl FnOnce() -> T) -> (u64, T) {
    let before = ALLOCS.with(Cell::get);
    let v = f();
    (ALLOCS.with(Cell::get) - before, v)
}

const SHORT: u64 = 100;
const LONG: u64 = 1100;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Engine {
    TreeWalker,
    Sir,
    SirOpt,
}

/// Allocations of one run of `src` (its `N` replaced by `n`) on `engine`,
/// and its output.
fn run_counted(src: &str, n: u64, engine: Engine) -> (u64, String) {
    let src = src.replace("{N}", &n.to_string());
    sky_ast::with_deep_stack(|| {
        let src = src.as_str();
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "parse errors");
        let checked = sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
        assert!(!checked.has_errors(), "check errors");
        let raw = sky_lower::lower(&module, src, &checked).expect("lowers");
        let mut opt = raw.clone();
        sky_opt::default_pipeline()
            .run(&mut opt)
            .expect("the pipeline runs");
        let mut out = Vec::new();
        let (allocs, result) = count(|| match engine {
            Engine::TreeWalker => {
                let mut env = RuntimeEnv::with_prelude();
                let mut derivatives = SirDerivatives::new(&module, src);
                run_module_with_derivatives(
                    &module,
                    src,
                    &mut env,
                    &mut out,
                    &checked.cholesky_solves,
                    &mut derivatives,
                )
                .map(|_| ())
                .map_err(|e| e.message.clone())
            }
            Engine::Sir => sky_ir_interp::run(&raw, &mut out).map_err(|e| e.to_string()),
            Engine::SirOpt => sky_ir_interp::run(&opt, &mut out).map_err(|e| e.to_string()),
        });
        result.expect("the program runs");
        (allocs, String::from_utf8(out).expect("utf-8 output"))
    })
}

/// Allocations per iteration in steady state, and the outputs of both runs.
fn per_iteration(src: &str, engine: Engine) -> (f64, String) {
    let (short, _) = run_counted(src, SHORT, engine);
    let (long, out) = run_counted(src, LONG, engine);
    ((long as f64 - short as f64) / (LONG - SHORT) as f64, out)
}

/// A velocity-Verlet-like update of eight bodies on a line: the loop
/// `v = v + dt * a` of a time-stepping simulation.
const NBODY: &str = "
N = {N}
dt = 0.001 s
k = 2 / (1 s)^2
x = [1 m, 2 m, 3 m, 4 m, 5 m, 6 m, 7 m, 8 m]
v = [0 m/s, 0 m/s, 0 m/s, 0 m/s, 0 m/s, 0 m/s, 0 m/s, 0 m/s]
for i in range(N):
    a = -k * x
    v = v + dt * a
    x = x + dt * v
print(x, v)
";

/// The same with a scalar in the loop, math calls, and an `if`.
const MIXED: &str = "
N = {N}
dt = 0.001 s
k = 2 / (1 s)^2
x = [1 m, 2 m, 3 m, 4 m]
v = [0 m/s, 0 m/s, 0 m/s, 0 m/s]
e = 0 J
m = 1 kg
for i in range(N):
    a = -k * x
    v = v + dt * a
    if i / 2 == floor(i / 2):
        x = x + dt * v
    else:
        x = dt * v + x
    e = e + 0.5 * m * sqrt(v[0] * v[0] / (1 m/s)^2) * (1 m/s)^2
print(x, v, e)
";

/// `A @ v` and `v @ v` allocate their results (the values are new), but no
/// per-operation `Vec` on top of it.
const MATMUL: &str = "
N = {N}
R = [[0.6, -0.8], [0.8, 0.6]]
v = [1 m, 0 m]
s = 0 m^2
for i in range(N):
    v = R @ v
    s = s + v @ v
print(v, s)
";

#[test]
fn nbody_loop_allocates_nothing_per_iteration_on_sir() {
    let (tree, tree_out) = per_iteration(NBODY, Engine::TreeWalker);
    let (raw, raw_out) = per_iteration(NBODY, Engine::Sir);
    let (opt, opt_out) = per_iteration(NBODY, Engine::SirOpt);
    println!("nbody: allocations per iteration: tree-walker {tree}, SIR {raw}, SIR+opt {opt}");
    assert_eq!(tree_out, raw_out);
    assert_eq!(tree_out, opt_out);
    assert_eq!(opt, 0.0, "the optimized loop allocates in steady state");
    // Unfused, `-k * x` and `dt * v` are new arrays (`x` and `v` are still
    // needed after them), two allocations each (the entries and the shape);
    // every other operation reuses a buffer.
    assert_eq!(raw, 4.0, "the unoptimized loop's allocations changed");
}

#[test]
fn mixed_loop_allocates_only_what_it_must_on_sir() {
    let (tree, tree_out) = per_iteration(MIXED, Engine::TreeWalker);
    let (raw, raw_out) = per_iteration(MIXED, Engine::Sir);
    let (opt, opt_out) = per_iteration(MIXED, Engine::SirOpt);
    println!("mixed: allocations per iteration: tree-walker {tree}, SIR {raw}, SIR+opt {opt}");
    assert_eq!(tree_out, raw_out);
    assert_eq!(tree_out, opt_out);
    // `a = -k * x` is one new array (`x` is still needed after it): the
    // entries and the shape. Nothing else allocates: the `if`, the math
    // calls, `v[0]` and the scalar arithmetic are free, and `x` and `v`
    // are updated in place.
    assert_eq!(opt, 2.0, "the optimized loop's allocations changed");
}

#[test]
fn matmul_loop_allocations_are_reported() {
    let (tree, tree_out) = per_iteration(MATMUL, Engine::TreeWalker);
    let (raw, raw_out) = per_iteration(MATMUL, Engine::Sir);
    let (opt, opt_out) = per_iteration(MATMUL, Engine::SirOpt);
    println!("matmul: allocations per iteration: tree-walker {tree}, SIR {raw}, SIR+opt {opt}");
    assert_eq!(tree_out, raw_out);
    assert_eq!(tree_out, opt_out);
}

/// Every workload of `benches/programs`: allocations per iteration on the
/// three engines, printed as a table (run with `--nocapture`), and the three
/// engines must print the same. Only the reuse of the array loops is asserted
/// (see the other tests); the rest is a report, since a loop that builds
/// arrays, calls functions or takes derivatives allocates for reasons of its
/// own.
#[test]
fn every_benchmark_program_allocations_are_reported() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/programs");
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("benches/programs exists")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "sky"))
        // In these `N` is a depth or a matrix size, not an iteration count.
        .filter(|p| {
            p.file_stem().is_some_and(|x| {
                !matches!(
                    x.to_str(),
                    Some("recursion" | "chain_order" | "chain_order_forced")
                )
            })
        })
        .collect();
    names.sort();
    println!(
        "{:<14} {:>12} {:>8} {:>10}",
        "workload", "tree-walker", "SIR", "SIR+opt"
    );
    for path in names {
        let src = std::fs::read_to_string(&path).expect("readable");
        // The size is the line `N = <number>`; `{N}` stands for it.
        let templated: String = src
            .lines()
            .map(|l| if l.starts_with("N = ") { "N = {N}" } else { l })
            .collect::<Vec<_>>()
            .join("\n");
        let (tree, tree_out) = per_iteration(&templated, Engine::TreeWalker);
        let (raw, raw_out) = per_iteration(&templated, Engine::Sir);
        let (opt, opt_out) = per_iteration(&templated, Engine::SirOpt);
        let name = path.file_stem().unwrap().to_string_lossy();
        println!("{name:<14} {tree:>12.1} {raw:>8.1} {opt:>10.1}");
        assert_eq!(tree_out, raw_out, "{name}");
        assert_eq!(tree_out, opt_out, "{name}");
    }
}
