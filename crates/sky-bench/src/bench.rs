//! The workloads and how one is measured.
//!
//! A workload is a Sky program in `benches/programs/*.sky`, embedded at
//! build time. Its first line is a header, `# bench: N=<full> quick=<small>`,
//! and the program sets its own size with a line `N = <full>`. [`Program::render`]
//! swaps that number for the `quick` one in smoke runs, so the tests exercise
//! the whole harness in milliseconds.
//!
//! Each workload runs through four engines:
//!
//! * `tree_walker` (a): `sky run`'s interpreter, exactly as the CLI drives it
//!   (derivatives go through `sky-derive`, Cholesky sites from the checker);
//! * `sir` (b): lowering, then the SIR reference interpreter, no passes;
//! * `sir_opt` (c): lowering, then `sky_opt::default_pipeline`, then the SIR
//!   interpreter;
//! * `native` (d): the same optimized SIR compiled to machine code by
//!   `sky-jit`, then run.
//!
//! Only the run itself is timed for an engine (the program is compiled
//! before). The compile-side phases (lex, parse, check, lower, pipeline,
//! verify, and the native compile) are timed separately. Engines alternate
//! within a repetition, so slow drift (thermal throttling, a noisy
//! neighbour) hits all four alike.

use std::time::Instant;

use sky_derive::SirDerivatives;
use sky_interp::{run_module_with_derivatives, RuntimeEnv};

use crate::stats::Stats;

/// The embedded workloads, by name. `every_program_file_is_registered`
/// checks this list against the directory.
const SOURCES: &[(&str, &str)] = &[
    (
        "scalar_loop",
        include_str!("../../../benches/programs/scalar_loop.sky"),
    ),
    (
        "recursion",
        include_str!("../../../benches/programs/recursion.sky"),
    ),
    (
        "math_calls",
        include_str!("../../../benches/programs/math_calls.sky"),
    ),
    (
        "array_chain",
        include_str!("../../../benches/programs/array_chain.sky"),
    ),
    (
        "matmul",
        include_str!("../../../benches/programs/matmul.sky"),
    ),
    (
        "chain_order",
        include_str!("../../../benches/programs/chain_order.sky"),
    ),
    (
        "chain_order_forced",
        include_str!("../../../benches/programs/chain_order_forced.sky"),
    ),
    (
        "cholesky",
        include_str!("../../../benches/programs/cholesky.sky"),
    ),
    (
        "derivative",
        include_str!("../../../benches/programs/derivative.sky"),
    ),
    (
        "gradient",
        include_str!("../../../benches/programs/gradient.sky"),
    ),
    (
        "vec3_physics",
        include_str!("../../../benches/programs/vec3_physics.sky"),
    ),
    ("nbody", include_str!("../../../benches/programs/nbody.sky")),
    (
        "licm_inverse",
        include_str!("../../../benches/programs/licm_inverse.sky"),
    ),
    (
        "cse_kernel",
        include_str!("../../../benches/programs/cse_kernel.sky"),
    ),
];

/// One workload.
#[derive(Clone, Debug)]
pub struct Program {
    pub name: &'static str,
    source: &'static str,
    /// The value of `N` in the file, and the one a `--quick` run uses.
    pub full: u64,
    pub quick: u64,
}

impl Program {
    fn parse(name: &'static str, source: &'static str) -> Result<Program, String> {
        let header = source
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("# bench:"))
            .ok_or_else(|| format!("{name}: the first line must be `# bench: N=<n> quick=<n>`"))?;
        let field = |key: &str| {
            header
                .split_whitespace()
                .find_map(|w| w.strip_prefix(key))
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| format!("{name}: no `{key}<number>` in the header"))
        };
        let program = Program {
            name,
            source,
            full: field("N=")?,
            quick: field("quick=")?,
        };
        let line = format!("N = {}", program.full);
        if !source.lines().any(|l| l == line) {
            return Err(format!("{name}: no line `{line}` to set the size"));
        }
        Ok(program)
    }

    /// The source text at the full or the quick size.
    pub fn render(&self, quick: bool) -> String {
        if !quick {
            return self.source.to_string();
        }
        let from = format!("N = {}", self.full);
        let to = format!("N = {}", self.quick);
        let mut out = String::with_capacity(self.source.len());
        // An array's size must be a literal (`identity(500)`), so a line that
        // ends in `# N` has its `(<full>)` swapped for `(<quick>)` as well.
        let (paren_from, paren_to) = (format!("({})", self.full), format!("({})", self.quick));
        for line in self.source.lines() {
            if line == from {
                out.push_str(&to);
            } else if line.ends_with("# N") {
                out.push_str(&line.replace(&paren_from, &paren_to));
            } else {
                out.push_str(line);
            }
            out.push('\n');
        }
        out
    }

    pub fn size(&self, quick: bool) -> u64 {
        if quick {
            self.quick
        } else {
            self.full
        }
    }
}

/// Every workload, in the order they are reported.
pub fn programs() -> Vec<Program> {
    SOURCES
        .iter()
        .map(|&(name, source)| Program::parse(name, source).unwrap_or_else(|e| panic!("{e}")))
        .collect()
}

/// How a run is measured.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Timed repetitions per measurement (at least 1).
    pub reps: usize,
    /// Untimed runs of each engine first (caches, allocator, page faults).
    pub warmup: usize,
    pub quick: bool,
}

/// The engines, in report order: JSON key, and the label in the table.
pub const ENGINES: [(&str, &str); 4] = [
    ("tree_walker", "tree-walk (a)"),
    ("sir", "SIR (b)"),
    ("sir_opt", "SIR+opt (c)"),
    ("native", "native (d)"),
];

/// The compile-side phases, in the order they run.
pub const PHASES: [&str; 7] = [
    "lex", "parse", "check", "lower", "pipeline", "verify", "native",
];

/// What measuring one workload found.
pub struct Measurement {
    pub name: &'static str,
    pub size: u64,
    /// Run times of the four [`ENGINES`].
    pub engines: Vec<Stats>,
    /// Compile times of the [`PHASES`].
    pub phases: Vec<Stats>,
    /// The program's stdout (the tree-walker's), the checksum.
    pub output: String,
    /// `None` if all four engines printed the same bytes in every
    /// repetition, else what differed.
    pub mismatch: Option<String>,
    /// Lines of printed SIR before and after the pipeline.
    pub ir_lines: (usize, usize),
}

fn nanos(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn timed<T>(f: impl FnOnce() -> T) -> (u64, T) {
    let t = Instant::now();
    let v = f();
    (nanos(t), v)
}

fn text(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|_| "the program printed invalid UTF-8".to_string())
}

/// Measures one workload: `Err` if it does not compile or run.
pub fn measure(program: &Program, cfg: Config) -> Result<Measurement, String> {
    let reps = cfg.reps.max(1);
    let src = program.render(cfg.quick);
    let name = program.name;
    let fail = |what: &str, why: String| format!("{name}: {what}: {why}");
    // Everything on the known-size stack, like `sky run`: the front end's
    // and the interpreters' depth limits assume it.
    sky_ast::with_deep_stack(|| {
        let src = src.as_str();
        let mut phases: Vec<Vec<u64>> = vec![Vec::new(); PHASES.len()];
        let mut engines: Vec<Vec<u64>> = vec![Vec::new(); ENGINES.len()];
        let mut outputs: Vec<Vec<String>> = vec![Vec::new(); ENGINES.len()];

        // Compile once for the runs, timing the phases each time.
        let compile = |record: Option<&mut Vec<Vec<u64>>>| -> Result<Compiled, String> {
            let (t_lex, lexed) = timed(|| sky_syntax::lex(src));
            let lexed = lexed.map_err(|_| "lexer error".to_string())?;
            let (t_parse, (module, errors)) = timed(|| sky_ast::parse_module(&lexed.tokens, src));
            if let Some(e) = errors.into_iter().next() {
                return Err(format!("parse error: {}", e.into_diagnostic().message));
            }
            let (t_check, checked) = timed(|| {
                sky_types::check_module_with(&module, src, sky_types::Env::with_prelude())
            });
            if checked.has_errors() {
                let first = checked
                    .diagnostics
                    .iter()
                    .find(|d| d.severity == sky_diag::Severity::Error)
                    .map_or_else(String::new, |d| d.message.clone());
                return Err(format!("check error: {first}"));
            }
            let (t_lower, raw) = timed(|| sky_lower::lower(&module, src, &checked));
            let raw = raw.map_err(|e| format!("lowering failed: {}", e.message))?;
            let mut opt = raw.clone();
            let (t_pipe, report) = timed(|| sky_opt::default_pipeline().run(&mut opt));
            report.map_err(|e| format!("pipeline failed: {e}"))?;
            let (t_verify, verified) = timed(|| sky_ir::verify(&opt));
            verified.map_err(|e| format!("invalid IR after the pipeline: {e:?}"))?;
            let (t_native, native) = timed(|| sky_jit::compile(&opt));
            let native = native.map_err(|e| e.to_string())?;
            if let Some(rec) = record {
                for (col, t) in rec
                    .iter_mut()
                    .zip([t_lex, t_parse, t_check, t_lower, t_pipe, t_verify, t_native])
                {
                    col.push(t);
                }
            }
            Ok(Compiled {
                module,
                solves: checked.cholesky_solves,
                chains: checked.chains,
                raw,
                opt,
                native,
            })
        };

        let compiled = compile(None).map_err(|e| fail("compile", e))?;
        let ir_lines = (
            sky_ir::print(&compiled.raw).lines().count(),
            sky_ir::print(&compiled.opt).lines().count(),
        );

        for _ in 0..cfg.warmup {
            for (engine, (key, _)) in ENGINES.iter().enumerate() {
                run_engine(engine, &compiled, src).map_err(|e| fail(key, e))?;
            }
        }
        for _ in 0..reps {
            // The compile-side phases are cheap: time them on every rep.
            compile(Some(&mut phases)).map_err(|e| fail("compile", e))?;
            for (engine, (key, _)) in ENGINES.iter().enumerate() {
                let (ns, out) = timed(|| run_engine(engine, &compiled, src));
                let out = out.map_err(|e| fail(key, e))?;
                engines[engine].push(ns);
                outputs[engine].push(text(out)?);
            }
        }

        let reference = outputs[0][0].clone();
        let mismatch = (0..ENGINES.len()).find_map(|e| {
            let bad = outputs[e].iter().find(|o| **o != reference)?;
            Some(format!(
                "{} printed {:?}, {} printed {:?}",
                ENGINES[e].0, bad, ENGINES[0].0, reference
            ))
        });
        Ok(Measurement {
            name,
            size: program.size(cfg.quick),
            engines: engines.into_iter().map(Stats::new).collect(),
            phases: phases.into_iter().map(Stats::new).collect(),
            output: reference,
            mismatch,
            ir_lines,
        })
    })
}

/// A program compiled once for the runs.
struct Compiled {
    module: sky_ast::Module,
    solves: std::collections::HashSet<sky_diag::Span>,
    chains: std::collections::HashMap<sky_diag::Span, std::sync::Arc<sky_types::ChainPlan>>,
    raw: sky_ir::Module,
    opt: sky_ir::Module,
    native: sky_jit::Compiled,
}

/// Runs one engine (index into [`ENGINES`]) and returns its stdout.
fn run_engine(engine: usize, c: &Compiled, src: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    match engine {
        0 => {
            let mut env = RuntimeEnv::with_prelude();
            env.add_chain_plans(&c.chains);
            // As in `sky run`: a derivative runs on SIR through `sky-derive`.
            let mut derivatives = SirDerivatives::new(&c.module, src);
            run_module_with_derivatives(
                &c.module,
                src,
                &mut env,
                &mut out,
                &c.solves,
                &mut derivatives,
            )
            .map_err(|e| e.message.clone())?;
        }
        1 => sky_ir_interp::run(&c.raw, &mut out).map_err(|e| e.to_string())?,
        3 => c.native.run(&mut out).map_err(|e| e.to_string())?,
        _ => sky_ir_interp::run(&c.opt, &mut out).map_err(|e| e.to_string())?,
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_program_file_is_registered() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/programs");
        let on_disk: BTreeSet<String> = std::fs::read_dir(dir)
            .expect("benches/programs exists")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .filter_map(|f| f.strip_suffix(".sky").map(str::to_string))
            .collect();
        let registered: BTreeSet<String> = SOURCES.iter().map(|(n, _)| n.to_string()).collect();
        assert_eq!(on_disk, registered);
    }

    #[test]
    fn headers_parse_and_quick_is_smaller() {
        for p in programs() {
            assert!(p.quick < p.full, "{}", p.name);
            let small = p.render(true);
            assert!(small.contains(&format!("N = {}\n", p.quick)), "{}", p.name);
            assert!(!small.contains(&format!("N = {}\n", p.full)), "{}", p.name);
        }
    }

    #[test]
    fn a_missing_header_is_an_error() {
        assert!(Program::parse("x", "print(1)\n").is_err());
        assert!(Program::parse("x", "# bench: N=5 quick=1\nprint(1)\n").is_err());
        assert!(Program::parse("x", "# bench: N=5 quick=1\nN = 5\nprint(N)\n").is_ok());
    }

    /// The whole harness on tiny sizes: every workload compiles, runs
    /// through all four engines, and the engines agree.
    #[test]
    fn smoke_every_workload_agrees_across_engines() {
        let cfg = Config {
            reps: 2,
            warmup: 0,
            quick: true,
        };
        for p in programs() {
            let m = measure(&p, cfg).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(m.mismatch, None, "{}", p.name);
            assert!(!m.output.is_empty(), "{} printed nothing", p.name);
            assert_eq!(m.engines.len(), ENGINES.len());
            assert_eq!(m.phases.len(), PHASES.len());
            assert!(m.engines.iter().all(|s| s.samples.len() == 2));
        }
    }

    #[test]
    fn a_broken_program_is_an_error_not_a_panic() {
        let p = Program {
            name: "broken",
            source: "# bench: N=2 quick=1\nN = 2\nprint(1 m + 1 s)\n",
            full: 2,
            quick: 1,
        };
        let cfg = Config {
            reps: 1,
            warmup: 0,
            quick: false,
        };
        let e = measure(&p, cfg).err().expect("does not check");
        assert!(e.contains("check error"), "{e}");
    }
}
