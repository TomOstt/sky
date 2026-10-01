//! `sky-bench compare`: Sky against C, Rust, Python and a BLAS/LAPACK
//! library on the same workloads, written to `benches/compare/RESULTS.md`.
//!
//! Every implementation is a separate program that prints its result (numbers
//! on the last line of stdout); the harness compiles what needs compiling,
//! runs each one as a subprocess with a warmup and repetitions, checks that
//! all results agree within the workload's tolerance, and reports medians of
//! wall time. Wall time includes process start-up and, for Sky, the whole
//! front end (lex, parse, check, lower): the `floor` rows in the report
//! measure the fixed part so it can be read off. A tool that is missing (no
//! NumPy, no BLAS, no `native` engine yet) means its column is skipped, never
//! an error.

mod report;
mod tools;
pub mod workloads;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::stats::Stats;
use tools::{Machine, Tool};
use workloads::Case;

const USAGE: &str = "usage: sky-bench compare [--filter <substring>] [--reps <n>] [--warmup <n>]\n\
                     \x20                      [--quick] [--out <file.md>] [--sky <path>] [--no-build]\n\
                     \x20                      [--no-sky] [--timeout <secs>] [--slow <secs>]\n\x20                      [--emit-sky <dir>] [--threads <n>] [--list]\n\n\
                     --filter    only cases whose id contains the text (see --list)\n\
                     --reps      timed repetitions per cell (default 5)\n\
                     --warmup    untimed runs first (default 1)\n\
                     --quick     tiny sizes, for a smoke run (writes no report unless --out)\n\
                     --out       where to write the Markdown report\n\
                     \x20           (default benches/compare/RESULTS.md, not written with --quick)\n\
                     --sky       the `sky` binary (default: built with cargo, next to this one)\n\
                     --no-build  do not run `cargo build --release -p sky-cli` first\n\
                     --no-sky    leave the Sky rows out\n\
                     --timeout   kill a single run after this many seconds (default 300)\n\
                     --slow      a cell whose first run takes longer than this is measured\n\
                     \x20           once instead of repeated (default 20)\n\
                     --emit-sky  also write each rendered Sky program into this directory\n\
                     --threads   run every Sky program with `--threads=N` (default: the\n\
                     \x20           machine's: SKY_THREADS, else the cores)\n\
                     --list      print the case ids and exit\n\n\
                     Run it with `cargo run --release -p sky-bench -- compare`.";

/// Engines of `sky run --engine=`, in report order.
pub const SKY_ENGINES: [&str; 4] = ["ast", "sir", "native", "auto"];

/// The id of the column that runs Sky's default engine (`ast`) with
/// `--numerics=relaxed` (fused multiply-add, reassociated sums, Accelerate on
/// macOS): not bit-reproducible, so it is a column of its own, never the
/// "best Sky engine" the interpretation compares with, and it is checked
/// against the same reference within the workload's tolerance.
pub const SKY_RELAXED: &str = "sky-relaxed";

/// Every implementation column, in report order: id and table label.
pub const IMPLS: [(&str, &str); 11] = [
    ("c", "C -O3"),
    ("c-fast", "C -O3 -ffast-math"),
    ("rust", "Rust -O3"),
    ("lib", "BLAS/LAPACK"),
    ("python", "CPython"),
    ("numpy", "NumPy"),
    ("sky-ast", "Sky ast"),
    ("sky-sir", "Sky sir"),
    ("sky-native", "Sky native"),
    ("sky-auto", "Sky auto"),
    (SKY_RELAXED, "Sky ast, relaxed"),
];

pub fn is_sky(id: &str) -> bool {
    id.starts_with("sky-")
}

/// A Sky column that is an engine with the default numerics (not
/// [`SKY_RELAXED`]).
pub fn is_sky_engine(id: &str) -> bool {
    is_sky(id) && id != SKY_RELAXED
}

/// How a run is configured.
#[derive(Clone, Debug)]
pub struct Options {
    pub filter: Option<String>,
    pub reps: usize,
    pub warmup: usize,
    pub quick: bool,
    pub out: Option<PathBuf>,
    pub sky: Option<PathBuf>,
    pub build_sky: bool,
    pub no_sky: bool,
    pub timeout: Duration,
    pub slow: Duration,
    pub list: bool,
    /// Also write each rendered Sky program here.
    pub emit_sky: Option<PathBuf>,
    /// Run every `sky run` with `--threads=N` (else the machine's default:
    /// `SKY_THREADS`, or the cores).
    pub threads: Option<usize>,
    /// `benches/compare`.
    pub dir: PathBuf,
}

/// The outcome of one (case, implementation) cell.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// Ran and printed a result.
    Ran(Ran),
    /// Not attempted: a tool is missing.
    Skipped(String),
    /// Attempted and failed: does not compile or check, crashed, timed out.
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct Ran {
    pub stats: Stats,
    /// The numbers on the last line of stdout.
    pub values: Vec<f64>,
    /// Measured once because the first run exceeded `--slow`.
    pub single: bool,
    /// Set when the result disagrees with the reference implementation's.
    pub mismatch: Option<String>,
}

pub struct CaseResult {
    pub case: Case,
    /// In [`IMPLS`] order.
    pub cells: Vec<(&'static str, Outcome)>,
}

impl CaseResult {
    pub fn cell(&self, id: &str) -> Option<&Outcome> {
        self.cells.iter().find(|(i, _)| *i == id).map(|(_, o)| o)
    }

    /// The median wall time of a cell that ran and agrees with the reference.
    pub fn time(&self, id: &str) -> Option<f64> {
        match self.cell(id)? {
            Outcome::Ran(r) if r.mismatch.is_none() => Some(r.stats.median as f64),
            _ => None,
        }
    }
}

/// What is installed, for the report.
pub struct ToolInfo {
    pub c: Option<Tool>,
    pub rust: Option<Tool>,
    pub python: Option<Tool>,
    pub numpy: Option<String>,
    pub lib: Option<String>,
    pub sky: Option<PathBuf>,
    /// Engines this `sky` accepts, and the ones it does not.
    pub engines: Vec<&'static str>,
    pub missing_engines: Vec<&'static str>,
}

/// Fixed costs: an empty process, and `sky run` on `print(1)`.
#[derive(Default)]
pub struct Floors {
    pub c_ns: Option<u64>,
    pub python_ns: Option<u64>,
    pub sky_ns: Vec<(&'static str, u64)>,
}

pub struct Run {
    pub machine: Machine,
    pub tools: ToolInfo,
    pub floors: Floors,
    pub cases: Vec<CaseResult>,
    pub opts: Options,
    pub rev: Option<String>,
    pub date: Option<String>,
    /// The 1-minute load average when the measurements started and ended.
    pub load: (Option<f64>, Option<f64>),
}

fn parse_args(args: &[String], dir: PathBuf) -> Result<Options, String> {
    let mut o = Options {
        filter: None,
        reps: 5,
        warmup: 1,
        quick: false,
        out: None,
        sky: None,
        build_sky: true,
        no_sky: false,
        timeout: Duration::from_secs(300),
        slow: Duration::from_secs(20),
        list: false,
        emit_sky: None,
        threads: None,
        dir,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        let number = |flag: &str, v: String| {
            v.parse::<u64>()
                .map_err(|_| format!("{flag} needs a whole number, found `{v}`"))
        };
        match arg.as_str() {
            "--filter" => o.filter = Some(value("--filter")?),
            "--reps" => {
                o.reps = number("--reps", value("--reps")?)? as usize;
                if o.reps == 0 {
                    return Err("--reps must be at least 1".to_string());
                }
            }
            "--warmup" => o.warmup = number("--warmup", value("--warmup")?)? as usize,
            "--quick" => o.quick = true,
            "--out" => o.out = Some(PathBuf::from(value("--out")?)),
            "--sky" => o.sky = Some(PathBuf::from(value("--sky")?)),
            "--no-build" => o.build_sky = false,
            "--no-sky" => o.no_sky = true,
            "--timeout" => {
                o.timeout = Duration::from_secs(number("--timeout", value("--timeout")?)?)
            }
            "--slow" => o.slow = Duration::from_secs(number("--slow", value("--slow")?)?),
            "--emit-sky" => o.emit_sky = Some(PathBuf::from(value("--emit-sky")?)),
            "--threads" => {
                let n = number("--threads", value("--threads")?)? as usize;
                if !(1..=sky_ir::parallel::MAX_THREADS).contains(&n) {
                    return Err(format!(
                        "--threads must be from 1 to {}",
                        sky_ir::parallel::MAX_THREADS
                    ));
                }
                o.threads = Some(n);
            }
            "--list" => o.list = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(o)
}

/// The directory holding the programs: `benches/compare` of this checkout.
pub fn default_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benches/compare")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Entry point of `sky-bench compare`; returns the process exit code.
pub fn main(args: &[String]) -> std::process::ExitCode {
    use std::process::ExitCode;
    let opts = match parse_args(args, default_dir()) {
        Ok(o) => o,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let selected = select(&opts);
    if opts.list {
        for c in &selected {
            println!("{}", c.id);
        }
        return ExitCode::SUCCESS;
    }
    if selected.is_empty() {
        eprintln!("error: no case matches the filter");
        return ExitCode::from(2);
    }
    if cfg!(debug_assertions) {
        eprintln!("warning: a debug build; the harness times other programs, but use `cargo run --release -p sky-bench -- compare`\n");
    }
    let run = match execute(&opts, &selected) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    print!("{}", report::terminal(&run));
    let target = opts
        .out
        .clone()
        .or_else(|| (!opts.quick && opts.filter.is_none()).then(|| opts.dir.join("RESULTS.md")));
    if let Some(path) = target {
        if let Err(e) = std::fs::write(&path, report::markdown(&run)) {
            eprintln!("error: cannot write {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {}", path.display());
    }
    let disagree = run
        .cases
        .iter()
        .flat_map(|c| c.cells.iter())
        .any(|(_, o)| matches!(o, Outcome::Ran(r) if r.mismatch.is_some()));
    if disagree {
        eprintln!("error: some implementations disagree with the reference result");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn select(opts: &Options) -> Vec<Case> {
    workloads::cases()
        .into_iter()
        .filter(|c| {
            opts.filter
                .as_ref()
                .is_none_or(|f| c.id.contains(f.as_str()))
        })
        .collect()
}

/// A compiled program, or why there is none.
type Built = Result<PathBuf, String>;

/// Everything a run needs to start programs.
struct Ctx<'a> {
    opts: &'a Options,
    build: PathBuf,
    c: Option<Tool>,
    rust: Option<Tool>,
    python: Option<Tool>,
    numpy: bool,
    sky: Option<PathBuf>,
    engines: Vec<&'static str>,
    /// Compiled programs by (implementation, kernel).
    built: HashMap<(&'static str, &'static str), Built>,
}

/// Runs the comparison for `cases`.
pub fn execute(opts: &Options, cases: &[Case]) -> Result<Run, String> {
    let build = std::env::temp_dir().join(format!("sky-compare-{}", std::process::id()));
    std::fs::create_dir_all(&build)
        .map_err(|e| format!("cannot create {}: {e}", build.display()))?;
    let result = execute_in(opts, cases, &build);
    let _ = std::fs::remove_dir_all(&build);
    result
}

fn execute_in(opts: &Options, cases: &[Case], build: &Path) -> Result<Run, String> {
    let c = tools::find_c();
    let rust = tools::find_rust(&workspace_root());
    let (python, numpy) = tools::find_python();

    let mut sky = None;
    if !opts.no_sky {
        if opts.sky.is_none() && opts.build_sky {
            match tools::build_sky(&workspace_root()) {
                Ok(p) => sky = Some(p),
                Err(e) => eprintln!("note: {e}; looking for an existing binary"),
            }
        }
        if sky.is_none() {
            sky = tools::find_sky(opts.sky.as_deref());
        }
        if sky.is_none() {
            eprintln!("note: no `sky` binary found: the Sky rows are skipped");
        }
    }

    // Which engines does this `sky` have?
    let probe = build.join("probe.sky");
    std::fs::write(&probe, "print(1)\n").map_err(|e| e.to_string())?;
    let mut engines = Vec::new();
    let mut missing_engines = Vec::new();
    if let Some(bin) = &sky {
        for e in SKY_ENGINES {
            let out = tools::run_timed(
                Command::new(bin)
                    .arg("run")
                    .arg(format!("--engine={e}"))
                    .arg(&probe),
                Duration::from_secs(30),
            );
            match out {
                Ok(o) if o.success => engines.push(e),
                _ => missing_engines.push(e),
            }
        }
    }

    let lib = lib_description(c.as_ref(), build);
    let mut ctx = Ctx {
        opts,
        build: build.to_path_buf(),
        c: c.clone(),
        rust: rust.clone(),
        python: python.clone(),
        numpy: numpy.is_some(),
        sky: sky.clone(),
        engines: engines.clone(),
        built: HashMap::new(),
    };

    let load_start = tools::load_average();
    let floors = measure_floors(&ctx, &probe);

    let mut results = Vec::new();
    for case in cases {
        eprintln!("== {} ==", case.id);
        results.push(run_case(&mut ctx, case)?);
    }
    for r in &mut results {
        verify(r);
    }
    let load_end = tools::load_average();

    Ok(Run {
        machine: tools::machine(),
        tools: ToolInfo {
            c,
            rust,
            python,
            numpy,
            lib,
            sky,
            engines,
            missing_engines,
        },
        floors,
        cases: results,
        opts: opts.clone(),
        rev: std::env::var("SKY_BENCH_REV")
            .ok()
            .or_else(|| tools::capture("git", &["rev-parse", "--short=12", "HEAD"])),
        date: tools::capture("date", &["-u", "+%Y-%m-%d"]),
        load: (load_start, load_end),
    })
}

/// Which BLAS/LAPACK the `lib` programs would link, if one links at all.
fn lib_description(cc: Option<&Tool>, build: &Path) -> Option<String> {
    let cc = cc?;
    let src = build.join("blas_probe.c");
    let out = build.join("blas_probe");
    let body = if cfg!(target_os = "macos") {
        "#define ACCELERATE_NEW_LAPACK\n#include <Accelerate/Accelerate.h>\nint main(void){double a=1,b=1,c=0;cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0,&a,1,&b,1,0.0,&c,1);return c==1.0?0:1;}\n"
    } else {
        "#include <cblas.h>\nint main(void){double a=1,b=1,c=0;cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0,&a,1,&b,1,0.0,&c,1);return c==1.0?0:1;}\n"
    };
    std::fs::write(&src, body).ok()?;
    tools::compile_c(cc, &src, &out, false, true).ok()?;
    if cfg!(target_os = "macos") {
        Some("Apple Accelerate (cblas_dgemm, LAPACK dpotrf/dpotrs)".to_string())
    } else {
        let v = tools::capture("pkg-config", &["--modversion", "openblas"]);
        Some(match v {
            Some(v) => format!("OpenBLAS {v} (cblas_dgemm, LAPACK dpotrf/dpotrs)"),
            None => "OpenBLAS (cblas_dgemm, LAPACK dpotrf/dpotrs)".to_string(),
        })
    }
}

fn median_of(samples: Vec<u64>) -> u64 {
    Stats::new(samples).median
}

fn measure_floors(ctx: &Ctx, probe: &Path) -> Floors {
    let t = Duration::from_secs(60);
    let n = if ctx.opts.quick { 3 } else { 15 };
    let mut floors = Floors::default();
    let time = |mk: &dyn Fn() -> Command| -> Option<u64> {
        let mut samples = Vec::new();
        for _ in 0..n {
            let out = tools::run_timed(&mut mk(), t).ok()?;
            if !out.success {
                return None;
            }
            samples.push(out.wall_ns);
        }
        Some(median_of(samples))
    };
    if let Some(cc) = &ctx.c {
        let src = ctx.build.join("floor.c");
        let bin = ctx.build.join("floor");
        if std::fs::write(&src, "int main(void){return 0;}\n").is_ok()
            && tools::compile_c(cc, &src, &bin, false, false).is_ok()
        {
            floors.c_ns = time(&|| Command::new(&bin));
        }
    }
    if let Some(py) = &ctx.python {
        floors.python_ns = time(&|| {
            let mut c = Command::new(&py.program);
            c.args(["-c", "pass"]);
            c
        });
    }
    if let Some(sky) = &ctx.sky {
        for e in &ctx.engines {
            if let Some(ns) = time(&|| {
                let mut c = Command::new(sky);
                c.arg("run").arg(format!("--engine={e}")).arg(probe);
                c
            }) {
                floors.sky_ns.push((e, ns));
            }
        }
    }
    floors
}

/// Compiles the C or Rust program for (`impl_id`, `case.kernel`), once.
fn build_program(ctx: &mut Ctx, impl_id: &'static str, case: &Case) -> Built {
    let key = (impl_id, case.kernel);
    if let Some(b) = ctx.built.get(&key) {
        return b.clone();
    }
    let d = &ctx.opts.dir;
    let out = ctx.build.join(format!("{impl_id}-{}", case.kernel));
    let built = match impl_id {
        "c" | "c-fast" => match &ctx.c {
            None => Err("no C compiler".to_string()),
            Some(cc) => tools::compile_c(
                cc,
                &d.join("c").join(format!("{}.c", case.kernel)),
                &out,
                impl_id == "c-fast",
                false,
            ),
        },
        "lib" => match &ctx.c {
            None => Err("no C compiler".to_string()),
            Some(cc) => {
                let src = d.join("lib").join(format!("{}.c", case.kernel));
                if !src.is_file() {
                    Err("no library program for this workload".to_string())
                } else {
                    tools::compile_c(cc, &src, &out, false, true)
                        .map_err(|e| format!("no BLAS/LAPACK to link: {e}"))
                }
            }
        },
        "rust" => match &ctx.rust {
            None => Err("no rustc".to_string()),
            Some(rc) => tools::compile_rust(
                rc,
                &d.join("rust").join(format!("{}.rs", case.kernel)),
                &out,
            ),
        },
        _ => Err(format!("`{impl_id}` is not compiled")),
    };
    let built = built.map(|_| out);
    ctx.built.insert(key, built.clone());
    built
}

/// The command for one cell, or why it cannot run (`Err`: skipped).
fn command_for(
    ctx: &mut Ctx,
    impl_id: &'static str,
    case: &Case,
    sky_file: &Path,
) -> Result<Command, String> {
    let args = case.args(ctx.opts.quick);
    match impl_id {
        "c" | "c-fast" | "rust" | "lib" => {
            let bin = build_program(ctx, impl_id, case)?;
            let mut c = Command::new(bin);
            c.args(&args);
            Ok(c)
        }
        "python" | "numpy" => {
            let py = ctx.python.as_ref().ok_or("no python3")?;
            let (sub, ext) = if impl_id == "numpy" {
                if !ctx.numpy {
                    return Err("numpy is not installed".to_string());
                }
                if !case.numpy {
                    return Err("no NumPy program for this workload".to_string());
                }
                ("numpy", "py")
            } else {
                ("python", "py")
            };
            let mut c = Command::new(&py.program);
            c.arg(
                ctx.opts
                    .dir
                    .join(sub)
                    .join(format!("{}.{ext}", case.kernel)),
            );
            c.args(&args);
            Ok(c)
        }
        _ => {
            let engine = impl_id
                .strip_prefix("sky-")
                .ok_or("unknown implementation")?;
            let relaxed = impl_id == SKY_RELAXED;
            let engine = if relaxed { "ast" } else { engine };
            let sky = ctx.sky.as_ref().ok_or("no sky binary")?;
            if !ctx.engines.contains(&engine_name(engine)) {
                return Err(format!("this `sky` has no `{engine}` engine"));
            }
            let mut c = Command::new(sky);
            c.arg("run").arg(format!("--engine={engine}"));
            if let Some(n) = ctx.opts.threads {
                c.arg(format!("--threads={n}"));
            }
            if relaxed {
                c.arg("--numerics=relaxed");
            }
            c.arg(sky_file);
            Ok(c)
        }
    }
}

fn engine_name(e: &str) -> &'static str {
    SKY_ENGINES.iter().copied().find(|x| *x == e).unwrap_or("")
}

fn run_case(ctx: &mut Ctx, case: &Case) -> Result<CaseResult, String> {
    let quick = ctx.opts.quick;
    // The Sky program for this case, rendered once.
    let template_path = ctx.opts.dir.join("sky").join(format!("{}.sky", case.sky));
    let sky_file = ctx.build.join(format!("{}.sky", case.id));
    if let Ok(template) = std::fs::read_to_string(&template_path) {
        let text = case.render_sky(&template, quick)?;
        std::fs::write(&sky_file, &text).map_err(|e| e.to_string())?;
        if let Some(dir) = &ctx.opts.emit_sky {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join(format!("{}.sky", case.id)), &text)
                .map_err(|e| e.to_string())?;
        }
    }
    let mut cells = Vec::new();
    for (id, _) in IMPLS {
        if is_sky(id) && ctx.opts.no_sky {
            continue;
        }
        if (id == "lib" && !case.lib) || (id == "numpy" && !case.numpy) {
            cells.push((
                id,
                Outcome::Skipped("no program for this workload".to_string()),
            ));
            continue;
        }
        let outcome = match command_for(ctx, id, case, &sky_file) {
            Err(why) => Outcome::Skipped(why),
            Ok(cmd) => {
                let mk = cmd_factory(&cmd);
                measure_cell(&mk, ctx.opts)
            }
        };
        match &outcome {
            Outcome::Ran(r) => eprintln!("  {id:<11} {}", tools_fmt(r.stats.median)),
            Outcome::Skipped(why) => eprintln!("  {id:<11} skipped: {why}"),
            Outcome::Failed(why) => eprintln!("  {id:<11} FAILED: {why}"),
        }
        cells.push((id, outcome));
    }
    Ok(CaseResult {
        case: case.clone(),
        cells,
    })
}

fn tools_fmt(ns: u64) -> String {
    format!("{:.1} ms", ns as f64 / 1e6)
}

/// A rebuildable command (a `Command` cannot be cloned).
fn cmd_factory(cmd: &Command) -> impl Fn() -> Command {
    let program = cmd.get_program().to_os_string();
    let args: Vec<_> = cmd.get_args().map(|a| a.to_os_string()).collect();
    move || {
        let mut c = Command::new(&program);
        c.args(&args);
        c
    }
}

/// The numbers on the last non-empty line of `stdout`.
pub fn parse_result(stdout: &str) -> Result<Vec<f64>, String> {
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or("printed nothing")?;
    line.split_whitespace()
        .map(|w| {
            w.parse::<f64>()
                .map_err(|_| format!("the last line is not numbers: `{}`", line.trim()))
        })
        .collect()
}

fn one_line(stderr: &str) -> String {
    let line = stderr
        .lines()
        .find(|l| l.contains("error") || l.contains("panicked"))
        .or_else(|| stderr.lines().find(|l| !l.trim().is_empty()))
        .unwrap_or("failed with no message");
    // Diagnostics start with `file:line:col: `; keep the message.
    let line = line.split_once(": error: ").map_or(line, |(_, m)| m);
    line.trim().chars().take(160).collect()
}

/// Times one cell: a first (warmup) run that also yields the result, then
/// repetitions, unless the first run was so slow that once is all we take.
fn measure_cell(mk: &dyn Fn() -> Command, opts: &Options) -> Outcome {
    let run = |mk: &dyn Fn() -> Command| -> Result<(u64, Vec<f64>), String> {
        let out = tools::run_timed(&mut mk(), opts.timeout)?;
        if out.timed_out {
            return Err(format!("timed out after {} s", opts.timeout.as_secs()));
        }
        if !out.success {
            return Err(one_line(&out.stderr));
        }
        Ok((out.wall_ns, parse_result(&out.stdout)?))
    };
    let (first_ns, values) = match run(mk) {
        Ok(v) => v,
        Err(e) => return Outcome::Failed(e),
    };
    if Duration::from_nanos(first_ns) > opts.slow {
        return Outcome::Ran(Ran {
            stats: Stats::new(vec![first_ns]),
            values,
            single: true,
            mismatch: None,
        });
    }
    let mut samples = Vec::new();
    if opts.warmup == 0 {
        samples.push(first_ns);
    }
    for _ in 1..opts.warmup {
        if let Err(e) = run(mk) {
            return Outcome::Failed(e);
        }
    }
    let mut mismatch = None;
    while samples.len() < opts.reps {
        match run(mk) {
            Ok((ns, v)) => {
                if v != values && mismatch.is_none() {
                    mismatch = Some("the result changed between runs".to_string());
                }
                samples.push(ns);
            }
            Err(e) => return Outcome::Failed(e),
        }
    }
    Outcome::Ran(Ran {
        stats: Stats::new(samples),
        values,
        single: false,
        mismatch,
    })
}

/// `a` and `b` agree within relative tolerance `tol` (exactly, for 0).
pub fn close(a: f64, b: f64, tol: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= tol * a.abs().max(b.abs())
}

/// Marks every implementation whose result disagrees with the reference
/// (the first that ran, in [`IMPLS`] order, so C when there is a compiler).
fn verify(r: &mut CaseResult) {
    let reference = r.cells.iter().find_map(|(id, o)| match o {
        Outcome::Ran(x) => Some((*id, x.values.clone())),
        _ => None,
    });
    let Some((ref_id, ref_values)) = reference else {
        return;
    };
    let tol = r.case.tolerance;
    for (id, o) in &mut r.cells {
        let Outcome::Ran(x) = o else { continue };
        if *id == ref_id {
            continue;
        }
        let ok = x.values.len() == ref_values.len()
            && x.values
                .iter()
                .zip(&ref_values)
                .all(|(a, b)| close(*a, *b, tol));
        if !ok && x.mismatch.is_none() {
            x.mismatch = Some(format!(
                "printed {:?}, {ref_id} printed {:?} (tolerance {tol:e})",
                x.values, ref_values
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Result<Options, String> {
        parse_args(
            &a.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            PathBuf::from("."),
        )
    }

    #[test]
    fn flags_parse() {
        let o = args(&[
            "--filter",
            "mat",
            "--reps",
            "3",
            "--quick",
            "--no-sky",
            "--timeout",
            "9",
        ])
        .expect("valid");
        assert_eq!(o.filter.as_deref(), Some("mat"));
        assert_eq!((o.reps, o.quick, o.no_sky), (3, true, true));
        assert_eq!(o.timeout, Duration::from_secs(9));
        assert!(args(&["--reps", "0"]).is_err());
        assert!(args(&["--reps"]).is_err());
        assert!(args(&["--nope"]).is_err());
    }

    #[test]
    fn results_parse_from_the_last_line() {
        assert_eq!(
            parse_result("noise\n1 2.5e-3 -4\n\n").unwrap(),
            vec![1.0, 0.0025, -4.0]
        );
        assert!(parse_result("").is_err());
        assert!(parse_result("1 two").is_err());
    }

    #[test]
    fn closeness_is_relative() {
        assert!(close(1.0, 1.0 + 1e-12, 1e-9));
        assert!(!close(1.0, 1.001, 1e-9));
        assert!(close(1e-5, 1e-5 * (1.0 + 1e-10), 1e-9));
        assert!(!close(1e-5, 2e-5, 1e-9));
        assert!(close(0.0, 0.0, 0.0));
        assert!(!close(1.0, 1.000_000_000_000_000_2, 0.0));
    }

    fn ran(values: &[f64]) -> Outcome {
        Outcome::Ran(Ran {
            stats: Stats::new(vec![1_000_000]),
            values: values.to_vec(),
            single: false,
            mismatch: None,
        })
    }

    #[test]
    fn verify_flags_the_odd_one_out() {
        let case = workloads::cases().remove(0);
        let mut r = CaseResult {
            case,
            cells: vec![
                ("c", ran(&[1.0, 2.0])),
                ("rust", ran(&[1.0, 2.0])),
                ("python", ran(&[1.0, 2.5])),
                ("sky-ast", Outcome::Failed("nope".to_string())),
            ],
        };
        verify(&mut r);
        assert!(r.time("c").is_some() && r.time("rust").is_some());
        assert!(
            r.time("python").is_none(),
            "a disagreeing cell has no usable time"
        );
        assert!(matches!(r.cell("python"), Some(Outcome::Ran(x)) if x.mismatch.is_some()));
    }

    #[test]
    fn the_program_directory_has_a_file_for_every_case() {
        let d = default_dir();
        for c in workloads::cases() {
            assert!(
                d.join("sky").join(format!("{}.sky", c.sky)).is_file(),
                "sky/{}",
                c.sky
            );
            for (sub, ext) in [("c", "c"), ("rust", "rs"), ("python", "py")] {
                assert!(
                    d.join(sub).join(format!("{}.{ext}", c.kernel)).is_file(),
                    "{sub}/{}",
                    c.kernel
                );
            }
            if c.lib {
                assert!(
                    d.join("lib").join(format!("{}.c", c.kernel)).is_file(),
                    "lib/{}",
                    c.kernel
                );
            }
            if c.numpy {
                assert!(
                    d.join("numpy").join(format!("{}.py", c.kernel)).is_file(),
                    "numpy/{}",
                    c.kernel
                );
            }
        }
    }

    /// Runs Sky source on the tree-walker, in process (no `sky` binary needed).
    fn run_ast(src: &str) -> Result<String, String> {
        sky_ast::with_deep_stack(|| {
            let lexed = sky_syntax::lex(src).map_err(|_| "lexer error".to_string())?;
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
            if let Some(e) = errors.into_iter().next() {
                return Err(format!("parse error: {}", e.into_diagnostic().message));
            }
            let checked =
                sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
            if let Some(d) = checked
                .diagnostics
                .iter()
                .find(|d| d.severity == sky_diag::Severity::Error)
            {
                return Err(format!("check error: {}", d.message));
            }
            let mut env = sky_interp::RuntimeEnv::with_prelude();
            let mut derivatives = sky_derive::SirDerivatives::new(&module, src);
            let mut out = Vec::new();
            sky_interp::run_module_with_derivatives(
                &module,
                src,
                &mut env,
                &mut out,
                &checked.cholesky_solves,
                &mut derivatives,
            )
            .map_err(|e| e.message.clone())?;
            String::from_utf8(out).map_err(|e| e.to_string())
        })
    }

    /// Every Sky template, rendered at the quick size, runs on the
    /// tree-walker and prints what the C program prints (when a C compiler is
    /// there).
    #[test]
    fn smoke_sky_templates_agree_with_c() {
        let dir = default_dir();
        let cc = tools::find_c();
        let build = std::env::temp_dir().join(format!("sky-compare-test-{}", std::process::id()));
        std::fs::create_dir_all(&build).expect("temp dir");
        let mut ran = 0;
        for case in workloads::cases() {
            let template =
                std::fs::read_to_string(dir.join("sky").join(format!("{}.sky", case.sky)))
                    .unwrap_or_else(|e| panic!("{}: {e}", case.sky));
            let src = case.render_sky(&template, true).expect("renders");
            let sky_out = match run_ast(&src) {
                Ok(o) => o,
                Err(e) => panic!("{}: Sky failed: {e}", case.id),
            };
            let sky_values = parse_result(&sky_out).expect("Sky prints numbers");
            ran += 1;
            let Some(cc) = &cc else { continue };
            let bin = build.join(&case.id);
            tools::compile_c(
                cc,
                &dir.join("c").join(format!("{}.c", case.kernel)),
                &bin,
                false,
                false,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
            let out = tools::run_timed(
                Command::new(&bin).args(case.args(true)),
                Duration::from_secs(60),
            )
            .expect("C runs");
            assert!(out.success, "{}: {}", case.id, out.stderr);
            let c_values = parse_result(&out.stdout).expect("C prints numbers");
            assert_eq!(c_values.len(), sky_values.len(), "{}", case.id);
            for (a, b) in sky_values.iter().zip(&c_values) {
                assert!(
                    close(*a, *b, case.tolerance),
                    "{}: Sky {a} vs C {b}",
                    case.id
                );
            }
        }
        let _ = std::fs::remove_dir_all(&build);
        assert!(ran >= 8, "only {ran} templates ran");
    }

    /// The harness end to end on the cheapest case, without Sky: whatever
    /// toolchains exist run, agree with each other, and make a report.
    #[test]
    fn smoke_harness_end_to_end() {
        let mut o = args(&[
            "--quick", "--reps", "1", "--warmup", "0", "--no-sky", "--filter", "fib",
        ])
        .expect("valid");
        o.dir = default_dir();
        let cases = select(&o);
        assert_eq!(cases.len(), 1);
        let run = execute(&o, &cases).expect("runs");
        let r = &run.cases[0];
        for (id, out) in &r.cells {
            match out {
                Outcome::Ran(x) => assert!(x.mismatch.is_none(), "{id}: {:?}", x.mismatch),
                Outcome::Failed(why) => panic!("{id} failed: {why}"),
                Outcome::Skipped(_) => {}
            }
        }
        let md = report::markdown(&run);
        assert!(md.contains("## Results") && md.contains("`fib-30`"), "{md}");
    }
}
