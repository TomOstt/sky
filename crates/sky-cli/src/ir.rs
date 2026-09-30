//! `sky check --emit-ir` and the pass half of `--explain`: lower a checked
//! module to SIR, run the default pipeline (`sky_opt::default_pipeline`),
//! and keep what the two flags print.
//!
//! Nothing here can make `sky check` fail because of the program: a program
//! that checks but cannot be lowered yet (`sky_lower::LowerError`, for
//! example array literals) has no IR, and [`build`] says so with a note. The
//! only error is a bug in a pass (invalid IR after it), which is reported as
//! an internal error.

use std::collections::HashSet;
use std::fmt;

use sky_ast::Module as Ast;
use sky_diag::{Diagnostic, Span};
use sky_ir::PassError;
use sky_lower::LowerError;
use sky_types::Checked;

use sky_opt::CHOLESKY_RULE;

/// Which IR `--emit-ir` prints.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum Stage {
    /// The IR after the whole pipeline (the default).
    #[default]
    Final,
    /// The IR straight out of lowering.
    Lowered,
    /// The IR after the named pass of `sky_opt::DEFAULT_PASSES`.
    After(String),
}

impl Stage {
    /// Parses the text after `--emit-ir=`: `lower`, `final`, `after:<pass>`,
    /// or a bare pass name.
    pub fn parse(text: &str) -> Result<Stage, StageError> {
        let pass = text.strip_prefix("after:").unwrap_or(text);
        match text {
            "lower" | "lowered" => Ok(Stage::Lowered),
            "final" | "end" => Ok(Stage::Final),
            _ if sky_opt::DEFAULT_PASSES.contains(&pass) => Ok(Stage::After(pass.to_string())),
            _ => Err(StageError(text.to_string())),
        }
    }
}

/// An unknown `--emit-ir` stage.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StageError(String);

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown IR stage `{}` (use `lower`, `final`, or `after:<pass>` with a pass from: {})",
            sky_diag::display_safe(&self.0),
            sky_opt::DEFAULT_PASSES.join(", ")
        )
    }
}

/// What lowering and the pipeline produced.
pub enum Outcome {
    /// The program has IR. `printed` holds the text of every stage when the
    /// caller asked for it.
    Ready(Box<Ready>),
    /// The program checks but has no IR yet: a note saying why, for
    /// `--emit-ir` to print.
    Unavailable(Diagnostic),
    /// A pass produced invalid IR: a bug in it, not in the program.
    Internal(String),
}

/// A lowered and optimized program.
pub struct Ready {
    /// The module after the whole pipeline: what `sky run` executes, and what
    /// `--emit=clif` compiles.
    pub module: sky_ir::Module,
    /// The text after lowering.
    pub lowered: String,
    /// The text after the whole pipeline.
    pub finished: String,
    /// The text after each pass, in pipeline order.
    pub after: Vec<(&'static str, String)>,
    /// The pass explanations as notes, for `--explain`.
    pub notes: Vec<Diagnostic>,
}

impl Ready {
    /// The text `--emit-ir` prints for `stage`.
    pub fn text(&self, stage: &Stage) -> &str {
        match stage {
            Stage::Final => &self.finished,
            Stage::Lowered => &self.lowered,
            // `Stage::parse` only accepts pipeline passes, and each runs.
            Stage::After(pass) => self
                .after
                .iter()
                .find(|(name, _)| name == pass)
                .map_or("", |(_, text)| text),
        }
    }
}

/// Lowers `module` (checked without errors, with `checked` its result) and
/// runs the default pipeline with verification after every pass.
///
/// `print` keeps the IR text of every stage (for `--emit-ir`); `explain`
/// turns the passes' explanations into notes, and adds what the parallelism
/// analysis (`sky_ir::parallel`) decided for `threads` threads.
///
/// # The Cholesky note is not shown twice
///
/// The checker reports `A⁻¹ @ b` sites itself, in operand source text ("`K⁻¹
/// @ f` ... K is Matrix[3,3] ..."), which the goldens pin word for word, and
/// `sky run` still needs its site set (`Checked::cholesky_solves`). The
/// `cholesky` pass reports the same decision, so its explanation is dropped
/// when it is at a span the checker rewrote. One it finds that the checker
/// did not (`B = inv(A)` and then `B @ b`, which only the IR sees through)
/// is shown, in the pass's own words.
pub fn build(
    module: &Ast,
    src: &str,
    checked: &Checked,
    print: bool,
    explain: bool,
    threads: usize,
) -> Outcome {
    let mut ir = match sky_lower::lower(module, src, checked) {
        Ok(ir) => ir,
        Err(e) => {
            let mut d = Diagnostic::note(format!("no IR: {}", e.message), e.span);
            if let Some(n) = e.note {
                d = d.with_note(n);
            }
            return Outcome::Unavailable(d.with_note(
                "the program checks, but lowering to SIR does not support it yet".to_string(),
            ));
        }
    };
    let lowered = if print {
        sky_ir::print(&ir)
    } else {
        String::new()
    };
    let mut pipeline = sky_opt::default_pipeline().verify_each(true);
    if print {
        pipeline = pipeline.print_after("*");
    }
    let report = match pipeline.run(&mut ir) {
        Ok(r) => r,
        Err(e) => return Outcome::Internal(internal(&e)),
    };
    let notes = if explain {
        let mut notes = explanation_notes(&report, src, &checked.cholesky_solves);
        notes.extend(parallel_notes(&ir, threads));
        notes.extend(reuse_notes(&ir, src));
        notes
    } else {
        Vec::new()
    };
    let finished = if print {
        sky_ir::print(&ir)
    } else {
        String::new()
    };
    Outcome::Ready(Box::new(Ready {
        module: ir,
        lowered,
        finished,
        after: report.printed,
        notes,
    }))
}

/// A checked program prepared for `sky run`'s SIR engine.
pub enum Runnable {
    /// The optimized, verified module, ready for `sky_ir_interp::run`.
    Ready(Box<sky_ir::Module>),
    /// SIR cannot run this program yet: the tree-walker must (`auto`), or
    /// the run fails (`--engine=sir`).
    Unsupported(LowerError),
    /// A pass produced invalid IR: a bug in it, not in the program.
    Internal(String),
}

/// Lowers `module` (checked without errors) and runs the default pipeline
/// with verification after every pass, for `sky run` to execute.
///
/// # Programs SIR does not run
///
/// A [`LowerError`] is one. (An input declaration, `A: Matrix[3, 3]` with
/// no value, runs on SIR: it lowers to an `input` operation that fails where
/// it is reached, after any earlier output, as on the tree-walker.)
pub fn prepare_run(module: &Ast, src: &str, checked: &Checked) -> Runnable {
    let mut ir = match sky_lower::lower_for_run(module, src, checked) {
        Ok(ir) => ir,
        Err(e) => return Runnable::Unsupported(e),
    };
    // Verified before the first pass and after each, so the interpreter
    // never sees IR the verifier rejects.
    match sky_opt::default_pipeline().verify_each(true).run(&mut ir) {
        Ok(_) => Runnable::Ready(Box::new(ir)),
        // IR the verifier rejects before any pass ran came out of lowering
        // itself: a program it does not handle yet (`"a" + "b"` types as
        // unknown to the checker, which leaves the error to run time, and
        // lowers to an `add` of strings), not one a pass broke. The
        // tree-walker gives it its run-time error.
        Err(PassError::InvalidInput(errors)) => Runnable::Unsupported(LowerError {
            message: "lowering produced SIR that does not verify".to_string(),
            span: errors.first().map_or(module.span, |e| e.span),
            note: errors.first().map(|e| e.message.clone()),
        }),
        // After a pass it is the pass that is wrong: a bug to surface.
        Err(e) => Runnable::Internal(internal(&e)),
    }
}

fn internal(e: &PassError) -> String {
    format!("internal error: {e}")
}

/// Adds what `sky tune` found for these linear algebra kernels on this machine
/// (`gemm`, `cholesky`) to the note of a decision that chooses them, e.g.
/// "GEMM blocks for sizes up to 256: tuned on this machine (2026-09-30):
/// mc 256, kc 128, nc 1024, 1.30x faster than the default ...". Says nothing
/// where nothing was tuned.
pub fn cite_tuning(mut d: Diagnostic, kernels: &[&str]) -> Diagnostic {
    for kernel in kernels {
        for line in sky_linalg::tuning::explain(kernel, None) {
            d = d.with_note(line);
        }
    }
    d
}

/// One explanation as a note. The Cholesky rewrite is worded like the
/// checker's note for it (same headline, same sentence), so it reads the
/// same whichever of the two reports it; everything else uses the generic
/// rendering.
fn note(x: &sky_ir::Explanation) -> Diagnostic {
    // A discharged contract reads "this `requires` always holds".
    if let Some(headline) = sky_opt::contract_headline(x.rule, &x.before) {
        let mut d = Diagnostic::note(headline, x.span)
            .with_note(format!("applied by the `{}` pass", x.pass));
        for reason in &x.because {
            d = d.with_note(format!("because {reason}"));
        }
        return d;
    }
    // A specialization reads "specialized `f` for N = 3 at this call".
    if let Some(headline) = sky_opt::specialize_headline(x) {
        let mut d = Diagnostic::note(headline, x.span)
            .with_note(format!("applied by the `{}` pass", x.pass));
        for reason in &x.because {
            d = d.with_note(format!("because {reason}"));
        }
        return d;
    }
    match sky_opt::cholesky_text(x) {
        Some(text) => Diagnostic::note(
            format!("`{}` is rewritten to a Cholesky solve", x.before),
            x.span,
        )
        .with_note(text)
        .with_note(format!("applied by the `{}` pass", x.pass)),
        None => x.to_diagnostic(),
    }
}

/// What can run in parallel, and what cannot and why, as notes
/// (`[PAR-001]`, `[PAR-002]`, `[PAR-003]`).
fn parallel_notes(ir: &sky_ir::Module, threads: usize) -> Vec<Diagnostic> {
    let cfg = sky_ir::parallel::Config::tuned(threads);
    sky_ir::parallel::explain(ir, &cfg)
        .into_iter()
        .map(|n| {
            let mut d = Diagnostic::note(format!("[{}] {}", n.rule, n.headline), n.span);
            for reason in n.because {
                d = d.with_note(format!("because {reason}"));
            }
            d
        })
        .collect()
}

/// The stable id of the note that an elementwise operation reuses the buffer
/// of an operand that dies there.
pub const MEM_RULE: &str = "MEM-001";

/// One note per operation that overwrites a dying operand's buffer with its
/// result (`sky_ir::Lifetimes::reuse_sites`), so `v = v + dt * a` in a
/// time-stepping loop says that it allocates nothing.
///
/// The note is about the source the user wrote: a fused kernel is quoted by
/// the expression it yields (`v + dt * a`), not by the statements it merged,
/// and the buffer is named by the variable, as written where it is read. A
/// site whose expression or variable has no source text to quote is left
/// out: a note about `%17` would say nothing to the user. These notes are
/// about what the SIR interpreter does (`sky run` on the tree-walker moves
/// the same buffers at a reassignment, but only `x = x + ...`).
fn reuse_notes(module: &sky_ir::Module, src: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for func in &module.functions {
        let life = sky_ir::Lifetimes::analyze(func);
        for site in life.reuse_sites() {
            let op = func.op(site.op);
            // The expression the result is: a fused kernel's yielded value.
            let expr_span = if let sky_ir::OpKind::Fused = op.kind {
                fused_result_span(func, op).unwrap_or(op.span)
            } else {
                op.span
            };
            let Some(expr) = one_line(src, expr_span) else {
                continue;
            };
            let operand = op.operands[site.operand];
            let Some(name) = func
                .value(operand)
                .name
                .clone()
                .filter(|n| n.chars().all(|c| c.is_alphanumeric() || c == '_'))
                .or_else(|| one_line(src, op.operand_span(site.operand)))
            else {
                continue;
            };
            let place = if site.in_loop {
                "no allocation in this loop"
            } else {
                "no allocation for this result"
            };
            out.push(
                Diagnostic::note(
                    format!("[{MEM_RULE}] `{expr}` reuses `{name}`'s buffer: {place}"),
                    expr_span,
                )
                .with_note(format!(
                    "`{name}` is not used again after this on any path, and has the shape of the result"
                ))
                .with_note("applied by the `lifetimes` analysis of the SIR interpreter".to_string()),
            );
        }
    }
    out
}

/// The span of the expression a `fused` kernel yields: the operation in its
/// body that produces the result (the kernel's own span covers every
/// statement it merged).
fn fused_result_span(func: &sky_ir::Function, op: &sky_ir::OpData) -> Option<Span> {
    let region = func.regions.get(op.regions.first()?.index())?;
    let block = func.blocks.get(region.blocks.first()?.index())?;
    let last = func.ops.get(block.ops.last()?.index())?;
    let yielded = *last.operands.first()?;
    match func.value(yielded).def {
        sky_ir::ValueDef::OpResult { op, .. } => Some(func.op(op).span),
        sky_ir::ValueDef::BlockArg { .. } => None,
    }
}

/// The source of `span` on one line, with no line breaks; `None` if it has
/// none or spans lines.
fn one_line(src: &str, span: Span) -> Option<String> {
    let text = sky_ast::ast::span_text(src, span)?;
    (!text.is_empty() && !text.contains('\n')).then(|| text.to_string())
}

/// The pass explanations as notes, in pipeline order (the caller sorts
/// them into source order with the other diagnostics).
///
/// Notes speak the user's language: a removal quotes the Sky source it came
/// from (the passes only see SIR, where it reads `%S = array.literal %0,
/// ...`), and a rewrite whose two sides read the same (folding `-2`, a
/// negated constant, into the constant `-2`) says nothing new and is left
/// out.
fn explanation_notes(
    report: &sky_ir::PipelineReport,
    src: &str,
    checker_solves: &HashSet<Span>,
) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = report
        .explanations
        .iter()
        .filter(|x| !(x.rule == CHOLESKY_RULE && checker_solves.contains(&x.span)))
        .filter(|x| x.after.is_empty() || x.before != x.after)
        .map(|x| {
            // `gvn` and `licm` say where the other place is: a reused value
            // was computed at line 4, a hoisted operation left the loop at
            // line 12.
            if let Some(headline) = sky_opt::motion_headline(x, src) {
                let mut d = Diagnostic::note(headline, x.span)
                    .with_note(format!("applied by the `{}` pass", x.pass));
                if let Some(related) = x.related {
                    d = d.with_secondary(
                        related,
                        if x.rule == sky_opt::GVN_RULE {
                            "computed here first"
                        } else {
                            "the loop"
                        },
                    );
                }
                for reason in &x.because {
                    d = d.with_note(format!("because {reason}"));
                }
                return d;
            }
            let source = sky_ast::ast::span_text(src, x.span).filter(|t| !t.is_empty());
            let d = match source {
                Some(text) if x.after.is_empty() => note(&sky_ir::Explanation {
                    before: text.to_string(),
                    ..x.clone()
                }),
                _ => note(x),
            };
            match x.pass {
                "algorithms" => cite_tuning(d, &["gemm"]),
                "cholesky" => cite_tuning(d, &["cholesky", "gemm"]),
                _ => d,
            }
        })
        .collect();
    if report.omitted_explanations > 0 {
        let n = report.omitted_explanations;
        let at = report
            .explanations
            .last()
            .map_or(Span::point(0), |x| x.span);
        out.push(Diagnostic::note(
            format!(
                "{n} more pass explanation{} not shown",
                if n == 1 { "" } else { "s" }
            ),
            at,
        ));
    }
    out
}
