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
/// turns the passes' explanations into notes.
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
pub fn build(module: &Ast, src: &str, checked: &Checked, print: bool, explain: bool) -> Outcome {
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
        explanation_notes(&report, &checked.cholesky_solves)
    } else {
        Vec::new()
    };
    Outcome::Ready(Box::new(Ready {
        lowered,
        finished: if print {
            sky_ir::print(&ir)
        } else {
            String::new()
        },
        after: report.printed,
        notes,
    }))
}

fn internal(e: &PassError) -> String {
    format!("internal error: {e}")
}

/// One explanation as a note. The Cholesky rewrite is worded like the
/// checker's note for it (same headline, same sentence), so it reads the
/// same whichever of the two reports it; everything else uses the generic
/// rendering.
fn note(x: &sky_ir::Explanation) -> Diagnostic {
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

/// The pass explanations as notes, in pipeline order (the caller sorts
/// them into source order with the other diagnostics).
fn explanation_notes(
    report: &sky_ir::PipelineReport,
    checker_solves: &HashSet<Span>,
) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = report
        .explanations
        .iter()
        .filter(|x| !(x.rule == CHOLESKY_RULE && checker_solves.contains(&x.span)))
        .map(note)
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
