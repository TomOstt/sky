//! Passes, the pass manager, and explanations.
//!
//! Every capability after lowering is a [`Pass`]: it reads or rewrites a
//! [`Module`] and says what it did. The roadmap's fourth principle ("always
//! explained") is enforced by the shape of the API: a transform reports each
//! decision through [`PassCx::explain`], which fills the pass name in
//! itself, so an explanation cannot be attributed to the wrong pass, and
//! `sky check --explain` needs no side channel to collect them.
//!
//! The [`PassManager`] runs a pipeline and, when verification is on (the
//! default in debug builds), runs [`verify`] before the first pass and after
//! every pass. A verifier failure after a pass is a bug in that pass, so the
//! error names it; for tests, [`PassManager::for_tests`] panics instead of
//! returning it, so a broken pass fails the test that ran it at the point of
//! the bug. (`cfg(test)` cannot do this for us: it is false in downstream
//! crates, which are where the real passes live.)
//!
//! The passes themselves live elsewhere: effect inference in this crate
//! ([`crate::effects`]), constant folding, dead-code elimination and the
//! Cholesky rewrite in `sky-opt`, e-graph rewriting in `sky-egraph`, and
//! `sky_opt::default_pipeline` puts them in order. The design document's
//! `level()` method is left out until there is a second level.

use std::fmt;

use sky_diag::{display_capped, Diagnostic, Span};

use crate::ir::Module;
use crate::print::print;
use crate::verify::{verify, VerifyError};

/// A stable identifier of a rewrite rule, like `"LA-CHOLESKY-001"`, so users
/// and documentation can refer to it.
pub type RuleId = &'static str;

/// Whether a pass modified the module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[must_use]
pub enum Changed {
    No,
    Yes,
}

impl Changed {
    pub fn is_yes(self) -> bool {
        self == Changed::Yes
    }

    /// `Yes` if either is.
    pub fn or(self, other: Changed) -> Changed {
        if self.is_yes() || other.is_yes() {
            Changed::Yes
        } else {
            Changed::No
        }
    }
}

impl From<bool> for Changed {
    fn from(b: bool) -> Self {
        if b {
            Changed::Yes
        } else {
            Changed::No
        }
    }
}

/// What a transform did, and why it was allowed to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Explanation {
    /// The pass that made the change.
    pub pass: &'static str,
    pub rule: RuleId,
    /// The Sky source of the code that changed.
    pub span: Span,
    /// The code before, in a form a user can read (usually its source form).
    pub before: String,
    /// The code after. Empty when the code was removed.
    pub after: String,
    /// The facts and type properties the rule relied on, one per entry
    /// ("`A` is PositiveDefinite", "`b` is Vector[3]").
    pub because: Vec<String>,
}

impl Explanation {
    /// A note for `sky check --explain`: points at the changed code, names
    /// the rule and pass, and lists the reasons.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let headline = if self.after.is_empty() {
            format!(
                "[{}] `{}` is removed",
                self.rule,
                display_capped(&self.before)
            )
        } else {
            format!(
                "[{}] `{}` is rewritten to `{}`",
                self.rule,
                display_capped(&self.before),
                display_capped(&self.after)
            )
        };
        let mut d = Diagnostic::note(headline, self.span)
            .with_note(format!("applied by the `{}` pass", self.pass));
        for reason in &self.because {
            d = d.with_note(format!("because {reason}"));
        }
        d
    }
}

/// Most explanations one [`PassCx`] keeps; later ones are counted in
/// [`PassCx::omitted`]. The cap never stops a pass and never changes what it
/// does, as with the checker's rewrite sites.
pub const MAX_EXPLANATIONS: usize = 10_000;

/// What a running pass can reach besides the module: the collector for
/// explanations.
#[derive(Debug, Default)]
pub struct PassCx {
    pass: &'static str,
    explanations: Vec<Explanation>,
    omitted: usize,
}

impl PassCx {
    pub fn new() -> Self {
        Self::default()
    }

    /// The name of the pass that is running (set by the manager).
    pub fn pass_name(&self) -> &'static str {
        self.pass
    }

    /// Records a decision of the running pass.
    pub fn explain(
        &mut self,
        rule: RuleId,
        span: Span,
        before: impl Into<String>,
        after: impl Into<String>,
        because: Vec<String>,
    ) {
        if self.explanations.len() >= MAX_EXPLANATIONS {
            self.omitted += 1;
            return;
        }
        self.explanations.push(Explanation {
            pass: self.pass,
            rule,
            span,
            before: before.into(),
            after: after.into(),
            because,
        });
    }

    pub fn explanations(&self) -> &[Explanation] {
        &self.explanations
    }

    /// How many explanations were dropped by the cap.
    pub fn omitted(&self) -> usize {
        self.omitted
    }

    pub fn into_explanations(self) -> Vec<Explanation> {
        self.explanations
    }
}

/// A transform or analysis over a module.
pub trait Pass {
    /// A stable, short name (`"cholesky"`, `"dce"`), used in errors,
    /// explanations, and `--emit-ir=after:<name>`.
    fn name(&self) -> &'static str;

    /// Runs the pass. Explain every change through `cx`.
    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed;
}

/// Why a pipeline stopped.
#[derive(Clone, PartialEq, Debug)]
pub enum PassError {
    /// The module was already invalid before the first pass: a bug in
    /// whatever produced it (lowering, the parser's caller).
    InvalidInput(Vec<VerifyError>),
    /// A pass left the module invalid: a bug in that pass.
    InvalidAfter {
        pass: &'static str,
        errors: Vec<VerifyError>,
    },
}

impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (head, errors) = match self {
            PassError::InvalidInput(e) => {
                ("the IR is invalid before the first pass".to_string(), e)
            }
            PassError::InvalidAfter { pass, errors } => {
                (format!("pass `{pass}` produced invalid IR"), errors)
            }
        };
        write!(f, "{head}")?;
        for e in errors {
            write!(f, "\n  {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for PassError {}

/// What a finished pipeline produced.
#[derive(Debug, Default)]
pub struct PipelineReport {
    /// Each pass that ran, in order, and whether it changed the module.
    pub ran: Vec<(&'static str, Changed)>,
    /// Every explanation, in the order the passes made them.
    pub explanations: Vec<Explanation>,
    /// Explanations dropped by [`MAX_EXPLANATIONS`].
    pub omitted_explanations: usize,
    /// The IR text after each pass chosen with [`PassManager::print_after`].
    pub printed: Vec<(&'static str, String)>,
}

/// Runs a pipeline of passes.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    verify: bool,
    panic_on_invalid: bool,
    print_after: Vec<String>,
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PassManager {
    /// An empty pipeline. Verifies between passes in debug builds only (a
    /// release `sky` trusts its passes; the tests already checked them), and
    /// returns a [`PassError`] on failure.
    pub fn new() -> Self {
        PassManager {
            passes: Vec::new(),
            verify: cfg!(debug_assertions),
            panic_on_invalid: false,
            print_after: Vec::new(),
        }
    }

    /// A pipeline for tests: always verifies, and panics (naming the pass)
    /// if the module is invalid.
    pub fn for_tests() -> Self {
        PassManager {
            verify: true,
            panic_on_invalid: true,
            ..Self::new()
        }
    }

    /// Appends a pass.
    pub fn with(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Turns verification between passes on or off.
    pub fn verify_each(mut self, on: bool) -> Self {
        self.verify = on;
        self
    }

    /// Makes a verification failure a panic instead of an error.
    pub fn panic_on_invalid(mut self, on: bool) -> Self {
        self.panic_on_invalid = on;
        self
    }

    /// Captures the printed IR after the pass called `name` (`"*"` for every
    /// pass) in [`PipelineReport::printed`].
    pub fn print_after(mut self, name: impl Into<String>) -> Self {
        self.print_after.push(name.into());
        self
    }

    fn fail(&self, e: PassError) -> Result<PipelineReport, PassError> {
        if self.panic_on_invalid {
            panic!("{e}");
        }
        Err(e)
    }

    /// Runs every pass in order on `m`.
    pub fn run(&self, m: &mut Module) -> Result<PipelineReport, PassError> {
        if self.verify {
            if let Err(errors) = verify(m) {
                return self.fail(PassError::InvalidInput(errors));
            }
        }
        let mut cx = PassCx::new();
        let mut report = PipelineReport::default();
        for pass in &self.passes {
            let name = pass.name();
            cx.pass = name;
            let changed = pass.run(m, &mut cx);
            report.ran.push((name, changed));
            if self.verify {
                if let Err(errors) = verify(m) {
                    return self.fail(PassError::InvalidAfter { pass: name, errors });
                }
            }
            if self.print_after.iter().any(|p| p == "*" || p == name) {
                report.printed.push((name, print(m)));
            }
        }
        report.omitted_explanations = cx.omitted;
        report.explanations = cx.explanations;
        Ok(report)
    }
}
