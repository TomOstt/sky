//! The record a pass leaves for each change it makes.

use std::fmt;

use sky_diag::Span;

/// One applied rewrite. The fields mirror the design document's
/// `Explanation { pass, rule, span, before, after, because }`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rewrite {
    /// The pass that applied it (`"fold"`, `"dce"`, `"cholesky"`).
    pub pass: &'static str,
    /// A stable rule id, such as `FOLD-EXACT-001`.
    pub rule: &'static str,
    /// The user's source the change is about.
    pub span: Span,
    /// The code before, as text.
    pub before: String,
    /// The code after, as text.
    pub after: String,
    /// The facts and type properties the rewrite relied on.
    pub because: Vec<String>,
}

impl fmt::Display for Rewrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: `{}` -> `{}`",
            self.rule, self.pass, self.before, self.after
        )?;
        if !self.because.is_empty() {
            write!(f, " because {}", self.because.join("; "))?;
        }
        Ok(())
    }
}
