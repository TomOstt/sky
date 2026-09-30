//! The record a pass leaves for each change it makes.

use std::fmt;

use sky_diag::Span;
use sky_ir::PassCx;

/// One applied rewrite. The fields mirror the design document's
/// `Explanation { pass, rule, span, before, after, because }`; the pass
/// adapters in [`crate::pipeline`] hand each one to
/// [`Rewrite::explain`], the single place that converts it.
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

impl Rewrite {
    /// Reports the rewrite through the pass manager's collector, as a
    /// `sky_ir::Explanation` with the same rule, span, texts and reasons.
    /// The explanation's pass name is the running pass's, which is this
    /// rewrite's `pass` when it runs through [`crate::pipeline`].
    pub fn explain(&self, cx: &mut PassCx) {
        debug_assert_eq!(self.pass, cx.pass_name(), "explained by a different pass");
        cx.explain(
            self.rule,
            self.span,
            self.before.clone(),
            self.after.clone(),
            self.because.clone(),
        );
    }
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

/// A [`Rewrite`] that is also about a second place in the source: the
/// earlier computation a later one reuses (`gvn`), the loop an operation
/// moved out of (`licm`). It reaches `sky_ir::Explanation::related`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Related {
    pub rewrite: Rewrite,
    pub related: Span,
}

impl Related {
    /// Reports the rewrite like [`Rewrite::explain`], with the second span.
    pub fn explain(&self, cx: &mut PassCx) {
        debug_assert_eq!(
            self.rewrite.pass,
            cx.pass_name(),
            "explained by a different pass"
        );
        cx.explain_related(
            self.rewrite.rule,
            self.rewrite.span,
            Some(self.related),
            self.rewrite.before.clone(),
            self.rewrite.after.clone(),
            self.rewrite.because.clone(),
        );
    }
}

/// The headline `sky check --explain` gives an explanation of `gvn` or
/// `licm`, which quote the user's source and say *where* the other place is
/// (`x * y` at line 7 reuses the value computed at line 4), or `None` for
/// any other explanation. `src` is the source text the spans index.
///
/// * `[GVN-001] `x * y` at line 7 reuses the value computed at line 4`
/// * `[LICM-001] `inv(K)` is computed once before the loop (line 12)`
/// * `[LICM-002] `inv(K)` is computed once before the loop, if the loop
///   runs at all (line 12)`
pub fn motion_headline(x: &sky_ir::Explanation, src: &str) -> Option<String> {
    let rule = x.rule;
    if ![
        crate::gvn::GVN_RULE,
        crate::licm::LICM_RULE,
        crate::licm::LICM_GUARDED_RULE,
    ]
    .contains(&rule)
    {
        return None;
    }
    let related = x.related?;
    let file = sky_diag::SourceFile::new("", src);
    let line = |s: Span| file.line_col(s.start).0;
    let text = src.get(x.span.start as usize..x.span.end as usize)?;
    let text = sky_diag::display_capped(text);
    let (here, there) = (line(x.span), line(related));
    Some(if rule == crate::gvn::GVN_RULE {
        if here == there {
            format!("[{rule}] `{text}` reuses the value computed earlier on line {there}")
        } else {
            format!("[{rule}] `{text}` at line {here} reuses the value computed at line {there}")
        }
    } else if rule == crate::licm::LICM_RULE {
        format!("[{rule}] `{text}` is computed once before the loop (line {there})")
    } else {
        format!(
            "[{rule}] `{text}` is computed once before the loop, if the loop runs at all \
             (line {there})"
        )
    })
}
