//! Glue between shape errors and the checker's diagnostics: which line and
//! operation to record as the origin of a size, and how a [`ShapeError`]
//! becomes a diagnostic with its label, notes and help.
//!
//! The unifier knows nothing about source text, so the checker brackets each
//! constraint it emits with [`Checker::with_origin`]. Keeping this in its own
//! module keeps the hooks in `check.rs` and `flow.rs` to a line or two each.

use std::sync::Arc;

use sky_diag::{Diagnostic, Span};

use crate::check::Checker;
use crate::shape::{Origin, ShapeError, ShapeUnifier};

impl Checker<'_> {
    /// The 1-based line `span` starts on. Spans are byte offsets into `src`;
    /// an out-of-range start counts as the last line rather than panicking.
    fn line_of(&self, span: Span) -> u32 {
        let start = (span.start as usize).min(self.src.len());
        let newlines = self.src.as_bytes()[..start]
            .iter()
            .filter(|b| **b == b'\n')
            .count();
        u32::try_from(newlines).map_or(u32::MAX, |n| n.saturating_add(1))
    }

    /// Installs the [`Origin`] for the constraints that follow; the caller
    /// clears it with `self.env.shapes.set_origin(None)`. Prefer
    /// [`Self::with_origin`] unless the unifier must be cloned in between.
    pub(crate) fn set_origin(&mut self, span: Span, what: String) {
        let origin = Origin {
            line: Some(self.line_of(span)),
            what: Arc::from(what),
        };
        self.env.shapes.set_origin(Some(origin));
    }

    /// Runs `f` on the unifier with an [`Origin`] installed, then clears it,
    /// so every size `f` binds is recorded as coming from `what` at `span`'s
    /// line. `what` is user text and must already be quoted.
    pub(crate) fn with_origin<T>(
        &mut self,
        span: Span,
        what: String,
        f: impl FnOnce(&mut ShapeUnifier) -> T,
    ) -> T {
        self.set_origin(span, what);
        let out = f(&mut self.env.shapes);
        self.env.shapes.set_origin(None);
        out
    }

    /// An error diagnostic at `span` with the shape error's headline, its
    /// primary label (if it has one), its notes and its help. Callers add
    /// their own secondary labels, and may replace the primary label.
    pub(crate) fn shape_diagnostic(&self, e: &ShapeError, span: Span) -> Diagnostic {
        let mut d = Diagnostic::error(e.to_string(), span);
        if let Some(label) = e.label() {
            d = d.with_label(label);
        }
        for note in e.notes() {
            d = d.with_note(note);
        }
        if let Some(help) = e.help() {
            d = d.with_help(help);
        }
        d
    }
}
