//! Module names read late must keep their type (spec "Functions").
//!
//! A function reads a module variable as it is when the function runs, and
//! an equation reads one when a `solve` runs it; but the body was checked
//! once, with the type the name had at the `def` or the `equation`. If the
//! module later rebinds the name to another dimension or shape, or to a
//! value without a tag the body relied on (a rewrite of `Q⁻¹` to `Qᵀ` for
//! an Orthogonal `Q`), the body would run against a value it was never
//! checked against: `return kk` in a `-> Hz` function returning metres, or
//! a wrong answer. So the checker remembers each module name a body reads,
//! with the type it read, and a later module-level binding of that name
//! must have that type (the same dimension and shape, and at least its
//! tags). Calls are not affected: they bind early, to the function the
//! checker saw (spec "Calls bind where the caller is defined").

use std::collections::BTreeMap;

use sky_diag::{display_capped as quote, Diagnostic, Span};

use crate::check::{Checker, Ty};

/// One body's read of a module name.
#[derive(Clone, Debug)]
pub(crate) struct ModuleRead {
    /// The type the body was checked with.
    ty: Ty,
    /// "`rate`" or "equation `decay`".
    reader: String,
    /// The first place the body reads it.
    at: Span,
}

/// Every module name some body reads, with how.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModuleReads {
    reads: BTreeMap<String, Vec<ModuleRead>>,
}

impl Checker<'_> {
    /// Records that the function being checked reads `name`, of type `ty`,
    /// at `at`, if `name` is a module name (no function around binds it).
    pub(crate) fn note_module_read(&mut self, name: &str, ty: &Ty, at: Span) {
        let Some(ctx) = self.fns.first() else {
            return;
        };
        if self.fns.iter().any(|c| c.locals.contains(name)) {
            return;
        }
        let reader = format!("`{}`", quote(ctx.name()));
        self.record_read(name, ty, reader, at);
    }

    /// Records that equation `eq` (at module level) reads `name` at `at`.
    pub(crate) fn note_equation_read(&mut self, eq: &str, name: &str, at: Span) {
        if !self.fns.is_empty() {
            return; // inside a function: its locals, not module names
        }
        if let Some(ty) = self.env.get(name) {
            self.record_read(name, &ty, format!("equation `{}`", quote(eq)), at);
        }
    }

    fn record_read(&mut self, name: &str, ty: &Ty, reader: String, at: Span) {
        if !matches!(ty, Ty::Value(_) | Ty::Bool) {
            return;
        }
        let reads = self.reads.reads.entry(name.to_string()).or_default();
        if reads.iter().any(|r| r.reader == reader) {
            return;
        }
        reads.push(ModuleRead {
            ty: ty.clone(),
            reader,
            at,
        });
    }

    /// A module-level binding of `name` to `ty` at `span`: an error if a
    /// body that reads `name` was checked with a type `ty` does not have.
    pub(crate) fn check_module_rebind(&mut self, name: &str, ty: &Ty, span: Span) {
        if !self.fns.is_empty() {
            return;
        }
        let Some(reads) = self.reads.reads.get(name) else {
            return;
        };
        let Some(read) = reads.iter().find(|r| !self.keeps_type(&r.ty, ty)).cloned() else {
            return;
        };
        let n = quote(name);
        let (old, new) = (self.ty_noun(&read.ty), self.ty_noun(ty));
        let lost_tags = matches!((&read.ty, ty), (Ty::Value(o), Ty::Value(v))
            if o.dim == v.dim && self.env.shapes.apply(o.ty.shape()) == self.env.shapes.apply(v.ty.shape()));
        let message = if lost_tags {
            format!(
                "`{n}` cannot lose its tags: {} reads it as {old}, and this is {new}",
                read.reader
            )
        } else {
            format!(
                "`{n}` cannot become {new}: {} reads it as {old}",
                read.reader
            )
        };
        self.error(
            Diagnostic::error(message, span)
                .with_label("this rebinding")
                .with_secondary(read.at, format!("{} reads `{n}` here, when it runs", read.reader))
                .with_note("a function or an equation reads module names when it runs, so they must keep the type its body was checked with (a rewrite there may rely on a tag)")
                .with_help(format!(
                    "use a new name, or pass `{n}` to the function as a parameter"
                )),
        );
    }

    /// Whether a value of type `new` can stand where a body read `old`.
    fn keeps_type(&self, old: &Ty, new: &Ty) -> bool {
        match (old, new) {
            (Ty::Unknown, _) | (_, Ty::Unknown) | (Ty::Bool, Ty::Bool) => true,
            (Ty::Value(o), Ty::Value(n)) => {
                o.dim == n.dim
                    && self.env.shapes.apply(o.ty.shape()) == self.env.shapes.apply(n.ty.shape())
                    && n.ty.tags().is_subtag_of(o.ty.tags())
            }
            _ => false,
        }
    }
}
