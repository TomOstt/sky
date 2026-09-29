//! The diagnostic data model and its builder methods.

use std::fmt;

use crate::Span;

/// How serious a diagnostic is. Printed lowercase in the header.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    /// The word printed in the header: `error`, `warning`, or `note`.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A span with an optional message printed after its underline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub span: Span,
    pub message: Option<String>,
}

/// One diagnostic about one source file.
///
/// Every span refers to the same [`crate::SourceFile`], the one passed to
/// [`Diagnostic::render`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    /// Where the header's `line:col` points; underlined with `^`.
    pub primary: Label,
    /// Related locations; underlined with `-`, rendered in source order.
    pub secondary: Vec<Label>,
    /// `= note:` lines, in insertion order.
    pub notes: Vec<String>,
    /// `= help:` lines, in insertion order, after the notes.
    pub help: Vec<String>,
}

impl Diagnostic {
    /// A diagnostic with the given severity, message, and unlabeled primary span.
    pub fn new(severity: Severity, message: impl Into<String>, primary: Span) -> Self {
        Diagnostic {
            severity,
            message: message.into(),
            primary: Label {
                span: primary,
                message: None,
            },
            secondary: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    /// Shorthand for `Diagnostic::new(Severity::Error, ..)`.
    pub fn error(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Error, message, primary)
    }

    /// Shorthand for `Diagnostic::new(Severity::Warning, ..)`.
    pub fn warning(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Warning, message, primary)
    }

    /// Shorthand for `Diagnostic::new(Severity::Note, ..)`.
    pub fn note(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Note, message, primary)
    }

    /// Sets the message printed after the primary span's underline.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.primary.message = Some(label.into());
        self
    }

    /// Adds a labeled secondary span.
    pub fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.secondary.push(Label {
            span,
            message: Some(label.into()),
        });
        self
    }

    /// Adds a `= note:` line.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Adds a `= help:` line.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_collects_everything() {
        let d = Diagnostic::error("bad", Span::new(0, 1))
            .with_label("here")
            .with_secondary(Span::new(4, 5), "there")
            .with_note("n1")
            .with_help("h1")
            .with_note("n2");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.message, "bad");
        assert_eq!(d.primary.message.as_deref(), Some("here"));
        assert_eq!(d.secondary.len(), 1);
        assert_eq!(d.secondary[0].span, Span::new(4, 5));
        assert_eq!(d.notes, ["n1", "n2"]);
        assert_eq!(d.help, ["h1"]);
    }

    #[test]
    fn severity_words() {
        assert_eq!(Severity::Error.to_string(), "error");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Note.to_string(), "note");
        assert_eq!(
            Diagnostic::warning("w", Span::point(0)).severity,
            Severity::Warning
        );
        assert_eq!(
            Diagnostic::note("n", Span::point(0)).severity,
            Severity::Note
        );
    }
}
