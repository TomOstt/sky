//! Runtime errors, convertible to ordinary diagnostics.

use std::fmt;
use std::ops::Deref;

use sky_diag::{Diagnostic, Span};

/// A failure while running a program, located at the expression that
/// failed. Rendered like any other diagnostic (see [`RuntimeError::to_diagnostic`]).
///
/// Boxed so `Result<_, RuntimeError>` stays small on the hot path; the
/// fields are reached through `Deref` (`err.message`, `err.span`).
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeError(Box<ErrorDetails>);

/// What a [`RuntimeError`] carries.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorDetails {
    /// Lowercase, no trailing period.
    pub message: String,
    /// The failing expression or statement.
    pub span: Span,
    /// Printed after the primary underline.
    pub label: Option<String>,
    /// Related locations, e.g. the zero divisor.
    pub secondary: Vec<(Span, String)>,
    pub notes: Vec<String>,
    pub help: Vec<String>,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        RuntimeError(Box::new(ErrorDetails {
            message: message.into(),
            span,
            label: None,
            secondary: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }))
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.0.label = Some(label.into());
        self
    }

    pub fn with_secondary(mut self, span: Span, label: impl Into<String>) -> Self {
        self.0.secondary.push((span, label.into()));
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.0.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.0.help.push(help.into());
        self
    }

    /// The error as an error-severity diagnostic, for the renderer.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let mut d = Diagnostic::error(self.message.clone(), self.span);
        if let Some(l) = &self.label {
            d = d.with_label(l.clone());
        }
        for (span, label) in &self.secondary {
            d = d.with_secondary(*span, label.clone());
        }
        for n in &self.notes {
            d = d.with_note(n.clone());
        }
        for h in &self.help {
            d = d.with_help(h.clone());
        }
        d
    }
}

impl Deref for RuntimeError {
    type Target = ErrorDetails;

    fn deref(&self) -> &ErrorDetails {
        &self.0
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}
