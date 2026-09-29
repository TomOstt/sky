//! The front end shared by `check`, `run` and `repl`: lex, parse, and
//! type-check one source text, and collect every diagnostic.

use std::collections::HashSet;

use sky_ast::{Module, ParseError};
use sky_diag::{Diagnostic, Severity, SourceFile, Span};
use sky_syntax::Edition;
use sky_types::Env;

/// The outcome of the front end on one source text.
pub struct FrontEnd {
    /// Lexer warnings, parse errors and checker diagnostics, in source order.
    pub diagnostics: Vec<Diagnostic>,
    /// The parsed module, only if lexing and parsing succeeded. It is
    /// only safe to run when [`FrontEnd::ok`] is also true.
    pub module: Option<Module>,
    /// The checker's environment after the last statement. Meaningful only
    /// when [`FrontEnd::ok`]; the REPL keeps it for the next line.
    pub env: Env,
    /// The `A⁻¹ @ b` sites the checker rewrote to Cholesky solves, which the
    /// interpreter must honour (see `sky_interp::run_module_with`).
    pub cholesky_solves: HashSet<Span>,
    /// Whether no diagnostic is an error.
    pub ok: bool,
}

impl FrontEnd {
    /// The diagnostics rendered against `file`, in order.
    pub fn render(&self, file: &SourceFile) -> Vec<String> {
        self.diagnostics
            .iter()
            .map(|d| d.render_to_string(file))
            .collect()
    }
}

/// Lexes, parses, and (if parsing succeeded) type-checks `src` starting
/// from `env`. With `explain`, the checker's rewrite decisions follow as
/// notes. Never executes anything. Runs inside `with_deep_stack`.
///
/// `edition` is the language edition the source is written for. Nothing
/// consults it yet (there is one edition); it is threaded through now so the
/// first edition-dependent rule has one place to read it, and no phase can
/// forget to ask.
pub fn analyze(src: &str, env: Env, explain: bool, _edition: Edition) -> FrontEnd {
    let mut module_out = None;
    let mut env_out = env.clone();
    let mut solves = HashSet::new();
    let mut diagnostics: Vec<Diagnostic> = match sky_syntax::lex(src) {
        Err(e) => e.diagnostics().cloned().collect(),
        Ok(lexed) => {
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
            let mut diagnostics = lexed.warnings;
            if errors.is_empty() {
                // Only a module that parsed cleanly is checked, so the
                // checker never reports on a statement the parser dropped.
                let checked = sky_types::check_module_with(&module, src, env);
                diagnostics.extend(checked.diagnostics);
                if explain {
                    diagnostics.extend(checked.rewrites.iter().map(|r| r.to_diagnostic(src)));
                    if checked.rewrites_omitted > 0 {
                        let last = checked.rewrites.last().map_or(Span::point(0), |r| r.span);
                        diagnostics.push(Diagnostic::note(
                            format!(
                                "{} more rewrite{} not shown",
                                checked.rewrites_omitted,
                                if checked.rewrites_omitted == 1 {
                                    ""
                                } else {
                                    "s"
                                }
                            ),
                            last,
                        ));
                    }
                }
                env_out = checked.env;
                solves = checked.cholesky_solves;
                module_out = Some(module);
            } else {
                diagnostics.extend(errors.into_iter().map(ParseError::into_diagnostic));
            }
            diagnostics
        }
    };
    // Lexer warnings with parse or check errors, interleaved by position.
    // The sort is stable, so diagnostics at one position keep their order.
    diagnostics.sort_by_key(|d| d.primary.span.start);
    let ok = !diagnostics.iter().any(|d| d.severity == Severity::Error);
    FrontEnd {
        diagnostics,
        module: module_out,
        env: env_out,
        cholesky_solves: solves,
        ok,
    }
}
