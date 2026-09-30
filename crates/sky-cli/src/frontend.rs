//! The front end shared by `check`, `run` and `repl`: lex, parse, and
//! type-check one source text, and collect every diagnostic.

use std::collections::HashSet;

use sky_ast::{Module, ParseError};
use sky_diag::{Diagnostic, Severity, SourceFile, Span};
use sky_syntax::Edition;
use sky_types::{Checked, Env};

use crate::ir;

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
    /// The checker's whole result, when the module parsed: `sky run`'s SIR
    /// engine lowers with it (lowering reads the types the checker recorded
    /// and must not re-derive them). Its diagnostics are also in
    /// [`FrontEnd::diagnostics`].
    pub checked: Option<Checked>,
    /// The IR outcome, when asked for ([`analyze_ir`]) and the program has
    /// no errors. Its notes (pass explanations, or why there is no IR) are
    /// already in `diagnostics`.
    pub ir: Option<ir::Outcome>,
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
pub fn analyze(src: &str, env: Env, explain: bool, edition: Edition) -> FrontEnd {
    analyze_ir(src, env, explain, false, false, edition)
}

/// [`analyze`], and with `explain` or `emit_ir`, also lowers a program the
/// checker accepts to SIR and runs the default pipeline (see
/// [`ir::build`]). With `explain`, the pass explanations join the checker's
/// notes in source order; with `emit_ir`, [`FrontEnd::ir`] keeps the text of
/// every stage. A program that cannot be lowered is not an error here:
/// [`FrontEnd::ir`] says why.
///
/// With `budgets` (`sky check`), the `accuracy` statements are checked by the
/// roundoff analysis (see [`crate::accuracy`]), and with `explain` too the
/// accuracy of printed values.
pub fn analyze_ir(
    src: &str,
    env: Env,
    explain: bool,
    emit_ir: bool,
    budgets: bool,
    _edition: Edition,
) -> FrontEnd {
    let mut ir_out = None;
    let mut pass_notes = Vec::new();
    let mut module_out = None;
    let mut checked_out = None;
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
                if (explain || emit_ir) && !checked.has_errors() {
                    let outcome = ir::build(&module, src, &checked, emit_ir, explain);
                    match &outcome {
                        ir::Outcome::Ready(ready) => pass_notes = ready.notes.clone(),
                        // Only `--emit-ir` says so: `--explain` alone stays
                        // as it was for programs that cannot be lowered.
                        ir::Outcome::Unavailable(why) if emit_ir => pass_notes.push(why.clone()),
                        _ => {}
                    }
                    ir_out = Some(outcome);
                }
                diagnostics.extend(checked.diagnostics.iter().cloned());
                if budgets && !checked.has_errors() {
                    diagnostics.extend(crate::accuracy::diagnostics(
                        &module,
                        src,
                        &checked,
                        &checked.diagnostics,
                        explain,
                    ));
                }
                if explain {
                    diagnostics.extend(checked.ode_notes.iter().cloned());
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
                // After the checker's notes: at one position they come first.
                diagnostics.append(&mut pass_notes);
                env_out = checked.env.clone();
                solves = checked.cholesky_solves.clone();
                checked_out = Some(checked);
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
        checked: checked_out,
        ir: ir_out,
        ok,
    }
}
