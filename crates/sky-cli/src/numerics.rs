//! Numerics modes in the command line: how `--numerics=` and `sky.toml`
//! choose one, and how a relaxed run reports itself (spec "Relaxed
//! numerics").
//!
//! The default is reproducible numerics: bit-identical results everywhere.
//! **Relaxed** numerics are opt-in, in two explicit places:
//!
//! * `sky run --numerics=relaxed` (and `sky check --numerics=relaxed` for
//!   `--explain` and `--emit=clif`), for one run;
//! * `numerics = "relaxed"` in the project's `sky.toml`, for everyone who
//!   runs the project.
//!
//! The flag wins over the manifest (either way: `--numerics=reproducible`
//! turns a relaxed project back to reproducible, for a check against the
//! reference). Nothing else selects a mode: no environment variable, no
//! per-file pragma (a mode is a property of a run, and a file that changed
//! it would change what the files that import it compute).

use std::path::PathBuf;

use sky_diag::{display_safe, Diagnostic, Span};
use sky_ir::{Module, OpData, OpKind};
use sky_project::{Discovery, Numerics, Project};

/// Where the mode of a run came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Nothing asked: the default.
    Default,
    /// `--numerics=`.
    Flag,
    /// The `numerics` key of this `sky.toml`.
    Manifest(PathBuf),
}

/// The numerics of a run, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mode {
    pub numerics: Numerics,
    pub origin: Origin,
}

impl Mode {
    /// The mode `--numerics=` (if given) and the project's `sky.toml` give:
    /// the flag first, then the manifest, then reproducible.
    pub fn resolve(flag: Option<Numerics>, discovery: &Discovery) -> Mode {
        if let Some(numerics) = flag {
            return Mode {
                numerics,
                origin: Origin::Flag,
            };
        }
        match &discovery.project {
            Project::Rooted { manifest, .. } if discovery.numerics == Numerics::Relaxed => Mode {
                numerics: discovery.numerics,
                origin: Origin::Manifest(manifest.clone()),
            },
            _ => Mode {
                numerics: Numerics::Reproducible,
                origin: Origin::Default,
            },
        }
    }

    /// Whether the run is relaxed.
    pub fn relaxed(&self) -> bool {
        self.numerics == Numerics::Relaxed
    }

    /// The kernels' name for the mode: what `sky_interp::linalg` and
    /// `sky_jit` are told.
    pub fn kernels(&self) -> sky_interp::linalg::Numerics {
        match self.numerics {
            Numerics::Reproducible => sky_interp::linalg::Numerics::Reproducible,
            Numerics::Relaxed => sky_interp::linalg::Numerics::Relaxed,
        }
    }

    /// What set the mode, for a message: `--numerics=relaxed`, or `sky.toml`
    /// (with its path for `check --explain`).
    fn cause(&self, with_path: bool) -> String {
        match &self.origin {
            Origin::Manifest(path) if with_path => {
                format!("declared in {}", display_safe(&path.display().to_string()))
            }
            Origin::Manifest(_) => "sky.toml".to_string(),
            Origin::Flag | Origin::Default => format!("--numerics={}", self.numerics.name()),
        }
    }

    /// The line `sky check --explain` prints before anything else when the
    /// mode is relaxed, in the manner of its `edition:` line.
    pub fn explain_line(&self) -> String {
        format!(
            "numerics: relaxed ({}): fused multiply-add, reassociated sums and the platform's BLAS/LAPACK; results are not bit-reproducible across machines",
            self.cause(true)
        )
    }

    /// The one line a relaxed `sky run` that printed results ends with, on
    /// stderr (stdout stays the program's).
    pub fn run_note(&self) -> String {
        format!(
            "note: results computed in relaxed numerics ({}): not bit-reproducible across machines",
            self.cause(false)
        )
    }
}

/// Parses the text after `--numerics=`.
pub fn parse(text: &str) -> Result<Numerics, String> {
    Numerics::parse(text).ok_or_else(|| {
        format!(
            "unknown numerics `{}` (use `reproducible` or `relaxed`)",
            display_safe(text)
        )
    })
}

/// The stable id of the note that an array operation runs on relaxed kernels.
pub const ARRAY_RULE: &str = "NUM-001";
/// The stable id of the note that native code fuses a multiply-add.
pub const FUSE_RULE: &str = "NUM-002";

/// `sky check --explain`'s notes for a relaxed run: every array operation
/// that runs on the relaxed kernels (`[NUM-001]`), and every `a*b + c` that
/// native code fuses (`[NUM-002]`), each saying what changes and what does
/// not.
pub fn notes(module: &Module, src: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for func in &module.functions {
        for op in reachable_ops(func) {
            let what = match op.kind {
                OpKind::MatMul | OpKind::MatMulEx { .. } => "a matrix product",
                OpKind::Dot => "a dot product",
                OpKind::Norm => "a norm",
                OpKind::Solve { .. } | OpKind::SolveRight { .. } => "a linear solve",
                _ => continue,
            };
            let text = one_line(src, op.span).unwrap_or_else(|| op.kind.mnemonic().to_string());
            let mut d = Diagnostic::note(
                format!("[{ARRAY_RULE}] `{text}` is {what} on relaxed numerics"),
                op.span,
            )
            .with_note(
                "it may use fused multiply-add and the platform's BLAS/LAPACK (Accelerate on macOS), and add its terms in an order that depends on the machine and on --threads"
                    .to_string(),
            )
            .with_note(
                "it agrees with reproducible numerics to rounding error (about 1e-12 relative for a well-conditioned problem), not bit for bit"
                    .to_string(),
            );
            if matches!(op.kind, OpKind::Solve { .. } | OpKind::SolveRight { .. }) {
                d = d.with_note(
                    "a singular or not-positive-definite matrix is still an error, with the same message at the same place"
                        .to_string(),
                );
            }
            out.push(d);
        }
    }
    for (product, sum) in sky_jit::fused_sums(module) {
        let text = one_line(src, sum).unwrap_or_else(|| "a * b + c".to_string());
        out.push(
            Diagnostic::note(
                format!("[{FUSE_RULE}] `{text}` is one fused multiply-add on the native engine"),
                sum,
            )
            .with_note(
                "the product is not rounded before the sum (one rounding instead of two), so the last bit can differ from `--engine=ast` and `--engine=sir`, which evaluate each operation separately"
                    .to_string(),
            )
            .with_note(format!(
                "an overflow is still an error at the operation that overflows; only a product that overflows while the fused sum does not (`{}`) is accepted",
                one_line(src, product).unwrap_or_else(|| "the product".to_string())
            )),
        );
    }
    out
}

/// Every operation of `func` that a block of its body (or of a region
/// nested in one) holds, in program order.
fn reachable_ops(func: &sky_ir::Function) -> Vec<&OpData> {
    fn region<'f>(func: &'f sky_ir::Function, r: sky_ir::RegionId, out: &mut Vec<&'f OpData>) {
        for &b in &func.region(r).blocks {
            for &id in &func.block(b).ops {
                let op = func.op(id);
                out.push(op);
                for &inner in &op.regions {
                    region(func, inner, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    region(func, func.body, &mut out);
    out
}

/// The source of `span` on one line, or `None` if it has none or spans lines.
fn one_line(src: &str, span: Span) -> Option<String> {
    let text = sky_ast::ast::span_text(src, span)?;
    (!text.is_empty() && !text.contains('\n')).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_two_modes_only() {
        assert_eq!(parse("relaxed"), Ok(Numerics::Relaxed));
        assert_eq!(parse("reproducible"), Ok(Numerics::Reproducible));
        assert!(parse("fast").is_err());
        assert!(parse("").is_err());
        assert!(parse("Relaxed").is_err());
    }

    #[test]
    fn the_run_note_names_its_cause() {
        let flag = Mode {
            numerics: Numerics::Relaxed,
            origin: Origin::Flag,
        };
        assert_eq!(
            flag.run_note(),
            "note: results computed in relaxed numerics (--numerics=relaxed): not bit-reproducible across machines"
        );
        let manifest = Mode {
            numerics: Numerics::Relaxed,
            origin: Origin::Manifest(PathBuf::from("/p/sky.toml")),
        };
        assert!(manifest.run_note().contains("(sky.toml)"));
        assert!(manifest.explain_line().contains("declared in /p/sky.toml"));
    }
}
