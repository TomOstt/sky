//! Condition numbers of constant matrices: the checker warns when a matrix
//! the source spells out entry by entry is inverted or used to solve a
//! system and its inverse would lose most of the 16 digits of an `f64`.
//!
//! # What is computed
//!
//! κ(A) = σmax(A) / σmin(A), the 2-norm condition number, from the singular
//! values of a dense `n` by `n` matrix ([`condition_number`]). The singular
//! values come from a one-sided Jacobi iteration (Hestenes): plane
//! rotations of column pairs until the columns are orthogonal, and the
//! column norms are the singular values. It is short, has no dependencies,
//! and computes small singular values to high relative accuracy for the
//! matrices that matter here (Hilbert 10 is 1.6e13 to about three digits).
//!
//! Whether the matrix is singular at all is **not** decided here: that is
//! [`crate::linalg::inverse`], the code `sky run` calls, so the check-time
//! verdict and the run-time error can never disagree (a matrix `sky run`
//! rejects as singular is reported singular here, and one it inverts gets a
//! κ).
//!
//! # Bounds and purity
//!
//! Only matrices up to [`MAX_SIZE`] by [`MAX_SIZE`] are analyzed (an
//! elimination and at most [`MAX_SWEEPS`] Jacobi sweeps: a fixed amount of
//! arithmetic, well under a millisecond at 64 by 64). Nothing is executed
//! from user code: the entries are the constants the source spells out
//! ([`crate::entry_value`]), and the analysis is arithmetic on them, which
//! the security rules allow. At most [`MAX_WARNINGS`] numerical warnings
//! (this module's and the cancellation lint's) are emitted per check.
//!
//! # Which matrices are constant
//!
//! A literal operand (`inv([[1, 2], [3, 4]])`) always is. A name is when its
//! *only* binding in the module is `A = <literal>` (or `A: T = <literal>`):
//! any other binding of the same name anywhere (a second assignment, `A +=`,
//! a parameter or local of a function with that name) forgets it for good,
//! whatever path it is on. That over-approximates on purpose: a warning
//! must be about the matrix `sky run` will invert, and following values
//! through branches is what this module avoids. Inside a `while` body only
//! literals are analyzed (a later statement of the body can rebind the name
//! for the next iteration, and the body is checked once), and inside a
//! function only the function's own locals are (a global can be rebound
//! between the `def` and a call).

use std::collections::{HashMap, HashSet};

use sky_ast::{Expr, ExprKind};
use sky_diag::{Diagnostic, Span};

use crate::check::{strip_parens, Checker};
use crate::constant::entry_value;
use crate::linalg;

/// Largest matrix (rows and columns) whose condition number is estimated.
pub const MAX_SIZE: usize = 64;

/// A condition number above this is warned about: about 8 of the 16 digits
/// of an `f64` are gone.
pub const WARN_KAPPA: f64 = 1e8;

/// Most Jacobi sweeps. Convergence is quadratic; a 64 by 64 matrix needs
/// about 8, so the cap only bounds a pathological input.
pub const MAX_SWEEPS: usize = 40;

/// Most numerical warnings (this module's and `cancellation`'s) per check,
/// so a generated program with thousands of sites cannot flood the output.
pub const MAX_WARNINGS: usize = 100;

/// How well conditioned a constant matrix is for inversion.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Conditioning {
    /// [`crate::linalg::inverse`] rejects the matrix: what `sky run` does.
    Singular,
    /// The matrix inverts. `kappa` is κ₂(A) (infinite only if the
    /// singular-value iteration underflows to a zero σmin); `scaled` is κ₂
    /// after scaling rows and then columns to unit maximum entry, which
    /// tells scaling trouble (`diag(1, 1e10)`) from real near-dependence.
    Invertible { kappa: f64, scaled: f64 },
}

/// The largest and smallest singular values of the `n` by `n` row-major
/// matrix `a` (`n >= 1`), by one-sided Jacobi. Entries must be finite. The
/// matrix is first divided by its largest entry so the sums of squares
/// cannot overflow; the values returned are for the original scale.
fn singular_range(a: &[f64], n: usize) -> (f64, f64) {
    debug_assert_eq!(a.len(), n * n);
    let scale = a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return (0.0, 0.0);
    }
    let mut u: Vec<f64> = a.iter().map(|x| x / scale).collect();
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..n {
                    let (x, y) = (u[i * n + p], u[i * n + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..n {
                    let (x, y) = (u[i * n + p], u[i * n + q]);
                    u[i * n + p] = c * x - s * y;
                    u[i * n + q] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let (mut hi, mut lo) = (0.0_f64, f64::INFINITY);
    for j in 0..n {
        let norm = (0..n)
            .map(|i| u[i * n + j] * u[i * n + j])
            .sum::<f64>()
            .sqrt();
        hi = hi.max(norm);
        lo = lo.min(norm);
    }
    (hi * scale, lo * scale)
}

/// κ₂ of the `n` by `n` matrix `a`: σmax / σmin, infinite if σmin is zero.
fn kappa(a: &[f64], n: usize) -> f64 {
    let (hi, lo) = singular_range(a, n);
    if lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

/// Scales every row of `a`, then every column, to unit maximum entry (a
/// zero row or column is left alone).
fn equilibrated(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    for i in 0..n {
        let max = (0..n).fold(0.0_f64, |s, j| s.max(m[i * n + j].abs()));
        if max > 0.0 {
            (0..n).for_each(|j| m[i * n + j] /= max);
        }
    }
    for j in 0..n {
        let max = (0..n).fold(0.0_f64, |s, i| s.max(m[i * n + j].abs()));
        if max > 0.0 {
            (0..n).for_each(|i| m[i * n + j] /= max);
        }
    }
    m
}

/// The conditioning of the `n` by `n` row-major matrix `a` for inversion.
/// `None` if `n` is 0 or above [`MAX_SIZE`], or an entry is not finite.
///
/// Singularity is [`crate::linalg::inverse`]'s verdict, the one `sky run`
/// reaches; κ is [`Conditioning::Invertible`]'s.
pub fn condition_number(a: &[f64], n: usize) -> Option<Conditioning> {
    if n == 0 || n > MAX_SIZE || a.len() != n * n || a.iter().any(|x| !x.is_finite()) {
        return None;
    }
    if linalg::inverse(a, n).is_err() {
        return Some(Conditioning::Singular);
    }
    Some(Conditioning::Invertible {
        kappa: kappa(a, n),
        scaled: kappa(&equilibrated(a, n), n),
    })
}

/// How many of an `f64`'s 16 significant digits a condition number `kappa`
/// can cost: `floor(log10 κ)`, at most 16.
pub fn digits_lost(kappa: f64) -> u32 {
    if kappa.is_nan() || kappa < 1.0 {
        return 0;
    }
    if !kappa.is_finite() {
        return 16;
    }
    (kappa.log10().floor() as u32).min(16)
}

/// A square matrix literal's entries, row-major.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct ConstMatrix {
    n: usize,
    data: Vec<f64>,
}

/// The entries of `e` if it is a square matrix literal of at most
/// [`MAX_SIZE`] rows whose entries are all numeric constants.
pub(crate) fn const_matrix(src: &str, e: &Expr) -> Option<ConstMatrix> {
    let ExprKind::Array(rows) = &strip_parens(e).kind else {
        return None;
    };
    let n = rows.len();
    if n == 0 || n > MAX_SIZE {
        return None;
    }
    let mut data = Vec::with_capacity(n * n);
    for row in rows {
        let ExprKind::Array(items) = &row.kind else {
            return None;
        };
        if items.len() != n {
            return None;
        }
        for item in items {
            data.push(entry_value(src, item)?);
        }
    }
    Some(ConstMatrix { n, data })
}

/// What the checker remembers for the numerical warnings: which names hold
/// constant matrices, and what has been warned about already.
#[derive(Default)]
pub(crate) struct NumState {
    /// Names bound so far, ever, on any path.
    bound: HashSet<String>,
    /// Names bound more than once: never constant.
    rebound: HashSet<String>,
    consts: HashMap<String, ConstMatrix>,
    /// Sites already warned about (a loop body's code is one site).
    pub(crate) warned: HashSet<Span>,
    /// Numerical warnings emitted, against [`MAX_WARNINGS`].
    pub(crate) emitted: usize,
}

impl NumState {
    /// The name `name` is being bound (or updated): the first binding is the
    /// only one that may make it constant.
    pub(crate) fn on_bind(&mut self, name: &str) {
        if !self.bound.insert(name.to_string()) {
            self.rebound.insert(name.to_string());
            self.consts.remove(name);
        }
    }

    fn set_const(&mut self, name: &str, m: ConstMatrix) {
        if !self.rebound.contains(name) {
            self.consts.insert(name.to_string(), m);
        }
    }
}

/// The κ of the matrix in a warning: `3.2e14`.
fn fmt_kappa(k: f64) -> String {
    if k.is_finite() {
        format!("{k:.1e}")
    } else {
        "∞".to_string()
    }
}

impl Checker<'_> {
    /// After `name = value` (or `name: T = value`) bound `name`: remembers
    /// the entries if `value` is a constant matrix literal.
    pub(crate) fn note_const_binding(&mut self, name: &str, value: &Expr) {
        if let Some(m) = const_matrix(self.src, value) {
            self.num.set_const(name, m);
        }
    }

    /// The constant matrix `operand` denotes at this point, if any: a
    /// literal, or a name with exactly one binding in the module (see the
    /// module docs for where names are not trusted).
    fn constant_operand(&self, operand: &Expr) -> Option<ConstMatrix> {
        let e = strip_parens(operand);
        if let ExprKind::Array(_) = e.kind {
            return const_matrix(self.src, e);
        }
        let ExprKind::Name(id) = &e.kind else {
            return None;
        };
        let name = self.ident(id);
        if !self.loops.is_empty() {
            return None;
        }
        if let Some(ctx) = self.fns.last() {
            if !ctx.locals.contains(name) {
                return None;
            }
        }
        self.num.consts.get(name).cloned()
    }

    /// Whether another numerical warning may be emitted for the site `key`,
    /// and books it if so.
    pub(crate) fn numeric_warning_budget(&mut self, key: Span) -> bool {
        if self.num.emitted >= MAX_WARNINGS || !self.num.warned.insert(key) {
            return false;
        }
        self.num.emitted += 1;
        true
    }

    /// The `A⁻¹` (or `inv(A)`) at `op_span` with operand `operand` type
    /// checked: if `A` is a constant matrix, warns when it is singular or
    /// ill-conditioned. One warning per site.
    pub(crate) fn warn_conditioning(&mut self, operand: &Expr, op_span: Span) {
        let Some(m) = self.constant_operand(operand) else {
            return;
        };
        let Some(verdict) = condition_number(&m.data, m.n) else {
            return;
        };
        let (bad, kappa_scaled) = match verdict {
            Conditioning::Singular => (true, None),
            Conditioning::Invertible { kappa, scaled } => {
                (kappa > WARN_KAPPA, Some((kappa, scaled)))
            }
        };
        if !bad || !self.numeric_warning_budget(op_span) {
            return;
        }
        let what = format!("a constant {n}×{n} matrix", n = m.n);
        let name = match &strip_parens(operand).kind {
            ExprKind::Name(id) => self.quoted(id.span).to_string(),
            _ => "A".to_string(),
        };
        let d = match kappa_scaled {
            None => Diagnostic::warning("this matrix is singular, so it has no inverse", op_span)
                .with_label("`sky run` stops here with \"the matrix is singular\"")
                .with_secondary(operand.span, format!("this is {what}"))
                .with_note(
                    "elimination finds a pivot that is zero, or no larger than n·ε·max|aᵢⱼ| \
                     (too small to tell from rounding error): the test `sky run` applies to the \
                     same numbers",
                )
                .with_help(
                    "look for a row that is a multiple or a sum of others; if this is a linear \
                     system, a least-squares or regularized formulation handles a rank-deficient \
                     matrix",
                ),
            Some((kappa, scaled)) => {
                let lost = digits_lost(kappa);
                let k = fmt_kappa(kappa);
                let digits = if lost >= 16 {
                    "all 16 significant digits may be lost".to_string()
                } else {
                    format!("about {lost} of 16 significant digits may be lost")
                };
                let scaling_only = scaled * 100.0 < kappa;
                let help = if scaling_only {
                    format!(
                        "most of this is scaling: rows and columns of very different size. \
                         Rescale them (choose units that make the unknowns and equations similar \
                         in size), which brings κ down to about {}",
                        fmt_kappa(scaled)
                    )
                } else {
                    "rescale rows and columns so their entries are similar in size; or \
                     reformulate so the matrix is better posed (a different basis, fewer nearly \
                     dependent unknowns, or a regularized or least-squares formulation); \
                     higher-precision arithmetic (planned) buys back digits but not a better \
                     problem"
                        .to_string()
                };
                Diagnostic::warning(
                    format!("estimated condition number κ({name}) ≈ {k}: {digits}"),
                    op_span,
                )
                .with_label(format!(
                    "inverting or solving with this magnifies errors by up to {k}"
                ))
                .with_secondary(operand.span, format!("this is {what}"))
                .with_note(format!(
                    "κ({name}) = σmax/σmin measures how much a matrix magnifies relative error: \
                     rounding error in the inputs (about 1.1e-16 for an `f64`) can change the \
                     solution of {name}·x = b by about κ({name}) times as much, so about \
                     log10 κ = {lost} of the 16 digits are lost. A well-conditioned matrix has \
                     κ near 1"
                ))
                .with_help(help)
            }
        };
        self.error(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hilbert(n: usize) -> Vec<f64> {
        (0..n * n)
            .map(|k| 1.0 / ((k / n + k % n) as f64 + 1.0))
            .collect()
    }

    fn kappa_of(a: &[f64], n: usize) -> f64 {
        match condition_number(a, n) {
            Some(Conditioning::Invertible { kappa, .. }) => kappa,
            other => panic!("expected an invertible matrix, got {other:?}"),
        }
    }

    fn near(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel * want,
            "κ = {got:e}, expected {want:e} within {rel}"
        );
    }

    #[test]
    fn hilbert_matrices_match_the_published_condition_numbers() {
        // κ₂ of the Hilbert matrices (Wolfram MathWorld / Higham, ASNA).
        near(kappa_of(&hilbert(3), 3), 524.0567775860627, 1e-6);
        near(kappa_of(&hilbert(5), 5), 476_607.250_242_5, 1e-6);
        near(kappa_of(&hilbert(8), 8), 1.525_756_4e10, 1e-4);
        near(kappa_of(&hilbert(10), 10), 1.602_5e13, 2e-2);
    }

    #[test]
    fn easy_matrices() {
        let eye = [1.0, 0.0, 0.0, 1.0];
        near(kappa_of(&eye, 2), 1.0, 1e-12);
        near(kappa_of(&[1.0, 0.0, 0.0, 1e6], 2), 1e6, 1e-12);
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        near(kappa_of(&[2.0, 1.0, 1.0, 2.0], 2), 3.0, 1e-12);
        // A rotation is perfectly conditioned.
        let (c, s) = (0.6, 0.8);
        near(kappa_of(&[c, -s, s, c], 2), 1.0, 1e-12);
        near(kappa_of(&[-7.0], 1), 1.0, 1e-15);
    }

    #[test]
    fn an_almost_singular_matrix_is_huge() {
        // det = 1e-10: κ ≈ 4/1e-10 · ... about 4e10.
        let a = [1.0, 1.0, 1.0, 1.0 + 1e-10];
        let k = kappa_of(&a, 2);
        assert!(k > 3.9e10 && k < 4.1e10, "{k:e}");
    }

    #[test]
    fn scaling_is_told_from_conditioning() {
        // diag(1, 1e10): κ = 1e10, but only because of scale.
        match condition_number(&[1.0, 0.0, 0.0, 1e10], 2) {
            Some(Conditioning::Invertible { kappa, scaled }) => {
                near(kappa, 1e10, 1e-12);
                near(scaled, 1.0, 1e-12);
            }
            other => panic!("{other:?}"),
        }
        // Hilbert 8 stays ill-conditioned after scaling.
        match condition_number(&hilbert(8), 8) {
            Some(Conditioning::Invertible { kappa, scaled }) => assert!(scaled * 100.0 > kappa),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn singular_agrees_with_the_interpreters_inverse() {
        // Exactly singular, numerically singular, zero: whatever
        // `linalg::inverse` (what `sky run` calls) rejects is singular here,
        // and whatever it inverts has a κ.
        let cases: [(&[f64], usize); 6] = [
            (&[1.0, 2.0, 2.0, 4.0], 2),
            (&[0.0; 4], 2),
            (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
            (&[1.0, 2.0, 3.0, 4.0], 2),
            (&[0.0, 1.0, 1.0, 0.0], 2),
            (&[5.0], 1),
        ];
        for (a, n) in cases {
            let rejected = linalg::inverse(a, n).is_err();
            assert_eq!(
                condition_number(a, n) == Some(Conditioning::Singular),
                rejected,
                "{a:?}"
            );
        }
        // Hilbert 20: κ ≈ 1e28, far past what an inverse can resolve.
        let h = hilbert(20);
        assert_eq!(linalg::inverse(&h, 20).is_err(), {
            condition_number(&h, 20) == Some(Conditioning::Singular)
        });
    }

    #[test]
    fn size_is_bounded() {
        assert_eq!(condition_number(&[], 0), None);
        let big = vec![1.0; 65 * 65];
        assert_eq!(condition_number(&big, 65), None);
        // 64 by 64 is analyzed: a diagonally dominant matrix is fine.
        let mut a = vec![0.0; 64 * 64];
        for i in 0..64 {
            a[i * 64 + i] = 1.0 + i as f64;
        }
        near(kappa_of(&a, 64), 64.0, 1e-12);
        assert_eq!(condition_number(&[f64::NAN], 1), None);
    }

    #[test]
    fn digits_lost_counts_orders_of_magnitude() {
        assert_eq!(digits_lost(1.0), 0);
        assert_eq!(digits_lost(3.2e14), 14);
        assert_eq!(digits_lost(1e8), 8);
        assert_eq!(digits_lost(1e30), 16);
        assert_eq!(digits_lost(f64::INFINITY), 16);
    }

    #[test]
    fn kappa_formats_like_the_spec() {
        assert_eq!(fmt_kappa(3.2e14), "3.2e14");
        assert_eq!(fmt_kappa(4.766e5), "4.8e5");
    }
}
