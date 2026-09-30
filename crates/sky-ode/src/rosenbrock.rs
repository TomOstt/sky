//! Rodas3: an L-stable, stiffly accurate Rosenbrock method (order 3 with an
//! embedded order 2 error estimator) with a finite-difference Jacobian.
//!
//! # Why a Rosenbrock method
//!
//! A stiff problem forces an explicit method to a step set by stability, not
//! accuracy (see `dopri.rs`). Implicit methods are not limited that way, but
//! a Newton iteration for every stage is delicate (convergence tests,
//! Jacobian reuse policies, failure handling). A *Rosenbrock* method is
//! linearly implicit: each stage solves one linear system with the matrix
//! `W = I/(gamma h) - J`, no iteration, so the cost and the result are
//! deterministic and the failure modes are few. It is the natural stiff
//! method for a small dense system where `J` is cheap enough to recompute
//! every step.
//!
//! # The method
//!
//! With the stages `k_i` defined by
//!
//! ```text
//! (I/(gamma h) - J) k_i = f(t + alpha_i h, y + sum_{j<i} a_ij k_j)
//!                         + sum_{j<i} (c_ij / h) k_j + gamma_i h f_t,
//! y_new = y + sum m_i k_i,      err = sum (m_i - mhat_i) k_i,
//! ```
//!
//! the coefficients of **Rodas3** (A. Sandu, J. G. Verwer, J. G. Blom, E. J.
//! Spee, G. R. Carmichael, F. A. Potra, "Benchmarking stiff ODE solvers for
//! atmospheric chemistry problems II: Rosenbrock solvers", Atmospheric
//! Environment 31 (1997) 3459-3472; also Hairer & Wanner, *Solving ODEs II*,
//! IV.7) give four stages, order 3, and stiff accuracy: `R(z) -> 0` as
//! `z -> -infinity` (L-stability), so the fast components are damped
//! completely instead of ringing, which is what makes it usable for
//! Robertson-type chemistry and relaxation oscillators at any tolerance.
//! `gamma = 1/2`. The table below is internally checked in the tests: the
//! `gamma_i` and `alpha_i` are *derived* from `c`, `a` and `gamma` and must
//! match, `sum m_i k^t_i = h` (consistency of the time variable), and the
//! observed convergence order on a nonlinear problem is 3.
//!
//! # The Jacobian
//!
//! The [`System`](crate::System) API offers no Jacobian, so `J = df/dy` is a
//! forward difference, one right-hand-side call per column, and `f_t` one
//! more call. The perturbation of column `j` is
//! `sqrt(eps) * max(|y_j|, atol/rtol, 1e-10)`:
//!
//! * proportional to `|y_j|` so the relative truncation and roundoff errors
//!   balance at about `sqrt(eps) ~ 1.5e-8`;
//! * floored by `atol/rtol`, the size below which the error control treats a
//!   component as "zero", so a component sitting at 0 (or at `1e-30`, as in
//!   chemistry) still gets a perturbation that survives roundoff in `f`;
//! * and by an absolute `1e-10` for the degenerate `atol = 0` case.
//!
//! `J` only enters through `W` and the stage right-hand sides, so its
//! errors act like a slightly different (stable) method; they do not spoil
//! the solution beyond the error the controller measures. The perturbed
//! point is `y_j + delta` with `delta` recomputed as `(y_j + delta) - y_j`,
//! so the divisor is exactly the step taken.
//!
//! The Jacobian and `f_t` are computed once per accepted state and reused
//! after a rejection (the state is the same; only `h` changed).

use crate::linalg::Lu;
use crate::run::{all_finite, initial_step, scaled_rms, step_underflow, Run};
use crate::SolveError;

const GAMMA: f64 = 0.5;
const S: usize = 4;
/// `a[i][j]`, the coefficients of the stage arguments.
const A: [[f64; S]; S] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 1.0, 0.0],
];
/// `c[i][j]`, the coefficients of the stage right-hand sides.
const C: [[f64; S]; S] = [
    [0.0, 0.0, 0.0, 0.0],
    [4.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0],
    [1.0, -1.0, -8.0 / 3.0, 0.0],
];
/// Solution weights, and the embedded (order 2) weights.
const M: [f64; S] = [2.0, 0.0, 1.0, 1.0];
const MHAT: [f64; S] = [2.0, 0.0, 1.0, 0.0];
/// Stage times `alpha_i` and time-derivative coefficients `gamma_i`.
const ALPHA: [f64; S] = [0.0, 0.0, 1.0, 1.0];
const GAMMA_I: [f64; S] = [0.5, 1.5, 0.0, 0.0];

const SAFETY: f64 = 0.9;
const FAC_MIN: f64 = 0.2;
const FAC_MAX: f64 = 5.0;

/// Which stage arguments and stage times coincide with `(t, y)`, so the
/// evaluation can reuse `f0`.
fn stage_is_start(i: usize) -> bool {
    i == 0 || (ALPHA[i] == 0.0 && A[i].iter().all(|&a| a == 0.0))
}

/// The finite-difference Jacobian `J[i * n + j] = d f_i / d y_j` at
/// `(t, y)`, given `f0 = f(t, y)`. `None` if it is not finite.
fn jacobian(
    run: &mut Run<'_>,
    t: f64,
    y: &[f64],
    f0: &[f64],
) -> Result<Option<Vec<f64>>, SolveError> {
    let n = y.len();
    let floor = if run.opts.rtol > 0.0 {
        run.opts.atol / run.opts.rtol
    } else {
        0.0
    }
    .max(1e-10);
    let root_eps = f64::EPSILON.sqrt();
    let mut jac = vec![0.0; n * n];
    let mut yp = y.to_vec();
    let mut fp = vec![0.0; n];
    for j in 0..n {
        let mut delta = root_eps * y[j].abs().max(floor);
        let mut ok = false;
        // Forward difference; if that leaves the domain of f (a NaN), try
        // the other side before giving up.
        for _ in 0..2 {
            yp[j] = y[j] + delta;
            let d = yp[j] - y[j];
            run.rhs(t, &yp, &mut fp)?;
            if all_finite(&fp) && d != 0.0 {
                for i in 0..n {
                    jac[i * n + j] = (fp[i] - f0[i]) / d;
                }
                ok = true;
                break;
            }
            delta = -delta;
        }
        yp[j] = y[j];
        if !ok {
            return Ok(None);
        }
    }
    Ok(Some(jac))
}

/// Integrates from `(t, y)` to `run.t1` with Rodas3, stepping exactly onto
/// each requested output time. Returns the final `(t, y)`.
pub(crate) fn rodas3(
    run: &mut Run<'_>,
    mut t: f64,
    mut y: Vec<f64>,
    h_start: Option<f64>,
) -> Result<(f64, Vec<f64>), SolveError> {
    let n = y.len();
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let t1 = run.t1;
    let span = t1 - run.t0;
    let hmax = run.opts.max_step.map_or(span, |m| m.min(span));
    let root_eps = f64::EPSILON.sqrt();

    let mut f0 = vec![0.0; n];
    run.rhs(t, &y, &mut f0)?;
    if !all_finite(&f0) {
        return Err(SolveError::NotFinite { t });
    }
    run.flush_outputs(t, &y);
    if t >= t1 {
        return Ok((t, y));
    }
    let mut h = match h_start.or(run.opts.first_step) {
        Some(h) => h,
        None => initial_step(run, t, &y, &f0, 3, hmax),
    }
    .min(hmax);

    let mut jac: Vec<f64> = Vec::new();
    let mut ft = vec![0.0; n];
    let mut have_jac = false;
    let mut reject = false;
    let mut bad = 0usize;
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; S];
    let (mut ystage, mut fstage, mut ynew, mut err) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    loop {
        run.budget(t)?;
        h = h.min(hmax);
        if step_underflow(t, h) {
            return Err(SolveError::StepUnderflow { t, step: h });
        }
        // Step exactly onto the next output or t1.
        let target = run.next_output().filter(|&o| o < t1).unwrap_or(t1);
        let mut hh = h;
        let mut clipped = false;
        if t + 1.01 * hh >= target {
            hh = target - t;
            clipped = true;
        }

        if !have_jac {
            match jacobian(run, t, &y, &f0)? {
                Some(j) => jac = j,
                None => return Err(SolveError::NotFinite { t }),
            }
            // f_t by a forward difference in time.
            let dt = root_eps * t.abs().max(hh).max(1e-10);
            let tp = t + dt;
            run.rhs(tp, &y, &mut ft)?;
            if !all_finite(&ft) {
                return Err(SolveError::NotFinite { t });
            }
            for i in 0..n {
                ft[i] = (ft[i] - f0[i]) / (tp - t);
            }
            have_jac = true;
        }

        // W = I/(gamma h) - J, factored once per attempt.
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                w[i * n + j] = -jac[i * n + j];
            }
            w[i * n + i] += 1.0 / (GAMMA * hh);
        }
        let Some(lu) = Lu::factor(w, n) else {
            bad += 1;
            if bad > 40 {
                return Err(SolveError::NotFinite { t });
            }
            run.stats.rejected += 1;
            reject = true;
            h = hh * 0.25;
            continue;
        };

        let ok = 'stages: {
            for i in 0..S {
                let fi: &[f64] = if stage_is_start(i) {
                    &f0
                } else {
                    for q in 0..n {
                        let mut s = y[q];
                        for j in 0..i {
                            s += A[i][j] * k[j][q];
                        }
                        ystage[q] = s;
                    }
                    run.rhs(t + ALPHA[i] * hh, &ystage, &mut fstage)?;
                    if !all_finite(&fstage) {
                        break 'stages false;
                    }
                    &fstage
                };
                let mut rhs = vec![0.0; n];
                for q in 0..n {
                    let mut s = fi[q] + GAMMA_I[i] * hh * ft[q];
                    for j in 0..i {
                        s += C[i][j] / hh * k[j][q];
                    }
                    rhs[q] = s;
                }
                lu.solve(&mut rhs);
                if !all_finite(&rhs) {
                    break 'stages false;
                }
                k[i] = rhs;
            }
            true
        };
        if !ok {
            bad += 1;
            if bad > 40 {
                return Err(SolveError::NotFinite { t });
            }
            run.stats.rejected += 1;
            reject = true;
            h = hh * 0.25;
            continue;
        }
        for q in 0..n {
            let mut s = y[q];
            let mut e = 0.0;
            for i in 0..S {
                s += M[i] * k[i][q];
                e += (M[i] - MHAT[i]) * k[i][q];
            }
            ynew[q] = s;
            err[q] = e;
        }
        let errn = scaled_rms(&err, &y, &ynew, rtol, atol);
        if !errn.is_finite() || !all_finite(&ynew) {
            bad += 1;
            if bad > 40 {
                return Err(SolveError::NotFinite { t });
            }
            run.stats.rejected += 1;
            reject = true;
            h = hh * 0.25;
            continue;
        }
        bad = 0;

        // The error estimate has order 2, so the step scales like err^(-1/3).
        let mut fac = if errn == 0.0 {
            FAC_MAX
        } else {
            (SAFETY * libm::pow(errn, -1.0 / 3.0)).clamp(FAC_MIN, FAC_MAX)
        };
        if errn <= 1.0 {
            run.stats.steps += 1;
            let tnew = if clipped { target } else { t + hh };
            std::mem::swap(&mut y, &mut ynew);
            t = tnew;
            run.monitor(t, &y)?;
            run.flush_outputs(t, &y);
            if t >= t1 {
                return Ok((t, y));
            }
            if reject {
                fac = fac.min(1.0);
            }
            reject = false;
            let mut hn = hh * fac;
            if clipped {
                // The step was cut short by an output: do not let that
                // shrink the step the dynamics would allow.
                hn = hn.max(h * fac.min(1.0));
            }
            h = hn;
            run.rhs(t, &y, &mut f0)?;
            if !all_finite(&f0) {
                return Err(SolveError::NotFinite { t });
            }
            have_jac = false;
        } else {
            run.stats.rejected += 1;
            reject = true;
            h = hh * fac.min(SAFETY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived quantities must match the tabulated `alpha_i`, `gamma_i`:
    /// this catches a mistyped entry of `a`, `c` or `gamma`, because the
    /// time variable is itself an ODE (`t' = 1`) that the method must
    /// integrate exactly.
    #[test]
    fn time_variable_is_integrated_consistently() {
        // k^t_i = gamma h + gamma sum_j c_ij k^t_j, for h = 1.
        let mut kt = [0.0; S];
        for i in 0..S {
            let mut s = GAMMA;
            for j in 0..i {
                s += GAMMA * C[i][j] * kt[j];
            }
            kt[i] = s;
        }
        for i in 0..S {
            assert!((kt[i] - GAMMA_I[i]).abs() < 1e-14, "gamma_{i}");
            let alpha: f64 = (0..i).map(|j| A[i][j] * kt[j]).sum();
            assert!((alpha - ALPHA[i]).abs() < 1e-14, "alpha_{i}");
        }
        let sum: f64 = (0..S).map(|i| M[i] * kt[i]).sum();
        assert!((sum - 1.0).abs() < 1e-14, "sum m_i k^t_i = {sum}");
    }
}
