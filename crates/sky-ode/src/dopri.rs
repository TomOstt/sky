//! Dormand-Prince 5(4): the workhorse adaptive explicit Runge-Kutta method.
//!
//! * Coefficients: J. R. Dormand and P. J. Prince, "A family of embedded
//!   Runge-Kutta formulae", J. Comput. Appl. Math. 6 (1980) 19-26. Seven
//!   stages, fifth order, with an embedded fourth-order solution for the
//!   error estimate. The last stage is the first stage of the next step
//!   (FSAL: first same as last), so a step costs six new evaluations.
//! * Step-size control: the classical scheme of Hairer, Nørsett & Wanner,
//!   *Solving ODEs I* (1993), II.4, as implemented in their `dopri5.f`: an
//!   error norm with mixed `atol`/`rtol` scaling, a step factor from a PI
//!   controller (Gustafsson; `beta = 0.04`, which smooths the step sequence
//!   over plain `err^(-1/5)` control), letting the step grow at most 10x or
//!   shrink at most 5x at a time, and never growing on the step right after
//!   a rejection.
//! * Dense output: the continuous extension of Hairer's `dopri5.f`
//!   (`contd5`; fourth-order accurate inside the step, as the general theory
//!   of Runge-Kutta interpolants allows for a 7-stage, order-5 method): a
//!   quintic in `theta` that matches `y_n`, `y_{n+1}`, `f_n` and `f_{n+1}` and
//!   uses the stage values for the remaining freedom. It lets
//!   `Options::outputs` fall anywhere inside a step without forcing the step
//!   to end there, so requesting outputs never changes the integration.
//! * Stiffness detection: Hairer's `dopri5.f` estimate of the dominant
//!   eigenvalue `|h lambda| ~ h |k7 - k6| / |y7 - y6|` (the two stages are
//!   both evaluated near the end of the step, so their difference quotient
//!   approximates the Jacobian's largest eigenvalue along the solution). The
//!   stability region of the method reaches about 3.3 on the negative real
//!   axis. If `|h lambda|` stays above 3.25 for 15 accepted steps (the
//!   counter is reset after 6 steps below it) the step is limited by
//!   stability, not accuracy, which is the definition of stiff.

use crate::run::{all_finite, initial_step, scaled_rms, step_underflow, Run};
use crate::SolveError;

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
// Row 7 is also the fifth-order solution (b): A72 = 0.
const A71: f64 = 35.0 / 384.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

// Error coefficients e = b - b_hat (fifth minus fourth order solution).
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// Dense-output coefficients (Hairer's `dopri5.f`, `d1 ... d7`).
const D1: f64 = -12715105075.0 / 11282082432.0;
const D3: f64 = 87487479700.0 / 32700410799.0;
const D4: f64 = -10690763975.0 / 1880347072.0;
const D5: f64 = 701980252875.0 / 199316789632.0;
const D6: f64 = -1453857185.0 / 822651844.0;
const D7: f64 = 69997945.0 / 29380423.0;

/// Step-size controller constants (Hairer's defaults).
const SAFETY: f64 = 0.9;
/// The step shrinks by at most this factor per attempt (5x) ...
const FAC_MIN: f64 = 0.2;
/// ... and grows by at most this factor per step.
const FAC_MAX: f64 = 10.0;
const BETA: f64 = 0.04;
const EXPO1: f64 = 0.2 - BETA * 0.75;

/// Stiffness detection thresholds (Hairer's `dopri5.f`).
const STIFF_LAMBDA: f64 = 3.25;
const STIFF_HITS: usize = 15;
const STIFF_RESET: usize = 6;

/// Why `dopri5` stopped.
pub(crate) enum Outcome {
    /// Reached `t1`.
    Done { t: f64, y: Vec<f64> },
    /// The step size is limited by stability: hand over to a stiff method.
    Stiff(StiffAlarm),
}

/// What was observed when stiffness was detected.
pub(crate) struct StiffAlarm {
    pub(crate) t: f64,
    pub(crate) y: Vec<f64>,
    /// The step just taken.
    pub(crate) h: f64,
    /// The estimate of `h * |lambda|` at the last step (the limit is ~3.3).
    pub(crate) hlamb: f64,
    /// The error estimate at that step, as a fraction of the tolerance.
    pub(crate) err: f64,
    /// The relative change of the state over that step.
    pub(crate) rel_change: f64,
}

/// Evaluates a stage; `false` if the result is not finite (the caller then
/// shrinks the step).
fn stage(run: &mut Run<'_>, t: f64, y: &[f64], k: &mut [f64]) -> Result<bool, SolveError> {
    run.rhs(t, y, k)?;
    Ok(all_finite(k))
}

/// Integrates from `(t, y)` to `run.t1` with adaptive steps and dense
/// output. `h_start` overrides the initial step (used after a hand-over);
/// with `detect_stiffness` the run stops early with [`Outcome::Stiff`] when
/// the step is limited by stability.
pub(crate) fn dopri5(
    run: &mut Run<'_>,
    mut t: f64,
    mut y: Vec<f64>,
    h_start: Option<f64>,
    detect_stiffness: bool,
) -> Result<Outcome, SolveError> {
    let n = y.len();
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let t1 = run.t1;
    let span = t1 - run.t0;
    let hmax = run.opts.max_step.map_or(span, |m| m.min(span));
    let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut k5, mut k6, mut k7) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut ytmp, mut ynew, mut err) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    run.rhs(t, &y, &mut k1)?;
    if !all_finite(&k1) {
        return Err(SolveError::NotFinite { t });
    }
    run.flush_outputs(t, &y);
    if t >= t1 {
        return Ok(Outcome::Done { t, y });
    }
    let mut h = match h_start.or(run.opts.first_step) {
        Some(h) => h,
        None => initial_step(run, t, &y, &k1, 5, hmax),
    }
    .min(hmax);

    let mut facold = 1e-4_f64;
    let mut reject = false;
    let mut nonfinite = 0usize;
    let (mut nonsti, mut iasti) = (0usize, 0usize);

    loop {
        run.budget(t)?;
        if step_underflow(t, h) {
            return Err(SolveError::StepUnderflow { t, step: h });
        }
        // Land exactly on t1 rather than leave a sliver for one more step.
        let mut last = false;
        let mut hh = h;
        if t + 1.01 * hh - t1 > 0.0 {
            hh = t1 - t;
            last = true;
        }

        let ok = 'stages: {
            for i in 0..n {
                ytmp[i] = y[i] + hh * A21 * k1[i];
            }
            if !stage(run, t + C2 * hh, &ytmp, &mut k2)? {
                break 'stages false;
            }
            for i in 0..n {
                ytmp[i] = y[i] + hh * (A31 * k1[i] + A32 * k2[i]);
            }
            if !stage(run, t + C3 * hh, &ytmp, &mut k3)? {
                break 'stages false;
            }
            for i in 0..n {
                ytmp[i] = y[i] + hh * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
            }
            if !stage(run, t + C4 * hh, &ytmp, &mut k4)? {
                break 'stages false;
            }
            for i in 0..n {
                ytmp[i] = y[i] + hh * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
            }
            if !stage(run, t + C5 * hh, &ytmp, &mut k5)? {
                break 'stages false;
            }
            for i in 0..n {
                ytmp[i] = y[i]
                    + hh * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
            }
            if !stage(run, t + hh, &ytmp, &mut k6)? {
                break 'stages false;
            }
            // `ytmp` now holds the argument of stage 6, kept for the
            // stiffness estimate below.
            for i in 0..n {
                ynew[i] = y[i]
                    + hh * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
            }
            if !all_finite(&ynew) || !stage(run, t + hh, &ynew, &mut k7)? {
                break 'stages false;
            }
            true
        };
        if !ok {
            // A trial stage left the finite numbers: the step was too bold.
            // (A right-hand side that *reports an error* is different, and
            // has already stopped the solve above.)
            nonfinite += 1;
            if nonfinite > 40 {
                return Err(SolveError::NotFinite { t });
            }
            run.stats.rejected += 1;
            reject = true;
            h = hh * 0.25;
            continue;
        }

        for i in 0..n {
            err[i] =
                hh * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        }
        let errn = scaled_rms(&err, &y, &ynew, rtol, atol);
        if !errn.is_finite() {
            nonfinite += 1;
            if nonfinite > 40 {
                return Err(SolveError::NotFinite { t });
            }
            run.stats.rejected += 1;
            reject = true;
            h = hh * 0.25;
            continue;
        }
        nonfinite = 0;

        // PI step-size controller (Hairer's `dopri5.f`): `fac` is the
        // factor by which the step is *divided*.
        let fac11 = libm::pow(errn, EXPO1);
        let fac = (fac11 / libm::pow(facold, BETA)).clamp(1.0 / FAC_MAX, 1.0 / FAC_MIN);
        let mut hnew = hh / fac;

        if errn <= 1.0 {
            facold = errn.max(1e-4);
            run.stats.steps += 1;
            let tnew = if last { t1 } else { t + hh };

            // Dense output for the outputs inside (t, tnew].
            if run.next_output().is_some_and(|o| o <= tnew) {
                let ydiff: Vec<f64> = (0..n).map(|i| ynew[i] - y[i]).collect();
                let bspl: Vec<f64> = (0..n).map(|i| hh * k1[i] - ydiff[i]).collect();
                let r4: Vec<f64> = (0..n).map(|i| ydiff[i] - hh * k7[i] - bspl[i]).collect();
                let r5: Vec<f64> = (0..n)
                    .map(|i| {
                        hh * (D1 * k1[i]
                            + D3 * k3[i]
                            + D4 * k4[i]
                            + D5 * k5[i]
                            + D6 * k6[i]
                            + D7 * k7[i])
                    })
                    .collect();
                while let Some(o) = run.next_output().filter(|&o| o <= tnew) {
                    let yo = if o >= tnew {
                        ynew.clone()
                    } else {
                        let th = (o - t) / hh;
                        let th1 = 1.0 - th;
                        (0..n)
                            .map(|i| {
                                y[i] + th
                                    * (ydiff[i] + th1 * (bspl[i] + th * (r4[i] + th1 * r5[i])))
                            })
                            .collect()
                    };
                    run.push_output(yo);
                }
            }

            // Stiffness: the difference quotient of the last two stages.
            let mut alarm = None;
            if detect_stiffness && !last {
                let (mut num, mut den) = (0.0, 0.0);
                for i in 0..n {
                    let dk = k7[i] - k6[i];
                    let dy = ynew[i] - ytmp[i];
                    num += dk * dk;
                    den += dy * dy;
                }
                if den > 0.0 {
                    let hlamb = hh * (num / den).sqrt();
                    if hlamb > STIFF_LAMBDA {
                        nonsti = 0;
                        iasti += 1;
                        if iasti == STIFF_HITS {
                            alarm = Some(hlamb);
                        }
                    } else {
                        nonsti += 1;
                        if nonsti == STIFF_RESET {
                            iasti = 0;
                        }
                    }
                }
            }

            let rel_change = {
                let d: f64 = (0..n).map(|i| (ynew[i] - y[i]).powi(2)).sum();
                let s: f64 = ynew.iter().map(|v| v * v).sum();
                (d / s.max(f64::MIN_POSITIVE)).sqrt()
            };
            std::mem::swap(&mut y, &mut ynew);
            t = tnew;
            std::mem::swap(&mut k1, &mut k7); // FSAL
            run.monitor(t, &y)?;
            if last {
                return Ok(Outcome::Done { t, y });
            }
            if reject {
                hnew = hnew.min(hh);
            }
            reject = false;
            h = hnew.min(hmax);
            if let Some(hlamb) = alarm {
                return Ok(Outcome::Stiff(StiffAlarm {
                    t,
                    y,
                    h: hh,
                    hlamb,
                    err: errn,
                    rel_change,
                }));
            }
        } else {
            // Rejected: shrink, at most by 1/FAC_MIN per attempt.
            run.stats.rejected += 1;
            reject = true;
            h = hh / (1.0 / FAC_MIN).min(fac11 / SAFETY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coefficient tables satisfy the order conditions that are cheap to
    /// check exactly: consistency `sum_j a_ij = c_i`, and the weights of the
    /// fifth-order solution summing to 1 (with the embedded error weights
    /// summing to 0).
    #[test]
    fn tableau_is_consistent() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-14;
        assert!(close(A21, C2));
        assert!(close(A31 + A32, C3));
        assert!(close(A41 + A42 + A43, C4));
        assert!(close(A51 + A52 + A53 + A54, C5));
        assert!(close(A61 + A62 + A63 + A64 + A65, 1.0));
        assert!(close(A71 + A73 + A74 + A75 + A76, 1.0));
        assert!(close(E1 + E3 + E4 + E5 + E6 + E7, 0.0));
        // The dense-output coefficients are checked by their accuracy in
        // `tests/accuracy.rs` (interpolated samples vs the exact solution).
    }
}
