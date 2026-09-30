//! `Method::Auto`: choosing the integrator, and saying why.
//!
//! The rules, in order (each choice is a pure function of the system's
//! declared structure, the options and the deterministic numbers the probes
//! produce, so a solve is reproducible):
//!
//! 1. **Second order and a conserved quantity is monitored** -> symplectic.
//!    A symplectic integrator conserves a nearby Hamiltonian exactly, so the
//!    energy error stays bounded for very long times where an ordinary
//!    method's drifts (Hairer, Lubich & Wanner 2006, IX.8). Yoshida's
//!    4th-order composition when `rtol <= 1e-4`, velocity Verlet above (its
//!    three force evaluations per step and large error constant only pay off
//!    at tight tolerances). The **step is derived from the tolerance** by
//!    [`probe_step`], checked against the monitored quantities themselves by
//!    trial runs (`trial`) so that their bounded drift stays within the
//!    tolerance (and within the user's `max_drift`), then held fixed.
//! 2. **Otherwise** Dormand-Prince 5(4) with Hairer's stiffness detection;
//!    when the step is found to be limited by stability, continue from the
//!    current state with the L-stable Rosenbrock method (Rodas3). The switch
//!    is one-way: a solve that turned stiff stays with the method that does
//!    not care.

use crate::dopri::{dopri5, Outcome};
use crate::fixed::{drive, verify_second_order, Stepper, SymKind, Symplectic};
use crate::rosenbrock::rodas3;
use crate::run::{all_finite, initial_step, num, scaled_rms, step_underflow, Run};
use crate::{Method, SolveError};

/// The result of an automatic solve: what ran, the final state, and the
/// sentences explaining the choice (before the drift report).
pub(crate) struct Chosen {
    pub(crate) method: Method,
    pub(crate) t: f64,
    pub(crate) y: Vec<f64>,
    pub(crate) explanation: Vec<String>,
}

/// Joins names as "a", "a and b", "a, b and c".
fn join_names(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [a] => (*a).to_string(),
        [init @ .., last] => format!("{} and {}", init.join(", "), last),
    }
}

pub(crate) fn auto(
    run: &mut Run<'_>,
    second_order: Option<usize>,
    y0: &[f64],
) -> Result<Chosen, SolveError> {
    if run.t1 == run.t0 {
        run.flush_outputs(run.t0, y0);
        return Ok(Chosen {
            method: Method::DormandPrince,
            t: run.t0,
            y: y0.to_vec(),
            explanation: vec![
                "the time interval is empty, so there was nothing to integrate".to_string(),
            ],
        });
    }
    match second_order {
        Some(n) if !run.opts.conserved.is_empty() => symplectic(run, n, y0),
        other => general(run, other.is_some(), y0),
    }
}

// ---------------------------------------------------------------------------
// Symplectic
// ---------------------------------------------------------------------------

/// At most this many accepted probe steps.
const PROBE_STEPS: usize = 2000;
/// The fixed step is this fraction of the smallest step the probe accepted.
const PROBE_SAFETY: f64 = 0.8;

fn symplectic(run: &mut Run<'_>, n: usize, y0: &[f64]) -> Result<Chosen, SolveError> {
    crate::check_tolerances(run.opts)?;
    verify_second_order(run, n, y0)?;
    let names: Vec<&str> = run.opts.conserved.iter().map(|c| c.name.as_str()).collect();
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let tol = if rtol > 0.0 { rtol } else { atol };
    let (kind, why_kind) = if tol <= 1e-4 {
        (
            SymKind::Yoshida4,
            format!(
                "at tolerance {} a fourth-order composition (Yoshida) needs far fewer steps than second-order velocity Verlet",
                num(tol)
            ),
        )
    } else {
        (
            SymKind::Verlet,
            format!(
                "at tolerance {} second-order velocity Verlet is cheaper per unit time than a fourth-order composition",
                num(tol)
            ),
        )
    };
    let probe = probe_step(run, kind, n, y0)?;
    let max_step = run.opts.max_step.unwrap_or(f64::INFINITY);
    let h_probe = probe.h.min(max_step);

    // Check the step against what was actually asked for: the monitored
    // quantities. Run the candidate on the probe's window and shrink the
    // step until each stays within its target.
    let targets = drift_targets(run, tol);
    let mut h = h_probe;
    let mut first: Option<Trial> = None;
    let mut last: Trial;
    let mut tries = 0;
    loop {
        last = trial(run, kind, n, y0, h, probe.covered, &targets)?;
        if first.is_none() {
            first = Some(last.clone());
        }
        tries += 1;
        if last.ratio <= 1.0 || tries == MAX_TRIALS {
            break;
        }
        // The bounded energy error of a p-th order method scales like h^p.
        let factor = (0.9 * last.ratio.powf(-1.0 / f64::from(kind.order()))).clamp(0.1, 0.9);
        h *= factor;
        if h < 1e-3 * h_probe {
            break; // the target is out of reach (or a roundoff floor): stop
        }
    }
    let first = first.expect("at least one trial ran");

    let mut stepper = Symplectic::new(kind, n, y0);
    let y = drive(run, &mut stepper, h)?;
    let method = match kind {
        SymKind::Verlet => Method::VelocityVerlet { step: h },
        SymKind::Yoshida4 => Method::Yoshida4 { step: h },
    };

    let plural = names.len() > 1;
    let mut explanation = vec![
        format!(
            "the system is second order and {} {} monitored: symplectic integrators keep {} error bounded over long times instead of letting it drift",
            join_names(&names),
            if plural { "are" } else { "is" },
            if plural { "their" } else { names[0] }
        ),
        format!("{why_kind}, so {} was chosen", method.name()),
    ];
    let covered = if probe.whole {
        "the whole interval".to_string()
    } else {
        format!("the first {} s of the interval", num(probe.covered))
    };
    explanation.push(format!(
        "the step comes from rtol = {} and atol = {}: a probe (adaptive step doubling, where one step compared with two half steps estimates the local error) ran over {covered} in {} steps and estimated the largest step the tolerance allows at each point; the smallest was {} s, and {} times that, {} s, is held fixed for the whole run, because varying the step would forfeit the bounded-energy property",
        num(rtol),
        num(atol),
        probe.steps,
        num(probe.h_min),
        num(PROBE_SAFETY),
        num(h_probe),
    ));
    let worst = names[first.worst];
    if h == h_probe {
        explanation.push(format!(
            "a trial run at that step over {} confirmed it: {worst} drifted by at most {} (relative), within the {} the tolerance asks",
            trial_window(&first, probe.whole),
            num(first.drift[first.worst]),
            num(targets[first.worst]),
        ));
    } else {
        explanation.push(format!(
            "but a trial run at that step over {} showed {worst} drifting by {} (relative), more than the {} the tolerance asks, so the step was reduced to {} s (a {}-order method's bounded error scales like the step to that power), where the trial drift was {}",
            trial_window(&first, probe.whole),
            num(first.drift[first.worst]),
            num(targets[first.worst]),
            num(h),
            match kind {
                SymKind::Verlet => "second",
                SymKind::Yoshida4 => "fourth",
            },
            num(last.drift[last.worst]),
        ));
    }
    explanation.push(format!(
        "took {} fixed steps of at most {} s",
        run.stats.steps,
        num(h)
    ));
    Ok(Chosen {
        method,
        t: run.t1,
        y,
        explanation,
    })
}

/// The most trial runs when shrinking the step.
const MAX_TRIALS: usize = 6;
/// The most steps of one trial run.
const TRIAL_STEPS: usize = 100_000;

/// The drift each monitored quantity may show, relative to its initial
/// value: the tolerance, and a quarter of the user's `max_drift` if that is
/// tighter (so the run stays clear of the limit that would abort it).
fn drift_targets(run: &Run<'_>, tol: f64) -> Vec<f64> {
    run.opts
        .conserved
        .iter()
        .map(|c| c.max_drift.map_or(tol, |m| tol.min(0.25 * m)))
        .collect()
}

#[derive(Clone)]
struct Trial {
    /// The largest relative drift of each conserved quantity.
    drift: Vec<f64>,
    /// The largest `drift / target`, and which quantity it belongs to.
    ratio: f64,
    worst: usize,
    /// The time the trial covered.
    covered: f64,
}

fn trial_window(t: &Trial, whole: bool) -> String {
    if whole {
        "the whole interval".to_string()
    } else {
        format!("the first {} s", num(t.covered))
    }
}

/// Runs the symplectic method at the fixed step `h` from `y0` for `window`
/// (at most [`TRIAL_STEPS`] steps) and records the drift of the monitored
/// quantities. Uses the same drift definition as the real run's monitor, but
/// touches neither its statistics nor its limits.
fn trial(
    run: &mut Run<'_>,
    kind: SymKind,
    n: usize,
    y0: &[f64],
    h: f64,
    window: f64,
    targets: &[f64],
) -> Result<Trial, SolveError> {
    let t0 = run.t0;
    let count = ((window / h).ceil().max(1.0) as usize).min(TRIAL_STEPS);
    let initial: Vec<f64> = run
        .opts
        .conserved
        .iter()
        .map(|c| (c.value)(t0, y0))
        .collect();
    let mut drift = vec![0.0_f64; initial.len()];
    let mut st = Symplectic::new(kind, n, y0);
    let mut covered = 0.0;
    for j in 0..count {
        st.advance(run, t0 + j as f64 * h, h)?;
        covered = (j + 1) as f64 * h;
        let y = st.state();
        let finite = all_finite(y);
        for (i, c) in run.opts.conserved.iter().enumerate() {
            let q = (c.value)(t0 + covered, y);
            let scale = if initial[i] != 0.0 {
                initial[i].abs()
            } else {
                1.0
            };
            let d = if finite && q.is_finite() {
                (q - initial[i]).abs() / scale
            } else {
                f64::INFINITY
            };
            drift[i] = drift[i].max(d);
        }
        if !finite {
            break;
        }
    }
    let (worst, ratio) = drift
        .iter()
        .zip(targets)
        .map(|(d, t)| d / t)
        .enumerate()
        .fold(
            (0, 0.0_f64),
            |best, (i, r)| if r > best.1 { (i, r) } else { best },
        );
    Ok(Trial {
        drift,
        ratio,
        worst,
        covered: covered.min(window.max(h)),
    })
}

struct Probe {
    /// The step to use.
    h: f64,
    /// The smallest step the probe found the tolerance to allow.
    h_min: f64,
    /// Accepted probe steps.
    steps: usize,
    /// Time covered by the probe, and whether that is all of the interval.
    covered: f64,
    whole: bool,
}

/// Chooses the fixed step for a symplectic method from the tolerance.
///
/// Step doubling: from the state `y` take one step of `h` and two of `h/2`
/// with the method itself. Their difference, divided by `2^p - 1` for a
/// method of order `p` (Richardson), estimates the local error of the
/// *finer* result. That estimate goes through the same mixed `atol`/`rtol`
/// norm as the adaptive methods, and the usual controller (`0.9 err^(-1/(p+1))`,
/// between 0.2x and 5x) adapts `h` so the norm is about 1: **the tolerance
/// is a per-step local error, exactly as for Dormand-Prince**, so the same
/// numbers mean the same thing for every method.
///
/// The probe walks the trajectory (advancing with the finer result) for up
/// to [`PROBE_STEPS`] accepted steps or the whole interval. At every
/// attempt it also computes the step `h err^(-1/(p+1))` that would have met
/// the tolerance exactly, and returns `PROBE_SAFETY` times the *smallest*
/// of those: the fixed step must satisfy the tolerance at the hardest place
/// the probe saw, for instance the pericentre of an eccentric orbit. If the
/// interval is longer than the probe, places beyond it are not covered, and
/// the explanation says so.
fn probe_step(run: &mut Run<'_>, kind: SymKind, n: usize, y0: &[f64]) -> Result<Probe, SolveError> {
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let (t0, t1) = (run.t0, run.t1);
    let span = t1 - t0;
    let p = kind.order();
    let denom = f64::from(2_i32.pow(p as u32) - 1);
    let expo = -1.0 / f64::from(p + 1);

    let mut f0 = vec![0.0; 2 * n];
    run.rhs(t0, y0, &mut f0)?;
    if !all_finite(&f0) {
        return Err(SolveError::NotFinite { t: t0 });
    }
    let mut h = match run.opts.first_step {
        Some(h) => h,
        None => initial_step(run, t0, y0, &f0, p as u32, span),
    }
    .min(span);

    let mut stepper = Symplectic::new(kind, n, y0);
    let (mut t, mut y) = (t0, y0.to_vec());
    let (mut steps, mut nonfinite) = (0usize, 0usize);
    let mut h_min = f64::INFINITY;
    let mut h_max_tried = 0.0_f64;
    let mut attempts = 0usize;
    while t < t1 && steps < PROBE_STEPS {
        attempts += 1;
        if attempts > 20 * PROBE_STEPS || step_underflow(t, h) {
            return Err(SolveError::StepUnderflow { t, step: h });
        }
        let mut hh = h;
        let mut last = false;
        if t + 1.01 * hh >= t1 {
            hh = t1 - t;
            last = true;
        }
        stepper.step_from(run, t, hh, &y)?;
        let ya = stepper.state().to_vec();
        stepper.step_from(run, t, 0.5 * hh, &y)?;
        let mid = stepper.state().to_vec();
        stepper.step_from(run, t + 0.5 * hh, 0.5 * hh, &mid)?;
        let yb = stepper.state().to_vec();
        if !all_finite(&ya) || !all_finite(&yb) {
            nonfinite += 1;
            if nonfinite > 40 {
                return Err(SolveError::NotFinite { t });
            }
            h = hh * 0.25;
            continue;
        }
        nonfinite = 0;
        let diff: Vec<f64> = (0..2 * n).map(|i| (yb[i] - ya[i]) / denom).collect();
        let err = scaled_rms(&diff, &y, &yb, rtol, atol);
        // The step that would have given exactly the tolerance here. Only
        // attempts with a measurable error say anything about that: below
        // about 1% of the tolerance the difference is mostly roundoff
        // (the small first steps of the ramp-up), which is not
        // proportional to `h^(p+1)` and would understate the allowed step.
        h_max_tried = h_max_tried.max(hh);
        if err >= 0.01 {
            h_min = h_min.min(hh * err.powf(expo));
        }
        let fac = (0.9 * err.powf(expo)).clamp(0.2, 5.0);
        if err <= 1.0 {
            y = yb;
            t = if last { t1 } else { t + hh };
            steps += 1;
            h = hh * fac;
        } else {
            h = hh * fac.min(0.9);
        }
    }
    // If no attempt ever had a measurable error (a nearly exact integrator,
    // no force, say) the tolerance never asked for a smaller step than the
    // largest tried.
    let h_min = if h_min.is_finite() {
        h_min
    } else {
        h_max_tried.min(span)
    };
    Ok(Probe {
        h: (PROBE_SAFETY * h_min).min(span),
        h_min,
        steps,
        covered: t - t0,
        whole: t >= t1,
    })
}

// ---------------------------------------------------------------------------
// General: Dormand-Prince, switching to the stiff method
// ---------------------------------------------------------------------------

fn general(run: &mut Run<'_>, second_order: bool, y0: &[f64]) -> Result<Chosen, SolveError> {
    crate::check_tolerances(run.opts)?;
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let lead = if second_order {
        "the system is second order, but no conserved quantity is monitored, so there is no reason to give up the higher accuracy per step of a general-purpose method; "
    } else {
        "no structure a specialised method could use is known, so "
    };
    let mut explanation = vec![format!(
        "{lead}the general-purpose Dormand-Prince 5(4) method was started, with adaptive steps for rtol = {} and atol = {}",
        num(rtol),
        num(atol)
    )];
    match dopri5(run, run.t0, y0.to_vec(), None, true)? {
        Outcome::Done { t, y } => {
            explanation.push(format!(
                "the step was limited by accuracy for the whole interval, never by stability, so the system is not stiff and no other method was needed; {} steps, {} rejected",
                run.stats.steps, run.stats.rejected
            ));
            Ok(Chosen {
                method: Method::DormandPrince,
                t,
                y,
                explanation,
            })
        }
        Outcome::Stiff(alarm) => {
            let (steps_dp, rej_dp) = (run.stats.steps, run.stats.rejected);
            explanation.push(format!(
                "by t = {} s the step size had shrunk to {} s while the solution changed slowly (by only {} relative per step) and the error estimate was only {} of the tolerance: the step was limited by stability, not accuracy (the estimated h*lambda stayed near {} for 15 steps, at Dormand-Prince's stability limit of about 3.3), which means the system is stiff",
                num(alarm.t),
                num(alarm.h),
                num(alarm.rel_change),
                num(alarm.err),
                num(alarm.hlamb),
            ));
            explanation.push(format!(
                "so from t = {} s the integration continued from that state with Rodas3, an L-stable Rosenbrock method (order 3, finite-difference Jacobian, no Newton iteration), whose step is set by accuracy alone: it damps the fast decaying components instead of resolving them",
                num(alarm.t)
            ));
            let (t, y) = rodas3(run, alarm.t, alarm.y, Some(alarm.h))?;
            explanation.push(format!(
                "took {} steps in all: {} with Dormand-Prince ({} rejected) before the switch and {} with Rodas3 ({} rejected) after it",
                run.stats.steps,
                steps_dp,
                rej_dp,
                run.stats.steps - steps_dp,
                run.stats.rejected - rej_dp
            ));
            Ok(Chosen {
                method: Method::Rodas3,
                t,
                y,
                explanation,
            })
        }
    }
}
