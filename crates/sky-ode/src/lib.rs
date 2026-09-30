//! Initial-value ODE solvers for Sky (ROADMAP Phase 3, ideas 11 and 12).
//!
//! The language's `equation`/`solve` lowers a system of first-order ODEs to
//! a [`System`] and calls [`solve`]. This crate is pure numerics on `f64`
//! slices: it knows nothing about units (the checker has already proved the
//! equations dimensionally consistent, and values are in SI).
//!
//! # The contract
//!
//! * [`System`] evaluates `dy/dt = f(t, y)` and may declare [`Structure`]
//!   (a second-order system `q'' = a(t, q)`, which symplectic methods
//!   need).
//! * [`Options`] carries the tolerances, the method choice (`Auto` or a
//!   named [`Method`]), conserved quantities to monitor, and output times.
//! * [`solve`] returns a [`Solution`] with the final state, samples at the
//!   requested output times, statistics, the method actually used, and an
//!   **explanation**: why that method was chosen, in words a scientist can
//!   check ("the system is second order and energy is monitored: symplectic
//!   integrators keep energy error bounded over long times instead of
//!   letting it drift ...").
//! * Every value stays finite; a non-finite state, a failing right-hand
//!   side, a step that shrinks to nothing, or exceeding
//!   [`Options::max_steps`] is a [`SolveError`] carrying the time, never NaN
//!   or infinity (Sky's rule for numbers).
//!
//! # The methods
//!
//! | [`Method`] | kind | use |
//! |---|---|---|
//! | `Rk4 { step }` | fixed step, order 4 | simple, predictable |
//! | `DormandPrince` | adaptive 5(4), FSAL, dense output | smooth, non-stiff problems |
//! | `VelocityVerlet { step }` | symplectic, order 2 | long orbits at loose tolerance |
//! | `Yoshida4 { step }` | symplectic, order 4 | long orbits at tight tolerance |
//! | `Rodas3` | adaptive L-stable Rosenbrock 3(2), finite-difference Jacobian | stiff problems |
//! | `Auto` | chooses among the above and explains why | the language's default |
//!
//! References and the reasoning behind each choice are in the module docs
//! (`dopri`, `fixed`, `rosenbrock`, `auto`).
//!
//! # Choosing automatically
//!
//! [`Method::Auto`] applies these rules in order:
//!
//! 1. A second-order system with a monitored conserved quantity gets a
//!    symplectic method, with a fixed step derived from the tolerance by a
//!    step-doubling probe.
//! 2. Otherwise Dormand-Prince 5(4) starts; if its step turns out to be
//!    limited by stability rather than accuracy (Hairer's stiffness test),
//!    the solve continues from the current state with Rodas3.
//!
//! Every choice and observation is a sentence in [`Solution::explanation`],
//! followed by the drift of each conserved quantity. Choices are
//! deterministic: the same inputs give bit-identical outputs.

#![forbid(unsafe_code)]
// Integrators index several parallel state and stage vectors with one loop
// variable (`y[i] + h * k1[i]`); iterator chains over four slices at once
// read worse than the formulas in the papers they implement.
#![allow(clippy::needless_range_loop)]

mod auto;
mod dopri;
mod fixed;
mod linalg;
mod rosenbrock;
mod run;

use std::fmt;

use fixed::{drive, verify_second_order, Rk4, SymKind, Symplectic};
use run::Run;

/// A first-order system `dy/dt = f(t, y)`.
pub trait System {
    /// The number of state variables.
    fn dim(&self) -> usize;
    /// Writes `f(t, y)` into `dydt` (length [`System::dim`]). An error (a
    /// runtime error of the user's right-hand side, say) stops the solve.
    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), String>;
    /// What the system is known to be, beyond first order. `General` is
    /// always correct; a more specific answer lets [`Method::Auto`] choose
    /// better.
    fn structure(&self) -> Structure {
        Structure::General
    }
}

/// Structure a [`System`] may declare.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Structure {
    /// Nothing known.
    General,
    /// A second-order system `q'' = a(t, q)` written as first order: the
    /// state is `[q (n values), v (n values)]`, `f` gives `q' = v` in the
    /// first half, and the acceleration in the second half depends only on
    /// `t` and `q` (not on `v`). Symplectic integrators need this.
    SecondOrder { n: usize },
}

/// An integration method.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    /// Choose from the system's structure, the conserved quantities asked
    /// for, and stiffness; the choice is explained in [`Solution::explanation`].
    Auto,
    /// Classical fourth-order Runge-Kutta with a fixed step (shortened
    /// slightly so a whole number of steps lands on each output time and on
    /// `t1`).
    Rk4 { step: f64 },
    /// Dormand-Prince 5(4) (Dormand & Prince 1980): adaptive steps from
    /// [`Options::rtol`] and [`Options::atol`] with the step-size control of
    /// Hairer, Nørsett & Wanner (*Solving ODEs I*, II.4), FSAL, and a dense
    /// output so [`Options::outputs`] never force a step. For smooth,
    /// non-stiff problems.
    DormandPrince,
    /// Velocity Verlet (Verlet 1967; kick-drift-kick leapfrog): symplectic,
    /// second order, one acceleration per step, fixed `step`. Needs
    /// [`Structure::SecondOrder`]. Energy error stays bounded (`O(step^2)`)
    /// for very long times instead of drifting.
    VelocityVerlet { step: f64 },
    /// Yoshida's fourth-order symplectic composition of Verlet steps
    /// (Yoshida 1990, *Phys. Lett. A* 150), three accelerations per step,
    /// fixed `step`. Needs [`Structure::SecondOrder`]. Pays off over Verlet
    /// at tight tolerances.
    Yoshida4 { step: f64 },
    /// Rodas3 (Sandu et al. 1997): an L-stable, stiffly accurate Rosenbrock
    /// method of order 3 with an embedded order 2 error estimate, adaptive
    /// steps from the tolerances, a finite-difference Jacobian and dense LU.
    /// For stiff problems, where the step of an explicit method is set by
    /// stability instead of accuracy.
    Rodas3,
}

impl Method {
    /// A human name for messages ("Dormand-Prince 5(4)").
    pub fn name(&self) -> &'static str {
        match self {
            Method::Auto => "automatic selection",
            Method::Rk4 { .. } => "classical Runge-Kutta 4",
            Method::DormandPrince => "Dormand-Prince 5(4)",
            Method::VelocityVerlet { .. } => "velocity Verlet (symplectic, order 2)",
            Method::Yoshida4 { .. } => "Yoshida's symplectic composition (order 4)",
            Method::Rodas3 => "Rodas3 (stiff, L-stable Rosenbrock, order 3)",
        }
    }
}

/// A function of time and state, such as a conserved energy.
pub type StateFn = Box<dyn Fn(f64, &[f64]) -> f64>;

/// A quantity to monitor, such as energy: a function of the state whose
/// relative drift over the solve is reported (and bounded by `max_drift`,
/// if given).
pub struct Conserved {
    pub name: String,
    pub value: StateFn,
    /// The largest allowed relative drift `|Q(t) - Q(t0)| / |Q(t0)|`, or
    /// `None` to only report it. (If `Q(t0)` is exactly zero the drift is
    /// absolute, there being no scale to divide by.)
    pub max_drift: Option<f64>,
}

impl fmt::Debug for Conserved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conserved")
            .field("name", &self.name)
            .field("max_drift", &self.max_drift)
            .finish()
    }
}

/// How to solve.
#[derive(Debug)]
pub struct Options {
    /// Relative tolerance per step for adaptive methods (`Auto` uses it for
    /// the symplectic step too). Zero is allowed if `atol` is positive;
    /// otherwise at least `1e-14`, about 50 machine epsilons: tighter cannot
    /// be met in double precision.
    pub rtol: f64,
    /// Absolute tolerance per step for adaptive methods: the error allowed
    /// in a component whose value is near zero.
    pub atol: f64,
    pub method: Method,
    pub conserved: Vec<Conserved>,
    /// Times (between `t0` and `t1`, increasing) at which to record the
    /// state in [`Solution::samples`].
    pub outputs: Vec<f64>,
    /// The most steps (accepted or rejected) before
    /// [`SolveError::TooManySteps`].
    pub max_steps: usize,
    /// The first step of an adaptive method, or `None` to estimate it.
    pub first_step: Option<f64>,
    /// The largest step of an adaptive method, or `None` for the interval.
    pub max_step: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            rtol: 1e-9,
            atol: 1e-12,
            method: Method::Auto,
            conserved: Vec::new(),
            outputs: Vec::new(),
            max_steps: 10_000_000,
            first_step: None,
            max_step: None,
        }
    }
}

/// Counters for a solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    /// Accepted steps, in total over every method a solve used.
    pub steps: usize,
    pub rejected: usize,
    /// Every call of the right-hand side, including finite-difference
    /// Jacobians and the step-size probe of `Auto`.
    pub rhs_evals: usize,
}

/// The result of a solve.
#[derive(Clone, PartialEq, Debug)]
pub struct Solution {
    /// The final time (`t1`) and state.
    pub t: f64,
    pub y: Vec<f64>,
    /// `(time, state)` at each requested output time, in order.
    pub samples: Vec<(f64, Vec<f64>)>,
    pub stats: Stats,
    /// The method used (never `Auto`): fully specified, so for a symplectic
    /// choice the step is the one derived from the tolerance, and for an
    /// automatic switch to the stiff method it is the method that finished.
    pub method: Method,
    /// Why this method, and what was observed (step counts, the drift of
    /// each conserved quantity), one sentence per entry.
    pub explanation: Vec<String>,
    /// The largest relative drift of each conserved quantity over the
    /// solve, in the order asked.
    pub drift: Vec<f64>,
}

/// Why a solve stopped.
#[derive(Clone, PartialEq, Debug)]
pub enum SolveError {
    /// The right-hand side reported an error at time `t`.
    Rhs { t: f64, message: String },
    /// The state stopped being finite at time `t`.
    NotFinite { t: f64 },
    /// More than `Options::max_steps` steps.
    TooManySteps { t: f64, steps: usize },
    /// A conserved quantity drifted more than allowed.
    Drift {
        name: String,
        drift: f64,
        max: f64,
        t: f64,
    },
    /// The inputs are unusable (a wrong state length, `t1 < t0`, a
    /// non-positive step or tolerance ...).
    BadInput(String),
    /// The requested method is not available (yet).
    Unsupported(String),
    /// An adaptive method needed a step so small that it no longer changes
    /// the time (or is denormal-small) at time `t`: `step` is the step it
    /// had shrunk to. Usually a singularity or blow-up of the solution at
    /// `t`, a right-hand side that is not smooth there, or a tolerance
    /// tighter than double precision can meet.
    StepUnderflow { t: f64, step: f64 },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::Rhs { t, message } => write!(f, "{message} (at t = {t} s)"),
            SolveError::NotFinite { t } => write!(f, "the solution is not finite at t = {t} s"),
            SolveError::TooManySteps { t, steps } => {
                write!(f, "gave up after {steps} steps at t = {t} s")
            }
            SolveError::Drift { name, drift, max, t } => write!(
                f,
                "{name} drifted by {drift:e} (relative), more than the allowed {max:e}, by t = {t} s"
            ),
            SolveError::BadInput(m) | SolveError::Unsupported(m) => f.write_str(m),
            SolveError::StepUnderflow { t, step } => write!(
                f,
                "step size underflow at t = {t} s: the step shrank to {step:e} s, too small to advance the time; the solution probably has a singularity or blows up here, the right-hand side is not smooth there, or the tolerance is tighter than double precision can meet"
            ),
        }
    }
}

impl std::error::Error for SolveError {}

/// Checks the tolerances of an adaptive method.
pub(crate) fn check_tolerances(opts: &Options) -> Result<(), SolveError> {
    let (rtol, atol) = (opts.rtol, opts.atol);
    if !(rtol.is_finite() && atol.is_finite() && rtol >= 0.0 && atol >= 0.0) {
        return Err(SolveError::BadInput(format!(
            "the tolerances rtol = {rtol} and atol = {atol} must be finite and not negative"
        )));
    }
    if rtol == 0.0 && atol == 0.0 {
        return Err(SolveError::BadInput(
            "rtol and atol are both zero: an adaptive method needs a positive tolerance".into(),
        ));
    }
    if rtol > 0.0 && rtol < 1e-14 {
        return Err(SolveError::BadInput(format!(
            "rtol = {rtol:e} is below 1e-14, about 50 machine epsilons: it cannot be met in double precision"
        )));
    }
    for (name, v) in [("first_step", opts.first_step), ("max_step", opts.max_step)] {
        if let Some(v) = v {
            if !(v.is_finite() && v > 0.0) {
                return Err(SolveError::BadInput(format!(
                    "{name} = {v} must be positive and finite"
                )));
            }
        }
    }
    Ok(())
}

fn check_step(step: f64) -> Result<(), SolveError> {
    if step.is_finite() && step > 0.0 {
        Ok(())
    } else {
        Err(SolveError::BadInput(format!(
            "the step {step} is not positive"
        )))
    }
}

/// Solves `dy/dt = f(t, y)`, `y(t0) = y0`, up to `t1`.
pub fn solve(
    sys: &mut dyn System,
    t0: f64,
    t1: f64,
    y0: &[f64],
    opts: &Options,
) -> Result<Solution, SolveError> {
    let n = sys.dim();
    if y0.len() != n {
        return Err(SolveError::BadInput(format!(
            "the initial state has {} values, the system {n}",
            y0.len()
        )));
    }
    if !(t0.is_finite() && t1.is_finite()) || t1 < t0 {
        return Err(SolveError::BadInput(format!(
            "the time span {t0} to {t1} is not an increasing finite interval"
        )));
    }
    if y0.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::BadInput(
            "the initial state contains NaN or infinity".to_string(),
        ));
    }
    let mut prev = t0;
    for &o in &opts.outputs {
        if !(o.is_finite() && o >= prev && o <= t1) {
            return Err(SolveError::BadInput(format!(
                "the output time {o} is not finite, in increasing order, and between {t0} and {t1}"
            )));
        }
        prev = o;
    }
    let second_order = match sys.structure() {
        Structure::General => None,
        Structure::SecondOrder { n: half } => {
            if 2 * half != n {
                return Err(SolveError::BadInput(format!(
                    "the system declares a second-order structure with {half} coordinates, so its state should have {} values, not {n}",
                    2 * half
                )));
            }
            Some(half)
        }
    };
    let mut run = Run::new(sys, opts, t0, t1, y0)?;

    let (method, t, y, explanation) = match opts.method {
        Method::Auto => {
            let c = auto::auto(&mut run, second_order, y0)?;
            (c.method, c.t, c.y, c.explanation)
        }
        Method::Rk4 { step } => {
            check_step(step)?;
            let y = drive(&mut run, &mut Rk4::new(y0), step)?;
            let e = format!(
                "classical RK4 with a fixed step of {step} s took {} steps",
                run.stats.steps
            );
            (opts.method, t1, y, vec![e])
        }
        Method::DormandPrince => {
            check_tolerances(opts)?;
            let y = match dopri::dopri5(&mut run, t0, y0.to_vec(), None, false)? {
                dopri::Outcome::Done { y, .. } => y,
                dopri::Outcome::Stiff(_) => unreachable!("stiffness detection is off"),
            };
            let e = format!(
                "Dormand-Prince 5(4) with adaptive steps for rtol = {} and atol = {} took {} steps ({} rejected)",
                run::num(opts.rtol),
                run::num(opts.atol),
                run.stats.steps,
                run.stats.rejected
            );
            (opts.method, t1, y, vec![e])
        }
        Method::Rodas3 => {
            check_tolerances(opts)?;
            let (_, y) = rosenbrock::rodas3(&mut run, t0, y0.to_vec(), None)?;
            let e = format!(
                "Rodas3 (L-stable Rosenbrock, finite-difference Jacobian) with adaptive steps for rtol = {} and atol = {} took {} steps ({} rejected)",
                run::num(opts.rtol),
                run::num(opts.atol),
                run.stats.steps,
                run.stats.rejected
            );
            (opts.method, t1, y, vec![e])
        }
        Method::VelocityVerlet { step } | Method::Yoshida4 { step } => {
            check_step(step)?;
            let kind = if matches!(opts.method, Method::Yoshida4 { .. }) {
                SymKind::Yoshida4
            } else {
                SymKind::Verlet
            };
            let Some(half) = second_order else {
                return Err(SolveError::Unsupported(format!(
                    "{} needs a second-order system (q'' = a(t, q)), but this system does not declare that structure",
                    opts.method.name()
                )));
            };
            verify_second_order(&mut run, half, y0)?;
            let y = drive(&mut run, &mut Symplectic::new(kind, half, y0), step)?;
            let e = format!(
                "{} with a fixed step of {step} s took {} steps",
                opts.method.name(),
                run.stats.steps
            );
            (opts.method, t1, y, vec![e])
        }
    };
    Ok(finish(run, method, t, y, explanation))
}

/// Assembles the solution and appends the drift report.
fn finish(
    run: Run<'_>,
    method: Method,
    t: f64,
    y: Vec<f64>,
    mut explanation: Vec<String>,
) -> Solution {
    for (c, d) in run.opts.conserved.iter().zip(&run.drift) {
        explanation.push(match c.max_drift {
            Some(max) => format!(
                "{} drifted by at most {d:.1e} (relative), within the allowed {max:.1e}",
                c.name
            ),
            None => format!("{} drifted by at most {d:.1e} (relative)", c.name),
        });
    }
    Solution {
        t,
        y,
        samples: run.samples,
        stats: run.stats,
        method,
        explanation,
        drift: run.drift,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `y' = -y`.
    struct Decay;
    impl System for Decay {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
            d[0] = -y[0];
            Ok(())
        }
    }

    #[test]
    fn rk4_solves_exponential_decay() {
        let opts = Options {
            method: Method::Rk4 { step: 0.01 },
            outputs: vec![0.5],
            ..Options::default()
        };
        let s = solve(&mut Decay, 0.0, 1.0, &[1.0], &opts).unwrap();
        assert!((s.y[0] - (-1.0f64).exp()).abs() < 1e-9, "{}", s.y[0]);
        assert_eq!(s.t, 1.0);
        assert_eq!(s.samples.len(), 1);
        assert!((s.samples[0].1[0] - (-0.5f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn bad_inputs_are_errors() {
        let opts = Options::default();
        assert!(matches!(
            solve(&mut Decay, 0.0, 1.0, &[1.0, 2.0], &opts),
            Err(SolveError::BadInput(_))
        ));
        assert!(matches!(
            solve(&mut Decay, 1.0, 0.0, &[1.0], &opts),
            Err(SolveError::BadInput(_))
        ));
    }

    #[test]
    fn number_formatting_for_explanations() {
        use run::num;
        assert_eq!(num(0.0), "0");
        assert_eq!(num(1.0), "1");
        assert_eq!(num(0.5), "0.5");
        assert_eq!(num(12.3456), "12.3");
        assert_eq!(num(1234.5), "1234");
        assert_eq!(num(1e-9), "1.00e-9");
        assert_eq!(num(2.5e7), "2.50e7");
    }
}
