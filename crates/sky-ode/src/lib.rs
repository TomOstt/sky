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
//!   check ("the system is second order and energy is to be conserved, so a
//!   symplectic velocity-Verlet integrator ...").
//! * Every value stays finite; a non-finite state, a failing right-hand
//!   side, or exceeding [`Options::max_steps`] is a [`SolveError`], never
//!   NaN or infinity (Sky's rule for numbers).
//!
//! v0 ships one method, fixed-step classical RK4, so the language side can
//! be built and tested end to end; the adaptive, symplectic and stiff
//! methods and automatic selection fill in behind this same API.

#![forbid(unsafe_code)]

use std::fmt;

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
    /// Classical fourth-order Runge-Kutta with a fixed step.
    Rk4 { step: f64 },
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
    /// `None` to only report it.
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
    /// Relative tolerance per step for adaptive methods.
    pub rtol: f64,
    /// Absolute tolerance per step for adaptive methods.
    pub atol: f64,
    pub method: Method,
    pub conserved: Vec<Conserved>,
    /// Times (between `t0` and `t1`, increasing) at which to record the
    /// state in [`Solution::samples`].
    pub outputs: Vec<f64>,
    /// The most steps before [`SolveError::TooManySteps`].
    pub max_steps: usize,
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
        }
    }
}

/// Counters for a solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    pub steps: usize,
    pub rejected: usize,
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
    /// The method used (never `Auto`).
    pub method: Method,
    /// Why this method, and what was observed (step counts, the drift of
    /// each conserved quantity), one sentence per entry.
    pub explanation: Vec<String>,
    /// The relative drift of each conserved quantity, in the order asked.
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
        }
    }
}

impl std::error::Error for SolveError {}

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
    match opts.method {
        Method::Rk4 { step } => rk4(sys, t0, t1, y0, step, opts),
        Method::Auto => {
            // v0: automatic selection is not built yet. A fixed RK4 step
            // of 1/1000 of the span keeps the language side testable.
            let step = if t1 > t0 { (t1 - t0) / 1000.0 } else { 1.0 };
            let mut s = rk4(sys, t0, t1, y0, step, opts)?;
            s.explanation.insert(
                0,
                "automatic method selection is not available yet, so classical RK4 with 1000 fixed steps was used".to_string(),
            );
            Ok(s)
        }
    }
}

fn rk4(
    sys: &mut dyn System,
    t0: f64,
    t1: f64,
    y0: &[f64],
    step: f64,
    opts: &Options,
) -> Result<Solution, SolveError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(SolveError::BadInput(format!(
            "the step {step} is not positive"
        )));
    }
    let n = y0.len();
    let mut stats = Stats::default();
    let eval = |sys: &mut dyn System, t: f64, y: &[f64], out: &mut [f64], stats: &mut Stats| {
        stats.rhs_evals += 1;
        sys.rhs(t, y, out)
            .map_err(|message| SolveError::Rhs { t, message })
    };
    let initial: Vec<f64> = opts.conserved.iter().map(|c| (c.value)(t0, y0)).collect();
    let mut drift = vec![0.0_f64; opts.conserved.len()];
    let (mut t, mut y) = (t0, y0.to_vec());
    let mut samples = Vec::new();
    let mut next_out = opts.outputs.iter().copied().peekable();
    let (mut k1, mut k2, mut k3, mut k4, mut tmp) = (
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
    );
    while t < t1 {
        if stats.steps >= opts.max_steps {
            return Err(SolveError::TooManySteps {
                t,
                steps: stats.steps,
            });
        }
        // Land exactly on the next output time or on t1.
        let target = next_out.peek().copied().filter(|&o| o < t1).unwrap_or(t1);
        let h = step.min(target - t);
        eval(sys, t, &y, &mut k1, &mut stats)?;
        for i in 0..n {
            tmp[i] = y[i] + 0.5 * h * k1[i];
        }
        eval(sys, t + 0.5 * h, &tmp, &mut k2, &mut stats)?;
        for i in 0..n {
            tmp[i] = y[i] + 0.5 * h * k2[i];
        }
        eval(sys, t + 0.5 * h, &tmp, &mut k3, &mut stats)?;
        for i in 0..n {
            tmp[i] = y[i] + h * k3[i];
        }
        eval(sys, t + h, &tmp, &mut k4, &mut stats)?;
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t = if h == target - t { target } else { t + h };
        stats.steps += 1;
        if y.iter().any(|v| !v.is_finite()) {
            return Err(SolveError::NotFinite { t });
        }
        for (i, c) in opts.conserved.iter().enumerate() {
            let q0 = initial[i];
            let d = ((c.value)(t, &y) - q0).abs() / q0.abs().max(f64::MIN_POSITIVE);
            drift[i] = drift[i].max(d);
            if let Some(max) = c.max_drift.filter(|&m| d > m) {
                return Err(SolveError::Drift {
                    name: c.name.clone(),
                    drift: d,
                    max,
                    t,
                });
            }
        }
        while next_out.peek().is_some_and(|&o| o <= t) {
            next_out.next();
            samples.push((t, y.clone()));
        }
    }
    let mut explanation = vec![format!(
        "classical RK4 with a fixed step of {step} s took {} steps",
        stats.steps
    )];
    for (c, d) in opts.conserved.iter().zip(&drift) {
        explanation.push(format!("{} drifted by at most {d:.1e} (relative)", c.name));
    }
    Ok(Solution {
        t,
        y,
        samples,
        stats,
        method: Method::Rk4 { step },
        explanation,
        drift,
    })
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
}
