//! Fixed-step methods: classical RK4 and the two symplectic integrators
//! (velocity Verlet, Yoshida's 4th-order composition).
//!
//! # Why fixed steps for the symplectic methods
//!
//! A symplectic integrator does not conserve the energy `H`; it conserves a
//! *modified* Hamiltonian `H~ = H + h^p H_p + ...` exactly (up to
//! exponentially small terms), which is why the energy error stays bounded
//! over exponentially long times (Hairer, Lubich & Wanner, *Geometric
//! Numerical Integration*, 2nd ed. 2006, IX.8 "Backward error analysis").
//! That statement needs the *same* step for every step: naive step-size
//! adaptation changes `H~` at every step and destroys the bound (ibid.,
//! VIII.1), leaving the secular drift of an ordinary method. So these
//! methods take a fixed step, and [`Method::Auto`](crate::Method::Auto)
//! chooses that step from the tolerance in advance (see `auto.rs`).
//!
//! # Landing on outputs
//!
//! Every fixed-step method here steps *onto* output times and `t1`: the
//! interval between two consecutive landing points is cut into the fewest
//! equal steps not larger than the requested step. The step therefore
//! varies slightly from one interval to the next but is constant inside one
//! (and the modified energy differs by only `O(h^p)` between intervals), and
//! time is computed as `start + j * (length / count)` rather than by
//! accumulating `h`, so there is no drift in `t` over a million steps.

use crate::run::{all_finite, Run};
use crate::SolveError;

/// A one-step method with internal state, driven on a fixed grid.
pub(crate) trait Stepper {
    /// Advances the state from `t` by `h`.
    fn advance(&mut self, run: &mut Run<'_>, t: f64, h: f64) -> Result<(), SolveError>;
    fn state(&self) -> &[f64];
}

/// Integrates from `(t0, y0)` to `run.t1` with steps of at most `step`,
/// landing on every output time. Returns the final state.
pub(crate) fn drive<S: Stepper>(
    run: &mut Run<'_>,
    stepper: &mut S,
    step: f64,
) -> Result<Vec<f64>, SolveError> {
    let t1 = run.t1;
    let mut t = run.t0;
    run.flush_outputs(t, stepper.state());
    while t < t1 {
        let b = run.next_output().filter(|&o| o < t1).unwrap_or(t1);
        let len = b - t;
        // The fewest equal steps of at most `step` (the 1e-9 forgives the
        // rounding in `len / step` when `step` divides `len`).
        let count = ((len / step) * (1.0 - 1e-9)).ceil().max(1.0);
        if count.is_nan() || count >= 9.0e15 {
            // More steps than a counter can hold: certainly over any budget.
            return Err(SolveError::TooManySteps {
                t,
                steps: run.stats.steps,
            });
        }
        let count = count as u64;
        let dt = len / count as f64;
        for j in 0..count {
            let ts = t + j as f64 * dt;
            run.budget(ts)?;
            stepper.advance(run, ts, dt)?;
            let tn = if j + 1 == count {
                b
            } else {
                t + (j + 1) as f64 * dt
            };
            run.stats.steps += 1;
            if !all_finite(stepper.state()) {
                return Err(SolveError::NotFinite { t: tn });
            }
            run.monitor(tn, stepper.state())?;
        }
        t = b;
        run.flush_outputs(t, stepper.state());
    }
    Ok(stepper.state().to_vec())
}

// ---------------------------------------------------------------------------
// Classical RK4
// ---------------------------------------------------------------------------

/// Kutta's classical fourth-order method (Kutta 1901; Hairer, Nørsett &
/// Wanner I, II.1).
pub(crate) struct Rk4 {
    y: Vec<f64>,
    k: [Vec<f64>; 4],
    tmp: Vec<f64>,
}

impl Rk4 {
    pub(crate) fn new(y0: &[f64]) -> Self {
        let n = y0.len();
        Rk4 {
            y: y0.to_vec(),
            k: [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            tmp: vec![0.0; n],
        }
    }
}

impl Stepper for Rk4 {
    fn advance(&mut self, run: &mut Run<'_>, t: f64, h: f64) -> Result<(), SolveError> {
        let n = self.y.len();
        let [k1, k2, k3, k4] = &mut self.k;
        run.rhs(t, &self.y, k1)?;
        for i in 0..n {
            self.tmp[i] = self.y[i] + 0.5 * h * k1[i];
        }
        run.rhs(t + 0.5 * h, &self.tmp, k2)?;
        for i in 0..n {
            self.tmp[i] = self.y[i] + 0.5 * h * k2[i];
        }
        run.rhs(t + 0.5 * h, &self.tmp, k3)?;
        for i in 0..n {
            self.tmp[i] = self.y[i] + h * k3[i];
        }
        run.rhs(t + h, &self.tmp, k4)?;
        for i in 0..n {
            self.y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        Ok(())
    }
    fn state(&self) -> &[f64] {
        &self.y
    }
}

// ---------------------------------------------------------------------------
// Symplectic methods
// ---------------------------------------------------------------------------

/// Which symplectic scheme.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SymKind {
    /// Velocity Verlet (kick-drift-kick leapfrog), order 2, one force
    /// evaluation per step.
    Verlet,
    /// Yoshida's triple-jump composition of Verlet, order 4, three force
    /// evaluations per step.
    Yoshida4,
}

impl SymKind {
    pub(crate) fn order(self) -> i32 {
        match self {
            SymKind::Verlet => 2,
            SymKind::Yoshida4 => 4,
        }
    }
}

/// Yoshida (1990), "Construction of higher order symplectic integrators",
/// Phys. Lett. A 150, 262-268: composing a symmetric second-order method
/// `S(h)` as `S(w1 h) S(w0 h) S(w1 h)` with `w1 = 1/(2 - 2^(1/3))`,
/// `w0 = -2^(1/3) w1` cancels the `h^3` error term and is 4th order. The
/// middle step is negative (`w0 ~ -1.70`), which is why the error constant
/// is large and Yoshida only pays off at tight tolerances.
fn yoshida_weights() -> (f64, f64) {
    let cbrt2 = 2.0_f64.cbrt();
    let w1 = 1.0 / (2.0 - cbrt2);
    (w1, -cbrt2 * w1)
}

/// A symplectic stepper for `q'' = a(t, q)` on the state `[q, v]`.
pub(crate) struct Symplectic {
    kind: SymKind,
    n: usize,
    /// `[q, v]`.
    y: Vec<f64>,
    /// The acceleration at the current `(t, q)` (used by Verlet only).
    a: Vec<f64>,
    ybuf: Vec<f64>,
    dbuf: Vec<f64>,
    /// `true` once `a` holds the acceleration at the current state.
    primed: bool,
}

impl Symplectic {
    pub(crate) fn new(kind: SymKind, n: usize, y0: &[f64]) -> Self {
        Symplectic {
            kind,
            n,
            y: y0.to_vec(),
            a: vec![0.0; n],
            ybuf: vec![0.0; 2 * n],
            dbuf: vec![0.0; 2 * n],
            primed: false,
        }
    }

    pub(crate) fn set_state(&mut self, y: &[f64]) {
        self.y.copy_from_slice(y);
        self.primed = false;
    }

    /// Acceleration at `(t, q)` from the current state into `self.a`. The
    /// contract says it does not depend on `v`; the current `v` is passed
    /// only because the right-hand side takes a full state.
    fn accel(&mut self, run: &mut Run<'_>, t: f64) -> Result<(), SolveError> {
        let n = self.n;
        self.ybuf.copy_from_slice(&self.y);
        run.rhs(t, &self.ybuf, &mut self.dbuf)?;
        self.a.copy_from_slice(&self.dbuf[n..]);
        // A non-finite `q'` half would otherwise go unnoticed, since only
        // the acceleration is used. Poison the step instead, so the caller's
        // usual finiteness check reports it (or the probe shrinks its step).
        if !all_finite(&self.dbuf[..n]) {
            self.a.fill(f64::NAN);
        }
        Ok(())
    }
}

/// Checks, at the initial state, the two things a
/// [`Structure::SecondOrder`](crate::Structure) declaration promises and a
/// symplectic method silently relies on: `q' = v` in the first half of the
/// right-hand side, and an acceleration that does not depend on `v`. A wrong
/// declaration would otherwise give a plausible but wrong orbit, the worst
/// kind of failure, so this costs two right-hand-side calls to rule it out.
pub(crate) fn verify_second_order(
    run: &mut Run<'_>,
    n: usize,
    y0: &[f64],
) -> Result<(), SolveError> {
    let t0 = run.t0;
    let mut f = vec![0.0; 2 * n];
    run.rhs(t0, y0, &mut f)?;
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (a.abs() + b.abs());
    for i in 0..n {
        if all_finite(&f) && !close(f[i], y0[n + i]) {
            return Err(SolveError::BadInput(format!(
                "the system declares a second-order structure, but its right-hand side gives q'[{i}] = {} where the velocity is {}: the first half of the derivative must be the velocity",
                f[i],
                y0[n + i]
            )));
        }
    }
    let mut bumped = y0.to_vec();
    for v in &mut bumped[n..] {
        *v = *v * 1.5 + 0.25;
    }
    let mut g = vec![0.0; 2 * n];
    run.rhs(t0, &bumped, &mut g)?;
    for i in n..2 * n {
        if all_finite(&f) && all_finite(&g) && !close(f[i], g[i]) {
            return Err(SolveError::BadInput(
                "the system declares a second-order structure, but its acceleration depends on the velocity: symplectic methods need q'' = a(t, q); declare the system without structure to use a general method".to_string(),
            ));
        }
    }
    Ok(())
}

impl Stepper for Symplectic {
    fn advance(&mut self, run: &mut Run<'_>, t: f64, h: f64) -> Result<(), SolveError> {
        let n = self.n;
        match self.kind {
            SymKind::Verlet => {
                if !self.primed {
                    self.accel(run, t)?;
                }
                for i in 0..n {
                    self.y[n + i] += 0.5 * h * self.a[i];
                }
                for i in 0..n {
                    self.y[i] += h * self.y[n + i];
                }
                self.accel(run, t + h)?;
                self.primed = true;
                for i in 0..n {
                    self.y[n + i] += 0.5 * h * self.a[i];
                }
            }
            SymKind::Yoshida4 => {
                // Drift-kick-drift form of S(w1 h) S(w0 h) S(w1 h): four
                // drifts, three kicks, and the adjacent half drifts merged.
                let (w1, w0) = yoshida_weights();
                let c = [0.5 * w1, 0.5 * (w0 + w1), 0.5 * (w0 + w1), 0.5 * w1];
                let d = [w1, w0, w1];
                let mut tau = 0.0;
                for s in 0..3 {
                    for i in 0..n {
                        self.y[i] += c[s] * h * self.y[n + i];
                    }
                    tau += c[s] * h;
                    self.accel(run, t + tau)?;
                    for i in 0..n {
                        self.y[n + i] += d[s] * h * self.a[i];
                    }
                }
                for i in 0..n {
                    self.y[i] += c[3] * h * self.y[n + i];
                }
            }
        }
        Ok(())
    }
    fn state(&self) -> &[f64] {
        &self.y
    }
}

impl Symplectic {
    /// One step of size `h` from the given state (for the step-size probe):
    /// leaves the stepper in the new state.
    pub(crate) fn step_from(
        &mut self,
        run: &mut Run<'_>,
        t: f64,
        h: f64,
        y: &[f64],
    ) -> Result<(), SolveError> {
        self.set_state(y);
        self.advance(run, t, h)
    }
}
