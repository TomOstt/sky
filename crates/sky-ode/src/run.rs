//! The state shared by every method during one solve.
//!
//! A [`Run`] owns the bookkeeping that must survive a change of method (the
//! statistics, the samples already recorded, the drift seen so far, how many
//! outputs are still pending), so `Auto` can start with Dormand-Prince and
//! hand over to the stiff method half way without losing any of it. The
//! integrators themselves only advance `(t, y)`.

use crate::{Options, SolveError, Stats, System};

pub(crate) struct Run<'a> {
    pub(crate) sys: &'a mut dyn System,
    pub(crate) opts: &'a Options,
    pub(crate) t0: f64,
    pub(crate) t1: f64,
    pub(crate) stats: Stats,
    pub(crate) samples: Vec<(f64, Vec<f64>)>,
    /// The largest relative drift seen so far, per conserved quantity.
    pub(crate) drift: Vec<f64>,
    /// The value of each conserved quantity at `t0`.
    initial: Vec<f64>,
    next_out: usize,
}

impl<'a> Run<'a> {
    pub(crate) fn new(
        sys: &'a mut dyn System,
        opts: &'a Options,
        t0: f64,
        t1: f64,
        y0: &[f64],
    ) -> Result<Self, SolveError> {
        let mut initial = Vec::with_capacity(opts.conserved.len());
        for c in &opts.conserved {
            let q = (c.value)(t0, y0);
            if !q.is_finite() {
                return Err(SolveError::NotFinite { t: t0 });
            }
            initial.push(q);
        }
        Ok(Run {
            sys,
            opts,
            t0,
            t1,
            stats: Stats::default(),
            samples: Vec::new(),
            drift: vec![0.0; opts.conserved.len()],
            initial,
            next_out: 0,
        })
    }

    /// Evaluates the right-hand side, counting it. A user error stops the
    /// solve; a non-finite result is left to the caller (a trial stage of an
    /// adaptive method may recover by shrinking the step).
    pub(crate) fn rhs(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
        self.stats.rhs_evals += 1;
        self.sys
            .rhs(t, y, out)
            .map_err(|message| SolveError::Rhs { t, message })
    }

    /// Fails if `max_steps` steps (accepted or rejected) have been tried.
    pub(crate) fn budget(&self, t: f64) -> Result<(), SolveError> {
        let used = self.stats.steps + self.stats.rejected;
        if used >= self.opts.max_steps {
            Err(SolveError::TooManySteps { t, steps: used })
        } else {
            Ok(())
        }
    }

    /// Updates the drift of every conserved quantity at an accepted state
    /// and fails if one exceeds its `max_drift`. Relative to the initial
    /// value; if that is exactly zero, absolute (there is no scale to
    /// divide by).
    pub(crate) fn monitor(&mut self, t: f64, y: &[f64]) -> Result<(), SolveError> {
        for (i, c) in self.opts.conserved.iter().enumerate() {
            let q = (c.value)(t, y);
            if !q.is_finite() {
                return Err(SolveError::NotFinite { t });
            }
            let q0 = self.initial[i];
            let scale = if q0 != 0.0 { q0.abs() } else { 1.0 };
            let d = (q - q0).abs() / scale;
            if d > self.drift[i] {
                self.drift[i] = d;
            }
            if let Some(max) = c.max_drift.filter(|&m| d > m) {
                return Err(SolveError::Drift {
                    name: c.name.clone(),
                    drift: d,
                    max,
                    t,
                });
            }
        }
        Ok(())
    }

    /// The next requested output time not yet recorded.
    pub(crate) fn next_output(&self) -> Option<f64> {
        self.opts.outputs.get(self.next_out).copied()
    }

    /// Records the state at the next pending output time.
    pub(crate) fn push_output(&mut self, y: Vec<f64>) {
        let o = self.opts.outputs[self.next_out];
        self.next_out += 1;
        self.samples.push((o, y));
    }

    /// Records `y` for every pending output time up to and including `t`
    /// (for methods that step exactly onto their outputs).
    pub(crate) fn flush_outputs(&mut self, t: f64, y: &[f64]) {
        while self.next_output().is_some_and(|o| o <= t) {
            self.push_output(y.to_vec());
        }
    }
}

/// `true` if every entry is finite.
pub(crate) fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Root mean square of `v[i] / (atol + rtol * max(|a[i]|, |b[i]|))`: the
/// error norm of Hairer & Wanner (1996, II.4), where the mixed tolerance
/// scales each component by its own size and `atol` covers components near
/// zero. Zero for an empty system.
pub(crate) fn scaled_rms(v: &[f64], a: &[f64], b: &[f64], rtol: f64, atol: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..v.len() {
        let sc = atol + rtol * a[i].abs().max(b[i].abs());
        let e = v[i] / sc;
        s += e * e;
    }
    (s / v.len() as f64).sqrt()
}

/// `true` if a step `h` at time `t` is too small to matter: it no longer
/// changes `t` in double precision (Hairer's `0.1 |h| <= |t| u` test, with
/// `u` the unit roundoff), or is subnormal-small.
pub(crate) fn step_underflow(t: f64, h: f64) -> bool {
    h.is_nan() || h <= 0.0 || h <= 10.0 * (f64::EPSILON / 2.0) * t.abs() || h < 1e-280
}

/// Hairer, Nørsett & Wanner's starting step (*Solving ODEs I*, II.4, "Starting
/// step size"): from the size of `y` and `f`, take a small explicit Euler
/// step, measure how fast `f` changes, and choose `h` so the leading error
/// term of an order-`order` method is about 1% of the tolerance scale. The
/// Euler probe is not a real step, so a failure there is not an error: fall
/// back to the cautious first guess.
pub(crate) fn initial_step(
    run: &mut Run<'_>,
    t: f64,
    y: &[f64],
    f0: &[f64],
    order: u32,
    hmax: f64,
) -> f64 {
    let (rtol, atol) = (run.opts.rtol, run.opts.atol);
    let n = y.len();
    if n == 0 {
        return hmax;
    }
    let sk: Vec<f64> = y.iter().map(|v| atol + rtol * v.abs()).collect();
    let rms = |v: &[f64]| -> f64 {
        (v.iter()
            .zip(&sk)
            .map(|(a, s)| (a / s) * (a / s))
            .sum::<f64>()
            / n as f64)
            .sqrt()
    };
    let d0 = rms(y);
    let d1 = rms(f0);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    }
    .min(hmax);
    let y1: Vec<f64> = (0..n).map(|i| y[i] + h0 * f0[i]).collect();
    let mut f1 = vec![0.0; n];
    if run.rhs(t + h0, &y1, &mut f1).is_err() || !all_finite(&f1) {
        return h0.min(hmax);
    }
    let diff: Vec<f64> = (0..n).map(|i| f1[i] - f0[i]).collect();
    let d2 = rms(&diff) / h0;
    let der12 = d2.max(d1);
    let h1 = if der12 <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        libm::pow(0.01 / der12, 1.0 / f64::from(order))
    };
    (100.0 * h0).min(h1).min(hmax)
}

/// A number for an explanation: plain if moderate, scientific otherwise,
/// three significant digits.
pub(crate) fn num(x: f64) -> String {
    let a = x.abs();
    if a == 0.0 {
        return "0".to_string();
    }
    if (1e-3..1e6).contains(&a) {
        let digits = (2 - a.log10().floor() as i32).max(0) as usize;
        let s = format!("{x:.digits$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    } else {
        format!("{x:.2e}")
    }
}
