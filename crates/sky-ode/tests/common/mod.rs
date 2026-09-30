//! Test systems shared by the integration tests.
#![allow(dead_code)]

use sky_ode::{Conserved, Method, Options, Structure, System};

/// `y' = -y`, exact `exp(-t)`.
pub struct Decay;
impl System for Decay {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = -y[0];
        Ok(())
    }
}

/// `q'' = -q`, state `[q, v]`; exact `q = cos t, v = -sin t` from `[1, 0]`.
pub struct Oscillator;
impl System for Oscillator {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = y[1];
        d[1] = -y[0];
        Ok(())
    }
    fn structure(&self) -> Structure {
        Structure::SecondOrder { n: 1 }
    }
}

/// The same oscillator without declaring the second-order structure.
pub struct PlainOscillator;
impl System for PlainOscillator {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = y[1];
        d[1] = -y[0];
        Ok(())
    }
}

/// The planar Kepler problem with `GM = 1`: state `[x, y, vx, vy]`.
pub struct Kepler;
impl System for Kepler {
    fn dim(&self) -> usize {
        4
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        let r2 = y[0] * y[0] + y[1] * y[1];
        let r3 = r2 * r2.sqrt();
        d[0] = y[2];
        d[1] = y[3];
        d[2] = -y[0] / r3;
        d[3] = -y[1] / r3;
        Ok(())
    }
    fn structure(&self) -> Structure {
        Structure::SecondOrder { n: 2 }
    }
}

/// Kepler's energy per unit mass.
pub fn kepler_energy(y: &[f64]) -> f64 {
    0.5 * (y[2] * y[2] + y[3] * y[3]) - 1.0 / (y[0] * y[0] + y[1] * y[1]).sqrt()
}

/// Kepler's angular momentum per unit mass.
pub fn kepler_angular_momentum(y: &[f64]) -> f64 {
    y[0] * y[3] - y[1] * y[2]
}

/// Semi-major axis 1 (so the period is `2 pi`), eccentricity `e`, starting
/// at pericentre.
pub fn kepler_start(e: f64) -> [f64; 4] {
    [1.0 - e, 0.0, 0.0, ((1.0 + e) / (1.0 - e)).sqrt()]
}

pub fn energy_conserved() -> Conserved {
    Conserved {
        name: "energy".to_string(),
        value: Box::new(|_, y| kepler_energy(y)),
        max_drift: None,
    }
}

/// Robertson's chemical kinetics problem, a classic stiff test.
pub struct Robertson;
impl System for Robertson {
    fn dim(&self) -> usize {
        3
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        d[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        d[2] = 3e7 * y[1] * y[1];
        Ok(())
    }
}

/// Van der Pol's oscillator `y'' = mu (1 - y^2) y' - y`.
pub struct VanDerPol {
    pub mu: f64,
}
impl System for VanDerPol {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = y[1];
        d[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        Ok(())
    }
}

/// A system defined by a closure.
pub struct Closure<F>(pub usize, pub F);
impl<F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), String>> System for Closure<F> {
    fn dim(&self) -> usize {
        self.0
    }
    fn rhs(&mut self, t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        (self.1)(t, y, d)
    }
}

pub fn opts(method: Method, rtol: f64, atol: f64) -> Options {
    Options {
        rtol,
        atol,
        method,
        ..Options::default()
    }
}

pub fn dopri(rtol: f64, atol: f64) -> Options {
    opts(Method::DormandPrince, rtol, atol)
}
