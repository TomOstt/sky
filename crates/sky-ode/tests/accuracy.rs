//! Accuracy against exact solutions: error scaling with the tolerance,
//! convergence orders, dense output, and landing on outputs.

mod common;

use common::*;
use sky_ode::*;

const TOLS: [f64; 4] = [1e-4, 1e-6, 1e-8, 1e-10];

/// Relative error of `exp(-10)`.
fn decay_error(method: Method, rtol: f64) -> (f64, Solution) {
    let s = solve(&mut Decay, 0.0, 10.0, &[1.0], &opts(method, rtol, 1e-16)).unwrap();
    let exact = (-10.0f64).exp();
    ((s.y[0] - exact).abs() / exact, s)
}

fn oscillator_error(method: Method, rtol: f64) -> (f64, Solution) {
    let t = 20.0;
    let s = solve(
        &mut Oscillator,
        0.0,
        t,
        &[1.0, 0.0],
        &opts(method, rtol, 1e-14),
    )
    .unwrap();
    let e = ((s.y[0] - t.cos()).powi(2) + (s.y[1] + t.sin()).powi(2)).sqrt();
    (e, s)
}

/// For an adaptive method the *global* error should follow the tolerance:
/// within a modest factor of `rtol`, and shrinking by about as much as the
/// tolerance does (the tolerance is per step, so a gentle growth of the
/// error/tolerance ratio over 20 periods is normal).
fn check_scaling(name: &str, errs: &[(f64, f64)]) {
    for &(rtol, e) in errs {
        assert!(
            e < 50.0 * rtol,
            "{name}: error {e:e} is not within 50x of rtol {rtol:e}"
        );
    }
    for w in errs.windows(2) {
        let (r0, e0) = w[0];
        let (r1, e1) = w[1];
        // The tolerance shrinks 100x between neighbours; the error must
        // shrink by at least 10x (it typically shrinks ~100x).
        assert!(
            e1 < e0 / 10.0,
            "{name}: tightening rtol {r0:e} -> {r1:e} only moved the error {e0:e} -> {e1:e}"
        );
    }
}

#[test]
fn exponential_decay_error_scales_with_rtol() {
    for (name, m) in [
        ("dopri", Method::DormandPrince),
        ("rodas3", Method::Rodas3),
        ("auto", Method::Auto),
    ] {
        let errs: Vec<(f64, f64)> = TOLS.iter().map(|&r| (r, decay_error(m, r).0)).collect();
        check_scaling(name, &errs);
    }
}

#[test]
fn harmonic_oscillator_error_scales_with_rtol() {
    for (name, m) in [("dopri", Method::DormandPrince), ("rodas3", Method::Rodas3)] {
        let errs: Vec<(f64, f64)> = TOLS
            .iter()
            .map(|&r| (r, oscillator_error(m, r).0))
            .collect();
        check_scaling(name, &errs);
    }
}

#[test]
fn tighter_tolerance_costs_more_steps_at_the_expected_rate() {
    // Order 5 (error ~ h^5): 100x tighter tolerance -> about 100^(1/5) =
    // 2.5x the steps. Order 2 estimator of Rodas3: 100^(1/3) = 4.6x.
    let steps = |m, r| oscillator_error(m, r).1.stats.steps as f64;
    let d = steps(Method::DormandPrince, 1e-10) / steps(Method::DormandPrince, 1e-8);
    assert!((1.8..3.5).contains(&d), "dopri step ratio {d}");
    let r = steps(Method::Rodas3, 1e-10) / steps(Method::Rodas3, 1e-8);
    assert!((3.5..6.0).contains(&r), "rodas3 step ratio {r}");
}

/// A nonlinear problem with a closed form: logistic growth
/// `y' = y (1 - y)`, `y = 1 / (1 + (1/y0 - 1) e^-t)`.
struct Logistic;
impl System for Logistic {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, _t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        d[0] = y[0] * (1.0 - y[0]);
        Ok(())
    }
}

fn logistic_exact(t: f64) -> f64 {
    1.0 / (1.0 + (1.0 / 0.1 - 1.0) * (-t).exp())
}

/// Forces an adaptive method to take (almost) uniform steps of `h`: an
/// enormous tolerance accepts every step and `max_step` caps it.
fn uniform_step_error(method: Method, h: f64) -> f64 {
    let o = Options {
        rtol: 1e3,
        atol: 1e3,
        method,
        first_step: Some(h),
        max_step: Some(h),
        ..Options::default()
    };
    let s = solve(&mut Logistic, 0.0, 4.0, &[0.1], &o).unwrap();
    (s.y[0] - logistic_exact(4.0)).abs()
}

fn observed_order(e_coarse: f64, e_fine: f64) -> f64 {
    (e_coarse / e_fine).log2()
}

#[test]
fn convergence_orders_of_the_adaptive_methods() {
    let p = observed_order(
        uniform_step_error(Method::DormandPrince, 0.1),
        uniform_step_error(Method::DormandPrince, 0.05),
    );
    assert!((4.6..5.6).contains(&p), "Dormand-Prince order {p}");
    // Rodas3 is order 3; the forward-difference Jacobian (relative error
    // ~1e-8) and roundoff blur the measured order below h ~ 3e-3, so
    // measure in the asymptotic range above that, where it is 2.8-2.9. An
    // order-2 method would give 2.0 there.
    let p = observed_order(
        uniform_step_error(Method::Rodas3, 0.0125),
        uniform_step_error(Method::Rodas3, 0.00625),
    );
    assert!((2.6..3.3).contains(&p), "Rodas3 order {p}");
}

#[test]
fn convergence_orders_of_the_fixed_step_methods() {
    let fixed = |m: Method| {
        let s = solve(
            &mut Oscillator,
            0.0,
            10.0,
            &[1.0, 0.0],
            &Options {
                method: m,
                ..Options::default()
            },
        )
        .unwrap();
        ((s.y[0] - 10f64.cos()).powi(2) + (s.y[1] + 10f64.sin()).powi(2)).sqrt()
    };
    let order = |mk: fn(f64) -> Method, h: f64| observed_order(fixed(mk(h)), fixed(mk(h / 2.0)));
    let p = order(|step| Method::Rk4 { step }, 0.1);
    assert!((3.8..4.3).contains(&p), "RK4 order {p}");
    let p = order(|step| Method::VelocityVerlet { step }, 0.02);
    assert!((1.8..2.2).contains(&p), "velocity Verlet order {p}");
    let p = order(|step| Method::Yoshida4 { step }, 0.1);
    assert!((3.8..4.3).contains(&p), "Yoshida order {p}");
}

#[test]
fn dense_output_is_accurate_and_does_not_change_the_integration() {
    let outputs: Vec<f64> = (0..=200).map(|i| i as f64 * 0.1).collect();
    let mut with = dopri(1e-8, 1e-12);
    with.outputs = outputs.clone();
    let a = solve(&mut Oscillator, 0.0, 20.0, &[1.0, 0.0], &with).unwrap();
    let b = solve(&mut Oscillator, 0.0, 20.0, &[1.0, 0.0], &dopri(1e-8, 1e-12)).unwrap();

    // Asking for outputs did not force a single extra step, nor change y.
    assert_eq!(a.stats, b.stats);
    assert_eq!(a.y, b.y);
    assert_eq!(a.samples.len(), outputs.len());
    // The interpolant is as accurate as the tolerance: the steps here are
    // ~0.5 s long, with 5 outputs inside each.
    let mut worst: f64 = 0.0;
    for ((t, y), o) in a.samples.iter().zip(&outputs) {
        assert_eq!(t, o);
        worst = worst
            .max((y[0] - t.cos()).abs())
            .max((y[1] + t.sin()).abs());
    }
    assert!(worst < 1e-6, "dense output error {worst:e}");
    assert!(
        a.stats.steps < 400,
        "{} steps: the outputs must not force steps",
        a.stats.steps
    );
}

#[test]
fn outputs_at_the_ends_and_repeated_outputs() {
    for method in [
        Method::DormandPrince,
        Method::Rodas3,
        Method::Rk4 { step: 0.1 },
        Method::Yoshida4 { step: 0.1 },
        Method::VelocityVerlet { step: 0.1 },
        Method::Auto,
    ] {
        let o = Options {
            method,
            outputs: vec![0.0, 0.5, 0.5, 1.0],
            rtol: 1e-8,
            atol: 1e-10,
            ..Options::default()
        };
        let s = solve(&mut Oscillator, 0.0, 1.0, &[1.0, 0.0], &o).unwrap();
        assert_eq!(s.samples.len(), 4, "{method:?}");
        assert_eq!(s.samples[0].1, vec![1.0, 0.0], "{method:?}: output at t0");
        for (t, y) in &s.samples {
            let tol = if matches!(method, Method::VelocityVerlet { .. }) {
                1e-2
            } else {
                1e-5
            };
            assert!((y[0] - t.cos()).abs() < tol, "{method:?} at {t}: {y:?}");
        }
        assert_eq!(
            s.samples[3].1, s.y,
            "{method:?}: output at t1 is the result"
        );
        assert_eq!(s.t, 1.0);
    }
}

#[test]
fn fixed_step_methods_land_exactly_on_outputs_and_t1() {
    // 0.3 does not divide 1.0: the step is shortened so that a whole
    // number of steps lands on every output and on t1 (no sliver step).
    let o = Options {
        method: Method::Rk4 { step: 0.3 },
        outputs: vec![0.5],
        ..Options::default()
    };
    let s = solve(&mut Decay, 0.0, 1.0, &[1.0], &o).unwrap();
    assert_eq!(s.t, 1.0);
    assert_eq!(s.samples[0].0, 0.5);
    // [0, 0.5] -> 2 steps, [0.5, 1] -> 2 steps.
    assert_eq!(s.stats.steps, 4);
}

#[test]
fn non_autonomous_systems_are_integrated_correctly() {
    // y' = cos t, y(0) = 0 -> sin t: the right-hand side depends on t alone,
    // which exercises the time-derivative term of the Rosenbrock method.
    let mut sys = Closure(1, |t: f64, _y: &[f64], d: &mut [f64]| {
        d[0] = t.cos() + 0.0 * t;
        Ok(())
    });
    for m in [Method::DormandPrince, Method::Rodas3, Method::Auto] {
        let s = solve(&mut sys, 0.0, 10.0, &[0.0], &opts(m, 1e-9, 1e-12)).unwrap();
        assert!((s.y[0] - 10f64.sin()).abs() < 1e-7, "{m:?}: {}", s.y[0]);
    }
    // A stiff non-autonomous problem: y' = -1000 (y - cos t) - sin t has
    // the exact solution cos t for y(0) = 1.
    let mut stiff = Closure(1, |t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -1000.0 * (y[0] - t.cos()) - t.sin();
        Ok(())
    });
    let s = solve(
        &mut stiff,
        0.0,
        3.0,
        &[1.0],
        &opts(Method::Rodas3, 1e-8, 1e-10),
    )
    .unwrap();
    assert!((s.y[0] - 3f64.cos()).abs() < 1e-6, "{}", s.y[0]);
}

#[test]
fn a_zero_length_interval_is_a_valid_solve() {
    let o = Options {
        outputs: vec![2.0],
        ..Options::default()
    };
    let s = solve(&mut Decay, 2.0, 2.0, &[3.0], &o).unwrap();
    assert_eq!(s.y, vec![3.0]);
    assert_eq!(s.samples, vec![(2.0, vec![3.0])]);
    assert_eq!(s.stats.steps, 0);
    for m in [
        Method::DormandPrince,
        Method::Rodas3,
        Method::Rk4 { step: 1.0 },
    ] {
        let o = Options {
            method: m,
            outputs: vec![2.0],
            ..Options::default()
        };
        let s = solve(&mut Decay, 2.0, 2.0, &[3.0], &o).unwrap();
        assert_eq!((s.t, s.y[0], s.samples.len()), (2.0, 3.0, 1), "{m:?}");
    }
}
