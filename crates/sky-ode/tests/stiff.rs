//! Stiff problems: Robertson's chemistry and the Van der Pol oscillator with
//! mu = 1000. `Auto` must notice the stiffness, switch, explain itself, and
//! finish in a reasonable number of steps with the right answer.
//!
//! Reference values:
//! * Robertson at t = 40 (Hairer & Wanner, *Solving ODEs II*, IV.1) and at
//!   t = 1e11 (the IVP test set of Mazzia & Magherini, "Test set for initial
//!   value problem solvers", Bari 2008).
//! * Van der Pol, mu = 1000, `y(0) = (2, 0)`, at t = 500, 1000, 3000:
//!   computed here with Dormand-Prince at rtol 1e-9 (an *explicit*
//!   method, taking 1.7 million steps, so independent of the stiff solver's
//!   machinery) and Rodas3 at rtol 1e-10, which agree to 10 digits; see the
//!   ignored test `van_der_pol_reference_by_explicit_integration`.

mod common;

use common::*;
use sky_ode::*;

fn mass() -> Conserved {
    Conserved {
        name: "mass".into(),
        value: Box::new(|_, y| y[0] + y[1] + y[2]),
        max_drift: None,
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn robertson(rtol: f64) -> Solution {
    let mut o = opts(Method::Auto, rtol, 1e-14);
    o.conserved = vec![mass()];
    o.outputs = vec![40.0, 1e11];
    solve(&mut Robertson, 0.0, 1e11, &[1.0, 0.0, 0.0], &o).unwrap()
}

#[test]
fn robertson_is_detected_as_stiff_and_solved_correctly() {
    let s = robertson(1e-8);
    assert_eq!(s.method, Method::Rodas3, "{:?}", s.explanation);
    // t = 40.
    let y = &s.samples[0].1;
    assert!(rel(y[0], 0.7158270687) < 1e-6, "{y:?}");
    assert!(rel(y[1], 9.185534764e-6) < 1e-5, "{y:?}");
    assert!(rel(y[2], 0.2841637457) < 1e-6, "{y:?}");
    // t = 1e11, over eleven decades of time.
    assert!(rel(s.y[0], 2.083340149701255e-8) < 1e-3, "{:?}", s.y);
    assert!(rel(s.y[1], 8.333360770334713e-14) < 1e-3, "{:?}", s.y);
    assert!((s.y[2] - 0.999999979166505).abs() < 1e-9, "{:?}", s.y);
    // The mass is conserved by any linear-invariant-preserving method.
    assert!(s.drift[0] < 1e-12, "{:e}", s.drift[0]);
    // A reasonable number of steps for 11 decades (an explicit method
    // would need ~1e9), and a handful of rejections.
    assert!(s.stats.steps < 8000, "{} steps", s.stats.steps);
    assert!(s.stats.rejected < s.stats.steps / 2);
    assert_eq!(s.t, 1e11);
}

#[test]
fn robertson_explanation_says_why_and_when() {
    let s = robertson(1e-6);
    let text = s.explanation.join("\n");
    // Starts with Dormand-Prince, says it became stiff, names the method it
    // switched to, and reports the step counts of both phases.
    assert!(
        s.explanation[0].contains("Dormand-Prince 5(4) method was started"),
        "{text}"
    );
    assert!(text.contains("the system is stiff"), "{text}");
    assert!(
        text.contains("limited by stability, not accuracy"),
        "{text}"
    );
    assert!(
        text.contains("continued from that state with Rodas3"),
        "{text}"
    );
    assert!(text.contains("L-stable"), "{text}");
    assert!(text.contains("with Dormand-Prince ("), "{text}");
    assert!(text.contains("mass drifted by at most"), "{text}");
    assert!(s.explanation.len() >= 5);
    // The values quoted are real: the DP5 phase was short, the rest is Rodas3.
    assert!(s.stats.steps < 2000);
}

#[test]
fn robertson_at_several_tolerances() {
    for rtol in [1e-4, 1e-6, 1e-8] {
        let s = robertson(rtol);
        assert_eq!(s.method, Method::Rodas3, "rtol {rtol:e}");
        let y = &s.samples[0].1;
        // The error at t = 40 tracks the tolerance (loosely).
        assert!(
            rel(y[0], 0.7158270687) < 100.0 * rtol,
            "rtol {rtol:e}: {y:?}"
        );
        assert!((s.y[2] - 0.999999979166505).abs() < 1e-6, "rtol {rtol:e}");
    }
}

#[test]
fn an_explicit_method_cannot_do_robertson_in_reasonable_steps() {
    // The point of switching: Dormand-Prince alone is stability-limited
    // and exhausts a generous budget on a fraction of the interval.
    let mut o = dopri(1e-6, 1e-14);
    o.max_steps = 20_000;
    let e = solve(&mut Robertson, 0.0, 1e3, &[1.0, 0.0, 0.0], &o).unwrap_err();
    assert!(matches!(e, SolveError::TooManySteps { .. }), "{e:?}");
}

fn van_der_pol(rtol: f64) -> Solution {
    let mut o = opts(Method::Auto, rtol, 1e-10);
    o.outputs = vec![500.0, 1000.0];
    solve(&mut VanDerPol { mu: 1000.0 }, 0.0, 3000.0, &[2.0, 0.0], &o).unwrap()
}

#[test]
fn van_der_pol_mu_1000_is_detected_as_stiff_and_solved_correctly() {
    let s = van_der_pol(1e-6);
    assert_eq!(s.method, Method::Rodas3, "{:?}", s.explanation);
    assert!(
        (s.samples[0].1[0] - 1.5967689508).abs() < 1e-5,
        "{:?}",
        s.samples[0]
    );
    assert!(
        (s.samples[1].1[0] + 1.8636462545).abs() < 1e-5,
        "{:?}",
        s.samples[1]
    );
    assert!((s.y[0] + 1.5106069355).abs() < 1e-5, "{:?}", s.y);
    assert!((s.y[1] - 1.17838000e-3).abs() < 1e-6, "{:?}", s.y);
    // ~1.9 periods of a relaxation oscillation with a stiffness of 1e6:
    // thousands of steps, not millions.
    assert!(s.stats.steps < 20_000, "{} steps", s.stats.steps);
    let text = s.explanation.join("\n");
    assert!(text.contains("the system is stiff"), "{text}");
    // The switch happened early, when the transient was over.
    assert!(text.contains("by t = 0.0"), "{text}");
}

#[test]
fn van_der_pol_tighter_tolerance_agrees_with_the_reference() {
    let s = van_der_pol(1e-8);
    assert_eq!(s.method, Method::Rodas3);
    assert!((s.y[0] + 1.5106069355).abs() < 2e-7, "{:?}", s.y);
    assert!((s.samples[0].1[0] - 1.5967689508).abs() < 2e-7);
    assert!((s.samples[1].1[0] + 1.8636462545).abs() < 2e-7);
}

/// The independent check of the reference values above: an explicit
/// method, no stiff machinery, 1.7 million steps.
#[test]
#[ignore = "slow in debug builds: 1.7 million Dormand-Prince steps"]
fn van_der_pol_reference_by_explicit_integration() {
    let mut o = dopri(1e-9, 1e-12);
    o.outputs = vec![500.0, 1000.0];
    let s = solve(&mut VanDerPol { mu: 1000.0 }, 0.0, 3000.0, &[2.0, 0.0], &o).unwrap();
    assert!((s.samples[0].1[0] - 1.5967689508).abs() < 1e-8);
    assert!((s.samples[1].1[0] + 1.8636462545).abs() < 1e-8);
    assert!((s.y[0] + 1.5106069355).abs() < 1e-8, "{:?}", s.y);
    assert!((s.y[1] - 1.17838000e-3).abs() < 1e-9, "{:?}", s.y);
    assert!(s.stats.steps > 1_000_000);
}

#[test]
fn non_stiff_problems_are_not_flagged() {
    // Stiffness detection must not fire on ordinary problems, including
    // oscillatory ones where h*lambda is large-ish but imaginary.
    let lv = |_t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = 1.5 * y[0] - y[0] * y[1];
        d[1] = -3.0 * y[1] + y[0] * y[1];
        Ok(())
    };
    let cases: Vec<(&str, Solution)> = vec![
        (
            "decay",
            solve(&mut Decay, 0.0, 30.0, &[1.0], &Options::default()).unwrap(),
        ),
        (
            "oscillator",
            solve(
                &mut PlainOscillator,
                0.0,
                200.0,
                &[1.0, 0.0],
                &Options::default(),
            )
            .unwrap(),
        ),
        (
            "kepler",
            solve(
                &mut Kepler,
                0.0,
                100.0,
                &kepler_start(0.5),
                &Options::default(),
            )
            .unwrap(),
        ),
        (
            "predator-prey",
            solve(
                &mut Closure(2, lv),
                0.0,
                50.0,
                &[10.0, 5.0],
                &Options::default(),
            )
            .unwrap(),
        ),
        (
            "van der Pol mu = 1",
            solve(
                &mut VanDerPol { mu: 1.0 },
                0.0,
                100.0,
                &[2.0, 0.0],
                &Options::default(),
            )
            .unwrap(),
        ),
    ];
    for (name, s) in cases {
        assert_eq!(
            s.method,
            Method::DormandPrince,
            "{name}: {:?}",
            s.explanation
        );
        assert!(
            s.explanation.iter().any(|e| e.contains("not stiff")),
            "{name}: {:?}",
            s.explanation
        );
    }
}

#[test]
fn a_linear_stiff_problem_with_a_known_solution() {
    // y' = -1e5 (y - cos t) - sin t, y(0) = 1: y = cos t exactly, with a
    // decay rate of 1e5 toward it.
    let mut sys = Closure(1, |t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -1e5 * (y[0] - t.cos()) - t.sin();
        Ok(())
    });
    let s = solve(
        &mut sys,
        0.0,
        10.0,
        &[1.0],
        &opts(Method::Auto, 1e-8, 1e-10),
    )
    .unwrap();
    assert_eq!(s.method, Method::Rodas3, "{:?}", s.explanation);
    assert!((s.y[0] - 10f64.cos()).abs() < 1e-6, "{}", s.y[0]);
    assert!(s.stats.steps < 10_000, "{}", s.stats.steps);
}

#[test]
fn rodas3_is_l_stable() {
    // y' = -1e10 y: the fast mode is damped completely (R(inf) = 0) even at
    // a step 1e10 times larger than 1/lambda; a merely A-stable method
    // (trapezoid) would ring at |R| ~ 1.
    let mut sys = Closure(1, |_t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -1e10 * y[0];
        Ok(())
    });
    let s = solve(
        &mut sys,
        0.0,
        1.0,
        &[1.0],
        &opts(Method::Rodas3, 1e-3, 1e-6),
    )
    .unwrap();
    assert!(s.y[0].abs() < 1e-6, "{}", s.y[0]);
    assert!(s.stats.steps < 100, "{} steps", s.stats.steps);
}

#[test]
fn the_jacobian_perturbation_is_scaled_to_each_component() {
    // Components 32 orders of magnitude apart, all stiff: the difference
    // quotients must be neither swamped by roundoff (tiny component) nor by
    // truncation (huge one). With a = 1e4:
    //   y0 = 1e12 e^{-a t},  y1 = e^{-a t} (1e-20 + 1e-11 t),
    //   y2' = -1e3 y2 + 1e3 y1  (closed form below).
    let mut sys = Closure(3, |_t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -1e4 * y[0];
        d[1] = -1e4 * y[1] + 1e-23 * y[0];
        d[2] = -1e3 * y[2] + 1e3 * y[1];
        Ok(())
    });
    let t = 1e-3;
    let s = solve(
        &mut sys,
        0.0,
        t,
        &[1e12, 1e-20, 0.0],
        &opts(Method::Rodas3, 1e-9, 1e-40),
    )
    .unwrap();
    let (a, c0, sl) = (1e4_f64, 1e-20, 1e-11);
    let e = (-a * t).exp();
    let b = a - 1e3;
    let integral =
        c0 * (1.0 - (-b * t).exp()) / b + sl * (1.0 - (-b * t).exp() * (1.0 + b * t)) / (b * b);
    let exact = [
        1e12 * e,
        e * (c0 + sl * t),
        1e3 * (-1e3 * t).exp() * integral,
    ];
    for (i, (got, want)) in s.y.iter().zip(exact).enumerate() {
        assert!(rel(*got, want) < 1e-5, "component {i}: {got:e} vs {want:e}");
    }
}

#[test]
fn mildly_stiff_problems_finish_with_either_method_and_agree() {
    // A stiffness ratio of ~30: DP5 alone is fine, Rodas3 alone is fine,
    // and they agree, which cross-checks the two implementations.
    let mut sys = Closure(2, |_t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -30.0 * y[0] + y[1];
        d[1] = -y[1] + 0.5 * y[0];
        Ok(())
    });
    let a = solve(&mut sys, 0.0, 20.0, &[1.0, 1.0], &dopri(1e-10, 1e-14)).unwrap();
    let b = solve(
        &mut sys,
        0.0,
        20.0,
        &[1.0, 1.0],
        &opts(Method::Rodas3, 1e-10, 1e-14),
    )
    .unwrap();
    assert!((a.y[0] - b.y[0]).abs() < 1e-9 && (a.y[1] - b.y[1]).abs() < 1e-9);
}
