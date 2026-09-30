//! The Kepler problem (eccentricity 0.5, 100 periods): the case for
//! symplectic integrators, and for `Auto` choosing them when energy is
//! monitored.

mod common;

use common::*;
use sky_ode::*;
use std::f64::consts::PI;

const PERIOD: f64 = 2.0 * PI;

fn solve_kepler(method: Method, rtol: f64, periods: f64, conserved: Vec<Conserved>) -> Solution {
    let mut o = opts(method, rtol, 1e-12);
    o.conserved = conserved;
    solve(&mut Kepler, 0.0, periods * PERIOD, &kepler_start(0.5), &o).unwrap()
}

fn energy() -> Vec<Conserved> {
    vec![energy_conserved()]
}

/// Position error after whole periods (the exact orbit is periodic).
fn position_error(s: &Solution) -> f64 {
    let y0 = kepler_start(0.5);
    ((s.y[0] - y0[0]).powi(2) + (s.y[1] - y0[1]).powi(2)).sqrt()
}

#[test]
fn dopri_energy_drift_grows_with_time() {
    // Secular drift: about linear in the number of periods.
    let d10 = solve_kepler(Method::DormandPrince, 1e-6, 10.0, energy()).drift[0];
    let d100 = solve_kepler(Method::DormandPrince, 1e-6, 100.0, energy()).drift[0];
    let growth = d100 / d10;
    assert!(
        (5.0..20.0).contains(&growth),
        "10 -> 100 periods: {d10:e} -> {d100:e} (x{growth})"
    );
    assert!(d100 > 1e-4, "{d100:e}");
}

#[test]
fn symplectic_energy_drift_stays_bounded() {
    for (name, method) in [
        ("auto", Method::Auto),
        ("yoshida", Method::Yoshida4 { step: 0.02 }),
        ("verlet", Method::VelocityVerlet { step: 0.002 }),
    ] {
        let d10 = solve_kepler(method, 1e-6, 10.0, energy()).drift[0];
        let d100 = solve_kepler(method, 1e-6, 100.0, energy()).drift[0];
        // Ten times as long: the *maximum* deviation barely moves (it is the
        // amplitude of a bounded oscillation, reached within a period).
        assert!(
            d100 < 1.5 * d10,
            "{name}: 10 -> 100 periods: {d10:e} -> {d100:e}"
        );
    }
}

#[test]
fn symplectic_beats_dopri_on_energy_at_the_same_tolerance() {
    let dp = solve_kepler(Method::DormandPrince, 1e-6, 100.0, energy());
    let auto = solve_kepler(Method::Auto, 1e-6, 100.0, energy());
    assert!(
        dp.drift[0] > 20.0 * auto.drift[0],
        "dopri {:e} vs symplectic {:e}",
        dp.drift[0],
        auto.drift[0]
    );
    // And it is not paying for that with the orbit: the positions after 100
    // periods are more accurate too.
    assert!(position_error(&auto) < position_error(&dp) / 5.0);
}

#[test]
fn auto_picks_a_symplectic_method_when_energy_is_monitored() {
    let s = solve_kepler(Method::Auto, 1e-9, 100.0, energy());
    match s.method {
        Method::Yoshida4 { step } => assert!(step > 1e-4 && step < 0.1, "step {step}"),
        m => panic!("expected Yoshida4 at rtol 1e-9, got {m:?}"),
    }
    let text = s.explanation.join("\n");
    assert!(
        s.explanation[0].starts_with(
            "the system is second order and energy is monitored: symplectic integrators keep energy error bounded over long times instead of letting it drift"
        ),
        "{text}"
    );
    assert!(text.contains("Yoshida"), "{text}");
    assert!(text.contains("step doubling"), "{text}");
    assert!(text.contains("rtol = 1.00e-9"), "{text}");
    assert!(
        s.explanation
            .last()
            .unwrap()
            .starts_with("energy drifted by at most"),
        "{text}"
    );
    assert_eq!(s.drift.len(), 1);
    assert!(s.drift[0] < 1e-7, "{:e}", s.drift[0]);
    // Accuracy: the orbit is right to ~1e-5 after 100 periods.
    assert!(position_error(&s) < 1e-4, "{:e}", position_error(&s));
}

#[test]
fn auto_uses_verlet_at_a_loose_tolerance() {
    let s = solve_kepler(Method::Auto, 1e-3, 20.0, energy());
    assert!(
        matches!(s.method, Method::VelocityVerlet { .. }),
        "{:?}",
        s.method
    );
    assert!(
        s.explanation.iter().any(|e| e.contains("velocity Verlet")),
        "{:?}",
        s.explanation
    );
    assert!(s.drift[0] < 1e-2, "{:e}", s.drift[0]);
}

#[test]
fn auto_names_every_monitored_quantity() {
    let l = Conserved {
        name: "angular momentum".to_string(),
        value: Box::new(|_, y| kepler_angular_momentum(y)),
        max_drift: None,
    };
    let s = solve_kepler(Method::Auto, 1e-8, 5.0, vec![energy_conserved(), l]);
    assert!(
        s.explanation[0].starts_with(
            "the system is second order and energy and angular momentum are monitored: symplectic integrators keep their error bounded"
        ),
        "{:?}",
        s.explanation
    );
    assert_eq!(s.drift.len(), 2);
    // Verlet and Yoshida conserve angular momentum to roundoff (a quadratic
    // invariant preserved by symmetric compositions of these maps).
    assert!(s.drift[1] < 1e-12, "{:e}", s.drift[1]);
    let text = s.explanation.join("\n");
    assert!(
        text.contains("angular momentum drifted by at most"),
        "{text}"
    );
}

#[test]
fn auto_without_a_conserved_quantity_starts_with_dormand_prince() {
    let s = solve_kepler(Method::Auto, 1e-9, 5.0, Vec::new());
    assert_eq!(s.method, Method::DormandPrince);
    let text = s.explanation.join("\n");
    assert!(
        text.contains("second order, but no conserved quantity"),
        "{text}"
    );
    assert!(text.contains("Dormand-Prince"), "{text}");
    assert!(text.contains("not stiff"), "{text}");
}

#[test]
fn a_general_system_is_never_given_a_symplectic_method() {
    // Same equations, structure not declared: symplectic is impossible.
    let mut o = opts(Method::Auto, 1e-8, 1e-10);
    o.conserved = vec![Conserved {
        name: "energy".into(),
        value: Box::new(|_, y| 0.5 * (y[0] * y[0] + y[1] * y[1])),
        max_drift: None,
    }];
    let s = solve(&mut PlainOscillator, 0.0, 50.0, &[1.0, 0.0], &o).unwrap();
    assert_eq!(s.method, Method::DormandPrince);
    assert!(
        s.explanation[0].contains("no structure"),
        "{:?}",
        s.explanation
    );
    let e = solve(
        &mut PlainOscillator,
        0.0,
        1.0,
        &[1.0, 0.0],
        &opts(Method::Yoshida4 { step: 0.1 }, 1e-8, 1e-10),
    );
    assert!(matches!(e, Err(SolveError::Unsupported(_))), "{e:?}");
}

#[test]
fn max_drift_is_enforced() {
    // DOPRI5 at rtol 1e-6 breaks a 1e-5 energy budget within 100 periods ...
    let mut o = opts(Method::DormandPrince, 1e-6, 1e-12);
    o.conserved = vec![Conserved {
        max_drift: Some(1e-5),
        ..energy_conserved()
    }];
    let e = solve(&mut Kepler, 0.0, 100.0 * PERIOD, &kepler_start(0.5), &o).unwrap_err();
    match e {
        SolveError::Drift {
            name,
            drift,
            max,
            t,
        } => {
            assert_eq!(name, "energy");
            assert!(drift > max && max == 1e-5);
            assert!(t > 0.0 && t < 100.0 * PERIOD, "{t}");
        }
        e => panic!("expected a drift error, got {e:?}"),
    }
    // ... where the symplectic method keeps it (at a tolerance that allows it).
    let mut o = opts(Method::Auto, 1e-8, 1e-12);
    o.conserved = vec![Conserved {
        max_drift: Some(1e-5),
        ..energy_conserved()
    }];
    let s = solve(&mut Kepler, 0.0, 100.0 * PERIOD, &kepler_start(0.5), &o).unwrap();
    assert!(s.drift[0] < 1e-5);
    assert!(s.explanation.last().unwrap().contains("within the allowed"));
}

#[test]
fn rk4_drifts_too() {
    // The baseline: fixed-step RK4 dissipates energy steadily.
    let d = |periods| solve_kepler(Method::Rk4 { step: 0.01 }, 1e-9, periods, energy()).drift[0];
    let (d10, d40) = (d(10.0), d(40.0));
    assert!(d40 > 2.5 * d10, "{d10:e} -> {d40:e}");
}

#[test]
fn auto_keeps_the_monitored_quantity_within_the_tolerance() {
    // The step is checked against the energy itself: over the whole 100
    // periods (not just the trial window) its drift stays at the tolerance.
    for rtol in [1e-5, 1e-7, 1e-9] {
        let s = solve_kepler(Method::Auto, rtol, 100.0, energy());
        assert!(
            s.drift[0] <= 1.5 * rtol,
            "rtol {rtol:e}: energy drifted by {:e}",
            s.drift[0]
        );
        let text = s.explanation.join("\n");
        assert!(text.contains("trial run"), "{text}");
    }
}

#[test]
fn auto_stays_clear_of_a_user_drift_limit() {
    // A max_drift far tighter than the tolerance makes the step smaller so
    // the run does not abort: the limit is honoured by choosing the step,
    // not just by failing.
    let mut o = opts(Method::Auto, 1e-6, 1e-12);
    o.conserved = vec![Conserved {
        max_drift: Some(1e-9),
        ..energy_conserved()
    }];
    let s = solve(&mut Kepler, 0.0, 30.0 * PERIOD, &kepler_start(0.5), &o).unwrap();
    assert!(s.drift[0] < 1e-9, "{:e}", s.drift[0]);
    let text = s.explanation.join("\n");
    assert!(text.contains("so the step was reduced"), "{text}");
}
