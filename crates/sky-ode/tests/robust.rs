//! Robustness: every failure is a `SolveError` naming a time, never NaN or
//! infinity; the step budget holds; bad input is rejected; results are
//! deterministic.

mod common;

use common::*;
use sky_ode::*;

/// All the methods, configured to run on the oscillator/decay-sized
/// problems below.
fn every_method() -> Vec<(&'static str, Method)> {
    vec![
        ("auto", Method::Auto),
        ("rk4", Method::Rk4 { step: 0.01 }),
        ("dopri", Method::DormandPrince),
        ("rodas3", Method::Rodas3),
        ("verlet", Method::VelocityVerlet { step: 0.01 }),
        ("yoshida", Method::Yoshida4 { step: 0.01 }),
    ]
}

fn with(method: Method) -> Options {
    Options {
        method,
        rtol: 1e-8,
        atol: 1e-10,
        ..Options::default()
    }
}

// ---- failing right-hand sides ----------------------------------------------

/// A second-order system (so every method, symplectic included, applies)
/// whose right-hand side fails after `t_fail`.
struct FailsAfter {
    t_fail: f64,
}
impl System for FailsAfter {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&mut self, t: f64, y: &[f64], d: &mut [f64]) -> Result<(), String> {
        if t > self.t_fail {
            return Err("division by zero in the force law".to_string());
        }
        d[0] = y[1];
        d[1] = -y[0];
        Ok(())
    }
    fn structure(&self) -> Structure {
        Structure::SecondOrder { n: 1 }
    }
}

#[test]
fn a_failing_rhs_stops_the_solve_with_its_message_and_time() {
    for (name, m) in every_method() {
        let mut o = with(m);
        // Auto needs a conserved quantity to choose symplectic; test both.
        if name == "auto" {
            o.conserved = vec![Conserved {
                name: "energy".into(),
                value: Box::new(|_, y| 0.5 * (y[0] * y[0] + y[1] * y[1])),
                max_drift: None,
            }];
        }
        let e = solve(&mut FailsAfter { t_fail: 0.5 }, 0.0, 1.0, &[1.0, 0.0], &o).unwrap_err();
        match &e {
            SolveError::Rhs { t, message } => {
                assert!(*t > 0.5 && *t <= 1.0, "{name}: t = {t}");
                assert_eq!(message, "division by zero in the force law");
            }
            e => panic!("{name}: expected an Rhs error, got {e:?}"),
        }
        let text = e.to_string();
        assert!(
            text.contains("division by zero") && text.contains("t = "),
            "{name}: {text}"
        );
    }
}

#[test]
fn a_failing_rhs_at_the_start_is_reported_at_t0() {
    for (name, m) in every_method() {
        let e = solve(
            &mut FailsAfter { t_fail: -1.0 },
            2.0,
            3.0,
            &[1.0, 0.0],
            &with(m),
        )
        .unwrap_err();
        assert!(
            matches!(e, SolveError::Rhs { t, .. } if t >= 2.0),
            "{name}: {e:?}"
        );
    }
}

// ---- NaN and infinity ---------------------------------------------------------

#[test]
fn a_nan_right_hand_side_is_not_finite_not_a_nan_result() {
    struct Nan;
    impl System for Nan {
        fn dim(&self) -> usize {
            2
        }
        fn rhs(&mut self, _t: f64, _y: &[f64], d: &mut [f64]) -> Result<(), String> {
            d[0] = f64::NAN;
            d[1] = 0.0;
            Ok(())
        }
        fn structure(&self) -> Structure {
            Structure::SecondOrder { n: 1 }
        }
    }
    for (name, m) in every_method() {
        let o = with(m);
        let r = solve(&mut Nan, 0.0, 1.0, &[1.0, 0.0], &o);
        match r {
            Err(SolveError::NotFinite { t }) => assert!(t.is_finite(), "{name}"),
            other => panic!("{name}: expected NotFinite, got {other:?}"),
        }
    }
}

#[test]
fn a_right_hand_side_that_becomes_nan_midway_is_an_error_near_that_time() {
    // d/dt y = sqrt(1.5 - t): NaN beyond t = 1.5 (the square root of a
    // negative). The adaptive methods shrink their steps against the edge
    // and give up; the fixed-step ones step into it.
    for (name, m) in [
        ("rk4", Method::Rk4 { step: 0.01 }),
        ("dopri", Method::DormandPrince),
        ("rodas3", Method::Rodas3),
        ("auto", Method::Auto),
    ] {
        let mut sys = Closure(1, |t: f64, _y: &[f64], d: &mut [f64]| {
            d[0] = (1.5 - t).sqrt();
            Ok(())
        });
        let r = solve(&mut sys, 0.0, 3.0, &[0.0], &with(m));
        match r {
            Err(SolveError::NotFinite { t }) | Err(SolveError::StepUnderflow { t, .. }) => {
                assert!((1.4..=1.51).contains(&t), "{name}: failed at t = {t}");
            }
            other => panic!("{name}: expected NotFinite or StepUnderflow, got {other:?}"),
        }
    }
}

#[test]
fn a_blow_up_is_a_step_underflow_with_a_helpful_message() {
    // y' = y^2, y(0) = 1: y = 1/(1 - t) blows up at t = 1.
    for (name, m) in [
        ("dopri", Method::DormandPrince),
        ("rodas3", Method::Rodas3),
        ("auto", Method::Auto),
    ] {
        let mut sys = Closure(1, |_t: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[0] * y[0];
            Ok(())
        });
        let e = solve(&mut sys, 0.0, 2.0, &[1.0], &with(m)).unwrap_err();
        match &e {
            SolveError::StepUnderflow { t, step } => {
                assert!((0.99..=1.0 + 1e-9).contains(t), "{name}: t = {t}");
                assert!(*step > 0.0 && *step < 1e-8, "{name}: step {step:e}");
                let text = e.to_string();
                assert!(text.contains("step size underflow"), "{text}");
                assert!(text.contains("singularity or blows up"), "{text}");
            }
            SolveError::NotFinite { t } => assert!((0.99..=1.0 + 1e-9).contains(t), "{name}: {t}"),
            e => panic!("{name}: {e:?}"),
        }
    }
}

#[test]
fn overflow_in_a_fixed_step_method_is_not_finite() {
    // y' = y^3 blows up in finite time; a fixed step runs into infinity.
    let mut sys = Closure(1, |_t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = y[0] * y[0] * y[0];
        Ok(())
    });
    let o = with(Method::Rk4 { step: 0.05 });
    let e = solve(&mut sys, 0.0, 10.0, &[2.0], &o).unwrap_err();
    assert!(
        matches!(e, SolveError::NotFinite { t } if t > 0.0 && t < 10.0),
        "{e:?}"
    );
}

#[test]
fn a_nan_conserved_quantity_is_an_error() {
    let nan_later = Conserved {
        name: "q".into(),
        value: Box::new(|t, _| if t > 0.5 { f64::NAN } else { 1.0 }),
        max_drift: None,
    };
    let mut o = with(Method::DormandPrince);
    o.conserved = vec![nan_later];
    let e = solve(&mut Decay, 0.0, 1.0, &[1.0], &o).unwrap_err();
    assert!(matches!(e, SolveError::NotFinite { t } if t > 0.5), "{e:?}");
    let nan_now = Conserved {
        name: "q".into(),
        value: Box::new(|_, _| f64::INFINITY),
        max_drift: None,
    };
    let mut o = with(Method::DormandPrince);
    o.conserved = vec![nan_now];
    let e = solve(&mut Decay, 0.0, 1.0, &[1.0], &o).unwrap_err();
    assert_eq!(e, SolveError::NotFinite { t: 0.0 });
}

// ---- step budget ------------------------------------------------------------------

#[test]
fn max_steps_is_enforced_by_every_method() {
    for (name, m) in every_method() {
        let mut o = with(m);
        o.max_steps = 50;
        // Needs far more than 50 steps at these settings.
        let mut o2 = o;
        if name == "auto" {
            o2.conserved = vec![Conserved {
                name: "energy".into(),
                value: Box::new(|_, y| 0.5 * (y[0] * y[0] + y[1] * y[1])),
                max_drift: None,
            }];
        }
        let e = solve(
            &mut FailsAfter { t_fail: 1e9 },
            0.0,
            1000.0,
            &[1.0, 0.0],
            &o2,
        )
        .unwrap_err();
        match e {
            SolveError::TooManySteps { t, steps } => {
                assert_eq!(steps, 50, "{name}");
                assert!(t > 0.0 && t < 1000.0, "{name}: t = {t}");
            }
            e => panic!("{name}: expected TooManySteps, got {e:?}"),
        }
    }
}

#[test]
fn a_huge_step_count_fails_fast_instead_of_hanging() {
    // A fixed step of 1e-300 over [0, 1] is 1e300 steps: the budget stops
    // it, and a count that overflows an integer is handled, too.
    let mut o = with(Method::Rk4 { step: 1e-300 });
    o.max_steps = 1000;
    let e = solve(&mut Decay, 0.0, 1.0, &[1.0], &o).unwrap_err();
    assert!(matches!(e, SolveError::TooManySteps { .. }), "{e:?}");
    let mut o = with(Method::Rk4 { step: 1e-3 });
    o.max_steps = 10;
    let e = solve(&mut Decay, 0.0, 1.0, &[1.0], &o).unwrap_err();
    assert!(
        matches!(e, SolveError::TooManySteps { steps: 10, .. }),
        "{e:?}"
    );
}

// ---- bad input ---------------------------------------------------------------------

fn bad(r: Result<Solution, SolveError>) -> String {
    match r {
        Err(SolveError::BadInput(m)) => m,
        other => panic!("expected BadInput, got {other:?}"),
    }
}

#[test]
fn bad_states_and_intervals() {
    let o = with(Method::DormandPrince);
    assert!(bad(solve(&mut Decay, 0.0, 1.0, &[], &o)).contains("0 values"));
    assert!(bad(solve(&mut Decay, 0.0, 1.0, &[f64::NAN], &o)).contains("NaN"));
    assert!(bad(solve(&mut Decay, 0.0, 1.0, &[f64::INFINITY], &o)).contains("infinity"));
    assert!(bad(solve(&mut Decay, 1.0, 0.0, &[1.0], &o)).contains("increasing"));
    assert!(bad(solve(&mut Decay, 0.0, f64::NAN, &[1.0], &o)).contains("finite"));
    assert!(bad(solve(&mut Decay, f64::NEG_INFINITY, 0.0, &[1.0], &o)).contains("finite"));
}

#[test]
fn bad_output_times() {
    for outputs in [
        vec![0.5, 0.4],
        vec![-0.1],
        vec![1.5],
        vec![f64::NAN],
        vec![0.2, f64::INFINITY],
    ] {
        let o = Options {
            outputs: outputs.clone(),
            ..with(Method::DormandPrince)
        };
        let m = bad(solve(&mut Decay, 0.0, 1.0, &[1.0], &o));
        assert!(m.contains("output time"), "{outputs:?}: {m}");
    }
}

#[test]
fn bad_steps_and_tolerances() {
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for m in [
            Method::Rk4 { step },
            Method::VelocityVerlet { step },
            Method::Yoshida4 { step },
        ] {
            let msg = bad(solve(&mut Oscillator, 0.0, 1.0, &[1.0, 0.0], &with(m)));
            assert!(msg.contains("step"), "{m:?}: {msg}");
        }
    }
    for (rtol, atol) in [
        (-1e-6, 1e-9),
        (1e-6, -1e-9),
        (f64::NAN, 1e-9),
        (1e-6, f64::INFINITY),
        (0.0, 0.0),
        (1e-15, 1e-12),
    ] {
        for m in [Method::DormandPrince, Method::Rodas3, Method::Auto] {
            let o = Options {
                rtol,
                atol,
                method: m,
                ..Options::default()
            };
            let msg = bad(solve(&mut Decay, 0.0, 1.0, &[1.0], &o));
            assert!(
                msg.contains("tolerance") || msg.contains("rtol"),
                "{m:?} {rtol} {atol}: {msg}"
            );
        }
    }
    // A pure absolute tolerance is legitimate.
    let o = Options {
        rtol: 0.0,
        atol: 1e-9,
        method: Method::DormandPrince,
        ..Options::default()
    };
    assert!(solve(&mut Decay, 0.0, 1.0, &[1.0], &o).is_ok());
    for (first, max) in [
        (Some(0.0), None),
        (None, Some(-1.0)),
        (Some(f64::NAN), None),
    ] {
        let o = Options {
            first_step: first,
            max_step: max,
            ..with(Method::DormandPrince)
        };
        bad(solve(&mut Decay, 0.0, 1.0, &[1.0], &o));
    }
}

#[test]
fn structure_mistakes() {
    // A symplectic method on a system that did not declare the structure.
    for m in [
        Method::VelocityVerlet { step: 0.1 },
        Method::Yoshida4 { step: 0.1 },
    ] {
        let e = solve(&mut PlainOscillator, 0.0, 1.0, &[1.0, 0.0], &with(m)).unwrap_err();
        match e {
            SolveError::Unsupported(msg) => {
                assert!(msg.contains("second-order"), "{msg}");
            }
            e => panic!("{e:?}"),
        }
    }
    // A declared structure that does not match the dimension.
    struct Wrong;
    impl System for Wrong {
        fn dim(&self) -> usize {
            3
        }
        fn rhs(&mut self, _t: f64, _y: &[f64], d: &mut [f64]) -> Result<(), String> {
            d.fill(0.0);
            Ok(())
        }
        fn structure(&self) -> Structure {
            Structure::SecondOrder { n: 1 }
        }
    }
    let m = bad(solve(&mut Wrong, 0.0, 1.0, &[0.0; 3], &with(Method::Auto)));
    assert!(m.contains("second-order"), "{m}");
}

#[test]
fn an_empty_system_is_solved_trivially() {
    let mut sys = Closure(0, |_t: f64, _y: &[f64], _d: &mut [f64]| Ok(()));
    for (name, m) in every_method() {
        if matches!(m, Method::VelocityVerlet { .. } | Method::Yoshida4 { .. }) {
            continue;
        }
        let s = solve(&mut sys, 0.0, 1.0, &[], &with(m)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((s.t, s.y.len()), (1.0, 0), "{name}");
    }
}

// ---- determinism -----------------------------------------------------------------------

#[test]
fn solves_are_deterministic() {
    let energy = || vec![energy_conserved()];
    let run = |which: u32| -> Solution {
        match which {
            0 => {
                let mut o = with(Method::Auto);
                o.conserved = energy();
                o.outputs = vec![1.0, 10.0, 30.0];
                solve(&mut Kepler, 0.0, 60.0, &kepler_start(0.5), &o).unwrap()
            }
            1 => {
                let mut o = with(Method::Rodas3);
                o.outputs = vec![40.0, 1e5];
                solve(&mut Robertson, 0.0, 1e8, &[1.0, 0.0, 0.0], &o).unwrap()
            }
            2 => solve(
                &mut VanDerPol { mu: 1000.0 },
                0.0,
                800.0,
                &[2.0, 0.0],
                &with(Method::Rodas3),
            )
            .unwrap(),
            _ => {
                let mut o = with(Method::DormandPrince);
                o.outputs = (1..50).map(|i| f64::from(i) * 0.37).collect();
                solve(&mut Kepler, 0.0, 20.0, &kepler_start(0.3), &o).unwrap()
            }
        }
    };
    for which in 0..4 {
        let (a, b) = (run(which), run(which));
        // `Solution: PartialEq` compares every f64 exactly and the whole
        // explanation text: the runs are bit-identical.
        assert_eq!(a, b, "case {which}");
        for (x, y) in a.y.iter().zip(&b.y) {
            assert_eq!(x.to_bits(), y.to_bits(), "case {which}");
        }
    }
}

#[test]
fn the_options_are_not_consumed_or_changed_by_a_solve() {
    let mut o = with(Method::Auto);
    o.conserved = vec![energy_conserved()];
    let a = solve(&mut Kepler, 0.0, 20.0, &kepler_start(0.5), &o).unwrap();
    let b = solve(&mut Kepler, 0.0, 20.0, &kepler_start(0.5), &o).unwrap();
    assert_eq!(a, b);
}
