//! Short benchmarks as tests. The time limits are deliberately generous
//! (dozens of times the measured release time, so they hold in unoptimised
//! debug builds and on slow CI machines): they catch an accidental
//! quadratic step loop or a runaway step count, not a 20% slowdown. The
//! `#[ignore]`d test prints throughput figures:
//! `cargo test -p sky-ode --release --test bench -- --ignored --nocapture`.

mod common;

use common::*;
use sky_ode::*;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

fn timed<T>(limit: Duration, f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let r = f();
    let took = start.elapsed();
    assert!(took < limit, "took {took:?}, limit {limit:?}");
    (r, took)
}

fn kepler_auto(periods: f64, rtol: f64) -> Solution {
    let mut o = opts(Method::Auto, rtol, 1e-12);
    o.conserved = vec![energy_conserved()];
    solve(&mut Kepler, 0.0, periods * 2.0 * PI, &kepler_start(0.5), &o).unwrap()
}

fn robertson_auto() -> Solution {
    let mut o = opts(Method::Auto, 1e-8, 1e-14);
    o.outputs = vec![40.0];
    solve(&mut Robertson, 0.0, 1e11, &[1.0, 0.0, 0.0], &o).unwrap()
}

fn van_der_pol_auto(rtol: f64) -> Solution {
    solve(
        &mut VanDerPol { mu: 1000.0 },
        0.0,
        3000.0,
        &[2.0, 0.0],
        &opts(Method::Auto, rtol, 1e-10),
    )
    .unwrap()
}

#[test]
fn kepler_100_periods_is_fast() {
    let (s, _) = timed(Duration::from_secs(30), || kepler_auto(100.0, 1e-9));
    assert!(s.stats.steps < 300_000, "{} steps", s.stats.steps);
    assert!(s.stats.rhs_evals < 1_000_000, "{} evals", s.stats.rhs_evals);
}

#[test]
fn stiff_problems_are_fast() {
    let (s, _) = timed(Duration::from_secs(30), robertson_auto);
    assert!(s.stats.rhs_evals < 100_000, "{} evals", s.stats.rhs_evals);
    let (s, _) = timed(Duration::from_secs(30), || van_der_pol_auto(1e-6));
    assert!(s.stats.rhs_evals < 100_000, "{} evals", s.stats.rhs_evals);
}

#[test]
fn a_million_cheap_steps_stay_linear() {
    // 1e6 Verlet steps of a 1-d oscillator: catches per-step allocation or
    // per-step work that grows with the step count.
    let o = Options {
        method: Method::VelocityVerlet { step: 1e-3 },
        max_steps: 2_000_000,
        ..Options::default()
    };
    let (s, _) = timed(Duration::from_secs(60), || {
        solve(&mut Oscillator, 0.0, 1000.0, &[1.0, 0.0], &o).unwrap()
    });
    assert_eq!(s.stats.steps, 1_000_000);
    assert!((s.y[0] - 1000f64.cos()).abs() < 1e-3);
}

#[test]
fn many_outputs_do_not_slow_the_adaptive_methods() {
    // 100k outputs on Dormand-Prince: dense output, one interpolation
    // each, no extra steps.
    let outputs: Vec<f64> = (1..=100_000).map(|i| f64::from(i) * 2e-4).collect();
    let mut o = dopri(1e-9, 1e-12);
    o.outputs = outputs;
    let (s, _) = timed(Duration::from_secs(30), || {
        solve(&mut Oscillator, 0.0, 20.0, &[1.0, 0.0], &o).unwrap()
    });
    assert_eq!(s.samples.len(), 100_000);
    assert!(s.stats.steps < 1000, "{} steps", s.stats.steps);
}

#[test]
#[ignore = "prints throughput; run with --release --ignored --nocapture"]
fn throughput_report() {
    for (name, rtol) in [
        ("kepler auto 1e-6", 1e-6),
        ("kepler auto 1e-9", 1e-9),
        ("kepler auto 1e-12", 1e-12),
    ] {
        let (s, took) = timed(Duration::from_secs(600), || kepler_auto(100.0, rtol));
        println!(
            "{name}: {took:?}, {} steps, {} rhs evals, {:.1} M evals/s, drift {:.1e}",
            s.stats.steps,
            s.stats.rhs_evals,
            s.stats.rhs_evals as f64 / took.as_secs_f64() / 1e6,
            s.drift[0]
        );
    }
    let mut o = dopri(1e-9, 1e-12);
    o.conserved = vec![energy_conserved()];
    let (s, took) = timed(Duration::from_secs(600), || {
        solve(&mut Kepler, 0.0, 100.0 * 2.0 * PI, &kepler_start(0.5), &o).unwrap()
    });
    println!(
        "kepler dopri 1e-9: {took:?}, {} steps, {} rhs evals, drift {:.1e}",
        s.stats.steps, s.stats.rhs_evals, s.drift[0]
    );
    let (s, took) = timed(Duration::from_secs(600), robertson_auto);
    println!(
        "robertson auto 1e-8: {took:?}, {} steps, {} evals",
        s.stats.steps, s.stats.rhs_evals
    );
    let (s, took) = timed(Duration::from_secs(600), || van_der_pol_auto(1e-8));
    println!(
        "van der pol 1000 auto 1e-8: {took:?}, {} steps, {} evals",
        s.stats.steps, s.stats.rhs_evals
    );
}
