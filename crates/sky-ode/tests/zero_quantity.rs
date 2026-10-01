//! A conserved quantity that is zero (up to rounding) has no scale of its
//! own: its drift is measured against the size of its terms, and a drift
//! that a smaller step does not reduce (rounding) stops setting the step.
//! Before, the relative drift of a zero quantity was noise divided by
//! noise, and `Auto` shrank the step a thousandfold for nothing (the
//! figure-eight orbit's angular momentum ran for minutes).

mod common;

use common::*;
use sky_ode::*;
use std::f64::consts::PI;

/// Kepler's angular momentum minus its value at the start, the start's
/// value taken one rounding off: a quantity that is zero up to rounding,
/// as `Lz` of the figure-eight orbit is.
fn lz_minus(offset: f64) -> Conserved {
    let l0 = kepler_angular_momentum(&kepler_start(0.5));
    let l0 = f64::from_bits(l0.to_bits() + 1);
    Conserved {
        name: "Lz".to_string(),
        value: Box::new(move |_, y| kepler_angular_momentum(y) - l0 + offset),
        max_drift: None,
    }
}

fn run(conserved: Vec<Conserved>) -> Solution {
    let mut o = opts(Method::Auto, 1e-8, 1e-12);
    o.conserved = conserved;
    solve(&mut Kepler, 0.0, 2.0 * PI, &kepler_start(0.5), &o).unwrap()
}

#[test]
fn a_zero_quantity_is_measured_against_its_terms_and_costs_nothing() {
    let alone = run(vec![energy_conserved()]);
    let with = run(vec![energy_conserved(), lz_minus(0.0)]);
    // Same step as without the zero quantity: it is not what sets it.
    assert_eq!(with.stats.steps, alone.stats.steps);
    // Rounding-sized against the size of x vy (about 0.87).
    assert!(with.drift[1] < 1e-12, "{:e}", with.drift[1]);
    let text = with.explanation.join("\n");
    assert!(
        text.contains("Lz drifted by at most")
            && text.contains("(relative to the size of its terms, since it starts at 0)"),
        "{text}"
    );
}

#[test]
fn a_drift_at_the_rounding_floor_stops_setting_the_step() {
    // 1e-10 is far above rounding, so the drift is relative to it, and the
    // rounding of the terms (about 1e-16) is then a relative 1e-6 that no
    // step can bring down to the 1e-8 asked.
    let alone = run(vec![energy_conserved()]);
    let with = run(vec![energy_conserved(), lz_minus(1e-10)]);
    assert!(
        with.stats.steps <= 2 * alone.stats.steps,
        "{} steps, {} without Lz",
        with.stats.steps,
        alone.stats.steps
    );
    let text = with.explanation.join("\n");
    assert!(
        text.contains("Lz drifted no less at the smaller step")
            && text.contains("which no step removes, so it no longer sets the step"),
        "{text}"
    );
    // Energy still gets the tolerance it asked for.
    assert!(with.drift[0] <= 1e-8, "{:e}", with.drift[0]);
}
