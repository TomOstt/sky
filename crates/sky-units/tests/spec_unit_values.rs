//! Pins the value claims in spec v0.1, "Unit values". The dimension side of
//! each example is a golden case (`tests/golden/pass_spec_*.case`).
//!
//! A unit literal `x u` is converted to SI once, as `x * lookup(u).scale`.

use sky_units::lookup;

/// `x u` converted to SI, as the spec says the literal is.
fn si(x: f64, unit: &str) -> f64 {
    x * lookup(unit)
        .unwrap_or_else(|| panic!("unit `{unit}` is not in the table"))
        .scale
}

#[test]
fn ten_min_is_600_s() {
    assert_eq!(si(10.0, "min"), 600.0);
}

#[test]
fn one_km_plus_one_m_is_1001_m() {
    assert_eq!(si(1.0, "km") + si(1.0, "m"), 1001.0);
}

#[test]
fn three_ev_is_not_exactly_4_806529902e_minus_19_j() {
    // 3 eV is exactly 4.806529902e-19 J by the SI definition of the
    // electronvolt, but the f64 product rounds one step low.
    let three_ev = si(3.0, "eV");
    let joules = si(4.806_529_902e-19, "J");
    assert_ne!(three_ev, joules);
    assert_eq!(three_ev.to_bits() + 1, joules.to_bits());
    assert_eq!(format!("{three_ev:e}"), "4.806529901999999e-19");
}
