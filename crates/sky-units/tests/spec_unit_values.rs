//! Pins the value claims in spec v0.1, "Unit values", for units already in
//! the table. The dimension side is a golden case
//! (`tests/golden/pass_spec_km_plus_m.case`). The `ms` examples are pinned
//! alongside `ms` itself, in `src/unit.rs`.
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
fn one_km_plus_one_m_is_1001_m() {
    assert_eq!(si(1.0, "km") + si(1.0, "m"), 1001.0);
}
