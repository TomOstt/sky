//! The v0.1 unit table: each unit is a dimension plus a scale factor to the
//! coherent SI unit of that dimension (so `km` is length with scale 1000).
//!
//! The derived SI units `N`, `Pa`, `W`, and `Hz` are coherent (scale 1).
//! Because `rad` is dimensionless, `Hz` and `rad/s` have the same dimension,
//! so a missing 2π between frequency and angular frequency is not caught (a
//! known v0.1 limitation).
//!
//! Affine units such as °C are deliberately absent in v0.1. Lookup is by
//! symbol only; full-name aliases (`meters`, `kelvin`) are not provided yet.

use crate::dimension::Dimension;

/// A unit: a dimension and a multiplicative scale factor to SI.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Unit {
    /// The unit symbol, e.g. `"km"`.
    pub name: &'static str,
    /// The unit's dimension.
    pub dim: Dimension,
    /// Value of one of this unit in the coherent SI unit of `dim`.
    pub scale: f64,
}

/// 1 eV in joules (exact since the 2019 SI redefinition).
pub const ELECTRONVOLT_IN_JOULES: f64 = 1.602_176_634e-19;
/// 1 au in metres (exact, IAU 2012 Resolution B2).
pub const ASTRONOMICAL_UNIT_IN_METRES: f64 = 149_597_870_700.0;
/// 1 pc in metres: 648000/π au (IAU 2015 Resolution B2).
pub const PARSEC_IN_METRES: f64 = 648_000.0 / std::f64::consts::PI * ASTRONOMICAL_UNIT_IN_METRES;

const fn unit(name: &'static str, dim: Dimension, scale: f64) -> Unit {
    Unit { name, dim, scale }
}

/// Every unit known to v0.1.
///
/// `h` is the hour (3600 s), not Planck's constant.
pub static UNITS: &[Unit] = &[
    unit("m", Dimension::LENGTH, 1.0),
    unit("km", Dimension::LENGTH, 1000.0),
    unit("s", Dimension::TIME, 1.0),
    unit("min", Dimension::TIME, 60.0),
    // `h` is the hour (3600 s), not Planck's constant. Physical constants
    // such as h and ħ are not units and will live in a separate table.
    unit("h", Dimension::TIME, 3600.0),
    unit("kg", Dimension::MASS, 1.0),
    unit("K", Dimension::TEMPERATURE, 1.0),
    unit("J", Dimension::ENERGY, 1.0),
    unit("eV", Dimension::ENERGY, ELECTRONVOLT_IN_JOULES),
    unit("pc", Dimension::LENGTH, PARSEC_IN_METRES),
    unit("AU", Dimension::LENGTH, ASTRONOMICAL_UNIT_IN_METRES),
    unit("rad", Dimension::DIMENSIONLESS, 1.0),
    // Derived SI units with special names, all coherent (scale 1).
    // `N` is the newton. It is unrelated to the base-dimension symbol N
    // (amount of substance) in `Dimension`'s display, and to a shape
    // variable written `N` in `Matrix[N, N]`: the parser only reads a unit
    // right after a numeric literal (`10 N`).
    unit("N", Dimension::FORCE, 1.0),
    unit("Pa", Dimension::PRESSURE, 1.0),
    unit("W", Dimension::POWER, 1.0),
    // `Hz` is T⁻¹, the same dimension as `rad/s` because `rad` is
    // dimensionless: a missing 2π between frequency and angular frequency is
    // not a dimension error. A known v0.1 limitation.
    unit("Hz", Dimension::FREQUENCY, 1.0),
];

/// Looks a unit up by its exact symbol (case-sensitive), e.g. `lookup("km")`.
///
/// This (together with [`is_unit_symbol`]) is the single source of truth for
/// which unit names exist: the parser uses it to decide whether an identifier
/// after a numeric literal is a unit, and the checker uses it to get that
/// unit's dimension and scale. Do not keep a second list anywhere else.
pub fn lookup(name: &str) -> Option<&'static Unit> {
    UNITS.iter().find(|u| u.name == name)
}

/// True if `name` is a unit symbol in [`UNITS`]; shorthand for
/// `lookup(name).is_some()`.
pub fn is_unit_symbol(name: &str) -> bool {
    lookup(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_units_share_a_dimension() {
        let m = lookup("m").unwrap();
        let km = lookup("km").unwrap();
        assert_eq!(m.dim, km.dim);
        assert_eq!(km.scale, 1000.0);
        assert_eq!(lookup("h").unwrap().scale, 3600.0);
        assert_eq!(lookup("min").unwrap().scale, 60.0);
    }

    #[test]
    fn exact_constants() {
        assert_eq!(lookup("eV").unwrap().scale, 1.602176634e-19);
        assert_eq!(lookup("eV").unwrap().dim, lookup("J").unwrap().dim);
        assert_eq!(lookup("AU").unwrap().scale, 149_597_870_700.0);
        let pc = lookup("pc").unwrap().scale;
        // IAU 2015: 1 pc = 3.085 677 581 491 367... e16 m.
        assert!((pc - 3.085_677_581_491_367e16).abs() / pc < 1e-15);
    }

    #[test]
    fn radian_is_dimensionless() {
        let rad = lookup("rad").unwrap();
        assert!(rad.dim.is_dimensionless());
        assert_eq!(rad.scale, 1.0);
    }

    #[test]
    fn table_is_exactly_the_v0_1_set() {
        let names: Vec<_> = UNITS.iter().map(|u| u.name).collect();
        assert_eq!(
            names,
            [
                "m", "km", "s", "min", "h", "kg", "K", "J", "eV", "pc", "AU", "rad", "N", "Pa",
                "W", "Hz"
            ]
        );
        for u in UNITS {
            assert!(u.scale.is_finite() && u.scale > 0.0, "{}", u.name);
        }
    }

    #[test]
    fn lookup_is_the_public_source_of_truth() {
        // Every table entry is reachable through the crate-root API, by its own symbol.
        for u in UNITS {
            let found = crate::lookup(u.name).expect(u.name);
            assert!(std::ptr::eq(found, u), "{}", u.name);
            assert!(crate::is_unit_symbol(u.name));
        }
        let km = crate::lookup("km").unwrap();
        assert_eq!(
            (km.name, km.dim, km.scale),
            ("km", Dimension::LENGTH, 1000.0)
        );
        // Case-sensitive, and neither base-dimension letters nor aliases are units.
        for not_unit in ["Km", "M", "L", "T", "meters", "kelvin", "°C", "km ", ""] {
            assert!(crate::lookup(not_unit).is_none(), "{not_unit:?}");
            assert!(!crate::is_unit_symbol(not_unit), "{not_unit:?}");
        }
    }

    fn dim(name: &str) -> Dimension {
        lookup(name).unwrap_or_else(|| panic!("{name}")).dim
    }

    #[test]
    fn derived_si_units_equal_their_si_composition() {
        let (m, kg, s) = (dim("m"), dim("kg"), dim("s"));
        let m2 = m.pow_int(2).unwrap();
        // N = kg·m·s⁻²
        let newton = kg
            .checked_mul(m)
            .unwrap()
            .checked_div(s.pow_int(2).unwrap())
            .unwrap();
        assert_eq!(dim("N"), newton);
        // Pa = N·m⁻²
        assert_eq!(dim("Pa"), dim("N").checked_div(m2).unwrap());
        // W = J·s⁻¹
        assert_eq!(dim("W"), dim("J").checked_div(s).unwrap());
        // Hz = s⁻¹
        assert_eq!(dim("Hz"), s.inverse().unwrap());
        // Cross-checks: N·m = J, W·s = J, Pa·m² = N.
        assert_eq!(dim("N").checked_mul(m).unwrap(), dim("J"));
        assert_eq!(dim("W").checked_mul(s).unwrap(), dim("J"));
        assert_eq!(dim("Pa").checked_mul(m2).unwrap(), dim("N"));
        for u in ["N", "Pa", "W", "Hz"] {
            assert_eq!(lookup(u).unwrap().scale, 1.0, "{u}");
        }
        assert_eq!(dim("N").name(), Some("force"));
        assert_eq!(dim("Pa").name(), Some("pressure"));
        assert_eq!(dim("W").name(), Some("power"));
        assert_eq!(dim("Hz").name(), Some("frequency"));
    }

    #[test]
    fn hertz_and_radians_per_second_share_a_dimension() {
        // Known v0.1 limitation, not a bug: rad is dimensionless, so a
        // missing 2π between Hz and rad/s is not a dimension error.
        let rad_per_s = dim("rad").checked_div(dim("s")).unwrap();
        assert_eq!(dim("Hz"), rad_per_s);
    }

    /// Shown by the guard tests below when they fail.
    const DISPLAY_COLLISION: &str = "if you add `mol` or `cd`, rename the `N`/`J` dimension \
         display symbols; they collide with newton/joule";

    #[test]
    fn newton_is_not_the_amount_dimension() {
        // The unit `N` (force) and the base-dimension symbol `N` (amount of
        // substance, shown by `Dimension`'s display) are different things.
        assert_ne!(dim("N"), Dimension::AMOUNT, "{DISPLAY_COLLISION}");
        assert_ne!(dim("J"), Dimension::LUMINOSITY, "{DISPLAY_COLLISION}");
        assert_eq!(Dimension::AMOUNT.to_string(), "N", "{DISPLAY_COLLISION}");
        assert_eq!(
            Dimension::LUMINOSITY.to_string(),
            "J",
            "{DISPLAY_COLLISION}"
        );
        assert_eq!(dim("N").to_string(), "L·M·T⁻²");
        assert_eq!(dim("J").to_string(), "L²·M·T⁻²");
    }

    /// Fails as soon as the table gains a unit (such as `mol` or `cd`)
    /// whose dimension involves amount of substance or luminous intensity,
    /// because error text would then show the bare symbols `N` and `J`,
    /// which read as newton and joule.
    #[test]
    fn no_unit_exposes_the_n_or_j_display_symbols() {
        use crate::dimension::BaseDim;
        for u in UNITS {
            for b in [BaseDim::Amount, BaseDim::Luminosity] {
                assert!(
                    u.dim.exponent(b).is_zero(),
                    "{DISPLAY_COLLISION} (unit `{}` has dimension {})",
                    u.name,
                    u.dim
                );
            }
        }
    }

    #[test]
    fn no_affine_units_or_aliases() {
        for missing in ["°C", "degC", "meters", "kelvin", "KM", ""] {
            assert!(lookup(missing).is_none(), "{missing}");
        }
    }
}
