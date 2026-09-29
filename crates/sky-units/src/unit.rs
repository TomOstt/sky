//! The v0.1 unit table: each unit is a dimension plus a scale factor to the
//! coherent SI unit of that dimension (so `km` is length with scale 1000).
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
pub static UNITS: &[Unit] = &[
    unit("m", Dimension::LENGTH, 1.0),
    unit("km", Dimension::LENGTH, 1000.0),
    unit("s", Dimension::TIME, 1.0),
    unit("min", Dimension::TIME, 60.0),
    unit("h", Dimension::TIME, 3600.0),
    unit("kg", Dimension::MASS, 1.0),
    unit("K", Dimension::TEMPERATURE, 1.0),
    unit("J", Dimension::ENERGY, 1.0),
    unit("eV", Dimension::ENERGY, ELECTRONVOLT_IN_JOULES),
    unit("pc", Dimension::LENGTH, PARSEC_IN_METRES),
    unit("AU", Dimension::LENGTH, ASTRONOMICAL_UNIT_IN_METRES),
    unit("rad", Dimension::DIMENSIONLESS, 1.0),
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
            ["m", "km", "s", "min", "h", "kg", "K", "J", "eV", "pc", "AU", "rad"]
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

    #[test]
    fn no_affine_units_or_aliases() {
        for missing in ["°C", "degC", "meters", "kelvin", "KM", ""] {
            assert!(lookup(missing).is_none(), "{missing}");
        }
    }
}
