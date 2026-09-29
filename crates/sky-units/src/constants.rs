//! Physical constants (CODATA 2018), as predefined names in scope.
//!
//! A constant is a value with a dimension, not a unit: `E = m * c^2` works
//! because `c` is a name bound to a scalar of dimension L·T⁻¹, while `10 c`
//! is an error (a unit may only follow a number, and `c` is not one). The
//! checker binds every entry of [`CONSTANTS`] before it walks a program
//! (`Env::with_prelude` in `sky-types`).
//!
//! Values are CODATA 2018 (Tiesinga et al., Rev. Mod. Phys. 93, 025010,
//! 2021). The seven SI defining constants and the values derived from them
//! alone are exact (SI Brochure, 9th edition, Table 1); the rest carry
//! measurement uncertainty, and [`Constant::exact`] says which.
//!
//! # Naming
//!
//! * Every Unicode name has an ASCII spelling ([`Constant::ascii`]), as the
//!   spec requires, and both are bound. Unicode names must be NFKC or the
//!   lexer rejects them, which is why `ħ` is U+0127 (`ℏ` U+210F is not NFKC)
//!   and why the permittivity and permeability are `ε0` and `μ0`, not `ε₀`
//!   and `μ₀`: a subscript zero normalises to `0`, so `ε₀` cannot be written
//!   in a Sky identifier at all.
//! * The elementary charge is `q_e`, not `e`. Bare `e` is too useful as a
//!   variable (eccentricity, error, unit vector) and is the natural name for
//!   Euler's number once Sky has math functions; `q_e` pairs with `m_e`, `m_p`
//!   and is unambiguous.
//! * The Stefan-Boltzmann constant is `σ_SB` (ASCII `sigma_SB`), not `σ`: a
//!   bare `σ` is a standard deviation, stress, or cross section in almost
//!   every physics program.
//! * `c` and `h` are the standard symbols and are kept. Both are common
//!   variable names too, so rebinding a constant is allowed but warned about
//!   once (see the checker); the hour `h` is a unit, which never conflicts,
//!   because units appear only after a number.

use crate::dimension::Dimension;
use crate::unit::{ELECTRONVOLT_IN_JOULES, SPEED_OF_LIGHT};

/// A physical constant: a name in scope with a dimension and an SI value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Constant {
    /// The name as written in Sky (Unicode where the physics symbol is).
    pub name: &'static str,
    /// The ASCII spelling, when `name` is not ASCII.
    pub ascii: Option<&'static str>,
    /// The constant's dimension.
    pub dim: Dimension,
    /// The CODATA 2018 value in coherent SI units.
    pub value: f64,
    /// True when the value has no uncertainty (SI defining constants and
    /// values derived from them).
    pub exact: bool,
    /// What it is, in words, for diagnostics ("the speed of light").
    pub description: &'static str,
}

const fn constant(
    name: &'static str,
    ascii: Option<&'static str>,
    dim: Dimension,
    value: f64,
    exact: bool,
    description: &'static str,
) -> Constant {
    Constant {
        name,
        ascii,
        dim,
        value,
        exact,
        description,
    }
}

/// The Planck constant in J·s (exact; SI Brochure, 9th edition, Table 1).
pub const PLANCK: f64 = 6.626_070_15e-34;

const ACTION: Dimension = Dimension::from_ints([2, 1, -1, 0, 0, 0, 0]);

/// Every predefined constant, in the order the spec lists them.
pub static CONSTANTS: &[Constant] = &[
    constant(
        "c",
        None,
        Dimension::VELOCITY,
        SPEED_OF_LIGHT,
        true,
        "the speed of light",
    ),
    constant("h", None, ACTION, PLANCK, true, "the Planck constant"),
    // ħ = h/2π, given as CODATA 2018 lists it (1.054 571 817 × 10⁻³⁴ J·s; exact
    // given h, and a test checks it agrees with h/2π to 9 digits).
    constant(
        "ħ",
        Some("hbar"),
        ACTION,
        1.054_571_817e-34,
        true,
        "the reduced Planck constant",
    ),
    constant(
        "k_B",
        None,
        Dimension::from_ints([2, 1, -2, 0, -1, 0, 0]),
        1.380_649e-23,
        true,
        "the Boltzmann constant",
    ),
    constant(
        "N_A",
        None,
        Dimension::from_ints([0, 0, 0, 0, 0, -1, 0]),
        6.022_140_76e23,
        true,
        "the Avogadro constant",
    ),
    constant(
        "q_e",
        None,
        Dimension::CHARGE,
        ELECTRONVOLT_IN_JOULES,
        true,
        "the elementary charge",
    ),
    constant(
        "G",
        None,
        Dimension::from_ints([3, -1, -2, 0, 0, 0, 0]),
        6.674_30e-11,
        false,
        "the Newtonian constant of gravitation",
    ),
    constant(
        "ε0",
        Some("eps0"),
        Dimension::from_ints([-3, -1, 4, 2, 0, 0, 0]),
        8.854_187_812_8e-12,
        false,
        "the vacuum electric permittivity",
    ),
    constant(
        "μ0",
        Some("mu0"),
        Dimension::from_ints([1, 1, -2, -2, 0, 0, 0]),
        1.256_637_062_12e-6,
        false,
        "the vacuum magnetic permeability",
    ),
    constant(
        "m_e",
        None,
        Dimension::MASS,
        9.109_383_701_5e-31,
        false,
        "the electron mass",
    ),
    constant(
        "m_p",
        None,
        Dimension::MASS,
        1.672_621_923_69e-27,
        false,
        "the proton mass",
    ),
    constant(
        "m_n",
        None,
        Dimension::MASS,
        1.674_927_498_04e-27,
        false,
        "the neutron mass",
    ),
    constant(
        "m_u",
        None,
        Dimension::MASS,
        1.660_539_066_60e-27,
        false,
        "the atomic mass constant",
    ),
    // σ = 2π⁵k⁴/(15h³c²) is exact given the defining constants; CODATA 2018
    // lists 5.670 374 419... × 10⁻⁸ W·m⁻²·K⁻⁴.
    constant(
        "σ_SB",
        Some("sigma_SB"),
        Dimension::from_ints([0, 1, -3, 0, -4, 0, 0]),
        5.670_374_419e-8,
        true,
        "the Stefan-Boltzmann constant",
    ),
    // Standard acceleration of gravity, exact by definition (CGPM 1901;
    // SI Brochure, 9th edition, Table 8 footnote).
    constant(
        "g_n",
        None,
        Dimension::from_ints([1, 0, -2, 0, 0, 0, 0]),
        9.806_65,
        true,
        "the standard acceleration of gravity",
    ),
];

/// Looks a constant up by its Unicode or ASCII spelling (case-sensitive).
pub fn lookup_constant(name: &str) -> Option<&'static Constant> {
    CONSTANTS
        .iter()
        .find(|c| c.name == name || c.ascii == Some(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(name: &str) -> f64 {
        lookup_constant(name)
            .unwrap_or_else(|| panic!("{name}"))
            .value
    }

    fn dim(name: &str) -> Dimension {
        lookup_constant(name)
            .unwrap_or_else(|| panic!("{name}"))
            .dim
    }

    #[test]
    fn names_are_unique_and_every_unicode_name_has_an_ascii_spelling() {
        let mut seen = Vec::new();
        for c in CONSTANTS {
            for s in [Some(c.name), c.ascii].into_iter().flatten() {
                assert!(!seen.contains(&s), "duplicate {s}");
                seen.push(s);
            }
            assert_eq!(c.name.is_ascii(), c.ascii.is_none(), "{}", c.name);
            if let Some(a) = c.ascii {
                assert!(a.is_ascii(), "{a}");
            }
            assert!(c.value.is_finite() && c.value > 0.0, "{}", c.name);
            assert!(!c.description.is_empty());
        }
        assert_eq!(lookup_constant("hbar"), lookup_constant("ħ"));
        assert_eq!(lookup_constant("sigma_SB"), lookup_constant("σ_SB"));
        assert_eq!(lookup_constant("eps0"), lookup_constant("ε0"));
        assert_eq!(lookup_constant("mu0"), lookup_constant("μ0"));
        // Not bare `e` (Euler), not bare `σ`, and no wrong-case spellings.
        for not_constant in ["e", "σ", "ℏ", "ε₀", "μ₀", "C", "H", "K_B", "hBar", ""] {
            assert!(lookup_constant(not_constant).is_none(), "{not_constant}");
        }
    }

    #[test]
    fn codata_2018_values() {
        assert_eq!(value("c"), 299_792_458.0);
        assert_eq!(value("h"), 6.626_070_15e-34);
        assert_eq!(value("k_B"), 1.380_649e-23);
        assert_eq!(value("N_A"), 6.022_140_76e23);
        assert_eq!(value("q_e"), 1.602_176_634e-19);
        assert_eq!(value("G"), 6.674_30e-11);
        assert_eq!(value("g_n"), 9.806_65);
        assert_eq!(value("m_e"), 9.109_383_701_5e-31);
        assert_eq!(value("m_p"), 1.672_621_923_69e-27);
        let hbar = value("ħ");
        assert!((hbar - 1.054_571_817e-34).abs() / hbar < 1e-9);
        for (name, exact) in [
            ("c", true),
            ("h", true),
            ("ħ", true),
            ("k_B", true),
            ("N_A", true),
            ("q_e", true),
            ("σ_SB", true),
            ("g_n", true),
            ("G", false),
            ("ε0", false),
            ("μ0", false),
            ("m_e", false),
            ("m_p", false),
        ] {
            assert_eq!(lookup_constant(name).unwrap().exact, exact, "{name}");
        }
    }

    #[test]
    fn dimensions_agree_with_physics() {
        let mul = |x: Dimension, y: Dimension| x.checked_mul(y).unwrap();
        let div = |x: Dimension, y: Dimension| x.checked_div(y).unwrap();
        // E = h ν, E = k_B T, E = m c², E = q_e V: all energy.
        assert_eq!(mul(dim("h"), Dimension::FREQUENCY), Dimension::ENERGY);
        assert_eq!(mul(dim("k_B"), Dimension::TEMPERATURE), Dimension::ENERGY);
        let c2 = dim("c").pow_int(2).unwrap();
        assert_eq!(mul(dim("m_e"), c2), Dimension::ENERGY);
        assert_eq!(mul(dim("q_e"), Dimension::VOLTAGE), Dimension::ENERGY);
        // F = G m² / r², Coulomb: F = q² / (4π ε0 r²), Ampère: F/L = μ0 I² / (2π r).
        let r2 = Dimension::LENGTH.pow_int(2).unwrap();
        let m2 = Dimension::MASS.pow_int(2).unwrap();
        assert_eq!(div(mul(dim("G"), m2), r2), Dimension::FORCE);
        let q2 = dim("q_e").pow_int(2).unwrap();
        assert_eq!(div(q2, mul(dim("ε0"), r2)), Dimension::FORCE);
        // c² ε0 μ0 = 1.
        assert!(mul(mul(c2, dim("ε0")), dim("μ0")).is_dimensionless());
        // Stefan-Boltzmann: P/A = σ T⁴.
        let t4 = Dimension::TEMPERATURE.pow_int(4).unwrap();
        assert_eq!(
            mul(dim("σ_SB"), t4),
            div(Dimension::POWER, r2),
            "power per area"
        );
        // N_A n = count.
        assert!(mul(dim("N_A"), Dimension::AMOUNT).is_dimensionless());
        // ħ has the dimension of angular momentum, h as well.
        assert_eq!(dim("ħ"), dim("h"));
        assert_eq!(dim("g_n"), div(Dimension::VELOCITY, Dimension::TIME));
    }

    #[test]
    fn derived_values_are_consistent() {
        let close = |a: f64, b: f64, tol: f64| (a - b).abs() / b.abs() < tol;
        let (c, h, k) = (value("c"), value("h"), value("k_B"));
        // c² ε0 μ0 = 1 to the precision of the measured pair.
        assert!(close(c * c * value("ε0") * value("μ0"), 1.0, 1e-9));
        // σ = 2π⁵ k⁴ / (15 h³ c²).
        let pi = std::f64::consts::PI;
        let sigma = 2.0 * pi.powi(5) * k.powi(4) / (15.0 * h.powi(3) * c * c);
        assert!(close(sigma, value("σ_SB"), 1e-9));
        assert!(close(value("ħ"), h / (2.0 * pi), 1e-9));
        // m_p is about 1836 m_e.
        assert!(close(value("m_p") / value("m_e"), 1_836.152_673_43, 1e-9));
        assert!(close(value("m_u"), 1.660_539_066_60e-27, 1e-15));
    }

    /// The names must survive the lexer, which rejects non-NFKC identifiers
    /// and keywords: every spelling lexes as one identifier.
    #[test]
    fn every_spelling_lexes_as_one_identifier() {
        for c in CONSTANTS {
            for s in [Some(c.name), c.ascii].into_iter().flatten() {
                let src = format!("{s} = 1\n");
                let lexed = sky_syntax::lex(&src).unwrap_or_else(|e| panic!("{s}: {e:?}"));
                assert!(lexed.warnings.is_empty(), "{s}");
                let (module, errors) = sky_ast::parse_module(&lexed.tokens, &src);
                assert!(errors.is_empty(), "{s}: {errors:?}");
                let sky_ast::StmtKind::Assign {
                    target: sky_ast::AssignTarget::Name(id),
                    ..
                } = &module.stmts[0].kind
                else {
                    panic!("{s}: not a name assignment");
                };
                assert_eq!(sky_ast::ast::span_text(&src, id.span), Some(s));
            }
        }
    }

    #[test]
    fn the_unicode_spellings_the_spec_warns_about_do_not_lex() {
        // `ℏ` (PLANCK CONSTANT OVER TWO PI) and subscript digits normalise to
        // something else, so they are not identifiers; the ASCII spelling or
        // the NFKC form is what users are told to write.
        for bad in ["ℏ", "ε₀", "μ₀", "\u{00B5}0"] {
            let src = format!("{bad} = 1\n");
            assert!(sky_syntax::lex(&src).is_err(), "{bad:?} should not lex");
        }
    }
}
