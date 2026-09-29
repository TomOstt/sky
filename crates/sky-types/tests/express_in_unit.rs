//! `express_in_unit` is the inverse of `apply_unit`: same unit table, same
//! factors, opposite direction.

use sky_ast::{parse_expr, ExprKind, UnitExpr};
use sky_types::{apply_unit, express_in_unit};

fn unit(text: &str) -> (String, UnitExpr) {
    let src = format!("1 {text}");
    let lexed = sky_syntax::lex(&src).expect("lexes");
    let (expr, errors) = parse_expr(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?}");
    match expr.expect("expression").kind {
        ExprKind::UnitLiteral { unit, .. } => (src, unit),
        other => panic!("not a unit literal: {other:?}"),
    }
}

#[test]
fn undoes_the_factors_in_reverse_order() {
    let (src, u) = unit("km/h");
    // 90 km/h is 25 m/s, and 25 m/s is exactly 90 km/h: 25 * 3600 / 1000.
    assert_eq!(apply_unit(&src, &u, 90.0).unwrap().0, 25.0);
    assert_eq!(express_in_unit(&src, &u, 25.0).unwrap().0, 90.0);
}

#[test]
fn both_directions_agree_on_the_dimension() {
    for text in ["m", "km/h", "kg*m^2/s^2", "eV", "m^(1/2)", "s^-2", "AU/yr"] {
        let (src, u) = unit(text);
        let (_, to) = apply_unit(&src, &u, 1.0).unwrap();
        let (_, from) = express_in_unit(&src, &u, 1.0).unwrap();
        assert_eq!(to, from, "{text}");
    }
}

#[test]
fn round_trips_within_rounding() {
    for text in [
        "m", "km", "mm", "km/h", "kg*m^2/s^2", "eV", "MeV", "day", "AU", "pc", "m^(1/2)", "s^-2",
        "m^2", "kW*h", "N*m", "AU/yr",
    ] {
        let (src, u) = unit(text);
        for x in [0.001, 1.0, 3.5, 90.0, 12345.678] {
            let si = apply_unit(&src, &u, x).unwrap().0;
            let back = express_in_unit(&src, &u, si).unwrap().0;
            assert!(
                ((back - x) / x).abs() < 1e-14,
                "{x} {text}: {si} SI is {back}"
            );
        }
    }
}

#[test]
fn unknown_units_are_errors_in_both_directions() {
    let (src, u) = unit("parsecs");
    assert!(apply_unit(&src, &u, 1.0).is_err());
    assert_eq!(
        express_in_unit(&src, &u, 1.0),
        apply_unit(&src, &u, 1.0),
        "same error"
    );
}
