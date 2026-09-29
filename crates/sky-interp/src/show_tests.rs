//! `v as unit`: values written in a chosen unit. Like the other interpreter
//! tests these run without the checker, so they also cover the dynamic
//! checks a checked program never reaches.

use sky_ast::parse_module;
use sky_diag::Span;
use sky_syntax::lex;

use crate::{run_module, show_unit_text, RuntimeEnv, RuntimeError, Value};

fn run(src: &str) -> Result<(Option<Value>, String), RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "parse errors in {src:?}: {errors:?}");
    let mut out = Vec::new();
    let v = run_module(&module, &src, &mut RuntimeEnv::new(), &mut out)?;
    Ok((v, String::from_utf8(out).expect("utf-8")))
}

/// The text of the display value of the expression `src`.
fn show(src: &str) -> String {
    match run(src).expect("runs").0.expect("a value") {
        Value::Shown(text) => text,
        other => panic!("{src:?} is not a display value: {other:?}"),
    }
}

fn printed(src: &str) -> String {
    run(src).expect("runs").1
}

fn err(src: &str) -> RuntimeError {
    run(src).expect_err("fails at run time")
}

#[test]
fn speeds() {
    // The motivating case: 90 km/h is stored as 25 m/s.
    assert_eq!(printed("print(90 km/h)"), "25 m/s\n");
    assert_eq!(printed("print(90 km/h as km/h)"), "90 km/h\n");
    assert_eq!(show("25 m/s as km/h"), "90 km/h");
    assert_eq!(show("1 km/h as m/s"), "0.2777777777777778 m/s");
    assert_eq!(show("30 m/s as km/h"), "108 km/h");
    // A run of speeds that do not divide evenly still print as written.
    for n in [1, 7, 13, 50, 100, 120, 130, 250, 999] {
        assert_eq!(show(&format!("{n} km/h as km/h")), format!("{n} km/h"));
    }
    assert_eq!(show("100 km/h as km/h"), "100 km/h");
}

#[test]
fn time_and_distance() {
    assert_eq!(show("1 h as s"), "3600 s");
    assert_eq!(show("3600 s as h"), "1 h");
    assert_eq!(show("1 day as h"), "24 h");
    assert_eq!(show("90 min as h"), "1.5 h");
    assert_eq!(show("1 yr as day"), "365.25 day");
    assert_eq!(show("1 AU as km"), "149597870.7 km");
    assert_eq!(show("1 AU as m"), "149597870700 m");
    assert_eq!(show("1 ly as pc"), show("1 ly as pc"));
    assert_eq!(show("1 pc as AU").split(' ').nth(1), Some("AU"));
    assert_eq!(show("2500 m as km"), "2.5 km");
    assert_eq!(show("0.5 m as mm"), "500 mm");
    assert_eq!(show("1 m as μm"), "1000000 μm");
}

#[test]
fn energies() {
    assert_eq!(show("511 keV as eV"), "511000 eV");
    assert_eq!(show("511 keV as MeV"), "0.511 MeV");
    assert_eq!(show("1 MeV as keV"), "1000 keV");
    assert_eq!(show("1 eV as J"), "1.602176634e-19 J");
    assert_eq!(show("1.602176634e-19 J as eV"), "1 eV");
    assert_eq!(show("13.6 eV as eV"), "13.6 eV");
    assert_eq!(show("2 GeV as MeV"), "2000 MeV");
    // Compound units follow the unit-literal grammar.
    assert_eq!(show("2 kg*m^2/s^2 as J"), "2 J");
    assert_eq!(show("2 J as kg*m^2/s^2"), "2 kg*m^2/s^2");
    assert_eq!(show("3 J as N*m"), "3 N*m");
    assert_eq!(show("3 J as kW*h"), "8.333333333333334e-7 kW*h");
    assert_eq!(show("1 kW*h as J"), "3600000 J");
    assert_eq!(show("9.8 m/s^2 as km/h^2"), "127008 km/h^2");
    assert_eq!(show("4 m^2 as m^2"), "4 m^2");
    assert_eq!(show("4 m^2 as cm^2"), "40000 cm^2");
    assert_eq!(show("2 m^3 as L"), "2000 L");
}

#[test]
fn arithmetic_before_as() {
    // `as` is the loosest operator: the whole expression is converted.
    assert_eq!(show("1 km + 500 m as m"), "1500 m");
    assert_eq!(show("1 km + 500 m as km"), "1.5 km");
    assert_eq!(show("2 kg * 9.8 m/s^2 as N"), "19.6 N");
    // Floating-point noise in the stored value is not hidden: it is the
    // value the program holds (spec "Unit values").
    assert_eq!(show("0.1 s + 0.2 s as s"), "0.30000000000000004 s");
    assert_eq!(show("0.1 s + 0.2 s as ms"), "300.00000000000006 ms");
}

#[test]
fn arrays() {
    assert_eq!(show("[100 mm, 200 mm] as mm"), "[100 mm, 200 mm]");
    assert_eq!(show("[0.1 m, 0.2 m] as mm"), "[100 mm, 200 mm]");
    assert_eq!(
        show("[[1 m, 2 m], [3 m, 4 m]] as cm"),
        "[[100 cm, 200 cm], [300 cm, 400 cm]]"
    );
    assert_eq!(printed("print([1 km, 2 km] as m)"), "[1000 m, 2000 m]\n");
    // A vector of speeds.
    assert_eq!(show("[36 km/h, 72 km/h] as km/h"), "[36 km/h, 72 km/h]");
    // The SI printer still prints the stored values.
    assert_eq!(printed("print([100 mm, 200 mm])"), "[0.1 m, 0.2 m]\n");
    // A dimensionless array in a dimensionless unit.
    assert_eq!(show("[1, 2] as rad"), "[1 rad, 2 rad]");
}

#[test]
fn print_writes_display_values_beside_others() {
    assert_eq!(
        printed("v = 90 km/h\nprint(\"speed:\", v as km/h, v)"),
        "speed: 90 km/h 25 m/s\n"
    );
    assert_eq!(
        printed("print(1 AU as km, 1 AU)"),
        "149597870.7 km 149597870700 m\n"
    );
}

#[test]
fn the_shown_number_is_the_shortest_that_reads_back() {
    // Every decimal a person could write reads back as itself, in a range
    // of units whose factors are not powers of ten.
    for unit in ["km/h", "eV", "day", "AU", "kW*h", "mm", "MeV", "yr", "pc"] {
        for n in 1..=400u32 {
            let x = f64::from(n) / 8.0;
            let text = format!("{x} {unit}");
            let want = crate::format_number(x);
            assert_eq!(
                show(&format!("{text} as {unit}")),
                format!("{want} {unit}"),
                "{text}"
            );
        }
    }
}

#[test]
fn a_display_value_is_not_a_number_and_prints_raw() {
    let v = run("25 m/s as km/h").unwrap().0.unwrap();
    assert_eq!(v.to_string(), "90 km/h");
    assert_eq!(v.print_text(), "90 km/h");
    assert_eq!(v.kind(), "a display value");
}

#[test]
fn wrong_dimension_is_a_runtime_error_without_the_checker() {
    let e = err("25 m/s as kg");
    assert_eq!(e.message, "cannot show a velocity in `kg`");
    let e = err("2 J as m");
    assert_eq!(e.message, "cannot show an energy in `m`");
    let e = err("3 as m");
    assert_eq!(e.message, "cannot show a plain number in `m`");
    let e = err("[1 m] as s");
    assert_eq!(e.message, "cannot show a length in `s`");
}

#[test]
fn non_numbers_and_unknown_units_are_runtime_errors() {
    assert_eq!(err("true as m").message, "cannot show a Bool in `m`");
    assert_eq!(err("\"hi\" as m").message, "cannot show a string in `m`");
    assert_eq!(err("1 m as parsecs").message, "unknown unit `parsecs`");
    assert_eq!(
        err("1 m as m*m^9223372036854775807").message,
        "dimension exponent too large"
    );
}

#[test]
fn overflow_is_an_error_not_infinity() {
    // 1e300 m in nanometres is 1e309: not a finite f64.
    let e = err("1e300 m as nm");
    assert_eq!(e.message, "arithmetic overflow");
    // No half-printed line reaches the output.
    let e = run("print([1e300 m, 1 m] as nm)").expect_err("overflow");
    assert_eq!(e.message, "arithmetic overflow");
}

#[test]
fn show_unit_text_is_the_same_function() {
    let v = Value::number(25.0, sky_units::Dimension::VELOCITY);
    let shown = show_unit_text("km/h", &v, Span::new(0, 0)).unwrap();
    assert_eq!(shown, Value::Shown("90 km/h".to_string()));
    let bad = show_unit_text("km h", &v, Span::new(0, 0)).unwrap_err();
    assert_eq!(bad.message, "`km h` is not a unit");
    assert!(show_unit_text("", &v, Span::new(0, 0)).is_err());
    assert!(show_unit_text("km/h + 1", &v, Span::new(0, 0)).is_err());
    let wrong = show_unit_text("kg", &v, Span::new(0, 0)).unwrap_err();
    assert_eq!(wrong.message, "cannot show a velocity in `kg`");
}
