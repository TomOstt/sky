//! Units as scientists write them, from the dogfooding reports: grouped
//! denominators (`W/(m*K)`), `1/m`, angular frequencies that must not
//! print as `Hz`, and spellings that are not what they look like (`ft`,
//! `°C`, `1 in`). Every program runs on every engine and prints the same.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("unit-spelling-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// `sky <args> prog.sky` on `src`, in a scratch directory.
fn sky(name: &str, args: &[&str], src: &str) -> Output {
    let dir = scratch(name);
    fs::write(dir.join("prog.sky"), src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg("prog.sky")
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

/// Runs `src` on every engine and checks each prints `expected`.
fn runs_everywhere(name: &str, src: &str, expected: &str) {
    for engine in ["ast", "sir", "native", "auto"] {
        let flag = format!("--engine={engine}");
        let o = sky(&format!("{name}-{engine}"), &["run", &flag], src);
        assert!(o.status.success(), "{engine}: {}", err(&o));
        assert_eq!(out(&o), expected, "{engine}");
    }
}

/// `sky check` on `src` fails, and its stderr contains every needle.
fn rejected(name: &str, src: &str, needles: &[&str]) {
    let o = sky(name, &["check"], src);
    assert!(!o.status.success(), "{name}: accepted");
    let e = err(&o);
    for needle in needles {
        assert!(e.contains(needle), "{name}: no {needle:?} in\n{e}");
    }
}

#[test]
fn grouped_denominators_and_one_over_a_unit() {
    runs_everywhere(
        "grouped",
        "k = 237 W/(m*K)\n\
         print(k)\n\
         print(k == 237 W/m/K)\n\
         Gc = 6.674e-11 m^3/(kg*s^2)\n\
         print(Gc as m^3/(kg*s^2))\n\
         print(8.314 J/(mol*K))\n\
         q = 0.001 1/m\n\
         print(q, q == 0.001 m^-1)\n\
         def f(kk: 1/m, x: m) -> Scalar:\n    return kk * x\n\
         print(f(2 1/m, 3 m))\n\
         v: Vector[2] 1/s = [1 1/s, 2 1/s]\n\
         print(v)\n\
         print(2 Hz as 1/min)\n",
        "237 kg*m/s^3/K\n\
         true\n\
         6.674e-11 m^3/(kg*s^2)\n\
         8.314 kg*m^2/mol/s^2/K\n\
         0.001 m^-1 true\n\
         6\n\
         [1 s^-1, 2 s^-1]\n\
         120 1/min\n",
    );
}

#[test]
fn angular_frequency_does_not_print_as_hertz() {
    // `rad` is dimensionless, so `rad/s` and `Hz` share a dimension; the
    // value prints as `s^-1`, which is right for both, and `as` picks one.
    runs_everywhere(
        "omega",
        "w = 3.9 rad/s\n\
         print(w)\n\
         print(w as rad/s)\n\
         print(50 Hz, 50 Hz as Hz)\n\
         print((2 s)⁻¹)\n",
        "3.9 s^-1\n\
         3.9 rad/s\n\
         50 s^-1 50 Hz\n\
         0.5 s^-1\n",
    );
}

#[test]
fn a_rate_constant_prints_per_mole_first() {
    runs_everywhere(
        "rate",
        "print(3e7 L/mol/s)\nprint(2 mol/L / 1 s)\n",
        "30000 m^3/mol/s\n2000 mol/m^3/s\n",
    );
}

#[test]
fn feet_are_not_femtotonnes() {
    rejected(
        "ft",
        "x = 3 ft + 2 m\n",
        &[
            "unknown unit `ft`",
            "`ft` would be femto-tonne",
            "`0.3048 m`",
        ],
    );
    rejected(
        "dt",
        "dt = 0.1 s\nv = 4 m/dt\n",
        &[
            "unknown unit `dt`",
            "put spaces around the operator (`x / dt`)",
        ],
    );
    rejected("molar", "c = 2 M\n", &["unknown unit `M`", "`2 mol/L`"]);
}

#[test]
fn temperatures_and_inches_explain_themselves() {
    rejected(
        "celsius",
        "T = 25 °C\n",
        &["unexpected character '°'", "write 25 °C as `298.15 K`"],
    );
    rejected(
        "degc",
        "T = 25 degC\n",
        &["unknown unit `degC`", "`298.15 K`"],
    );
    rejected(
        "angle",
        "a = 30°\n",
        &["unexpected character '°'", "`30 deg`"],
    );
    rejected(
        "inch",
        "L = 1 in\n",
        &["`in` is a keyword, not a unit", "`0.0254 m`"],
    );
}

#[test]
fn dimension_messages_say_mol_and_dimensionless() {
    rejected("amount", "x: mol/m^3/s = 1 mol/m^3\n", &["mol"]);
    let o = sky("amount-text", &["check"], "x: mol/m^3/s = 1 mol/m^3\n");
    assert!(!err(&o).contains("·N"), "{}", err(&o));
    rejected(
        "dimensionless",
        "def step(u: Vector[2]) -> Vector[2]:\n    return u\nprint(step([1 K, 2 K]))\n",
        &["Vector[2] [dimensionless]", "Vector[2] [Θ]"],
    );
}
