//! The unit table: each unit is a dimension plus a scale factor to the
//! coherent SI unit of that dimension, and a flag saying whether SI prefixes
//! apply to it.
//!
//! # Prefixes
//!
//! Only units marked `prefixable` take a prefix (`km`, `μs`, `GeV`, `kpc`);
//! `min`, `h`, `day`, and `AU` do not. [`lookup`] resolves a name in two
//! steps: an exact table name (or ASCII alias) always wins; only if there is
//! none is the name split into a prefix and a prefixable unit. The gram is
//! the prefixable mass unit, so `kg` is `k` + `g`, and it is still the
//! coherent SI unit (its scale is exactly 1).
//!
//! Exact-wins settles every case where a prefix reading is also possible or
//! plausible:
//!
//! * `min` is the minute, never `m` + `in` (milli-inch).
//! * `h` is the hour; hecto is only a prefix (`hPa`, `hm`), and the hour does
//!   not take prefixes, so there is no `hh`.
//! * `pc` is the parsec, not pico-`c`; `Mpc` is mega-parsec.
//! * `Pa` is the pascal, not peta-`a`; `hPa` is hectopascal.
//! * `cd` is the candela, not centi-day (there is no unit `d`; the day is
//!   `day`).
//! * `T` is the tesla and also the tera prefix (`TT`, `Tm`, `mT`); `t` is the
//!   tonne (`kt`, `Mt`).
//! * `da` is the deca prefix, so `dam` is decametres; alone it is not a
//!   unit. `day` is a table name and so never `da` + `y`.
//! * `ms`, `mm`, `nm` are prefix + unit (`m` alone is the metre). `Gyr` is
//!   giga-year; `Gy` is not a unit (there is no gray), and the checker
//!   suggests `Gyr`.
//!
//! The test `every_spelling_is_unambiguous` enumerates every prefix and unit
//! spelling to prove no other collision exists.
//!
//! # Spellings that are not units
//!
//! A few prefix readings are absurd units that people write for something
//! else: `ft` would be femto-tonne, but everyone means the foot. Sky does
//! not guess. These spellings ([`NOT_UNITS`]) are not units at all, so
//! `3 ft + 2 m` is an "unknown unit" error whose help says what `ft` would
//! have meant and what to write instead, never a silent mass. They are
//! `ft` (foot), `pt` (pint, point), `ct` (carat), `qt` (quart), `at`
//! (technical atmosphere), `dt` (usually a time-step variable written
//! without spaces, `x/dt`), and `myr` (milli-year, but people mean `Myr`).
//! The test `no_prefix_reading_hides_a_common_unit` pins the list.
//!
//! Lookup is allocation-free and never panics: it only compares string
//! prefixes with `strip_prefix`, which respects character boundaries.
//!
//! Because `rad` is dimensionless, `Hz` and `rad/s` have the same dimension,
//! so a missing 2π between frequency and angular frequency is not caught (a
//! known v0.1 limitation). For the same reason a value of that dimension
//! never prints as `Hz` ([`si_unit_text`]): it may be an angular frequency.
//!
//! Affine units such as °C are deliberately absent in v0.1. Full-name
//! aliases (`meters`, `kelvin`) are not units; [`suggest_unit`] turns them
//! into a "did you mean" hint.

use crate::dimension::Dimension;

/// A base unit in the table (one that prefixes may attach to).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Unit {
    /// The unit symbol, e.g. `"m"` (Unicode where the SI symbol is, like `"Ω"`).
    pub name: &'static str,
    /// An ASCII spelling, when `name` is not ASCII (`Ω` is also `ohm`).
    pub ascii: Option<&'static str>,
    /// The unit's dimension.
    pub dim: Dimension,
    /// Value of one of this unit in the coherent SI unit of `dim`.
    pub scale: f64,
    /// Whether SI prefixes may be attached (`km` yes, `kmin` no).
    pub prefixable: bool,
    /// `Some(e)` when `scale` is exactly `10^e`, so prefixing can use an
    /// exact power of ten instead of a rounded product.
    exp10: Option<i8>,
}

/// An SI prefix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prefix {
    /// The prefix symbol, e.g. `"k"`, `"μ"`.
    pub name: &'static str,
    /// An ASCII spelling (`u` for `μ`).
    pub ascii: Option<&'static str>,
    /// The full name, e.g. `"kilo"`, used only for did-you-mean help.
    pub full: &'static str,
    /// The power of ten.
    pub exp10: i8,
}

/// The result of [`lookup`]: a table unit, possibly with a prefix, and the
/// dimension and SI scale of the whole spelling.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Resolved {
    /// The table entry (for `km`, the metre).
    pub unit: &'static Unit,
    /// The prefix, if the spelling had one (for `km`, kilo).
    pub prefix: Option<&'static Prefix>,
    /// The dimension (a prefix never changes it).
    pub dim: Dimension,
    /// Value of one of this in the coherent SI unit of `dim`.
    pub scale: f64,
}

/// 1 eV in joules (exact since the 2019 SI redefinition; SI Brochure, 9th
/// edition, Table 8: e = 1.602 176 634 × 10⁻¹⁹ C).
pub const ELECTRONVOLT_IN_JOULES: f64 = 1.602_176_634e-19;
/// 1 au in metres (exact, IAU 2012 Resolution B2).
pub const ASTRONOMICAL_UNIT_IN_METRES: f64 = 149_597_870_700.0;
/// 1 pc in metres: 648000/π au (IAU 2015 Resolution B2).
pub const PARSEC_IN_METRES: f64 = 648_000.0 / std::f64::consts::PI * ASTRONOMICAL_UNIT_IN_METRES;
/// 1 degree of arc in radians: π/180 (SI Brochure, 9th edition, Table 8).
pub const DEGREE_IN_RADIANS: f64 = std::f64::consts::PI / 180.0;
/// 1 min in seconds (SI Brochure, 9th edition, Table 8).
pub const MINUTE_IN_SECONDS: f64 = 60.0;
/// 1 h in seconds (SI Brochure, 9th edition, Table 8).
pub const HOUR_IN_SECONDS: f64 = 3600.0;
/// 1 day in seconds (SI Brochure, 9th edition, Table 8: 1 d = 86 400 s).
pub const DAY_IN_SECONDS: f64 = 86_400.0;
/// 1 Julian year in seconds: exactly 365.25 days. The Julian year is the
/// astronomers' year (IAU convention); the SI Brochure, 9th edition, Table 8
/// supplies the day it is counted in.
pub const JULIAN_YEAR_IN_SECONDS: f64 = 365.25 * DAY_IN_SECONDS;
/// The speed of light in vacuum in m/s (exact; SI Brochure, 9th edition,
/// Table 1).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// 1 ly in metres: the distance light travels in one Julian year, exactly
/// `c` × 365.25 d (IAU convention).
pub const LIGHT_YEAR_IN_METRES: f64 = SPEED_OF_LIGHT * JULIAN_YEAR_IN_SECONDS;

/// Exact powers of ten, `POW10[i]` = 10^(i - 33). Literals, so each entry is
/// the correctly rounded double; a computed power would not be. The range
/// covers every prefix (±30) applied to the gram and the litre (10⁻³).
const POW10_MIN: i32 = -33;
static POW10: [f64; 67] = [
    1e-33, 1e-32, 1e-31, 1e-30, 1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21,
    1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
    1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26,
    1e27, 1e28, 1e29, 1e30, 1e31, 1e32, 1e33,
];

/// `10^e` as the correctly rounded double, or `None` outside the table.
fn pow10(e: i32) -> Option<f64> {
    let i = usize::try_from(e.checked_sub(POW10_MIN)?).ok()?;
    POW10.get(i).copied()
}

const fn unit(name: &'static str, dim: Dimension, scale: f64) -> Unit {
    Unit {
        name,
        ascii: None,
        dim,
        scale,
        prefixable: false,
        exp10: None,
    }
}

/// A unit whose scale is exactly `10^exp10`. Indexing a const table with an
/// out-of-range exponent is a compile error, not a runtime panic.
const fn decimal(name: &'static str, dim: Dimension, exp10: i8) -> Unit {
    Unit {
        name,
        ascii: None,
        dim,
        scale: POW10[(exp10 as i32 - POW10_MIN) as usize],
        prefixable: false,
        exp10: Some(exp10),
    }
}

impl Unit {
    const fn prefixable(mut self) -> Self {
        self.prefixable = true;
        self
    }

    const fn ascii(mut self, ascii: &'static str) -> Self {
        self.ascii = Some(ascii);
        self
    }

    fn is_spelled(&self, s: &str) -> bool {
        self.name == s || self.ascii == Some(s)
    }
}

const fn prefix(
    name: &'static str,
    full: &'static str,
    exp10: i8,
    ascii: Option<&'static str>,
) -> Prefix {
    Prefix {
        name,
        ascii,
        full,
        exp10,
    }
}

/// Every SI prefix (SI Brochure, 9th edition, Table 7, including the 2022
/// additions ronna, quetta, ronto, and quecto from CGPM 27, Resolution 3).
///
/// `μ` is GREEK SMALL LETTER MU U+03BC, the only mu the lexer accepts (the
/// MICRO SIGN U+00B5 is not NFKC); ASCII `u` is an alias for it.
pub static PREFIXES: &[Prefix] = &[
    prefix("q", "quecto", -30, None),
    prefix("r", "ronto", -27, None),
    prefix("y", "yocto", -24, None),
    prefix("z", "zepto", -21, None),
    prefix("a", "atto", -18, None),
    prefix("f", "femto", -15, None),
    prefix("p", "pico", -12, None),
    prefix("n", "nano", -9, None),
    prefix("μ", "micro", -6, Some("u")),
    prefix("m", "milli", -3, None),
    prefix("c", "centi", -2, None),
    prefix("d", "deci", -1, None),
    prefix("da", "deca", 1, None),
    prefix("h", "hecto", 2, None),
    prefix("k", "kilo", 3, None),
    prefix("M", "mega", 6, None),
    prefix("G", "giga", 9, None),
    prefix("T", "tera", 12, None),
    prefix("P", "peta", 15, None),
    prefix("E", "exa", 18, None),
    prefix("Z", "zetta", 21, None),
    prefix("Y", "yotta", 24, None),
    prefix("R", "ronna", 27, None),
    prefix("Q", "quetta", 30, None),
];

/// Every base unit known to Sky. Prefixed spellings (`km`, `ms`) are not
/// listed; [`lookup`] derives them.
pub static UNITS: &[Unit] = &[
    // The seven SI base units (SI Brochure, 9th edition, Table 2). The
    // gram, not the kilogram, is the prefixable mass unit, so `kg` is
    // `k` + `g` and still has scale exactly 1.
    decimal("m", Dimension::LENGTH, 0).prefixable(),
    decimal("s", Dimension::TIME, 0).prefixable(),
    decimal("g", Dimension::MASS, -3).prefixable(),
    decimal("K", Dimension::TEMPERATURE, 0).prefixable(),
    decimal("A", Dimension::CURRENT, 0).prefixable(),
    decimal("mol", Dimension::AMOUNT, 0).prefixable(),
    decimal("cd", Dimension::LUMINOSITY, 0).prefixable(),
    // Derived SI units with special names, all coherent (SI Brochure, 9th
    // edition, Table 4). `N` is the newton. It is unrelated to the
    // base-dimension symbol N (amount of substance) in `Dimension`'s
    // display, and to a shape variable written `N` in `Matrix[N, N]`: the
    // parser only reads a unit right after a numeric literal (`10 N`).
    decimal("J", Dimension::ENERGY, 0).prefixable(),
    decimal("N", Dimension::FORCE, 0).prefixable(),
    decimal("Pa", Dimension::PRESSURE, 0).prefixable(),
    decimal("W", Dimension::POWER, 0).prefixable(),
    // `Hz` is T⁻¹, the same dimension as `rad/s` because `rad` is
    // dimensionless: a missing 2π between frequency and angular frequency is
    // not a dimension error. A known v0.1 limitation.
    decimal("Hz", Dimension::FREQUENCY, 0).prefixable(),
    decimal("C", Dimension::CHARGE, 0).prefixable(),
    decimal("V", Dimension::VOLTAGE, 0).prefixable(),
    // `Ω` is GREEK CAPITAL LETTER OMEGA U+03A9; the OHM SIGN U+2126 is not
    // NFKC and the lexer rejects it. `ohm` is the ASCII spelling.
    decimal("Ω", Dimension::RESISTANCE, 0)
        .prefixable()
        .ascii("ohm"),
    decimal("F", Dimension::CAPACITANCE, 0).prefixable(),
    decimal("T", Dimension::MAGNETIC_FLUX_DENSITY, 0).prefixable(),
    decimal("Wb", Dimension::MAGNETIC_FLUX, 0).prefixable(),
    decimal("H", Dimension::INDUCTANCE, 0).prefixable(),
    decimal("S", Dimension::CONDUCTANCE, 0).prefixable(),
    // The radian is dimensionless (SI Brochure, 9th edition, Table 4), so
    // `mrad` and `μrad` are allowed and are plain numbers.
    decimal("rad", Dimension::DIMENSIONLESS, 0).prefixable(),
    // The degree of arc (SI Brochure, 9th edition, Table 8: 1° = (π/180)
    // rad) is a dimensionless multiple of the radian, so `sin(30 deg)`
    // works. It takes no SI prefixes (the Brochure lists it among the
    // non-SI units accepted for use with SI, and no one writes `kdeg`;
    // sub-degree angles are arcminutes and arcseconds, not `mdeg`). `deg`
    // is not `d` (deci) + `eg`, and the table has no `eg`, so no spelling
    // is ambiguous.
    unit("deg", Dimension::DIMENSIONLESS, DEGREE_IN_RADIANS),
    // Accepted for use with SI (SI Brochure, 9th edition, Table 8): the
    // litre is exactly 1 dm³ = 10⁻³ m³ and the tonne exactly 10³ kg.
    decimal("L", Dimension::VOLUME, -3).prefixable(),
    decimal("t", Dimension::MASS, 3).prefixable(),
    // Time units without prefixes (`min`, `h`, `day`), and the year, which
    // takes them (`kyr`, `Myr`, `Gyr`). `h` is the hour, not Planck's
    // constant: the constant `h` is a name in scope, not a unit.
    unit("min", Dimension::TIME, MINUTE_IN_SECONDS),
    unit("h", Dimension::TIME, HOUR_IN_SECONDS),
    unit("day", Dimension::TIME, DAY_IN_SECONDS),
    unit("yr", Dimension::TIME, JULIAN_YEAR_IN_SECONDS).prefixable(),
    // Length and energy units for astronomy and particle physics.
    unit("ly", Dimension::LENGTH, LIGHT_YEAR_IN_METRES).prefixable(),
    unit("pc", Dimension::LENGTH, PARSEC_IN_METRES).prefixable(),
    unit("AU", Dimension::LENGTH, ASTRONOMICAL_UNIT_IN_METRES),
    unit("eV", Dimension::ENERGY, ELECTRONVOLT_IN_JOULES).prefixable(),
];

/// Spellings that an SI prefix and a table unit would read as a unit, but
/// that people write for a different unit (or a variable), with what the
/// help says instead. [`lookup`] refuses them; [`unit_hint`] explains them.
pub static NOT_UNITS: &[(&str, &str)] = &[
    (
        "ft",
        "`ft` would be femto-tonne (10⁻¹² kg), not feet; Sky has no foot, so write `0.3048 m` for 1 ft",
    ),
    (
        "pt",
        "`pt` would be pico-tonne (10⁻⁹ kg), not a pint or a point; Sky has neither, so write the value in SI (1 US pint is `0.473176473 L`)",
    ),
    (
        "ct",
        "`ct` would be centi-tonne (10 kg), not a carat; Sky has no carat, so write `0.2 g` for 1 ct",
    ),
    (
        "qt",
        "`qt` would be quecto-tonne (10⁻²⁷ kg), not a quart; Sky has no quart, so write the value in SI (1 US quart is `0.946352946 L`)",
    ),
    (
        "at",
        "`at` would be atto-tonne (10⁻¹⁵ kg), not a technical atmosphere; write `98066.5 Pa` for 1 at",
    ),
    (
        "dt",
        "`dt` would be deci-tonne (100 kg); to divide by a variable `dt`, put spaces around the operator (`x / dt`), and for 100 kg write `100 kg`",
    ),
    (
        "myr",
        "`myr` would be milli-year; for a million years write `Myr` (capital M)",
    ),
];

fn exact(name: &str) -> Option<&'static Unit> {
    UNITS.iter().find(|u| u.is_spelled(name))
}

fn resolved(unit: &'static Unit, prefix: Option<&'static Prefix>) -> Resolved {
    let scale = match prefix {
        None => unit.scale,
        Some(p) => unit
            .exp10
            // Exact power of ten: `mg` is the literal 1e-6, not 1e-3 × 1e-3.
            .and_then(|e| pow10(i32::from(e) + i32::from(p.exp10)))
            .or_else(|| pow10(i32::from(p.exp10)).map(|f| f * unit.scale))
            .unwrap_or(unit.scale),
    };
    Resolved {
        unit,
        prefix,
        dim: unit.dim,
        scale,
    }
}

/// Looks a unit up by its exact spelling (case-sensitive), e.g. `lookup("km")`.
///
/// An exact table name (or ASCII alias) always wins; otherwise the name is
/// split into an SI prefix and a prefixable unit (see the module docs).
///
/// This (together with [`is_unit_symbol`]) is the single source of truth for
/// which unit names exist: the checker uses it to get a unit's dimension and
/// scale. Do not keep a second list anywhere else.
pub fn lookup(name: &str) -> Option<Resolved> {
    if let Some(u) = exact(name) {
        return Some(resolved(u, None));
    }
    if NOT_UNITS.iter().any(|&(n, _)| n == name) {
        return None;
    }
    for p in PREFIXES {
        for spelling in [Some(p.name), p.ascii].into_iter().flatten() {
            let Some(rest) = name.strip_prefix(spelling) else {
                continue;
            };
            if let Some(u) = exact(rest).filter(|u| u.prefixable) {
                return Some(resolved(u, Some(p)));
            }
        }
    }
    None
}

/// True if `name` is a unit spelling; shorthand for `lookup(name).is_some()`.
pub fn is_unit_symbol(name: &str) -> bool {
    lookup(name).is_some()
}

/// Full names people are likely to write, mapped to a table symbol.
static FULL_NAMES: &[(&str, &str)] = &[
    ("meter", "m"),
    ("metre", "m"),
    ("second", "s"),
    ("sec", "s"),
    ("gram", "g"),
    ("kelvin", "K"),
    ("ampere", "A"),
    ("amp", "A"),
    ("mole", "mol"),
    ("candela", "cd"),
    ("joule", "J"),
    ("newton", "N"),
    ("pascal", "Pa"),
    ("watt", "W"),
    ("hertz", "Hz"),
    ("coulomb", "C"),
    ("volt", "V"),
    ("ohm", "Ω"),
    ("farad", "F"),
    ("tesla", "T"),
    ("weber", "Wb"),
    ("henry", "H"),
    ("siemens", "S"),
    ("liter", "L"),
    ("litre", "L"),
    ("tonne", "t"),
    ("radian", "rad"),
    ("degree", "deg"),
    ("parsec", "pc"),
    ("electronvolt", "eV"),
    ("lightyear", "ly"),
    ("year", "yr"),
    ("minute", "min"),
    ("hour", "h"),
    ("hr", "h"),
    ("day", "day"),
];

/// Help for a name that is not a unit but that people write as one: a
/// spelling in [`NOT_UNITS`], a temperature scale with an offset (°C, °F),
/// molar `M`, or the inch. `None` for anything else ([`suggest_unit`] may
/// still find a near miss).
pub fn unit_hint(name: &str) -> Option<String> {
    if let Some(&(_, why)) = NOT_UNITS.iter().find(|&&(n, _)| n == name) {
        return Some(why.to_string());
    }
    let lower = name.to_lowercase();
    let hint = match lower.as_str() {
        "degc" | "celsius" | "°c" | "℃" | "degf" | "fahrenheit" | "°f" | "℉" => {
            temperature_hint()
        }
        "inch" | "inches" | "in" => "Sky has no inch; write `0.0254 m` for 1 in",
        _ if name == "M" || name == "molar" => {
            "`M` (molar) is not a unit; write `mol/L` (2 M is `2 mol/L`)"
        }
        _ => return None,
    };
    Some(hint.to_string())
}

/// The help for °C and °F, which the lexer also gives for a `°`.
pub fn temperature_hint() -> &'static str {
    "temperatures with an offset (°C, °F) are not units in Sky, which measures temperature in kelvin: write 25 °C as `298.15 K`; a temperature difference of 1 °C is `1 K`, and of 1 °F is 5/9 of a kelvin (`5 K / 9`)"
}

/// A unit spelling to suggest for something that is not a unit, or `None`.
///
/// Handles full names with an optional plural and SI prefix name
/// (`kilometers` → `km`, `micrometer` → `μm`, `amps` → `A`), a wrong case
/// (`hz` → `Hz`), and `Gy` → `Gyr`. Only used for `help:` text; every
/// suggestion is checked against [`lookup`], so it is always a real unit.
pub fn suggest_unit(name: &str) -> Option<String> {
    if name == "Gy" {
        return Some("Gyr".to_string());
    }
    if let Some(u) = UNITS.iter().find(|u| u.name.eq_ignore_ascii_case(name)) {
        return Some(u.name.to_string());
    }
    let singular = name.strip_suffix('s').filter(|s| !s.is_empty());
    for candidate in [Some(name), singular].into_iter().flatten() {
        let lower = candidate.to_lowercase();
        let (prefix, rest) = PREFIXES
            .iter()
            .find_map(|p| lower.strip_prefix(p.full).map(|r| (p.name, r)))
            .or_else(|| lower.strip_prefix("deka").map(|r| ("da", r)))
            .filter(|(_, r)| !r.is_empty())
            .unwrap_or(("", lower.as_str()));
        if let Some(&(_, sym)) = FULL_NAMES.iter().find(|(full, _)| *full == rest) {
            let suggestion = format!("{prefix}{sym}");
            if lookup(&suggestion).is_some() {
                return Some(suggestion);
            }
        }
    }
    None
}

/// The SI unit text for `dim` as valid Sky unit syntax, or `None` if the
/// dimension is dimensionless (a bare number). This is how every value
/// prints, so all engines and the checker's help text agree.
///
/// If a unit in the table has exactly this dimension and scale 1 (`N`,
/// `J`, `W`, `Pa`, and whatever the table gains), that symbol is used.
/// Otherwise base units are composed: positive powers first, in the order
/// kg, m, s, K, A, mol, cd (`kg*m^2/s^2`), then one `/` per negative power,
/// with `mol` first and the rest in that order, the way chemistry writes
/// "per mole" (`kg*m^2/mol/s^2/K` for J/(mol·K), `m^3/mol/s` for a rate
/// constant). A dimension with only negative powers is `m^-1`, `s^-1*K^-1`.
///
/// `Hz` is never chosen: `rad` is dimensionless, so a value of dimension
/// T⁻¹ may be a frequency, an angular frequency (`rad/s`) or a rate, and
/// naming it `Hz` would be a 2π error for an angular frequency. It prints
/// as `s^-1`, which is right for all three.
pub fn si_unit_text(dim: &Dimension) -> Option<String> {
    use crate::dimension::BaseDim;
    use crate::rational::Rational;
    if dim.is_dimensionless() {
        return None;
    }
    let ambiguous = *dim == Dimension::FREQUENCY;
    if let Some(u) = UNITS
        .iter()
        .find(|u| u.dim == *dim && u.scale == 1.0 && !ambiguous)
    {
        return Some(u.name.to_string());
    }
    let symbol = |b: BaseDim| match b {
        BaseDim::Mass => "kg",
        BaseDim::Length => "m",
        BaseDim::Time => "s",
        BaseDim::Temperature => "K",
        BaseDim::Current => "A",
        BaseDim::Amount => "mol",
        BaseDim::Luminosity => "cd",
    };
    let power = |sym: &str, e: Rational| {
        if e == Rational::ONE {
            sym.to_string()
        } else if e.is_integer() {
            format!("{sym}^{}", e.numer())
        } else {
            format!("{sym}^({}/{})", e.numer(), e.denom())
        }
    };
    const NUMERATOR: [BaseDim; 7] = [
        BaseDim::Mass,
        BaseDim::Length,
        BaseDim::Time,
        BaseDim::Temperature,
        BaseDim::Current,
        BaseDim::Amount,
        BaseDim::Luminosity,
    ];
    const DENOMINATOR: [BaseDim; 7] = [
        BaseDim::Amount,
        BaseDim::Mass,
        BaseDim::Length,
        BaseDim::Time,
        BaseDim::Temperature,
        BaseDim::Current,
        BaseDim::Luminosity,
    ];
    let pos: Vec<String> = NUMERATOR
        .into_iter()
        .map(|b| (b, dim.exponent(b)))
        .filter(|(_, e)| e.numer() > 0)
        .map(|(b, e)| power(symbol(b), e))
        .collect();
    if pos.is_empty() {
        // No numerator: write the negative exponents, `s^-1*K^-1`.
        let parts: Vec<String> = NUMERATOR
            .into_iter()
            .map(|b| (symbol(b), dim.exponent(b)))
            .filter(|(_, e)| e.numer() < 0)
            .map(|(s, e)| power(s, e))
            .collect();
        return Some(parts.join("*"));
    }
    let mut out = pos.join("*");
    for b in DENOMINATOR {
        let e = dim.exponent(b);
        if e.numer() >= 0 {
            continue;
        }
        out.push('/');
        // `e` is negative, so it can be negated unless it is i64::MIN,
        // which the checker's exponent limits keep out of any value.
        out.push_str(&power(symbol(b), e.checked_neg().unwrap_or(e)));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(name: &str) -> Dimension {
        lookup(name).unwrap_or_else(|| panic!("{name}")).dim
    }

    fn scale(name: &str) -> f64 {
        lookup(name).unwrap_or_else(|| panic!("{name}")).scale
    }

    /// The prefix symbol of `name`, or `""` if it resolved unprefixed.
    fn prefix_of(name: &str) -> &'static str {
        lookup(name).unwrap().prefix.map_or("", |p| p.name)
    }

    fn base_of(name: &str) -> &'static str {
        lookup(name).unwrap().unit.name
    }

    #[test]
    fn prefixed_units_share_a_dimension() {
        assert_eq!(dim("m"), dim("km"));
        assert_eq!(scale("km"), 1000.0);
        assert_eq!(scale("h"), 3600.0);
        assert_eq!(scale("min"), 60.0);
        assert_eq!((prefix_of("km"), base_of("km")), ("k", "m"));
    }

    #[test]
    fn millisecond_is_time_with_scale_one_thousandth() {
        let ms = lookup("ms").unwrap();
        assert_eq!(ms.dim, Dimension::TIME);
        assert_eq!(ms.scale, 0.001);
        assert_eq!((prefix_of("ms"), base_of("ms")), ("m", "s"));
        // `ms` is one symbol, not metre times second.
        assert_ne!(
            ms.dim,
            Dimension::LENGTH.checked_mul(Dimension::TIME).unwrap()
        );
    }

    #[test]
    fn millisecond_values_in_the_spec() {
        // spec v0.1, "Unit values": a literal converts to SI once, as
        // `x * scale`, and the result is an ordinary f64. `10 ms` is exactly
        // 0.01 and `100 ms` is 0.1, but `3 * (100 ms)` is
        // 0.30000000000000004, so `3 * (100 ms) == 0.3 s` is false.
        let ms = scale("ms");
        assert_eq!(10.0 * ms, 0.01);
        assert_eq!(100.0 * ms, 0.1);
        let lhs = 3.0 * (100.0 * ms);
        assert_ne!(lhs, 0.3 * scale("s"));
        assert_eq!(lhs.to_bits(), (0.3_f64).to_bits() + 1);
        assert_eq!(format!("{lhs:?}"), "0.30000000000000004");
    }

    #[test]
    fn exact_constants() {
        assert_eq!(scale("eV"), 1.602176634e-19);
        assert_eq!(dim("eV"), dim("J"));
        assert_eq!(scale("AU"), 149_597_870_700.0);
        let pc = scale("pc");
        // IAU 2015: 1 pc = 3.085 677 581 491 367... e16 m.
        assert!((pc - 3.085_677_581_491_367e16).abs() / pc < 1e-15);
        assert_eq!(scale("day"), 86_400.0);
        assert_eq!(scale("yr"), 31_557_600.0);
        // IAU: 1 ly = 9 460 730 472 580 800 m exactly.
        assert_eq!(scale("ly"), 9_460_730_472_580_800.0);
        assert_eq!(scale("L"), 1e-3);
        assert_eq!(scale("t"), 1e3);
    }

    #[test]
    fn radian_is_dimensionless() {
        let rad = lookup("rad").unwrap();
        assert!(rad.dim.is_dimensionless());
        assert_eq!(rad.scale, 1.0);
        assert_eq!(scale("mrad"), 1e-3);
    }

    #[test]
    fn degree_is_a_dimensionless_multiple_of_the_radian_without_prefixes() {
        let deg = lookup("deg").unwrap();
        assert!(deg.dim.is_dimensionless());
        assert_eq!(deg.scale, std::f64::consts::PI / 180.0);
        assert!(!deg.unit.prefixable);
        for p in PREFIXES {
            assert_eq!(lookup(&format!("{}deg", p.name)), None, "{}deg", p.name);
        }
        assert_eq!(suggest_unit("degrees").as_deref(), Some("deg"));
        // 180 deg is a half turn.
        assert!((180.0 * deg.scale - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn table_is_exactly_this_set() {
        let names: Vec<_> = UNITS.iter().map(|u| u.name).collect();
        assert_eq!(
            names,
            [
                "m", "s", "g", "K", "A", "mol", "cd", "J", "N", "Pa", "W", "Hz", "C", "V", "Ω",
                "F", "T", "Wb", "H", "S", "rad", "deg", "L", "t", "min", "h", "day", "yr", "ly",
                "pc", "AU", "eV"
            ]
        );
        for u in UNITS {
            assert!(u.scale.is_finite() && u.scale > 0.0, "{}", u.name);
            if let Some(e) = u.exp10 {
                assert_eq!(u.scale, pow10(i32::from(e)).unwrap(), "{}", u.name);
            }
        }
    }

    #[test]
    fn prefix_table_is_the_full_si_set() {
        let syms: Vec<_> = PREFIXES.iter().map(|p| p.name).collect();
        assert_eq!(
            syms,
            [
                "q", "r", "y", "z", "a", "f", "p", "n", "μ", "m", "c", "d", "da", "h", "k", "M",
                "G", "T", "P", "E", "Z", "Y", "R", "Q"
            ]
        );
        // Strictly increasing exponents: no duplicates, none out of order.
        assert!(PREFIXES.windows(2).all(|w| w[0].exp10 < w[1].exp10));
        assert_eq!(scale("qm"), 1e-30);
        assert_eq!(scale("Qm"), 1e30);
        assert_eq!(scale("Rm"), 1e27);
        assert_eq!(scale("rm"), 1e-27);
    }

    #[test]
    fn every_prefix_applies_exactly_to_every_decimal_unit() {
        for u in UNITS.iter().filter(|u| u.prefixable && u.exp10.is_some()) {
            for p in PREFIXES {
                let name = format!("{}{}", p.name, u.name);
                if NOT_UNITS.iter().any(|&(n, _)| n == name) {
                    continue;
                }
                let want: f64 = format!("1e{}", i32::from(u.exp10.unwrap()) + i32::from(p.exp10))
                    .parse()
                    .unwrap();
                assert_eq!(scale(&name), want, "{name}");
                assert_eq!(dim(&name), u.dim, "{name}");
            }
        }
    }

    #[test]
    fn prefixed_non_decimal_units_are_within_an_ulp() {
        for (name, want) in [
            ("keV", 1.602176634e-16),
            ("GeV", 1.602176634e-10),
            ("kpc", 3.085_677_581_491_367e19),
            ("Myr", 3.15576e13),
            ("Gyr", 3.15576e16),
            ("Mly", 9.460_730_472_580_8e21),
        ] {
            let got = scale(name);
            assert!((got - want).abs() / want < 4.0 * f64::EPSILON, "{name}");
        }
    }

    #[test]
    fn kilogram_is_kilo_gram_and_coherent() {
        let kg = lookup("kg").unwrap();
        assert_eq!((kg.prefix.unwrap().name, kg.unit.name), ("k", "g"));
        assert_eq!(kg.dim, Dimension::MASS);
        // The gram is 10⁻³ of the coherent unit, and `kg` is exactly 1.
        assert_eq!(scale("g"), 1e-3);
        assert_eq!(kg.scale, 1.0);
        assert_eq!(scale("mg"), 1e-6);
        assert_eq!(scale("μg"), 1e-9);
        assert_eq!(scale("Mg"), 1e3);
    }

    #[test]
    fn ascii_u_is_micro() {
        let (u, mu) = (lookup("um").unwrap(), lookup("μm").unwrap());
        assert_eq!(u, mu);
        assert_eq!(u.scale, 1e-6);
        assert_eq!(scale("us"), 1e-6);
        assert_eq!(scale("uohm"), 1e-6);
        assert_eq!(scale("μΩ"), 1e-6);
    }

    #[test]
    fn ohm_has_an_ascii_spelling() {
        assert_eq!(lookup("ohm").unwrap(), lookup("Ω").unwrap());
        assert_eq!(scale("kohm"), 1e3);
        assert_eq!(scale("MΩ"), 1e6);
        assert_eq!(dim("Ω"), Dimension::RESISTANCE);
    }

    // ----- ambiguities: one test per resolution -----

    #[test]
    fn min_is_the_minute_not_milli_inch() {
        assert_eq!(base_of("min"), "min");
        assert_eq!(prefix_of("min"), "");
        assert_eq!(dim("min"), Dimension::TIME);
        assert_eq!(scale("min"), 60.0);
        // `m` + `in` is not a reading: there is no inch, and `min` takes
        // no prefixes.
        assert!(lookup("mmin").is_none());
        assert!(lookup("kmin").is_none());
    }

    #[test]
    fn h_is_the_hour_and_hecto_is_a_prefix() {
        assert_eq!((base_of("h"), prefix_of("h")), ("h", ""));
        assert_eq!(scale("h"), 3600.0);
        assert_eq!((base_of("hPa"), prefix_of("hPa")), ("Pa", "h"));
        assert_eq!(scale("hPa"), 100.0);
        assert_eq!(dim("hPa"), Dimension::PRESSURE);
        assert_eq!(scale("hm"), 100.0);
        // The hour takes no prefix.
        assert!(lookup("hh").is_none());
        assert!(lookup("kh").is_none());
    }

    #[test]
    fn pc_is_the_parsec_not_pico_c() {
        assert_eq!((base_of("pc"), prefix_of("pc")), ("pc", ""));
        assert_eq!(dim("pc"), Dimension::LENGTH);
        assert_eq!(scale("pc"), PARSEC_IN_METRES);
        assert_eq!((base_of("Mpc"), prefix_of("Mpc")), ("pc", "M"));
        assert_eq!((base_of("kpc"), prefix_of("kpc")), ("pc", "k"));
    }

    #[test]
    fn pa_is_the_pascal_not_peta_a() {
        assert_eq!((base_of("Pa"), prefix_of("Pa")), ("Pa", ""));
        assert_eq!(dim("Pa"), Dimension::PRESSURE);
        // Peta + ampere is the capitalised `PA`.
        assert_eq!((base_of("PA"), prefix_of("PA")), ("A", "P"));
        assert_eq!(scale("kPa"), 1e3);
        assert_eq!(scale("GPa"), 1e9);
    }

    #[test]
    fn cd_is_the_candela_not_centi_day() {
        assert_eq!((base_of("cd"), prefix_of("cd")), ("cd", ""));
        assert_eq!(dim("cd"), Dimension::LUMINOSITY);
        assert_eq!(scale("mcd"), 1e-3);
        // There is no unit `d`, so centi-day cannot be spelled `cd`; days are
        // `day` and take no prefix.
        assert!(lookup("d").is_none());
        assert!(lookup("cday").is_none());
    }

    #[test]
    fn ms_mm_nm_are_prefix_plus_unit() {
        assert_eq!((base_of("ms"), prefix_of("ms")), ("s", "m"));
        assert_eq!((base_of("mm"), prefix_of("mm")), ("m", "m"));
        assert_eq!((base_of("nm"), prefix_of("nm")), ("m", "n"));
        assert_eq!(dim("ms"), Dimension::TIME);
        assert_eq!(dim("mm"), Dimension::LENGTH);
        assert_eq!(scale("mm"), 1e-3);
        assert_eq!(scale("nm"), 1e-9);
        // `m` alone is the metre, not milli of nothing.
        assert_eq!((base_of("m"), prefix_of("m")), ("m", ""));
    }

    #[test]
    fn gyr_is_giga_year_and_gy_is_not_a_unit() {
        assert_eq!((base_of("Gyr"), prefix_of("Gyr")), ("yr", "G"));
        assert_eq!(dim("Gyr"), Dimension::TIME);
        assert!(lookup("Gy").is_none());
        assert_eq!(suggest_unit("Gy").as_deref(), Some("Gyr"));
    }

    #[test]
    fn da_is_deca_and_day_is_the_day() {
        assert_eq!((base_of("dam"), prefix_of("dam")), ("m", "da"));
        assert_eq!(scale("dam"), 10.0);
        assert_eq!((base_of("day"), prefix_of("day")), ("day", ""));
        assert_eq!(scale("day"), 86_400.0);
        // `da` alone is a prefix with nothing to prefix; `dm` is deci-metre.
        assert!(lookup("da").is_none());
        assert_eq!(prefix_of("dm"), "d");
        assert_eq!(scale("dm"), 0.1);
    }

    #[test]
    fn t_is_tesla_and_tera_and_t_is_the_tonne() {
        assert_eq!((base_of("T"), prefix_of("T")), ("T", ""));
        assert_eq!(dim("T"), Dimension::MAGNETIC_FLUX_DENSITY);
        assert_eq!((base_of("TT"), prefix_of("TT")), ("T", "T"));
        assert_eq!((base_of("Tm"), prefix_of("Tm")), ("m", "T"));
        assert_eq!((base_of("mT"), prefix_of("mT")), ("T", "m"));
        assert_eq!(scale("mT"), 1e-3);
        assert_eq!(scale("Tm"), 1e12);
        // Lowercase `t` is the tonne (1000 kg), a different dimension.
        assert_eq!((base_of("t"), prefix_of("t")), ("t", ""));
        assert_eq!(dim("t"), Dimension::MASS);
        assert_eq!(scale("kt"), 1e6);
        assert_eq!(scale("Mt"), 1e9);
        assert_eq!(base_of("mt"), "t");
        assert_eq!(scale("mt"), 1.0);
    }

    #[test]
    fn every_spelling_is_unambiguous() {
        // Every (prefix spelling, prefixable unit spelling) pair: the
        // concatenation must be produced by exactly one pair, and if it is
        // also an exact table name, exact must win (and we list those).
        let mut seen: Vec<(String, &str, &str)> = Vec::new();
        let mut shadowed = Vec::new();
        for p in PREFIXES {
            for ps in [Some(p.name), p.ascii].into_iter().flatten() {
                for u in UNITS.iter().filter(|u| u.prefixable) {
                    for us in [Some(u.name), u.ascii].into_iter().flatten() {
                        let joined = format!("{ps}{us}");
                        if let Some(e) = exact(&joined) {
                            shadowed.push(joined.clone());
                            assert!(lookup(&joined).unwrap().prefix.is_none());
                            assert_eq!(lookup(&joined).unwrap().unit.name, e.name);
                        }
                        seen.push((joined, p.name, u.name));
                    }
                }
            }
        }
        for (i, (a, pa, ua)) in seen.iter().enumerate() {
            for (b, pb, ub) in &seen[i + 1..] {
                if a == b {
                    // Same string from two decompositions is only fine
                    // when they are the same prefix and unit (an alias).
                    assert_eq!((pa, ua), (pb, ub), "{a} is ambiguous");
                }
            }
        }
        // Pinned: the prefix readings that an exact table name overrides.
        // `Pa` (peta + a) does not appear: there is no unit `a`.
        shadowed.sort();
        shadowed.dedup();
        assert_eq!(shadowed, Vec::<String>::new(), "new shadowing; document it");
    }

    #[test]
    fn lookup_is_the_public_source_of_truth() {
        for u in UNITS {
            for spelling in [Some(u.name), u.ascii].into_iter().flatten() {
                let found = crate::lookup(spelling).expect(spelling);
                assert!(std::ptr::eq(found.unit, u), "{spelling}");
                assert!(found.prefix.is_none());
                assert!(crate::is_unit_symbol(spelling));
            }
        }
        let km = crate::lookup("km").unwrap();
        assert_eq!((km.dim, km.scale), (Dimension::LENGTH, 1000.0));
        // Case-sensitive, and neither base-dimension letters nor aliases are
        // units. (`L`, `T`, `N`, `J` are units now, but different ones.)
        for not_unit in [
            "Km", "M", "I", "meters", "kelvin", "°C", "km ", "", "µm", "k", "da", "Gy", "kmin",
            "ohms", "hh", "μ", "u",
        ] {
            assert!(crate::lookup(not_unit).is_none(), "{not_unit:?}");
            assert!(!crate::is_unit_symbol(not_unit), "{not_unit:?}");
        }
    }

    #[test]
    fn lookup_never_panics_on_odd_input() {
        // Multi-byte characters where a prefix would end, and lone prefixes.
        for s in [
            "μ",
            "μμ",
            "μ\u{301}m",
            "kμ",
            "日本",
            "k日",
            "😀m",
            "m😀",
            "d\u{0}",
            "\u{10FFFF}",
            "daa",
            "dda",
            "Ωm",
        ] {
            let _ = lookup(s);
            let _ = suggest_unit(s);
        }
    }

    #[test]
    fn micro_sign_is_not_accepted() {
        // U+00B5 is what keyboards produce; the lexer rejects it and points to
        // U+03BC. The table follows the lexer.
        assert!(lookup("\u{00B5}m").is_none());
        assert!(lookup("\u{03BC}m").is_some());
        // Likewise OHM SIGN U+2126 (not NFKC), while U+03A9 is the unit.
        assert!(lookup("\u{2126}").is_none());
        assert!(lookup("\u{03A9}").is_some());
    }

    #[test]
    fn suggestions() {
        for (wrong, right) in [
            ("meter", "m"),
            ("meters", "m"),
            ("kilometers", "km"),
            ("micrometer", "μm"),
            ("micrometers", "μm"),
            ("nanometre", "nm"),
            ("gram", "g"),
            ("kilogram", "kg"),
            ("milligrams", "mg"),
            ("amp", "A"),
            ("amps", "A"),
            ("ampere", "A"),
            ("milliamp", "mA"),
            ("volt", "V"),
            ("volts", "V"),
            ("ohms", "Ω"),
            ("liter", "L"),
            ("litres", "L"),
            ("milliliter", "mL"),
            ("kelvin", "K"),
            ("mole", "mol"),
            ("hertz", "Hz"),
            ("gigahertz", "GHz"),
            ("hz", "Hz"),
            ("ev", "eV"),
            ("megaparsec", "Mpc"),
            ("year", "yr"),
            ("seconds", "s"),
            ("hours", "h"),
            ("hr", "h"),
            ("tesla", "T"),
            ("Gy", "Gyr"),
        ] {
            assert_eq!(suggest_unit(wrong).as_deref(), Some(right), "{wrong}");
            assert!(is_unit_symbol(right), "{right}");
        }
        for nothing in ["", "banana", "kilo", "kilobanana", "s2", "furlong"] {
            assert_eq!(suggest_unit(nothing), None, "{nothing:?}");
        }
        // A prefix name on a unit that takes none is not suggested.
        assert_eq!(suggest_unit("kilohour"), None);
    }

    #[test]
    fn derived_si_units_equal_their_si_composition() {
        let (m, kg, s, a) = (dim("m"), dim("kg"), dim("s"), dim("A"));
        let m2 = m.pow_int(2).unwrap();
        let mul = |x: Dimension, y: Dimension| x.checked_mul(y).unwrap();
        let div = |x: Dimension, y: Dimension| x.checked_div(y).unwrap();
        // N = kg·m·s⁻²
        assert_eq!(dim("N"), div(mul(kg, m), s.pow_int(2).unwrap()));
        // Pa = N·m⁻²
        assert_eq!(dim("Pa"), div(dim("N"), m2));
        // W = J·s⁻¹
        assert_eq!(dim("W"), div(dim("J"), s));
        // Hz = s⁻¹
        assert_eq!(dim("Hz"), s.inverse().unwrap());
        // Cross-checks: N·m = J, W·s = J, Pa·m² = N.
        assert_eq!(mul(dim("N"), m), dim("J"));
        assert_eq!(mul(dim("W"), s), dim("J"));
        assert_eq!(mul(dim("Pa"), m2), dim("N"));
        // Electromagnetism: C = A·s, V = W/A, Ω = V/A, S = 1/Ω, F = C/V,
        // Wb = V·s, T = Wb/m², H = Wb/A.
        assert_eq!(dim("C"), mul(a, s));
        assert_eq!(dim("V"), div(dim("W"), a));
        assert_eq!(dim("Ω"), div(dim("V"), a));
        assert_eq!(dim("S"), dim("Ω").inverse().unwrap());
        assert_eq!(dim("F"), div(dim("C"), dim("V")));
        assert_eq!(dim("Wb"), mul(dim("V"), s));
        assert_eq!(dim("T"), div(dim("Wb"), m2));
        assert_eq!(dim("H"), div(dim("Wb"), a));
        // L = dm³, g = 10⁻³ kg.
        assert_eq!(dim("L"), m.pow_int(3).unwrap());
        assert_eq!(dim("g"), kg);
        for u in [
            "N", "Pa", "W", "Hz", "C", "V", "Ω", "F", "T", "Wb", "H", "S", "kg",
        ] {
            assert_eq!(scale(u), 1.0, "{u}");
        }
        assert_eq!(dim("N").name(), Some("force"));
        assert_eq!(dim("Pa").name(), Some("pressure"));
        assert_eq!(dim("W").name(), Some("power"));
        assert_eq!(dim("Hz").name(), Some("frequency"));
        assert_eq!(dim("V").name(), Some("voltage"));
        assert_eq!(dim("T").name(), Some("magnetic flux density"));
    }

    #[test]
    fn hertz_and_radians_per_second_share_a_dimension() {
        // Known v0.1 limitation, not a bug: rad is dimensionless, so a
        // missing 2π between Hz and rad/s is not a dimension error.
        let rad_per_s = dim("rad").checked_div(dim("s")).unwrap();
        assert_eq!(dim("Hz"), rad_per_s);
    }

    #[test]
    fn amount_and_luminosity_never_print_as_newton_or_joule() {
        // Dogfooding: a rate law's dimension printed as `L⁻³·T⁻¹·N`, which
        // reads as a force. Their SI symbols `N` and `J` are the newton and
        // the joule, so messages show `mol` and `cd` instead.
        assert_eq!(Dimension::AMOUNT.to_string(), "mol");
        assert_eq!(Dimension::LUMINOSITY.to_string(), "cd");
        let rate = dim("mol")
            .checked_div(dim("m").pow_int(3).unwrap())
            .unwrap()
            .checked_div(dim("s"))
            .unwrap();
        assert_eq!(rate.to_string(), "L⁻³·T⁻¹·mol");
        assert_eq!(rate.pow_int(2).unwrap().to_string(), "L⁻⁶·T⁻²·mol²");
        assert_eq!(dim("N").to_string(), "L·M·T⁻²");
        assert_eq!(dim("J").to_string(), "L²·M·T⁻²");
        assert_eq!(dim("mol").name(), Some("amount of substance"));
        assert_eq!(dim("cd").name(), Some("luminous intensity"));
    }

    #[test]
    fn hertz_is_never_the_printed_unit() {
        let hz = dim("Hz");
        assert_eq!(si_unit_text(&hz).as_deref(), Some("s^-1"));
        let rad_per_s = dim("rad").checked_div(dim("s")).unwrap();
        assert_eq!(si_unit_text(&rad_per_s).as_deref(), Some("s^-1"));
        // Other named units are still used.
        for name in ["N", "J", "W", "Pa", "C", "V", "Ω", "T"] {
            assert_eq!(si_unit_text(&dim(name)).as_deref(), Some(name));
        }
        assert_eq!(si_unit_text(&Dimension::DIMENSIONLESS), None);
    }

    #[test]
    fn per_mole_comes_first_in_a_denominator() {
        let per = |a: &str, b: &str| dim(a).checked_div(dim(b)).unwrap();
        // J/(mol·K), m³/(mol·s), mol/(m³·s): written the chemist's way.
        let r = per("J", "mol").checked_div(dim("K")).unwrap();
        assert_eq!(si_unit_text(&r).as_deref(), Some("kg*m^2/mol/s^2/K"));
        let k = per("L", "mol").checked_div(dim("s")).unwrap();
        assert_eq!(si_unit_text(&k).as_deref(), Some("m^3/mol/s"));
        let rate = per("mol", "L").checked_div(dim("s")).unwrap();
        assert_eq!(si_unit_text(&rate).as_deref(), Some("mol/m^3/s"));
        // Unchanged where there is no amount.
        let w_per_m_k = per("W", "m").checked_div(dim("K")).unwrap();
        assert_eq!(si_unit_text(&w_per_m_k).as_deref(), Some("kg*m/s^3/K"));
        let per_mol = dim("mol").inverse().unwrap();
        assert_eq!(si_unit_text(&per_mol).as_deref(), Some("mol^-1"));
    }

    #[test]
    fn no_prefix_reading_hides_a_common_unit() {
        // Each of these would be a prefix plus a table unit, a reading nobody
        // means. Pinned, so a change to the list is deliberate.
        let names: Vec<&str> = NOT_UNITS.iter().map(|&(n, _)| n).collect();
        assert_eq!(names, ["ft", "pt", "ct", "qt", "at", "dt", "myr"]);
        for &(name, why) in NOT_UNITS {
            assert!(exact(name).is_none(), "{name} is a table unit");
            assert!(lookup(name).is_none(), "{name}");
            assert_eq!(unit_hint(name).as_deref(), Some(why));
            assert!(why.starts_with(&format!("`{name}` would be")), "{why}");
        }
        // The readings they block were real: the prefix and unit exist.
        for (prefix, unit) in [
            ("f", "t"),
            ("p", "t"),
            ("c", "t"),
            ("q", "t"),
            ("a", "t"),
            ("d", "t"),
            ("m", "yr"),
        ] {
            assert!(PREFIXES.iter().any(|p| p.name == prefix));
            assert!(exact(unit).is_some_and(|u| u.prefixable));
        }
        // Neighbours stay units.
        for name in ["kt", "Mt", "Gt", "mt", "Myr", "kyr", "ms", "mm"] {
            assert!(lookup(name).is_some(), "{name}");
        }
    }

    #[test]
    fn hints_for_things_people_write_as_units() {
        for name in ["degC", "celsius", "degF", "fahrenheit", "°C"] {
            assert_eq!(
                unit_hint(name).as_deref(),
                Some(temperature_hint()),
                "{name}"
            );
        }
        assert!(unit_hint("M").unwrap().contains("`mol/L`"));
        assert!(unit_hint("inch").unwrap().contains("`0.0254 m`"));
        assert_eq!(unit_hint("meters"), None);
        assert_eq!(unit_hint("m"), None);
    }

    #[test]
    fn no_affine_units_or_aliases() {
        for missing in ["°C", "degC", "meters", "kelvin", "KM", ""] {
            assert!(lookup(missing).is_none(), "{missing}");
        }
    }
}
