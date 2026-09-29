//! Confusable-identifier checks (spec v0.1, security non-negotiable 4).
//!
//! This is a deliberately small, curated approximation for v0.1. A full
//! UTS #39 implementation (confusable skeletons from `confusables.txt`,
//! Script_Extensions based mixed-script detection, restriction levels, and
//! comparing skeletons *across* identifiers in a program) is follow-up work.
//!
//! Rules, checked per identifier, at most one warning each:
//!
//! 1. **Letter variants**: compatibility look-alikes of letters that
//!    scientific code uses, e.g. MICRO SIGN `µ` vs Greek `μ`, KELVIN SIGN
//!    vs `K`, OHM SIGN vs `Ω`, fullwidth and mathematical alphanumeric
//!    Latin letters (`ｘ`, `𝑥`).
//! 2. **Look-alike letters**: a Cyrillic/Greek/Armenian letter from
//!    small curated table that renders like a Latin letter, when the identifier
//!    also contains Latin letters (`pаth` with Cyrillic `а`), or when every
//!    letter of the identifier is such a look-alike (`х` alone, which reads
//!    as `x`). Greek letters that are only *somewhat* Latin-like (`α`, `ν`,
//!    `ρ`, `ι`) warn only in the first case, so standalone `α` or `ν` is fine.
//! 3. **Mixed scripts**: Cyrillic together with Latin or Greek in one
//!    identifier. Latin + Greek is allowed on purpose: `Δt`, `λ_max`, `μm`
//!    are normal in scientific code, and rule 2 still catches Greek letters
//!    that impersonate Latin ones.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Greek,
    Cyrillic,
    Armenian,
}

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Script::Latin => "Latin",
            Script::Greek => "Greek",
            Script::Cyrillic => "Cyrillic",
            Script::Armenian => "Armenian",
        })
    }
}

/// Coarse script of a letter, by Unicode block. `None` for digits, `_`,
/// combining marks and scripts this check does not look at.
fn script(c: char) -> Option<Script> {
    match c as u32 {
        0x41..=0x5A | 0x61..=0x7A => Some(Script::Latin),
        0xC0..=0x24F
        | 0x250..=0x2AF
        | 0x1E00..=0x1EFF
        | 0x2C60..=0x2C7F
        | 0xA720..=0xA7FF
        | 0xAB30..=0xAB6F => Some(Script::Latin),
        0x370..=0x3FF | 0x1F00..=0x1FFF => Some(Script::Greek),
        0x400..=0x52F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => {
            Some(Script::Cyrillic)
        }
        0x530..=0x58F => Some(Script::Armenian),
        _ => None,
    }
}

/// `(char, Latin look-alike, script, strong)`. `strong` letters are
/// practically identical to the Latin letter in common fonts and also warn
/// when a whole identifier is made of them.
const LOOKALIKES: &[(char, char, Script, bool)] = &[
    // Cyrillic lowercase
    ('а', 'a', Script::Cyrillic, true),
    ('е', 'e', Script::Cyrillic, true),
    ('о', 'o', Script::Cyrillic, true),
    ('р', 'p', Script::Cyrillic, true),
    ('с', 'c', Script::Cyrillic, true),
    ('у', 'y', Script::Cyrillic, true),
    ('х', 'x', Script::Cyrillic, true),
    ('і', 'i', Script::Cyrillic, true),
    ('ј', 'j', Script::Cyrillic, true),
    ('ѕ', 's', Script::Cyrillic, true),
    ('һ', 'h', Script::Cyrillic, true),
    ('ԁ', 'd', Script::Cyrillic, true),
    ('ԛ', 'q', Script::Cyrillic, true),
    ('ԝ', 'w', Script::Cyrillic, true),
    ('ӏ', 'l', Script::Cyrillic, true),
    // Cyrillic uppercase
    ('А', 'A', Script::Cyrillic, true),
    ('В', 'B', Script::Cyrillic, true),
    ('Е', 'E', Script::Cyrillic, true),
    ('К', 'K', Script::Cyrillic, true),
    ('М', 'M', Script::Cyrillic, true),
    ('Н', 'H', Script::Cyrillic, true),
    ('О', 'O', Script::Cyrillic, true),
    ('Р', 'P', Script::Cyrillic, true),
    ('С', 'C', Script::Cyrillic, true),
    ('Т', 'T', Script::Cyrillic, true),
    ('Х', 'X', Script::Cyrillic, true),
    ('І', 'I', Script::Cyrillic, true),
    ('Ј', 'J', Script::Cyrillic, true),
    ('Ѕ', 'S', Script::Cyrillic, true),
    // Greek
    ('ο', 'o', Script::Greek, true),
    ('Α', 'A', Script::Greek, true),
    ('Β', 'B', Script::Greek, true),
    ('Ε', 'E', Script::Greek, true),
    ('Ζ', 'Z', Script::Greek, true),
    ('Η', 'H', Script::Greek, true),
    ('Ι', 'I', Script::Greek, true),
    ('Κ', 'K', Script::Greek, true),
    ('Μ', 'M', Script::Greek, true),
    ('Ν', 'N', Script::Greek, true),
    ('Ο', 'O', Script::Greek, true),
    ('Ρ', 'P', Script::Greek, true),
    ('Τ', 'T', Script::Greek, true),
    ('Υ', 'Y', Script::Greek, true),
    ('Χ', 'X', Script::Greek, true),
    ('α', 'a', Script::Greek, false),
    ('ν', 'v', Script::Greek, false),
    ('ρ', 'p', Script::Greek, false),
    ('ι', 'i', Script::Greek, false),
    // Armenian
    ('օ', 'o', Script::Armenian, true),
    ('ս', 'u', Script::Armenian, true),
    ('հ', 'h', Script::Armenian, true),
];

/// Single-character compatibility variants: `(char, name, preferred)`.
const VARIANTS: &[(char, &str, char)] = &[
    ('\u{00B5}', "MICRO SIGN", 'μ'),
    ('\u{2126}', "OHM SIGN", 'Ω'),
    ('\u{212A}', "KELVIN SIGN", 'K'),
    ('\u{212B}', "ANGSTROM SIGN", 'Å'),
    ('\u{210F}', "PLANCK CONSTANT OVER TWO PI", 'ħ'),
    ('\u{210E}', "PLANCK CONSTANT", 'h'),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confusable {
    /// A compatibility variant of a letter, e.g. MICRO SIGN for `μ`.
    Variant {
        ch: char,
        name: &'static str,
        preferred: Option<char>,
    },
    /// A letter that looks like a Latin letter.
    Lookalike {
        ch: char,
        script: Script,
        looks_like: char,
    },
    /// Letters from two scripts that should not mix.
    MixedScript { first: Script, second: Script },
}

impl fmt::Display for Confusable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Confusable::Variant {
                ch,
                name,
                preferred: Some(p),
            } => write!(
                f,
                "contains {name} U+{:04X}, which looks like '{p}' (U+{:04X}); use '{p}' instead",
                ch as u32, p as u32
            ),
            Confusable::Variant {
                ch,
                name,
                preferred: None,
            } => write!(
                f,
                "contains '{ch}' (U+{:04X}), a {name} variant of a Latin letter",
                ch as u32
            ),
            Confusable::Lookalike {
                ch,
                script,
                looks_like,
            } => write!(
                f,
                "contains {script} '{ch}' (U+{:04X}), which looks like Latin '{looks_like}'",
                ch as u32
            ),
            Confusable::MixedScript { first, second } => {
                write!(f, "mixes {first} and {second} scripts")
            }
        }
    }
}

fn lookalike(c: char) -> Option<(char, Script, bool)> {
    LOOKALIKES
        .iter()
        .find(|(ch, ..)| *ch == c)
        .map(|&(_, latin, script, strong)| (latin, script, strong))
}

fn variant(c: char) -> Option<Confusable> {
    if let Some(&(ch, name, preferred)) = VARIANTS.iter().find(|(ch, ..)| *ch == c) {
        return Some(Confusable::Variant {
            ch,
            name,
            preferred: Some(preferred),
        });
    }
    let name = match c as u32 {
        0xFF21..=0xFF3A | 0xFF41..=0xFF5A => "fullwidth",
        0x1D400..=0x1D6A5 => "mathematical",
        _ => return None,
    };
    Some(Confusable::Variant {
        ch: c,
        name,
        preferred: None,
    })
}

/// Checks one identifier. Returns the most specific finding, if any.
pub fn check_identifier(ident: &str) -> Option<Confusable> {
    if ident.is_ascii() {
        return None;
    }
    if let Some(v) = ident.chars().find_map(variant) {
        return Some(v);
    }

    let has_latin = ident.chars().any(|c| script(c) == Some(Script::Latin));
    let mut letters = ident
        .chars()
        .filter(|&c| c != '_' && !c.is_ascii_digit())
        .peekable();
    let all_strong =
        letters.peek().is_some() && letters.all(|c| matches!(lookalike(c), Some((_, _, true))));
    for c in ident.chars() {
        if let Some((looks_like, script, _)) = lookalike(c) {
            if has_latin || all_strong {
                return Some(Confusable::Lookalike {
                    ch: c,
                    script,
                    looks_like,
                });
            }
        }
    }

    let mut first = None;
    for s in ident.chars().filter_map(script) {
        match first {
            None => first = Some(s),
            Some(f) if f == s => {}
            Some(f) => {
                let pair_ok = matches!(
                    (f, s),
                    (Script::Latin, Script::Greek) | (Script::Greek, Script::Latin)
                );
                if !pair_ok {
                    return Some(Confusable::MixedScript {
                        first: f,
                        second: s,
                    });
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warns(s: &str) -> bool {
        check_identifier(s).is_some()
    }

    #[test]
    fn ascii_and_scientific_greek_are_clean() {
        for ok in [
            "x",
            "velocity",
            "d",
            "inv",
            "grad",
            "ψ",
            "ω",
            "ħ",
            "λ",
            "α",
            "ν",
            "ρ",
            "Δt",
            "λ_max",
            "μm",
            "σx",
            "Ω",
            "ψ2",
            "привет",
            "变量",
            "é",
        ] {
            assert_eq!(check_identifier(ok), None, "{ok}");
        }
    }

    #[test]
    fn cyrillic_lookalike_in_latin_word() {
        assert_eq!(
            check_identifier("pаth"),
            Some(Confusable::Lookalike {
                ch: 'а',
                script: Script::Cyrillic,
                looks_like: 'a'
            })
        );
    }

    #[test]
    fn whole_script_lookalikes() {
        assert!(warns("х"));
        assert!(warns("ох"));
        assert!(warns("х_1"));
        assert!(warns("ο")); // Greek omicron
        assert!(warns("Α")); // Greek capital alpha
    }

    #[test]
    fn weak_greek_lookalike_only_next_to_latin() {
        assert!(!warns("ν"));
        assert_eq!(
            check_identifier("νmax"),
            Some(Confusable::Lookalike {
                ch: 'ν',
                script: Script::Greek,
                looks_like: 'v'
            })
        );
        assert!(warns("vαlue"));
    }

    #[test]
    fn mixed_scripts() {
        assert_eq!(
            check_identifier("xЖ"),
            Some(Confusable::MixedScript {
                first: Script::Latin,
                second: Script::Cyrillic
            })
        );
        assert_eq!(
            check_identifier("ψж"),
            Some(Confusable::MixedScript {
                first: Script::Greek,
                second: Script::Cyrillic
            })
        );
    }

    #[test]
    fn compatibility_variants() {
        assert!(matches!(
            check_identifier("\u{00B5}m"),
            Some(Confusable::Variant {
                preferred: Some('μ'),
                ..
            })
        ));
        assert!(warns("\u{212A}"));
        assert!(warns("\u{210F}"));
        assert!(warns("ｘ"));
        assert!(warns("\u{1D465}"));
    }
}
