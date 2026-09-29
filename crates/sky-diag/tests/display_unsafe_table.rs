//! The generated table must equal the property definition in
//! spec/diagnostics.md ("Terminal safety") for every code point.
//!
//! The reference is built here, independently of the generator, from the
//! verbatim UCD records in `tests/data/ucd-display-unsafe-source.txt`
//! (Default_Ignorable_Code_Point and General_Category=Cf), plus U+2028/U+2029
//! and std's `char::is_control` for the C0/DEL/C1 controls.

use sky_diag::{display_safe, is_display_unsafe, DISPLAY_UNSAFE_UNICODE_VERSION};

const SOURCE: &str = include_str!("data/ucd-display-unsafe-source.txt");

/// Inclusive ranges for records whose property field is `prop`.
fn ranges(prop: &str) -> Vec<(u32, u32)> {
    let out: Vec<(u32, u32)> = SOURCE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let data = l.split('#').next().unwrap();
            let (cps, p) = data.split_once(';')?;
            (p.trim() == prop).then(|| {
                let cps = cps.trim();
                let (lo, hi) = cps.split_once("..").unwrap_or((cps, cps));
                (
                    u32::from_str_radix(lo, 16).unwrap(),
                    u32::from_str_radix(hi, 16).unwrap(),
                )
            })
        })
        .collect();
    assert!(!out.is_empty(), "no {prop} records");
    out
}

#[test]
fn source_data_is_the_pinned_unicode_version() {
    let (a, b, c) = DISPLAY_UNSAFE_UNICODE_VERSION;
    let v = format!("Unicode {a}.{b}.{c} UCD");
    assert!(SOURCE.lines().next().unwrap().contains(&v), "{v}");
    assert_eq!(DISPLAY_UNSAFE_UNICODE_VERSION, (18, 0, 0));
}

#[test]
fn table_matches_the_property_definition_for_every_code_point() {
    let di = ranges("Default_Ignorable_Code_Point");
    let cf = ranges("Cf");
    let within = |rs: &[(u32, u32)], cp: u32| rs.iter().any(|&(lo, hi)| lo <= cp && cp <= hi);
    let mut escaped = 0;
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else {
            continue; // surrogates
        };
        let expected = within(&di, cp)
            || within(&cf, cp)
            || cp == 0x2028
            || cp == 0x2029
            || (c.is_control() && c != '\t');
        assert_eq!(is_display_unsafe(c), expected, "U+{cp:04X}");
        escaped += usize::from(expected);
    }
    assert_eq!(escaped, 4272);
}

#[test]
fn every_unsafe_character_is_escaped_to_the_width_of_its_escape() {
    for c in (0..=0x10FFFFu32)
        .filter_map(char::from_u32)
        .filter(|&c| is_display_unsafe(c))
    {
        let s = c.to_string();
        let e = display_safe(&s);
        assert_eq!(e, format!("\\u{{{:04X}}}", c as u32));
        assert!(!e.chars().any(is_display_unsafe));
    }
    assert_eq!(display_safe("\t"), "\\u{0009}");
}
