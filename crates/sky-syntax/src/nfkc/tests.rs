use super::*;
use crate::lexer::{is_ident_continue, is_ident_start};

const CONFORMANCE: &str = include_str!("../../tests/data/nfkc-conformance.txt");

fn parse_cps(field: &str) -> String {
    field
        .split_whitespace()
        .map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap())
        .collect()
}

fn hex(s: &str) -> String {
    s.chars()
        .map(|c| format!("{:04X}", c as u32))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn unicode_version_is_the_pinned_one() {
    assert_eq!(UNICODE_VERSION, (18, 0, 0));
    assert_eq!(UNICODE_VERSION, sky_diag::DISPLAY_UNSAFE_UNICODE_VERSION);
    let (a, b, c) = UNICODE_VERSION;
    assert!(CONFORMANCE
        .lines()
        .next()
        .unwrap()
        .contains(&format!("Unicode {a}.{b}.{c} NormalizationTest.txt")));
}

/// Every NFKC case in the Unicode NormalizationTest.txt.
#[test]
fn nfkc_conforms_to_normalization_test() {
    let mut cases = 0;
    for line in CONFORMANCE.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (src, expected) = match line.split_once(';') {
            Some((a, b)) => (parse_cps(a), parse_cps(b)),
            None => (parse_cps(line), parse_cps(line)),
        };
        let got = nfkc(&src);
        assert_eq!(hex(&got), hex(&expected), "NFKC({})", hex(&src));
        // The quick check never contradicts the full comparison.
        match quick_check(&src) {
            QuickCheck::Yes => assert_eq!(src, expected, "quick check Yes for {}", hex(&src)),
            QuickCheck::No => assert_ne!(src, expected, "quick check No for {}", hex(&src)),
            QuickCheck::Maybe => {}
        }
        assert_eq!(is_nfkc(&src), src == expected, "{}", hex(&src));
        cases += 1;
    }
    assert!(cases > 30_000, "{cases} cases");
}

/// For every single code point, NFKC_QC agrees with the decomposition data:
/// `No` exactly when the code point on its own is not its own NFKC.
#[test]
fn quick_check_agrees_with_nfkc_for_every_code_point() {
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        let s = c.to_string();
        let changes = nfkc(&s) != s;
        let qc = quick_check_char(c as u32);
        assert_eq!(
            qc == QuickCheck::No,
            changes,
            "U+{:04X}: NFKC_QC {qc:?}, NFKC {}",
            c as u32,
            hex(&nfkc(&s))
        );
    }
}

/// The Unicode property the lexer relies on, checked for the pinned data:
/// XID_Start and XID_Continue are closed under NFKC. For every identifier
/// character, if it is XID_Start its NFKC form is one XID_Start followed by
/// XID_Continue characters; otherwise every character of its NFKC form is
/// XID_Continue.
#[test]
fn nfkc_preserves_xid_for_every_identifier_character() {
    use unicode_ident::{is_xid_continue, is_xid_start};
    let mut failures = Vec::new();
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if !is_ident_continue(c) && !is_ident_start(c) {
            continue;
        }
        let n = nfkc(&c.to_string());
        let ok = if is_xid_start(c) {
            let mut it = n.chars();
            it.next().is_some_and(is_xid_start) && it.all(is_xid_continue)
        } else {
            !n.is_empty() && n.chars().all(is_xid_continue)
        };
        if !ok {
            failures.push(format!("U+{:04X} -> {}", c as u32, hex(&n)));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures: {failures:?}",
        failures.len()
    );
}

/// XID_Start and XID_Continue are closed under NFKC, and so are the
/// lexer's identifier rules (which exclude invisible characters and `·`):
/// for every identifier character, an identifier start normalizes to one
/// start followed by continue characters, and any other identifier
/// character normalizes to continue characters only. This is what makes the
/// `debug_assert!` in the lexer's non-NFKC error unreachable.
#[test]
fn nfkc_of_every_identifier_character_is_an_identifier() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if !is_ident_continue(c) && !is_ident_start(c) {
            continue;
        }
        checked += 1;
        let n = nfkc(&c.to_string());
        let ok = if is_ident_start(c) {
            let mut it = n.chars();
            it.next().is_some_and(is_ident_start) && it.all(is_ident_continue)
        } else {
            !n.is_empty() && n.chars().all(is_ident_continue)
        };
        if !ok {
            failures.push(format!("U+{:04X} -> {}", c as u32, hex(&n)));
        }
    }
    assert!(checked > 100_000, "{checked}");
    assert!(
        failures.is_empty(),
        "{} failures: {failures:?}",
        failures.len()
    );
}

#[test]
fn examples() {
    assert_eq!(nfkc("e\u{0301}"), "\u{00E9}");
    assert_eq!(nfkc("\u{FB01}"), "fi");
    assert_eq!(nfkc("\u{FF58}"), "x");
    assert_eq!(nfkc("\u{1D431}"), "x"); // MATHEMATICAL BOLD SMALL X
    assert_eq!(nfkc("\u{1D465}"), "x"); // MATHEMATICAL ITALIC SMALL X
    assert_eq!(nfkc("\u{00B5}"), "\u{03BC}");
    assert_eq!(nfkc("\u{212A}"), "K");
    assert_eq!(nfkc("\u{210F}"), "\u{0127}");
    assert_eq!(nfkc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}"); // Hangul L V T
    assert_eq!(nfkc("\u{AC00}\u{11A8}"), "\u{AC01}"); // Hangul LV + T
    assert_eq!(nfkc("x\u{0300}"), "x\u{0300}"); // no precomposed form
    assert!(is_nfkc("x\u{0300}"));
    assert!(!is_nfkc("a\u{0300}"));
    assert_eq!(quick_check("a\u{0300}"), QuickCheck::Maybe);
    assert_eq!(quick_check("\u{FB01}"), QuickCheck::No);
    assert_eq!(quick_check("caf\u{00E9}"), QuickCheck::Yes);
    // Canonical order: a dot below (220) sorts before an acute (230).
    assert_eq!(nfkc("q\u{0301}\u{0323}"), "q\u{0323}\u{0301}");
    assert_eq!(quick_check("q\u{0301}\u{0323}"), QuickCheck::No);
}

#[test]
fn tables_are_sorted() {
    for t in [NFKC_QC_NO, NFKC_QC_MAYBE] {
        assert!(!t.is_empty());
        assert!(t.windows(2).all(|w| w[0].1 + 1 < w[1].0));
    }
    assert!(CCC.windows(2).all(|w| w[0].1 < w[1].0));
    assert!(DECOMPOSITION.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(COMPOSITION
        .windows(2)
        .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)));
}
