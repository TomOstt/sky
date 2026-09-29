use super::*;
use crate::lexer::{is_ident_continue, is_ident_start};
use std::cell::Cell;

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
    // The identifier tables (`unicode-ident`, whose Cargo requirement is
    // just "1") must be the same Unicode version as the NFKC tables, or a
    // `cargo update` could move XID to a newer version than NFKC.
    assert_eq!(UNICODE_VERSION, unicode_ident::UNICODE_VERSION);
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

/// Counts allocations made by the current thread, so the fast-path test
/// below is not disturbed by tests running in parallel.
mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct Counting;

    thread_local! {
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            // SAFETY: forwarded unchanged to the system allocator.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: forwarded unchanged to the system allocator.
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            // SAFETY: forwarded unchanged to the system allocator.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    /// Allocations made by `f` on this thread.
    pub fn count<T>(f: impl FnOnce() -> T) -> (usize, T) {
        let before = COUNT.with(Cell::get);
        let out = f();
        (COUNT.with(Cell::get) - before, out)
    }
}

/// The check the lexer runs on every non-ASCII identifier allocates nothing
/// when the quick check answers Yes (the common case) or No; only a Maybe
/// answer builds the NFKC form to compare.
#[test]
fn fast_path_allocates_nothing() {
    for s in [
        "caf\u{00E9}",
        "\u{03BC}m",
        "\u{0127}",
        "\u{AC00}\u{AC01}",
        "\u{0394}t_max",
        "x_1",
    ] {
        let (n, yes) = alloc_count::count(|| is_nfkc(s));
        assert!(yes, "{s:?}");
        assert_eq!(n, 0, "is_nfkc({s:?}) allocated");
        let (n, _) = alloc_count::count(|| quick_check(s));
        assert_eq!(n, 0, "quick_check({s:?}) allocated");
    }
    for s in ["\u{FB01}", "e\u{0301}\u{0323}"] {
        let (n, yes) = alloc_count::count(|| is_nfkc(s));
        assert!(!yes);
        assert_eq!(n, 0, "is_nfkc({s:?}) allocated");
    }
    // Maybe does allocate (the NFKC form); documents the known cost.
    let (n, _) = alloc_count::count(|| is_nfkc("x\u{0300}"));
    assert!(n > 0);
    // In the lexer, an NFKC-Yes identifier costs no more allocations than
    // an ASCII one of the same shape.
    let (ascii, _) = alloc_count::count(|| crate::lex("cafe = 1\n").unwrap());
    let (yes, _) = alloc_count::count(|| crate::lex("caf\u{00E9} = 1\n").unwrap());
    assert_eq!(ascii, yes);
}

/// Normalizing one starter followed by out-of-order combining marks (the
/// canonical-ordering worst case) takes a number of steps bounded by
/// n log n: counted as `ccc` lookups, not time, so the test is not flaky.
#[test]
fn canonical_ordering_is_n_log_n() {
    for n in [1_000usize, 16_000, 256_000] {
        let mut s = String::from("a");
        for i in 0..n {
            s.push(if i % 2 == 0 { '\u{0301}' } else { '\u{0323}' });
        }
        CCC_LOOKUPS.with(|c| c.set(0));
        let out = nfkc(&s);
        let lookups = CCC_LOOKUPS.with(Cell::get);
        // 220s sort before 230s; `a` + U+0323 composes to U+1EA1.
        assert!(out.starts_with('\u{1EA1}'));
        let bound = 8 * n * (usize::BITS - n.leading_zeros()) as usize;
        assert!(lookups <= bound, "n={n}: {lookups} ccc lookups > {bound}");
    }
}
