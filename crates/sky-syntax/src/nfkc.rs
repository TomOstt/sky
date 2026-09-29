//! Unicode Normalization Form KC (NFKC), for rejecting identifiers that are
//! not already in NFKC. The lexer never normalizes source: [`check`] decides
//! whether an identifier is in NFKC (normalizing only the short segments
//! around NFKC_QC=Maybe characters, in a small reused buffer), and [`nfkc`]
//! is used only to spell the expected identifier in the error message.
//!
//! The tables in `nfkc_table.rs` are generated from the pinned Unicode
//! version (see [`UNICODE_VERSION`]) by
//! `crates/sky-diag/tools/gen_display_unsafe.py`. This is an in-crate
//! implementation rather than the `unicode-normalization` crate because that
//! crate's Unicode version (17.0.0 in 0.1.25) does not match the pinned
//! 18.0.0 data the rest of the lexer uses.

use std::cmp::Ordering;

pub(crate) use crate::nfkc_table::UNICODE_VERSION;

// The normalizer, the lexer's other tables, and the renderer's table come
// from one generator run over one pinned Unicode version.
const _: () = {
    let (a, b, c) = UNICODE_VERSION;
    let (x, y, z) = sky_diag::DISPLAY_UNSAFE_UNICODE_VERSION;
    assert!(a == x && b == y && c == z);
};
use crate::nfkc_table::{CCC, CHAR_PROPS, COMPOSITION, DECOMPOSITION, DECOMPOSITION_DATA};
#[cfg(test)]
use crate::nfkc_table::{NFKC_QC_MAYBE, NFKC_QC_NO};

/// The stream-safe limit (UAX #15, section 13): the most non-starters
/// (characters with a nonzero combining class, counted in each character's
/// full decomposition) an identifier may have in a row. It bounds the work
/// and memory of [`check`].
pub(crate) const MAX_NON_STARTERS: usize = 30;

// Hangul syllable constants (Unicode section 3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

/// The result of the NFKC quick check (UAX #15, section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuickCheck {
    Yes,
    No,
    Maybe,
}

#[cfg(test)]
fn in_ranges(ranges: &[(u32, u32)], c: u32) -> bool {
    ranges
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                Ordering::Less
            } else if lo > c {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

#[cfg(test)]
thread_local! {
    /// [`ccc`] calls on this thread, for the step-bound test.
    pub(crate) static CCC_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Canonical_Combining_Class.
pub(crate) fn ccc(c: u32) -> u8 {
    #[cfg(test)]
    CCC_LOOKUPS.with(|n| n.set(n.get() + 1));
    match CCC.binary_search_by(|&(lo, hi, _)| {
        if hi < c {
            Ordering::Less
        } else if lo > c {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }) {
        Ok(i) => CCC[i].2,
        Err(_) => 0,
    }
}

/// NFKC_QC of one code point, from the NFKC_QC tables (tests use it to
/// check [`CharInfo`], which reads `CHAR_PROPS`).
#[cfg(test)]
pub(crate) fn quick_check_char(c: u32) -> QuickCheck {
    if in_ranges(NFKC_QC_NO, c) {
        QuickCheck::No
    } else if in_ranges(NFKC_QC_MAYBE, c) {
        QuickCheck::Maybe
    } else {
        QuickCheck::Yes
    }
}

/// The NFKC quick check of a whole string: `Yes` and `No` are definite,
/// `Maybe` needs a comparison with [`nfkc`].
#[cfg(test)]
pub(crate) fn quick_check(s: &str) -> QuickCheck {
    let mut last_ccc = 0;
    let mut result = QuickCheck::Yes;
    for c in s.chars() {
        let c = c as u32;
        if c < 0x80 {
            last_ccc = 0;
            continue;
        }
        let cc = ccc(c);
        if last_ccc > cc && cc != 0 {
            return QuickCheck::No;
        }
        match quick_check_char(c) {
            QuickCheck::No => return QuickCheck::No,
            QuickCheck::Maybe => result = QuickCheck::Maybe,
            QuickCheck::Yes => {}
        }
        last_ccc = cc;
    }
    result
}

/// Stream-safe counts of `c`: how many non-starters begin and end its full
/// decomposition, and whether it has no starter at all.
#[cfg(test)]
pub(crate) fn non_starters(c: u32) -> (u8, u8, bool) {
    let info = CharInfo::of(c);
    (info.lead, info.trail, info.all_non_starters)
}

/// True if `c` is a non-starter or its full decomposition starts with one
/// (a combining mark, for the stream-safe count).
pub(crate) fn is_non_starter(c: char) -> bool {
    CharInfo::of(c as u32).lead > 0
}

/// What the lexer needs to know about one character, from one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CharInfo {
    ccc: u8,
    qc: QuickCheck,
    lead: u8,
    trail: u8,
    all_non_starters: bool,
}

impl CharInfo {
    pub(crate) fn of(c: u32) -> CharInfo {
        if c < 0x80 {
            return CharInfo {
                ccc: 0,
                qc: QuickCheck::Yes,
                lead: 0,
                trail: 0,
                all_non_starters: false,
            };
        }
        // One lookup for everything (`CHAR_PROPS` agrees with `CCC` and
        // the NFKC_QC tables; tests check every code point).
        let found = CHAR_PROPS.binary_search_by(|&(lo, hi, ..)| {
            if hi < c {
                Ordering::Less
            } else if lo > c {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        let Ok(i) = found else {
            return CharInfo::of(0);
        };
        let (_, _, ccc, qc, lead, trail, all_non_starters) = CHAR_PROPS[i];
        CharInfo {
            ccc,
            qc: match qc {
                0 => QuickCheck::Yes,
                1 => QuickCheck::No,
                _ => QuickCheck::Maybe,
            },
            lead,
            trail,
            all_non_starters,
        }
    }

    /// Stable for NFKC: text before a stable character and text from it on
    /// normalize independently. It is a starter (class 0) whose full
    /// decomposition starts with a starter, and NFKC_QC=Yes, so it never
    /// combines with a preceding character (every second character of a
    /// composition, Hangul V and T jamo included, is NFKC_QC=Maybe; tests
    /// check this against the composition data for every code point).
    fn is_stable(self) -> bool {
        self.ccc == 0 && self.lead == 0 && self.qc == QuickCheck::Yes
    }
}

/// The result of [`check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Check {
    Nfkc,
    NotNfkc,
    /// More than [`MAX_NON_STARTERS`] non-starters in a row; the byte
    /// offset of the character that holds the first one over the limit.
    TooManyNonStarters {
        at: usize,
    },
}

/// Whether `s` is in NFKC, in one pass and constant memory, or that it
/// breaks the stream-safe limit (checked first, character by character, so
/// the error is the first problem in the string).
///
/// The quick check (UAX #15, section 9) decides every string without an
/// NFKC_QC=Maybe character. A Maybe character is verified by normalizing
/// only its segment: from the last stable character before it to the next
/// stable character after it (see [`CharInfo::is_stable`]). The segment is
/// normalized one starter and its non-starters at a time, in one small
/// reused buffer, and compared with the source as it goes; the limit keeps
/// that buffer small.
pub(crate) fn check(s: &str) -> Check {
    check_with_limit(s, MAX_NON_STARTERS)
}

/// True if `s` is in NFKC, whatever its non-starter runs.
#[cfg(test)]
pub(crate) fn is_nfkc(s: &str) -> bool {
    check_with_limit(s, usize::MAX) == Check::Nfkc
}

fn check_with_limit(s: &str, max_non_starters: usize) -> Check {
    let mut run = 0usize;
    let mut last_ccc = 0u8;
    let mut segment_start = 0;
    let mut maybe = false;
    let mut unit = Vec::new();
    for (i, ch) in s.char_indices() {
        let info = CharInfo::of(ch as u32);
        // Stream-safe count (UAX #15, section 13).
        if usize::from(info.lead) > max_non_starters - run {
            return Check::TooManyNonStarters { at: i };
        }
        run = if info.all_non_starters {
            run + usize::from(info.lead)
        } else {
            usize::from(info.trail)
        };
        if info.is_stable() {
            if maybe && !segment_is_nfkc(&s[segment_start..i], &mut unit) {
                return Check::NotNfkc;
            }
            maybe = false;
            segment_start = i;
        }
        if info.ccc != 0 && last_ccc > info.ccc {
            return Check::NotNfkc;
        }
        match info.qc {
            QuickCheck::No => return Check::NotNfkc,
            QuickCheck::Maybe => maybe = true,
            QuickCheck::Yes => {}
        }
        last_ccc = info.ccc;
    }
    if maybe && !segment_is_nfkc(&s[segment_start..], &mut unit) {
        return Check::NotNfkc;
    }
    Check::Nfkc
}

/// True if `seg` is its own NFKC. `seg` starts at a stable character or at
/// the start of the string. It is normalized one unit at a time (a starter
/// and the non-starters after it, or the leading non-starters) in `unit`,
/// which is reused, and compared with `seg` as it goes. A unit only grows
/// past one starter plus the non-starters of one run, so with the
/// stream-safe limit it stays small.
fn segment_is_nfkc(seg: &str, unit: &mut Vec<(u32, u8)>) -> bool {
    if unit.capacity() == 0 {
        unit.reserve(MAX_NON_STARTERS + 20);
    }
    unit.clear();
    let mut source = seg.chars().map(|c| c as u32);
    let mut tmp = [0; 3];
    for ch in seg.chars() {
        for &x in decomposition(ch as u32, &mut tmp) {
            let cc = ccc(x);
            if cc == 0 && !unit.is_empty() {
                compose_unit(unit);
                // A starter composes with the unit's starter only if nothing
                // is left between them (Hangul L + V, LV + T, ...).
                if unit.len() == 1 && unit[0].1 == 0 {
                    if let Some(p) = compose_pair(unit[0].0, x) {
                        unit[0].0 = p;
                        continue;
                    }
                }
                if !unit.drain(..).all(|(c, _)| source.next() == Some(c)) {
                    return false;
                }
            }
            unit.push((x, cc));
        }
    }
    compose_unit(unit);
    unit.drain(..).all(|(c, _)| source.next() == Some(c)) && source.next().is_none()
}

/// Canonical ordering and composition of one unit, in place: an optional
/// starter followed by non-starters only.
fn compose_unit(unit: &mut Vec<(u32, u8)>) {
    let first = usize::from(unit.first().is_some_and(|&(_, cc)| cc == 0));
    // Stable insertion sort by class; the run has at most
    // `MAX_NON_STARTERS` characters (in the lexer).
    for i in first + 1..unit.len() {
        let mut j = i;
        while j > first && unit[j - 1].1 > unit[j].1 {
            unit.swap(j - 1, j);
            j -= 1;
        }
    }
    if first == 0 {
        return;
    }
    let mut last_ccc: Option<u8> = None;
    let mut kept = 1;
    for r in 1..unit.len() {
        let (c, cc) = unit[r];
        let blocked = matches!(last_ccc, Some(lc) if lc == 0 || lc >= cc);
        if !blocked {
            if let Some(p) = compose_pair(unit[0].0, c) {
                unit[0].0 = p;
                continue;
            }
        }
        last_ccc = Some(cc);
        unit[kept] = unit[r];
        kept += 1;
    }
    unit.truncate(kept);
}

/// The full compatibility decomposition of `c` (itself if it has none);
/// Hangul syllables are decomposed into `tmp`.
fn decomposition(c: u32, tmp: &mut [u32; 3]) -> &[u32] {
    let s = c.wrapping_sub(S_BASE);
    if s < S_COUNT {
        tmp[0] = L_BASE + s / N_COUNT;
        tmp[1] = V_BASE + (s % N_COUNT) / T_COUNT;
        if s % T_COUNT != 0 {
            tmp[2] = T_BASE + s % T_COUNT;
            return &tmp[..3];
        }
        return &tmp[..2];
    }
    match DECOMPOSITION.binary_search_by_key(&c, |&(cp, _, _)| cp) {
        Ok(i) => {
            let (_, off, len) = DECOMPOSITION[i];
            &DECOMPOSITION_DATA[off as usize..off as usize + len as usize]
        }
        Err(_) => {
            tmp[0] = c;
            &tmp[..1]
        }
    }
}

fn decompose_into(c: u32, out: &mut Vec<u32>) {
    out.extend_from_slice(decomposition(c, &mut [0; 3]));
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    // Hangul L + V, then LV + T.
    let l = a.wrapping_sub(L_BASE);
    let v = b.wrapping_sub(V_BASE);
    if l < L_COUNT && v < V_COUNT {
        return Some(S_BASE + (l * V_COUNT + v) * T_COUNT);
    }
    let s = a.wrapping_sub(S_BASE);
    let t = b.wrapping_sub(T_BASE);
    if s < S_COUNT && s % T_COUNT == 0 && t > 0 && t < T_COUNT {
        return Some(a + t);
    }
    COMPOSITION
        .binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b)))
        .ok()
        .map(|i| COMPOSITION[i].2)
}

/// The NFKC form of `s`: full compatibility decomposition, canonical
/// ordering, then canonical composition (UAX #15).
pub(crate) fn nfkc(s: &str) -> String {
    let mut d = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c as u32, &mut d);
    }
    // Canonical ordering: stable-sort each run of non-starters by class.
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < d.len() && ccc(d[i]) != 0 {
            i += 1;
        }
        d[start..i].sort_by_key(|&c| ccc(c));
    }
    // Canonical composition.
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    // Class of the last character appended since the starter, if any.
    let mut last_ccc: Option<u8> = None;
    for c in d {
        let cc = ccc(c);
        if let Some(si) = starter {
            let blocked = matches!(last_ccc, Some(lc) if lc == 0 || lc >= cc);
            if !blocked {
                if let Some(p) = compose_pair(out[si], c) {
                    out[si] = p;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
            last_ccc = None;
        } else {
            last_ccc = Some(cc);
        }
        out.push(c);
    }
    out.into_iter()
        .map(|c| char::from_u32(c).expect("tables hold only scalar values"))
        .collect()
}

#[cfg(test)]
mod tests;
