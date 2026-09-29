//! Unicode Normalization Form KC (NFKC), for rejecting identifiers that are
//! not already in NFKC. The lexer never normalizes source: [`nfkc`] is used
//! only to spell the expected identifier in the error message.
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
use crate::nfkc_table::{
    CCC, COMPOSITION, DECOMPOSITION, DECOMPOSITION_DATA, NFKC_QC_MAYBE, NFKC_QC_NO,
};

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

/// NFKC_QC of one code point.
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

/// True if `s` is in NFKC.
pub(crate) fn is_nfkc(s: &str) -> bool {
    match quick_check(s) {
        QuickCheck::Yes => true,
        QuickCheck::No => false,
        QuickCheck::Maybe => nfkc(s) == s,
    }
}

fn decompose_into(c: u32, out: &mut Vec<u32>) {
    let s = c.wrapping_sub(S_BASE);
    if s < S_COUNT {
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if s % T_COUNT != 0 {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    match DECOMPOSITION.binary_search_by_key(&c, |&(cp, _, _)| cp) {
        Ok(i) => {
            let (_, off, len) = DECOMPOSITION[i];
            out.extend_from_slice(&DECOMPOSITION_DATA[off as usize..off as usize + len as usize]);
        }
        Err(_) => out.push(c),
    }
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
