/// Unicode bidi controls: LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI, LRM, RLM, ALM.
pub(crate) const BIDI_CONTROLS: &[char] = &[
    '\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}',
    '\u{2069}', '\u{200E}', '\u{200F}', '\u{061C}',
];

#[derive(Debug, PartialEq, Eq)]
pub struct BidiControl {
    pub byte_offset: usize,
    pub ch: char,
}

/// True for the bidi controls [`find_bidi_control`] looks for.
pub(crate) fn is_bidi_control(c: char) -> bool {
    BIDI_CONTROLS.contains(&c)
}

/// Returns the first bidi control character in `src`, if any.
pub fn find_bidi_control(src: &str) -> Option<BidiControl> {
    src.char_indices()
        .find(|&(_, c)| is_bidi_control(c))
        .map(|(byte_offset, ch)| BidiControl { byte_offset, ch })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_source_is_accepted() {
        assert_eq!(find_bidi_control("velocity = 10 km / 30 min\n"), None);
    }

    #[test]
    fn unicode_operators_are_accepted() {
        assert_eq!(find_bidi_control("df = ∂f/∂x\nx = A⁻¹ @ b\n"), None);
    }

    #[test]
    fn rlo_is_rejected() {
        let src = "x = 1 \u{202E} # comment";
        assert_eq!(
            find_bidi_control(src),
            Some(BidiControl {
                byte_offset: 6,
                ch: '\u{202E}'
            })
        );
    }
}
