//! Byte-offset spans into a single source file.

use std::ops::Range;

/// A half-open byte range `start..end` into one source file's text.
///
/// Offsets are `u32` to keep AST nodes small, so a source file is addressable
/// up to `u32::MAX` bytes. `start <= end` is expected: [`Span::new`]
/// debug-asserts it, and the renderer tolerates spans built by hand that
/// violate it (they render as zero-width at `start`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Byte offset of the first byte in the span.
    pub start: u32,
    /// Byte offset one past the last byte in the span.
    pub end: u32,
}

impl Span {
    /// Creates the span `start..end`.
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "Span::new: start {start} > end {end}");
        Span { start, end }
    }

    /// Creates the zero-width span `offset..offset`, e.g. for "expected `)` here".
    pub fn point(offset: u32) -> Self {
        Span {
            start: offset,
            end: offset,
        }
    }

    /// Length in bytes (0 for an inverted span).
    pub fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// True if the span covers no bytes.
    pub fn is_empty(self) -> bool {
        self.end <= self.start
    }

    /// The smallest span covering both `self` and `other`.
    pub fn join(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// The span as a `usize` range, for slicing the source text.
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_and_is_empty() {
        assert_eq!(Span::new(3, 7).len(), 4);
        assert!(!Span::new(3, 7).is_empty());
        assert_eq!(Span::point(5).len(), 0);
        assert!(Span::point(5).is_empty());
    }

    #[test]
    fn inverted_span_built_by_hand_is_empty_without_panicking() {
        let s = Span { start: 9, end: 2 };
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn join_covers_both_and_is_commutative() {
        let a = Span::new(2, 5);
        let b = Span::new(8, 12);
        assert_eq!(a.join(b), Span::new(2, 12));
        assert_eq!(b.join(a), Span::new(2, 12));
        assert_eq!(a.join(Span::new(3, 4)), a);
    }

    #[test]
    fn range_matches_offsets() {
        assert_eq!(Span::new(1, 4).range(), 1..4);
        assert_eq!(&"x = 10 km"[Span::new(4, 6).range()], "10");
    }
}
