//! Language editions (`ROADMAP.md` idea 75; spec v0.1 "Editions").
//!
//! An edition may add syntax, keywords and diagnostics, but never changes the
//! meaning of a program that compiled before, so old code never silently means
//! something new. It lives here, at the bottom of the crate graph next to the
//! lexer and parser, because those are the first phases an edition will
//! change (a new keyword is a lexer decision), while `sky-project` (which
//! reads the manifest that names it) and `sky-ir` (which records it on every
//! module) can both depend on `sky-syntax` cheaply and neither needs to
//! depend on the other.

use std::fmt;

/// The language edition a project or module was written for. v1 has one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum Edition {
    /// The first edition.
    #[default]
    E2026,
}

impl Edition {
    /// The newest edition this Sky supports: what a file checked without a
    /// `sky.toml` uses.
    pub const LATEST: Edition = Edition::E2026;

    /// The edition a manifest that names none means. It is the *first*
    /// edition and never a later one, so adding editions cannot change what
    /// an old manifest means.
    pub const FIRST: Edition = Edition::E2026;

    /// The edition's year, as written in `sky.toml`.
    pub fn year(self) -> u16 {
        match self {
            Edition::E2026 => 2026,
        }
    }

    /// The edition named by `year`, if this Sky supports it.
    pub fn from_year(year: u16) -> Option<Edition> {
        match year {
            2026 => Some(Edition::E2026),
            _ => None,
        }
    }
}

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.year())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn year_round_trips() {
        assert_eq!(Edition::from_year(2026), Some(Edition::E2026));
        assert_eq!(Edition::LATEST.year(), 2026);
        assert_eq!(Edition::from_year(2031), None);
        assert_eq!(Edition::from_year(0), None);
        assert_eq!(Edition::LATEST.to_string(), "2026");
    }
}
