//! The `sir_parse` fuzz check, shared by the fuzz target
//! (`fuzz/fuzz_targets/sir_parse.rs`) and a stable test in sky-ir
//! (`crates/sky-ir/tests/fuzz_sir_parse.rs`), which both include this file
//! with `#[path]`. It depends only on sky-ir and sky-diag.
//!
//! The input is SIR text (UTF-8, lossy). Invariants, checked on every input:
//! - `parse`, `verify`, `print` and rendering diagnostics never panic and
//!   never overflow the stack (their recursion is bounded by
//!   `MAX_REGION_DEPTH`).
//! - A parse error's span satisfies `start <= end <= len` and lies on char
//!   boundaries, so the diagnostic renderer can slice the source.
//! - On success: `print(m)` parses back to exactly `m`, printing that gives
//!   the same text (round-trip exactness), and if `m` verifies it is already
//!   in canonical order (`canonicalize(m) == m`).

use sky_diag::SourceFile;
use sky_ir::{canonicalize, parse, print, verify};

/// What a run produced, for tests that want more than "didn't crash".
/// The fuzz target ignores it, hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The parse error, if any.
    pub error: Option<String>,
    /// Whether the parsed module verified.
    pub verified: bool,
}

pub fn check(data: &[u8]) -> Outcome {
    let src = String::from_utf8_lossy(data);
    let m = match parse(&src) {
        Ok(m) => m,
        Err(e) => {
            let (start, end) = (e.span.start as usize, e.span.end as usize);
            assert!(
                start <= end && end <= src.len(),
                "span {:?} out of bounds",
                e.span
            );
            assert!(src.is_char_boundary(start) && src.is_char_boundary(end));
            let file = SourceFile::new("fuzz.sir", src.to_string());
            let _ = e.to_diagnostic().render_to_string(&file);
            return Outcome {
                error: Some(e.message),
                verified: false,
            };
        }
    };
    let verified = match verify(&m) {
        Ok(()) => {
            assert_eq!(canonicalize(&m), m, "a parsed module is canonical");
            true
        }
        Err(errors) => {
            for e in errors {
                let _ = e.to_string();
            }
            false
        }
    };
    let text = print(&m);
    let again = parse(&text).unwrap_or_else(|e| panic!("printed text does not parse: {e}\n{text}"));
    assert!(
        again == m,
        "print/parse round trip changed the module:\n{text}"
    );
    assert_eq!(print(&again), text);
    Outcome {
        error: None,
        verified,
    }
}
