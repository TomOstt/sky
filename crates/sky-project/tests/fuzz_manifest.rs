//! Runs the `manifest` fuzz check on its seed corpus with stable Rust, so
//! the seeds are covered where `cargo fuzz` (nightly) isn't. The check itself
//! is `fuzz/harness/manifest.rs`, the same file the fuzz target includes.

#[path = "../../../fuzz/harness/manifest.rs"]
mod harness;

use harness::Outcome;
use sky_project::{Edition, MAX_MANIFEST_BYTES};
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus/manifest")
}

fn seed(name: &str) -> Outcome {
    harness::run(&std::fs::read(corpus_dir().join(name)).expect("seed file"))
}

fn rejected(name: &str) -> String {
    match seed(name) {
        Outcome::Rejected(m) => m,
        other => panic!("{name}: {other:?}"),
    }
}

#[test]
fn accepted_seeds() {
    for name in ["ok_edition", "ok_empty", "ok_crlf_comments"] {
        assert_eq!(seed(name), Outcome::Accepted(Edition::E2026), "{name}");
    }
}

#[test]
fn rejected_seeds_say_why() {
    assert_eq!(
        rejected("err_too_new"),
        "edition 2031 is newer than this Sky (supports 2026)"
    );
    assert_eq!(rejected("err_typo_key"), "unknown key `editon`");
    assert_eq!(rejected("err_duplicate"), "duplicate key `edition`");
    assert_eq!(rejected("err_table"), "expected a key");
    assert_eq!(
        rejected("err_escape"),
        "escapes are not supported in sky.toml strings"
    );
    assert_eq!(rejected("err_unterminated"), "unterminated string");
    assert_eq!(
        rejected("err_bidi_comment"),
        "bidi control character U+202E in sky.toml"
    );
    assert_eq!(
        rejected("err_not_string"),
        "expected a double-quoted string"
    );
    assert_eq!(
        rejected("err_lone_cr"),
        "control character U+000D in sky.toml"
    );
    assert_eq!(
        rejected("err_edition_overflow"),
        "edition 65535 is newer than this Sky (supports 2026)"
    );
    assert_eq!(rejected("err_non_ascii_key"), "expected a key");
    assert_eq!(seed("err_not_utf8"), Outcome::NotUtf8);
}

#[test]
fn oversized_input_is_rejected_not_parsed() {
    let mut big = b"# ".to_vec();
    big.resize(MAX_MANIFEST_BYTES + 1, b'a');
    assert!(matches!(harness::run(&big), Outcome::Rejected(_)));
}

#[test]
fn every_seed_and_every_prefix_passes_the_fuzz_check() {
    let mut n = 0;
    for entry in std::fs::read_dir(corpus_dir()).expect("corpus dir") {
        let data = std::fs::read(entry.expect("entry").path()).expect("seed file");
        // Truncation at every byte, including mid-UTF-8, is the cheapest
        // way to reach every error path from a valid manifest.
        for end in 0..=data.len() {
            let _ = harness::run(&data[..end]);
        }
        n += 1;
    }
    assert!(n >= 15, "{n}");
}
