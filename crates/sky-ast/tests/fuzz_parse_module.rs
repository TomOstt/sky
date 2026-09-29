//! Runs the `parse_module` fuzz check on its seed corpus with stable Rust,
//! so the deep seeds are covered even where `cargo fuzz` (nightly) isn't.
//! The check itself is `fuzz/harness/parse_module.rs`, the same file the
//! fuzz target includes, and it runs every input inside `with_deep_stack`.

#[path = "../../../fuzz/harness/parse_module.rs"]
mod harness;

use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus/parse_module")
}

/// Reads a seed and checks it is exactly `expected` (mode byte `0`, then
/// source), so the seed files can't drift from what this test asserts.
fn seed(name: &str, expected_src: &str) -> Vec<u8> {
    let data = std::fs::read(corpus_dir().join(name)).expect("seed file");
    let mut expected = vec![b'0'];
    expected.extend_from_slice(expected_src.as_bytes());
    assert!(data == expected, "seed {name} differs from the test's copy");
    data
}

#[test]
fn nested_300_levels_is_a_depth_error() {
    let src = format!("{}1{}\n", "(".repeat(300), ")".repeat(300));
    let out = harness::run(&seed("nest_300", &src));
    assert_eq!(out.stmts, Some(0));
    assert_eq!(
        out.errors,
        ["expression is nested too deeply (the limit is 256 levels)"]
    );
}

#[test]
fn nested_256_levels_parses() {
    let src = format!("{}1{}\n", "(".repeat(256), ")".repeat(256));
    let out = harness::run(&seed("nest_256", &src));
    assert_eq!(out.stmts, Some(1));
    assert!(out.errors.is_empty(), "{:?}", out.errors);
}

#[test]
fn nested_256_index_levels_parses() {
    let src = format!("{}1{}\n", "a[".repeat(256), "]".repeat(256));
    let out = harness::run(&seed("nest_index_256", &src));
    assert_eq!(out.stmts, Some(1));
    assert!(out.errors.is_empty(), "{:?}", out.errors);
}

#[test]
fn every_compound_operator_on_every_target_kind_parses() {
    let mut src = String::new();
    for op in ["+=", "-=", "*=", "/=", "^=", "@="] {
        for target in ["x", "A[i, j]", "a.b"] {
            src.push_str(&format!("{target} {op} y\n"));
        }
    }
    let out = harness::run(&seed("compound_assign", &src));
    assert_eq!(out.stmts, Some(18));
    assert!(out.errors.is_empty(), "{:?}", out.errors);
}

#[test]
fn and_chain_of_10000_parses() {
    let src = format!("{}\n", vec!["a"; 10_000].join(" and "));
    let out = harness::run(&seed("and_chain_10000", &src));
    assert_eq!(out.stmts, Some(1));
    assert!(out.errors.is_empty(), "{:?}", out.errors);
}

#[test]
fn every_seed_passes_the_fuzz_check() {
    let mut names: Vec<_> = std::fs::read_dir(corpus_dir())
        .expect("corpus dir")
        .map(|e| e.expect("dir entry").path())
        .collect();
    names.sort();
    assert!(names.len() >= 10, "{names:?}");
    for path in names {
        let data = std::fs::read(&path).expect("seed file");
        let _ = harness::run(&data);
    }
}
