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

fn nested_ifs(n: usize) -> String {
    let mut src = String::new();
    for i in 0..n {
        src.push_str(&"\t".repeat(i));
        src.push_str("if a:\n");
    }
    src.push_str(&"\t".repeat(n));
    src.push_str("pass\n");
    src
}

#[test]
fn nested_256_blocks_parse() {
    let out = harness::run(&seed("blocks_256", &nested_ifs(256)));
    assert_eq!(out.stmts, Some(1));
    assert!(out.errors.is_empty(), "{:?}", out.errors);
}

#[test]
fn nested_257_blocks_are_a_depth_error() {
    let out = harness::run(&seed("blocks_257", &nested_ifs(257)));
    assert_eq!(out.stmts, Some(1));
    assert_eq!(
        out.errors,
        ["blocks are nested too deeply (the limit is 256 levels)"]
    );
}

#[test]
fn block_forms_parse_and_block_errors_recover() {
    let src = "\
def f(a, b: Vector[3]) -> Matrix[3, 3]:
    if a > 0 m:
        return b
    elif a < 0 m:
        pass
    else:
        while true:
            break
    return
x = 1 if c else 2 if d else 3
for i in range(1, n):
    for row in A:
        continue
    break
";
    let out = harness::run(&seed("block_forms", src));
    assert_eq!(out.stmts, Some(3));
    assert!(out.errors.is_empty(), "{:?}", out.errors);

    let src = "\
def f(x):
    if x = 1:
        y = 2 +
    z = 3 +
    while x
        pass
    return z
if a: b = 1
for i in x:
    pass
else:
    pass
if c:
d = 1
    e = 2
equation u:
    solve
";
    let out = harness::run(&seed("block_errors", src));
    assert!(out.errors.len() >= 8, "{:?}", out.errors);
    assert!(out.stmts.is_some());
}

#[test]
fn every_seed_passes_the_fuzz_check() {
    let mut names: Vec<_> = std::fs::read_dir(corpus_dir())
        .expect("corpus dir")
        .map(|e| e.expect("dir entry").path())
        .collect();
    names.sort();
    assert!(names.len() >= 14, "{names:?}");
    for path in names {
        let data = std::fs::read(&path).expect("seed file");
        let _ = harness::run(&data);
    }
}

#[test]
fn type_annotations_parse_and_recover() {
    let src = "t: s\nv: Vector[3] m/s\nA: PositiveDefinite Matrix[N, N]\nK: Symmetric Diagonal Matrix[3, 3] N/m\nx: Scalar\nM: Tensor[B, T, D,] kg\nbad1: Matrix[N+1]\nbad2: Matrix[3,3] PositiveDefinite m s\nbad3: 5 m\n";
    let out = harness::run(&seed("type_annotations", src));
    assert_eq!(out.stmts, Some(6));
    assert_eq!(out.errors.len(), 3, "{:?}", out.errors);
}

#[test]
fn as_display_parses_and_recovers() {
    let src = "v = 25 m/s\nprint(v as km/h, v as kg*m^2/s^2)\nv as km/h\nx = a + b as km\nbad = v as\nbad2 = v as km as m\nbad3 = v as km + 1\nbad4 = v as m / s\nbad5 = 3 as 4\n";
    let out = harness::run(&seed("as_display", src));
    assert_eq!(out.stmts, Some(4));
    assert_eq!(out.errors.len(), 5, "{:?}", out.errors);
}
