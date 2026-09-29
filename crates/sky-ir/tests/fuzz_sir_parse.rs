//! Runs the `sir_parse` fuzz check with stable Rust: on its seed corpus, on
//! deterministic mutations of every seed and golden file, and on the
//! adversarial inputs a fuzzer finds first. The check itself is
//! `fuzz/harness/sir_parse.rs`, the same file the fuzz target includes.

#[path = "../../../fuzz/harness/sir_parse.rs"]
mod harness;

mod common;

use std::path::{Path, PathBuf};

fn read_dir(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "sir"))
        .collect();
    v.sort();
    v
}

fn seeds() -> Vec<(String, Vec<u8>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut dirs = vec![
        root.join("../../fuzz/corpus/sir_parse"),
        root.join("tests/sir"),
        root.join("tests/sir/invalid"),
    ];
    dirs.retain(|d| d.exists());
    let mut out = Vec::new();
    for d in dirs {
        for p in read_dir(&d) {
            out.push((
                p.display().to_string(),
                std::fs::read(&p).expect("seed file"),
            ));
        }
    }
    // A module built in code, so the seeds cover every operation and type.
    out.push((
        "all_ops".into(),
        sky_ir::print(&common::all_ops()).into_bytes(),
    ));
    out
}

#[test]
fn every_seed_passes_the_fuzz_check() {
    let seeds = seeds();
    assert!(seeds.len() >= 10, "{}", seeds.len());
    for (name, data) in &seeds {
        let out = harness::check(data);
        if !name.contains("invalid") {
            assert_eq!(out.error, None, "{name}");
        }
    }
    let ok = harness::check(seeds.last().unwrap().1.as_slice());
    assert!(ok.verified);
}

/// xorshift64*: deterministic, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const SNIPPETS: &[&str] = &[
    "%",
    "^",
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ":",
    ";",
    ",",
    "#",
    "@",
    "..",
    "\"",
    "\\",
    "\n",
    "  ",
    "//",
    "num[",
    "if ",
    "while ",
    "return",
    "yield",
    "condition",
    "0x",
    "-",
    "1/0",
    "Θ",
    "⁻",
    "¹",
    "·",
    "^(",
    "é",
    "\u{202e}",
    "\0",
    "f64",
    "exact",
    "{Symmetric}",
    "18446744073709551615",
    "N#0",
    "\"x\"#1",
    "%0",
    "%x",
    "call @f(",
    "solve[cholesky]",
    "expect.bool[and]",
];

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut v = seed.to_vec();
    for _ in 0..1 + rng.below(4) {
        if v.is_empty() {
            v.push(b'%');
        }
        let at = rng.below(v.len());
        match rng.below(6) {
            0 => v[at] = (rng.next() & 0xff) as u8,
            1 => {
                let end = (at + 1 + rng.below(16)).min(v.len());
                v.drain(at..end);
            }
            2 => {
                let s = SNIPPETS[rng.below(SNIPPETS.len())];
                v.splice(at..at, s.bytes());
            }
            3 => {
                let end = (at + 1 + rng.below(64)).min(v.len());
                let piece = v[at..end].to_vec();
                let to = rng.below(v.len());
                v.splice(to..to, piece);
            }
            4 => v.truncate(at),
            _ => {
                let s = SNIPPETS[rng.below(SNIPPETS.len())];
                let end = (at + 1 + rng.below(4)).min(v.len());
                v.splice(at..end, s.bytes());
            }
        }
    }
    v
}

#[test]
fn mutations_of_the_seeds_never_panic() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let seeds = seeds();
    let (mut parsed, mut total) = (0, 0);
    for (_, seed) in &seeds {
        for _ in 0..1500 {
            let input = mutate(&mut rng, seed);
            let out = harness::check(&input);
            total += 1;
            parsed += usize::from(out.error.is_none());
        }
    }
    // Some mutations must survive parsing, or the test explores nothing
    // past the first error.
    assert!(parsed * 50 > total, "{parsed} of {total} parsed");
}

#[test]
fn deep_nesting_is_an_error_not_a_crash() {
    let max = sky_ir::MAX_REGION_DEPTH;
    for levels in [10, max - 2, max - 1, max, 5 * max, 100_000] {
        let mut s = String::from("sir edition 2026\nfn @f(%c: bool) -> () {\n");
        for _ in 0..levels {
            s.push_str("if %c {\n^0:\n");
        }
        for _ in 0..levels {
            s.push_str("}\n");
        }
        s.push_str("}\n");
        let out = harness::check(s.as_bytes());
        if levels >= max {
            assert!(out.error.unwrap().contains("nested too deeply"));
        }
    }
}

#[test]
fn long_and_repetitive_inputs_are_fast() {
    // 50 000 operations, and one function with 20 000 values sharing a name
    // (uniquifying must not be quadratic).
    let mut s = String::from("sir edition 2026\nfn @f(%x: num[L; ()]) -> () {\n");
    for i in 0..50_000 {
        s.push_str(&format!("  %v{i} = neg %x : num[L; ()]\n"));
    }
    s.push_str("  return\n}\n");
    let out = harness::check(s.as_bytes());
    assert_eq!(out.error, None);

    let mut b = sky_ir::FunctionBuilder::new("g", vec![], vec![], sky_diag::Span::point(0));
    for _ in 0..20_000 {
        let v = b.op1(
            sky_ir::OpKind::ConstBool(true),
            vec![],
            sky_ir::Type::Bool,
            sky_diag::Span::point(0),
        );
        b.set_name(v, "x");
    }
    b.op(
        sky_ir::OpKind::Return,
        vec![],
        vec![],
        sky_diag::Span::point(0),
    );
    let m = sky_ir::Module {
        edition: sky_ir::Edition::E2026,
        functions: vec![b.finish()],
    };
    let text = sky_ir::print(&m);
    assert!(text.contains("%x_19999 = "));
    let back = sky_ir::parse(&text).unwrap();
    assert_eq!(back, sky_ir::canonicalize(&m));
}

#[test]
fn hostile_tokens() {
    let cases: &[&str] = &[
        "",
        "\0",
        "sir edition 2026\nfn @f(",
        "sir edition 2026\nfn @\"",
        "sir edition 2026\nfn @f() -> (num[L^(",
        "sir edition 2026\nfn @f() -> (num[L¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹; ()]) {}",
        "sir edition 2026\nfn @f() -> (num[L; (99999999999999999999999)]) {}",
        "sir edition 2026\nfn @f() -> (num[1; (\"\\u{110000}\"#1)]) {}",
        "sir edition 2026\nfn @f() -> () {\n  %a = const.float 0x\n}",
        "sir edition 2026\nfn @f() -> () {\n  %a = const.float ++--..ee\n}",
        "sir edition 2026\nfn @f() -> () {\n  %a = const.str \"\u{202e}\"\n}",
        "sir edition 2026\nfn @f() -> () {\n  %é = const.bool true : bool\n  %θ² = not %é : bool\n}",
        "\u{feff}sir edition 2026\n",
        "sir edition 2026\nfn @f() -> () @ 4294967295..4294967295 {\n}",
        "sir edition 2026\nfn @f() -> () @ 9..1 {\n  return @ 5..2\n}",
    ];
    for c in cases {
        harness::check(c.as_bytes());
    }
}
