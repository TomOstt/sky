//! Golden `.sir` files.
//!
//! * `tests/sir/*.sir` must parse, verify, and print back exactly as they
//!   are on disk (the files are stored in printed form, so the text format
//!   cannot drift unnoticed).
//! * `tests/sir/invalid/*.sir` must parse but fail verification; the
//!   verifier's messages are compared with the `.errors` file next to each.
//!
//! Bless with `SKY_BLESS=1 cargo test -p sky-ir --test golden` and review
//! the diff.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sky_ir::{parse, print, verify};

fn files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "sir"))
        .collect();
    v.sort();
    v
}

fn bless() -> bool {
    std::env::var_os("SKY_BLESS").is_some()
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sir")
}

#[test]
fn valid_files_parse_verify_and_reprint_identically() {
    let cases = files(&dir());
    assert!(cases.len() >= 4, "{cases:?}");
    let mut problems = Vec::new();
    for path in cases {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read case");
        let m = match parse(&src) {
            Ok(m) => m,
            Err(e) => {
                problems.push(format!("{name}: {e}"));
                continue;
            }
        };
        if let Err(errors) = verify(&m) {
            let list: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            problems.push(format!("{name}: does not verify:\n  {}", list.join("\n  ")));
            continue;
        }
        let printed = print(&m);
        // Printing what was printed is a no-op, whatever the file says.
        assert_eq!(print(&parse(&printed).unwrap()), printed, "{name}");
        if bless() {
            // Keep the leading comment, which the printer drops.
            let comment: String = src
                .lines()
                .take_while(|l| l.starts_with("//"))
                .map(|l| l.to_owned() + "\n")
                .collect();
            std::fs::write(&path, format!("{comment}{printed}")).expect("bless");
        } else if strip_comments(&src) != printed {
            problems.push(format!(
                "{name}: differs from its reprint (bless to update)\n--- printed\n{printed}"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

fn strip_comments(src: &str) -> String {
    src.lines()
        .skip_while(|l| l.starts_with("//"))
        .map(|l| l.to_owned() + "\n")
        .collect()
}

#[test]
fn invalid_files_parse_but_fail_verification_with_the_expected_errors() {
    let dir = dir().join("invalid");
    let cases = files(&dir);
    assert!(cases.len() >= 2, "{cases:?}");
    let mut problems = Vec::new();
    for path in cases {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read case");
        let m = parse(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let Err(errors) = verify(&m) else {
            problems.push(format!("{name}: expected verification errors"));
            continue;
        };
        let mut out = String::new();
        for e in &errors {
            writeln!(out, "{} @ {}..{}", e, e.span.start, e.span.end).unwrap();
        }
        let expected_path = path.with_extension("errors");
        if bless() {
            std::fs::write(&expected_path, &out).expect("bless");
            continue;
        }
        match std::fs::read_to_string(&expected_path) {
            Ok(expected) if expected == out => {}
            Ok(expected) => problems.push(format!(
                "{name}: errors differ\n--- expected\n{expected}\n--- actual\n{out}"
            )),
            Err(_) => problems.push(format!("{name}: missing {}", expected_path.display())),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
