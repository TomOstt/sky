//! Every `examples/*.sky` runs cleanly and prints exactly its `.out` file on
//! the tree-walker and under `--engine=auto`, and on SIR too unless SIR
//! declines the program with its reason (equations and `±` run on the tree
//! walker), so the examples in the README cannot drift from the language.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

#[test]
fn examples_print_their_expected_output_on_both_engines() {
    let mut seen = 0;
    for entry in fs::read_dir(examples_dir()).expect("read examples/") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "sky") {
            continue;
        }
        seen += 1;
        let expected = fs::read_to_string(path.with_extension("out"))
            .unwrap_or_else(|e| panic!("{}: missing .out file: {e}", path.display()));
        for engine in ["ast", "auto", "sir"] {
            let out = Command::new(env!("CARGO_BIN_EXE_sky"))
                .arg("run")
                .arg(format!("--engine={engine}"))
                .arg(&path)
                .output()
                .expect("run sky");
            let stderr = String::from_utf8_lossy(&out.stderr);
            if engine == "sir" && stderr.contains(": error: cannot run on SIR: ") {
                assert_eq!(out.status.code(), Some(1), "{stderr}");
                assert!(out.stdout.is_empty(), "a declined program prints nothing");
                continue;
            }
            assert!(
                out.status.success() && stderr.is_empty(),
                "{} on {engine}: exit {:?}\n{stderr}",
                path.display(),
                out.status.code()
            );
            assert_eq!(
                String::from_utf8_lossy(&out.stdout),
                expected,
                "{} on {engine}",
                path.display()
            );
        }
    }
    assert!(seen > 0, "no examples found");
}
