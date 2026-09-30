//! Every `examples/*.sky` runs cleanly on both engines and prints exactly its
//! `.out` file, so the examples in the README cannot drift from the language.

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
        for engine in ["ast", "sir"] {
            let out = Command::new(env!("CARGO_BIN_EXE_sky"))
                .arg("run")
                .arg(format!("--engine={engine}"))
                .arg(&path)
                .output()
                .expect("run sky");
            let stderr = String::from_utf8_lossy(&out.stderr);
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
