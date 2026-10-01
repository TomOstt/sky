//! What `--engine=sir` and `--engine=native` say about a program they do not
//! take: what is missing, in the user's words, a way around it, and that the
//! tree-walker runs it (never that nothing can).
//!
//! Scripts are written to a scratch directory that has a `.git`, so project
//! discovery stops there and never looks at the real checkout.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("refuse-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

fn run_engine(name: &str, src: &str, engine: &str) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("run")
        .arg(format!("--engine={engine}"))
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// Every engine but the tree-walker refuses `src`; the tree-walker runs
/// it. Returns the SIR and native messages.
fn refused(name: &str, src: &str) -> [String; 2] {
    let ast = run_engine(&format!("{name}_ast"), src, "ast");
    assert_eq!(ast.status.code(), Some(0), "{}", err(&ast));
    ["sir", "native"].map(|engine| {
        let o = run_engine(&format!("{name}_{engine}"), src, engine);
        assert_eq!(o.status.code(), Some(1));
        let e = err(&o);
        assert!(e.contains("`--engine=ast` runs this one"), "{e}");
        assert!(!e.contains("cannot execute this either"), "{e}");
        assert!(!e.contains("stand-in"), "{e}");
        e
    })
}

#[test]
fn a_loop_over_an_unannotated_parameter_says_what_to_annotate() {
    // A loop over an array compiles; over a parameter of unknown type it
    // cannot, and the refusal says how to fix that.
    let src = "def f(w):\n    for x in w:\n        print(x)\nf([1 m, 2 m, 3 m])\n";
    for e in refused("array_loop", src) {
        assert!(
            e.contains("iteration over a value whose type is not known"),
            "{e}"
        );
        assert!(e.contains("`w: Vector[N]`"), "{e}");
    }
}

#[test]
fn a_slice_with_a_computed_bound_names_the_variable() {
    let src = "v = [1 m, 2 m, 3 m, 4 m]\nfor i in range(2):\n    d = v[i : i + 2] - v[i + 1 : i + 3]\n    print(d[0])\n";
    for e in refused("slice", src) {
        assert!(
            e.contains("`d` is an array whose size is known only while the program runs"),
            "{e}"
        );
        assert!(e.contains("read its entries one by one"), "{e}");
    }
}
