//! `sky check --emit=clif` and `sky run --engine=native`, end to end.
//!
//! The Cranelift IR depends on the machine (its calling convention names
//! it), so there are no goldens: these tests look for what is the same
//! everywhere.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("clif-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

fn sky(name: &str, args: &[&str], src: &str) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const PROGRAM: &str = "def area(w: m, h: m) -> m^2:\n    return w * h\nprint(area(3 m, 4 m))\n";

#[test]
fn emit_clif_prints_every_function_on_stdout() {
    let o = sky("emit", &["check", "--emit=clif"], PROGRAM);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    assert!(text.contains("sky_f0__main_"), "{text}");
    assert!(text.contains("sky_f1_area"), "{text}");
    // The product of two typed scalars is an inline multiplication with a
    // finiteness check, not a call.
    assert!(text.contains("fmul"), "{text}");
    assert!(text.contains("fabs"), "{text}");
    // Only the CLIF is on stdout; the `ok` line is on stderr.
    assert!(err(&o).contains(": ok"), "{}", err(&o));
    assert!(!text.contains("ok\n"), "{text}");
}

#[test]
fn emit_clif_and_emit_ir_together_print_the_ir_first() {
    let o = sky("both", &["check", "--emit-ir", "--emit=clif"], PROGRAM);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    let sir = text.find("sir edition").expect("the SIR text");
    let clif = text.find("function %").expect("the CLIF text");
    assert!(sir < clif, "{text}");
}

#[test]
fn a_program_without_ir_has_no_clif_and_a_note_says_why() {
    let o = sky(
        "noir",
        &["check", "--emit=clif"],
        "for x in [1, 2]:\n    print(x)\n",
    );
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("no IR:"), "{}", err(&o));
}

#[test]
fn only_clif_is_an_emit_target_and_it_may_not_repeat() {
    let o = sky("asm", &["check", "--emit=asm"], PROGRAM);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        err(&o).contains("unknown --emit target `asm`"),
        "{}",
        err(&o)
    );
    let o = sky("twice", &["check", "--emit=clif", "--emit=clif"], PROGRAM);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn native_runs_the_program_and_auto_prefers_it() {
    let native = sky("run_native", &["run", "--engine=native"], PROGRAM);
    assert_eq!(native.status.code(), Some(0), "{}", err(&native));
    assert_eq!(out(&native), "12 m^2\n");
    let auto = sky("run_auto", &["run", "--engine=auto"], PROGRAM);
    assert_eq!(out(&auto), out(&native));
    let ast = sky("run_ast", &["run", "--engine=ast"], PROGRAM);
    assert_eq!(out(&ast), out(&native));
}

#[test]
fn a_native_run_fails_like_the_others() {
    let src = "print(1)\nz = 2 - 2\nprint(4 / z)\n";
    let native = sky("fail", &["run", "--engine=native"], src);
    let sir = sky("fail", &["run", "--engine=sir"], src);
    assert_eq!(native.status.code(), Some(1));
    assert_eq!(out(&native), "1\n");
    assert_eq!(out(&native), out(&sir));
    assert_eq!(err(&native), err(&sir));
}
