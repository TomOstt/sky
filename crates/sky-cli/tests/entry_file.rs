//! The file `sky check` is given is read like an imported module: one open
//! (non-blocking), an fstat of that handle (regular files only), and a read
//! capped at MAX_SOURCE_BYTES.
#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use sky_syntax::MAX_SOURCE_BYTES;

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("entry-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// Runs `sky check <path>`, killing it (and failing the test) if it takes
/// more than 10 seconds, so a blocking open fails instead of hanging CI.
fn sky_check(path: &PathBuf) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("check")
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run sky");
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("sky check {} did not finish within 10 s", path.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn fifo_is_refused_without_blocking() {
    let d = scratch("fifo");
    let fifo = d.join("mod.sky");
    let st = Command::new("mkfifo").arg(&fifo).status().expect("mkfifo");
    assert!(st.success());
    let out = sky_check(&fifo);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "error: refusing to read {}: it is not a regular file\n",
            fifo.display()
        )
    );
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn directory_is_refused() {
    let d = scratch("dir");
    let dir = d.join("pkg.sky");
    fs::create_dir(&dir).unwrap();
    let out = sky_check(&dir);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("is not a regular file"));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn oversized_file_is_refused_with_the_limit() {
    let d = scratch("big");
    let big = d.join("big.sky");
    // Sparse: set_len writes no data, and sky refuses on the fstat size
    // before reading or allocating anything.
    fs::File::create(&big)
        .unwrap()
        .set_len(MAX_SOURCE_BYTES as u64 + 1)
        .unwrap();
    let out = sky_check(&big);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "error: refusing to read {}: it is 16777217 bytes, over the limit of 16777216 bytes (16 MiB)\n",
            big.display()
        )
    );
    assert!(out.stdout.is_empty());
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn small_file_is_checked() {
    let d = scratch("small");
    let f = d.join("ok.sky");
    fs::write(&f, "x = 1\n").unwrap();
    let out = sky_check(&f);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{}: ok\n", f.display())
    );
    fs::remove_dir_all(&d).unwrap();
}
