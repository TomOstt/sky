//! Running programs and finding tools: subprocesses with a timeout, compiler
//! detection and versions, and the machine description.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// What one process run produced.
pub struct RunOut {
    /// Wall time from spawn to exit, in nanoseconds.
    pub wall_ns: u64,
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub timed_out: bool,
}

/// Runs `cmd` to completion, killing it after `timeout`. The wall time covers
/// process start-up as well as the program: that is what a user waits for,
/// and it is the same for every implementation measured.
pub fn run_timed(cmd: &mut Command, timeout: Duration) -> Result<RunOut, String> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let start = Instant::now();
    let child = cmd.spawn().map_err(|e| format!("cannot start: {e}"))?;
    let pid = child.id();
    let (done, waiting) = mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if waiting.recv_timeout(timeout).is_err() {
            // Only the watchdog kills, and only when the run outlived the
            // timeout (a dropped sender means it finished).
            let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
            true
        } else {
            false
        }
    });
    let output = child.wait_with_output();
    let wall_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let _ = done.send(());
    let killed = watchdog.join().unwrap_or(false);
    let output = output.map_err(|e| format!("cannot wait: {e}"))?;
    Ok(RunOut {
        wall_ns,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        success: output.status.success(),
        timed_out: killed && !output.status.success(),
    })
}

/// The trimmed stdout of a quick command, `None` if it is missing or fails.
pub fn capture(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// The first line of a tool's `--version`-style output (stdout, else stderr).
pub fn first_line(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&out.stderr).into_owned();
    }
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
}

/// A tool found on this machine.
#[derive(Clone, Debug)]
pub struct Tool {
    pub program: String,
    pub version: String,
    /// Where to run it from (rustup reads the toolchain pin there).
    pub cwd: Option<PathBuf>,
}

fn find(candidates: &[&str], version_args: &[&str]) -> Option<Tool> {
    candidates.iter().find_map(|c| {
        first_line(c, version_args).map(|version| Tool {
            program: (*c).to_string(),
            version,
            cwd: None,
        })
    })
}

/// The C compiler: `clang`, else whatever `cc` is.
pub fn find_c() -> Option<Tool> {
    find(&["clang", "cc", "gcc"], &["--version"])
}

/// `rustc`, run from `root` so that rustup picks the repository's pinned
/// toolchain: `$RUSTC`, else `~/.cargo/bin/rustc` (rustup's proxy), else
/// whichever `rustc` is on `PATH`.
pub fn find_rust(root: &Path) -> Option<Tool> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(e) = std::env::var("RUSTC") {
        candidates.push(e);
    }
    let home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| Path::new(&h).join(".cargo"))
        });
    if let Some(h) = home {
        let proxy = h.join("bin").join("rustc");
        if proxy.is_file() {
            candidates.push(proxy.to_string_lossy().into_owned());
        }
    }
    candidates.push("rustc".to_string());
    candidates.iter().find_map(|c| {
        let out = Command::new(c)
            .arg("--version")
            .current_dir(root)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Some(Tool {
            program: c.clone(),
            version,
            cwd: Some(root.to_path_buf()),
        })
    })
}

/// `python3` (or `python`), and its NumPy version if it can import it.
pub fn find_python() -> (Option<Tool>, Option<String>) {
    let py = find(&["python3", "python"], &["--version"]);
    let numpy = py.as_ref().and_then(|p| {
        capture(
            &p.program,
            &["-c", "import numpy; print(numpy.__version__)"],
        )
    });
    (py, numpy)
}

/// Compiles a C program. `fast` adds `-ffast-math`; `lib` links the platform
/// BLAS/LAPACK (Accelerate on macOS, OpenBLAS elsewhere).
pub fn compile_c(cc: &Tool, src: &Path, out: &Path, fast: bool, lib: bool) -> Result<(), String> {
    let mut cmd = Command::new(&cc.program);
    cmd.args(["-O3", "-march=native"]);
    if fast {
        cmd.arg("-ffast-math");
    }
    cmd.arg(src).arg("-o").arg(out);
    if lib {
        if cfg!(target_os = "macos") {
            cmd.args(["-framework", "Accelerate", "-Wno-deprecated-declarations"]);
        } else {
            cmd.arg("-lopenblas");
        }
    }
    cmd.arg("-lm");
    run_compiler(cmd)
}

/// Compiles a standalone Rust program the way the task states.
pub fn compile_rust(rustc: &Tool, src: &Path, out: &Path) -> Result<(), String> {
    let mut cmd = Command::new(&rustc.program);
    cmd.args([
        "-C",
        "opt-level=3",
        "-C",
        "target-cpu=native",
        "--edition",
        "2021",
    ])
    .arg(src)
    .arg("-o")
    .arg(out);
    if let Some(cwd) = &rustc.cwd {
        cmd.current_dir(cwd);
    }
    run_compiler(cmd)
}

fn run_compiler(mut cmd: Command) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("cannot start: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let line = err
        .lines()
        .find(|l| l.contains("error"))
        .or_else(|| err.lines().next())
        .unwrap_or("compilation failed");
    Err(line.trim().chars().take(200).collect())
}

/// Locates the `sky` binary: `explicit`, else next to this executable (the
/// same `target/<profile>` directory).
pub fn find_sky(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return p.is_file().then(|| p.to_path_buf());
    }
    let exe = std::env::current_exe().ok()?;
    let sibling = exe.parent()?.join("sky");
    sibling.is_file().then_some(sibling)
}

/// Builds `sky` in release mode (a no-op when up to date) so the run measures
/// the current source. Returns the binary, or why not.
pub fn build_sky(root: &Path) -> Result<PathBuf, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(&cargo)
        .args(["build", "--release", "--locked", "-p", "sky-cli"])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("cannot run {cargo}: {e}"))?;
    if !status.success() {
        return Err("`cargo build --release -p sky-cli` failed".to_string());
    }
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("target"));
    let bin = target.join("release").join("sky");
    bin.is_file()
        .then_some(bin)
        .ok_or_else(|| "built sky-cli but found no target/release/sky".to_string())
}

/// The machine, as far as the OS says: `sysctl` on macOS, `/proc` on Linux.
#[derive(Clone, Debug, Default)]
pub struct Machine {
    pub cpu: String,
    pub cores: String,
    pub memory: String,
    pub os: String,
    pub arch: String,
}

pub fn machine() -> Machine {
    let arch = std::env::consts::ARCH.to_string();
    if cfg!(target_os = "macos") {
        let sysctl = |k: &str| capture("sysctl", &["-n", k]);
        let cpu = sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown CPU".to_string());
        let logical = sysctl("hw.logicalcpu").or_else(|| sysctl("hw.ncpu"));
        let physical = sysctl("hw.physicalcpu");
        let perf = sysctl("hw.perflevel0.physicalcpu");
        let eff = sysctl("hw.perflevel1.physicalcpu");
        let mut cores = match (&physical, &logical) {
            (Some(p), Some(l)) => format!("{p} physical, {l} logical"),
            (Some(n), None) | (None, Some(n)) => n.clone(),
            _ => "unknown".to_string(),
        };
        if let (Some(p), Some(e)) = (perf, eff) {
            cores.push_str(&format!(" ({p} performance + {e} efficiency)"));
        }
        let memory = sysctl("hw.memsize")
            .and_then(|m| m.parse::<u64>().ok())
            .map_or_else(|| "unknown".to_string(), gib);
        let version = capture("sw_vers", &["-productVersion"]).unwrap_or_default();
        let kernel = capture("uname", &["-sr"]).unwrap_or_default();
        return Machine {
            cpu,
            cores,
            memory,
            os: format!("macOS {version} ({kernel})"),
            arch,
        };
    }
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu = cpuinfo
        .lines()
        .find(|l| {
            l.starts_with("model name") || l.starts_with("Model") || l.starts_with("Hardware")
        })
        .and_then(|l| l.split(':').nth(1))
        .map_or_else(|| "unknown CPU".to_string(), |s| s.trim().to_string());
    let logical = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let memory = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<u64>().ok())
        .map_or_else(|| "unknown".to_string(), |kb| gib(kb * 1024));
    let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    let pretty = os_release
        .lines()
        .find_map(|l| l.strip_prefix("PRETTY_NAME="))
        .map_or_else(
            || std::env::consts::OS.to_string(),
            |s| s.trim_matches('"').to_string(),
        );
    let kernel = capture("uname", &["-r"]).unwrap_or_default();
    Machine {
        cpu,
        cores: if logical == 0 {
            "unknown".to_string()
        } else {
            format!("{logical} logical")
        },
        memory,
        os: format!("{pretty} (kernel {kernel})"),
        arch,
    }
}

/// The 1-minute load average, `None` where the OS does not say.
pub fn load_average() -> Option<f64> {
    if cfg!(target_os = "macos") {
        // `{ 1.23 1.10 0.95 }`
        capture("sysctl", &["-n", "vm.loadavg"])?
            .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    } else {
        std::fs::read_to_string("/proc/loadavg")
            .ok()?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.0} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_process_is_timed_and_its_output_captured() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello");
        let out = run_timed(&mut cmd, Duration::from_secs(10)).expect("echo runs");
        assert!(out.success && !out.timed_out);
        assert_eq!(out.stdout.trim(), "hello");
        assert!(out.wall_ns > 0);
    }

    #[test]
    fn a_runaway_process_is_killed() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let start = Instant::now();
        let out = run_timed(&mut cmd, Duration::from_millis(300)).expect("sleep starts");
        assert!(out.timed_out && !out.success);
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn a_missing_program_is_an_error() {
        let mut cmd = Command::new("definitely-not-a-program-sky");
        assert!(run_timed(&mut cmd, Duration::from_secs(1)).is_err());
        assert!(first_line("definitely-not-a-program-sky", &["--version"]).is_none());
    }

    #[test]
    fn the_load_average_is_a_number_when_known() {
        if let Some(l) = load_average() {
            assert!(l >= 0.0);
        }
    }

    #[test]
    fn the_machine_has_a_cpu_and_an_os() {
        let m = machine();
        assert!(!m.cpu.is_empty() && !m.os.is_empty() && !m.arch.is_empty());
    }
}
