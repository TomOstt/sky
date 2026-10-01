//! Derivatives and late-read module names, from the dogfooding logs: a
//! size-generic function differentiates on every engine; a derivative at an
//! uncertain point is refused on the argument, not inside the function; a
//! function passed as a value can be differentiated where it runs; and a
//! module name a body reads cannot change type underneath it.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("derivative-bugs-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// `sky <args> prog.sky` on `src`, in a scratch directory.
fn sky(name: &str, args: &[&str], src: &str) -> Output {
    let dir = scratch(name);
    fs::write(dir.join("prog.sky"), src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg("prog.sky")
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

/// Runs `src` on every engine and checks each prints `expected`.
fn runs_everywhere(name: &str, src: &str, expected: &str) {
    for engine in ["ast", "sir", "native", "auto"] {
        let flag = format!("--engine={engine}");
        let o = sky(&format!("{name}-{engine}"), &["run", &flag], src);
        assert!(o.status.success(), "{engine}: {}", err(&o));
        assert_eq!(out(&o), expected, "{engine}");
    }
}

/// Fails `sky check` with `message` (and so `sky run` too).
fn check_fails(name: &str, src: &str, message: &str) -> String {
    let o = sky(name, &["check"], src);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    let e = err(&o);
    assert!(e.contains(message), "{e}");
    e
}

#[test]
fn a_size_generic_function_differentiates_on_every_engine() {
    // Before: SIR did not verify ("result 0 of the call to `dmodel/dA` is
    // num[L; (N#0)], but it returns num[L; (3)]").
    runs_everywhere(
        "generic",
        "def model(A: m, g: Hz, t: Vector[N] s) -> Vector[N] m:
    return A * exp(-g * t)

def sweep(t: Vector[K] s) -> Vector[K] m*s:
    return derivative(model, g)(2 m, 0.5 Hz, t)

t = [0 s, 1 s, 2 s]
print(derivative(model, A)(2 m, 0.5 Hz, t))
print(sweep(t))
",
        "[1, 0.6065306597126334, 0.36787944117144233]\n\
         [-0 m*s, -1.2130613194252668 m*s, -1.4715177646857693 m*s]\n",
    );
}

const UNCERTAIN_POINT: &str = "def f(A: Scalar, γ: Hz, t: s) -> Scalar:
    return A * exp(-γ * t)
x = 0.5 Hz ± 0.01 Hz
print(derivative(f, γ)(2.0, x, 1 s))
";

#[test]
fn a_derivative_at_an_uncertain_point_is_refused_on_the_argument() {
    let e = check_fails(
        "uncertain-point",
        UNCERTAIN_POINT,
        "`derivative(f, γ)` needs exact arguments, but `γ` may be an uncertain number",
    );
    assert!(e.contains("prog.sky:4:29"), "{e}");
    assert!(e.contains("pass `nominal(...)`"), "{e}");
}

#[test]
fn run_time_refuses_an_uncertain_argument_the_check_could_not_see() {
    // A parameter is not tracked by the check; run time finds the value.
    let o = sky(
        "uncertain-param",
        &["run"],
        "def f(γ: Hz, t: s) -> Scalar:
    return exp(-γ * t)
def at(y: Hz) -> s:
    return derivative(f, γ)(y, 1 s)
print(at(0.5 Hz ± 0.01 Hz))
",
    );
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("`derivative(f, γ)` needs exact arguments, but `γ` is an uncertain number"),
        "{e}"
    );
    // On the argument, not on `-γ` inside `f`.
    assert!(e.contains("prog.sky:4:29"), "{e}");
}

#[test]
fn a_function_passed_as_a_value_can_be_differentiated() {
    // Before: `sky check` passed and `sky run` failed ("only functions
    // defined with `def` can be called"): every `def` had to lower to SIR.
    let src = "def f(p: Vector[2], t: s) -> Scalar:
    return p[0] * exp(-p[1] * t / 1 s)

def step(h, p: Vector[2], t: s) -> Vector[2]:
    return derivative(h, p)(p, t)

print(step(f, [1.0, 2.0], 1 s))
";
    for engine in ["--engine=ast", "--engine=auto"] {
        let o = sky("fn-value", &["run", engine], src);
        assert_eq!(err(&o), "", "{engine}");
        assert_eq!(
            out(&o),
            "[0.1353352832366127, -0.1353352832366127]\n",
            "{engine}"
        );
    }
    let o = sky("fn-value-sir", &["run", "--engine=sir"], src);
    assert!(
        err(&o).contains("cannot differentiate a function passed as a value"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_name_a_function_reads_cannot_change_type() {
    // Before: `rate()` printed `2 m` from a `-> Hz` function.
    check_fails(
        "retype-def",
        "kk = 1 s^-1
def rate() -> Hz:
    return kk
kk = 2 m
print(rate())
",
        "`kk` cannot become a length: `rate` reads it as a frequency",
    );
}

#[test]
fn a_name_an_equation_reads_cannot_change_type() {
    // Before: the solve used 2 m as a rate, silently.
    check_fails(
        "retype-equation",
        "kk = 1 s^-1
equation decay(t: s):
    n: mol
    n' = -kk * n
kk = 2 m
sol = solve decay from 0 s to 1 s with n = 1 mol
print(sol.n)
",
        "`kk` cannot become a length: equation `decay` reads it as a frequency",
    );
}

#[test]
fn a_name_a_function_reads_cannot_lose_a_tag_the_body_relied_on() {
    // Before: the body's `Q⁻¹ @ b` had become `Qᵀ @ b` (Q orthogonal), so
    // after `Q = 2 * Q` the tree walker printed [2, 2], not [0.5, 0.5],
    // and SIR failed to verify.
    check_fails(
        "retag",
        "Q = identity(2)
b = [1, 1]
def solveq() -> Vector[2]:
    return Q⁻¹ @ b
print(solveq())
Q = 2 * Q
print(solveq())
",
        "`Q` cannot lose its tags: `solveq` reads it as",
    );
}

#[test]
fn a_sweep_rebinding_a_value_of_the_same_type_reads_it_late() {
    // Equations, like functions, read values when they run: the sweep the
    // spec describes.
    runs_everywhere(
        "sweep",
        "kk = 1 s^-1
def rate() -> Hz:
    return kk
for i in range(2):
    kk = (i + 1) * 1 s^-1
    print(rate())
",
        "1 s^-1\n2 s^-1\n",
    );
    let o = sky(
        "sweep-eq",
        &["run"],
        "kk = 1 s^-1
equation decay(t: s):
    n: mol
    n' = -kk * n
kk = 2 s^-1
sol = solve decay from 0 s to 1 s with n = 1 mol
print(sol.n)
",
    );
    assert_eq!(err(&o), "");
    let n: f64 = out(&o).trim().trim_end_matches(" mol").parse().unwrap();
    assert!((n - (-2.0f64).exp()).abs() < 1e-6, "{n}");
}
