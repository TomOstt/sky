//! Relaxed numerics on the native engine: `a*b + c` is contracted into one
//! fused multiply-add, and everything else that native code promises holds:
//! the errors are the reproducible ones, at the same place, after the same
//! output. (The kernels behind arrays are tested in `sky-linalg` and
//! `sky-cli`'s `relaxed` tests; this file only runs scalar programs, so it
//! never touches the process-wide numerics mode.)

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::Module;
use sky_jit::{compile, compile_with, emit_clif, emit_clif_with, fused_sums, Numerics};

/// What a run did: stdout, and the failure as the whole rendered diagnostic.
#[derive(Debug, PartialEq)]
struct Run {
    stdout: String,
    failure: Option<String>,
}

fn render(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("test.sky", src))
}

/// The program as lowered, before any pass.
fn lower_plain(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("does not lower:\n{src}\n{e}"))
}

fn lower(src: &str) -> Module {
    let mut ir =
        sky_lower::lower_source(src).unwrap_or_else(|e| panic!("does not lower:\n{src}\n{e}"));
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut ir)
        .expect("the pipeline runs");
    ir
}

fn native(m: &Module, src: &str, numerics: Numerics) -> Run {
    sky_ast::with_deep_stack(|| {
        let code = compile_with(m, numerics).unwrap_or_else(|e| panic!("{e}"));
        let mut out = Vec::new();
        let r = code.run(&mut out);
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| match e {
                sky_jit::RunFailure::Error(e) => render(&e.to_diagnostic(), src),
                other => panic!("{other:?}"),
            }),
        }
    })
}

fn sir(m: &Module, src: &str) -> Run {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        Run {
            stdout: String::from_utf8(out).unwrap(),
            failure: r.err().map(|e| render(&e.to_diagnostic(), src)),
        }
    })
}

fn host_fuses() -> bool {
    let m = lower("x = 1\nfor i in range(2):\n    x = x * i + 1\nprint(x)\n");
    emit_clif_with(&m, Numerics::Relaxed)
        .map(|t| t.contains("fma"))
        .unwrap_or(false)
}

/// A loop-carried value keeps the optimizer from folding the arithmetic
/// away (constants are folded with the interpreter's unfused operations).
// Tested on the lowered SIR, before the optimizer: value numbering merges the
// two equal products `e * (i + 1)` into one value with two uses, which is
// then fused into neither sum.
const FUSED_DIFFERS: &str = "\
e = 1 / 1073741824
t = 0
for i in range(3):
    a = 1 + e * (i + 1)
    b = 1 - e * (i + 1)
    d = a * b - 1
    t = t + d
print(t)
";

#[test]
fn reproducible_native_code_never_fuses() {
    let m = lower(FUSED_DIFFERS);
    assert!(!emit_clif(&m).unwrap().contains("fma"));
    assert!(!emit_clif_with(&m, Numerics::Reproducible)
        .unwrap()
        .contains("fma"));
    assert_eq!(
        native(&m, FUSED_DIFFERS, Numerics::Reproducible),
        sir(&m, FUSED_DIFFERS)
    );
    // `compile` is `compile_with(Reproducible)`.
    let out = sky_ast::with_deep_stack(|| {
        let code = compile(&m).unwrap();
        let mut out = Vec::new();
        code.run(&mut out).unwrap();
        out
    });
    assert_eq!(String::from_utf8(out).unwrap(), sir(&m, "").stdout);
}

#[test]
fn relaxed_native_code_fuses_a_product_into_its_sum() {
    if !host_fuses() {
        return;
    }
    let m = lower_plain(FUSED_DIFFERS);
    let clif = emit_clif_with(&m, Numerics::Relaxed).unwrap();
    assert!(clif.contains("fma"), "{clif}");
    // `e * k + 1`, `1 - e * k` and `a * b - 1`.
    assert_eq!(fused_sums(&m).len(), 3);
    let want = sir(&m, FUSED_DIFFERS);
    let got = native(&m, FUSED_DIFFERS, Numerics::Relaxed);
    assert!(got.failure.is_none());
    // (1 + x)(1 - x) - 1 is -x^2: 0 unfused (1 - x^2 rounds to 1), -x^2 fused.
    assert_ne!(got.stdout, want.stdout, "fusing must show in this program");
    let value = |r: &Run| r.stdout.trim().parse::<f64>().unwrap();
    // Sum of -(k * 2^-30)^2 for k = 1..3 is -14 * 2^-60.
    let exact = -14.0 * 2f64.powi(-60);
    assert!((value(&got) - exact).abs() <= 1e-3 * exact.abs(), "{got:?}");
}

#[test]
fn a_fused_program_agrees_with_the_reproducible_one_to_rounding_error() {
    let src = "\
x = 0.7
v = 0.3
i = 0
while i < 2000:
    a = -3.1 * x - 0.05 * v
    v = v + a * 0.001
    x = x + v * 0.001
    i += 1
print(x)
print(v)
";
    let m = lower(src);
    let want = sir(&m, src);
    let got = native(&m, src, Numerics::Relaxed);
    assert!(got.failure.is_none() && want.failure.is_none());
    for (g, w) in got.stdout.lines().zip(want.stdout.lines()) {
        let (g, w): (f64, f64) = (g.parse().unwrap(), w.parse().unwrap());
        assert!((g - w).abs() <= 1e-12 * w.abs().max(1e-3), "{g} vs {w}");
    }
}

/// An error is the reproducible one, at the same place, after the same
/// output, whether it is in the product or in the sum.
#[test]
fn errors_are_the_same_kind_at_the_same_place() {
    let programs = [
        // The product overflows.
        "x = 1e200\nprint(1)\nfor i in range(2):\n    y = x * (x + i) + i\n    print(y)\n",
        // The sum overflows.
        "x = 1e308\nprint(1)\nfor i in range(2):\n    y = x * (i + 1) + x\n    print(y)\n",
        // Division by zero after a fusable pair.
        "x = 3\nfor i in range(2):\n    y = x * i + 1\n    print(y)\n    print(1 / (i - 1))\n",
        // Dimensions: a product of lengths is an area, added to an area.
        "x = 1e200 m\nprint(1)\nfor i in range(2):\n    y = x * (x + i * 1 m) + 1 m^2\n    print(y)\n",
    ];
    for src in programs {
        let ir = lower(src);
        assert!(!fused_sums(&ir).is_empty(), "nothing to fuse in:\n{src}");
        let want = sir(&ir, src);
        assert!(want.failure.is_some(), "the program should fail:\n{src}");
        let got = native(&ir, src, Numerics::Relaxed);
        assert_eq!(got, want, "relaxed native code differs on:\n{src}");
        assert_eq!(native(&ir, src, Numerics::Reproducible), want, "{src}");
    }
}

/// The one documented relaxation: a product that overflows while the fused
/// sum does not is accepted.
#[test]
fn a_product_rescued_by_its_addend_is_the_documented_relaxation() {
    if !host_fuses() {
        return;
    }
    let src = "\
x = 2e154
z = 1.7e308
for i in range(1):
    y = x * (1e154 * (i + 1)) - z
    print(y)
";
    let m = lower(src);
    let strict = native(&m, src, Numerics::Reproducible);
    assert!(
        strict
            .failure
            .as_deref()
            .is_some_and(|f| f.contains("overflow")),
        "{strict:?}"
    );
    assert_eq!(strict, sir(&m, src));
    let relaxed = native(&m, src, Numerics::Relaxed);
    assert!(relaxed.failure.is_none(), "{relaxed:?}");
    let y: f64 = relaxed.stdout.trim().parse().unwrap();
    assert!(y > 2.9e307 && y < 3.1e307, "{y}");
}

/// Relaxed numerics and SLP together: a sum that fuses a deferred product is
/// never paired into vector lanes (SLP would read the product before it is
/// emitted: "used before it is defined"). With SLP forced on, the program
/// still compiles, on the lowered and on the optimized SIR.
#[test]
fn fused_sums_and_vector_lanes_do_not_collide() {
    for m in [lower_plain(FUSED_DIFFERS), lower(FUSED_DIFFERS)] {
        for slp in [sky_jit::Slp::Off, sky_jit::Slp::Always] {
            sky_jit::emit_clif_with_slp(&m, Numerics::Relaxed, slp)
                .unwrap_or_else(|e| panic!("{slp:?}: {e}"));
            // Compiled and dropped on the deep-stack thread (the code is not
            // `Send`).
            sky_ast::with_deep_stack(|| {
                let compiled = sky_jit::compile_with_slp(&m, Numerics::Relaxed, slp);
                assert!(compiled.is_ok(), "{slp:?}: {:?}", compiled.err());
            });
        }
    }
}
