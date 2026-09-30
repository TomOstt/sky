//! Dead loop-carried values (`dce`): a name a loop assigns before it reads
//! it, and nothing reads afterwards, is not carried. The pipeline must drop
//! exactly those, keep every slot something reads (the loop's own state, a
//! name used after the loop, a value only the next iteration reads), and
//! change no output or error.

use sky_ir::{Module, OpKind, PassManager};
use sky_opt::default_passes;

fn run(m: &Module, src: &str) -> (String, Option<String>) {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        (
            String::from_utf8(out).expect("UTF-8 output"),
            r.err().map(|e| {
                e.to_diagnostic()
                    .render_to_string(&sky_diag::SourceFile::new("test.sky", src))
            }),
        )
    })
}

/// Lowers and optimizes `src`, asserts the runs agree, and returns the
/// optimized module.
fn optimized(src: &str) -> Module {
    let mut m = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let before = run(&m, src);
    default_passes(PassManager::for_tests())
        .run(&mut m)
        .expect("valid IR");
    assert_eq!(run(&m, src), before, "the pipeline changed:\n{src}");
    m
}

/// How many values each reachable `while` of `<main>` carries, outermost
/// first.
fn carried(m: &Module) -> Vec<usize> {
    fn walk(f: &sky_ir::Function, r: sky_ir::RegionId, out: &mut Vec<usize>) {
        for &b in &f.region(r).blocks {
            for &o in &f.block(b).ops {
                let op = f.op(o);
                if matches!(op.kind, OpKind::While) {
                    out.push(op.operands.len());
                }
                for &inner in &op.regions {
                    walk(f, inner, out);
                }
            }
        }
    }
    let f = m.function(Module::MAIN).expect("main");
    let mut out = Vec::new();
    walk(f, f.body, &mut out);
    out
}

#[test]
fn temporaries_a_loop_defines_before_reading_are_not_carried() {
    let src = "total = 0\n\
        for py in range(3):\n    ci = py * 0.5\n    for px in range(3):\n        \
        zr = 0.0\n        zi = 0.0\n        i = 0\n        while i < 10:\n            \
        zr2 = zr * zr\n            zi2 = zi * zi\n            \
        if zr2 + zi2 > 4.0:\n                break\n            \
        zi = 2.0 * zr * zi + ci\n            zr = zr2 - zi2 + px\n            i = i + 1\n        \
        total = total + i\nprint(total)\n";
    let m = optimized(src);
    let text = sky_ir::print(&m);
    assert!(!text.contains("unknown"), "boxed values remain:\n{text}");
    // py and total; px and total; i, zi, zr and the break flag.
    assert_eq!(carried(&m), vec![2, 2, 4], "{text}");
}

#[test]
fn a_name_read_after_the_loop_stays_carried() {
    // `last` is assigned in the loop and printed after it: live. `scratch`
    // is not.
    let src = "n = 0\nlast = 0\nwhile n < 4:\n    scratch = n * 2\n    last = scratch + 1\n    n = n + 1\n\
        print(last, n)\n";
    let m = optimized(src);
    assert_eq!(carried(&m), vec![2], "{}", sky_ir::print(&m));
}

#[test]
fn a_value_only_the_next_iteration_reads_stays_carried() {
    // `prev` is read in the body before it is assigned: it carries state.
    let src = "prev = 1.0\ncur = 1.0\nk = 0\nwhile k < 10:\n    nxt = prev + cur\n    \
        prev = cur\n    cur = nxt\n    k = k + 1\nprint(cur)\n";
    let m = optimized(src);
    assert_eq!(carried(&m), vec![3], "{}", sky_ir::print(&m));
}

#[test]
fn a_name_the_checker_lets_be_read_later_is_never_dropped() {
    // Assigned before the loop, so it may be read after it, even when the
    // loop never runs.
    let src = "t = 5\nn = 0\nwhile n < 0:\n    t = 1\n    n = n + 1\nprint(t)\n";
    let m = optimized(src);
    assert_eq!(carried(&m), vec![2], "{}", sky_ir::print(&m));
}

#[test]
fn a_failing_operation_feeding_a_dead_slot_still_fails() {
    // The slot `w` is dead (nothing reads it), but `1e300 * 1e300 * n`
    // overflows in the loop: only the carried value goes, not the failure.
    let src = "n = 1\nwhile n < 3:\n    w = 1e300 * 1e300 * n\n    n = n + 1\nprint(n)\n";
    let m = optimized(src);
    let (out, err) = run(&m, src);
    assert!(out.is_empty());
    assert!(err.is_some());
}
