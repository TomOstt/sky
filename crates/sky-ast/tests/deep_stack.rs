//! `with_deep_stack`: results, borrowing, panic propagation, and depth.

mod common;
use std::panic::{catch_unwind, AssertUnwindSafe};

use sky_ast::{parse_module, with_deep_stack, DEEP_STACK_BYTES, MAX_DEPTH};

#[test]
fn returns_the_closure_value() {
    assert_eq!(DEEP_STACK_BYTES, 8 * 1024 * 1024);
    assert_eq!(with_deep_stack(|| 6 * 7), 42);
    let v: Vec<String> = with_deep_stack(|| vec!["a".to_string(), "b".to_string()]);
    assert_eq!(v, ["a", "b"]);
}

#[test]
fn runs_on_a_named_thread() {
    let name = with_deep_stack(|| std::thread::current().name().map(str::to_string));
    assert_eq!(name.as_deref(), Some("sky-deep-stack"));
}

#[test]
fn borrows_non_static_data() {
    // Built at run time, so it can't be 'static.
    let src = format!("x = {}", "1 + 2");
    let tokens = common::lex(&src);
    let printed = with_deep_stack(|| {
        let (module, errors) = parse_module(&tokens, &src);
        assert!(errors.is_empty(), "{errors:#?}");
        module.to_sexpr(&src)
    });
    assert_eq!(printed, "(= x (+ 1 2))");
    // Still usable after the call.
    assert_eq!(src.len(), 9);
}

#[test]
fn panics_propagate_with_the_original_payload() {
    let err = catch_unwind(|| with_deep_stack(|| -> u8 { panic!("static message") }))
        .expect_err("the panic propagates");
    assert_eq!(err.downcast_ref::<&str>(), Some(&"static message"));

    let n = 7;
    let err = catch_unwind(|| with_deep_stack(|| -> u8 { panic!("formatted {n}") }))
        .expect_err("the panic propagates");
    assert_eq!(
        err.downcast_ref::<String>().map(String::as_str),
        Some("formatted 7")
    );

    // A custom payload type survives too.
    #[derive(Debug, PartialEq)]
    struct Custom(u32);
    let err = catch_unwind(|| with_deep_stack(|| -> u8 { std::panic::panic_any(Custom(3)) }))
        .expect_err("the panic propagates");
    assert_eq!(err.downcast_ref::<Custom>(), Some(&Custom(3)));

    // The caller keeps working afterwards.
    let mut hits = 0;
    let _ = catch_unwind(AssertUnwindSafe(|| with_deep_stack(|| panic!("again"))));
    hits += with_deep_stack(|| 1);
    assert_eq!(hits, 1);
}

fn nested(kind: &str, n: usize) -> String {
    match kind {
        "paren" => format!("{}x{}", "(".repeat(n), ")".repeat(n)),
        "call" => format!("{}x{}", "f(".repeat(n), ")".repeat(n)),
        "power" => format!("x{}", "^x".repeat(n)),
        "not" => format!("{}x", "not ".repeat(n)),
        // A mix of every kind, MAX_DEPTH levels in total.
        "mixed" => {
            let quarter = n / 4;
            format!(
                "{}{}{}{}x{}{}{}",
                "(".repeat(quarter),
                "f(".repeat(quarter),
                "not ".repeat(quarter),
                "-".repeat(n - 3 * quarter - 1),
                "^x",
                ")".repeat(quarter),
                ")".repeat(quarter),
            )
        }
        _ => unreachable!(),
    }
}

#[test]
fn max_depth_parses_inside_with_deep_stack() {
    let limit = MAX_DEPTH as usize;
    for kind in ["paren", "call", "power", "not", "mixed"] {
        let ok = format!("y = {}\n", nested(kind, limit));
        let too_deep = format!("y = {}\n", nested(kind, limit + 1));
        let (ok_errors, deep_errors) = with_deep_stack(|| {
            let (m, ok_errors) = parse_module(&common::lex(&ok), &ok);
            let _ = m.to_sexpr(&ok);
            let (m, deep_errors) = parse_module(&common::lex(&too_deep), &too_deep);
            drop(m);
            (ok_errors, deep_errors)
        });
        assert!(ok_errors.is_empty(), "{kind}: {ok_errors:#?}");
        assert_eq!(deep_errors.len(), 1, "{kind}: {deep_errors:#?}");
        assert_eq!(
            deep_errors[0].message, "expression is nested too deeply (the limit is 256 levels)",
            "{kind}"
        );
    }
}

/// Recurses with 1 KiB frames until `target` bytes of stack are in use
/// (measured from the address of a local in the first frame) and returns
/// the number of frames. `black_box` keeps each frame's array alive.
fn burn(base: usize, target: usize, depth: u32) -> u32 {
    let frame = std::hint::black_box([depth as u8; 1024]);
    let here = std::hint::black_box(&frame) as *const [u8; 1024] as usize;
    if base.abs_diff(here) >= target {
        return depth;
    }
    let frames = burn(base, target, depth + 1);
    // Use the array after the call so the frame can't be reused.
    std::hint::black_box(&frame);
    frames
}

#[test]
fn deep_but_legal_recursion_fits() {
    // 4 MiB of frames: twice what a default spawned thread has, half of
    // DEEP_STACK_BYTES.
    let target = 4 * 1024 * 1024;
    let frames = with_deep_stack(|| {
        let marker = 0u8;
        let base = std::hint::black_box(&marker) as *const u8 as usize;
        burn(base, target, 0)
    });
    assert!(frames > 100, "{frames}");
}
