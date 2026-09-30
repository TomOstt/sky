//! Parser errors: the message, and the span it points at (the source text
//! the span covers is compared, so the tests read as documentation).

mod common;

use common::*;
use sky_ir::{parse, print, MAX_REGION_DEPTH};

const HEAD: &str = "sir edition 2026\n";

/// Parses `src`, which must fail with a message containing `msg`, and
/// whose span covers exactly `at`.
#[track_caller]
fn fails(src: &str, msg: &str, at: &str) {
    let e = parse(src).expect_err(src);
    assert!(
        e.message.contains(msg),
        "message {:?} does not contain {msg:?}\n{src}",
        e.message
    );
    assert!(e.span.start <= e.span.end && e.span.end as usize <= src.len());
    assert_eq!(&src[e.span.range()], at, "span of {:?}\n{src}", e.message);
}

/// A function around one body line.
fn body(line: &str) -> String {
    format!("{HEAD}fn @f(%x: num[L; ()]) -> () {{\n  {line}\n  return\n}}\n")
}

#[test]
fn header_errors() {
    fails("", "expected `sir`", "");
    fails("fn @f() -> () {}", "expected `sir`", "fn");
    fails("sir\n", "expected `edition`", "");
    fails("sir edition", "expected a number", "");
    fails("sir edition 2027\n", "unknown edition 2027", "2027");
    fails(
        "sir edition 99999999999999999999\n",
        "number too large",
        "99999999999999999999",
    );
    fails("sir edition 2026\nhello", "expected `fn`", "hello");
}

#[test]
fn function_header_errors() {
    fails(&format!("{HEAD}fn f() -> () {{\n}}"), "expected `@`", "f");
    fails(
        &format!("{HEAD}fn @() -> () {{\n}}"),
        "expected a function name",
        "(",
    );
    fails(
        &format!("{HEAD}fn @f(x: bool) -> () {{\n}}"),
        "expected a value like",
        "x",
    );
    fails(
        &format!("{HEAD}fn @f(%x bool) -> () {{\n}}"),
        "expected `:`",
        "b",
    );
    fails(
        &format!("{HEAD}fn @f(%x: nope) -> () {{\n}}"),
        "expected a type",
        "nope",
    );
    fails(&format!("{HEAD}fn @f() {{\n}}"), "expected `-`", "{");
    fails(
        &format!("{HEAD}fn @f() -> () effects(disk) {{\n}}"),
        "expected `io`, `fail` or `call`",
        "disk",
    );
    fails(
        &format!("{HEAD}fn @f(%x: bool, %x: bool) -> () {{\n}}"),
        "`%x` is defined twice",
        "%x",
    );
    fails(
        &format!("{HEAD}fn @f() -> () @ 5.. {{\n}}"),
        "expected a number",
        "{",
    );
    fails(
        &format!("{HEAD}fn @f() -> () @ 5..4294967296 {{\n}}"),
        "span offset too large",
        "5..4294967296",
    );
    fails(
        &format!("{HEAD}fn @f() -> () {{\n"),
        "expected an operation, a `^` block, or `}`",
        "",
    );
    fails(
        &format!("{HEAD}fn @f() -> () {{\n  return\n"),
        "expected an operation, a `^` block, or `}`",
        "",
    );
}

#[test]
fn operation_errors() {
    fails(
        &body("frobnicate %x"),
        "unknown operation `frobnicate`",
        "frobnicate",
    );
    fails(
        &body("%a = neg %y : num[L; ()]"),
        "`%y` is used before it is defined",
        "%y",
    );
    fails(
        &body("%a = neg %a : num[L; ()]"),
        "`%a` is used before it is defined",
        "%a",
    );
    fails(
        &body("%a = neg %x : num[L; ()]\n  %a = neg %x : num[L; ()]"),
        "`%a` is defined twice",
        "%a",
    );
    fails(&body("%a = neg %x"), "needs `: types`", "");
    fails(
        &body("%a, %b = neg %x : num[L; ()]"),
        "1 result type(s) for 2 result value(s)",
        ": num[L; ()]",
    );
    fails(
        &body("neg %x : num[L; ()]"),
        "1 result type(s) for 0 result value(s)",
        ": num[L; ()]",
    );
    fails(
        &body("%a = neg %x : num[L; ()] extra"),
        "expected the end of the line",
        "e",
    );
    fails(
        &body("%a = neg %x, : num[L; ()]"),
        "expected a value like",
        ":",
    );
    fails(&body("%a = = neg %x"), "expected an operation", "=");
    fails(&body("%a neg %x"), "expected `=`", "n");
    fails(&body("%3"), "expected `=`", "");
    fails(&body("%a = neg %x : num[L; ()] @ 3"), "expected `..`", "");
}

#[test]
fn attribute_errors() {
    fails(
        &body("%a = const.float : num[1; ()]"),
        "expected a decimal or `0x` float",
        ":",
    );
    fails(
        &body("%a = const.float 1.2.3 : num[1; ()]"),
        "malformed float",
        "1.2.3",
    );
    fails(
        &body("%a = const.float 0x123456789abcdef01 : num[1; ()]"),
        "16 hex digits",
        "0x123456789abcdef01",
    );
    fails(
        &body("%a = const.float nan : num[1; ()]"),
        "expected a decimal",
        "n",
    );
    fails(
        &body("%a = const.rational 1/0 : num[1; ()]"),
        "division by zero",
        "1/0",
    );
    fails(
        &body("%a = const.rational 1/ : num[1; ()]"),
        "expected a number",
        ":",
    );
    fails(
        &body("%a = const.rational 99999999999999999999 : num[1; ()]"),
        "number too large",
        "99999999999999999999",
    );
    fails(
        &body("%a = const.bool maybe : bool"),
        "expected `true` or `false`",
        "maybe",
    );
    fails(
        &body("%a = const.str \"abc : str"),
        "unterminated string",
        "\"abc : str",
    );
    fails(&body("%a = const.str abc : str"), "expected a string", "a");
    fails(
        &body("%a = const.str \"\\q\" : str"),
        "unknown escape",
        "\\q",
    );
    fails(
        &body("%a = const.str \"\\u{110000}\" : str"),
        "invalid unicode escape",
        "\\u{110000}",
    );
    fails(
        &body("%a = const.str \"\\u12\" : str"),
        "expected `{` after `\\u`",
        "1",
    );
    fails(
        &body("%a = pow.const 2 %x : num[L; ()]"),
        "expected `[`",
        "2",
    );
    fails(
        &body("%a = solve[qr] %x, %x : num[L; ()]"),
        "expected `cholesky`, `lu` or `diagonal`",
        "qr",
    );
    fails(&body("%a = call f(%x) : num[L; ()]"), "expected `@`", "f");
    fails(&body("%a = call @f(%x : num[L; ()]"), "expected `)`", ":");
}

#[test]
fn type_errors() {
    let ty = |t: &str| body(&format!("%a = neg %x : {t}"));
    fails(&ty("float"), "expected a type", "float");
    fails(&ty("num"), "expected `[`", "");
    fails(&ty("num[L]"), "expected `;`", "]");
    fails(&ty("num[Q; ()]"), "expected a dimension symbol", "Q");
    fails(&ty("num[L·L; ()]"), "appears twice", "L");
    fails(&ty("num[L^(1/0); ()]"), "division by zero", "1/0");
    fails(&ty("num[L⁻; ()]"), "expected exponent digits", "⁻");
    fails(
        &ty("num[L¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹; ()]"),
        "exponent too large",
        "¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹",
    );
    fails(&ty("num[L; 3]"), "expected `(`", "3");
    fails(&ty("num[L; (3,)]"), "expected a size", ")");
    fails(&ty("num[L; (x)]"), "expected a size", "x");
    fails(
        &ty("num[L; (99999999999999999999)]"),
        "size too large",
        "99999999999999999999",
    );
    fails(&ty("num[L; (N#)]"), "expected a number", ")");
    fails(&ty("num[L; (); {Skew}]"), "expected a tag name", "Skew");
    fails(&ty("num[L; (); {Symmetric]"), "expected `}`", "]");
    fails(
        &ty("num[L; (); {Symmetric}; {Diagonal}]"),
        "tags given twice",
        "{Diagonal}",
    );
    fails(&ty("num[L; (); f16]"), "expected tags", "f16");
    fails(
        &ty("num[L; (); f32; f64]"),
        "representation given twice",
        "f64",
    );
    fails(&ty("num[L; () f64]"), "expected `;` or `]`", "f");
}

#[test]
fn region_errors() {
    let text =
        format!("{HEAD}fn @f(%c: bool) -> () {{\n  if %c {{\n    return\n  }}\n  return\n}}\n");
    fails(&text, "expected a `^` block or `}`", "r");
    // The `}` closes the region, not the function.
    let text = format!("{HEAD}fn @f(%c: bool) -> () {{\n  if %c {{\n    ^0:\n    return\n  }}\n");
    fails(&text, "expected an operation, a `^` block, or `}`", "");
    let text = format!(
        "{HEAD}fn @f() -> () {{\n  %a = const.bool true : bool\n  ^0(%a: bool):\n  return\n}}\n"
    );
    fails(&text, "`%a` is defined twice", "%a");
}

#[test]
fn regions_are_limited_in_depth() {
    let nest = |levels: usize| {
        let mut s = format!("{HEAD}fn @f(%c: bool) -> () {{\n");
        for _ in 0..levels {
            s.push_str("if %c {\n^0:\n");
        }
        s.push_str("return\n");
        for _ in 0..levels {
            s.push_str("}\n");
        }
        s.push_str("return\n}\n");
        s
    };
    // At the limit the text parses (the verifier has its own opinion); one
    // more level is a parse error, not a stack overflow.
    parse(&nest(MAX_REGION_DEPTH - 1)).unwrap();
    let e = parse(&nest(MAX_REGION_DEPTH)).unwrap_err();
    assert!(e.message.contains("nested too deeply"), "{e}");
    let e = parse(&nest(100_000)).unwrap_err();
    assert!(e.message.contains("nested too deeply"), "{e}");
}

#[test]
fn oversized_input_is_refused_up_front() {
    let big = " ".repeat(sky_syntax::MAX_SOURCE_BYTES + 1);
    let e = parse(&big).unwrap_err();
    assert!(e.message.contains("larger than"), "{e}");
    assert_eq!(e.span.start, 0);
    // Exactly at the limit is processed (and fails for its content).
    let at_limit = " ".repeat(sky_syntax::MAX_SOURCE_BYTES);
    let e = parse(&at_limit).unwrap_err();
    assert!(e.message.contains("expected `sir`"), "{e}");
}

#[test]
fn errors_render_as_diagnostics() {
    let src = body("frobnicate %x");
    let e = parse(&src).unwrap_err();
    let d = e.to_diagnostic();
    let file = sky_diag::SourceFile::new("t.sir", &src);
    let out = d.render_to_string(&file);
    assert!(
        out.contains("error: unknown operation `frobnicate`"),
        "{out}"
    );
    assert!(out.contains("t.sir:3:3"), "{out}");
    assert!(e.to_string().contains("at bytes"));
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let src = format!(
        "// a module\n{HEAD}\n// a function\nfn @f(%x: num[L; ()]) -> (num[L; ()]) {{ // open\n\n  %y = neg %x : num[L; ()] // negate\n  return %y\n}}\n"
    );
    parse(&src).unwrap();
}

#[test]
fn every_prefix_of_a_valid_module_parses_or_errors_without_panicking() {
    let text = print(&all_ops());
    let mut ok = 0;
    for i in 0..=text.len() {
        if !text.is_char_boundary(i) {
            continue;
        }
        match parse(&text[..i]) {
            Ok(_) => ok += 1,
            Err(e) => assert!(e.span.end as usize <= i),
        }
    }
    assert!(ok >= 1);
}

#[test]
fn slice_item_errors() {
    fails(
        &body("%a = slice[first] %x, %x : num[L; ()]"),
        "expected `at`, `:`, `lo:`, `:hi` or `lo:hi`",
        "first",
    );
    fails(
        &body("%a = slice[lo:up] %x, %x : num[L; ()]"),
        "expected `hi` or the end of the item",
        "up",
    );
    fails(&body("%a = slice %x : num[L; ()]"), "expected `[`", "%");
    fails(
        &body("%a = slice[at %x, %x : num[L; ()]"),
        "expected `]`",
        "%",
    );
    let ok = body("%a = slice[at, :, lo:, :hi, lo:hi] %x, %x, %x, %x, %x, %x : num[L; ()]");
    let printed = print(&parse(&ok).expect("parses"));
    assert!(printed.contains("slice[at, :, lo:, :hi, lo:hi] %x, %x, %x, %x, %x, %x"));
}

#[test]
fn math_operations_parse() {
    let src = body("%a = math.sqrt %x : num[L; ()]");
    assert!(print(&parse(&src).expect("parses")).contains("math.sqrt %x"));
}
