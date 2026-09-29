//! Checker behaviour for array literals, indexing, the array builtins and
//! the literal tag proofs, beyond what the goldens in `tests/ui/types/`
//! pin: exact types, the proof boundaries, and the rewrite spans the
//! interpreter reads.

use sky_ast::{with_deep_stack, MAX_DEPTH};
use sky_types::{check_module, Checked};

fn check(src: &str) -> Checked {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module(&module, src)
    })
}

fn ty(c: &Checked, name: &str) -> String {
    c.env
        .get(name)
        .map_or_else(|| "<unbound>".to_string(), |t| t.to_string())
}

fn messages(c: &Checked) -> Vec<&str> {
    c.diagnostics.iter().map(|d| d.message.as_str()).collect()
}

#[test]
fn literals_have_exact_shapes_and_no_tags() {
    let c = check("v = [1 m, 2 m, 3 m]\nA = [[1, 2], [2, 1]]\nB = [[1 s, 2 s, 3 s]]\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "v"), "Vector[3] [L]");
    // Symmetric entries, but a literal is untagged until declared.
    assert_eq!(ty(&c, "A"), "Matrix[2,2] [1]");
    assert_eq!(ty(&c, "B"), "Matrix[1,3] [T]");
}

#[test]
fn a_trailing_comma_and_units_of_the_same_dimension_are_fine() {
    let c = check("v = [1 m, 2 km, 3 cm,]\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "v"), "Vector[3] [L]");
}

#[test]
fn the_odd_element_out_is_the_one_reported() {
    // The dimension most elements have is expected, wherever the odd one is.
    for (lit, odd) in [
        ("[2 s, 1 m, 3 m]", "2 s"),
        ("[1 m, 2 m, 3 s]", "3 s"),
        ("[1 m, 3 s, 2 m]", "3 s"),
    ] {
        let src = format!("v = {lit}\n");
        let c = check(&src);
        assert_eq!(c.diagnostics.len(), 1, "{lit}: {:#?}", c.diagnostics);
        let d = &c.diagnostics[0];
        assert!(d.message.contains("expected length, found time"), "{lit}");
        assert_eq!(&src[d.primary.span.range()], odd, "{lit}");
    }
}

#[test]
fn many_bad_elements_are_capped_with_a_count() {
    let mut src = String::from("v = [");
    for i in 0..40 {
        src.push_str(if i % 4 == 0 { "1 s, " } else { "1 m, " });
    }
    src.push_str("]\n");
    let c = check(&src);
    assert_eq!(c.diagnostics.len(), 5, "{:#?}", c.diagnostics);
    assert!(
        c.diagnostics[4].notes[0].contains("5 more"),
        "{:#?}",
        c.diagnostics[4]
    );
}

#[test]
fn a_failed_literal_does_not_cascade() {
    let c = check("v = [1 m, 2 s, 3 m]\nw = v + v\nx = v[0]\n");
    assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
}

#[test]
fn indexing_types() {
    let c = check(
        "A = [[1 m, 2 m, 3 m], [4 m, 5 m, 6 m]]\ni = 1\n\
         a = A[0, 1]\nb = A[i, i]\nr = A[1, :]\nq = A[:, 2]\ns = A[0:2, 1:3]\nu = A[i:2, :]\nw = A[0]\n",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "a"), "Scalar [L]");
    assert_eq!(ty(&c, "b"), "Scalar [L]");
    assert_eq!(ty(&c, "r"), "Vector[3] [L]");
    assert_eq!(ty(&c, "q"), "Vector[2] [L]");
    assert_eq!(ty(&c, "s"), "Matrix[2,2] [L]");
    assert_eq!(ty(&c, "w"), "Vector[3] [L]");
    // A slice with a variable bound has an unknown (but consistent) length.
    assert!(ty(&c, "u").starts_with("Matrix[_,3]"), "{}", ty(&c, "u"));
}

#[test]
fn indexing_a_tagged_matrix_drops_the_tags() {
    let c = check("K: PositiveDefinite Matrix[3, 3] N/m\nk = K[0:2, 0:2]\nd = K[1, 1]\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "k"), "Matrix[2,2] [M·T⁻²]");
    assert_eq!(ty(&c, "d"), "Scalar [M·T⁻²]");
}

#[test]
fn index_bounds_are_checked_only_when_constant() {
    let c = check("v = [1, 2, 3]\ni = 7\na = v[i]\nb = v[3]\n");
    assert_eq!(
        messages(&c),
        ["index 3 is out of bounds for axis 0 of length 3"]
    );
    // Symbolic axis lengths cannot be checked either.
    let c = check("v: Vector[N]\na = v[100]\nb = v[0:100]\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}

#[test]
fn builtins_are_typed_and_can_be_shadowed() {
    let c = check("z = zeros(3)\nZ = zeros(2, 3)\nI = identity(2)\nn = len(Z)\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "z"), "Vector[3] [1]");
    assert_eq!(ty(&c, "Z"), "Matrix[2,3] [1]");
    assert_eq!(
        ty(&c, "I"),
        "Matrix[2,2] {Symmetric, PositiveDefinite, Diagonal, Orthogonal} [1]"
    );
    assert_eq!(ty(&c, "n"), "Scalar [1]");
    // A user's own `zeros` wins, as with `print` and `inv`.
    let c = check("def zeros(n):\n    return n\nz = zeros(3)\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "z"), "?");
}

#[test]
fn identity_and_zeros_can_carry_their_tags_into_a_declaration() {
    let c = check(
        "D: Diagonal Matrix[3, 3] = identity(3)\nQ: Orthogonal Matrix[2, 2] = identity(2)\n\
         Z: Diagonal Matrix[2, 2] = zeros(2, 2)\nP: PositiveDefinite Matrix[2, 2] = zeros(2, 2)\n",
    );
    assert_eq!(
        messages(&c),
        ["`zeros(2, 2)` is not known to be PositiveDefinite"]
    );
}

#[test]
fn literal_tag_proofs() {
    let ok = |src: &str| {
        let c = check(src);
        assert!(c.diagnostics.is_empty(), "{src}: {:#?}", c.diagnostics);
    };
    let bad = |src: &str, what: &str| {
        let c = check(src);
        assert_eq!(c.diagnostics.len(), 1, "{src}: {:#?}", c.diagnostics);
        assert!(c.diagnostics[0].message.contains(what), "{src}");
    };
    ok("S: Symmetric Matrix[3, 3] = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]\n");
    ok("S: Symmetric Matrix[2, 2] = [[1, -2], [-2, (1)]]\n");
    // The same value through different spellings and units is equal.
    ok("S: Symmetric Matrix[2, 2] m = [[0 m, 1 m], [100 cm, 0 m]]\n");
    ok("D: Diagonal Matrix[2, 2] = [[3, 0.0], [-0, 1e-3]]\n");
    // Diagonal implies Symmetric, and PositiveDefinite does too.
    ok("D: Symmetric Matrix[2, 2] = [[1, 0], [0, 2]]\n");
    ok("P: PositiveDefinite Matrix[2, 2] = [[2, -1], [-1, 2]]\n");
    ok("P: PositiveDefinite Symmetric Matrix[2, 2] = [[2, -1], [-1, 2]]\n");
    bad(
        "S: Symmetric Matrix[2, 2] = [[1, 2], [2.0000001, 1]]\n",
        "not known to be Symmetric",
    );
    bad(
        "D: Diagonal Matrix[2, 2] = [[1, 0], [1e-300, 1]]\n",
        "not known to be Diagonal",
    );
    // Symmetric but indefinite, semidefinite, and negative definite.
    bad(
        "P: PositiveDefinite Matrix[2, 2] = [[1, 2], [2, 1]]\n",
        "PositiveDefinite",
    );
    bad(
        "P: PositiveDefinite Matrix[2, 2] = [[1, 1], [1, 1]]\n",
        "PositiveDefinite",
    );
    bad(
        "P: PositiveDefinite Matrix[2, 2] = [[-1, 0], [0, -1]]\n",
        "PositiveDefinite",
    );
    // Positive definite but within the margin of singular: not proven.
    bad(
        "P: PositiveDefinite Matrix[2, 2] = [[1, 1], [1, 1.0000000001]]\n",
        "PositiveDefinite",
    );
    // Not every entry is a numeric literal.
    bad(
        "x = 1\nS: Symmetric Matrix[2, 2] = [[x, 0], [0, x]]\n",
        "Symmetric",
    );
    bad(
        "S: Symmetric Matrix[2, 2] = [[1 + 1, 0], [0, 2]]\n",
        "Symmetric",
    );
    bad(
        "O: Orthogonal Matrix[2, 2] = [[1, 0], [0, 1]]\n",
        "Orthogonal",
    );
}

#[test]
fn a_big_literal_is_not_proven_positive_definite() {
    // 33 by 33 identity: past the size the checker will factor.
    let row = |i: usize| {
        let cells: Vec<&str> = (0..33).map(|j| if i == j { "1" } else { "0" }).collect();
        format!("[{}]", cells.join(", "))
    };
    let rows: Vec<String> = (0..33).map(row).collect();
    let src = format!(
        "P: PositiveDefinite Matrix[33, 33] = [{}]\nD: Diagonal Matrix[33, 33] = [{}]\n",
        rows.join(", "),
        rows.join(", ")
    );
    let c = check(&src);
    // Diagonal is exact and has no size limit; PositiveDefinite does.
    assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
    assert!(c.diagnostics[0].message.contains("PositiveDefinite"));
}

#[test]
fn a_proven_literal_takes_part_in_the_cholesky_rewrite() {
    let src = "K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nf = [1, 2]\nu = K⁻¹ @ f\nw = K⁻¹ @ (f + f)\ns: Symmetric Matrix[2, 2] = [[4, 1], [1, 3]]\nv = s⁻¹ @ f\n";
    let c = check(src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // Two sites; the Symmetric-only matrix is not rewritten.
    assert_eq!(c.rewrites.len(), 2);
    assert_eq!(c.cholesky_solves.len(), 2);
    for site in &c.rewrites {
        assert!(c.cholesky_solves.contains(&site.span));
    }
}

#[test]
fn rewrite_spans_are_not_capped() {
    let mut src = String::from("K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nf = [1, 2]\n");
    for i in 0..(sky_types::MAX_STORED_REWRITES + 20) {
        src.push_str(&format!("u{i} = K⁻¹ @ f\n"));
    }
    let c = check(&src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.rewrites.len(), sky_types::MAX_STORED_REWRITES);
    assert_eq!(c.rewrites_omitted, 20);
    assert_eq!(c.cholesky_solves.len(), sky_types::MAX_STORED_REWRITES + 20);
}

#[test]
fn nested_literals_at_the_depth_limit_do_not_overflow_the_stack() {
    let n = MAX_DEPTH as usize - 1;
    let src = format!("x = {}1{}\n", "[".repeat(n), "]".repeat(n));
    let c = check(&src);
    assert!(!c.diagnostics.is_empty()); // "at most two levels", not a crash
}
