//! The checker's chain planning (`sky_types::chain`) as the interpreters
//! read it: which `@` chains get a plan in `Checked::chains`, the text of
//! the `--explain` note, what parentheses do, and how the older per-site
//! Cholesky rewrite and nested plans coexist with it.

use sky_ast::with_deep_stack;
use sky_diag::Span;
use sky_types::{check_module, Checked, LeafForm, Node, MAX_STORED_REWRITES};

fn check(src: &str) -> Checked {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let c = check_module(&module, src);
        assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
        c
    })
}

/// The span of the first occurrence of `text` in `src`.
fn span_of(src: &str, text: &str) -> Span {
    let start = src
        .find(text)
        .unwrap_or_else(|| panic!("no `{text}` in the source"));
    Span::new(start as u32, (start + text.len()) as u32)
}

const BIG: &str = "A: Matrix[500, 500]\nB: Matrix[500, 500]\nC: Matrix[500, 500]\nv: Vector[500]\nb: Vector[500]\n";

fn with_big(line: &str) -> (String, Checked) {
    let src = format!("{BIG}{line}\n");
    let c = check(&src);
    (src, c)
}

fn notes_of(c: &Checked) -> Vec<&str> {
    c.chain_notes
        .iter()
        .flat_map(|d| d.notes.iter().map(String::as_str))
        .collect()
}

#[test]
fn a_matrix_chain_ending_in_a_vector_is_reassociated() {
    let (src, c) = with_big("x = A @ B @ C @ v");
    let span = span_of(&src, "A @ B @ C @ v");
    assert_eq!(c.chains.len(), 1);
    let plan = &c.chains[&span];
    assert_eq!(plan.factors.len(), 4);
    assert_eq!(plan.span_of(&plan.root), span);
    // A @ (B @ (C @ v)): the root's right child is a product of B and C @ v.
    let Node::Mul { lhs, rhs } = &plan.root else {
        panic!("{:?}", plan.root);
    };
    assert!(matches!(
        **lhs,
        Node::Leaf {
            factor: 0,
            form: LeafForm::Plain
        }
    ));
    assert_eq!(rhs.range(), (1, 3));
    assert!(!plan.root.is_left_associated());

    assert_eq!(c.chain_notes.len(), 1);
    let note = &c.chain_notes[0];
    assert_eq!(
        headline(note),
        "`A @ B @ C @ v` is rewritten to `A @ (B @ (C @ v))`"
    );
    assert_eq!(note.primary.span, span);
    let notes = notes_of(&c);
    assert!(
        notes.iter().any(|n| n.contains(
            "estimated cost: 500,500,000 flops as written, 1,500,000 flops after the rewrite"
        )),
        "{notes:#?}"
    );
    assert!(notes
        .iter()
        .any(|n| n.contains("write parentheses to force an order")));
    assert!(notes.iter().any(|n| n.contains("not bit-identical")));
    // No per-site Cholesky rewrite is involved.
    assert!(c.cholesky_solves.is_empty() && c.rewrites.is_empty());
}

#[test]
fn a_chain_already_in_the_best_order_records_nothing() {
    let (_, c) = with_big("x = A @ B");
    assert!(c.chains.is_empty() && c.chain_notes.is_empty());
    let (_, c) = with_big("x = A @ v");
    assert!(c.chains.is_empty());
    // A row vector on the left is already cheapest left to right.
    let (_, c) = with_big("x = v @ A @ B");
    assert!(c.chains.is_empty());
}

#[test]
fn parentheses_are_one_factor_and_are_respected() {
    // ((A @ B) @ C) @ v: every group is a two-factor chain, so the order the
    // user wrote is the order computed, expensive as it is.
    let (_, c) = with_big("x = ((A @ B) @ C) @ v");
    assert!(
        c.chains.is_empty(),
        "{:?}",
        c.chains.keys().collect::<Vec<_>>()
    );
    assert!(c.chain_notes.is_empty());
    // (A @ B) @ C @ v: the outer chain has the factors (A @ B), C and v: the
    // group stays whole and the rest is reordered.
    let (src, c) = with_big("x = (A @ B) @ C @ v");
    let span = span_of(&src, "(A @ B) @ C @ v");
    let plan = &c.chains[&span];
    assert_eq!(c.chains.len(), 1);
    assert_eq!(plan.factors.len(), 3);
    assert_eq!(plan.factors[0].span, span_of(&src, "(A @ B)"));
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`(A @ B) @ C @ v` is rewritten to `(A @ B) @ (C @ v)`"
    );
    // A @ (B @ C @ v): the group's own chain is planned on its own.
    let (src, c) = with_big("x = A @ (B @ C @ v)");
    let inner = span_of(&src, "B @ C @ v");
    assert_eq!(c.chains.len(), 1);
    assert!(c.chains.contains_key(&inner));
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`B @ C @ v` is rewritten to `B @ (C @ v)`"
    );
}

#[test]
fn symbolic_shapes_are_compared_for_every_size() {
    let src = "A: Matrix[N, N]\nB: Matrix[N, N]\nv: Vector[N]\nx = A @ B @ v\n";
    let c = check(src);
    assert_eq!(c.chain_notes.len(), 1);
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`A @ B @ v` is rewritten to `A @ (B @ v)`"
    );
    let notes = notes_of(&c);
    assert!(
        notes
            .iter()
            .any(|n| n.contains("estimated cost: 2·N³ + 2·N² flops as written, 4·N² flops after")),
        "{notes:#?}"
    );
    // Which order wins depends on N, M and K: the written one stays.
    let src = "A: Matrix[N, N]\nB: Matrix[N, M]\nC: Matrix[M, K]\nx = A @ B @ C\n";
    let c = check(src);
    assert!(c.chains.is_empty() && c.chain_notes.is_empty());
}

#[test]
fn an_inverse_is_a_solve_by_the_tags() {
    let src = "A: Matrix[3, 3]\nS: Symmetric Matrix[3, 3]\nD: Diagonal Matrix[3, 3]\nQ: Orthogonal Matrix[3, 3]\nb: Vector[3]\nx1 = inv(A) @ b\nx2 = b @ inv(A)\nx3 = S⁻¹ @ b\nx4 = D⁻¹ @ b\nx5 = Q⁻¹ @ b\nx6 = b @ D⁻¹\n";
    let c = check(src);
    let text: Vec<&str> = c.chain_notes.iter().map(headline).collect();
    assert_eq!(
        text,
        [
            "`inv(A) @ b` is rewritten to `solve[lu](A, b)`",
            "`b @ inv(A)` is rewritten to `solve.right[lu](b, A)`",
            "`S⁻¹ @ b` is rewritten to `solve[lu](S, b)`",
            "`D⁻¹ @ b` is rewritten to `solve[diagonal](D, b)`",
            "`Q⁻¹ @ b` is rewritten to `Qᵀ @ b`",
            "`b @ D⁻¹` is rewritten to `solve.right[diagonal](b, D)`",
        ]
    );
    assert_eq!(c.chains.len(), 6);
    // Q⁻¹ is Qᵀ, exactly by the tag; the note says so.
    let q = c.chain_notes[4].notes.join("\n");
    assert!(
        q.contains("Q is Orthogonal (Qᵀ·Q = Q·Qᵀ = I), so Q⁻¹ is Qᵀ"),
        "{q}"
    );
    // Every plan is a solve or a product and never forms the inverse.
    let span = span_of(src, "inv(A) @ b");
    let plan = &c.chains[&span];
    assert!(matches!(plan.root, Node::Solve { .. }));
    assert_eq!(plan.factor_forms(), [LeafForm::Deferred, LeafForm::Plain]);
    assert!(c.cholesky_solves.is_empty());
}

#[test]
fn positive_definite_stays_the_per_site_cholesky_rewrite() {
    let src = "P: PositiveDefinite Matrix[3, 3]\nb: Vector[3]\nx = P⁻¹ @ b\n";
    let c = check(src);
    assert!(c.chains.is_empty() && c.chain_notes.is_empty());
    assert_eq!(c.rewrites.len(), 1);
    assert_eq!(c.cholesky_solves.len(), 1);
    assert_eq!(
        c.rewrites[0].rewrite.explanation(),
        "rewrite `P⁻¹ @ b` to a Cholesky solve: P is Matrix[3,3] and PositiveDefinite, b is Vector[3]. Factor P = L·Lᵀ (L lower-triangular), solve L·y = b, then Lᵀ·x = y. x = P⁻¹·b is Vector[3], and P⁻¹ is never formed."
    );
    // With b on the left there is no per-site rewrite: the chain planner
    // solves it, by Cholesky.
    let src = "P: PositiveDefinite Matrix[3, 3]\nb: Vector[3]\ny = b @ P⁻¹\n";
    let c = check(src);
    assert_eq!(c.chain_notes.len(), 1);
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`b @ P⁻¹` is rewritten to `solve.right[cholesky](b, P)`"
    );
    assert!(c.cholesky_solves.is_empty());
    // Diagonal and PositiveDefinite: division, not a factorization.
    let src = "P: PositiveDefinite Diagonal Matrix[3, 3]\nb: Vector[3]\nx = P⁻¹ @ b\n";
    let c = check(src);
    assert!(c.cholesky_solves.is_empty() && c.rewrites.is_empty());
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`P⁻¹ @ b` is rewritten to `solve[diagonal](P, b)`"
    );
}

#[test]
fn a_product_of_inverses_is_the_inverse_of_the_reversed_product() {
    let src = "A: Matrix[4, 4]\nB: Matrix[4, 4]\nx = inv(A) @ inv(B)\n";
    let c = check(src);
    assert_eq!(c.chain_notes.len(), 1);
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`inv(A) @ inv(B)` is rewritten to `(B @ A)⁻¹`"
    );
    let span = span_of(src, "inv(A) @ inv(B)");
    let Node::InvOfProduct { left, right } = c.chains[&span].root else {
        panic!("{:?}", c.chains[&span].root);
    };
    assert_eq!((left, right), (0, 1));
}

#[test]
fn inverses_and_a_vector_solve_right_to_left() {
    let src = "A: Matrix[50, 50]\nB: Matrix[50, 50]\nb: Vector[50]\nx = inv(A) @ inv(B) @ b\n";
    let c = check(src);
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`inv(A) @ inv(B) @ b` is rewritten to `solve[lu](A, solve[lu](B, b))`"
    );
}

#[test]
fn an_outer_plan_withdraws_the_inner_ones() {
    // The inner `inv(P) @ Bm` is a per-site Cholesky rewrite; the outer chain
    // is planned as inv(P) @ (Bm @ v), which replaces it.
    let src = "P: PositiveDefinite Matrix[500, 500]\nBm: Matrix[500, 500]\nv: Vector[500]\nx = inv(P) @ Bm @ v\n";
    let c = check(src);
    assert_eq!(c.chains.len(), 1);
    assert_eq!(c.chain_notes.len(), 1);
    assert!(c.cholesky_solves.is_empty(), "{:?}", c.cholesky_solves);
    assert!(c.rewrites.is_empty());
    assert_eq!(
        headline(&c.chain_notes[0]),
        "`inv(P) @ Bm @ v` is rewritten to `solve[cholesky](P, Bm @ v)`"
    );
    assert!(c.chains.contains_key(&span_of(src, "inv(P) @ Bm @ v")));

    // The inner `inv(A) @ Bm` would be an LU plan of its own.
    let src = "A: Matrix[500, 500]\nBm: Matrix[500, 500]\nv: Vector[500]\nx = inv(A) @ Bm @ v\n";
    let c = check(src);
    assert_eq!(
        c.chains.len(),
        1,
        "{:?}",
        c.chains.keys().collect::<Vec<_>>()
    );
    assert_eq!(c.chain_notes.len(), 1);
    assert!(c.chains.contains_key(&span_of(src, "inv(A) @ Bm @ v")));
    assert!(!c.chains.contains_key(&span_of(src, "inv(A) @ Bm")));

    // Nothing to reorder outside: the per-site Cholesky rewrite of the inner
    // product stands, and no chain plan is recorded.
    let src = "P: PositiveDefinite Matrix[3, 3]\nb: Vector[3]\nx = P⁻¹ @ b @ b\n";
    let check = with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module(&module, src)
    });
    // (P⁻¹ @ b) @ b is a vector times a vector: a dot product, so the
    // Cholesky site remains and nothing else is planned.
    assert!(check.chains.is_empty());
    assert_eq!(check.cholesky_solves.len(), 1);
}

#[test]
fn a_chain_checked_twice_is_recorded_once() {
    let src = format!("{BIG}i = 0\nwhile i < 2:\n    x = A @ B @ C @ v\n    i = i + 1\n");
    let c = check(&src);
    assert_eq!(c.chains.len(), 1);
    assert_eq!(c.chain_notes.len(), 1);
    // And inside a function body, once per definition.
    let src = "def f(A: Matrix[N, N], B: Matrix[N, N], v: Vector[N]):\n    return A @ B @ v\n";
    let c = check(src);
    assert_eq!(c.chains.len(), 1, "{:?}", c.diagnostics);
    assert_eq!(c.chain_notes.len(), 1);
}

#[test]
fn stored_notes_are_capped_but_plans_are_not() {
    let mut src = String::from(BIG);
    for i in 0..(MAX_STORED_REWRITES + 20) {
        src.push_str(&format!("x{i} = A @ B @ C @ v\n"));
    }
    let c = check(&src);
    assert_eq!(c.chains.len(), MAX_STORED_REWRITES + 20);
    assert_eq!(c.chain_notes.len(), MAX_STORED_REWRITES);
    assert_eq!(c.chain_notes_omitted, 20);
}

#[test]
fn plans_carry_the_operand_spans_the_executors_check() {
    let src = "A: Matrix[3, 3]\nb: Vector[3]\nx = inv(A) @ b\n";
    let c = check(src);
    let plan = &c.chains[&span_of(src, "inv(A) @ b")];
    assert_eq!(plan.factors[0].span, span_of(src, "inv(A)"));
    // The `A` inside `inv(A)`, not the declaration's.
    let a = Span::new(
        (src.find("inv(A)").unwrap() + 4) as u32,
        (src.find("inv(A)").unwrap() + 5) as u32,
    );
    assert_eq!(plan.factors[0].inverse_operand, Some(a));
    assert_eq!(plan.factors[1].inverse_operand, None);
}

/// The note's headline without its `[RULE-ID] ` prefix.
fn headline(d: &sky_diag::Diagnostic) -> &str {
    match d.message.strip_prefix('[').and_then(|m| m.split_once("] ")) {
        Some((_, rest)) => rest,
        None => &d.message,
    }
}
