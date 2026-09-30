//! The exact algorithm rewrites of `sky-opt` (`transposes`, `structure`,
//! `powers`) and the algorithm-choice framework (`algorithms`).
//!
//! **Exact** means every printed digit and every error is the same before and
//! after. Each rewrite is checked that way, on random data that includes the
//! cases that could differ: zeros of both signs, negative diagonals, huge
//! values that overflow. `print` writes the shortest text that reads back as
//! the same `f64`, so equal output is equal bits (`-0` prints as `-0`).

use proptest::prelude::*;
use sky_ir::{Module, PassManager};
use sky_opt::{
    algorithms, default_passes, structure, transposes, AlgorithmPass, MatMulShape, ShapeClass,
    StructurePass, TransposesPass,
};
use sky_types::Algorithm;

/// What running a module printed, and the error that stopped it.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    diagnostic: Option<String>,
}

fn run(m: &Module, src: &str) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let result = sky_ir_interp::run(m, &mut out);
        Outcome {
            stdout: String::from_utf8(out).expect("UTF-8 output"),
            diagnostic: result.err().map(|e| {
                e.to_diagnostic()
                    .render_to_string(&sky_diag::SourceFile::new("test.sky", src))
            }),
        }
    })
}

fn lower(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"))
}

/// Runs the whole default pipeline, verified after every pass.
fn optimize(m: &mut Module) -> sky_ir::PipelineReport {
    default_passes(PassManager::for_tests())
        .run(m)
        .expect("valid IR")
}

/// The pipeline changes the text of the IR but not the behaviour.
fn same_behaviour(src: &str) -> (Outcome, Module) {
    let mut m = lower(src);
    let before = run(&m, src);
    optimize(&mut m);
    let after = run(&m, src);
    assert_eq!(
        after, before,
        "the pipeline changed the behaviour of:\n{src}"
    );
    (before, m)
}

fn text(m: &Module) -> String {
    sky_ir::print(m)
}

// ---------------------------------------------------------------- literals

/// A literal entry: small integers (so sums and products are easy to read
/// and zeros of both signs arise from `-1 * 0`), and a few values that round.
fn entry() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => (-4i32..5).prop_map(|n| n.to_string()),
        1 => Just("0.1".to_string()),
        1 => Just("-0.3".to_string()),
        1 => Just("1e-160".to_string()),
        1 => Just("2.5e150".to_string()),
    ]
}

fn matrix_text(rows: usize, cols: usize, entries: &[String]) -> String {
    let body: Vec<String> = (0..rows)
        .map(|i| format!("[{}]", entries[i * cols..(i + 1) * cols].join(", ")))
        .collect();
    format!("[{}]", body.join(", "))
}

fn diagonal_text(n: usize, entries: &[String]) -> String {
    let body: Vec<String> = (0..n)
        .map(|i| {
            let row: Vec<String> = (0..n)
                .map(|j| {
                    if i == j {
                        entries[i].clone()
                    } else {
                        "0".to_string()
                    }
                })
                .collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    format!("[{}]", body.join(", "))
}

// ------------------------------------------------------------- transposes

#[test]
fn a_transpose_feeding_a_product_is_read_transposed() {
    let src = "A = [[1, 2, 3], [4, 5, 6]]\nB = [[1, 0], [0, 2]]\nx = Aᵀ @ B\nprint(x)\n";
    let mut m = lower(src);
    let rws = transposes(&mut m);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].rule, "LA-TRANSPOSE-001");
    assert_eq!(
        (rws[0].before.as_str(), rws[0].after.as_str()),
        ("Aᵀ @ B", "matmul[ta](A, B)")
    );
    sky_ir::verify(&m).expect("valid");
    let t = text(&m);
    assert!(t.contains("matmul[ta] %A, %B"), "{t}");
    assert!(
        !t.contains("transpose"),
        "the unused transpose is gone:\n{t}"
    );
    // The same digits.
    let plain = lower(src);
    assert_eq!(run(&m, src), run(&plain, src));
}

#[test]
fn both_operands_transposed_and_a_transpose_with_other_users() {
    let src = "A = [[1, 2, 3], [4, 5, 6]]\nB = [[1, 0], [0, 2], [3, 1], [1, 1]]\n\
               x = Aᵀ @ Bᵀ\nt = Aᵀ\nprint(x, t)\n";
    let mut m = lower(src);
    let rws = transposes(&mut m);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].after, "matmul[ta,tb](A, B)");
    sky_ir::verify(&m).expect("valid");
    let t = text(&m);
    // `t = Aᵀ` still needs its transpose, and only that one is left.
    assert_eq!(t.matches("= transpose").count(), 1, "{t}");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

#[test]
fn a_vector_is_never_transposed() {
    // `v @ A` has no transpose in it, and `Aᵀ @ v` transposes the matrix.
    let src = "A = [[1, 2], [3, 4]]\nv = [1, 2]\nprint(v @ A, Aᵀ @ v)\n";
    let mut m = lower(src);
    let rws = transposes(&mut m);
    assert_eq!(rws.len(), 1);
    assert_eq!(rws[0].after, "matmul[ta](A, v)");
    sky_ir::verify(&m).expect("valid");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    #[test]
    fn transposed_products_are_bit_identical(
        m in 1usize..4, k in 1usize..4, n in 1usize..4,
        a in proptest::collection::vec(entry(), 9),
        b in proptest::collection::vec(entry(), 9),
        c in proptest::collection::vec(entry(), 9),
    ) {
        // A is m x k, B is n x k, C is m x n, E is n x m.
        let src = format!(
            "A = {}\nB = {}\nC = {}\nE = {}\nprint(Aᵀ @ C)\nprint(A @ Bᵀ)\nprint(Aᵀ @ Cᵀᵀ)\nprint(Aᵀ @ Eᵀ)\nprint(Aᵀ @ Aᵀᵀ)\n",
            matrix_text(m, k, &a[..m * k]),
            matrix_text(n, k, &b[..n * k]),
            matrix_text(m, n, &c[..m * n]),
            matrix_text(n, m, &c[..m * n]),
        );
        same_behaviour(&src);
    }
}

// -------------------------------------------------------------- structure

#[test]
fn gram_products_compute_one_triangle_and_say_so() {
    let src = "A = [[1, 2, 3], [4, 5, 6]]\ng = Aᵀ @ A\nh = A @ Aᵀ\nprint(g, h)\n";
    let mut m = lower(src);
    transposes(&mut m);
    let rws = structure(&mut m);
    let rules: Vec<&str> = rws.iter().map(|r| r.rule).collect();
    assert_eq!(rules, ["LA-GRAM-001", "LA-GRAM-001"]);
    sky_ir::verify(&m).expect("valid");
    let t = text(&m);
    assert!(t.contains("matmul[ta,alg=syrk] %A, %A"), "{t}");
    assert!(t.contains("matmul[tb,alg=syrk] %A, %A"), "{t}");
    // The result is Symmetric, so the next rule can use it.
    assert!(t.contains("{Symmetric}"), "{t}");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

#[test]
fn the_transpose_of_a_gram_product_is_itself() {
    let src = "A = [[1, 2, 3], [4, 5, 6]]\ng = Aᵀ @ A\nt = transpose(g)\nprint(t)\n";
    let mut m = lower(src);
    transposes(&mut m);
    let rws = structure(&mut m);
    assert!(rws.iter().any(|r| r.rule == "LA-SYMMETRIC-001"), "{rws:?}");
    sky_ir::verify(&m).expect("valid");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

#[test]
fn diagonal_products_scale_rows_and_columns() {
    let src = "D: Diagonal Matrix[2, 2] = [[2, 0], [0, -3]]\nX = [[1, 0], [0, 4]]\n\
               v = [5, 6]\nprint(D @ X, X @ D, D @ v, v @ D, D @ D)\n";
    let mut m = lower(src);
    let rws = structure(&mut m);
    assert_eq!(rws.len(), 5, "{rws:?}");
    assert!(rws.iter().all(|r| r.rule == "LA-DIAGONAL-001"));
    sky_ir::verify(&m).expect("valid");
    let t = text(&m);
    assert!(t.contains("diag.mul[left] %D, %X"), "{t}");
    assert!(t.contains("diag.mul[right] %X, %D"), "{t}");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

#[test]
fn a_matrix_that_is_only_symmetric_is_not_scaled() {
    let src = "S: Symmetric Matrix[2, 2] = [[1, 2], [2, 3]]\nprint(S @ S)\n";
    let mut m = lower(src);
    assert!(structure(&mut m).is_empty());
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Every entry of a diagonal product is `0.0 + d·x`: the same bits as the
    /// general product, including the sign of a zero and the overflow error.
    #[test]
    fn diagonal_scaling_is_bit_identical(
        n in 1usize..4, k in 1usize..4,
        d in proptest::collection::vec(entry(), 3),
        x in proptest::collection::vec(entry(), 9),
    ) {
        let src = format!(
            "D: Diagonal Matrix[{n}, {n}] = {}\nX = {}\nY = {}\nprint(D @ X)\nprint(Y @ D)\n",
            diagonal_text(n, &d),
            matrix_text(n, k, &x[..n * k]),
            matrix_text(k, n, &x[..n * k]),
        );
        same_behaviour(&src);
    }

    /// One triangle mirrored: the same bits as the general Gram product.
    #[test]
    fn gram_products_are_bit_identical(
        m in 1usize..4, n in 1usize..4,
        a in proptest::collection::vec(entry(), 9),
    ) {
        let src = format!(
            "A = {}\nprint(Aᵀ @ A)\nprint(A @ Aᵀ)\nprint(transpose(Aᵀ @ A))\n",
            matrix_text(m, n, &a[..m * n]),
        );
        let (_, opt) = same_behaviour(&src);
        prop_assert!(text(&opt).contains("alg=syrk"));
    }
}

// ------------------------------------------------------------------ powers

/// `powi`'s chain: what the pass must reproduce.
fn chain_value(x: f64, n: u32) -> f64 {
    let mut vals = vec![(1u32, x)];
    for (l, r) in sky_opt::powers::chain(n) {
        let get = |e: u32| vals.iter().rev().find(|(k, _)| *k == e).unwrap().1;
        vals.push((l + r, get(l) * get(r)));
    }
    vals.last().unwrap().1
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    /// The multiplication chain is bit-identical to `f64::powi` for every
    /// exponent the pass expands, for any finite input (a result that
    /// overflows is `inf` in both).
    #[test]
    fn powi_chains_are_bit_identical(bits in any::<u64>(), n in 2u32..=14) {
        let x = f64::from_bits(bits);
        prop_assume!(x.is_finite());
        let (want, got) = (x.powi(n as i32), chain_value(x, n));
        prop_assert_eq!(want.to_bits(), got.to_bits(), "x = {:e}, n = {}", x, n);
    }
}

#[test]
fn a_power_is_expanded_and_behaves_the_same() {
    for n in 2..=14 {
        let src = format!(
            "def p(x: Scalar):\n    return x ^ {n}\nprint(p(1.1))\nprint(p(-1.7))\nprint(p(0.3))\nprint(p(1e20))\n"
        );
        let mut m = lower(&src);
        let before = run(&m, &src);
        let rws = sky_opt::powers(&mut m);
        assert_eq!(rws.len(), 1, "x^{n}");
        assert_eq!(rws[0].rule, "POW-CHAIN-001");
        sky_ir::verify(&m).expect("valid");
        let t = text(&m);
        assert!(!t.contains("pow.const"), "x^{n}:\n{t}");
        let want = sky_opt::powers::chain(n).len();
        assert_eq!(t.matches("= mul ").count(), want, "x^{n}:\n{t}");
        assert_eq!(run(&m, &src), before, "x^{n}");
    }
}

#[test]
fn overflow_is_the_same_error_at_the_same_place() {
    for n in [2, 3, 5, 14] {
        let src = format!("def p(x: Scalar):\n    return x ^ {n}\nprint(1)\nprint(p(1e200))\n");
        let (o, _) = same_behaviour(&src);
        assert_eq!(o.stdout, "1\n");
        let d = o.diagnostic.expect("overflows");
        assert!(d.contains("arithmetic overflow"), "{d}");
    }
}

#[test]
fn other_exponents_are_left_alone() {
    let src =
        "def p(x: Scalar):\n    return x ^ 15 + x ^ 1 + x ^ -2 + x ^ 0.5 + x ^ 0\nprint(p(2))\n";
    let mut m = lower(src);
    let rws = sky_opt::powers(&mut m);
    assert!(rws.is_empty(), "{rws:?}");
}

#[test]
fn a_power_of_a_quantity_keeps_its_dimension() {
    let src = "def volume(l: m):\n    return l ^ 3\nprint(volume(2 m))\n";
    let mut m = lower(src);
    let before = run(&m, src);
    assert_eq!(sky_opt::powers(&mut m).len(), 1);
    sky_ir::verify(&m).expect("valid: the intermediate square is an area");
    assert_eq!(run(&m, src), before);
}

// ------------------------------------------------------------- algorithms

fn gemm_policy(p: &MatMulShape) -> Algorithm {
    match p.class {
        ShapeClass::Gemm if p.all_at_most(8) => Algorithm::Gemm,
        ShapeClass::MatVec => Algorithm::MatVec,
        // A choice that does not fit the class is ignored.
        ShapeClass::VecMat => Algorithm::Gemm,
        _ => Algorithm::Auto,
    }
}

#[test]
fn a_policy_records_its_choice_and_the_result_is_unchanged() {
    let src = "A = [[1, 2], [3, 4]]\nv = [1, 2]\nprint(A @ A, A @ v, v @ A, v @ v)\n";
    let mut m = lower(src);
    let rws = algorithms(&mut m, gemm_policy);
    let after: Vec<&str> = rws.iter().map(|r| r.after.as_str()).collect();
    assert_eq!(after.len(), 2, "{rws:?}");
    assert!(after[0].contains("alg=gemm"), "{after:?}");
    assert!(after[1].contains("alg=matvec"), "{after:?}");
    assert!(rws.iter().all(|r| r.rule == "ALG-MATMUL-001"));
    sky_ir::verify(&m).expect("valid");
    let t = text(&m);
    assert!(t.contains("matmul[alg=gemm] %A, %A"), "{t}");
    assert!(t.contains("matmul[alg=matvec] %A, %v"), "{t}");
    // v @ A, whose class the policy got wrong, and v @ v stay plain.
    assert_eq!(t.matches("= matmul ").count(), 2, "{t}");
    assert_eq!(run(&m, src), run(&lower(src), src));
}

#[test]
fn the_default_policy_chooses_nothing_and_a_pass_runs_a_custom_one() {
    let src = "A = [[1, 2], [3, 4]]\nprint(A @ A)\n";
    let mut m = lower(src);
    assert!(algorithms(&mut m, sky_opt::default_policy).is_empty());
    let mut m = lower(src);
    let report = PassManager::for_tests()
        .with(AlgorithmPass {
            matmul: gemm_policy,
        })
        .run(&mut m)
        .unwrap();
    assert_eq!(report.explanations.len(), 1);
    assert_eq!(report.explanations[0].pass, "algorithms");
    assert!(text(&m).contains("matmul[alg=gemm]"));
}

#[test]
fn a_recorded_algorithm_is_not_overwritten_and_transposes_keep_theirs() {
    let src = "A = [[1, 2], [3, 4]]\nprint(Aᵀ @ A)\n";
    let mut m = lower(src);
    let report = PassManager::for_tests()
        .with(TransposesPass)
        .with(StructurePass)
        .with(AlgorithmPass {
            matmul: gemm_policy,
        })
        .run(&mut m)
        .unwrap();
    // syrk was chosen by `structure`; the policy leaves it.
    assert!(text(&m).contains("matmul[ta,alg=syrk]"), "{}", text(&m));
    assert!(report
        .explanations
        .iter()
        .all(|x| x.rule != "ALG-MATMUL-001"));
}
