//! Rounding bounds of the algorithm rewrites: a planned chain, and a solve
//! instead of an inverse, are not bit-identical to the written form, so
//! these tests state how far apart they may be and check it on random data.
//!
//! Each program is run end to end (lex, check, lower, SIR interpreter), so
//! what is measured is what `sky run` computes, and the reference is the
//! same program with the order forced by parentheses, or with the inverse
//! named (`M = inv(A)` is never rewritten by the checker).
//!
//! # The bounds
//!
//! With `ε = 2⁻⁵³` (the unit roundoff) and `γₖ = kε / (1 - kε)`:
//!
//! * a floating-point matrix product satisfies `|fl(AB) - AB| ≤ γₙ |A||B|`
//!   componentwise (`n` the inner dimension; Higham, "Accuracy and Stability
//!   of Numerical Algorithms", 3.5). For `A(B(Cv))` and `((AB)C)v` each
//!   is within `γₖ |A||B||C||v|` of the exact product, `k` the total of the
//!   inner dimensions plus the 3 products' own roundings (bounded by
//!   `k = 3n + 3`), so **their difference is at most `2·γ₃ₙ₊₃ |A||B||C||v|`
//!   componentwise**;
//! * `solve[lu]` on a diagonally dominant matrix (no pivot growth: the
//!   growth factor is at most 2) has a small **residual**:
//!   `|A x̂ - b| ≤ c·γₙ (|A||x̂| + |b|)` componentwise with a modest constant
//!   `c` (we use 8, generous by the theory's 3), and it is close to the
//!   inverse-then-matmul answer, `|x̂ - x| ≤ 1e-9 (1 + |x|)`, because the
//!   matrices are well conditioned (condition number below about 30).

use proptest::prelude::*;
use sky_interp::linalg;

const EPS: f64 = f64::EPSILON / 2.0;

fn gamma(k: usize) -> f64 {
    let ke = k as f64 * EPS;
    ke / (1.0 - ke)
}

/// A `Matrix` literal of `n` rows.
fn matrix_lit(a: &[f64], n: usize) -> String {
    let rows: Vec<String> = (0..n)
        .map(|i| {
            let row: Vec<String> = (0..n).map(|j| format!("{}", a[i * n + j])).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

fn vector_lit(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", items.join(", "))
}

/// Runs `src` (checked, lowered, interpreted on SIR) and returns each printed
/// line as the numbers on it.
fn run(src: &str) -> Vec<Vec<f64>> {
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let mut out = Vec::new();
    sky_ast::with_deep_stack(|| sky_ir_interp::run(&ir, &mut out))
        .unwrap_or_else(|e| panic!("{e}\n{src}"));
    String::from_utf8(out)
        .expect("utf-8")
        .lines()
        .map(|l| {
            l.replace(['[', ']'], "")
                .split(", ")
                .map(|x| {
                    x.trim()
                        .parse::<f64>()
                        .unwrap_or_else(|_| panic!("`{x}` in `{l}`"))
                })
                .collect()
        })
        .collect()
}

/// Entries that are not exactly representable, so products round.
fn entry() -> impl Strategy<Value = f64> {
    (-700i32..=700).prop_map(|i| f64::from(i) / 7.0)
}

fn matrix(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(entry(), n * n)
}

/// A diagonally dominant matrix with entries off the diagonal of size at
/// most 1 and a diagonal of at least `n + 1`: well conditioned.
fn dominant(n: usize) -> impl Strategy<Value = Vec<f64>> {
    (
        proptest::collection::vec(-1.0f64..1.0, n * n),
        proptest::collection::vec(any::<bool>(), n),
    )
        .prop_map(move |(mut a, signs)| {
            for i in 0..n {
                let d = n as f64 + 1.0;
                a[i * n + i] += if signs[i] { d } else { -d };
            }
            a
        })
}

fn abs(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.abs()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(120))]

    /// `A @ B @ C @ v` (planned as `A @ (B @ (C @ v))`) against
    /// `((A @ B) @ C) @ v` (the written order, forced by parentheses).
    #[test]
    fn a_reassociated_chain_is_within_the_product_bound(
        (n, a, b, c, v) in (2usize..=8).prop_flat_map(|n| {
            (Just(n), matrix(n), matrix(n), matrix(n), proptest::collection::vec(entry(), n))
        })
    ) {
        let data = format!(
            "A = {}\nB = {}\nC = {}\nv = {}\n",
            matrix_lit(&a, n), matrix_lit(&b, n), matrix_lit(&c, n), vector_lit(&v)
        );
        let planned = &run(&format!("{data}print(A @ B @ C @ v)\n"))[0];
        let written = &run(&format!("{data}print(((A @ B) @ C) @ v)\n"))[0];

        // The written order is the plain left-to-right product, bit for bit.
        let ab = linalg::matmul(&a, n, n, &b, n);
        let abc = linalg::matmul(&ab, n, n, &c, n);
        let direct = linalg::matmul(&abc, n, n, &v, 1);
        prop_assert_eq!(written, &direct);

        // The planned one is within 2·γ of it, componentwise.
        let mag = linalg::matmul(
            &linalg::matmul(&linalg::matmul(&abs(&a), n, n, &abs(&b), n), n, n, &abs(&c), n),
            n, n, &abs(&v), 1,
        );
        let k = 3 * n + 3;
        for i in 0..n {
            let bound = 2.0 * gamma(k) * mag[i];
            prop_assert!(
                (planned[i] - written[i]).abs() <= bound,
                "entry {i}: {} vs {} differ by more than {bound}", planned[i], written[i]
            );
        }
    }

    /// `A @ B @ A @ u`, the same bound with a repeated factor.
    #[test]
    fn a_chain_with_a_repeated_factor_is_within_the_product_bound(
        (n, a, b, u) in (2usize..=6).prop_flat_map(|n| {
            (Just(n), matrix(n), matrix(n), proptest::collection::vec(entry(), n))
        })
    ) {
        // u @ A @ B is already left to right; put a column vector on the
        // right of a longer chain so the order changes: A @ B @ A @ u.
        let data = format!(
            "A = {}\nB = {}\nu = {}\n",
            matrix_lit(&a, n), matrix_lit(&b, n), vector_lit(&u)
        );
        let planned = &run(&format!("{data}print(A @ B @ A @ u)\n"))[0];
        let written = &run(&format!("{data}print(((A @ B) @ A) @ u)\n"))[0];
        let mag = linalg::matmul(
            &linalg::matmul(&linalg::matmul(&abs(&a), n, n, &abs(&b), n), n, n, &abs(&a), n),
            n, n, &abs(&u), 1,
        );
        for i in 0..n {
            let bound = 2.0 * gamma(3 * n + 3) * mag[i];
            prop_assert!((planned[i] - written[i]).abs() <= bound);
        }
    }

    /// `inv(A) @ b` (planned as `solve[lu](A, b)`) against `M @ b` with
    /// `M = inv(A)` named, which the checker leaves as an inverse and a
    /// product.
    #[test]
    fn an_lu_solve_has_a_small_residual_and_matches_the_inverse(
        (n, a, b) in (1usize..=8).prop_flat_map(|n| {
            (Just(n), dominant(n), proptest::collection::vec(entry(), n))
        })
    ) {
        let data = format!("A = {}\nb = {}\n", matrix_lit(&a, n), vector_lit(&b));
        let solved = &run(&format!("{data}print(inv(A) @ b)\n"))[0];
        let via_inverse = &run(&format!("{data}M = inv(A)\nprint(M @ b)\n"))[0];
        for i in 0..n {
            prop_assert!(
                (solved[i] - via_inverse[i]).abs() <= 1e-9 * (1.0 + via_inverse[i].abs()),
                "{} vs {}", solved[i], via_inverse[i]
            );
        }
        // |A x - b| <= 8 γₙ (|A||x| + |b|), componentwise.
        let ax = linalg::matmul(&a, n, n, solved, 1);
        let mag = linalg::matmul(&abs(&a), n, n, &abs(solved), 1);
        for i in 0..n {
            let bound = 8.0 * gamma(n) * (mag[i] + b[i].abs());
            prop_assert!((ax[i] - b[i]).abs() <= bound, "residual {} > {bound}", (ax[i] - b[i]).abs());
        }
    }

    /// `b @ inv(A)` (a transposed solve): `x A = b` to the same bound.
    #[test]
    fn a_transposed_lu_solve_has_a_small_residual(
        (n, a, b) in (1usize..=8).prop_flat_map(|n| {
            (Just(n), dominant(n), proptest::collection::vec(entry(), n))
        })
    ) {
        let data = format!("A = {}\nb = {}\n", matrix_lit(&a, n), vector_lit(&b));
        let solved = &run(&format!("{data}print(b @ inv(A))\n"))[0];
        let via_inverse = &run(&format!("{data}M = inv(A)\nprint(b @ M)\n"))[0];
        for i in 0..n {
            prop_assert!((solved[i] - via_inverse[i]).abs() <= 1e-9 * (1.0 + via_inverse[i].abs()));
        }
        // (x A)ⱼ = Σᵢ xᵢ aᵢⱼ.
        let xa = linalg::matmul(solved, 1, n, &a, n);
        let mag = linalg::matmul(&abs(solved), 1, n, &abs(&a), n);
        for j in 0..n {
            let bound = 8.0 * gamma(n) * (mag[j] + b[j].abs());
            prop_assert!((xa[j] - b[j]).abs() <= bound, "residual {} > {bound}", (xa[j] - b[j]).abs());
        }
    }

    /// `inv(A) @ inv(B) @ b`, planned as two solves, against the inverses
    /// formed and multiplied: both are the same number to within the
    /// conditioning (well-conditioned matrices here).
    #[test]
    fn two_solves_match_two_inverses_on_well_conditioned_matrices(
        (n, a, b, x) in (2usize..=6).prop_flat_map(|n| {
            (Just(n), dominant(n), dominant(n), proptest::collection::vec(entry(), n))
        })
    ) {
        let data = format!(
            "A = {}\nB = {}\nx = {}\n",
            matrix_lit(&a, n), matrix_lit(&b, n), vector_lit(&x)
        );
        let planned = &run(&format!("{data}print(inv(A) @ inv(B) @ x)\n"))[0];
        let named = &run(&format!("{data}P = inv(A)\nQ = inv(B)\nprint(P @ Q @ x)\n"))[0];
        for i in 0..n {
            prop_assert!(
                (planned[i] - named[i]).abs() <= 1e-9 * (1.0 + named[i].abs()),
                "{} vs {}", planned[i], named[i]
            );
        }
    }

    /// A Diagonal solve is the division `bᵢ / dᵢ`, one rounding: within
    /// `ε|x|` of the exact quotient, and never further from it than the
    /// inverse's two roundings (`bᵢ · fl(1/dᵢ)`) allow.
    #[test]
    fn a_diagonal_solve_is_correctly_rounded(
        (n, d, b) in (1usize..=6).prop_flat_map(|n| {
            (
                Just(n),
                proptest::collection::vec(
                    prop_oneof![(1i32..=200).prop_map(|i| f64::from(i) / 7.0), (-200i32..=-1).prop_map(|i| f64::from(i) / 7.0)],
                    n,
                ),
                proptest::collection::vec(entry(), n),
            )
        })
    ) {
        let mut a = vec![0.0; n * n];
        for i in 0..n { a[i * n + i] = d[i]; }
        // The Diagonal tag needs a literal the checker can prove; every
        // off-diagonal entry is a literal zero here.
        let data = format!("D: Diagonal Matrix[{n}, {n}] = {}\nb = {}\n", matrix_lit(&a, n), vector_lit(&b));
        let solved = &run(&format!("{data}print(inv(D) @ b)\n"))[0];
        for i in 0..n {
            prop_assert_eq!(solved[i].to_bits(), (b[i] / d[i] + 0.0).to_bits());
        }
    }
}
