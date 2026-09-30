//! Array behaviour: literals, printing, arithmetic, `@`, the inverse, the
//! Cholesky solve, indexing and the builtins. Known results are compared
//! exactly where the arithmetic is exact and to a tolerance where it is not.

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::{run_module, run_module_with, RuntimeEnv, RuntimeError, Value};

/// Runs `src` (no checker), returning the value of the last expression.
fn run(src: &str) -> Result<Value, RuntimeError> {
    run_in(&mut RuntimeEnv::new(), src)
}

fn run_in(env: &mut RuntimeEnv, src: &str) -> Result<Value, RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?} in {src:?}");
    let mut out = Vec::new();
    run_module(&module, &src, env, &mut out).map(|v| v.unwrap_or(Value::None))
}

fn show(src: &str) -> String {
    run(src).expect("runs").to_string()
}

fn err(src: &str) -> String {
    run(src).expect_err("fails").message.clone()
}

fn data(v: &Value) -> Vec<f64> {
    match v {
        Value::Array { data, .. } => data.clone(),
        other => vec![match other {
            Value::Number { value, .. } => *value,
            _ => panic!("not numeric: {other:?}"),
        }],
    }
}

fn close(v: &Value, want: &[f64], tol: f64) {
    let got = data(v);
    assert_eq!(got.len(), want.len(), "{v}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= tol, "{v} vs {want:?}");
    }
}

#[test]
fn literals_print_as_sky_source() {
    assert_eq!(show("[1 m, 2 m, 3 m]"), "[1 m, 2 m, 3 m]");
    assert_eq!(show("[1, 2, 3]"), "[1, 2, 3]");
    assert_eq!(show("[[1, 2], [3, 4]]"), "[[1, 2], [3, 4]]");
    assert_eq!(show("[[1 N, 2 N]]"), "[[1 N, 2 N]]");
    // Entries convert to SI, and every entry carries the unit.
    assert_eq!(show("[1 km, 250 cm]"), "[1000 m, 2.5 m]");
    assert_eq!(show("[3 kg*m/s^2]"), "[3 N]");
    assert_eq!(show("[0.5 m/s, 2 m/s]"), "[0.5 m/s, 2 m/s]");
    assert_eq!(show("[1e-5, 1e20]"), "[1e-5, 1e20]");
}

#[test]
fn printed_arrays_read_back_as_the_same_value() {
    for src in [
        "[1 m, 2.5 m]",
        "[[1 N, -2 N], [0.1 N, 4e-9 N]]",
        "[[1e30, 2e-30, 3]]",
        "[1 kg*m^2/s^2, 7 kg*m^2/s^2]",
    ] {
        let v = run(src).unwrap();
        assert_eq!(run(&v.to_string()).unwrap(), v, "{src}");
    }
}

#[test]
fn literal_errors_are_defensive() {
    assert_eq!(
        err("[1 m, 2 s]"),
        "array elements must have the same dimension: expected length, found time"
    );
    assert_eq!(
        err("[[1, 2], [3]]"),
        "matrix rows must have the same length: expected 2, found 1"
    );
    assert_eq!(
        err("[true, false]"),
        "an array element must be a number, found a Bool"
    );
    assert_eq!(
        err("[1, [2]]"),
        "an array element must be a number, found a vector"
    );
}

#[test]
fn elementwise_and_scaling() {
    assert_eq!(show("[1 m, 2 m] + [3 m, 4 m]"), "[4 m, 6 m]");
    assert_eq!(
        show("[[1, 2], [3, 4]] - [[1, 1], [1, 1]]"),
        "[[0, 1], [2, 3]]"
    );
    assert_eq!(show("2 * [1 m, 2 m]"), "[2 m, 4 m]");
    assert_eq!(show("[1 m, 2 m] * 3 s"), "[3 m*s, 6 m*s]");
    assert_eq!(show("[2 m, 4 m] / 2 s"), "[1 m/s, 2 m/s]");
    assert_eq!(show("-[1 m, -2 m]"), "[-1 m, 2 m]");
    assert_eq!(show("+[1, 2]"), "[1, 2]");
    // Negating zero does not produce -0.
    assert_eq!(show("-[0, 1]"), "[0, -1]");
    assert_eq!(err("[1 m] + [1 s]"), "cannot add length and time");
    assert_eq!(
        err("[1, 2] + [1, 2, 3]"),
        "cannot add Vector[2] and Vector[3]"
    );
    assert_eq!(
        err("[1, 2] - [[1, 2]]"),
        "cannot subtract Matrix[1,2] from Vector[2]"
    );
    assert_eq!(err("[1, 2] + 1"), "cannot add a vector and a number");
    assert_eq!(
        err("[1, 2] * [1, 2, 3]"),
        "cannot apply `*` to Vector[2] and Vector[3]"
    );
    assert_eq!(
        err("[1, 2] / [[1, 2]]"),
        "cannot apply `/` to Vector[2] and Matrix[1,2]"
    );
    assert_eq!(err("1 / [1, 2]"), "cannot divide a number by a vector");
    assert_eq!(err("[1, 2] / 0"), "division by zero");
}

#[test]
fn results_are_finite() {
    assert_eq!(err("[1e308] + [1e308]"), "arithmetic overflow");
    assert_eq!(err("1e200 * [1e200, 1]"), "arithmetic overflow");
    assert_eq!(err("[1e308, 1] / 1e-10"), "arithmetic overflow");
    assert_eq!(
        err("[1e200, 1e200] @ [1e200, 1e200]"),
        "arithmetic overflow"
    );
    assert_eq!(err("[[1e-320, 0], [0, 1e-320]]⁻¹"), "arithmetic overflow");
}

#[test]
fn matmul_known_products() {
    assert_eq!(
        show("[[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]"),
        "[[58, 64], [139, 154]]"
    );
    assert_eq!(show("[[1, 2], [3, 4]] @ [1, 1]"), "[3, 7]");
    assert_eq!(show("[1, 1] @ [[1, 2], [3, 4]]"), "[4, 6]");
    // A dot product is a number, and units multiply.
    assert_eq!(show("[1 N, 2 N] @ [3 m, 4 m]"), "11 J");
    assert_eq!(show("[[2 m, 0 m], [0 m, 2 m]] @ [1 N, 1 N]"), "[2 J, 2 J]");
    assert_eq!(
        err("[[1, 2, 3]] @ [[1, 2, 3]]"),
        "cannot multiply Matrix[1,3] and Matrix[1,3]: the inner dimensions 3 and 1 differ"
    );
    assert_eq!(
        err("[1, 2] @ [1, 2, 3]"),
        "cannot multiply Vector[2] and Vector[3]: the inner dimensions 2 and 3 differ"
    );
}

#[test]
fn inverse_of_known_matrices() {
    let v = run("[[4, 7], [2, 6]]⁻¹").unwrap();
    close(&v, &[0.6, -0.7, -0.2, 0.4], 1e-15);
    // `inv(A)` is the same operator.
    assert_eq!(run("inv([[4, 7], [2, 6]])").unwrap(), v);
    // The inverse of a matrix in N/m is in m/N, printed in base units.
    let flex = run("[[2 N/m, 0 N/m], [0 N/m, 4 N/m]]⁻¹").unwrap();
    assert_eq!(
        flex.to_string(),
        "[[0.5 s^2/kg, 0 s^2/kg], [0 s^2/kg, 0.25 s^2/kg]]"
    );
    // A times its inverse is the identity.
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]").unwrap();
    let prod = run_in(&mut env, "A @ A⁻¹").unwrap();
    close(&prod, &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 1e-15);
}

#[test]
fn transpose_swaps_the_axes_and_keeps_dimension_and_values() {
    assert_eq!(show("[[1, 2, 3], [4, 5, 6]]ᵀ"), "[[1, 4], [2, 5], [3, 6]]");
    // `transpose(A)` is the same operator, like `inv(A)` for `⁻¹`.
    assert_eq!(
        run("transpose([[1, 2, 3], [4, 5, 6]])").unwrap(),
        run("[[1, 2, 3], [4, 5, 6]]ᵀ").unwrap()
    );
    // Units ride along untouched.
    assert_eq!(
        show("[[1 m, 2 m], [3 m, 4 m]]ᵀ"),
        "[[1 m, 3 m], [2 m, 4 m]]"
    );
    // A double transpose is the original, and a number is its own transpose.
    assert_eq!(show("[[1, 2, 3], [4, 5, 6]]ᵀᵀ"), "[[1, 2, 3], [4, 5, 6]]");
    assert_eq!(show("5 kg ᵀ"), "5 kg");
    assert_eq!(show("5 kgᵀ"), "5 kg");
    assert_eq!(show("transpose(2)"), "2");
    // Postfix binds tighter than `@`: `Aᵀ @ A` is Gram, and it is symmetric.
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "A = [[1, 2], [3, 4], [5, 6]]").unwrap();
    assert_eq!(
        run_in(&mut env, "Aᵀ @ A").unwrap().to_string(),
        "[[35, 44], [44, 56]]"
    );
    assert_eq!(
        run_in(&mut env, "A @ Aᵀ").unwrap().to_string(),
        "[[5, 11, 17], [11, 25, 39], [17, 39, 61]]"
    );
    // (A B)ᵀ = Bᵀ Aᵀ.
    run_in(&mut env, "B = [[1, 0, 2], [0, 1, 3]]").unwrap();
    assert_eq!(
        run_in(&mut env, "(B @ A)ᵀ").unwrap(),
        run_in(&mut env, "Aᵀ @ Bᵀ").unwrap()
    );
}

#[test]
fn transpose_of_a_vector_or_a_non_array_is_a_runtime_error() {
    assert_eq!(
        err("[1, 2, 3]ᵀ"),
        "cannot transpose Vector[3]: transpose needs a matrix"
    );
    assert_eq!(
        err("transpose([1, 2])"),
        "cannot transpose Vector[2]: transpose needs a matrix"
    );
    assert_eq!(err("trueᵀ"), "cannot transpose a Bool");
    assert_eq!(
        err("transpose(1, 2)"),
        "`transpose` takes 1 argument, found 2"
    );
}

#[test]
fn a_user_function_named_transpose_shadows_the_builtin() {
    assert_eq!(
        show("def transpose(x):\n    return x + 1\ntranspose(1)"),
        "2"
    );
}

#[test]
fn inverse_of_a_symmetric_matrix_is_bitwise_symmetric() {
    let v = run("[[2, -1, 0.3, 0.1], [-1, 2, -1, 0.7], [0.3, -1, 2, -1], [0.1, 0.7, -1, 2]]⁻¹")
        .unwrap();
    let d = data(&v);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(d[i * 4 + j].to_bits(), d[j * 4 + i].to_bits(), "({i}, {j})");
        }
    }
}

#[test]
fn singular_matrices_are_runtime_errors() {
    let e = run("[[1, 2], [2, 4]]⁻¹").unwrap_err();
    assert_eq!(e.message, "the matrix is singular");
    let e = run("inv([[1, 2, 3], [4, 5, 6], [7, 8, 9]])").unwrap_err();
    assert_eq!(e.message, "the matrix is singular");
    assert_eq!(err("[[0, 0], [0, 0]]⁻¹"), "the matrix is singular");
    assert_eq!(
        err("[1, 2]⁻¹"),
        "the inverse needs a square matrix, found Vector[2]"
    );
    assert_eq!(
        err("[[1, 2, 3], [4, 5, 6]]⁻¹"),
        "the inverse needs a square matrix, found Matrix[2,3]"
    );
}

/// Checks `src`, then runs it honouring the checker's Cholesky rewrites,
/// against `env` (whose values may differ from what the checker assumed,
/// which is how these tests reach the rewrite with a matrix that lies).
fn run_checked(
    env: &mut RuntimeEnv,
    types: sky_types::Env,
    src: &str,
) -> Result<Value, RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?}");
    let checked = sky_types::check_module_with(&module, &src, types);
    assert!(!checked.has_errors(), "{:#?}", checked.diagnostics);
    let mut out = Vec::new();
    run_module_with(&module, &src, env, &mut out, &checked.cholesky_solves)
        .map(|v| v.expect("an expression"))
}

fn pd_types(name: &str, n: u64) -> sky_types::Env {
    let mut types = sky_types::Env::new();
    types
        .bind_array(
            name,
            sky_types::Shape::matrix(n, n),
            sky_types::TagSet::of(sky_types::Tag::PositiveDefinite),
            sky_units::Dimension::DIMENSIONLESS,
        )
        .expect("square");
    types
}

#[test]
fn cholesky_solve_matches_a_known_solution() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] = L·Lᵀ with
    // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]; A·[1, 2, 3] = [-20, -43, 192].
    let mut env = RuntimeEnv::new();
    let src = "K: PositiveDefinite Matrix[3, 3] = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]\n\
               K⁻¹ @ [-20, -43, 192]";
    let x = run_checked(&mut env, sky_types::Env::new(), src).unwrap();
    assert_eq!(x.to_string(), "[1, 2, 3]");
}

#[test]
fn cholesky_solve_handles_units_and_several_right_hand_sides() {
    let mut env = RuntimeEnv::new();
    let src =
        "K: PositiveDefinite Matrix[2, 2] N/m = [[3000 N/m, -1000 N/m], [-1000 N/m, 2000 N/m]]\n\
               K⁻¹ @ [[1 N, 0 N], [0 N, 5 N]]";
    let x = run_checked(&mut env, sky_types::Env::new(), src).unwrap();
    // K⁻¹ = 1/5e6 * [[2000, 1000], [1000, 3000]]
    close(&x, &[4e-4, 1e-3, 2e-4, 3e-3], 1e-15);
    assert!(x.to_string().ends_with(" m]]"), "{x}");
}

#[test]
fn the_rewrite_is_honoured_by_span() {
    // The checker is told `K` is PositiveDefinite; the value it gets at run
    // time is not. Honouring the rewrite means the Cholesky factorization
    // runs, and fails on that value; without the rewrite the interpreter
    // just inverts and multiplies.
    let src = "K⁻¹ @ [1, 1]";
    let mut env = RuntimeEnv::new();
    env.set("K", run("[[1, 2], [2, 1]]").unwrap());
    let e = run_checked(&mut env, pd_types("K", 2), src).unwrap_err();
    assert_eq!(e.message, "the matrix is not positive definite");
    assert_eq!(&src[e.span.range()], "K");
    // No rewrite recorded: the general path.
    let x = run_in(&mut env, src).unwrap();
    close(&x, &[1.0 / 3.0, 1.0 / 3.0], 1e-15);
    // `inv(K) @ b` and a parenthesized `(K⁻¹) @ b` are rewritten too.
    for src in ["inv(K) @ [1, 1]", "(K⁻¹) @ [1, 1]"] {
        let e = run_checked(&mut env, pd_types("K", 2), src).unwrap_err();
        assert_eq!(e.message, "the matrix is not positive definite", "{src}");
    }
    // Not symmetric: also refused, with its own message.
    env.set("K", run("[[2, 1], [0, 2]]").unwrap());
    let e = run_checked(&mut env, pd_types("K", 2), "K⁻¹ @ [1, 1]").unwrap_err();
    assert_eq!(e.message, "the matrix is not symmetric");
}

#[test]
fn a_rewrite_only_applies_at_its_own_span() {
    // Two `K⁻¹ @ b`, one on a Symmetric matrix the checker does not
    // rewrite: it must not be solved by Cholesky.
    let mut types = pd_types("P", 2);
    types
        .bind_array(
            "S",
            sky_types::Shape::matrix(2u64, 2u64),
            sky_types::TagSet::of(sky_types::Tag::Symmetric),
            sky_units::Dimension::DIMENSIONLESS,
        )
        .unwrap();
    let mut env = RuntimeEnv::new();
    env.set("P", run("[[2, 1], [1, 2]]").unwrap());
    // Symmetric but indefinite: fine for the general path.
    env.set("S", run("[[1, 2], [2, 1]]").unwrap());
    let x = run_checked(&mut env, types, "S⁻¹ @ [3, 3]").unwrap();
    close(&x, &[1.0, 1.0], 1e-15);
}

#[test]
fn indexing() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "v = [10 m, 20 m, 30 m]").unwrap();
    run_in(&mut env, "A = [[1, 2, 3], [4, 5, 6]]").unwrap();
    fn at(env: &mut RuntimeEnv, src: &str) -> String {
        run_in(env, src).unwrap().to_string()
    }
    assert_eq!(at(&mut env, "v[0]"), "10 m");
    assert_eq!(at(&mut env, "v[2]"), "30 m");
    assert_eq!(at(&mut env, "A[1, 2]"), "6");
    assert_eq!(at(&mut env, "A[0, :]"), "[1, 2, 3]");
    assert_eq!(at(&mut env, "A[:, 1]"), "[2, 5]");
    assert_eq!(at(&mut env, "A[1]"), "[4, 5, 6]");
    assert_eq!(at(&mut env, "v[1:]"), "[20 m, 30 m]");
    assert_eq!(at(&mut env, "v[:2]"), "[10 m, 20 m]");
    assert_eq!(at(&mut env, "v[0:3]"), "[10 m, 20 m, 30 m]");
    assert_eq!(at(&mut env, "A[0:2, 1:3]"), "[[2, 3], [5, 6]]");
    assert_eq!(at(&mut env, "A[0:1, 1:3]"), "[[2, 3]]");
    assert_eq!(at(&mut env, "A[1, 1:]"), "[5, 6]");
    // Computed indices.
    run_in(&mut env, "i = 1").unwrap();
    assert_eq!(at(&mut env, "v[i + 1]"), "30 m");
    assert_eq!(at(&mut env, "[7, 8, 9][i]"), "8");
}

#[test]
fn index_errors() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "v = [1, 2, 3]").unwrap();
    run_in(&mut env, "A = [[1, 2], [3, 4]]").unwrap();
    run_in(&mut env, "i = 5").unwrap();
    let mut fail = |src: &str| run_in(&mut env, src).unwrap_err();
    assert_eq!(
        fail("v[i]").message,
        "index 5 is out of bounds for axis 0 with length 3"
    );
    assert_eq!(
        fail("A[0, i]").message,
        "index 5 is out of bounds for axis 1 with length 2"
    );
    assert_eq!(fail("v[-1]").message, "an index cannot be negative");
    assert_eq!(
        fail("v[1.5]").message,
        "an index must be an integer, found 1.5"
    );
    assert_eq!(fail("v[1 m]").message, "an index must be dimensionless");
    assert_eq!(
        fail("v[0, 0]").message,
        "too many indices: Vector[3] has 1 axis, found 2 indices"
    );
    assert_eq!(fail("v[2:1]").message, "this slice is empty");
    assert_eq!(
        fail("v[0:i]").message,
        "slice 0..5 is out of bounds for axis 0 with length 3"
    );
    assert_eq!(
        fail("v[true]").message,
        "an index must be a number, found a Bool"
    );
    let e = fail("v[i]");
    assert_eq!(e.span.len(), 1); // on the index itself
}

#[test]
fn builtins() {
    assert_eq!(show("zeros(3)"), "[0, 0, 0]");
    assert_eq!(show("zeros(2, 3)"), "[[0, 0, 0], [0, 0, 0]]");
    assert_eq!(show("identity(2)"), "[[1, 0], [0, 1]]");
    assert_eq!(show("len([1, 2, 3])"), "3");
    assert_eq!(show("len(zeros(4, 2))"), "4");
    assert_eq!(show("3 m * zeros(2)"), "[0 m, 0 m]");
    assert_eq!(show("identity(2) @ [5 m, 6 m]"), "[5 m, 6 m]");
    assert_eq!(show("[1, 1] + zeros(2)"), "[1, 1]");
    assert_eq!(
        err("zeros(0)"),
        "`zeros` needs a size, a positive integer, found 0"
    );
    assert_eq!(
        err("zeros(2.5)"),
        "`zeros` needs a size, a positive integer, found 2.5"
    );
    assert_eq!(err("zeros(1 m)"), "`zeros` needs a size, found length");
    assert_eq!(
        err("identity(1, 2)"),
        "`identity` takes 1 argument, found 2"
    );
    assert_eq!(
        err("zeros(1, 2, 3)"),
        "`zeros` takes 1 or 2 arguments, found 3"
    );
    assert_eq!(
        err("len(1)"),
        "`len` needs a vector or a matrix, found a number"
    );
    assert_eq!(
        err("zeros(100000, 100000)"),
        "the array from `zeros` is too large"
    );
    assert_eq!(
        err("identity(100000)"),
        "the array from `identity` is too large"
    );
    // A user's own name wins, as for `print`.
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "zeros = 5").unwrap();
    assert_eq!(run_in(&mut env, "zeros").unwrap().to_string(), "5");
}

#[test]
fn comparing_arrays_is_an_error() {
    assert!(err("[1, 2] == [1, 2]").starts_with("cannot compare"));
    assert!(err("[1, 2] < [1, 2]").starts_with("cannot order"));
}

#[test]
fn the_spring_example() {
    // Two springs in series: K u = f, so u = K⁻¹ f.
    let mut env = RuntimeEnv::new();
    let src =
        "K: PositiveDefinite Matrix[2, 2] N/m = [[3000 N/m, -1000 N/m], [-1000 N/m, 1000 N/m]]\n\
               f: Vector[2] N = [0 N, 100 N]\n\
               K⁻¹ @ f";
    let u = run_checked(&mut env, sky_types::Env::new(), src).unwrap();
    // K⁻¹ = 1/2e6 * [[1000, 1000], [1000, 3000]], so u = [0.05, 0.15] m.
    close(&u, &[0.05, 0.15], 1e-15);
    assert!(u.to_string().ends_with(" m]"), "{u}");
}

// ---------------------------------------------------------------------------
// Reproducible arithmetic (spec "Arrays: reproducible arithmetic").

/// The dot product in the spec's order: left to right below 8 terms,
/// otherwise 8 interleaved partial sums and the fixed tree.
fn spec_dot(a: &[f64], b: &[f64]) -> f64 {
    let p: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    if p.len() < 8 {
        return p.iter().fold(0.0, |s, x| s + x);
    }
    let mut s = [0.0_f64; 8];
    for (i, x) in p.iter().enumerate() {
        s[i % 8] += x;
    }
    ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]))
}

fn literal(xs: &[f64]) -> String {
    let items: Vec<String> = xs.iter().map(|x| format!("{x:e}")).collect();
    format!("[{}]", items.join(", "))
}

#[test]
fn dots_sum_in_the_fixed_order() {
    // Sums that round differently left to right and by the tree.
    let a = [1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5];
    let b = [1.0; 9];
    let tree = spec_dot(&a, &b);
    let sequential = a.iter().fold(0.0, |s, x| s + x);
    assert_ne!(
        tree, sequential,
        "the test input must tell the orders apart"
    );
    let (la, lb) = (literal(&a), literal(&b));
    assert_eq!(data(&run(&format!("dot({la}, {lb})")).unwrap()), [tree]);
    // `a @ b` of two vectors is the same reduction, so `dot(a, b)` and
    // `a @ b` agree to the bit.
    assert_eq!(data(&run(&format!("{la} @ {lb}")).unwrap()), [tree]);
    // `norm` sums the squares in the same order.
    let squares: Vec<f64> = a.iter().map(|x| x * x).collect();
    let want = spec_dot(&[1.0; 9], &squares).sqrt();
    let got = data(&run(&format!("norm({la})")).unwrap())[0];
    assert!(
        (got - want).abs() <= 1e-15 * want || got.to_bits() == want.to_bits(),
        "{got} vs {want}"
    );
    // Fewer than 8 terms: left to right, as before.
    let c = [0.1, 0.2, 0.3];
    let sequential = c.iter().fold(0.0, |s, x| s + x * 1.0);
    assert_eq!(
        data(&run(&format!("dot({}, [1, 1, 1])", literal(&c))).unwrap()),
        [sequential]
    );
}

#[test]
fn a_sum_of_negative_zeros_is_zero() {
    assert_eq!(data(&run("dot([0, 0], [-1, -1])").unwrap())[0].to_bits(), 0);
    assert_eq!(data(&run("[0, 0] @ [-1, -1]").unwrap())[0].to_bits(), 0);
}

#[test]
fn matrix_products_keep_the_naive_order_at_every_size() {
    use crate::linalg;
    // Big enough for blocking, packing and more than one thread.
    let (m, k, n) = (140, 100, 150);
    let a: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7919 + 13) % 2003) as f64 / 1000.0 - 1.0)
        .collect();
    let b: Vec<f64> = (0..k * n)
        .map(|i| ((i * 104_729 + 7) % 1999) as f64 / 1000.0 - 1.0)
        .collect();
    let mut naive = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            naive[i * n + j] = s;
        }
    }
    let got = linalg::matmul(&a, m, k, &b, n);
    assert!(got
        .iter()
        .zip(&naive)
        .all(|(g, w)| g.to_bits() == w.to_bits()));
}

#[test]
fn products_and_quotients_of_arrays_are_entry_by_entry() {
    assert_eq!(show("[1, 2, 3] * [4, 5, 6]"), "[4, 10, 18]");
    assert_eq!(show("[1 m, 2 m] * [3 s, 4 s]"), "[3 m*s, 8 m*s]");
    assert_eq!(show("[6 m, 8 m] / [2 s, 4 s]"), "[3 m/s, 2 m/s]");
    assert_eq!(
        show("[[1, 2], [3, 4]] * [[5, 6], [7, 8]]"),
        "[[5, 12], [21, 32]]"
    );
    assert_eq!(
        show("[[1, 2], [3, 4]] / [[1, 2], [3, 4]]"),
        "[[1, 1], [1, 1]]"
    );
    // `@` is still the matrix product.
    assert_eq!(
        show("[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]"),
        "[[19, 22], [43, 50]]"
    );
    // A length-1 vector is not a number: the shapes must agree.
    assert_eq!(
        err("[[1, 2, 3], [4, 5, 6]] * [[1, 2], [3, 4], [5, 6]]"),
        "cannot apply `*` to Matrix[2,3] and Matrix[3,2]"
    );
}

#[test]
fn entry_errors_are_the_scalar_errors() {
    assert_eq!(err("[1, 2] / [1, 0]"), "division by zero");
    assert_eq!(err("[1e200, 1] * [1e200, 1]"), "arithmetic overflow");
    assert_eq!(err("[1e300, 1] / [1e-300, 1]"), "arithmetic overflow");
    // The lowest failing entry decides: an overflow at entry 0 comes before a
    // zero divisor at entry 1, and the other way round.
    assert_eq!(err("[1e300, 1] / [1e-300, 0]"), "arithmetic overflow");
    assert_eq!(err("[1, 1e300] / [0, 1e-300]"), "division by zero");
}

#[test]
fn math_functions_apply_to_every_entry() {
    assert_eq!(show("sqrt([4 m^2, 9 m^2])"), "[2 m, 3 m]");
    assert_eq!(show("sqrt([[1, 4], [9, 16]])"), "[[1, 2], [3, 4]]");
    assert_eq!(show("abs([-1 m, 2 m])"), "[1 m, 2 m]");
    assert_eq!(show("floor([1.5, -1.5])"), "[1, -2]");
    assert_eq!(show("max([1 m, 5 m], [3 m, 2 m])"), "[3 m, 5 m]");
    // A number is every entry.
    assert_eq!(show("min([1 m, 5 m], 3 m)"), "[1 m, 3 m]");
    assert_eq!(show("min(3 m, [1 m, 5 m])"), "[1 m, 3 m]");
    assert_eq!(show("hypot([3 m, 5 m], [4 m, 12 m])"), "[5 m, 13 m]");
    assert_eq!(show("sign([-2 m, 0 m, 3 m])"), "[-1, 0, 1]");
    assert_eq!(show("exp([0, 0])"), "[1, 1]");
    // Each entry is the scalar call, bit for bit.
    let v = run("atan2([1, 2, 3], [4, 5, 6])").expect("runs");
    let want: Vec<f64> = [(1.0f64, 4.0f64), (2.0, 5.0), (3.0, 6.0)]
        .iter()
        .map(|&(y, x)| y.atan2(x))
        .collect();
    assert_eq!(data(&v), want);
}

#[test]
fn math_errors_are_the_scalar_errors() {
    assert_eq!(
        err("sqrt([1, -1])"),
        "`sqrt` of a negative number has no real value"
    );
    assert_eq!(err("exp([1, 1000])"), "arithmetic overflow");
    assert_eq!(err("log([1, 0])"), "`log` of zero has no real value");
    // The lowest failing entry decides the wording.
    assert_eq!(
        err("log([-1, 0])"),
        "`log` of a negative number has no real value"
    );
    assert_eq!(
        err("atan2([1, 2], [1, 2, 3])"),
        "cannot apply `atan2` to Vector[2] and Vector[3]"
    );
    assert_eq!(
        err("exp([1 m])"),
        "the argument of `exp` must be dimensionless, found length"
    );
}
