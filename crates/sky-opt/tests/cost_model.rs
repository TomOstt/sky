//! The static cost model (`sky_ir::cost`) end to end on Sky source: each
//! rule of the model in the module documentation, checked as the polynomial
//! it produces. The model reads the optimized program (spec "Performance
//! contracts"), so these run the pipeline `sky check` uses for them first:
//! the default one without `specialize`, which would replace a function by
//! its clones for particular calls.

use sky_ir::cost::{analyze, Amount, ReasonKind};

/// The cost of function `name` of `src`, as `sky check` computes it.
fn cost(src: &str, name: &str) -> sky_ir::cost::FunctionCost {
    sky_ast::with_deep_stack(|| {
        let mut ir = sky_lower::lower_source(src).expect("lowers");
        sky_opt::default_passes_without_specialize(sky_ir::PassManager::new())
            .verify_each(true)
            .run(&mut ir)
            .expect("optimizes");
        analyze(&ir).get(name).expect("function").clone()
    })
}

/// An exact amount as text.
fn exact(a: &Amount) -> String {
    assert!(a.unbounded.is_empty(), "unbounded: {:?}", a.unbounded);
    assert_eq!(a.lo, a.hi, "not exact");
    a.hi.to_string()
}

#[test]
fn matmul_is_two_n_k_m_flops() {
    let c = cost(
        "def mm(A: Matrix[N, K], B: Matrix[K, M]) -> Matrix[N, M]:\n    return A @ B\n",
        "mm",
    );
    assert_eq!(exact(&c.cost.flops), "2·K·M·N");
    assert_eq!(exact(&c.cost.bytes_read), "8·K·M + 8·K·N");
    assert_eq!(exact(&c.cost.bytes_written), "8·M·N");
    assert_eq!(exact(&c.cost.allocations), "1");
    assert_eq!(exact(&c.cost.allocated_bytes), "8·M·N");
}

#[test]
fn a_matrix_times_a_vector_is_two_n_k() {
    let c = cost(
        "def mv(A: Matrix[N, K], x: Vector[K]) -> Vector[N]:\n    return A @ x\n",
        "mv",
    );
    assert_eq!(exact(&c.cost.flops), "2·K·N");
}

#[test]
fn the_cholesky_solve_is_n_cubed_over_three_plus_two_n_squared() {
    let c = cost(
        "def s(A: PositiveDefinite Matrix[N, N], b: Vector[N]) -> Vector[N]:\n    return A⁻¹ @ b\n",
        "s",
    );
    assert_eq!(exact(&c.cost.flops), "N³/3 + 2·N²");
    // The result and the factor.
    assert_eq!(exact(&c.cost.allocations), "2");
}

#[test]
fn an_inverse_is_two_n_cubed() {
    let c = cost(
        "def i(A: Matrix[N, N]) -> Matrix[N, N]:\n    return inv(A)\n",
        "i",
    );
    assert_eq!(exact(&c.cost.flops), "2·N³");
}

#[test]
fn elementwise_operations_are_n_per_operation() {
    let c = cost(
        "def e(v: Vector[N], w: Vector[N]) -> Vector[N]:\n    return v - w\n",
        "e",
    );
    assert_eq!(exact(&c.cost.flops), "N");
    assert_eq!(exact(&c.cost.bytes_read), "16·N");
    assert_eq!(exact(&c.cost.bytes_written), "8·N");
}

#[test]
fn a_fused_kernel_counts_its_body_per_element_in_one_allocation() {
    let c = cost(
        "def step(v: Vector[N], w: Vector[N]) -> Vector[N]:\n    return v + 2 * w\n",
        "step",
    );
    assert_eq!(exact(&c.cost.flops), "2·N");
    assert_eq!(exact(&c.cost.allocations), "1");
    assert_eq!(exact(&c.cost.allocated_bytes), "8·N");
}

#[test]
fn scalar_operations_have_weights() {
    let c = cost("def w(x: m/m) -> m/m:\n    return sqrt(x * x + x)\n", "w");
    // mul, add, sqrt.
    assert_eq!(exact(&c.cost.flops), "3");
    assert!(c.cost.allocations.is_zero() && c.cost.bytes_read.is_zero());
    // A transcendental function is 20.
    let c = cost("def u(x: m/m) -> m/m:\n    return exp(x)\n", "u");
    assert_eq!(exact(&c.cost.flops), "20");
    // Whole powers: square-and-multiply.
    for (e, flops) in [(2, "1"), (3, "2"), (4, "2"), (5, "3"), (8, "3")] {
        let c = cost(&format!("def p(x: m/m) -> m/m:\n    return x^{e}\n"), "p");
        assert_eq!(exact(&c.cost.flops), flops, "x^{e}");
    }
}

#[test]
fn a_counted_loop_multiplies() {
    let c = cost(
        "def f(x: m/m) -> m/m:\n    for i in range(10):\n        x = x * 2\n    return x\n",
        "f",
    );
    // The index is bookkeeping: only the multiplication counts.
    assert_eq!(exact(&c.cost.flops), "10");
    let c = cost(
        "def g(v: Vector[N]) -> Vector[N]:\n    for i in range(len(v)):\n        v = v * 2\n    return v\n",
        "g",
    );
    assert_eq!(exact(&c.cost.flops), "N²");
    assert_eq!(exact(&c.cost.allocations), "N");
    // A start and a step.
    let c = cost(
        "def h(x: m/m) -> m/m:\n    for i in range(2, 12, 3):\n        x = x * 2\n    return x\n",
        "h",
    );
    assert_eq!(exact(&c.cost.flops), "4");
    let c = cost(
        "def k(v: Vector[N]) -> Vector[N]:\n    for i in range(1, len(v)):\n        v = v * 2\n    return v\n",
        "k",
    );
    assert_eq!(exact(&c.cost.flops), "N² - N");
    // A loop that never runs costs nothing.
    let c = cost(
        "def z(x: m/m) -> m/m:\n    for i in range(5, 2):\n        x = x * 2\n    return x\n",
        "z",
    );
    assert!(c.cost.flops.is_zero());
}

#[test]
fn a_while_with_a_counter_is_counted_and_a_break_bounds_it() {
    let c = cost(
        "def f(x: m/m) -> m/m:\n    i = 0\n    while i < 4:\n        x = x * 2\n        i += 1\n    return x\n",
        "f",
    );
    assert_eq!(exact(&c.cost.flops), "4");
    let c = cost(
        "def g(v: Vector[N]) -> Vector[N]:\n    for i in range(len(v)):\n        if v[0] > 5:\n            break\n        v = v * 2\n    return v\n",
        "g",
    );
    assert_eq!(c.cost.flops.hi.to_string(), "N²");
    assert!(c.cost.flops.lo.is_zero());
}

#[test]
fn any_other_loop_is_unbounded_and_says_where() {
    let c = cost(
        "def f(v: Vector[N]) -> Vector[N]:\n    while v[0] > 1:\n        v = v * 2\n    return v\n",
        "f",
    );
    assert!(c.cost.flops.is_unbounded());
    let r = &c.cost.flops.unbounded[0];
    assert_eq!(r.kind, ReasonKind::Loop);
    assert_eq!(r.function, "f");
    // A loop whose body costs nothing is not unbounded.
    let c = cost(
        "def g(x: m/m) -> m/m:\n    while x > 1:\n        x = 0\n    return x\n",
        "g",
    );
    assert!(c.cost.flops.is_zero());
}

#[test]
fn an_if_costs_the_dearer_branch_and_the_cheaper_is_the_lower_bound() {
    let c = cost(
        "def f(v: Vector[N], w: Vector[N]) -> Vector[N]:\n    r = v\n    if v[0] > 1:\n        r = v + w\n    return r\n",
        "f",
    );
    assert_eq!(c.cost.flops.hi.to_string(), "N");
    assert!(c.cost.flops.lo.is_zero());
    assert!(c.cost.flops.exact_poly().is_none());
}

#[test]
fn calls_compose_with_the_sizes_of_the_arguments() {
    let src = "def dbl(v: Vector[N]) -> Vector[N]:\n    return v * 2\n\ndef twice(w: Vector[M]) -> Vector[M]:\n    return dbl(dbl(w))\n\ndef three() -> Vector[3]:\n    return dbl([1, 2, 3])\n";
    assert_eq!(exact(&cost(src, "twice").cost.flops), "2·M");
    assert_eq!(exact(&cost(src, "three").cost.flops), "3");
}

#[test]
fn recursion_is_unbounded() {
    let c = cost("def f(v: Vector[N]) -> Vector[N]:\n    return f(v)\n", "f");
    assert!(matches!(
        c.cost.flops.unbounded[0].kind,
        ReasonKind::Recursion { .. }
    ));
}

#[test]
fn a_value_of_unknown_shape_is_unbounded() {
    let c = cost("def f(x, y):\n    return x * y\n", "f");
    assert!(c.cost.flops.is_unbounded());
    assert_eq!(c.cost.flops.unbounded[0].kind, ReasonKind::UnknownShape);
}

#[test]
fn array_operations_allocate_and_move_bytes() {
    let c = cost(
        "def t(A: Matrix[N, M]) -> Matrix[M, N]:\n    return Aᵀ\n",
        "t",
    );
    assert_eq!(exact(&c.cost.bytes_read), "8·M·N");
    assert_eq!(exact(&c.cost.allocations), "1");
    let c = cost("def a() -> Vector[3]:\n    return [1 m, 2 m, 3 m]\n", "a");
    assert_eq!(exact(&c.cost.bytes_written), "24");
    assert!(c.cost.flops.is_zero());
    let c = cost("def z() -> Matrix[2, 3]:\n    return zeros(2, 3)\n", "z");
    assert_eq!(exact(&c.cost.allocated_bytes), "48");
}

#[test]
fn every_site_of_a_loop_carries_its_multiplier() {
    let c = cost(
        "def g(v: Vector[N]) -> Vector[N]:\n    for i in range(len(v)):\n        v = v * 2\n    return v\n",
        "g",
    );
    let mul = c
        .sites
        .iter()
        .find(|s| s.what == "mul")
        .expect("the multiplication is a site");
    assert_eq!(exact(&mul.cost.flops), "N²");
}
