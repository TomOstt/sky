# Automatic differentiation on SIR: design

Status: forward mode (scalar and array values) and reverse mode for straight-line code, `if` and calls are implemented in `crates/sky-ad`; reverse mode through `while` is design only, and a loop falls back to forward mode. Roadmap ideas 32 (differentiable programming) and 33 (adjoints). Date: 2026-09-29, updated for arrays.

## Why a program transformation on SIR

AD on SIR sees what an operator-overloading library or a tree-walker cannot: the dimension of every value, structured control flow that is not hidden in jumps, and pure operations with known effects. The result is SIR again, so the verifier, the optimizer (CSE removes the duplicated primal work) and every backend treat a derivative like any other function. Units come for free and are checked: the tangent of `v` with respect to `x` has dimension `dim(v)/dim(x)`, so d(position)/d(time) is a velocity and a rule that gets a dimension wrong produces ill-typed IR that the verifier rejects.

## Forward mode (implemented)

`derivative(m, f, wrt)` builds `df/dx(params...) -> (value, ∂f/∂x)`; a function with several results returns its values and then their tangents, so a derivative function (which has two results) can itself be differentiated and higher derivatives work. Each value `v` gets a tangent `tv = dv/dx` emitted next to it (a dual number spelled out as ops). Tangents that are structurally zero are `None` and cost nothing. An activity analysis (a fixpoint over the function) finds which values depend on `x`, so loop counters and flags carry no tangents.

* `if`: both arms are differentiated and yield tangents; the condition does not flow.
* `while`: a tangent slot beside each active loop-carried value, threaded through `before`, `condition`, `after`, `yield` and the results. No tape is needed: the tangent is computed in the same iteration as the value.
* `call`: the callee is differentiated into `d<callee>/d<param>` and called; recursion works because the derivative calls itself. Derivatives are memoized by name.

* Arrays: the tangent of an array value is an array of the same shape with dimension `dim(value)/dim(x)`, and every rule in `rules.rs` is written on values that may be arrays. `add`/`sub`/`neg` and scaling by a scalar are linear; `mul`/`div` by a scalar follow the product and quotient rules with an array-valued tangent; `matmul` is `ta @ b + a @ tb`; `transpose` transposes the tangent; `norm(a)` is `(a·ta)/r` (summed over all entries, with `r = ‖a‖`; a zero norm fails with the division by zero of `sqrt`'s rule at 0); `dot(a, b)` is `dot(ta, b) + dot(a, tb)`; `inverse` of a matrix is `-(r ta r)` (with `r = A⁻¹`), scalar `inverse` unchanged; `solve` is `A⁻¹ (tb - tA r)`, a solve with the same `A`; `array.literal` gathers the entries' tangents (a zero for an entry that does not depend on `x`); `index` and `slice` apply to the tangent; `zeros`, `identity` and `len` have no tangent (integers).
* `fused` kernels are differentiated without unfusing: the tangent is a second `fused` op over the same operands and the operands' tangents, whose body is the primal body with the scalar tangent rules applied elementwise (a scalar operand that is broadcast stays a scalar operand). Unfusing first is not always possible (the body may mix a broadcast scalar with `add`, which has no array form) and would throw away what the optimizer did; the elementwise rule is the same rule, one level down.
* `mul` and `div` of two arrays of one shape, and the math functions of arrays, are elementwise SIR operations, and the scalar rules apply entry by entry. A rule states the dimension it knows and the shape follows the operands (`Diff::emit` gives an elementwise result the array shape of its operands, and spreads a number over an array as `s * ones` before an `add` or `sub`, which take equal shapes only). In reverse mode the adjoint of an elementwise function is `r̄ ⊙ f'(a)`, made by running the tangent rule with the tangent of `a` set to `r̄`; a number operand spread over the array is seeded with ones and its contributions are summed. `abs` of an array (and inside a kernel body, where no `if` may go) uses `s + (1 - s*s)` for `s = sign(a)`, which is exactly the `sign(0) = 1` slope. `min` and `max` of arrays need a choice per entry that cannot be made exactly without an `if` (a difference of two finite entries can overflow), so their derivative is refused with `AdError::Unsupported`, never invalid SIR.
* `call`: a call that passes or returns an array is **inlined** before differentiating (`inline.rs`; the callee's body is spliced in, recursion and depth 16 excepted). The chain rule through a call is a product of the callee's partial derivatives with the argument tangents, which for an array argument is a Jacobian that SIR cannot write; inlining makes the derivative flow through the callee like any other code, so a helper `displacement(p)` in a potential is part of the potential. Calls with scalar operands and results stay calls (their callee may loop).

Cost: one pass over the program, about 2 to 3 times the work of `f`, per input parameter. That is the right trade for one input and many outputs, or a handful of inputs. For an *array* input `p` with `n` entries it is `n` passes (below: reverse mode is one).

## In the language

`derivative(f, x)(args)` (spec v0.1, "Derivatives") is the user-facing form. `f` is a `def`, `x` names one of its parameters, and the call passes `f`'s arguments; the checker types it `dim(f) / dim(x)` from the annotations, so a velocity added to a length is a unit error before anything runs. Both engines run the derivative that this crate builds: lowering emits a `call` of `df/dx` and keeps its second result, and `sky run` (the tree walker, which has no AD) lowers the program, builds the same function and runs it on the SIR interpreter (`sky-derive`). There is one set of rules, and a differential test compares the printed bytes. Higher derivatives are the derivative of a `def` that calls the first.

`grad(f)(args)` is the same machinery once per parameter: the arguments are evaluated once, `df/dxᵢ` is called for each parameter, and the second results are gathered in a `Vector[n]` (`array.literal`). It exists when every parameter has the same dimension, so the vector has one element unit, `dim(f)/dim(x)`; with mixed dimensions the checker refuses and points at `derivative(f, x)` per parameter. A function of one array parameter has one gradient, an array of the parameter's shape (a force for a potential of a position vector), which is `derivative(f, p)(args)` for that parameter: `d<f>/d<p>(params...) -> (value, gradient)`, computed by reverse mode (below). `derivative(f, t)` of an array-valued `f` with a scalar `t` is forward mode with array tangents (a velocity vector). `(∂f/∂x)(args)` is sugar the checker, interpreter and lowering read as `derivative(f, x)(args)`.

## Why forward mode first

1. It needs no storage of intermediate values, so it needs no answer to the hard question of reverse mode (see below), and it is correct on every structured construct on day one.
2. It is a local, compositional rewrite per op. It gives the rule table (`rules.rs`) that reverse mode reuses: an adjoint rule is the transpose of the tangent rule.
3. It is the oracle for reverse mode. `∇f · v` from reverse mode must equal the forward tangent in direction `v`, which makes property tests for the hard part cheap.
4. The typical Sky programs (a physical model differentiated with respect to a few parameters, sensitivity of one output) are the forward-mode sweet spot. The gradient of a scalar loss with respect to many parameters (fitting, optimization, the force from a potential of many coordinates) is the reverse-mode case, which now exists for array parameters.

## Reverse mode (implemented for straight-line code, `if` and calls)

`crates/sky-ad/src/reverse.rs`. `derivative_functions` builds the gradient of a scalar function with respect to an **array** parameter with it, and falls back to forward mode (`gradient.rs`) when it cannot. A scalar parameter always uses forward mode: for one input reverse mode has no advantage, and forward mode handles loops.

What is implemented:

* **Sweeps.** After inlining array calls and running the activity analysis, the forward sweep copies the function (so the value is computed exactly once and in the original order), and the backward sweep walks the operations in reverse seeding `F̄ = 1` and accumulating adjoints with `+`. Adjoints are typed `dim(F)/dim(v)` with the shape of `v`. The result is `(F, ∂F/∂p)`, in one function `d<f>/d<p>`.
* **Adjoint rules** are the transposes of the tangent rules: `matmul` (`Ā = r̄ Bᵀ`, `B̄ = Aᵀ r̄`, and for a matrix-vector product the outer product `r̄ xᵀ`), `transpose`, `norm` (`ā = (r̄/r) a`), `dot` (`ā = r̄ b`, `b̄ = r̄ a`), `inverse` (`Ā = -rᵀ r̄ rᵀ`), `solve` (`b̄ = solve(A, r̄)` since `A` is symmetric, `Ā = -b̄ rᵀ`), `array.literal` (entries of `r̄`), `index` and `slice` (a scatter, below), `fused` (per operand, a kernel `r̄ ⊙ ∂body/∂arg`, itself made with the forward rules), scaling (a scalar factor of an array gets `Σ r̄ ⊙ array`), and the scalar operations. A scalar rule that is a function of one operand (`pow`, the math functions, scalar `inverse`) reuses its forward rule: the partial `∂r/∂a` is the tangent rule run with `ta = 1` and `dim(x) = dim(a)`.
* **SIR has no scatter, no outer product, no sum and no reshape**, so these are built from what it has, with no new operation: the sum of a vector is `v @ ones(n)` (and of a matrix `ones(m) @ A @ ones(n)`, with `ones(n)` a `fused` add over `zeros(n)`, so a run-time length works); scattering the adjoint of `A[lo:hi]` back is `identity(n)[:, lo:hi] @ r̄` (a selection matrix), and the adjoint of `A[i]` is row `i` of `identity(n)` scaled; the adjoint of a slice or index of a matrix is `Sᵣ R̄ S_cᵀ` with two selection matrices; an outer product `u vᵀ` is `(n×1) @ (1×m)`, and a vector has no reshape, so both factors are gathered with `array.literal` from the entries, which needs the sizes known at compile time (`Vector[3]`, not `Vector[N]`) and at most 4096 entries in the product. Where a size is needed and unknown, reverse mode reports "not supported" and the forward fallback is tried (which has the same need).
* **`if`**: the forward sweep copies the `if`; the backward sweep emits an `if` on the same condition whose arms **replay** the forward arm (its values were defined inside the forward `if`, out of scope), sweep it backwards seeded with the adjoints of the `if`'s results, and `yield` one adjoint contribution per outside value the arm uses (a zero where an arm does not use it). The replay costs a second evaluation of an arm. An arm that prints (directly, or through a call whose callee prints) is refused so that nothing prints twice, and the gradient falls back to forward mode.
* **`call`**: after inlining, the calls that remain have scalar operands and results (the callee may loop) or are recursive. The forward sweep calls the callee's *forward* derivative `d<callee>/d<param>` for each argument the gradient depends on, and keeps its partial derivatives; the backward sweep multiplies them by the result's adjoint. The first such call also gives the value. A recursive callee with an array argument uses the callee's gradient function, `d<callee>/d<param>` of kind gradient, memoized by name.
* **`while`** with a result the gradient depends on is **not** handled. A loop the gradient does not depend on is copied.

The fallback is `gradient.rs`: the same function `d<f>/d<p>` with the same signature, whose body calls the directional derivative `d<f>/d<p>#jvp(params..., e)` once for every unit direction `e` (a one-hot array of `p`'s shape, from `identity`) and gathers the `n` partial derivatives in an array. The directional derivative is forward mode with an extra last parameter that seeds the tangent of `p` (dimensionless, of `p`'s shape). It costs `n` evaluations, unrolls the loop over the entries (so it needs `p`'s size, at most 256), and prints `n` times if `f` prints.

Which is used, for the record: a straight-line or `if` function of an array parameter: reverse, one forward and one backward sweep, a small multiple of one evaluation regardless of `n`; a function with a `while` that carries the result: forward, `n` evaluations of about 2 to 3 times the cost each. Measured on a release build, 200 gradients of a potential of a 200-entry vector took 15 ms by reverse mode and 300 ms by the forward fallback (`sky run`, the same function with the computation wrapped in a one-iteration loop), and `benches/programs/gradient.sky` times the reverse-mode gradient of a spring-network potential (about 4 µs per gradient on the tree-walker, including the call into SIR). The test `reverse_mode_cost_does_not_grow_with_n` in `crates/sky-lower/tests/array_ad.rs` pins that the size of the reverse function is the same for a 3-vector and a 40-vector.

### Design notes that still stand for the rest


Reverse mode computes `∂f/∂xᵢ` for all inputs in one backward sweep: run `f` forward, then propagate an *adjoint* `v̄ = ∂f/∂v` from the result to the inputs. The adjoint of `v` has the dimension `dim(f)/dim(v)` (the mirror image of the tangent's `dim(v)/dim(x)`), so the gradient of an energy with respect to a length is a force and the same typing argument holds.

Straight-line code is easy: each op `r = op(a, b)` contributes `ā += r̄ · ∂r/∂a`, in reverse order of the ops. Fan-out (a value used twice) accumulates with `+`. The reverse of SIR's regions is where the design is:

### `if`

The reverse of `if c { A } else { B }` is `if c { reverse(A) } else { reverse(B) }`, so the condition value must be available in the backward sweep. Values used by `reverse(A)` are needed only when that arm ran. Adjoints of values captured from outside accumulate in both arms, and an arm that does not use a captured value contributes a zero.

### `while`: tape versus recomputation

The backward sweep visits the iterations in reverse order and needs each iteration's primal values. Three strategies, in the order we would implement them:

1. **Tape.** The forward pass pushes what the backward pass will need (the values the adjoint rules read, plus the trip count) onto a per-loop stack; the backward loop pops. Memory is `O(trips × live values)`. Simple and fast, and the only strategy needing new IR (a stack value or a preallocated array of trip count length, so it needs arrays or a dynamic-array op).
2. **Recomputation.** Store nothing, and rebuild iteration `k` by re-running the loop from the start to `k`. Memory `O(1)`, time `O(trips²)`. Right only for tiny trip counts, and always correct because SIR ops are pure (an `Io` effect inside makes the loop unfit for it).
3. **Checkpointing** (binomial / revolve). Store the loop-carried state every `s` iterations; in the backward sweep recompute the segment between checkpoints and tape only that segment. With `s = √trips` it is `O(√trips)` memory for one extra forward pass; with a chosen budget of `c` checkpoints the extra work is logarithmic in the trip count (Griewank and Walther). The pass takes a memory budget as a parameter and defaults to the tape below a threshold and checkpointing above it.

Only the loop-carried values need saving, not the intermediates: the backward iteration recomputes the intermediates of one iteration from the carried values, which is `O(1)` in the body. That "save carried values, recompute the body" rule is the default unit of taping. Loops whose body is an affine update need no tape at all (the adjoint is a fixed matrix power), which a later pass can recognise.

Data-dependent trip counts are stored (the tape length), and the backward loop is a `while` over the counter. Bool flags from lowered `break` and `continue` are carried values like any other, so the structure stays single-entry, single-exit.

### `call` and recursion

Each function gets a *reverse* twin: the forward part that records what its backward part needs, and the backward part. For non-recursive callees that is inlining-equivalent, and memoized by name as forward mode is. A recursive function's tape is the call stack; the twin uses the same tape-or-recompute choice, and the `MAX_CALL_DEPTH` limit bounds it.

### Effects and failure

`Fail` ops stay in the forward pass; the backward pass runs only if the forward one succeeded. A backward rule that divides by a value the forward rule did not (`√` at 0) fails there, as forward mode does. `Io` inside a differentiated region is kept in the forward pass only.

## Adjoints for solvers and implicit functions (idea 33)

For `x = A⁻¹ b` (op `solve`), never differentiate the algorithm. The adjoints are `b̄ = A⁻ᵀ x̄` and `Ā = -b̄ xᵀ`, one extra solve that reuses the factorization the forward pass made (a Cholesky factor is kept as a checkpoint), and the matrix-property tags say which transposes are free (`A` symmetric: `A⁻ᵀ = A⁻¹`). The same shape applies to any implicit function `F(x, y) = 0` (a root-finder, a steady-state solve, a fixed point): `ȳ ↦ x̄ = -(∂F/∂x)ᵀ (∂F/∂y)⁻ᵀ ȳ`, evaluated at the converged `y`, with no tape over the solver's iterations. For an ODE, the continuous adjoint integrates a backward equation for `λ` with the same solver, trading the tape for a second integration and a numerical mismatch with the discretized forward pass (the discrete adjoint is the exact gradient of the computed answer). Which to use is a solver-synthesis (idea 12) decision, recorded like any strategy choice so `--explain` can say it.

## Decisions taken in the forward implementation

* Unknown types are rejected with a clear error when the derivative flows through them, never treated dynamically (a dynamic derivative would have no static unit).
* The derivative of an `if` at its discontinuity is the taken arm's derivative; conditions never contribute.
* A general power `a ^ e` differentiates in both `a` and `e` (`r * ln(a) * te`). Every math function has a rule (`math_rules.rs`, table in the header of `rules.rs`); `floor`, `ceil` and `round` have a zero tangent, `abs`, `min` and `max` take the one-sided slope of the arm that runs, and a rule that divides by zero at the edge of a domain (`sqrt` at 0) fails there as the primal does.
* A call with several dependent arguments calls one derivative per argument. A directional-derivative function (all tangents in one call) would avoid that for recursion and is a later optimization.
* Arrays are differentiated (both modes) as described above. A Jacobian (an array-valued function of an array) is not: the tangent of a vector-valued function with respect to a *vector* would be a matrix, and neither mode has a use for it here (forward mode gives one Jacobian-vector product per pass, and there is no user-facing type for the matrix).
* `derivative(f, p)` for an array `p` is the *gradient* of a scalar `f`, not a directional derivative: it has no second argument, the physical reading is the force, and a directional derivative is the dot product of the gradient with the direction.
