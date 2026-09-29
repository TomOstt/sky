# Automatic differentiation on SIR: design

Status: draft. Forward mode is implemented in `crates/sky-ad`; reverse mode and adjoints are design only. Roadmap ideas 32 (differentiable programming) and 33 (adjoints). Date: 2026-09-29.

## Why a program transformation on SIR

AD on SIR sees what an operator-overloading library or a tree-walker cannot: the dimension of every value, structured control flow that is not hidden in jumps, and pure operations with known effects. The result is SIR again, so the verifier, the optimizer (CSE removes the duplicated primal work) and every backend treat a derivative like any other function. Units come for free and are checked: the tangent of `v` with respect to `x` has dimension `dim(v)/dim(x)`, so d(position)/d(time) is a velocity and a rule that gets a dimension wrong produces ill-typed IR that the verifier rejects.

## Forward mode (implemented)

`derivative(m, f, wrt)` builds `df/dx(params...) -> (value, ∂f/∂x)`. Each value `v` gets a tangent `tv = dv/dx` emitted next to it (a dual number spelled out as ops). Tangents that are structurally zero are `None` and cost nothing. An activity analysis (a fixpoint over the function) finds which values depend on `x`, so loop counters and flags carry no tangents.

* `if`: both arms are differentiated and yield tangents; the condition does not flow.
* `while`: a tangent slot beside each active loop-carried value, threaded through `before`, `condition`, `after`, `yield` and the results. No tape is needed: the tangent is computed in the same iteration as the value.
* `call`: the callee is differentiated into `d<callee>/d<param>` and called; recursion works because the derivative calls itself. Derivatives are memoized by name.

Cost: one pass over the program, about 2 to 3 times the work of `f`, per input parameter. That is the right trade for one input and many outputs, or a handful of inputs.

## Why forward mode first

1. It needs no storage of intermediate values, so it needs no answer to the hard question of reverse mode (see below), and it is correct on every structured construct on day one.
2. It is a local, compositional rewrite per op. It gives the rule table (`rules.rs`) that reverse mode reuses: an adjoint rule is the transpose of the tangent rule.
3. It is the oracle for reverse mode. `∇f · v` from reverse mode must equal the forward tangent in direction `v`, which makes property tests for the hard part cheap.
4. The typical Sky programs (a physical model differentiated with respect to a few parameters, sensitivity of one output) are the forward-mode sweet spot. The gradient of a scalar loss with respect to many parameters (fitting, optimization) is the reverse-mode case and comes after arrays run.

## Reverse mode (design)

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
* Arrays are out of scope until the interpreter runs them; the adjoint rules for `matmul`, `transpose` and `solve` go in with array AD.
