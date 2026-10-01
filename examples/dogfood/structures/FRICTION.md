# Friction log: linear systems and structures

Dogfooding Sky as a structural/mechanical engineer: a 1D finite-element bar
(N = 10, 200, 2000), a 2D truss with reactions, the natural frequencies of a
3-DOF spring-mass chain, and performance contracts on solvers. Built from
`origin/main` (b18f4d6); engines `ast` (default), `sir` and `native` were
compared on every program that is committed here, and they print identical
bytes on all of them, apart from the engine errors listed below.

Programs in this directory (each has a `.out`):

| file | what |
|---|---|
| `bar_fem.sky` | 10-element bar, loop assembly, LU vs Cholesky |
| `bar_fem_200.sky` | 200 elements, Cholesky through the `trust_pd` workaround |
| `bar_fem_gram.sky` | 2000 elements assembled as `k Bᵀ B` (50x faster than the loop) |
| `bar_fem_dynamic.sky` | run-time `n` through unannotated functions |
| `bar_fixed_middle.sky` | a clamp in the middle: non-contiguous fixed DOFs |
| `truss2d.sky` | 6-node, 9-bar truss, reactions, bar forces, equilibrium check |
| `modal_chain.sky` | natural frequencies by hand-written inverse/power iteration |
| `solver_costs.sky` | `cost` contracts on solvers |

Severity counts: **2 blocker, 9 major, 12 minor** (23 entries).

---

## Blockers

### 1. No eigen-solver, and almost no dense linear algebra beyond `@`, `⁻¹`, `ᵀ`
- **Severity:** blocker. **Category:** missing feature.
- **Tried:** the first thing a modal analysis does.
  ```sky
  K: PositiveDefinite Matrix[3, 3] N/m = [[2000 N/m, ...]]
  a = eig(K)          b = eigvals(K)      c = diag(K)       d = trace(K)
  e = det(K)          g = cholesky(K)     h = ones(3)       i = sum([1, 2, 3])
  j = outer([1, 2], [3, 4])   k = hstack([1, 2], [3, 4])   l = linspace(0, 1, 5)
  m = cond(K)         s = eye(3)          u = mean([1, 2])
  ```
- **What happened:** every one is `error: cannot find `eig` in this scope` / `... not defined`
  (and `max(K)` is "`max` takes 2 arguments"). Nothing says "this does not exist
  in v0.1", and nothing suggests the closest thing that does exist. Only `zeros`,
  `identity`, `len`, `norm`, `dot`, `transpose`, `inv` and the elementwise math
  exist.
- **Workaround:** `modal_chain.sky`: inverse iteration for mode 1, inverse
  iteration with M-orthogonal deflation for mode 2, power iteration on `M⁻¹K`
  for mode 3, 80-200 fixed iterations, `v @ (K @ v) / (v @ (M @ v))` for the
  Rayleigh quotient. It reproduces the exact frequencies of the equal-mass
  chain to the last printed digit and passes a `trace` identity check for
  unequal masses (the trace itself had to be summed by hand, there is no
  `trace`). That is fine for 3 DOF and unusable for a model with 500 DOF
  and 20 modes (no Lanczos, no shift-and-invert, no generalised problem).
- **Expect:** `eigh(K, M)` for the symmetric generalised problem (values and
  vectors), returning frequencies in the unit `1/s^2` for `N/m` over `kg`;
  `diag`, `trace`, `det`, `cholesky`, `sum`, `outer`, `ones`, `linspace` and
  concatenation at least.

### 2. No sparse or banded matrices; dense arrays stop at 4096 x 4096
- **Severity:** blocker (for anything beyond a toy). **Category:** missing feature / performance.
- **Tried:** the same bar with 4100 elements (a 1D bar!).
  ```sky
  B = identity(4100)[0:4099, :] - identity(4100)[1:4100, :]
  ```
- **What happened:** `error: the array from `identity` is too large ... more than 16777216 elements`
  (the 2^24 cap in the spec). A tridiagonal system, which is O(n) in any FE
  code, is a dense `n x n` matrix here: 2000 DOF means a 32 MB `K`, 4000 DOF
  is the ceiling, and a 3D mesh has 10^5 DOF.
- **Workaround:** none. At 4000 DOF dense LU/Cholesky still runs in about
  1 s total (`bar_fem_gram.sky` style), so the dense kernels are good; it is
  the absence of a banded/sparse type that hurts.
- **Expect:** a `Banded`/`Tridiagonal` tag or a sparse matrix type in the
  same tag lattice as `Diagonal` and `PositiveDefinite`, so that `K⁻¹ @ f`
  on a banded `K` is O(n b^2) and assembly does not need a dense `n x n`.

---

## Major

### 3. Arrays are immutable: there is no scatter-add, so assembly is a matrix product
- **Severity:** major. **Category:** missing feature / ergonomics.
- **Tried:**
  ```sky
  K = zeros(11, 11)
  K[0:2, 0:2] += ke
  K2[0, 0] = 1 N/m
  ```
- **What happened:** `error: cannot assign to an element of `K` ... written with `+=` ... = note: arrays are immutable in v0.1`.
  The message is accurate but gives no way forward; there is no help line.
  The spec calls it an open question ("pending Tom's confirmation").
- **Workaround:** the connectivity-matrix form of assembly, `K = K + Pᵀ @ ke @ P` with
  `P = identity(11)[e:e+2, :]` (a 2 x 11 slice). It is the textbook
  `K = Σ Pₑᵀ kₑ Pₑ`, it type-checks with units, and for the 2D truss
  `(Pᵢ - Pⱼ)ᵀ B (Pᵢ - Pⱼ)` is the whole member contribution (`truss2d.sky`). It
  reads fine once you know it; but it makes every `K[i, j] += v` into an
  O(n²) matrix operation (entry 4).
- **Expect:** either a functional update spelled naturally and compiled in
  place when the old value is dead (`K = K.with(i, j, K[i, j] + v)`, or
  `K[i, j] += v` that desugars to it, using the lifetime analysis Sky
  already has: it reports "reuses `K`'s buffer" on `K = K + ...`), or an
  `assemble(...)`/`scatter_add` primitive. Without it, FE codes are O(n³).

### 4. Loop assembly is O(n³); the fast form is a trick you have to find
- **Severity:** major. **Category:** performance.
- **Tried:** the loop of entry 3 at n = 500, 1000, 2000 elements
  (`K = K + Pᵀ @ ke @ P` over a dense `(n+1) x (n+1)` `K`).
- **What happened (this machine, `--engine=native`):** 0.35 s, 1.7 s, 10.7 s;
  `ast` takes 19 s at n = 2000; the solve is a few percent of that.
  Each iteration allocates/adds an entire matrix (`sky check --explain` says
  the buffer is reused, but the add still touches n² entries; the cost model
  honestly says `cost of assemble: 40763201 flops` for 200 elements).
  The same matrix as one Gram product, `K = k * (Bᵀ @ B)` with the
  difference operator `B = identity(n+1)[0:n, :] - identity(n+1)[1:n+1, :]`,
  gives bit-identical displacements and runs in 0.2 s at n = 2000 and 1.1 s
  at n = 4000 (`bar_fem_gram.sky`): about 50x faster, `Bᵀ @ B` being a symmetric
  rank-k update. That form only exists for uniform stiffness (no `diag(v)`
  to weight elements, no broadcasting of a vector against a matrix, entry 1);
  a variable cross-section needs the loop again.
- **Workaround:** `bar_fem_gram.sky` for uniform bars.
- **Expect:** a `cost` hint or a note when a loop is building a dense matrix
  one outer product at a time ("this loop builds an n x n matrix n times: O(n³)"), and
  a way to write weighted assembly (`diag`, entry 1).

### 5. Array sizes must be literal numbers, even when you named the number
- **Severity:** major. **Category:** missing feature / ergonomics.
- **Tried:**
  ```sky
  n = 4
  K = zeros(n, n)
  ```
- **What happened:** `error: `zeros` needs a constant size, found `n` ... not a constant ... = note: an array's size is part of its type, so it is known before the program runs ... = help: write the number, e.g. `zeros(3)``.
  `n = 4` *is* known before the program runs; the checker evidently only accepts a
  literal token. `ne = 10; zeros(ne + 1, ne + 1)` fails the same way, and so does `identity(ne + 1)`.
- **Workaround:** repeat the literal everywhere (`bar_fem.sky` has `11`, `10` and
  `[1:11, 1:11]` hard-coded, a typo-prone chore when the mesh changes), or
  generate the source with a script (that is how the N = 200 and 2000 files were made), or entry 6.
- **Expect:** constant propagation: any binding whose value is a literal, or an
  arithmetic expression of literals (`n = 10`, `n + 1`, `2 * n`), usable as a
  size (a `const n = 10` form would also do). Better: size parameters
  (`def assemble[n](...)`), since "N from a variable" is the first thing a mesh
  generator does.

### 6. Run-time sizes work only if you give up the type checker
- **Severity:** major. **Category:** spec gap.
- **Tried:** `bar_fem_dynamic.sky`: `def assemble(n, ke): ... zeros(n + 1, n + 1) ...` with
  *unannotated* parameters, called with `10`, `20`, `100`.
- **What happened:** it works, on all three engines, and gives the right
  tip displacement at every n. But the checker now types nothing inside:
  `tip_displacement(10, 20 MN, 2 m, 1 m)` (a length where a force belongs) passes
  `sky check` ("ok") and fails at run time with
  `error: cannot show a quantity of dimension L·M⁻¹·T² in `mm``, on the *caller's* print and
  not at the bad argument. A non-integer size is a run-time error too:
  `` `zeros` needs a size, a positive integer, found 3.5 ``. Sizes, units and tags
  are all unchecked until the line runs, which is the Python behaviour Sky
  exists to avoid.
- **Workaround:** none that keeps the checking; annotate and use literal sizes (entry 5).
- **Expect:** size parameters in the type system (`def assemble(n: Size, ke: Matrix[2, 2] N/m) -> Matrix[n + 1, n + 1] N/m`).
  Dimension arithmetic on sizes is declared out of scope for v0.1, but a
  plain `Size` parameter used in `zeros(n)` and `range(n)` would cover FE use.

### 7. WRONG RESULT: a tag that a loop invalidates is still trusted inside the loop
- **Severity:** major (silently wrong answer). **Category:** wrong result / bug.
- **Tried:**
  ```sky
  Q = identity(2)
  b = [1, 1]
  for i in range(2):
      x = Q⁻¹ @ b
      print(x)
      Q = 2 * Q
  ```
- **What happened:** prints `[1, 1]` then `[2, 2]` (all engines). The second answer
  is `Qᵀ b` with `Q = 2I`; the correct `Q⁻¹ b` is `[0.5, 0.5]`
  (`Q2 = 2 * Q; print(Q2⁻¹ @ b)` outside a loop correctly prints `[0.5, 0.5]`).
  The checker types a loop body once, from the state before the loop (spec,
  "Blocks and control flow"), so the `Orthogonal` tag of `identity(2)` is still
  believed on iteration 2, and `Orthogonal` is the one tag whose promise is not verified at run time.
  `Diagonal` and `PositiveDefinite` are verified, so they become a run-time
  error instead of a wrong number; only `Orthogonal` goes silently wrong.
- **Workaround:** none needed for FE; do not carry an `Orthogonal` matrix through a loop that changes it.
- **Expect:** a variable the body rebinds loses its tags before the body is checked
  (check the body with the join of the entry and the end-of-body types).

### 8. Tags on a function parameter are not checked at the call, and the failure is blamed on the callee
- **Severity:** major. **Category:** wrong result / spec gap.
- **Tried:**
  ```sky
  def solve_pd(K: PositiveDefinite Matrix[N, N] N/m, f: Vector[N] N) -> Vector[N] m:
      return K⁻¹ @ f
  B = [[1 N/m, 5 N/m], [-3 N/m, 2 N/m]]
  print(solve_pd(B, [1 N, 2 N]))
  ```
- **What happened:** `sky check` says `ok`. `sky run` fails at the callee:
  `error: the matrix is not symmetric ... declared PositiveDefinite, which implies symmetric` pointing at `return K⁻¹ @ f`,
  not at the call that passed an untagged matrix. Shape and dimension of arguments *are* checked at the call,
  tags are not (the spec lists "shape, then dimension" and is silent on tags). So
  an annotation `K: PositiveDefinite ...` on a parameter is a promise the
  checker does not hold the caller to.
- **Side effect, used as a workaround:** this is a free cast.
  ```sky
  def trust_pd(K: PositiveDefinite Matrix[N, N] N/m) -> PositiveDefinite Matrix[N, N] N/m:
      return K
  Kr = trust_pd(K[1:201, 1:201])      # `K⁻¹ @ f` is now a Cholesky solve
  ```
  Cholesky is chosen (`--explain` confirms) and the run-time check still verifies symmetry and positive
  pivots, so a wrong promise is an error, never a wrong number (`bar_fem_200.sky`).
- **Expect:** either the call is an error ("`B` is not known to be PositiveDefinite"), or an explicit,
  checked-at-run-time cast with a name (`assume PositiveDefinite K`), which is what `trust_pd` is doing by accident.

### 9. A PositiveDefinite matrix cannot be built from anything but a small literal
- **Severity:** major. **Category:** missing feature.
- **Tried:** every way an FE matrix becomes positive definite:
  ```sky
  S2: PositiveDefinite Matrix[2,2] = P + Q            # P, Q both PositiveDefinite
  S3: PositiveDefinite Matrix[2,2] = 2 * P
  S9: PositiveDefinite Matrix[2,2] = Pᵀ @ Q @ P
  Kr: PositiveDefinite Matrix[10, 10] N/m = K[1:11, 1:11]
  G:  Symmetric Matrix[2,2] = Aᵀ @ A
  C:  Symmetric Matrix[2,2] N/m = K + K
  ```
- **What happened:** all rejected ("is not known to be PositiveDefinite" / "... Symmetric"). In particular:
  `P + Q` of two symmetric matrices has *no* tags (not even Symmetric; the spec
  says `+`/`-` claim nothing); `2 * P` keeps `{Symmetric}` but drops PositiveDefinite
  although `2 > 0`; `Aᵀ @ A` is not Symmetric in the checker even though the
  spec's "Algorithm choice" says a Gram product is symmetric bit for bit
  (only the SIR pass knows); a slice loses everything. Literals are
  proven only up to 32 x 32 (`the checker proves PositiveDefinite only for literals up to 32 by 32`),
  and the help text `declare the PositiveDefinite property on an input the value is computed from`
  points at inputs, which cannot be read yet (`error: `K` is declared but has no value`).
  Net effect: Cholesky is reachable only for hand-typed matrices of at most 32 x 32 (`bar_fem.sky`, 10 x 10 literal),
  or through entry 8's accident.
- **Workaround:** `trust_pd` (entry 8); otherwise the assembled `K` is untagged and gets LU
  (2n³/3 instead of n³/3 flops, and `K⁻¹ @ f` with an assembled `K` never claims PD).
- **Expect:** propagation rules for the structural sums: `PD + PD` is PD, `PSD + PD` is PD,
  `c * PD` with a positive literal `c` is PD, `Aᵀ @ D @ A` with `D` PD and `A` square-invertible, symmetric + symmetric is symmetric,
  Gram products are Symmetric; and an `assert_pd`/`assume` form for assembled
  matrices that is checked when run.

### 10. No way to keep a factorization: every `K⁻¹ @ b` refactors `K`
- **Severity:** major. **Category:** missing feature / performance.
- **Tried:** iterative refinement and inverse iteration on a PD `K`:
  ```sky
  def refine(K: PositiveDefinite Matrix[N, N] N/m, f: Vector[N] N) -> Vector[N] m:
      cost flops <= 3 * N^3
      u = K⁻¹ @ f
      for it in range(3):
          r = f - K @ u
          u = u + K⁻¹ @ r
      return u
  ```
- **What happened:** the contract is refuted:
  `error: `refine` costs 4·N³/3 + 14·N² + 7·N flops, which is more than `3 * N^3``;
  each solve refactors (N³/3 each, four of them). The same in `modal_chain.sky`
  (80 Cholesky factorizations of one 3 x 3 for one mode). `sky check` is right, and it is useful that it
  tells you; but there is nothing to write instead: no `L = cholesky(K)` / `lu(K)` object and
  no `solve(L, b)`.
- **Workaround:** put several right-hand sides in one matrix (`K⁻¹ @ F` with `F` a Matrix[N, k] of load cases: one factorization,
  `2·n²·k` more flops, which works and prints a Matrix[N, k] of displacements); impossible
  when the next right-hand side depends on the previous solve (iteration, time stepping).
- **Expect:** a factorization value (`F = factor(K)`, then `F⁻¹ @ b`, with its
  cost: n³/3 once, 2n² per solve) and `cost` lines that can state it.

### 11. A slice with a computed bound cannot run on SIR/native, with a message about "stand-ins"
- **Severity:** major. **Category:** confusing/wrong error message + engine gap.
- **Tried:**
  ```sky
  v = [1 m, 2 m, 3 m, 4 m]
  for i in range(2):
      d = v[i : i + 2] - v[i + 1 : i + 3]
      print(d[0])
  ```
  (the natural way to pull an element's two DOFs out of `u`; `truss2d.sky` first had `u_full[2*j : 2*j + 2] - u_full[2*i : 2*i + 2]`.)
- **What happened:** `sky check` and `sky run` (ast) are fine and print `-1 m` twice.
  `--engine=sir` and `--engine=native`: `error: cannot run natively: cannot build a stand-in for an array of type num[L; (_#1)]: its size on axis 0 is known only as a variable that no value in scope has`.
  `--engine=auto` silently falls back to the tree-walker (here, harmless; on a
  big program, a 10x slowdown with no word about why).
  The cause is the spec's rule: with non-constant bounds the slice length is a fresh size
  variable that equals only itself, so `v[i : i + 2]` is not `Vector[2]` even though the
  type system could see `(i + 2) - i`.
- **Workaround:** index the two entries separately (`u_full[2*j]`, `u_full[2*j + 1]`) as `truss2d.sky` now does.
  (`identity(11)[e : e + 2, :]` is fine because a later `@ ke` with a `2 x 2` pins it.)
- **Expect:** `v[a : a + k]` typed `Vector[k]` (stop minus start when both are the same name plus constants);
  a message that says "this slice has a size the compiler cannot work out" rather than "stand-in".

---

## Minor

### 12. Removing a non-contiguous set of fixed DOFs needs a hand-built selection matrix
- **Severity:** minor (a workaround exists, but it is not obvious). **Category:** missing feature.
- **Tried:** fix the middle node of a bar (`bar_fixed_middle.sky`); free DOFs 0-4 and 6-10.
  No fancy indexing (`K[free, free]`), no `concat`/`hstack`, no row/column assignment
  (`K[5, :] = 0`), and `[K[0, :], K[1, :]]` is `error: an array element must be a number, found a vector`.
- **Workaround:** `S = Σ_k e_k e_free[k]ᵀ` as a loop of outer products of
  identity rows (`S = S + identity(10)[r:r+1, :]ᵀ @ identity(11)[free[r]:free[r]+1, :]`),
  then `S @ K @ Sᵀ`, `Sᵀ @ u`, and reactions `K @ u_full - f_full`. Works on all engines and
  matches the hand answer. In `truss2d.sky` the supports were chosen so the fixed DOFs are the first three:
  the free block is a plain slice, which is the only reason that file is short.
  A penalty (adding `1e15 N/m` on a diagonal entry) needs the same trick for the diagonal entry.
- **Expect:** `K[free, free]` with an index vector, `concat`, and `A.with_row(i, v)`.

### 13. Wrong help text: suggests `ones(...)`, which does not exist
- **Severity:** minor. **Category:** confusing/wrong error message.
- **Tried:** `f = 0 N * zeros(10) + 0 N` (a scalar plus a vector).
- **What happened:** `error: cannot add a scalar to a vector ... = help: write `0 N * zeros(10) + 0 N * ones(0 N * zeros(10))` for every element`.
  There is no `ones` (entry 1), and the suggested expression is not something anybody would write.
- **Expect:** the help should name what exists, or be left out.

### 14. A false note on engine errors: "`sky run` cannot execute this either"
- **Severity:** minor. **Category:** confusing/wrong error message.
- **Tried:** `for row in bars:` over a Matrix with `--engine=native` (or `sir`).
- **What happened:** `error: cannot run natively: iteration over arrays lowers once SIR has array values`
  followed by `= note: `sky run` cannot execute this either: arrays are immutable and attributes are not supported in v0.1`.
  `sky run` (ast) runs this program fine and prints the right numbers. The first line also says SIR has no array
  values, which is stale (these programs use arrays on SIR everywhere).
- **Expect:** delete the second note; reword the first ("iteration over an array is not lowered yet").

### 15. A string passed to a `Scalar`-annotated parameter checks, then SIR fails verification
- **Severity:** minor. **Category:** wrong result/bug (engine mismatch).
- **Tried:** `def report(label: Scalar, w2: Hz^2, ...)` then `report("mode 1", ...)`.
- **What happened:** `sky check` ok and `ast` prints correctly; `--engine=sir`/`native`:
  `error: cannot run on SIR: lowering produced SIR that does not verify ... argument 0 of the call to `report` is str, expected num[1; ()]`.
  Leaving `label` unannotated works on every engine. The spec says the checker does not type strings,
  but a *declared* `Scalar` receiving a string is accepted by the checker and then breaks a different engine.
- **Expect:** the checker rejects it ("expected a number, found a string").

### 16. Derived units print and report as base units; matrices print on one line
- **Severity:** minor. **Category:** ergonomics.
- **Tried:** `print(K)` for an N/m matrix; errors and `--explain` involving stiffness.
- **What happened:** `print(K)` gives `[[3 kg/s^2, -1 kg/s^2], [-1 kg/s^2, 1 kg/s^2]]` (needs `print(K as N/m)`);
  `k = E * A / Le` prints `10000000000 kg/s^2`; errors say `cannot add M·T⁻² and dimensionless` instead of "stiffness (N/m)";
  `FOLD` notes say `2000000000 kg/s^2`; and `res as N` on a stiffness-dimension value says
  `help: `as` changes the unit a value is shown in ... choose a unit of M·T⁻²`
  (which is no help; the user wrote `N/m` a line earlier). A 10 x 10 or 200-entry
  value prints as one very long line.
- **Expect:** remember the unit the user wrote, or pick a named derived unit (N, Pa, N/m) when the dimensions match one,
  and wrap or truncate large arrays (`Vector[200] m [0.00001 ... 0.1]`).

### 17. `solve` is a keyword, so `solve(K, f)` is a syntax error
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** `f = solve(K, [1, 2, 3])`.
- **What happened:** `error: expected an expression, found keyword `solve` ... help: `equation` and `solve` start statements: `sol = solve orbit from 0 s to 1 yr with ...``.
  The message is good, but an engineer reaches for `solve(K, f)` first, and `K⁻¹ @ f` reads like an inverse.
- **Expect:** `solve` as a contextual keyword (only at the start of a statement or after `=` followed by an equation name),
  so `solve(K, f)` can exist as the function everybody knows.

### 18. Natural single-letter names trigger "shadows a predefined constant" warnings
- **Severity:** minor. **Category:** ergonomics.
- **Tried:** `c = dx / Lb` (direction cosine), `h = ones(3)`, `G: Symmetric Matrix[2, 2] = ...`, `s`/`c` for sin/cos in a truss.
- **What happened:** `warning: `c` shadows the speed of light` (also `h`: Planck constant, `G`: gravitational
  constant) with the help "rename your variable". The warning is accurate and harmless, but `c` (damping,
  cosine, chord), `h` (height, step), `G` (shear modulus!) are the standard names in
  mechanics; shear modulus `G` and `E`, `A`, `L` are exactly what a structure program uses
  (`E`, `A` and `L` are fine, `G` is not). There is no way to silence a warning
  (spec, "No way to silence a warning (yet)"), so a program with `G = 80 GPa` is never warning-free.
- **Workaround:** `cx`, `cy`, `shear_G`.
- **Expect:** warn only when the shadowed constant is *used* later in the same scope, or allow a local-scope shadow quietly.

### 19. `N` means both "newton" and "a size" in the same signature
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** `def solve_spd(K: PositiveDefinite Matrix[N, N] N/m, f: Vector[N] N) -> Vector[N] m:` and `cost flops <= N^3 / 3 + 2 * N^2`.
- **What happened:** it parses and checks (by position: `N` inside `[...]` is a size, a bare `N` is a newton),
  and the cost proof is right. It is a trap for readers: `Vector[N] N` in a line about forces,
  and `-> N` (a force) next to `Matrix[N, N]`. `sum_loads(f: Vector[N] N) -> N` in `solver_costs.sky` is hard to read.
- **Workaround:** use `n`/`M` for sizes by habit.
- **Expect:** nothing to change in the grammar, but the style guide / error messages might say "use a lower-case `n` for sizes".

### 20. Dimensioned matrices are built with `0 N/m * zeros(n, n)`; PD literals need a unit on every entry
- **Severity:** minor. **Category:** ergonomics.
- **Tried:** `K = zeros(11, 11)` then `K = K + Pᵀ @ ke @ P`; and
  `K: PositiveDefinite Matrix[3, 3] N/m = [[2, -1, 0], [-1, 2, -1], [0, -1, 1]] * 1000`.
- **What happened:** `error: cannot add dimensionless and M·T⁻²` (with a note chain that is genuinely good, and a
  `help: did you mean `K * ke``, which is wrong here); `zeros`/`identity` are dimensionless by design.
  The second fails twice: `expected M·T⁻², found dimensionless` and then `is not known to be PositiveDefinite`
  (the proof needs every entry to be a literal with its own unit, so a scale factor on a literal matrix loses the proof).
  The 10 x 10 literal in `bar_fem.sky` is 10 rows of `0 N/m`.
- **Workaround:** `0 N/m * zeros(11, 11)` (it even carries `{Symmetric, Diagonal}` tags: `zeros` is Diagonal) and writing `0 N/m` in every cell.
- **Expect:** `zeros(11, 11) N/m` (or `zeros(11, 11, unit=N/m)`), and the tag proof looking through a scale factor.

### 21. No negative indices and no way to build coordinate vectors
- **Severity:** minor. **Category:** missing feature.
- **Tried:** `u[-1]` for the tip, `linspace(0, L, 11)`/`arange` for node coordinates and the analytic solution.
- **What happened:** `error: an index cannot be negative ... v0.1 counts from 0 and has no negative indices`; no `linspace`.
- **Workaround:** `u[len(u) - 1]` or the literal `u[9]`; the analytic comparison is a scalar loop
  `for i in range(10): err = max(err, abs(u[i] - F * ((i + 1) * Le) / (E * A)))` (that works well, loop and all, on every engine).
- **Expect:** `u[-1]`, `linspace`.

### 22. `--explain` buries the one note you asked for
- **Severity:** minor. **Category:** ergonomics.
- **Tried:** `sky check --explain bar_fem.sky` to see whether the solve is LU or Cholesky.
- **What happened:** about 25 notes (`DCE-001` "`200 GPa` is removed because `E` is never used", `FOLD-FLOAT-001`, `PAR-003`
  "not parallelized: it may fail and is small", `MEM-001`) for a 15-line program, and the `LA-CHAIN-001` /
  Cholesky line is in the middle. The content is good (and the flop estimate "4,200 flops as written, 867 after the rewrite" is exactly what I wanted).
- **Workaround:** `| grep -E 'LA-|Cholesky|cost of'`.
- **Expect:** `--explain=linalg` / `--explain=cost` filters, or the solve choice and cost summary first.

### 23. "The matrix is singular" does not say which degree of freedom
- **Severity:** minor. **Category:** ergonomics.
- **Tried:** an unconstrained bar (forgot the boundary condition), `K⁻¹ @ f` on the full 5 x 5.
- **What happened:** `error: the matrix is singular ... this matrix has no inverse ... a pivot of the elimination is zero, or too small`.
  With the PD tag (via `trust_pd`): `the matrix is not positive definite ... the factorization failed at row 4: its pivot is not positive`;
  that one is better (it names a row). An engineer's #1 error (a rigid-body mode / missing support) deserves a hint.
- **Expect:** name the pivot row for LU too, and mention "a structure with no supports has a rigid-body mode".

---

## What was pleasant

- **Units on structures are real.** `E = 200 GPa`, `A = 1 cm^2`, `k = E * A / Le`, `F * L / (E * A) as mm`:
  no unit bookkeeping at all, and the error for `F + L` ("cannot add force and length ... `F` is a force because `F = 1 kN`") shows the chain of reasoning.
  The 2D truss had four unit-carrying arrays flowing through a loop and the results printed in `kN` and `mm`.
- **Shape errors are legible.** `cannot multiply Matrix[2,2] by Vector[3] ... inner dimensions differ: 2 vs 3` with
  the axis table, and the `Matrix[10,10] + Matrix[11,11]` mismatch inside an assembly loop, are at the line, on the operand.
- **`K⁻¹ @ f` is the right spelling.** `--explain` says it became a Cholesky or an LU solve, why, and what it saves
  (`4,200 flops as written, 867 after`), `K⁻¹ @ F` with a matrix of load cases works, and the two failure
  modes (singular matrix; PositiveDefinite promise broken) are run-time errors that name the problem and are the
  same on every engine.
- **Performance contracts work and are informative.** `sky check` proved `N³/3 + 2·N²` for a Cholesky solve,
  `2·N³/3 + 2·N²` for LU, `== 2·N²` for `K @ u`, `<= N` for a bounded loop; refuted a loose
  `cost flops <= N^3` for LU with the size at which it fails (N = 1) and the exact bound to write; refused a
  `cost allocations <= 0`; reported an unbounded `while` as a warning; and showed that `refine` refactors four times,
  which is what a person would not notice. A `cost runtime <= 10 ms` is a parse error that explains why time
  is not a contract.
- **Engines agree to the bit.** Every committed program prints identical bytes on `ast`, `sir` and `native`
  (apart from the engine gaps listed above), including long LU/Cholesky/Gram solves at 2000 DOF,
  so a result can be trusted and quoted. The dense kernels are fast: 2000 DOF assembled with a Gram product and solved by LU in 0.2 s;
  4000 DOF in about 1 s; dense LU of 200 DOF in a few milliseconds on any engine.
- **Syntax stays out of the way.** `for`, `while`, `def` with unit-annotated parameters, `assert err < 1e-12 m, "..."`,
  `Aᵀ @ B @ A`, `identity(11)[e:e+2, :]` and `hypot(dx, dy)` read like the blackboard; `x as mm` at the end of a print is quick to learn; the
  precision of the exact-solution comparison (`1.5e-19 m` of error on a 10-element bar) is reassuring.
- **Tag proofs on literals work and are explained.** `[[1 N/m, 2 N/m], [2 N/m, 1 N/m]]` declared PositiveDefinite:
  "the leading 2x2 block is not positive definite (its Cholesky pivot is -3)", pointing at the entry.
