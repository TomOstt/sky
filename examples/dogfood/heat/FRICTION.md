# Sky dogfooding: heat diffusion

Written as a physicist who found Sky and tried to do heat diffusion with it. Sky is at commit `b18f4d6`, built with
`cargo build --release -p sky-cli`. No compiler or crate was modified.

Programs (all run, and all give byte-identical output on `ast`, `sir`, `native` and `auto`, and as `sky build` binaries,
except where noted):

| file | what it does |
|------|--------------|
| `rod1d.sky` | 1D rod, FTCS, stability check, steady state vs the analytic line (max error 3.1e-8 K) |
| `plate2d.sky` | 33x33 aluminium plate with units (alpha = k/(rho c_p)), FTCS, vs a Fourier series (centre 324.9999974 K vs 325 K) |
| `implicit1d.sky` | backward Euler and Crank-Nicolson via `A⁻¹ @ b`; the error halves / quarters as dt halves (1st / 2nd order, as it should) |
| `rod_mol.sky` | method of lines with `equation` / `solve` (adaptive Dormand-Prince); AST engine only, by design |

Counts: **0 blocker, 11 major, 13 minor.** Two things are strictly impossible as written (element writes, a hand-written
Thomas algorithm) but each has a workaround, so they are rated major.

Summary of the shape of the experience: the units and shape checking are excellent and caught every deliberate mistake I
made with a clear message. Almost everything that hurt is about **building and reducing arrays**: there is no way to write
an element, concatenate, generate a grid, broadcast a scalar, or reduce (`sum`, `max`). The whole of every program below is
bent around that.

---

## Major

### F1. Arrays are immutable and there is no concatenation: a Dirichlet FTCS step has to be written with matrix products
- **Severity:** major (would be a blocker without the identity-slice trick)
- **Category:** missing feature
- **What I tried:** the obvious stencil.
  ```sky
  u = 300 K * exp(zeros(21))
  u[0] = 400 K
  for i in range(1, 20):
      u[i] = u[i] + r * (u[i-1] - 2 * u[i] + u[i+1])
  ```
- **What happened:** `error: cannot assign to an element of 'u' ... = note: arrays are immutable in v0.1`. There is also no
  `concat`, `append`, `vstack`, `cat` (`cannot find 'concat' in this scope`). Slicing `u[1:20]` gives the interior update
  fine, but I then have no way to put the two boundary values back.
- **Workaround:** embed the interior with a slice of the identity matrix and add a boundary vector:
  ```sky
  E  = identity(21)[:, 1:20]                              # 21 x 19 embedding
  bc = T_hot * identity(21)[:, 0] + T_cold * identity(21)[:, 20]
  u  = E @ (u[1:20] + r * (u[0:19] - 2*u[1:20] + u[2:21])) + bc
  ```
  In 2D: `E @ interior @ Eᵀ + Bc`, with `Bc` assembled from outer products of identity columns and `exp(zeros(1, 33))`.
  It works and is bit-identical on all engines, but it is a puzzle a newcomer would not solve without help, and it costs
  O(N²) (1D) / O(N³) (2D) flops per step for an O(N) / O(N²) stencil (see F23).
- **Expected:** `u[i] = x` (or a functional `with`/`set(u, i, x)`), `concat(a, b, c)`, and a way to build a vector from a
  function of the index. Ghost-node boundary conditions (Neumann/insulated ends) are the same problem and I did not even try.

### F2. No tridiagonal / banded solve, and none can be written by hand (Thomas algorithm is inexpressible)
- **Severity:** major (the Thomas algorithm itself is a blocker: it needs in-place array sweeps)
- **Category:** missing feature / performance
- **What I tried:** Crank-Nicolson needs `(I - r/2 Tm) u_new = rhs` with `Tm` tridiagonal. I wrote `A⁻¹ @ rhs`.
- **What happened:** it works and is correct, but `sky check --explain` says
  `[LA-CHAIN-001] ... is rewritten to solve[lu](A, ...)`: a dense LU, refactored every step (2/3 n³ flops each time, nothing
  hoisted). A 399x399 system, 200 steps: 0.55 s on every engine; it will be cubic in the grid size. I cannot write the O(n)
  Thomas algorithm myself, because the forward sweep updates `c'[i]` and `d'[i]` in place.
- **Workaround:** dense `A⁻¹ @ rhs`; fine for the 19-node demo. `inv(A)` once outside the loop plus `Ainv @ rhs` would give
  O(n²) per step but I did not rely on it.
- **Expected:** a `Tridiagonal`/`Banded` tag (like `Diagonal`) with an O(n) solve, or `tridiag(lower, diag, upper)` plus a
  solver, and factorisation reuse for a loop-invariant `A`.

### F3. No scalar broadcasting: `u - T0` is an error, and the suggested fix uses a function that does not exist
- **Severity:** major (broadcast), see F4 for the wrong help text
- **Category:** ergonomics / design choice (documented as deliberate, spec "No broadcasting between arrays")
- **What I tried:**
  ```sky
  w = (x - 0.5 m) / (0.1 m)
  u = zeros(5) + 300 K
  v + 1 K
  ```
- **What happened:** every one is `error: cannot add a scalar to a vector ... no broadcasting between a scalar and an array`.
  The error is clear, but it hits in every initial condition and every shifted profile.
- **Workaround:** build a ones vector once, `ones21 = exp(zeros(21))`, and write `x - 0.5 m * ones21`,
  `T_cold * ones21 + ...`.
- **Expected:** `array ± scalar` to broadcast when the dimensions agree (the spec already lets `array * scalar` scale),
  and an `ones(n)` builtin.

### F5. No linspace / arange / grid generation: node coordinates are a hand-typed literal
- **Severity:** major
- **Category:** missing feature
- **What I tried:** `x = linspace(0 m, 1 m, 21)`, `arange(21)`, `range(21)` as a value.
- **What happened:** `cannot find 'linspace' in this scope`, `cannot find 'arange'`; `x = range(3)` is documented as an error
  ("`range` is not a value"). Nothing else produces `0, 1, 2, ...`.
- **Workaround:** `idx = [0, 1, 2, ..., 20]` typed out (21 numbers; for a 257-node grid I used a profile that needs no
  coordinates). Any grid refinement study means rewriting a literal.
- **Expected:** `linspace(a, b, n)` (dimension-aware) and `arange(n)`. Since sizes are types, `linspace(0 m, 1 m, 21)` is
  perfectly checkable.

### F6. No reductions: `sum`, `max`, `min`, `argmax`, `mean`, max-norm
- **Severity:** major
- **Category:** missing feature
- **What I tried:** `max(abs(u - exact))` to report "the max error", the first thing the task asks for.
- **What happened:** `error: 'max' takes 2 arguments, found 1`; `sum`, `cumsum`, `trace`, `diag`, `outer` are all
  `cannot find ... in this scope`.
- **Workaround:** a scalar loop, `err = 0 K; for i in range(21): err = max(err, abs(u[i] - exact[i]))`. (`for e in abs(u -
  exact)` is nicer but is rejected by the SIR and native engines, see F16.) A sum is `ones @ v` with a ones vector.
- **Expected:** `max(v)`, `min(v)`, `sum(v)`, `mean(v)`, and `norm(v, inf)`; they are one-liners the checker can type.

### F7. A size held in a variable is not "constant": `N = 21; zeros(N)` fails, though slices accept it
- **Severity:** major (forces `21` to be repeated everywhere)
- **Category:** ergonomics / inconsistency
- **What I tried:**
  ```sky
  N = 21
  I = identity(N)
  u = zeros(N)
  w = v[1:N-1]          # accepted
  ```
- **What happened:** `error: 'identity' needs a constant size, found 'N' ... not a constant ... help: write the number, e.g.
  'identity(3)'`, and the same for `zeros(N)`. Yet `v[1:N-1]` is accepted (it is checked at run time, with the length as a
  fresh size variable). Also `v: Vector[N] K = ...` after `N = 21` is `error: size variable 'N' has the same name as a value
  in scope`.
- **Workaround:** write `21`, `20`, `19` literally in each program, with comments saying they must agree. Changing the
  resolution means editing about 15 numbers in `rod1d.sky`.
- **Expected:** constant propagation of integer literal bindings into sizes (`N = 21` is as constant as `21` is). The checker
  already folds `L / 20` in `--explain`.

### F8. No formatted output; floats print with all digits; nothing to export or plot
- **Severity:** major
- **Category:** missing feature / ergonomics
- **What I tried:** a results table, fixed precision, CSV rows, `print("a", "b", sep=",")`, `print(f"T = {x}")`, `format(x, 3)`,
  `"a" + "b"`.
- **What happened:** `f"..."` is a parse error (`unexpected string literal after an expression`), `sep=` is a parse error,
  `format`/`str` are undefined, `"T = " + "x"` is `cannot add a string and a string`. `print` always ends with a newline and
  separates with a space (so `print("x", "y")` cannot be made a CSV line, and an ASCII heat-map row cannot be drawn).
  Floating-point noise is printed in full: `alpha dt / dx^2 = 0.3999999999999999`, `stability limit: dt <= 12.500000000000002 s`.
- **Workaround:** round by hand and drop the unit: `print(round(u[i] / (1 nK)) / 1e9)`. The unit disappears in the printed
  value, so the header must carry it. Whole vectors print on one line with all digits (`print(u as K)` of the 21-node
  rod is one 400-character line of 17-digit floats, which is why `rod1d.sky` prints a hand-rounded table instead).
- **Expected:** `print(x, digits=4)` or `fmt(x, 4)`/`x as K:.3f`, plus some way to write a data file or plot. I found none
  (see F9).

### F9. No input: parameters cannot be read from a file, the command line, stdin or the environment; no file output
- **Severity:** major
- **Category:** missing feature
- **What I tried:** `n = input("n?")`, `args[0]`, `open("out.csv", "w")`, `write("out.csv", [1, 2])`.
- **What happened:** all `cannot find '...' in this scope`. The README is honest that inputs "have nothing to read it from at
  run time".
- **Workaround:** edit constants in the source and redirect stdout (`sky run f.sky > data.txt`).
- **Expected:** at least `args`/environment parameters with units (`alpha = param("alpha", 1e-4 m^2/s)`) and a CSV writer.
  A parameter study (different alpha, dt, grid) cannot be scripted at all today.

### F10. No tuples: a function cannot return two values, no unpacking
- **Severity:** major
- **Category:** missing feature (+ unhelpful message)
- **What I tried:**
  ```sky
  def minmax(u: Vector[3]):
      return min(u[0], u[1]), max(u[0], u[1])
  a, b = minmax([1, 2, 3])
  ```
- **What happened:** `error: expected end of line, found ','` at the comma of the return, and again at the comma of the
  assignment. It does not say "tuples are not supported".
- **Workaround:** return one vector of the values, or call two functions. A time stepper that returns `(u, steps)` has to
  keep the second in a global (which a `def` cannot rebind), so the step count has to be recomputed by the caller.
- **Expected:** at minimum a message "multiple return values are not supported; return a Vector", at best tuples.

### F11. Generic code over grid size is hard: no `Vector[N - 2]`, and `N` is not a value in a body
- **Severity:** major
- **Category:** missing feature / spec limitation (spec: "v0.1 shapes are equality constraints only")
- **What I tried:** a reusable stencil.
  ```sky
  def laplacian(u: Vector[N]) -> Vector[N - 2]:
      return u[0:N-2] - 2 * u[1:N-1] + u[2:N]
  ```
- **What happened:** `error: shape arithmetic like 'N - 2' is not supported in v0.1`. Using `N` in the body gives `cannot find
  'N' in this scope`. Inside the body I must write `n = len(u)` and slice with `n`, and then `identity(n)` is not allowed
  (F7), so the embedding matrix has to be passed in as an argument: `def step(u: Vector[N] K, E: Matrix[N, M], bc: Vector[N] K, r)`.
  This works (`step(u, E, bc, 0.4)`), with the size mismatch only caught when the program runs.
- **Workaround:** as above, or hard-code `Vector[21] K` (and a `Vector[21]` without a unit is *dimensionless*; the
  error for passing a temperature says so, with the help "write the unit in the annotation").
- **Expected:** `N`-arithmetic in signatures (`Vector[N-2]`), and size variables usable as values in the body.

### F13. `W/(m*K)` does not parse: thermal conductivity cannot be written the textbook way
- **Severity:** major (it is the first line of the 2D plate problem; trivial workaround, but it makes every first attempt fail)
- **Category:** ergonomics/syntax
- **What I tried:** `k = 237 W/(m*K)`, `c_p = 897 J/(kg*K)`, `237 W/(m K)`.
- **What happened:**
  ```
  error: expected a unit name after `/` in a unit literal, found `(`
  1 | k = 237 W/(m*K)
  help: to divide the quantity, put spaces around `/`
  ```
  The help ("put spaces around `/`") is for a different intent (dividing a quantity), and would silently make `237 W / (m*K)` a
  *computed quantity* instead of a unit literal (a dimension-preserving but easy-to-misread change).
- **Workaround:** `237 W/m/K`, `897 J/kg/K`. Note `237 W/m*K` is accepted and *silently* means W·K/m (printed `237
  kg*m*K/s^3`), the classic trap; the checker only catches it later when the dimension of a derived quantity is wrong.
- **Expected:** parentheses in unit literals, or a specific message: "write `W/m/K` or `W*m^-1*K^-1`".

---

## Minor

### F4. Error help recommends `ones(...)`, which does not exist
- **Severity:** minor
- **Category:** confusing / wrong error message
- **What happened:** `u + u[0]` gives `help: write 'u + u[0] * ones(u)' for every element`. `ones` is `cannot find 'ones' in
  this scope`. I used `exp(zeros(n))`.
- **Expected:** either add `ones`, or the help should suggest something that compiles.

### F14. `°C` is a lexer error, not an explanation
- **Severity:** minor
- **Category:** confusing error message (+ spec gap, documented as out of scope)
- **What I tried:** `T = 25 °C`, and `25 degC`.
- **What happened:** `error: unexpected character '°' (U+00B0)`; `error: unknown unit 'degC'`. The spec says offset units are
  out of scope (temperature is K); the message does not.
- **Workaround:** use K everywhere, convert by hand (`T + 273.15 K`); only differences are unaffected.
- **Expected:** "offset units like °C are not supported in v0.1, write `298.15 K`".

### F15. `c` (specific heat), and other short names, shadow physical constants
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** `c = 897 J/kg/K` gives `warning: 'c' shadows the speed of light`. It is only a warning, and for heat
  conduction `c` is the textbook symbol for specific heat (also `h` the heat-transfer coefficient is fine; `k` is free).
  I used `c_p`, `cp`.
- **Expected:** fine as designed, but worth noting for a heat/thermo audience.

### F16. SIR/native error says "`sky run` cannot execute this either" when `sky run` does execute it
- **Severity:** minor
- **Category:** confusing error message / docs
- **What I tried:** `for e in abs(u - exact):` (iterate over an array), and `for dt in [100 s, 50 s, 25 s]:`.
- **What happened:** `--engine=ast` and `auto` run it; `--engine=sir`/`native` give
  ```
  error: cannot run on SIR: iteration over arrays lowers once SIR has array values
  = note: `--engine=sir` runs only programs that lower to SIR
  = note: `sky run` cannot execute this either: arrays are immutable and attributes are not supported in v0.1
  ```
  The last note is false here (the tree-walker ran it), and mentions attributes though nothing used one. Also: the README
  says "Not there yet: `for` loops", yet `for` works (only over `range` and arrays); stale.
- **Workaround:** `for i in range(n): ... u[i] ...` (runs on all engines).
- **Expected:** drop the misleading note when the program is fine on `ast`; update the README.

### F17. The checker cannot be told a computed matrix is symmetric or positive definite; its own help is circular
- **Severity:** minor
- **Category:** confusing error message / missing feature
- **What I tried:** `A: PositiveDefinite Matrix[19, 19] = I19 - 0.5 * r * Tm` (the Crank-Nicolson matrix is SPD), and
  `S: Symmetric Matrix[5, 5] = sub + subᵀ`, and `I5 + I5`.
- **What happened:** `error: 'I5 - 0.1 * T' is not known to be PositiveDefinite ... help: declare the PositiveDefinite property
  on an input the value is computed from`. But an input has no way to be read at run time (README), so that help cannot be
  followed. Even `identity(5) + identity(5)` and `M + Mᵀ` are not known symmetric. As a result the SPD solve takes the LU path
  (`solve[lu]`), not the Cholesky rewrite the showcase advertises.
- **Workaround:** none; use LU.
- **Expected:** `assume PositiveDefinite A` or `A: PositiveDefinite ... = expr` with a runtime verification (the interpreter
  already verifies Cholesky promises), and `M + Mᵀ` / `a*I + b*M` of symmetric operands should keep `Symmetric`.

### F18. `solve` is a keyword, so the natural `solve(A, b)` cannot be a name
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** `l = solve(identity(3), [1, 2, 3])` gives `error: expected an expression, found keyword 'solve'`. The
  right Sky spelling is `A⁻¹ @ b`, which I found in the showcase, and it is nice. A user porting `solve(A, b)` just sees a
  parse error that does not suggest `A⁻¹ @ b`.
- **Expected:** help text "for a linear solve write `A⁻¹ @ b`; `solve` starts a differential-equation solve".

### F19. Derived units print as base SI composites
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** `print(0.897 kJ/kg/K)` prints `897 m^2/s^2/K`; `print(237 W/m/K)` prints `237 kg*m/s^3/K`. (This is per spec: only
  dimensions that match a unit table entry get a name.) `print(k as W/m/K)` works, so each print needs `as`. A dimensionless
  ratio that was a `W/m^3` comes out as `kg/s^3`.
- **Expected:** a way to say "show this variable in these units from now on" (a declared display unit on the binding:
  `k: W/m/K = ...` already declares the unit but `print(k)` ignores it).

### F20. Compile-time constants asserts are not checked at compile time
- **Severity:** minor
- **Category:** missing feature
- **What I tried:** `dt = 20 s`, `r = alpha*dt/dx^2` (0.8), `assert r <= 0.5, "FTCS unstable"`.
- **What happened:** `sky check`: `ok`. `sky run`: `error: assertion failed: FTCS unstable ('r <= 0.5', r = 0.7999999999999998)`.
  Everything is constant, `--explain` already folds `0.3999999999999999 <= 0.5` to `true` (`[CONTRACT-001] this assert always
  holds, so its check is removed`), so the failing case is known statically too.
- **Expected:** `sky check` reporting "this assertion is always false" on a constant condition. (The run-time message is
  otherwise very good: it prints the value of `r`.)
- **Related, pleasant:** without the assert, an unstable FTCS (r = 0.6) overflows after some iterations and stops with
  `error: arithmetic overflow` pointing at the stencil expression, instead of NaNs. But the message does not say *which
  iteration*, nor hint at instability.

### F21. `--explain` is very noisy, and one note is misleading
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** `sky check --explain` of the 70-line plate program prints about 140 notes, most of them constant-folding
  of the setup. For `alpha = 1e-4 m^2/s` it says `[DCE-001] '1e-4 m^2/s' is removed ... because 'alpha' is never used`, but
  `alpha` *is* used (after folding replaced the use by the constant).
- **Expected:** a quieter default (`--explain=linalg`), and no "never used" for a folded value.

### F22. `sky build` fails after the documented single build command
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** after `cargo build --release -p sky-cli`: `error: the runtime library 'libsky_rt.a' was not found next to
  the 'sky' executable; build it with 'cargo build -p sky-rt'`. The message is clear and fixes it; `-p sky-cli` is the build
  line given in the task, and the README says `cargo build --workspace` builds it.
- After `cargo build --release -p sky-rt`, `sky build` produced binaries with output identical to `sky run` for the three
  array programs. `rod_mol.sky` is refused (`cannot build: differential equations are not supported by SIR yet`), as documented.

### F23. The identity-embedding workaround wastes flops and memory traffic
- **Severity:** minor
- **Category:** performance
- **What happened:** per step the 2D update is 2·(N·(N-2)² + N²(N-2)) flops from the two embedding products, where the stencil
  needs about 6·N². A 257x257 plate, 1000 steps: 1.8 s on `ast` and `native` alike (`user 5.4s sys 2.7s`: the big
  temporaries are allocated and zeroed each step), 3.6 s on `sir --threads=1`. A 129x129 plate takes
  0.65 s and a 1601-node rod 1000 steps 0.2 s, so it is usable, just 100x slower than a stencil kernel would be. I also saw
  that the three engines were not faster than each other on this, because everything is inside `@`.
- **Expected:** shifted-slice arithmetic that avoids the matmul (a `pad`/`concat`), or the fusion pass to treat
  `E @ x` with a 0/1 selection matrix as a copy.

### F24. Errors show dimensionless as `[1]`
- **Severity:** minor
- **Category:** ergonomics
- **What happened:** `error: 'u' of 'step' expects 'Vector[21] [1]', found 'Vector[21] [Θ]'`. A physicist reads `[1]` and
  `[Θ]` as a unit bracket; "dimensionless" and "temperature" would be easier. The explanatory note
  ("an annotation without a unit is dimensionless") is good.

### F25. `r` vs `L / dx` rounding: `dx = L / 20` then `r = alpha*dt/dx^2` gives `0.3999999999999999`, not `0.4`
- **Severity:** minor
- **Category:** ergonomics (inherent to f64, see F8)
- **What happened:** that is the right IEEE answer; I just have no way to print `0.4`. (`--explain` helpfully says the value
  is accurate to 2.7e-16 relative.) The `assert r <= 0.5` is fine because there is margin; a user choosing `r = 0.5`
  exactly could see rounding push it just past the limit and fail the assertion; there is no tolerance idiom.

---

## Notes on engines and numerics
- All four `sky run` engines (`ast`, `sir`, `native`, `auto`) and `sky build` binaries printed identical bytes for `rod1d`,
  `plate2d` and `implicit1d`. `rod_mol` only runs on `ast`/`auto`; `--engine=sir|native` refuse with a clear error (documented).
- `--numerics=relaxed` gave the same printed digits on `plate2d`, with a stderr notice.
- Times, whole programs: `rod1d` 0.03 s, `plate2d` 0.25-0.33 s, `implicit1d` 0.03 s on every engine. There is no
  noticeable gain from `native` on `@`-heavy programs, only on scalar loops.
- The explicit rod needs 2141 steps to reach steady state at r = 0.4 (t = 21410 s), the plate 6860 steps; both are quick.

## What was pleasant
- **Units caught my mistakes with the reasons**: forgetting to square `dx` gave `cannot compare length and dimensionless`
  plus "`r` is a length because `r = alpha * dt / dx` (line 4), where `alpha` is of dimension L²·T⁻¹ ...". `print(w as m)` of
  an area explained `as` changes the unit shown, not what it measures. Adding `300 K + 5 m` is caught at check time.
- **`α = k/(ρ c_p)` came out as m²/s by itself**; `dt <= dx^2 / (4 * alpha)` is a legal, dimension-checked assertion, and
  `(0.2 m)^2 / alpha as min` just works.
- **Shape errors are excellent**: `cannot add Vector[19] to Vector[18]; axis 0 differs: 18 vs 19` and the `@` contraction
  note (`[21, *19*] @ [*21*] needs 19 == 21`) found my off-by-one slice mistakes before running.
- **`A⁻¹ @ b` reads like the maths**, and `--explain` told me it was an LU solve with a flop estimate.
- **`equation` / `solve` made a method-of-lines rod in 5 lines** (`u' = alpha / dx^2 * (Tm @ u + bc)`), adaptive, with an
  explanation of why it was not stiff. The semi-discrete error was 8e-9 K.
- **No NaN, ever**: an unstable FTCS stops with an overflow error instead of silently printing garbage.
- **Fast and bit-identical across engines and as `sky build` binaries**; start-up is instantaneous.
- `for i in range(a, b, step)`, `x += 1`, `while change > 1e-9 K:`, `for x in vector` (rows of a matrix) all read like Python.
- Slices with constant bounds are checked at check time, and `identity(21)[:, 1:20]` works as a selector, which is what saved
  the day.

---

## Tally
- **Blocker (0).**
- **Major (11):** F1 (no array writes/concat), F2 (no tridiagonal/banded solve; Thomas inexpressible), F3 (no scalar
  broadcast), F5 (no linspace/arange), F6 (no reductions), F7 (`N = 21` not a constant size), F8 (no formatted output),
  F9 (no input / file output / plotting), F10 (no tuples), F11 (no `N-2` in signatures, `N` not a value in the body),
  F13 (`W/(m*K)` does not parse). (F-numbers are not contiguous: the numbering follows the order I hit them.)
- **Minor (13):** F4, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25.
