# Friction log: classical mechanics and ODEs

Dogfooding Sky as a physics student/researcher. Domain: projectile with drag, double pendulum, three-body figure-eight, Robertson stiff kinetics. All programs are in this directory with their `.out`. Everything ran with `./target/release/sky` built from `origin/main` (b18f4d6). Timings are on a shared, busy laptop.

Summary: **0 blockers, 6 major, 12 minor.** Every blocker-looking item had an in-language workaround, but several of those workarounds are heavy. The ODE solver itself is impressive (stiffness detection, symplectic selection, unit-checked right-hand sides, good error messages). The pain is almost all in what `solve` returns and in how it composes with the rest of the language.

Engine comparison: `equation`/`solve` run only on the tree-walker (`--engine=native` and `--engine=sir` refuse; `auto` falls back). The programs without equations (`drag_ad.sky`) print identical bytes on `ast`, `sir`, `native` and `auto` (0.38 s, 0.23 s, 0.14 s, 0.14 s). No engine disagreements were found.

Files: `drag.sky` (angle of max range via `solve` + Newton on landing time + golden section), `drag_ad.sky` (hand-written RK4 so `derivative` can differentiate through it; Newton on dR/dtheta with a second derivative), `drag_table.sky` (sampled trajectory by chaining solves), `double_pendulum.sky`, `double_pendulum_unc.sky`, `figure8.sky`, `figure8_matrix.sky`, `figure8_lz.sky` (bug repro), `robertson.sky`, `param_equation.sky` (an `equation` defined inside a `def`).

---

## Major

### 1. `conserve` of a quantity whose value is 0 makes the solver run (almost) forever
- **Severity:** major (a blocker for that exact program; the workaround is to drop the `conserve`)
- **Category:** wrong result/bug, performance
- **What I tried:** the figure-eight orbit has zero total angular momentum, so I declared it, as I would on paper:
  ```sky
  sol = solve fig8 from 0 s to T with r1 = p1, ...:
      tolerance 1e-8
      conserve Lz = mb * (r1[0] * v1[1] - r1[1] * v1[0] + r2[0] * v2[1] - ... )
  ```
- **What happened:** no output for more than 100 s (killed). Shortened to 0.1 s (`figure8_lz.sky`), the explanation says: `Lz drifted by at most 2.3e2 (relative)` and `the step was reduced to 6.64e-6 s ... where the trial drift was 170`, `took 15060 fixed steps` for 0.1 s of simulated time at tolerance 1e-6. At tolerance 1e-9 a 1 s span took 42 s and 1.6 million steps; 2 s did not finish in 60 s. The drift is `|Q(t) - Q(0)| / |Q(0)|` with `Q(0)` = 0 up to rounding, so it is rounding noise divided by ~1e-17, and no step size can satisfy it. The step-shrinking loop ("a fourth-order method's bounded error scales like the step to that power") then pushes the step down on a quantity that can never pass. It is not an error and not a warning; `sky check --explain` does not mention it. The same happens for any quantity that passes through or starts at 0 (net momentum of a system at rest, energy measured from a reference level, `Lz` of a radial orbit).
- **Workaround:** remove `conserve Lz` (`figure8.sky`: 0.5 s for one period at 1e-9, 0.75 s for ten periods).
- **Expected:** a relative drift of a quantity with `|Q(0)|` below `atol`-ish should fall back to an absolute drift (using the quantity's own scale, e.g. the norm of the terms, or `atol`), or `sky` should say "`Lz` is 0 at the start, so a relative drift is meaningless" and stop shrinking the step. At minimum, a cap on the step-reduction loop with a warning.

### 2. No events or stopping conditions: cannot say "stop when y = 0"
- **Severity:** major
- **Category:** missing feature
- **What I tried:**
  ```sky
  sol = solve flight from 0 s to 5 s with x = 0 m, ...:
      tolerance 1e-9
      until y < 0 m
  ```
- **What happened:** `error: unknown solve option until` / `help: the options are tolerance 1e-9 and conserve energy = <expression>`. Clear, but there is nothing to try next.
- **Workaround:** guess a flight time `T` (vacuum value), then Newton on `T`: `solve ... from 0 s to T`, `T = T - sol.y / sol.vy`, repeated 12 times, each time re-solving from t = 0 (`drag.sky`). Or chain short solves until `y < 0` and bisect (`drag_table.sky`). Both are fine for a toy but are O(iterations) full solves and give the landing time only as accurate as the loop count.
- **Expected:** `until y < 0 m` (or `stop when y <= 0 m`) in the solve block, with the solution holding the state at the event (`sol.t` is the landing time, `sol.x` the range). Root-finding on the dense output is what every ODE library (SciPy `events=`, Julia callbacks, MATLAB `Events`) offers, and it is the first thing a projectile problem needs.

### 3. No sampled output and no solution as a function of time
- **Severity:** major
- **Category:** missing feature (the spec lists it as "left to a later one")
- **What I tried:**
  ```sky
  sol = solve flight from 0 s to 5 s with ... at [1 s, 2 s]      # or
  print(sol(2 s))
  ```
- **What happened:** `at [...]`: `error: unexpected name 'at' after the unit 'm/s'` with `help: compound units are written with * and no spaces: m/s*at`. That message is misleading: the problem is that `at` is not a solve clause, not that I wrote a bad unit. `sol(2 s)`: `error: cannot call a solution`. `print(sol)` prints `<solution of flight>`.
- **Workaround:** chain solves, each starting from the previous final state (`sol.x`, `sol.vx`, ... re-fed into `with`). It works, and `with x = x` even works when the outer variable has the same name as the state (`drag_table.sky`), but it is manual bookkeeping for N state variables, and it restarts the adaptive step selector at every sample.
- **Expected:** `solve ... at [0 s, 0.5 s, ...]` or `every 0.5 s`, with `sol.x` then a `Vector[N] m` of samples (or `sol(t)` interpolating), so one can print a table, plot, or find a landing point by scanning.

### 4. `derivative` cannot go through a `solve`, and one `solve` in any `def` breaks every `derivative` in the program
- **Severity:** major
- **Category:** missing feature + confusing error message
- **What I tried:** (a) optimise the launch angle with Newton using `derivative(land, theta)` where `land` calls `solve`:
  ```sky
  def land(theta: rad) -> m:
      sol = solve flight from 0 s to 5 s with ..., vx = 30 m/s * cos(theta), ...
      return sol.x
  print(derivative(land, theta)(0.7 rad))
  ```
  (b) an unrelated `def sq(x: m)` differentiated in a program that also has a `def` containing a `solve`.
- **What happened:** (a) `error: cannot evaluate a derivative: differential equations are not supported by SIR yet`. (b) the same error on `derivative(sq, x)(3 m)` even though `sq` has nothing to do with the ODE (`note: derivatives run on SIR, so every def in the program must lower to SIR`). If the `solve` is at top level and not inside a `def`, (b) works.
- **Workaround:** (a) write the integrator by hand in plain `def`s (RK4 with `Vector[2]` state, landing found by a Newton sub-step) and differentiate that: `drag_ad.sky`. It works very well: exact `dR/dtheta`, Newton with a second derivative from a wrapper `def`, best angle 40.1229436 deg agrees with the golden-section-over-`solve` answer to 8 digits, identical bytes on all four engines. But that means giving up `equation` and the solver's step control. (b) keep `solve` out of `def`s when a derivative is needed.
- **Expected:** sensitivity of a solution to initial values or parameters (forward sensitivities or discrete-adjoint AD through the solver) so `derivative(land, theta)` just works; failing that, the error for (b) should be removed: only the `def`s reachable from the differentiated function need to lower, and the message should say which one does not.

### 5. Functions return one value, and an array has one unit: states with mixed units are painful
- **Severity:** major
- **Category:** ergonomics/syntax
- **What I tried:** a helper returning position and velocity after one RK4 step: `return p, v` (a tuple), or a state vector `[1 m, 2 m/s]`.
- **What happened:** `error: expected end of line, found ','` (no tuples; the message does not say so) and `error: array elements must have the same dimension: expected length, found velocity`.
- **Workaround:** two functions, `step_p` and `step_v`, duplicating the same four acceleration evaluations (`drag_ad.sky`), or a `Vector[2] m` position and a separate `Vector[2] m/s` velocity.
- **Expected:** tuple (or record) returns, `a, b = f(x)`, which every mechanics code needs the moment a step produces `(q, p)`. Mixed-unit state vectors are probably too hard to type; tuples would remove the need.

### 6. Many-body systems: `equation` has no "for each body", and arrays of vectors cannot be assembled
- **Severity:** major
- **Category:** ergonomics/syntax, missing feature
- **What I tried:** three bodies with `Vector[2]` positions. (a) six separate state variables `r1 r2 r3 v1 v2 v3` and one acceleration formula per body, written out by hand (`figure8.sky`, works, but it is 3x the code and does not extend to 4 bodies without rewriting). (b) one `Matrix[3, 2] m` of positions and one of velocities (`figure8_matrix.sky`): the state declaration is natural, and the second-order structure is recognised. But the acceleration function has to return a `Matrix[3, 2]` and:
  ```sky
  return [pull(r[0], r[1]) + pull(r[0], r[2]), ...]   # rows are Vector[2]
  ```
  gives `error: an array element must be a number, found a vector` (`help: write a matrix as rows: [[1, 2], [3, 4]]`).
- **What happened / workaround:** I had to unpack every row to scalars: `return [[a0[0], a0[1]], [a1[0], a1[1]], [a2[0], a2[1]]]`. With arrays immutable (no element writes), no `stack`/`vstack`, and no matrix comprehension, a force loop over N bodies cannot be written at all; N is fixed by hand.
- **Expected:** `[a0, a1, a2]` of equal-length vectors is a matrix (rows), or a `stack(...)`; ideally a pairwise helper such as a comprehension, or `r' = v` and `v' = accel(r)` with `accel` written with a `for` building rows. A general N-body system should be writable once.

---

## Minor

### 7. Unit literal syntax: no parentheses, no `1/m`
- **Severity:** minor
- **Category:** ergonomics/syntax, confusing error message
- **Tried / happened:**
  - `Gs = 1 m^3/(kg*s^2)` gives `error: expected a unit name after '/' in a unit literal, found '('` with `help: to divide the quantity, put spaces around '/'`. That help is wrong for what I meant. (`m^3/kg/s^2` works.)
  - `k = 0.001 1/m` gives `unexpected integer literal '1' after an expression; expected end of line`. (`0.001 m^-1` works.)
  - `def f(kk: 1/m, ...)` gives `an annotation is a type, not a value` with `help: a type is a unit like m/s ...`. `1/m` is clearly meant as a unit, so the message is confusing. (`kk: m^-1` works.)
  - `30 m/s*sin(theta)` (forgot the spaces) is read as the unit `m/s*sin`: `unknown unit sin`. The message is correct but surprising.
- **Expected:** `m^3/(kg s^2)` and `1/m` accepted in unit literals and annotations, or error messages that suggest `m^3/kg/s^2` and `m^-1`.

### 8. Rate constants: `M` (molar) is not a unit; derived units print in an odd order
- **Severity:** minor
- **Category:** missing feature / ergonomics
- **Tried:** `c0 = 2 M` gives `unknown unit M` (`help: did you mean m?`). `L` and `mol/L` work. `print(3e7 L/mol/s)` prints `30000 m^3/s/mol`, while the same thing written `m^3/mol/s` prints `m^3/mol/s`: the printed order of a derived unit varies.
- **Workaround:** `mol/L`.
- **Expected:** `M` (molar) for chemistry, or a help that says `use mol/L`; and a stable print order for derived units.

### 9. Amount-of-substance dimension prints as `N` in dimension messages
- **Severity:** minor
- **Category:** confusing error message
- **Tried:** a rate law with a wrong rate constant unit (`3e7 s^-1` instead of `m^3/mol/s`).
- **What happened:** `` `cc'` is a quantity of dimension L⁻³·T⁻¹·N, but the right side is a quantity of dimension L⁻⁶·T⁻¹·N² ``. The explanation of the error is excellent (it points at the `3e7 s^-1` line and says the constant is a frequency), but `N` here is amount of substance, and in Sky `N` is also the newton, so `N²` reads like a force squared.
- **Expected:** `mol` in dimension text (`L⁻³·T⁻¹·mol`).

### 10. `rad/s` prints as `Hz`, angles print without a unit, `rad` is dimensionless
- **Severity:** minor
- **Category:** ergonomics
- **Tried:** the double pendulum: `th1: rad`, `w1: rad/s`, `print(sol.w1)`.
- **What happened:** `velocities: 0.409 Hz -5.633 Hz`. The angle prints `-1.93` with no unit. `dR/dtheta` of a range (m per rad) prints as `-22.03 m`.
- **Expected:** this follows from "rad is dimensionless" (spec: `rad` and `deg` are dimensionless units) and `as rad/s` can force it, but an angular velocity printed in hertz looks like a bug to a physicist (a 2π mismatch). Maybe track "angle" as a display tag, or print `rad/s` when the declared type said `rad/s`.

### 11. Common physics names shadow constants: `c`, `h`, ...
- **Severity:** minor
- **Category:** ergonomics
- **Tried:** a chemist's species `c: mol/m^3` in an `equation`, and a time step `h = 0.01 s`.
- **What happened:** `warning: 'c' shadows the speed of light` / `warning: 'h' shadows the Planck constant`, printed on every run, including inside an `equation` state declaration. I renamed them (`cc`, `dt0`). Also `g` is the gram unit and `e` is undefined; `g_n` works but a student writes `g`.
- **Expected:** a warning is fair, but shadowing in a local scope (an `equation` body, a `def` parameter) is routine in physics code and should be quieter; or a per-file opt-out. (The spec says there is deliberately no way to silence warnings yet.)

### 12. `print` has no number formatting; `explain` output is one very long line per sentence
- **Severity:** minor
- **Category:** ergonomics
- **Tried:** printing a table of a trajectory.
- **What happened:** `print(x)` always writes shortest-round-trip digits (`86.4589043884601 m`). To get two decimals I used `round(x / 1 m * 100) / 100`, which also drops the unit. The explanation lines are 300-700 characters each, with the key numbers (`took 15060 fixed steps`) at the end.
- **Expected:** a way to say digits (`print(x as m, digits=3)` or `round(x, 2 m)`), and wrapped `explain` output or an `sol.steps` attribute.

### 13. `solve` is a statement, not an expression
- **Severity:** minor
- **Category:** ergonomics/syntax
- **Tried:** `return solve flight ... .x`, `x = (solve flight ...).x`.
- **What happened:** `expected an expression, found keyword solve` (`help: equation and solve start statements`). It works inside a `def` and inside a `while` (that part is good), but each call needs a temporary `sol = ...; return sol.x`.
- **Expected:** `solve` usable as an expression, so `range = solve(...).x` or `land(theta) = (solve ...).x` reads like the maths. Low priority; the workaround is two lines.

### 14. No root finding or optimisation in the standard library, and a generic Newton cannot be written
- **Severity:** minor
- **Category:** missing feature
- **Tried:** the optimal launch angle. I wrote golden-section search by hand. A function can be passed to a `def` through an unannotated parameter, so a reusable `argmax(f, lo, hi)` works (`drag.sky`; tree-walker only). But a reusable Newton step does not:
  ```sky
  def newton(f, x0: m) -> m:
      ...
          x = x - f(x) / derivative(f, x)(x)
  print(newton(g2, 1 m))
  ```
  gives `error: cannot evaluate a derivative: 'f' is not a function known at compile time` / `SIR has no function values`.
- **Workaround:** write Newton inline for each function (`drag_ad.sky`).
- **Expected:** `argmax`, `find_root` and a Newton helper in the standard library; for the last, `derivative` on a function parameter (specialised per call site, as the checker already specialises sizes).

### 15. Uncertain initial conditions: a clear error, but only at run time, and no propagation through `solve`
- **Severity:** minor (clear message, easy workaround)
- **Category:** missing feature, confusing check/run split
- **Tried:** the double pendulum with `th1 = 2.0 ± 0.01 rad`, and with an uncertain rod length `L1 = 1 ± 0.01 m` used inside the equation.
- **What happened:** `sky check` says `ok` (exit 0) for both; `sky run` stops with `error: the initial value of 'th1' must be a number, found an uncertain number` / `note: solve does not propagate uncertainties (±) yet` and a help that explains the workaround (`nominal(v)`, solve again with `nominal(v) + uncertainty(v)`). For the uncertain `L1` the error appears on the `conserve` line ("a conserved quantity must be a number"), not on the `L1` that caused it.
- **Workaround:** `double_pendulum_unc.sky`: solve at `theta` and `theta + eps` (inside a `def` with a `solve`), take the finite-difference sensitivity `s`, and attach `y0 ± abs(s) * uncertainty(a0)`. That reproduces what Sky's linear propagation would do, and it shows the chaos nicely (the sensitivity grows 233, 5900, 3.4e5, 4.4e6 at T = 4, 6, 8, 10 s, so past about 5 s the first-order uncertainty is meaningless: `theta1 = 0 ± 60`). A `Ensemble`/Monte Carlo would be the honest tool there; spec lists it as future work.
- **Expected:** `sky check` should already know that a `±` expression flows into `solve`'s `with` (the checker tracks "may be uncertain" for warnings). The real fix is `solve` propagating `±` by AD of the solution, which is the natural companion to item 4.

### 16. Equation reads names late, `def`s read them early
- **Severity:** minor
- **Category:** spec gap
- **Tried:**
  ```sky
  kk = 1 s^-1
  equation decay(t: s):
      n: mol
      n' = -kk * n
  kk = 2 s^-1
  sol = solve decay from 0 s to 1 s with n = 1 mol
  ```
- **What happened:** `sol.n` = 0.1353 mol (e^-2): the equation used `kk` as it is at the `solve`, not where the equation is written. The spec says right sides are typed with "the names in scope where the equation is written" and, for `def`s, is explicit that calls bind early. For the check (types) both agree; for values they differ. Reassigning a constant between definition and use is rare, but it is exactly what happens in a parameter sweep loop.
- **Expected:** say so in the spec. Late binding is actually what a sweep wants (`kk = ...` in a loop, then `solve`), so documenting it as intended would be fine.

### 17. README is stale on `for`, and does not say what `solve` cannot do
- **Severity:** minor
- **Category:** spec gap
- **What happened:** README says "Not there yet: `for` loops"; `for T in [1 s, 2 s, 4 s]:` and `for i in range(3)` both work (and the spec has a long section). Also the README example shows `explain(sol)` but nothing about what `solve` cannot do (events, samples), which I only found by trying.

### 18. Tolerance semantics: `tolerance` is a local-step control, not a global error
- **Severity:** minor
- **Category:** spec gap
- **Tried:** double pendulum with `tolerance 1e-10` and `conserve energy ... within 1e-8`: the energy drifted by 1.1e-9 (10x the tolerance), correctly within `within`. Figure-eight at `tolerance 1e-12` closes to 2e-8 m after one period, because my `T = 6.32591398 s` has only 9 digits. This is all as it should be, but the word "tolerance" suggests a global error bound to a student. The explanation does say "rtol" and shows what was observed, which helps a lot.
- **Expected:** one line in the spec/README: "tolerance is the per-step relative tolerance; look at the drift of a conserved quantity for the real accuracy."

---

## What was pleasant

- **An ODE inside a function just works.** `solve` in a `def`, in a `while`, with a `def` argument as the initial condition; 1600 solves for the golden-section search ran in 0.37 s. An `equation` can even be defined inside a `def` and use its parameters (`param_equation.sky`: `ln(2)/0.01 = 69.31 m`, exact).
- **`conserve` + `explain(sol)` are the best part of the language.** Reading why Dormand-Prince was chosen, that the step was limited "by accuracy for the whole interval, never by stability, so the system is not stiff", or that Yoshida was chosen for the three-body problem and the step then reduced because the trial run drifted too much, is better than any other ODE library's debugging experience. The numbers (steps, rejects, drift) are all there.
- **Stiffness is detected and reported.** Robertson (units: mol/m^3, `m^3/mol/s`, `s^-1`): `the step was limited by stability, not accuracy (the estimated h*lambda stayed near 3.42 for 15 steps, at Dormand-Prince's stability limit of about 3.3), which means the system is stiff`, then a switch to the L-stable Rodas3. At t = 40 s the results match the ten published digits of Hairer & Wanner (a = 0.7158270688 vs 0.7158270687). Mass balance drifted 8e-13.
- **Units are really checked in the right-hand sides, with good messages.** The wrong rate-constant unit error points at the line where the constant was defined and explains the dimension derivation.
- **Second-order structure is found automatically**: the pendulum and the three-body problem got a symplectic method with no annotations; the energy stayed within 6.6e-10 over ten periods (14763 steps).
- **`derivative` is exact and surprisingly capable.** Differentiating a while-loop RK4 trajectory with a data-dependent stopping step, with `Vector[2]` state, twice (Newton on dR/dtheta needs the second derivative through a wrapper `def`), converged to dR/dtheta = 0 in six iterations, identical on all four engines.
- **Uncertainty printing and `as deg`/`as ms`** make results readable; GUM rounding of `3.1 ± 2.3` is right.
- **Error messages for what Sky does not support are precise and say what to do** (`second derivatives are not supported: write the velocity as a variable of its own`, `no initial value for v ... give every state variable a starting value`, `the solution of e has no nothing; it has t, q and v`, `rtol = 1e-15 is below 1e-14, about 50 machine epsilons`).
