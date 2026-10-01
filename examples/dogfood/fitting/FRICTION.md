# Friction log: fitting models to lab data with Sky

Written as an experimental physicist who found Sky today and tried three jobs: a
Levenberg-Marquardt fit of a damped pendulum (50 points), a weighted straight-line fit with
`±`, and a mixed-units lab pipeline. Everything here was run on the build of
`origin/main` at `b18f4d6` (release build, `./target/release/sky`). Nothing in the compiler
was touched.

Programs: `fit_oscillator.sky` (vectorised model, one `derivative` per parameter),
`fit_oscillator_vec.sky` (scalar model of a parameter vector, explicit loops),
`fit_line.sky`, `units_pipeline.sky`, `stats_helpers.sky`; `gen_data.py` makes the data.
All five print identical bytes on `--engine=ast` and `--engine=auto`; the `±`-free
parts also agree to the last bit on `--engine=sir` and `--engine=native` (see entries 13, 14).

Severity: **blocker** = cannot be expressed; **major** = painful workaround; **minor** = a nuisance.

---

## A. Data in and out

### 1. There is no way to read data  (blocker, missing feature)
**Tried:** `x = read_csv("x.csv")`, `x = input()`, `import data`, `cat data.sky prog.sky | sky run /dev/stdin`, `sky run -`.
**Happened:**
```
error: cannot find `read_csv` in this scope
error: cannot find `input` in this scope
error: `import` is not supported yet   (help: there is no module loader yet; keep the code in one file for now)
error: refusing to read /dev/stdin: it is not a regular file
error: cannot read -: No such file or directory
```
`sky repl` does read stdin, so `printf 'x = [1.0 m]\nprint(x)\n' | sky repl` works, but only as a
trick (no multi-line, no file). A declared-but-unvalued input (`A: Matrix[N, N]`) is a
runtime error ("declared but has no value").
**Workaround:** a Python script (`gen_data.py`) prints Sky array literals and I pasted them
into the source (50 times and 50 angles as `0.25 s, ...` / `18.33 deg, ...`). Real lab data
(a CSV from the scope, 10^4 rows) would mean generating the `.sky` file with another tool.
**Expect:** `read_csv("run3.csv")` giving named columns (with a unit given at the call:
`read_csv("run3.csv", t: s, theta: deg)`), or at least `sky run prog.sky --input data.csv`;
and `import` so a `data.sky` of literals can live in its own file.

### 2. No random numbers, no way to build arrays in code  (major, missing feature)
**Tried:** `t = linspace(0 s, 12 s, 50)`, `arange(10)`, `append(v, x)`, `concat(v, w)`, `v[1] = 2.0`, `[v, 3.0]`, `rand()`.
**Happened:** `cannot find linspace/append/concat/rand in this scope`; `cannot assign to an element of v (arrays are immutable in v0.1)`; `an array element must be a number, found a vector`.
**Workaround:** none in Sky. The time grid and the noise were made in Python. There is also no
way to accumulate results in a loop into an array (so no residual vector, no list of fit results
per trial); everything is a scalar accumulator or a hand-written literal.
**Expect:** `linspace`, `arange`, `range(...)` as a value, array comprehension or `append`, and a
seeded `normal(mu, sigma)` so a pseudo-experiment (Monte Carlo of the fit) can be done in the language.

### 3. No statistics helpers: sum, mean, std, weighted mean, min/max of an array, sort, median  (major, missing feature)
**Tried:** `mean(y)`, `sum(y)`, `std(y)`, `min(x)`, `sort(x)`, `det(A)`, `trace(A)`, `diag(A)`.
**Happened:** `cannot find mean in this scope` (and so on); `min(x)` is `min takes 2 arguments, found 1`.
**Workaround:** `stats_helpers.sky`: `total`, `mean`, `std`, `wmean` written as loops. They take
an unannotated vector so they work for any unit, but the accumulator must start from `v[0]`
(`s = 0` would not be a length) and the loop from 1. Ten lines each time, per program, because
there is no `import`.
**Expect:** `sum`, `mean`, `std`, `var`, `median`, `min`/`max` over a vector, `dot` already exists so
`sum(w * x) / sum(w)` is the weighted mean. Unit-preserving, uncertainty-aware where relevant.

### 4. Printing tables: no formatting at all  (minor, missing feature)
**Tried:** a residual table, columns `t, data, fit, resid/sigma`.
**Happened:** `print(a, b, ...)` separates by one space, floats print with up to 17 digits, and
columns do not align:
```
0 s    18.33 deg    18.338640868171833 deg    -0.014401446953055124
0.25 s    3.48 deg    3.5175765334943145 deg    -0.06262755582385732
```
`"a" + "b"` is an error (`cannot add a string and a string`), there is no `format`, no `\t`
alignment, no way to say "4 significant digits" for a plain number. (Values *with* a `±` print
very well; plain numbers have no equivalent.)
**Workaround:** `round(x * 100) / 100` and `as` for units; accept ragged columns.
**Expect:** `print(f"{t:8.2f} {theta:8.2f}")` or a `format(x, digits)`, `round(x, 3)`; same
digit rules as the `±` printer for plain numbers.

### 5. Display and float noise  (minor, wrong result)
- `[32.45 cm, 46.05 cm, 61.34999999999999 cm, ...]`: `wire + bob` in SI then `as cm` shows the
  noise from the SI conversion (`31.2 cm + 12.5 mm`). The "shortest decimal that converts back"
  rule helps for single literals but not for sums.
- The uncertainty printer rounds on that noise: `L = 61.35 ± 0.1 cm` prints as `61.3 ± 0.1 cm`
  while `32.45 ± 0.1` prints `32.5`; the digit flips depending on which side of the tie the
  sum landed. (Correct to the float, surprising to a person.)
- `derivative(...) * 0` prints `-0 s` (negative zero with a unit). Also seen as `-0 s` in an
  untouched Jacobian column (`t = 0`).
- `k` came out as `64.88 kg/s^2`; N/m is not recognised as a printed unit unless you say `as N/m`.
  Same for `rad/s`: see entry 18.

---

## B. Arrays, loops, and broadcasting

### 6. A scalar cannot be added to or divided into a vector  (major, ergonomics / spec gap)
**Tried:** the model `A * exp(-γ * t) * cos(ω * t + φ)` over a `Vector[N]` of times;
`w = 1 / (s * s)` (weights); `x - mean(x)` (centring); `1 + x`.
**Happened:**
```
error: cannot add a scalar to a vector   (no broadcasting between a scalar and an array)
  = help: write `ω * t + φ * ones(ω * t)` for every element
error: cannot divide a scalar by a vector
error: cannot raise a vector to a power   (`^` ... help: `s * s`)
```
The scalar-times-vector case is fine (`2 * v`), but `+`/`-` with a scalar and `1 / v` are
errors, and these are the most common operations in a weighted fit.
**Workaround:** for the phase, the angle-sum identity `cos(ωt)cos(φ) - sin(ωt)sin(φ)` (a physicist
who does not remember it is stuck); for weights, loop over the data; for "ones" I found
`exp(zeros(3))`.
**Expect:** a scalar broadcasts across a vector for `+ - /` (it is what every other
scientific language does, and the dimension rules are unambiguous: same dimension as the vector).
At minimum `ones(n)`.

### 7. The help text above names a function that does not exist  (minor, confusing/wrong error message)
The help in entry 6 says ``write `ω * t + φ * ones(ω * t)` ``. There is no `ones`
(`cannot find ones in this scope`; the same advice appears for `1 + x`). Following the
compiler's own suggestion fails.
**Expect:** either add `ones` or change the suggestion (e.g. `exp(zeros(n))`, ugly) / fix by 6.

### 8. `zeros(len(t))` is rejected, so a helper cannot be generic in the vector length  (major, spec gap)
**Tried:** inside `def model(..., t: Vector[N] s) -> Vector[N]`, `zeros(len(t))` for a ones-vector.
**Happened:** ``error: `zeros` needs a constant size, found `len(t)` (an array's size is part of its type)``.
So the workaround of entry 6 (`exp(zeros(50))`) forces a hard-coded size and kills generic code.
**Expect:** `zeros_like(t)` / `ones_like(t)`, or `zeros(N)` with the size variable `N` in scope.

### 9. Size variable and value name clash  (minor, ergonomics)
`npts = 50` is fine, but `N = 50` followed by `def f(t: Vector[N] s)` is
`error: size variable N has the same name as a value in scope`. Natural for a physicist to
write `N = len(t)`. Also `Vector[N]` cannot use a value `N`. Renamed to `npts`.

### 10. `v[i]` in a loop is O(n): quadratic on the default engine  (major, performance)
**Tried:** `v = zeros(40000)`, `for i in range(n): s += v[i] + 1.5`.
**Measured (`sky run`, i.e. `--engine=ast`):** 20 000 entries 0.066 s, 40 000 entries 0.216 s
(x3.3 for x2: quadratic), 1 000 000 entries: still running after 2 minutes (killed).
`for x in v:` is linear (1e6 in 0.3 s) and `--engine=sir` / `--engine=native` index in linear time (40 000 in 0.02 s).
The default engine is the slow one, and index loops are exactly how sums over data are written
(entry 3). Not a problem at 50 points, a cliff at 10^5.

### 11. Loops over arrays do not run on SIR/native, and the refusal message is wrong  (minor, confusing message / performance)
`for x in v:` under `--engine=native` says:
```
error: cannot run natively: iteration over arrays lowers once SIR has array values
  = note: `sky run` cannot execute this either: arrays are immutable and attributes are not supported in v0.1
```
The last note is false (`sky run` executes it fine on the default engine). `for i in range(len(v)): v[i]`
does lower and run natively (checked), so there is an escape, but a person reading the message
concludes the program is broken.

### 12. Functions return one value; no tuples, no structs  (major, missing feature)
**Tried:** `def f(a: m, b: s) -> m: return a, b` -> `expected end of line, found ','`.
**Consequence:** a fit routine cannot return `(params, covariance, chi2)`. A weighted line fit
returns intercept (mm) and slope (mm/g): different units, so they cannot even be packed into a
`Vector`. I kept everything as top-level globals and wrote two functions' worth of code inline.
**Expect:** tuple returns (`a, b = fit(...)`), or named records `fit.slope`, `fit.cov`.

---

## C. Jacobians, `derivative`, `grad`, matrices

### 13. `derivative(model, ...)` on a size-generic function fails to lower to SIR  (major, bug)
**Tried:** `def model(..., t: Vector[N] s) -> Vector[N]` with `derivative(model, A)(...)`, then
`--engine=sir` or `--engine=native`.
**Happened:**
```
error: cannot run on SIR: lowering produced SIR that does not verify
  = note: result 0 of the call to `dmodel/dA` is num[1; (N#0)], but it returns num[1; (50)]
```
(on `--engine=ast` and `auto` it runs, by falling back.) With `Vector[50]` written out everywhere,
SIR and native run it and give the same bytes. The message leaks an internal verifier failure.
**Expect:** generic-size functions differentiate on every engine.

### 14. Any use of `±` takes the whole program off SIR/native  (minor, performance / ergonomics)
`--engine=sir` / `native` refuse the entire file as soon as one `±` appears
(`cannot run on SIR: uncertainty (±) is not supported by SIR yet`), though the Gauss-Newton loop
itself has no `±`. `auto` silently falls back to the tree-walker. For a fit that ends in
`A ± sqrt(C[0,0])` I have to remove the last lines to compare engines (I did: same bits on all four).
**Expect:** `±` lowered where used, or at least per-function granularity.

### 15. No Jacobian: derivative of a vector-valued model with respect to a parameter vector is an error  (major, missing feature)
**Tried:** `derivative(model, p)([1.0, 0.5], t)` for `model(p: Vector[2], t: Vector[3] s) -> Vector[3]`.
**Happened:**
```
error: cannot differentiate `model` with respect to `p`: the derivative of an array with respect to an array is a Jacobian
  = note: ... Sky does not have Jacobians yet
```
(An honest message, which I appreciate.) The model of a fit *is* vector-valued over the data.
**Workaround:** two styles, both used:
- `fit_oscillator.sky`: parameters as separate scalars, one `derivative(model, A)(...)`, `(model, γ)`, ... per
  column, each re-evaluating the whole model (4x the cost, and each call repeats the full argument list);
- `fit_oscillator_vec.sky`: scalar model `f(p: Vector[4], t: s)`, `derivative(f, p)(p, t[i])` is a
  gradient (a Jacobian row), accumulated point by point in a loop.
**Expect:** `jacobian(model, p)(p, t)` giving a `Matrix[N, 4]`; at the least `derivative(model, p)` for
a vector-valued model when `p` and the result are both arrays.

### 16. Parameters with different units cannot form a Jacobian / `grad` / normal matrix  (major, spec gap)
**Tried:** `grad(f)(A, γ, ω, φ)` where `A` is dimensionless, `γ` is Hz, `ω` is rad/s; and a
4x4 normal matrix `[[dot(j0,j0), dot(j0,j1), ...]]`.
**Happened:** ``cannot take the gradient of `f`: its parameters have different units`` (`A` dimensionless,
`γ` frequency, `t` time), and a matrix literal needs one dimension for all entries; `∂f/∂γ` has unit `s`
while `∂f/∂A` has none. This is *correct* dimensional analysis, and the compiler's help (take one
derivative per parameter) is right, but it makes the Gauss-Newton normal matrix inexpressible with units.
**Workaround:** multiply each Jacobian column by its parameter's unit so it is dimensionless:
`j1 = derivative(model, γ)(...) * (1 Hz)`, step `δ` is a plain vector,
and apply it back as `γ + δ[1] * (1 Hz)`. The model keeps its unit checks, but this
scaling is written by hand for every parameter in three places (Jacobian, update, covariance
output), and I found no error if I forget one (the units *do* catch some of it).
**Expect:** a covariance type that carries per-entry units (`Matrix` with row/column units), or a
helper that scales parameters to dimensionless for the solver and back.

### 17. No outer product; `Jᵀ @ J` is not known to be symmetric, so no Cholesky for the normal equations  (major, missing feature / spec gap)
**Tried:** `g @ transpose(g)` (outer product), then `K: Symmetric Matrix[2, 2] = Jᵀ @ J`, and
`H: PositiveDefinite Matrix[2, 2] = [[dot(j0, j0), ...]]`.
**Happened:**
```
error: cannot transpose Vector[2]: transpose needs a matrix  (vectors have no orientation; ...)
error: `Jᵀ @ J` is not known to be Symmetric   (help: declare the Symmetric property on an input the value is computed from)
error: ... is not known to be PositiveDefinite   (every entry must be a numeric literal)
```
`outer(a, b)` does not exist. The Gram matrix `JᵀJ` is the standard case of a matrix that is
symmetric positive (semi-)definite *by construction*, and Sky's headline feature is exploiting that
(Cholesky). It does not know it; `--explain` rewrites my `(H + λD)⁻¹ @ g` to `solve[lu]`.
The "declare an input" advice cannot be followed because inputs have no run-time value (entry 1).
**Workaround:** `fit_oscillator_vec.sky` has a 16-entry `outer(g)` literal (`g[0]*g[0], g[0]*g[1], ...`) for
exactly 4 parameters; it does not scale to 10. In `fit_oscillator.sky` the matrix is a 16-entry literal of `dot`s.
**Expect:** `outer(a, b)`; `Aᵀ @ A` known `Symmetric` (and PD for a full-rank `A` with a run-time Cholesky
check, as the checker already does for inputs); a `Symmetric`-preserving builder from a
bilinear form.

### 18. No function values: a reusable `lm(model, data, p0)` cannot be written; the failure appears only at run time  (major, spec gap; and minor, wrong error timing)
**Tried:**
```
def step(h, p: Vector[2], t: s) -> Vector[2]:
    return derivative(h, p)(p, t)
print(step(f, [1.0, 2.0], 1 s))
```
**Happened:** `sky check` says `ok`; `sky run` says
```
error: cannot evaluate a derivative: only functions defined with `def` can be called
  = note: derivatives run on SIR, so every `def` in the program must lower to SIR
```
(passing `f` as an argument and calling `h(p, t)` works on the default engine and fails on `--engine=sir`,
"SIR has no function values".) So the solver must be pasted into each program with the model
named literally in the `derivative` calls: a generic Levenberg-Marquardt library is not possible, and
the check passes a program that can never run.
**Expect:** `derivative(h, p)` with `h` a function argument, or at least a check error where `h` is
not a known `def`; and the second note should say what is actually unsupported.

### 19. `derivative(f, x)` helpers read poorly for dimensionless functions  (minor, ergonomics / wrong help)
- Every parameter must carry a unit annotation, including dimensionless ones: `A: Scalar`. The help for an
  unannotated `A` says "annotate the parameter with its unit, for example `A: s`" (a time for an amplitude); it
  should say `Scalar` for the dimensionless case.
- A function whose return is annotated `Vector[N]` but whose body has an earlier error reports
  "the unit of its result is not known" on every `derivative` call: a cascade that hides the real error
  (`zeros` needs a constant size, line above it). I fixed the first error and the rest vanished.
- A derivative cannot be named (`dA = derivative(model, A)` is an error, spec: "must be called where it is
  written"), so the four-column Jacobian repeats `(A, γ, ω, φ, t)` four times.

---

## D. `±` with arrays, loops, functions, derivatives

### 20. Arrays cannot carry uncertainties; an uncertain scalar does not combine with a vector  (major, missing feature)
**Tried:** `[1 m ± 0.1 m, 2 m]`, `d ± 0.1 m` with a vector `d`, and a fitted model evaluated at an uncertain
parameter: `A = 1.0 ± 0.1`, `model(A, 0.2 Hz, t)` with `t: Vector[2]`.
**Happened:**
```
error: an array element must be a number, found an uncertain number
error: `±` needs scalar operands  (arrays cannot carry an uncertainty yet; help: write ± on each element you need)
error: an uncertain number does not combine with a vector     <- at run time, inside the model
```
**Consequence:** the "error band" of the fitted curve (`model(p ± σ)` over 50 times), propagation
of data noise through a vectorised formula, the covariance of fitted parameters as correlated `±` values: none
of it works. `σ` per point in a data table cannot be a vector of `±` readings.
**Workaround:** scalars in a loop (`yi = L[i] ± sL[i]`) as in `fit_line.sky`; for the oscillator I took
`C = H⁻¹` by hand and wrapped the final numbers with `A ± sqrt(C[0, 0])` (independent
errors: the parameter correlations are lost in the `±` values, so `A_fit * γ_fit` would be wrong).
**Expect:** `±` on arrays (the spec lists this as "not in this step", fine, but it is the biggest
hole for this domain); a multivariate constructor `params = with_covariance(p, C)` so correlated parameters
propagate through `model(params, t)`.

### 21. `derivative` rejects uncertain arguments, with a message that points at the wrong place  (major, confusing error message / bug)
**Tried:**
```
def f(A: Scalar, γ: Hz, t: s) -> Scalar:
    return A * exp(-γ * t)
x = 0.5 Hz ± 0.01 Hz
print(derivative(f, γ)(2.0, x, 1 s))
```
**Happened:** a run-time error on line 2, `` `-` needs a number, found an uncertain number `` pointing at `-γ`
inside `f`, with no mention of `derivative` or that `±` is the cause. (`f(2.0, x, 1 s)` alone gives
`1.213 ± 0.012`; `derivative(f, γ)(2.0, 0.5 Hz, 1 s)` alone works.)
**Workaround:** strip with `nominal(x)` before differentiating, and multiply by `uncertainty(x)` by hand
(first-order propagation, which is what `±` does anyway).
**Expect:** a derivative evaluated at an uncertain point has an uncertainty (`f'(x ± dx)`), or a check
error ("`derivative` needs exact arguments: use `nominal(x)`") instead of a runtime error inside the callee.

### 22. `±` through loops and sums: works, and the correlation handling is excellent  (not a friction; see "pleasant")
Recorded because it was a pleasant surprise: accumulating `y_i ± σ_i` in a `for` loop gives the covariance-correct
slope and intercept uncertainty (matches the closed-form formulas to 1e-9, asserted in `fit_line.sky`).

### 23. `round`, comparisons and `==` on uncertain values  (minor, spec corner)
`round(6.00 ± 0.17)` is `6 ± 0` (slope 0 by spec: constant between jumps), `a == 1.0` compares the
nominal value only. The comparison has a good warning with the right advice; `round` silently dropping the
uncertainty is surprising for a number that is about to go in a table.

---

## E. Units and the lab pipeline

### 24. Angles are dimensionless; `rad/s` prints as `Hz`  (major, wrong result / spec gap)
**Tried:** `ω = 3.9 rad/s`, `print(ω)`; `theta = [18.33 deg, ...]`, `print(theta[0])`.
**Happened:** `3.9 Hz` (an angular frequency of 3.9 rad/s labelled as a frequency; as a *frequency* it would be
0.62 Hz, a 2π error) and `0.3199...` for an angle (radians, no unit, as it must be if `deg` is
dimensionless). `as rad/s` and `as deg` bring them back. The spec admits this ("Hz and rad/s share a dimension").
For a pendulum fit the mislabel is very easy to read as physics: my decay rate `γ` and angular frequency `ω` both print
as `Hz`; I print `ω_fit as rad/s` explicitly.
**Expect:** `rad` as a (dimensionless-compatible) tagged unit so that plain printing keeps `rad/s`; or print
angular quantities with `rad`.

### 25. Missing common lab units  (minor, missing feature)
`inch` (`unknown unit inch`) and `rpm` (`unknown unit rpm`) are not units; also `°C` (out of scope per spec). Mixed-unit
data from a lab will contain some of these.

### 26. Raw columns of bare numbers: attaching units is clean  (not a friction; see "pleasant")
`wire = wire_cm * (1 cm)` and `t20 = t20_ms * (1 ms)` convert a whole bare vector in one expression.

### 27. Constant names shadow easily; `solve` is a keyword  (minor, ergonomics)
`G`, `h`, `c`, `g_n` are predefined: calling a Gram matrix `G` or a step size `h` gives a warning
(`G shadows the Newtonian constant of gravitation`; fine, a warning). But `solve` is a keyword (`solve(A, b)` is
`expected an expression, found keyword solve`), the one name every linear-algebra user reaches for. `e` is
undefined (`exp(1)`).

### 28. Negative indices and slices of arrays  (minor, missing feature)
`x[-1]` is an error ("v0.1 counts from 0 and has no negative indices"); there is no `last(x)` or `x[len(x) - 1]`
shortcut other than the explicit one, which works.

---

## Tally

27 frictions: 1 blocker (entry 1), 14 major (2, 3, 6, 8, 10, 12, 13, 15, 16, 17, 18, 20, 21, 24),
12 minor (4, 5, 7, 9, 11, 14, 18b, 19, 23, 25, 27, 28). No crash was found, and the engines never
disagreed on a program they both accepted.

---

## What was pleasant

- **The unit checker caught my mistake immediately and explained it.** In `units_pipeline.sky` I accumulated
  `w * g` into an accumulator I had written as `0 m/s^2`; the error showed the dimensions of both sides,
  where each came from, and the hint `did you mean g_sum * w`. It was a genuine bug in my weighted mean.
- **`±` is a joy for propagation.** `L = (wire[i] ± 1 mm) + (bob[i] ± 0.05 mm)`, `T = ... / 20`, `g = 4π²L/T²`
  and then `gi as m/s^2` printed `9.85 ± 0.05 m/s^2` with GUM-style rounding; the shared-source correlations make
  `x - x = 0` and make the slope/intercept correlation in `fit_line.sky` come out right with no covariance code
  (prediction at 325 g: `0.254 mm` by `±`, identical to the covariance formula, versus `0.535 mm` if you wrongly
  ignore the correlation).
- **`derivative`/`grad` are exact and the units are right.** `derivative(model, γ)` has unit `s`, `j * (1 Hz)`
  is a plain number; the Jacobian had no step-size tuning and the fit converged (well inside 40 iterations) from a poor
  guess (chi2 5174 -> 29.6). The two formulations of the oscillator fit agree to 1e-15 and match the truth
  (A = 19.9 ± 0.4 deg vs 20, ω = 3.935 ± 0.005 vs 3.93, γ = 0.180 ± 0.005 vs 0.18).
- **Engines agree.** Every program (the parts with no `±`) printed the same bytes on `ast`, `sir`, `native`, and `auto`
  (checked by `md5` on the full output, including the residual table). The 50-point LM fit runs in ~13 ms.
- **`as` for output is intuitive** (`t[i] as ms`, `(A ± dA) as deg`, `k as N/m`), and `sky check --explain` told me
  why my solve was an LU, with a flop estimate.
- **Honest error messages for what is missing**: "Sky does not have Jacobians yet", "arrays cannot carry an
  uncertainty yet", with the workaround in the help. I knew immediately what to do next.
- **A generic helper over any unit works with an unannotated parameter** (`def mean(v)` gave `9.813 m` for lengths and
  `1 s` for times), which made the missing `mean`/`std` bearable.
