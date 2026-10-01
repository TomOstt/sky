# Sky first-time-user friction log

Persona: a scientist who knows Python and NumPy, has read only `README.md`, and
learns the rest from error messages and `sky --help`. Built from
`origin/main` at `b18f4d6`, release build (`cargo build --release -p sky-cli`
took 2 min 40 s from cold). The compiler was not modified.

Programs (each runs, with the expected output next to it):

| Program | What it does | Time to a working version (judgement) |
|---|---|---|
| `units.sky` | km/L to mpg, kWh to J and back, recipe scaling | about 10 min: two compile rounds, then I discovered `mi`, `gal`, `lb`, `kWh` do not exist |
| `stats.sky` | mean, sd, sem, min, max, text histogram | about 10 min: `sum`/`min`/`max` on arrays missing, loops fixed it; a further 10 to make output tidy |
| `montecarlo.sky` | pi by darts, A -> B -> C decay chain | about 15 min: no RNG, so I wrote an LCG. Then 10 more to find out why it was slow |
| `planets.sky` | planet table, surface gravity and density | about 15 min: no records or string arrays, no padding, so alignment is by tabs and still ragged |
| `errors/e01..e10_*.sky` | ten deliberate mistakes | about 15 min in all |

Times where I had to leave the README and open `spec/v0.1.md` (5 times, each
because the README or an error left me stuck):

1. `mi` unknown, and no unit list anywhere in README or `--help`. Spec line 18 has it.
2. `min(lengths)` rejected: "takes 2 arguments". The README never says what builtins exist; spec "Math functions" table.
3. No `f"..."`, `str`, `format`. Spec "Values and printing" says strings are only printed (no formatting).
4. `linspace`, `append`, etc. missing. Spec "Arrays: indexing, builtins" is the only list of array builtins.
5. How to write a dimensionless annotation (`-> 1` fails). Spec "Type annotations" says `Scalar`.

Where the README failed me: it lists `for` loops as "Not there yet" (they work,
over `range` and over arrays); it never lists units, builtins, string rules or
how to build an array; the example snippets never show a loop that
accumulates, which is the thing a Python user reaches for first.

---

## Friction entries

Severity counts: **1 blocker, 10 major, 14 minor** (25 entries).

### F1. Arrays cannot be built at run time
- **Severity:** blocker. **Category:** missing feature.
- **Tried:** `linspace(0, 1, 5)`, `arange(5)`, `ones(3)`, `append(v, x)`, `concat`, `stack`, `reshape`, `sort`, a list comprehension `[i*2 for i in range(3)]`, and `v[0] = 5 m` in a loop.
- **Happened:** each of the named functions is "cannot find `x` in this scope"; the comprehension is the parse error "expected `,` or `]`, found keyword `for`"; element writes are "arrays are immutable in v0.1". Only literals, `zeros(n)` and `identity(n)` make arrays.
- **Workaround:** none for storing a series. You can only hold a scalar accumulator in a loop. (Parallel arrays must be typed out as literals.)
- **Python user expects:** `np.linspace`, `np.arange`, `np.ones`, `append`, comprehensions, `x[i] = ...`. A simulation that stores its trajectory, or a histogram that fills bins, cannot be written.

### F2. No random number generator
- **Severity:** major. **Category:** missing feature.
- **Tried:** `random()`; grepped README, spec and ROADMAP for random/rand/seed (nothing).
- **Happened:** "cannot find `random` in this scope".
- **Workaround:** a Park-Miller LCG written in Sky (`a = 16807`, `m = 2^31-1`, products stay below 2^53 so f64 is exact), state threaded by hand since there are no multi-value returns. It works: pi to 3.135 from 200 000 darts, and the decay-chain counts match the Bateman solution within about 1 percent.
- **Python user expects:** `random.random()`, `np.random.default_rng(seed).normal(...)`, seeded and reproducible.

### F3. No reductions: `sum`, `mean`, `std`, `min`/`max` of an array
- **Severity:** major. **Category:** missing feature.
- **Tried:** `sum(x)`, `np.sum(x)`, `mean`, `std`, `min(lengths)`, `max(lengths)`, `cumsum`, `sort`.
- **Happened:** `sum` and the rest are "cannot find ... in this scope". `min(lengths)` is "`min` takes 2 arguments, found 1 ... call it as `min(a, b)`", which is correct but a dead end (the help does not say there is no reduction form).
- **Workaround:** `for x in v:` loops with an accumulator, and `lo = min(lo, x)`. `v @ ones` is not possible (no `ones`), so even a sum by dot product needs a literal.
- **Python user expects:** `sum`, `min`, `max`, `len` on a list (`len` exists), `statistics.mean`, `np.std`.

### F4. `ft` silently means femtotonne, so `3 ft + 2 m` is a mass error
- **Severity:** major. **Category:** bug (a trap in the unit table), also error message.
- **Tried:** `3 ft + 2 m`, then `1 ft` in the REPL.
- **Happened:** the unit parses as `f` (femto) + `t` (tonne). The error was "cannot add mass and length" with no hint; in the REPL `1 ft` echoes `1e-12 kg`, with no warning. `1 ft` and `1 min`, `1 cm`, `1 mg` all check cleanly, so there is no signal that `ft` is not a foot. (`1 mi`, by contrast, is rejected, so `mi`/`ft` look inconsistent.)
- **Workaround:** `foot = 0.3048 m` and write `3 * foot`.
- **Python user expects:** either feet, or "unknown unit `ft`". A prefix-plus-unit match that yields an absurd unit should at least be a note ("read as femto-tonne").

### F5. Many everyday units are missing
- **Severity:** major. **Category:** missing feature.
- **Tried (each "unknown unit"):** `mi`, `inch`, `lb`, `oz`, `gal`, `l`, `kWh`, `Wh`, `cal`, `kcal`, `bar`, `atm`, `psi`, `mmHg`, `Torr`, `mph`, `kn`, `degC`, `hr`, `hour`, `d`, `week`, `year`, `tonne`, `ha`, `Å`, `Da`, `u`, `amu`, `dB`, `byte`, `bit`, `Bq`, `Gy`, `lm`. Also the characters `°` and `%` abort the lexer. Present: `km`, `cm`, `mm`, `L`, `mL`, `min`, `h`, `day`, `yr`, `ly`, `eV`, `deg`, `rad`, `mol`, `t`.
- **Happened:** units with a near miss get "did you mean `h`?" (for `hr`), others get nothing. The error never says where the list of units is, and no list is printed by `--help`, README, or any subcommand.
- **Workaround:** named constants: `mile = 1609.344 m`, `gallon = 3.785411784 L`, `kWh = 1000 W * 1 h`. See F6 for the cost.
- **Python user expects:** `pint`-style `ureg.mile`, `ureg.kWh`, `ureg.psi`. For fuel economy and electricity, which the README's own audience asks for first, `mi`, `gal`, `kWh` are the obvious ones.

### F6. User-defined quantities cannot be used as units
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** after `mile = 1609.344 m`, writing `5 mile`, and `x as mile`.
- **Happened:** a unit may only follow a number and must be in the built-in table ("unexpected name `mile` after an expression; a unit may only follow a numeric literal", then the help says multiply explicitly). The message is good, but it means `5 mile` is `5 * mile` and `as mile` is impossible: you divide instead (`x / mile`) and the result is a plain number, not tagged.
- **Workaround:** `5 * mile`, `x / mile`.
- **Python user expects:** a way to define a unit once and use it everywhere, including `as`.

### F7. `in` is a keyword, so inches cannot be a unit
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** `1 in`.
- **Happened:** "expected end of line, found keyword `in`". Even if `in` were a unit it would conflict with `for i in ...`.
- **Workaround:** `inch = 0.0254 m`.
- **Python user expects:** something. `inch` and `inches` would be the way out; both are unknown units too.

### F8. No string formatting, no padding, no print-without-newline
- **Severity:** major. **Category:** missing feature.
- **Tried:** `f"x = {x}"`, `str(x)`, `format(x, ".2f")`, `"%.2f" % x`, `"a" * 3`, `"a" + "b"`, `len("hello")`, `print(..., end="")`.
- **Happened:** f-string: "unexpected string literal after an expression; ... is an operator missing?" (misleading, see F20); `str`, `format`: not defined; `%`: "unexpected character '%'"; `"a" * 3` and `"a" + "b"`: "cannot multiply/add a string and a string"; `len("hello")`: "`len` needs a vector or a matrix, found a string". `print(a, b)` always joins with one space and ends with a newline.
- **Workaround:** `\t` works inside string literals, so tab-separated columns are possible; numbers are rounded by `round(v * 100) / 100`; the histogram bar is picked by an `if` ladder of literal strings (`"#"`, `"##"`, ...) because a bar cannot be built.
- **Python user expects:** `f"{x:8.2f}"`, `str.ljust`, `"#" * n`, `end=""`. The planet table (`planets.out`) shows the result: columns are ragged because `print` inserts a space around every tab.

### F9. Printing is SI-only and shows float noise
- **Severity:** major. **Category:** ergonomics/syntax.
- **Tried:** `print(lengths[0])` where the element was `10.2 mm`, and the mean and sd of the 16 lengths.
- **Happened:** `0.010199999999999999 m`, and `sd = 0.24999999999999983 mm` even with `as mm`. Values are stored in SI, so `10.2 mm` is `0.0102` in f64 and round-trips with noise. `print(x as mm)` puts the right unit on but not a sensible number of digits.
- **Workaround:** `def r2(v: mm): return round(v / (1 mm) * 100) / 100` and print the number plus the literal `"mm"`. This works but loses the unit check at the print.
- **Python user expects:** `print(f"{x:.2f}")` or `x.to("mm")` printing `10.2 mm`. Scientists see this on the first program they run. The unit-aware `as` display is the right idea; it needs a precision (or a shortest-repr that does not go through SI).

### F10. No records, dicts, tuples, arrays of strings, or multiple return values
- **Severity:** major. **Category:** missing feature.
- **Tried:** `["Mercury", "Venus"]`, `{"name": "Earth", "mass": 5.972e24 kg}`, `("Earth", 5.972e24 kg)`, `return a, a + 1`.
- **Happened:** "an array element must be a number, found a string"; "unexpected character '{'"; "expected `)`, found `,`"; "expected end of line, found `,`". The last two do not explain that tuples do not exist.
- **Workaround:** parallel numeric arrays plus a `def name_of(i)` ladder of `if`s for the names; generator state threaded by passing and returning one number. The mini data table (task 4) is written like this.
- **Python user expects:** a list of dicts (or a dataclass) for the table, and tuple returns for anything with state.

### F11. The README is stale and incomplete
- **Severity:** major. **Category:** docs/discoverability.
- **Tried:** following the README's "Not there yet" list.
- **Happened:** it says "`for` loops and `import`" are not there; `for x in array` and `for i in range(n)` work on the default engine (spec has a whole "Blocks and control flow" section; the benchmarks use `for`). It lists array writes and negative indices as missing, which is right, but never says what builtins, string operations, or units exist. A first reading suggests `for` is unusable; I only tried it because the showcase uses `while`.
- **Workaround:** read the spec.
- **Python user expects:** a short "Sky for Python users" page: what exists, what does not (list, dict, string formatting, sum), and the idiom for each.

### F12. `sky --help` documents compiler flags only; no language reference, `--version`, or unit list
- **Severity:** major. **Category:** docs/discoverability.
- **Tried:** `sky --help`, `sky help`, `sky units`, `sky --version`, `sky run` with no file.
- **Happened:** all four print the same usage block (the unknown words are not treated as errors; `sky units` and `sky --version` both just print usage). The usage is about IR stages and Cranelift. Nothing mentions that `sky repl` exists with `:help` unless you read the usage line; nothing lists units or builtins.
- **Workaround:** the spec.
- **Python user expects:** `sky --version`, and at least `sky units` / `sky builtins`, or a `:units` command in the REPL.

### F13. Functions without annotations run 90x slower on the fast engine, with no hint; the default engine is the slow one
- **Severity:** major. **Category:** ergonomics/syntax (performance).
- **Tried:** the Park-Miller step as `def next_state(s):`, called 200 000 times, with `sky run` (default `ast` engine), `--engine=auto`, `--engine=native`.
- **Happened:** 0.69 s, 0.73 s, 0.74 s: native gave no speed-up. The same step inlined in the loop is 0.017 s native (70x faster than ast at 1.2 s). Only after trying `def next_state(s: Scalar) -> Scalar:` did the function call run at 0.018 s. Nothing in a message, warning or `--explain` says that an unannotated parameter keeps the whole call on a slow path.
- **Workaround:** annotate every function with `Scalar` or a unit. (The README's "over 100 times faster" is for scalar loops; a user who wrote a helper function will not see it.)
- **Python user expects:** that a helper function is as fast as inline code. At least `sky check --explain` could say "`next_state` has untyped parameters and runs on the slow path".

### F14. `%` and `//` do not exist, and there are no integers
- **Severity:** minor. **Category:** missing feature.
- **Tried:** `10 % 3`, `7 // 2`, then `range(7 / 2)`.
- **Happened:** "unexpected character '%' (U+0025)" and "expected an expression, found `/`". `7 / 2` is `3.5` everywhere (a good choice). `range(7 / 2)` is a nice error ("`range` needs an integer ... round it first"); but `n = 10; range(n / 2)` is accepted, because 5.0 is integer-valued: the check depends on the value, which is a surprise.
- **Workaround:** `x - floor(x / m) * m` for modulo; `floor(a / b)` for integer division.
- **Python user expects:** `%` and `//`. The messages never mention `floor`.

### F15. A syntax error hides all other errors; the lexer error hides everything
- **Severity:** minor. **Category:** error message.
- **Tried:** a file with several different mistakes.
- **Happened:** a single lexer error (`%`, `°`, `{`) is the only message; the first parse error is the only message; type errors come together only when the file parses. To find all unknown units I had to delete lines one at a time.
- **Workaround:** fix one at a time.
- **Python user expects:** about the same as Python (first syntax error only), but the lexer error for `%` should not stop the checker from reporting a missing unit on line 3.

### F16. Unknown names get no "did you mean"
- **Severity:** minor. **Category:** error message.
- **Tried:** `print(velocty)` after `velocity = ...`; `sqroot(4)`.
- **Happened:** "cannot find `velocty` in this scope: not defined". Unknown units do get "did you mean `h`?" (so the machinery exists).
- **Workaround:** none needed, but a one-character typo takes a re-read.
- **Python user expects:** Python 3.10+ says "Did you mean: 'velocity'?".

### F17. `e` and `math.pi` / `np.pi`
- **Severity:** minor. **Category:** error message.
- **Tried:** `e^2`, `1e3 + e`, `math.pi`, `np.pi`.
- **Happened:** "cannot find `e` in this scope: not defined". `math` and `np` likewise. `pi` and `π` exist. The spec says "`e` is not defined; write `exp(1)`", but the error does not.
- **Workaround:** `exp(1)`.
- **Python user expects:** the error to say "Euler's number is `exp(1)`", and `pi` (not `math.pi`) to be called out in the README.

### F18. `x.mean()` error uses implementation words
- **Severity:** minor. **Category:** error message.
- **Tried:** `x.mean()` on a vector (a NumPy habit).
- **Happened:** "attribute `mean` is not supported by the interpreter: a vector has no attributes". "By the interpreter" suggests an engine limitation that a later version may lift; the reality is that attributes are reserved for later.
- **Python user expects:** "no methods; use ... (there is no mean yet)".

### F19. Immutable arrays: error gives no alternative
- **Severity:** minor. **Category:** error message.
- **Tried:** `v[0] = 5 m`, `v[1] += 1 m`.
- **Happened:** "cannot assign to an element of `v` ... arrays are immutable in v0.1". No help line, and the README lists "array writes" as not there yet only in the status list.
- **Workaround:** none (see F1).
- **Python user expects:** a hint such as "build a new array with a literal" (which does not exist either, so the honest message is "cannot be done yet").

### F20. An f-string gets the "missing operator" help
- **Severity:** minor. **Category:** error message.
- **Tried:** `print(f"x = {x}")`.
- **Happened:** "unexpected string literal after an expression; expected `,` or `)` ... two expressions cannot be written side by side; is an operator missing?" The cause is that `f` is read as a name.
- **Python user expects:** "f-strings do not exist; there is no string formatting".

### F21. Assigning to a global name inside a function silently makes a local
- **Severity:** minor. **Category:** bug (semantics trap).
- **Tried:** `counter = 0; def bump(): counter = counter + 1; return counter`.
- **Happened:** `bump()` returns 1 each time and `counter` stays 0. No warning. Python raises `UnboundLocalError`, which at least tells you.
- **Workaround:** pass and return state.
- **Python user expects:** an error or a warning ("`counter` is read before it is assigned in this function; assignment makes a local").

### F22. How to annotate a dimensionless value is not in the error
- **Severity:** minor. **Category:** error message.
- **Tried:** `def r2(v: mm) -> 1:`.
- **Happened:** "an annotation is a type, not a value ... a type is a unit like `m/s` or a shape like `Matrix[3, 3]`". The word `Scalar` (the dimensionless type) is not mentioned.
- **Workaround:** drop the annotation (then see F13), or `Scalar` after reading the spec.

### F23. The checker does not type strings, so errors appear at run time after output
- **Severity:** minor. **Category:** bug.
- **Tried:** `len("hello")` and `"a" + "b"` in a file with earlier prints.
- **Happened:** `sky check` says ok; `sky run` prints the earlier lines, then fails ("`len` needs a vector or a matrix, found a string").
- **Workaround:** none needed.

### F24. Volumes and ratios print in SI even when written in L
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** `print(1 L)` and 7.5 L/100 km (European fuel consumption).
- **Happened:** `0.001 m^3`, and `7.5e-8 m^2` (litres per distance is an area). Correct, but a reader sees "m^2" for a fuel figure. `as L/km` prints `0.075 L/km`, which is what the user wanted.
- **Workaround:** always use `as`.
- **Python user expects:** a quantity to print in the unit it was written in (`pint` does).

### F25. Warnings for ordinary variable names (`c`, `h`, `G`)
- **Severity:** minor. **Category:** ergonomics/syntax.
- **Tried:** `c = np.sum(x)`, `h = 0.1 s` (a step size), `G = 5`.
- **Happened:** "warning: `c` shadows the speed of light", `h` the Planck constant, `G` the gravitation constant. Reasonable and the help tells you how to avoid it, but `c` and `h` are the first names a user reaches for (counts, step size). `k`, `g`, `R`, `N`, `m`, `q`, `mu` did not warn, so the rule is inconsistent for a new user (`g` is not a constant; `g_n` is). The units `h` and `m` also work as variable names without warning, and `m = 3 kg; print(2 m)` still means metres.
- **Workaround:** other names.

---

## The 10 deliberate mistakes, graded

Grade 1 = misleading, 5 = says exactly what is wrong and what to write.

| # | Mistake | What Sky said | Grade | Comment |
|---|---|---|---|---|
| 1 | `5 m + 3 s`; `speed(10 s, 2 m)` for `(d: m, t: s)` | "cannot add length and time", both operands underlined and labelled; "`d` of `speed` expects a length, found a time", with the annotation shown | 5 | Both errors reported at once, with the reason. |
| 2 | typo `velocty` | "cannot find `velocty` in this scope: not defined" | 3 | No suggestion, though unknown units get one. |
| 3 | `def double(x)` with no colon | "expected `:` after `def`, found end of line", `def` underlined, "end the header line with `:`" | 5 | |
| 4 | over-indented `print(total)`; `if` with unindented body | "unexpected indentation ... only the body of an `if`, `while`, `for`, or `def` is indented; otherwise align this line with the one before it"; "expected an indented block after `if`; use `pass`..." | 5 | Both reported. |
| 5 | NumPy habits: `x**2`, `x^2` on a vector, `np.sum(x)`, `sum(x)`, `x.mean()` | `**`: "not an operator; Sky writes powers as `^`" (5). `x^2`: explains the ambiguity and offers `x @ x` and `x * x` (5). `np.sum`, `sum`: "cannot find ... not defined" (2). `x.mean()`: "attribute `mean` is not supported by the interpreter" (2) | 3 | Excellent where the language chose, flat where a function is simply missing. |
| 6 | `A @ v` of `[2,3]` and `[2]`; `v + w` of 2 and 3 | "cannot multiply Matrix[2,3] by Vector[2]: inner dimensions differ: 3 vs 2", the rule, the pair, `3 == 2`; for `A @ A` it suggests transposing | 5 | Shape debugger is the best thing here. Unit label `[1]` on dimensionless is slightly odd. |
| 7 | `7 // 2`, `7 % 2`, `range(7 / 2)` | "expected an expression, found `/`"; "unexpected character '%'"; "`range` needs an integer, found `7 / 2` ... round it first: `round(x)`, `floor(x)` or `ceil(x)`" | 3 | `range` is a 5; `//` and `%` are a 2 each: no sign that the operators do not exist. |
| 8 | `range(len(v) + 1)` indexing `v[i]` | "index 3 is out of bounds for axis 0 with length 3: valid indices are 0 to 2", at the index | 4 | Runtime, after the first three lines printed. The line number of the `for` is not shown. |
| 9 | `v[0] = 5 m`, `v[1] += 1 m` | "cannot assign to an element of `v`: arrays are immutable in v0.1" | 3 | Clear, and the `+=` form is understood. No help line, and no way forward. |
| 10 | `e^2`, `math.pi` | "cannot find `e` in this scope: not defined" | 2 | The spec knows the answer is `exp(1)`; the compiler does not say. |

Mean grade: 3.8 / 5. The messages Sky has designed (units, shapes, indentation,
`**`, `x^2`, `range`) are the best I have seen from a new language; the
messages for things that are simply not there yet are all "not defined".

## The REPL (`sky repl`)

- **Nice:** expressions echo (`1 km + 500 m` gives `1500 m`; `v as km/h` keeps `90 km/h`); a failing line changes nothing, so you can retry; blocks (`for`, `def`) continue until an empty line; `:help` is short and complete; `range(3)` alone gets a helpful note ("`range` is not a value: it only tells a `for` loop what to count"); `sqrt(2)`, `pi`, `c`, `h` print at once; uncertainty (`10.14 mm ± 0.06 mm`) and `as mm` work. For poking at units it was the fastest way to find that `1 ft` is `1e-12 kg`.
- **Annoying:** echo is SI-only (`1 L` shows `0.001 m^3`, `1 ly` shows `9460730472580800 m`), so each exploration needs `as ...` appended; no tab completion or listing of defined names or units that I could find; `x = 3 m` then re-binding to another dimension is fine silently; I could not test history or line editing because the tool environment has no terminal (piped input prints no prompt and echoes only results).
- **Missing:** `:units`, `:vars`, `:type x`, `:load file.sky`.

## What was pleasant

- Unit errors are exactly what a scientist wants: both sides underlined, dimension named ("length", "time"), and the `+` rule stated. A call with the wrong units points at the parameter annotation.
- `as` is a clean idea: `print(v as km/h)`, and it carries through `±` (`10.14 ± 0.06 mm`). The error when you try to compute with an `as` value says exactly how (`divide by one`).
- Constants (`G`, `c`, `pi`, `g_n`) are built in, with CODATA values, and the surface-gravity formula in `planets.sky` was a one-liner with the right dimension checked: `G * M / R^2` gives `m/s^2` without any conversion work.
- Shape errors explain the contraction rule and suggest the transpose.
- `x^2` on a vector gets a short explanation of why it is refused and both alternatives; so does `**`.
- `for` over `range(n)` or an array works, with `if`/`elif`/`else`, `while`, `break`, `continue`, `+=`, and `a if c else b`.
- Output is deterministic: every run of the Monte Carlo matches byte for byte on `ast`, `sir`, and `native` (the programs that iterate over an array correctly say they cannot run on SIR/native, and `--engine=auto` falls back).
- With annotations the native engine is genuinely fast: 1 million LCG steps in 17 ms.
- Compiler warnings that shadow a constant (`c`, `G`) are friendly: they say what the name is and how to avoid it.
