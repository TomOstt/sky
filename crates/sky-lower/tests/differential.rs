//! Lowering preserves meaning: run programs with the tree-walking
//! interpreter and, after lowering, with the SIR reference interpreter
//! (`sky-ir-interp`), and compare what they print and the error that stops
//! them, message for message. Covers hand-written tricky control flow, the repository's runnable
//! UI programs, and a few hundred generated programs full of early returns,
//! `break`, `continue`, nested loops and module-level names.

mod common;

use common::eval::{normalized, run_main};
use common::*;
use sky_interp::RuntimeEnv;

/// What the tree-walking interpreter prints and whether it finished.
fn interpret(src: &str) -> (String, Result<(), String>) {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        // `sky run` honours the checker's Cholesky rewrites.
        let checked = sky_types::check_module(&ast, src);
        let mut env = RuntimeEnv::with_prelude();
        // ... and its chain plans (order, solves), as SIR does.
        env.add_chain_plans(&checked.chains);
        let mut out = Vec::new();
        let r =
            sky_interp::run_module_with(&ast, src, &mut env, &mut out, &checked.cholesky_solves);
        (
            String::from_utf8(out).expect("utf8"),
            r.map(|_| ()).map_err(|e| e.message.clone()),
        )
    })
}

/// Asserts SIR and the interpreter agree on `src`; returns the output.
fn agree(src: &str) -> String {
    let m = lower_ok(src);
    let (want_out, want) = interpret(src);
    let (got_out, got) = run_main(&m);
    assert_eq!(
        got_out,
        want_out,
        "printed output differs\nsource:\n{src}\nSIR:\n{}",
        dump_module(&m)
    );
    assert_eq!(
        normalized(&got),
        normalized(&want),
        "failure differs (interpreter {want:?}, SIR {got:?})\nsource:\n{src}\nSIR:\n{}",
        dump_module(&m)
    );
    want_out
}

#[test]
fn arithmetic_units_and_constants() {
    let out = agree(
        "v = 36 km/h\nprint(v)\nm = 2 kg\nprint(m * c^2)\nprint(3 m / 2 s, -v, (2 s)⁻¹, inv(4 m))\nprint(2^10, 4^0.5, 8^(1/3), (0 - 8)^3)\nx = 5 m\nx += 2 m\nx *= 3\nx -= 1 m\nx /= 4\nprint(x)\n",
    );
    assert!(out.starts_with("10 m/s\n"), "{out}");
}

#[test]
fn booleans_strings_and_short_circuit() {
    let out = agree(
        "a = true\nb = false\nprint(a and b, a or b, not a, a and not b or b)\nx = 0 m\nprint(x != 0 m and 1 m / x > 1)\nprint(x == 0 m or 1 m / x > 1)\ns = \"a\\tb\"\nprint(s, s == \"a\\tb\", s != \"c\")\nprint(1 if a else 2, 3 if b else 4)\n",
    );
    assert!(
        out.contains("false\ntrue\n") || out.contains("false true"),
        "{out}"
    );
}

#[test]
fn the_right_operand_of_and_is_not_evaluated_when_the_left_decides() {
    // 1/0 would fail: it must not run. Both implementations succeed.
    let out = agree("x = 0 m\nok = x > 1 m and 1 m / x > 1\nprint(ok)\n");
    assert_eq!(out, "false\n");
    // And when it does run, both fail.
    let m = lower_ok("x = 0 m\nok = x < 1 m and 1 m / x > 1\n");
    assert!(run_main(&m).1.is_err());
}

#[test]
fn division_by_zero_and_overflow_fail_in_both() {
    for src in [
        "print(1 / 0)\n",
        "print(1e308 * 10)\n",
        "x = 0\nprint(inv(x))\n",
        "print((0 - 8)^0.5)\n",
        "print(0^(0 - 1))\n",
        "print(1)\nprint(2 / 0)\nprint(3)\n",
    ] {
        let m = lower_ok(src);
        let (want_out, want) = interpret(src);
        let (got_out, got) = run_main(&m);
        assert!(want.is_err() && got.is_err(), "{src}: {want:?} / {got:?}");
        assert_eq!(got_out, want_out, "{src}");
    }
}

#[test]
fn control_flow_if_elif_else_while() {
    let out = agree(
        "speed = 3 m/s\nlimit = 10 m/s\nsteps = 0\nwhile speed < limit:\n    speed = speed + 2 m/s\n    steps = steps + 1\n    if steps == 3:\n        continue\n    if steps > 100:\n        break\nif speed > limit:\n    label = 1\nelif speed == limit:\n    label = 2\nelse:\n    label = 3\ncapped = limit if speed > limit else speed\nprint(steps, label, speed, capped)\n",
    );
    assert_eq!(out, "4 1 11 m/s 10 m/s\n");
}

#[test]
fn break_and_continue_in_nested_loops() {
    agree(
        "i = 0\ntotal = 0\nwhile i < 6:\n    i += 1\n    if i == 2:\n        continue\n    j = 0\n    while true:\n        j += 1\n        if j > 3:\n            break\n        if j == 2:\n            continue\n        total += i * j\n    if i == 5:\n        break\nprint(i, total)\n",
    );
}

#[test]
fn a_name_first_assigned_in_while_true_is_read_after_it() {
    let out = agree(
        "n = 0\nwhile true:\n    n += 1\n    found = n * 10\n    if found > 25:\n        break\nprint(found, n)\n",
    );
    assert_eq!(out, "30 3\n");
}

#[test]
fn early_return_inside_while_inside_if_inside_def() {
    let src = "\
def find(n, limit):
    if n > 0:
        i = 0
        while i < limit:
            if i * i > n:
                return i
            i = i + 1
        return 0 - 1
    return 100
print(find(10, 20), find(10, 2), find(0 - 5, 3), find(1, 5))
";
    assert_eq!(agree(src), "4 -1 100 2\n");
}

#[test]
fn return_from_nested_loops() {
    let src = "\
def first_pair(n):
    i = 1
    while i < n:
        j = 1
        while j < n:
            if i * j > 6:
                return i * 100 + j
            j = j + 1
        i = i + 1
    return 0
print(first_pair(5), first_pair(3))
";
    assert_eq!(agree(src), "204 0\n");
}

#[test]
fn returns_with_flags_and_merges_across_arms() {
    let src = "\
def f(a, b):
    if a:
        if b:
            return 1
        y = 2
    else:
        y = 3
    return y
print(f(true, true), f(true, false), f(false, true), f(false, false))
";
    assert_eq!(agree(src), "1 2 3 3\n");
}

#[test]
fn early_returns_in_a_row() {
    let src = "\
def classify(x):
    if x < 0:
        return 0 - 1
    if x == 0:
        return 0
    if x < 10:
        return 1
    if x < 100:
        return 2
    return 3
print(classify(0 - 4), classify(0), classify(7), classify(70), classify(700))
";
    assert_eq!(agree(src), "-1 0 1 2 3\n");
}

#[test]
fn functions_read_the_modules_current_names() {
    let src = "\
k = 2
def scaled(x):
    return x * k
a = scaled(10)
k = 3
b = scaled(10)
def outer(x):
    return scaled(x) + 1
c = outer(1)
def shadow(k):
    return scaled(k)
d = shadow(100)
def local_first(x):
    k = x + 1
    return k * 2
e = local_first(4)
print(a, b, c, d, e, k)
";
    assert_eq!(agree(src), "20 30 4 300 10 3\n");
}

#[test]
fn a_function_that_assigns_a_module_name_on_one_path_only() {
    let src = "\
base = 10
def f(flag):
    if flag:
        base = 1
    return base + 1
print(f(true), f(false))
";
    assert_eq!(agree(src), "2 11\n");
}

#[test]
fn constants_rebound_after_a_function_that_reads_them() {
    let src = "\
def speed():
    return c
before = speed()
c = 2 m/s
after = speed()
print(before, after)
";
    agree(src);
}

#[test]
fn recursion_and_calls_as_values_of_expressions() {
    let src = "\
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
print(fact(10), fib(12), fact(3) + fib(4) * 2)
";
    assert_eq!(agree(src), "3628800 144 12\n");
}

#[test]
fn a_function_without_a_value_and_print_as_a_value() {
    let src = "def shout(x):\n    print(x)\n    if x > 1:\n        return\n    print(0)\nshout(5)\nshout(1)\nr = print(9)\nprint(r)\n";
    agree(src);
}

#[test]
fn print_shadowing_and_inv_shadowing() {
    let src = "def inv(x):\n    return x + 1\nprint(inv(2))\ndef twice(x):\n    return inv(x) * 2\nprint(twice(1))\n";
    assert_eq!(agree(src), "3\n4\n");
}

#[test]
fn display_units_print_the_same_in_both() {
    let out = agree(
        "v = 90 km/h\nprint(v)\nprint(v as km/h)\nprint(v as m/s, v as km/h, 3)\nprint(1 AU as km, 1 AU as m)\nprint(511 keV as eV, 511 keV as MeV)\nprint(2 kg * 9.8 m/s^2 as N)\nprint(0.1 s + 0.2 s as ms)\nv as km/h\nprint(1 h + 30 min as min, 1 day as h)\n",
    );
    assert_eq!(
        out,
        "25 m/s\n90 km/h\n25 m/s 90 km/h 3\n149597870.7 km 149597870700 m\n511000 eV 0.511 MeV\n19.6 N\n300.00000000000006 ms\n90 min 24 h\n"
    );
}

#[test]
fn display_units_inside_functions_loops_and_branches() {
    let out = agree(
        "def show_speed(v):\n    print(v as km/h)\n    if v > 5 m/s:\n        print(\"fast\", v as mm/s)\n    return v\nx = show_speed(36 km/h)\nn = 0\nwhile n < 3:\n    n += 1\n    print(n * 1 km as m)\n    n * 1 km as km\nprint(x as m/s, (x as km/h))\nprint((1 m if true else 2 m) as cm)\n",
    );
    assert!(out.starts_with("36 km/h\nfast 10000 mm/s\n"), "{out}");
}

#[test]
fn display_units_fail_the_same_way_in_both() {
    for src in [
        // 1e300 m is 1e309 nm.
        "print(1)\nprint(1e300 m as nm)\nprint(2)\n",
        "x = 1e300 m\nx as nm\n",
        "def f(a):\n    print(a as nm)\nf(1e300 m)\n",
    ] {
        let m = lower_ok(src);
        let (want_out, want) = interpret(src);
        let (got_out, got) = run_main(&m);
        assert_eq!(want, Err("arithmetic overflow".to_string()), "{src}");
        assert_eq!(got, want, "{src}");
        assert_eq!(got_out, want_out, "{src}");
    }
    // A wrong dimension is a check error before anything runs.
    let e = lower_err("print(90 km/h as kg)\n");
    assert!(e.contains("cannot show a velocity in kg"), "{e}");
}

#[test]
fn every_value_and_unit_prints_alike_in_both() {
    let units = [
        "m", "km", "mm", "AU", "pc", "ly", "km/h", "m/s", "eV", "MeV", "J", "kW*h", "day", "h",
        "min", "yr",
    ];
    for unit in units {
        let mut src = String::new();
        for n in 1..=40u32 {
            let x = f64::from(n) * 0.75 + 0.125;
            src.push_str(&format!("print({x} {unit} as {unit}, {x} {unit})\n"));
        }
        // Every value reads back as itself.
        let out = agree(&src);
        for (n, line) in (1..=40u32).zip(out.lines()) {
            let x = sky_interp::format_number(f64::from(n) * 0.75 + 0.125);
            assert!(line.starts_with(&format!("{x} {unit} ")), "{unit}: {line}");
        }
    }
}

#[test]
fn the_runnable_ui_programs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "tests/ui/check/ok_control_flow.sky",
        "tests/ui/check/ok_kinetic.sky",
        "tests/ui/check/ok_constants.sky",
        "tests/ui/check/ok_parses.sky",
        "tests/ui/check/ok_bool_literals.sky",
        "tests/ui/check/ok_shadow_constant.sky",
        "crates/sky-types/tests/ui/ok_booleans.sky",
        "crates/sky-types/tests/ui/ok_branch_merge_scalar.sky",
        "crates/sky-types/tests/ui/ok_conditional_expression.sky",
        "crates/sky-types/tests/ui/ok_decimal_exponent.sky",
        "crates/sky-types/tests/ui/ok_electrical_units.sky",
        "crates/sky-types/tests/ui/ok_prefixed_units.sky",
        "crates/sky-types/tests/ui/ok_prelude.sky",
        "crates/sky-types/tests/ui/ok_print_builtin.sky",
        "crates/sky-types/tests/ui/ok_scalar_inverse.sky",
        "crates/sky-types/tests/ui/ok_user_bound_true.sky",
        "tests/ui/types/ok_array_builtins.sky",
        "tests/ui/types/ok_array_index.sky",
        "tests/ui/types/ok_array_literals.sky",
        "tests/ui/types/ok_cholesky_literal.sky",
        "tests/ui/types/ok_literal_tag_proofs.sky",
    ] {
        let path = root.join(name);
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        let m = lower_ok(&src);
        let (want_out, want) = interpret(&src);
        let (got_out, got) = run_main(&m);
        assert_eq!(got_out, want_out, "{name}");
        assert_eq!(normalized(&got), normalized(&want), "{name}");
    }
}

/// The names a program binds at its top level (`x = ...`, `x: T = ...`).
fn top_level_names(src: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for line in src.lines() {
        let ident: String = line
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let rest = line[ident.len()..].trim_start();
        let binds = !ident.is_empty()
            && ((rest.starts_with('=') && !rest.starts_with("==")) || rest.starts_with(':'));
        if binds && ident != "def" && !names.contains(&ident) {
            names.push(ident);
        }
    }
    names
}

/// The array programs of the UI suites print nothing, so print every name
/// they bind: the arrays SIR computes must be the ones `sky run` computes,
/// entry for entry, in the same text.
#[test]
fn the_array_ui_programs_bind_the_same_values() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "ok_array_builtins",
        "ok_array_index",
        "ok_array_literals",
        "ok_cholesky_literal",
        "ok_literal_tag_proofs",
    ] {
        let path = root.join("tests/ui/types").join(format!("{name}.sky"));
        let mut src = std::fs::read_to_string(&path).expect("reads");
        let names = top_level_names(&src);
        assert!(!names.is_empty(), "{name}");
        for n in &names {
            src.push_str(&format!("print({n})\n"));
        }
        let out = agree(&src);
        assert_eq!(out.lines().count(), names.len(), "{name}:\n{out}");
    }
}

// ----- generated programs -----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// Generates a well-formed program of dimensionless numbers: every name is
/// bound before it is read, every loop has a counter, every function ends in
/// `return`. Failures at run time (division by zero, overflow) are allowed:
/// both implementations must fail alike.
struct Gen {
    rng: Rng,
    out: String,
    indent: usize,
    /// Names that may be read here.
    readable: Vec<String>,
    /// Names that plain assignment may target.
    assignable: Vec<String>,
    /// Names `+=` etc. may target (bound in this function or at top level).
    updatable: Vec<String>,
    /// Functions defined so far, with their arity.
    funcs: Vec<(String, usize)>,
    modnames: Vec<String>,
    loop_depth: usize,
    in_fn: bool,
    counter: usize,
}

impl Gen {
    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn pick(&mut self, v: &[String]) -> String {
        v[self.rng.below(v.len())].clone()
    }

    fn expr(&mut self, depth: usize) -> String {
        if depth == 0 || self.rng.chance(30) {
            return if self.rng.chance(45) || self.readable.is_empty() {
                format!("{}", self.rng.below(10))
            } else {
                let r = self.readable.clone();
                self.pick(&r)
            };
        }
        match self.rng.below(12) {
            0..=3 => format!("({} + {})", self.expr(depth - 1), self.expr(depth - 1)),
            4..=5 => format!("({} - {})", self.expr(depth - 1), self.expr(depth - 1)),
            6..=7 => format!("({} * {})", self.expr(depth - 1), self.expr(depth - 1)),
            8 => format!("(0 - {})", self.expr(depth - 1)),
            9 => {
                if self.rng.chance(50) {
                    format!(
                        "({} / ({} + 1))",
                        self.expr(depth - 1),
                        self.expr(depth - 1)
                    )
                } else {
                    format!("({}^2)", self.expr(depth - 1))
                }
            }
            10 => {
                let c = self.cond(depth - 1);
                format!(
                    "({} if {c} else {})",
                    self.expr(depth - 1),
                    self.expr(depth - 1)
                )
            }
            _ => {
                if self.funcs.is_empty() {
                    return self.expr(depth - 1);
                }
                let (name, arity) = self.funcs[self.rng.below(self.funcs.len())].clone();
                let args: Vec<String> = (0..arity).map(|_| self.expr(depth - 1)).collect();
                format!("{name}({})", args.join(", "))
            }
        }
    }

    fn cond(&mut self, depth: usize) -> String {
        match self.rng.below(10) {
            0..=4 => {
                let op = ["<", "<=", ">", ">=", "==", "!="][self.rng.below(6)];
                format!(
                    "{} {op} {}",
                    self.expr(depth.min(1)),
                    self.expr(depth.min(1))
                )
            }
            5 => format!(
                "({} and {})",
                self.cond(depth - depth.min(1)),
                self.cond(depth - depth.min(1))
            ),
            6 => format!(
                "({} or {})",
                self.cond(depth - depth.min(1)),
                self.cond(depth - depth.min(1))
            ),
            7 => format!("not ({})", self.cond(depth - depth.min(1))),
            8 => "true".to_string(),
            _ => format!("{} < {}", self.expr(1), self.expr(1)),
        }
    }

    /// Statements of a block; leaves the name pools as it found them.
    fn block(&mut self, depth: usize, exits_ok: bool) {
        let saved = (
            self.readable.len(),
            self.assignable.len(),
            self.updatable.len(),
        );
        self.indent += 1;
        let n = 1 + self.rng.below(3);
        for _ in 0..n {
            self.stmt(depth, exits_ok);
        }
        self.indent -= 1;
        self.readable.truncate(saved.0);
        self.assignable.truncate(saved.1);
        self.updatable.truncate(saved.2);
    }

    fn stmt(&mut self, depth: usize, exits_ok: bool) {
        let choice = self.rng.below(100);
        match choice {
            0..=24 => {
                let a = self.assignable.clone();
                let t = self.pick(&a);
                let e = self.expr(2);
                self.line(&format!("{t} = {e}"));
            }
            25..=34 => {
                let u = self.updatable.clone();
                if u.is_empty() {
                    return;
                }
                let t = self.pick(&u);
                let op = ["+=", "-=", "*="][self.rng.below(3)];
                let e = self.expr(1);
                self.line(&format!("{t} {op} {e}"));
            }
            35..=46 => {
                let e = self.expr(2);
                if self.rng.chance(30) {
                    let e2 = self.expr(1);
                    self.line(&format!("print({e}, {e2})"));
                } else {
                    self.line(&format!("print({e})"));
                }
            }
            47..=62 if depth < 3 => self.if_stmt(depth),
            63..=76 if depth < 3 => self.while_stmt(depth),
            77..=82 if depth < 3 => self.merge_stmt(depth),
            83..=89 if self.in_fn && exits_ok => {
                let e = self.expr(2);
                self.line(&format!("return {e}"));
            }
            90..=94 if self.loop_depth > 0 && exits_ok => self.line("break"),
            95..=99 if self.loop_depth > 0 && exits_ok => self.line("continue"),
            _ => {
                let e = self.expr(1);
                self.line(&format!("print({e})"));
            }
        }
    }

    fn if_stmt(&mut self, depth: usize) {
        let c = self.cond(2);
        self.line(&format!("if {c}:"));
        self.block(depth + 1, true);
        for _ in 0..self.rng.below(2) {
            let c = self.cond(2);
            self.line(&format!("elif {c}:"));
            self.block(depth + 1, true);
        }
        if self.rng.chance(50) {
            self.line("else:");
            self.block(depth + 1, true);
        }
    }

    /// `if c: ...; t = e else: ...; t = e`, then `t` is readable.
    fn merge_stmt(&mut self, depth: usize) {
        let t = self.fresh("t");
        let c = self.cond(2);
        self.line(&format!("if {c}:"));
        self.arm_assigning(depth, &t);
        if self.rng.chance(40) {
            let c = self.cond(1);
            self.line(&format!("elif {c}:"));
            self.arm_assigning(depth, &t);
        }
        self.line("else:");
        self.arm_assigning(depth, &t);
        self.readable.push(t.clone());
        self.assignable.push(t);
    }

    fn arm_assigning(&mut self, depth: usize, t: &str) {
        let saved = (
            self.readable.len(),
            self.assignable.len(),
            self.updatable.len(),
        );
        self.indent += 1;
        for _ in 0..self.rng.below(3) {
            let exits = self.rng.chance(50);
            self.stmt(depth + 1, exits);
        }
        // Not an unconditional exit at this level: the arm must fall through
        // so the checker binds `t` after the `if`.
        let e = self.expr(2);
        self.line(&format!("{t} = {e}"));
        self.indent -= 1;
        self.readable.truncate(saved.0);
        self.assignable.truncate(saved.1);
        self.updatable.truncate(saved.2);
    }

    fn while_stmt(&mut self, depth: usize) {
        let k = self.fresh("k");
        self.line(&format!("{k} = 0"));
        let forever = self.rng.chance(25);
        if forever {
            self.line("while true:");
        } else {
            let bound = 1 + self.rng.below(4);
            let extra = if self.rng.chance(30) {
                format!(" and {}", self.cond(1))
            } else {
                String::new()
            };
            self.line(&format!("while {k} < {bound}{extra}:"));
        }
        self.indent += 1;
        self.line(&format!("{k} = {k} + 1"));
        if forever {
            self.line(&format!("if {k} > 3:"));
            self.indent += 1;
            self.line("break");
            self.indent -= 1;
        }
        self.indent -= 1;
        self.loop_depth += 1;
        let saved = (
            self.readable.len(),
            self.assignable.len(),
            self.updatable.len(),
        );
        self.readable.push(k);
        self.block_body_only(depth + 1);
        self.readable.truncate(saved.0);
        self.assignable.truncate(saved.1);
        self.updatable.truncate(saved.2);
        self.loop_depth -= 1;
    }

    /// Statements at the loop body's indentation (after the counter line).
    fn block_body_only(&mut self, depth: usize) {
        self.indent += 1;
        for _ in 0..1 + self.rng.below(3) {
            self.stmt(depth, true);
        }
        self.indent -= 1;
    }

    fn function(&mut self, name: &str, arity: usize) {
        let params: Vec<String> = (0..arity).map(|i| format!("p{i}")).collect();
        self.line(&format!("def {name}({}):", params.join(", ")));
        self.indent += 1;
        self.in_fn = true;
        let saved = (
            self.readable.clone(),
            self.assignable.clone(),
            self.updatable.clone(),
        );
        self.readable = self.modnames.clone();
        self.readable.extend(params.iter().cloned());
        self.assignable = params.clone();
        self.updatable = params.clone();
        // Module names may be assigned too: that makes a local.
        for m in self.modnames.clone() {
            if self.rng.chance(30) {
                self.assignable.push(m);
            }
        }
        for v in ["v0", "v1"] {
            let e = self.expr(1);
            self.line(&format!("{v} = {e}"));
            self.readable.push(v.to_string());
            self.assignable.push(v.to_string());
            self.updatable.push(v.to_string());
        }
        for _ in 0..2 + self.rng.below(4) {
            self.stmt(0, true);
        }
        let e = self.expr(2);
        self.line(&format!("return {e}"));
        self.in_fn = false;
        self.indent -= 1;
        (self.readable, self.assignable, self.updatable) = saved;
    }

    fn program(seed: u64) -> String {
        let mut g = Gen {
            rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
            out: String::new(),
            indent: 0,
            readable: Vec::new(),
            assignable: Vec::new(),
            updatable: Vec::new(),
            funcs: Vec::new(),
            modnames: Vec::new(),
            loop_depth: 0,
            in_fn: false,
            counter: 0,
        };
        let nmod = 2 + g.rng.below(3);
        for i in 0..nmod {
            let name = format!("x{i}");
            let v = g.rng.below(6);
            g.line(&format!("{name} = {v}"));
            g.modnames.push(name);
        }
        g.readable = g.modnames.clone();
        g.assignable = g.modnames.clone();
        g.updatable = g.modnames.clone();
        for i in 0..g.rng.below(4) {
            let arity = 1 + g.rng.below(2);
            let name = format!("f{i}");
            g.function(&name, arity);
            g.funcs.push((name, arity));
        }
        for _ in 0..6 + g.rng.below(8) {
            g.stmt(0, true);
        }
        let names: Vec<String> = g.modnames.clone();
        g.line(&format!("print({})", names.join(", ")));
        g.out
    }
}

#[test]
fn show_generated_program() {
    if let Ok(seed) = std::env::var("SKY_SEED") {
        let src = Gen::program(seed.parse().expect("seed"));
        println!("{src}");
    }
}

#[test]
fn generated_programs_mean_the_same_lowered() {
    let (mut compared, mut skipped, mut failed) = (0, 0, 0);
    for seed in 1..=600u64 {
        let src = Gen::program(seed);
        let m = match sky_lower::lower_source(&src) {
            Ok(m) => m,
            Err(e) => {
                // The generator only skips programs the checker rejects
                // (say, a merge it could not prove); lowering itself must
                // never be what fails.
                assert!(
                    !e.contains("internal error") && !e.contains("is not supported"),
                    "seed {seed}: {e}\n{src}"
                );
                skipped += 1;
                continue;
            }
        };
        check_structure(&m);
        let (want_out, want) = interpret(&src);
        let (got_out, got) = run_main(&m);
        assert_eq!(
            got_out,
            want_out,
            "seed {seed}: printed output differs\n{src}\nSIR:\n{}",
            dump_module(&m)
        );
        assert_eq!(
            got.is_ok(),
            want.is_ok(),
            "seed {seed}: failure differs (interpreter {want:?}, SIR {got:?})\n{src}\nSIR:\n{}",
            dump_module(&m)
        );
        compared += 1;
        failed += usize::from(want.is_err());
    }
    eprintln!("generated: compared {compared}, skipped {skipped}, failing at run time {failed}");
    // Most programs must reach the comparison, and enough must run to the
    // end for the comparison to mean something.
    assert!(compared >= 400, "compared {compared}, skipped {skipped}");
    assert!(
        failed * 3 < compared * 2,
        "{failed} of {compared} fail at run time"
    );
}

#[test]
fn planned_chains_run_alike_in_both() {
    // The checker plans these (order, solves, a division, a transpose) and
    // both engines follow the plan: the same output, bit for bit.
    let src = "\
A = [[1, 2], [3, 4]]
B = [[0.5, 1], [1, 0.25]]
C = [[2, 0.1], [0, 2]]
v = [1, 2]
print(A @ B @ C @ v)
print(inv(A) @ inv(B) @ v)
print(v @ inv(A), inv(A) @ B, inv(A) @ inv(B))
D: Diagonal Matrix[2, 2] = [[2, 0], [0, 8]]
print(inv(D) @ v, v @ inv(D), inv(D) @ A)
Q = identity(2)
print(inv(Q) @ v)
";
    let out = agree(src);
    assert_eq!(out.lines().count(), 5, "{out}");
    let m = lower_ok(src);
    let text = sky_ir::print(&m);
    for op in ["solve[lu]", "solve.right[lu]", "solve[diagonal]"] {
        assert!(text.contains(op), "{op} in\n{text}");
    }
}

#[test]
fn a_planned_chain_that_fails_fails_alike() {
    for src in [
        "A = [[1, 2], [2, 4]]\nprint(1)\nprint(inv(A) @ [1, 2])\n",
        "A = [[1e200, 2], [3, 4]]\nprint(A @ A @ A @ [1e200, 1])\n",
    ] {
        let m = lower_ok(src);
        let (want_out, want) = interpret(src);
        let (got_out, got) = run_main(&m);
        assert_eq!(got_out, want_out, "{src}");
        assert!(want.is_err(), "{src}");
        assert_eq!(normalized(&got), normalized(&want), "{src}");
    }
}
