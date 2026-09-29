//! The IR's region nesting is bounded by a constant times the source's
//! nesting, whatever the length of a chain: `region depth <= 2 * D + 4`,
//! where `D` is the source depth (see "Region depth" in the crate docs).
//! Checked on generated programs full of long `elif` chains, long `and`/`or`
//! chains and early exits, which the old scheme nested one region per arm,
//! operand and exit. Each program also runs through the tree-walking
//! interpreter and the SIR interpreter, which must agree.

mod common;

use common::eval::{normalized, run_main};
use common::*;
use sky_interp::RuntimeEnv;

/// `region_depth <= FACTOR * D + SLACK`: two levels per source level (the
/// guard before a statement, and the arm of the `if`/`while` it is), and a
/// constant for the function's own guard and the `if` of a chain's
/// condition.
const FACTOR: usize = 2;
const SLACK: usize = 4;

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

/// Generates a checked program of dimensionless numbers and tracks its
/// source depth: the deepest point of block nesting (bodies of `def`, `if`,
/// `elif`, `else` and `while`) plus short-circuit nesting in expressions
/// (`and`/`or` operands and `if`-`else` expressions inside one another). A
/// chain of `elif`s or of operands is not nesting, however long.
struct Gen {
    rng: Rng,
    out: String,
    indent: usize,
    /// The deepest source point so far.
    depth: usize,
    /// Names that may be read and assigned here.
    names: Vec<String>,
    counter: usize,
    loops: usize,
    in_fn: bool,
    /// The longest chain the generator may write.
    max_chain: usize,
}

impl Gen {
    fn line(&mut self, s: &str) {
        self.out.push_str(&"    ".repeat(self.indent));
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn name(&mut self) -> String {
        let i = self.rng.below(self.names.len());
        self.names[i].clone()
    }

    /// An expression's text and its short-circuit nesting.
    fn atom(&mut self) -> (String, usize) {
        match self.rng.below(4) {
            0 => (format!("{}", self.rng.below(9)), 0),
            1 => (self.name(), 0),
            2 => {
                let (a, b) = (self.name(), self.name());
                (format!("{a} + {b}"), 0)
            }
            _ => {
                let (a, k) = (self.name(), self.rng.below(5));
                (format!("{a} * {k}"), 0)
            }
        }
    }

    /// A number: an atom, or `(x if c else y)`.
    fn number(&mut self, nest: usize) -> (String, usize) {
        if nest == 0 || !self.rng.chance(20) {
            return self.atom();
        }
        let (c, nc) = self.cond(nest - 1);
        let (a, na) = self.number(nest - 1);
        let (b, nb) = self.number(nest - 1);
        (format!("({a} if {c} else {b})"), 1 + nc.max(na).max(nb))
    }

    /// A Bool: a comparison, or a chain of them joined by `and`/`or` whose
    /// operands may be parenthesized chains. `nest` is how much nesting is
    /// left to spend.
    fn cond(&mut self, nest: usize) -> (String, usize) {
        if nest == 0 || self.rng.chance(35) {
            let (a, b) = (self.name(), self.rng.below(6));
            let op = ["<", ">", "==", "!=", "<=", ">="][self.rng.below(6)];
            return (format!("{a} {op} {b}"), 0);
        }
        let n = if nest >= 2 && self.rng.chance(15) {
            2 + self.rng.below(self.max_chain)
        } else {
            2 + self.rng.below(4)
        };
        let word = if self.rng.chance(50) { "and" } else { "or" };
        let mut deepest = 0;
        let mut parts = Vec::new();
        for _ in 0..n {
            // Mixing `and` and `or` in one chain needs parentheses; a
            // nested chain is parenthesized, so it is one level down.
            let (p, np) = if self.rng.chance(25) {
                let (c, nc) = self.cond(nest - 1);
                (format!("({c})"), nc)
            } else {
                self.cond(0)
            };
            deepest = deepest.max(np);
            parts.push(p);
        }
        (parts.join(&format!(" {word} ")), 1 + deepest)
    }

    /// Records a source point with expression nesting `e` in the block
    /// being written; the indentation is the block depth (0 at the top
    /// of the file, 1 in a `def`'s body).
    fn at(&mut self, e: usize) {
        self.depth = self.depth.max(self.indent + e);
    }

    /// An indented block of one to three statements (and `pass`, so it is
    /// never empty), starting with `prelude` if any.
    fn block(&mut self, budget: usize, prelude: Option<String>) {
        self.indent += 1;
        self.at(0);
        if let Some(p) = prelude {
            self.line(&p);
        }
        for _ in 0..1 + self.rng.below(3) {
            self.stmt(budget);
        }
        self.line("pass");
        self.indent -= 1;
    }

    fn stmt(&mut self, budget: usize) {
        let kind = self.rng.below(100);
        if budget == 0 || kind < 30 {
            let (v, n) = (self.name(), self.number(2));
            self.at(n.1);
            self.line(&format!("{v} = {}", n.0));
        } else if kind < 65 {
            // if / elif ... / else, with a chain of any length.
            // Long chains only near the top: a long chain in every arm of a
            // long chain would make programs of millions of lines.
            let arms = if budget >= 3 && self.rng.chance(20) {
                1 + self.rng.below(self.max_chain)
            } else {
                1 + self.rng.below(3)
            };
            for k in 0..arms {
                let (c, n) = self.cond(2);
                self.at(n);
                self.line(&format!("{}{c}:", if k == 0 { "if " } else { "elif " }));
                self.block(budget - 1, None);
            }
            if self.rng.chance(50) {
                self.line("else:");
                self.block(budget - 1, None);
            }
        } else if kind < 78 && self.loops < 3 {
            self.counter += 1;
            let k = format!("k{}", self.counter);
            self.line(&format!("{k} = 0"));
            let (c, n) = self.cond(1);
            self.at(n + 1);
            self.line(&format!("while {k} < 3 and ({c}):"));
            self.loops += 1;
            self.block(budget - 1, Some(format!("{k} += 1")));
            self.loops -= 1;
        } else if kind < 90 && self.in_fn {
            let (c, n) = self.cond(2);
            self.at(n);
            self.line(&format!("if {c}:"));
            self.indent += 1;
            self.at(0);
            let (v, n) = self.number(2);
            self.at(n);
            self.line(&format!("return {v}"));
            self.indent -= 1;
        } else if self.loops > 0 {
            let (c, n) = self.cond(2);
            self.at(n);
            let word = if self.rng.chance(50) {
                "break"
            } else {
                "continue"
            };
            self.line(&format!("if {c}:"));
            self.indent += 1;
            self.at(0);
            self.line(word);
            self.indent -= 1;
        } else {
            let (v, n) = (self.name(), self.number(2));
            self.at(n.1);
            self.line(&format!("{v} += {}", n.0));
        }
    }

    fn program(seed: u64, max_chain: usize) -> (String, usize) {
        let mut g = Gen {
            rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
            out: String::new(),
            indent: 0,
            depth: 0,
            names: Vec::new(),
            counter: 0,
            loops: 0,
            in_fn: false,
            max_chain,
        };
        let module: Vec<String> = (0..3).map(|i| format!("m{i}")).collect();
        for (i, m) in module.iter().enumerate() {
            g.line(&format!("{m} = {}", i + 1));
        }
        for f in 0..1 + g.rng.below(2) {
            // A function: locals bound first, statements with early
            // returns, and a final return.
            g.line(&format!("def f{f}(a, b):"));
            g.indent += 1;
            g.at(0);
            g.line("t0 = a");
            g.line("t1 = b");
            g.indent -= 1;
            g.names = vec!["a".into(), "b".into(), "t0".into(), "t1".into()];
            g.in_fn = true;
            g.block_body(3);
            g.in_fn = false;
            g.line("");
        }
        g.names = module.clone();
        for _ in 0..4 {
            g.stmt(3);
        }
        for f in 0..g.out.matches("def f").count() {
            let (a, b) = (g.name(), g.name());
            let m = &module[f % 3];
            g.line(&format!("{m} = f{f}({a}, {b})"));
        }
        g.line(&format!("print({})", module.join(", ")));
        (g.out, g.depth)
    }

    /// The statements of a function body, then its `return`.
    fn block_body(&mut self, budget: usize) {
        self.indent += 1;
        self.at(0);
        for _ in 0..3 + self.rng.below(4) {
            self.stmt(budget);
        }
        let (v, n) = self.number(2);
        self.at(n);
        self.line(&format!("return {v}"));
        self.indent -= 1;
    }
}

fn interpret(src: &str) -> (String, Result<(), String>) {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let mut env = RuntimeEnv::with_prelude();
        let mut out = Vec::new();
        let r = sky_interp::run_module(&ast, src, &mut env, &mut out);
        (
            String::from_utf8(out).expect("utf8"),
            r.map(|_| ()).map_err(|e| e.message.clone()),
        )
    })
}

/// The deepest region nesting over the functions of a lowered program.
fn ir_depth(m: &sky_ir::Module) -> usize {
    m.functions.iter().map(region_depth).max().unwrap_or(0)
}

#[test]
fn region_nesting_is_bounded_by_the_source_depth_whatever_the_chain_length() {
    let (mut checked, mut worst_slack, mut deepest, mut longest) = (0, usize::MAX, 0, 0);
    for seed in 1..=250u64 {
        // Every fourth program has chains of up to 40: a chain's length
        // must not show up in the depth.
        let max_chain = if seed % 4 == 0 { 40 } else { 8 };
        let (src, d) = Gen::program(seed, max_chain);
        let m = match sky_lower::lower_source(&src) {
            Ok(m) => m,
            Err(e) => {
                assert!(
                    !e.contains("internal error") && !e.contains("nested more than"),
                    "seed {seed}: {e}\n{src}"
                );
                continue;
            }
        };
        let depth = ir_depth(&m);
        assert!(
            depth <= FACTOR * d + SLACK,
            "seed {seed}: region depth {depth} > {FACTOR} * {d} + {SLACK}\n{src}\n{}",
            dump_module(&m)
        );
        worst_slack = worst_slack.min(FACTOR * d + SLACK - depth);
        deepest = deepest.max(depth);
        longest = longest.max(src.lines().count());
        // The bound must not come from lowering less: the program means
        // the same.
        let (want_out, want) = interpret(&src);
        let (got_out, got) = run_main(&m);
        assert_eq!(got_out, want_out, "seed {seed}\n{src}");
        assert_eq!(normalized(&got), normalized(&want), "seed {seed}\n{src}");
        checked += 1;
    }
    eprintln!("depth: {checked} programs, deepest IR {deepest}, least slack {worst_slack}, longest {longest} lines");
    assert!(checked >= 200, "only {checked} programs lowered");
}

/// The chains themselves, measured directly: for `n` arms, operands or
/// returns the depth is the same constant.
#[test]
fn depth_does_not_grow_with_chain_length() {
    let depth_of = |src: &str| ir_depth(&sky_lower::lower_source(src).expect("lowers"));
    let elif = |n: usize| {
        let mut src = String::from("x = 3\nif x == 0:\n    y = 0\n");
        for i in 1..n {
            src.push_str(&format!("elif x == {i}:\n    y = {i}\n"));
        }
        src.push_str("else:\n    y = 0 - 1\nprint(y)\n");
        src
    };
    let and_or = |n: usize| {
        let mut src = String::from("x = 1\nok = ");
        for _ in 0..n {
            src.push_str("x > 0 and x < 5 or ");
        }
        src.push_str("false\nprint(ok)\n");
        src
    };
    let returns = |n: usize| {
        let mut src = String::from("def f(x):\n");
        for i in 0..n {
            src.push_str(&format!("    if x == {i}:\n        return {i}\n"));
        }
        src.push_str("    return 0 - 1\nprint(f(3))\n");
        src
    };
    let loop_exits = |n: usize| {
        let mut src = String::from("def f(x):\n    i = 0\n    while i < 10:\n        i += 1\n");
        for k in 0..n {
            src.push_str(&format!(
                "        if x == {k}:\n            continue\n        if i == {}:\n            return {k}\n",
                k + 100
            ));
        }
        src.push_str("        x += 1\n    return 0 - 1\nprint(f(3))\n");
        src
    };
    for (name, make) in [
        ("elif", &elif as &dyn Fn(usize) -> String),
        ("and/or", &and_or),
        ("returns", &returns),
        ("loop exits", &loop_exits),
    ] {
        let d = depth_of(&make(3));
        for n in [10, 100, 1000] {
            assert_eq!(depth_of(&make(n)), d, "{name}: {n} arms deepen the IR");
        }
    }
}
