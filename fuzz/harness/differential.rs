//! The `differential` fuzz check, shared by the fuzz target
//! (`fuzz/fuzz_targets/differential.rs`) and a stable test in sky-ir-interp
//! (`crates/sky-ir-interp/tests/fuzz_differential.rs`), which both include
//! this file with `#[path]`.
//!
//! Two parts:
//!
//! 1. [`generate`] turns bytes into a Sky program that passes `sky check`.
//!    The bytes are a *choice stream*: every decision the generator makes
//!    (which statement, which operator, which unit) reads the next byte, and
//!    once the bytes run out every decision takes its first option, so the
//!    program ends. A libFuzzer mutation of the bytes is therefore a local
//!    change to the program, and every input is a valid program. The
//!    generator tracks the dimension of every variable and picks operands so
//!    that `+`, `-`, comparisons, `if` arms, assignments, arguments and
//!    returns are dimensionally consistent. Loops always terminate (each
//!    `while` has a counter the body cannot touch) and recursion is bounded
//!    by a depth argument, except for one deliberate unbounded kind, which
//!    must end in the resource-limit error in both engines. Run-time
//!    failures (division by zero, overflow, math domain errors) are
//!    generated on purpose: how often depends on a per-program "risk".
//!
//! 2. [`check_source`] is the oracle. It lexes, parses and checks the source
//!    (all must succeed: a rejection is a generator bug), lowers it, verifies
//!    the SIR, runs the AST interpreter and the SIR interpreter, and requires
//!    the same stdout and the same error message (or both succeed). Then it
//!    prints the SIR, parses the text back, requires the result to equal
//!    `canonicalize(m)`, verifies it, and runs it again for the same result.
//!    The two resource-limit errors compare as one, exactly as in the
//!    hand-written differential tests.
//!
//! Bugs this fuzzer already found are reported, not repeated: the generator
//! steers around most (the `GATE_*` constants) and [`KNOWN`] lets a lenient
//! run tolerate the rest, so a failure of the target or the test is a new
//! one. Each has an `#[ignore]`d regression test in the sky-ir-interp test.
//!
//! The file depends only on `std` and the sky crates (path dependencies of
//! the fuzz crate; sky-ir-interp's tests have them all).
#![allow(dead_code)]

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// The choice stream
// ---------------------------------------------------------------------------

/// Bytes read as decisions. Past the end, every decision is "the first
/// option" and every `chance` is false, so a generator loop that stops on
/// [`Choices::exhausted`] always ends.
pub struct Choices<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Choices<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Choices { data, pos: 0 }
    }

    pub fn exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn byte(&mut self) -> usize {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        usize::from(b)
    }

    /// A number in `0..n` (0 when `n <= 1`, without reading a byte).
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        if n <= 256 {
            self.byte() % n
        } else {
            ((self.byte() << 8) | self.byte()) % n
        }
    }

    /// True with probability about `percent`%; false once the bytes ran out.
    pub fn chance(&mut self, percent: usize) -> bool {
        if self.exhausted() {
            self.pos += 1;
            return false;
        }
        self.below(100) < percent
    }

    pub fn pick<'b, T>(&mut self, xs: &'b [T]) -> &'b T {
        &xs[self.below(xs.len())]
    }

    /// An index into `weights`, chosen in proportion to them.
    pub fn weighted(&mut self, weights: &[usize]) -> usize {
        let total: usize = weights.iter().sum();
        let mut r = self.below(total);
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= w;
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Dimensions and units
// ---------------------------------------------------------------------------

/// A dimension: exponents of m, s, kg, K, A in sixths, so `sqrt` and `cbrt`
/// of anything reasonable stay whole numbers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Dim([i16; 5]);

const ZERO: Dim = Dim([0; 5]);
/// The largest exponent magnitude (in sixths) an expression may reach.
const MAX_EXP: i16 = 24;

const fn dim(m: i16, s: i16, kg: i16, k: i16, a: i16) -> Dim {
    Dim([6 * m, 6 * s, 6 * kg, 6 * k, 6 * a])
}

impl Dim {
    fn add(self, o: Dim) -> Dim {
        let mut r = self.0;
        for (x, y) in r.iter_mut().zip(o.0) {
            *x += y;
        }
        Dim(r)
    }

    fn neg(self) -> Dim {
        Dim(self.0.map(|x| -x))
    }

    fn sub(self, o: Dim) -> Dim {
        self.add(o.neg())
    }

    /// `self * p / q` if every exponent stays a whole number of sixths.
    fn scale(self, p: i16, q: i16) -> Option<Dim> {
        let mut r = self.0;
        for x in &mut r {
            let v = i32::from(*x) * i32::from(p);
            if v % i32::from(q) != 0 {
                return None;
            }
            *x = i16::try_from(v / i32::from(q)).ok()?;
        }
        Some(Dim(r))
    }

    fn ok(self) -> bool {
        self.0.iter().all(|e| e.abs() <= MAX_EXP)
    }

    fn is_zero(self) -> bool {
        self == ZERO
    }
}

/// Base units, in `Dim` order, with the spellings the generator uses (the
/// first is the plain SI one).
const BASES: [&[&str]; 5] = [
    &[
        "m", "km", "cm", "mm", "μm", "um", "nm", "pm", "dm", "hm", "dam", "Mm", "fm", "m",
    ],
    &[
        "s", "ms", "μs", "ns", "min", "h", "day", "yr", "ks", "cs", "s",
    ],
    &["kg", "g", "mg", "μg", "t", "Mg", "cg", "kg"],
    &["K", "mK", "kK", "K"],
    &["A", "mA", "kA", "nA", "A"],
];

/// Units with their own names, by dimension.
fn named_units() -> Vec<(Dim, &'static [&'static str])> {
    vec![
        (
            dim(1, 0, 0, 0, 0),
            &["m", "km", "cm", "mm", "μm", "nm", "pc", "AU", "ly"],
        ),
        (
            dim(0, 1, 0, 0, 0),
            &["s", "ms", "μs", "ns", "min", "h", "day", "yr", "Gyr"],
        ),
        (dim(0, 0, 1, 0, 0), &["kg", "g", "mg", "t"]),
        (dim(0, 0, 0, 1, 0), &["K", "mK"]),
        (dim(0, 0, 0, 0, 1), &["A", "mA"]),
        (dim(0, -1, 0, 0, 0), &["Hz", "kHz", "MHz", "GHz"]),
        (dim(1, -2, 1, 0, 0), &["N", "kN", "mN"]),
        (
            dim(2, -2, 1, 0, 0),
            &["J", "kJ", "mJ", "eV", "keV", "MeV", "GeV", "meV"],
        ),
        (dim(2, -3, 1, 0, 0), &["W", "kW", "mW", "MW"]),
        (dim(-1, -2, 1, 0, 0), &["Pa", "kPa", "hPa", "MPa"]),
        (dim(0, 1, 0, 0, 1), &["C", "mC"]),
        (dim(2, -3, 1, 0, -1), &["V", "mV", "kV"]),
        (dim(2, -3, 1, 0, -2), &["ohm", "kohm", "Ω"]),
        (dim(-2, 4, -1, 0, 2), &["F", "mF", "μF"]),
        (dim(0, -2, 1, 0, -1), &["T"]),
        (dim(3, 0, 0, 0, 0), &["L", "mL"]),
        (dim(2, 0, 0, 0, 0), &["m^2", "km^2", "cm^2"]),
        (dim(1, -1, 0, 0, 0), &["m/s", "km/h", "cm/s"]),
        (dim(1, -2, 0, 0, 0), &["m/s^2", "km/h^2"]),
        (dim(-1, 0, 0, 0, 0), &["m^-1", "cm^-1"]),
    ]
}

/// The predefined constants: name, dimension.
const CONSTANTS: &[(&str, Dim)] = &[
    ("c", dim(1, -1, 0, 0, 0)),
    ("g_n", dim(1, -2, 0, 0, 0)),
    ("pi", ZERO),
    ("π", ZERO),
    ("h", dim(2, -1, 1, 0, 0)),
    ("hbar", dim(2, -1, 1, 0, 0)),
    ("ħ", dim(2, -1, 1, 0, 0)),
    ("k_B", dim(2, -2, 1, -1, 0)),
    ("G", dim(3, -2, -1, 0, 0)),
    ("m_e", dim(0, 0, 1, 0, 0)),
    ("m_p", dim(0, 0, 1, 0, 0)),
    ("q_e", dim(0, 1, 0, 0, 1)),
    ("sigma_SB", dim(0, -3, 1, -4, 0)),
];

/// Dimensions worth having variables of.
fn palette() -> Vec<Dim> {
    vec![
        ZERO,
        ZERO,
        ZERO,
        dim(1, 0, 0, 0, 0),
        dim(1, 0, 0, 0, 0),
        dim(0, 1, 0, 0, 0),
        dim(0, 1, 0, 0, 0),
        dim(0, 0, 1, 0, 0),
        dim(1, -1, 0, 0, 0),
        dim(1, -2, 0, 0, 0),
        dim(1, -2, 1, 0, 0),
        dim(2, -2, 1, 0, 0),
        dim(0, -1, 0, 0, 0),
        dim(2, 0, 0, 0, 0),
        dim(0, 0, 0, 1, 0),
        dim(0, 0, 0, 0, 1),
    ]
}

fn exponent_text(sixths: i16) -> String {
    // `sixths` is positive here.
    if sixths % 6 == 0 {
        return (sixths / 6).to_string();
    }
    let g = gcd(sixths, 6);
    format!("({}/{})", sixths / g, 6 / g)
}

fn gcd(a: i16, b: i16) -> i16 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Num(Dim),
    Bool,
    Str,
}

#[derive(Clone, Debug)]
struct Var {
    name: String,
    ty: Ty,
    /// Plain assignment (same type) may target it.
    assign: bool,
    /// `+=` and friends may target it.
    update: bool,
    /// Bound at module level: what a function body sees.
    module: bool,
    /// A later plain assignment at the top of its scope may give it another
    /// type (not annotated, not a parameter or counter or constant).
    retype: bool,
}

#[derive(Clone, Debug)]
struct FnSig {
    name: String,
    params: Vec<Dim>,
    /// `None`: a procedure (no value).
    ret: Option<Ty>,
    /// Depth-bounded: the first parameter counts down.
    recursive: bool,
    /// Unannotated parameters: `params` are the dimensions the body was
    /// written for (its "instance"); other arguments give run-time errors
    /// (or values) that both engines must agree on.
    poly: bool,
    /// A rough count of the operations one call runs.
    weight: u64,
}

/// An expression's text and how tightly it binds (the levels of the
/// precedence table in the spec: 0 `if`/`else`, 1 `or`, 2 `and`, 3 `not`,
/// 4 comparison, 5 `+ -`, 6 `* /`, 7 prefix, 8 `^` and unit literals, 9
/// atoms and postfix forms).
struct E {
    s: String,
    p: u8,
}

fn atom(s: impl Into<String>) -> E {
    E { s: s.into(), p: 9 }
}

/// The text of `e`, parenthesized unless it binds at least as tightly as `p`.
fn at(e: E, p: u8) -> String {
    if e.p >= p {
        e.s
    } else {
        format!("({})", e.s)
    }
}

/// The state of the recursive function being generated.
struct Rec {
    idx: usize,
    n: String,
    sites: u32,
    /// No base case: recursion runs into the depth limit.
    unbounded: bool,
}

struct Gen<'a> {
    c: Choices<'a>,
    out: String,
    ind: usize,
    vars: Vec<Var>,
    fns: Vec<FnSig>,
    counter: usize,
    /// Percent of risky operations left unguarded.
    risk: usize,
    /// Statements left to generate.
    budget: i32,
    in_fn: bool,
    ret: Option<Ty>,
    loops: usize,
    /// Product of the loop bounds around the current statement.
    mult: u64,
    /// Estimated cost of the function or module generated so far.
    weight: u64,
    rec: Option<Rec>,
    /// The function being generated.
    cur: Option<usize>,
    /// Builtins a `def` shadowed (the generator stops calling them).
    shadowed: Vec<&'static str>,
    /// Constants may be rebound.
    shadow_consts: bool,
    /// No math functions in this program (see GATE_SIR_TEXT_MATH).
    no_math: bool,
    /// The identity function `def u(a): return a` of this program, if it has
    /// one: its result has a type the checker cannot know, so a call of it
    /// can stand where a Bool goes whatever its argument is.
    unknown: Option<String>,
}

/// Known bugs the generator steers around, so the fuzzer keeps finding new
/// ones. Each gate is `true` while its bug is open; the `#[ignore]`d
/// regression test named in its comment (in
/// `crates/sky-ir-interp/tests/fuzz_differential.rs`) reproduces it. Flip a
/// gate to `false` when the bug is fixed.
///
/// Some reported differences the generator never produces, by construction
/// (see the tests of the same file): a conditionally defined function, a
/// nested `def`, an input declaration, and a literal that overflows `f64` in
/// a program that must run (huge numbers get plain SI units). A function may
/// be defined again, and a `def` may shadow a builtin, after functions that
/// call the name (calls bind where the caller is defined,
/// `lowering_binds_a_call_to_the_definition_seen_at_check_time`).
///
/// `lowering_loses_a_name_bound_beside_a_break_arm`: fixed (a name bound only
/// by the arms of an `if` that fall through, when another arm `break`s or
/// `continue`s, was "cannot find `t` in this scope" if read after the `if`
/// inside the loop).
const GATE_MERGE_AFTER_JUMP: bool = false;

/// `lowering_ignores_a_constant_rebound_in_a_block`: fixed (rebinding a
/// predefined constant (`c = 2 m/s`) inside an `if` or `while`, at module
/// level or in a function, was ignored after the block).
const GATE_CONST_REBIND_IN_BLOCK: bool = false;

/// `sir_text_parse_accepts_the_math_operations_the_printer_writes`: fixed
/// (the SIR parser reads `math.*`), so the generator uses math freely.
const GATE_SIR_TEXT_MATH: bool = false;

/// `sir_does_not_check_the_right_operand_of_and_or_for_bool` and
/// `the_left_operand_of_and_is_worded_as_an_if_in_sir`: fixed (a value the
/// checker cannot type, from an unannotated parameter, that is not a Bool, as
/// an operand of `and`/`or`: the SIR interpreter did not check a right
/// operand at all, and worded a left one as an `if`; lowering now emits
/// `expect.bool` for it). With the gate on, the generator only puts
/// Bool-returning functions in such places.
const GATE_NON_BOOL_OPERAND: bool = false;

const MATH_FNS: &[&str] = &[
    "sqrt", "cbrt", "abs", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos",
    "atan", "atan2", "hypot", "min", "max", "floor", "ceil", "round",
];

/// A budget for one program's estimated run time, in "operations".
const COST_CAP: u64 = 500;

const NAMES: &[&str] = &["v", "w", "x", "y", "z", "θ", "ω", "val", "tmp"];
const SHADOWABLE: &[&str] = &["inv", "abs", "max", "min", "hypot", "sqrt", "exp", "round"];

impl Gen<'_> {
    // ----- output helpers -----

    fn line(&mut self, s: &str) {
        for _ in 0..self.ind {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn fresh_name(&mut self) -> String {
        let p = *self.c.pick(NAMES);
        self.fresh(p)
    }

    /// Whether the builtin `name` may be called: not shadowed by a `def`,
    /// and not a math function in a program that avoids them.
    fn builtin(&self, name: &str) -> bool {
        !(self.shadowed.contains(&name) || (self.no_math && MATH_FNS.contains(&name)))
    }

    // ----- numbers and units -----

    fn number(&mut self) -> String {
        match self.c.weighted(&[40, 12, 20, 8, 3, 4]) {
            0 => {
                // Zero is rare unless the program is risky.
                if self.c.chance(self.risk / 2 + 3) {
                    "0".to_string()
                } else {
                    (1 + self.c.below(9)).to_string()
                }
            }
            1 => self
                .c
                .pick(&["10", "12", "100", "360", "1000", "255"])
                .to_string(),
            2 => self
                .c
                .pick(&[
                    "0.5",
                    "2.5",
                    "0.25",
                    "1.5",
                    "3.14",
                    "0.1",
                    "9.81",
                    "0.75",
                    "2.0",
                    "0.0001",
                    "0.00009",
                    "12345.678",
                    "0.3",
                    "1.0",
                ])
                .to_string(),
            3 => self
                .c
                .pick(&[
                    "1e3",
                    "2.5e-2",
                    "1e-3",
                    "6e2",
                    "1.5e1",
                    "1E2",
                    "4e0",
                    "1e15",
                    "1e16",
                    "1e17",
                    "1e-4",
                    "1e-5",
                    "1e21",
                    "1e22",
                    "9007199254740993",
                    "4.35e-320",
                    "5e-324",
                ])
                .to_string(),
            4 => self.c.pick(&["1_000", "2_5", "1_0.5"]).to_string(),
            _ => {
                // Big and small: overflow and underflow.
                if self.c.chance(self.risk) {
                    self.c
                        .pick(&["1e150", "1e200", "1e300", "1e-300", "1e-200", "1e100"])
                        .to_string()
                } else {
                    (1 + self.c.below(5)).to_string()
                }
            }
        }
    }

    /// A unit of dimension `d`, spelled with a prefix or a named form.
    fn unit(&mut self, d: Dim) -> String {
        self.unit_with(d, false)
    }

    /// `plain`: SI base units only (scale 1), so a huge number times the unit
    /// cannot overflow while the literal is converted.
    fn unit_with(&mut self, d: Dim, plain: bool) -> String {
        if !plain {
            if let Some((_, names)) = named_units().into_iter().find(|(nd, _)| *nd == d) {
                if self.c.chance(60) {
                    return self.c.pick(names).to_string();
                }
            }
        }
        let prefixed = !plain && self.c.chance(50);
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for (i, e) in d.0.iter().enumerate() {
            let spelling = if prefixed {
                *self.c.pick(BASES[i])
            } else {
                BASES[i][0]
            };
            match e.cmp(&0) {
                std::cmp::Ordering::Greater => pos.push((spelling, *e)),
                std::cmp::Ordering::Less => neg.push((spelling, -*e)),
                std::cmp::Ordering::Equal => {}
            }
        }
        let mut s = String::new();
        if pos.is_empty() {
            let (n, e) = neg.remove(0);
            // Only negative exponents: the first factor carries its sign.
            s.push_str(&if e % 6 == 0 {
                format!("{n}^-{}", e / 6)
            } else {
                format!("{n}^(-{}", &exponent_text(e)[1..])
            });
        } else {
            let parts: Vec<String> = pos
                .iter()
                .map(|(n, e)| {
                    if *e == 6 {
                        (*n).to_string()
                    } else {
                        format!("{n}^{}", exponent_text(*e))
                    }
                })
                .collect();
            s.push_str(&parts.join("*"));
        }
        for (n, e) in neg {
            s.push('/');
            s.push_str(&if e == 6 {
                n.to_string()
            } else {
                format!("{n}^{}", exponent_text(e))
            });
        }
        s
    }

    /// The annotation for a value of dimension `d`.
    fn annotation(&mut self, d: Dim) -> String {
        if d.is_zero() {
            "Scalar".to_string()
        } else {
            self.unit(d)
        }
    }

    fn literal(&mut self, d: Dim) -> E {
        let n = self.number();
        self.literal_with(d, n)
    }

    /// A literal that is not zero, for guards.
    fn positive_literal(&mut self, d: Dim) -> E {
        let n = self.c.pick(&["1", "2", "5", "0.5", "10", "3.14", "1e3"]);
        self.literal_with(d, (*n).to_string())
    }

    fn literal_with(&mut self, d: Dim, n: String) -> E {
        if d.is_zero() {
            if self.c.chance(4) {
                return E {
                    s: format!("{n} rad"),
                    p: 8,
                };
            }
            return atom(n);
        }
        // A literal whose conversion to SI overflows is refused by lowering
        // as a whole (see KNOWN), so huge numbers get plain SI units.
        let big = n.starts_with("1e") && n.len() > 3 && !n.starts_with("1e-");
        let u = self.unit_with(d, big);
        E {
            s: format!("{n} {u}"),
            p: 8,
        }
    }

    fn rand_dim(&mut self) -> Dim {
        let known: Vec<Dim> = self
            .vars
            .iter()
            .filter_map(|v| match v.ty {
                Ty::Num(d) => Some(d),
                Ty::Bool | Ty::Str => None,
            })
            .collect();
        if !known.is_empty() && self.c.chance(60) {
            return *self.c.pick(&known);
        }
        let p = palette();
        *self.c.pick(&p)
    }

    // ----- expressions -----

    fn leaf(&mut self, d: Dim) -> E {
        let names: Vec<String> = self
            .vars
            .iter()
            .filter(|v| v.ty == Ty::Num(d))
            .map(|v| v.name.clone())
            .collect();
        if !names.is_empty() && self.c.chance(75) {
            return atom(self.c.pick(&names).clone());
        }
        self.literal(d)
    }

    /// A numeric expression of dimension `d`.
    fn num(&mut self, d: Dim, depth: u32) -> E {
        if depth == 0 || self.c.exhausted() || self.c.chance(20) {
            return self.leaf(d);
        }
        let w = [22, 14, 12, 5, 8, 14, 9, 6, 3];
        match self.c.weighted(&w) {
            0 => {
                let plus = self.c.chance(60);
                let a = self.num(d, depth - 1);
                let mut b = self.num(d, depth - 1);
                if b.s == a.s && self.c.chance(85) {
                    // `x - x` is zero: keep it rare.
                    b = self.literal(d);
                }
                E {
                    s: format!("{} {} {}", at(a, 5), if plus { "+" } else { "-" }, at(b, 6)),
                    p: 5,
                }
            }
            1 => {
                let d1 = self.rand_dim();
                let d2 = d.sub(d1);
                if !d2.ok() {
                    return self.leaf(d);
                }
                let a = self.num(d1, depth - 1);
                let b = self.num(d2, depth - 1);
                E {
                    s: format!("{} * {}", at(a, 6), at(b, 7)),
                    p: 6,
                }
            }
            2 => {
                let d1 = self.rand_dim();
                let d2 = d.add(d1);
                if !d2.ok() {
                    return self.leaf(d);
                }
                let a = self.num(d2, depth - 1);
                let b = self.denominator(d1, depth - 1);
                E {
                    s: format!("{} / {}", at(a, 6), at(b, 7)),
                    p: 6,
                }
            }
            3 => {
                let a = self.num(d, depth - 1);
                if self.c.chance(70) || !self.builtin("abs") {
                    E {
                        s: format!("-{}", at(a, 8)),
                        p: 7,
                    }
                } else if self.c.chance(30) {
                    E {
                        s: format!("+{}", at(a, 8)),
                        p: 7,
                    }
                } else {
                    atom(format!("abs({})", a.s))
                }
            }
            4 => self.power(d, depth),
            5 => self.math(d, depth),
            6 => self.call(d, depth),
            7 => {
                let cnd = self.boolean(depth - 1);
                let a = self.num(d, depth - 1);
                let b = self.num(d, depth - 1);
                E {
                    s: format!("{} if {} else {}", at(a, 1), at(cnd, 1), at(b, 0)),
                    p: 0,
                }
            }
            _ => {
                // The inverse has the inverse dimension.
                let a = self.denominator(d.neg(), depth - 1);
                if self.c.chance(50) && self.builtin("inv") {
                    atom(format!("inv({})", a.s))
                } else {
                    // `x⁻¹` after a unit literal is ambiguous: parenthesize.
                    let base = if a.p >= 9 { a.s } else { format!("({})", a.s) };
                    atom(format!("{base}⁻¹"))
                }
            }
        }
    }

    /// A divisor of dimension `d`: usually not zero, sometimes anything.
    fn denominator(&mut self, d: Dim, depth: u32) -> E {
        let e = self.num(d, depth);
        if self.c.chance(self.risk) || !self.builtin("abs") {
            return e;
        }
        let lit = self.positive_literal(d);
        E {
            s: format!("abs({}) + {}", e.s, at(lit, 6)),
            p: 5,
        }
    }

    fn power(&mut self, d: Dim, depth: u32) -> E {
        // (numerator, denominator, spelling)
        const EXPS: &[(i16, i16, &str)] = &[
            (2, 1, "2"),
            (3, 1, "3"),
            (-1, 1, "-1"),
            (-2, 1, "-2"),
            (1, 2, "(1/2)"),
            (1, 2, "0.5"),
            (1, 3, "(1/3)"),
            (3, 2, "(3/2)"),
            (3, 2, "1.5"),
            (2, 3, "(2/3)"),
            (-1, 2, "-(1/2)"),
            (4, 1, "4"),
            (1, 1, "1"),
            (-3, 2, "(-3/2)"),
            (5, 1, "5"),
        ];
        if d.is_zero() && self.c.chance(45) {
            // A dimensionless base takes any dimensionless exponent, a
            // constant one such as 64 and 100 (repeated multiplication or
            // `powf`) or one computed at run time.
            if self.c.chance(45) {
                let exp = *self.c.pick(&[
                    "64", "65", "100", "-64", "10", "0.5", "2.5", "-0.5", "0", "1e2", "7", "-7",
                    "63", "0.1", "1e-3", "20",
                ]);
                // A negative or fractional exponent needs a positive base.
                let base = if exp.starts_with('-') || exp.contains('.') || exp.contains("e-") {
                    self.denominator(ZERO, depth - 1)
                } else {
                    self.num(ZERO, depth - 1)
                };
                return E {
                    s: format!("{}^{exp}", at(base, 9)),
                    p: 8,
                };
            }
            let base = self.num(ZERO, depth - 1);
            let e = self.num(ZERO, depth - 1);
            return E {
                s: format!("{}^{}", at(base, 9), at(e, 8)),
                p: 8,
            };
        }
        let (p, q, text) = *self.c.pick(EXPS);
        let Some(base_dim) = d.scale(q, p) else {
            return self.leaf(d);
        };
        if !base_dim.ok() {
            return self.leaf(d);
        }
        let base = if p < 0 || q != 1 {
            self.denominator(base_dim, depth - 1)
        } else {
            self.num(base_dim, depth - 1)
        };
        E {
            s: format!("{}^{text}", at(base, 9)),
            p: 8,
        }
    }

    fn math(&mut self, d: Dim, depth: u32) -> E {
        if self.no_math {
            return self.leaf(d);
        }
        let dl = depth - 1;
        let guard = self.risk;
        if !d.is_zero() {
            return match self.c.below(5) {
                0 if self.builtin("sqrt") => {
                    let Some(ad) = d.scale(2, 1).filter(|x| x.ok()) else {
                        return self.leaf(d);
                    };
                    let a = self.num(ad, dl);
                    if self.c.chance(guard) || !self.builtin("abs") {
                        atom(format!("sqrt({})", a.s))
                    } else {
                        atom(format!("sqrt(abs({}))", a.s))
                    }
                }
                1 => {
                    let Some(ad) = d.scale(3, 1).filter(|x| x.ok()) else {
                        return self.leaf(d);
                    };
                    let a = self.num(ad, dl);
                    atom(format!("cbrt({})", a.s))
                }
                2 if self.builtin("abs") => {
                    let a = self.num(d, dl);
                    atom(format!("abs({})", a.s))
                }
                3 if self.builtin("hypot") => {
                    let a = self.num(d, dl);
                    let b = self.num(d, dl);
                    atom(format!("hypot({}, {})", a.s, b.s))
                }
                _ => {
                    let f = if self.c.chance(50) && self.builtin("max") {
                        "max"
                    } else if self.builtin("min") {
                        "min"
                    } else {
                        return self.leaf(d);
                    };
                    let a = self.num(d, dl);
                    let b = self.num(d, dl);
                    atom(format!("{f}({}, {})", a.s, b.s))
                }
            };
        }
        let risky = self.c.chance(guard);
        match self.c.below(14) {
            0 if self.builtin("exp") => {
                let a = self.num(ZERO, dl);
                if risky {
                    // Can overflow.
                    atom(format!("exp({} * 100)", at(a, 6)))
                } else {
                    atom(format!("exp({})", a.s))
                }
            }
            1 => {
                let a = self.num(ZERO, dl);
                atom(format!("sin({})", a.s))
            }
            2 => {
                let a = self.num(ZERO, dl);
                atom(format!("cos({})", a.s))
            }
            3 => {
                let a = self.num(ZERO, dl);
                atom(format!("tan({})", a.s))
            }
            4..=6 => {
                let f = *self.c.pick(&["log", "log10", "log2"]);
                let a = self.num(ZERO, dl);
                if risky || !self.builtin("abs") {
                    atom(format!("{f}({})", a.s))
                } else {
                    atom(format!("{f}(abs({}) + 1)", a.s))
                }
            }
            7 | 8 => {
                let f = *self.c.pick(&["asin", "acos"]);
                let a = self.num(ZERO, dl);
                if risky {
                    atom(format!("{f}({})", a.s))
                } else {
                    atom(format!("{f}(sin({}))", a.s))
                }
            }
            9 => {
                let a = self.num(ZERO, dl);
                atom(format!("atan({})", a.s))
            }
            10 => {
                let ad = self.rand_dim();
                let a = self.num(ad, dl);
                let b = self.num(ad, dl);
                atom(format!("atan2({}, {})", a.s, b.s))
            }
            11..=13 => {
                let f = *self.c.pick(&["floor", "ceil", "round"]);
                if f == "round" && !self.builtin("round") {
                    return self.leaf(d);
                }
                let a = self.num(ZERO, dl);
                atom(format!("{f}({})", a.s))
            }
            _ => self.leaf(d),
        }
    }

    /// A call of a user function returning `d`, if one fits the cost budget.
    fn call(&mut self, d: Dim, depth: u32) -> E {
        let want = Some(Ty::Num(d));
        let mut cands: Vec<usize> = Vec::new();
        for (i, f) in self.fns.iter().enumerate() {
            if f.ret != want {
                continue;
            }
            let is_current = self.rec.as_ref().is_some_and(|r| r.idx == i);
            if is_current {
                let r = self.rec.as_ref().expect("rec");
                if r.sites < 2 && self.mult == 1 {
                    cands.push(i);
                }
            } else if self.cur == Some(i) {
                // A plain function does not call itself.
            } else if self.weight + f.weight * self.mult <= COST_CAP {
                cands.push(i);
            }
        }
        if cands.is_empty() {
            return self.leaf(d);
        }
        let i = *self.c.pick(&cands);
        self.call_expr(i, depth)
    }

    fn call_expr(&mut self, i: usize, depth: u32) -> E {
        let f = self.fns[i].clone();
        let is_current = self.rec.as_ref().is_some_and(|r| r.idx == i);
        let mut args = Vec::new();
        for (k, pd) in f.params.iter().enumerate() {
            if k == 0 && f.recursive {
                if is_current {
                    let r = self.rec.as_mut().expect("rec");
                    r.sites += 1;
                    let n = r.n.clone();
                    args.push(if r.unbounded {
                        format!("{n} + 1")
                    } else {
                        format!("{n} - 1")
                    });
                } else if self.c.chance(60) {
                    args.push(self.c.below(5).to_string());
                } else {
                    let e = self.num(ZERO, depth.saturating_sub(1));
                    args.push(format!("min({}, 4)", e.s));
                    if !self.builtin("min") {
                        // `min` is user code now: use a literal instead.
                        args.pop();
                        args.push(self.c.below(5).to_string());
                    }
                }
                continue;
            }
            // The other arguments of an unbounded recursion are plain
            // values: nothing may print at each level, because where the
            // depth limit trips differs between the engines.
            let unbounded = is_current && self.rec.as_ref().is_some_and(|r| r.unbounded);
            let e = if unbounded {
                self.leaf(*pd)
            } else {
                self.num(*pd, depth.saturating_sub(1))
            };
            args.push(e.s);
        }
        if !is_current {
            self.weight += f.weight * self.mult;
        } else {
            self.weight += self.mult;
        }
        atom(format!("{}({})", f.name, args.join(", ")))
    }

    fn boolean(&mut self, depth: u32) -> E {
        if depth == 0 || self.c.exhausted() || self.c.chance(15) {
            let bools: Vec<String> = self
                .vars
                .iter()
                .filter(|v| v.ty == Ty::Bool)
                .map(|v| v.name.clone())
                .collect();
            if self.c.chance(12) {
                return atom(if self.c.chance(50) { "true" } else { "false" });
            }
            if !bools.is_empty() && self.c.chance(50) {
                return atom(self.c.pick(&bools).clone());
            }
            return self.comparison(depth.min(2));
        }
        match self.c.weighted(&[45, 22, 8, 22, 6, 6, 4]) {
            0 => self.comparison(depth),
            5 => self.bool_call(depth),
            6 => {
                let a = self.boolean(depth - 1);
                let cnd = self.boolean(depth - 1);
                let b = self.boolean(depth - 1);
                E {
                    s: format!("{} if {} else {}", at(a, 1), at(cnd, 1), at(b, 0)),
                    p: 0,
                }
            }
            1 => {
                let n = 2 + usize::from(self.c.chance(25));
                let ops: Vec<String> = (0..n)
                    .map(|_| {
                        let e = self.boolean(depth - 1);
                        at(e, 3)
                    })
                    .collect();
                E {
                    s: ops.join(" and "),
                    p: 2,
                }
            }
            2 => {
                let e = self.boolean(depth - 1);
                E {
                    s: format!("not {}", at(e, 3)),
                    p: 3,
                }
            }
            3 => {
                let n = 2 + usize::from(self.c.chance(25));
                let ops: Vec<String> = (0..n)
                    .map(|_| {
                        let e = self.boolean(depth - 1);
                        at(e, 2)
                    })
                    .collect();
                E {
                    s: ops.join(" or "),
                    p: 1,
                }
            }
            _ => {
                // A guarded division: the right operand must not run (or
                // must, for `or`) depending on the left.
                let d1 = self.rand_dim();
                let x = self.num(d1, 1);
                let x_text = at(x, 5);
                let zero = self.zero(d1);
                let num = self.num(d1, 1);
                let lit = self.literal(ZERO);
                let quotient = format!("{} / {} > {}", at(num, 6), paren(&x_text, 7), at(lit, 5));
                if self.c.chance(50) {
                    E {
                        s: format!("{x_text} != {zero} and {quotient}"),
                        p: 2,
                    }
                } else {
                    E {
                        s: format!("{x_text} == {zero} or {quotient}"),
                        p: 1,
                    }
                }
            }
        }
    }

    /// A call of a Bool-returning function, or a comparison.
    fn bool_call(&mut self, depth: u32) -> E {
        let cands: Vec<usize> = (0..self.fns.len())
            .filter(|&i| {
                let f = &self.fns[i];
                f.ret == Some(Ty::Bool)
                    && self.cur != Some(i)
                    && self.weight + f.weight * self.mult <= COST_CAP
            })
            .collect();
        if cands.is_empty() {
            return self.comparison(depth.min(2));
        }
        let i = *self.c.pick(&cands);
        self.call_expr(i, depth)
    }

    fn zero(&mut self, d: Dim) -> String {
        if d.is_zero() {
            "0".to_string()
        } else {
            let u = self.unit(d);
            format!("0 {u}")
        }
    }

    fn comparison(&mut self, depth: u32) -> E {
        if self.c.chance(5) {
            // Strings and Bools compare for equality.
            let op = *self.c.pick(&["==", "!="]);
            if self.c.chance(50) {
                let a = self.string(1);
                let b = self.string(1);
                return E {
                    s: format!("{} {op} {}", at(a, 5), at(b, 5)),
                    p: 4,
                };
            }
            let a = self.boolean(1);
            let b = self.boolean(1);
            return E {
                s: format!("{} {op} {}", at(a, 5), at(b, 5)),
                p: 4,
            };
        }
        let d = self.rand_dim();
        let a = self.num(d, depth.min(2));
        let b = if self.c.chance(15) {
            let z = self.zero(d);
            E {
                s: z,
                p: if d.is_zero() { 9 } else { 8 },
            }
        } else {
            self.num(d, depth.min(2))
        };
        let op = *self
            .c
            .pick(&["<", "<=", ">", ">=", "==", "!=", "≤", "≥", "≠"]);
        E {
            s: format!("{} {op} {}", at(a, 5), at(b, 5)),
            p: 4,
        }
    }
}

/// `text` in parentheses unless it is an atom or a call (a helper for text
/// that is already flattened).
fn paren(text: &str, _p: u8) -> String {
    if text.chars().all(|ch| ch.is_alphanumeric() || ch == '_') {
        text.to_string()
    } else {
        format!("({text})")
    }
}

// ----- statements -----

/// How a statement or block ends.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// Control can reach the next statement.
    Falls,
    /// Every path returns, breaks or continues.
    Diverges,
}

impl Gen<'_> {
    fn ty_expr(&mut self, ty: Ty, depth: u32) -> E {
        match ty {
            Ty::Num(d) => self.num(d, depth),
            Ty::Bool => self.boolean(depth),
            Ty::Str => self.string(depth.min(2)),
        }
    }

    fn rand_ty(&mut self) -> Ty {
        match self.c.weighted(&[84, 12, 4]) {
            0 => Ty::Num(self.rand_dim()),
            1 => Ty::Bool,
            _ => Ty::Str,
        }
    }

    fn string(&mut self, depth: u32) -> E {
        let vars: Vec<String> = self
            .vars
            .iter()
            .filter(|v| v.ty == Ty::Str)
            .map(|v| v.name.clone())
            .collect();
        if depth > 0 && self.c.chance(15) {
            let a = self.string(depth - 1);
            let cnd = self.boolean(depth - 1);
            let b = self.string(depth - 1);
            return E {
                s: format!("{} if {} else {}", at(a, 1), at(cnd, 1), at(b, 0)),
                p: 0,
            };
        }
        if !vars.is_empty() && self.c.chance(50) {
            return atom(self.c.pick(&vars).clone());
        }
        let lit = *self.c.pick(&[
            "\"hi\"",
            "\"a\\tb\"",
            "\"line\\n\"",
            "\"é∂\"",
            "\"x =\"",
            "\"\"",
            "\"q\\\"q\"",
            "\"back\\\\slash\"",
        ]);
        atom(lit)
    }

    fn tick(&mut self) {
        self.weight += self.mult;
    }

    /// The statements of an indented block. Never empty; stops after a
    /// statement that diverges (nothing follows an unconditional exit).
    fn block(&mut self, depth: usize) -> Exit {
        let mark = self.vars.len();
        self.ind += 1;
        let before = self.out.len();
        let n = 1 + self.c.below(3);
        let mut exit = Exit::Falls;
        for _ in 0..n {
            if self.budget <= 0 || self.c.exhausted() {
                break;
            }
            exit = self.stmt(depth);
            if exit == Exit::Diverges {
                break;
            }
        }
        if self.out.len() == before {
            self.line("pass");
        }
        self.ind -= 1;
        self.vars.truncate(mark);
        exit
    }

    /// May a statement here `return`?
    fn can_return(&self) -> bool {
        self.in_fn
    }

    /// May a statement here `break` or `continue`?
    fn can_jump(&self) -> bool {
        self.loops > 0
    }

    fn stmt(&mut self, depth: usize) -> Exit {
        self.budget -= 1;
        self.tick();
        if self.c.chance(4) {
            let c = *self
                .c
                .pick(&["# note", "# x = 1 / 0", "#", "# ok: 1 m + 2 s"]);
            self.line(c);
        }
        if self.c.chance(3) {
            self.out.push('\n');
        }
        let nested = depth < 3 && self.budget > 3;
        let loopable = nested && self.loops < 3 && self.mult <= 16;
        let exits = self.can_return() || self.can_jump();
        let w = [
            14,                            // 0 new name
            14,                            // 1 assign
            8,                             // 2 compound
            16,                            // 3 print
            3,                             // 4 expression statement
            if nested { 13 } else { 0 },   // 5 if
            if nested { 8 } else { 0 },    // 6 merge
            if loopable { 11 } else { 0 }, // 7 while
            if exits { 7 } else { 0 },     // 8 exit
            1,                             // 9 pass
            5,                             // 10 annotated
            if self.fns.iter().any(|f| f.poly) {
                14
            } else {
                0
            }, // 11 probe
        ];
        match self.c.weighted(&w) {
            0 => {
                self.new_name(None);
                Exit::Falls
            }
            1 => {
                self.assign();
                Exit::Falls
            }
            2 => {
                self.compound();
                Exit::Falls
            }
            3 => {
                self.print();
                Exit::Falls
            }
            4 => {
                let ty = self.rand_ty();
                let e = self.ty_expr(ty, 2);
                self.line(&e.s);
                Exit::Falls
            }
            5 => self.if_stmt(depth),
            6 => self.merge_stmt(depth),
            7 => self.while_stmt(depth),
            8 => self.exit_stmt(),
            9 => {
                self.line("pass");
                Exit::Falls
            }
            10 => {
                self.annotated();
                Exit::Falls
            }
            _ => {
                self.poly_probe();
                Exit::Falls
            }
        }
    }

    /// Calls a function with unannotated parameters with whatever arguments:
    /// often the dimensions its body was written for, often not (another
    /// dimension, a Bool, a string). The result is not kept, so the
    /// generator's types stay sound; run-time errors must agree.
    fn poly_probe(&mut self) {
        let cands: Vec<usize> = (0..self.fns.len())
            .filter(|&i| {
                let f = &self.fns[i];
                f.poly && self.cur != Some(i) && self.weight + f.weight * self.mult <= COST_CAP
            })
            .collect();
        if cands.is_empty() {
            return self.print();
        }
        let f = self.fns[*self.c.pick(&cands)].clone();
        let mut args = Vec::new();
        for pd in &f.params {
            if self.c.chance(35) {
                args.push(self.num(*pd, 1).s);
            } else {
                let ty = self.rand_ty();
                args.push(self.ty_expr(ty, 1).s);
            }
        }
        self.weight += f.weight * self.mult;
        let mut call = format!("{}({})", f.name, args.join(", "));
        // A call can be an operand or a condition where the checker does not
        // know its type: a Bool works, and any other result is the run-time
        // error "`kw` needs a Bool operand", the same in both engines (see
        // GATE_NON_BOOL_OPERAND). What it returns is only unknown to the
        // checker through the identity function.
        let mut unknown = f.ret == Some(Ty::Bool);
        if let (Some(u), Some(_)) = (&self.unknown, &f.ret) {
            if self.c.chance(60) {
                call = format!("{u}({call})");
                unknown = true;
            }
        }
        if unknown && self.c.chance(50) {
            match self.c.below(8) {
                0 => {
                    self.line(&format!("if {call}:"));
                    self.ind += 1;
                    self.line("pass");
                    self.ind -= 1;
                }
                1 => self.line(&format!("print({call} and true)")),
                2 => self.line(&format!("print(false or {call})")),
                3 => self.line(&format!("print(true and {call} and true)")),
                4 => self.line(&format!("print(1 if {call} else 2)")),
                5 => {
                    self.line("if false:");
                    self.ind += 1;
                    self.line("pass");
                    self.ind -= 1;
                    self.line(&format!("elif {call}:"));
                    self.ind += 1;
                    self.line("pass");
                    self.ind -= 1;
                }
                6 => {
                    self.line(&format!("while {call}:"));
                    self.ind += 1;
                    self.line("break");
                    self.ind -= 1;
                }
                _ => self.line(&format!("print(not {call})")),
            }
        } else if self.c.chance(70) {
            self.line(&format!("print({call})"));
        } else {
            self.line(&call);
        }
    }

    fn bind(&mut self, name: String, ty: Ty, retype: bool) {
        let module = !self.in_fn && self.ind == 0;
        let update = matches!(ty, Ty::Num(_));
        self.vars.push(Var {
            name,
            ty,
            assign: true,
            update,
            module,
            retype,
        });
    }

    fn new_name(&mut self, ty: Option<Ty>) {
        let ty = ty.unwrap_or_else(|| self.rand_ty());
        let name = self.fresh_name();
        let e = self.ty_expr(ty, 3);
        self.line(&format!("{name} = {}", e.s));
        self.bind(name, ty, true);
    }

    fn annotated(&mut self) {
        let d = self.rand_dim();
        let name = self.fresh_name();
        let ann = self.annotation(d);
        let e = self.num(d, 3);
        self.line(&format!("{name}: {ann} = {}", e.s));
        self.bind(name, Ty::Num(d), false);
    }

    fn assign(&mut self) {
        // Known bug (GATE_CONST_REBIND_IN_BLOCK): a predefined constant is
        // rebound only at the top of a scope, not inside an `if` or a loop.
        let gated = GATE_CONST_REBIND_IN_BLOCK && self.ind > usize::from(self.in_fn);
        let targets: Vec<usize> = (0..self.vars.len())
            .filter(|&i| {
                self.vars[i].assign
                    && !(gated && CONSTANTS.iter().any(|(n, _)| *n == self.vars[i].name))
            })
            .collect();
        if targets.is_empty() {
            return self.new_name(None);
        }
        let i = *self.c.pick(&targets);
        // Straight-line code at the top of a scope may rebind a name with a
        // different type (`x = 1 m`, later `x = 2 s`): the checker follows the
        // flow. Not once a function could read the module's name.
        let top = if self.in_fn {
            self.ind == 1
        } else {
            self.ind == 0 && self.fns.is_empty()
        };
        if self.vars[i].retype && top && self.loops == 0 && self.c.chance(15) {
            let ty = self.rand_ty();
            let e = self.ty_expr(ty, 3);
            let name = self.vars[i].name.clone();
            self.line(&format!("{name} = {}", e.s));
            self.vars[i].ty = ty;
            self.vars[i].update = matches!(ty, Ty::Num(_));
            return;
        }
        let (name, ty) = (self.vars[i].name.clone(), self.vars[i].ty);
        let e = self.ty_expr(ty, 3);
        self.line(&format!("{name} = {}", e.s));
    }

    fn compound(&mut self) {
        let targets: Vec<usize> = (0..self.vars.len())
            .filter(|&i| self.vars[i].update && matches!(self.vars[i].ty, Ty::Num(_)))
            .collect();
        if targets.is_empty() {
            return self.new_name(None);
        }
        let i = *self.c.pick(&targets);
        let name = self.vars[i].name.clone();
        let Ty::Num(d) = self.vars[i].ty else {
            unreachable!("filtered to numbers")
        };
        // Dimension-preserving forms only: `+=`/`-=` with the same
        // dimension, `*=`/`/=` with a scalar, `^=` on a scalar.
        let (op, rhs) = match self.c.below(5) {
            0 | 1 => ("+=", self.num(d, 2)),
            2 => ("-=", self.num(d, 2)),
            3 => {
                if self.c.chance(50) {
                    ("*=", self.num(ZERO, 2))
                } else {
                    ("/=", self.denominator(ZERO, 2))
                }
            }
            _ if d.is_zero() => {
                let e = *self.c.pick(&["2", "3", "0.5", "1", "0"]);
                ("^=", atom(e))
            }
            _ => ("+=", self.num(d, 1)),
        };
        self.line(&format!("{name} {op} {}", rhs.s));
    }

    fn print(&mut self) {
        if self.c.chance(2) {
            self.line("print()");
            return;
        }
        let n = 1 + self.c.below(3);
        let mut args = Vec::new();
        for _ in 0..n {
            let ty = self.rand_ty();
            args.push(self.ty_expr(ty, 3).s);
        }
        self.line(&format!("print({})", args.join(", ")));
    }

    fn exit_stmt(&mut self) -> Exit {
        let mut opts: Vec<u8> = Vec::new();
        if self.can_return() {
            opts.push(0);
        }
        if self.can_jump() {
            opts.push(1);
            opts.push(2);
        }
        match *self.c.pick(&opts) {
            0 => match self.ret {
                Some(ty) => {
                    let e = self.ty_expr(ty, 3);
                    self.line(&format!("return {}", e.s));
                }
                None => self.line("return"),
            },
            1 => self.line("break"),
            _ => self.line("continue"),
        }
        Exit::Diverges
    }

    fn if_stmt(&mut self, depth: usize) -> Exit {
        let cnd = self.boolean(2);
        self.line(&format!("if {}:", cnd.s));
        let mut all = self.block(depth + 1) == Exit::Diverges;
        for _ in 0..self.c.below(3) {
            let cnd = self.boolean(2);
            self.line(&format!("elif {}:", cnd.s));
            all &= self.block(depth + 1) == Exit::Diverges;
        }
        if self.c.chance(45) {
            self.line("else:");
            all &= self.block(depth + 1) == Exit::Diverges;
            if all {
                return Exit::Diverges;
            }
        }
        Exit::Falls
    }

    /// `if`/`elif`/`else` where every arm that falls through binds a new
    /// name `t`, so `t` is defined afterwards; the other arms exit.
    fn merge_stmt(&mut self, depth: usize) -> Exit {
        let ty = self.rand_ty();
        let t = self.fresh("t");
        let arms = 2 + self.c.below(2);
        let keeper = self.c.below(arms);
        let can_exit = self.can_return() || self.can_jump();
        let mut exited = false;
        let mut assigned = false;
        for i in 0..arms {
            if i == 0 {
                let cnd = self.boolean(2);
                self.line(&format!("if {}:", cnd.s));
            } else if i + 1 == arms {
                self.line("else:");
            } else {
                let cnd = self.boolean(2);
                self.line(&format!("elif {}:", cnd.s));
            }
            let mark = self.vars.len();
            self.ind += 1;
            let mut exit = Exit::Falls;
            for _ in 0..self.c.below(3) {
                if self.budget <= 0 || self.c.exhausted() {
                    break;
                }
                if self.stmt(depth + 1) == Exit::Diverges {
                    exit = Exit::Diverges;
                    break;
                }
            }
            // An arm that diverged binds nothing; any other arm binds `t`,
            // or (except the keeper) exits.
            if exit == Exit::Diverges {
                exited = true;
            } else if i != keeper && can_exit && self.c.chance(45) {
                self.exit_stmt();
                exited = true;
            } else {
                let e = self.ty_expr(ty, 2);
                self.line(&format!("{t} = {}", e.s));
                assigned = true;
            }
            self.ind -= 1;
            self.vars.truncate(mark);
        }
        if !assigned {
            // Every arm exited: nothing follows.
            return Exit::Diverges;
        }
        // Known bug (GATE_MERGE_AFTER_JUMP): lowering loses a name that only
        // the arms that do not `break`/`continue` bind, when it is read after.
        if !(GATE_MERGE_AFTER_JUMP && exited && self.loops > 0) {
            self.bind(t, ty, false);
        }
        Exit::Falls
    }

    fn while_stmt(&mut self, depth: usize) -> Exit {
        let k = self.fresh("k");
        self.line(&format!("{k} = 0"));
        self.vars.push(Var {
            name: k.clone(),
            ty: Ty::Num(ZERO),
            assign: false,
            update: false,
            module: !self.in_fn && self.ind == 0,
            retype: false,
        });
        let forever = self.c.chance(25);
        let bound = 1 + self.c.below(4) as u64;
        if forever {
            self.line("while true:");
        } else {
            let extra = if self.c.chance(30) {
                let e = self.boolean(1);
                format!(" and {}", at(e, 3))
            } else {
                String::new()
            };
            self.line(&format!("while {k} < {bound}{extra}:"));
        }
        let mult = self.mult;
        self.mult *= if forever { 4 } else { bound };
        self.loops += 1;
        self.ind += 1;
        if self.c.chance(50) {
            self.line(&format!("{k} = {k} + 1"));
        } else {
            self.line(&format!("{k} += 1"));
        }
        if forever {
            // A loop only `break` leaves defines what its body bound before
            // the `break`: the name is readable after it.
            if self.c.chance(50) {
                let ty = self.rand_ty();
                let name = self.fresh("t");
                let e = self.ty_expr(ty, 2);
                self.line(&format!("{name} = {}", e.s));
                self.bind(name, ty, false);
            }
            self.line(&format!("if {k} > 3:"));
            self.ind += 1;
            self.line("break");
            self.ind -= 1;
        }
        self.ind -= 1;
        self.block(depth + 1);
        self.loops -= 1;
        self.mult = mult;
        Exit::Falls
    }
}

// ----- functions and the program -----

impl Gen<'_> {
    /// A top-level `def`: annotated parameters, a return annotation (mostly),
    /// early returns, and, sometimes, recursion bounded by a depth argument.
    fn function(&mut self) {
        // 0 plain, 1 recursive, 2 procedure, 3 returns a Bool, 4 unbounded,
        // 5 unannotated parameters
        let kind = self
            .c
            .weighted(&[50, 25, 10, 6, if self.risk > 0 { 3 } else { 0 }, 12]);
        let poly = kind == 5;
        // A `def` of a name an earlier function already calls (a user
        // function, or a builtin the `def` shadows): calls bind to the
        // definition in scope where the caller is defined, so the earlier
        // callers keep the old one.
        let redefine = !self.fns.is_empty() && self.c.chance(10);
        let redefined = if redefine {
            self.c.below(self.fns.len())
        } else {
            0
        };
        let shadow_name = *self.c.pick(SHADOWABLE);
        let shadow = !redefine
            && kind == 0
            && self.c.chance(8)
            && !self.fns.iter().any(|f| f.name == shadow_name);
        let name = if redefine {
            self.fns[redefined].name.clone()
        } else if shadow {
            self.shadowed.push(shadow_name);
            shadow_name.to_string()
        } else {
            self.fresh("f")
        };
        let recursive = kind == 1 || kind == 4;
        let nparams = 1 + self.c.below(3);
        let mut params: Vec<Dim> = Vec::new();
        for _ in 0..nparams {
            params.push(self.rand_dim());
        }
        if recursive {
            params[0] = ZERO;
        }
        let ret = match kind {
            2 => None,
            3 => Some(Ty::Bool),
            5 if self.c.chance(35) => Some(Ty::Bool),
            _ => Some(Ty::Num(self.rand_dim())),
        };
        let annotate_ret = !poly && matches!(ret, Some(Ty::Num(_))) && self.c.chance(75);
        // A parameter sometimes takes the name of a module variable or
        // constant, of any dimension: inside the body the parameter wins.
        let module_names: Vec<String> = self
            .vars
            .iter()
            .filter(|v| v.module)
            .map(|v| v.name.clone())
            .collect();
        let mut pnames: Vec<String> = Vec::new();
        for _ in 0..params.len() {
            let taken = !module_names.is_empty() && self.c.chance(20);
            let n = if taken {
                self.c.pick(&module_names).clone()
            } else {
                self.fresh("p")
            };
            if pnames.contains(&n) {
                pnames.push(self.fresh("p"));
            } else {
                pnames.push(n);
            }
        }
        let mut header = Vec::new();
        for (n, d) in pnames.iter().zip(&params) {
            if poly {
                header.push(n.clone());
            } else {
                let ann = self.annotation(*d);
                header.push(format!("{n}: {ann}"));
            }
        }
        let mut def = format!("def {name}({})", header.join(", "));
        if annotate_ret {
            if let Some(Ty::Num(d)) = ret {
                let ann = self.annotation(d);
                let _ = write!(def, " -> {ann}");
            }
        }
        def.push(':');
        self.line(&def);

        let sig = FnSig {
            name,
            params: params.clone(),
            ret,
            recursive,
            poly,
            weight: 0,
        };
        let idx = if redefine {
            self.fns[redefined] = sig;
            redefined
        } else {
            self.fns.push(sig);
            self.fns.len() - 1
        };

        // Enter the function's scope: it sees the module's names.
        let saved_vars = std::mem::take(&mut self.vars);
        let saved = (
            self.in_fn,
            self.ret,
            self.loops,
            self.mult,
            self.weight,
            std::mem::take(&mut self.rec),
            self.cur.replace(idx),
        );
        let module_vars: Vec<Var> = saved_vars
            .iter()
            .filter(|v| v.module && !pnames.contains(&v.name))
            .cloned()
            .collect();
        for v in module_vars {
            let assign = v.assign && self.c.chance(30);
            self.vars.push(Var {
                assign,
                update: false,
                retype: false,
                ..v
            });
        }
        for (i, (n, d)) in pnames.iter().zip(&params).enumerate() {
            let counter = recursive && i == 0;
            self.vars.push(Var {
                name: n.clone(),
                ty: Ty::Num(*d),
                assign: !counter,
                update: !counter,
                module: false,
                retype: false,
            });
        }
        self.in_fn = true;
        self.ret = ret;
        self.loops = 0;
        self.mult = 1;
        self.weight = 0;
        self.ind = 1;
        if recursive {
            self.rec = Some(Rec {
                idx,
                n: pnames[0].clone(),
                sites: 0,
                unbounded: kind == 4,
            });
        }

        // Sometimes a local takes a module name too, with another type. The
        // right side still reads the module's value (a name is read from the
        // locals, then from the module).
        if self.c.chance(20) {
            let candidates: Vec<String> = self
                .vars
                .iter()
                .filter(|v| v.module && v.assign)
                .map(|v| v.name.clone())
                .collect();
            if !candidates.is_empty() {
                let name = self.c.pick(&candidates).clone();
                let ty = self.rand_ty();
                let e = self.ty_expr(ty, 2);
                self.line(&format!("{name} = {}", e.s));
                self.vars.retain(|v| v.name != name);
                self.bind(name, ty, false);
            }
        }

        let mut ends = Exit::Falls;
        if kind == 4 {
            // No base case: recursion runs into the depth limit.
            let Some(Ty::Num(d)) = ret else {
                unreachable!("unbounded recursion returns a number")
            };
            let call = self.call_expr(idx, 1);
            let rest = self.num(d, 1);
            self.line(&format!("return {} + {}", call.s, at(rest, 6)));
            ends = Exit::Diverges;
        } else {
            for _ in 0..self.c.below(3) {
                self.new_name(None);
            }
            if recursive {
                // Base case first, without recursion.
                let n = pnames[0].clone();
                self.line(&format!("if {n} <= 0:"));
                self.ind += 1;
                let rec = self.rec.take();
                match ret {
                    Some(ty) => {
                        let e = self.ty_expr(ty, 2);
                        self.line(&format!("return {}", e.s));
                    }
                    None => self.line("return"),
                }
                self.rec = rec;
                self.ind -= 1;
            }
            for _ in 0..1 + self.c.below(4) {
                if self.budget <= 0 || self.c.exhausted() {
                    break;
                }
                ends = self.stmt(0);
                if ends == Exit::Diverges {
                    break;
                }
            }
            if ends == Exit::Falls {
                match ret {
                    Some(ty) => {
                        let e = self.ty_expr(ty, 3);
                        self.line(&format!("return {}", e.s));
                        ends = Exit::Diverges;
                    }
                    None => {
                        if self.c.chance(30) {
                            self.line("return");
                        } else if self.out.ends_with(":\n") {
                            self.line("pass");
                        }
                    }
                }
            }
        }
        let _ = ends;

        let weight = self.weight.max(1) * if recursive { 32 } else { 1 };
        self.vars = saved_vars;
        (
            self.in_fn,
            self.ret,
            self.loops,
            self.mult,
            self.weight,
            self.rec,
            self.cur,
        ) = saved;
        self.ind = 0;
        self.fns[idx].weight = weight;
    }
}

/// The generated program for `data`.
pub fn generate(data: &[u8]) -> String {
    let mut c = Choices::new(data);
    let risk = *c.pick(&[0, 0, 2, 5, 10, 25]);
    let shadow_consts = c.chance(15);
    // GATE_SIR_TEXT_MATH is off now that the SIR parser reads `math.*`; the
    // gate stays so a regression can be steered around again.
    // The choice is drawn either way, so turning the gate off doesn't shift
    // every later choice (and the fixed seeds' programs) along the stream.
    let no_math = c.chance(40) && GATE_SIR_TEXT_MATH;
    let mut g = Gen {
        c,
        out: String::new(),
        ind: 0,
        vars: Vec::new(),
        fns: Vec::new(),
        counter: 0,
        risk,
        budget: 70,
        in_fn: false,
        ret: None,
        loops: 0,
        mult: 1,
        weight: 0,
        rec: None,
        cur: None,
        shadowed: Vec::new(),
        shadow_consts,
        no_math,
        unknown: None,
    };
    if !GATE_NON_BOOL_OPERAND && g.c.chance(50) {
        let name = g.fresh("u");
        g.line(&format!("def {name}(a):"));
        g.line("    return a");
        g.unknown = Some(name);
    }
    // A few of the predefined constants.
    for _ in 0..g.c.below(6) {
        let (name, d) = *g.c.pick(CONSTANTS);
        if g.vars.iter().any(|v| v.name == name) {
            continue;
        }
        let assign = g.shadow_consts && g.c.chance(60);
        g.vars.push(Var {
            name: name.to_string(),
            ty: Ty::Num(d),
            assign,
            update: false,
            module: true,
            retype: false,
        });
    }
    for _ in 0..2 + g.c.below(3) {
        g.new_name(None);
    }
    while g.budget > 0 && !g.c.exhausted() {
        if g.fns.len() < 4 && g.c.chance(10) {
            g.function();
        } else {
            g.stmt(0);
        }
    }
    // Show the module's final state.
    let shown: Vec<String> = g
        .vars
        .iter()
        .filter(|v| v.module && !CONSTANTS.iter().any(|(n, _)| *n == v.name))
        .take(6)
        .map(|v| v.name.clone())
        .collect();
    if !shown.is_empty() {
        g.line(&format!("print({})", shown.join(", ")));
    }
    g.out
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

/// What one program did, when the engines agreed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// It stopped with a run-time error (the same one in both engines).
    pub failed: bool,
    /// The resource-limit error (recursion depth or nesting).
    pub limit: bool,
    /// The run-time error both engines stopped with, if any.
    pub error: Option<String>,
    /// Bytes printed.
    pub printed: usize,
    /// A known, reported bug the run tolerated (see [`KNOWN`]); the rest of
    /// the check after it was skipped.
    pub known: Option<&'static str>,
}

/// Bugs this fuzzer found that are not fixed yet: an id and the text of the
/// error message that identifies each. A lenient run ([`check_source`])
/// skips the rest of the check when it hits one, so the fuzzer keeps looking
/// for new bugs; a strict run ([`check_source_strict`], used by the ignored
/// regression tests in `fuzz_differential.rs`) reports it. Delete the entry
/// when the bug is fixed.
const KNOWN: &[(&str, &str)] = &[
    // Documented in sky-lower: a number literal that overflows f64 (here
    // through its unit's scale) is refused when lowering, while `sky run`
    // reports it only if the literal is evaluated.
    (
        "lowering-literal-overflow",
        "lowering failed on a checked program: arithmetic overflow",
    ),
];

fn known_issue(msg: &str) -> Option<&'static str> {
    KNOWN
        .iter()
        .find(|(_, needle)| msg.contains(needle))
        .map(|(id, _)| *id)
}

/// The two resource-limit errors compare as one: which trips first depends
/// on each interpreter's stack use per call, not on the program.
fn normalized(r: &Result<(), String>) -> Result<(), String> {
    r.clone().map_err(|e| {
        if e.starts_with("recursion is too deep")
            || e == "the program nests too deeply for the interpreter"
        {
            "<resource limit>".to_string()
        } else {
            e
        }
    })
}

/// What the AST interpreter (`sky run`) does.
fn interpret(ast: &sky_ast::Module, src: &str) -> (String, Result<(), String>) {
    let mut env = sky_interp::RuntimeEnv::with_prelude();
    let mut out = Vec::new();
    let r = sky_interp::run_module(ast, src, &mut env, &mut out);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.map(|_| ()).map_err(|e| e.message.clone()),
    )
}

/// What the SIR reference interpreter does with `m`.
fn run_sir(m: &sky_ir::Module) -> (String, Result<(), String>) {
    let mut out = Vec::new();
    let r = sky_ir_interp::run(m, &mut out);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.map_err(|e| e.message.clone()),
    )
}

/// Runs the whole oracle on `src`. `Err` describes what went wrong: a
/// divergence between the engines, a lowering or verification failure, or a
/// generator bug (the program does not pass `sky check`; the message starts
/// with `generator bug`).
pub fn check_source(src: &str) -> Result<Outcome, String> {
    sky_ast::with_deep_stack(|| check_inner(src, false))
}

/// [`check_source`] without tolerating the bugs in [`KNOWN`].
pub fn check_source_strict(src: &str) -> Result<Outcome, String> {
    sky_ast::with_deep_stack(|| check_inner(src, true))
}

fn check_inner(src: &str, strict: bool) -> Result<Outcome, String> {
    let lexed = sky_syntax::lex(src).map_err(|e| {
        let ds: Vec<String> = e.diagnostics().map(|d| d.message.clone()).collect();
        format!("generator bug: the lexer rejects the program: {ds:?}")
    })?;
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        let ms: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
        return Err(format!("generator bug: parse errors: {ms:?}"));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        let ms: Vec<&str> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == sky_diag::Severity::Error)
            .map(|d| d.message.as_str())
            .collect();
        return Err(format!(
            "generator bug: sky check rejects the program: {ms:?}"
        ));
    }

    let m = match sky_lower::lower(&ast, src, &checked) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("lowering failed on a checked program: {e}");
            return match known_issue(&msg) {
                Some(id) if !strict => Ok(Outcome {
                    failed: true,
                    limit: false,
                    error: None,
                    printed: 0,
                    known: Some(id),
                }),
                _ => Err(msg),
            };
        }
    };
    if let Err(errors) = sky_ir::verify(&m) {
        let ms: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
        return Err(format!("the lowered SIR does not verify: {ms:?}"));
    }

    let (want_out, want) = interpret(&ast, src);
    let (got_out, got) = run_sir(&m);
    let dump = |m: &sky_ir::Module| sky_ir::print(m);
    if got_out != want_out {
        return Err(format!(
            "stdout differs\n--- sky run ---\n{want_out}(error: {want:?})\n--- SIR ---\n{got_out}(error: {got:?})\n--- SIR text ---\n{}",
            dump(&m)
        ));
    }
    if normalized(&got) != normalized(&want) {
        return Err(format!(
            "the error differs\nsky run: {want:?}\nSIR:     {got:?}\nstdout (both):\n{want_out}\n--- SIR text ---\n{}",
            dump(&m)
        ));
    }

    let mut known = None;
    let outcome = |known| Outcome {
        failed: want.is_err(),
        limit: normalized(&want) == Err("<resource limit>".to_string()),
        error: want.clone().err(),
        printed: want_out.len(),
        known,
    };

    // The text format: print, parse back, canonical, verifies, same run.
    let text = sky_ir::print(&m);
    let back = match sky_ir::parse(&text) {
        Ok(back) => back,
        Err(e) => {
            let msg = format!("the printed SIR does not parse: {e}\n{text}");
            match known_issue(&msg) {
                Some(id) if !strict => {
                    known = Some(id);
                    return Ok(outcome(known));
                }
                _ => return Err(msg),
            }
        }
    };
    if back != sky_ir::canonicalize(&m) {
        return Err(format!(
            "parse(print(m)) differs from canonicalize(m)\n--- SIR text ---\n{text}\n--- reprinted ---\n{}",
            sky_ir::print(&back)
        ));
    }
    if let Err(errors) = sky_ir::verify(&back) {
        let ms: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
        return Err(format!("the reparsed SIR does not verify: {ms:?}\n{text}"));
    }
    let (back_out, back_r) = run_sir(&back);
    if back_out != got_out || normalized(&back_r) != normalized(&got) {
        return Err(format!(
            "the reparsed SIR runs differently\nbefore: {got_out:?} {got:?}\nafter:  {back_out:?} {back_r:?}\n--- SIR text ---\n{text}"
        ));
    }

    Ok(outcome(known))
}

/// The fuzz check: generate a program from `data` and require the engines to
/// agree. Panics, with the program and both outputs, when they do not.
pub fn check(data: &[u8]) -> Outcome {
    let src = generate(data);
    match check_source(&src) {
        Ok(o) => o,
        Err(msg) => panic!("{msg}\n--- program ---\n{src}"),
    }
}
