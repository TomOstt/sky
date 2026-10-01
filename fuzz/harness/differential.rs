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
//!    Contracts are generated too (`assert` statements, and `requires` and
//!    `ensures` lines on functions), mostly conditions that hold so the
//!    program goes on, sometimes one that fails, whose message (the values of
//!    its names) and span the engines must agree on.
//!
//! 2. [`check_source`] is the oracle. It lexes, parses and checks the source
//!    (all must succeed: a rejection is a generator bug), lowers it, verifies
//!    the SIR, runs the AST interpreter and the SIR interpreter, and requires
//!    the same stdout and the same error, compared as the whole rendered
//!    diagnostic (message, spans, labels, secondary labels, notes; the SIR
//!    operations carry the spans of their operands), or both succeed. Then it
//!    prints the SIR, parses the text back, requires the result to equal
//!    `canonicalize(m)`, verifies it, and runs it again for the same result.
//!    The two resource-limit errors compare as one, exactly as in the
//!    hand-written differential tests. The native engine (`sky-jit`) is held
//!    to the same comparison, on the lowered SIR and on the SIR after the
//!    default pipeline, whenever it compiles the program (a refusal is
//!    reported in [`Outcome::native_refusal`], not a failure).
//!
//! Bugs this fuzzer found are fixed and each has a regression test in the
//! sky-ir-interp test. A bug found later that is not fixed at once is
//! reported, not repeated: the generator steers around it (a `GATE_*`
//! constant, `true` while the bug is open) and [`KNOWN`] lets a lenient run
//! tolerate it, so a failure of the target or the test is a new one. Both are
//! empty or off now.
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
    /// The array variables in scope, in a program of small arrays
    /// ([`generate_vectors`]); always empty in the other programs.
    vecs: Vec<VecVar>,
    /// The functions over arrays defined so far.
    vfns: Vec<VecFn>,
    /// The loop counters in scope, which may index arrays.
    counters: Vec<Counter>,
    /// How deep the scalars made from arrays nest.
    vec_depth: u32,
}

/// Known bugs the generator steers around, so the fuzzer keeps finding new
/// ones. Each gate is `true` while its bug is open; the `#[ignore]`d
/// regression test named in its comment (in
/// `crates/sky-ir-interp/tests/fuzz_differential.rs`) reproduces it. Flip a
/// gate to `false` when the bug is fixed.
///
/// Some reported differences the generator never produces, by construction
/// (see the tests of the same file): a conditionally defined function and a
/// nested `def` (lowering refuses them, `sky run` runs them). An input
/// declaration is generated (rarely: the run ends at it), and so is a literal
/// too large for an `f64` (a run-time error where it is evaluated). A function may
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

/// `a_unary_plus_of_an_unknown_non_number_stops_at_the_plus` (and
/// `unary_plus_of_an_unknown_non_number_is_checked_in_sir`): fixed. `+x` was
/// lowered as `x`, so where `x` is a value the checker cannot type (an
/// unannotated parameter) and is not a number, `sky run` stopped with "`+`
/// needs a number, found a Bool" and SIR went on (and failed later, or
/// never); lowering now emits `expect.number` for such an operand. With the
/// gate on, the generator writes no unary `+` inside a function with
/// unannotated parameters.
const GATE_UNARY_PLUS_UNKNOWN: bool = false;

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
        // A huge number with a prefixed unit overflows when converted to SI:
        // a run-time error where the literal is evaluated, in both engines.
        let u = self.unit_with(d, false);
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
        // In a program of small arrays, a scalar is often read from one (the
        // choice is drawn only there, so the other programs are unchanged).
        if (!self.vecs.is_empty() || !self.vfns.is_empty()) && self.c.chance(30) {
            if let Some(e) = self.vec_scalar(d) {
                return e;
            }
        }
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
                } else if self.c.chance(30)
                    && !(GATE_UNARY_PLUS_UNKNOWN && self.cur.is_some_and(|i| self.fns[i].poly))
                {
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
            // 12 an input declaration: the run ends there, so it is rare
            if self.ind == 0 && !self.in_fn { 1 } else { 0 },
            2,                            // 13 an assertion
            if loopable { 6 } else { 0 }, // 14 for
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
            12 => {
                self.input_declaration();
                Exit::Falls
            }
            13 => {
                self.assert_stmt();
                Exit::Falls
            }
            14 => self.for_stmt(depth),
            _ => {
                self.poly_probe();
                Exit::Falls
            }
        }
    }

    /// A condition for `assert`, `requires` or `ensures`. Mostly one that
    /// holds whatever the values are (`v == v`), so the program goes on;
    /// often a comparison of the constants the `contracts` pass can decide;
    /// sometimes a random Bool expression, which may fail (or fail to
    /// compute); rarely one that is false. A quarter of them go through the
    /// identity function, so the checker cannot type the condition and the
    /// engines must check it at run time.
    fn contract_cond(&mut self, depth: u32) -> String {
        let vars: Vec<(String, Ty)> = self.vars.iter().map(|v| (v.name.clone(), v.ty)).collect();
        let cond = match self.c.weighted(&[40, 16, 14 + self.risk, 4]) {
            0 if !vars.is_empty() => {
                let (n, ty) = self.c.pick(&vars).clone();
                match ty {
                    Ty::Num(_) => {
                        let form = *self.c.pick(&[
                            "{0} == {0}",
                            "{0} <= {0}",
                            "{0} >= {0}",
                            "not {0} != {0}",
                        ]);
                        form.replace("{0}", &n)
                    }
                    Ty::Bool | Ty::Str => format!("{n} == {n}"),
                }
            }
            0 | 1 => {
                let d = self.rand_dim();
                let (a, b) = (self.positive_literal(d).s, self.positive_literal(d).s);
                // The same literal twice is equal; two may or may not be ordered.
                match self.c.below(4) {
                    0 => format!("{a} == {a}"),
                    1 => format!("{a} <= {a}"),
                    2 => "true".to_string(),
                    _ => format!("{a} != {b} or {a} == {b}"),
                }
            }
            2 => self.boolean(depth).s,
            _ => "1 > 2".to_string(),
        };
        match &self.unknown {
            Some(u) if self.c.chance(25) => {
                if self.c.chance(8) {
                    // Not a Bool: the same run-time error in both engines.
                    format!("{u}(1)")
                } else {
                    format!("{u}({cond})")
                }
            }
            _ => cond,
        }
    }

    /// `, "message"` for a contract, one time in three.
    fn contract_message(&mut self) -> String {
        if self.c.chance(33) {
            let m = *self.c.pick(&[
                "\"invariant\"",
                "\"energy must be positive\"",
                "\"a longer message, with: punctuation (and parentheses)\"",
                "\"say \\\"hi\\\"\"",
            ]);
            format!(", {m}")
        } else {
            String::new()
        }
    }

    /// `assert cond`, with a message sometimes.
    fn assert_stmt(&mut self) {
        let cond = self.contract_cond(2);
        let message = self.contract_message();
        self.line(&format!("assert {cond}{message}"));
    }

    /// The `requires` and `ensures` lines of the function being generated,
    /// right after its `def` line: the parameters and the module's names are
    /// in scope, and `result` (the returned value) in an `ensures`.
    fn function_contracts(&mut self, ret: Option<Ty>) {
        for _ in 0..1 + self.c.below(3) {
            if self.c.chance(55) {
                let cond = self.contract_cond(1);
                let message = self.contract_message();
                self.line(&format!("requires {cond}{message}"));
            } else {
                let had = self.vars.len();
                if let Some(ty) = ret {
                    self.vars.push(Var {
                        name: "result".to_string(),
                        ty,
                        assign: false,
                        update: false,
                        module: false,
                        retype: false,
                    });
                }
                let cond = self.contract_cond(1);
                let message = self.contract_message();
                self.vars.truncate(had);
                self.line(&format!("ensures {cond}{message}"));
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

    /// `x: m`: a name declared without a value, an input nothing supplies. The
    /// run stops here with "`x` is declared but has no value", so what the
    /// program printed before it must be the same in both engines.
    fn input_declaration(&mut self) {
        let d = self.rand_dim();
        let name = self.fresh_name();
        let ann = self.annotation(d);
        self.line(&format!("{name}: {ann}"));
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

    /// The header of a `for` over `range`: its text after `range` and the
    /// number of iterations to budget for. Bounds are mostly literals (which
    /// the checker checks), but also names and calls the checker cannot read
    /// as constants, so the interpreter's checks of a computed bound or step
    /// (a whole number, not too large, no unit, a step that is not zero) are
    /// compared too: a fraction, a huge bound, a zero step, a value of unknown
    /// type that is a Bool, an array or has a unit.
    fn range_header(&mut self) -> (String, u64) {
        let count = |lo: i64, hi: i64, step: i64| -> u64 {
            let n = if step > 0 {
                (hi - lo + step - 1) / step
            } else {
                (lo - hi - step - 1) / -step
            };
            n.max(0) as u64
        };
        match self.c.weighted(&[6, 3, 3, 7, 4, 3, 3, 2, 3]) {
            // range(n)
            0 => {
                let n = self.small(0, 5);
                (format!("({n})"), n as u64)
            }
            // range(a, b)
            1 => {
                let a = self.small(-2, 4);
                let b = a + self.small(-1, 5);
                (format!("({a}, {b})"), count(a, b, 1))
            }
            // range(a, b, s), literals
            2 => {
                let a = self.small(-3, 7);
                let b = self.small(-3, 7);
                let s = *self.c.pick(&[1, 2, 3, -1, -2]);
                (format!("({a}, {b}, {s})"), count(a, b, s))
            }
            // Computed whole numbers.
            3 => {
                let (a, b, s) = (
                    self.small(0, 3),
                    self.small(2, 4),
                    *self.c.pick(&[1, 2, -1]),
                );
                let na = self.bound(&format!("{} - 1", a + 1));
                let nb = self.bound(&format!("{b} * 1"));
                let ns = self.bound(&format!("{} / {}", s * 2, 2));
                let (lo, hi) = if s > 0 { (a, b) } else { (b, a) };
                let text = match self.c.below(4) {
                    0 => format!("({nb})"),
                    1 => format!("({na}, {nb})"),
                    2 => format!("({na}, {nb}, {ns})"),
                    _ => format!("({lo}, {hi}, {ns})"),
                };
                let n = count(lo, hi, s).max(count(0, b, 1)).min(5);
                (text, n)
            }
            // A bound that is not a whole number.
            4 => {
                let n = self.bound_of(&["5 / 2", "1 / 3", "-0.5", "2.000001"]);
                match self.c.below(3) {
                    0 => (format!("({n})"), 0),
                    1 => (format!("(0, {n})"), 0),
                    _ => (format!("(0, 3, {n})"), 0),
                }
            }
            // A zero step, computed.
            5 => {
                let z = self.bound_of(&["3 - 3", "0 * 5", "-(2 - 2)"]);
                (format!("(0, 3, {z})"), 0)
            }
            // Too large, or huge but fine (the range is then empty).
            6 => match self.c.below(3) {
                0 => {
                    let n = self.bound("2 ^ 60");
                    (format!("({n})"), 0)
                }
                1 => {
                    let n = self.bound("10 ^ 300");
                    (format!("(0, 2, {n})"), 0)
                }
                _ => {
                    let n = self.bound("2 ^ 53");
                    (format!("({n}, {n})"), 0)
                }
            },
            // A value the checker cannot type: whatever it is fails or runs
            // at run time.
            7 => match self.unknown.clone() {
                Some(u) => {
                    let arg = *self
                        .c
                        .pick(&["3", "2.5", "1 m", "true", "[1, 2]", "-1", "2 ^ 54"]);
                    let n = self.bound(&format!("{u}({arg})"));
                    match self.c.below(2) {
                        0 => (format!("({n})"), 3),
                        _ => (format!("(0, {n})"), 3),
                    }
                }
                None => ("(2)".to_string(), 2),
            },
            // Computed, negative direction and empty.
            _ => {
                let lo = self.small(0, 4);
                let a = self.bound(&format!("{} + 1", lo));
                let s = self.bound("-1 * 1");
                (format!("({a}, 0, {s})"), 5)
            }
        }
    }

    /// A name bound just before a loop to `value`, which the checker does
    /// not read as a constant.
    fn bound(&mut self, value: &str) -> String {
        let n = self.fresh("n");
        self.line(&format!("{n} = {value}"));
        n
    }

    /// [`Self::bound`] of one of `values`.
    fn bound_of(&mut self, values: &[&str]) -> String {
        let value = *self.c.pick(values);
        self.bound(value)
    }

    /// A small integer: `lo` and up to `span - 1` more.
    fn small(&mut self, lo: i64, span: usize) -> i64 {
        lo + self.c.below(span) as i64
    }

    fn for_stmt(&mut self, depth: usize) -> Exit {
        let (header, iterations) = self.range_header();
        let i = self.fresh("i");
        self.line(&format!("for {i} in range{header}:"));
        let mult = self.mult;
        self.mult *= iterations.max(1);
        self.loops += 1;
        let mark = self.vars.len();
        self.vars.push(Var {
            name: i,
            ty: Ty::Num(ZERO),
            assign: false,
            update: false,
            module: !self.in_fn && self.ind == 0,
            retype: false,
        });
        self.block(depth + 1);
        self.vars.truncate(mark);
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
            // A function never assigns a module name it can read: a name the
            // body binds is its local throughout, so reading the module's
            // value first is a check error. (The draw is kept so programs
            // stay what they were otherwise.)
            let _ = self.c.chance(30);
            self.vars.push(Var {
                assign: false,
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
        // Sometimes a local takes a module name too, with another type. The
        // body binds it, so it is a local of the whole body: nothing in the
        // body may read the module's value, not even the contracts or the
        // right side of that first binding.
        let shadow = if self.c.chance(20) {
            let candidates: Vec<String> = self
                .vars
                .iter()
                .filter(|v| v.module)
                .map(|v| v.name.clone())
                .collect();
            (!candidates.is_empty()).then(|| {
                let name = self.c.pick(&candidates).clone();
                self.vars.retain(|v| v.name != name);
                name
            })
        } else {
            None
        };
        // A contract cannot call the function it belongs to (`rec` is not
        // set yet, and `cur` keeps the others from it).
        if self.c.chance(15) {
            self.function_contracts(ret);
        }
        if recursive {
            self.rec = Some(Rec {
                idx,
                n: pnames[0].clone(),
                sites: 0,
                unbounded: kind == 4,
            });
        }

        if let Some(name) = shadow {
            let ty = self.rand_ty();
            let e = self.ty_expr(ty, 2);
            self.line(&format!("{name} = {}", e.s));
            self.bind(name, ty, false);
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
    // About one input in five makes a program of arrays and the
    // entry-by-entry operations on them instead (`generate_arrays`); the
    // others are read as before, so their programs do not change.
    if data.first().is_some_and(|b| b % 5 == 0) {
        return generate_arrays(data);
    }
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
        vecs: Vec::new(),
        vfns: Vec::new(),
        counters: Vec::new(),
        vec_depth: 0,
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
// Programs of small vectors and matrices
// ---------------------------------------------------------------------------
//
// The native engine holds a vector or matrix of known shape and at most 16
// entries as separate numbers (`sky_jit`, "Small arrays"), so it must agree
// with the interpreters on every operation of such arrays, the way values
// cross `if`, `while` and calls, and every way they fail. [`generate_vectors`]
// writes programs for that: declarations of vectors and matrices of a few
// shapes (mostly of 2 to 5 entries, sometimes 8 to 17, on both sides of the
// limit) and dimensions; assignments of well-typed array expressions
// (`+ - *` by a scalar, `/`, negation, `@`, transpose, a row, `if`
// expressions, calls); scalars made from arrays (`dot`, `norm`, `@`, `len`,
// constant and run-time indexes, some out of range); `for` and `while` loops
// and `if` statements that carry arrays; typed functions with array
// parameters and results; functions without annotations, whose values the
// checker cannot type; contracts over arrays; and the scalar generator's
// expressions around them. Values are chosen to overflow at some entry, and
// scales to be zero, when the program is risky.

/// A vector or matrix variable of a generated program.
#[derive(Clone, Debug)]
struct VecVar {
    name: String,
    d: Dim,
    /// The rows of a matrix, the length of a vector.
    r: usize,
    /// The columns of a matrix; 0 for a vector.
    c: usize,
}

/// The type of a parameter or result of a generated array function.
#[derive(Clone, Copy, Debug)]
enum VTy {
    Scalar(Dim),
    Arr(Dim, usize, usize),
}

/// A generated function over arrays.
#[derive(Clone, Debug)]
struct VecFn {
    name: String,
    params: Vec<VTy>,
    ret: VTy,
}

/// An integer that may index: a counter, and how many iterations it runs.
#[derive(Clone, Debug)]
struct Counter {
    name: String,
    bound: usize,
}

/// Parenthesizes unless `e` is an atom or a postfix form.
fn par(e: E) -> String {
    at(e, 9)
}

impl Gen<'_> {
    // ----- shapes and types -----

    /// A shape `(rows or length, columns or 0)`: mostly small.
    fn vec_shape(&mut self) -> (usize, usize) {
        match self.c.weighted(&[16, 22, 30, 8, 6, 5, 3, 3, 3, 4]) {
            0 => (2, 0),
            1 => (3, 0),
            2 => (3 + usize::from(self.c.chance(30)), 0),
            3 => (1 + self.c.below(5), 0),
            4 => (*self.c.pick(&[2, 3, 4]), *self.c.pick(&[1, 2, 3, 4])),
            5 => (3, 3),
            6 => (*self.c.pick(&[7, 8, 9, 12]), 0),
            7 => (*self.c.pick(&[15, 16, 17]), 0),
            8 => (*self.c.pick(&[1, 2, 4, 5]), *self.c.pick(&[4, 5, 6, 8])),
            _ => (4, 4),
        }
    }

    /// A dimension for arrays: often dimensionless, often a length or a time.
    fn vec_dim(&mut self) -> Dim {
        match self.c.weighted(&[40, 20, 12, 10, 18]) {
            0 => ZERO,
            1 => dim(1, 0, 0, 0, 0),
            2 => dim(0, 1, 0, 0, 0),
            3 => dim(1, -1, 0, 0, 0),
            _ => self.rand_dim(),
        }
    }

    /// `Vector[3] m`, `Matrix[2,3]`: the annotation of an array type.
    fn vec_type_text(&mut self, d: Dim, r: usize, c: usize) -> String {
        let shape = if c == 0 {
            format!("Vector[{r}]")
        } else {
            format!("Matrix[{r}, {c}]")
        };
        if d.is_zero() {
            shape
        } else {
            let u = self.unit(d);
            format!("{shape} {u}")
        }
    }

    fn vty_text(&mut self, t: VTy) -> String {
        match t {
            VTy::Scalar(d) => self.annotation(d),
            VTy::Arr(d, r, c) => self.vec_type_text(d, r, c),
        }
    }

    // ----- literals and leaves -----

    fn vec_entry(&mut self, d: Dim) -> String {
        // A zero is likelier in a vector than in a scalar: it exercises the
        // sign of zero and division.
        if self.c.chance(8) {
            return self.literal_with(d, "0".to_string()).s;
        }
        self.literal(d).s
    }

    fn vec_literal(&mut self, d: Dim, r: usize, c: usize) -> E {
        if c == 0 {
            let items: Vec<String> = (0..r).map(|_| self.vec_entry(d)).collect();
            return atom(format!("[{}]", items.join(", ")));
        }
        let rows: Vec<String> = (0..r)
            .map(|_| {
                let items: Vec<String> = (0..c).map(|_| self.vec_entry(d)).collect();
                format!("[{}]", items.join(", "))
            })
            .collect();
        atom(format!("[{}]", rows.join(", ")))
    }

    /// An array variable of this shape and dimension, or a literal.
    fn vec_leaf(&mut self, d: Dim, r: usize, c: usize) -> E {
        let names: Vec<String> = self
            .vecs
            .iter()
            .filter(|v| v.d == d && v.r == r && v.c == c)
            .map(|v| v.name.clone())
            .collect();
        if !names.is_empty() && self.c.chance(75) {
            return atom(self.c.pick(&names).clone());
        }
        self.vec_literal(d, r, c)
    }

    /// A valid position on an axis of length `n`: a constant, or a counter
    /// that (mostly) stays within it.
    fn vec_index(&mut self, n: usize) -> String {
        let counters: Vec<Counter> = self
            .counters
            .iter()
            .filter(|k| k.bound <= n || self.risk >= 10)
            .cloned()
            .collect();
        if !counters.is_empty() && self.c.chance(45) {
            return self.c.pick(&counters).name.clone();
        }
        // A counter can be shifted, which stays in range for the smaller ones.
        if !self.counters.is_empty() && self.c.chance(8) {
            let k = self.c.pick(&self.counters.clone()).clone();
            return format!("{} + 1", k.name);
        }
        self.c.below(n).to_string()
    }

    // ----- array expressions -----

    fn vec_expr(&mut self, d: Dim, r: usize, c: usize, depth: u32) -> E {
        if depth == 0 || self.c.exhausted() || self.c.chance(18) {
            return self.vec_leaf(d, r, c);
        }
        let matrix = c != 0;
        let w = [
            22,                                       // 0 a + b, a - b
            8,                                        // 1 -a
            12,                                       // 2 a * s
            7,                                        // 3 s * a
            6,                                        // 4 a / s
            if r * c.max(1) <= 16 { 12 } else { 0 },  // 5 a product
            if matrix { 6 } else { 0 },               // 6 transpose
            if !matrix { 6 } else { 0 },              // 7 a row of a matrix
            4,                                        // 8 an `if` expression
            if self.vfns.is_empty() { 0 } else { 8 }, // 9 a call
            3,                                        // 10 `+a`-like identity: a - a * 0
        ];
        match self.c.weighted(&w) {
            0 => {
                let plus = self.c.chance(60);
                let a = self.vec_expr(d, r, c, depth - 1);
                let mut b = self.vec_expr(d, r, c, depth - 1);
                if b.s == a.s && self.c.chance(85) {
                    b = self.vec_literal(d, r, c);
                }
                E {
                    s: format!("{} {} {}", at(a, 5), if plus { "+" } else { "-" }, at(b, 6)),
                    p: 5,
                }
            }
            1 => {
                let mut a = self.vec_expr(d, r, c, depth - 1);
                // Known bug, reported and not fixed here: the pipeline folds
                // `-(-a)` of an array to `a`, which loses the sign of a zero
                // (a negation is `0 - x`), so no double negation is written.
                if a.s.starts_with('-') {
                    a = self.vec_leaf(d, r, c);
                }
                E {
                    s: format!("-{}", at(a, 7)),
                    p: 7,
                }
            }
            2 | 3 => {
                let d1 = self.vec_dim();
                let inner = d.sub(d1);
                if !inner.ok() {
                    return self.vec_leaf(d, r, c);
                }
                let a = self.vec_expr(inner, r, c, depth - 1);
                let s = self.num(d1, depth.saturating_sub(2));
                let (a, s) = (par(a), par(s));
                E {
                    s: if self.c.chance(65) {
                        format!("{a} * {s}")
                    } else {
                        format!("{s} * {a}")
                    },
                    p: 6,
                }
            }
            4 => {
                let d1 = self.vec_dim();
                let inner = d.add(d1);
                if !inner.ok() {
                    return self.vec_leaf(d, r, c);
                }
                let a = self.vec_expr(inner, r, c, depth - 1);
                // A zero divisor is a run-time error; only a risky program
                // divides by anything but a positive literal.
                let s = if self.c.chance(self.risk + 4) {
                    self.num(d1, 1)
                } else {
                    self.positive_literal(d1)
                };
                E {
                    s: format!("{} / {}", par(a), par(s)),
                    p: 6,
                }
            }
            5 => self.vec_product(d, r, c, depth),
            6 => {
                // The transpose of a matrix of the opposite shape.
                let a = self.vec_expr(d, c, r, depth - 1);
                if self.c.chance(50) {
                    atom(format!("transpose({})", a.s))
                } else {
                    atom(format!("{}ᵀ", par(a)))
                }
            }
            7 => {
                // A row of a matrix: `M[i]`.
                let rows = 1 + self.c.below(4);
                let m = self.vec_expr(d, rows, r, depth - 1);
                let i = self.vec_index(rows);
                atom(format!("{}[{i}]", par(m)))
            }
            8 => {
                let a = self.vec_expr(d, r, c, depth - 1);
                let cnd = self.boolean(1);
                let b = self.vec_expr(d, r, c, depth - 1);
                E {
                    s: format!("{} if {} else {}", at(a, 1), at(cnd, 1), at(b, 0)),
                    p: 0,
                }
            }
            9 => {
                let cands: Vec<usize> = (0..self.vfns.len())
                    .filter(|&i| matches!(self.vfns[i].ret, VTy::Arr(fd, fr, fc) if fd == d && fr == r && fc == c))
                    .collect();
                if cands.is_empty() {
                    return self.vec_leaf(d, r, c);
                }
                let i = *self.c.pick(&cands);
                self.vec_call(i, depth)
            }
            _ => {
                // `a - a * 0`-like forms keep every entry finite and give
                // signed zeros.
                let a = self.vec_expr(d, r, c, depth - 1);
                E {
                    s: format!("{} * 0", par(a)),
                    p: 6,
                }
            }
        }
    }

    /// A product `x @ y` of shape `(r, c)` and dimension `d`.
    fn vec_product(&mut self, d: Dim, r: usize, c: usize, depth: u32) -> E {
        let d1 = self.vec_dim();
        let d2 = d.sub(d1);
        if !d2.ok() {
            return self.vec_leaf(d, r, c);
        }
        let k = 1 + self.c.below(4);
        let (x, y) = if c == 0 {
            if self.c.chance(50) {
                // matrix @ vector
                (
                    self.vec_expr(d1, r, k, depth - 1),
                    self.vec_expr(d2, k, 0, depth - 1),
                )
            } else {
                // vector @ matrix
                (
                    self.vec_expr(d1, k, 0, depth - 1),
                    self.vec_expr(d2, k, r, depth - 1),
                )
            }
        } else {
            (
                self.vec_expr(d1, r, k, depth - 1),
                self.vec_expr(d2, k, c, depth - 1),
            )
        };
        E {
            s: format!("{} @ {}", at(x, 6), at(y, 7)),
            p: 6,
        }
    }

    fn vec_call(&mut self, i: usize, depth: u32) -> E {
        let f = self.vfns[i].clone();
        let mut args = Vec::new();
        for p in &f.params {
            args.push(match *p {
                VTy::Scalar(d) => self.num(d, depth.saturating_sub(2)).s,
                VTy::Arr(d, r, c) => self.vec_expr(d, r, c, depth.saturating_sub(2)).s,
            });
        }
        atom(format!("{}({})", f.name, args.join(", ")))
    }

    // ----- scalars made from arrays -----

    /// A scalar of dimension `d` that reads an array, for the scalar
    /// generator's leaves. `None` when there is no array to read.
    fn vec_scalar(&mut self, d: Dim) -> Option<E> {
        if self.vecs.is_empty() && self.vfns.is_empty() {
            return None;
        }
        if self.vec_depth >= 2 {
            return None;
        }
        self.vec_depth += 1;
        let e = self.vec_scalar_inner(d);
        self.vec_depth -= 1;
        Some(e)
    }

    fn vec_scalar_inner(&mut self, d: Dim) -> E {
        let (r, c) = self.vec_shape();
        let depth = 1;
        match self
            .c
            .weighted(&[26, 20, 24, 12, if d.is_zero() { 6 } else { 0 }, 12])
        {
            0 => {
                // dot(a, b) or a @ b
                let d1 = self.vec_dim();
                let d2 = d.sub(d1);
                if !d2.ok() {
                    return self.literal(d);
                }
                let n = if c == 0 { r } else { c };
                let (a, b) = (
                    self.vec_expr(d1, n, 0, depth),
                    self.vec_expr(d2, n, 0, depth),
                );
                if self.c.chance(50) {
                    atom(format!("dot({}, {})", a.s, b.s))
                } else {
                    E {
                        s: format!("{} @ {}", at(a, 6), at(b, 7)),
                        p: 6,
                    }
                }
            }
            1 => {
                let a = self.vec_expr(d, r, c, depth);
                atom(format!("norm({})", a.s))
            }
            2 => {
                // v[i]
                let a = self.vec_expr(d, r, 0, depth);
                let i = self.vec_index(r);
                atom(format!("{}[{i}]", par(a)))
            }
            3 => {
                // M[i, j]
                let (r, c) = (r.min(4), if c == 0 { 1 + self.c.below(4) } else { c });
                let a = self.vec_expr(d, r, c, depth);
                let (i, j) = (self.vec_index(r), self.vec_index(c));
                atom(format!("{}[{i}, {j}]", par(a)))
            }
            4 => {
                let a = self.vec_expr(ZERO, r, c, 0);
                atom(format!("len({})", a.s))
            }
            _ => {
                // An entry of a product.
                let a = self.vec_expr(d, 2, 0, depth);
                let i = self.vec_index(2);
                atom(format!("{}[{i}]", par(a)))
            }
        }
    }

    // ----- statements -----

    /// A new array variable, bound where it is written (never inside a loop
    /// or an `if`: lowering does not take an array first bound there).
    fn vec_decl(&mut self) {
        let (r, c) = self.vec_shape();
        let d = self.vec_dim();
        let name = self.fresh_name();
        let e = if self.c.chance(70) || self.vecs.is_empty() {
            self.vec_literal(d, r, c)
        } else {
            self.vec_expr(d, r, c, 3)
        };
        self.line(&format!("{name} = {}", e.s));
        self.vecs.push(VecVar { name, d, r, c });
    }

    fn vec_assign(&mut self) {
        if self.vecs.is_empty() {
            return self.print();
        }
        let v = self.c.pick(&self.vecs.clone()).clone();
        match self.c.weighted(&[60, 14, 8, 6, 6, 6]) {
            0 => {
                let e = self.vec_expr(v.d, v.r, v.c, 3);
                self.line(&format!("{} = {}", v.name, e.s));
            }
            1 => {
                let e = self.vec_expr(v.d, v.r, v.c, 2);
                self.line(&format!("{} += {}", v.name, e.s));
            }
            2 => {
                let e = self.vec_expr(v.d, v.r, v.c, 2);
                self.line(&format!("{} -= {}", v.name, e.s));
            }
            3 => {
                let s = self.num(ZERO, 1);
                self.line(&format!("{} *= {}", v.name, s.s));
            }
            4 => {
                let s = if self.c.chance(self.risk + 4) {
                    self.num(ZERO, 1)
                } else {
                    self.positive_literal(ZERO)
                };
                self.line(&format!("{} /= {}", v.name, s.s));
            }
            _ => {
                // The same shape from a function of an array of another.
                let e = self.vec_expr(v.d, v.r, v.c, 3);
                self.line(&format!("{} = {} + {}", v.name, v.name, par(e)));
            }
        }
    }

    fn vec_print(&mut self) {
        let n = 1 + self.c.below(3);
        let mut args = Vec::new();
        for _ in 0..n {
            if self.c.chance(60) && !self.vecs.is_empty() {
                let v = self.c.pick(&self.vecs.clone()).clone();
                args.push(self.vec_expr(v.d, v.r, v.c, 2).s);
            } else {
                let d = self.vec_dim();
                args.push(self.num(d, 2).s);
            }
        }
        self.line(&format!("print({})", args.join(", ")));
    }

    fn vec_assert(&mut self) {
        let d = self.vec_dim();
        let (r, c) = self.vec_shape();
        let a = self.vec_expr(d, r, c, 2);
        let bound = self.positive_literal(d);
        // `norm(a) >= 0` always holds; the other two may not.
        let cond = match self.c.weighted(&[50, 25, 25]) {
            0 => {
                let zero = self.literal_with(d, "0".to_string());
                format!("norm({}) >= {}", a.s, zero.s)
            }
            1 => format!("norm({}) < {}", a.s, bound.s),
            _ => format!("norm({}) >= {}", a.s, bound.s),
        };
        let msg = if self.c.chance(40) {
            ", \"vectors\""
        } else {
            ""
        };
        self.line(&format!("assert {cond}{msg}"));
    }

    /// `for i in range(n):` over statements that read `i` as an index and
    /// assign arrays.
    fn vec_for(&mut self, depth: usize) {
        let n = 1 + self.c.below(5);
        let name = *self.c.pick(&["i", "j", "k", "n"]);
        let name = self.fresh(name);
        self.line(&format!("for {name} in range({n}):"));
        self.counters.push(Counter {
            name: name.clone(),
            bound: n,
        });
        self.vec_block(depth + 1);
        self.counters.pop();
    }

    fn vec_while(&mut self, depth: usize) {
        let n = 1 + self.c.below(4);
        let name = self.fresh("t");
        self.line(&format!("{name} = 0"));
        self.line(&format!("while {name} < {n}:"));
        self.counters.push(Counter {
            name: name.clone(),
            bound: n,
        });
        self.vec_block_with(depth + 1, &format!("{name} += 1"));
        self.counters.pop();
    }

    fn vec_if(&mut self, depth: usize) {
        let cnd = self.boolean(1);
        self.line(&format!("if {}:", cnd.s));
        self.vec_block(depth + 1);
        if self.c.chance(45) {
            self.line("else:");
            self.vec_block(depth + 1);
        }
    }

    fn vec_block(&mut self, depth: usize) {
        self.vec_block_with(depth, "");
    }

    /// An indented block of array statements, then `tail` if not empty.
    fn vec_block_with(&mut self, depth: usize, tail: &str) {
        let mark = self.vars.len();
        self.ind += 1;
        for _ in 0..1 + self.c.below(3) {
            if self.budget <= 0 || self.c.exhausted() {
                break;
            }
            self.vec_stmt(depth);
        }
        if !tail.is_empty() {
            self.line(tail);
        } else if self.out.ends_with(":\n") {
            self.line("pass");
        }
        self.ind -= 1;
        self.vars.truncate(mark);
    }

    fn vec_stmt(&mut self, depth: usize) {
        self.budget -= 1;
        let nested = depth < 3 && self.budget > 3;
        let top = self.ind == 0 || (self.in_fn && self.ind == 1);
        let w = [
            10, // 0 declaration (top only)
            28, // 1 assignment
            14, // 2 print
            if nested && self.counters.len() < 2 {
                9
            } else {
                0
            }, // 3 for
            if nested && self.counters.len() < 2 {
                6
            } else {
                0
            }, // 4 while
            if nested { 9 } else { 0 }, // 5 if
            8,  // 6 a scalar statement
            3,  // 7 assert
            if top && !self.in_fn && self.vfns.len() < 3 {
                7
            } else {
                0
            }, // 8 function
            if top && !self.in_fn && !self.vfns.is_empty() {
                6
            } else {
                0
            }, // 9 typed call
            if top && !self.in_fn { 3 } else { 0 }, // 10 an untyped function
        ];
        match self.c.weighted(&w) {
            0 if top => self.vec_decl(),
            0 | 1 => self.vec_assign(),
            2 => self.vec_print(),
            3 => self.vec_for(depth),
            4 => self.vec_while(depth),
            5 => self.vec_if(depth),
            6 => {
                // A scalar name from arrays, or the scalar generator's own.
                if top || self.c.chance(50) {
                    let d = self.vec_dim();
                    let name = self.fresh_name();
                    let e = self.num(d, 3);
                    self.line(&format!("{name} = {}", e.s));
                    self.bind(name, Ty::Num(d), true);
                } else {
                    self.assign();
                }
            }
            7 => self.vec_assert(),
            8 => self.vec_function(),
            9 => {
                let i = self.c.below(self.vfns.len());
                let call = self.vec_call(i, 2);
                let f = self.vfns[i].clone();
                let name = self.fresh_name();
                let ann = self.vty_text(f.ret);
                self.line(&format!("{name}: {ann} = {}", call.s));
                match f.ret {
                    VTy::Scalar(d) => self.bind(name, Ty::Num(d), false),
                    VTy::Arr(d, r, c) => self.vecs.push(VecVar { name, d, r, c }),
                }
            }
            _ => self.vec_untyped_function(),
        }
    }

    // ----- functions -----

    /// A function with array parameters and result, all annotated: its body
    /// is a few array statements and a `return` of an expression over the
    /// parameters.
    fn vec_function(&mut self) {
        let name = self.fresh("vf");
        let nparams = 1 + self.c.below(3);
        let mut params = Vec::new();
        for _ in 0..nparams {
            params.push(if self.c.chance(22) {
                VTy::Scalar(self.vec_dim())
            } else {
                let (r, c) = self.vec_shape();
                VTy::Arr(self.vec_dim(), r, c)
            });
        }
        let ret = if self.c.chance(15) {
            VTy::Scalar(self.vec_dim())
        } else {
            let (r, c) = self.vec_shape();
            VTy::Arr(self.vec_dim(), r, c)
        };
        let pnames: Vec<String> = (0..nparams).map(|_| self.fresh("p")).collect();
        let mut header = Vec::new();
        for (n, t) in pnames.iter().zip(&params) {
            let ann = self.vty_text(*t);
            header.push(format!("{n}: {ann}"));
        }
        let ret_ann = self.vty_text(ret);
        self.line(&format!("def {name}({}) -> {ret_ann}:", header.join(", ")));

        // Enter the function: only its parameters (and scalars) are in scope.
        let saved_vars = std::mem::take(&mut self.vars);
        let saved_vecs = std::mem::take(&mut self.vecs);
        let saved_counters = std::mem::take(&mut self.counters);
        let saved_fns = std::mem::take(&mut self.fns);
        let saved_in_fn = std::mem::replace(&mut self.in_fn, true);
        let saved_ret = self.ret.take();
        self.ind = 1;
        for (n, t) in pnames.iter().zip(&params) {
            match *t {
                VTy::Scalar(d) => self.vars.push(Var {
                    name: n.clone(),
                    ty: Ty::Num(d),
                    assign: false,
                    update: false,
                    module: false,
                    retype: false,
                }),
                VTy::Arr(d, r, c) => self.vecs.push(VecVar {
                    name: n.clone(),
                    d,
                    r,
                    c,
                }),
            }
        }
        // Locals, bound at the top of the body.
        for _ in 0..self.c.below(3) {
            self.vec_decl();
        }
        if self.c.chance(30) {
            self.vec_for(1);
        }
        let ret_expr = match ret {
            VTy::Scalar(d) => self.num(d, 3),
            VTy::Arr(d, r, c) => self.vec_expr(d, r, c, 3),
        };
        self.line(&format!("return {}", ret_expr.s));
        self.ind = 0;
        self.vars = saved_vars;
        self.vecs = saved_vecs;
        self.counters = saved_counters;
        self.fns = saved_fns;
        self.in_fn = saved_in_fn;
        self.ret = saved_ret;
        self.vfns.push(VecFn { name, params, ret });
    }

    /// `def f(x): return x * 2 + x`-like functions with no annotations: their
    /// parameters and results are `unknown` to the checker, so the values are
    /// boxed on the way in and checked on the way out. Called with an array
    /// of the shape it was written for, into a typed declaration.
    fn vec_untyped_function(&mut self) {
        let name = self.fresh("uf");
        let (r, c) = self.vec_shape();
        let d = self.vec_dim();
        let k = self.num(ZERO, 1);
        let body = match self.c.below(4) {
            0 => format!("x * {}", par(k)),
            1 => "x + x".to_string(),
            2 => format!("-x * {} + x", par(k)),
            _ => "x - x * 0.5".to_string(),
        };
        self.line(&format!("def {name}(x):"));
        self.line(&format!("    return {body}"));
        let arg = self.vec_expr(d, r, c, 2);
        let out = self.fresh_name();
        let ann = self.vec_type_text(d, r, c);
        self.line(&format!("{out}: {ann} = {name}({})", arg.s));
        self.vecs.push(VecVar { name: out, d, r, c });
    }
}

/// A program of small arrays for `data` (see the notes above).
pub fn generate_vectors(data: &[u8]) -> String {
    let mut c = Choices::new(data);
    let risk = *c.pick(&[0, 0, 2, 5, 10, 25]);
    let mut g = Gen {
        c,
        out: String::new(),
        ind: 0,
        vars: Vec::new(),
        fns: Vec::new(),
        counter: 0,
        risk,
        budget: 40,
        in_fn: false,
        ret: None,
        loops: 0,
        mult: 1,
        weight: 0,
        rec: None,
        cur: None,
        shadowed: Vec::new(),
        shadow_consts: false,
        no_math: false,
        unknown: None,
        vecs: Vec::new(),
        vfns: Vec::new(),
        counters: Vec::new(),
        vec_depth: 0,
    };
    for _ in 0..1 + g.c.below(3) {
        g.new_name(Some(Ty::Num(ZERO)));
    }
    for _ in 0..2 + g.c.below(3) {
        g.vec_decl();
    }
    while g.budget > 0 && !g.c.exhausted() {
        g.vec_stmt(0);
    }
    let shown: Vec<String> = g.vecs.iter().take(6).map(|v| v.name.clone()).collect();
    if !shown.is_empty() {
        g.line(&format!("print({})", shown.join(", ")));
    }
    g.out
}

// ---------------------------------------------------------------------------
// The array generator
// ---------------------------------------------------------------------------
//
// Programs made of arrays and the entry-by-entry operations on them: `+`,
// `-`, `*` and `/` of arrays of one shape, scaling by numbers, unary minus,
// the math functions of one array (or of an array and a number), and the
// reductions back to numbers (`norm`, `dot`, `@`, indexing). Shapes are a few
// fixed ones, dimensions are tracked (a small palette of `m` and `s`
// exponents, so `*`, `/`, `sqrt` and `cbrt` move them around), and every
// program type-checks by construction. Some programs are risky on purpose:
// zeros, negatives, and entries near the ends of the `f64` range make
// overflow, division by zero and math domain errors happen at some entries.
// Bindings are reused (so the optimizer fuses the single-use trees and
// leaves the shared ones), a `for` loop feeds an array back into itself, and
// a function or two of arrays are defined and called. About one program in
// four has a vector of several hundred entries, long enough for the native
// kernels.

/// Exponents of `m` and `s`.
type AD = (i8, i8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AShape {
    V3,
    V5,
    M23,
    /// A vector of `Ar::long` entries.
    Long,
}

impl AShape {
    fn is_vector(self) -> bool {
        self != AShape::M23
    }
}

#[derive(Clone, Debug)]
struct AVar {
    name: String,
    shape: AShape,
    ad: AD,
}

#[derive(Clone, Debug)]
struct SVar {
    name: String,
    ad: AD,
}

#[derive(Clone, Debug)]
struct AFn {
    name: String,
    params: Vec<(AShape, AD)>,
    ret: (AShape, AD),
}

struct Ar<'a> {
    c: Choices<'a>,
    out: String,
    ind: usize,
    vars: Vec<AVar>,
    svars: Vec<SVar>,
    fns: Vec<AFn>,
    /// Percent of numbers that are risky (huge, tiny, zero).
    risk: usize,
    counter: usize,
    /// The length of `AShape::Long`, or 0 when the program has none.
    long: usize,
    /// Inside a function body: the module's bindings are not visible.
    in_fn: bool,
}

/// Dimensions that have a plain spelling in an annotation.
const NICE: &[AD] = &[(0, 0), (1, 0), (2, 0), (0, 1), (1, -1), (0, 2), (1, 1)];
/// Every dimension an expression may have.
const EXP_LIMIT: i8 = 6;

fn ad_add(a: AD, b: AD) -> AD {
    (a.0 + b.0, a.1 + b.1)
}

fn ad_sub(a: AD, b: AD) -> AD {
    (a.0 - b.0, a.1 - b.1)
}

fn ad_in_range(a: AD) -> bool {
    a.0.abs() <= EXP_LIMIT && a.1.abs() <= EXP_LIMIT
}

fn ad_is_zero(a: AD) -> bool {
    a == (0, 0)
}

/// The unit as a literal writes it: `m^2*s^-1`.
fn ad_unit(a: AD) -> String {
    let part = |name: &str, e: i8| match e {
        0 => None,
        1 => Some(name.to_string()),
        e => Some(format!("{name}^{e}")),
    };
    [part("m", a.0), part("s", a.1)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("*")
}

/// The unit as an annotation writes it (only for the `NICE` dimensions).
fn ad_annotation(a: AD) -> String {
    match a {
        (0, 0) => String::new(),
        (1, 0) => " m".into(),
        (2, 0) => " m^2".into(),
        (0, 1) => " s".into(),
        (1, -1) => " m/s".into(),
        (0, 2) => " s^2".into(),
        (1, 1) => " m*s".into(),
        _ => unreachable!("not a nice dimension"),
    }
}

impl Ar<'_> {
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

    fn shape_text(&self, s: AShape) -> String {
        match s {
            AShape::V3 => "Vector[3]".into(),
            AShape::V5 => "Vector[5]".into(),
            AShape::M23 => "Matrix[2, 3]".into(),
            AShape::Long => format!("Vector[{}]", self.long),
        }
    }

    fn any_shape(&mut self) -> AShape {
        if self.long > 0 && self.c.chance(35) {
            return AShape::Long;
        }
        *self.c.pick(&[AShape::V3, AShape::V5, AShape::M23])
    }

    fn any_ad(&mut self) -> AD {
        NICE[self.c.weighted(&[40, 25, 10, 10, 8, 4, 3])]
    }

    // ----- numbers -----

    /// A number with the unit of `ad`, as source text (not parenthesized).
    fn number(&mut self, ad: AD) -> String {
        let n = match self.c.weighted(&[40, 8, 20, 8, 4]) {
            0 => {
                if self.c.chance(self.risk / 2 + 2) {
                    "0".to_string()
                } else {
                    (1 + self.c.below(9)).to_string()
                }
            }
            1 => self.c.pick(&["10", "12", "100", "255", "1000"]).to_string(),
            2 => self
                .c
                .pick(&[
                    "0.5", "2.5", "0.25", "1.5", "3.14", "0.1", "9.81", "0.75", "2.0", "0.3", "1.0",
                ])
                .to_string(),
            3 => self
                .c
                .pick(&["1e3", "2.5e-2", "1e-3", "1e15", "1e-5"])
                .to_string(),
            _ => {
                if self.c.chance(self.risk) {
                    self.c
                        .pick(&[
                            "1e150", "1e200", "1e300", "1e-300", "1e-200", "1e100", "1e154",
                        ])
                        .to_string()
                } else {
                    (1 + self.c.below(5)).to_string()
                }
            }
        };
        let n = if self.c.chance(if self.risk == 0 { 10 } else { 30 }) {
            format!("-{n}")
        } else {
            n
        };
        if ad_is_zero(ad) {
            n
        } else {
            format!("{n} {}", ad_unit(ad))
        }
    }

    /// An array literal of shape `s` with the unit of `ad`.
    fn literal(&mut self, s: AShape, ad: AD) -> String {
        match s {
            AShape::V3 | AShape::V5 => {
                let n = if s == AShape::V3 { 3 } else { 5 };
                let es: Vec<String> = (0..n).map(|_| self.number(ad)).collect();
                format!("[{}]", es.join(", "))
            }
            AShape::M23 => {
                let rows: Vec<String> = (0..2)
                    .map(|_| {
                        let es: Vec<String> = (0..3).map(|_| self.number(ad)).collect();
                        format!("[{}]", es.join(", "))
                    })
                    .collect();
                format!("[{}]", rows.join(", "))
            }
            AShape::Long => {
                // A few numbers, cycled: a long literal without a long stream.
                let base: Vec<String> = (0..7).map(|_| self.number(ad)).collect();
                let es: Vec<&str> = (0..self.long).map(|i| base[i * 5 % 7].as_str()).collect();
                format!("[{}]", es.join(", "))
            }
        }
    }

    /// A number expression of dimension `ad`.
    fn scalar(&mut self, ad: AD, depth: u32) -> String {
        let pool: Vec<String> = self
            .svars
            .iter()
            .filter(|v| v.ad == ad)
            .map(|v| v.name.clone())
            .collect();
        let pool = if self.in_fn { Vec::new() } else { pool };
        match self.c.weighted(&[45, 30, if depth > 0 { 25 } else { 0 }]) {
            0 if !pool.is_empty() => self.c.pick(&pool).clone(),
            0 | 1 => {
                let n = self.number(ad);
                if ad_is_zero(ad) && !n.starts_with('-') {
                    n
                } else {
                    format!("({n})")
                }
            }
            _ => self.reduction(ad, depth - 1),
        }
    }

    /// A number computed from arrays: `norm`, `dot`, `@`, an entry.
    fn reduction(&mut self, ad: AD, depth: u32) -> String {
        match self.c.weighted(&[3, 3, 2, 3]) {
            0 => {
                let s = self.any_shape();
                format!("norm({})", self.array(s, ad, depth))
            }
            1 | 2 => {
                // `dot` and `@` of two vectors of one length, dimensions
                // adding up to `ad`.
                let s = self.vector_shape();
                let d1 = self.any_ad();
                let d2 = ad_sub(ad, d1);
                if !ad_in_range(d2) {
                    return self.number_parenthesized(ad);
                }
                let (a, b) = (self.array(s, d1, depth), self.array(s, d2, depth));
                if self.c.chance(50) {
                    format!("dot({a}, {b})")
                } else {
                    format!("({a} @ {b})")
                }
            }
            _ => {
                let s = self.any_shape();
                let e = self.array(s, ad, depth);
                let index = match s {
                    AShape::V3 => format!("[{}]", self.c.below(3)),
                    AShape::V5 => format!("[{}]", self.c.below(5)),
                    AShape::M23 => format!("[{}, {}]", self.c.below(2), self.c.below(3)),
                    AShape::Long => format!("[{}]", self.c.below(self.long)),
                };
                format!("({e}){index}")
            }
        }
    }

    fn number_parenthesized(&mut self, ad: AD) -> String {
        let n = self.number(ad);
        format!("({n})")
    }

    fn vector_shape(&mut self) -> AShape {
        if self.long > 0 && self.c.chance(30) {
            return AShape::Long;
        }
        *self.c.pick(&[AShape::V3, AShape::V5])
    }

    // ----- array expressions -----

    fn leaf(&mut self, s: AShape, ad: AD) -> String {
        let pool: Vec<String> = self
            .vars
            .iter()
            .filter(|v| v.shape == s && v.ad == ad)
            .map(|v| v.name.clone())
            .collect();
        if !pool.is_empty() && self.c.chance(85) {
            return self.c.pick(&pool).clone();
        }
        // A call of a function that returns this type.
        let calls: Vec<AFn> = self
            .fns
            .iter()
            .filter(|f| f.ret == (s, ad))
            .cloned()
            .collect();
        if !calls.is_empty() && self.c.chance(50) {
            let f = self.c.pick(&calls).clone();
            let args: Vec<String> = f
                .params
                .iter()
                .map(|&(ps, pad)| self.leaf_or_literal(ps, pad))
                .collect();
            return format!("{}({})", f.name, args.join(", "));
        }
        self.literal(s, ad)
    }

    fn leaf_or_literal(&mut self, s: AShape, ad: AD) -> String {
        let pool: Vec<String> = self
            .vars
            .iter()
            .filter(|v| v.shape == s && v.ad == ad)
            .map(|v| v.name.clone())
            .collect();
        if !pool.is_empty() && self.c.chance(80) {
            self.c.pick(&pool).clone()
        } else {
            self.literal(s, ad)
        }
    }

    /// An array expression of shape `s` and dimension `ad`.
    fn array(&mut self, s: AShape, ad: AD, depth: u32) -> String {
        self.array_e(s, ad, depth).s_or_paren()
    }

    fn array_e(&mut self, s: AShape, ad: AD, depth: u32) -> E {
        if depth == 0 || self.c.chance(12) {
            return atom(self.leaf(s, ad));
        }
        let d = depth - 1;
        match self.c.weighted(&[3, 3, 4, 3, 4, 1, 6, 5]) {
            // `a + b`, `a - b`
            0 => {
                let sym = if self.c.chance(50) { "+" } else { "-" };
                let (a, b) = (self.array_e(s, ad, d), self.array_e(s, ad, d));
                E {
                    s: format!("{} {sym} {}", at(a, 5), at(b, 6)),
                    p: 5,
                }
            }
            // `a * b`: dimensions add up
            1 | 2 => {
                let d1 = self.any_ad();
                let d2 = ad_sub(ad, d1);
                if !ad_in_range(d2) {
                    return atom(self.leaf(s, ad));
                }
                let (a, b) = (self.array_e(s, d1, d), self.array_e(s, d2, d));
                E {
                    s: format!("{} * {}", at(a, 6), at(b, 7)),
                    p: 6,
                }
            }
            // `a / b`
            3 => {
                let d2 = self.any_ad();
                let d1 = ad_add(ad, d2);
                if !ad_in_range(d1) {
                    return atom(self.leaf(s, ad));
                }
                let (a, b) = (self.array_e(s, d1, d), self.array_e(s, d2, d));
                E {
                    s: format!("{} / {}", at(a, 6), at(b, 7)),
                    p: 6,
                }
            }
            // a number scales an array
            4 => {
                let k = self.any_ad();
                match self.c.below(3) {
                    0 => {
                        let kd = ad_sub(ad, k);
                        if !ad_in_range(kd) {
                            return atom(self.leaf(s, ad));
                        }
                        let (x, a) = (self.scalar(kd, d), self.array_e(s, k, d));
                        E {
                            s: format!("{x} * {}", at(a, 7)),
                            p: 6,
                        }
                    }
                    1 => {
                        let kd = ad_sub(ad, k);
                        if !ad_in_range(kd) {
                            return atom(self.leaf(s, ad));
                        }
                        let (a, x) = (self.array_e(s, k, d), self.scalar(kd, d));
                        E {
                            s: format!("{} * {x}", at(a, 6)),
                            p: 6,
                        }
                    }
                    _ => {
                        let ka = ad_add(ad, k);
                        if !ad_in_range(ka) {
                            return atom(self.leaf(s, ad));
                        }
                        let (a, x) = (self.array_e(s, ka, d), self.scalar(k, d));
                        E {
                            s: format!("{} / {x}", at(a, 6)),
                            p: 6,
                        }
                    }
                }
            }
            // `-a`
            5 => {
                let a = self.array_e(s, ad, d);
                E {
                    s: format!("-{}", at(a, 7)),
                    p: 7,
                }
            }
            // a function of one array
            6 => self.math1(s, ad, d),
            // a function of two
            _ => self.math2(s, ad, d),
        }
    }

    fn guarded(&mut self, s: AShape, ad: AD, d: u32, guard: &str) -> String {
        let a = self.array(s, ad, d);
        if self.c.chance(100 - self.risk.min(60)) {
            format!("{guard}({a})")
        } else {
            a
        }
    }

    fn math1(&mut self, s: AShape, ad: AD, d: u32) -> E {
        // The functions whose result has dimension `ad`.
        let mut options: Vec<&str> = vec!["abs"];
        if ad_in_range((ad.0 * 2, ad.1 * 2)) {
            options.push("sqrt");
        }
        if ad_in_range((ad.0 * 3, ad.1 * 3)) {
            options.push("cbrt");
        }
        if ad_is_zero(ad) {
            options.extend([
                "sign", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
                "floor", "ceil", "round", "expm1", "log1p", "sinh", "cosh", "tanh", "asinh",
                "acosh", "atanh", "erf",
            ]);
        }
        let f = *self.c.pick(&options);
        let text = match f {
            "abs" => {
                let a = self.array(s, ad, d);
                format!("abs({a})")
            }
            "sqrt" => {
                let arg = self.guarded(s, (ad.0 * 2, ad.1 * 2), d, "abs");
                format!("sqrt({arg})")
            }
            "cbrt" => {
                let a = self.array(s, (ad.0 * 3, ad.1 * 3), d);
                format!("cbrt({a})")
            }
            // Any dimension in, a plain number out.
            "sign" => {
                let k = self.any_ad();
                let a = self.array(s, k, d);
                format!("sign({a})")
            }
            "log" | "log10" | "log2" | "log1p" => {
                let arg = self.guarded(s, ad, d, "abs");
                format!("{f}({arg})")
            }
            "asin" | "acos" | "atanh" => {
                let arg = self.guarded(s, ad, d, "tanh");
                format!("{f}({arg})")
            }
            "acosh" => {
                let a = self.array(s, ad, d);
                if self.c.chance(90) {
                    format!("acosh(cosh(tanh({a})))")
                } else {
                    format!("acosh({a})")
                }
            }
            "exp" | "expm1" | "sinh" | "cosh" => {
                let arg = self.guarded(s, ad, d, "tanh");
                format!("{f}({arg})")
            }
            _ => {
                let a = self.array(s, ad, d);
                format!("{f}({a})")
            }
        };
        atom(text)
    }

    fn math2(&mut self, s: AShape, ad: AD, d: u32) -> E {
        let f = if ad_is_zero(ad) {
            *self.c.pick(&["atan2", "hypot", "min", "max"])
        } else {
            *self.c.pick(&["hypot", "min", "max"])
        };
        // `atan2` takes any common dimension and gives a plain number; the
        // others keep the dimension.
        let common = if f == "atan2" { self.any_ad() } else { ad };
        // At least one argument is an array; the other may be a number.
        let (a, b) = match self.c.weighted(&[6, 2, 2]) {
            0 => (self.array(s, common, d), self.array(s, common, d)),
            1 => (self.array(s, common, d), self.scalar(common, d)),
            _ => (self.scalar(common, d), self.array(s, common, d)),
        };
        atom(format!("{f}({a}, {b})"))
    }

    // ----- statements -----

    fn print_line(&mut self) {
        let mut parts = Vec::new();
        for _ in 0..1 + self.c.below(3) {
            if self.c.chance(75) && !self.vars.is_empty() {
                let i = self.c.below(self.vars.len());
                let v = &self.vars[i];
                // A long vector prints as its length-independent summary.
                if v.shape == AShape::Long && self.c.chance(80) {
                    parts.push(format!("{}[{}]", v.name, 1));
                } else {
                    parts.push(v.name.clone());
                }
            } else {
                let s = self.any_shape();
                let ad = self.any_ad();
                let e = self.array(s, ad, 2);
                if s == AShape::Long {
                    parts.push(format!("norm({e})"));
                } else {
                    parts.push(e);
                }
            }
        }
        let text = format!("print({})", parts.join(", "));
        self.line(&text);
    }

    fn bind_array(&mut self) {
        let s = self.any_shape();
        let ad = self.any_ad();
        let depth = 1 + self.c.below(4) as u32;
        let e = self.array(s, ad, depth);
        let name = self.fresh("a");
        self.line(&format!("{name} = {e}"));
        self.vars.push(AVar { name, shape: s, ad });
    }

    fn bind_scalar(&mut self) {
        let ad = self.any_ad();
        let e = self.scalar(ad, 2);
        let name = self.fresh("k");
        self.line(&format!("{name} = {e}"));
        self.svars.push(SVar {
            name: name.clone(),
            ad,
        });
        if self.c.chance(50) {
            self.line(&format!("print({name})"));
        }
    }

    /// A loop that feeds an array back into itself.
    fn loop_stmt(&mut self) {
        if self.vars.is_empty() {
            return;
        }
        let i = self.c.below(self.vars.len());
        let (name, s, ad) = (
            self.vars[i].name.clone(),
            self.vars[i].shape,
            self.vars[i].ad,
        );
        let n = 1 + self.c.below(5);
        let counter = self.fresh("i");
        self.line(&format!("for {counter} in range({n}):"));
        self.ind += 1;
        for _ in 0..1 + self.c.below(2) {
            let depth = 2 + self.c.below(2) as u32;
            let e = self.array(s, ad, depth);
            self.line(&format!("{name} = {e}"));
        }
        if self.c.chance(30) {
            self.line(&format!("print({name}[0])"));
        }
        self.ind -= 1;
    }

    /// `def f(p0: Vector[3] m, ...) -> Vector[3] m: return <expression>`.
    fn function(&mut self) {
        let nparams = 1 + self.c.below(3);
        let mut params = Vec::new();
        for _ in 0..nparams {
            let s = self.any_shape();
            let ad = *self.c.pick(NICE);
            params.push((s, ad));
        }
        let ret = (params[0].0, *self.c.pick(NICE));
        let name = self.fresh("f");
        let text: Vec<String> = params
            .iter()
            .enumerate()
            .map(|(i, &(s, ad))| format!("p{i}: {}{}", self.shape_text(s), ad_annotation(ad)))
            .collect();
        self.line(&format!(
            "def {name}({}) -> {}{}:",
            text.join(", "),
            self.shape_text(ret.0),
            ad_annotation(ret.1)
        ));
        // The body sees its parameters and the functions defined before.
        let saved = std::mem::take(&mut self.vars);
        let saved_scalars = std::mem::take(&mut self.svars);
        self.in_fn = true;
        for (i, &(s, ad)) in params.iter().enumerate() {
            self.vars.push(AVar {
                name: format!("p{i}"),
                shape: s,
                ad,
            });
        }
        self.ind += 1;
        let depth = 2 + self.c.below(3) as u32;
        let e = self.array(ret.0, ret.1, depth);
        self.line(&format!("return {e}"));
        self.ind -= 1;
        self.in_fn = false;
        self.vars = saved;
        self.svars = saved_scalars;
        self.fns.push(AFn { name, params, ret });
    }
}

impl E {
    /// The text, parenthesized unless it is an atom: for places that take a
    /// whole expression but where the caller does not track binding
    /// strength (function arguments and statements never need it, but
    /// operands do).
    fn s_or_paren(self) -> String {
        if self.p >= 9 {
            self.s
        } else {
            format!("({})", self.s)
        }
    }
}

/// A program of arrays and the entry-by-entry operations on them, from
/// `data` (see the section comment above).
pub fn generate_arrays(data: &[u8]) -> String {
    let mut c = Choices::new(data);
    let risk = *c.pick(&[0, 0, 0, 3, 8, 20, 45]);
    let long = if c.chance(28) { 520 + c.below(180) } else { 0 };
    let mut g = Ar {
        c,
        out: String::new(),
        ind: 0,
        vars: Vec::new(),
        svars: Vec::new(),
        fns: Vec::new(),
        risk,
        counter: 0,
        long,
        in_fn: false,
    };
    // Some scalars first, then the arrays the expressions start from.
    for _ in 0..g.c.below(3) {
        g.bind_scalar();
    }
    for _ in 0..2 + g.c.below(4) {
        let s = g.any_shape();
        let ad = g.any_ad();
        let lit = g.literal(s, ad);
        let name = g.fresh("a");
        g.line(&format!("{name} = {lit}"));
        g.vars.push(AVar { name, shape: s, ad });
    }
    if g.long > 0 && !g.vars.iter().any(|v| v.shape == AShape::Long) {
        let ad = g.any_ad();
        let lit = g.literal(AShape::Long, ad);
        let name = g.fresh("a");
        g.line(&format!("{name} = {lit}"));
        g.vars.push(AVar {
            name,
            shape: AShape::Long,
            ad,
        });
    }
    let mut budget = 3 + g.c.below(12) as i32;
    while budget > 0 && !g.c.exhausted() {
        budget -= 1;
        match g.c.weighted(&[40, 8, 12, 6, 6]) {
            0 => g.bind_array(),
            1 => g.bind_scalar(),
            2 => g.print_line(),
            3 => g.loop_stmt(),
            _ => g.function(),
        }
    }
    // The final state.
    let last: Vec<String> = g
        .vars
        .iter()
        .rev()
        .take(4)
        .map(|v| {
            if v.shape == AShape::Long {
                format!("norm({})", v.name)
            } else {
                v.name.clone()
            }
        })
        .collect();
    if !last.is_empty() {
        g.line(&format!("print({})", last.join(", ")));
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
    /// Why native code refused to compile the program (lowered or
    /// optimized), if it did; `None` when it compiled both and agreed.
    pub native_refusal: Option<String>,
    /// Native code deoptimized (a value the checker could not type was not
    /// what a declaration says) and the SIR interpreter finished the run.
    pub native_deopt: bool,
}

/// Bugs this fuzzer found that are not fixed yet: an id and the text of the
/// error message that identifies each. A lenient run ([`check_source`])
/// skips the rest of the check when it hits one, so the fuzzer keeps looking
/// for new bugs; a strict run ([`check_source_strict`], used by the ignored
/// regression tests in `fuzz_differential.rs`) reports it. Delete the entry
/// when the bug is fixed.
const KNOWN: &[(&str, &str)] = &[];

fn known_issue(msg: &str) -> Option<&'static str> {
    KNOWN
        .iter()
        .find(|(_, needle)| msg.contains(needle))
        .map(|(id, _)| *id)
}

/// A run-time error: its message, and the diagnostic as `sky run` prints it
/// (rendered against the program, so a span, a label or a note that differs
/// is a difference).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub message: String,
    pub rendered: String,
}

impl Failure {
    fn new(d: &sky_diag::Diagnostic, message: &str, src: &str) -> Failure {
        Failure {
            message: message.to_string(),
            rendered: d.render_to_string(&sky_diag::SourceFile::new("program.sky", src)),
        }
    }
}

/// A run's result for a report: `Ok`, or the rendered diagnostic.
fn shown(r: &Result<(), Failure>) -> String {
    match r {
        Ok(()) => "ok".to_string(),
        Err(f) => f.rendered.clone(),
    }
}

/// The two resource-limit errors compare as one: which trips first depends
/// on each interpreter's stack use per call, not on the program. Any other
/// error compares as the whole rendered diagnostic.
fn normalized(r: &Result<(), Failure>) -> Result<(), String> {
    r.clone().map_err(|f| {
        if f.message.starts_with("recursion is too deep")
            || f.message == "the program nests too deeply for the interpreter"
        {
            "<resource limit>".to_string()
        } else {
            f.rendered
        }
    })
}

/// Whether `msg` is the failure of a contract (spec "Contracts"): a false
/// condition, or one that is not a Bool.
fn is_contract_failure(msg: &str) -> bool {
    msg.starts_with("assertion failed")
        || msg.starts_with("precondition of ")
        || msg.starts_with("postcondition of ")
        || ["assert", "requires", "ensures"]
            .iter()
            .any(|k| msg.starts_with(&format!("`{k}` needs a Bool condition")))
}

/// Where the engines stop is part of the failure of a contract: both point
/// at the condition. (The other errors' spans are the tests' business.)
fn check_contract_span(
    ast: &sky_ast::Module,
    src: &str,
    checked: &sky_types::Checked,
    m: &sky_ir::Module,
) -> Result<(), String> {
    let mut env = sky_interp::RuntimeEnv::with_prelude();
    env.add_chain_plans(&checked.chains);
    let walker = sky_interp::run_module_with(
        ast,
        src,
        &mut env,
        &mut Vec::new(),
        &checked.cholesky_solves,
    )
    .err()
    .map(|e| (e.message.clone(), e.span));
    let sir = sky_ir_interp::run(m, &mut Vec::new())
        .err()
        .map(|e| (e.message.clone(), e.span));
    if walker != sir {
        return Err(format!(
            "a contract failure differs in message or span\nsky run: {walker:?}\nSIR:     {sir:?}\n--- SIR text ---\n{}",
            sky_ir::print(m)
        ));
    }
    Ok(())
}

/// What the AST interpreter (`sky run`) does.
fn interpret(
    ast: &sky_ast::Module,
    src: &str,
    checked: &sky_types::Checked,
) -> (String, Result<(), Failure>) {
    let mut env = sky_interp::RuntimeEnv::with_prelude();
    // What `sky run` does after checking: it follows the checker's rewrites
    // (the association of an `@` chain, Cholesky solves), as lowering does.
    env.add_chain_plans(&checked.chains);
    let mut out = Vec::new();
    let r = sky_interp::run_module_with(ast, src, &mut env, &mut out, &checked.cholesky_solves);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.map(|_| ())
            .map_err(|e| Failure::new(&e.to_diagnostic(), &e.message, src)),
    )
}

/// What the SIR reference interpreter does with `m`.
fn run_sir(m: &sky_ir::Module, src: &str) -> (String, Result<(), Failure>) {
    let mut out = Vec::new();
    let r = sky_ir_interp::run(m, &mut out);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.map_err(|e| Failure::new(&e.to_diagnostic(), &e.message, src)),
    )
}

/// What native code did with a module.
enum Native {
    /// It compiled and did what the tree-walker did.
    Same,
    /// It refused to compile.
    Refused(String),
    /// It deoptimized (a value of unknown type was not what a declaration
    /// says), and the SIR interpreter finished the run: the same stdout and
    /// failure.
    Deopt,
}

/// What native code does with `m`, compared with the tree-walker's `want`
/// (same stdout, same failure as a whole diagnostic); `Err` if it differs.
fn check_native(
    m: &sky_ir::Module,
    which: &str,
    src: &str,
    want_out: &str,
    want: &Result<(), Failure>,
) -> Result<Native, String> {
    // Native code without vectorizing, with it where it pays, and with it
    // everywhere (pairs of entries and of scalar operations as two lanes):
    // all three must do what the tree-walker did.
    let mut first = None;
    for slp in [sky_jit::Slp::Off, sky_jit::Slp::Auto, sky_jit::Slp::Always] {
        let r = check_native_with(m, which, src, want_out, want, slp)?;
        first.get_or_insert(r);
    }
    Ok(first.unwrap_or(Native::Same))
}

fn check_native_with(
    m: &sky_ir::Module,
    which: &str,
    src: &str,
    want_out: &str,
    want: &Result<(), Failure>,
    slp: sky_jit::Slp,
) -> Result<Native, String> {
    let compiled = match sky_jit::compile_with_slp(m, sky_jit::Numerics::Reproducible, slp) {
        Ok(c) => c,
        Err(e) => return Ok(Native::Refused(e.message)),
    };
    let mut out = Vec::new();
    let (r, deopt) = compiled.run_or_rerun(&mut out, |w| sky_ir_interp::run(m, &mut { w }));
    let got_out = String::from_utf8_lossy(&out).into_owned();
    let got = r.map_err(|e| Failure::new(&e.to_diagnostic(), &e.message, src));
    if got_out != want_out {
        return Err(format!(
            "native code ({which} SIR, {slp:?}) prints differently\n--- sky run ---\n{want_out}(error: {})\n--- native ---\n{got_out}(error: {})\n--- SIR text ---\n{}",
            shown(want),
            shown(&got),
            sky_ir::print(m)
        ));
    }
    if normalized(&got) != normalized(want) {
        return Err(format!(
            "native code ({which} SIR, {slp:?}) fails differently\n--- sky run ---\n{}--- native ---\n{}--- stdout (both) ---\n{want_out}\n--- SIR text ---\n{}",
            shown(want),
            shown(&got),
            sky_ir::print(m)
        ));
    }
    Ok(if deopt { Native::Deopt } else { Native::Same })
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
                    native_refusal: None,
                    native_deopt: false,
                }),
                _ => Err(msg),
            };
        }
    };
    if let Err(errors) = sky_ir::verify(&m) {
        let ms: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
        return Err(format!("the lowered SIR does not verify: {ms:?}"));
    }

    let (want_out, want) = interpret(&ast, src, &checked);
    let (got_out, got) = run_sir(&m, src);
    if let Err(msg) = &want {
        if is_contract_failure(&msg.message) {
            check_contract_span(&ast, src, &checked, &m)?;
        }
    }
    let dump = |m: &sky_ir::Module| sky_ir::print(m);
    if got_out != want_out {
        return Err(format!(
            "stdout differs\n--- sky run ---\n{want_out}(error: {})\n--- SIR ---\n{got_out}(error: {})\n--- SIR text ---\n{}",
            shown(&want),
            shown(&got),
            dump(&m)
        ));
    }
    if normalized(&got) != normalized(&want) {
        return Err(format!(
            "the error differs\n--- sky run ---\n{}--- SIR ---\n{}--- stdout (both) ---\n{want_out}\n--- SIR text ---\n{}",
            shown(&want),
            shown(&got),
            dump(&m)
        ));
    }

    // The native engine, when it compiles the program.
    let mut native_refusal = None;
    let mut native_deopt = false;
    let mut optimized = m.clone();
    if let Err(e) = sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut optimized)
    {
        return Err(format!(
            "the pipeline fails on a lowered program: {e}\n--- SIR text ---\n{}",
            sky_ir::print(&m)
        ));
    }
    for (which, module) in [("lowered", &m), ("optimized", &optimized)] {
        match check_native(module, which, src, &want_out, &want)? {
            Native::Same => {}
            Native::Refused(why) => {
                native_refusal.get_or_insert(why);
            }
            Native::Deopt => native_deopt = true,
        }
    }

    let mut known = None;
    let outcome = |known| Outcome {
        failed: want.is_err(),
        limit: normalized(&want) == Err("<resource limit>".to_string()),
        error: want.clone().err().map(|f| f.message),
        printed: want_out.len(),
        known,
        native_refusal: native_refusal.clone(),
        native_deopt,
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
    let (back_out, back_r) = run_sir(&back, src);
    if back_out != got_out || normalized(&back_r) != normalized(&got) {
        return Err(format!(
            "the reparsed SIR runs differently\nbefore: {got_out:?} {}\nafter:  {back_out:?} {}\n--- SIR text ---\n{text}",
            shown(&got),
            shown(&back_r)
        ));
    }

    Ok(outcome(known))
}

/// [`check`] on a program of small arrays ([`generate_vectors`]).
pub fn check_vectors(data: &[u8]) -> Outcome {
    let src = generate_vectors(data);
    match check_source(&src) {
        Ok(o) => o,
        Err(msg) => panic!("{msg}\n--- program ---\n{src}"),
    }
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
