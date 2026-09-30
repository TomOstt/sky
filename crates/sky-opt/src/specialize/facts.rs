//! What a call site knows that the callee's signature does not.
//!
//! A [`Fact`] is one such piece of knowledge about the arguments of one
//! call. Three kinds:
//!
//! * **Size**: the callee has a symbolic size (`Vector[N]`) and the argument
//!   has a concrete one (`Vector[3]`): `N = 3`;
//! * **Const**: a scalar parameter and an argument that is a constant: a
//!   literal (`const.float`, `const.rational`, `const.bool`), or a number the
//!   exactness analysis proves is one value (`2 * 3`). (A Bool parameter has
//!   no annotation in v0.1, so it is `unknown` and gets no fact from Sky
//!   source; the case is here for IR that types one.)
//! * **Sign**: a scalar parameter whose argument the analysis proves strictly
//!   positive or strictly negative (a range that excludes zero and does not
//!   cross it).
//!
//! Only parameters the callee's body reads matter, so a fact about an unused
//! parameter is never recorded (two calls that differ only there share one
//! specialization, or none is made).
//!
//! Facts are trusted as far as the types they come from are: a `Vector[3]`
//! argument is a three-element vector, as lowering already assumes when it
//! writes a constant for `len`. A parameter that the callee declares with an
//! `unknown` type gets no fact (its operations check their operands at run
//! time, and a constant of a more precise type could change which check
//! fails).
//!
//! The name of a specialization is `f[` + its facts, sorted (sizes by
//! variable, then parameters in order) and separated by commas + `]`:
//! `f[N=3]`, `f[x=2]`, `f[N=3,x>0]`. The same facts always give the same
//! name, which is how duplicates are found.

use sky_ir::exact::{self, Analysis, Range};
use sky_ir::{Function, OpData, OpKind, Repr, Type, ValueDef, ValueId};
use sky_types::{ShapeDim, ShapeVar};
use sky_units::Rational;

use crate::util::reachable_ops;

/// A constant argument.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Const {
    Rational(Rational),
    /// The bits of a finite `f64` (kept as bits so `-0` and `0` differ).
    Float(u64),
    Bool(bool),
}

impl Const {
    /// The value as the name of a specialization writes it.
    pub(crate) fn text(self) -> String {
        match self {
            Const::Rational(q) if q.is_integer() => q.numer().to_string(),
            Const::Rational(q) => format!("{}/{}", q.numer(), q.denom()),
            Const::Float(bits) => {
                let s = format!("{:?}", f64::from_bits(bits));
                s.strip_suffix(".0").map_or(s.clone(), str::to_string)
            }
            Const::Bool(b) => b.to_string(),
        }
    }
}

/// What is known about one parameter (or size variable).
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Kind {
    Size { var: ShapeVar, n: u64 },
    Const { param: usize, value: Const },
    Sign { param: usize, positive: bool },
}

/// One fact, and the reason to show for it.
#[derive(Clone, Debug)]
pub(crate) struct Fact {
    pub(crate) kind: Kind,
    /// The parameter's name, or `argN`.
    pub(crate) name: String,
    /// One sentence for `--explain`: "the argument for `a` is a Vector[3]".
    pub(crate) why: String,
}

impl Fact {
    /// The fact as it appears in the name: `N=3`, `x=2`, `x>0`.
    pub(crate) fn key(&self) -> String {
        match &self.kind {
            Kind::Size { var, n } => format!("{}={n}", var.name()),
            Kind::Const { value, .. } => format!("{}={}", self.name, value.text()),
            Kind::Sign { positive, .. } => {
                format!("{}{}0", self.name, if *positive { '>' } else { '<' })
            }
        }
    }

    fn order(&self) -> (u8, u64) {
        match &self.kind {
            Kind::Size { var, .. } => (0, var.id()),
            Kind::Const { param, .. } | Kind::Sign { param, .. } => (1, *param as u64),
        }
    }

    pub(crate) fn is_sign(&self) -> bool {
        matches!(self.kind, Kind::Sign { .. })
    }

    /// The range a sign fact gives its parameter.
    pub(crate) fn range(&self) -> Option<Range> {
        match self.kind {
            Kind::Sign { positive: true, .. } => {
                Some(Range::new(exact::range::next_up(0.0), f64::MAX))
            }
            Kind::Sign {
                positive: false, ..
            } => Some(Range::new(-f64::MAX, exact::range::next_down(0.0))),
            _ => None,
        }
    }
}

/// Puts facts in their canonical order.
pub(crate) fn sort(facts: &mut [Fact]) {
    facts.sort_by_key(Fact::order);
}

/// The name of `base` specialized for `facts`: `f[N=3]`. A specialization
/// specialized again adds its facts to the same brackets.
pub(crate) fn clone_name(base: &str, facts: &[Fact]) -> String {
    let keys: Vec<String> = facts.iter().map(Fact::key).collect();
    match base.strip_suffix(']') {
        Some(open) if base.contains('[') => format!("{open},{}]", keys.join(",")),
        _ => format!("{base}[{}]", keys.join(",")),
    }
}

/// Which parameters of `f` its body reads.
pub(crate) fn used_params(f: &Function) -> Vec<bool> {
    let params = f.param_values();
    let mut used = vec![false; params.len()];
    for o in reachable_ops(f) {
        for v in &f.op(o).operands {
            if let Some(i) = params.iter().position(|p| p == v) {
                used[i] = true;
            }
        }
    }
    used
}

/// The name a parameter is shown by.
pub(crate) fn param_name(f: &Function, i: usize) -> String {
    f.param_values()
        .get(i)
        .and_then(|v| f.value(*v).name.clone())
        .unwrap_or_else(|| format!("arg{i}"))
}

/// Whether a scalar parameter type takes part in constant and sign facts:
/// a typed `f64` scalar.
fn is_scalar_f64(t: &Type) -> bool {
    matches!(t, Type::Number(n) if n.shape.rank() == 0 && matches!(n.repr, Repr::Unchosen | Repr::F64))
}

/// A constant an argument is, if it is one: a literal, or a nonzero number
/// the analysis proves is a single `f64` (a zero could be `+0` or `-0`, and
/// the range cannot say which).
fn constant_of(caller: &Function, an: &Analysis, arg: ValueId, param: &Type) -> Option<Const> {
    let ValueDef::OpResult { op, .. } = caller.value(arg).def else {
        return None;
    };
    let kind = &caller.op(op).kind;
    match (kind, param) {
        (OpKind::ConstBool(b), Type::Bool) => return Some(Const::Bool(*b)),
        (_, Type::Bool) => return None,
        _ => {}
    }
    if !is_scalar_f64(param) || !matches!(caller.ty(arg), Type::Number(a) if a.shape.rank() == 0) {
        return None;
    }
    match kind {
        OpKind::ConstRational(q) => Some(Const::Rational(*q)),
        OpKind::ConstFloat { bits } if f64::from_bits(*bits).is_finite() => {
            Some(Const::Float(*bits))
        }
        _ => {
            let x = an.known(arg)?.range.as_point()?;
            (x != 0.0).then(|| Const::Float(x.to_bits()))
        }
    }
}

/// Every fact the call `call` (in `caller`, analysed as `an`) gives about
/// the parameters of `callee` that its body reads. Empty when the call
/// gives no usable fact (or contradicts itself).
pub(crate) fn derive(
    caller: &Function,
    an: &Analysis,
    call: &OpData,
    callee: &Function,
) -> Vec<Fact> {
    if call.operands.len() != callee.params.len() {
        return Vec::new();
    }
    let used = used_params(callee);
    let mut facts: Vec<Fact> = Vec::new();
    for (i, (&arg, param)) in call.operands.iter().zip(&callee.params).enumerate() {
        if !used[i] {
            continue;
        }
        let name = param_name(callee, i);
        let (Type::Number(_) | Type::Bool, arg_ty) = (param, caller.ty(arg)) else {
            continue;
        };
        // Sizes: a variable the parameter's shape has and the argument's
        // shape fixes.
        if let (Type::Number(p), Type::Number(a)) = (param, arg_ty) {
            if p.shape.rank() == a.shape.rank() {
                for (pd, ad) in p.shape.dims().iter().zip(a.shape.dims()) {
                    let (ShapeDim::Var(var), ShapeDim::Known(n)) = (pd, ad) else {
                        continue;
                    };
                    let already = facts.iter().find_map(|f| match &f.kind {
                        Kind::Size { var: v, n: m } if v == var => Some(*m),
                        _ => None,
                    });
                    match already {
                        Some(m) if m != *n => return Vec::new(),
                        Some(_) => {}
                        None => facts.push(Fact {
                            kind: Kind::Size {
                                var: var.clone(),
                                n: *n,
                            },
                            name: name.clone(),
                            why: format!(
                                "the argument for `{name}` is a {}, so {} = {n}",
                                a.shape,
                                var.name()
                            ),
                        }),
                    }
                }
            }
        }
        if let Some(value) = constant_of(caller, an, arg, param) {
            facts.push(Fact {
                kind: Kind::Const { param: i, value },
                why: format!("the argument for `{name}` is the constant {}", value.text()),
                name,
            });
        } else if is_scalar_f64(param) {
            let Some(k) = an.known(arg) else { continue };
            let positive = k.is_positive();
            if positive || k.is_negative() {
                facts.push(Fact {
                    kind: Kind::Sign { param: i, positive },
                    why: format!(
                        "the argument for `{name}` is {} ({})",
                        if positive { "positive" } else { "negative" },
                        k.range
                    ),
                    name,
                });
            }
        }
    }
    sort(&mut facts);
    facts
}
