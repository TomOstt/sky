//! What a call site knows that the callee's signature does not.
//!
//! A [`Fact`] is one such piece of knowledge about the arguments of one
//! call. Four kinds:
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
//!   cross it);
//! * **Type**: an `unknown` parameter (unannotated) and an argument whose
//!   type is known exactly: a number of a dimension and a concrete shape, a
//!   Bool, a string. The clone's parameter has that type, and the body is
//!   typed again from it ([`super::retype`]). The clone is for arguments of
//!   exactly that type: a different dimension (`3` and `3 m`), shape or kind
//!   is another clone, and the generic function stays for every call that
//!   gives no type.
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
//! fails). Such a parameter gets a **type fact** instead, and only that: once
//! the clone knows the type, its checks are the typed ones.
//!
//! The name of a specialization is `f[` + its facts, sorted (sizes by
//! variable, then parameters in order) and separated by commas + `]`:
//! `f[N=3]`, `f[x=2]`, `f[N=3,x>0]`, and for a type `f[n:Scalar]`
//! (dimensionless), `f[n:Scalar(length)]`, `f[v:Vector[3](time)]`, `f[b:Bool]`.
//! The same facts always give the same name, which is how duplicates are
//! found.

use sky_ir::exact::{self, Analysis, Range};
use sky_ir::{Function, NumberType, OpData, OpKind, Repr, Type, ValueDef, ValueId};
use sky_types::{ShapeDim, ShapeVar, TagSet};
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
    Size {
        var: ShapeVar,
        n: u64,
    },
    Const {
        param: usize,
        value: Const,
    },
    Sign {
        param: usize,
        positive: bool,
    },
    /// The parameter, which the callee declares `unknown`, has this type
    /// (no tags, representation unchosen, every size a number).
    Type {
        param: usize,
        ty: Type,
    },
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
            Kind::Type { ty, .. } => format!("{}:{}", self.name, type_key(ty)),
        }
    }

    fn order(&self) -> (u8, u64) {
        match &self.kind {
            Kind::Size { var, .. } => (0, var.id()),
            Kind::Const { param, .. } | Kind::Sign { param, .. } | Kind::Type { param, .. } => {
                (1, *param as u64)
            }
        }
    }

    pub(crate) fn is_sign(&self) -> bool {
        matches!(self.kind, Kind::Sign { .. })
    }

    pub(crate) fn is_type(&self) -> bool {
        matches!(self.kind, Kind::Type { .. })
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

/// The type as the name of a specialization writes it: `Scalar`,
/// `Vector[3]`, `Matrix[2,3]`, `Bool`, `Str`, and after a number's shape its
/// dimension in parentheses unless it is dimensionless (`Scalar(length)`).
pub(crate) fn type_key(t: &Type) -> String {
    match t {
        Type::Number(n) => {
            let dims: Vec<String> = n.shape.dims().iter().map(|d| d.to_string()).collect();
            let shape = match dims.len() {
                0 => "Scalar".to_string(),
                1 => format!("Vector[{}]", dims[0]),
                _ => format!("Matrix[{}]", dims.join(",")),
            };
            if n.dim.is_dimensionless() {
                shape
            } else {
                let dim = n
                    .dim
                    .name()
                    .map_or_else(|| n.dim.to_string(), str::to_string);
                format!("{shape}({dim})")
            }
        }
        Type::Bool => "Bool".to_string(),
        Type::Str => "Str".to_string(),
        other => other.to_string(),
    }
}

/// [`type_key`] as words: `Scalar (dimensionless)`, `Vector[3] (length)`,
/// `Bool`.
pub(crate) fn type_words(t: &Type) -> String {
    words_of_type_key(&type_key(t))
}

/// The words for a type as [`type_key`] writes it.
pub(crate) fn words_of_type_key(key: &str) -> String {
    match key.find('(') {
        Some(i) => format!("{} {}", &key[..i], &key[i..]),
        None if key.starts_with("Scalar")
            || key.starts_with("Vector")
            || key.starts_with("Matrix") =>
        {
            format!("{key} (dimensionless)")
        }
        None => key.to_string(),
    }
}

/// The type a type fact gives a parameter for an argument of type `t`: a
/// number of unchosen representation with a concrete shape of at most two
/// axes (tags dropped: a clone promises nothing about them), a Bool or a
/// string. `None` for anything else.
fn fact_type(t: &Type) -> Option<Type> {
    match t {
        Type::Number(n)
            if n.repr == Repr::Unchosen
                && n.shape.rank() <= 2
                && n.shape
                    .dims()
                    .iter()
                    .all(|d| matches!(d, ShapeDim::Known(_))) =>
        {
            Some(Type::Number(NumberType {
                tags: TagSet::EMPTY,
                ..n.clone()
            }))
        }
        Type::Bool | Type::Str => Some(t.clone()),
        _ => None,
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

/// The type facts a call of `callee` would give if its arguments had the
/// types `args` (as inference has them, not as declared): one per
/// `unknown` parameter the body reads, as [`derive`] makes them. `None` if
/// such a parameter's argument has no fact type (it stays unknown), or if
/// no parameter gets one.
pub(crate) fn type_facts(callee: &Function, args: &[&Type]) -> Option<Vec<Fact>> {
    if args.len() != callee.params.len() {
        return None;
    }
    let used = used_params(callee);
    let mut facts = Vec::new();
    for (i, (arg, param)) in args.iter().zip(&callee.params).enumerate() {
        if !used[i] || !matches!(param, Type::Unknown) {
            continue;
        }
        let ty = fact_type(arg)?;
        let name = param_name(callee, i);
        facts.push(Fact {
            kind: Kind::Type {
                param: i,
                ty: ty.clone(),
            },
            why: format!("the argument for `{name}` is a {}", type_words(&ty)),
            name,
        });
    }
    if facts.is_empty() {
        return None;
    }
    sort(&mut facts);
    Some(facts)
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
        if matches!(param, Type::Unknown) {
            if let Some(ty) = fact_type(caller.ty(arg)) {
                facts.push(Fact {
                    kind: Kind::Type {
                        param: i,
                        ty: ty.clone(),
                    },
                    why: format!("the argument for `{name}` is a {}", type_words(&ty)),
                    name,
                });
            }
            continue;
        }
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
