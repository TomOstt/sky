//! Constant folding.
//!
//! The one rule that governs every fold here: **the folded program must
//! behave exactly like the original under the reference interpreter.**
//! That is why folding never computes anything itself. It builds the
//! operands as interpreter [`Value`]s and calls the same functions the
//! interpreters call (`sky_interp::arith`). If they report an error
//! (division by zero, overflow, a dimension mismatch, a non-real power),
//! the operation is left in place, so the run-time error still happens, at
//! the same operation. If they succeed, the result is what the interpreter
//! would have computed, bit for bit.
//!
//! # Exact rationals (roadmap idea 9)
//!
//! When every operand is a `const.rational`, the result is also computed
//! with `Rational`'s checked operations (overflow means "not exact",
//! leave it as float). Because the interpreter evaluates in `f64`, an
//! exact result can differ from the interpreter's in the last bit (`1/10 +
//! 2/10` is exactly `3/10`, but `0.1 + 0.2` is not `0.3` in floats). The
//! mode decides what wins:
//!
//! * [`FoldMode::BitIdentical`] (the default, used by [`fold`]): the exact
//!   result is kept as a `const.rational` only if converting it to `f64`
//!   gives the interpreter's bits. Otherwise the interpreter's float is
//!   folded in as `const.float`. Output never changes.
//! * [`FoldMode::Exact`]: the exact result is always kept (when the
//!   interpreter would not have failed). Last-bit differences from the
//!   float interpreter are possible; for programs that opt into exactness.
//!
//! # What the explanations say about exactness
//!
//! A float result whose value over the reals has a closed form gets one more
//! line from the exactness analysis (`sky_ir::exact`), for example "over the
//! reals the result is exactly 1/2, and the f64 result is exactly that
//! value", or "exactly 3/10; the f64 result 0.30000000000000004 is that value
//! rounded". It is text only: what is folded in is always the
//! interpreter's number.
//!
//! # Not folded, deliberately
//!
//! * anything the interpreter functions reject (see above);
//! * operations with a non-scalar or non-number typed result (arrays);
//! * `while`, `call`, `print`, and everything with an effect;
//! * `pow` with a non-constant-exponent base that has a unit (the
//!   interpreter fails it), and `matmul`/`solve`/`transpose` (no array
//!   values at run time yet);
//! * reassociation or algebraic identities (`x * 1`, `x + 0`): those are
//!   the e-graph's business and are not always float-exact.

use sky_diag::Span;
use sky_interp::arith::{
    arith, equal, expect_bool, order, pow, range_bound, range_step, reciprocal, sign, ArithOp,
};
use sky_interp::Value;
use sky_ir::exact::{self, Analysis};
use sky_ir::{CmpOp, Function, Module, OpId, OpKind, RangePart, Type, ValueId};
use sky_units::{Dimension, Rational};

use crate::rewrite::Rewrite;
use crate::util::{defining_op, in_fused_body, reachable_ops, replace_uses, value_text};

/// What to do when an exact rational result differs from the interpreter's
/// `f64` result. See the module documentation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FoldMode {
    /// Never change what a program prints.
    #[default]
    BitIdentical,
    /// Prefer exact rational results.
    Exact,
}

/// Folds constants in every function, in [`FoldMode::BitIdentical`].
pub fn fold(m: &mut Module) -> Vec<Rewrite> {
    fold_with(m, FoldMode::BitIdentical)
}

/// Folds constants in every function.
pub fn fold_with(m: &mut Module, mode: FoldMode) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        fold_function(f, mode, &mut out);
    }
    out
}

pub(crate) fn fold_function(f: &mut Function, mode: FoldMode, out: &mut Vec<Rewrite>) {
    // What each value is over the reals, from the operations as written.
    // Folding replaces an operation by a constant for the same value, so
    // the table stays valid for the whole fixed point.
    let facts = exact::analyze(f);
    loop {
        let mut restart = false;
        let mut changed = false;
        for op in reachable_ops(f) {
            if in_fused_body(f, op) {
                continue;
            }
            let did = match f.op(op).kind {
                OpKind::If => {
                    restart = fold_if(f, op, out);
                    restart
                }
                _ => fold_op(f, op, mode, &facts, out),
            };
            changed |= did;
            // Inlining an `if` moved operations, so the list is stale.
            if restart {
                break;
            }
        }
        if !changed {
            return;
        }
    }
}

/// A constant operand.
#[derive(Clone, Debug)]
enum Const {
    Rat(Rational, Dimension),
    Flt(f64, Dimension),
    Bool(bool),
    Str(String),
}

impl Const {
    fn value(&self) -> Value {
        match self {
            Const::Rat(q, d) => Value::number(q.numer() as f64 / q.denom() as f64, *d),
            Const::Flt(x, d) => Value::number(*x, *d),
            Const::Bool(b) => Value::Bool(*b),
            Const::Str(s) => Value::Str(s.clone()),
        }
    }

    /// Source-like text; rationals print exactly (`1/3`), with the unit
    /// suffix the interpreter would print.
    fn text(&self) -> String {
        match self {
            Const::Rat(q, _) => {
                let shown = self.value().to_string();
                match shown.find(' ') {
                    Some(i) => format!("{q}{}", &shown[i..]),
                    None => q.to_string(),
                }
            }
            _ => self.value().to_string(),
        }
    }
}

/// The dimension of a scalar-number-typed value. `Unknown` is
/// dimensionless, as the interpreter treats a constant of unknown type.
fn scalar_dim(f: &Function, v: ValueId) -> Option<Dimension> {
    match f.ty(v) {
        Type::Number(n) if n.shape.rank() == 0 => Some(n.dim),
        Type::Unknown => Some(Dimension::DIMENSIONLESS),
        _ => None,
    }
}

fn const_of(f: &Function, v: ValueId) -> Option<Const> {
    let op = defining_op(f, v)?;
    match &f.op(op).kind {
        OpKind::ConstRational(q) => Some(Const::Rat(*q, scalar_dim(f, v)?)),
        OpKind::ConstFloat { bits } => {
            let x = f64::from_bits(*bits);
            x.is_finite().then_some(Const::Flt(x, scalar_dim(f, v)?))
        }
        OpKind::ConstBool(b) => Some(Const::Bool(*b)),
        OpKind::ConstStr(s) => Some(Const::Str(s.clone())),
        _ => None,
    }
}

/// Runs the operation through the interpreters' own functions. `None`
/// means "would fail at run time" (or is not a foldable kind).
fn eval(kind: &OpKind, args: &[Value], s: Span) -> Option<Value> {
    let arith_op = |op| arith(op, &args[0], &args[1], s, s, s).ok();
    match (kind, args.len()) {
        (OpKind::Add, 2) => arith_op(ArithOp::Add),
        (OpKind::Sub, 2) => arith_op(ArithOp::Sub),
        (OpKind::Mul, 2) => arith_op(ArithOp::Mul),
        (OpKind::Div, 2) => arith_op(ArithOp::Div),
        (OpKind::Neg, 1) => sign(true, args[0].clone(), s).ok(),
        (OpKind::PowConst { exponent }, 1) => {
            let e = Value::scalar(exponent.numer() as f64 / exponent.denom() as f64);
            pow(&args[0], &e, || Some(*exponent), s, s, s).ok()
        }
        (OpKind::Pow, 2) => pow(&args[0], &args[1], || None::<Rational>, s, s, s).ok(),
        (OpKind::Inverse, 1) => reciprocal("⁻¹", &args[0], s, s).ok(),
        // A scalar is its own length, `norm(-3 m)` is `3 m`, through the
        // function `sky run` calls. (Arrays are never constants here, so
        // `dot` never folds.)
        (OpKind::Norm, 1) => sky_interp::vecfn::norm(&args[0], s, s).ok(),
        (OpKind::Cmp(c), 2) => {
            let (l, r) = (&args[0], &args[1]);
            let holds = match c {
                CmpOp::Eq => equal(l, r, s, s, s).ok()?,
                CmpOp::Ne => !equal(l, r, s, s, s).ok()?,
                CmpOp::Lt => order("<", |x, y| x < y, l, r, s, s, s).ok()?,
                CmpOp::Le => order("<=", |x, y| x <= y, l, r, s, s, s).ok()?,
                CmpOp::Gt => order(">", |x, y| x > y, l, r, s, s, s).ok()?,
                CmpOp::Ge => order(">=", |x, y| x >= y, l, r, s, s, s).ok()?,
            };
            Some(Value::Bool(holds))
        }
        (OpKind::ExpectRange { part }, 1) => match part {
            RangePart::Bound => range_bound(&args[0], s).ok(),
            RangePart::Step => range_step(&args[0], s).ok(),
        },
        (OpKind::Not, 1) => expect_bool("not", &args[0], s)
            .ok()
            .map(|b| Value::Bool(!b)),
        _ => None,
    }
}

/// The exact result of `kind` on rationals, if there is one that does not
/// overflow.
fn exact(kind: &OpKind, q: &[Rational]) -> Option<Rational> {
    match (kind, q) {
        (OpKind::Add, [a, b]) => a.checked_add(*b).ok(),
        (OpKind::Sub, [a, b]) => a.checked_sub(*b).ok(),
        (OpKind::Mul, [a, b]) => a.checked_mul(*b).ok(),
        (OpKind::Div, [a, b]) => a.checked_div(*b).ok(),
        (OpKind::Neg, [a]) => a.checked_neg().ok(),
        (OpKind::Inverse, [a]) => a.checked_recip().ok(),
        (OpKind::PowConst { exponent }, [a]) if exponent.is_integer() => {
            let n = exponent.numer();
            if n.unsigned_abs() > 64 {
                return None;
            }
            let mut acc = Rational::ONE;
            for _ in 0..n.unsigned_abs() {
                acc = acc.checked_mul(*a).ok()?;
            }
            if n < 0 {
                acc.checked_recip().ok()
            } else {
                Some(acc)
            }
        }
        _ => None,
    }
}

/// `before` text with the operands spelled out.
fn expr_text(kind: &OpKind, a: &[Const]) -> String {
    let t: Vec<String> = a.iter().map(Const::text).collect();
    let infix = |op: &str| format!("{} {op} {}", t[0], t[1]);
    match kind {
        OpKind::Add => infix("+"),
        OpKind::Sub => infix("-"),
        OpKind::Mul => infix("*"),
        OpKind::Div => infix("/"),
        OpKind::Pow => infix("^"),
        OpKind::PowConst { exponent } => format!("{} ^ {exponent}", t[0]),
        OpKind::Neg => format!("-{}", t[0]),
        OpKind::Inverse => format!("({})⁻¹", t[0]),
        OpKind::Norm => format!("norm({})", t[0]),
        OpKind::Not => format!("not {}", t[0]),
        OpKind::Cmp(c) => infix(match c {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }),
        other => other.mnemonic().to_string(),
    }
}

fn fold_op(
    f: &mut Function,
    id: OpId,
    mode: FoldMode,
    facts: &Analysis,
    out: &mut Vec<Rewrite>,
) -> bool {
    let op = f.op(id).clone();
    let foldable = matches!(
        op.kind,
        OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Neg
            | OpKind::PowConst { .. }
            | OpKind::Pow
            | OpKind::Inverse
            | OpKind::Norm
            | OpKind::Cmp(_)
            | OpKind::Not
            | OpKind::ExpectRange { .. }
    );
    let [result] = op.results[..] else {
        return false;
    };
    if !foldable || op.operands.is_empty() {
        return false;
    }
    let Some(consts) = op
        .operands
        .iter()
        .map(|v| const_of(f, *v))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let values: Vec<Value> = consts.iter().map(Const::value).collect();
    let Some(value) = eval(&op.kind, &values, op.span) else {
        return false;
    };
    let before = expr_text(&op.kind, &consts);
    let (kind, rule, after, because) = match value {
        Value::Bool(b) => {
            // A constant is typed exactly (the verifier says so): a result the
            // checker could not type stays as it is.
            if !matches!(f.ty(result), Type::Bool) {
                return false;
            }
            let rule = if matches!(op.kind, OpKind::Not) {
                "FOLD-NOT-001"
            } else {
                "FOLD-CMP-001"
            };
            let because = vec![
                "every operand is a constant".to_string(),
                "the comparison is computed by the reference interpreter's own rule".to_string(),
            ];
            (OpKind::ConstBool(b), rule, b.to_string(), because)
        }
        Value::Number { value: x, dim } => {
            if !matches!(f.ty(result), Type::Number(_)) || scalar_dim(f, result) != Some(dim) {
                return false;
            }
            let rationals: Option<Vec<Rational>> = consts
                .iter()
                .map(|c| match c {
                    Const::Rat(q, _) => Some(*q),
                    _ => None,
                })
                .collect();
            let exact_result = rationals.and_then(|q| exact(&op.kind, &q));
            let float_bits = x.to_bits();
            match exact_result {
                Some(q)
                    if mode == FoldMode::Exact
                        || (q.numer() as f64 / q.denom() as f64).to_bits() == float_bits =>
                {
                    let shown = Const::Rat(q, dim).text();
                    let mut because = vec![
                        "every operand is an exact rational".to_string(),
                        "Rational arithmetic is checked, and this did not overflow".to_string(),
                    ];
                    if mode == FoldMode::BitIdentical {
                        because.push(
                            "the exact result rounds to the interpreter's f64 result".to_string(),
                        );
                    }
                    (OpKind::ConstRational(q), "FOLD-EXACT-001", shown, because)
                }
                _ => {
                    let shown = Const::Flt(x, dim).text();
                    let mut because = vec![
                        "every operand is a constant".to_string(),
                        "the result is the reference interpreter's own, bit for bit, and it \
                         cannot fail at run time"
                            .to_string(),
                    ];
                    // Say what the value is over the reals, when that is
                    // news: `1/3 + 1/6` is exactly `1/2`.
                    because.extend(facts.known(result).and_then(exact::Known::note));
                    (
                        OpKind::ConstFloat { bits: float_bits },
                        "FOLD-FLOAT-001",
                        shown,
                        because,
                    )
                }
            }
        }
        _ => return false,
    };
    let data = &mut f.ops[id.index()];
    data.kind = kind;
    data.operands.clear();
    // A constant reads nothing, so it has no operand spans either.
    data.operand_spans.clear();
    out.push(Rewrite {
        pass: "fold",
        rule,
        span: op.span,
        before,
        after,
        because,
    });
    true
}

/// Whether the users of an `if`'s results accept the values the taken arm
/// yields in their place. The users were checked against the results' types.
/// Where a yielded value has another type (the results of a variable that
/// only one arm assigns are `unknown`, and the arm that leaves it unassigned
/// yields `nothing`), an operation that needs a number (`tan(t)`, `t > 0`,
/// on a path the return flags keep from ever running, but that the verifier
/// still reads) would see the yielded type: not valid IR. Only users that
/// pass values on to a slot that takes the yielded type (a `yield` to an
/// `if`'s result, a `return`, `print`) are known to be fine.
fn arm_fits(f: &Function, results: &[ValueId], yielded: &[ValueId]) -> bool {
    let mismatched: Vec<(ValueId, ValueId)> = results
        .iter()
        .zip(yielded)
        .filter(|(r, y)| f.ty(**r) != f.ty(**y))
        .map(|(r, y)| (*r, *y))
        .collect();
    if mismatched.is_empty() {
        return true;
    }
    reachable_ops(f).into_iter().all(|o| {
        let op = f.op(o);
        op.operands.iter().enumerate().all(|(i, v)| {
            let Some((_, y)) = mismatched.iter().find(|(r, _)| r == v) else {
                return true;
            };
            let target = match op.kind {
                OpKind::Print => return true,
                OpKind::Return => f.results.get(i),
                OpKind::Yield => f
                    .block(op.parent)
                    .parent
                    .and_then(|r| f.region(r).parent)
                    .map(|owner| f.op(owner))
                    .filter(|owner| owner.kind == OpKind::If)
                    .and_then(|owner| owner.results.get(i))
                    .map(|res| f.ty(*res)),
                _ => None,
            };
            target.is_some_and(|t| sky_ir::assignable(f.ty(*y), t))
        })
    })
}

/// `if` on a constant condition: inline the taken region in place of the
/// `if`. Returns whether it did.
fn fold_if(f: &mut Function, id: OpId, out: &mut Vec<Rewrite>) -> bool {
    let op = f.op(id).clone();
    let (&[cond], &[then_r, else_r]) = (&op.operands[..], &op.regions[..]) else {
        return false;
    };
    let Some(Const::Bool(taken)) = const_of(f, cond) else {
        return false;
    };
    let region = if taken { then_r } else { else_r };
    let &[block] = &f.region(region).blocks[..] else {
        return false;
    };
    let body = f.block(block).ops.clone();
    let Some((&last, inner)) = body.split_last() else {
        return false;
    };
    let term = f.op(last);
    if !matches!(term.kind, OpKind::Yield)
        || term.operands.len() != op.results.len()
        || !f.block(block).args.is_empty()
    {
        return false;
    }
    let yielded = term.operands.clone();
    if !arm_fits(f, &op.results, &yielded) {
        return false;
    }

    // Splice the region's operations in where the `if` was.
    let parent = op.parent;
    let pos = f
        .block(parent)
        .ops
        .iter()
        .position(|o| *o == id)
        .expect("an if is in its parent block");
    for &o in inner {
        f.ops[o.index()].parent = parent;
    }
    f.blocks[parent.index()]
        .ops
        .splice(pos..=pos, inner.iter().copied());
    for (r, y) in op.results.iter().zip(&yielded) {
        replace_uses(f, *r, *y);
        // Keep the user's name for the value that now stands for it.
        if f.values[y.index()].name.is_none() {
            f.values[y.index()].name = f.values[r.index()].name.clone();
        }
    }
    let ifd = &mut f.ops[id.index()];
    ifd.regions.clear();
    ifd.operands.clear();
    ifd.operand_spans.clear();

    let branch = if taken { "then" } else { "else" };
    let other = if taken { "else" } else { "then" };
    out.push(Rewrite {
        pass: "fold",
        rule: "FOLD-IF-001",
        span: op.span,
        before: format!("if {} {{ .. }} else {{ .. }}", taken),
        after: format!("the `{branch}` branch, inlined"),
        because: vec![
            format!(
                "the condition {} is the constant {taken}",
                value_text(f, cond)
            ),
            format!("the `{other}` branch can never run"),
        ],
    });
    true
}
