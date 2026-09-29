//! The evaluator: a value table per activation, ops in order, regions by
//! recursion.
//!
//! The `match` over [`OpKind`] is exhaustive on purpose (no `_ =>` arm), so
//! an operation added to the IR fails to compile here instead of being
//! silently ignored.
//!
//! Every access to IR data goes through a checked accessor (`Act::op`,
//! `Act::get`, `fixed`, ...) that reports `malformed IR` instead of
//! indexing, so no input can make this module panic.

use std::io::Write;

use sky_diag::Span;
use sky_interp::arith::{
    arith, arity_mismatch, equal, expect_bool, order, pow, reciprocal, recursion_too_deep, sign,
    ArithOp, StackMeter, MAX_CALL_DEPTH,
};
use sky_interp::array::{self, Ix};
use sky_interp::Value;
use sky_ir::{
    BlockData, BlockId, CmpOp, Function, Module, NumberType, OpData, OpId, OpKind, RegionId,
    SliceAxis, SolveMethod, Type, ValueId,
};
use sky_units::{Dimension, Rational};

use crate::IrRuntimeError;

type R<T> = Result<T, IrRuntimeError>;

/// How a block finished: by the terminator it ended in, with its operands'
/// values.
enum Exit {
    Yield(Vec<Value>),
    Condition(Vec<Value>),
    Return(Vec<Value>),
}

/// One function activation: the function and its value table, indexed by
/// [`ValueId`]. `None` is a value not defined (yet), which reading reports
/// as malformed IR. SSA scoping is not enforced here (a value defined in a
/// region stays readable afterwards); that is the verifier's job.
struct Act<'m> {
    func: &'m Function,
    vals: Vec<Option<Value>>,
}

impl<'m> Act<'m> {
    fn new(func: &'m Function) -> Self {
        Act {
            func,
            vals: vec![None; func.values.len()],
        }
    }

    fn get(&self, v: ValueId, span: Span) -> R<&Value> {
        self.vals
            .get(v.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                IrRuntimeError::malformed(format!("{v} is used before it is defined"), span)
            })
    }

    fn set(&mut self, v: ValueId, value: Value, span: Span) -> R<()> {
        match self.vals.get_mut(v.index()) {
            Some(slot) => {
                *slot = Some(value);
                Ok(())
            }
            None => Err(IrRuntimeError::malformed(
                format!("{v} is not a value of `{}`", self.func.name),
                span,
            )),
        }
    }

    fn values(&self, vs: &[ValueId], span: Span) -> R<Vec<Value>> {
        vs.iter().map(|v| self.get(*v, span).cloned()).collect()
    }

    /// Binds `op`'s results to `values`, which must match in number.
    fn define(&mut self, op: &OpData, values: Vec<Value>) -> R<()> {
        if op.results.len() != values.len() {
            return Err(IrRuntimeError::malformed(
                format!(
                    "`{}` has {} results, expected {}",
                    op.kind.mnemonic(),
                    op.results.len(),
                    values.len()
                ),
                op.span,
            ));
        }
        for (r, v) in op.results.iter().zip(values) {
            self.set(*r, v, op.span)?;
        }
        Ok(())
    }

    fn op(&self, id: OpId, span: Span) -> R<&'m OpData> {
        self.func
            .ops
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no operation {id}"), span))
    }

    fn block(&self, id: BlockId, span: Span) -> R<&'m BlockData> {
        self.func
            .blocks
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no block {id}"), span))
    }

    /// The single block of a region (v1 regions have exactly one).
    fn entry_of(&self, id: RegionId, span: Span) -> R<BlockId> {
        let region = self
            .func
            .regions
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no region {id}"), span))?;
        match region.blocks[..] {
            [b] => Ok(b),
            [] => Err(IrRuntimeError::malformed(
                format!("{id} has no block"),
                span,
            )),
            _ => Err(IrRuntimeError::malformed(
                format!("{id} has several blocks, but v1 regions have one"),
                span,
            )),
        }
    }

    /// The type of `v`, if `v` exists.
    fn ty(&self, v: ValueId) -> Option<&'m Type> {
        self.func.values.get(v.index()).map(|d| &d.ty)
    }
}

/// `op`'s operands as exactly `N` values' ids.
fn fixed<const N: usize>(op: &OpData) -> R<[ValueId; N]> {
    op.operands.as_slice().try_into().map_err(|_| {
        IrRuntimeError::malformed(
            format!(
                "`{}` takes {N} operands, found {}",
                op.kind.mnemonic(),
                op.operands.len()
            ),
            op.span,
        )
    })
}

/// `op`'s regions as exactly `N` ids.
fn regions<const N: usize>(op: &OpData) -> R<[RegionId; N]> {
    op.regions.as_slice().try_into().map_err(|_| {
        IrRuntimeError::malformed(
            format!(
                "`{}` takes {N} regions, found {}",
                op.kind.mnemonic(),
                op.regions.len()
            ),
            op.span,
        )
    })
}

/// The dimension of a constant, from its result's type. An `Unknown`
/// result is dimensionless (a constant's value carries no other unit).
fn const_dim(act: &Act<'_>, op: &OpData) -> R<Dimension> {
    let [r] = op.results[..] else {
        return Err(IrRuntimeError::malformed(
            format!(
                "`{}` has {} results, expected 1",
                op.kind.mnemonic(),
                op.results.len()
            ),
            op.span,
        ));
    };
    match act.ty(r) {
        Some(Type::Number(NumberType { dim, shape, .. })) if shape.rank() == 0 => Ok(*dim),
        Some(Type::Number(_)) => Err(IrRuntimeError::malformed(
            format!("`{}` must have a scalar result", op.kind.mnemonic()),
            op.span,
        )),
        Some(Type::Unknown) => Ok(Dimension::DIMENSIONLESS),
        _ => Err(IrRuntimeError::malformed(
            format!("`{}` must have a number result", op.kind.mnemonic()),
            op.span,
        )),
    }
}

struct Machine<'m, 'w> {
    module: &'m Module,
    out: &'w mut dyn Write,
    /// User-function calls active now; see [`MAX_CALL_DEPTH`].
    depth: usize,
    stack: StackMeter,
    /// Ops left to run, in tests only: random malformed IR (a `while` whose
    /// condition is always true) must not hang the suite. Production has no
    /// step limit.
    #[cfg(test)]
    fuel: u64,
}

/// Calls `name` in `module` with `args`.
pub(crate) fn call(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
) -> R<Vec<Value>> {
    call_metered(module, name, args, out, StackMeter::new())
}

/// [`call`] measuring the stack from `stack`, a meter another interpreter
/// started, so a run nested in it stays inside one budget.
pub(crate) fn call_metered(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    stack: StackMeter,
) -> R<Vec<Value>> {
    let mut m = Machine {
        module,
        out,
        depth: 0,
        stack,
        #[cfg(test)]
        fuel: u64::MAX,
    };
    m.entry(name, args, Span::point(0))
}

/// [`call`] with a bounded number of ops, for fuzzing.
#[cfg(test)]
pub(crate) fn call_with_fuel(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    fuel: u64,
) -> R<Vec<Value>> {
    let mut m = Machine {
        module,
        out,
        depth: 0,
        stack: StackMeter::new(),
        fuel,
    };
    m.entry(name, args, Span::point(0))
}

impl<'m> Machine<'m, '_> {
    /// Looks `name` up for a call with `argc` arguments made at `span`.
    fn resolve(&self, name: &str, argc: usize, span: Span) -> R<&'m Function> {
        let module = self.module;
        let Some(func) = module.function(name) else {
            return Err(IrRuntimeError::malformed(
                format!("call to unknown function `{name}`"),
                span,
            ));
        };
        if argc != func.params.len() {
            return Err(arity_mismatch(name, func.params.len(), argc, span).into());
        }
        Ok(func)
    }

    /// The entry call of a run. Like `sky run`'s top level, it is not a
    /// nested call and does not count against [`MAX_CALL_DEPTH`].
    fn entry(&mut self, name: &str, args: Vec<Value>, span: Span) -> R<Vec<Value>> {
        let func = self.resolve(name, args.len(), span)?;
        self.activate(func, args)
    }

    /// A call made by a `call` op.
    fn call_named(&mut self, name: &str, args: Vec<Value>, span: Span) -> R<Vec<Value>> {
        let func = self.resolve(name, args.len(), span)?;
        if self.depth >= MAX_CALL_DEPTH {
            return Err(recursion_too_deep(name, span).into());
        }
        self.depth += 1;
        let result = self.activate(func, args);
        self.depth -= 1;
        result
    }

    /// Runs `func`'s body in a new activation.
    fn activate(&mut self, func: &Function, args: Vec<Value>) -> R<Vec<Value>> {
        let mut act = Act::new(func);
        let entry = act.entry_of(func.body, func.span)?;
        match self.block(&mut act, entry, args, func.span)? {
            Exit::Return(values) if values.len() == func.results.len() => Ok(values),
            Exit::Return(values) => Err(IrRuntimeError::malformed(
                format!(
                    "`{}` returns {} values, its signature has {}",
                    func.name,
                    values.len(),
                    func.results.len()
                ),
                func.span,
            )),
            Exit::Yield(_) | Exit::Condition(_) => Err(IrRuntimeError::malformed(
                format!("the body of `{}` must end in `return`", func.name),
                func.span,
            )),
        }
    }

    /// Runs a block: binds its arguments, runs its ops in order, and
    /// returns how it ended.
    fn block(&mut self, act: &mut Act<'_>, id: BlockId, args: Vec<Value>, span: Span) -> R<Exit> {
        self.stack.check(span)?;
        let block = act.block(id, span)?;
        if block.args.len() != args.len() {
            return Err(IrRuntimeError::malformed(
                format!(
                    "{id} takes {} arguments, given {}",
                    block.args.len(),
                    args.len()
                ),
                span,
            ));
        }
        for (a, v) in block.args.iter().zip(args) {
            act.set(*a, v, span)?;
        }
        for op_id in &block.ops {
            let op = act.op(*op_id, span)?;
            #[cfg(test)]
            {
                if self.fuel == 0 {
                    return Err(IrRuntimeError::malformed("out of fuel (test)", op.span));
                }
                self.fuel -= 1;
            }
            match op.kind {
                OpKind::Yield => return Ok(Exit::Yield(act.values(&op.operands, op.span)?)),
                OpKind::Condition => {
                    return Ok(Exit::Condition(act.values(&op.operands, op.span)?))
                }
                OpKind::Return => return Ok(Exit::Return(act.values(&op.operands, op.span)?)),
                _ => self.op(act, op)?,
            }
        }
        Err(IrRuntimeError::malformed(
            format!("{id} does not end in a terminator"),
            span,
        ))
    }

    /// Runs the (single) block of `region` with `args`.
    fn region(&mut self, act: &mut Act<'_>, id: RegionId, args: Vec<Value>, span: Span) -> R<Exit> {
        let entry = act.entry_of(id, span)?;
        self.block(act, entry, args, span)
    }

    /// Runs one operation that is not a terminator. Only operations that
    /// can recurse are handled here: `pure_op` has a large frame (one arm
    /// per operation), and keeping it off the recursion path is what lets
    /// [`MAX_CALL_DEPTH`] nested calls fit the stack budget in a debug
    /// build.
    fn op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        match &op.kind {
            OpKind::If => self.if_op(act, op),
            OpKind::While => self.while_op(act, op),
            OpKind::Call { callee } => {
                let args = act.values(&op.operands, op.span)?;
                let results = self.call_named(callee, args, op.span)?;
                act.define(op, results)
            }
            OpKind::Print => self.print(act, op),
            _ => pure_op(act, op),
        }
    }

    /// `if`: run the arm the condition selects; its `yield` gives the
    /// results.
    fn if_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let [c] = fixed(op)?;
        let [then, els] = regions(op)?;
        let arm = if expect_bool("if", act.get(c, s)?, s)? {
            then
        } else {
            els
        };
        match self.region(act, arm, Vec::new(), s)? {
            Exit::Yield(values) => act.define(op, values),
            Exit::Condition(_) | Exit::Return(_) => Err(IrRuntimeError::malformed(
                "an `if` region must end in `yield`",
                s,
            )),
        }
    }

    /// `while` (see `OpKind::While`): `before` computes the condition and
    /// the forwarded values; if true, `after` computes the next loop-carried
    /// values; if false, the forwarded values are the results.
    fn while_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let [before, after] = regions(op)?;
        let mut carried = act.values(&op.operands, s)?;
        loop {
            let Exit::Condition(mut forwarded) = self.region(act, before, carried, s)? else {
                return Err(IrRuntimeError::malformed(
                    "a `while` `before` region must end in `condition`",
                    s,
                ));
            };
            if forwarded.is_empty() {
                return Err(IrRuntimeError::malformed(
                    "`condition` needs a Bool operand",
                    s,
                ));
            }
            let flag = forwarded.remove(0);
            if !expect_bool("while", &flag, s)? {
                return act.define(op, forwarded);
            }
            match self.region(act, after, forwarded, s)? {
                Exit::Yield(next) => carried = next,
                Exit::Condition(_) | Exit::Return(_) => {
                    return Err(IrRuntimeError::malformed(
                        "a `while` `after` region must end in `yield`",
                        s,
                    ));
                }
            }
        }
    }

    /// `print(a, b, ...)`: the values separated by single spaces, then a
    /// newline, written in one piece as `sky run` does.
    fn print(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let mut line = String::new();
        for (i, v) in op.operands.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(&act.get(*v, s)?.print_text());
        }
        line.push('\n');
        self.out
            .write_all(line.as_bytes())
            .map_err(|e| sky_interp::RuntimeError::new(format!("cannot write output: {e}"), s))?;
        act.define(op, Vec::new())
    }
}

/// Runs an operation that never recurses into regions or calls.
fn pure_op(act: &mut Act<'_>, op: &OpData) -> R<()> {
    let s = op.span;
    match &op.kind {
        OpKind::ConstFloat { bits } => {
            let x = f64::from_bits(*bits);
            if !x.is_finite() {
                return Err(IrRuntimeError::malformed("a constant is not finite", s));
            }
            let dim = const_dim(act, op)?;
            act.define(op, vec![Value::number(x, dim)])
        }
        OpKind::ConstRational(q) => {
            let x = q.numer() as f64 / q.denom() as f64;
            if !x.is_finite() {
                return Err(IrRuntimeError::malformed(
                    "a rational constant is not finite",
                    s,
                ));
            }
            let dim = const_dim(act, op)?;
            act.define(op, vec![Value::number(x, dim)])
        }
        OpKind::ConstBool(b) => act.define(op, vec![Value::Bool(*b)]),
        OpKind::ConstStr(text) => act.define(op, vec![Value::Str(text.clone())]),
        OpKind::ConstNothing => act.define(op, vec![Value::None]),

        OpKind::Add => arith_op(act, op, ArithOp::Add),
        OpKind::Sub => arith_op(act, op, ArithOp::Sub),
        OpKind::Mul => arith_op(act, op, ArithOp::Mul),
        OpKind::Div => arith_op(act, op, ArithOp::Div),
        OpKind::Neg => {
            let [a] = fixed(op)?;
            let v = sign(true, act.get(a, s)?.clone(), s)?;
            act.define(op, vec![v])
        }
        OpKind::Math(f) => {
            let vals = act.values(&op.operands, s)?;
            if vals.len() != f.arity() {
                return Err(IrRuntimeError::malformed(
                    format!(
                        "`{}` takes {} operands, found {}",
                        f.mnemonic(),
                        f.arity(),
                        vals.len()
                    ),
                    s,
                ));
            }
            let refs: Vec<&Value> = vals.iter().collect();
            let v = crate::math::eval(*f, &refs, s)?;
            act.define(op, vec![v])
        }
        // The same function `sky run` calls for `v as unit`, so both print
        // the same digits and fail the same way.
        OpKind::Show { unit } => {
            let [a] = fixed(op)?;
            let v = sky_interp::show_unit_text(unit, act.get(a, s)?, s)?;
            act.define(op, vec![v])
        }
        OpKind::PowConst { exponent } => {
            let [a] = fixed(op)?;
            let e = Value::scalar(exponent.numer() as f64 / exponent.denom() as f64);
            let v = pow(act.get(a, s)?, &e, || Some(*exponent), s, s, s)?;
            act.define(op, vec![v])
        }
        OpKind::Pow => {
            let [a, e] = fixed(op)?;
            // A computed exponent gives a base with a unit no constant to
            // take the result's dimension from, as in `sky run`.
            let v = pow(act.get(a, s)?, act.get(e, s)?, || None::<Rational>, s, s, s)?;
            act.define(op, vec![v])
        }
        // The IR does not remember whether the source said `x⁻¹` or
        // `inv(x)`, so the (unreachable in checked programs) error for a
        // non-number always uses the `⁻¹` spelling.
        OpKind::Inverse => {
            let [a] = fixed(op)?;
            let v = reciprocal("⁻¹", act.get(a, s)?, s, s)?;
            act.define(op, vec![v])
        }
        OpKind::MatMul => {
            let [a, b] = fixed(op)?;
            let v = array::matmul(act.get(a, s)?, act.get(b, s)?, s, s, s)?;
            act.define(op, vec![v])
        }
        OpKind::Transpose => {
            let [a] = fixed(op)?;
            let v = array::transpose(act.get(a, s)?, s)?;
            act.define(op, vec![v])
        }
        // The Cholesky solve `sky run` performs where the checker rewrote
        // `A⁻¹ @ b`: the same function, so the same bits and errors.
        OpKind::Solve {
            method: SolveMethod::Cholesky,
        } => {
            let [a, b] = fixed(op)?;
            let v = array::cholesky_solve(act.get(a, s)?, act.get(b, s)?, s, s, s)?;
            act.define(op, vec![v])
        }
        OpKind::ArrayLiteral { cols } => array_literal(act, op, *cols),
        OpKind::Index => {
            let Some((&base, positions)) = op.operands.split_first() else {
                return Err(IrRuntimeError::malformed("`index` has no operands", s));
            };
            let mut items = Vec::with_capacity(positions.len());
            for &p in positions {
                items.push(Ix::At {
                    i: array::to_index(act.get(p, s)?, s)?,
                    span: s,
                });
            }
            let v = array::index(act.get(base, s)?, &items, s, s)?;
            act.define(op, vec![v])
        }
        OpKind::Slice { axes } => slice(act, op, axes),
        OpKind::Zeros | OpKind::Identity => {
            let mut sizes = Vec::with_capacity(2);
            let name = op.kind.mnemonic();
            for &v in &op.operands {
                sizes.push(array::size_arg(name, act.get(v, s)?, s)?);
            }
            let v = match (&op.kind, sizes.as_slice()) {
                (OpKind::Zeros, [_] | [_, _]) => array::zeros(&sizes, s)?,
                (OpKind::Identity, [n]) => array::identity(*n, s)?,
                _ => {
                    return Err(IrRuntimeError::malformed(
                        format!(
                            "`{name}` takes a size (`zeros`: two), found {}",
                            sizes.len()
                        ),
                        s,
                    ))
                }
            };
            act.define(op, vec![v])
        }
        OpKind::Len => {
            let [a] = fixed(op)?;
            let v = array::len(act.get(a, s)?, s)?;
            act.define(op, vec![v])
        }

        OpKind::Cmp(c) => {
            let [a, b] = fixed(op)?;
            let (l, r) = (act.get(a, s)?, act.get(b, s)?);
            let holds = match c {
                CmpOp::Eq => equal(l, r, s, s, s)?,
                CmpOp::Ne => !equal(l, r, s, s, s)?,
                CmpOp::Lt => order("<", |x, y| x < y, l, r, s, s, s)?,
                CmpOp::Le => order("<=", |x, y| x <= y, l, r, s, s, s)?,
                CmpOp::Gt => order(">", |x, y| x > y, l, r, s, s, s)?,
                CmpOp::Ge => order(">=", |x, y| x >= y, l, r, s, s, s)?,
            };
            act.define(op, vec![Value::Bool(holds)])
        }
        OpKind::Not => {
            let [a] = fixed(op)?;
            let b = expect_bool("not", act.get(a, s)?, s)?;
            act.define(op, vec![Value::Bool(!b)])
        }
        OpKind::Input { name } => Err(sky_interp::arith::declared_without_value(name, s).into()),
        OpKind::ExpectBool { context } => {
            let [a] = fixed(op)?;
            let b = expect_bool(context.keyword(), act.get(a, s)?, s)?;
            act.define(op, vec![Value::Bool(b)])
        }

        // Handled by `Machine::op` and `Machine::block`, which never send
        // them here; the arm keeps this `match` exhaustive.
        OpKind::If
        | OpKind::While
        | OpKind::Call { .. }
        | OpKind::Print
        | OpKind::Yield
        | OpKind::Condition
        | OpKind::Return => Err(IrRuntimeError::malformed(
            format!("unexpected `{}`", op.kind.mnemonic()),
            s,
        )),
    }
}

fn arith_op(act: &mut Act<'_>, op: &OpData, kind: ArithOp) -> R<()> {
    let [a, b] = fixed(op)?;
    let s = op.span;
    let v = arith(kind, act.get(a, s)?, act.get(b, s)?, s, s, s)?;
    act.define(op, vec![v])
}

/// `array.literal`: the operands, in row-major order, arranged as `cols`
/// says (see [`OpKind::ArrayLiteral`]) and built by `sky_interp`'s literal
/// rule, which checks that every entry is a number of one dimension.
fn array_literal(act: &mut Act<'_>, op: &OpData, cols: Option<u64>) -> R<()> {
    let s = op.span;
    let entries: Vec<(Value, Span)> = act
        .values(&op.operands, s)?
        .into_iter()
        .map(|v| (v, s))
        .collect();
    let width = match cols {
        None => entries.len(),
        Some(c) => usize::try_from(c).unwrap_or(0),
    };
    if width == 0 || entries.len() % width != 0 {
        return Err(IrRuntimeError::malformed(
            format!(
                "`array.literal` cannot arrange {} operands in rows of {width}",
                entries.len()
            ),
            s,
        ));
    }
    let rows: Vec<Vec<(Value, Span)>> = entries.chunks(width).map(<[_]>::to_vec).collect();
    let v = array::literal(rows, cols.is_none(), s)?;
    act.define(op, vec![v])
}

/// `slice`: the base, then each item's operands in order.
fn slice(act: &mut Act<'_>, op: &OpData, axes: &[SliceAxis]) -> R<()> {
    let s = op.span;
    let Some((&base, rest)) = op.operands.split_first() else {
        return Err(IrRuntimeError::malformed("`slice` has no operands", s));
    };
    let mut rest = rest.iter();
    let mut bound = |act: &Act<'_>| -> R<usize> {
        let Some(&v) = rest.next() else {
            return Err(IrRuntimeError::malformed(
                "`slice` has fewer operands than its items need",
                s,
            ));
        };
        Ok(array::to_index(act.get(v, s)?, s)?)
    };
    let mut items = Vec::with_capacity(axes.len());
    for axis in axes {
        items.push(match *axis {
            SliceAxis::At => Ix::At {
                i: bound(act)?,
                span: s,
            },
            SliceAxis::Range { start, stop } => {
                let start = if start { Some(bound(act)?) } else { None };
                let stop = if stop { Some(bound(act)?) } else { None };
                Ix::Slice {
                    start,
                    stop,
                    span: s,
                }
            }
        });
    }
    if rest.next().is_some() {
        return Err(IrRuntimeError::malformed(
            "`slice` has more operands than its items need",
            s,
        ));
    }
    let v = array::index(act.get(base, s)?, &items, s, s)?;
    act.define(op, vec![v])
}
