//! The cost of each operation, of loops and branches, and of calls.

use std::collections::{BTreeMap, HashMap};

use sky_diag::Span;
use sky_types::poly::Poly;
use sky_types::{Shape, ShapeDim};
use sky_units::Rational;

use super::amount::{Amount, At, Reason, ReasonKind, Trips};
use super::{Cost, FunctionCost, Site};
use crate::ir::{Function, OpData, OpId, RegionId, Type, ValueDef, ValueId};
use crate::op::{CmpOp, MathFn, OpKind, Side, SolveMethod};

/// Bytes per array entry: every entry is an `f64`.
const ENTRY_BYTES: i64 = 8;
/// The weight of a math function that is not one of the cheap ones.
const TRANSCENDENTAL: i64 = 20;
/// The most sites a function's cost keeps.
const MAX_SITES: usize = 2048;
/// How deep a size expression is followed.
const MAX_SIZE_DEPTH: u32 = 64;

/// Why an operation's cost is not a polynomial.
enum Fail {
    /// The shape of an operand is unknown: nothing about the cost is.
    Shape,
    /// The size of the array made is unknown: memory is, flops are not.
    Memory,
    TooLarge,
}

/// The parts of one operation's cost.
#[derive(Default)]
struct Parts {
    flops: Poly,
    read: Poly,
    written: Poly,
    allocs: Poly,
    alloc_bytes: Poly,
}

type P<T> = Result<T, Fail>;

fn mul(a: &Poly, b: &Poly) -> P<Poly> {
    a.checked_mul(b).ok_or(Fail::TooLarge)
}

fn add(a: &Poly, b: &Poly) -> P<Poly> {
    a.checked_add(b).ok_or(Fail::TooLarge)
}

fn scale(a: &Poly, k: i64) -> P<Poly> {
    a.checked_scale(Rational::integer(k)).ok_or(Fail::TooLarge)
}

fn frac(a: &Poly, num: i64, den: i64) -> P<Poly> {
    Rational::checked_new(num, den)
        .ok()
        .and_then(|r| a.checked_scale(r))
        .ok_or(Fail::TooLarge)
}

/// The number of entries a dimension list holds.
fn product(dims: &[Poly]) -> P<Poly> {
    dims.iter()
        .try_fold(Poly::integer(1), |acc, d| mul(&acc, d))
}

fn dim_poly(d: &ShapeDim) -> Poly {
    match d {
        ShapeDim::Known(n) => i64::try_from(*n).map_or_else(|_| Poly::zero(), Poly::integer),
        ShapeDim::Var(v) => Poly::var(v.name()),
    }
}

fn shape_dims(s: &Shape) -> Vec<Poly> {
    s.dims().iter().map(dim_poly).collect()
}

/// The weight of one scalar operation, in flops. `None` for an operation
/// that is not floating-point arithmetic.
pub fn flop_weight(kind: &OpKind) -> Option<i64> {
    Some(match kind {
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Neg => 1,
        OpKind::PowConst { exponent } => pow_const_weight(*exponent),
        OpKind::Pow => TRANSCENDENTAL,
        OpKind::Math(f) => math_weight(*f),
        _ => return None,
    })
}

fn math_weight(f: MathFn) -> i64 {
    match f {
        MathFn::Sqrt
        | MathFn::Abs
        | MathFn::Min
        | MathFn::Max
        | MathFn::Floor
        | MathFn::Ceil
        | MathFn::Round
        | MathFn::Sign => 1,
        _ => TRANSCENDENTAL,
    }
}

/// Square-and-multiply for a whole exponent, `sqrt` for one half, a library
/// call for anything else.
fn pow_const_weight(e: Rational) -> i64 {
    if e.is_integer() {
        let n = e.numer().unsigned_abs();
        let mults = match n {
            0 => 0,
            _ => i64::from(n.ilog2()) + i64::from(n.count_ones()) - 1,
        };
        mults + i64::from(e.numer() < 0)
    } else if e == Rational::new(1, 2) {
        1
    } else {
        TRANSCENDENTAL
    }
}

/// What the analysis of one function reads.
pub struct Cx<'a> {
    pub module: &'a crate::ir::Module,
    pub f: &'a Function,
    pub done: &'a HashMap<String, FunctionCost>,
    pub recursive: &'a dyn Fn(&str, &str) -> bool,
}

impl Cx<'_> {
    fn at(&self, span: Span) -> At<'_> {
        At {
            function: &self.f.name,
            span,
        }
    }

    pub fn function(&self) -> FunctionCost {
        let (cost, mut sites) = self.region(self.f.body);
        if sites.len() > MAX_SITES {
            // Keep the costliest, in source order: a huge function does not
            // make the report cost memory in proportion.
            let weight = |s: &Site| {
                s.cost.flops.hi.weight_at(1000.0)
                    + s.cost.bytes_read.hi.weight_at(1000.0)
                    + s.cost.bytes_written.hi.weight_at(1000.0)
            };
            sites.sort_by(|a, b| weight(b).total_cmp(&weight(a)));
            sites.truncate(MAX_SITES);
            sites.sort_by_key(|s| s.span.start);
        }
        FunctionCost {
            name: self.f.name.clone(),
            span: self.f.span,
            cost,
            sites,
        }
    }

    fn ops(&self, r: RegionId) -> &[crate::ir::OpId] {
        self.f
            .regions
            .get(r.index())
            .and_then(|region| region.blocks.first())
            .and_then(|b| self.f.blocks.get(b.index()))
            .map_or(&[], |b| b.ops.as_slice())
    }

    /// The cost of the operations of a region, and its costly operations.
    fn region(&self, r: RegionId) -> (Cost, Vec<Site>) {
        self.region_skipping(r, None)
    }

    /// [`Self::region`] without the operation `skip` (the step of a counted
    /// loop's index, which is bookkeeping and not the program's arithmetic).
    fn region_skipping(&self, r: RegionId, skip: Option<OpId>) -> (Cost, Vec<Site>) {
        let mut total = Cost::zero();
        let mut sites = Vec::new();
        for &o in self.ops(r) {
            if Some(o) == skip {
                continue;
            }
            let op = self.f.op(o);
            let at = self.at(op.span);
            let (cost, inner) = match &op.kind {
                OpKind::If => self.branch(op),
                OpKind::While => self.loop_(op),
                OpKind::Call { callee } => (self.call(op, callee), Vec::new()),
                _ => (self.simple(op), Vec::new()),
            };
            if cost.is_zero() && inner.is_empty() {
                continue;
            }
            if inner.is_empty() {
                sites.push(Site {
                    span: op.span,
                    what: op.kind.mnemonic().to_string(),
                    cost: cost.clone(),
                });
            } else {
                sites.extend(inner);
            }
            total = total.add(&cost, at);
        }
        (total, sites)
    }

    /// An `if`: the branch that runs.
    fn branch(&self, op: &OpData) -> (Cost, Vec<Site>) {
        let (Some(&then), Some(&otherwise)) = (op.regions.first(), op.regions.get(1)) else {
            return (Cost::zero(), Vec::new());
        };
        let (a, mut sites) = self.region(then);
        let (b, more) = self.region(otherwise);
        sites.extend(more);
        (a.join(&b), sites)
    }

    /// A `while`: the condition once more than the body.
    fn loop_(&self, op: &OpData) -> (Cost, Vec<Site>) {
        let (Some(&before), Some(&after)) = (op.regions.first(), op.regions.get(1)) else {
            return (Cost::zero(), Vec::new());
        };
        let at = self.at(op.span);
        let counted = self.trips(op);
        let (head, head_sites) = self.region(before);
        let (body, body_sites) = self.region_skipping(after, counted.as_ref().map(|c| c.1));
        let per_iteration = head.add(&body, at);
        let mut sites = head_sites;
        sites.extend(body_sites);
        match counted.map(|c| c.0) {
            Some(trips) => {
                let total = per_iteration.times(&trips, at).add(&head, at);
                for s in &mut sites {
                    s.cost = s.cost.times(&trips, at);
                }
                (total, sites)
            }
            None => {
                let reason = at.reason(ReasonKind::Loop);
                let total = per_iteration.times_unbounded(&reason).add(&head, at);
                for s in &mut sites {
                    s.cost = s.cost.times_unbounded(&reason);
                }
                (total, sites)
            }
        }
    }

    // ----- trip counts -----

    /// A size the model can name: an integer constant, the length of an
    /// array (a size of its type), and sums, differences and products of
    /// those.
    fn size(&self, v: ValueId, depth: u32) -> Option<Poly> {
        if depth > MAX_SIZE_DEPTH {
            return None;
        }
        let ValueDef::OpResult { op, .. } = self.f.value(v).def else {
            return None;
        };
        let op = self.f.op(op);
        let go = |i: usize| self.size(*op.operands.get(i)?, depth + 1);
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                let x = f64::from_bits(*bits);
                (x.fract() == 0.0 && x.abs() < 9e15).then(|| Poly::integer(x as i64))
            }
            OpKind::ConstRational(r) => Some(Poly::constant(*r)),
            OpKind::Len => {
                let Type::Number(n) = self.f.ty(*op.operands.first()?) else {
                    return None;
                };
                n.shape.dims().first().map(dim_poly)
            }
            // A run-time check of a `range` bound gives its operand back.
            OpKind::ExpectRange { .. } => go(0),
            OpKind::Add => go(0)?.checked_add(&go(1)?),
            OpKind::Sub => go(0)?.checked_sub(&go(1)?),
            OpKind::Mul => go(0)?.checked_mul(&go(1)?),
            OpKind::Neg => go(0)?.checked_neg(),
            _ => None,
        }
    }

    /// How often a `while` runs its body, when it is a counted loop: the
    /// condition compares a carried index with a bound the model can name,
    /// and the body advances the index by a constant.
    fn trips(&self, w: &OpData) -> Option<(Trips, OpId)> {
        let before = self
            .f
            .block(*self.f.region(*w.regions.first()?).blocks.first()?);
        let after = self
            .f
            .block(*self.f.region(*w.regions.get(1)?).blocks.first()?);
        let condition = self.f.op(*before.ops.last()?);
        if condition.kind != OpKind::Condition {
            return None;
        }
        // The index the condition tests is forwarded to the body unchanged.
        let (cmp, early_exit) = self.counted_comparison(*condition.operands.first()?)?;
        let OpKind::Cmp(rel @ (CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge)) = cmp.kind else {
            return None;
        };
        let index_arg = *cmp.operands.first()?;
        let j = before.args.iter().position(|a| *a == index_arg)?;
        if condition.operands.get(1 + j) != before.args.get(j) {
            return None;
        }
        let stop = self.size(*cmp.operands.get(1)?, 0)?;
        let start = self.size(*w.operands.get(j)?, 0)?;
        // The body's last operation yields the next index: `i + c` or `i - c`.
        let next = self.f.op(*after.ops.last()?);
        if next.kind != OpKind::Yield {
            return None;
        }
        let ValueDef::OpResult { op: step_op, .. } = self.f.value(*next.operands.get(j)?).def
        else {
            return None;
        };
        let step_id = step_op;
        let step_op = self.f.op(step_op);
        let (a, b) = (*step_op.operands.first()?, *step_op.operands.get(1)?);
        let index = *after.args.get(j)?;
        let step = match step_op.kind {
            OpKind::Add if a == index => self.size(b, 0)?.as_constant()?,
            OpKind::Add if b == index => self.size(a, 0)?.as_constant()?,
            OpKind::Sub if a == index => self.size(b, 0)?.as_constant()?.checked_neg().ok()?,
            _ => return None,
        };
        let increasing = matches!(rel, CmpOp::Lt | CmpOp::Le);
        if step.is_zero() || (step.numer() > 0) != increasing {
            return None;
        }
        let c = if increasing {
            step
        } else {
            step.checked_neg().ok()?
        };
        // The distance the index has to cover.
        let d = if increasing {
            stop.checked_sub(&start)?
        } else {
            start.checked_sub(&stop)?
        };
        let inclusive = matches!(rel, CmpOp::Le | CmpOp::Ge);
        let mut trips = trip_count(&d, c, inclusive)?;
        if early_exit {
            // A `break` may end the loop sooner.
            trips.lo = Poly::zero();
        }
        Some((trips, step_id))
    }

    /// The comparison a loop's condition is, or is the last test of: `i < n`
    /// itself, or (when the body has a `break`, which the lowering carries
    /// as a flag) `if not broke { i < n } else { false }`. The second form
    /// runs at most as often as the comparison holds, and says so.
    fn counted_comparison(&self, cond: ValueId) -> Option<(&OpData, bool)> {
        let ValueDef::OpResult { op, .. } = self.f.value(cond).def else {
            return None;
        };
        let op = self.f.op(op);
        match op.kind {
            OpKind::Cmp(_) => Some((op, false)),
            OpKind::If => {
                let yielded = |region: RegionId| {
                    let last = self.f.op(*self.ops(region).last()?);
                    (last.kind == OpKind::Yield && last.operands.len() == 1)
                        .then(|| last.operands[0])
                };
                let (then, otherwise) = (*op.regions.first()?, *op.regions.get(1)?);
                let ValueDef::OpResult { op: off, .. } = self.f.value(yielded(otherwise)?).def
                else {
                    return None;
                };
                if self.f.op(off).kind != OpKind::ConstBool(false) {
                    return None;
                }
                let ValueDef::OpResult { op: cmp, .. } = self.f.value(yielded(then)?).def else {
                    return None;
                };
                let cmp = self.f.op(cmp);
                matches!(cmp.kind, OpKind::Cmp(_)).then_some((cmp, true))
            }
            _ => None,
        }
    }

    // ----- calls -----

    fn call(&self, op: &OpData, callee: &str) -> Cost {
        let at = self.at(op.span);
        if (self.recursive)(&self.f.name, callee) {
            let reason = at.reason(ReasonKind::Recursion {
                callee: callee.to_string(),
            });
            return unbounded_everything(&reason);
        }
        let Some(callee_cost) = self.done.get(callee) else {
            // A callee that is not in the module: nothing is known.
            return unbounded_everything(&at.reason(ReasonKind::UnknownShape));
        };
        let Some(g) = self.module.function(callee) else {
            return unbounded_everything(&at.reason(ReasonKind::UnknownShape));
        };
        // The callee's sizes are the sizes of the arguments.
        let mut map: BTreeMap<String, Poly> = BTreeMap::new();
        for (param, arg) in g.param_values().iter().zip(&op.operands) {
            let (Type::Number(p), Type::Number(a)) = (g.ty(*param), self.f.ty(*arg)) else {
                continue;
            };
            for (pd, ad) in p.shape.dims().iter().zip(a.shape.dims()) {
                if let ShapeDim::Var(v) = pd {
                    map.entry(v.name().to_string())
                        .or_insert_with(|| dim_poly(ad));
                }
            }
        }
        callee_cost.cost.subst(&map, at)
    }

    // ----- operations -----

    /// The cost of an operation with no region and no callee.
    fn simple(&self, op: &OpData) -> Cost {
        let at = self.at(op.span);
        match self.parts(op) {
            Ok(p) => Cost {
                flops: Amount::exact(p.flops),
                bytes_read: Amount::exact(p.read),
                bytes_written: Amount::exact(p.written),
                allocations: Amount::exact(p.allocs),
                allocated_bytes: Amount::exact(p.alloc_bytes),
            },
            Err(Fail::Shape) => unbounded_everything(&at.reason(ReasonKind::UnknownShape)),
            Err(Fail::Memory) => {
                let reason = at.reason(ReasonKind::UnknownSize);
                Cost {
                    bytes_read: Amount::unbounded(reason.clone()),
                    bytes_written: Amount::unbounded(reason.clone()),
                    allocations: Amount::unbounded(reason.clone()),
                    allocated_bytes: Amount::unbounded(reason),
                    ..Cost::zero()
                }
            }
            Err(Fail::TooLarge) => unbounded_everything(&at.reason(ReasonKind::TooLarge)),
        }
    }

    /// The dimensions of a value's shape, or `Shape` failure if the value's
    /// type is unknown.
    fn dims(&self, v: ValueId) -> P<Vec<Poly>> {
        match self.f.ty(v) {
            Type::Number(n) => Ok(shape_dims(&n.shape)),
            Type::Unknown => Err(Fail::Shape),
            _ => Ok(Vec::new()),
        }
    }

    /// The dimensions of operand `i`.
    fn arg(&self, op: &OpData, i: usize) -> P<Vec<Poly>> {
        self.dims(*op.operands.get(i).ok_or(Fail::Shape)?)
    }

    fn result(&self, op: &OpData) -> P<ValueId> {
        op.results.first().copied().ok_or(Fail::Shape)
    }

    /// The array made by an operation: its entries, or `None` for a scalar.
    fn made(&self, op: &OpData, on_unknown: Fail) -> P<Option<Poly>> {
        let r = self.result(op)?;
        match self.f.ty(r) {
            Type::Unknown => Err(on_unknown),
            Type::Number(n) if n.shape.rank() > 0 => Ok(Some(product(&shape_dims(&n.shape))?)),
            _ => Ok(None),
        }
    }

    /// The parts of an array copy or creation of `n` entries.
    fn make(n: &Poly, read: &Poly) -> P<Parts> {
        Ok(Parts {
            read: scale(read, ENTRY_BYTES)?,
            written: scale(n, ENTRY_BYTES)?,
            allocs: Poly::integer(1),
            alloc_bytes: scale(n, ENTRY_BYTES)?,
            ..Parts::default()
        })
    }

    fn parts(&self, op: &OpData) -> P<Parts> {
        if let Some(w) = flop_weight(&op.kind) {
            return self.elementwise(op, w);
        }
        match &op.kind {
            OpKind::MatMul => self.matmul(op),
            OpKind::MatMulEx { ta, tb, algorithm } => {
                self.matmul_ex(op, *ta, *tb, *algorithm == sky_types::algo::Algorithm::Syrk)
            }
            OpKind::DiagMul { side } => self.diag_mul(op, *side),
            OpKind::Inverse => self.inverse(op),
            OpKind::Solve { method } => self.solve(op, *method, false),
            OpKind::SolveRight { method } => self.solve(op, *method, true),
            OpKind::Transpose => match self.made(op, Fail::Shape)? {
                Some(n) => Self::make(&n, &n),
                None => Ok(Parts::default()),
            },
            OpKind::ArrayLiteral { .. } => {
                let n =
                    Poly::integer(i64::try_from(op.operands.len()).map_err(|_| Fail::TooLarge)?);
                Self::make(&n, &Poly::zero())
            }
            OpKind::Index | OpKind::Slice { .. } => match self.made(op, Fail::Memory)? {
                // One entry read out of the array.
                None => Ok(Parts {
                    read: Poly::integer(ENTRY_BYTES),
                    ..Parts::default()
                }),
                Some(n) => Self::make(&n, &n),
            },
            OpKind::Zeros => match self.made(op, Fail::Memory)? {
                Some(n) => Self::make(&n, &Poly::zero()),
                None => Err(Fail::Memory),
            },
            OpKind::Identity => match self.made(op, Fail::Memory)? {
                Some(n) => Self::make(&n, &Poly::zero()),
                None => Err(Fail::Memory),
            },
            OpKind::Fused => self.fused(op),
            // `dot(a, b)`: a product and a sum per entry, 2·n; `norm(x)`:
            // a square and a sum per entry and one `sqrt`, 2·n + 1 (the
            // scaling that guards against overflow is bookkeeping, as the
            // loop index is). Both read their operands and make a scalar.
            OpKind::Dot => {
                let (a, b) = (product(&self.arg(op, 0)?)?, product(&self.arg(op, 1)?)?);
                Ok(Parts {
                    flops: scale(&a, 2)?,
                    read: scale(&add(&a, &b)?, ENTRY_BYTES)?,
                    ..Parts::default()
                })
            }
            OpKind::Norm => {
                let n = product(&self.arg(op, 0)?)?;
                Ok(Parts {
                    flops: add(&scale(&n, 2)?, &Poly::integer(1))?,
                    read: scale(&n, ENTRY_BYTES)?,
                    ..Parts::default()
                })
            }
            // Constants, comparisons, `len`, `print`, `show`, contract
            // checks, control-flow terminators: no floating-point work and
            // no array traffic.
            _ => Ok(Parts::default()),
        }
    }

    /// An arithmetic operation of weight `w` per scalar operation:
    /// elementwise on arrays.
    fn elementwise(&self, op: &OpData, w: i64) -> P<Parts> {
        let Some(n) = self.made(op, Fail::Shape)? else {
            // A scalar: in registers.
            return Ok(Parts {
                flops: Poly::integer(w),
                ..Parts::default()
            });
        };
        let mut read = Poly::zero();
        for &v in &op.operands {
            let dims = self.dims(v)?;
            if !dims.is_empty() {
                read = add(&read, &product(&dims)?)?;
            }
        }
        let mut p = Self::make(&n, &read)?;
        p.flops = scale(&n, w)?;
        Ok(p)
    }

    /// `A @ B`: `2·n·k·m`.
    fn matmul(&self, op: &OpData) -> P<Parts> {
        let (a, b) = (self.arg(op, 0)?, self.arg(op, 1)?);
        if a.is_empty() || b.is_empty() {
            return Err(Fail::Shape);
        }
        // `k`, the axis they share, is `B`'s first axis.
        let rest = b.get(1..).unwrap_or(&[]);
        let flops = scale(&mul(&product(&a)?, &product(rest)?)?, 2)?;
        let read = add(&product(&a)?, &product(&b)?)?;
        let mut p = match self.made(op, Fail::Shape)? {
            Some(n) => Self::make(&n, &read)?,
            None => Parts {
                read: scale(&read, ENTRY_BYTES)?,
                ..Parts::default()
            },
        };
        p.flops = flops;
        Ok(p)
    }

    /// `A⁻¹`: a scalar reciprocal, or `2·n³`.
    fn inverse(&self, op: &OpData) -> P<Parts> {
        let dims = self.arg(op, 0)?;
        let Some(n) = dims.first() else {
            return Ok(Parts {
                flops: Poly::integer(1),
                ..Parts::default()
            });
        };
        let entries = product(&dims)?;
        let mut p = Self::make(&entries, &entries)?;
        p.flops = scale(&mul(&mul(n, n)?, n)?, 2)?;
        Ok(p)
    }

    /// A Cholesky solve: `n³/3 + 2·n²·k`, two allocations.
    /// `solve[method]` (`A⁻¹ @ b`, operands `A`, `b`) and `solve.right`
    /// (`x @ A⁻¹`, operands `x`, `A`): the factorization, then two
    /// triangular solves per right-hand side. Cholesky factors in n³/3, LU
    /// in 2·n³/3; a Diagonal `A` is one division per entry of the result.
    fn solve(&self, op: &OpData, method: SolveMethod, right: bool) -> P<Parts> {
        let (ia, ib) = if right { (1, 0) } else { (0, 1) };
        let a = self.arg(op, ia)?;
        let b = self.arg(op, ib)?;
        let n = a.first().ok_or(Fail::Shape)?;
        let n2 = mul(n, n)?;
        // The right-hand sides: one, or the columns of `b` (the rows of
        // `x` for `solve.right`).
        let columns = if right {
            if b.len() > 1 {
                b.first().cloned().unwrap_or_else(|| Poly::integer(1))
            } else {
                Poly::integer(1)
            }
        } else {
            b.get(1).cloned().unwrap_or_else(|| Poly::integer(1))
        };
        let result = product(&b)?;
        let read = scale(&add(&product(&a)?, &result)?, ENTRY_BYTES)?;
        if method == SolveMethod::Diagonal {
            return Ok(Parts {
                flops: result.clone(),
                read,
                written: scale(&result, ENTRY_BYTES)?,
                allocs: Poly::integer(1),
                alloc_bytes: scale(&result, ENTRY_BYTES)?,
            });
        }
        let n3 = mul(&n2, n)?;
        let factor = match method {
            SolveMethod::Cholesky => frac(&n3, 1, 3)?,
            _ => frac(&n3, 2, 3)?,
        };
        let solves = scale(&mul(&n2, &columns)?, 2)?;
        Ok(Parts {
            flops: add(&factor, &solves)?,
            read,
            written: scale(&add(&result, &n2)?, ENTRY_BYTES)?,
            allocs: Poly::integer(2),
            alloc_bytes: scale(&add(&result, &n2)?, ENTRY_BYTES)?,
        })
    }

    /// `matmul[ta, tb]`: the product of the (possibly transposed) operands,
    /// 2·n·k·m as `matmul`; the transposes are read in place, not made. A
    /// Gram product (`alg=syrk`, an m×m result) computes one triangle and
    /// mirrors it: m·(m+1)/2 entries of 2·k flops, k·m² + k·m.
    fn matmul_ex(&self, op: &OpData, ta: bool, tb: bool, syrk: bool) -> P<Parts> {
        let flip = |d: Vec<Poly>, t: bool| {
            if t && d.len() == 2 {
                vec![d[1].clone(), d[0].clone()]
            } else {
                d
            }
        };
        let a = flip(self.arg(op, 0)?, ta);
        let b = flip(self.arg(op, 1)?, tb);
        if a.is_empty() || b.is_empty() {
            return Err(Fail::Shape);
        }
        let rest = b.get(1..).unwrap_or(&[]);
        let flops = match (syrk, rest) {
            (true, [m]) => mul(&product(&a)?, &add(m, &Poly::integer(1))?)?,
            _ => scale(&mul(&product(&a)?, &product(rest)?)?, 2)?,
        };
        let read = add(&product(&a)?, &product(&b)?)?;
        let mut p = match self.made(op, Fail::Shape)? {
            Some(n) => Self::make(&n, &read)?,
            None => Parts {
                read: scale(&read, ENTRY_BYTES)?,
                ..Parts::default()
            },
        };
        p.flops = flops;
        Ok(p)
    }

    /// `diag.mul`: each entry of `X` scaled by a diagonal entry, computed as
    /// `0.0 + d·x`, two flops per entry.
    fn diag_mul(&self, op: &OpData, side: Side) -> P<Parts> {
        let (id, ix) = match side {
            Side::Left => (0, 1),
            Side::Right => (1, 0),
        };
        let d = product(&self.arg(op, id)?)?;
        let x = product(&self.arg(op, ix)?)?;
        let mut p = Self::make(&x, &add(&d, &x)?)?;
        p.flops = scale(&x, 2)?;
        Ok(p)
    }

    /// A fused kernel: its body once per entry.
    fn fused(&self, op: &OpData) -> P<Parts> {
        let n = self.made(op, Fail::Shape)?.ok_or(Fail::Shape)?;
        let body = op.regions.first().map_or(&[][..], |r| self.ops(*r));
        let per_entry: i64 = body
            .iter()
            .filter_map(|o| flop_weight(&self.f.op(*o).kind))
            .sum();
        let mut read = Poly::zero();
        for &v in &op.operands {
            let dims = self.dims(v)?;
            if !dims.is_empty() {
                read = add(&read, &product(&dims)?)?;
            }
        }
        let mut p = Self::make(&n, &read)?;
        p.flops = scale(&n, per_entry)?;
        Ok(p)
    }
}

fn unbounded_everything(reason: &Reason) -> Cost {
    let a = || Amount::unbounded(reason.clone());
    Cost {
        flops: a(),
        bytes_read: a(),
        bytes_written: a(),
        allocations: a(),
        allocated_bytes: a(),
    }
}

/// How many steps of `c` fit in the distance `d` (`< d` or `<= d`).
fn trip_count(d: &Poly, c: Rational, inclusive: bool) -> Option<Trips> {
    if let Some(dist) = d.as_constant() {
        // Everything is a number: count exactly, as the loop does.
        let q = dist.checked_div(c).ok()?;
        let (n, den) = (q.numer(), q.denom());
        let floor = n.div_euclid(den);
        let n = if inclusive {
            floor + 1
        } else if n.rem_euclid(den) == 0 {
            floor
        } else {
            floor + 1
        };
        return Some(Trips::exact(Poly::integer(n.max(0))));
    }
    let over_c = |p: &Poly| p.checked_scale(c.checked_recip().ok()?);
    let one = Poly::integer(1);
    // Only a distance that is non-negative for every size gives a lower
    // bound other than 0 (the count is `max(0, ...)`, and a lower bound must
    // itself be non-negative).
    let never_negative = d.nonnegative_for_all_sizes();
    if c == Rational::ONE {
        // Whole steps of one: `d` (or `d + 1`) runs, or none if `d` is negative.
        let exact = if inclusive {
            d.checked_add(&one)?
        } else {
            d.clone()
        };
        return Some(if never_negative {
            Trips::exact(exact)
        } else {
            Trips {
                lo: Poly::zero(),
                hi: exact.upper_join(&Poly::zero()),
            }
        });
    }
    // At least `d/c`, and fewer than `d/c + 1`, runs.
    let lo = over_c(d)?;
    let hi = lo.checked_add(&one)?;
    Some(if never_negative {
        Trips { lo, hi }
    } else {
        Trips {
            lo: Poly::zero(),
            hi: hi.upper_join(&Poly::zero()),
        }
    })
}
