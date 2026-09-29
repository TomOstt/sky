//! Building the derivative function: the walk over the source function that
//! copies every operation (the primal computation) and, for the values that
//! depend on the differentiated parameter, emits their tangents. The
//! per-operation rules are in `rules.rs`; this file is the structure
//! (blocks, `if`, `while`, `call`) and the typing helpers.

use sky_diag::Span;
use sky_ir::{
    BlockId, Function, FunctionBuilder, Module, NumberType, OpData, OpId, OpKind, RegionId, Repr,
    Type, ValueId,
};
use sky_units::Dimension;

use crate::analysis::analyze;
use crate::{derivative_name, AdError};

pub(crate) type Res<T> = Result<T, AdError>;

/// The state of one derivative function being built.
pub(crate) struct Diff<'m> {
    pub module: &'m Module,
    pub src: &'m Function,
    pub b: FunctionBuilder,
    /// The dimension of the parameter differentiated against.
    pub x_dim: Dimension,
    /// From the activity analysis: which source values depend on `x`.
    pub active: Vec<bool>,
    /// Source value to its copy in the derivative function.
    pub map: Vec<Option<ValueId>>,
    /// Source value to its tangent, when it has one (`None` is a
    /// structural zero, which costs no code until something needs it).
    pub tan: Vec<Option<ValueId>>,
    /// Derivative functions this one calls: `(callee, parameter)`.
    pub needs: Vec<(String, usize)>,
}

/// Builds the derivative of `f` with respect to parameter `wrt`, and
/// returns the derivative functions it calls.
pub(crate) fn build(
    module: &Module,
    f: &Function,
    wrt: usize,
) -> Res<(Function, Vec<(String, usize)>)> {
    let name = f.name.clone();
    let params = f.param_values();
    let Some(&x) = params.get(wrt) else {
        return Err(AdError::ParamOutOfRange {
            func: name,
            wrt,
            params: params.len(),
        });
    };
    let x_dim = match f.ty(x) {
        Type::Number(n) if n.shape.rank() == 0 => n.dim,
        Type::Number(_) => {
            return Err(AdError::NonScalar {
                func: name,
                what: format!("parameter {wrt}"),
            })
        }
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: format!("parameter {wrt}"),
            })
        }
        _ => return Err(AdError::NonNumberParam { func: name, wrt }),
    };
    let [res_ty] = &f.results[..] else {
        return Err(AdError::BadResult {
            func: name,
            why: format!("it returns {} values, not one", f.results.len()),
        });
    };
    match res_ty {
        Type::Number(n) if n.shape.rank() == 0 => {}
        Type::Number(_) => {
            return Err(AdError::NonScalar {
                func: name,
                what: "the result".into(),
            })
        }
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: "the result".into(),
            })
        }
        _ => {
            return Err(AdError::BadResult {
                func: name,
                why: "its result is not a number".into(),
            })
        }
    }

    let mut d = Diff {
        module,
        src: f,
        // The result types are filled in below once we can compute them.
        b: FunctionBuilder::new(derivative_name(f, wrt), f.params.clone(), vec![], f.span),
        x_dim,
        active: analyze(f, wrt),
        map: vec![None; f.values.len()],
        tan: vec![None; f.values.len()],
        needs: Vec::new(),
    };
    let res_tan = tangent_type_of(&d, res_ty)?;
    let mut b = FunctionBuilder::new(
        derivative_name(f, wrt),
        f.params.clone(),
        vec![res_ty.clone(), res_tan],
        f.span,
    );
    b.set_effects(f.effects);
    d.b = b;

    for (old, new) in f.param_values().iter().zip(d.b.params()) {
        d.map[old.index()] = Some(new);
        if let Some(n) = &f.value(*old).name {
            d.b.set_name(new, n.clone());
        }
    }
    // The seed: dx/dx = 1, dimensionless.
    let seed_ty = d.tangent_ty(x)?;
    let seed = d.konst(1.0, seed_ty, f.span);
    d.tan[x.index()] = Some(seed);

    let term = d.process_ops(f.entry())?;
    let t = f.op(term);
    let [r] = t.operands[..] else {
        return Err(d.malformed("`return` must have one operand"));
    };
    if t.kind != OpKind::Return {
        return Err(d.malformed("the body must end in `return`"));
    }
    let value = d.primal(r)?;
    let tangent = d.tangent_or_zero(r, t.span)?;
    d.b.op(OpKind::Return, vec![value, tangent], vec![], t.span);
    Ok((d.b.finish(), d.needs))
}

fn tangent_type_of(d: &Diff<'_>, ty: &Type) -> Res<Type> {
    let Type::Number(n) = ty else {
        unreachable!("checked by the caller")
    };
    let dim = d.dim_div(n.dim, d.x_dim)?;
    Ok(number_type(dim, n.repr))
}

pub(crate) fn number_type(dim: Dimension, repr: Repr) -> Type {
    Type::Number(NumberType {
        repr,
        ..NumberType::scalar(dim)
    })
}

impl Diff<'_> {
    // ----- errors -----

    pub fn malformed(&self, why: &str) -> AdError {
        AdError::Malformed {
            func: self.src.name.clone(),
            why: why.into(),
        }
    }

    pub fn unsupported(&self, op: &str, why: &str) -> AdError {
        AdError::Unsupported {
            func: self.src.name.clone(),
            op: op.into(),
            why: why.into(),
        }
    }

    // ----- typing -----

    pub fn dim_div(&self, a: Dimension, b: Dimension) -> Res<Dimension> {
        a.checked_div(b).map_err(|e| self.dim_err(e))
    }

    pub fn dim_mul(&self, a: Dimension, b: Dimension) -> Res<Dimension> {
        a.checked_mul(b).map_err(|e| self.dim_err(e))
    }

    fn dim_err(&self, e: impl std::fmt::Display) -> AdError {
        AdError::Dimension {
            func: self.src.name.clone(),
            why: e.to_string(),
        }
    }

    pub fn dim_pow(&self, a: Dimension, e: sky_units::Rational) -> Res<Dimension> {
        a.pow_rat(e).map_err(|e| self.dim_err(e))
    }

    /// The dimension of source value `v`, which must be a scalar number.
    pub fn dim_of(&self, v: ValueId) -> Res<Dimension> {
        match self.src.ty(v) {
            Type::Number(n) if n.shape.rank() == 0 => Ok(n.dim),
            Type::Number(_) => Err(AdError::NonScalar {
                func: self.src.name.clone(),
                what: format!("{v}"),
            }),
            Type::Unknown => Err(AdError::UnknownType {
                func: self.src.name.clone(),
                what: format!("{v}"),
            }),
            _ => Err(self.malformed(&format!("{v} is not a number but has a tangent"))),
        }
    }

    pub fn repr_of(&self, v: ValueId) -> Repr {
        self.src
            .ty(v)
            .as_number()
            .map_or(Repr::Unchosen, |n| n.repr)
    }

    /// The type of `v`'s tangent: `dim(v) / dim(x)`, a scalar.
    pub fn tangent_ty(&self, v: ValueId) -> Res<Type> {
        let dim = self.dim_div(self.dim_of(v)?, self.x_dim)?;
        Ok(number_type(dim, self.repr_of(v)))
    }

    // ----- values -----

    pub fn primal(&self, v: ValueId) -> Res<ValueId> {
        self.map
            .get(v.index())
            .copied()
            .flatten()
            .ok_or_else(|| self.malformed(&format!("{v} is used before it is defined")))
    }

    pub fn tan_of(&self, v: ValueId) -> Option<ValueId> {
        self.tan.get(v.index()).copied().flatten()
    }

    pub fn konst(&mut self, x: f64, ty: Type, span: Span) -> ValueId {
        self.b
            .op1(OpKind::ConstFloat { bits: x.to_bits() }, vec![], ty, span)
    }

    pub fn emit(&mut self, kind: OpKind, operands: Vec<ValueId>, ty: Type, span: Span) -> ValueId {
        self.b.op1(kind, operands, ty, span)
    }

    /// The tangent of `v`, materializing a zero of the right type if it is
    /// structurally zero. Not cached: a zero is only valid in the block it
    /// is emitted in.
    pub fn tangent_or_zero(&mut self, v: ValueId, span: Span) -> Res<ValueId> {
        match self.tan_of(v) {
            Some(t) => Ok(t),
            None => {
                let ty = self.tangent_ty(v)?;
                Ok(self.konst(0.0, ty, span))
            }
        }
    }

    // ----- structure -----

    fn single_block(&self, r: RegionId) -> Res<BlockId> {
        match self.src.region(r).blocks[..] {
            [b] => Ok(b),
            _ => Err(self.malformed("a region must have exactly one block")),
        }
    }

    /// Emits the operations of `block` except its terminator, which is
    /// returned for the caller to translate.
    pub fn process_ops(&mut self, block: BlockId) -> Res<OpId> {
        let src = self.src;
        let Some((last, body)) = src.block(block).ops.split_last() else {
            return Err(self.malformed("a block has no terminator"));
        };
        if !src.op(*last).kind.properties().terminator {
            return Err(self.malformed("a block does not end in a terminator"));
        }
        for id in body {
            self.op(src.op(*id))?;
        }
        Ok(*last)
    }

    fn op(&mut self, op: &OpData) -> Res<()> {
        match &op.kind {
            OpKind::If => self.if_op(op),
            OpKind::While => self.while_op(op),
            OpKind::Call { callee } => self.call_op(op, callee),
            OpKind::Yield | OpKind::Condition | OpKind::Return => {
                Err(self.malformed("a terminator in the middle of a block"))
            }
            _ => {
                self.copy_primal(op)?;
                if let [r] = op.results[..] {
                    if self.active[r.index()]
                        && op.operands.iter().any(|v| self.tan_of(*v).is_some())
                    {
                        self.tan[r.index()] = self.tangent_rule(op, r)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Copies `op` (without regions) with mapped operands and records its
    /// results. Tangents are the caller's business.
    fn copy_primal(&mut self, op: &OpData) -> Res<Vec<ValueId>> {
        let operands = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        let tys = op.results.iter().map(|r| self.src.ty(*r).clone()).collect();
        let (_, results) = self.b.op(op.kind.clone(), operands, tys, op.span);
        for (old, new) in op.results.iter().zip(&results) {
            self.map[old.index()] = Some(*new);
            if let Some(n) = &self.src.value(*old).name {
                self.b.set_name(*new, n.clone());
            }
        }
        Ok(results)
    }

    /// `if`: both arms are differentiated. Each active result gets a
    /// tangent result, which each arm yields (a zero if that arm's value
    /// does not depend on `x`). The condition's tangent does not flow.
    fn if_op(&mut self, op: &OpData) -> Res<()> {
        let src = self.src;
        let [c] = op.operands[..] else {
            return Err(self.malformed("`if` takes one operand"));
        };
        if op.regions.len() != 2 {
            return Err(self.malformed("`if` takes two regions"));
        }
        let n = op.results.len();
        let slots: Vec<usize> = (0..n)
            .filter(|i| self.active[op.results[*i].index()])
            .collect();
        let mut tys: Vec<Type> = op.results.iter().map(|r| src.ty(*r).clone()).collect();
        for &i in &slots {
            tys.push(self.tangent_ty(op.results[i])?);
        }
        let cond = self.primal(c)?;
        let outer = self.b.insertion_block();
        let (new_op, results) = self.b.op(OpKind::If, vec![cond], tys, op.span);
        for (i, r) in op.results.iter().enumerate() {
            self.map[r.index()] = Some(results[i]);
        }
        for (k, &i) in slots.iter().enumerate() {
            self.tan[op.results[i].index()] = Some(results[n + k]);
        }
        for &region in &op.regions {
            let old = self.single_block(region)?;
            let r = self.b.add_region(new_op);
            let blk = self.b.add_block(r, vec![]);
            self.b.set_insertion_block(blk);
            let term = src.op(self.process_ops(old)?);
            if term.kind != OpKind::Yield || term.operands.len() != n {
                return Err(self.malformed("an `if` arm must yield one value per result"));
            }
            let mut ys = term
                .operands
                .iter()
                .map(|v| self.primal(*v))
                .collect::<Res<Vec<_>>>()?;
            for &i in &slots {
                ys.push(self.tangent_or_zero(term.operands[i], term.span)?);
            }
            self.b.op(OpKind::Yield, ys, vec![], term.span);
        }
        self.b.set_insertion_block(outer);
        Ok(())
    }

    /// `while`: a tangent slot next to every active loop-carried value,
    /// through the initial values, the `before` arguments, the forwarded
    /// values, the `after` arguments, the yields and the results.
    fn while_op(&mut self, op: &OpData) -> Res<()> {
        let src = self.src;
        let [rb, ra] = op.regions[..] else {
            return Err(self.malformed("`while` takes two regions"));
        };
        let (bb, ab) = (self.single_block(rb)?, self.single_block(ra)?);
        let n = op.operands.len();
        let (bargs, aargs) = (&src.block(bb).args, &src.block(ab).args);
        if bargs.len() != n || aargs.len() != n || op.results.len() != n {
            return Err(self.malformed("`while` operands, arguments and results must match"));
        }
        let slots: Vec<usize> = (0..n).filter(|j| self.active[bargs[*j].index()]).collect();

        let mut operands = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        for &j in &slots {
            operands.push(self.tangent_or_zero(op.operands[j], op.span)?);
        }
        let mut rtys: Vec<Type> = op.results.iter().map(|r| src.ty(*r).clone()).collect();
        for &j in &slots {
            rtys.push(self.tangent_ty(op.results[j])?);
        }
        let outer = self.b.insertion_block();
        let (new_op, results) = self.b.op(OpKind::While, operands, rtys, op.span);
        for (j, r) in op.results.iter().enumerate() {
            self.map[r.index()] = Some(results[j]);
        }
        for (k, &j) in slots.iter().enumerate() {
            self.tan[op.results[j].index()] = Some(results[n + k]);
        }

        // before
        let region = self.b.add_region(new_op);
        self.region_block(region, bargs, &slots)?;
        let term = src.op(self.process_ops(bb)?);
        let Some((&c, fwd)) = term.operands.split_first() else {
            return Err(self.malformed("`condition` needs a Bool operand"));
        };
        if term.kind != OpKind::Condition || fwd.len() != n {
            return Err(self.malformed("`before` must end in `condition` forwarding every value"));
        }
        let mut ops = vec![self.primal(c)?];
        for v in fwd {
            ops.push(self.primal(*v)?);
        }
        for &j in &slots {
            ops.push(self.tangent_or_zero(fwd[j], term.span)?);
        }
        self.b.op(OpKind::Condition, ops, vec![], term.span);

        // after
        let region = self.b.add_region(new_op);
        self.region_block(region, aargs, &slots)?;
        let term = src.op(self.process_ops(ab)?);
        if term.kind != OpKind::Yield || term.operands.len() != n {
            return Err(self.malformed("`after` must yield the next loop-carried values"));
        }
        let mut ys = term
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        for &j in &slots {
            ys.push(self.tangent_or_zero(term.operands[j], term.span)?);
        }
        self.b.op(OpKind::Yield, ys, vec![], term.span);

        self.b.set_insertion_block(outer);
        Ok(())
    }

    /// A new block in `region` whose arguments are copies of `old_args`
    /// followed by a tangent for each of `slots`; maps them and moves the
    /// insertion point in.
    fn region_block(&mut self, region: RegionId, old_args: &[ValueId], slots: &[usize]) -> Res<()> {
        let mut tys: Vec<Type> = old_args.iter().map(|a| self.src.ty(*a).clone()).collect();
        for &j in slots {
            tys.push(self.tangent_ty(old_args[j])?);
        }
        let blk = self.b.add_block(region, tys);
        let new = self.b.func().block(blk).args.clone();
        let n = old_args.len();
        for (j, a) in old_args.iter().enumerate() {
            self.map[a.index()] = Some(new[j]);
        }
        for (k, &j) in slots.iter().enumerate() {
            self.tan[old_args[j].index()] = Some(new[n + k]);
        }
        self.b.set_insertion_block(blk);
        Ok(())
    }

    /// `call`: a call whose arguments do not depend on `x` is copied. One
    /// that does becomes calls to the callee's derivatives, one per
    /// argument with a tangent, chained by the rule
    /// `d f(a, b) = ∂f/∂a · ta + ∂f/∂b · tb`. The first call also gives the
    /// primal result.
    fn call_op(&mut self, op: &OpData, callee: &str) -> Res<()> {
        let live: Vec<usize> = (0..op.operands.len())
            .filter(|i| self.tan_of(op.operands[*i]).is_some())
            .collect();
        if live.is_empty() || op.results.is_empty() {
            self.copy_primal(op)?;
            return Ok(());
        }
        let Some(target) = self.module.function(callee) else {
            return Err(AdError::NoSuchFunction(callee.into()));
        };
        let [r] = op.results[..] else {
            return Err(AdError::BadResult {
                func: self.src.name.clone(),
                why: format!("`{callee}` must return exactly one number to be differentiated"),
            });
        };
        let args = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        let r_dim = self.dim_of(r)?;
        let result_ty = self.tangent_ty(r)?;
        let mut sum: Option<ValueId> = None;
        for (n, &k) in live.iter().enumerate() {
            let dname = derivative_name(target, k);
            self.needs.push((callee.to_string(), k));
            // dim(∂f/∂a_k) = dim(f) / dim(a_k)
            let partial = number_type(
                self.dim_div(r_dim, self.dim_of(op.operands[k])?)?,
                self.repr_of(r),
            );
            let (_, res) = self.b.op(
                OpKind::Call { callee: dname },
                args.clone(),
                vec![self.src.ty(r).clone(), partial],
                op.span,
            );
            if n == 0 {
                self.map[r.index()] = Some(res[0]);
            }
            let t = self.tan_of(op.operands[k]).expect("live has a tangent");
            let term = self.emit(OpKind::Mul, vec![res[1], t], result_ty.clone(), op.span);
            sum = Some(match sum {
                None => term,
                Some(s) => self.emit(OpKind::Add, vec![s, term], result_ty.clone(), op.span),
            });
        }
        self.tan[r.index()] = sum;
        Ok(())
    }
}
