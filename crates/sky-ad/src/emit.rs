//! Building the derivative function: the walk over the source function that
//! copies every operation (the primal computation) and, for the values that
//! depend on the differentiated parameter, emits their tangents. The
//! per-operation rules are in `rules.rs`; this file is the structure
//! (blocks, `if`, `while`, `call`) and the typing helpers.

use sky_diag::Span;
use sky_ir::{
    BlockId, Function, FunctionBuilder, Module, NumberType, OpData, OpId, OpKind, RegionId, Repr,
    Type, ValueDef, ValueId,
};
use sky_types::{Shape, TagSet};
use sky_units::Dimension;

use crate::analysis::analyze;
use crate::inline::inline_array_calls;
use crate::{derivative_name, derivative_name_for, AdError, Kind, Task};

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
    /// Derivative functions this one calls.
    pub needs: Vec<Task>,
    /// Inside the scalar body of a `fused` kernel: no `if` and no array
    /// operation may be emitted there, so a rule that needs one refuses or
    /// uses branch-free arithmetic.
    pub in_body: bool,
}

/// What seeds the tangent of the parameter differentiated against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Seed {
    /// A scalar parameter `x`: `dx/dx = 1`.
    Unit,
    /// An array parameter `p`: the derivative in the direction of an extra
    /// trailing parameter, a dimensionless array of `p`'s shape (a
    /// Jacobian-vector product). A gradient is `n` of these with unit
    /// vectors as directions, when reverse mode cannot be used.
    Direction,
}

/// Builds the derivative of `f` with respect to parameter `wrt`, and
/// returns the derivative functions it calls.
///
/// A scalar parameter ([`Seed::Unit`]) gives `df/dx(params...) -> (values...,
/// tangents...)`. An array parameter ([`Seed::Direction`]) gives the
/// directional derivative `df/dp#jvp(params..., direction)`, with the
/// direction an extra trailing parameter.
pub(crate) fn build(
    module: &Module,
    f: &Function,
    wrt: usize,
    seed: Seed,
) -> Res<(Function, Vec<Task>)> {
    let name = f.name.clone();
    let params = f.param_values();
    let Some(&x) = params.get(wrt) else {
        return Err(AdError::ParamOutOfRange {
            func: name,
            wrt,
            params: params.len(),
        });
    };
    let x_ty = match f.ty(x) {
        Type::Number(n) => n.clone(),
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: format!("parameter {wrt}"),
            })
        }
        _ => return Err(AdError::NonNumberParam { func: name, wrt }),
    };
    if (x_ty.shape.rank() == 0) != (seed == Seed::Unit) {
        return Err(AdError::Malformed {
            func: name,
            why: "the seed does not fit the parameter's shape".into(),
        });
    }
    if f.results.is_empty() {
        return Err(AdError::BadResult {
            func: name,
            why: "it returns no value".into(),
        });
    }
    for (i, res_ty) in f.results.iter().enumerate() {
        let what = if f.results.len() == 1 {
            "the result".to_string()
        } else {
            format!("result {i}")
        };
        match res_ty {
            Type::Number(_) => {}
            Type::Unknown => return Err(AdError::UnknownType { func: name, what }),
            _ => {
                return Err(AdError::BadResult {
                    func: name,
                    why: format!("{what} is not a number"),
                })
            }
        }
    }

    // Calls that pass or return arrays are inlined, so the derivative sees
    // through them (a helper `displacement(p)` is part of the potential).
    let inlined = inline_array_calls(module, f)?;
    let f = &inlined;

    let mut params_ty = f.params.clone();
    if seed == Seed::Direction {
        params_ty.push(Type::Number(NumberType {
            dim: Dimension::DIMENSIONLESS,
            shape: x_ty.shape.clone(),
            tags: TagSet::EMPTY,
            repr: x_ty.repr,
        }));
    }
    let out_name = derivative_name_for(f, wrt, seed_kind(seed));
    let mut d = Diff {
        module,
        src: f,
        // The result types are filled in below once we can compute them.
        b: FunctionBuilder::new(out_name.clone(), params_ty.clone(), vec![], f.span),
        x_dim: x_ty.dim,
        active: analyze(f, wrt),
        map: vec![None; f.values.len()],
        tan: vec![None; f.values.len()],
        needs: Vec::new(),
        in_body: false,
    };
    let mut results = f.results.clone();
    for res_ty in &f.results {
        results.push(tangent_type_of(&d, res_ty)?);
    }
    let mut b = FunctionBuilder::new(out_name, params_ty, results, f.span);
    b.set_effects(f.effects);
    d.b = b;

    let new_params = d.b.params();
    for (old, new) in f.param_values().iter().zip(&new_params) {
        d.map[old.index()] = Some(*new);
        if let Some(n) = &f.value(*old).name {
            d.b.set_name(*new, n.clone());
        }
    }
    let seed_value = match seed {
        // The seed: dx/dx = 1, dimensionless.
        Seed::Unit => {
            let seed_ty = d.tangent_ty(x)?;
            d.konst(1.0, seed_ty, f.span)
        }
        Seed::Direction => new_params[f.params.len()],
    };
    d.tan[x.index()] = Some(seed_value);

    let term = d.process_ops(f.entry())?;
    let t = f.op(term);
    if t.kind != OpKind::Return {
        return Err(d.malformed("the body must end in `return`"));
    }
    if t.operands.len() != f.results.len() {
        return Err(d.malformed("`return` must have one operand per result"));
    }
    // The values, then their tangents.
    let mut returned = Vec::with_capacity(2 * t.operands.len());
    for r in &t.operands {
        returned.push(d.primal(*r)?);
    }
    for r in &t.operands {
        returned.push(d.tangent_or_zero(*r, t.span)?);
    }
    d.b.op(OpKind::Return, returned, vec![], t.span);
    Ok((d.b.finish(), d.needs))
}

fn seed_kind(seed: Seed) -> Kind {
    match seed {
        Seed::Unit => Kind::Forward,
        Seed::Direction => Kind::Jvp,
    }
}

fn tangent_type_of(d: &Diff<'_>, ty: &Type) -> Res<Type> {
    let Type::Number(n) = ty else {
        unreachable!("checked by the caller")
    };
    let dim = d.dim_div(n.dim, d.x_dim)?;
    Ok(shaped_type(dim, n.shape.clone(), n.repr))
}

/// A scalar of dimension `dim`.
pub(crate) fn number_type(dim: Dimension, repr: Repr) -> Type {
    Type::Number(NumberType {
        repr,
        ..NumberType::scalar(dim)
    })
}

/// A number of dimension `dim` and shape `shape`, with no tags: what a
/// tangent or an adjoint is.
pub(crate) fn shaped_type(dim: Dimension, shape: Shape, repr: Repr) -> Type {
    Type::Number(NumberType {
        dim,
        shape,
        tags: TagSet::EMPTY,
        repr,
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

    /// The dimension of source value `v`, which must be a number (of any
    /// shape: an array's entries share one dimension).
    pub fn dim_of(&self, v: ValueId) -> Res<Dimension> {
        match self.src.ty(v) {
            Type::Number(n) => Ok(n.dim),
            Type::Unknown => Err(AdError::UnknownType {
                func: self.src.name.clone(),
                what: self.describe(v),
            }),
            _ => Err(self.malformed(&format!("{v} is not a number but has a tangent"))),
        }
    }

    /// Source value `v` for an error message: its source name, or the
    /// operation that computes it (`the result of the call to `g``).
    fn describe(&self, v: ValueId) -> String {
        let data = self.src.value(v);
        if let Some(name) = &data.name {
            return format!("`{name}`");
        }
        match data.def {
            ValueDef::OpResult { op, .. } => match &self.src.op(op).kind {
                OpKind::Call { callee } => format!("the result of the call to `{callee}`"),
                other => format!("the result of `{}`", other.mnemonic()),
            },
            ValueDef::BlockArg { .. } => format!("{v}"),
        }
    }

    pub fn repr_of(&self, v: ValueId) -> Repr {
        self.src
            .ty(v)
            .as_number()
            .map_or(Repr::Unchosen, |n| n.repr)
    }

    /// The shape of source value `v` (a scalar's is `()`).
    pub fn shape_of(&self, v: ValueId) -> Shape {
        self.src
            .ty(v)
            .as_number()
            .map_or_else(Shape::scalar, |n| n.shape.clone())
    }

    /// The type of `v`'s tangent: `dim(v) / dim(x)`, with the shape of `v`.
    pub fn tangent_ty(&self, v: ValueId) -> Res<Type> {
        let dim = self.dim_div(self.dim_of(v)?, self.x_dim)?;
        Ok(shaped_type(dim, self.shape_of(v), self.repr_of(v)))
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

    /// Emits an operation. For the elementwise operations the rules state
    /// the dimension they know and may state a scalar shape where an operand
    /// is an array (a number operand broadcasts): the result takes the
    /// array's shape, and for `add`/`sub` a number operand is first spread
    /// over the array (`s * ones`), since those take equal shapes only.
    pub fn emit(&mut self, kind: OpKind, operands: Vec<ValueId>, ty: Type, span: Span) -> ValueId {
        let (operands, ty) = self.conform(&kind, operands, ty, span);
        self.b.op1(kind, operands, ty, span)
    }

    fn conform(
        &mut self,
        kind: &OpKind,
        mut operands: Vec<ValueId>,
        ty: Type,
        span: Span,
    ) -> (Vec<ValueId>, Type) {
        let elementwise = matches!(
            kind,
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Neg | OpKind::Math(_)
        );
        let Type::Number(mut n) = ty else {
            return (operands, ty);
        };
        if !elementwise || n.shape.rank() != 0 {
            return (operands, Type::Number(n));
        }
        let shape_of_new =
            |d: &Self, v: ValueId| d.b.func().ty(v).as_number().map(|x| x.shape.clone());
        let Some(array) = operands
            .iter()
            .copied()
            .find(|v| shape_of_new(self, *v).is_some_and(|s| s.rank() > 0))
        else {
            return (operands, Type::Number(n));
        };
        n.shape = shape_of_new(self, array).expect("an array operand");
        if matches!(kind, OpKind::Add | OpKind::Sub) {
            for slot in operands.iter_mut() {
                let v = *slot;
                if shape_of_new(self, v).is_none_or(|s| s.rank() != 0) {
                    continue;
                }
                let Ok(ones) = self.ones_like(array, span) else {
                    continue;
                };
                if let Type::Number(sn) = self.b.func().ty(v).clone() {
                    let spread = Type::Number(NumberType {
                        shape: n.shape.clone(),
                        ..sn
                    });
                    *slot = self.b.op1(OpKind::Mul, vec![v, ones], spread, span);
                }
            }
        }
        (operands, Type::Number(n))
    }

    /// The tangent of `v`, materializing a zero of the right type if it is
    /// structurally zero. Not cached: a zero is only valid in the block it
    /// is emitted in.
    pub fn tangent_or_zero(&mut self, v: ValueId, span: Span) -> Res<ValueId> {
        match self.tan_of(v) {
            Some(t) => Ok(t),
            None => {
                let ty = self.tangent_ty(v)?;
                let like = self.primal(v).ok();
                self.zero_of(&ty, like, span)
            }
        }
    }

    // ----- structure -----

    pub fn single_block(&self, r: RegionId) -> Res<BlockId> {
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
            OpKind::Fused => self.fused_op(op),
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
    pub fn copy_primal(&mut self, op: &OpData) -> Res<Vec<ValueId>> {
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
    ///
    /// The chain rule is a product of numbers, so the arguments with
    /// tangents must be scalars (a result may be an array: its derivative
    /// scales by the scalar tangent). A call with an array argument is
    /// inlined before differentiation ([`crate::inline`]), so one that gets
    /// here is a recursive function's.
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
        // Every result is a number with a tangent; a callee's derivative
        // function returns its values, then the partial derivative of each
        // (see `build`).
        let n = op.results.len();
        let mut result_tys = Vec::with_capacity(n);
        for r in &op.results {
            if !matches!(self.src.ty(*r), Type::Number(_) | Type::Unknown) {
                return Err(AdError::BadResult {
                    func: self.src.name.clone(),
                    why: format!("`{callee}` must return only numbers to be differentiated"),
                });
            }
            result_tys.push(self.tangent_ty(*r)?);
        }
        let args = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        let mut sums: Vec<Option<ValueId>> = vec![None; n];
        for (m, &k) in live.iter().enumerate() {
            if self.shape_of(op.operands[k]).rank() != 0 {
                return Err(self.unsupported(
                    "call",
                    &format!(
                        "an argument of the call to `{callee}` is an array that depends on the \
                         variable, and the call cannot be inlined (the function is recursive)"
                    ),
                ));
            }
            let dname = derivative_name(target, k);
            self.needs.push(Task {
                func: callee.to_string(),
                wrt: k,
                kind: Kind::Forward,
            });
            let mut tys: Vec<Type> = op.results.iter().map(|r| self.src.ty(*r).clone()).collect();
            for r in &op.results {
                // dim(∂r/∂a_k) = dim(r) / dim(a_k)
                tys.push(shaped_type(
                    self.dim_div(self.dim_of(*r)?, self.dim_of(op.operands[k])?)?,
                    self.shape_of(*r),
                    self.repr_of(*r),
                ));
            }
            let (_, res) = self
                .b
                .op(OpKind::Call { callee: dname }, args.clone(), tys, op.span);
            if m == 0 {
                for (r, v) in op.results.iter().zip(&res) {
                    self.map[r.index()] = Some(*v);
                }
            }
            let t = self.tan_of(op.operands[k]).expect("live has a tangent");
            for i in 0..n {
                let term = self.emit(
                    OpKind::Mul,
                    vec![res[n + i], t],
                    result_tys[i].clone(),
                    op.span,
                );
                sums[i] = Some(match sums[i] {
                    None => term,
                    Some(s) => {
                        self.emit(OpKind::Add, vec![s, term], result_tys[i].clone(), op.span)
                    }
                });
            }
        }
        for (r, sum) in op.results.iter().zip(sums) {
            self.tan[r.index()] = sum;
        }
        Ok(())
    }

    /// `fused`: the primal kernel is copied, and the tangent is a second
    /// kernel over the same operands and the operands' tangents, whose body
    /// is the primal body with the scalar tangent rules applied elementwise
    /// (the tangent of an elementwise function is the elementwise tangent).
    /// A scalar operand (broadcast) with a tangent gives a scalar tangent
    /// operand, broadcast in the same way.
    fn fused_op(&mut self, op: &OpData) -> Res<()> {
        let src = self.src;
        let [r] = op.results[..] else {
            return Err(self.malformed("`fused` has one result"));
        };
        self.copy_deep(op)?;
        let live: Vec<usize> = (0..op.operands.len())
            .filter(|i| self.tan_of(op.operands[*i]).is_some())
            .collect();
        if !self.active[r.index()] || live.is_empty() {
            return Ok(());
        }
        let body = self.single_block(op.regions[0])?;
        let ty = self.tangent_ty(r)?;
        let mut operands = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        for &j in &live {
            operands.push(self.tan_of(op.operands[j]).expect("live"));
        }
        let outer = self.b.insertion_block();
        let (new_op, results) = self.b.op(OpKind::Fused, operands, vec![ty], op.span);
        let region = self.b.add_region(new_op);
        let args = src.block(body).args.clone();
        let mut arg_tys: Vec<Type> = args.iter().map(|a| src.ty(*a).clone()).collect();
        for &j in &live {
            // The tangent of a body argument is a scalar of its dimension
            // over `dim(x)`.
            arg_tys.push(self.tangent_ty(args[j])?);
        }
        let blk = self.b.add_block(region, arg_tys);
        let new_args = self.b.func().block(blk).args.clone();
        for (j, a) in args.iter().enumerate() {
            self.map[a.index()] = Some(new_args[j]);
        }
        for (k, &j) in live.iter().enumerate() {
            self.tan[args[j].index()] = Some(new_args[args.len() + k]);
        }
        self.b.set_insertion_block(blk);
        self.in_body = true;
        let term = self.process_ops(body);
        self.in_body = false;
        let term = src.op(term?);
        let [y] = term.operands[..] else {
            return Err(self.malformed("the body of `fused` yields one value"));
        };
        let ty = self.tangent_or_zero(y, term.span)?;
        self.b.op(OpKind::Yield, vec![ty], vec![], term.span);
        self.b.set_insertion_block(outer);
        self.tan[r.index()] = Some(results[0]);
        // The body's values are only meaningful inside it.
        for a in &args {
            self.tan[a.index()] = None;
        }
        Ok(())
    }

    /// Copies `op` with its regions (an `if` or `while` arm, a `fused`
    /// body), without tangents: the primal computation only.
    pub fn copy_deep(&mut self, op: &OpData) -> Res<Vec<ValueId>> {
        let src = self.src;
        let operands = op
            .operands
            .iter()
            .map(|v| self.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        let tys = op.results.iter().map(|r| src.ty(*r).clone()).collect();
        let outer = self.b.insertion_block();
        let (new_op, results) = self.b.op(op.kind.clone(), operands, tys, op.span);
        for (old, new) in op.results.iter().zip(&results) {
            self.map[old.index()] = Some(*new);
            if let Some(n) = &src.value(*old).name {
                self.b.set_name(*new, n.clone());
            }
        }
        for &region in &op.regions {
            let blk = self.single_block(region)?;
            let r = self.b.add_region(new_op);
            let args = &src.block(blk).args;
            let arg_tys = args.iter().map(|a| src.ty(*a).clone()).collect();
            let nb = self.b.add_block(r, arg_tys);
            let new_args = self.b.func().block(nb).args.clone();
            for (a, n) in args.iter().zip(&new_args) {
                self.map[a.index()] = Some(*n);
            }
            self.b.set_insertion_block(nb);
            for id in &src.block(blk).ops {
                self.copy_deep(src.op(*id))?;
            }
        }
        self.b.set_insertion_block(outer);
        Ok(results)
    }
}
