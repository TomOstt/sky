//! Reverse mode: the gradient of a scalar function with respect to an array
//! parameter in one forward and one backward sweep.
//!
//! Forward mode gives a directional derivative per pass, so the gradient of
//! `f(p)` for `p: Vector[n]` costs `n` passes. Reverse mode runs `f`
//! forward once, then walks its operations backwards propagating the
//! *adjoint* `v̄ = ∂F/∂v` of every value `v` (`F` is the function's result),
//! which reaches the parameter as its gradient: a constant number of sweeps
//! however large `n` is.
//!
//! # Units
//!
//! The adjoint of `v` has dimension `dim(F)/dim(v)` and the shape of `v`
//! (the mirror image of a tangent's `dim(v)/dim(x)`). So the gradient of an
//! energy with respect to a position vector has dimension `J/m = N`: a
//! force, and a rule that gets a dimension wrong gives ill-typed SIR.
//!
//! # What it handles
//!
//! Straight-line code and `if`, which is the case a physical potential is
//! in. Every operation of the array rules (the table in `rules.rs` has the
//! tangent side; each adjoint rule below is its transpose) and calls.
//!
//! * `if`: the forward sweep copies it. The backward sweep runs an `if` on
//!   the same condition whose arm first *replays* the arm's forward
//!   operations (their values were computed inside the forward `if` and
//!   are out of scope), then runs the arm's backward sweep, and yields the
//!   adjoint contributions to the values defined outside. Replaying costs
//!   a second evaluation of the arm; it prints nothing twice because an
//!   arm that prints is refused.
//! * `call`: calls with array operands or results were inlined
//!   ([`crate::inline`]), so a remaining call is a scalar function; its
//!   partial derivatives come from its forward derivative
//!   `d<callee>/d<param>` (which may loop), computed in the forward sweep.
//!   A recursive function with an array argument uses the callee's
//!   gradient function.
//! * `while`: not supported when it carries a value the gradient depends
//!   on (a reverse sweep needs the values of every iteration, a tape). A
//!   loop the gradient does not depend on is copied. [`crate::gradient`]
//!   is the fallback for the rest: forward mode, one pass per entry.

use std::collections::{HashMap, HashSet};

use sky_diag::Span;
use sky_ir::{Effect, Function, FunctionBuilder, Module, OpData, OpKind, SliceAxis, Type, ValueId};
use sky_types::{Shape, ShapeDim};
use sky_units::Dimension;

use crate::analysis::analyze;
use crate::emit::{number_type, shaped_type, Diff, Res};
use crate::inline::inline_array_calls;
use crate::{derivative_name, derivative_name_for, AdError, Kind, Task};

/// The gradient function `d<f>/d<p>(params...) -> (value, ∂f/∂p)` of a
/// function with one scalar result, with respect to an array parameter.
pub(crate) fn build(module: &Module, f: &Function, wrt: usize) -> Res<(Function, Vec<Task>)> {
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
        Type::Number(n) if n.shape.rank() > 0 => n.clone(),
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
            why: "a gradient needs a function with one result".into(),
        });
    };
    let f_ty = match res_ty {
        Type::Number(n) if n.shape.rank() == 0 => n.clone(),
        Type::Unknown => {
            return Err(AdError::UnknownType {
                func: name,
                what: "the result".into(),
            })
        }
        _ => {
            return Err(AdError::BadResult {
                func: name,
                why: "a gradient needs a result that is a scalar number".into(),
            })
        }
    };

    let inlined = inline_array_calls(module, f)?;
    let f = &inlined;
    let out_name = derivative_name_for(f, wrt, Kind::Gradient);
    let d = Diff {
        module,
        src: f,
        b: FunctionBuilder::new(out_name.clone(), f.params.clone(), vec![], f.span),
        x_dim: x_ty.dim,
        active: analyze(f, wrt),
        map: vec![None; f.values.len()],
        tan: vec![None; f.values.len()],
        needs: Vec::new(),
    };
    let grad_ty = shaped_type(
        d.dim_div(f_ty.dim, x_ty.dim)?,
        x_ty.shape.clone(),
        x_ty.repr,
    );
    let mut b = FunctionBuilder::new(
        out_name,
        f.params.clone(),
        vec![res_ty.clone(), grad_ty.clone()],
        f.span,
    );
    b.set_effects(f.effects);
    let mut rev = Rev {
        adj: vec![None; f.values.len()],
        f_dim: f_ty.dim,
        partials: HashMap::new(),
        d,
    };
    rev.d.b = b;
    for (old, new) in f.param_values().iter().zip(rev.d.b.params()) {
        rev.d.map[old.index()] = Some(new);
        if let Some(n) = &f.value(*old).name {
            rev.d.b.set_name(new, n.clone());
        }
    }

    // Forward sweep.
    let entry = f.entry();
    let Some((&last, body)) = f.block(entry).ops.split_last() else {
        return Err(rev.d.malformed("a block has no terminator"));
    };
    for id in body {
        rev.fwd_op(f.op(*id))?;
    }
    let term = f.op(last);
    let [result] = term.operands[..] else {
        return Err(rev.d.malformed("`return` must have one operand per result"));
    };
    if term.kind != OpKind::Return {
        return Err(rev.d.malformed("the body must end in `return`"));
    }

    // Backward sweep, seeded with dF/dF = 1.
    if rev.d.active[result.index()] {
        let one = rev.d.konst_dimless(1.0, f_ty.repr, term.span);
        rev.adj[result.index()] = Some(one);
    }
    for id in body.iter().rev() {
        rev.bwd_op(f.op(*id))?;
    }
    let grad = match rev.adj[x.index()] {
        Some(g) => g,
        None => {
            let like = rev.d.primal(x)?;
            rev.d.zero_of(&grad_ty, Some(like), term.span)?
        }
    };
    let value = rev.d.primal(result)?;
    rev.d
        .b
        .op(OpKind::Return, vec![value, grad], vec![], term.span);
    Ok((rev.d.b.finish(), rev.d.needs))
}

struct Rev<'m> {
    /// The forward machinery: the builder, the value map and the
    /// activity flags, and the scalar tangent rules the adjoints of
    /// scalar functions are made from.
    d: Diff<'m>,
    /// Source value to its adjoint (`None` is a structural zero).
    adj: Vec<Option<ValueId>>,
    /// The dimension of the function's result.
    f_dim: Dimension,
    /// For a call with active arguments, keyed by its first result: for
    /// each such argument `k`, the partial derivatives of every result
    /// with respect to it.
    partials: HashMap<ValueId, Vec<(usize, Vec<ValueId>)>>,
}

/// One subscripted axis of an `index` or `slice`.
#[derive(Clone, Copy)]
enum Axis {
    /// The whole axis.
    Full,
    /// `A[i]`: one position (a value of the derivative function).
    At(ValueId),
    /// `A[lo:hi]`, with either bound absent.
    Range {
        lo: Option<ValueId>,
        hi: Option<ValueId>,
    },
}

impl Rev<'_> {
    // ----- types -----

    /// The type of the adjoint of `v`: `dim(F)/dim(v)` with `v`'s shape.
    fn adj_ty(&self, v: ValueId) -> Res<Type> {
        let dim = self.d.dim_div(self.f_dim, self.d.dim_of(v)?)?;
        Ok(shaped_type(dim, self.d.shape_of(v), self.d.repr_of(v)))
    }

    // ----- arithmetic on values of the derivative function -----
    //
    // Each computes its result type from its operands', so a rule states
    // the operation and the verifier checks the rule's dimensions.

    fn mul(&mut self, x: ValueId, y: ValueId, span: Span) -> Res<ValueId> {
        let (nx, ny) = (self.d.new_num(x)?, self.d.new_num(y)?);
        let dim = self.d.dim_mul(nx.dim, ny.dim)?;
        let shape = if nx.shape.rank() >= ny.shape.rank() {
            nx.shape.clone()
        } else {
            ny.shape.clone()
        };
        Ok(self.d.emit(
            OpKind::Mul,
            vec![x, y],
            shaped_type(dim, shape, nx.repr),
            span,
        ))
    }

    fn div(&mut self, x: ValueId, y: ValueId, span: Span) -> Res<ValueId> {
        let (nx, ny) = (self.d.new_num(x)?, self.d.new_num(y)?);
        let dim = self.d.dim_div(nx.dim, ny.dim)?;
        Ok(self.d.emit(
            OpKind::Div,
            vec![x, y],
            shaped_type(dim, nx.shape.clone(), nx.repr),
            span,
        ))
    }

    fn neg(&mut self, x: ValueId, span: Span) -> Res<ValueId> {
        let n = self.d.new_num(x)?;
        let ty = self.d.untagged(&n);
        Ok(self.d.emit(OpKind::Neg, vec![x], ty, span))
    }

    fn add(&mut self, x: ValueId, y: ValueId, span: Span) -> Res<ValueId> {
        let n = self.d.new_num(x)?;
        let ty = self.d.untagged(&n);
        Ok(self.d.emit(OpKind::Add, vec![x, y], ty, span))
    }

    fn transpose(&mut self, x: ValueId, span: Span) -> Res<ValueId> {
        let n = self.d.new_num(x)?;
        let dims = n.shape.dims();
        let shape = match dims {
            [m, k] => Shape::new(vec![k.clone(), m.clone()]),
            _ => {
                return Err(self
                    .d
                    .malformed("a transpose of something that is not a matrix"))
            }
        };
        Ok(self.d.emit(
            OpKind::Transpose,
            vec![x],
            shaped_type(n.dim, shape, n.repr),
            span,
        ))
    }

    fn matmul(&mut self, x: ValueId, y: ValueId, span: Span) -> Res<ValueId> {
        let (nx, ny) = (self.d.new_num(x)?, self.d.new_num(y)?);
        let dim = self.d.dim_mul(nx.dim, ny.dim)?;
        let (dx, dy) = (nx.shape.dims(), ny.shape.dims());
        let shape = match (dx, dy) {
            ([_], [_]) => Shape::scalar(),
            ([m, _], [_]) => Shape::vector(m.clone()),
            ([_], [_, n]) => Shape::vector(n.clone()),
            ([m, _], [_, n]) => Shape::matrix(m.clone(), n.clone()),
            _ => return Err(self.d.malformed("a product of things that are not arrays")),
        };
        Ok(self.d.emit(
            OpKind::MatMul,
            vec![x, y],
            shaped_type(dim, shape, nx.repr),
            span,
        ))
    }

    // ----- adjoint bookkeeping -----

    /// Adds `c` to the adjoint of source value `v`, if `v` has one (depends
    /// on the parameter).
    fn acc(&mut self, v: ValueId, c: ValueId, span: Span) -> Res<()> {
        if !self.d.active[v.index()] {
            return Ok(());
        }
        self.adj[v.index()] = Some(match self.adj[v.index()] {
            None => c,
            Some(old) => self.add(old, c, span)?,
        });
        Ok(())
    }

    // ----- forward sweep -----

    /// Emits the primal computation of `op` (the forward sweep, and the
    /// replay of an `if` arm).
    fn fwd_op(&mut self, op: &OpData) -> Res<()> {
        match &op.kind {
            OpKind::While => {
                if op.results.iter().any(|r| self.d.active[r.index()]) {
                    return Err(self.d.unsupported(
                        "while",
                        "reverse mode does not go through a loop that carries a value the \
                         gradient depends on",
                    ));
                }
                self.d.copy_deep(op).map(|_| ())
            }
            OpKind::Call { callee } => self.fwd_call(op, callee),
            _ => self.d.copy_deep(op).map(|_| ()),
        }
    }

    /// A call with an argument the gradient depends on calls the callee's
    /// derivative with respect to each such argument (the first also gives
    /// the value), keeping the partial derivatives for the backward sweep.
    fn fwd_call(&mut self, op: &OpData, callee: &str) -> Res<()> {
        let live: Vec<usize> = (0..op.operands.len())
            .filter(|i| self.d.active[op.operands[*i].index()])
            .collect();
        if live.is_empty() || op.results.is_empty() {
            return self.d.copy_deep(op).map(|_| ());
        }
        let Some(target) = self.d.module.function(callee) else {
            return Err(AdError::NoSuchFunction(callee.into()));
        };
        for r in &op.results {
            if self.d.shape_of(*r).rank() != 0 {
                return Err(self.d.unsupported(
                    "call",
                    &format!("`{callee}` returns an array and cannot be inlined (it is recursive)"),
                ));
            }
        }
        let n = op.results.len();
        let args = op
            .operands
            .iter()
            .map(|v| self.d.primal(*v))
            .collect::<Res<Vec<_>>>()?;
        let mut all = Vec::new();
        for (m, &k) in live.iter().enumerate() {
            let a = op.operands[k];
            let kind = if self.d.shape_of(a).rank() > 0 {
                Kind::Gradient
            } else {
                Kind::Forward
            };
            self.d.needs.push(Task {
                func: callee.to_string(),
                wrt: k,
                kind,
            });
            let mut tys: Vec<Type> = op
                .results
                .iter()
                .map(|r| self.d.src.ty(*r).clone())
                .collect();
            for r in &op.results {
                // dim(∂r/∂a_k) = dim(r) / dim(a_k), shaped like a_k.
                tys.push(shaped_type(
                    self.d.dim_div(self.d.dim_of(*r)?, self.d.dim_of(a)?)?,
                    self.d.shape_of(a),
                    self.d.repr_of(*r),
                ));
            }
            let (_, res) = self.d.b.op(
                OpKind::Call {
                    callee: derivative_name(target, k),
                },
                args.clone(),
                tys,
                op.span,
            );
            if m == 0 {
                for (r, v) in op.results.iter().zip(&res) {
                    self.d.map[r.index()] = Some(*v);
                    if let Some(name) = &self.d.src.value(*r).name {
                        self.d.b.set_name(*v, name.clone());
                    }
                }
            }
            all.push((k, res[n..].to_vec()));
        }
        self.partials.insert(op.results[0], all);
        Ok(())
    }

    // ----- backward sweep -----

    fn bwd_op(&mut self, op: &OpData) -> Res<()> {
        match &op.kind {
            OpKind::If => self.bwd_if(op),
            OpKind::While => Ok(()),
            OpKind::Call { .. } => self.bwd_call(op),
            _ => {
                let [r] = op.results[..] else {
                    return Ok(());
                };
                if !self.d.active[r.index()] {
                    return Ok(());
                }
                let Some(rb) = self.adj[r.index()] else {
                    return Ok(());
                };
                self.adjoint_rule(op, r, rb)
            }
        }
    }

    fn bwd_call(&mut self, op: &OpData) -> Res<()> {
        let Some(first) = op.results.first() else {
            return Ok(());
        };
        let Some(all) = self.partials.get(first).cloned() else {
            return Ok(());
        };
        for (k, partial) in all {
            let mut total: Option<ValueId> = None;
            for (i, r) in op.results.iter().enumerate() {
                let Some(rb) = self.adj[r.index()] else {
                    continue;
                };
                let term = self.mul(partial[i], rb, op.span)?;
                total = Some(match total {
                    None => term,
                    Some(t) => self.add(t, term, op.span)?,
                });
            }
            if let Some(t) = total {
                self.acc(op.operands[k], t, op.span)?;
            }
        }
        Ok(())
    }

    /// The values defined outside `op`'s regions, that its regions use and
    /// the gradient depends on, in a fixed order.
    fn free_vars(&self, op: &OpData) -> Vec<ValueId> {
        fn walk(
            f: &Function,
            op: &OpData,
            defined: &mut HashSet<ValueId>,
            used: &mut Vec<ValueId>,
        ) {
            for region in &op.regions {
                for blk in &f.region(*region).blocks {
                    defined.extend(f.block(*blk).args.iter().copied());
                    for id in &f.block(*blk).ops {
                        let inner = f.op(*id);
                        used.extend(inner.operands.iter().copied());
                        defined.extend(inner.results.iter().copied());
                        walk(f, inner, defined, used);
                    }
                }
            }
        }
        let (mut defined, mut used) = (HashSet::new(), Vec::new());
        walk(self.d.src, op, &mut defined, &mut used);
        let mut free: Vec<ValueId> = used
            .into_iter()
            .filter(|v| !defined.contains(v) && self.d.active[v.index()])
            .collect();
        free.sort_by_key(|v| v.index());
        free.dedup();
        free
    }

    /// Whether the regions of `op` write output: an arm that is replayed
    /// must not print twice.
    fn writes_output(&self, op: &OpData) -> bool {
        let f = self.d.src;
        op.regions.iter().any(|region| {
            f.region(*region).blocks.iter().any(|blk| {
                f.block(*blk).ops.iter().any(|id| {
                    let inner = f.op(*id);
                    let io = match &inner.kind {
                        OpKind::Print => true,
                        OpKind::Call { callee } => self
                            .d
                            .module
                            .function(callee)
                            .is_some_and(|g| g.effects.contains(Effect::Io)),
                        _ => false,
                    };
                    io || self.writes_output(inner)
                })
            })
        })
    }

    /// `if`: a backward `if` on the same condition, whose arms replay the
    /// forward arm, sweep it backwards, and yield the adjoint contribution
    /// to every outside value the arm uses.
    fn bwd_if(&mut self, op: &OpData) -> Res<()> {
        let src = self.d.src;
        let seeded: Vec<usize> = (0..op.results.len())
            .filter(|i| {
                let r = op.results[*i];
                self.d.active[r.index()] && self.adj[r.index()].is_some()
            })
            .collect();
        if seeded.is_empty() {
            return Ok(());
        }
        let free = self.free_vars(op);
        if free.is_empty() {
            return Ok(());
        }
        if self.writes_output(op) {
            return Err(self.d.unsupported(
                "if",
                "reverse mode replays the arms of an `if`, which would print twice",
            ));
        }
        let tys = free
            .iter()
            .map(|v| self.adj_ty(*v))
            .collect::<Res<Vec<_>>>()?;
        let cond = self.d.primal(op.operands[0])?;
        let outer = self.d.b.insertion_block();
        let (new_op, results) = self.d.b.op(OpKind::If, vec![cond], tys, op.span);
        for &region in &op.regions {
            let old = self.d.single_block(region)?;
            let r = self.d.b.add_region(new_op);
            let blk = self.d.b.add_block(r, vec![]);
            self.d.b.set_insertion_block(blk);
            let saved = std::mem::replace(&mut self.adj, vec![None; src.values.len()]);
            let Some((&last, body)) = src.block(old).ops.split_last() else {
                return Err(self.d.malformed("a block has no terminator"));
            };
            let term = src.op(last);
            for &i in &seeded {
                let seed = saved[op.results[i].index()].expect("seeded");
                self.acc(term.operands[i], seed, term.span)?;
            }
            for id in body {
                self.fwd_op(src.op(*id))?;
            }
            for id in body.iter().rev() {
                self.bwd_op(src.op(*id))?;
            }
            let mut ys = Vec::with_capacity(free.len());
            for &v in &free {
                ys.push(match self.adj[v.index()] {
                    Some(a) => a,
                    None => {
                        let ty = self.adj_ty(v)?;
                        let like = self.d.primal(v)?;
                        self.d.zero_of(&ty, Some(like), term.span)?
                    }
                });
            }
            self.d.b.op(OpKind::Yield, ys, vec![], term.span);
            self.adj = saved;
        }
        self.d.b.set_insertion_block(outer);
        for (v, r) in free.iter().zip(results) {
            self.acc(*v, r, op.span)?;
        }
        Ok(())
    }

    // ----- the adjoint rules -----

    /// The adjoint contributions of `r = op(operands)` to its operands,
    /// given the adjoint `rb` of `r`. Each rule is the transpose of the
    /// tangent rule in `rules.rs`.
    fn adjoint_rule(&mut self, op: &OpData, r: ValueId, rb: ValueId) -> Res<()> {
        let span = op.span;
        match &op.kind {
            // r = a + b: ā += r̄, b̄ += r̄.
            OpKind::Add => {
                let [a, b] = self.d.operands(op)?;
                self.acc(a, rb, span)?;
                self.acc(b, rb, span)
            }
            OpKind::Sub => {
                let [a, b] = self.d.operands(op)?;
                self.acc(a, rb, span)?;
                if self.d.active[b.index()] {
                    let n = self.neg(rb, span)?;
                    self.acc(b, n, span)?;
                }
                Ok(())
            }
            OpKind::Neg => {
                let [a] = self.d.operands(op)?;
                let n = self.neg(rb, span)?;
                self.acc(a, n, span)
            }
            // r = a * b: ā += r̄ b, b̄ += r̄ a. A scalar factor of an
            // array is used in every entry, so its adjoint sums r̄ ⊙ (the
            // array).
            OpKind::Mul => {
                let [a, b] = self.d.operands(op)?;
                let r_rank = self.d.shape_of(r).rank();
                for (x, y) in [(a, b), (b, a)] {
                    if !self.d.active[x.index()] {
                        continue;
                    }
                    let py = self.d.primal(y)?;
                    let c = if self.d.shape_of(x).rank() == r_rank {
                        self.mul(rb, py, span)?
                    } else {
                        self.d.hadamard_sum(rb, py, span)?
                    };
                    self.acc(x, c, span)?;
                }
                Ok(())
            }
            // r = a / b: ā += r̄ / b, b̄ += -(Σ r̄ ⊙ r) / b.
            OpKind::Div => {
                let [a, b] = self.d.operands(op)?;
                let (pb, pr) = (self.d.primal(b)?, self.d.primal(r)?);
                if self.d.active[a.index()] {
                    let c = self.div(rb, pb, span)?;
                    self.acc(a, c, span)?;
                }
                if self.d.active[b.index()] {
                    let s = if self.d.shape_of(r).rank() == 0 {
                        self.mul(rb, pr, span)?
                    } else {
                        self.d.hadamard_sum(rb, pr, span)?
                    };
                    let q = self.div(s, pb, span)?;
                    let c = self.neg(q, span)?;
                    self.acc(b, c, span)?;
                }
                Ok(())
            }
            OpKind::PowConst { .. } | OpKind::Pow | OpKind::Math(_) => {
                self.scalar_partials(op, r, rb)
            }
            OpKind::Inverse => {
                let [a] = self.d.operands(op)?;
                if self.d.shape_of(a).rank() == 0 {
                    return self.scalar_partials(op, r, rb);
                }
                // r = A⁻¹: Ā += -(rᵀ r̄ rᵀ).
                let pr = self.d.primal(r)?;
                let rt = self.transpose(pr, span)?;
                let left = self.matmul(rt, rb, span)?;
                let prod = self.matmul(left, rt, span)?;
                let c = self.neg(prod, span)?;
                self.acc(a, c, span)
            }
            OpKind::MatMul => self.adjoint_matmul(op, rb),
            // r = ‖a‖: ā += (r̄ / r) a. r̄ has dim(F)/dim(a), so r̄ / r is
            // dim(F)/dim(a)², and times `a` gives dim(F)/dim(a), the
            // adjoint's. At ‖a‖ = 0 the division fails, as in the tangent.
            OpKind::Norm => {
                let [a] = self.d.operands(op)?;
                let (pa, pr) = (self.d.primal(a)?, self.d.primal(r)?);
                let q = self.div(rb, pr, span)?;
                let c = self.mul(pa, q, span)?;
                self.acc(a, c, span)
            }
            // r = a·b: ā += r̄ b, b̄ += r̄ a, as for the product of two
            // vectors.
            OpKind::Dot => {
                let [a, b] = self.d.operands(op)?;
                let (pa, pb) = (self.d.primal(a)?, self.d.primal(b)?);
                if self.d.active[a.index()] {
                    let c = self.mul(pb, rb, span)?;
                    self.acc(a, c, span)?;
                }
                if self.d.active[b.index()] {
                    let c = self.mul(pa, rb, span)?;
                    self.acc(b, c, span)?;
                }
                Ok(())
            }
            OpKind::Transpose => {
                let [a] = self.d.operands(op)?;
                let c = if self.d.shape_of(a).rank() == 0 {
                    rb
                } else {
                    self.transpose(rb, span)?
                };
                self.acc(a, c, span)
            }
            // r = A⁻¹ b: b̄ += A⁻ᵀ r̄ = solve(A, r̄) (A is symmetric), and
            // Ā += -b̄ rᵀ.
            OpKind::Solve { method } => {
                let [a, b] = self.d.operands(op)?;
                let (pa, pr) = (self.d.primal(a)?, self.d.primal(r)?);
                let ty = self.adj_ty(b)?;
                let bbar = self
                    .d
                    .emit(OpKind::Solve { method: *method }, vec![pa, rb], ty, span);
                self.acc(b, bbar, span)?;
                if self.d.active[a.index()] {
                    let prod = if self.d.shape_of(r).rank() == 1 {
                        self.d.outer(bbar, pr, span)?
                    } else {
                        let rt = self.transpose(pr, span)?;
                        self.matmul(bbar, rt, span)?
                    };
                    let c = self.neg(prod, span)?;
                    self.acc(a, c, span)?;
                }
                Ok(())
            }
            // The entries of the literal are entries of r̄.
            OpKind::ArrayLiteral { cols } => {
                for (k, &e) in op.operands.iter().enumerate() {
                    if !self.d.active[e.index()] {
                        continue;
                    }
                    let k = k as u64;
                    let c = match cols {
                        None => self.d.element(rb, k, span)?,
                        Some(c) => self.d.element2(rb, k / c, k % c, span)?,
                    };
                    self.acc(e, c, span)?;
                }
                Ok(())
            }
            OpKind::Index | OpKind::Slice { .. } => {
                let base = op.operands[0];
                let c = self.scatter(op, r, rb)?;
                self.acc(base, c, span)
            }
            OpKind::Fused => self.adjoint_fused(op, r, rb),
            // Integers, Bools and constants carry no adjoint.
            OpKind::Zeros | OpKind::Identity | OpKind::Len => Ok(()),
            other => Err(self
                .d
                .unsupported(other.mnemonic(), "no adjoint rule for this operation")),
        }
    }

    /// The adjoints of a scalar function of scalars (`pow`, math, the
    /// scalar reciprocal) from its tangent rule: the partial derivative
    /// `∂r/∂a` is the tangent rule run with `ta = 1` measured against `a`
    /// itself (`dim(x) = dim(a)`), so it has dimension `dim(r)/dim(a)`,
    /// and `ā += r̄ ∂r/∂a`. One operand at a time.
    fn scalar_partials(&mut self, op: &OpData, r: ValueId, rb: ValueId) -> Res<()> {
        let span = op.span;
        let mut seen = HashSet::new();
        for &a in &op.operands {
            if !self.d.active[a.index()] || !seen.insert(a) {
                continue;
            }
            let saved = self.d.x_dim;
            self.d.x_dim = self.d.dim_of(a)?;
            let seed = self.d.konst_dimless(1.0, self.d.repr_of(a), span);
            self.d.tan[a.index()] = Some(seed);
            let partial = self.d.tangent_rule(op, r);
            self.d.tan[a.index()] = None;
            self.d.x_dim = saved;
            if let Some(p) = partial? {
                let c = self.mul(rb, p, span)?;
                self.acc(a, c, span)?;
            }
        }
        Ok(())
    }

    fn adjoint_matmul(&mut self, op: &OpData, rb: ValueId) -> Res<()> {
        let span = op.span;
        let [a, b] = self.d.operands(op)?;
        let (pa, pb) = (self.d.primal(a)?, self.d.primal(b)?);
        let (ra, rbk) = (self.d.shape_of(a).rank(), self.d.shape_of(b).rank());
        match (ra, rbk) {
            // r = a·b: ā += r̄ b, b̄ += r̄ a.
            (1, 1) => {
                if self.d.active[a.index()] {
                    let c = self.mul(pb, rb, span)?;
                    self.acc(a, c, span)?;
                }
                if self.d.active[b.index()] {
                    let c = self.mul(pa, rb, span)?;
                    self.acc(b, c, span)?;
                }
            }
            // r = A x: x̄ += Aᵀ r̄, Ā += r̄ xᵀ.
            (2, 1) => {
                if self.d.active[b.index()] {
                    let at = self.transpose(pa, span)?;
                    let c = self.matmul(at, rb, span)?;
                    self.acc(b, c, span)?;
                }
                if self.d.active[a.index()] {
                    let c = self.d.outer(rb, pb, span)?;
                    self.acc(a, c, span)?;
                }
            }
            // r = x B: x̄ += B r̄, B̄ += x r̄ᵀ.
            (1, 2) => {
                if self.d.active[a.index()] {
                    let c = self.matmul(pb, rb, span)?;
                    self.acc(a, c, span)?;
                }
                if self.d.active[b.index()] {
                    let c = self.d.outer(pa, rb, span)?;
                    self.acc(b, c, span)?;
                }
            }
            // r = A B: Ā += r̄ Bᵀ, B̄ += Aᵀ r̄.
            _ => {
                if self.d.active[a.index()] {
                    let bt = self.transpose(pb, span)?;
                    let c = self.matmul(rb, bt, span)?;
                    self.acc(a, c, span)?;
                }
                if self.d.active[b.index()] {
                    let at = self.transpose(pa, span)?;
                    let c = self.matmul(at, rb, span)?;
                    self.acc(b, c, span)?;
                }
            }
        }
        Ok(())
    }

    // ----- indexing -----

    /// The subscript of `op` (`index` or `slice`) as one item per axis of
    /// its base, with the operands mapped into the derivative function.
    fn axes(&self, op: &OpData, rank: usize) -> Res<Vec<Axis>> {
        let mut items = Vec::new();
        let mut next = 1;
        let mut take = |this: &Self| -> Res<ValueId> {
            let v = op
                .operands
                .get(next)
                .copied()
                .ok_or_else(|| this.d.malformed("a subscript has too few operands"))?;
            next += 1;
            this.d.primal(v)
        };
        match &op.kind {
            OpKind::Index => {
                for _ in 1..op.operands.len() {
                    items.push(Axis::At(take(self)?));
                }
            }
            OpKind::Slice { axes } => {
                for a in axes {
                    items.push(match *a {
                        SliceAxis::At => Axis::At(take(self)?),
                        SliceAxis::Range { start, stop } => {
                            let lo = if start { Some(take(self)?) } else { None };
                            let hi = if stop { Some(take(self)?) } else { None };
                            if lo.is_none() && hi.is_none() {
                                Axis::Full
                            } else {
                                Axis::Range { lo, hi }
                            }
                        }
                    });
                }
            }
            _ => return Err(self.d.malformed("not a subscript")),
        }
        items.resize(rank, Axis::Full);
        Ok(items)
    }

    /// The selection matrix of `axis` for an axis of length `len_axis` (the
    /// `axis`th of the base `pbase`): the columns of `identity(n)` the
    /// subscript picks, an `n×w` matrix with `w` entries in `kept`. `None`
    /// for the whole axis (the identity).
    fn selector(
        &mut self,
        axis: Axis,
        n_axis: usize,
        base_shape: &Shape,
        pbase: ValueId,
        kept: ShapeDim,
        span: Span,
    ) -> Res<Option<ValueId>> {
        let repr = self.d.new_num(pbase)?.repr;
        let n_dim = base_shape.dims()[n_axis].clone();
        let n = self
            .d
            .size_value(base_shape, n_axis, Some(pbase), repr, span)?;
        let dimless = |shape: Shape| shaped_type(Dimension::DIMENSIONLESS, shape, repr);
        let ident = || dimless(Shape::matrix(n_dim.clone(), n_dim.clone()));
        let (lo, hi) = match axis {
            Axis::Full => return Ok(None),
            Axis::At(i) => {
                let one = self.d.konst_dimless(1.0, repr, span);
                let next = self.d.emit(
                    OpKind::Add,
                    vec![i, one],
                    number_type(Dimension::DIMENSIONLESS, repr),
                    span,
                );
                (Some(i), Some(next))
            }
            Axis::Range { lo, hi } => (lo, hi),
        };
        let id = self.d.emit(OpKind::Identity, vec![n], ident(), span);
        let mut operands = vec![id];
        operands.extend(lo);
        operands.extend(hi);
        let sel = self.d.emit(
            OpKind::Slice {
                axes: vec![
                    SliceAxis::Range {
                        start: false,
                        stop: false,
                    },
                    SliceAxis::Range {
                        start: lo.is_some(),
                        stop: hi.is_some(),
                    },
                ],
            },
            operands,
            dimless(Shape::matrix(n_dim, kept)),
            span,
        );
        Ok(Some(sel))
    }

    /// The adjoint contribution of `r = A[...]` to `A`: an array of `A`'s
    /// shape that holds `r̄` where the subscript picked entries and zeros
    /// elsewhere. With `S` a selection matrix (`identity(n)` restricted to
    /// the picked columns) it is `Sᵣ R̄ S_cᵀ` for a matrix and `S r̄` for a
    /// vector; a position is a selection of one column, so `e_i r̄` is a
    /// `matmul` too.
    fn scatter(&mut self, op: &OpData, r: ValueId, rb: ValueId) -> Res<ValueId> {
        let span = op.span;
        let base = op.operands[0];
        let pbase = self.d.primal(base)?;
        let base_shape = self.d.shape_of(base);
        let rank = base_shape.rank();
        let axes = self.axes(op, rank)?;
        let r_dims: Vec<ShapeDim> = self.d.shape_of(r).dims().to_vec();
        // The length of each axis in the result (1 for a position: the
        // axis is a column of the selection).
        let mut kept_iter = r_dims.iter();
        let mut kept = Vec::new();
        for a in &axes {
            kept.push(match a {
                Axis::At(_) => ShapeDim::Known(1),
                _ => kept_iter
                    .next()
                    .cloned()
                    .ok_or_else(|| self.d.malformed("a subscript result has too few axes"))?,
            });
        }
        if rank == 1 {
            return Ok(match axes[0] {
                Axis::Full => rb,
                Axis::At(i) => {
                    // r̄ is a scalar: r̄ e_i, where e_i is row i of the identity.
                    let repr = self.d.new_num(pbase)?.repr;
                    let n = self.d.size_value(&base_shape, 0, Some(pbase), repr, span)?;
                    let n_dim = base_shape.dims()[0].clone();
                    let unit = |shape: Shape| shaped_type(Dimension::DIMENSIONLESS, shape, repr);
                    let id = self.d.emit(
                        OpKind::Identity,
                        vec![n],
                        unit(Shape::matrix(n_dim.clone(), n_dim.clone())),
                        span,
                    );
                    let e =
                        self.d
                            .emit(OpKind::Index, vec![id, i], unit(Shape::vector(n_dim)), span);
                    self.mul(e, rb, span)?
                }
                Axis::Range { .. } => {
                    let sel = self
                        .selector(axes[0], 0, &base_shape, pbase, kept[0].clone(), span)?
                        .expect("a range selects");
                    self.matmul(sel, rb, span)?
                }
            });
        }
        let row_sel = self.selector(axes[0], 0, &base_shape, pbase, kept[0].clone(), span)?;
        let col_sel = self.selector(axes[1], 1, &base_shape, pbase, kept[1].clone(), span)?;
        let at_row = matches!(axes[0], Axis::At(_));
        let at_col = matches!(axes[1], Axis::At(_));
        let rn = self.d.new_num(rb)?;
        let mut cur = match (at_row, at_col) {
            (true, true) => self.d.emit(
                OpKind::ArrayLiteral { cols: Some(1) },
                vec![rb],
                shaped_type(rn.dim, Shape::matrix(1u64, 1u64), rn.repr),
                span,
            ),
            (true, false) => self.d.as_matrix(rb, false, span)?,
            (false, true) => self.d.as_matrix(rb, true, span)?,
            (false, false) => rb,
        };
        if let Some(s) = row_sel {
            cur = self.matmul(s, cur, span)?;
        }
        if let Some(s) = col_sel {
            let st = self.transpose(s, span)?;
            cur = self.matmul(cur, st, span)?;
        }
        Ok(cur)
    }

    // ----- fused kernels -----

    /// `r = fused(operands){body}`: the adjoint of each operand `a_k` is
    /// `r̄ ⊙ ∂body/∂arg_k`, one `fused` kernel whose body is the primal body
    /// followed by the scalar tangent rules seeded on argument `k` (so the
    /// partial derivative is that kernel's tangent) and multiplied by the
    /// element of `r̄`. A broadcast scalar operand sums the result.
    fn adjoint_fused(&mut self, op: &OpData, r: ValueId, rb: ValueId) -> Res<()> {
        let src = self.d.src;
        let span = op.span;
        let body = self.d.single_block(op.regions[0])?;
        let args = src.block(body).args.clone();
        let rb_num = self.d.new_num(rb)?;
        for (k, &operand) in op.operands.iter().enumerate() {
            if !self.d.active[operand.index()] {
                continue;
            }
            let mut operands = op
                .operands
                .iter()
                .map(|v| self.d.primal(*v))
                .collect::<Res<Vec<_>>>()?;
            operands.push(rb);
            let dim = self.d.dim_div(self.f_dim, self.d.dim_of(operand)?)?;
            let ty = shaped_type(dim, self.d.shape_of(r), self.d.repr_of(r));
            let outer = self.d.b.insertion_block();
            let (new_op, results) = self.d.b.op(OpKind::Fused, operands, vec![ty], span);
            let region = self.d.b.add_region(new_op);
            let mut arg_tys: Vec<Type> = args.iter().map(|a| src.ty(*a).clone()).collect();
            arg_tys.push(number_type(rb_num.dim, rb_num.repr));
            let blk = self.d.b.add_block(region, arg_tys);
            let new_args = self.d.b.func().block(blk).args.clone();
            for (a, n) in args.iter().zip(&new_args) {
                self.d.map[a.index()] = Some(*n);
            }
            self.d.b.set_insertion_block(blk);
            // The partial derivative of the body with respect to argument k.
            let saved = self.d.x_dim;
            self.d.x_dim = self.d.dim_of(args[k])?;
            let seed = self.d.konst_dimless(1.0, self.d.repr_of(args[k]), span);
            self.d.tan[args[k].index()] = Some(seed);
            let term = src.op(self.d.process_ops(body)?);
            let [y] = term.operands[..] else {
                return Err(self.d.malformed("the body of `fused` yields one value"));
            };
            let partial = self.d.tangent_or_zero(y, term.span)?;
            let prod = self.mul(partial, new_args[args.len()], term.span)?;
            self.d.b.op(OpKind::Yield, vec![prod], vec![], term.span);
            self.d.x_dim = saved;
            self.clear_tangents(body);
            self.d.b.set_insertion_block(outer);
            let c = if self.d.shape_of(operand).rank() == 0 {
                self.d.sum_all(results[0], span)?
            } else {
                results[0]
            };
            self.acc(operand, c, span)?;
        }
        Ok(())
    }

    /// Forgets the tangents of the values of a `fused` body (they belong
    /// to one kernel).
    fn clear_tangents(&mut self, body: sky_ir::BlockId) {
        let src = self.d.src;
        for a in &src.block(body).args {
            self.d.tan[a.index()] = None;
        }
        for id in &src.block(body).ops {
            for r in &src.op(*id).results {
                self.d.tan[r.index()] = None;
            }
        }
    }
}
