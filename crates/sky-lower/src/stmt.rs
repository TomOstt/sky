//! Lowering statements, and with them all of structured control flow.
//! The scheme is described in the crate docs; the names used here:
//!
//! * a **guard** is the `if not flag { rest of the block }` inserted after a
//!   statement that may have set an exit flag;
//! * a **merge** ([`Lower::close_if`]) gives an `if` its results: every
//!   variable or flag that differs from its value before the `if`, on some
//!   path, becomes a result, and each region yields its current values.

use std::collections::BTreeSet;

use sky_ast::{AssignTarget, Block, CompoundOp, Expr, ExprKind, Ident, Stmt, StmtKind};
use sky_diag::Span;
use sky_ir::{BlockId, OpId, OpKind, Type, ValueId};
use sky_types::Ty;

use crate::analysis::{
    assigned_vars, block_diverges, exits_stmt, exits_stmts, is_forever, stmt_diverges, Exits,
};
use crate::ctx::{assignable, brk_flag, join_ty, skip_flag, Lower, Needed, Scope, R, RET, RETVAL};
use crate::expr::ARRAYS;
use crate::LowerError;

/// One way control can reach the end of an `if`: the block it ends in, the
/// variables at that point, and whether the checker counts it as reaching
/// the code after the `if` (see [`Lower::if_stmt`]).
struct Path {
    block: BlockId,
    scope: Scope,
    contrib: bool,
}

/// Which names become results of a merge.
enum Mode<'a> {
    /// A source `if`: a name is a result if it was bound before the `if`
    /// (and changed), or is bound on every path that reaches the code after
    /// it, which is when the checker says it is bound.
    User,
    /// A guard: the skipping path is "already exited", so any name the
    /// code after needs (`needed`) is a result, with a stand-in value on
    /// the skipping path.
    Guard(&'a Needed),
}

/// A guard that has been opened and not yet closed.
struct GuardFrame {
    op: OpId,
    then_b: BlockId,
    else_b: BlockId,
    outer: BlockId,
    base: Scope,
    span: Span,
}

/// One `if`/`elif` of a chain: an `if` whose else arm holds the next.
struct Level {
    op: OpId,
    then_b: BlockId,
    else_b: BlockId,
    then_end: Scope,
    then_contrib: bool,
}

/// An error for a write through an index or attribute, which the checker
/// rejects and no checked program contains.
fn write_through(span: Span) -> LowerError {
    LowerError::new("cannot assign through an index or attribute", span).with_note(ARRAYS)
}

impl Lower<'_> {
    /// Lowers a block's statements in the current scope. After a statement
    /// that may have exited (set a flag), the rest of the block is guarded;
    /// after one that always exits the rest is dead and is dropped. Guards
    /// nest, so they are opened in a loop and closed in reverse: a long run
    /// of early exits is depth in the IR, not in lowering's recursion.
    pub(crate) fn block(&mut self, stmts: &[Stmt], needed: &Needed) -> R<()> {
        let mut frames: Vec<GuardFrame> = Vec::new();
        for (i, s) in stmts.iter().enumerate() {
            self.stmt(s, needed)?;
            let rest = &stmts[i + 1..];
            if stmt_diverges(s) || rest.is_empty() {
                break;
            }
            let Some(flag) = self.guard_flag(exits_stmt(s)) else {
                continue;
            };
            let span = rest[0].span.join(rest[rest.len() - 1].span);
            frames.push(self.open_guard(&flag, span)?);
        }
        while let Some(frame) = frames.pop() {
            self.close_guard(frame, needed);
        }
        Ok(())
    }

    /// The flag that says "the rest of this block must not run", if a
    /// statement with `exits` can set it: the loop's `skip` flag inside a
    /// loop, the function's `ret` flag outside.
    fn guard_flag(&self, exits: Exits) -> Option<String> {
        let d = self.st.loops;
        if d == 0 {
            exits.ret.then(|| RET.to_string())
        } else {
            exits.any().then(|| skip_flag(d))
        }
    }

    fn open_guard(&mut self, flag: &str, span: Span) -> R<GuardFrame> {
        let Some(&fv) = self.st.scope.get(flag) else {
            return self.err("internal error: an exit flag is not in scope", span);
        };
        let go = self.op1(OpKind::Not, vec![fv], Type::Bool, span);
        let base = self.st.scope.clone();
        let outer = self.st.b.at();
        let (op, _) = self.st.b.op(OpKind::If, vec![go], Vec::new(), span);
        let then_b = self.new_arm(op);
        let else_b = self.new_arm(op);
        self.st.b.set_at(then_b);
        self.st.nest += 1;
        Ok(GuardFrame {
            op,
            then_b,
            else_b,
            outer,
            base,
            span,
        })
    }

    fn close_guard(&mut self, f: GuardFrame, needed: &Needed) {
        self.st.nest -= 1;
        let then_end = std::mem::take(&mut self.st.scope);
        let paths = vec![
            Path {
                block: f.then_b,
                scope: then_end,
                contrib: true,
            },
            Path {
                block: f.else_b,
                scope: f.base.clone(),
                contrib: true,
            },
        ];
        let merged = self.close_if(f.op, f.span, paths, &f.base, Mode::Guard(needed));
        self.st.b.set_at(f.outer);
        self.st.scope = merged;
    }

    fn stmt(&mut self, s: &Stmt, needed: &Needed) -> R<()> {
        match &s.kind {
            StmtKind::Expr(e) => self.expr_stmt(e),
            StmtKind::Assign { target, value } => match target {
                AssignTarget::Name(id) => {
                    let v = self.expr(value)?;
                    self.bind(self.text(id.span), v);
                    Ok(())
                }
                _ => Err(write_through(s.span)),
            },
            StmtKind::Binding { name, value, .. } => match value {
                Some(value) => {
                    let v = self.expr(value)?;
                    self.bind(self.text(name.span), v);
                    Ok(())
                }
                None => self.declare_input(name, s.span),
            },
            StmtKind::CompoundAssign {
                target: AssignTarget::Name(id),
                op,
                value,
                ..
            } => self.compound(s.span, id, *op, value),
            StmtKind::CompoundAssign { .. } => Err(write_through(s.span)),
            StmtKind::If {
                branches,
                else_block,
            } => self.if_stmt(s.span, branches, else_block.as_ref(), needed),
            StmtKind::While { cond, body } => self.while_stmt(s.span, cond, body, needed),
            StmtKind::Def {
                name, params, body, ..
            } => self.lower_def(name, params, body, s.span),
            StmtKind::Return(value) => self.return_stmt(s.span, value.as_ref()),
            StmtKind::Pass => Ok(()),
            StmtKind::Break => {
                let d = self.st.loops;
                if d == 0 {
                    return self.err("`break` outside a loop", s.span);
                }
                let t = self.const_bool(true, s.span);
                self.st.scope.insert(brk_flag(d), t);
                self.st.scope.insert(skip_flag(d), t);
                Ok(())
            }
            StmtKind::Continue => {
                let d = self.st.loops;
                if d == 0 {
                    return self.err("`continue` outside a loop", s.span);
                }
                let t = self.const_bool(true, s.span);
                self.st.scope.insert(skip_flag(d), t);
                Ok(())
            }
        }
    }

    /// An expression statement. `print(...)` makes no value.
    fn expr_stmt(&mut self, e: &Expr) -> R<()> {
        if let ExprKind::Call { callee, args } = &e.kind {
            let builtin_print = matches!(&callee.kind, ExprKind::Name(id)
                if self.text(id.span) == "print"
                    && !self.checked.expr_types.contains_key(&callee.span));
            if builtin_print {
                return self.print_call(args, e.span);
            }
        }
        self.expr(e).map(|_| ())
    }

    /// `x op= value`: `x op value`, rebound to `x`.
    fn compound(&mut self, span: Span, id: &Ident, op: CompoundOp, value: &Expr) -> R<()> {
        let name = self.text(id.span);
        let cur = match self.st.scope.get(name) {
            Some(v) => *v,
            None => match sky_units::lookup_constant(name) {
                Some(c) => {
                    let ty = self.ty_at(id.span)?;
                    self.const_float(c.value, ty, id.span)
                }
                None => {
                    return self.err(
                        format!(
                            "cannot find `{}` in this scope",
                            sky_diag::display_capped(name)
                        ),
                        id.span,
                    )
                }
            },
        };
        // The checker typed the synthesized `x op value` at this span. Where
        // it could not (an unknown operand), `x` keeps the type it has:
        // `x op= value` may never change it.
        let ty = match self.checked.expr_ty(id.span.join(value.span)) {
            Some(t @ (Ty::Value(_) | Ty::Bool)) => crate::ctx::map_ty(&t).unwrap_or(Type::Unknown),
            _ => self.st.b.ty(cur).clone(),
        };
        let v = self.binary_with(op.binary(), cur, value, ty, span)?;
        self.bind(name, v);
        Ok(())
    }

    /// `A: PositiveDefinite Matrix[N, N]`: an input. Inputs are the
    /// parameters of `<main>`, in declaration order, typed with the
    /// declared type the checker recorded (tags included). Only a
    /// declaration at the top level of the file is an input.
    fn declare_input(&mut self, name: &Ident, span: Span) -> R<()> {
        if !(self.st.is_main && self.st.nest == 0) {
            return Err(LowerError::new(
                "an input declaration must be at the top level of the file",
                span,
            )
            .with_note("a name declared without a value is an input of the program; SIR gives inputs to `<main>` only"));
        }
        let Some(t) = self.checked.binding_types.get(&name.span) else {
            return self.err("internal error: the checker recorded no type here", span);
        };
        let ty = crate::ctx::map_ty(&self.checked.resolve_ty(t)).unwrap_or(Type::Unknown);
        let v = self.st.b.add_param(ty);
        self.bind(self.text(name.span), v);
        Ok(())
    }

    /// `return` that is not the last statement of a function: sets the
    /// `ret` flag and the result slot, and ends every enclosing loop.
    fn return_stmt(&mut self, span: Span, value: Option<&Expr>) -> R<()> {
        if !self.st.flags {
            return self.err("internal error: `return` without the return flags", span);
        }
        let v = match value {
            Some(e) => Some(self.expr(e)?),
            None => None,
        };
        let t = self.const_bool(true, span);
        self.st.scope.insert(RET.to_string(), t);
        if let (true, Some(v)) = (self.st.ret_value, v) {
            self.st.scope.insert(RETVAL.to_string(), v);
        }
        for k in 1..=self.st.loops {
            self.st.scope.insert(brk_flag(k), t);
            self.st.scope.insert(skip_flag(k), t);
        }
        Ok(())
    }

    /// The flags a statement with `exits` may set, at loop depth `d`.
    fn flag_names(&self, ex: Exits, d: usize) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut push = |name: String| {
            if !out.contains(&name) {
                out.push(name);
            }
        };
        if ex.ret {
            push(RET.to_string());
            if self.st.ret_value {
                push(RETVAL.to_string());
            }
            for k in 1..=d {
                push(brk_flag(k));
                push(skip_flag(k));
            }
        }
        if d >= 1 {
            if ex.brk {
                push(brk_flag(d));
                push(skip_flag(d));
            }
            if ex.cont {
                push(skip_flag(d));
            }
        }
        out
    }

    /// `if`/`elif`/`else`. Each arm is lowered from the scope before the
    /// `if`; the arms of the chain nest as `if c1 {..} else { if c2 {..}
    /// else {..} }`, built in a loop (a long `elif` chain is a loop here,
    /// depth only in the IR). Then the merges are closed from the innermost
    /// `if` outwards.
    #[inline(never)]
    fn if_stmt(
        &mut self,
        span: Span,
        branches: &[(Expr, Block)],
        else_block: Option<&Block>,
        needed: &Needed,
    ) -> R<()> {
        let base = self.st.scope.clone();
        let outer = self.st.b.at();
        // An arm reaches the code after the `if` if it can fall through, or
        // leaves the *loop* by `break`/`continue` (its variables then flow
        // to the loop's exit). One that only returns takes no part.
        let contrib = |stmts: &[Stmt]| {
            let ex = exits_stmts(stmts);
            !block_diverges(stmts) || ex.brk || ex.cont
        };
        // Names the arms that reach the code after the `if` all assign: a
        // guard inside an arm must keep them alive on its skipping path.
        let mut all: Option<BTreeSet<String>> = None;
        let mut intersect = |stmts: &[Stmt], src: &str| {
            let mut a = BTreeSet::new();
            assigned_vars(stmts, src, &mut a);
            all = Some(match all.take() {
                None => a,
                Some(prev) => prev.intersection(&a).cloned().collect(),
            });
        };
        for (_, b) in branches {
            if contrib(&b.stmts) {
                intersect(&b.stmts, self.src);
            }
        }
        match else_block {
            Some(b) if contrib(&b.stmts) => intersect(&b.stmts, self.src),
            Some(_) => {}
            // No `else`: the path where no condition held assigns nothing.
            None => intersect(&[], self.src),
        }
        let mut arm_needed = needed.clone();
        arm_needed.extend(all.unwrap_or_default());

        let mut levels: Vec<Level> = Vec::new();
        for (cond, block) in branches {
            // Each condition is evaluated on the path where the earlier
            // ones failed: in the previous else arm.
            self.st.scope = base.clone();
            let c = self.expr(cond)?;
            let (op, _) = self.st.b.op(OpKind::If, vec![c], Vec::new(), span);
            let then_b = self.new_arm(op);
            let else_b = self.new_arm(op);
            self.st.b.set_at(then_b);
            self.st.nest += 1;
            self.block(&block.stmts, &arm_needed)?;
            self.st.nest -= 1;
            levels.push(Level {
                op,
                then_b,
                else_b,
                then_end: std::mem::take(&mut self.st.scope),
                then_contrib: contrib(&block.stmts),
            });
            self.st.b.set_at(else_b);
        }
        self.st.scope = base.clone();
        if let Some(b) = else_block {
            self.st.nest += 1;
            self.block(&b.stmts, &arm_needed)?;
            self.st.nest -= 1;
        }
        let mut else_end = std::mem::take(&mut self.st.scope);
        let mut else_contrib = else_block.is_none_or(|b| contrib(&b.stmts));
        while let Some(level) = levels.pop() {
            let paths = vec![
                Path {
                    block: level.then_b,
                    scope: level.then_end,
                    contrib: level.then_contrib,
                },
                Path {
                    block: level.else_b,
                    scope: else_end,
                    contrib: else_contrib,
                },
            ];
            else_contrib |= level.then_contrib;
            else_end = self.close_if(level.op, span, paths, &base, Mode::User);
        }
        self.st.b.set_at(outer);
        self.st.scope = else_end;
        Ok(())
    }

    /// Gives `op` (an `if` with both arms lowered) its results and the
    /// arms their `yield`s, and returns the scope after it: `base` with
    /// each result bound. See [`Mode`] for which names are results.
    ///
    /// A result's type is the join of its values on the contributing paths
    /// (dimension and shape agree, tags are the common ones, as in the
    /// checker's merge). A path that has no value of that type for it (it
    /// returned, or never assigned the name) yields a stand-in that is
    /// never observed, because the flags make the code after skip it.
    #[inline(never)]
    fn close_if(
        &mut self,
        op: OpId,
        span: Span,
        paths: Vec<Path>,
        base: &Scope,
        mode: Mode<'_>,
    ) -> Scope {
        let mut cand: BTreeSet<&String> = BTreeSet::new();
        for p in &paths {
            for (name, v) in &p.scope {
                if base.get(name) != Some(v) {
                    cand.insert(name);
                }
            }
        }
        let mut results: Vec<(String, Type)> = Vec::new();
        for name in cand {
            let keep = base.contains_key(name)
                || match &mode {
                    Mode::User => {
                        let mut reaching = paths.iter().filter(|p| p.contrib).peekable();
                        reaching.peek().is_some() && reaching.all(|p| p.scope.contains_key(name))
                    }
                    Mode::Guard(needed) => needed.contains(name),
                };
            if !keep {
                continue;
            }
            // The flags and the result slot describe how the function or
            // loop is left, so every path counts for them, even one that
            // returned; a variable of a path that returned is dead.
            let is_flag = name.starts_with('<');
            let mut ty: Option<Type> = None;
            for p in paths.iter().filter(|p| is_flag || p.contrib) {
                if let Some(v) = p.scope.get(name) {
                    let t = self.st.b.ty(*v).clone();
                    ty = Some(match ty {
                        None => t,
                        Some(u) => join_ty(&u, &t),
                    });
                }
            }
            if ty.is_none() {
                ty = base.get(name).map(|v| self.st.b.ty(*v).clone());
            }
            if let Some(ty) = ty {
                results.push((name.clone(), ty));
            }
        }
        for p in &paths {
            self.st.b.set_at(p.block);
            let mut vals: Vec<ValueId> = Vec::with_capacity(results.len());
            for (name, ty) in &results {
                let v = match p.scope.get(name) {
                    Some(v) if assignable(self.st.b.ty(*v), ty) => *v,
                    _ => self.placeholder(ty, span),
                };
                vals.push(v);
            }
            self.st.b.op(OpKind::Yield, vals, Vec::new(), span);
        }
        let mut merged = base.clone();
        for (name, ty) in results {
            let v = self.st.b.add_result(op, ty);
            if !name.starts_with('<') {
                self.st.b.name_value(v, &name);
            }
            merged.insert(name, v);
        }
        merged
    }

    /// `while cond: body` as `scf.while`: the loop-carried values are the
    /// variables the body assigns (that are bound before the loop) plus the
    /// flags the body may set. `before` computes the condition, `after` is
    /// the body. Carried values get their final types after the body is
    /// lowered (the join of the initial and the next value).
    #[inline(never)]
    fn while_stmt(&mut self, span: Span, cond: &Expr, body: &Block, needed: &Needed) -> R<()> {
        let d = self.st.loops;
        let n = d + 1;
        let ex = exits_stmts(&body.stmts);
        let forever = is_forever(cond);
        let outer = self.st.b.at();
        let mut base = self.st.scope.clone();

        // What is carried. A name first assigned in the loop is carried
        // only by `while true`, the one loop after which the checker can
        // bind it (from the `break`s).
        let mut vars = BTreeSet::new();
        assigned_vars(&body.stmts, self.src, &mut vars);
        let mut carried: Vec<String> = vars
            .into_iter()
            .filter(|v| forever || base.contains_key(v))
            .collect();
        let brk = brk_flag(n);
        let skip = skip_flag(n);
        let mut flags = self.flag_names(ex, n);
        flags.retain(|f| *f != skip);
        let has_brk = flags.contains(&brk);
        carried.extend(flags);
        if has_brk {
            let f = self.const_bool(false, span);
            base.insert(brk.clone(), f);
        }
        let init: Vec<Option<ValueId>> = carried.iter().map(|c| base.get(c).copied()).collect();
        let arg_tys: Vec<Type> = init
            .iter()
            .map(|v| v.map_or(Type::Unknown, |v| self.st.b.ty(v).clone()))
            .collect();

        // The operation. A missing initial value (a name first assigned in
        // the loop) is filled in once its type is known.
        let hole = ValueId(u32::MAX);
        let operands: Vec<ValueId> = init.iter().map(|v| v.unwrap_or(hole)).collect();
        let (wop, _) = self.st.b.op(OpKind::While, operands, Vec::new(), span);
        let before_r = self.st.b.new_region(wop);
        let before_b = self.st.b.new_block(before_r, arg_tys.clone());
        let after_r = self.st.b.new_region(wop);
        let after_b = self.st.b.new_block(after_r, arg_tys);
        let before_args = self.st.b.block_args(before_b);
        let after_args = self.st.b.block_args(after_b);

        // before: the condition (not evaluated once the loop was told to stop).
        self.st.b.set_at(before_b);
        self.st.scope = base.clone();
        for (c, a) in carried.iter().zip(&before_args) {
            self.name_carried(c, *a);
            self.st.scope.insert(c.clone(), *a);
        }
        let c = if has_brk {
            let Some(bi) = carried.iter().position(|c| *c == brk) else {
                return self.err("internal error: the loop's break flag is not carried", span);
            };
            let go = self.op1(OpKind::Not, vec![before_args[bi]], Type::Bool, cond.span);
            let (ifop, _) = self.st.b.op(OpKind::If, vec![go], Vec::new(), cond.span);
            let then_b = self.new_arm(ifop);
            let else_b = self.new_arm(ifop);
            self.st.b.set_at(then_b);
            let cv = self.expr(cond)?;
            self.st.b.op(OpKind::Yield, vec![cv], Vec::new(), cond.span);
            self.st.b.set_at(else_b);
            let f = self.const_bool(false, cond.span);
            self.st.b.op(OpKind::Yield, vec![f], Vec::new(), cond.span);
            self.st.b.set_at(before_b);
            self.st.b.add_result(ifop, Type::Bool)
        } else {
            self.expr(cond)?
        };
        let mut forward = vec![c];
        forward.extend(&before_args);
        self.st
            .b
            .op(OpKind::Condition, forward, Vec::new(), cond.span);

        // after: the body.
        self.st.b.set_at(after_b);
        self.st.scope = base.clone();
        for (c, a) in carried.iter().zip(&after_args) {
            self.name_carried(c, *a);
            self.st.scope.insert(c.clone(), *a);
        }
        if ex.any() {
            let f = self.const_bool(false, span);
            self.st.scope.insert(skip.clone(), f);
        }
        let mut body_needed = needed.clone();
        if forever {
            body_needed.extend(carried.iter().cloned());
        }
        self.st.loops = n;
        self.st.nest += 1;
        self.block(&body.stmts, &body_needed)?;
        self.st.nest -= 1;
        self.st.loops = d;
        let mut next: Vec<ValueId> = Vec::with_capacity(carried.len());
        for (i, c) in carried.iter().enumerate() {
            let v = match self.st.scope.get(c) {
                Some(v) => *v,
                None => {
                    let ty = self.st.b.ty(after_args[i]).clone();
                    self.placeholder(&ty, span)
                }
            };
            next.push(v);
        }
        self.st.b.op(OpKind::Yield, next.clone(), Vec::new(), span);

        // Types, initial values that had to wait, results.
        self.st.b.set_at(outer);
        let mut after_scope = base;
        after_scope.remove(&brk);
        for (i, c) in carried.iter().enumerate() {
            let next_ty = self.st.b.ty(next[i]).clone();
            let ty = match init[i] {
                Some(v) => join_ty(&self.st.b.ty(v).clone(), &next_ty),
                None => next_ty,
            };
            self.st.b.set_ty(before_args[i], ty.clone());
            self.st.b.set_ty(after_args[i], ty.clone());
            if init[i].is_none() {
                let (kind, vty) = Self::placeholder_kind(&ty);
                let v = self.st.b.op1_before(wop, kind, vty, span);
                self.st.b.set_operand(wop, i, v);
            }
            let res = self.st.b.add_result(wop, ty);
            if *c != brk {
                if !c.starts_with('<') {
                    self.st.b.name_value(res, c);
                }
                after_scope.insert(c.clone(), res);
            }
        }
        self.st.scope = after_scope;
        Ok(())
    }

    /// Names a loop-carried block argument after its variable (flags stay
    /// anonymous).
    fn name_carried(&mut self, name: &str, arg: ValueId) {
        if !name.starts_with('<') {
            self.st.b.name_value(arg, name);
        }
    }
}
