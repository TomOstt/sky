//! Functions: lowering a `def`, and the module-level names a function reads.
//!
//! SIR functions have no globals in v1, but a Sky function reads module
//! names when it is *called* (the interpreter looks a name up in the
//! function's frame, then in the module). So a function gets one extra
//! trailing parameter per module-level name it can read, and every call
//! passes the module's current value of each. [`Lower::globals_of`] decides
//! which names.

use std::collections::{BTreeSet, HashSet};

use sky_ast::{AssertKind, AssignTarget, Block, Expr, ExprKind, IndexItem, Param, Stmt, StmtKind};
use sky_diag::Span;
use sky_ir::{OpKind, Type};
use sky_types::Ty;
use sky_units::lookup_constant;

use crate::analysis::{count_returns, live_prefix, returns_value};
use crate::build::Builder;
use crate::contract::has_ensures;
use crate::ctx::{map_ty, DefInfo, FnInfo, FnState, Lower, Needed, Scope, R, RET, RETVAL};
use sky_ast::Ident;

impl Lower<'_> {
    /// A name for the SIR function of the `def` `base`: the def's own name,
    /// or `name#2`, `name#3` for a later def of the same name (SIR function
    /// names are unique; `#` cannot occur in a Sky identifier).
    fn unique_name(&mut self, base: &str) -> String {
        let n = self.used_names.entry(base.to_string()).or_insert(0);
        *n += 1;
        if *n == 1 {
            base.to_string()
        } else {
            format!("{base}#{n}")
        }
    }

    /// Lowers every `def` in `stmts`, looking inside the blocks of `if`,
    /// `while` and `for` but not into a `def` (its nested `def`s are lowered
    /// with its body): [`crate::lower_defs`]. Nothing else is lowered, but a
    /// name a statement binds is bound in `<main>` to an `input` of the type
    /// the checker gave that binding, so each `def` sees the module names
    /// bound before it, with their types, exactly as when the whole file is
    /// lowered, and takes the same trailing parameters for them.
    pub(crate) fn defs_in(&mut self, stmts: &[Stmt]) -> R<()> {
        for s in stmts {
            match &s.kind {
                StmtKind::Def {
                    name, params, body, ..
                } => self.lower_def_or_record(name, params, body, s.span)?,
                StmtKind::Assign {
                    target: AssignTarget::Name(name),
                    ..
                }
                | StmtKind::Binding { name, .. } => self.bind_as_input(name),
                StmtKind::If {
                    branches,
                    else_block,
                } => {
                    for (_, b) in branches {
                        self.defs_in(&b.stmts)?;
                    }
                    if let Some(b) = else_block {
                        self.defs_in(&b.stmts)?;
                    }
                }
                StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                    self.defs_in(&body.stmts)?
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// [`Self::lower_def`], or, when failures are being collected, the
    /// failure recorded and everything the failed `def` started undone, so
    /// the next `def` lowers as if it had not been there.
    fn lower_def_or_record(
        &mut self,
        name: &Ident,
        params: &[Param],
        body: &Block,
        span: Span,
    ) -> R<()> {
        if self.failed.is_none() {
            return self.lower_def(name, params, body, span);
        }
        let mark = (
            self.stack.len(),
            self.out.len(),
            self.derivatives.len(),
            self.goals.len(),
        );
        let Err(e) = self.lower_def(name, params, body, span) else {
            return Ok(());
        };
        while self.stack.len() > mark.0 {
            if let Some(parent) = self.stack.pop() {
                self.st = parent;
            }
        }
        self.out.truncate(mark.1);
        self.derivatives.truncate(mark.2);
        self.goals.truncate(mark.3);
        if let Some(failed) = &mut self.failed {
            failed.push((name.span, e));
        }
        Ok(())
    }

    /// Binds `name` in `<main>` to an `input` of its binding's type, for
    /// [`Self::defs_in`]. A binding SIR has no type for (an equation, a
    /// solution) is left unbound: a `def` that reads it cannot be lowered
    /// anyway.
    fn bind_as_input(&mut self, name: &Ident) {
        let Some(t) = self.checked.binding_types.get(&name.span) else {
            return;
        };
        let Some(ty) = map_ty(&self.checked.resolve_ty(t)) else {
            return;
        };
        let text = self.text(name.span);
        let v = self.op1(
            OpKind::Input {
                name: text.to_string(),
            },
            Vec::new(),
            ty,
            name.span,
        );
        self.bind(text, v);
    }

    /// Lowers `def name(params): body` to a function of the module. The
    /// enclosing function's state is parked while it is built. The
    /// signature comes from the checker's final `FuncSig`: annotated
    /// parameters have their annotated type, the others are `Unknown`, and
    /// the single result is the return type (`Unknown` if the checker could
    /// not say, `Nothing` for a function that returns no value).
    pub(crate) fn lower_def(
        &mut self,
        name: &Ident,
        params: &[Param],
        body: &Block,
        span: Span,
    ) -> R<()> {
        let Some(sig) = self.checked.funcs.get(&name.span).cloned() else {
            return self.err(
                "internal error: the checker recorded no signature",
                name.span,
            );
        };
        let has_value = returns_value(&body.stmts);
        let ret_ty = match sig.ret() {
            Ty::Unknown if has_value => Type::Unknown,
            Ty::Unknown => Type::Nothing,
            Ty::Func(_) => {
                return self.err(
                    "a function that returns a function is not supported",
                    name.span,
                )
            }
            t => map_ty(&self.checked.resolve_ty(t)).unwrap_or(Type::Unknown),
        };
        let globals = self.globals_of(name.span);
        let fname = {
            let text = self.text(name.span);
            self.unique_name(text)
        };
        // Registered before the body is lowered, so it can call itself.
        self.fns.insert(
            name.span,
            FnInfo {
                name: fname.clone(),
                result: ret_ty.clone(),
                params: params.len(),
                globals: globals.clone(),
            },
        );

        // Early `return`s use flags; a function whose only `return` is its
        // last statement (or that has none) does not need them. Statements
        // after one that always leaves are dead and are not lowered.
        let live = live_prefix(&body.stmts);
        let nret = count_returns(live);
        let last_is_return = matches!(live.last().map(|s| &s.kind), Some(StmtKind::Return(_)));
        let flags = !(nret == 0 || (nret == 1 && last_is_return));
        let state = FnState {
            b: Builder::new(fname, span),
            scope: Default::default(),
            module_vals: Default::default(),
            loops: 0,
            nest: 0,
            is_main: false,
            flags,
            ret_value: flags && has_value,
            fn_name: Some(self.text(name.span).to_string()),
        };
        let parent = std::mem::replace(&mut self.st, state);
        self.stack.push(parent);

        for (i, p) in params.iter().enumerate() {
            let ty = match sig.param_ty(i) {
                Some(v) => {
                    map_ty(&self.checked.resolve_ty(&Ty::Value(v.clone()))).unwrap_or(Type::Unknown)
                }
                None => Type::Unknown,
            };
            let v = self.st.b.add_param(ty);
            self.bind(self.text(p.name.span), v);
        }
        for g in &globals {
            let ty = self.global_type(g);
            let v = self.st.b.add_param(ty);
            self.st.b.name_value(v, g);
            self.st.module_vals.insert(g.clone(), v);
            // A parameter of the same name shadows the module's.
            self.st.scope.entry(g.clone()).or_insert(v);
        }
        self.seed_constants(&body.stmts, span);
        // What an `ensures` sees: the function as it starts.
        let entry = has_ensures(live).then(|| self.st.scope.clone());
        if flags {
            let f = self.const_bool(false, span);
            self.st.scope.insert(RET.to_string(), f);
            if self.st.ret_value {
                let scope = self.st.scope.clone();
                let v = self.stand_in(&ret_ty, span, &scope)?;
                self.st.scope.insert(RETVAL.to_string(), v);
            }
        }

        self.lower_body(live, span, entry.map(|e| (e, has_value)))?;
        self.st.b.set_results(vec![ret_ty]);
        let Some(parent) = self.stack.pop() else {
            return self.err("internal error: lost the enclosing function", span);
        };
        let done = std::mem::replace(&mut self.st, parent);
        self.out.push((name.span.start, done.b.finish()));
        Ok(())
    }

    /// Puts the predefined constants that `stmts` (a function's body, or the
    /// file) bind inside an `if`, `while` or `for` into scope as literals, so
    /// that they are ordinary variables: the merge after the block, and the
    /// loop that carries them, treat them like any name bound before. A
    /// constant read as a bare literal is only right where no path binds it
    /// (see [`Lower::name`]); the interpreter keeps constants in the same
    /// environment as variables, so a rebinding on one path changes the
    /// value after the block. Names already in scope (a parameter, a module
    /// name the function takes) keep their value.
    pub(crate) fn seed_constants(&mut self, stmts: &[Stmt], span: Span) {
        let mut names = BTreeSet::new();
        crate::analysis::conditionally_bound(stmts, self.src, &mut names);
        for n in names {
            if self.st.scope.contains_key(&n) {
                continue;
            }
            if let Some(c) = lookup_constant(&n) {
                let v = self.const_float(c.value, Type::scalar(c.dim), span);
                self.bind(&n, v);
            }
        }
    }

    /// A function body: the statements, then `return`.
    pub(crate) fn lower_body(
        &mut self,
        stmts: &[Stmt],
        span: Span,
        ensures: Option<(Scope, bool)>,
    ) -> R<()> {
        let needed = Needed::new();
        let value = if self.st.flags {
            self.block(stmts, &needed)?;
            match self.st.scope.get(RETVAL) {
                Some(v) => *v,
                None => self.op1(OpKind::ConstNothing, Vec::new(), Type::Nothing, span),
            }
        } else {
            // The last statement is the only `return`, if there is one.
            let (prefix, tail) = match stmts.split_last() {
                Some((last, prefix)) => match &last.kind {
                    StmtKind::Return(v) => (prefix, Some(v)),
                    _ => (stmts, None),
                },
                None => (stmts, None),
            };
            self.block(prefix, &needed)?;
            match tail {
                Some(Some(e)) => self.expr(e)?,
                _ => self.op1(OpKind::ConstNothing, Vec::new(), Type::Nothing, span),
            }
        };
        if let Some((entry, has_value)) = ensures {
            self.lower_ensures(stmts, entry, has_value.then_some(value))?;
        }
        self.st.b.op(OpKind::Return, vec![value], Vec::new(), span);
        Ok(())
    }

    /// The type of module-level name `name` as a parameter: what the module
    /// holds now (the function is being defined), or the constant's.
    fn global_type(&self, name: &str) -> Type {
        let m = self.module_state();
        if let Some(v) = m.scope.get(name) {
            return m.b.ty(*v).clone();
        }
        match lookup_constant(name) {
            Some(c) => Type::scalar(c.dim),
            None => Type::Unknown,
        }
    }

    /// The module-level names the function whose `def` name is at `def`
    /// takes as extra trailing parameters, sorted: the names it reads, or
    /// that a function it calls (transitively) reads, that the module has
    /// bound. A predefined constant nobody rebound is a literal wherever it
    /// is read, so it is not a parameter.
    pub(crate) fn globals_of(&self, def: Span) -> Vec<String> {
        let mut seen: HashSet<Span> = HashSet::new();
        let mut todo = vec![def];
        let mut names: BTreeSet<&String> = BTreeSet::new();
        while let Some(d) = todo.pop() {
            if !seen.insert(d) {
                continue;
            }
            if let Some(info) = self.defs.get(&d) {
                names.extend(info.direct.iter());
                todo.extend(info.callees.iter().copied());
            }
        }
        let module = &self.module_state().scope;
        names
            .into_iter()
            .filter(|n| {
                self.mod_assigned.contains(*n)
                    && (module.contains_key(*n) || lookup_constant(n).is_some())
            })
            .cloned()
            .collect()
    }

    // ----- the pre-pass -----

    /// Records every `def` of the file (however deeply nested) with the
    /// names its body may read and the functions it calls.
    pub(crate) fn collect_defs(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match &s.kind {
                StmtKind::Def {
                    name, params, body, ..
                } => {
                    let info = self.scan_def(params, body);
                    self.defs.insert(name.span, info);
                    self.collect_defs(&body.stmts);
                }
                StmtKind::If {
                    branches,
                    else_block,
                } => {
                    for (_, b) in branches {
                        self.collect_defs(&b.stmts);
                    }
                    if let Some(b) = else_block {
                        self.collect_defs(&b.stmts);
                    }
                }
                StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                    self.collect_defs(&body.stmts)
                }
                _ => {}
            }
        }
    }

    /// The names assigned or declared at module level anywhere in the file
    /// (not inside a `def`).
    pub(crate) fn collect_module_names(&mut self, stmts: &[Stmt]) {
        let mut names = BTreeSet::new();
        crate::analysis::assigned_vars(stmts, self.src, &mut names);
        for s in stmts {
            if let StmtKind::Binding { name, .. } = &s.kind {
                names.insert(self.text(name.span).to_string());
            }
        }
        self.mod_assigned.extend(names);
        // `assigned_vars` looks into `if` and `while`, but a bare
        // declaration there is an error anyway.
    }

    /// What a function body may read from the module. Reads are the names
    /// used in the body that are not parameters and not already assigned by
    /// an earlier top-level statement of the body (an assignment in a
    /// branch or loop does not count: the name may still be read from the
    /// module first). That over-approximates: a name it wrongly includes
    /// costs an unused parameter, never a wrong value.
    fn scan_def(&self, params: &[Param], body: &Block) -> DefInfo {
        let mut defined: BTreeSet<String> = params
            .iter()
            .map(|p| self.text(p.name.span).to_string())
            .collect();
        let mut direct = BTreeSet::new();
        let mut callees = Vec::new();
        for s in &body.stmts {
            self.scan_stmt(s, &defined, &mut direct, &mut callees);
            match &s.kind {
                StmtKind::Assign {
                    target: AssignTarget::Name(id),
                    ..
                }
                | StmtKind::CompoundAssign {
                    target: AssignTarget::Name(id),
                    ..
                }
                | StmtKind::Binding {
                    name: id,
                    value: Some(_),
                    ..
                } => {
                    defined.insert(self.text(id.span).to_string());
                }
                _ => {}
            }
        }
        DefInfo { direct, callees }
    }

    fn scan_block(
        &self,
        b: &Block,
        defined: &BTreeSet<String>,
        direct: &mut BTreeSet<String>,
        callees: &mut Vec<Span>,
    ) {
        for s in &b.stmts {
            self.scan_stmt(s, defined, direct, callees);
        }
    }

    fn scan_stmt(
        &self,
        s: &Stmt,
        defined: &BTreeSet<String>,
        direct: &mut BTreeSet<String>,
        callees: &mut Vec<Span>,
    ) {
        match &s.kind {
            StmtKind::Expr(e) => self.scan_expr(e, defined, direct, callees),
            StmtKind::Assign { target, value } => {
                self.scan_expr(value, defined, direct, callees);
                self.scan_target(target, defined, direct, callees);
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                self.scan_expr(value, defined, direct, callees);
                if let AssignTarget::Name(id) = target {
                    let n = self.text(id.span);
                    if !defined.contains(n) {
                        direct.insert(n.to_string());
                    }
                } else {
                    self.scan_target(target, defined, direct, callees);
                }
            }
            StmtKind::Binding { value, .. } => {
                if let Some(v) = value {
                    self.scan_expr(v, defined, direct, callees);
                }
            }
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (c, b) in branches {
                    self.scan_expr(c, defined, direct, callees);
                    self.scan_block(b, defined, direct, callees);
                }
                if let Some(b) = else_block {
                    self.scan_block(b, defined, direct, callees);
                }
            }
            StmtKind::While { cond, body } => {
                self.scan_expr(cond, defined, direct, callees);
                self.scan_block(body, defined, direct, callees);
            }
            StmtKind::For { iter, body, .. } => {
                self.scan_expr(iter, defined, direct, callees);
                self.scan_block(body, defined, direct, callees);
            }
            StmtKind::Return(v) => {
                if let Some(v) = v {
                    self.scan_expr(v, defined, direct, callees);
                }
            }
            // `result` in an `ensures` is the returned value, not a name of
            // the module.
            StmtKind::Assert { kind, cond, .. } => {
                if *kind == AssertKind::Ensures {
                    let mut with_result = defined.clone();
                    with_result.insert("result".to_string());
                    self.scan_expr(cond, &with_result, direct, callees);
                } else {
                    self.scan_expr(cond, defined, direct, callees);
                }
            }
            // Reads the name, like any use of it.
            StmtKind::Accuracy { name, .. } => {
                let n = self.text(name.span);
                if !defined.contains(n) {
                    direct.insert(n.to_string());
                }
            }
            // Only the checker reads a `cost` line (its names are sizes).
            StmtKind::Cost { .. } => {}
            // A nested `def` is its own function. Equations do not lower
            // (see `stmt`), so what they read does not matter.
            StmtKind::Def { .. }
            | StmtKind::Equation(_)
            | StmtKind::Solve(_)
            | StmtKind::Pass
            | StmtKind::Break
            | StmtKind::Continue => {}
        }
    }

    fn scan_target(
        &self,
        t: &AssignTarget,
        defined: &BTreeSet<String>,
        direct: &mut BTreeSet<String>,
        callees: &mut Vec<Span>,
    ) {
        match t {
            AssignTarget::Name(_) => {}
            AssignTarget::Index { base, items, .. } => {
                self.scan_expr(base, defined, direct, callees);
                self.scan_items(items, defined, direct, callees);
            }
            AssignTarget::Attribute { base, .. } => {
                self.scan_expr(base, defined, direct, callees);
            }
        }
    }

    fn scan_items(
        &self,
        items: &[IndexItem],
        defined: &BTreeSet<String>,
        direct: &mut BTreeSet<String>,
        callees: &mut Vec<Span>,
    ) {
        for item in items {
            match item {
                IndexItem::Expr(e) => self.scan_expr(e, defined, direct, callees),
                IndexItem::Slice { start, stop, .. } => {
                    for b in [start, stop].into_iter().flatten() {
                        self.scan_expr(b, defined, direct, callees);
                    }
                }
            }
        }
    }

    fn scan_expr(
        &self,
        e: &Expr,
        defined: &BTreeSet<String>,
        direct: &mut BTreeSet<String>,
        callees: &mut Vec<Span>,
    ) {
        match &e.kind {
            ExprKind::Name(id) => {
                let n = self.text(id.span);
                if !defined.contains(n) {
                    direct.insert(n.to_string());
                }
            }
            ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => {}
            ExprKind::Unary { operand, .. }
            | ExprKind::Inverse { operand, .. }
            | ExprKind::Transpose { operand, .. } => {
                self.scan_expr(operand, defined, direct, callees)
            }
            ExprKind::Paren(inner) => self.scan_expr(inner, defined, direct, callees),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.scan_expr(lhs, defined, direct, callees);
                self.scan_expr(rhs, defined, direct, callees);
            }
            ExprKind::BoolOp { operands, .. } => {
                for o in operands {
                    self.scan_expr(o, defined, direct, callees);
                }
            }
            ExprKind::IfElse { then, cond, else_ } => {
                for x in [then, cond, else_] {
                    self.scan_expr(x, defined, direct, callees);
                }
            }
            ExprKind::As { value, .. } => self.scan_expr(value, defined, direct, callees),
            ExprKind::PlusMinus { value, sigma, .. } => {
                self.scan_expr(value, defined, direct, callees);
                self.scan_expr(sigma, defined, direct, callees);
            }
            ExprKind::Array(elements) => {
                for el in elements {
                    self.scan_expr(el, defined, direct, callees);
                }
            }
            ExprKind::Index { base, items } => {
                self.scan_expr(base, defined, direct, callees);
                self.scan_items(items, defined, direct, callees);
            }
            ExprKind::Attribute { base, .. } => self.scan_expr(base, defined, direct, callees),
            ExprKind::Call { callee, args } => {
                // A callee that is a name is a function (or a builtin), not
                // a variable read; the checker says which function.
                let mut target = &**callee;
                while let ExprKind::Paren(inner) = &target.kind {
                    target = inner;
                }
                if matches!(target.kind, ExprKind::Name(_)) {
                    if let Some(Ty::Func(sig)) = self.checked.expr_types.get(&target.span) {
                        callees.push(sig.def_span());
                    }
                } else if let Some(def) = self.derivative_def(target) {
                    // `derivative(f, x)(...)` calls (a derivative of) `f`;
                    // its arguments name `f` and a parameter, not variables.
                    callees.push(def);
                } else {
                    self.scan_expr(callee, defined, direct, callees);
                }
                for a in args {
                    self.scan_expr(a, defined, direct, callees);
                }
            }
        }
    }
}
