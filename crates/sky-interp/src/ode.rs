//! `equation` and `solve` on the tree-walker (spec "Differential
//! equations"). Included by `eval.rs` as a child module, so it reaches the
//! interpreter's private state.
//!
//! # How a solve runs
//!
//! * `equation` binds an [`EquationValue`]: its syntax, and the functions in
//!   scope where it was written.
//! * `solve` evaluates the time bounds, the initial values and the options
//!   in the current scope, **flattens** the state variables into one `f64`
//!   vector (a scalar is one entry, a vector or matrix its entries
//!   row-major), and hands `sky-ode` a [`System`] whose right-hand side runs
//!   the equation's body on the interpreter: a frame holding `t` and the
//!   state variables, the local names evaluated in order, each derivative
//!   evaluated and written back into the flat vector. Everything the solver
//!   sees is in SI, as everywhere in the interpreter.
//! * If the equation is second order (`q' = v`, `v' = a(t, q)`, found by
//!   `sky_types::analyze_equation`, which the checker also uses), the state
//!   is laid out as `[q..., v...]` and declared
//!   [`Structure::SecondOrder`], so the solver may choose a symplectic
//!   method; otherwise the layout is declaration order and the system is
//!   [`Structure::General`].
//! * A conserved quantity is an expression over the state; the solver wants
//!   it as a plain function of `(t, y)` that outlives the call, so it is
//!   evaluated by a private interpreter that owns a snapshot of the
//!   module's names.
//! * The result is a [`SolutionValue`] with the final time, state and
//!   conserved values, and the solver's explanation (`explain(sol)`).
//!
//! A failure of the solver is a [`RuntimeError`] on the `solve`; a failure
//! of the equation's own arithmetic (a division by zero in a right-hand
//! side, say) is reported where it happened, with the `solve` as a
//! secondary location.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::io;
use std::rc::Rc;
use std::sync::Arc;

use sky_ast::{Block, Equation, EquationItem, Expr, Solve, SolveOption};
use sky_diag::Span;
use sky_ode::{Conserved, Options, SolveError, Structure, System};
use sky_units::Dimension;

use super::{Frame, Interp};
use crate::arith::{StackMeter, R};
use crate::value::{EquationValue, Function, OdeValue, SolutionValue};
use crate::{RuntimeEnv, RuntimeError, Value};

/// Where one state variable lives in the flat vector.
#[derive(Clone, Debug)]
struct Slot {
    name: String,
    /// `[]` for a scalar.
    dims: Vec<usize>,
    dim: Dimension,
    offset: usize,
    len: usize,
}

impl Slot {
    /// The variable's value in the flat state `y`. `y` is finite (the
    /// caller checked), so the value is too.
    fn value(&self, y: &[f64]) -> Value {
        let data = &y[self.offset..self.offset + self.len];
        if self.dims.is_empty() {
            Value::number(data[0], self.dim)
        } else {
            Value::array(self.dims.clone(), data.to_vec(), self.dim)
        }
    }
}

/// `t` and the state variables as a frame's variables.
fn frame_vars(
    base: &BTreeMap<String, Value>,
    time: (&str, Dimension),
    slots: &[Slot],
    t: f64,
    y: &[f64],
) -> BTreeMap<String, Value> {
    let mut vars = base.clone();
    vars.insert(time.0.to_string(), Value::number(t, time.1));
    for s in slots {
        vars.insert(s.name.clone(), s.value(y));
    }
    vars
}

fn all_finite(t: f64, y: &[f64]) -> bool {
    t.is_finite() && y.iter().all(|v| v.is_finite())
}

/// A `conserve` expression that can be evaluated as a function of `(t, y)`
/// at any time, with no reference to the interpreter that made it.
struct Conserve {
    src: String,
    expr: Expr,
    env: RefCell<RuntimeEnv>,
    scope: Arc<Function>,
    base: BTreeMap<String, Value>,
    time_name: String,
    time_dim: Dimension,
    slots: Vec<Slot>,
}

impl Conserve {
    fn eval(&self, t: f64, y: &[f64]) -> R<Value> {
        let non_finite = || {
            RuntimeError::new("the state is not finite", self.expr.span)
                .with_label("while evaluating this")
        };
        if !all_finite(t, y) {
            return Err(non_finite());
        }
        let vars = frame_vars(
            &self.base,
            (&self.time_name, self.time_dim),
            &self.slots,
            t,
            y,
        );
        let mut env = self.env.borrow_mut();
        let mut sink = io::sink();
        let solves = HashSet::new();
        let mut it = Interp {
            src: &self.src,
            env: &mut env,
            out: &mut sink,
            solves: &solves,
            frames: vec![Frame {
                vars,
                func: self.scope.clone(),
            }],
            stack: StackMeter::new(),
            derivatives: None,
        };
        it.eval(&self.expr)
    }

    /// The value as an `f64`, or the error.
    fn number(&self, t: f64, y: &[f64]) -> R<f64> {
        match self.eval(t, y)? {
            Value::Number { value, .. } => Ok(value),
            other => Err(RuntimeError::new(
                format!(
                    "a conserved quantity must be a number, found {}",
                    other.kind()
                ),
                self.expr.span,
            )
            .explain_uncertain(&other)),
        }
    }
}

/// The right-hand side of an equation, run on the interpreter.
struct OdeSystem<'i, 'a> {
    it: &'i mut Interp<'a>,
    steps: Vec<Step<'i>>,
    scope: Arc<Function>,
    /// The enclosing function's variables, if the solve is inside one.
    base: BTreeMap<String, Value>,
    time_name: String,
    time_dim: Dimension,
    slots: Vec<Slot>,
    structure: Structure,
    n: usize,
    /// The interpreter error behind the last `Err` from `rhs`.
    error: Option<RuntimeError>,
}

/// One line of the equation's body that runs at every evaluation.
enum Step<'e> {
    Local(String, &'e Expr),
    Derivative(usize, &'e Expr),
}

impl OdeSystem<'_, '_> {
    fn run_steps(&mut self, dydt: &mut [f64]) -> R<()> {
        for step in &self.steps {
            match step {
                Step::Local(name, value) => {
                    let v = self.it.eval(value)?;
                    if let Some(f) = self.it.frames.last_mut() {
                        f.vars.insert(name.clone(), v);
                    }
                }
                Step::Derivative(slot, value) => {
                    let v = self.it.eval(value)?;
                    let s = &self.slots[*slot];
                    write_derivative(s, &v, value.span, dydt)?;
                }
            }
        }
        Ok(())
    }
}

/// Stores the value of `s'` at the slot's place in `dydt`.
fn write_derivative(s: &Slot, v: &Value, span: Span, dydt: &mut [f64]) -> R<()> {
    let out = &mut dydt[s.offset..s.offset + s.len];
    match v {
        Value::Number { value, .. } if s.dims.is_empty() => out[0] = *value,
        Value::Array { dims, data, .. } if *dims == s.dims => out.copy_from_slice(data),
        other => {
            let want = if s.dims.is_empty() {
                "a number".to_string()
            } else {
                crate::array::shape_text(&s.dims)
            };
            let found = match other {
                Value::Array { dims, .. } => crate::array::shape_text(dims),
                o => o.kind().to_string(),
            };
            return Err(RuntimeError::new(
                format!("`{}'` must be {want}, found {found}", s.name),
                span,
            )
            .with_label(format!("this is {found}"))
            .explain_uncertain(other));
        }
    }
    Ok(())
}

/// Adds why an uncertain value cannot enter `solve`, and the way out, when
/// `v` is the uncertain value an error found.
trait ExplainUncertain {
    fn explain_uncertain(self, v: &Value) -> Self;
}

impl ExplainUncertain for RuntimeError {
    fn explain_uncertain(self, v: &Value) -> Self {
        if !matches!(v, Value::Uncertain(_)) {
            return self;
        }
        self.with_note("`solve` does not propagate uncertainties (`±`) yet")
            .with_help("pass `nominal(v)` for each uncertain value `v`, and solve again with `nominal(v) + uncertainty(v)` to see how far the result moves")
    }
}

impl System for OdeSystem<'_, '_> {
    fn dim(&self) -> usize {
        self.n
    }

    fn structure(&self) -> Structure {
        self.structure
    }

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), String> {
        if !all_finite(t, y) {
            return Err("the state is not finite".to_string());
        }
        let vars = frame_vars(
            &self.base,
            (&self.time_name, self.time_dim),
            &self.slots,
            t,
            y,
        );
        self.it.frames.push(Frame {
            vars,
            func: self.scope.clone(),
        });
        let result = self.run_steps(dydt);
        self.it.frames.pop();
        result.map_err(|e| {
            let message = e.message.clone();
            self.error.get_or_insert(e);
            message
        })
    }
}

impl Interp<'_> {
    /// `equation name(t: s): ...`: binds the name to the equation.
    pub(super) fn define_equation(&mut self, eq: &Equation) {
        let name = self.text(eq.name.span);
        let scope = Function {
            name: name.to_string(),
            def_span: eq.name.span,
            params: Vec::new(),
            body: Block {
                stmts: Vec::new(),
                span: eq.name.span,
            },
            captured: self.visible_functions(),
        };
        let value = EquationValue {
            equation: eq.clone(),
            scope: Arc::new(scope),
        };
        self.assign(name, Value::Ode(Arc::new(OdeValue::Equation(value))));
    }

    /// `sol = solve name from a to b with ...: options`.
    pub(super) fn solve_stmt(&mut self, span: Span, s: &Solve) -> R<()> {
        let eq_name = self.text(s.equation.span);
        let eqv = match self.lookup(eq_name) {
            Some(Value::Ode(o)) => match &**o {
                OdeValue::Equation(e) => e.clone(),
                other => return Err(not_an_equation(eq_name, other_kind(other), s.equation.span)),
            },
            Some(other) => return Err(not_an_equation(eq_name, other.kind(), s.equation.span)),
            None => {
                return Err(RuntimeError::new(
                    format!("cannot find equation `{eq_name}` in this scope"),
                    s.equation.span,
                )
                .with_label("not defined"));
            }
        };
        let eq = &eqv.equation;
        let info = sky_types::analyze_equation(self.src, eq);

        // The time span.
        let (t0, time_dim) = self.number_of("the start of the time span", &s.from)?;
        let (t1, _) = self.number_of("the end of the time span", &s.to)?;
        if t1 < t0 {
            return Err(RuntimeError::new(
                "the time span runs backwards: it ends before it starts",
                s.to.span,
            )
            .with_label("this is before the start")
            .with_secondary(s.from.span, "the start")
            .with_help("solving backwards in time is not supported yet"));
        }

        // The initial values, in the order of the equation's states.
        let mut initial: BTreeMap<String, (Value, Span)> = BTreeMap::new();
        for init in &s.initial {
            let v = self.eval(&init.value)?;
            initial.insert(self.text(init.name.span).to_string(), (v, init.value.span));
        }
        let mut slots: Vec<Slot> = Vec::new();
        for name in &info.states {
            let Some((v, vspan)) = initial.get(name) else {
                return Err(
                    RuntimeError::new(format!("no initial value for `{name}`"), s.keyword)
                        .with_label("initial values are missing"),
                );
            };
            let (dims, dim) = match v {
                Value::Number { dim, .. } => (Vec::new(), *dim),
                Value::Array { dims, dim, .. } => (dims.clone(), *dim),
                other => {
                    return Err(RuntimeError::new(
                        format!(
                            "the initial value of `{name}` must be a number, found {}",
                            other.kind()
                        ),
                        *vspan,
                    )
                    .explain_uncertain(other));
                }
            };
            let len = dims.iter().product::<usize>().max(1);
            slots.push(Slot {
                name: name.clone(),
                dims,
                dim,
                offset: 0,
                len,
            });
        }

        // The layout: declaration order, or `[q..., v...]` for a second
        // order system.
        let mut structure = Structure::General;
        let mut order: Vec<usize> = (0..slots.len()).collect();
        if let Some(pairs) = &info.second_order {
            let same_size = pairs.iter().all(|&(q, v)| slots[q].len == slots[v].len);
            if same_size {
                let n: usize = pairs.iter().map(|&(q, _)| slots[q].len).sum();
                order = pairs.iter().map(|&(q, _)| q).collect();
                order.extend(pairs.iter().map(|&(_, v)| v));
                structure = Structure::SecondOrder { n };
            }
        }
        let mut offset = 0;
        for &i in &order {
            slots[i].offset = offset;
            offset += slots[i].len;
        }
        let total = offset;
        let mut y0 = vec![0.0; total];
        for slot in &slots {
            let (v, _) = &initial[&slot.name];
            let data: &[f64] = match v {
                Value::Number { value, .. } => std::slice::from_ref(value),
                Value::Array { data, .. } => data,
                _ => &[],
            };
            y0[slot.offset..slot.offset + slot.len].copy_from_slice(data);
        }

        // The options.
        let time_name = info.time.clone();
        let base = self
            .frames
            .last()
            .map(|f| f.vars.clone())
            .unwrap_or_default();
        let mut opts = Options::default();
        let mut conserves: Vec<(String, Rc<Conserve>)> = Vec::new();
        let errors: Rc<RefCell<Option<RuntimeError>>> = Rc::new(RefCell::new(None));
        for option in &s.options {
            match option {
                SolveOption::Tolerance { value, .. } => {
                    let (tol, _) = self.number_of("the tolerance", value)?;
                    if !(tol > 0.0 && tol < 1.0) {
                        return Err(RuntimeError::new(
                            "the tolerance must be greater than 0 and less than 1",
                            value.span,
                        )
                        .with_label(format!("this is {tol}"))
                        .with_help("a tolerance is relative: `1e-9` asks for nine digits"));
                    }
                    opts.rtol = tol;
                    opts.atol = tol * 1e-3;
                }
                SolveOption::Conserve {
                    name,
                    value,
                    within,
                    ..
                } => {
                    let max_drift = match within {
                        Some(w) => {
                            let (x, _) = self.number_of("the allowed drift", w)?;
                            if x <= 0.0 {
                                return Err(RuntimeError::new(
                                    "the allowed drift must be greater than 0",
                                    w.span,
                                )
                                .with_label(format!("this is {x}")));
                            }
                            Some(x)
                        }
                        None => None,
                    };
                    let conserve = Rc::new(Conserve {
                        src: self.src.to_string(),
                        expr: value.clone(),
                        env: RefCell::new(self.env.clone()),
                        scope: eqv.scope.clone(),
                        base: base.clone(),
                        time_name: time_name.clone(),
                        time_dim,
                        slots: slots.clone(),
                    });
                    // Fail on the spot if the expression cannot be
                    // evaluated at the start, with its own diagnostic.
                    conserve.number(t0, &y0)?;
                    let name = self.text(name.span).to_string();
                    let (c, errs) = (conserve.clone(), errors.clone());
                    opts.conserved.push(Conserved {
                        name: name.clone(),
                        value: Box::new(move |t, y| match c.number(t, y) {
                            Ok(x) => x,
                            Err(e) => {
                                errs.borrow_mut().get_or_insert(e);
                                f64::NAN
                            }
                        }),
                        max_drift,
                    });
                    conserves.push((name, conserve));
                }
            }
        }

        // Solve.
        let scope = eqv.scope.clone();
        let steps: Vec<Step<'_>> = eq
            .items
            .iter()
            .filter_map(|item| match item {
                EquationItem::State { .. } => None,
                EquationItem::Local { name, value, .. } => {
                    Some(Step::Local(self.text(name.span).to_string(), value))
                }
                EquationItem::Derivative { name, value, .. } => {
                    let n = self.text(name.span);
                    slots
                        .iter()
                        .position(|s| s.name == n)
                        .map(|i| Step::Derivative(i, value))
                }
            })
            .collect();
        let mut system = OdeSystem {
            it: self,
            steps,
            scope,
            base,
            time_name: time_name.clone(),
            time_dim,
            slots: slots.clone(),
            structure,
            n: total,
            error: None,
        };
        let outcome = sky_ode::solve(&mut system, t0, t1, &y0, &opts);
        let rhs_error = system.error.take();
        if let Some(e) = errors.borrow_mut().take() {
            return Err(e.with_secondary(span, format!("while solving `{eq_name}`")));
        }
        let solution = match outcome {
            Ok(s) => s,
            Err(e) => {
                return Err(match (rhs_error, &e) {
                    (Some(inner), SolveError::Rhs { t, .. }) => inner
                        .with_secondary(span, format!("while solving `{eq_name}` (at t = {t} s)")),
                    (Some(inner), _) => {
                        inner.with_secondary(span, format!("while solving `{eq_name}`"))
                    }
                    (None, _) => solve_failure(&e, eq_name, span),
                });
            }
        };

        // The solution: final time, state, conserved values.
        let mut fields = vec![(time_name, Value::number(solution.t, time_dim))];
        for name in &info.states {
            if let Some(slot) = slots.iter().find(|s| s.name == *name) {
                fields.push((name.clone(), slot.value(&solution.y)));
            }
        }
        for (name, c) in &conserves {
            let v = c
                .eval(solution.t, &solution.y)
                .map_err(|e| e.with_secondary(span, format!("while solving `{eq_name}`")))?;
            fields.push((name.clone(), v));
        }
        let value = SolutionValue {
            equation: eq_name.to_string(),
            fields,
            explanation: solution.explanation,
        };
        self.assign(
            self.text(s.target.span),
            Value::Ode(Arc::new(OdeValue::Solution(value))),
        );
        Ok(())
    }

    /// A finite number and its dimension, or an error naming `what`.
    fn number_of(&mut self, what: &str, e: &Expr) -> R<(f64, Dimension)> {
        match self.eval(e)? {
            Value::Number { value, dim } => Ok((value, dim)),
            other => Err(RuntimeError::new(
                format!("{what} must be a number, found {}", other.kind()),
                e.span,
            )
            .explain_uncertain(&other)),
        }
    }

    /// `sol.name`. Only a solution has attributes.
    pub(super) fn attribute(&mut self, e: &Expr, base: &Expr, name: &sky_ast::Ident) -> R<Value> {
        let b = self.eval(base)?;
        let attr = self.text(name.span);
        match &b {
            Value::Ode(o) => match &**o {
                OdeValue::Solution(sol) => match sol.fields.iter().find(|(f, _)| f == attr) {
                    Some((_, v)) => Ok(v.clone()),
                    None => Err(RuntimeError::new(
                        format!("the solution of `{}` has no `{attr}`", sol.equation),
                        name.span,
                    )
                    .with_label("no such attribute")),
                },
                OdeValue::Equation(_) => Err(RuntimeError::new(
                    format!("an equation has no attribute `{attr}`"),
                    name.span,
                )
                .with_help("solve it first, then read the solution")),
            },
            other => Err(RuntimeError::new(
                format!(
                    "attribute `{attr}` is not supported by the interpreter: {} has no attributes",
                    other.kind()
                ),
                e.span,
            )),
        }
    }

    /// `explain(sol)`: the solver's explanation, one sentence per line.
    pub(super) fn explain_builtin(&mut self, call: &Expr, args: &[Expr]) -> R<Value> {
        let [arg] = args else {
            return Err(RuntimeError::new(
                format!("`explain` takes 1 argument, found {}", args.len()),
                call.span,
            ));
        };
        match self.eval(arg)? {
            Value::Ode(o) => match &*o {
                OdeValue::Solution(sol) => Ok(Value::Str(sol.explanation.join("\n"))),
                other => Err(RuntimeError::new(
                    format!("`explain` needs a solution, found {}", other_kind(other)),
                    arg.span,
                )),
            },
            other => Err(RuntimeError::new(
                format!("`explain` needs a solution, found {}", other.kind()),
                arg.span,
            )),
        }
    }
}

fn other_kind(o: &OdeValue) -> &'static str {
    match o {
        OdeValue::Equation(_) => "an equation",
        OdeValue::Solution(_) => "a solution",
    }
}

fn not_an_equation(name: &str, kind: &str, span: Span) -> RuntimeError {
    RuntimeError::new(format!("`{name}` is {kind}, not an equation"), span)
        .with_label("`solve` needs an equation")
}

/// The solver gave up: the diagnostic on the `solve`.
fn solve_failure(e: &SolveError, eq: &str, span: Span) -> RuntimeError {
    let err = RuntimeError::new(format!("cannot solve `{eq}`: {e}"), span);
    match e {
        SolveError::TooManySteps { .. } => err
            .with_label("the solver did not reach the end of the time span")
            .with_help("shorten the time span, loosen the tolerance, or check that the equations are not stiff or blowing up"),
        SolveError::NotFinite { .. } => err
            .with_label("the solution overflowed")
            .with_note("Sky numbers never become inf or NaN")
            .with_help("the equations may blow up in finite time, or the initial values may be far outside what they describe"),
        SolveError::Drift { name, .. } => err
            .with_label(format!("`{name}` was not conserved to the requested accuracy"))
            .with_help("tighten `tolerance`, or loosen `within`"),
        SolveError::Rhs { .. } => err.with_label("while evaluating the equations"),
        // The input errors and whatever the solver learns to report next:
        // its own words say what is wrong.
        _ => err.with_label("here"),
    }
}
