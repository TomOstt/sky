//! Differential equations in the checker: `equation` and `solve` (spec
//! "Differential equations").
//!
//! # What is checked
//!
//! * In `equation name(t: T):` every state variable has a declared type
//!   (`r: Vector[2] m`, a fixed size), and every state variable has exactly
//!   one derivative equation `r' = ...`. The derivative of `x` has the
//!   dimension of `x` divided by that of `t`, and the shape of `x`, so
//!   `v' = <velocity>` for a velocity `v` is the error the spec shows. The
//!   right sides see the state variables, `t`, the local names defined
//!   above them, and the names in scope where the equation is written.
//! * In `sol = solve name from a to b with x = ..., ...:` the bounds have
//!   the dimension of `t`; every state variable gets one initial value of
//!   its type; `tolerance` is a plain number; a `conserve` expression is a
//!   scalar over the state variables (and `t`), and its dimension is what
//!   `sol.<name>` will have.
//! * `sol.name` reads the final `t`, a final state variable, or the final
//!   value of a conserved quantity, typed like the thing it reads.
//!
//! Nothing here evaluates user code.
//!
//! An equation is a value the program can name (it is bound like a `def`),
//! so a REPL can define one on a line and solve it on the next; the
//! [`OdeTy`] in the environment carries what `solve` needs.

use std::collections::BTreeSet;
use std::sync::Arc;

use sky_ast::{Equation, EquationItem, Expr, ExprKind, Ident, Solve, SolveOption};
use sky_diag::{display_capped as quote, Diagnostic, Span};
use sky_units::Dimension;

use crate::check::{Checker, Ty, ValueTy};
use crate::display::a_dimension;
use crate::ode_analysis::{analyze_equation, name_uses, plus_minus_uses, EquationInfo};
use crate::provenance::label_dim;
use crate::rules::TensorType;
use crate::shape::{Shape, ShapeContext, ShapeDim};

/// What an `equation` or the result of a `solve` is, for the checker.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum OdeTy {
    Equation(Box<EquationSig>),
    Solution(SolutionSig),
}

/// An equation as `solve` sees it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EquationSig {
    pub name: String,
    /// The independent variable's name and dimension.
    pub time_name: String,
    pub time: Dimension,
    /// The state variables in declaration order.
    pub states: Vec<StateSig>,
    /// The structure read from the source (see [`EquationInfo`]).
    pub info: EquationInfo,
    /// The names from outside its right sides read (each once, where it is
    /// first read), and the `±`s written in them: they are evaluated by each
    /// `solve`, which cannot take an uncertain number.
    pub reads: Vec<(String, Span)>,
    pub plus_minus: Vec<Span>,
}

/// A state variable: its name, dimension and (fixed) shape.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StateSig {
    pub name: String,
    pub dim: Dimension,
    pub shape: Shape,
    /// The span of its declared type, for labels.
    pub decl: Span,
}

impl StateSig {
    fn ty(&self) -> Ty {
        Ty::value(self.dim, TensorType::untagged(self.shape.clone()))
    }
}

/// What `sol` holds: attributes, each a final value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SolutionSig {
    pub equation: String,
    /// The final time, each state variable, and each conserved quantity.
    pub fields: Vec<(String, ValueTy)>,
}

impl OdeTy {
    /// `equation `orbit`` or `solution of `orbit``, for messages.
    pub fn describe(&self) -> String {
        match self {
            OdeTy::Equation(e) => format!("equation `{}`", quote(&e.name)),
            OdeTy::Solution(s) => format!("solution of `{}`", quote(&s.equation)),
        }
    }
}

impl std::fmt::Display for OdeTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Words of a comma list: `a`, `a and b`, `a, b and c`.
/// The names from outside that the right sides of `eq` read (not the time,
/// a state variable or a local), each with its first use, and the `±`s in
/// them.
fn outside_reads(
    src: &str,
    eq: &Equation,
    time: &str,
    states: &[StateSig],
) -> (Vec<(String, Span)>, Vec<Span>) {
    let mut own: BTreeSet<String> = states.iter().map(|s| s.name.clone()).collect();
    own.insert(time.to_string());
    let (mut reads, mut pm) = (Vec::new(), Vec::new());
    for item in &eq.items {
        let value = match item {
            EquationItem::Derivative { value, .. } => value,
            EquationItem::Local { name, value, .. } => {
                if let Some(t) = sky_ast::ast::span_text(src, name.span) {
                    own.insert(t.to_string());
                }
                value
            }
            EquationItem::State { .. } => continue,
        };
        for (n, at) in name_uses(src, value) {
            if !own.contains(&n) && !reads.iter().any(|(r, _)| *r == n) {
                reads.push((n, at));
            }
        }
        pm.extend(plus_minus_uses(src, value));
    }
    (reads, pm)
}

fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => format!("`{one}`"),
        [init @ .., last] => {
            let init: Vec<String> = init.iter().map(|i| format!("`{i}`")).collect();
            format!("{} and `{last}`", init.join(", "))
        }
    }
}

impl Checker<'_> {
    // ----- equation -----

    pub(crate) fn equation_stmt(&mut self, eq: &Equation) {
        let saved = self.env.snapshot();
        let time_name = self.ident(&eq.time).to_string();
        let time_dim = self.time_type(eq);
        self.bind_name(&eq.time, Ty::scalar(time_dim));

        // Pass 1: the state variables, so their order in the source does not
        // matter to the equations.
        let mut states: Vec<StateSig> = Vec::new();
        let mut broken: BTreeSet<String> = BTreeSet::new();
        for item in &eq.items {
            let EquationItem::State { name, ty, .. } = item else {
                continue;
            };
            let n = self.ident(name).to_string();
            if n == time_name || states.iter().any(|s| s.name == n) || broken.contains(&n) {
                let what = if n == time_name {
                    format!("`{}` is the independent variable", quote(&n))
                } else {
                    format!("`{}` is already a state variable", quote(&n))
                };
                self.error(
                    Diagnostic::error(format!("{what}, so it cannot be declared again"), name.span)
                        .with_label("declared twice"),
                );
                continue;
            }
            match self.state_type(name, ty) {
                Some(sig) => {
                    self.bind_name(name, sig.ty());
                    states.push(sig);
                }
                None => {
                    self.bind_name(name, Ty::Unknown);
                    broken.insert(n);
                }
            }
        }
        if states.is_empty() && broken.is_empty() {
            self.error(
                Diagnostic::error(
                    format!(
                        "equation `{}` has no state variable",
                        self.quoted(eq.name.span)
                    ),
                    eq.name.span,
                )
                .with_help("declare what changes with a type, e.g. `x: m`, then give its derivative `x' = ...`"),
            );
        }

        // Pass 2: the equations and local names, in order.
        let mut derived: BTreeSet<String> = BTreeSet::new();
        let mut locals: BTreeSet<String> = BTreeSet::new();
        for item in &eq.items {
            match item {
                EquationItem::State { .. } => {}
                EquationItem::Derivative {
                    name, prime, value, ..
                } => self.derivative_item(
                    eq,
                    name,
                    *prime,
                    value,
                    time_dim,
                    &states,
                    &broken,
                    &mut derived,
                ),
                EquationItem::Local { name, value, .. } => {
                    let t = self.expr(value);
                    let n = self.ident(name).to_string();
                    if n == time_name || states.iter().any(|s| s.name == n) || broken.contains(&n) {
                        self.error(
                            Diagnostic::error(
                                format!("`{}` is already a name in this equation", quote(&n)),
                                name.span,
                            )
                            .with_label(
                                "state variables and the independent variable cannot be redefined",
                            )
                            .with_help("pick another name for this quantity"),
                        );
                    } else if !locals.insert(n) {
                        self.error(
                            Diagnostic::error(
                                format!(
                                    "`{}` is defined twice in this equation",
                                    self.quoted(name.span)
                                ),
                                name.span,
                            )
                            .with_label("already defined above"),
                        );
                    }
                    self.note_binding(name.span, Some(value), None);
                    self.bind_name(name, t);
                }
            }
        }
        let missing: Vec<&StateSig> = states
            .iter()
            .filter(|s| !derived.contains(&s.name))
            .collect();
        for s in missing {
            self.error(
                Diagnostic::error(
                    format!("no equation for `{}'`", quote(&s.name)),
                    eq.name.span,
                )
                .with_label(format!(
                    "`{}` is declared but never differentiated",
                    quote(&s.name)
                ))
                .with_help(format!(
                    "add a line `{}' = ...` (its right side has the dimension of `{}` per `{}`)",
                    quote(&s.name),
                    quote(&s.name),
                    quote(&time_name)
                )),
            );
        }

        self.env.restore(saved);
        let (reads, plus_minus) = outside_reads(self.src, eq, &time_name, &states);
        let eq_name = self.ident(&eq.name).to_string();
        for (n, at) in &reads {
            self.note_equation_read(&eq_name, n, *at);
        }
        let sig = EquationSig {
            name: self.ident(&eq.name).to_string(),
            time_name,
            time: time_dim,
            info: analyze_equation(self.src, eq),
            states,
            reads,
            plus_minus,
        };
        let ty = Ty::Ode(Arc::new(OdeTy::Equation(Box::new(sig))));
        self.bind_name(&eq.name, ty);
    }

    /// The dimension of the independent variable: a scalar, usually a unit.
    fn time_type(&mut self, eq: &Equation) -> Dimension {
        let Some(declared) = self.annotation(&eq.time_ty) else {
            return Dimension::DIMENSIONLESS;
        };
        if declared.ty.shape().rank() != 0 {
            self.error(
                Diagnostic::error(
                    format!(
                        "the independent variable `{}` must be a scalar",
                        self.quoted(eq.time.span)
                    ),
                    eq.time_ty.span,
                )
                .with_label("a single number, like `s`")
                .with_help("write `equation orbit(t: s):`"),
            );
        }
        declared.dim.unwrap_or(Dimension::DIMENSIONLESS)
    }

    /// A state variable's declared type: a fixed shape and a unit. Tags are
    /// not tracked through a solve, so they are dropped.
    fn state_type(&mut self, name: &Ident, ty: &sky_ast::TypeExpr) -> Option<StateSig> {
        let declared = self.annotation(ty)?;
        let shape = self.env.shapes.apply(declared.ty.shape());
        let mut size: u64 = 1;
        for d in shape.dims() {
            let ShapeDim::Known(n) = d else {
                self.error(
                    Diagnostic::error(
                        format!(
                            "the size of state variable `{}` must be a number",
                            self.quoted(name.span)
                        ),
                        ty.span,
                    )
                    .with_label(format!("{shape} has a size that is not fixed"))
                    .with_help("write the size out: `r: Vector[2] m`"),
                );
                return None;
            };
            size = size.saturating_mul(*n);
        }
        if size > crate::constant::MAX_ARRAY_ELEMENTS {
            self.error(
                Diagnostic::error(
                    format!("state variable `{}` is too large", self.quoted(name.span)),
                    ty.span,
                )
                .with_label(format!(
                    "more than {} values",
                    crate::constant::MAX_ARRAY_ELEMENTS
                )),
            );
            return None;
        }
        Some(StateSig {
            name: self.ident(name).to_string(),
            dim: declared.dim.unwrap_or(Dimension::DIMENSIONLESS),
            shape,
            decl: ty.span,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn derivative_item(
        &mut self,
        eq: &Equation,
        name: &Ident,
        prime: Span,
        value: &Expr,
        time: Dimension,
        states: &[StateSig],
        broken: &BTreeSet<String>,
        derived: &mut BTreeSet<String>,
    ) {
        let n = self.ident(name).to_string();
        let found = self.expr(value);
        let Some(state) = states.iter().find(|s| s.name == n) else {
            if !broken.contains(&n) {
                let mut d = Diagnostic::error(
                    format!(
                        "`{}'` is the derivative of `{}`, which is not declared",
                        quote(&n),
                        quote(&n)
                    ),
                    name.span.join(prime),
                )
                .with_label("no state variable of this name");
                if self.ident(&eq.time) == n {
                    d = d.with_help("the independent variable has no equation of its own");
                } else {
                    d = d.with_help(format!("declare it first, with a type: `{}: m`", quote(&n)));
                }
                self.error(d);
            }
            return;
        };
        if !derived.insert(n.clone()) {
            self.error(
                Diagnostic::error(
                    format!("`{}'` has more than one equation", quote(&n)),
                    name.span.join(prime),
                )
                .with_label("already given above")
                .with_help("a state variable has exactly one derivative"),
            );
            return;
        }
        let Ty::Value(v) = found else {
            if !matches!(found, Ty::Unknown) {
                self.error(
                    Diagnostic::error(
                        format!(
                            "the right side of `{}'` must be a number, found {}",
                            quote(&n),
                            self.describe_ty(&found)
                        ),
                        value.span,
                    )
                    .with_label(format!("this is {}", self.describe_ty(&found))),
                );
            }
            return;
        };
        let want = match state.dim.checked_div(time) {
            Ok(d) => d,
            Err(_) => {
                self.error(Diagnostic::error(
                    "dimension exponent too large",
                    name.span.join(prime),
                ));
                return;
            }
        };
        if v.dim != want {
            let time_span = eq.time.span;
            let d = Diagnostic::error(
                format!(
                    "`{n}'` is {}, but the right side is {}",
                    a_dimension(&want),
                    a_dimension(&v.dim)
                ),
                value.span,
            )
            .with_label(format!("this is {}", label_dim(&v.dim)))
            .with_secondary(state.decl, format!("`{n}` is {}", label_dim(&state.dim)))
            .with_secondary(
                time_span,
                format!("`{}` is {}", quote(self.ident(&eq.time)), label_dim(&time)),
            )
            .with_note(format!(
                "`{n}'` is the rate of change of `{n}`: {} per {} is {}",
                label_dim(&state.dim),
                label_dim(&time),
                label_dim(&want)
            ));
            let d = self.with_derivations(d, &[(value, v.dim)]);
            self.error(d);
            return;
        }
        let what = format!("`{}'`", quote(&n));
        let unified = self.with_origin(value.span, what, |u| u.unify(&state.shape, v.ty.shape()));
        if let Err(e) = unified {
            let e = e.in_context(ShapeContext::Declaration);
            let label = self.this_is(&v);
            self.error(
                self.shape_diagnostic(&e, value.span)
                    .with_label(label)
                    .with_secondary(state.decl, format!("the shape of `{n}`, and so of `{n}'`")),
            );
        }
    }

    fn describe_ty(&self, t: &Ty) -> String {
        match t {
            Ty::Bool => "a Bool".to_string(),
            other => other.to_string(),
        }
    }

    // ----- solve -----

    pub(crate) fn solve_stmt(&mut self, s: &Solve) {
        let sig = self.equation_named(&s.equation);
        let from = self.expr(&s.from);
        let to = self.expr(&s.to);
        let initial: Vec<Ty> = s.initial.iter().map(|i| self.expr(&i.value)).collect();
        let Some(sig) = sig else {
            self.bind_name(&s.target, Ty::Unknown);
            return;
        };

        for (which, e, t) in [("start", &s.from, from), ("end", &s.to, to)] {
            self.time_bound(&sig, which, e, t);
        }
        self.initial_values(&sig, s, &initial);
        self.uncertain_solve_inputs(&sig, s);

        let mut fields: Vec<(String, ValueTy)> =
            vec![(sig.time_name.clone(), ValueTy::scalar(sig.time))];
        for st in &sig.states {
            fields.push((
                st.name.clone(),
                ValueTy {
                    dim: st.dim,
                    ty: TensorType::untagged(st.shape.clone()),
                },
            ));
        }
        let (mut has_tolerance, mut conserved) = (false, Vec::<String>::new());
        for option in &s.options {
            match option {
                SolveOption::Tolerance { keyword, value } => {
                    if std::mem::replace(&mut has_tolerance, true) {
                        self.duplicate_option(*keyword, "tolerance");
                    }
                    self.plain_number(
                        value,
                        "the tolerance",
                        "a tolerance is relative: write `1e-9`, not a quantity",
                    );
                }
                SolveOption::Conserve {
                    name,
                    value,
                    within,
                    ..
                } => {
                    let n = self.ident(name).to_string();
                    let dim = self.conserved_expr(&sig, value);
                    if let Some(w) = within {
                        self.plain_number(
                            w,
                            "the allowed drift",
                            "`within` is a relative drift: write `1e-6`",
                        );
                    }
                    let clash = fields.iter().any(|(f, _)| *f == n);
                    if clash {
                        self.error(
                            Diagnostic::error(
                                format!("`{}` is already an attribute of the solution", quote(&n)),
                                name.span,
                            )
                            .with_label("the name of a conserved quantity must be new")
                            .with_help(format!(
                                "the solution already has {}",
                                list(&fields.iter().map(|(f, _)| f.clone()).collect::<Vec<_>>())
                            )),
                        );
                    } else if let Some(dim) = dim {
                        fields.push((n.clone(), ValueTy::scalar(dim)));
                        conserved.push(n);
                    }
                }
            }
        }
        self.explain_solve(&sig, s, &conserved, has_tolerance);
        let ty = Ty::Ode(Arc::new(OdeTy::Solution(SolutionSig {
            equation: sig.name.clone(),
            fields,
        })));
        self.bind_name(&s.target, ty);
    }

    /// The equation `id` names, or an error and `None`.
    fn equation_named(&mut self, id: &Ident) -> Option<EquationSig> {
        let name = self.ident(id);
        match self.env.get_symbolic(name) {
            Some(Ty::Ode(o)) => match &*o {
                OdeTy::Equation(e) => Some((**e).clone()),
                other => {
                    let d = Diagnostic::error(
                        format!(
                            "`{}` is a {}, not an equation",
                            quote(name),
                            other.describe()
                        ),
                        id.span,
                    )
                    .with_label("`solve` needs an equation")
                    .with_help("an equation is written `equation name(t: s): ...`");
                    self.error(d);
                    None
                }
            },
            Some(Ty::Unknown) => None,
            Some(other) => {
                let d = Diagnostic::error(
                    format!(
                        "`{}` is not an equation, it is {}",
                        quote(name),
                        self.describe_ty(&other)
                    ),
                    id.span,
                )
                .with_label("`solve` needs an equation")
                .with_help("an equation is written `equation name(t: s): ...`");
                self.error(d);
                None
            }
            None => {
                let mut d = Diagnostic::error(
                    format!("cannot find equation `{}` in this scope", quote(name)),
                    id.span,
                )
                .with_label("not defined");
                let known: Vec<String> = self
                    .env
                    .bindings()
                    .filter(|(_, t)| matches!(t, Ty::Ode(o) if matches!(**o, OdeTy::Equation(_))))
                    .map(|(n, _)| n.to_string())
                    .collect();
                if !known.is_empty() {
                    d = d.with_help(format!("the equations defined so far are {}", list(&known)));
                }
                self.error(d);
                None
            }
        }
    }

    /// The bound of the time span: the dimension of the independent
    /// variable.
    fn time_bound(&mut self, sig: &EquationSig, which: &str, e: &Expr, t: Ty) {
        let Ty::Value(v) = t else {
            if !matches!(t, Ty::Unknown) {
                self.error(
                    Diagnostic::error(
                        format!(
                            "the {which} of the time span must be {}, found {}",
                            a_dimension(&sig.time),
                            self.describe_ty(&t)
                        ),
                        e.span,
                    )
                    .with_label(format!("this is {}", self.describe_ty(&t))),
                );
            }
            return;
        };
        let bad_shape = !v.is_scalar();
        if v.dim != sig.time || bad_shape {
            let (headline, label) = if bad_shape {
                (
                    format!(
                        "the {which} of the time span must be a single number, found {}",
                        crate::check::rank_noun(v.ty.shape().rank())
                    ),
                    self.this_is(&v),
                )
            } else {
                (
                    format!(
                        "the {which} of the time span must be {} (the dimension of `{}`), found {}",
                        a_dimension(&sig.time),
                        quote(&sig.time_name),
                        a_dimension(&v.dim)
                    ),
                    format!("this is {}", label_dim(&v.dim)),
                )
            };
            let d = Diagnostic::error(headline, e.span)
                .with_label(label)
                .with_note(format!(
                    "`{}` is the independent variable of `{}`, and it is {}",
                    quote(&sig.time_name),
                    quote(&sig.name),
                    label_dim(&sig.time)
                ));
            let d = self.with_derivations(d, &[(e, v.dim)]);
            self.error(d);
        }
    }

    /// Every state variable gets one initial value of its type.
    fn initial_values(&mut self, sig: &EquationSig, s: &Solve, types: &[Ty]) {
        let mut given: BTreeSet<String> = BTreeSet::new();
        for (init, t) in s.initial.iter().zip(types) {
            let n = self.ident(&init.name).to_string();
            let Some(state) = sig.states.iter().find(|st| st.name == n) else {
                let names: Vec<String> = sig.states.iter().map(|st| st.name.clone()).collect();
                let mut d = Diagnostic::error(
                    format!(
                        "`{}` has no state variable `{}`",
                        quote(&sig.name),
                        quote(&n)
                    ),
                    init.name.span,
                )
                .with_label("not a state variable");
                if sig.time_name == n {
                    d = d.with_help(format!(
                        "`{}` is the independent variable: the time span is `from ... to ...`",
                        quote(&n)
                    ));
                } else if !names.is_empty() {
                    d = d.with_help(format!("its state variables are {}", list(&names)));
                }
                self.error(d);
                continue;
            };
            if !given.insert(n.clone()) {
                self.error(
                    Diagnostic::error(
                        format!("`{}` is given an initial value twice", quote(&n)),
                        init.name.span,
                    )
                    .with_label("already given"),
                );
                continue;
            }
            self.check_initial(state, &init.value, t);
        }
        let missing: Vec<String> = sig
            .states
            .iter()
            .filter(|st| !given.contains(&st.name))
            .map(|st| st.name.clone())
            .collect();
        if let Some(first) = missing.first() {
            let state = sig.states.iter().find(|st| st.name == *first);
            let example = state.map_or(String::new(), |st| {
                format!(
                    "`with ..., {first} = <{} of {}>`",
                    if st.shape.rank() == 0 {
                        "a value".to_string()
                    } else {
                        format!("{}", st.shape)
                    },
                    label_dim(&st.dim)
                )
            });
            self.error(
                Diagnostic::error(
                    format!(
                        "no initial value for {} in the `solve` of `{}`",
                        list(&missing),
                        quote(&sig.name)
                    ),
                    s.keyword,
                )
                .with_label("initial values are missing")
                .with_help(format!(
                    "give every state variable a starting value: {example}"
                )),
            );
        }
    }

    fn check_initial(&mut self, state: &StateSig, value: &Expr, t: &Ty) {
        let n = quote(&state.name);
        let Ty::Value(v) = t else {
            if !matches!(t, Ty::Unknown) {
                self.error(
                    Diagnostic::error(
                        format!(
                            "the initial value of `{n}` must be a number, found {}",
                            self.describe_ty(t)
                        ),
                        value.span,
                    )
                    .with_label(format!("this is {}", self.describe_ty(t))),
                );
            }
            return;
        };
        if v.dim != state.dim {
            let d = Diagnostic::error(
                format!(
                    "the initial value of `{n}` must be {}, found {}",
                    a_dimension(&state.dim),
                    a_dimension(&v.dim)
                ),
                value.span,
            )
            .with_label(format!("this is {}", label_dim(&v.dim)))
            .with_secondary(
                state.decl,
                format!("`{n}` is declared as {}", label_dim(&state.dim)),
            );
            let d = self.with_derivations(d, &[(value, v.dim)]);
            self.error(d);
            return;
        }
        let what = format!("the initial value of `{n}`");
        let unified = self.with_origin(value.span, what, |u| u.unify(&state.shape, v.ty.shape()));
        if let Err(e) = unified {
            let e = e.in_context(ShapeContext::Declaration);
            let label = self.this_is(v);
            self.error(
                self.shape_diagnostic(&e, value.span)
                    .with_label(label)
                    .with_secondary(state.decl, format!("`{n}` is declared here")),
            );
        }
    }

    /// A number without a unit: a tolerance, an allowed drift.
    fn plain_number(&mut self, e: &Expr, what: &str, help: &str) {
        let t = self.expr(e);
        let Ty::Value(v) = t else {
            if !matches!(t, Ty::Unknown) {
                self.error(
                    Diagnostic::error(
                        format!(
                            "{what} must be a plain number, found {}",
                            self.describe_ty(&t)
                        ),
                        e.span,
                    )
                    .with_label(format!("this is {}", self.describe_ty(&t))),
                );
            }
            return;
        };
        if !v.dim.is_dimensionless() || !v.is_scalar() {
            let found = if v.is_scalar() {
                a_dimension(&v.dim)
            } else {
                crate::check::rank_noun(v.ty.shape().rank())
            };
            let d = Diagnostic::error(
                format!("{what} must be a plain number, found {found}"),
                e.span,
            )
            .with_label(if v.is_scalar() {
                format!("this is {}", label_dim(&v.dim))
            } else {
                self.this_is(&v)
            })
            .with_help(help.to_string());
            let d = self.with_derivations(d, &[(e, v.dim)]);
            self.error(d);
        }
    }

    /// Types a `conserve` expression with the equation's names in scope.
    /// Returns its dimension if it is a scalar.
    fn conserved_expr(&mut self, sig: &EquationSig, value: &Expr) -> Option<Dimension> {
        let saved = self.env.snapshot();
        self.env.bind(sig.time_name.clone(), Ty::scalar(sig.time));
        for st in &sig.states {
            self.env.bind(st.name.clone(), st.ty());
        }
        let t = self.expr(value);
        self.env.restore(saved);
        match t {
            Ty::Value(v) if v.is_scalar() => Some(v.dim),
            Ty::Value(v) => {
                let label = self.this_is(&v);
                self.error(
                    Diagnostic::error(
                        format!(
                            "a conserved quantity must be a single number, found {}",
                            crate::check::rank_noun(v.ty.shape().rank())
                        ),
                        value.span,
                    )
                    .with_label(label)
                    .with_help("reduce it to a number, e.g. with `norm(...)` or `dot(a, b)`"),
                );
                None
            }
            Ty::Unknown => None,
            other => {
                self.error(
                    Diagnostic::error(
                        format!(
                            "a conserved quantity must be a number, found {}",
                            self.describe_ty(&other)
                        ),
                        value.span,
                    )
                    .with_label(format!("this is {}", self.describe_ty(&other))),
                );
                None
            }
        }
    }

    fn duplicate_option(&mut self, keyword: Span, name: &str) {
        self.error(
            Diagnostic::error(format!("`{name}` is given twice"), keyword)
                .with_label("already given above")
                .with_help("a `solve` takes one of these"),
        );
    }

    /// The note `sky check --explain` prints for a `solve`: what the checker
    /// knows that the solver will use.
    fn explain_solve(
        &mut self,
        sig: &EquationSig,
        s: &Solve,
        conserved: &[String],
        has_tolerance: bool,
    ) {
        let info = &sig.info;
        let name = quote(&sig.name);
        let mut notes: Vec<String> = Vec::new();
        let headline = match info.pair_names().as_slice() {
            [] => {
                notes.push(format!(
                    "the equations of `{name}` have no second-order structure (no `q' = v` with `v'` free of `v`), so only general-purpose methods apply"
                ));
                format!("`{name}` is a general first-order system")
            }
            pairs => {
                let rendered: Vec<String> =
                    pairs.iter().map(|(q, v)| format!("`{q}' = {v}`")).collect();
                notes.push(format!(
                    "{}, and the accelerations depend only on `{}` and the positions, not on the velocities",
                    rendered.join(" and "),
                    quote(&info.time)
                ));
                format!("`{name}` is a second-order system, so a symplectic integrator applies")
            }
        };
        if let Some(SolveOption::Tolerance { value, .. }) = s
            .options
            .iter()
            .find(|o| matches!(o, SolveOption::Tolerance { .. }))
        {
            notes.push(format!("tolerance: {}", self.quoted(value.span)));
        } else if !has_tolerance {
            notes.push("no tolerance given: the solver's default applies".to_string());
        }
        if !conserved.is_empty() {
            notes.push(format!(
                "to be conserved: {}; the solver monitors the drift of each and reports it",
                list(conserved)
            ));
        }
        notes.push(
            "the method is chosen when the program runs; `print(explain(sol))` says which, and why"
                .to_string(),
        );
        let mut d = Diagnostic::note(headline, s.keyword.join(s.equation.span));
        for n in notes {
            d = d.with_note(n);
        }
        self.ode_notes.push(d);
    }

    // ----- attributes and builtins -----

    /// `base.name`: reads a field of a solution; anything else is untyped
    /// (there are no other attributes yet).
    pub(crate) fn attribute_expr(&mut self, base: &Expr, name: &Ident) -> Ty {
        let t = self.expr(base);
        let attr = self.ident(name);
        let Ty::Ode(o) = t else {
            let what = match &t {
                Ty::Value(v) => format!("a {}", crate::check::rank_noun(v.ty.shape().rank())),
                Ty::Bool => "a comparison".to_string(),
                Ty::Func(_) => "a function".to_string(),
                // Strings and earlier errors: not typed here.
                Ty::Ode(_) | Ty::Unknown => return Ty::Unknown,
            };
            let b = self.operand_text(base);
            let rank = match &t {
                Ty::Value(v) => v.ty.shape().rank(),
                _ => 0,
            };
            let help = match attr {
                "T" if rank == 2 => format!("write `{b}ᵀ` for the transpose"),
                "T" if rank == 1 => {
                    "a vector has no row or column orientation, so there is nothing to transpose: `u @ v` is already the dot product".to_string()
                }
                "shape" | "size" | "len" => format!("write `len({b})` for its length"),
                "mean" | "sum" | "std" | "var" | "min" | "max" | "argmin" | "argmax"
                | "median" | "prod" => format!(
                    "Sky has no methods, and no `{attr}` function yet; a sum over the entries can be written as a loop over `range(len({b}))`"
                ),
                _ => "Sky values have no attributes or methods: functions are called as `f(x)`"
                    .to_string(),
            };
            self.error(
                Diagnostic::error(
                    format!("{what} has no attribute `{}`", quote(attr)),
                    name.span,
                )
                .with_label("no such attribute")
                .with_help(help),
            );
            return Ty::Unknown;
        };
        match &*o {
            OdeTy::Solution(sol) => match sol.fields.iter().find(|(f, _)| f == attr) {
                Some((_, v)) => Ty::Value(v.clone()),
                None => {
                    let names: Vec<String> = sol.fields.iter().map(|(f, _)| f.clone()).collect();
                    self.error(
                        Diagnostic::error(
                            format!(
                                "the solution of `{}` has no `{}`",
                                quote(&sol.equation),
                                quote(attr)
                            ),
                            name.span,
                        )
                        .with_label("no such attribute")
                        .with_help(format!("it has {}", list(&names))),
                    );
                    Ty::Unknown
                }
            },
            OdeTy::Equation(eq) => {
                self.error(
                    Diagnostic::error(
                        format!("an equation has no attribute `{}`", quote(attr)),
                        name.span,
                    )
                    .with_label(format!("`{}` is an equation", quote(&eq.name)))
                    .with_help(format!(
                        "solve it first: `sol = solve {} from ... to ...`, then read `sol.{}`",
                        quote(&eq.name),
                        quote(attr)
                    )),
                );
                Ty::Unknown
            }
        }
    }

    /// `explain(sol)`: the solver's explanation as text (a string, which
    /// the checker does not type).
    pub(crate) fn ode_call(&mut self, callee: &Expr, args: &[Expr], call: Span) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        if self.ident(id) != "explain" || self.env.get("explain").is_some() {
            return None;
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        let [arg] = args else {
            self.error(
                Diagnostic::error(
                    format!("`explain` takes 1 argument, found {}", args.len()),
                    call,
                )
                .with_label("wrong number of arguments")
                .with_help("call it as `explain(sol)`"),
            );
            return Some(Ty::Unknown);
        };
        match tys.first() {
            Some(Ty::Ode(o)) if matches!(**o, OdeTy::Solution(_)) => {}
            Some(Ty::Unknown) | None => {}
            Some(other) => {
                let what = self.describe_ty(other);
                self.error(
                    Diagnostic::error(
                        format!("`explain` needs a solution, found {what}"),
                        arg.span,
                    )
                    .with_label(format!("this is {what}"))
                    .with_help("explain what a solve did: `sol = solve ...`, then `explain(sol)`"),
                );
            }
        }
        Some(Ty::Unknown)
    }
}
