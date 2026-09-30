//! `Checked::expr_types` has an entry for every expression, so later stages
//! (lowering to SIR) can use the checker's types without re-deriving them.
//! Also pins the side tables for bindings and functions.

use std::path::Path;

use sky_ast::{
    with_deep_stack, AssignTarget, Block, Expr, ExprKind, IndexItem, Module, Stmt, StmtKind,
};
use sky_types::{check_module, Checked, Ty};
use sky_units::Dimension;

fn check(src: &str) -> (Module, Checked) {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    let checked = check_module(&module, src);
    (module, checked)
}

/// Every expression node of the module, callee-of-builtin flagged.
struct Walk<'a> {
    src: &'a str,
    /// Expressions with no entry, as (source text, is a builtin callee).
    missing: Vec<String>,
    checked: &'a Checked,
    count: usize,
}

impl Walk<'_> {
    fn expr(&mut self, e: &Expr, builtin_callee: bool) {
        self.count += 1;
        let has = self.checked.expr_types.contains_key(&e.span);
        // The callee of a builtin call (`print(...)`, `inv(x)`, `sqrt(x)`, `derivative(f, x)`)
        // is the one documented expression that is never typed.
        if !has && !builtin_callee {
            self.missing.push(self.src[e.span.range()].to_string());
        }
        match &e.kind {
            ExprKind::Literal(_) | ExprKind::Name(_) | ExprKind::UnitLiteral { .. } => {}
            ExprKind::Unary { operand, .. }
            | ExprKind::Inverse { operand, .. }
            | ExprKind::Transpose { operand, .. } => self.expr(operand, false),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs, false);
                self.expr(rhs, false);
            }
            ExprKind::BoolOp { operands, .. } => {
                for o in operands {
                    self.expr(o, false);
                }
            }
            ExprKind::Array(elements) => {
                for el in elements {
                    self.expr(el, false);
                }
            }
            ExprKind::Call { callee, args } => {
                // Builtins (`print`, `inv`, `range`, the array and math functions) are
                // recognised by name, so their callee is never typed.
                let builtin = matches!(&callee.kind, ExprKind::Name(id)
                    if matches!(&self.src[id.span.range()], "print" | "inv" | "transpose" | "zeros" | "identity" | "len" | "range" | "derivative" | "grad" | "nominal" | "uncertainty")
                        || sky_units::MathFn::from_name(&self.src[id.span.range()]).is_some())
                    && !self.checked.expr_types.contains_key(&callee.span);
                self.expr(callee, builtin);
                for a in args {
                    self.expr(a, false);
                }
            }
            ExprKind::Index { base, items } => {
                self.expr(base, false);
                self.items(items);
            }
            ExprKind::Attribute { base, .. } => self.expr(base, false),
            ExprKind::Paren(inner) => self.expr(inner, false),
            ExprKind::IfElse { then, cond, else_ } => {
                self.expr(then, false);
                self.expr(cond, false);
                self.expr(else_, false);
            }
            ExprKind::As { value, .. } => self.expr(value, false),
            ExprKind::PlusMinus { value, sigma, .. } => {
                self.expr(value, false);
                self.expr(sigma, false);
            }
        }
    }

    fn items(&mut self, items: &[IndexItem]) {
        for item in items {
            match item {
                IndexItem::Expr(e) => self.expr(e, false),
                IndexItem::Slice { start, stop, .. } => {
                    for b in [start, stop].into_iter().flatten() {
                        self.expr(b, false);
                    }
                }
            }
        }
    }

    fn block(&mut self, b: &Block) {
        for s in &b.stmts {
            self.stmt(s);
        }
    }

    fn target(&mut self, t: &AssignTarget) {
        match t {
            AssignTarget::Name(_) => {}
            AssignTarget::Index { base, items, .. } => {
                self.expr(base, false);
                self.items(items);
            }
            AssignTarget::Attribute { base, .. } => self.expr(base, false),
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Expr(e) => self.expr(e, false),
            StmtKind::Assign { target, value } => {
                self.expr(value, false);
                self.target(target);
            }
            StmtKind::CompoundAssign { target, value, .. } => {
                self.expr(value, false);
                self.target(target);
            }
            StmtKind::Binding { value, .. } => {
                if let Some(v) = value {
                    self.expr(v, false);
                }
            }
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (c, b) in branches {
                    self.expr(c, false);
                    self.block(b);
                }
                if let Some(b) = else_block {
                    self.block(b);
                }
            }
            StmtKind::While { cond, body } => {
                self.expr(cond, false);
                self.block(body);
            }
            StmtKind::For { iter, body, .. } => {
                // `for i in range(...)`: the call is a loop header, not a
                // value, so neither it nor its callee is typed; its
                // arguments are.
                match &iter.kind {
                    ExprKind::Call { callee, args }
                        if matches!(&callee.kind, ExprKind::Name(id)
                            if &self.src[id.span.range()] == "range")
                            && !self.checked.expr_types.contains_key(&callee.span) =>
                    {
                        for a in args {
                            self.expr(a, false);
                        }
                    }
                    _ => self.expr(iter, false),
                }
                self.block(body);
            }
            StmtKind::Def { body, .. } => self.block(body),
            StmtKind::Return(v) => {
                if let Some(v) = v {
                    self.expr(v, false);
                }
            }
            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Equation(_) | StmtKind::Solve(_) => {}
        }
    }
}

/// Asserts every expression of `src` has an entry; returns how many there were.
fn assert_complete(name: &str, src: &str) -> usize {
    with_deep_stack(|| {
        let (module, checked) = check(src);
        let mut w = Walk {
            src,
            missing: Vec::new(),
            checked: &checked,
            count: 0,
        };
        for s in &module.stmts {
            w.stmt(s);
        }
        assert!(
            w.missing.is_empty(),
            "{name}: expressions without a type entry: {:?}",
            w.missing
        );
        w.count
    })
}

#[test]
fn every_expression_of_the_ui_programs_has_a_type() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut total = 0;
    let mut files = 0;
    for dir in [
        root.join("tests/ui"),
        root.join("../../tests/ui/check"),
        root.join("../../tests/ui/types"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Programs that check (ok_*) and ones with errors (err_*) that
            // still parse: the table is complete either way.
            if !file.ends_with(".sky") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("reads");
            // Skip the ones that don't lex or parse (the lexer/parser error
            // goldens): they never reach the checker.
            let parsed = with_deep_stack(|| {
                let Ok(lexed) = sky_syntax::lex(&src) else {
                    return false;
                };
                sky_ast::parse_module(&lexed.tokens, &src).1.is_empty()
            });
            if !parsed {
                continue;
            }
            total += assert_complete(file, &src);
            files += 1;
        }
    }
    assert!(files >= 40 && total > 100, "{files} files, {total} exprs");
}

#[test]
fn every_expression_of_a_mixed_program_has_a_type() {
    let src = "\
def f(a, b):
    return a + b

def g(n):
    total = 0
    while n > 0:
        total += n
        n -= 1
        if n == 3 or n == 4 and total > 2:
            continue
        else:
            pass
    return (total if total > 1 else 2) ^ 2

x = f(1 m, 2 m)
print(x, \"s\", g(4))
A: PositiveDefinite Matrix[N, N]
b: Vector[3] m
y = (A⁻¹) @ b
z = inv(A) @ b
w = -x + +x
q = A[0, 1:2]
r = b.len
i = inv(2 s)
j = (3 m)⁻¹
k = 2 ^ 0.5
for t in range(1, g(2), 2):
    w += t * 1 m
for row in [[1, 2], [3, 4]]:
    for e in row:
        w += e * 1 m
";
    let n = assert_complete("mixed", src);
    assert!(n > 60, "{n}");
}

#[test]
fn display_values_and_their_misuse_have_type_entries() {
    // `v as unit` is typed (as unknown: it has no type to compute with), in
    // the places it is allowed and in the ones that are errors.
    let src = "v = 25 m/s\nprint(v as km/h, (v as m/s))\nv as km/h\nx = 2 * (v as km/h)\n";
    let n = assert_complete("show", src);
    assert!(n >= 12, "{n}");
}

#[test]
fn compound_assignment_types_the_synthesized_node() {
    let src = "x = 5 m\nx += 2 m\n";
    with_deep_stack(|| {
        let (_, c) = check(src);
        // `x += 2 m` synthesizes `x + 2 m` spanning `x` .. `m`.
        let start = src.find("x +=").expect("stmt") as u32;
        let end = src.rfind('m').expect("unit") as u32 + 1;
        let whole = c
            .expr_ty(sky_diag::Span::new(start, end))
            .expect("synthesized node is typed");
        assert_eq!(whole, Ty::scalar(Dimension::LENGTH));
        // The name node is keyed by the target.
        assert_eq!(
            c.expr_ty(sky_diag::Span::new(start, start + 1)),
            Some(Ty::scalar(Dimension::LENGTH))
        );
    });
}

#[test]
fn shape_variables_are_resolved_on_lookup() {
    let src = "A: PositiveDefinite Matrix[N, N]\nb: Vector[3] m\nx = A⁻¹ @ b\n";
    with_deep_stack(|| {
        let (module, c) = check(src);
        assert!(!c.has_errors());
        let StmtKind::Assign { value, .. } = &module.stmts[2].kind else {
            panic!("assignment")
        };
        // `N` was solved to 3 by the product; the resolved type says so.
        assert_eq!(
            c.expr_ty(value.span).expect("typed").to_string(),
            "Vector[3] [L]"
        );
        // The declared input's type is recorded at its name.
        let StmtKind::Binding { name, .. } = &module.stmts[0].kind else {
            panic!("binding")
        };
        let t = c.binding_types.get(&name.span).expect("declared type");
        assert_eq!(
            c.resolve_ty(t).to_string(),
            "Matrix[3,3] {Symmetric, PositiveDefinite} [1]"
        );
    });
}

#[test]
fn functions_are_recorded_by_def_name_span() {
    let src = "def f(x: Vector[N]) -> Vector[N]:\n    return x\ndef g():\n    pass\nv: Vector[2]\ny = f(v)\n";
    with_deep_stack(|| {
        let (module, c) = check(src);
        assert!(!c.has_errors(), "{:#?}", c.diagnostics);
        let StmtKind::Def { name, .. } = &module.stmts[0].kind else {
            panic!("def")
        };
        let sig = c.funcs.get(&name.span).expect("recorded");
        assert_eq!(sig.name(), "f");
        assert_eq!(sig.def_span(), name.span);
        assert_eq!(sig.param_name(0), Some("x"));
        assert!(sig.param_ty(0).is_some());
        // The call's callee has this very signature.
        let StmtKind::Assign { value, .. } = &module.stmts[3].kind else {
            panic!("assign")
        };
        let ExprKind::Call { callee, .. } = &value.kind else {
            panic!("call")
        };
        match c.expr_types.get(&callee.span) {
            Some(Ty::Func(s)) => assert_eq!(s.def_span(), name.span),
            other => panic!("{other:?}"),
        }
        // `g` returns nothing.
        let StmtKind::Def { name: g, .. } = &module.stmts[1].kind else {
            panic!("def")
        };
        assert_eq!(c.funcs.get(&g.span).expect("g").ret(), &Ty::Unknown);
    });
}

#[test]
fn a_gradient_call_is_a_vector_and_records_the_function() {
    let src = "def U(x: m, y: m) -> J:\n    return 3 N/m * (x^2 + y^2)\nF = grad(U)(1 m, 2 m)\n";
    with_deep_stack(|| {
        let (module, c) = check(src);
        assert!(!c.has_errors(), "{:#?}", c.diagnostics);
        assert_complete("grad", src);
        let StmtKind::Assign { value, .. } = &module.stmts[1].kind else {
            panic!("assign")
        };
        // The call is a Vector[2] of forces ...
        match c.expr_ty(value.span) {
            Some(Ty::Value(v)) => {
                assert_eq!(v.dim, Dimension::FORCE);
                assert_eq!(v.ty.shape().rank(), 1);
            }
            other => panic!("{other:?}"),
        }
        // ... and its callee says it is the gradient of `U`.
        let ExprKind::Call { callee, .. } = &value.kind else {
            panic!("call")
        };
        let Some(Ty::Func(sig)) = c.expr_types.get(&callee.span) else {
            panic!("the callee is a function")
        };
        let d = sig.derivative_of().expect("a gradient");
        assert_eq!((d.func().name(), d.is_gradient()), ("U", true));
        assert_eq!(d.wrts(), 0..2);
        assert_eq!(sig.name(), "grad(U)");
    });
}

#[test]
fn partial_notation_records_the_derived_function_and_types_every_node() {
    let src = "def position(t: s) -> m:\n    return 5 m/s^2 * t^2\nv = (∂position/∂t)(2 s)\nw = ∂position/∂t(2 s)\n";
    with_deep_stack(|| {
        let (module, c) = check(src);
        // The second form is an error (a call binds tighter than ∂), and
        // both leave no expression untyped.
        assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
        assert_complete("partial", src);
        let StmtKind::Assign { value, .. } = &module.stmts[1].kind else {
            panic!("assign")
        };
        match c.expr_ty(value.span) {
            Some(Ty::Value(v)) => assert_eq!(v.dim, Dimension::VELOCITY),
            other => panic!("{other:?}"),
        }
        let ExprKind::Call { callee, .. } = &value.kind else {
            panic!("call")
        };
        let Some(Ty::Func(sig)) = c.expr_types.get(&callee.span) else {
            panic!("the callee is a function")
        };
        let d = sig.derivative_of().expect("a derivative");
        assert_eq!(
            (d.func().name(), d.wrt(), d.is_gradient()),
            ("position", 0, false)
        );
        assert_eq!(sig.name(), "derivative(position, t)");
    });
}

#[test]
fn a_derivative_call_records_the_derived_function() {
    let src =
        "def position(t: s) -> m:\n    return 5 m/s^2 * t^2\nv = derivative(position, t)(2 s)\n";
    with_deep_stack(|| {
        let (module, c) = check(src);
        assert!(!c.has_errors(), "{:#?}", c.diagnostics);
        assert_eq!(assert_complete("derivative", src), 11);
        let StmtKind::Assign { value, .. } = &module.stmts[1].kind else {
            panic!("assign")
        };
        // The call is a velocity ...
        match c.expr_ty(value.span) {
            Some(Ty::Value(v)) => assert_eq!(v.dim, Dimension::VELOCITY),
            other => panic!("{other:?}"),
        }
        // ... and its callee `derivative(position, t)` is a function that
        // says what it differentiates.
        let ExprKind::Call { callee, args } = &value.kind else {
            panic!("call")
        };
        assert_eq!(args.len(), 1);
        let Some(Ty::Func(sig)) = c.expr_types.get(&callee.span) else {
            panic!("the callee is a function")
        };
        let d = sig.derivative_of().expect("a derivative");
        assert_eq!((d.func().name(), d.wrt()), ("position", 0));
        assert_eq!(sig.name(), "derivative(position, t)");
        assert_eq!(sig.arity(), 1);
        assert!(c.funcs.values().all(|f| f.derivative_of().is_none()));
    });
}
