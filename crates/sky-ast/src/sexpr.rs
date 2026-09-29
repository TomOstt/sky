//! S-expression printer for the AST, used by snapshot-style tests.
//!
//! Format: `(op child...)`. Leaves print their source text. Examples:
//! `-x^2` → `(neg (^ x 2))`, `9.8 m/s^2` → `(unit 9.8 (/ m (^ s 2)))`,
//! `m^(1/2)` → `(^ m 1/2)`, `A⁻¹ @ b` → `(@ (inverse A) b)`,
//! `f(a)` → `(call f a)`, `M[3, 4]` → `(index M 3 4)`, `a.b` → `(attr a b)`,
//! `A[1:3]` → `(index A (: 1 3))`, `A[0, :]` → `(index A 0 (: _ _))`,
//! `(a)` → `(paren a)`, `x = 1` → `(= x 1)`,
//! `A: Symmetric Matrix[3, 3] N = v` → `(: A (type (tags Symmetric) (Matrix 3 3) (unit N)) v)`,
//! `a.b = x` → `(= (attr a b) x)`, `x += e` → `(+= x e)`,
//! `true` → `true`, `not a` → `(not a)`, `a and b and c` → `(and a b c)`,
//! `a and b or c` → `(or (and a b) c)`.

use sky_syntax::Span;

use crate::ast::{
    span_text, AssignTarget, Expr, ExprKind, IndexItem, Module, Stmt, StmtKind, TypeExpr, UnitExpr,
};

fn text(src: &str, span: Span) -> &str {
    span_text(src, span).unwrap_or("<?>")
}

impl Expr {
    pub fn to_sexpr(&self, src: &str) -> String {
        let mut out = String::new();
        write_expr(&mut out, self, src);
        out
    }
}

impl UnitExpr {
    pub fn to_sexpr(&self, src: &str) -> String {
        let mut out = String::new();
        write_unit(&mut out, self, src);
        out
    }
}

impl TypeExpr {
    /// `(type (tags T...)? (Matrix d...)? (unit u)?)`: absent parts are
    /// left out, so `x: m/s` is `(type (unit (/ m s)))` and `x: Scalar` is
    /// `(type (Scalar))`.
    pub fn to_sexpr(&self, src: &str) -> String {
        let mut out = String::from("(type");
        if !self.tags.is_empty() {
            out.push_str(" (tags");
            for t in &self.tags {
                out.push(' ');
                out.push_str(text(src, t.span));
            }
            out.push(')');
        }
        if let Some(shape) = &self.shape {
            out.push_str(" (");
            out.push_str(shape.kind.name());
            for d in &shape.dims {
                out.push(' ');
                out.push_str(text(src, d.span()));
            }
            out.push(')');
        }
        if let Some(u) = &self.unit {
            out.push_str(" (unit ");
            write_unit(&mut out, u, src);
            out.push(')');
        }
        out.push(')');
        out
    }
}

impl Stmt {
    pub fn to_sexpr(&self, src: &str) -> String {
        let mut out = String::new();
        match &self.kind {
            StmtKind::Expr(e) => write_expr(&mut out, e, src),
            StmtKind::Assign { target, value } => {
                write_assign(&mut out, "=", target, value, src);
            }
            StmtKind::CompoundAssign {
                target, op, value, ..
            } => {
                write_assign(&mut out, op.symbol(), target, value, src);
            }
            StmtKind::Binding { name, ty, value } => {
                out.push_str("(: ");
                out.push_str(text(src, name.span));
                out.push(' ');
                out.push_str(&ty.to_sexpr(src));
                if let Some(v) = value {
                    out.push(' ');
                    write_expr(&mut out, v, src);
                }
                out.push(')');
            }
        }
        out
    }
}

impl Module {
    /// One statement per line.
    pub fn to_sexpr(&self, src: &str) -> String {
        self.stmts
            .iter()
            .map(|s| s.to_sexpr(src))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn write_assign(out: &mut String, head: &str, target: &AssignTarget, value: &Expr, src: &str) {
    out.push('(');
    out.push_str(head);
    out.push(' ');
    match target {
        AssignTarget::Name(id) => out.push_str(text(src, id.span)),
        AssignTarget::Index { base, items, .. } => write_index(out, base, items, src),
        AssignTarget::Attribute { base, name, .. } => write_attr(out, base, name.span, src),
    }
    out.push(' ');
    write_expr(out, value, src);
    out.push(')');
}

fn write_index(out: &mut String, base: &Expr, items: &[IndexItem], src: &str) {
    out.push_str("(index ");
    write_expr(out, base, src);
    for item in items {
        out.push(' ');
        match item {
            IndexItem::Expr(e) => write_expr(out, e, src),
            IndexItem::Slice { start, stop, .. } => {
                out.push_str("(:");
                for bound in [start, stop] {
                    out.push(' ');
                    match bound {
                        Some(e) => write_expr(out, e, src),
                        None => out.push('_'),
                    }
                }
                out.push(')');
            }
        }
    }
    out.push(')');
}

fn write_attr(out: &mut String, base: &Expr, name: Span, src: &str) {
    out.push_str("(attr ");
    write_expr(out, base, src);
    out.push(' ');
    out.push_str(text(src, name));
    out.push(')');
}

fn node(out: &mut String, head: &str, src: &str, children: &[&Expr]) {
    out.push('(');
    out.push_str(head);
    for child in children {
        out.push(' ');
        write_expr(out, child, src);
    }
    out.push(')');
}

fn write_expr(out: &mut String, e: &Expr, src: &str) {
    match &e.kind {
        ExprKind::Literal(lit) => out.push_str(text(src, lit.span())),
        ExprKind::Name(id) => out.push_str(text(src, id.span)),
        ExprKind::UnitLiteral { value, unit } => {
            out.push_str("(unit ");
            out.push_str(text(src, value.span()));
            out.push(' ');
            write_unit(out, unit, src);
            out.push(')');
        }
        ExprKind::Unary { op, operand, .. } => node(out, op.name(), src, &[operand]),
        ExprKind::Binary { op, lhs, rhs, .. } => node(out, op.symbol(), src, &[lhs, rhs]),
        ExprKind::BoolOp { op, operands, .. } => {
            let children: Vec<&Expr> = operands.iter().collect();
            node(out, op.keyword(), src, &children);
        }
        ExprKind::Inverse { operand, .. } => node(out, "inverse", src, &[operand]),
        ExprKind::Call { callee, args } => {
            let mut children = vec![&**callee];
            children.extend(args.iter());
            node(out, "call", src, &children);
        }
        ExprKind::Index { base, items } => write_index(out, base, items, src),
        ExprKind::Attribute { base, name } => write_attr(out, base, name.span, src),
        ExprKind::Paren(inner) => node(out, "paren", src, &[inner]),
    }
}

fn write_unit(out: &mut String, u: &UnitExpr, src: &str) {
    match u {
        UnitExpr::Name(id) => out.push_str(text(src, id.span)),
        UnitExpr::Mul { lhs, rhs, .. } | UnitExpr::Div { lhs, rhs, .. } => {
            out.push_str(if matches!(u, UnitExpr::Mul { .. }) {
                "(* "
            } else {
                "(/ "
            });
            write_unit(out, lhs, src);
            out.push(' ');
            write_unit(out, rhs, src);
            out.push(')');
        }
        UnitExpr::Pow { base, num, den, .. } => {
            out.push_str("(^ ");
            write_unit(out, base, src);
            out.push(' ');
            out.push_str(&num.to_string());
            if *den != 1 {
                out.push('/');
                out.push_str(&den.to_string());
            }
            out.push(')');
        }
    }
}
