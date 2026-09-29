//! S-expression printer for the AST, used by snapshot-style tests.
//!
//! Format: `(op child...)`. Leaves print their source text. Examples:
//! `-x^2` → `(neg (^ x 2))`, `9.8 m/s^2` → `(unit 9.8 (/ m (^ s 2)))`,
//! `m^(1/2)` → `(^ m 1/2)`, `A⁻¹ @ b` → `(@ (inverse A) b)`,
//! `f(a)` → `(call f a)`, `M[3, 4]` → `(index M 3 4)`, `a.b` → `(attr a b)`,
//! `(a)` → `(paren a)`, `x = 1` → `(= x 1)`, `A: T = v` → `(: A T v)`.

use sky_diag::Span;

use crate::ast::{span_text, Expr, ExprKind, Module, Stmt, StmtKind, UnitExpr};

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

impl Stmt {
    pub fn to_sexpr(&self, src: &str) -> String {
        let mut out = String::new();
        match &self.kind {
            StmtKind::Expr(e) => write_expr(&mut out, e, src),
            StmtKind::Assign { target, value } => {
                out.push_str("(= ");
                out.push_str(text(src, target.span));
                out.push(' ');
                write_expr(&mut out, value, src);
                out.push(')');
            }
            StmtKind::Binding { name, ty, value } => {
                out.push_str("(: ");
                out.push_str(text(src, name.span));
                out.push(' ');
                write_expr(&mut out, ty, src);
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
        ExprKind::Inverse { operand, .. } => node(out, "inverse", src, &[operand]),
        ExprKind::Call { callee, args } => {
            let mut children = vec![&**callee];
            children.extend(args.iter());
            node(out, "call", src, &children);
        }
        ExprKind::Index { base, indices } => {
            let mut children = vec![&**base];
            children.extend(indices.iter());
            node(out, "index", src, &children);
        }
        ExprKind::Attribute { base, name } => {
            out.push_str("(attr ");
            write_expr(out, base, src);
            out.push(' ');
            out.push_str(text(src, name.span));
            out.push(')');
        }
        ExprKind::Paren(inner) => node(out, "paren", src, &[inner]),
    }
}

fn write_unit(out: &mut String, u: &UnitExpr, src: &str) {
    match u {
        UnitExpr::Name(id) => out.push_str(text(src, id.span)),
        UnitExpr::Mul { lhs, rhs, .. } | UnitExpr::Div { lhs, rhs, .. } => {
            out.push_str(if matches!(u, UnitExpr::Mul { .. }) { "(* " } else { "(/ " });
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
