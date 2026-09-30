//! Syntactic facts about statements that lowering needs before it builds
//! the regions they end up in: which names a loop assigns (its carried
//! variables), which of `break`, `continue` and `return` a statement may
//! execute (which flags it touches), and whether it always leaves its block.
//!
//! These are properties of the tree, not of types, so nothing here
//! re-derives what the checker knows. Every function recurses once per
//! nesting level, which the parser bounds.

use std::collections::BTreeSet;

use sky_ast::ast::span_text;
use sky_ast::{AssignTarget, Expr, ExprKind, Literal, Stmt, StmtKind};

/// Which of `break`, `continue` and `return` a statement may execute.
/// `break` and `continue` are counted only if no loop inside the statement
/// catches them (they leave the loop the statement sits in); `return`
/// always leaves the function. Nested `def`s are separate functions.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Exits {
    pub(crate) ret: bool,
    pub(crate) brk: bool,
    pub(crate) cont: bool,
}

impl Exits {
    pub(crate) fn any(self) -> bool {
        self.ret || self.brk || self.cont
    }

    fn union(self, other: Exits) -> Exits {
        Exits {
            ret: self.ret || other.ret,
            brk: self.brk || other.brk,
            cont: self.cont || other.cont,
        }
    }
}

pub(crate) fn exits_stmt(s: &Stmt) -> Exits {
    match &s.kind {
        StmtKind::Return(_) => Exits {
            ret: true,
            ..Exits::default()
        },
        StmtKind::Break => Exits {
            brk: true,
            ..Exits::default()
        },
        StmtKind::Continue => Exits {
            cont: true,
            ..Exits::default()
        },
        StmtKind::If {
            branches,
            else_block,
        } => {
            let mut e = Exits::default();
            for (_, b) in branches {
                e = e.union(exits_stmts(&b.stmts));
            }
            if let Some(b) = else_block {
                e = e.union(exits_stmts(&b.stmts));
            }
            e
        }
        // The loop catches its own break and continue.
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => Exits {
            ret: exits_stmts(&body.stmts).ret,
            ..Exits::default()
        },
        StmtKind::Def { .. }
        | StmtKind::Expr(_)
        | StmtKind::Assign { .. }
        | StmtKind::Binding { .. }
        | StmtKind::CompoundAssign { .. }
        | StmtKind::Equation(_)
        | StmtKind::Solve(_)
        | StmtKind::Assert { .. }
        | StmtKind::Pass => Exits::default(),
    }
}

pub(crate) fn exits_stmts(stmts: &[Stmt]) -> Exits {
    stmts
        .iter()
        .fold(Exits::default(), |e, s| e.union(exits_stmt(s)))
}

/// The names assigned anywhere in `stmts` (`x = ...`, `x: T = ...`,
/// `x += ...`), not looking into nested `def`s (their names are their own).
pub(crate) fn assigned_vars(stmts: &[Stmt], src: &str, out: &mut BTreeSet<String>) {
    for s in stmts {
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
                out.insert(span_text(src, id.span).unwrap_or("").to_string());
            }
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (_, b) in branches {
                    assigned_vars(&b.stmts, src, out);
                }
                if let Some(b) = else_block {
                    assigned_vars(&b.stmts, src, out);
                }
            }
            StmtKind::While { body, .. } => assigned_vars(&body.stmts, src, out),
            StmtKind::For { var, body, .. } => {
                out.insert(span_text(src, var.span).unwrap_or("").to_string());
                assigned_vars(&body.stmts, src, out);
            }
            _ => {}
        }
    }
}

/// The names bound inside the `if`, `while` and `for` statements of `stmts`
/// (however deeply), and the variables of the `for`s among them: the names
/// whose binding may or may not happen, depending on the path. Nested `def`s
/// are separate.
pub(crate) fn conditionally_bound(stmts: &[Stmt], src: &str, out: &mut BTreeSet<String>) {
    for s in stmts {
        match &s.kind {
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (_, b) in branches {
                    assigned_vars(&b.stmts, src, out);
                }
                if let Some(b) = else_block {
                    assigned_vars(&b.stmts, src, out);
                }
            }
            StmtKind::While { body, .. } => assigned_vars(&body.stmts, src, out),
            StmtKind::For { var, body, .. } => {
                out.insert(span_text(src, var.span).unwrap_or("").to_string());
                assigned_vars(&body.stmts, src, out);
            }
            _ => {}
        }
    }
}

/// Whether a statement cannot fall through, by the checker's rule (its
/// `diverged` flag): `return`, `break`, `continue`, an `if` with an `else`
/// whose every branch diverges, and `while true` with no `break`. The
/// checker lets a diverging branch out of merges, so lowering must too. (A
/// `for` whose body always returns diverges in the checker when it is known
/// to run; here it does not, which only costs a guard after it.)
pub(crate) fn stmt_diverges(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Return(_) | StmtKind::Break | StmtKind::Continue => true,
        StmtKind::If {
            branches,
            else_block: Some(e),
        } => branches.iter().all(|(_, b)| block_diverges(&b.stmts)) && block_diverges(&e.stmts),
        StmtKind::While { cond, body } => is_forever(cond) && !exits_stmts(&body.stmts).brk,
        _ => false,
    }
}

/// Whether a block cannot fall through: some statement diverges.
pub(crate) fn block_diverges(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_diverges)
}

/// The statements of a block up to and including the first one that always
/// leaves it: what follows is dead code.
pub(crate) fn live_prefix(stmts: &[Stmt]) -> &[Stmt] {
    match stmts.iter().position(stmt_diverges) {
        Some(i) => &stmts[..=i],
        None => stmts,
    }
}

/// The condition is the literal `true` (possibly parenthesized): the loop
/// can only be left by `break` or `return`.
pub(crate) fn is_forever(cond: &Expr) -> bool {
    let mut e = cond;
    while let ExprKind::Paren(inner) = &e.kind {
        e = inner;
    }
    matches!(e.kind, ExprKind::Literal(Literal::Bool { value: true, .. }))
}

/// How many `return` statements are in `stmts`, not looking into nested
/// `def`s.
pub(crate) fn count_returns(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Return(_) => 1,
            StmtKind::If {
                branches,
                else_block,
            } => {
                branches
                    .iter()
                    .map(|(_, b)| count_returns(&b.stmts))
                    .sum::<usize>()
                    + else_block.as_ref().map_or(0, |b| count_returns(&b.stmts))
            }
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => count_returns(&body.stmts),
            _ => 0,
        })
        .sum()
}

/// Whether some `return` in `stmts` carries a value.
pub(crate) fn returns_value(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::Return(v) => v.is_some(),
        StmtKind::If {
            branches,
            else_block,
        } => {
            branches.iter().any(|(_, b)| returns_value(&b.stmts))
                || else_block.as_ref().is_some_and(|b| returns_value(&b.stmts))
        }
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => returns_value(&body.stmts),
        _ => false,
    })
}
