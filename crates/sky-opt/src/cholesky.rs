//! `A⁻¹ @ b` to a Cholesky solve.
//!
//! This is the SIR version of the decision `sky_types::cholesky_rewrite`
//! makes in the checker, and its explanation keeps that text.
//!
//! A `matmul(inverse(A), b)` becomes `solve[cholesky](A, b)` when
//!
//! 1. the left operand of the `matmul` is the result of an `inverse` (a
//!    `matmul` of some other value, or of `A ^ -1` lowered as a power, is
//!    never rewritten);
//! 2. `A`'s tags, closed, include `PositiveDefinite`;
//! 3. the types fit: `A` is a square matrix, `b` a vector or a matrix with
//!    as many rows as `A`, and the `matmul`'s result has `b`'s shape. Shape
//!    dimensions are compared as they are (the checker already resolved
//!    them); an unresolved mismatch means no rewrite;
//! 4. `A` and `b` are `Number`-typed, never `Unknown`.
//!
//! The `matmul` operation is rewritten in place, so its result value, name
//! and users are unchanged, and its span is joined with the `inverse`'s.
//! The `inverse` is deleted when nothing else uses it. If something does,
//! it is kept, still computed for those users, and the rewrite notes it.

use sky_ir::{Function, Module, OpId, OpKind, SolveMethod, Type};
use sky_types::Tag;

use crate::rewrite::Rewrite;
use crate::util::{defining_op, detach, reachable_ops, use_counts};

/// Applies the Cholesky rewrite everywhere it holds.
pub fn cholesky(m: &mut Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        for op in reachable_ops(f) {
            if let Some(rw) = rewrite_matmul(f, op) {
                out.push(rw);
            }
        }
    }
    out
}

fn rewrite_matmul(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    if !matches!(op.kind, OpKind::MatMul) {
        return None;
    }
    let &[lhs, b] = &op.operands[..] else {
        return None;
    };
    let inv_id = defining_op(f, lhs)?;
    let inv = f.op(inv_id).clone();
    if !matches!(inv.kind, OpKind::Inverse) {
        return None;
    }
    let &[a] = &inv.operands[..] else {
        return None;
    };
    let (Type::Number(a_ty), Type::Number(b_ty)) = (f.ty(a), f.ty(b)) else {
        return None;
    };
    let &[result] = &op.results[..] else {
        return None;
    };
    let Type::Number(r_ty) = f.ty(result) else {
        return None;
    };
    if !a_ty.tags.closure().contains(Tag::PositiveDefinite) {
        return None;
    }
    let (ad, bd) = (a_ty.shape.dims(), b_ty.shape.dims());
    let square = ad.len() == 2 && ad[0] == ad[1];
    let fits = matches!(bd.len(), 1 | 2) && !bd.is_empty() && !ad.is_empty() && bd[0] == ad[0];
    if !square || !fits || r_ty.shape != b_ty.shape {
        return None;
    }

    let name_of = |v| f.value(v).name.clone();
    let (a_name, b_name) = (name_of(a), name_of(b));
    let (a_shape, b_shape, r_shape) = (
        a_ty.shape.to_string(),
        b_ty.shape.to_string(),
        r_ty.shape.to_string(),
    );

    // Rewrite in place; the result value keeps its identity.
    let span = op.span.join(inv.span);
    {
        let data = &mut f.ops[id.index()];
        data.kind = OpKind::Solve {
            method: SolveMethod::Cholesky,
        };
        data.operands = vec![a, b];
        data.span = span;
    }
    let other_users = use_counts(f)[lhs.index()] > 0;
    if !other_users {
        detach(f, inv_id);
    }

    let (an, bn) = (
        a_name.clone().unwrap_or_else(|| "A".to_string()),
        b_name.clone().unwrap_or_else(|| "b".to_string()),
    );
    let mut because = Vec::new();
    if a_name.is_some() || b_name.is_some() {
        because.push(format!("A is `{an}` and b is `{bn}`"));
    }
    because.push(format!("A is {a_shape} and PositiveDefinite"));
    because.push(format!("b is {b_shape}"));
    because.push("Factor A = L·Lᵀ (L lower-triangular), solve L·y = b, then Lᵀ·x = y".to_string());
    because.push(if other_users {
        format!("x = A⁻¹·b is {r_shape}; A⁻¹ is still computed for its other users")
    } else {
        format!("x = A⁻¹·b is {r_shape}, and A⁻¹ is never formed")
    });
    Some(Rewrite {
        pass: "cholesky",
        rule: "LA-CHOLESKY-001",
        span,
        before: format!("{an}⁻¹ @ {bn}"),
        after: format!("solve[cholesky]({an}, {bn})"),
        because,
    })
}
