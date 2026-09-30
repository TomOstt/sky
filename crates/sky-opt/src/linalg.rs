//! Exact linear-algebra rewrites: transposes that are never materialized,
//! and products that use a matrix's structure.
//!
//! Everything here produces **bit-for-bit the same values and the same
//! errors** as the program it rewrites, under the reference interpreter, so
//! neither the tree-walker nor the checker needs to know about it (the
//! rewrites that change rounding are decided by the checker, in
//! `sky_types::chain`, so both engines follow them). Each explanation says
//! why it is exact.
//!
//! | pass | rule | rewrite |
//! |---|---|---|
//! | [`transposes`] | `LA-TRANSPOSE-001` | `transpose(A) @ B` to `matmul[ta](A, B)` (and `tb`) |
//! | [`structure`] | `LA-GRAM-001` | `Aᵀ @ A` and `A @ Aᵀ` to `matmul[ta,alg=syrk]`: one triangle, mirrored |
//! | [`structure`] | `LA-DIAGONAL-001` | `D @ X`, `X @ D` for a Diagonal `D` to `diag.mul` (a scaling) |
//! | [`structure`] | `LA-SYMMETRIC-001` | `transpose(S)` of a Symmetric `S` to `S` |
//!
//! # Why each is exact
//!
//! * **`matmul[ta]`** is defined as the transpose followed by the product
//!   (`sky_interp::solve::matmul_ex` does exactly that), so it is the same
//!   computation. A fast kernel reads the operand with swapped strides
//!   instead of copying it; the entries and the accumulation order are
//!   unchanged.
//! * **Gram products.** Entry `(i, j)` of `AᵀA` is `Σₚ aₚᵢ·aₚⱼ` and entry
//!   `(j, i)` is `Σₚ aₚⱼ·aₚᵢ`: IEEE multiplication commutes, so the terms
//!   are bitwise equal and are added in the same order; the two entries are
//!   the same bits. Computing one triangle and mirroring it halves the
//!   multiplications. The result is bitwise symmetric, so its type gains the
//!   `Symmetric` tag (which lets the next rule fire on its transpose).
//! * **Diagonal scaling.** Entry `(i, j)` of `D @ X` is a sum started at
//!   `+0.0` of `dᵢₚ·xₚⱼ`. Every term but `p = i` multiplies an exact zero
//!   (all values are finite), which is `±0.0`; adding `±0.0` to a sum never
//!   changes it, except that `+0.0 + -0.0` is `+0.0`. So the sum is
//!   `0.0 + dᵢᵢ·xᵢⱼ`, the value `diag.mul` computes (the `0.0 +` is what
//!   keeps the sign of a zero the same). Overflow of the one product is the
//!   same error.
//! * **`transpose(S)` is `S`** because Symmetric means `aᵢⱼ == aⱼᵢ`
//!   exactly (the spec's definition), so the transpose has the same bits.
//!
//! A tag is a promise the checker proved or the program declared: the
//! interpreter verifies `Diagonal` when it scales (a nonzero off-diagonal
//! entry is a run-time error, as a broken PositiveDefinite is for a
//! Cholesky solve).

use sky_ir::{Function, Module, OpId, OpKind, Side, Type, ValueId};
use sky_types::flops::Flops;
use sky_types::shape::{Shape, ShapeDim};
use sky_types::{Algorithm, Tag, TagSet};

use crate::rewrite::Rewrite;
use crate::util::{defining_op, detach, reachable_ops, replace_uses, use_counts};

pub const TRANSPOSE_RULE: &str = "LA-TRANSPOSE-001";
pub const GRAM_RULE: &str = "LA-GRAM-001";
pub const DIAGONAL_RULE: &str = "LA-DIAGONAL-001";
pub const SYMMETRIC_RULE: &str = "LA-SYMMETRIC-001";

/// Whether `op` is still in its block (a rewrite may have detached it since
/// the list of operations was taken).
fn live(f: &Function, op: OpId) -> bool {
    f.block(f.op(op).parent).ops.contains(&op)
}

/// The source name of `v`, or `fallback` for a temporary (the explanations
/// are read by users, who never saw `%3`).
fn name(f: &Function, v: ValueId, fallback: &str) -> String {
    match &f.value(v).name {
        Some(n) => n.clone(),
        None => fallback.to_string(),
    }
}

fn number(f: &Function, v: ValueId) -> Option<&sky_ir::NumberType> {
    match f.ty(v) {
        Type::Number(n) => Some(n),
        _ => None,
    }
}

/// `(rows, cols)` of an operand of a product: a matrix's own, a vector as a
/// row on the left and a column on the right.
fn as_matrix(shape: &Shape, left: bool) -> Option<(ShapeDim, ShapeDim)> {
    match shape.dims() {
        [r, c] => Some((r.clone(), c.clone())),
        [n] if left => Some((ShapeDim::Known(1), n.clone())),
        [n] => Some((n.clone(), ShapeDim::Known(1))),
        _ => None,
    }
}

/// The flops of `a @ b` for shapes `a` and `b`.
fn matmul_flops(a: &Shape, b: &Shape) -> Option<Flops> {
    let (m, k) = as_matrix(a, true)?;
    let (_, n) = as_matrix(b, false)?;
    Some(Flops::matmul(&m, &k, &n))
}

fn is_diagonal(t: &Type) -> bool {
    matches!(t, Type::Number(n)
        if matches!(n.shape.dims(), [a, b] if a == b) && n.tags.closure().contains(Tag::Diagonal))
}

fn is_symmetric(t: &Type) -> bool {
    matches!(t, Type::Number(n)
        if matches!(n.shape.dims(), [a, b] if a == b) && n.tags.closure().contains(Tag::Symmetric))
}

// ----------------------------------------------------------- transposes

/// `transpose(A) @ B` and `A @ transpose(B)` to `matmul[ta]` and
/// `matmul[tb]`, everywhere. A transpose nothing else uses is removed.
pub fn transposes(m: &mut Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        for op in reachable_ops(f) {
            if live(f, op) {
                if let Some(rw) = fold_transposes(f, op) {
                    out.push(rw);
                }
            }
        }
    }
    out
}

/// The transposed matrix `x` and the op, if `v` is `transpose(x)` of a
/// matrix.
fn transposed(f: &Function, v: ValueId) -> Option<(OpId, ValueId)> {
    let t = defining_op(f, v)?;
    let op = f.op(t);
    if !matches!(op.kind, OpKind::Transpose) {
        return None;
    }
    let &[x] = &op.operands[..] else {
        return None;
    };
    (number(f, x)?.shape.rank() == 2).then_some((t, x))
}

fn fold_transposes(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    if !matches!(op.kind, OpKind::MatMul) {
        return None;
    }
    let &[a, b] = &op.operands[..] else {
        return None;
    };
    let (ta, tb) = (transposed(f, a), transposed(f, b));
    if ta.is_none() && tb.is_none() {
        return None;
    }
    let (x, y) = (ta.map_or(a, |t| t.1), tb.map_or(b, |t| t.1));

    // What the operation reads, and where.
    let spans = |i: usize, t: Option<(OpId, ValueId)>| match t {
        Some((tid, _)) => f.op(tid).operand_spans.first().copied(),
        None => op.operand_spans.get(i).copied(),
    };
    let operand_spans = match (spans(0, ta), spans(1, tb)) {
        (Some(l), Some(r)) if op.operand_spans.len() == 2 => vec![l, r],
        _ => Vec::new(),
    };
    let flops = number(f, a)
        .zip(number(f, b))
        .and_then(|(p, q)| matmul_flops(&p.shape, &q.shape));
    let copies: Vec<String> = [ta, tb]
        .iter()
        .flatten()
        .filter_map(|(_, m)| number(f, *m))
        .map(|n| {
            let dims = n.shape.dims();
            format!("{}·{}", dims[0], dims[1])
        })
        .collect();
    let (xn, yn) = (name(f, x, "A"), name(f, y, "B"));
    let before = format!(
        "{}{} @ {}{}",
        xn,
        if ta.is_some() { "ᵀ" } else { "" },
        yn,
        if tb.is_some() { "ᵀ" } else { "" }
    );
    let flag = match (ta.is_some(), tb.is_some()) {
        (true, true) => "ta,tb",
        (true, false) => "ta",
        _ => "tb",
    };
    let after = format!("matmul[{flag}]({xn}, {yn})");
    let span = op.span;

    {
        let data = &mut f.ops[id.index()];
        data.kind = OpKind::MatMulEx {
            ta: ta.is_some(),
            tb: tb.is_some(),
            algorithm: Algorithm::Auto,
        };
        data.operands = vec![x, y];
        data.operand_spans = operand_spans;
    }
    let counts = use_counts(f);
    for (tid, tv) in [ta.map(|t| (t.0, a)), tb.map(|t| (t.0, b))]
        .into_iter()
        .flatten()
    {
        if counts[tv.index()] == 0 {
            detach(f, tid);
        }
    }

    let mut because = vec![
        format!(
            "the transposed operand is a matrix: `matmul[{flag}]` reads it transposed, so the {} element copy that `transpose` makes is never made",
            copies.join(" and ")
        ),
        format!(
            "exact: `matmul[{flag}]` is defined as the transpose followed by the product, so every bit and every error is the same; a kernel may read the operand with swapped strides"
        ),
    ];
    if let Some(fl) = flops {
        because.push(format!("the flops are unchanged: {fl}"));
    }
    Some(Rewrite {
        pass: "transposes",
        rule: TRANSPOSE_RULE,
        span,
        before,
        after,
        because,
    })
}

// ------------------------------------------------------------ structure

/// Gram products, Diagonal scaling, and transposes of Symmetric matrices,
/// in this order (the first gives the third something to do).
pub fn structure(m: &mut Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        let ops = reachable_ops(f);
        for &op in &ops {
            if live(f, op) {
                if let Some(rw) = gram(f, op) {
                    out.push(rw);
                }
            }
        }
        for &op in &ops {
            if live(f, op) {
                if let Some(rw) = diagonal(f, op) {
                    out.push(rw);
                }
            }
        }
        for &op in &ops {
            if live(f, op) {
                if let Some(rw) = symmetric_transpose(f, op) {
                    out.push(rw);
                }
            }
        }
    }
    out
}

fn gram(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    let OpKind::MatMulEx {
        ta,
        tb,
        algorithm: Algorithm::Auto,
    } = op.kind
    else {
        return None;
    };
    let &[a, b] = &op.operands[..] else {
        return None;
    };
    if a != b || ta == tb {
        return None;
    }
    let n = number(f, a)?.clone();
    let [rows, cols] = n.shape.dims() else {
        return None;
    };
    let result = *op.results.first()?;
    // `Aᵀ @ A` is cols × cols with `rows` terms per entry; `A @ Aᵀ` is
    // rows × rows with `cols` terms.
    let (size, terms) = if ta { (cols, rows) } else { (rows, cols) };
    let before_flops = Flops::matmul(size, terms, size);
    let after_flops =
        Flops::product(3, &[size, size, terms]).plus(&Flops::product(3, &[size, terms]));
    let an = name(f, a, "A");
    let (before, after) = if ta {
        (
            format!("{an}ᵀ @ {an}"),
            format!("matmul[ta,alg=syrk]({an}, {an})"),
        )
    } else {
        (
            format!("{an} @ {an}ᵀ"),
            format!("matmul[tb,alg=syrk]({an}, {an})"),
        )
    };
    if let OpKind::MatMulEx { algorithm, .. } = &mut f.ops[id.index()].kind {
        *algorithm = Algorithm::Syrk;
    }
    // The result is bitwise symmetric: say so.
    if let Type::Number(r) = &mut f.values[result.index()].ty {
        r.tags = r.tags.meet(TagSet::of(Tag::Symmetric));
    }
    Some(Rewrite {
        pass: "structure",
        rule: GRAM_RULE,
        span: op.span,
        before,
        after,
        because: vec![
            format!("{an} is multiplied by its own transpose: entries (i, j) and (j, i) sum the same products (`aₚᵢ·aₚⱼ` and `aₚⱼ·aₚᵢ`, and multiplication commutes) in the same order"),
            format!("one triangle is computed and mirrored: {after_flops} flops instead of {before_flops}"),
            "exact: the result is bit for bit the general product's, and bitwise symmetric (its type gains the Symmetric tag)".to_string(),
        ],
    })
}

fn diagonal(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    if !matches!(op.kind, OpKind::MatMul) {
        return None;
    }
    let &[a, b] = &op.operands[..] else {
        return None;
    };
    let (side, d, x) = if is_diagonal(f.ty(a)) {
        (Side::Left, a, b)
    } else if is_diagonal(f.ty(b)) {
        (Side::Right, b, a)
    } else {
        return None;
    };
    let (dn, xn) = (name(f, d, "D"), name(f, x, "X"));
    let (dt, xt) = (number(f, d)?.clone(), number(f, x)?.clone());
    let before_flops = number(f, a)
        .zip(number(f, b))
        .and_then(|(p, q)| matmul_flops(&p.shape, &q.shape));
    // `n·k` multiplications: the entries of `X`.
    let entries: Vec<&ShapeDim> = xt.shape.dims().iter().collect();
    let after_flops = Flops::product(3, &entries);
    let (before, after) = match side {
        Side::Left => (
            format!("{dn} @ {xn}"),
            format!("diag.mul[left]({dn}, {xn})"),
        ),
        Side::Right => (
            format!("{xn} @ {dn}"),
            format!("diag.mul[right]({xn}, {dn})"),
        ),
    };
    f.ops[id.index()].kind = OpKind::DiagMul { side };
    let mut because = vec![
        format!(
            "{dn} is {} and Diagonal: every off-diagonal entry is exactly zero, so the product scales the {} of {xn} by its diagonal",
            dt.shape,
            if xt.shape.rank() == 1 {
                "entries"
            } else if side == Side::Left {
                "rows"
            } else {
                "columns"
            }
        ),
        "exact: entry (i, j) is `0.0 + d·x` in either form (the other terms add ±0 to a sum that starts at +0.0), so every bit is the same, including the sign of a zero".to_string(),
    ];
    if let Some(fl) = before_flops {
        because.push(format!(
            "{after_flops} multiplications instead of {fl} flops"
        ));
    }
    Some(Rewrite {
        pass: "structure",
        rule: DIAGONAL_RULE,
        span: op.span,
        before,
        after,
        because,
    })
}

fn symmetric_transpose(f: &mut Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id).clone();
    if !matches!(op.kind, OpKind::Transpose) {
        return None;
    }
    let &[s] = &op.operands[..] else {
        return None;
    };
    let &[result] = &op.results[..] else {
        return None;
    };
    if !is_symmetric(f.ty(s)) {
        return None;
    }
    let sn = name(f, s, "S");
    let shape = number(f, s)?.shape.clone();
    replace_uses(f, result, s);
    detach(f, id);
    Some(Rewrite {
        pass: "structure",
        rule: SYMMETRIC_RULE,
        span: op.span,
        before: format!("{sn}ᵀ"),
        after: sn.clone(),
        because: vec![
            format!("{sn} is {shape} and Symmetric: aᵢⱼ == aⱼᵢ exactly, so its transpose has the same bits"),
            format!("exact: no {} element copy is made", element_count(&shape)),
        ],
    })
}

fn element_count(shape: &Shape) -> String {
    shape
        .dims()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("·")
}
