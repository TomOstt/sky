//! Traversal and printing helpers shared by the passes.

use sky_ir::{BlockId, Function, OpId, OpKind, RegionId, ValueDef, ValueId};

/// Every operation reachable from the function body, in program order,
/// nested regions right after the operation that owns them. Operations that
/// earlier rewrites detached from their block are not visited.
pub(crate) fn reachable_ops(f: &Function) -> Vec<OpId> {
    let mut out = Vec::new();
    walk_region(f, f.body, &mut out);
    out
}

fn walk_region(f: &Function, r: RegionId, out: &mut Vec<OpId>) {
    for &b in &f.region(r).blocks {
        for &o in &f.block(b).ops {
            out.push(o);
            for &inner in &f.op(o).regions {
                walk_region(f, inner, out);
            }
        }
    }
}

/// How many operands (in reachable operations) name each value.
pub(crate) fn use_counts(f: &Function) -> Vec<u32> {
    let mut counts = vec![0; f.values.len()];
    for o in reachable_ops(f) {
        for v in &f.op(o).operands {
            counts[v.index()] += 1;
        }
    }
    counts
}

/// Rewrites every use of `from` (in reachable operations) to `to`.
pub(crate) fn replace_uses(f: &mut Function, from: ValueId, to: ValueId) {
    for o in reachable_ops(f) {
        for v in &mut f.ops[o.index()].operands {
            if *v == from {
                *v = to;
            }
        }
    }
}

/// The operation that defines `v`, if it is an operation result.
pub(crate) fn defining_op(f: &Function, v: ValueId) -> Option<OpId> {
    match f.value(v).def {
        ValueDef::OpResult { op, .. } => Some(op),
        ValueDef::BlockArg { .. } => None,
    }
}

/// Removes `op` from the block it is in. The operation stays in the arena
/// (nothing refers to it any more).
pub(crate) fn detach(f: &mut Function, op: OpId) {
    let block: BlockId = f.op(op).parent;
    f.blocks[block.index()].ops.retain(|o| *o != op);
}

/// `%name` for a named value, `%3` otherwise.
pub(crate) fn value_text(f: &Function, v: ValueId) -> String {
    match &f.value(v).name {
        Some(n) => format!("%{n}"),
        None => v.to_string(),
    }
}

/// One line of text for an operation, for `before` fields.
pub(crate) fn op_text(f: &Function, op: OpId) -> String {
    let o = f.op(op);
    let args: Vec<String> = o.operands.iter().map(|v| value_text(f, *v)).collect();
    let attr = match &o.kind {
        OpKind::ConstFloat { bits } => format!(" {}", f64::from_bits(*bits)),
        OpKind::ConstRational(q) => format!(" {q}"),
        OpKind::ConstBool(b) => format!(" {b}"),
        OpKind::ConstStr(s) => format!(" {s:?}"),
        OpKind::PowConst { exponent } => format!("[{exponent}]"),
        OpKind::Call { callee } => format!(" @{callee}"),
        OpKind::Solve { method } => format!("[{}]", method.mnemonic()),
        _ => String::new(),
    };
    let body = if matches!(o.kind, OpKind::If) {
        format!(
            "{} {} {{ .. }} else {{ .. }}",
            o.kind.mnemonic(),
            args.join("")
        )
    } else if args.is_empty() {
        format!("{}{attr}", o.kind.mnemonic())
    } else {
        format!("{}{attr} {}", o.kind.mnemonic(), args.join(", "))
    };
    if o.results.is_empty() {
        body
    } else {
        let rs: Vec<String> = o.results.iter().map(|v| value_text(f, *v)).collect();
        format!("{} = {body}", rs.join(", "))
    }
}
