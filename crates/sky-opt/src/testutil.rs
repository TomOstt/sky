//! Test-only helpers: a verifier-lite that keeps the hand-built and
//! rewritten IR in these tests sane, and small builders. The real verifier
//! lives elsewhere; this one is deliberately not exported.

use std::collections::HashSet;

use sky_diag::Span;
use sky_ir::{
    BlockId, Function, FunctionBuilder, Module, OpKind, RegionId, Type, ValueDef, ValueId,
};
use sky_units::Dimension;

pub(crate) fn sp() -> Span {
    Span::point(0)
}

pub(crate) fn num() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

pub(crate) fn module_of(f: Function) -> Module {
    Module {
        functions: vec![f],
        ..Module::default()
    }
}

pub(crate) fn main_builder() -> FunctionBuilder {
    FunctionBuilder::new(Module::MAIN, vec![], vec![], sp())
}

pub(crate) fn finish_main(mut b: FunctionBuilder) -> Module {
    b.op(OpKind::Return, vec![], vec![], sp());
    module_of(b.finish())
}

/// The mnemonics of the operations reachable in the entry block, in order.
pub(crate) fn entry_mnemonics(f: &Function) -> Vec<&'static str> {
    f.block(f.entry())
        .ops
        .iter()
        .map(|o| f.op(*o).kind.mnemonic())
        .collect()
}

/// Checks SSA scoping (every operand defined earlier in an enclosing
/// scope), parent links, terminator placement and the shape of `if` and
/// `while`. Returns the first problem.
pub(crate) fn verify(m: &Module) -> Result<(), String> {
    m.functions.iter().try_for_each(verify_function)
}

fn verify_function(f: &Function) -> Result<(), String> {
    let name = &f.name;
    let mut scope = HashSet::new();
    let region = f.region(f.body);
    if region.blocks.len() != 1 {
        return Err(format!("{name}: body must have one block"));
    }
    block(f, region.blocks[0], &mut scope, true).map_err(|e| format!("{name}: {e}"))
}

fn block(
    f: &Function,
    b: BlockId,
    scope: &mut HashSet<ValueId>,
    is_body: bool,
) -> Result<(), String> {
    let data = f.block(b);
    for (i, a) in data.args.iter().enumerate() {
        match f.value(*a).def {
            ValueDef::BlockArg { block, index } if block == b && index as usize == i => {}
            _ => return Err(format!("{a} has a wrong def for arg {i} of {b}")),
        }
        scope.insert(*a);
    }
    let Some(last) = data.ops.last() else {
        return Err(format!("{b} is empty"));
    };
    if !f.op(*last).kind.properties().terminator {
        return Err(format!("{b} does not end in a terminator"));
    }
    if is_body && !matches!(f.op(*last).kind, OpKind::Return) {
        return Err("the body must end in return".into());
    }
    for (pos, &id) in data.ops.iter().enumerate() {
        let op = f.op(id);
        if op.parent != b {
            return Err(format!("{id} has parent {} but is in {b}", op.parent));
        }
        let terminator = op.kind.properties().terminator;
        if terminator != (pos + 1 == data.ops.len()) {
            return Err(format!("{id}: terminator in the wrong place"));
        }
        if matches!(op.kind, OpKind::Return) && !is_body {
            return Err(format!("{id}: return inside a region"));
        }
        for v in &op.operands {
            if !scope.contains(v) {
                return Err(format!(
                    "{id} ({}) uses {v} out of scope",
                    op.kind.mnemonic()
                ));
            }
        }
        match &op.kind {
            OpKind::If => {
                if op.operands.len() != 1 || op.regions.len() != 2 {
                    return Err(format!("{id}: an if takes 1 operand and 2 regions"));
                }
                for r in &op.regions {
                    let ys = region_yield_arity(f, *r, scope)?;
                    if ys != op.results.len() {
                        return Err(format!("{id}: an arm yields {ys}, if has results"));
                    }
                }
            }
            OpKind::While => {
                for r in &op.regions {
                    let mut inner = scope.clone();
                    let blocks = &f.region(*r).blocks;
                    if blocks.len() != 1 {
                        return Err(format!("{id}: region needs one block"));
                    }
                    block(f, blocks[0], &mut inner, false)?;
                }
            }
            _ if !op.regions.is_empty() => return Err(format!("{id}: unexpected regions")),
            _ => {}
        }
        for (i, r) in op.results.iter().enumerate() {
            match f.value(*r).def {
                ValueDef::OpResult { op: o, index } if o == id && index as usize == i => {}
                _ => return Err(format!("{r} has a wrong def")),
            }
            scope.insert(*r);
        }
    }
    Ok(())
}

fn region_yield_arity(
    f: &Function,
    r: RegionId,
    scope: &HashSet<ValueId>,
) -> Result<usize, String> {
    let blocks = &f.region(r).blocks;
    if blocks.len() != 1 {
        return Err(format!("{r} needs one block"));
    }
    let mut inner = scope.clone();
    block(f, blocks[0], &mut inner, false)?;
    let last = *f.block(blocks[0]).ops.last().expect("checked non-empty");
    match f.op(last).kind {
        OpKind::Yield => Ok(f.op(last).operands.len()),
        _ => Err(format!("{r} must end in yield")),
    }
}
