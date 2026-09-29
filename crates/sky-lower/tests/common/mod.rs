//! Helpers shared by the lowering tests: lower a source string, and look at
//! the result through the `sky_ir` API (a compact indented dump, flat op
//! listings, counts). `sky_lower::lower_source` verifies every module it
//! returns, so `lower_ok` only ever sees valid SIR.

#![allow(dead_code)]

pub mod eval;

use std::fmt::Write as _;

use sky_ir::{BlockId, Function, Module, OpId, OpKind, RegionId, Type, ValueId};

/// Lowers `src` (lex, parse, check, lower), panicking with the rendered
/// diagnostics on failure.
pub fn lower_ok(src: &str) -> Module {
    match sky_lower::lower_source(src) {
        Ok(m) => {
            check_structure(&m);
            m
        }
        Err(e) => panic!("lowering failed:\n{e}\nsource:\n{src}"),
    }
}

/// The rendered error of a source that must not lower.
pub fn lower_err(src: &str) -> String {
    match sky_lower::lower_source(src) {
        Ok(m) => panic!("expected an error, lowered:\n{}", dump_module(&m)),
        Err(e) => e,
    }
}

pub fn main_fn(m: &Module) -> &Function {
    m.function(Module::MAIN).expect("<main>")
}

pub fn func<'m>(m: &'m Module, name: &str) -> &'m Function {
    m.function(name)
        .unwrap_or_else(|| panic!("no function {name}: {:?}", names(m)))
}

pub fn names(m: &Module) -> Vec<&str> {
    m.functions.iter().map(|f| f.name.as_str()).collect()
}

/// A stand-in for the verifier: the structural invariants lowering must keep
/// (every block ends in exactly one terminator, and no value is used before
/// its definition in the block order).
pub fn check_structure(m: &Module) {
    for f in &m.functions {
        let mut defined = std::collections::HashSet::new();
        walk_region(f, f.body, &mut defined, true);
        // The body's last op is `return`.
        let entry = f.block(f.entry());
        let last = *entry.ops.last().expect("a body has ops");
        assert!(
            matches!(f.op(last).kind, OpKind::Return),
            "{}: the body must end in return\n{}",
            f.name,
            dump(f)
        );
        assert_eq!(
            f.op(last).operands.len(),
            f.results.len(),
            "{}: return arity\n{}",
            f.name,
            dump(f)
        );
    }
}

fn walk_region(
    f: &Function,
    r: RegionId,
    defined: &mut std::collections::HashSet<ValueId>,
    _top: bool,
) {
    for &b in &f.region(r).blocks {
        walk_block(f, b, defined);
    }
}

fn walk_block(f: &Function, b: BlockId, defined: &mut std::collections::HashSet<ValueId>) {
    let block = f.block(b);
    defined.extend(block.args.iter().copied());
    let n = block.ops.len();
    for (i, &o) in block.ops.iter().enumerate() {
        let op = f.op(o);
        let term = op.kind.properties().terminator;
        assert_eq!(
            term,
            i + 1 == n,
            "{}: a terminator must be last, and only last: {}\n{}",
            f.name,
            op.kind.mnemonic(),
            dump(f)
        );
        for v in &op.operands {
            assert!(
                defined.contains(v),
                "{}: {v} used before it is defined by {}\n{}",
                f.name,
                op.kind.mnemonic(),
                dump(f)
            );
        }
        for &r in &op.regions {
            // Values defined in a region are not visible outside it.
            let mut inner = defined.clone();
            walk_region(f, r, &mut inner, false);
        }
        defined.extend(op.results.iter().copied());
    }
}

pub fn ty(t: &Type) -> String {
    match t {
        Type::Number(n) => {
            let dim = n
                .dim
                .name()
                .map_or_else(|| n.dim.to_string(), str::to_string);
            let mut s = dim;
            if n.shape.rank() > 0 {
                s = format!("{}:{s}", n.shape);
            }
            if !n.tags.is_empty() {
                s = format!("{s}{}", n.tags);
            }
            s
        }
        Type::Bool => "bool".into(),
        Type::Str => "str".into(),
        Type::Nothing => "nothing".into(),
        Type::Unknown => "?".into(),
    }
}

fn attrs(k: &OpKind) -> String {
    match k {
        OpKind::ConstFloat { bits } => format!(" {}", f64::from_bits(*bits)),
        OpKind::ConstRational(r) => format!(" {r}"),
        OpKind::ConstBool(b) => format!(" {b}"),
        OpKind::ConstStr(s) => format!(" {s:?}"),
        OpKind::PowConst { exponent } => format!(" [{exponent}]"),
        OpKind::Call { callee } => format!(" @{callee}"),
        _ => String::new(),
    }
}

fn dump_block(f: &Function, b: BlockId, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    let block = f.block(b);
    if !block.args.is_empty() {
        let args: Vec<String> = block
            .args
            .iter()
            .map(|v| format!("{v}{}: {}", name_of(f, *v), ty(f.ty(*v))))
            .collect();
        let _ = writeln!(out, "{pad}^({})", args.join(", "));
    }
    for &o in &block.ops {
        dump_op(f, o, indent, out);
    }
}

fn name_of(f: &Function, v: ValueId) -> String {
    f.value(v)
        .name
        .as_ref()
        .map_or(String::new(), |n| format!("'{n}"))
}

fn dump_op(f: &Function, o: OpId, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    let op = f.op(o);
    let res: Vec<String> = op
        .results
        .iter()
        .map(|v| format!("{v}{}: {}", name_of(f, *v), ty(f.ty(*v))))
        .collect();
    let opnds: Vec<String> = op.operands.iter().map(|v| v.to_string()).collect();
    let lhs = if res.is_empty() {
        String::new()
    } else {
        format!("{} = ", res.join(", "))
    };
    let _ = writeln!(
        out,
        "{pad}{lhs}{}{} {}",
        op.kind.mnemonic(),
        attrs(&op.kind),
        opnds.join(", ")
    );
    for (i, &r) in op.regions.iter().enumerate() {
        let _ = writeln!(out, "{pad}  region {i} {{");
        for &b in &f.region(r).blocks {
            dump_block(f, b, indent + 2, out);
        }
        let _ = writeln!(out, "{pad}  }}");
    }
}

pub fn dump(f: &Function) -> String {
    let mut out = String::new();
    let params: Vec<String> = f
        .param_values()
        .iter()
        .map(|v| format!("{v}{}: {}", name_of(f, *v), ty(f.ty(*v))))
        .collect();
    let results: Vec<String> = f.results.iter().map(ty).collect();
    let effects: Vec<String> = f.effects.iter().map(|e| format!("{e:?}")).collect();
    let _ = writeln!(
        out,
        "fn {}({}) -> ({}) effects({}) {{",
        f.name,
        params.join(", "),
        results.join(", "),
        effects.join(", ")
    );
    let entry = f.entry();
    for &o in &f.block(entry).ops {
        dump_op(f, o, 1, &mut out);
    }
    out.push_str("}\n");
    out
}

pub fn dump_module(m: &Module) -> String {
    m.functions.iter().map(dump).collect::<Vec<_>>().join("\n")
}

/// Mnemonics of the entry block's operations, in order.
pub fn top_kinds(f: &Function) -> Vec<&'static str> {
    f.block(f.entry())
        .ops
        .iter()
        .map(|&o| f.op(o).kind.mnemonic())
        .collect()
}

/// Mnemonics of the operations of block `b`, in order.
pub fn block_kinds(f: &Function, b: BlockId) -> Vec<&'static str> {
    f.block(b)
        .ops
        .iter()
        .map(|&o| f.op(o).kind.mnemonic())
        .collect()
}

/// How many operations of the function (nested ones included) have this
/// mnemonic.
pub fn count(f: &Function, mnemonic: &str) -> usize {
    f.ops
        .iter()
        .filter(|o| o.kind.mnemonic() == mnemonic)
        .count()
}

/// How deeply regions nest in `f`: 0 for a function with no control flow,
/// otherwise the number of regions on the longest path from the body down to
/// an operation (the verifier's `MAX_REGION_DEPTH` bounds this). Iterative:
/// the IR it measures may be thousands of levels deep.
pub fn region_depth(f: &Function) -> usize {
    // A block's depth is its owner's block depth plus one; blocks are
    // created after the operation that owns their region, so one pass over
    // the arena in index order sees every owner first.
    let mut depth = vec![0usize; f.blocks.len()];
    let mut deepest = 0;
    for b in 0..f.blocks.len() {
        let region = f.blocks[b].parent.expect("a block has a region");
        depth[b] = match f.regions[region.index()].parent {
            None => 0,
            Some(owner) => depth[f.ops[owner.index()].parent.index()] + 1,
        };
        deepest = deepest.max(depth[b]);
    }
    deepest
}

/// The operations of the entry block with this mnemonic.
pub fn top_ops(f: &Function, mnemonic: &str) -> Vec<OpId> {
    f.block(f.entry())
        .ops
        .iter()
        .copied()
        .filter(|&o| f.op(o).kind.mnemonic() == mnemonic)
        .collect()
}

/// Lexes, parses and checks `src` (which must check without errors) and
/// lowers it, returning the `LowerError` itself.
pub fn try_lower(src: &str) -> Result<Module, sky_lower::LowerError> {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let checked = sky_types::check_module(&ast, src);
        assert!(!checked.has_errors(), "{:#?}", checked.diagnostics);
        sky_lower::lower(&ast, src, &checked)
    })
}

/// The last value named `name` in `f` (a variable's latest binding).
pub fn named(f: &Function, name: &str) -> ValueId {
    (0..f.values.len() as u32)
        .rev()
        .map(ValueId)
        .find(|v| f.value(*v).name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no value named {name}\n{}", dump(f)))
}

/// The source text of an operation's span.
pub fn span_text<'s>(src: &'s str, f: &Function, o: OpId) -> &'s str {
    &src[f.op(o).span.range()]
}

/// The operation that defines `v`, if it is an operation result.
pub fn def_op(f: &Function, v: ValueId) -> Option<OpId> {
    match f.value(v).def {
        sky_ir::ValueDef::OpResult { op, .. } => Some(op),
        sky_ir::ValueDef::BlockArg { .. } => None,
    }
}

/// The first top-level operation with this mnemonic.
pub fn top_op(f: &Function, mnemonic: &str) -> OpId {
    top_ops(f, mnemonic)
        .first()
        .copied()
        .unwrap_or_else(|| panic!("no top-level {mnemonic}\n{}", dump(f)))
}

/// The blocks of an operation's regions, one per region (v1 regions have one
/// block).
pub fn arms(f: &Function, o: OpId) -> Vec<BlockId> {
    f.op(o)
        .regions
        .iter()
        .map(|&r| f.region(r).blocks[0])
        .collect()
}
