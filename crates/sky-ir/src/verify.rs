//! The SIR verifier.
//!
//! [`verify`] checks every invariant `docs/design/ir.md` and the data-model
//! docs promise, so a pass that breaks one is caught right after it runs
//! (the pass manager calls this between passes in debug builds and tests).
//! It is total: any `Module`, however malformed (ids out of range, regions
//! that contain themselves, arenas that disagree with the tree), gives a
//! list of errors, never a panic and never unbounded recursion.
//!
//! # What is checked
//!
//! * **Arena integrity.** Every id is in range; each value's `def` names
//!   the place it actually appears; `parent` links of operations, blocks and
//!   regions agree with the tree; every operation, block and region is
//!   reachable from the function body exactly once. Arena entries not
//!   reachable from the body (an operation a pass removed from its block)
//!   are ignored: passes may leave garbage, but a reachable operation that
//!   uses a value defined only by such garbage is an error, because the
//!   value's definition is not in scope.
//! * **SSA and scoping.** A use is legal if the value is defined earlier in
//!   the same block, is an argument of an enclosing block, or is defined
//!   earlier in an enclosing block. An operation's own results are not
//!   visible in its own regions, and values defined inside a region are not
//!   visible after it (in particular not from a `while`'s `before` region
//!   in its `after` region: values cross through the region arguments).
//! * **Region shape.** v1 regions have exactly one block; every block ends
//!   in exactly one terminator, its last operation; `return` ends the
//!   function body, `yield` ends `if` arms, `while` `after` regions and
//!   `fused` bodies, `condition` ends `while` `before` regions; `if`,
//!   `while` and `fused` follow the rules in [`crate::op`]. A `fused` body
//!   is scalar-only, pure or failing only, without regions or captures.
//! * **Types per operation.** Operand and result counts, dimensions
//!   (`checked_mul`, `pow_rat`, ...), shapes (through [`sky_types::rules`],
//!   so the checker's rules and the verifier's cannot drift), calls against
//!   the callee's signature, and `return` against the function's results.
//!   [`Type::Unknown`] is compatible with anything: an operation with an
//!   unknown operand skips the checks that need its type, and an unknown
//!   result is accepted (only constants must be typed exactly).
//! * **Tags.** Every [`NumberType`]'s tag set is closed, and a non-empty tag
//!   set sits on a square matrix (the invariant of `TensorType`).
//!
//! * **Effects.** A function's declared `effects` include what
//!   [`crate::effects::infer`] computes from its body and its callees (the
//!   design's "effect consistency"). Declaring *more* than inferred is fine:
//!   a pass that removes code leaves the old declaration valid, and
//!   [`crate::effects::InferEffects`] tightens it. This check runs only when
//!   the module has no other error, because inference reads the structure
//!   the other checks establish.

mod arrays;
mod fused;

use std::fmt;

use sky_diag::{Diagnostic, Span};
use sky_types::rules::{self, TensorType};
use std::collections::BTreeMap;

use sky_types::{Shape, ShapeDim, ShapeUnifier, Tag};
use sky_units::Dimension;

use crate::effects;
use crate::ir::{
    BlockId, EffectSet, Function, Module, NumberType, OpData, OpId, RegionId, Type, ValueDef,
    ValueId,
};
use crate::op::{Algorithm, AssertKind, CmpOp, OpKind, Side, SolveMethod};
use crate::print::effect_name;

/// The deepest nesting of regions the verifier, printer and parser accept.
/// It bounds their recursion. It must leave room above Sky's own source
/// limit (256 nested blocks): lowering adds regions per source level (a
/// `while` has two, the function body one), so 256 source levels become a
/// little more than 256 region levels. Lowering keeps the ratio a small
/// constant (it never nests one region per `elif` arm or `and` operand), so
/// 4× the source limit is ample.
pub const MAX_REGION_DEPTH: usize = 1024;

/// Most errors reported for one module; later ones are dropped so a badly
/// broken function cannot produce megabytes of output.
const MAX_ERRORS: usize = 100;

/// One violated invariant.
#[derive(Clone, PartialEq, Debug)]
pub struct VerifyError {
    pub message: String,
    /// The function the error is in (empty for module-level errors).
    pub function: String,
    /// The offending operation, if the error is about one.
    pub op: Option<OpId>,
    /// The offending value, if the error is about one.
    pub value: Option<ValueId>,
    /// The Sky source of the offending operation (or of the function).
    pub span: Span,
}

impl VerifyError {
    /// An error diagnostic pointing at the source the operation came from.
    pub fn to_diagnostic(&self) -> Diagnostic {
        Diagnostic::error(format!("invalid IR: {}", self.message), self.span)
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.function.is_empty() {
            write!(f, "@{}: ", self.function)?;
        }
        f.write_str(&self.message)?;
        match (self.op, self.value) {
            (Some(o), Some(v)) => write!(f, " ({o}, {v})"),
            (Some(o), None) => write!(f, " ({o})"),
            (None, Some(v)) => write!(f, " ({v})"),
            (None, None) => Ok(()),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Checks all of `m`'s invariants. `Ok(())` means the module is well formed.
pub fn verify(m: &Module) -> Result<(), Vec<VerifyError>> {
    // Region nesting (up to MAX_REGION_DEPTH) is walked recursively: run on a
    // stack of known size, whatever thread the caller is on.
    sky_diag::with_deep_stack(|| verify_inner(m))
}

fn verify_inner(m: &Module) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    for (i, f) in m.functions.iter().enumerate() {
        if m.functions[..i].iter().any(|g| g.name == f.name) {
            push(
                &mut errors,
                VerifyError {
                    message: format!("function `{}` is defined twice", f.name),
                    function: f.name.clone(),
                    op: None,
                    value: None,
                    span: f.span,
                },
            );
        }
        FnVerifier::new(m, f, &mut errors).run();
    }
    if errors.is_empty() {
        check_effects(m, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Every function's declared effects include the inferred ones.
fn check_effects(m: &Module, errors: &mut Vec<VerifyError>) {
    let names = |e: EffectSet| {
        let list: Vec<&str> = e.iter().map(effect_name).collect();
        if list.is_empty() {
            "none".to_string()
        } else {
            list.join(", ")
        }
    };
    for (f, inferred) in m.functions.iter().zip(effects::infer(m)) {
        let missing = inferred.difference(f.effects);
        if !missing.is_pure() {
            push(
                errors,
                VerifyError {
                    message: format!(
                        "the declared effects ({}) miss the inferred ones: {}",
                        names(f.effects),
                        names(missing)
                    ),
                    function: f.name.clone(),
                    op: None,
                    value: None,
                    span: f.span,
                },
            );
        }
    }
}

fn push(errors: &mut Vec<VerifyError>, e: VerifyError) {
    if errors.len() < MAX_ERRORS {
        errors.push(e);
    }
}

/// What a region is for, which decides its terminator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Body,
    IfArm,
    /// The body of a `fused`: scalar operations ending in `yield`.
    FusedBody,
    WhileBefore,
    WhileAfter,
}

impl Ctx {
    fn terminator(self) -> &'static str {
        match self {
            Ctx::Body => "return",
            Ctx::IfArm | Ctx::FusedBody | Ctx::WhileAfter => "yield",
            Ctx::WhileBefore => "condition",
        }
    }
}

/// Where a value stands while the walk is inside the function.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Vis {
    /// Not defined yet at this point of the walk.
    Unseen,
    /// Defined and in scope.
    Visible,
    /// Defined in a block the walk has left.
    Ended,
}

/// An operand type, classified for the number rules.
enum Slot<'a> {
    Num(&'a NumberType),
    Unknown,
}

struct FnVerifier<'a> {
    m: &'a Module,
    f: &'a Function,
    errors: &'a mut Vec<VerifyError>,
    vis: Vec<Vis>,
    op_seen: Vec<bool>,
    block_seen: Vec<bool>,
    region_seen: Vec<bool>,
}

impl<'a> FnVerifier<'a> {
    fn new(m: &'a Module, f: &'a Function, errors: &'a mut Vec<VerifyError>) -> Self {
        FnVerifier {
            m,
            f,
            errors,
            vis: vec![Vis::Unseen; f.values.len()],
            op_seen: vec![false; f.ops.len()],
            block_seen: vec![false; f.blocks.len()],
            region_seen: vec![false; f.regions.len()],
        }
    }

    fn report(&mut self, message: String, op: Option<OpId>, value: Option<ValueId>, span: Span) {
        push(
            self.errors,
            VerifyError {
                message,
                function: self.f.name.clone(),
                op,
                value,
                span,
            },
        );
    }

    fn err_fn(&mut self, message: String) {
        self.report(message, None, None, self.f.span);
    }

    fn err_op(&mut self, o: OpId, message: String) {
        let span = self.f.ops.get(o.index()).map_or(self.f.span, |d| d.span);
        self.report(message, Some(o), None, span);
    }

    fn run(&mut self) {
        let f = self.f;
        for (i, t) in f.params.iter().chain(&f.results).enumerate() {
            self.check_type(t, None, None, &format!("signature type {i}"));
        }
        if f.body.index() >= f.regions.len() {
            self.err_fn(format!("the body region {} is out of range", f.body));
            return;
        }
        if f.regions[f.body.index()].parent.is_some() {
            self.err_fn("the body region must not have a parent operation".into());
        }
        self.walk_region(f.body, Ctx::Body, None, 0);
        self.check_body_relations();
    }

    // ----- structure -----

    /// The span of the operation owning a region (the function for its body).
    fn owner_span(&self, owner: Option<OpId>) -> Span {
        owner
            .and_then(|o| self.f.ops.get(o.index()))
            .map_or(self.f.span, |d| d.span)
    }

    fn walk_region(&mut self, r: RegionId, ctx: Ctx, owner: Option<OpId>, depth: usize) {
        let f = self.f;
        let span = self.owner_span(owner);
        if depth >= MAX_REGION_DEPTH {
            self.report(
                format!("regions are nested more than {MAX_REGION_DEPTH} levels deep"),
                owner,
                None,
                span,
            );
            return;
        }
        let Some(region) = f.regions.get(r.index()) else {
            self.report(format!("{r} is out of range"), owner, None, span);
            return;
        };
        if std::mem::replace(&mut self.region_seen[r.index()], true) {
            self.report(
                format!("{r} appears in the tree more than once"),
                owner,
                None,
                span,
            );
            return;
        }
        if region.parent != owner {
            self.report(
                format!(
                    "{r} says its parent is {:?} but it is owned by {:?}",
                    region.parent, owner
                ),
                owner,
                None,
                span,
            );
        }
        if region.blocks.len() != 1 {
            self.report(
                format!(
                    "a region has exactly one block in v1, found {}",
                    region.blocks.len()
                ),
                owner,
                None,
                span,
            );
        }
        for &b in &region.blocks {
            self.walk_block(b, r, ctx, owner, depth);
        }
    }

    fn walk_block(&mut self, b: BlockId, r: RegionId, ctx: Ctx, owner: Option<OpId>, depth: usize) {
        let f = self.f;
        let span = self.owner_span(owner);
        let Some(block) = f.blocks.get(b.index()) else {
            self.report(format!("{b} is out of range"), owner, None, span);
            return;
        };
        if std::mem::replace(&mut self.block_seen[b.index()], true) {
            self.report(
                format!("{b} appears in the tree more than once"),
                owner,
                None,
                span,
            );
            return;
        }
        if block.parent != Some(r) {
            self.report(
                format!("{b} says its parent is {:?} but it is in {r}", block.parent),
                owner,
                None,
                span,
            );
        }
        let mut defined: Vec<ValueId> = Vec::new();
        for (i, &v) in block.args.iter().enumerate() {
            let want = ValueDef::BlockArg {
                block: b,
                index: i as u32,
            };
            if self.define(v, want, owner, span) {
                defined.push(v);
            }
        }
        if block.ops.is_empty() {
            self.report(
                format!("{b} is empty: it must end in `{}`", ctx.terminator()),
                owner,
                None,
                span,
            );
        }
        let last = block.ops.len().wrapping_sub(1);
        for (idx, &o) in block.ops.iter().enumerate() {
            let Some(op) = f.ops.get(o.index()) else {
                self.err_op(o, format!("{o} is out of range"));
                continue;
            };
            if std::mem::replace(&mut self.op_seen[o.index()], true) {
                self.err_op(o, format!("{o} appears in the tree more than once"));
                continue;
            }
            if op.parent != b {
                self.err_op(
                    o,
                    format!("{o} says its parent is {} but it is in {b}", op.parent),
                );
            }
            let terminator = op.kind.properties().terminator;
            if terminator && idx != last {
                self.err_op(
                    o,
                    format!(
                        "`{}` is a terminator but is not the last operation of {b}",
                        op.kind.mnemonic()
                    ),
                );
            } else if idx == last && !terminator {
                self.err_op(
                    o,
                    format!(
                        "{b} does not end with a terminator: found `{}`, expected `{}`",
                        op.kind.mnemonic(),
                        ctx.terminator()
                    ),
                );
            } else if terminator && op.kind.mnemonic() != ctx.terminator() {
                self.err_op(
                    o,
                    format!(
                        "`{}` is not allowed here: this region must end in `{}`",
                        op.kind.mnemonic(),
                        ctx.terminator()
                    ),
                );
            }
            self.visit_op(o, op, depth);
            defined.extend(op.results.iter().copied());
        }
        for v in defined {
            if let Some(vis) = self.vis.get_mut(v.index()) {
                if *vis == Vis::Visible {
                    *vis = Vis::Ended;
                }
            }
        }
    }

    /// Checks that `v` is in range and defined at `want`, and makes it
    /// visible. False if it isn't usable.
    fn define(&mut self, v: ValueId, want: ValueDef, owner: Option<OpId>, span: Span) -> bool {
        let Some(data) = self.f.values.get(v.index()) else {
            self.report(format!("{v} is out of range"), owner, Some(v), span);
            return false;
        };
        if data.def != want {
            self.report(
                format!("{v} appears as {want:?} but its `def` says {:?}", data.def),
                owner,
                Some(v),
                span,
            );
            return false;
        }
        self.check_type(&data.ty, owner, Some(v), &format!("the type of {v}"));
        self.vis[v.index()] = Vis::Visible;
        true
    }

    // ----- operations -----

    fn visit_op(&mut self, o: OpId, op: &'a OpData, depth: usize) {
        let f = self.f;
        let mut operands_ok = true;
        for (i, &v) in op.operands.iter().enumerate() {
            if v.index() >= f.values.len() {
                self.err_op(o, format!("operand {i} ({v}) is out of range"));
                operands_ok = false;
                continue;
            }
            match self.vis[v.index()] {
                Vis::Visible => {}
                Vis::Unseen => {
                    self.report(
                        format!("operand {i}, {v}, is used before its definition"),
                        Some(o),
                        Some(v),
                        op.span,
                    );
                    operands_ok = false;
                }
                Vis::Ended => {
                    self.report(
                        format!(
                            "operand {i}, {v}, is not visible here: it is defined in a region that has ended"
                        ),
                        Some(o),
                        Some(v),
                        op.span,
                    );
                    operands_ok = false;
                }
            }
        }
        if !op.operand_spans.is_empty() && op.operand_spans.len() != op.expected_operand_spans() {
            self.err_op(
                o,
                format!(
                    "`{}` has {} operand span(s), but {} operand(s){}: they must be one per operand (or none)",
                    op.kind.mnemonic(),
                    op.operand_spans.len(),
                    op.operands.len(),
                    match &op.kind {
                        OpKind::Slice { axes } => format!(" and {} axis item(s)", axes.len()),
                        _ => String::new(),
                    }
                ),
            );
        }
        // Results are defined after the operation's regions have been walked
        // (they are not visible inside them), but their `def` links and
        // types are checked now.
        let mut results_ok = true;
        for (i, &v) in op.results.iter().enumerate() {
            let want = ValueDef::OpResult {
                op: o,
                index: i as u32,
            };
            let Some(data) = f.values.get(v.index()) else {
                self.err_op(o, format!("result {i} ({v}) is out of range"));
                results_ok = false;
                continue;
            };
            if data.def != want {
                self.report(
                    format!("{v} is result {i} of {o} but its `def` says {:?}", data.def),
                    Some(o),
                    Some(v),
                    op.span,
                );
                results_ok = false;
                continue;
            }
            self.check_type(&data.ty, Some(o), Some(v), &format!("the type of {v}"));
        }

        let region_ctx: &[Ctx] = match op.kind {
            OpKind::If => &[Ctx::IfArm, Ctx::IfArm],
            OpKind::While => &[Ctx::WhileBefore, Ctx::WhileAfter],
            OpKind::Fused => &[Ctx::FusedBody],
            _ => &[],
        };
        if op.regions.len() != region_ctx.len() {
            self.err_op(
                o,
                format!(
                    "`{}` takes {} region(s), found {}",
                    op.kind.mnemonic(),
                    region_ctx.len(),
                    op.regions.len()
                ),
            );
        }
        for (i, &r) in op.regions.iter().enumerate() {
            // Extra regions on the wrong kind of operation are walked as if
            // they were plain arms, so their contents are still checked.
            let ctx = region_ctx.get(i).copied().unwrap_or(Ctx::IfArm);
            self.walk_region(r, ctx, Some(o), depth + 1);
        }

        if operands_ok && results_ok {
            self.check_op(o, op);
        }
        for &v in &op.results {
            if let Some(vis) = self.vis.get_mut(v.index()) {
                *vis = Vis::Visible;
            }
        }
    }

    fn check_type(&mut self, t: &Type, op: Option<OpId>, value: Option<ValueId>, what: &str) {
        let Type::Number(n) = t else { return };
        let span = self.owner_span(op);
        if !n.tags.is_closed() {
            self.report(
                format!(
                    "{what} has tags {} that are not closed (closure is {})",
                    n.tags,
                    n.tags.closure()
                ),
                op,
                value,
                span,
            );
        }
        if !n.tags.is_empty() {
            let square = matches!(n.shape.dims(), [a, b] if a == b);
            if !square {
                self.report(
                    format!(
                        "{what} has tags {} but its shape {} is not a square matrix",
                        n.tags, n.shape
                    ),
                    op,
                    value,
                    span,
                );
            }
        }
    }

    /// Operand and result counts, then the kind's type rules.
    fn check_op(&mut self, o: OpId, op: &'a OpData) {
        let name = op.kind.mnemonic();
        let (operands, results): (Option<usize>, Option<usize>) = match &op.kind {
            OpKind::ConstFloat { .. }
            | OpKind::ConstRational(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstStr(_)
            | OpKind::ConstNothing
            | OpKind::LiteralError { .. }
            | OpKind::Input { .. } => (Some(0), Some(1)),
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Pow
            | OpKind::MatMul
            | OpKind::Dot
            | OpKind::MatMulEx { .. }
            | OpKind::DiagMul { .. }
            | OpKind::Solve { .. }
            | OpKind::SolveRight { .. }
            | OpKind::Cmp(_) => (Some(2), Some(1)),
            OpKind::Neg
            | OpKind::PowConst { .. }
            | OpKind::Inverse
            | OpKind::Transpose
            | OpKind::Norm
            | OpKind::Show { .. }
            | OpKind::Identity
            | OpKind::Len
            | OpKind::ExpectBool { .. }
            | OpKind::ExpectNumber { .. }
            | OpKind::Assume
            | OpKind::Not => (Some(1), Some(1)),
            OpKind::ArrayLiteral { .. } | OpKind::Index | OpKind::Slice { .. } | OpKind::Zeros => {
                (None, Some(1))
            }
            OpKind::If => (Some(1), None),
            OpKind::Fused => (None, Some(1)),
            OpKind::While => (None, Some(op.operands.len())),
            OpKind::Yield | OpKind::Return | OpKind::Print | OpKind::Assert { .. } => {
                (None, Some(0))
            }
            OpKind::Condition => (None, Some(0)),
            OpKind::Call { .. } => (None, None),
            OpKind::Math(f) => (Some(f.arity()), Some(1)),
        };
        if let Some(n) = operands {
            if op.operands.len() != n {
                self.err_op(
                    o,
                    format!("`{name}` takes {n} operand(s), found {}", op.operands.len()),
                );
                return;
            }
        }
        if let Some(n) = results {
            if op.results.len() != n {
                self.err_op(
                    o,
                    format!("`{name}` has {n} result(s), found {}", op.results.len()),
                );
                return;
            }
        }
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                if !f64::from_bits(*bits).is_finite() {
                    self.err_op(
                        o,
                        format!("`const.float` must be finite, found bits {bits:#018x}"),
                    );
                }
                self.check_const_number(o, op);
            }
            OpKind::ConstRational(_) => self.check_const_number(o, op),
            OpKind::ConstBool(_) => self.expect_exact(o, op, &Type::Bool),
            OpKind::ConstStr(_) => self.expect_exact(o, op, &Type::Str),
            OpKind::ConstNothing => self.expect_exact(o, op, &Type::Nothing),
            // Any type: the one the program declared for the name, or the
            // literal's.
            OpKind::Input { .. } | OpKind::LiteralError { .. } => {}
            OpKind::Add | OpKind::Sub => self.check_additive(o, op),
            OpKind::Mul | OpKind::Div => self.check_multiplicative(o, op),
            OpKind::Neg => {
                if let Some(Slot::Num(a)) = self.operand_num(o, op, 0) {
                    self.expect_result(o, op, Some(a.dim), Some(a.shape.clone()));
                } else {
                    self.expect_result_loose(o, op);
                }
            }
            OpKind::PowConst { exponent } => {
                if let Some(Slot::Num(a)) = self.operand_num(o, op, 0) {
                    match a.dim.pow_rat(*exponent) {
                        Ok(d) => self.expect_result(o, op, Some(d), Some(a.shape.clone())),
                        Err(e) => self.err_op(o, format!("`pow.const` {e}")),
                    }
                } else {
                    self.expect_result_loose(o, op);
                }
            }
            OpKind::Math(f) => self.check_math(o, op, *f),
            OpKind::Show { unit } => self.check_show(o, op, unit),
            OpKind::Pow => self.check_pow(o, op),
            OpKind::Inverse => self.check_inverse(o, op),
            OpKind::MatMul => self.check_matmul(o, op),
            OpKind::Transpose => self.check_transpose(o, op),
            OpKind::Norm => self.check_norm(o, op),
            OpKind::Dot => self.check_dot(o, op),
            OpKind::Solve { method } => self.check_solve(o, op, *method, false),
            OpKind::SolveRight { method } => self.check_solve(o, op, *method, true),
            OpKind::MatMulEx { ta, tb, algorithm } => {
                self.check_matmul_ex(o, op, *ta, *tb, *algorithm)
            }
            OpKind::DiagMul { side } => self.check_diag_mul(o, op, *side),
            OpKind::ArrayLiteral { cols } => self.check_array_literal(o, op, *cols),
            OpKind::Index => self.check_index(o, op),
            OpKind::Slice { axes } => self.check_slice(o, op, axes),
            OpKind::Zeros | OpKind::Identity => self.check_zeros_or_identity(o, op),
            OpKind::Len => self.check_len(o, op),
            OpKind::Cmp(c) => self.check_cmp(o, op, *c),
            OpKind::Not => {
                let t = self.f.ty(op.operands[0]);
                if !matches!(t, Type::Bool | Type::Unknown) {
                    self.err_op(o, format!("`not` needs a Bool operand, found {t}"));
                }
                self.expect_bool_result(o, op);
            }
            OpKind::ExpectBool { context } => {
                // Only an operand the checker could not type needs the
                // check; a typed one that is not a Bool can never pass it.
                let t = self.f.ty(op.operands[0]);
                if !matches!(t, Type::Bool | Type::Unknown) {
                    self.err_op(
                        o,
                        format!(
                            "`expect.bool` needs a Bool or unknown operand, found {t} (`{}`)",
                            context.keyword()
                        ),
                    );
                }
                self.expect_bool_result(o, op);
            }
            OpKind::Assume => {
                // Only the tags may differ: the value is passed on as it is.
                let (t, r) = (self.f.ty(op.operands[0]), self.f.ty(op.results[0]));
                let same = match (t, r) {
                    (Type::Number(a), Type::Number(b)) => {
                        a.dim == b.dim && a.shape == b.shape && a.repr == b.repr
                    }
                    _ => false,
                };
                if !same {
                    self.err_op(
                        o,
                        format!(
                            "`assume` needs a number and gives it back with other tags, found {t} and {r}"
                        ),
                    );
                }
            }
            OpKind::ExpectNumber { context } => {
                // Only an operand the checker could not type needs the
                // check; a typed one that is not a number can never pass it.
                let t = self.f.ty(op.operands[0]);
                if !matches!(t, Type::Number(_) | Type::Unknown) {
                    self.err_op(
                        o,
                        format!(
                            "`expect.number` needs a number or unknown operand, found {t} (`{}`)",
                            context.keyword()
                        ),
                    );
                } else if !matches!(self.f.ty(op.results[0]), Type::Unknown)
                    && self.f.ty(op.results[0]) != t
                {
                    // Unknown is always fine: a pass that learns the
                    // operand's type (folding an `if` away) leaves the
                    // result as it was.
                    self.err_op(
                        o,
                        format!(
                            "the result of `expect.number` must be unknown or {t} like its operand, found {}",
                            self.f.ty(op.results[0])
                        ),
                    );
                }
            }
            OpKind::Assert {
                kind,
                text,
                func,
                names,
                ..
            } => {
                if op.operands.len() != 1 + names.len() {
                    self.err_op(
                        o,
                        format!(
                            "`assert` takes a condition and {} value(s) for its names, found {} operand(s)",
                            names.len(),
                            op.operands.len()
                        ),
                    );
                } else {
                    let t = self.f.ty(op.operands[0]);
                    if !matches!(t, Type::Bool | Type::Unknown) {
                        self.err_op(
                            o,
                            format!("`assert` needs a Bool or unknown condition, found {t}"),
                        );
                    }
                }
                if text.is_empty() {
                    self.err_op(o, "`assert` needs the text of its condition".to_string());
                }
                if func.is_some() != (*kind != AssertKind::Assert) {
                    self.err_op(
                        o,
                        format!(
                            "`assert[{}]` {} the function it belongs to",
                            kind.keyword(),
                            if *kind == AssertKind::Assert {
                                "takes no name of"
                            } else {
                                "needs the name of"
                            }
                        ),
                    );
                }
            }
            OpKind::If => self.check_if(o, op),
            OpKind::While => self.check_while(o, op),
            OpKind::Fused => self.check_fused(o, op),
            OpKind::Yield | OpKind::Condition | OpKind::Return => {
                // Their operand rules are checked by the owner (the `if`,
                // the `while`, or the function) once its regions are walked.
            }
            OpKind::Call { callee } => self.check_call(o, op, callee),
            OpKind::Print => {}
        }
    }

    // ----- helpers for the per-kind rules -----

    /// Operand `i` as a number, `Unknown`, or (with an error) `None`.
    fn operand_num(&mut self, o: OpId, op: &OpData, i: usize) -> Option<Slot<'a>> {
        let f = self.f;
        match f.ty(op.operands[i]) {
            Type::Number(n) => Some(Slot::Num(n)),
            Type::Unknown => Some(Slot::Unknown),
            other => {
                self.err_op(
                    o,
                    format!(
                        "operand {i} of `{}` must be a number, found {other}",
                        op.kind.mnemonic()
                    ),
                );
                None
            }
        }
    }

    /// Both operands as numbers; `None` if either is unknown or not a number
    /// (a wrong kind of operand is reported here).
    fn two_nums(&mut self, o: OpId, op: &OpData) -> Option<(&'a NumberType, &'a NumberType)> {
        let a = self.operand_num(o, op, 0);
        let b = self.operand_num(o, op, 1);
        match (a, b) {
            (Some(Slot::Num(a)), Some(Slot::Num(b))) => Some((a, b)),
            _ => {
                self.expect_result_loose(o, op);
                None
            }
        }
    }

    /// Result 0 must be a number or Unknown.
    /// `show "unit"`: a number operand whose dimension is the unit's (the
    /// unit text is read by the parser's own unit grammar and priced by the
    /// unit table, as `sky_interp::show_unit_text` does), and a `Str` result.
    fn check_show(&mut self, o: OpId, op: &OpData, unit: &str) {
        let Some(u) = sky_ast::parse_unit_text(unit) else {
            self.err_op(o, format!("`show` needs a unit, found `{unit}`"));
            return;
        };
        if let Some(Slot::Num(n)) = self.operand_num(o, op, 0) {
            match sky_types::apply_unit(&u.src, &u.unit, 1.0) {
                Ok((_, dim)) if dim == n.dim => {}
                Ok((_, dim)) => self.err_op(
                    o,
                    format!(
                        "`show` of `{unit}` ({dim}) needs an operand of that dimension, found {}",
                        n.dim
                    ),
                ),
                Err(_) => self.err_op(o, format!("`show` needs a unit, found `{unit}`")),
            }
        }
        self.expect_exact(o, op, &Type::Str);
    }

    /// `math.*`: scalar operands (the checker rejects arrays), and the result
    /// dimension from the one rule table, `MathFn::result_dim`.
    fn check_math(&mut self, o: OpId, op: &OpData, f: crate::op::MathFn) {
        let mut dims = Vec::with_capacity(op.operands.len());
        for i in 0..op.operands.len() {
            match self.operand_num(o, op, i) {
                Some(Slot::Num(n)) if n.shape.rank() == 0 => dims.push(n.dim),
                Some(Slot::Num(n)) => {
                    let (name, shape) = (f.name(), n.shape.clone());
                    self.err_op(
                        o,
                        format!("`{name}` takes scalars, operand {i} has shape {shape:?}"),
                    );
                    return;
                }
                // Unknown operand, or already reported.
                _ => return self.expect_result_loose(o, op),
            }
        }
        match f.result_dim(&dims) {
            Ok(d) => self.expect_result(o, op, Some(d), Some(Shape::scalar())),
            Err(e) => self.err_op(
                o,
                format!("`{}`: {e:?} for operand dimensions {dims:?}", f.name()),
            ),
        }
    }

    fn expect_result_loose(&mut self, o: OpId, op: &OpData) {
        self.expect_result(o, op, None, None);
    }

    /// Result 0 must be Unknown, or a number with the given dimension and
    /// shape (each only if given).
    fn expect_result(
        &mut self,
        o: OpId,
        op: &OpData,
        dim: Option<Dimension>,
        shape: Option<Shape>,
    ) {
        let name = op.kind.mnemonic();
        match self.f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(n) => {
                if let Some(d) = dim {
                    if n.dim != d {
                        self.err_op(
                            o,
                            format!(
                                "the result of `{name}` should have dimension {d}, found {}",
                                n.dim
                            ),
                        );
                    }
                }
                if let Some(s) = shape {
                    if n.shape != s {
                        self.err_op(
                            o,
                            format!(
                                "the result of `{name}` should have shape {s}, found {}",
                                n.shape
                            ),
                        );
                    }
                }
            }
            other => self.err_op(
                o,
                format!("the result of `{name}` must be a number, found {other}"),
            ),
        }
    }

    fn expect_bool_result(&mut self, o: OpId, op: &OpData) {
        let t = self.f.ty(op.results[0]);
        if !matches!(t, Type::Bool | Type::Unknown) {
            self.err_op(
                o,
                format!(
                    "the result of `{}` must be Bool, found {t}",
                    op.kind.mnemonic()
                ),
            );
        }
    }

    /// Constants are typed exactly: a scalar of any dimension.
    fn check_const_number(&mut self, o: OpId, op: &OpData) {
        match self.f.ty(op.results[0]) {
            Type::Number(n) if n.shape.rank() == 0 => {}
            other => self.err_op(
                o,
                format!(
                    "the result of `{}` must be a scalar number, found {other}",
                    op.kind.mnemonic()
                ),
            ),
        }
    }

    fn expect_exact(&mut self, o: OpId, op: &OpData, want: &Type) {
        let got = self.f.ty(op.results[0]);
        if got != want {
            self.err_op(
                o,
                format!(
                    "the result of `{}` must be {want}, found {got}",
                    op.kind.mnemonic()
                ),
            );
        }
    }

    /// Runs a `sky_types::rules` function on a scratch unifier. The types in
    /// SIR are already resolved, so a rule that only succeeds by *binding* a
    /// shape variable (`N` vs `3`) means the shapes differ, and is an error.
    fn with_rules<T>(
        &mut self,
        o: OpId,
        run: impl FnOnce(&mut ShapeUnifier) -> Result<T, sky_types::shape::ShapeError>,
    ) -> Option<T> {
        let mut u = ShapeUnifier::new();
        let before = u.clone();
        match run(&mut u) {
            Err(e) => {
                self.err_op(o, e.to_string());
                None
            }
            Ok(_) if u != before => {
                self.err_op(
                    o,
                    "shapes only agree if a shape variable is bound to another size".into(),
                );
                None
            }
            Ok(v) => Some(v),
        }
    }

    fn dim_result(
        &mut self,
        o: OpId,
        r: Result<Dimension, sky_units::DimError>,
    ) -> Option<Dimension> {
        match r {
            Ok(d) => Some(d),
            Err(e) => {
                self.err_op(o, e.to_string());
                None
            }
        }
    }

    // ----- per-kind rules -----

    fn check_additive(&mut self, o: OpId, op: &OpData) {
        let Some((a, b)) = self.two_nums(o, op) else {
            return;
        };
        let name = op.kind.mnemonic();
        let mut ok = true;
        if a.dim != b.dim {
            self.err_op(
                o,
                format!(
                    "operands of `{name}` have different dimensions: {} and {}",
                    a.dim, b.dim
                ),
            );
            ok = false;
        }
        let (ta, tb) = (
            TensorType::untagged(a.shape.clone()),
            TensorType::untagged(b.shape.clone()),
        );
        if self
            .with_rules(o, |u| rules::elementwise(u, &ta, &tb))
            .is_none()
        {
            ok = false;
        }
        if ok {
            self.expect_result(o, op, Some(a.dim), Some(a.shape.clone()));
        }
    }

    fn check_multiplicative(&mut self, o: OpId, op: &OpData) {
        let Some((a, b)) = self.two_nums(o, op) else {
            return;
        };
        let mul = matches!(op.kind, OpKind::Mul);
        let name = op.kind.mnemonic();
        let dim = if mul {
            a.dim.checked_mul(b.dim)
        } else {
            a.dim.checked_div(b.dim)
        };
        let dim = self.dim_result(o, dim);
        let shape = match (a.shape.rank() == 0, b.shape.rank() == 0) {
            (true, true) => Some(Shape::scalar()),
            (true, false) if mul => Some(b.shape.clone()),
            (false, true) => Some(a.shape.clone()),
            _ => {
                self.err_op(
                    o,
                    format!(
                        "`{name}` needs a scalar {}, found {} and {}",
                        if mul { "operand" } else { "divisor" },
                        a.shape,
                        b.shape
                    ),
                );
                None
            }
        };
        if let (Some(d), Some(s)) = (dim, shape) {
            self.expect_result(o, op, Some(d), Some(s));
        }
    }

    fn check_pow(&mut self, o: OpId, op: &OpData) {
        let Some((a, e)) = self.two_nums(o, op) else {
            return;
        };
        let mut ok = true;
        for (i, n) in [a, e].into_iter().enumerate() {
            if !n.dim.is_dimensionless() || n.shape.rank() != 0 {
                self.err_op(
                    o,
                    format!(
                        "operand {i} of `pow` must be a dimensionless scalar, found {} with dimension {}",
                        n.shape, n.dim
                    ),
                );
                ok = false;
            }
        }
        if ok {
            self.expect_result(o, op, Some(Dimension::DIMENSIONLESS), Some(Shape::scalar()));
        }
    }

    fn check_inverse(&mut self, o: OpId, op: &OpData) {
        let Some(Slot::Num(a)) = self.operand_num(o, op, 0) else {
            self.expect_result_loose(o, op);
            return;
        };
        let t = TensorType::untagged(a.shape.clone());
        let shape = self.with_rules(o, |u| rules::inverse(u, &t));
        let dim = self.dim_result(o, a.dim.inverse());
        if let (Some(t), Some(d)) = (shape, dim) {
            self.expect_result(o, op, Some(d), Some(t.shape().clone()));
        }
    }

    fn check_matmul(&mut self, o: OpId, op: &OpData) {
        let Some((a, b)) = self.two_nums(o, op) else {
            return;
        };
        let (ta, tb) = (
            TensorType::untagged(a.shape.clone()),
            TensorType::untagged(b.shape.clone()),
        );
        let shape = self.with_rules(o, |u| rules::matmul(u, &ta, &tb));
        let dim = self.dim_result(o, a.dim.checked_mul(b.dim));
        if let (Some(t), Some(d)) = (shape, dim) {
            self.expect_result(o, op, Some(d), Some(t.shape().clone()));
        }
    }

    fn check_transpose(&mut self, o: OpId, op: &OpData) {
        let Some(Slot::Num(a)) = self.operand_num(o, op, 0) else {
            self.expect_result_loose(o, op);
            return;
        };
        let t = TensorType::untagged(a.shape.clone());
        // `transpose` never touches a unifier, but goes through the same
        // wrapper for its error text.
        if let Some(t) = self.with_rules(o, |_| rules::transpose(&t)) {
            self.expect_result(o, op, Some(a.dim), Some(t.shape().clone()));
        }
    }

    /// `norm(x)`: any number; the result is a scalar of the operand's
    /// dimension.
    fn check_norm(&mut self, o: OpId, op: &OpData) {
        match self.operand_num(o, op, 0) {
            Some(Slot::Num(a)) => self.expect_result(o, op, Some(a.dim), Some(Shape::scalar())),
            _ => self.expect_result_loose(o, op),
        }
    }

    /// `dot(a, b)`: two vectors of one length; the result is a scalar in the
    /// product of the dimensions.
    fn check_dot(&mut self, o: OpId, op: &OpData) {
        let Some((a, b)) = self.two_nums(o, op) else {
            self.expect_result_loose(o, op);
            return;
        };
        let mut ok = true;
        for (i, n) in [a, b].into_iter().enumerate() {
            if n.shape.rank() != 1 {
                self.err_op(
                    o,
                    format!("operand {i} of `dot` must be a vector, found {}", n.shape),
                );
                ok = false;
            }
        }
        if ok {
            let (ta, tb) = (
                TensorType::untagged(a.shape.clone()),
                TensorType::untagged(b.shape.clone()),
            );
            if self
                .with_rules(o, |u| rules::elementwise(u, &ta, &tb))
                .is_none()
            {
                ok = false;
            }
        }
        let dim = self.dim_result(o, a.dim.checked_mul(b.dim));
        if let (true, Some(d)) = (ok, dim) {
            self.expect_result(o, op, Some(d), Some(Shape::scalar()));
        }
    }

    /// `solve` (`A⁻¹ @ b`, operands `A`, `b`) and `solve.right` (`x @ A⁻¹`,
    /// operands `x`, `A`).
    fn check_solve(&mut self, o: OpId, op: &OpData, method: SolveMethod, right: bool) {
        let Some((first, second)) = self.two_nums(o, op) else {
            return;
        };
        let (a, b) = if right {
            (second, first)
        } else {
            (first, second)
        };
        let name = op.kind.mnemonic();
        let needs = match method {
            SolveMethod::Cholesky => Some(Tag::PositiveDefinite),
            SolveMethod::Diagonal => Some(Tag::Diagonal),
            SolveMethod::Lu => None,
        };
        if let Some(tag) = needs {
            if !a.tags.closure().contains(tag) {
                self.err_op(
                    o,
                    format!(
                        "`{name}[{}]` needs a {tag} matrix, but A has tags {}",
                        method.mnemonic(),
                        a.tags
                    ),
                );
            }
        }
        // `solve(A, b)` is `A⁻¹ @ b` and `solve.right(x, A)` is `x @ A⁻¹`:
        // reuse both rules, on one unifier so a variable bound by the first
        // is seen by the second.
        let (ta, tb) = (
            TensorType::untagged(a.shape.clone()),
            TensorType::untagged(b.shape.clone()),
        );
        let shape = self.with_rules(o, |u| {
            let inv = rules::inverse(u, &ta)?;
            if right {
                rules::matmul(u, &tb, &inv)
            } else {
                rules::matmul(u, &inv, &tb)
            }
        });
        let dim = a.dim.inverse().and_then(|inv| inv.checked_mul(b.dim));
        let dim = self.dim_result(o, dim);
        if let (Some(t), Some(d)) = (shape, dim) {
            self.expect_result(o, op, Some(d), Some(t.shape().clone()));
        }
    }

    /// `matmul[ta,tb,alg=...]`: the product of the operands after the
    /// flagged ones are transposed, of the algorithm class's shapes.
    fn check_matmul_ex(&mut self, o: OpId, op: &OpData, ta: bool, tb: bool, alg: Algorithm) {
        let Some((a, b)) = self.two_nums(o, op) else {
            return;
        };
        if !ta && !tb && alg == Algorithm::Auto {
            self.err_op(
                o,
                "`matmul` without attributes is written plain `matmul`, not `matmul[]`".to_string(),
            );
        }
        let mut ok = true;
        let mut eff = Vec::new();
        for (n, flag, side) in [(a, ta, "left"), (b, tb, "right")] {
            let t = TensorType::untagged(n.shape.clone());
            if !flag {
                eff.push(t);
                continue;
            }
            match rules::transpose(&t) {
                Ok(t) if n.shape.rank() == 2 => eff.push(t),
                _ => {
                    self.err_op(
                        o,
                        format!(
                            "`matmul` transposes its {side} operand, which is {}: only a matrix has a transpose",
                            n.shape
                        ),
                    );
                    ok = false;
                    eff.push(t);
                }
            }
        }
        let (ra, rb) = (eff[0].shape().rank(), eff[1].shape().rank());
        let fits = match alg {
            Algorithm::Auto => true,
            Algorithm::Dot => (ra, rb) == (1, 1),
            Algorithm::MatVec => (ra, rb) == (2, 1),
            Algorithm::VecMat => (ra, rb) == (1, 2),
            Algorithm::Gemm | Algorithm::Syrk => (ra, rb) == (2, 2),
        };
        if !fits {
            self.err_op(
                o,
                format!(
                    "the algorithm `{}` does not fit operands of rank {ra} and {rb}",
                    alg.mnemonic()
                ),
            );
        }
        if alg == Algorithm::Syrk && (op.operands[0] != op.operands[1] || ta == tb) {
            self.err_op(
                o,
                "the algorithm `syrk` needs one value multiplied by its own transpose (`Aᵀ @ A` or `A @ Aᵀ`)"
                    .to_string(),
            );
        }
        if !ok {
            return;
        }
        let shape = self.with_rules(o, |u| rules::matmul(u, &eff[0], &eff[1]));
        let dim = self.dim_result(o, a.dim.checked_mul(b.dim));
        if let (Some(t), Some(d)) = (shape, dim) {
            self.expect_result(o, op, Some(d), Some(t.shape().clone()));
        }
    }

    /// `diag.mul[side]`: a matmul whose `side` operand is a Diagonal matrix.
    fn check_diag_mul(&mut self, o: OpId, op: &OpData, side: Side) {
        let Some((a, b)) = self.two_nums(o, op) else {
            return;
        };
        let d = if side == Side::Left { a } else { b };
        if !d.tags.closure().contains(Tag::Diagonal) {
            self.err_op(
                o,
                format!(
                    "`diag.mul[{}]` needs a Diagonal {} operand, but it has tags {}",
                    side.text(),
                    side.text(),
                    d.tags
                ),
            );
        }
        let (ta, tb) = (
            TensorType::untagged(a.shape.clone()),
            TensorType::untagged(b.shape.clone()),
        );
        let shape = self.with_rules(o, |u| rules::matmul(u, &ta, &tb));
        let dim = self.dim_result(o, a.dim.checked_mul(b.dim));
        if let (Some(t), Some(dim)) = (shape, dim) {
            self.expect_result(o, op, Some(dim), Some(t.shape().clone()));
        }
    }

    fn check_cmp(&mut self, o: OpId, op: &OpData, c: CmpOp) {
        let (ta, tb) = (self.f.ty(op.operands[0]), self.f.ty(op.operands[1]));
        let name = op.kind.mnemonic();
        match (ta, tb) {
            (Type::Unknown, _) | (_, Type::Unknown) => {}
            (Type::Number(a), Type::Number(b)) => {
                if a.shape.rank() != 0 || b.shape.rank() != 0 {
                    self.err_op(
                        o,
                        format!(
                            "`{name}` compares scalars, found {} and {}",
                            a.shape, b.shape
                        ),
                    );
                }
                if a.dim != b.dim {
                    self.err_op(
                        o,
                        format!(
                            "operands of `{name}` have different dimensions: {} and {}",
                            a.dim, b.dim
                        ),
                    );
                }
            }
            (Type::Bool, Type::Bool) | (Type::Str, Type::Str)
                if matches!(c, CmpOp::Eq | CmpOp::Ne) => {}
            (a, b) => self.err_op(o, format!("`{name}` cannot compare {a} with {b}")),
        }
        self.expect_bool_result(o, op);
    }

    fn check_call(&mut self, o: OpId, op: &OpData, callee: &str) {
        let Some(g) = self.m.function(callee) else {
            self.err_op(o, format!("call to undefined function `{callee}`"));
            return;
        };
        // The callee's size variables are generic: each call instantiates
        // them (`N = 3`), consistently across its arguments, as the checker
        // does with fresh copies per call. The results use the same binding.
        let mut subst = Subst::new();
        if op.operands.len() != g.params.len() {
            self.err_op(
                o,
                format!(
                    "`{callee}` takes {} argument(s), the call passes {}",
                    g.params.len(),
                    op.operands.len()
                ),
            );
        } else {
            for (i, (&v, p)) in op.operands.iter().zip(&g.params).enumerate() {
                let t = self.f.ty(v);
                if !subst.bind(t, p) {
                    let p = subst.apply(p);
                    self.err_op(
                        o,
                        format!("argument {i} of the call to `{callee}` is {t}, expected {p}"),
                    );
                }
            }
        }
        if op.results.len() != g.results.len() {
            self.err_op(
                o,
                format!(
                    "`{callee}` returns {} value(s), the call has {} result(s)",
                    g.results.len(),
                    op.results.len()
                ),
            );
        } else {
            for (i, (&v, r)) in op.results.iter().zip(&g.results).enumerate() {
                let t = self.f.ty(v);
                let r = &subst.apply(r);
                if !assignable(r, t) {
                    self.err_op(
                        o,
                        format!("result {i} of the call to `{callee}` is {t}, but it returns {r}"),
                    );
                }
            }
        }
    }

    // ----- control flow -----

    /// The terminator of a region's first block, if it has one in range.
    fn terminator(&self, r: RegionId) -> Option<(OpId, &'a OpData)> {
        let f = self.f;
        let block = f
            .blocks
            .get(f.regions.get(r.index())?.blocks.first()?.index())?;
        let &o = block.ops.last()?;
        Some((o, f.ops.get(o.index())?))
    }

    fn first_block(&self, r: RegionId) -> Option<&'a crate::ir::BlockData> {
        let f = self.f;
        f.blocks
            .get(f.regions.get(r.index())?.blocks.first()?.index())
    }

    fn check_if(&mut self, o: OpId, op: &OpData) {
        let f = self.f;
        let c = f.ty(op.operands[0]);
        if !matches!(c, Type::Bool | Type::Unknown) {
            self.err_op(o, format!("the condition of `if` must be Bool, found {c}"));
        }
        for (arm, &r) in op.regions.iter().enumerate() {
            let name = ["then", "else"].get(arm).copied().unwrap_or("extra");
            if let Some(b) = self.first_block(r) {
                if !b.args.is_empty() {
                    self.err_op(
                        o,
                        format!(
                            "the `{name}` block of `if` takes no arguments, found {}",
                            b.args.len()
                        ),
                    );
                }
            }
            let Some((y, yop)) = self.terminator(r) else {
                continue;
            };
            if !matches!(yop.kind, OpKind::Yield) {
                continue;
            }
            self.check_values_assignable(
                y,
                &yop.operands,
                &op.results,
                &format!("`yield` of the `{name}` arm"),
                "the result of the `if`",
            );
        }
    }

    fn check_while(&mut self, o: OpId, op: &OpData) {
        let f = self.f;
        let n = op.operands.len();
        let [before, after] = op.regions[..] else {
            return;
        };
        // Region arguments against the carried values.
        if let Some(b) = self.first_block(before) {
            self.check_args(o, "before", &b.args, &op.operands);
        }
        if let Some(b) = self.first_block(after) {
            self.check_args(o, "after", &b.args, &op.results);
        }
        // `condition`: a Bool, then values forwarded to `after` and to the
        // loop's results.
        if let Some((c, cop)) = self.terminator(before) {
            if matches!(cop.kind, OpKind::Condition) {
                if cop.operands.len() != n + 1 {
                    self.err_op(
                        c,
                        format!(
                            "`condition` takes a Bool and {n} forwarded value(s), found {} operand(s)",
                            cop.operands.len()
                        ),
                    );
                } else {
                    let t = f.ty(cop.operands[0]);
                    if !matches!(t, Type::Bool | Type::Unknown) {
                        self.err_op(
                            c,
                            format!("the first operand of `condition` must be Bool, found {t}"),
                        );
                    }
                    let fwd = &cop.operands[1..];
                    self.check_values_assignable(
                        c,
                        fwd,
                        &op.results,
                        "a value forwarded by `condition`",
                        "the result of the `while`",
                    );
                }
            }
        }
        // `yield` of `after`: the next carried values.
        if let Some((y, yop)) = self.terminator(after) {
            if matches!(yop.kind, OpKind::Yield) {
                let carried: Vec<ValueId> = self
                    .first_block(before)
                    .map(|b| b.args.clone())
                    .unwrap_or_default();
                self.check_values_assignable(
                    y,
                    &yop.operands,
                    &carried,
                    "`yield` of the `after` region",
                    "the loop-carried value",
                );
            }
        }
    }

    /// A region's block arguments must be one per value in `expected`, and
    /// each expected value must be assignable to it.
    fn check_args(&mut self, o: OpId, which: &str, args: &[ValueId], expected: &[ValueId]) {
        let f = self.f;
        if args.len() != expected.len() {
            self.err_op(
                o,
                format!(
                    "the `{which}` block of `while` takes {} argument(s), one per carried value; found {}",
                    expected.len(),
                    args.len()
                ),
            );
            return;
        }
        for (i, (&a, &e)) in args.iter().zip(expected).enumerate() {
            let (Some(at), Some(et)) = (f.values.get(a.index()), f.values.get(e.index())) else {
                continue;
            };
            if !assignable(&et.ty, &at.ty) {
                self.err_op(
                    o,
                    format!(
                        "argument {i} of the `{which}` block of `while` is {}, but the carried value is {}",
                        at.ty, et.ty
                    ),
                );
            }
        }
    }

    /// `op`'s operands must match `targets` one to one (count, then
    /// assignability of each type). Reports on `at`.
    fn check_values_assignable(
        &mut self,
        at: OpId,
        operands: &[ValueId],
        targets: &[ValueId],
        from: &str,
        to: &str,
    ) {
        let f = self.f;
        if operands.len() != targets.len() {
            self.err_op(
                at,
                format!(
                    "{from} passes {} value(s) but there are {} {to}",
                    operands.len(),
                    targets.len()
                ),
            );
            return;
        }
        for (i, (&v, &t)) in operands.iter().zip(targets).enumerate() {
            let (Some(vd), Some(td)) = (f.values.get(v.index()), f.values.get(t.index())) else {
                continue;
            };
            if !assignable(&vd.ty, &td.ty) {
                self.err_op(
                    at,
                    format!("{from}: value {i} is {}, but {to} is {}", vd.ty, td.ty),
                );
            }
        }
    }

    /// The function-level rules that need the body walked first: entry
    /// arguments against the parameters, `return` against the results.
    fn check_body_relations(&mut self) {
        let f = self.f;
        if let Some(b) = self.first_block(f.body) {
            let args: Vec<&Type> = b
                .args
                .iter()
                .filter_map(|v| f.values.get(v.index()).map(|d| &d.ty))
                .collect();
            if args.len() != f.params.len() || args.iter().zip(&f.params).any(|(a, p)| *a != p) {
                self.err_fn(format!(
                    "the entry block's arguments do not match the parameters ({} parameter(s))",
                    f.params.len()
                ));
            }
        }
        let Some((r, rop)) = self.terminator(f.body) else {
            return;
        };
        if !matches!(rop.kind, OpKind::Return) {
            return;
        }
        if rop.operands.len() != f.results.len() {
            self.err_op(
                r,
                format!(
                    "`return` passes {} value(s), the function returns {}",
                    rop.operands.len(),
                    f.results.len()
                ),
            );
            return;
        }
        for (i, (&v, want)) in rop.operands.iter().zip(&f.results).enumerate() {
            let Some(vd) = f.values.get(v.index()) else {
                continue;
            };
            if !assignable(&vd.ty, want) {
                self.err_op(
                    r,
                    format!(
                        "`return` value {i} is {}, the function returns {want}",
                        vd.ty
                    ),
                );
            }
        }
    }
}

/// A per-call instantiation of a callee's size variables (by variable id).
#[derive(Default)]
struct Subst {
    map: BTreeMap<u64, ShapeDim>,
}

impl Subst {
    fn new() -> Self {
        Subst::default()
    }

    /// Whether an argument of type `arg` fits parameter type `param`,
    /// binding `param`'s size variables on first use and requiring later
    /// uses to agree. Otherwise the rules of [`assignable`].
    fn bind(&mut self, arg: &Type, param: &Type) -> bool {
        let (Type::Number(a), Type::Number(p)) = (arg, param) else {
            return assignable(arg, param);
        };
        if a.dim != p.dim || a.repr != p.repr || !a.tags.is_subtag_of(p.tags) {
            return false;
        }
        let (ad, pd) = (a.shape.dims(), p.shape.dims());
        if ad.len() != pd.len() {
            return false;
        }
        // Tentative: commit only if every axis agrees, so a failed argument
        // leaves no binding behind for the next one.
        let mut next = self.map.clone();
        for (x, want) in ad.iter().zip(pd) {
            match want {
                ShapeDim::Var(v) => match next.get(&v.id()) {
                    Some(bound) if bound != x => return false,
                    Some(_) => {}
                    None => {
                        next.insert(v.id(), x.clone());
                    }
                },
                known => {
                    if x != known {
                        return false;
                    }
                }
            }
        }
        self.map = next;
        true
    }

    /// `t` with the bound size variables replaced.
    fn apply(&self, t: &Type) -> Type {
        match t {
            Type::Number(n) => {
                let dims = n
                    .shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        ShapeDim::Var(v) => {
                            self.map.get(&v.id()).cloned().unwrap_or_else(|| d.clone())
                        }
                        known => known.clone(),
                    })
                    .collect();
                Type::Number(crate::ir::NumberType {
                    shape: Shape::new(dims),
                    ..n.clone()
                })
            }
            other => other.clone(),
        }
    }
}

/// Can a value of type `from` be used where `to` is expected? Unknown fits
/// anywhere. Numbers need equal dimension, shape and representation; the
/// value may guarantee *more* tags than the target asks for (a
/// PositiveDefinite matrix is a Symmetric one), never fewer.
pub fn assignable(from: &Type, to: &Type) -> bool {
    match (from, to) {
        (Type::Unknown, _) | (_, Type::Unknown) => true,
        (Type::Number(a), Type::Number(b)) => {
            a.dim == b.dim && a.shape == b.shape && a.repr == b.repr && a.tags.is_subtag_of(b.tags)
        }
        (a, b) => a == b,
    }
}
