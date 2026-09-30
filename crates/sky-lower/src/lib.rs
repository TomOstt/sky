//! Lowering from the checked Sky AST to SIR (`docs/design/ir.md`).
//!
//! [`lower`] takes a module the checker accepted, and the checker's results,
//! and builds a [`sky_ir::Module`]. It **uses the checker's types and never
//! re-derives them**: the type of every operation's result is the type the
//! checker recorded for the expression (`Checked::expr_types`), with solved
//! shape variables applied. The only types lowering decides itself are the
//! ones fixed by an operation's kind where the checker says "unknown" (a
//! comparison is a Bool, a string literal a Str, `print` gives Nothing) and
//! the join of two paths' types at a merge.
//!
//! It is total over what `sky run` can execute today, and rejects the rest
//! with a [`LowerError`] (see below). [`lower_source`] runs the whole front
//! end and lowering on a source string, for tests in other crates. Run both
//! under `sky_ast::with_deep_stack`: lowering recurses once per nesting
//! level, which the parser bounds.
//!
//! # Module layout
//!
//! * The file's top-level statements are the function [`Module::MAIN`]
//!   (`<main>`), first in the module; it has no results and no parameters.
//!   A declaration without a value at the top level
//!   (`A: PositiveDefinite Matrix[N, N]`) is an *input*: an `input`
//!   operation, at the declaration, whose result has the declared type. It
//!   fails when it runs ("`A` is declared but has no value": nothing
//!   supplies an input in v0.1), so the program prints what comes before it
//!   and stops there, as `sky run` does.
//! * Every `def` (however nested) is a function with the def's name (a
//!   later def of the same name is `name#2`, ...), in source order after
//!   `<main>`. Its parameters are typed by the annotations
//!   (unannotated: `Unknown`), and it has exactly one result: the return
//!   type, `Unknown` when the checker cannot say, `Nothing` when it returns
//!   no value. Calls resolve the way the checker resolved them (by the
//!   callee's signature), so a callee is always a function known at compile
//!   time.
//!
//! # Globals
//!
//! SIR functions have no globals in v1, but the interpreter reads module
//! names when a function is *called*. So a function takes one extra
//! **trailing parameter for each module-level name it may read** (directly
//! or through a function it calls), and every call passes the module's
//! current value of each. Names the function assigns first (or takes as
//! parameters) are not counted; a predefined constant nobody rebound is a
//! literal (`const.float`), not a parameter. The parameters start as the
//! values of those names in the function's scope, so "the function's own
//! variable, else the module's" is an ordinary lookup, and a call passes
//! the *module's* values, never the caller's locals.
//!
//! A predefined constant is a literal only where **no binding of its name
//! can be in scope**: the interpreter keeps constants in the same
//! environment as variables, so `if c: pi = 2` changes `pi` after the `if`.
//! A constant that the file or a function binds inside an `if`, `while` or
//! `for` is therefore put in scope as a `const.float` at the start of that
//! function (`Lower::seed_constants`) and is an ordinary variable from then
//! on (merged after the `if`, carried by the loop); one the module rebinds
//! anywhere is a global as above.
//!
//! # Variables are SSA
//!
//! Lowering keeps a map from name to the current SSA value, with the
//! scoping the checker and interpreter define: `if`, `while` and `for` share the
//! enclosing scope, a `def` opens a new one.
//!
//! * `if`: each arm is lowered from the scope before the `if`. Afterwards
//!   every name whose value differs on some path becomes a **result** of the
//!   `if`, and each arm yields its current values. A name that is bound
//!   after the `if` only on some paths (the checker calls it "maybe
//!   undefined") is not a result: a checked program cannot read it. "Some
//!   paths" are the arms that *fall through*: an arm that `break`s,
//!   `continue`s or returns takes no part, as in the checker's merge
//!   (`sky-types/src/flow.rs`), so `t` is a result of an `if` whose other
//!   arm `break`s (with a stand-in on the breaking path, which the flags
//!   make unobservable). A name the enclosing loop's exit reads (a new name
//!   of a `while true`) is a result too, if a `break`ing arm binds it.
//! * `while` is `scf.while`: the variables the body assigns (that are bound
//!   before the loop) are the **loop-carried** values, block arguments of
//!   `before` and `after` and results of the loop. `before` computes the
//!   condition; `after` is the body and yields the next values.
//! * an `elif` chain is **flat**: a `done` flag and one sibling `if` per
//!   arm (below).
//! * `and`/`or` are a running Bool updated by sibling `if`s, one per
//!   operand (below); `a if c else b` is an `if` with a result.
//! * An operand of `and`/`or`/`not`, or a condition, that the checker could
//!   not type (`Unknown`: the result of a function with unannotated
//!   parameters) gets an `expect.bool[kw]` at the operand's span, with the
//!   construct's keyword (`elif` and the conditional expression are `if`).
//!   It fails with `sky run`'s message on `sky run`'s span, and it covers the
//!   operand that is only yielded (the last of an `and`/`or`), which no `if`
//!   tests. A typed Bool gets none.
//! * The operand of a unary `+`, and the argument of `inv(...)`, that the
//!   checker could not type get an `expect.number[pos|inv]` likewise (`+x`
//!   is otherwise just `x`; `inv(x)` is worded with its own spelling).
//! * An operation that reads operands records where, in the source
//!   (`OpData::operand_spans`), with the spans the tree-walker uses for the
//!   same operands: `b` of `a / b`, `x` of `-x`, `i` of `v[i]` (and the
//!   whole `1:3` item of a slice), the entries of an array literal, the
//!   arguments of a math function. The SIR interpreter labels its errors
//!   with them, so they are the tree-walker's.
//! * `for i in range(a, b, s)` is that `while` over a hidden carried index
//!   (`<i1>`, `<i2>`, ...) that the body's first operations copy into `i`
//!   and advance; see `for_loop.rs`. `for x in v` over an array is a
//!   [`LowerError`] until SIR has array values.
//!
//! # Flat chains
//!
//! Regions nest, and the verifier caps their depth
//! (`sky_ir::MAX_REGION_DEPTH`), so anything a *chain* of a source construct
//! produces must be siblings, never nested:
//!
//! * `a and b and c`: `t = a; t = if t { b } else { false };
//!   t = if t { c } else { false }` (`or`: `if t { true } else { b }`).
//!   Operand `k` is evaluated only in the arm that runs while `t` has not
//!   decided the result, so short-circuiting is exact.
//! * `if c1 { A } elif c2 { B } else { E }`: `done = false;
//!   if c1 { A; done = true }; g = if done { false } else { c2 };
//!   if g { B; done = true }; if not done { E }`. A condition is evaluated
//!   only if no earlier arm was taken, in order, as the interpreter does.
//!   Each arm's `if` merges its own variables (a name bound first in the
//!   chain is kept when every arm reaching the code after assigns it, with
//!   a stand-in where the arm did not run); a plain `if`/`else` is one `if`.
//! * early exits: see the next section; each statement after a possible
//!   exit is guarded by its own `if not flag`.
//!
//! # Region depth
//!
//! **The IR's region nesting is at most `2 * D + 4`**, where `D` is the
//! source depth: the deepest point of block nesting (the bodies of `def`,
//! `if`/`elif`/`else` and `while`, the top of the file being 0) plus
//! short-circuit nesting in expressions (an `and`/`or` operand or ternary
//! inside another). It does not depend on the length of any chain: an
//! `elif` chain, an `and`/`or` chain or a run of early exits adds a
//! constant. Each level of source nesting costs at most two IR levels: the
//! arm of its `if`/`while`, and the guard around the statement (below); the
//! constant covers the guard of a function's last `return`, and the `if`
//! that gates a loop's condition or an `elif` condition. `tests/depth.rs`
//! checks the bound on generated programs, and the chain tests in
//! `tests/robustness.rs` lower chains of thousands.
//!
//! # `break`, `continue` and early `return`
//!
//! Regions are single-entry and single-exit, and `return` may only be the
//! body's last operation, so these become Bool **flags** that are ordinary
//! SSA variables (named with a `<`, which no identifier has), carried by
//! the enclosing `if`s and loops like any variable:
//!
//! * `<ret>` (function level) and `<retval>` (the result so far, only for a
//!   function that returns a value). Both are set by `return`; `<retval>`
//!   starts as a stand-in of the result type that is never observed.
//! * `<brkN>` and `<skipN>` for the loop at depth `N`. `<brkN>` starts false
//!   and is carried by the loop; `<skipN>` is reset to false at the start of
//!   each iteration. `break` sets both, `continue` sets `<skipN>`, and
//!   `return` sets `<ret>`, `<retval>` and both flags of **every**
//!   enclosing loop (so each loop stops and the code after each is skipped).
//! * After a statement that may have set a flag, every statement after it
//!   in the block is a **guard**: `if not <flag> { stmt }` (`<skipN>` inside
//!   a loop, `<ret>` outside), so nothing after an exit runs. A statement
//!   that may itself exit gets its own guard; a maximal run of statements
//!   that cannot exit shares one. Guards are siblings, so a block adds one
//!   level of nesting for them however many exits it has. A guarded group
//!   passes its values on through the `if`'s results: every name it assigns
//!   is a result, with a stand-in on the skipping path. Code after a
//!   statement that always exits is dropped.
//! * A loop whose body can `break` or `return` evaluates its condition only
//!   while `<brkN>` is false: `if not <brkN> { cond } else { false }`.
//! * At the end of the function, `return <retval>`. A function whose only
//!   `return` is its last statement, or that has none, uses no flags: the
//!   returned value is `return`'s operand directly.
//!
//! Where a path has no value for a name the merge needs (it returned, or
//! skipped by a guard), it yields a **stand-in** (`const.float 0`, `false`,
//! `""`, `nothing`, of the result's type). It is never observed: the flags
//! make every reader skip it. Names the checker keeps out of merges (an arm
//! that returned takes no part) are kept out here too.
//!
//! # Derivatives
//!
//! `derivative(f, x)(args)` is a `call` of `df/dx`, the function
//! `sky_ad::differentiate_into` adds to the module once all functions exist
//! (see `derivative.rs`); the call keeps the second result, the
//! derivative. A function `sky-ad` cannot differentiate is a
//! [`LowerError`] on the call.
//!
//! # Rejected with a [`LowerError`]
//!
//! Anything `sky run` cannot execute either: attributes, `∂` and `∇`,
//! writes through an index or attribute, and an input declaration anywhere
//! but the top level of the file; and anything SIR v1 has no
//! representation for: a function used as a value (passed, stored,
//! returned), calling anything that is not a function defined with `def`.
//!
//! Two programs `sky run` runs are refused for that reason, by design: a
//! function defined in more than one arm of an `if` and called afterwards
//! (a SIR call names one function, fixed at compile time: "`f` is not a
//! function known at compile time"), and a nested `def` that reads a local
//! of its enclosing function (a SIR function sees only its own values and
//! the module's names, which lowering passes; "cannot find `a` in this
//! scope", even when the inner function is never called, where `sky run`
//! fails only if it is). `sky run` falls back to the tree-walker on a
//! [`LowerError`]. The spec's "Functions" lists both.
//!
//! Not rejected, though `sky run` reports it only when it evaluates the
//! literal: a number literal too large for an `f64` (or whose SI value is).
//! It is a `literal.error` operation that fails when it runs, with
//! `sky run`'s error on `sky run`'s span, so a literal in a branch that never
//! runs is harmless.
//!
//! # Effects
//!
//! Lowering leaves every function's effects to `sky_ir::effects`, which it
//! runs once over the finished module: the union over the operations, with
//! calls taking their callee's effects (so a function that only calls a
//! printing function is `io` too).
//!
//! # Where the verifier plugs in
//!
//! `lower` returns an unverified module. When `sky-ir` has its verifier,
//! [`lower_source`] and the tests call it on every lowered module; that is
//! the single place to add the call (`lower_source`, below, and
//! `tests/common/mod.rs`).

#![forbid(unsafe_code)]

mod accuracy;
mod analysis;
mod array;
mod build;
mod chain;
mod contract;
mod ctx;
mod derivative;
mod expr;
mod for_loop;
mod func;
mod stmt;

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use sky_ast::Module as Ast;
use sky_diag::{Diagnostic, SourceFile, Span};
use sky_ir::{Edition, Module};
use sky_types::Checked;

use crate::build::Builder;
use crate::ctx::{FnState, Lower, Needed};

/// Why a checked module could not be lowered.
#[derive(Clone, Debug, PartialEq)]
pub struct LowerError {
    /// Lowercase, no trailing period.
    pub message: String,
    /// The expression or statement lowering stopped at.
    pub span: Span,
    /// Why, when the message alone does not say.
    pub note: Option<String>,
}

impl LowerError {
    pub(crate) fn new(message: impl Into<String>, span: Span) -> Self {
        LowerError {
            message: message.into(),
            span,
            note: None,
        }
    }

    pub(crate) fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// The error as an ordinary diagnostic.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let d = Diagnostic::error(self.message.clone(), self.span);
        match &self.note {
            Some(n) => d.with_note(n.clone()),
            None => d,
        }
    }
}

impl From<sky_interp::RuntimeError> for LowerError {
    /// A literal the interpreter cannot read (too large, unit overflow).
    fn from(e: sky_interp::RuntimeError) -> Self {
        LowerError {
            message: e.message.clone(),
            span: e.span,
            note: e.notes.first().cloned(),
        }
    }
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(n) = &self.note {
            write!(f, " ({n})")?;
        }
        Ok(())
    }
}

impl std::error::Error for LowerError {}

/// Lowers `module` (parsed from `src` and checked without errors, with
/// `checked` the checker's result for it) to SIR. The module is unverified.
pub fn lower(module: &Ast, src: &str, checked: &Checked) -> Result<Module, LowerError> {
    lower_with_defs(module, src, checked).map(|(m, _)| m)
}

/// [`lower`], and which SIR function each `def` became: the span of the
/// `def`'s name (the key of `Checked::funcs`, and `Function::def_span` in
/// the interpreter) to the function's name in the module.
pub fn lower_with_defs(
    module: &Ast,
    src: &str,
    checked: &Checked,
) -> Result<(Module, HashMap<Span, String>), LowerError> {
    lower_impl(module, src, checked, false, false).map(|(m, d, _)| (m, d))
}

/// [`lower_with_defs`] for the `def`s alone: `<main>` is empty, and the
/// file's other top-level statements are not lowered. A function reads
/// module names through its trailing parameters (see "Globals"), so it
/// needs none of them to be lowered, only their types, which the checker
/// gives. This is what `sky run` differentiates (`sky-derive`): the tree
/// walker supplies the module's values, so a derivative works in a program
/// whose top level SIR cannot express yet (`equation`, `solve`, `±`), as
/// long as the functions it differentiates can be.
pub fn lower_defs(
    module: &Ast,
    src: &str,
    checked: &Checked,
) -> Result<(Module, HashMap<Span, String>), LowerError> {
    lower_impl(module, src, checked, false, true).map(|(m, d, _)| (m, d))
}

/// [`lower`], and the `accuracy` declarations of the program as goals for
/// the roundoff analysis (`sky_ir::roundoff`). The goals refer to the
/// module returned, as lowering produced it: run the analysis on it before
/// any pass.
pub fn lower_with_accuracy(
    module: &Ast,
    src: &str,
    checked: &Checked,
) -> Result<(Module, Vec<sky_ir::roundoff::Goal>), LowerError> {
    lower_impl(module, src, checked, false, false).map(|(m, _, g)| (m, g))
}

/// [`lower`] for a module that is about to be executed in place of the tree
/// walker (`sky run`'s SIR engine), which must behave exactly as the tree
/// walker does. It refuses, with a [`LowerError`], the programs whose SIR
/// would run differently because SIR cannot express a check the tree walker
/// makes at run time: today, a `for` over `range` with a bound or step
/// computed at run time (see `for_loop.rs`). Everything else is [`lower`].
///
/// Tools that only look at the IR (`sky check --emit-ir`) keep using
/// [`lower`], which lowers such a loop as far as `f64` arithmetic goes.
pub fn lower_for_run(module: &Ast, src: &str, checked: &Checked) -> Result<Module, LowerError> {
    lower_impl(module, src, checked, true, false).map(|(m, _, _)| m)
}

/// Everything one lowering produces: the module, which SIR function each
/// `def` became, and the `accuracy` goals.
type Lowered = (Module, HashMap<Span, String>, Vec<sky_ir::roundoff::Goal>);

fn lower_impl(
    module: &Ast,
    src: &str,
    checked: &Checked,
    for_run: bool,
    defs_only: bool,
) -> Result<Lowered, LowerError> {
    let main = FnState {
        b: Builder::new(Module::MAIN.to_string(), module.span),
        scope: Default::default(),
        module_vals: Default::default(),
        loops: 0,
        nest: 0,
        is_main: true,
        flags: false,
        ret_value: false,
        fn_name: None,
    };
    let mut lw = Lower {
        src,
        checked,
        defs: HashMap::new(),
        mod_assigned: BTreeSet::new(),
        fns: HashMap::new(),
        literal_bindings: HashMap::new(),
        out: Vec::new(),
        used_names: HashMap::new(),
        st: main,
        stack: Vec::new(),
        derivatives: Vec::new(),
        for_run,
        goals: Vec::new(),
    };
    lw.collect_defs(&module.stmts);
    lw.collect_module_names(&module.stmts);
    lw.collect_literal_bindings(&module.stmts);
    lw.seed_constants(&module.stmts, module.span);
    if defs_only {
        lw.defs_in(&module.stmts)?;
    } else {
        lw.block(&module.stmts, &Needed::new())?;
    }
    lw.st
        .b
        .op(sky_ir::OpKind::Return, Vec::new(), Vec::new(), module.span);
    let Lower {
        st,
        mut out,
        derivatives,
        fns,
        goals,
        ..
    } = lw;
    out.sort_by_key(|(start, _)| *start);
    let mut functions = vec![st.b.finish()];
    functions.extend(out.into_iter().map(|(_, f)| f));
    let mut module = Module {
        edition: Edition::default(),
        functions,
    };
    // The derivative functions the program asked for, now that the
    // functions they differentiate exist.
    for want in derivatives {
        sky_ad::differentiate_into(&mut module, &want.func, want.wrt)
            .map_err(|e| derivative::ad_error(&e, want.span))?;
    }
    // Effects follow from the bodies (the derivative functions included)
    // and the calls between them; one pass owns that rule
    // (`sky_ir::effects`), lowering does not repeat it.
    let _ = sky_ir::effects::infer_effects(&mut module);
    let defs = fns
        .into_iter()
        .map(|(def, info)| (def, info.name))
        .collect();
    Ok((module, defs, goals))
}

/// Lexes, parses, checks and lowers `src`: the whole path from source text
/// to SIR, for tests and tools. The error is the rendered diagnostics
/// (lexer, parser or checker errors, or the [`LowerError`]).
pub fn lower_source(src: &str) -> Result<Module, String> {
    sky_ast::with_deep_stack(|| lower_source_inner(src))
}

fn lower_source_inner(src: &str) -> Result<Module, String> {
    let file = SourceFile::new("<source>", src);
    let render = |ds: &[&Diagnostic]| {
        ds.iter()
            .map(|d| d.render_to_string(&file))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let lexed = match sky_syntax::lex(src) {
        Ok(l) => l,
        Err(e) => return Err(render(&e.diagnostics().collect::<Vec<_>>())),
    };
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        let ds: Vec<Diagnostic> = errors
            .into_iter()
            .map(sky_ast::ParseError::into_diagnostic)
            .collect();
        return Err(render(&ds.iter().collect::<Vec<_>>()));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        let ds: Vec<&Diagnostic> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == sky_diag::Severity::Error)
            .collect();
        return Err(render(&ds));
    }
    let module = lower(&ast, src, &checked).map_err(|e| render(&[&e.to_diagnostic()]))?;
    // Every module this function returns has been checked against SIR's
    // invariants: a verifier error here is a lowering bug, reported with the
    // verifier's own messages.
    if let Err(errors) = sky_ir::verify(&module) {
        let msgs: Vec<String> = errors
            .iter()
            .map(|e| {
                format!(
                    "internal error: lowering produced invalid SIR: {}",
                    e.message
                )
            })
            .collect();
        return Err(msgs.join("\n"));
    }
    Ok(module)
}
