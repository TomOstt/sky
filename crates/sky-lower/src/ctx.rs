//! Lowering state and the small helpers every part of lowering shares.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use sky_ast::ast::span_text;
use sky_ast::Expr;
use sky_diag::Span;
use sky_ir::{Function, NumberType, OpKind, Repr, Type, ValueId};
use sky_types::{Checked, Ty};

use crate::build::Builder;
use crate::derivative::Wanted;
use crate::LowerError;

pub(crate) type R<T> = Result<T, LowerError>;

/// Names in scope and the SSA value each currently holds. Besides source
/// variables it holds the control-flow flags, whose names start with `<` so
/// they cannot clash with an identifier (see the crate docs).
pub(crate) type Scope = BTreeMap<String, ValueId>;

/// New names some enclosing construct will read after the code being
/// lowered: a guard that skips code (see `stmt.rs`) must still give them a
/// value on the skipping path.
pub(crate) type Needed = BTreeSet<String>;

/// The flag set when the function has returned.
pub(crate) const RET: &str = "<ret>";
/// The function's result so far (only for functions that return a value).
pub(crate) const RETVAL: &str = "<retval>";

/// The flag "loop `n` must stop" (`break`, or a `return` inside it).
pub(crate) fn brk_flag(n: usize) -> String {
    format!("<brk{n}>")
}

/// The flag "skip the rest of this iteration of loop `n`" (`break`,
/// `continue` or `return`).
pub(crate) fn skip_flag(n: usize) -> String {
    format!("<skip{n}>")
}

/// The function being built and where lowering is in it.
pub(crate) struct FnState {
    pub(crate) b: Builder,
    /// Names in scope. In a function this starts with the parameters and
    /// the module-level names the function reads, so "frame first, then
    /// module" (the interpreter's rule) is just a lookup.
    pub(crate) scope: Scope,
    /// The module-level names as passed to this function (its extra
    /// trailing parameters). Calls pass these on, never `scope`, because a
    /// callee sees module names, not its caller's locals.
    pub(crate) module_vals: Scope,
    /// How many `while` loops enclose the code being lowered.
    pub(crate) loops: usize,
    /// How many `if`/`while` bodies enclose it (0: the function's top).
    pub(crate) nest: usize,
    pub(crate) is_main: bool,
    /// Whether `return` lowers to flags (see the crate docs). False for a
    /// function whose only `return` is its last statement.
    pub(crate) flags: bool,
    /// Whether the function has a `<retval>` slot.
    pub(crate) ret_value: bool,
}

/// What a call needs to know about a lowered (or being lowered) function.
pub(crate) struct FnInfo {
    /// Its name in the SIR module (unique).
    pub(crate) name: String,
    pub(crate) result: Type,
    pub(crate) params: usize,
    /// The module-level names passed as trailing parameters, in order.
    pub(crate) globals: Vec<String>,
}

/// What the pre-pass found out about a `def`.
pub(crate) struct DefInfo {
    /// Names its body may read from the module (before filtering by what
    /// the module has bound).
    pub(crate) direct: BTreeSet<String>,
    /// The functions its body calls, by `def` name span.
    pub(crate) callees: Vec<Span>,
}

pub(crate) struct Lower<'a> {
    pub(crate) src: &'a str,
    pub(crate) checked: &'a Checked,
    pub(crate) defs: HashMap<Span, DefInfo>,
    /// Names assigned or declared at module level anywhere in the file.
    pub(crate) mod_assigned: BTreeSet<String>,
    pub(crate) fns: HashMap<Span, FnInfo>,
    /// The array literals that initialize a typed declaration, by the
    /// literal's span, with the span of the declared name (see `array.rs`).
    pub(crate) literal_bindings: HashMap<Span, Span>,
    /// Finished user functions, with their `def` position for ordering.
    pub(crate) out: Vec<(u32, Function)>,
    pub(crate) used_names: HashMap<String, usize>,
    pub(crate) st: FnState,
    /// The states of the functions around the one being lowered (the
    /// module's `<main>` first).
    pub(crate) stack: Vec<FnState>,
    /// The derivative functions `derivative(f, x)(...)` calls need, built
    /// once the module's functions exist (see `derivative.rs`).
    pub(crate) derivatives: Vec<Wanted>,
}

/// A checker type as a SIR type. `None` for a function value.
pub(crate) fn map_ty(t: &Ty) -> Option<Type> {
    match t {
        Ty::Value(v) => Some(Type::Number(NumberType {
            dim: v.dim,
            shape: v.ty.shape().clone(),
            tags: v.ty.tags().closure(),
            repr: Repr::Unchosen,
        })),
        Ty::Bool => Some(Type::Bool),
        Ty::Unknown => Some(Type::Unknown),
        Ty::Func(_) => None,
    }
}

/// The type of a value that is either `a` or `b` (two branches, or a loop's
/// initial and next value): equal types stay; numbers of one dimension and
/// shape keep the tags both have (the checker's merge rule); anything else
/// is [`Type::Unknown`].
pub(crate) fn join_ty(a: &Type, b: &Type) -> Type {
    match (a, b) {
        (Type::Number(x), Type::Number(y)) if x.dim == y.dim && x.shape == y.shape => {
            Type::Number(NumberType {
                tags: x.tags.join(y.tags),
                ..x.clone()
            })
        }
        _ if a == b => a.clone(),
        _ => Type::Unknown,
    }
}

/// Whether a value of type `from` can flow into a slot of type `to`.
pub(crate) fn assignable(from: &Type, to: &Type) -> bool {
    join_ty(from, to) == *to
}

impl<'a> Lower<'a> {
    pub(crate) fn text(&self, span: Span) -> &'a str {
        span_text(self.src, span).unwrap_or("")
    }

    pub(crate) fn err<T>(&self, message: impl Into<String>, span: Span) -> R<T> {
        Err(LowerError::new(message, span))
    }

    /// The checker's type of the expression at `span`, as a SIR type.
    pub(crate) fn ty_at(&self, span: Span) -> R<Type> {
        match self.checked.expr_ty(span) {
            Some(Ty::Func(sig)) => self.err(
                format!(
                    "the function `{}` is used as a value",
                    sky_diag::display_capped(sig.name())
                ),
                span,
            ),
            Some(t) => Ok(map_ty(&t).unwrap_or(Type::Unknown)),
            None => self.err("internal error: the checker recorded no type here", span),
        }
    }

    pub(crate) fn ty_of(&self, e: &Expr) -> R<Type> {
        self.ty_at(e.span)
    }

    pub(crate) fn op1(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        ty: Type,
        span: Span,
    ) -> ValueId {
        self.st.b.op1(kind, operands, ty, span)
    }

    pub(crate) fn const_bool(&mut self, value: bool, span: Span) -> ValueId {
        self.op1(OpKind::ConstBool(value), Vec::new(), Type::Bool, span)
    }

    /// The operation and value type of a stand-in of type `ty`: a value
    /// that is never observed, for a path that will not use it (a branch
    /// that returned, a loop's initial value of a name first assigned in
    /// the loop).
    pub(crate) fn placeholder_kind(ty: &Type) -> (OpKind, Type) {
        match ty {
            Type::Number(_) => (OpKind::ConstFloat { bits: 0 }, ty.clone()),
            Type::Bool => (OpKind::ConstBool(false), Type::Bool),
            Type::Str => (OpKind::ConstStr(String::new()), Type::Str),
            Type::Nothing | Type::Unknown => (OpKind::ConstNothing, Type::Nothing),
        }
    }

    pub(crate) fn placeholder(&mut self, ty: &Type, span: Span) -> ValueId {
        let (kind, vty) = Self::placeholder_kind(ty);
        self.op1(kind, Vec::new(), vty, span)
    }

    /// Binds `name` to `v` in the current scope.
    pub(crate) fn bind(&mut self, name: &str, v: ValueId) {
        self.st.b.name_value(v, name);
        self.st.scope.insert(name.to_string(), v);
    }

    /// The state of the module's own function, which holds the
    /// module-level names.
    pub(crate) fn module_state(&self) -> &FnState {
        self.stack.first().unwrap_or(&self.st)
    }
}
