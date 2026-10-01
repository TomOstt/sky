//! Typing a caller again, once the functions it calls have typed
//! specializations.
//!
//! A value that comes back from a function with unannotated parameters is
//! `unknown`, and so is a loop variable that one feeds:
//!
//! ```text
//! def next_state(s): return s * 16807 - ...
//! s = 42
//! for i in range(n): s = next_state(s)   # s: unknown, so the call has no type fact
//! ```
//!
//! The call gives no type fact, because its argument's type depends on the
//! call's own result. [`Specializer::retype_callers`] breaks the cycle
//! optimistically, the way [`super::retype`] types a loop: it infers the
//! caller again ([`infer`]), with every call of a function with `unknown`
//! parameters typed as its specialization for the arguments' inferred types
//! returns. A specialization not made yet counts as returning nothing so far
//! ([`Ty::Bot`]); it is then made like any other ([`Specializer::adopt`]:
//! the same limits, the same "worth it" and validity tests, the same
//! `SPEC-005` explanation at the call), and the caller inferred again, until
//! every call it needs exists. `s` starts as a number, `next_state[s:Scalar]`
//! returns a number, so `s` stays one: the loop is typed and the call goes to
//! the clone.
//!
//! The typed caller replaces the old one only if the verifier accepts it,
//! with the calls retargeted at their clones; a value whose inferred type
//! is still `unknown` (a specialization that was not made, a variable that
//! really holds different types) keeps its run-time checks. As everywhere in
//! this pass, the code and its order are unchanged: only types become known,
//! and the type checks they answer go.

use std::cell::RefCell;

use sky_diag::Span;
use sky_ir::{verify_function, Function, OpData, OpKind, Type};

use super::facts::{clone_name, type_facts, Fact};
use super::retype::{apply, infer, Env, Inference, Ty};
use super::Specializer;
use crate::util::reachable_ops;

/// The most times every function is looked at again.
const MAX_CALLER_ROUNDS: usize = 4;

/// The most times one caller is inferred again for specializations it
/// asked for.
const MAX_CALLER_TRIES: usize = 8;

/// A specialization a caller's typing asks for: the callee, the type facts
/// and the call's span.
type Wanted = (String, Vec<Fact>, Span);

/// The specialization a call of `g` with arguments typed `args` goes to,
/// and its type facts: `None` if `g` has no `unknown` parameter that the
/// arguments type. `Err(())` while an argument is still [`Ty::Bot`].
fn typed_target(g: &Function, args: &[Ty]) -> Result<Option<(String, Vec<Fact>)>, ()> {
    if !g.params.iter().any(|p| matches!(p, Type::Unknown)) || args.len() != g.params.len() {
        return Ok(None);
    }
    let mut tys: Vec<&Type> = Vec::with_capacity(args.len());
    for (a, p) in args.iter().zip(&g.params) {
        match (a, p) {
            (Ty::Bot, Type::Unknown) => return Err(()),
            (Ty::T(t), Type::Unknown) => tys.push(t),
            (_, p) => tys.push(p),
        }
    }
    Ok(type_facts(g, &tys).map(|facts| (clone_name(&g.name, &facts), facts)))
}

/// Whether `f` has a value of unknown type that is not a parameter.
fn has_unknown(f: &Function) -> bool {
    let params = f.param_values();
    f.values
        .iter()
        .enumerate()
        .any(|(i, v)| matches!(v.ty, Type::Unknown) && !params.iter().any(|p| p.index() == i))
}

impl Specializer<'_> {
    /// Types every function again whose calls of functions with `unknown`
    /// parameters can go to typed specializations (see the module
    /// documentation), making the specializations it needs.
    pub(super) fn retype_callers(&mut self) {
        for _ in 0..MAX_CALLER_ROUNDS {
            let mut changed = false;
            for fi in 0..self.m.functions.len() {
                if has_unknown(&self.m.functions[fi]) && self.retype_caller(fi) {
                    changed = true;
                }
            }
            self.drain();
            if !changed {
                break;
            }
        }
    }

    /// Types function `fi` again; whether it changed.
    fn retype_caller(&mut self, fi: usize) -> bool {
        let mut tries = 0;
        let inferred = loop {
            let (inferred, wanted) = self.infer_caller(fi);
            if wanted.is_empty() || tries >= MAX_CALLER_TRIES {
                break inferred;
            }
            tries += 1;
            for (callee, facts, span) in wanted {
                let name = clone_name(&callee, &facts);
                if self.m.functions.iter().any(|g| g.name == name) {
                    continue;
                }
                let Some(idx) = self.m.functions.iter().position(|g| g.name == callee) else {
                    continue;
                };
                // The specialization may come out with other facts, or not
                // at all: either way this name will not exist, and the call
                // keeps its unknown result.
                if self
                    .adopt(idx, facts, span)
                    .is_none_or(|(made, _)| made != name)
                {
                    self.rejected.insert(name);
                }
            }
            self.drain();
        };
        let f = &self.m.functions[fi];
        let gains = f.values.iter().enumerate().any(|(i, v)| {
            inferred.seen[i]
                && matches!(v.ty, Type::Unknown)
                && matches!(&inferred.tys[i], Ty::T(t) if !matches!(t, Type::Unknown))
        });
        if !gains {
            return false;
        }
        let mut g = f.clone();
        apply(&mut g, &inferred, None);
        self.retarget_typed(&mut g);
        if verify_function(&self.m.functions, &g).is_err() {
            return false;
        }
        self.m.functions[fi] = g;
        true
    }

    /// Infers function `fi` with calls typed by their specializations, and
    /// the specializations that do not exist yet.
    fn infer_caller(&self, fi: usize) -> (Inference, Vec<Wanted>) {
        let funcs = &self.m.functions;
        let wanted: RefCell<Vec<Wanted>> = RefCell::new(Vec::new());
        let typed = |op: &OpData, args: &[Ty]| -> Option<Ty> {
            let OpKind::Call { callee } = &op.kind else {
                return None;
            };
            let g = funcs.iter().find(|g| g.name == *callee)?;
            let (name, facts) = match typed_target(g, args) {
                Err(()) => return Some(Ty::Bot),
                Ok(target) => target?,
            };
            if let Some(clone) = funcs.iter().find(|h| h.name == name) {
                return match clone.results.as_slice() {
                    [t] => Some(Ty::T(t.clone())),
                    _ => None,
                };
            }
            if self.rejected.contains(&name) {
                return None;
            }
            wanted.borrow_mut().push((callee.clone(), facts, op.span));
            Some(Ty::Bot)
        };
        let results = std::collections::HashMap::new();
        let env = Env {
            funcs,
            results: &results,
            typed_call: Some(&typed),
        };
        let inferred = infer(&funcs[fi], &env);
        let mut wanted = wanted.into_inner();
        wanted.dedup_by(|a, b| a.0 == b.0 && clone_name(&a.0, &a.1) == clone_name(&b.0, &b.1));
        (inferred, wanted)
    }

    /// Points every call of `g` whose arguments now type a specialization
    /// that exists at it.
    fn retarget_typed(&self, g: &mut Function) {
        for o in reachable_ops(g) {
            let OpKind::Call { callee } = &g.op(o).kind else {
                continue;
            };
            let Some(callee) = self.m.functions.iter().find(|h| h.name == *callee) else {
                continue;
            };
            let args: Vec<Ty> = g
                .op(o)
                .operands
                .iter()
                .map(|v| Ty::T(g.ty(*v).clone()))
                .collect();
            let Ok(Some((name, _))) = typed_target(callee, &args) else {
                continue;
            };
            if self.m.functions.iter().any(|h| h.name == name) {
                g.ops[o.index()].kind = OpKind::Call { callee: name };
            }
        }
    }
}
