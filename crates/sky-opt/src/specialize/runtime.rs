//! Specializing for values known at run time: the hook an engine calls.
//!
//! The pass in [`super`] uses what a call site *proves*. An engine that runs
//! a function (the native engine compiling it, say) also learns what its
//! arguments *are*: it has the values of the first call. [`specialize_for`]
//! turns those values into the same facts, and returns a module with a
//! specialized copy of the function, which the engine compiles or runs
//! instead of the generic one **for calls whose arguments match**:
//!
//! ```text
//! let args = [Some(Value::scalar(2.0)), None];      // x = 2, y unknown
//! let name = specialized_name(&module, "f", &args); // "f[x=2]"
//! let module2 = specialize_for(&module, "f", &args);
//! // module2 has everything module has, plus `f[x=2]` (and the
//! // specializations of the calls inside it), verified.
//! // Call `f[x=2]` with the same arguments (all of them: the signature is
//! // unchanged) whenever `x == 2`; call `f` otherwise.
//! ```
//!
//! # The guard is the caller's
//!
//! The specialized function computes what `f` computes **only for
//! arguments equal to the values it was specialized for** (bitwise, for
//! numbers; equal shapes for arrays). Checking that, and falling back to
//! `f`, is the engine's job: it is one comparison per specialized argument.
//! The signature is unchanged (a constant parameter stays, unused), so the
//! call is made exactly as `f`'s would be.
//!
//! # What a value gives
//!
//! * a finite number for a scalar `f64` parameter of the same dimension: the
//!   parameter is that constant (`-0` and `0` are different constants);
//! * a Bool for a Bool parameter;
//! * an array for a parameter whose shape has size variables: they take the
//!   array's sizes (the values in the array are not facts);
//! * `None`, or a value that does not fit the parameter (wrong kind or
//!   dimension, a parameter the body never reads), gives nothing.
//!
//! With no fact, [`specialize_for`] returns the module unchanged and
//! [`specialized_name`] is the function's own name.
//!
//! # The cache
//!
//! [`SpecializationCache`] keeps the results for one module, keyed by
//! function and facts, so repeated calls with the same values specialize
//! once. It holds at most [`SpecializationCache::CAPACITY`] entries and drops
//! the oldest first.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use sky_interp::Value;
use sky_ir::{Function, Module, Repr, Type};
use sky_types::ShapeDim;

use super::facts::{clone_name, sort, used_params, Const, Fact, Kind};
use super::{build, Specializer, MAX_CLONE_OPS};

/// The facts the values `args` give about the parameters of `f`.
fn facts_of(f: &Function, args: &[Option<Value>]) -> Vec<Fact> {
    if args.len() != f.params.len() {
        return Vec::new();
    }
    let used = used_params(f);
    let mut facts: Vec<Fact> = Vec::new();
    for (i, (arg, param)) in args.iter().zip(&f.params).enumerate() {
        let (Some(arg), true) = (arg, used[i]) else {
            continue;
        };
        let name = super::facts::param_name(f, i);
        match (arg, param) {
            (Value::Number { value, dim }, Type::Number(p))
                if p.shape.rank() == 0
                    && matches!(p.repr, Repr::Unchosen | Repr::F64)
                    && p.dim == *dim
                    && value.is_finite() =>
            {
                let value = Const::Float(value.to_bits());
                facts.push(Fact {
                    kind: Kind::Const { param: i, value },
                    why: format!("the argument for `{name}` is {}", value.text()),
                    name,
                });
            }
            (Value::Bool(b), Type::Bool) => {
                let value = Const::Bool(*b);
                facts.push(Fact {
                    kind: Kind::Const { param: i, value },
                    why: format!("the argument for `{name}` is {b}"),
                    name,
                });
            }
            (Value::Array { dims, dim, .. }, Type::Number(p))
                if p.shape.rank() == dims.len() && p.dim == *dim =>
            {
                for (pd, n) in p.shape.dims().iter().zip(dims) {
                    if let ShapeDim::Var(var) = pd {
                        let n = *n as u64;
                        if facts
                            .iter()
                            .any(|f| matches!(&f.kind, Kind::Size { var: v, .. } if v == var))
                        {
                            continue;
                        }
                        facts.push(Fact {
                            kind: Kind::Size {
                                var: var.clone(),
                                n,
                            },
                            why: format!("the argument for `{name}` has {} = {n}", var.name()),
                            name: name.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    sort(&mut facts);
    facts
}

/// The name [`specialize_for`] gives its specialization of `func` for
/// `args`: `func` itself when no argument gives a fact (or `func` is not in
/// `module`).
pub fn specialized_name(module: &Module, func: &str, args: &[Option<Value>]) -> String {
    match module.function(func) {
        Some(f) => {
            let facts = facts_of(f, args);
            if facts.is_empty() {
                func.to_string()
            } else {
                clone_name(func, &facts)
            }
        }
        None => func.to_string(),
    }
}

/// `module` plus a specialized copy of `func` for the argument values
/// `args` (`None` is unknown), named [`specialized_name`], and the
/// specializations of the calls inside it. The module is verified by the
/// tests, not here; run the pass manager's verification if it matters.
///
/// Nothing in `module` changes: `func` stays as it was, and other functions
/// still call it. If no argument gives a fact, or the body is larger than
/// [`MAX_CLONE_OPS`], the result is `module` unchanged.
pub fn specialize_for(module: &Module, func: &str, args: &[Option<Value>]) -> Module {
    let mut out = module.clone();
    let Some(base) = module.function(func) else {
        return out;
    };
    let facts = facts_of(base, args);
    if facts.is_empty() || super::size(base) > MAX_CLONE_OPS {
        return out;
    }
    let name = clone_name(func, &facts);
    if out.function(&name).is_some() {
        return out;
    }
    let built = build(base, name, &facts, true, &module.functions);
    out.functions.push(built.func);
    let idx = out.functions.len() - 1;
    // The calls inside the copy are specialized like any other.
    let mut s = Specializer::new(&mut out);
    s.run(vec![idx]);
    out
}

/// One cached specialization.
#[derive(Debug)]
pub struct Specialization {
    /// The module: the original plus the specialized function(s).
    pub module: Module,
    /// The function to call ([`specialized_name`]).
    pub name: String,
}

/// Specializations of one module, by function and facts.
#[derive(Debug)]
pub struct SpecializationCache {
    module: Module,
    entries: HashMap<String, Arc<Specialization>>,
    order: VecDeque<String>,
    hits: usize,
    misses: usize,
}

impl SpecializationCache {
    /// The most entries kept.
    pub const CAPACITY: usize = 64;

    /// A cache for specializations of `module`.
    pub fn new(module: Module) -> Self {
        SpecializationCache {
            module,
            entries: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// The module the cache specializes.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// The specialization of `func` for `args`, made on the first request
    /// and reused after. `None` if `func` is not in the module or `args`
    /// give no fact (run the generic function).
    pub fn get(&mut self, func: &str, args: &[Option<Value>]) -> Option<Arc<Specialization>> {
        let name = specialized_name(&self.module, func, args);
        if name == func {
            return None;
        }
        if let Some(hit) = self.entries.get(&name) {
            self.hits += 1;
            return Some(Arc::clone(hit));
        }
        self.misses += 1;
        let made = Arc::new(Specialization {
            module: specialize_for(&self.module, func, args),
            name: name.clone(),
        });
        if self.order.len() >= Self::CAPACITY {
            if let Some(old) = self.order.pop_front() {
                self.entries.remove(&old);
            }
        }
        self.order.push_back(name.clone());
        self.entries.insert(name, Arc::clone(&made));
        Some(made)
    }

    /// Requests answered from the cache, and requests that specialized.
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// How many specializations are kept.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
