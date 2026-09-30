//! Runtime bindings.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use sky_diag::Span;
use sky_types::ChainPlan;

use crate::Value;

/// Names bound to values. `Clone` so the REPL can snapshot it before a line
/// and restore the snapshot if the line fails. Never serialized: v0.1 does
/// not save REPL sessions (spec non-negotiable 3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeEnv {
    vars: BTreeMap<String, Value>,
    /// The checker's plans for `@` chains (`Checked::chains`), by the span
    /// of the whole chain expression: the tree-walker computes a planned
    /// chain in the plan's order and by its methods, as SIR does. Not
    /// program state; a REPL adds the plans of each input as it goes.
    chains: HashMap<Span, Arc<ChainPlan>>,
}

impl RuntimeEnv {
    pub fn new() -> Self {
        RuntimeEnv::default()
    }

    /// The environment every program starts in: each physical constant from
    /// `sky_units::CONSTANTS`, under its Unicode and ASCII names, with its SI
    /// value. The runtime twin of `sky_types::Env::with_prelude`, so a name
    /// the checker accepts as a constant also has a value at run time.
    pub fn with_prelude() -> Self {
        let mut env = RuntimeEnv::new();
        for c in sky_units::CONSTANTS {
            for name in [Some(c.name), c.ascii].into_iter().flatten() {
                env.set(name, Value::number(c.value, c.dim));
            }
        }
        env
    }

    /// Adds the chain plans of a checked input (`Checked::chains`, from
    /// checking the very source that will run). Spans must not clash with
    /// earlier plans (a REPL's are session offsets, so they do not).
    pub fn add_chain_plans(&mut self, plans: &HashMap<Span, Arc<ChainPlan>>) {
        self.chains
            .extend(plans.iter().map(|(s, p)| (*s, Arc::clone(p))));
    }

    /// The plan for the chain expression at `span`, if the checker made one.
    pub fn chain_plan(&self, span: Span) -> Option<&Arc<ChainPlan>> {
        self.chains.get(&span)
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }

    /// Moves the value out of `name`, leaving `Value::None` bound there; the
    /// caller stores the next value right away. `None` if `name` is unbound.
    pub(crate) fn take(&mut self, name: &str) -> Option<Value> {
        self.vars
            .get_mut(name)
            .map(|slot| std::mem::replace(slot, Value::None))
    }

    /// [`RuntimeEnv::set`] for a name that is usually bound already (a loop
    /// updating a variable): the key is not allocated again.
    pub(crate) fn assign(&mut self, name: &str, value: Value) {
        match self.vars.get_mut(name) {
            Some(slot) => *slot = value,
            None => {
                self.vars.insert(name.to_string(), value);
            }
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: Value) {
        self.vars.insert(name.into(), value);
    }

    /// Every binding, in name order.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.vars.iter().map(|(k, v)| (k.as_str(), v))
    }
}
