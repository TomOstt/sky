//! Runtime bindings.

use std::collections::BTreeMap;

use crate::Value;

/// Names bound to values. `Clone` so the REPL can snapshot it before a line
/// and restore the snapshot if the line fails. Never serialized: v0.1 does
/// not save REPL sessions (spec non-negotiable 3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeEnv {
    vars: BTreeMap<String, Value>,
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

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }

    pub fn set(&mut self, name: impl Into<String>, value: Value) {
        self.vars.insert(name.into(), value);
    }

    /// Every binding, in name order.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.vars.iter().map(|(k, v)| (k.as_str(), v))
    }
}
