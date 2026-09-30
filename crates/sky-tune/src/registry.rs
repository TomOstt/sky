//! The registry of things `sky tune` can tune.
//!
//! Rust has no life-before-main, so crates cannot register themselves. The
//! `sky` binary builds a [`Registry`] and each crate that has tunable kernels
//! offers a `register(&mut Registry)` function it calls:
//!
//! ```
//! use sky_tune::{grid, Registry, Tunable, Work};
//!
//! fn tunables() -> Vec<Tunable> {
//!     let space = grid(&[("block", &[32, 64, 128])]);
//!     vec![Tunable::new("my_kernel", "2^16", space, Work::Unknown, |p| {
//!         // Run the kernel with `p.get("block")`; return something that
//!         // depends on the answer.
//!         p.get_or("block", 0) as u64
//!     })
//!     .unwrap()]
//! }
//!
//! let mut registry = Registry::new();
//! registry.add("my_kernel", tunables);
//! assert_eq!(registry.names(), ["my_kernel"]);
//! ```
//!
//! A builder is a plain `fn`, called only when the kernel is about to be
//! tuned, so allocating benchmark buffers costs nothing for `sky hw` or
//! `sky tune --show`.

use crate::measure::Tunable;

/// Builds the tunables (one per problem size worth tuning) of one kernel.
pub type Builder = fn() -> Vec<Tunable>;

/// The tunables `sky tune` knows about.
#[derive(Default)]
pub struct Registry {
    entries: Vec<(&'static str, Builder)>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Registry {
        Registry::default()
    }

    /// A registry holding Sky's own demo tunables (see [`crate::demo`]).
    pub fn with_demos() -> Registry {
        let mut registry = Registry::new();
        crate::demo::register(&mut registry);
        registry
    }

    /// Adds a builder under `name`, which is what `sky tune <name>` matches.
    /// A second builder with the same name replaces the first.
    pub fn add(&mut self, name: &'static str, builder: Builder) -> &mut Registry {
        self.entries.retain(|(n, _)| *n != name);
        self.entries.push((name, builder));
        self
    }

    /// The registered names, in the order they were added.
    pub fn names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|(n, _)| *n).collect()
    }

    /// Builds the tunables of every entry whose name contains `filter` (all
    /// entries for `None`), as `(name, tunables)`.
    pub fn build(&self, filter: Option<&str>) -> Vec<(&'static str, Vec<Tunable>)> {
        self.entries
            .iter()
            .filter(|(name, _)| filter.is_none_or(|f| name.contains(f)))
            .map(|(name, builder)| (*name, builder()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Params, Work};

    fn one() -> Vec<Tunable> {
        vec![Tunable::new(
            "one",
            "s",
            vec![Params::new().with("a", 1)],
            Work::Unknown,
            |_| 0,
        )
        .unwrap()]
    }

    #[test]
    fn add_filter_and_replace() {
        let mut r = Registry::new();
        r.add("alpha_one", one).add("beta_one", one);
        assert_eq!(r.names(), ["alpha_one", "beta_one"]);
        assert_eq!(r.build(None).len(), 2);
        let only = r.build(Some("alpha"));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].0, "alpha_one");
        assert!(r.build(Some("gamma")).is_empty());
        r.add("alpha_one", one);
        assert_eq!(r.names(), ["beta_one", "alpha_one"]);
    }

    #[test]
    fn the_demos_are_registered() {
        let r = Registry::with_demos();
        assert!(r.names().len() >= 2);
        assert!(r.names().contains(&"dot_blocked"));
        assert!(r.names().contains(&"transpose_tiled"));
    }
}
