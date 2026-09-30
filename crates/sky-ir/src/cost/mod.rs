//! The static cost model: what a function costs, as a polynomial in the
//! sizes of its arrays (spec "Performance contracts").
//!
//! [`analyze`] reads a SIR module and gives every function a [`Cost`]: five
//! measures, each an [`Amount`] (an interval of polynomials in the symbolic
//! sizes, exact for straight-line code).
//!
//! | measure | what it counts |
//! | --- | --- |
//! | `flops` | floating-point operations, weighted as below |
//! | `bytes_read` | bytes of arrays that array operations read (8 per entry) |
//! | `bytes_written` | bytes of arrays that array operations write |
//! | `allocations` | arrays created |
//! | `allocated_bytes` | bytes of the arrays created |
//!
//! `bytes` in a contract is `bytes_read + bytes_written`. The model counts
//! what the reference implementation does; it is not a prediction of the
//! time a machine takes. It ignores scalar traffic (a scalar lives in a
//! register), control flow, comparisons, `print`, and the checks of
//! contracts (their conditions are operations like any other, and count).
//!
//! # Weights
//!
//! * `+ - * /`, negation, `sqrt`, `abs`, `min`, `max`, `floor`, `ceil`,
//!   `round`, `sign`: 1 flop per scalar operation. An elementwise array
//!   operation costs that per entry.
//! * Any other math function (`exp`, `log`, `sin`, `atan2`, `erf`, `cbrt`,
//!   `hypot`, the hyperbolics and their inverses, ...): 20 flops, the rough
//!   cost of a library call. `a ^ b` with a computed exponent: 20.
//! * `a ^ n` with a constant whole exponent: the multiplications of
//!   square-and-multiply (`x^2` is 1, `x^3` is 2, `x^4` is 2, `x^5` is 3),
//!   plus one division for a negative exponent; `x^(1/2)` is 1; another
//!   fraction is 20.
//! * `A @ B` with `A` of shape `(n, k)` and `B` of `(k, m)`: `2·n·k·m` flops
//!   (a vector is the matrix with one axis missing). Reads both operands,
//!   writes and allocates the result.
//! * `A⁻¹` of an `n × n` matrix: `2·n³` flops (a scalar reciprocal is 1).
//! * A Cholesky solve of `A x = b` with `b` a vector: `n³/3 + 2·n²` flops
//!   (factor, then two triangular solves; `2·n²·k` for `k` right sides), and
//!   two allocations: the result and the factor.
//! * A `fused` kernel: its body per entry, so `v + 2 * w` is `2·N` flops in
//!   one allocation of `N` floats, where the unfused chain has two of each.
//! * A transpose, slice or row read is a copy: reads and writes its
//!   entries and allocates them (a later backend may do better; a contract
//!   proven against this model then holds all the more).
//!
//! # Control flow and calls
//!
//! * A loop whose trip count is known multiplies its body: a counted
//!   `for i in range(10)`, `range(len(v))` (the size `N` of a `Vector[N]`),
//!   any of them with a start and a constant step. The counts are exact or
//!   bounded on both sides ([`Trips`]). A loop with any other condition is
//!   *unbounded* (a `while x > 1`, a `break`): its cost has no upper bound,
//!   and says the line of the loop.
//! * An `if` costs its condition plus the branch that runs: the lower bound
//!   is the cheaper branch and the upper bound is the dearer one, so the
//!   model gives the *maximum* of the branches as the cost.
//! * A call costs the callee's cost with the callee's sizes replaced by the
//!   sizes of the arguments. Recursion, direct or mutual, is unbounded.
//! * An operation on a value of unknown shape (an unannotated parameter) is
//!   unbounded.
//!
//! # What the bounds mean
//!
//! An [`Amount`] is a pair of polynomials with `lo <= actual <= hi` for every
//! run that finishes (one that stops with an error has done only part of the
//! work), at every size from 1 up. Polynomials compare with the sound test of
//! [`sky_types::poly`]: two polynomials are ordered only when they are so
//! for all sizes at least 1.

mod amount;
mod ops;

pub use amount::{Amount, At, Reason, ReasonKind, Trips};
pub use sky_types::poly::Poly;

use std::collections::{HashMap, HashSet};

use sky_ast::CostMetric;
use sky_diag::Span;

use crate::ir::{Function, Module};
use crate::op::OpKind;

/// The five measures of a piece of code.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Cost {
    pub flops: Amount,
    pub bytes_read: Amount,
    pub bytes_written: Amount,
    pub allocations: Amount,
    pub allocated_bytes: Amount,
}

impl Cost {
    pub fn zero() -> Cost {
        Cost::default()
    }

    pub fn is_zero(&self) -> bool {
        self.fields().iter().all(|a| a.is_zero())
    }

    fn fields(&self) -> [&Amount; 5] {
        [
            &self.flops,
            &self.bytes_read,
            &self.bytes_written,
            &self.allocations,
            &self.allocated_bytes,
        ]
    }

    fn from_fields(f: [Amount; 5]) -> Cost {
        let [flops, bytes_read, bytes_written, allocations, allocated_bytes] = f;
        Cost {
            flops,
            bytes_read,
            bytes_written,
            allocations,
            allocated_bytes,
        }
    }

    fn map(&self, f: impl Fn(&Amount) -> Amount) -> Cost {
        Cost::from_fields(self.fields().map(f))
    }

    fn zip(&self, other: &Cost, f: impl Fn(&Amount, &Amount) -> Amount) -> Cost {
        let (a, b) = (self.fields(), other.fields());
        Cost::from_fields(std::array::from_fn(|i| f(a[i], b[i])))
    }

    pub fn add(&self, other: &Cost, at: At<'_>) -> Cost {
        self.zip(other, |a, b| a.add(b, at))
    }

    pub fn times(&self, trips: &Trips, at: At<'_>) -> Cost {
        self.map(|a| a.times(trips, at))
    }

    pub fn times_unbounded(&self, reason: &Reason) -> Cost {
        self.map(|a| a.times_unbounded(reason.clone()))
    }

    pub fn join(&self, other: &Cost) -> Cost {
        self.zip(other, Amount::join)
    }

    pub fn subst(&self, map: &std::collections::BTreeMap<String, Poly>, at: At<'_>) -> Cost {
        self.map(|a| a.subst(map, at))
    }

    /// One measure as a contract names it. `bytes` is read plus written.
    pub fn metric(&self, m: CostMetric, at: At<'_>) -> Amount {
        match m {
            CostMetric::Flops => self.flops.clone(),
            CostMetric::BytesRead => self.bytes_read.clone(),
            CostMetric::BytesWritten => self.bytes_written.clone(),
            CostMetric::Bytes => self.bytes_read.add(&self.bytes_written, at),
            CostMetric::Allocations => self.allocations.clone(),
            CostMetric::AllocatedBytes => self.allocated_bytes.clone(),
        }
    }
}

/// One operation (or a call, or a whole loop) and what it contributes, for
/// saying which part of a function costs the most.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Site {
    /// The source of the operation.
    pub span: Span,
    /// The operation's name in the text format (`matmul`, `fused`, `call`).
    pub what: String,
    /// What it contributes to the function, loops around it included.
    pub cost: Cost,
}

/// The cost of one function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FunctionCost {
    pub name: String,
    pub span: Span,
    pub cost: Cost,
    /// The costly operations of the body, in source order.
    pub sites: Vec<Site>,
}

/// The cost of every function of a module, in module order.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Analysis {
    pub functions: Vec<FunctionCost>,
}

impl Analysis {
    pub fn get(&self, name: &str) -> Option<&FunctionCost> {
        self.functions.iter().find(|f| f.name == name)
    }
}

/// The names of the functions `f` calls, anywhere in its body.
fn callees(f: &Function) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut seen = vec![false; f.regions.len()];
    let mut todo = vec![f.body];
    while let Some(r) = todo.pop() {
        let Some(region) = f.regions.get(r.index()) else {
            continue;
        };
        if std::mem::replace(&mut seen[r.index()], true) {
            continue;
        }
        for b in &region.blocks {
            let Some(block) = f.blocks.get(b.index()) else {
                continue;
            };
            for o in &block.ops {
                let Some(op) = f.ops.get(o.index()) else {
                    continue;
                };
                if let OpKind::Call { callee } = &op.kind {
                    if !out.contains(&callee.as_str()) {
                        out.push(callee);
                    }
                }
                todo.extend(op.regions.iter().copied());
            }
        }
    }
    out
}

/// Computes the cost of every function of `module`, which must verify.
/// Run it under `sky_ast::with_deep_stack`, like the passes: it walks
/// nested regions recursively (the verifier bounds their depth).
pub fn analyze(module: &Module) -> Analysis {
    let graph: HashMap<&str, Vec<&str>> = module
        .functions
        .iter()
        .map(|f| (f.name.as_str(), callees(f)))
        .collect();
    // What each function reaches through one or more calls.
    let mut reaches: HashMap<&str, HashSet<&str>> = HashMap::new();
    for f in &module.functions {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut todo: Vec<&str> = graph[f.name.as_str()].clone();
        while let Some(g) = todo.pop() {
            if seen.insert(g) {
                if let Some(next) = graph.get(g) {
                    todo.extend(next.iter().copied());
                }
            }
        }
        reaches.insert(f.name.as_str(), seen);
    }
    let recursive = |from: &str, callee: &str| -> bool {
        from == callee || reaches.get(callee).is_some_and(|r| r.contains(from))
    };
    let mut done: HashMap<String, FunctionCost> = HashMap::new();
    let mut pending: Vec<&Function> = module.functions.iter().collect();
    while !pending.is_empty() {
        let before = pending.len();
        let mut rest = Vec::new();
        for f in pending {
            let ready = graph[f.name.as_str()]
                .iter()
                .all(|c| recursive(&f.name, c) || done.contains_key(*c) || !graph.contains_key(c));
            if ready {
                let cx = ops::Cx {
                    module,
                    f,
                    done: &done,
                    recursive: &recursive,
                };
                let cost = cx.function();
                done.insert(f.name.clone(), cost);
            } else {
                rest.push(f);
            }
        }
        pending = rest;
        // The call graph without its cycles is acyclic, so something is
        // always ready; a stall would be a bug, and must not loop forever.
        if pending.len() == before {
            break;
        }
    }
    let functions = module
        .functions
        .iter()
        .filter_map(|f| done.remove(&f.name))
        .collect();
    Analysis { functions }
}
