//! Fuzz target: a grammar-based differential fuzzer for Sky's two engines.
//!
//! The input bytes are a choice stream for a generator of valid, well-typed
//! Sky programs (units and prefixes, constants, arithmetic, `^` with
//! rational exponents, math functions, comparisons, `and`/`or`/`not`,
//! `if`/`elif`/`else`, bounded `while` with `break`/`continue`, `def` with
//! annotations, bounded recursion, early returns, `print`, and inputs that
//! fail at run time). Every program is lexed, parsed and checked (a
//! rejection is a generator bug), lowered to SIR and verified, then run by
//! the AST interpreter (`sky run`) and by the SIR interpreter. The check
//! panics unless both print the same stdout and stop with the same error,
//! compared as the whole rendered diagnostic (spans, labels and notes, not
//! only the message; or both succeed), and unless the SIR text format
//! round-trips
//! (`parse(print(m)) == canonicalize(m)`) and the re-parsed module runs the
//! same. The generator and the oracle are in `fuzz/harness/differential.rs`,
//! which a stable sky-ir-interp test also runs, on fixed seeds.
//!
//! The oracle runs the engines under `sky_ast::with_deep_stack`, as `sky
//! run` does, so this target needs no worker thread of its own. Bugs already
//! reported are listed (and, where the generator can steer around them,
//! gated) at the top of the harness, so a crash here is a new one.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../harness/differential.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    harness::check(data);
});
