//! Fuzz target for the SIR text parser (`sky_ir::parse`) and the printer and
//! verifier behind it.
//!
//! The input is SIR text; the invariants (no panic, error spans in bounds,
//! `parse(print(m)) == m`) are in `fuzz/harness/sir_parse.rs`, which a stable
//! sky-ir test also runs on the seed corpus and on mutations of it. The
//! parser, printer and verifier bound their own recursion, so the target
//! needs no big-stack worker thread.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../harness/sir_parse.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    harness::check(data);
});
