//! Fuzz target for the Sky parser (`sky_ast::parse_module`).
//!
//! The first byte picks the mode: even means the rest is source text, lexed
//! with the real lexer and then parsed; odd means the rest is decoded into
//! an arbitrary token sequence (any order, overlapping spans, `Eof` anywhere
//! or nowhere) that goes straight to the parser. Every input runs inside
//! `sky_ast::with_deep_stack`. The invariants (no panic, every AST and
//! diagnostic span in bounds, char boundaries for lexed source) are in
//! `fuzz/harness/parse_module.rs`, which a stable sky-ast test also runs on
//! the seed corpus.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../harness/parse_module.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    harness::run(data);
});
