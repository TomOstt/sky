//! Fuzz target for the `sky.toml` manifest parser
//! (`sky_project::parse_manifest`).
//!
//! The input is the manifest's bytes. The invariants (no panic, every span in
//! bounds and on char boundaries, rendered errors free of terminal-unsafe
//! characters, bidi and invisible characters never accepted) are in
//! `fuzz/harness/manifest.rs`, which a stable sky-project test also runs on
//! the seed corpus.
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../harness/manifest.rs"]
mod harness;

fuzz_target!(|data: &[u8]| {
    let _ = harness::run(data);
});
