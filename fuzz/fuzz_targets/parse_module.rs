//! Fuzz target for the Sky parser (`sky_ast::parse_module`).
//!
//! The first byte picks the mode: even means the rest is source text, lexed
//! with the real lexer and then parsed; odd means the rest is decoded into
//! an arbitrary token sequence (any order, overlapping spans, `Eof` anywhere
//! or nowhere) that goes straight to the parser. The invariants (no panic,
//! every AST and diagnostic span in bounds, char boundaries for lexed
//! source) are in `fuzz/harness/parse_module.rs`, which a stable sky-ast
//! test also runs on the seed corpus through `sky_ast::with_deep_stack`.
//!
//! Crashes: libfuzzer-sys's panic hook is process-wide, so a panic on the
//! worker aborts the process like one on the fuzzer thread. As a fallback
//! (if that hook were replaced) the worker catches the panic and it is
//! re-raised here with `resume_unwind`, which aborts at libfuzzer-sys's
//! `extern "C"` boundary; a worker that stops answering aborts too. A stack
//! overflow on the worker hits its guard page: ASan (the CI build) reports
//! it as a crash, and without a sanitizer the process dies with SIGSEGV,
//! exactly as it would on `with_deep_stack`'s thread.
#![no_main]

use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread;

use libfuzzer_sys::fuzz_target;

#[path = "../harness/parse_module.rs"]
mod harness;

struct Worker {
    inputs: Sender<Vec<u8>>,
    results: Receiver<thread::Result<()>>,
}

// One reused thread with with_deep_stack's DEEP_STACK_BYTES: same stack, no per-input spawn.
fn worker() -> &'static Mutex<Worker> {
    static WORKER: OnceLock<Mutex<Worker>> = OnceLock::new();
    WORKER.get_or_init(|| {
        let (inputs, input_rx) = mpsc::channel::<Vec<u8>>();
        let (result_tx, results) = mpsc::channel();
        thread::Builder::new()
            .name("sky-fuzz-worker".to_string())
            .stack_size(sky_ast::DEEP_STACK_BYTES)
            .spawn(move || {
                for data in input_rx {
                    let result = panic::catch_unwind(AssertUnwindSafe(|| {
                        harness::check(&data);
                    }));
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .unwrap_or_else(|e| panic!("failed to spawn the fuzz worker thread: {e}"));
        Mutex::new(Worker { inputs, results })
    })
}

fuzz_target!(|data: &[u8]| {
    let worker = worker().lock().unwrap_or_else(PoisonError::into_inner);
    if worker.inputs.send(data.to_vec()).is_err() {
        std::process::abort();
    }
    match worker.results.recv() {
        Ok(Ok(())) => {}
        Ok(Err(payload)) => {
            drop(worker);
            panic::resume_unwind(payload);
        }
        Err(_) => std::process::abort(),
    }
});
