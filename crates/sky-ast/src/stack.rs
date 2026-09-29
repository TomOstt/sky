//! Running the front end on a stack of known size.
//!
//! The parser bounds its own recursion with [`crate::MAX_DEPTH`], and later
//! passes (checker, interpreter) recurse over trees whose height that bounds.
//! Those bounds only mean something on a stack of known size: the main
//! thread's is set by the OS (often 8 MiB, sometimes 1 MiB), a spawned
//! thread's defaults to 2 MiB, and debug builds use several times more stack
//! per level than release builds. [`with_deep_stack`] removes the guesswork
//! by running the work on a fresh thread with [`DEEP_STACK_BYTES`] of stack.
//!
//! **Every entry point that parses, checks, or interprets Sky code must go
//! through [`with_deep_stack`]:** `sky check`, `sky run`, `sky repl` (per
//! input), and the fuzz harness. Library callers that parse directly get the
//! same guarantee by wrapping their call.

use std::thread;

/// Stack size for [`with_deep_stack`]: 8 MiB. Parsing [`crate::MAX_DEPTH`]
/// levels needs well under 1 MiB even in a debug build, so this leaves room
/// for the passes that run after the parser.
pub const DEEP_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Runs `f` on a new thread named `sky-deep-stack` with a
/// [`DEEP_STACK_BYTES`] stack, waits for it, and returns its result.
///
/// The thread is scoped, so `f` may borrow from the caller (source text,
/// tokens, a module). `f` and its result must be `Send`: state built on
/// `Rc` or `RefCell` (an interpreter environment, say) has to be created
/// inside the closure, not captured.
///
/// # Panics
///
/// If `f` panics, the panic is resumed on the calling thread with the
/// original payload, so `catch_unwind` and test harnesses see the same
/// message as if `f` had run inline.
///
/// If the thread cannot be spawned (the OS refused the stack or a thread),
/// this panics with "failed to spawn 8 MiB parser thread: ...". It does not
/// fall back to running `f` inline: on a stack of unknown size, deep but
/// legal input could overflow it and abort the process, which silently
/// breaks the guarantee this function exists for.
pub fn with_deep_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    thread::scope(|scope| {
        let handle = thread::Builder::new()
            .name("sky-deep-stack".to_string())
            .stack_size(DEEP_STACK_BYTES)
            .spawn_scoped(scope, f)
            .unwrap_or_else(|e| panic!("failed to spawn 8 MiB parser thread: {e}"));
        match handle.join() {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}
