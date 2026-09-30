//! Keeping the instruction cache in step with freshly written machine code.
//!
//! Cranelift's JIT relies on `mprotect` to do it on Apple silicon (its
//! `clear_cache` is a no-op there), which is not enough when the memory was
//! code before, as it is when a module's memory is freed and reused by the
//! next one: the old instructions can still be fetched, and the fuzz
//! campaigns crashed (`SIGBUS`, a jump to a garbage address) within minutes
//! on eight threads compiling and freeing modules. Calling [`flush`] on the
//! code after `finalize_definitions` does what the architecture asks: clean
//! the data cache, invalidate the instruction cache on every core, and
//! synchronise this one. Every module that finalizes code and later frees it
//! does this.

#![allow(unsafe_code)]

/// Makes the instruction cache see the code just written at `ptr..ptr+len`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn flush(ptr: *const u8, len: usize) {
    extern "C" {
        fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
    }
    if len == 0 {
        return;
    }
    // SAFETY: `sys_icache_invalidate` (libSystem) only maintains caches for
    // the range, which is code Cranelift just finalized, in memory this
    // process owns; it neither reads nor writes through the pointer.
    unsafe { sys_icache_invalidate(ptr as *mut std::ffi::c_void, len) };
}

/// Other targets: Cranelift's own handling is what those need.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn flush(_ptr: *const u8, _len: usize) {}
