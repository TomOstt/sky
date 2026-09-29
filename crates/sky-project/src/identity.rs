//! Who the process runs as, and where that user's home directory is.
//!
//! Injectable so tests can simulate a foreign owner without root.

use std::path::PathBuf;

/// The identity facts project discovery depends on.
pub trait Identity {
    /// The effective uid of this process. A `sky.toml`, or the directory
    /// holding it, must be owned by this uid.
    fn effective_uid(&self) -> u32;

    /// The effective user's home directory, which is a discovery boundary.
    /// `None` means there is no home boundary; `.git` and the filesystem root
    /// still stop the walk.
    fn home_dir(&self) -> Option<PathBuf>;
}

/// The real process identity. The home directory comes from the passwd
/// database (`getpwuid_r`), never from `$HOME` or any other environment
/// variable.
#[derive(Copy, Clone, Debug, Default)]
pub struct SystemIdentity;

#[cfg(unix)]
impl Identity for SystemIdentity {
    fn effective_uid(&self) -> u32 {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }

    fn home_dir(&self) -> Option<PathBuf> {
        passwd_home(self.effective_uid())
    }
}

/// Fails closed off Unix: no file is ever treated as owned by us, so any
/// `sky.toml` is refused, and there is no home boundary.
#[cfg(not(unix))]
impl Identity for SystemIdentity {
    fn effective_uid(&self) -> u32 {
        u32::MAX
    }

    fn home_dir(&self) -> Option<PathBuf> {
        None
    }
}

#[cfg(unix)]
fn passwd_home(uid: u32) -> Option<PathBuf> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;

    // SAFETY: sysconf has no preconditions; -1 means "no limit / unknown".
    let hint = unsafe { libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) };
    let mut len = usize::try_from(hint)
        .ok()
        .filter(|&n| n > 0)
        .unwrap_or(1024);
    loop {
        let mut buf: Vec<libc::c_char> = vec![0; len];
        // SAFETY: passwd is a plain C struct; all-zero is a valid value for
        // getpwuid_r to overwrite.
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::passwd = std::ptr::null_mut();
        // SAFETY: every pointer is valid for the duration of the call and
        // `buf.len()` is the real length of `buf`.
        let rc =
            unsafe { libc::getpwuid_r(uid, &mut pwd, buf.as_mut_ptr(), buf.len(), &mut result) };
        if rc == libc::ERANGE && len < (1 << 20) {
            len *= 2;
            continue;
        }
        if rc != 0 || result.is_null() || pwd.pw_dir.is_null() {
            return None;
        }
        // SAFETY: on success pw_dir points at a NUL-terminated string inside
        // `buf`, which is still alive here; we copy it out before returning.
        let dir = unsafe { CStr::from_ptr(pwd.pw_dir) }.to_bytes();
        if dir.is_empty() {
            return None;
        }
        return Some(PathBuf::from(OsStr::from_bytes(dir)));
    }
}
