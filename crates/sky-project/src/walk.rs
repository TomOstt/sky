//! Handle-based filesystem primitives for import resolution (Unix only).
//!
//! Every lookup below the project root is an `openat` relative to a
//! directory handle we already hold and have checked, with `O_NOFOLLOW`, so
//! no path is checked by name and then reopened, and no symbolic link is
//! ever followed.

use std::ffi::CString;
use std::fs::{File, Metadata, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

use crate::import::Facts;

/// Opens the project root: `O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`.
pub(crate) fn open_root(root: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(root)
}

/// Opens the subdirectory `name` of `dir`:
/// `openat(dir, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)`.
pub(crate) fn open_dir_at(dir: &File, name: &str) -> io::Result<File> {
    open_at(dir, name, libc::O_RDONLY | libc::O_DIRECTORY)
}

/// Opens the module file `name` in `dir`:
/// `openat(dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK | O_NOCTTY)`.
///
/// `O_NONBLOCK` keeps a FIFO from blocking the open (the caller then
/// rejects it by `fstat`); it has no effect on reads from a regular file.
/// `O_NOCTTY` keeps a terminal device from becoming the controlling
/// terminal.
pub(crate) fn open_file_at(dir: &File, name: &str) -> io::Result<File> {
    open_at(
        dir,
        name,
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOCTTY,
    )
}

fn open_at(dir: &File, name: &str, flags: libc::c_int) -> io::Result<File> {
    let name = component(name);
    // SAFETY: `dir` is an open descriptor for the duration of the call and
    // `name` is a NUL-terminated string. No O_CREAT, so no mode argument.
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a fresh descriptor that nothing else owns.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// What `name` in `dir` is, without following a symbolic link
/// (`fstatat(..., AT_SYMLINK_NOFOLLOW)`). Only used to explain a failed
/// open or to test existence; never to decide what to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Symlink,
    Directory,
    Other,
}

pub(crate) fn kind_at(dir: &File, name: &str) -> io::Result<Kind> {
    let name = component(name);
    let mut st = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dir` is open, `name` is NUL-terminated, and `st` is valid
    // for writes of one `stat`.
    let rc = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            name.as_ptr(),
            st.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fstatat succeeded, so it initialized `st`.
    let fmt = unsafe { st.assume_init() }.st_mode & libc::S_IFMT;
    Ok(if fmt == libc::S_IFLNK {
        Kind::Symlink
    } else if fmt == libc::S_IFDIR {
        Kind::Directory
    } else {
        Kind::Other
    })
}

/// A single path component. Module segments are ASCII identifiers (checked
/// by `parse_module_path`), so this never fails; it is asserted anyway so
/// that `openat` can never be handed a separator, `.`, `..`, or a NUL.
fn component(name: &str) -> CString {
    assert!(
        !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\0']),
        "not a single path component: {name:?}"
    );
    CString::new(name).expect("no NUL")
}

/// The facts the per-handle rule looks at, from `fstat` of a held handle.
pub(crate) fn facts(meta: &Metadata) -> Facts {
    Facts {
        is_dir: meta.is_dir(),
        is_file: meta.is_file(),
        uid: meta.uid(),
        mode: meta.mode(),
    }
}
