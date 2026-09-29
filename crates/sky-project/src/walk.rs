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

use crate::import::{Facts, ImportError};

/// Opens the project root: `O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`.
pub(crate) fn open_root(root: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(root)
}

/// Opens the subdirectory `name` of `dir`:
/// `openat(dir, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)`.
pub(crate) fn open_dir_at(dir: &File, name: &Component) -> io::Result<File> {
    open_at(dir, name, libc::O_RDONLY | libc::O_DIRECTORY)
}

/// Opens the module file `name` in `dir`:
/// `openat(dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK | O_NOCTTY)`.
///
/// `O_NONBLOCK` keeps a FIFO from blocking the open (the caller then
/// rejects it by `fstat`); it has no effect on reads from a regular file.
/// `O_NOCTTY` keeps a terminal device from becoming the controlling
/// terminal.
pub(crate) fn open_file_at(dir: &File, name: &Component) -> io::Result<File> {
    open_at(
        dir,
        name,
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOCTTY,
    )
}

fn open_at(dir: &File, name: &Component, flags: libc::c_int) -> io::Result<File> {
    // SAFETY: `dir` is an open descriptor for the duration of the call and
    // `name` is a NUL-terminated string. No O_CREAT, so no mode argument.
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.c.as_ptr(),
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

pub(crate) fn kind_at(dir: &File, name: &Component) -> io::Result<Kind> {
    let mut st = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dir` is open, `name` is NUL-terminated, and `st` is valid
    // for writes of one `stat`.
    let rc = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            name.c.as_ptr(),
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

/// A name that is a single path component: not empty, `.`, or `..`, and
/// with no `/` or NUL. The walk functions take only this, so `openat` and
/// `fstatat` can never be handed a separator or a parent reference.
#[derive(Debug)]
pub(crate) struct Component<'a> {
    name: &'a str,
    c: CString,
}

impl Component<'_> {
    pub(crate) fn name(&self) -> &str {
        self.name
    }
}

/// Validates `name`, a component of the import `module`. Module segments are
/// ASCII identifiers (checked by `parse_module_path`), so this does not fail
/// for a parsed import; it is checked anyway, as an error rather than a
/// panic, so a future caller can't get past it.
pub(crate) fn component<'a>(module: &str, name: &'a str) -> Result<Component<'a>, ImportError> {
    let bad = || ImportError::InvalidComponent {
        module: module.to_string(),
        component: name.to_string(),
    };
    if name.is_empty() || name == "." || name == ".." || name.contains('/') {
        return Err(bad());
    }
    let c = CString::new(name).map_err(|_| bad())?;
    Ok(Component { name, c })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_components_are_an_error() {
        for name in ["..", ".", "", "a/b", "a\0b", "/", "a/", "\0"] {
            match component("m", name) {
                Err(ImportError::InvalidComponent { module, component }) => {
                    assert_eq!(module, "m");
                    assert_eq!(component, name);
                }
                other => panic!("{name:?}: expected InvalidComponent, got {other:?}"),
            }
        }
        for name in ["a", "_a1", "kinematics.sky", "..a", "a.."] {
            assert_eq!(component("m", name).unwrap().name(), name);
        }
    }

    #[test]
    fn invalid_component_diagnostic_is_display_safe() {
        let err = component("m", "a\0b\u{1b}[2J").unwrap_err();
        let msg = err.to_string();
        assert_eq!(
            msg,
            "refusing to import `m`: `a\\u{0000}b\\u{001B}[2J` is not a single path component"
        );
        let rendered = err
            .to_diagnostic(sky_diag::Span::new(7, 8))
            .render_to_string(&sky_diag::SourceFile::new("main.sky", "import m\n"));
        assert!(
            rendered.contains("`a\\u{0000}b\\u{001B}[2J`"),
            "{rendered:?}"
        );
        assert!(!rendered.contains(['\0', '\u{1b}']), "{rendered:?}");
    }
}
