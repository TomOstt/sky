//! Reading a Sky source file from an open handle, with a size limit.
//!
//! Both the file `sky check` is given and every imported module are read
//! here, so they get the same rules: the handle must be a regular file, and
//! at most [`MAX_SOURCE_BYTES`] are read from it.

use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sky_diag::{display_safe, Diagnostic, SourceFile, Span};
use sky_syntax::MAX_SOURCE_BYTES;

// `sky_diag::Span` holds `u32` byte offsets.
const _: () = assert!(MAX_SOURCE_BYTES < u32::MAX as usize);

/// Why a source file could not be read.
#[derive(Debug)]
pub enum SourceError {
    /// Opening, inspecting, or reading the file failed.
    Io { path: PathBuf, error: io::Error },
    /// The handle is not a regular file (a directory, FIFO, device, ...).
    NotAFile { path: PathBuf },
    /// The file is larger than `limit` bytes. `size` is its size from
    /// `fstat`, or `None` if `fstat` was within the limit but more than
    /// `limit` bytes could be read (the file grew).
    TooLarge {
        path: PathBuf,
        size: Option<u64>,
        limit: u64,
    },
    /// The contents are not UTF-8.
    NotUtf8 { path: PathBuf },
}

/// Opens and reads the file `sky check` (or `sky run`) was given.
///
/// On Unix the open uses `O_NONBLOCK | O_NOCTTY`, so a FIFO or terminal
/// device can't block or become the controlling terminal before it is
/// rejected. Symbolic links are followed: the user named this file. The
/// contents come from [`read_source`] on the same handle.
pub fn read_entry_file(path: &Path) -> Result<String, SourceError> {
    let file = open_entry(path).map_err(|error| SourceError::Io {
        path: path.to_path_buf(),
        error,
    })?;
    read_source(&file, path)
}

#[cfg(unix)]
fn open_entry(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(unix))]
fn open_entry(path: &Path) -> io::Result<File> {
    File::open(path)
}

/// Reads a whole source file from `file`, whose path is `path` (used only
/// in errors).
///
/// `fstat`s the handle first: it must be a regular file of at most
/// [`MAX_SOURCE_BYTES`], so an oversized file is refused without reading
/// or allocating. The read itself is also capped at the limit, because the
/// file can grow after the `fstat`. The contents must be UTF-8.
pub fn read_source(file: &File, path: &Path) -> Result<String, SourceError> {
    read_source_limited(file, path, MAX_SOURCE_BYTES as u64)
}

pub(crate) fn read_source_limited(
    file: &File,
    path: &Path,
    limit: u64,
) -> Result<String, SourceError> {
    let io_err = |error| SourceError::Io {
        path: path.to_path_buf(),
        error,
    };
    let meta = file.metadata().map_err(io_err)?;
    if !meta.is_file() {
        return Err(SourceError::NotAFile {
            path: path.to_path_buf(),
        });
    }
    read_capped(file, meta.len(), path, limit)
}

/// Reads at most `limit` bytes from `r`, refusing if the declared size or
/// the bytes actually available exceed it.
pub(crate) fn read_capped(
    r: impl Read,
    declared: u64,
    path: &Path,
    limit: u64,
) -> Result<String, SourceError> {
    let too_large = |size| SourceError::TooLarge {
        path: path.to_path_buf(),
        size,
        limit,
    };
    if declared > limit {
        return Err(too_large(Some(declared)));
    }
    // `declared <= limit`, so this allocation is bounded by the limit.
    let mut bytes = Vec::with_capacity(declared as usize);
    r.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| SourceError::Io {
            path: path.to_path_buf(),
            error,
        })?;
    if bytes.len() as u64 > limit {
        return Err(too_large(None));
    }
    String::from_utf8(bytes).map_err(|_| SourceError::NotUtf8 {
        path: path.to_path_buf(),
    })
}

impl SourceError {
    /// The path of the file that could not be read.
    pub fn path(&self) -> &Path {
        match self {
            SourceError::Io { path, .. }
            | SourceError::NotAFile { path }
            | SourceError::TooLarge { path, .. }
            | SourceError::NotUtf8 { path } => path,
        }
    }

    /// The message, unescaped. The renderer escapes it; [`fmt::Display`]
    /// escapes it too.
    fn message(&self) -> String {
        match self {
            SourceError::Io { path, error } => format!("cannot read {}: {error}", path.display()),
            SourceError::NotAFile { path } => {
                format!(
                    "refusing to read {}: it is not a regular file",
                    path.display()
                )
            }
            SourceError::TooLarge {
                path,
                size: Some(size),
                limit,
            } => format!(
                "refusing to read {}: it is {size} bytes, over the limit of {limit} bytes ({})",
                path.display(),
                human(*limit)
            ),
            SourceError::TooLarge {
                path,
                size: None,
                limit,
            } => format!(
                "refusing to read {}: it is over the limit of {limit} bytes ({})",
                path.display(),
                human(*limit)
            ),
            SourceError::NotUtf8 { path } => {
                format!("cannot read {}: it is not valid UTF-8", path.display())
            }
        }
    }

    /// The file the diagnostic is reported against. Its text is empty on
    /// purpose: the file was not (fully) read.
    pub fn source_file(&self) -> SourceFile {
        SourceFile::new(self.path().display().to_string(), "")
    }

    /// This error as a diagnostic at `span`: for an imported module, the
    /// import's module path in the importing file; for the entry file,
    /// `Span::point(0)` in [`SourceError::source_file`].
    pub fn to_diagnostic(&self, span: Span) -> Diagnostic {
        let d = Diagnostic::error(self.message(), span);
        match self {
            SourceError::TooLarge { limit, .. } => d.with_note(format!(
                "Sky reads at most {} from a source file",
                human(*limit)
            )),
            _ => d,
        }
    }
}

/// `limit` in MiB when it is a whole number of MiB, else in bytes.
fn human(limit: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if limit % MIB == 0 {
        format!("{} MiB", limit / MIB)
    } else {
        format!("{limit} bytes")
    }
}

/// The message with terminal-unsafe characters escaped, so a hostile file
/// name can't reach the terminal raw through `{}`.
impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&display_safe(&self.message()))
    }
}

impl std::error::Error for SourceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn p() -> &'static Path {
        Path::new("m.sky")
    }

    #[test]
    fn exactly_the_limit_is_read() {
        assert_eq!(
            read_capped(Cursor::new("12345678"), 8, p(), 8).unwrap(),
            "12345678"
        );
    }

    #[test]
    fn declared_size_over_the_limit_is_refused_before_reading() {
        struct Unreadable;
        impl Read for Unreadable {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                panic!("must not read");
            }
        }
        assert!(matches!(
            read_capped(Unreadable, 9, p(), 8),
            Err(SourceError::TooLarge {
                size: Some(9),
                limit: 8,
                ..
            })
        ));
    }

    #[test]
    fn a_file_that_grew_after_fstat_is_refused_by_the_read_cap() {
        // fstat said 4 bytes, but 9 are there by the time we read.
        assert!(matches!(
            read_capped(Cursor::new("123456789"), 4, p(), 8),
            Err(SourceError::TooLarge {
                size: None,
                limit: 8,
                ..
            })
        ));
    }

    #[test]
    fn invalid_utf8_is_refused() {
        assert!(matches!(
            read_capped(Cursor::new(b"\xff"), 1, p(), 8),
            Err(SourceError::NotUtf8 { .. })
        ));
    }

    #[test]
    fn limit_message_names_the_file_and_the_limit() {
        let e = SourceError::TooLarge {
            path: "big\u{1b}[2J.sky".into(),
            size: Some(MAX_SOURCE_BYTES as u64 + 1),
            limit: MAX_SOURCE_BYTES as u64,
        };
        assert_eq!(
            e.to_string(),
            "refusing to read big\\u{001B}[2J.sky: it is 16777217 bytes, over the limit of 16777216 bytes (16 MiB)"
        );
    }
}
