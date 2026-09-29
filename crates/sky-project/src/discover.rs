//! Finding the project root: walk up from the checked file to `sky.toml`.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sky_diag::{display_safe, Diagnostic, SourceFile, Span};

use crate::Identity;

/// The manifest file that marks a project root.
pub const MANIFEST: &str = "sky.toml";

/// Where imports for a file come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Project {
    /// A trusted `sky.toml` was found; project imports resolve under `root`.
    Rooted {
        /// Canonical directory containing `manifest`.
        root: PathBuf,
        /// `root/sky.toml`.
        manifest: PathBuf,
    },
    /// No `sky.toml` before a boundary: only stdlib imports resolve, and every
    /// project import is an error. The file's own directory is deliberately
    /// NOT used as a root, so a stray `.sky` file next to the checked one is
    /// never picked up implicitly.
    SingleFile,
}

/// The result of discovery for one checked file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Discovery {
    /// The checked file, canonicalized.
    pub file: PathBuf,
    pub project: Project,
}

/// Why discovery refused to continue. Every variant fails closed.
#[derive(Debug)]
pub enum DiscoverError {
    /// A filesystem error other than "not found" (e.g. permission denied).
    Io { path: PathBuf, error: io::Error },
    /// `sky.toml` is a symbolic link.
    ManifestSymlink { manifest: PathBuf },
    /// `sky.toml` exists but is not a regular file.
    ManifestNotFile { manifest: PathBuf },
    /// `path` (the manifest or its directory) is owned by someone else.
    ForeignOwner {
        manifest: PathBuf,
        path: PathBuf,
        owner: u32,
        euid: u32,
    },
    /// `path` (the manifest or its directory) is writable by every user: any
    /// user could replace `sky.toml`, or put one in the directory. The mode
    /// rules are those of the import walk (`check_dir`, `check_file`): a
    /// directory is refused unless the sticky bit is set, a file always.
    WorldWritable {
        manifest: PathBuf,
        path: PathBuf,
        mode: u32,
    },
    /// The checked file's path no longer names the file that was opened
    /// (it was replaced or renamed between the open and discovery).
    FileChanged { path: PathBuf },
}

/// Discovers the project for `file`.
///
/// Starts at the canonicalized directory of `file` and walks up looking for
/// `sky.toml`. At each directory `sky.toml` is checked first; then the walk
/// stops if the directory contains `.git` (directory, file, or anything
/// else), is the user's home directory, or is the filesystem root. So a
/// boundary directory's own `sky.toml` counts, but nothing above it does.
///
/// The current working directory is used only by the OS to interpret a
/// relative `file` argument; it is never a candidate root. No environment
/// variable is read.
pub fn discover(file: &Path, identity: &dyn Identity) -> Result<Discovery, DiscoverError> {
    let file = file.canonicalize().map_err(|error| DiscoverError::Io {
        path: file.to_path_buf(),
        error,
    })?;
    // A relative passwd home would be resolved against the cwd; ignore it.
    let home = identity
        .home_dir()
        .filter(|h| h.is_absolute())
        .map(|h| h.canonicalize().unwrap_or(h));
    let euid = identity.effective_uid();

    let mut dir = file.parent();
    while let Some(d) = dir {
        let manifest = d.join(MANIFEST);
        match fs::symlink_metadata(&manifest) {
            Ok(meta) => {
                check_manifest(&manifest, &meta, d, euid)?;
                return Ok(Discovery {
                    file: file.clone(),
                    project: Project::Rooted {
                        root: d.to_path_buf(),
                        manifest,
                    },
                });
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(DiscoverError::Io {
                    path: manifest,
                    error,
                })
            }
        }
        if is_boundary(d, home.as_deref()) {
            break;
        }
        dir = d.parent();
    }
    Ok(Discovery {
        file,
        project: Project::SingleFile,
    })
}

/// Like [`discover`], for a file already open as `opened`: also checks that
/// the canonical path found for `file` still names that same file (same
/// device and inode), so the project root belongs to the source that was
/// read, not to whatever was swapped in at `file` afterwards.
pub fn discover_opened(
    file: &Path,
    opened: &fs::File,
    identity: &dyn Identity,
) -> Result<Discovery, DiscoverError> {
    let discovery = discover(file, identity)?;
    let changed = || DiscoverError::FileChanged {
        path: file.to_path_buf(),
    };
    let io = |error| DiscoverError::Io {
        path: file.to_path_buf(),
        error,
    };
    let now = fs::metadata(&discovery.file).map_err(|_| changed())?;
    let then = opened.metadata().map_err(io)?;
    if same_file(&now, &then) {
        Ok(discovery)
    } else {
        Err(changed())
    }
}

#[cfg(unix)]
fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

/// No file identity off Unix (where rooted projects are refused anyway):
/// fail closed.
#[cfg(not(unix))]
fn same_file(_a: &fs::Metadata, _b: &fs::Metadata) -> bool {
    false
}

fn is_boundary(dir: &Path, home: Option<&Path>) -> bool {
    if home == Some(dir) {
        return true;
    }
    match fs::symlink_metadata(dir.join(".git")) {
        Ok(_) => true,
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        // Can't tell (e.g. permission denied): stop rather than climb further.
        Err(_) => true,
    }
}

fn check_manifest(
    manifest: &Path,
    meta: &fs::Metadata,
    dir: &Path,
    euid: u32,
) -> Result<(), DiscoverError> {
    if meta.file_type().is_symlink() {
        return Err(DiscoverError::ManifestSymlink {
            manifest: manifest.to_path_buf(),
        });
    }
    if !meta.is_file() {
        return Err(DiscoverError::ManifestNotFile {
            manifest: manifest.to_path_buf(),
        });
    }
    let dir_meta = fs::symlink_metadata(dir).map_err(|error| DiscoverError::Io {
        path: dir.to_path_buf(),
        error,
    })?;
    // The same mode rules as the import walk applies below the root, so a
    // root that discovery accepts is never one the walk then refuses: a
    // directory may not be world-writable unless sticky, a file never.
    for (path, m, is_dir) in [(manifest, meta, false), (dir, &dir_meta, true)] {
        let owner = owner_uid(m);
        if owner != Some(euid) {
            return Err(DiscoverError::ForeignOwner {
                manifest: manifest.to_path_buf(),
                path: path.to_path_buf(),
                owner: owner.unwrap_or(u32::MAX),
                euid,
            });
        }
        let mode = mode_bits(m);
        if mode & 0o002 != 0 && (!is_dir || mode & 0o1000 == 0) {
            return Err(DiscoverError::WorldWritable {
                manifest: manifest.to_path_buf(),
                path: path.to_path_buf(),
                mode,
            });
        }
    }
    Ok(())
}

#[cfg(unix)]
fn owner_uid(meta: &fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(meta.uid())
}

/// No ownership information off Unix: fail closed.
#[cfg(not(unix))]
fn owner_uid(_meta: &fs::Metadata) -> Option<u32> {
    None
}

#[cfg(unix)]
fn mode_bits(meta: &fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    meta.mode() & 0o7777
}

/// No mode bits off Unix; unreachable in practice, because [`owner_uid`]
/// already refused the path.
#[cfg(not(unix))]
fn mode_bits(_meta: &fs::Metadata) -> u32 {
    0
}

impl DiscoverError {
    /// The file the diagnostic is reported against. Its text is empty on
    /// purpose: a refused manifest is never read.
    pub fn source_file(&self) -> SourceFile {
        let path = match self {
            DiscoverError::Io { path, .. } | DiscoverError::FileChanged { path } => path,
            DiscoverError::ManifestSymlink { manifest }
            | DiscoverError::ManifestNotFile { manifest }
            | DiscoverError::ForeignOwner { manifest, .. }
            | DiscoverError::WorldWritable { manifest, .. } => manifest,
        };
        SourceFile::new(path.display().to_string(), "")
    }

    /// This refusal as an error diagnostic for [`DiscoverError::source_file`].
    pub fn to_diagnostic(&self) -> Diagnostic {
        let at = Span::point(0);
        match self {
            DiscoverError::Io { path, error } => {
                Diagnostic::error(format!("cannot inspect {}: {error}", path.display()), at)
            }
            DiscoverError::FileChanged { path } => Diagnostic::error(
                format!(
                    "refusing to check {}: the file was replaced while it was being opened",
                    path.display()
                ),
                at,
            )
            .with_help("run the check again once nothing is modifying the file"),
            DiscoverError::ManifestSymlink { manifest } => Diagnostic::error(
                format!("refusing to use {}: it is a symbolic link", manifest.display()),
                at,
            )
            .with_help("replace the link with a regular sky.toml file owned by you"),
            DiscoverError::ManifestNotFile { manifest } => Diagnostic::error(
                format!("refusing to use {}: it is not a regular file", manifest.display()),
                at,
            )
            .with_help("sky.toml must be a regular file"),
            DiscoverError::ForeignOwner {
                manifest,
                path,
                owner,
                euid,
            } => Diagnostic::error(
                format!(
                    "refusing to use project root: {} is owned by uid {owner}, not by you (uid {euid})",
                    path.display()
                ),
                at,
            )
            .with_note("a sky.toml owned by another user could redirect imports to code you did not write")
            .with_help(format!(
                "if you trust this project, have {} and the directory containing it owned by uid {euid}, or copy the project into a directory you own",
                manifest.display()
            )),
            DiscoverError::WorldWritable { path, mode, .. } => Diagnostic::error(
                format!(
                    "refusing to use project root: {} is writable by every user (mode {:04o})",
                    path.display(),
                    mode & 0o7777
                ),
                at,
            )
            .with_note("any user could replace a world-writable sky.toml, or add one to a world-writable directory without the sticky bit, and redirect imports to code you did not write")
            .with_help(format!("remove write permission for others: chmod o-w {}", path.display())),
        }
    }
}

/// The message with terminal-unsafe characters escaped, so a hostile path
/// can't reach the terminal raw through `{}` (like [`crate::SourceError`]).
impl fmt::Display for DiscoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&display_safe(&self.to_diagnostic().message))
    }
}

impl std::error::Error for DiscoverError {}
