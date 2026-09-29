//! Finding the project root: walk up from the checked file to `sky.toml`.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sky_diag::{Diagnostic, SourceFile, Span};

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
    for (path, m) in [(manifest, meta), (dir, &dir_meta)] {
        let owner = owner_uid(m);
        if owner != Some(euid) {
            return Err(DiscoverError::ForeignOwner {
                manifest: manifest.to_path_buf(),
                path: path.to_path_buf(),
                owner: owner.unwrap_or(u32::MAX),
                euid,
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

impl DiscoverError {
    /// The file the diagnostic is reported against. Its text is empty on
    /// purpose: a refused manifest is never read.
    pub fn source_file(&self) -> SourceFile {
        let path = match self {
            DiscoverError::Io { path, .. } => path,
            DiscoverError::ManifestSymlink { manifest }
            | DiscoverError::ManifestNotFile { manifest }
            | DiscoverError::ForeignOwner { manifest, .. } => manifest,
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
        }
    }
}

impl fmt::Display for DiscoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_diagnostic().message)
    }
}

impl std::error::Error for DiscoverError {}
