//! Mapping a dotted module path to a stdlib module or a file under the root.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use sky_diag::{display_safe, Diagnostic, Span};

use crate::source::{read_source, SourceError};
use crate::{Identity, Project};

/// What an import resolved to.
#[derive(Debug)]
pub enum Module {
    /// A standard-library module, by its dotted name.
    Stdlib(String),
    /// A project module, already opened.
    Project(ProjectModule),
}

/// A project module file, opened and checked during resolution.
///
/// The loader reads the module from [`ProjectModule::file`] (or
/// [`ProjectModule::read_source`]) and never reopens it by name, so what it
/// reads is exactly the file that passed the checks, even if the path is
/// swapped afterwards.
#[derive(Debug)]
pub struct ProjectModule {
    path: PathBuf,
    file: File,
}

impl ProjectModule {
    /// `<root>/a/b/c.sky`: the path the module was opened by, for
    /// diagnostics only. Do not reopen it.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The open module file: a regular file owned by the effective uid. On
    /// Unix it was opened with `O_NONBLOCK`, which has no effect on reads
    /// from a regular file.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Reads the module's source from the held handle, with the regular-file
    /// check and [`sky_syntax::MAX_SOURCE_BYTES`] cap of [`read_source`].
    pub fn read_source(&self) -> Result<String, SourceError> {
        read_source(&self.file, &self.path)
    }

    /// The path and the handle.
    pub fn into_parts(self) -> (PathBuf, File) {
        (self.path, self.file)
    }
}

/// Why an import did not resolve.
#[derive(Debug)]
pub enum ImportError {
    /// The module path is empty.
    Empty,
    /// The path looks absolute (`/a`, `\a`, `C:a`).
    Absolute { module: String },
    /// The path contains `..` (relative import or parent traversal).
    ParentTraversal { module: String },
    /// A segment is empty (`a..b` is reported as `ParentTraversal`; this is
    /// `.a` or `a.`).
    EmptySegment { module: String },
    /// A segment is not an ASCII identifier `[A-Za-z_][A-Za-z0-9_]*`.
    InvalidSegment { module: String, segment: String },
    /// The first segment names a stdlib module, but the path is not one.
    UnknownStdlib { module: String },
    /// The project has `<root>/<name>.sky` or `<root>/<name>/`, where `name` is
    /// a reserved stdlib top-level name.
    ShadowsStdlib { name: String, path: PathBuf },
    /// Single-file mode (no `sky.toml`): project imports are not allowed.
    NoProject { module: String },
    /// `path` (a directory on the way, or the module file) does not exist.
    NotFound { module: String, path: PathBuf },
    /// `path` is on the way to the module but is not a directory.
    NotADirectory { module: String, path: PathBuf },
    /// The module path exists but is not a regular file.
    NotAFile { module: String, path: PathBuf },
    /// `path` (the root, a directory on the way, or the module file) is a
    /// symbolic link. Resolution never follows links.
    Symlink { module: String, path: PathBuf },
    /// `path` (the root, a directory on the way, or the module file) is
    /// owned by a uid other than the effective uid.
    ForeignOwner {
        module: String,
        path: PathBuf,
        owner: u32,
        euid: u32,
    },
    /// `path` (the root or a directory on the way) is world-writable and
    /// does not have the sticky bit.
    WorldWritable {
        module: String,
        path: PathBuf,
        mode: u32,
    },
    /// The module file is world-writable (the sticky bit is irrelevant for
    /// files).
    WorldWritableFile {
        module: String,
        path: PathBuf,
        mode: u32,
    },
    /// Project imports need Unix handle-relative opens; elsewhere they are
    /// refused (fail closed).
    Unsupported { module: String },
    /// A filesystem error other than the above.
    Io { path: PathBuf, error: io::Error },
}

/// Resolves the dotted module path `module` (as written in an import).
///
/// - Syntax first: rejects empty paths, absolute paths, `..`, empty segments,
///   and segments that are not ASCII identifiers.
/// - Stdlib names are reserved: if the first segment is the first segment of
///   any entry in `stdlib`, the import is a stdlib import. It resolves only if
///   `module` is in `stdlib`, and it is an error if the project contains
///   `<root>/<first>.sky` or `<root>/<first>/` (a project module may not shadow
///   the stdlib).
/// - Otherwise `a.b.c` is `<root>/a/b/c.sky`, opened by a handle-relative
///   walk: the root is opened with `O_DIRECTORY | O_NOFOLLOW`, then each
///   directory with `openat(parent, name, O_DIRECTORY | O_NOFOLLOW)`, then
///   the file with `openat(dir, "c.sky", O_RDONLY | O_NOFOLLOW | O_NONBLOCK)`.
///   Every handle is `fstat`ed as soon as it is opened. The root and each
///   directory must be a directory owned by `identity`'s effective uid and
///   must not be world-writable unless the sticky bit is set; the file must
///   be a regular file owned by the effective uid and not world-writable. No symbolic link is
///   followed anywhere below the root, so nothing can resolve outside it.
/// - In [`Project::SingleFile`] mode every project import is an error.
///
/// The result holds the open module file; read it from the handle.
pub fn resolve_import(
    module: &str,
    project: &Project,
    stdlib: &[&str],
    identity: &dyn Identity,
) -> Result<Module, ImportError> {
    resolve_with_hook(module, project, stdlib, identity, &mut |_| {})
}

/// [`resolve_import`], calling `after_dir(path)` each time a directory handle
/// (the root, then each directory on the way) has been opened and checked,
/// before the next `openat`. Tests use it to change the filesystem at a
/// precise point; in `resolve_import` it does nothing.
pub(crate) fn resolve_with_hook(
    module: &str,
    project: &Project,
    stdlib: &[&str],
    identity: &dyn Identity,
    after_dir: &mut dyn FnMut(&Path),
) -> Result<Module, ImportError> {
    let segments = parse_module_path(module)?;
    let top = segments[0];
    let is_stdlib = stdlib.iter().any(|m| m.split('.').next() == Some(top));
    let stdlib_result = || {
        if stdlib.contains(&module) {
            Ok(Module::Stdlib(module.to_string()))
        } else {
            Err(ImportError::UnknownStdlib {
                module: module.to_string(),
            })
        }
    };

    let root = match project {
        Project::Rooted { root, .. } => root,
        Project::SingleFile if is_stdlib => return stdlib_result(),
        Project::SingleFile => {
            return Err(ImportError::NoProject {
                module: module.to_string(),
            })
        }
    };
    let found = rooted::resolve(
        module,
        &segments,
        is_stdlib,
        root,
        identity.effective_uid(),
        after_dir,
    )?;
    match found {
        Some(m) => Ok(Module::Project(m)),
        None => stdlib_result(),
    }
}

/// What `fstat` of a held handle says, for the per-handle rule.
#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Facts {
    pub is_dir: bool,
    pub is_file: bool,
    pub uid: u32,
    pub mode: u32,
}

/// Why a handle failed the per-handle rule.
#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    NotADirectory,
    NotAFile,
    ForeignOwner { owner: u32 },
    WorldWritable { mode: u32 },
    WorldWritableFile { mode: u32 },
}

/// The rule for the root and every directory on the way to a module: a
/// directory, owned by `euid`, and not world-writable unless sticky.
#[cfg(unix)]
pub(crate) fn check_dir(f: Facts, euid: u32) -> Result<(), Refusal> {
    if !f.is_dir {
        return Err(Refusal::NotADirectory);
    }
    if f.uid != euid {
        return Err(Refusal::ForeignOwner { owner: f.uid });
    }
    if f.mode & 0o002 != 0 && f.mode & 0o1000 == 0 {
        return Err(Refusal::WorldWritable { mode: f.mode });
    }
    Ok(())
}

/// The rule for the module file: a regular file, owned by `euid`, and not
/// world-writable (the sticky bit means nothing for a file). Group-writable
/// is allowed, as for directories.
#[cfg(unix)]
pub(crate) fn check_file(f: Facts, euid: u32) -> Result<(), Refusal> {
    if !f.is_file {
        return Err(Refusal::NotAFile);
    }
    if f.uid != euid {
        return Err(Refusal::ForeignOwner { owner: f.uid });
    }
    if f.mode & 0o002 != 0 {
        return Err(Refusal::WorldWritableFile { mode: f.mode });
    }
    Ok(())
}

#[cfg(unix)]
impl Refusal {
    fn into_error(self, module: &str, path: &Path, euid: u32) -> ImportError {
        let (module, path) = (module.to_string(), path.to_path_buf());
        match self {
            Refusal::NotADirectory => ImportError::NotADirectory { module, path },
            Refusal::NotAFile => ImportError::NotAFile { module, path },
            Refusal::ForeignOwner { owner } => ImportError::ForeignOwner {
                module,
                path,
                owner,
                euid,
            },
            Refusal::WorldWritable { mode } => ImportError::WorldWritable { module, path, mode },
            Refusal::WorldWritableFile { mode } => {
                ImportError::WorldWritableFile { module, path, mode }
            }
        }
    }
}

#[cfg(unix)]
mod rooted {
    use std::fs::File;
    use std::io;
    use std::path::Path;

    use super::{check_dir, check_file, ImportError, ProjectModule};
    use crate::walk::{self, Kind};

    /// Opens and checks the root; for a stdlib import checks shadowing and
    /// returns `None`, else walks to the module and returns it.
    pub(super) fn resolve(
        module: &str,
        segments: &[&str],
        is_stdlib: bool,
        root: &Path,
        euid: u32,
        after_dir: &mut dyn FnMut(&Path),
    ) -> Result<Option<ProjectModule>, ImportError> {
        let mut dir = walk::open_root(root).map_err(|e| root_open_error(module, root, e))?;
        check_dir_handle(module, &dir, root, euid)?;
        after_dir(root);

        if is_stdlib {
            let top = segments[0];
            for name in [format!("{top}.sky"), top.to_string()] {
                match walk::kind_at(&dir, &name) {
                    Ok(_) => {
                        return Err(ImportError::ShadowsStdlib {
                            name: top.to_string(),
                            path: root.join(&name),
                        })
                    }
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(ImportError::Io {
                            path: root.join(&name),
                            error,
                        })
                    }
                }
            }
            return Ok(None);
        }

        let (last, dirs) = segments.split_last().expect("at least one segment");
        let mut path = root.to_path_buf();
        for name in dirs {
            path.push(name);
            let next = walk::open_dir_at(&dir, name)
                .map_err(|e| open_error(module, &dir, name, &path, e, true))?;
            check_dir_handle(module, &next, &path, euid)?;
            dir = next;
            after_dir(&path);
        }
        let name = format!("{last}.sky");
        path.push(&name);
        let file = walk::open_file_at(&dir, &name)
            .map_err(|e| open_error(module, &dir, &name, &path, e, false))?;
        let facts = walk::facts(&file.metadata().map_err(|error| ImportError::Io {
            path: path.clone(),
            error,
        })?);
        check_file(facts, euid).map_err(|r| r.into_error(module, &path, euid))?;
        Ok(Some(ProjectModule { path, file }))
    }

    fn check_dir_handle(
        module: &str,
        dir: &File,
        path: &Path,
        euid: u32,
    ) -> Result<(), ImportError> {
        let meta = dir.metadata().map_err(|error| ImportError::Io {
            path: path.to_path_buf(),
            error,
        })?;
        check_dir(walk::facts(&meta), euid).map_err(|r| r.into_error(module, path, euid))
    }

    fn root_open_error(module: &str, root: &Path, error: io::Error) -> ImportError {
        match std::fs::symlink_metadata(root) {
            Ok(m) if m.file_type().is_symlink() => ImportError::Symlink {
                module: module.to_string(),
                path: root.to_path_buf(),
            },
            Ok(m) if !m.is_dir() => ImportError::NotADirectory {
                module: module.to_string(),
                path: root.to_path_buf(),
            },
            _ => ImportError::Io {
                path: root.to_path_buf(),
                error,
            },
        }
    }

    /// Explains a failed `openat` of `name` in `dir`. The open has already
    /// failed; the `fstatat` here only picks the message.
    fn open_error(
        module: &str,
        dir: &File,
        name: &str,
        path: &Path,
        error: io::Error,
        want_dir: bool,
    ) -> ImportError {
        let (module, path) = (module.to_string(), path.to_path_buf());
        if error.kind() == io::ErrorKind::NotFound {
            return ImportError::NotFound { module, path };
        }
        match walk::kind_at(dir, name) {
            Ok(Kind::Symlink) => ImportError::Symlink { module, path },
            Ok(Kind::Other) if want_dir => ImportError::NotADirectory { module, path },
            Err(e) if e.kind() == io::ErrorKind::NotFound => ImportError::NotFound { module, path },
            _ => ImportError::Io { path, error },
        }
    }
}

/// Off Unix there are no handle-relative opens, so a rooted project's
/// imports (and the stdlib shadowing check) fail closed.
#[cfg(not(unix))]
mod rooted {
    use std::path::Path;

    use super::{ImportError, ProjectModule};

    pub(super) fn resolve(
        module: &str,
        _segments: &[&str],
        _is_stdlib: bool,
        _root: &Path,
        _euid: u32,
        _after_dir: &mut dyn FnMut(&Path),
    ) -> Result<Option<ProjectModule>, ImportError> {
        Err(ImportError::Unsupported {
            module: module.to_string(),
        })
    }
}

/// Validates the syntax of a dotted module path and splits it.
pub fn parse_module_path(module: &str) -> Result<Vec<&str>, ImportError> {
    let owned = || module.to_string();
    if module.is_empty() {
        return Err(ImportError::Empty);
    }
    let b = module.as_bytes();
    if b[0] == b'/' || b[0] == b'\\' || (b.len() >= 2 && b[1] == b':') {
        return Err(ImportError::Absolute { module: owned() });
    }
    if module.contains("..") {
        return Err(ImportError::ParentTraversal { module: owned() });
    }
    let segments: Vec<&str> = module.split('.').collect();
    if segments.iter().any(|s| s.is_empty()) {
        return Err(ImportError::EmptySegment { module: owned() });
    }
    if let Some(bad) = segments.iter().find(|s| !is_identifier(s)) {
        return Err(ImportError::InvalidSegment {
            module: owned(),
            segment: bad.to_string(),
        });
    }
    Ok(segments)
}

/// ASCII identifiers only, so a module path never contains separators,
/// confusable characters, or anything a filesystem might normalize.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl ImportError {
    /// This error as a diagnostic whose primary span is the import's module
    /// path in the importing file.
    pub fn to_diagnostic(&self, span: Span) -> Diagnostic {
        let d = |msg: String| Diagnostic::error(msg, span);
        match self {
            ImportError::Empty => d("empty module path".into()),
            ImportError::Absolute { module } => d(format!("module path `{module}` is absolute"))
                .with_help("imports are dotted names relative to the project root, like `physics.kinematics`"),
            ImportError::ParentTraversal { module } => d(format!("module path `{module}` contains `..`"))
                .with_note("relative imports and parent-directory traversal are not supported")
                .with_help("import by the full dotted name from the project root"),
            ImportError::EmptySegment { module } => d(format!("module path `{module}` has an empty segment")),
            ImportError::InvalidSegment { module, segment } => d(format!(
                "module path `{module}`: `{segment}` is not a valid module name"
            ))
            .with_note("module names are ASCII identifiers: letters, digits, and `_`, not starting with a digit"),
            ImportError::UnknownStdlib { module } => d(format!("no standard library module `{module}`")),
            ImportError::ShadowsStdlib { name, path } => d(format!(
                "project module {} shadows the standard library module `{name}`",
                path.display()
            ))
            .with_note(format!("`{name}` is reserved for the standard library"))
            .with_help(format!("rename {}", path.display())),
            ImportError::NoProject { module } => d(format!(
                "cannot import project module `{module}`: no sky.toml found"
            ))
            .with_note("without a project root only standard library imports resolve")
            .with_help("create a sky.toml in the project's top directory"),
            ImportError::NotFound { module, path } => {
                d(format!("module `{module}` not found: {} does not exist", path.display()))
            }
            ImportError::NotADirectory { module, path } => {
                d(format!("module `{module}` not found: {} is not a directory", path.display()))
            }
            ImportError::NotAFile { module, path } => {
                d(format!("module `{module}`: {} is not a regular file", path.display()))
            }
            ImportError::Symlink { module, path } => d(format!(
                "refusing to import `{module}`: {} is a symbolic link",
                path.display()
            ))
            .with_note("imports never follow symbolic links, so a link cannot point an import outside the project")
            .with_help(format!("replace {} with the file or directory it points to", path.display())),
            ImportError::ForeignOwner {
                module,
                path,
                owner,
                euid,
            } => d(format!(
                "refusing to import `{module}`: {} is owned by uid {owner}, not by you (uid {euid})",
                path.display()
            ))
            .with_note("another user could replace this module with code you did not write")
            .with_help(format!(
                "if you trust it, have {} owned by uid {euid}, or copy the project into a directory you own",
                path.display()
            )),
            ImportError::WorldWritable { module, path, mode } => d(format!(
                "refusing to import `{module}`: {} is writable by every user (mode {:04o})",
                path.display(),
                mode & 0o7777
            ))
            .with_note("any user can replace files in a world-writable directory that does not have the sticky bit")
            .with_help(format!("remove write permission for others: chmod o-w {}", path.display())),
            ImportError::WorldWritableFile { module, path, mode } => d(format!(
                "refusing to import `{module}`: {} is writable by every user (mode {:04o})",
                path.display(),
                mode & 0o7777
            ))
            .with_note("any user can change a world-writable module file to code you did not write")
            .with_help(format!("remove write permission for others: chmod o-w {}", path.display())),
            ImportError::Unsupported { module } => d(format!(
                "cannot import project module `{module}`: project imports are only supported on Unix"
            )),
            ImportError::Io { path, error } => d(format!("cannot inspect {}: {error}", path.display())),
        }
    }
}

/// The message with terminal-unsafe characters escaped, so a hostile path
/// can't reach the terminal raw through `{}`.
impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&display_safe(&self.to_diagnostic(Span::point(0)).message))
    }
}

impl std::error::Error for ImportError {}

#[cfg(all(test, unix))]
mod tests;
