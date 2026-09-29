//! Mapping a dotted module path to a stdlib module or a file under the root.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use sky_diag::{Diagnostic, Span};

use crate::Project;

/// What an import resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Module {
    /// A standard-library module, by its dotted name.
    Stdlib(String),
    /// A project module: the canonical path of a `.sky` file inside the root.
    Project(PathBuf),
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
    /// No such file under the root.
    NotFound { module: String, path: PathBuf },
    /// The path exists but is not a regular file.
    NotAFile { module: String, path: PathBuf },
    /// The path resolves (via a symlink) outside the project root.
    EscapesRoot {
        module: String,
        path: PathBuf,
        resolved: PathBuf,
    },
    /// A filesystem error other than "not found".
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
/// - Otherwise `a.b.c` maps to `<root>/a/b/c.sky`, which must exist, be a
///   regular file, and canonicalize to a path inside the canonical root. In
///   [`Project::SingleFile`] mode every project import is an error.
///
/// No I/O besides existence checks and canonicalization.
pub fn resolve_import(
    module: &str,
    project: &Project,
    stdlib: &[&str],
) -> Result<Module, ImportError> {
    let segments = parse_module_path(module)?;
    let top = segments[0];

    let root = match project {
        Project::Rooted { root, .. } => Some(canonical_root(root)?),
        Project::SingleFile => None,
    };

    if stdlib.iter().any(|m| m.split('.').next() == Some(top)) {
        if let Some(root) = &root {
            for candidate in [root.join(format!("{top}.sky")), root.join(top)] {
                if exists(&candidate)? {
                    return Err(ImportError::ShadowsStdlib {
                        name: top.to_string(),
                        path: candidate,
                    });
                }
            }
        }
        return if stdlib.contains(&module) {
            Ok(Module::Stdlib(module.to_string()))
        } else {
            Err(ImportError::UnknownStdlib {
                module: module.to_string(),
            })
        };
    }

    let Some(root) = root else {
        return Err(ImportError::NoProject {
            module: module.to_string(),
        });
    };
    let (last, dirs) = segments.split_last().expect("at least one segment");
    let mut path = root.clone();
    path.extend(dirs);
    path.push(format!("{last}.sky"));

    let resolved = match path.canonicalize() {
        Ok(p) => p,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(ImportError::NotFound {
                module: module.to_string(),
                path,
            })
        }
        Err(error) => return Err(ImportError::Io { path, error }),
    };
    if !resolved.starts_with(&root) {
        return Err(ImportError::EscapesRoot {
            module: module.to_string(),
            path,
            resolved,
        });
    }
    if !resolved.is_file() {
        return Err(ImportError::NotAFile {
            module: module.to_string(),
            path,
        });
    }
    Ok(Module::Project(resolved))
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

fn canonical_root(root: &Path) -> Result<PathBuf, ImportError> {
    root.canonicalize().map_err(|error| ImportError::Io {
        path: root.to_path_buf(),
        error,
    })
}

/// Existence without following a final symlink, so a dangling link still
/// counts as "the project has something with this name".
fn exists(path: &Path) -> Result<bool, ImportError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(ImportError::Io {
            path: path.to_path_buf(),
            error,
        }),
    }
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
            ImportError::NotAFile { module, path } => {
                d(format!("module `{module}`: {} is not a regular file", path.display()))
            }
            ImportError::EscapesRoot {
                module,
                path,
                resolved,
            } => d(format!(
                "module `{module}` resolves outside the project root: {} -> {}",
                path.display(),
                resolved.display()
            ))
            .with_help("imports must stay inside the project; remove the symbolic link"),
            ImportError::Io { path, error } => d(format!("cannot inspect {}: {error}", path.display())),
        }
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_diagnostic(Span::point(0)).message)
    }
}

impl std::error::Error for ImportError {}
