//! Project-root discovery and import resolution for Sky.
//!
//! Shared by `sky check`, `sky run`, `sky repl`, and (later) the type
//! checker's module loader, so it lives in its own small crate rather than
//! inside the `sky` binary.
//!
//! Spec v0.1 non-negotiable 2: imports resolve only from the project root and
//! the standard library, never from the working directory or environment
//! variables. This crate reads no environment variable and never uses the
//! current directory as a root.
//!
//! - [`discover`] walks up from the checked file to a trusted `sky.toml`,
//!   stopping at a `.git` entry, the user's home directory (from the passwd
//!   database), or the filesystem root. A `sky.toml`, or its directory, owned
//!   by a uid other than the effective uid is refused, as is a symlinked
//!   `sky.toml` (the lesson of Git's CVE-2022-24765).
//! - Without a `sky.toml` a file is checked in [`Project::SingleFile`] mode:
//!   only stdlib imports resolve.
//! - [`resolve_import`] maps `a.b.c` to `<root>/a/b/c.sky`, rejecting `..`,
//!   absolute or malformed paths, symlink escapes, and stdlib shadowing.

mod discover;
mod identity;
mod import;

pub use discover::{discover, DiscoverError, Discovery, Project, MANIFEST};
pub use identity::{Identity, SystemIdentity};
pub use import::{parse_module_path, resolve_import, ImportError, Module};
