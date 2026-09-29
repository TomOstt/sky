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
//!   absolute or malformed paths, and stdlib shadowing. It opens the module
//!   by a handle-relative walk from the root (`openat` with `O_NOFOLLOW` at
//!   every level), checks each handle it holds (owned by the effective uid;
//!   directories not world-writable unless sticky; the module file not
//!   world-writable at all), and returns the open file, so the module is
//!   never reopened by name.
//! - The manifest is read (at most 64 KiB, UTF-8, a strict `key = "string"`
//!   subset, see [`parse_manifest`]) from a handle opened with `O_NOFOLLOW`
//!   and re-checked by `fstat`, never by a second open of the path. It sets
//!   the [`Edition`]; no field can affect imports.
//! - [`read_source`] and [`read_entry_file`] read a source from a handle:
//!   regular files only, at most [`sky_syntax::MAX_SOURCE_BYTES`].

mod discover;
mod identity;
mod import;
mod manifest;
mod source;
#[cfg(unix)]
mod walk;

pub use discover::{discover, discover_opened, DiscoverError, Discovery, Project, MANIFEST};
pub use identity::{Identity, SystemIdentity};
pub use import::{parse_module_path, resolve_import, ImportError, Module, ProjectModule};
pub use manifest::{
    parse_manifest, Manifest, ManifestError, ManifestErrorKind, MAX_MANIFEST_BYTES,
};
pub use sky_syntax::Edition;
pub use source::{open_entry_file, read_entry_file, read_source, SourceError};
