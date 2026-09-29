//! Shared test helpers: std-only temp dirs and an injectable identity.
#![allow(dead_code)]

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use sky_project::Identity;

/// A fresh, canonical temp directory, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("sky-project-{}-{}-{n}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir.canonicalize().unwrap())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Creates `rel` (and parents) as a file with `text`; returns its path.
    pub fn file(&self, rel: &str, text: &str) -> PathBuf {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, text).unwrap();
        p
    }

    /// Creates `rel` (and parents) as a directory; returns its path.
    pub fn dir(&self, rel: &str) -> PathBuf {
        let p = self.0.join(rel);
        fs::create_dir_all(&p).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// An identity with a chosen euid and home. Every test injects a home inside
/// its own temp dir, so no walk ever leaves the test's directory tree.
pub struct FakeIdentity {
    pub euid: u32,
    pub home: Option<PathBuf>,
}

impl Identity for FakeIdentity {
    fn effective_uid(&self) -> u32 {
        self.euid
    }
    fn home_dir(&self) -> Option<PathBuf> {
        self.home.clone()
    }
}

/// The uid that owns files this process creates.
pub fn my_uid(t: &TempDir) -> u32 {
    fs::metadata(t.path()).unwrap().uid()
}

/// Identity of the current user with home at the temp dir itself.
pub fn me(t: &TempDir) -> FakeIdentity {
    FakeIdentity {
        euid: my_uid(t),
        home: Some(t.path().to_path_buf()),
    }
}
