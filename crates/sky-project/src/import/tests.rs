//! Deterministic TOCTOU tests through the `after_dir` hook, and the
//! per-handle rule on synthetic `fstat` facts (a file owned by another uid
//! can't be created without root).

use std::fs;
use std::os::unix::fs::{symlink, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::Identity;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        // The mode checks refuse world-writable directories: don't let an
        // unusual umask (e.g. 000) make the fixtures themselves unsafe.
        // SAFETY: umask has no preconditions and cannot fail.
        unsafe { libc::umask(0o022) };
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "sky-project-unit-{}-{name}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir.canonicalize().unwrap())
    }
    fn file(&self, rel: &str, text: &str) -> PathBuf {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, text).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Euid(u32);

impl Identity for Euid {
    fn effective_uid(&self) -> u32 {
        self.0
    }
    fn home_dir(&self) -> Option<PathBuf> {
        None
    }
}

/// A project `<t>/proj` with `physics/kinematics.sky` = "ORIGINAL", and
/// `<t>/evil/kinematics.sky` = "EVIL" outside it.
fn setup(name: &str) -> (TempDir, PathBuf, Project, Euid) {
    let t = TempDir::new(name);
    t.file("proj/sky.toml", "");
    t.file("proj/physics/kinematics.sky", "ORIGINAL");
    t.file("evil/kinematics.sky", "EVIL");
    let root = t.0.join("proj");
    let project = Project::Rooted {
        root: root.clone(),
        manifest: root.join("sky.toml"),
    };
    let euid = Euid(fs::metadata(&t.0).unwrap().uid());
    (t, root, project, euid)
}

/// Replaces the directory `root/physics` with a symlink to `<t>/evil`.
fn swap_dir_for_symlink(t: &TempDir, root: &Path) {
    fs::rename(root.join("physics"), t.0.join("moved")).unwrap();
    symlink(t.0.join("evil"), root.join("physics")).unwrap();
}

fn resolve(
    project: &Project,
    euid: &Euid,
    hook: &mut dyn FnMut(&Path),
) -> Result<Module, ImportError> {
    resolve_with_hook("physics.kinematics", project, &[], euid, hook)
}

#[test]
fn hook_sees_the_root_then_each_directory() {
    let (_t, root, project, euid) = setup("hook-order");
    let mut seen = Vec::new();
    let m = resolve(&project, &euid, &mut |p| seen.push(p.to_path_buf())).unwrap();
    assert_eq!(seen, [root.clone(), root.join("physics")]);
    let Module::Project(m) = m else { panic!() };
    assert_eq!(m.path(), root.join("physics/kinematics.sky"));
    assert_eq!(m.read_source().unwrap(), "ORIGINAL");
}

#[test]
fn directory_swapped_for_symlink_after_it_was_opened_reads_the_original() {
    // The swap happens while we hold the `physics` handle: the module is
    // opened relative to that handle, so it is the original file.
    let (t, root, project, euid) = setup("toctou-held");
    let dir = root.join("physics");
    let mut swapped = false;
    let result = resolve(&project, &euid, &mut |p| {
        if p == dir {
            swap_dir_for_symlink(&t, &root);
            swapped = true;
        }
    });
    assert!(swapped);
    match result {
        Ok(Module::Project(m)) => assert_eq!(m.read_source().unwrap(), "ORIGINAL"),
        Err(ImportError::Symlink { .. }) => {}
        other => panic!("expected the original file or a refusal, got {other:?}"),
    }
    // By name, the path now leads to the other file.
    assert_eq!(
        fs::read_to_string(root.join("physics/kinematics.sky")).unwrap(),
        "EVIL"
    );
}

#[test]
fn directory_swapped_for_symlink_before_it_is_opened_is_refused() {
    // The swap happens after the root is opened, before `openat(physics)`:
    // O_NOFOLLOW refuses the link.
    let (t, root, project, euid) = setup("toctou-next");
    let mut swapped = false;
    let result = resolve(&project, &euid, &mut |p| {
        if p == root {
            swap_dir_for_symlink(&t, &root);
            swapped = true;
        }
    });
    assert!(swapped);
    match result {
        Err(ImportError::Symlink { path, .. }) => assert_eq!(path, root.join("physics")),
        other => panic!("expected Symlink, got {other:?}"),
    }
}

#[test]
fn module_file_swapped_for_symlink_before_it_is_opened_is_refused() {
    let (t, root, project, euid) = setup("toctou-file");
    let dir = root.join("physics");
    let target = dir.join("kinematics.sky");
    let result = resolve(&project, &euid, &mut |p| {
        if p == dir {
            fs::remove_file(&target).unwrap();
            symlink(t.0.join("evil/kinematics.sky"), &target).unwrap();
        }
    });
    match result {
        Err(ImportError::Symlink { path, .. }) => assert_eq!(path, target),
        other => panic!("expected Symlink, got {other:?}"),
    }
}

#[test]
fn directory_made_world_writable_before_it_is_opened_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let (_t, root, project, euid) = setup("toctou-mode");
    let dir = root.join("physics");
    // Made world-writable before `openat(physics)`: the fstat of the new
    // handle sees it.
    let result = resolve(&project, &euid, &mut |p| {
        if p == root {
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o777)).unwrap();
        }
    });
    assert!(
        matches!(&result, Err(ImportError::WorldWritable { path, .. }) if path == &dir),
        "{result:?}"
    );
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
}

fn facts(is_dir: bool, uid: u32, mode: u32) -> Facts {
    Facts {
        is_dir,
        is_file: !is_dir,
        uid,
        mode,
    }
}

#[test]
fn directory_rule() {
    assert_eq!(check_dir(facts(true, 7, 0o755), 7), Ok(()));
    assert_eq!(check_dir(facts(true, 7, 0o775), 7), Ok(())); // group-writable: out of scope
    assert_eq!(check_dir(facts(true, 7, 0o1777), 7), Ok(())); // sticky
    assert_eq!(
        check_dir(facts(true, 7, 0o777), 7),
        Err(Refusal::WorldWritable { mode: 0o777 })
    );
    assert_eq!(
        check_dir(facts(true, 7, 0o40757), 7),
        Err(Refusal::WorldWritable { mode: 0o40757 })
    );
    assert_eq!(
        check_dir(facts(true, 0, 0o755), 7),
        Err(Refusal::ForeignOwner { owner: 0 })
    );
    // Owner is checked before mode.
    assert_eq!(
        check_dir(facts(true, 0, 0o1777), 7),
        Err(Refusal::ForeignOwner { owner: 0 })
    );
    assert_eq!(
        check_dir(facts(false, 7, 0o644), 7),
        Err(Refusal::NotADirectory)
    );
}

#[test]
fn file_rule() {
    assert_eq!(check_file(facts(false, 7, 0o644), 7), Ok(()));
    assert_eq!(
        check_file(facts(false, 1000, 0o644), 7),
        Err(Refusal::ForeignOwner { owner: 1000 })
    );
    assert_eq!(check_file(facts(true, 7, 0o755), 7), Err(Refusal::NotAFile));
}

#[test]
#[should_panic(expected = "not a single path component")]
fn openat_refuses_a_non_component() {
    let t = TempDir::new("component");
    let dir = crate::walk::open_root(&t.0).unwrap();
    let _ = crate::walk::open_dir_at(&dir, "..");
}
