#![cfg(unix)]

mod common;

use std::os::unix::fs::symlink;

use common::{me, my_uid, FakeIdentity, TempDir};
use sky_project::{discover, DiscoverError, Project};

fn rooted(root: &std::path::Path) -> Project {
    Project::Rooted {
        root: root.to_path_buf(),
        manifest: root.join("sky.toml"),
    }
}

#[test]
fn finds_sky_toml_in_same_directory() {
    let t = TempDir::new("same");
    t.file("home/proj/sky.toml", "");
    let f = t.file("home/proj/main.sky", "x = 1\n");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    let d = discover(&f, &id).unwrap();
    assert_eq!(d.file, f);
    assert_eq!(d.project, rooted(&t.path().join("home/proj")));
}

#[test]
fn finds_sky_toml_in_a_parent() {
    let t = TempDir::new("parent");
    t.file("home/proj/sky.toml", "");
    let f = t.file("home/proj/src/physics/kinematics.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(
        discover(&f, &id).unwrap().project,
        rooted(&t.path().join("home/proj"))
    );
}

#[test]
fn relative_components_in_the_file_path_are_canonicalized() {
    let t = TempDir::new("canon");
    t.file("home/proj/sky.toml", "");
    t.file("home/proj/src/a.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    let d = discover(&t.path().join("home/proj/src/../src/./a.sky"), &id).unwrap();
    assert_eq!(d.file, t.path().join("home/proj/src/a.sky"));
    assert_eq!(d.project, rooted(&t.path().join("home/proj")));
}

#[test]
fn sky_toml_above_a_git_directory_is_ignored() {
    let t = TempDir::new("gitdir");
    t.file("home/sky.toml", ""); // planted above the repo
    t.dir("home/repo/.git");
    let f = t.file("home/repo/src/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(discover(&f, &id).unwrap().project, Project::SingleFile);
}

#[test]
fn sky_toml_above_a_git_file_is_ignored() {
    let t = TempDir::new("gitfile");
    t.file("home/sky.toml", "");
    t.file(
        "home/worktree/.git",
        "gitdir: /elsewhere/.git/worktrees/wt\n",
    );
    let f = t.file("home/worktree/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(discover(&f, &id).unwrap().project, Project::SingleFile);
}

#[test]
fn sky_toml_in_the_git_boundary_directory_itself_counts() {
    let t = TempDir::new("gitself");
    t.file("home/repo/sky.toml", "");
    t.dir("home/repo/.git");
    let f = t.file("home/repo/src/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(
        discover(&f, &id).unwrap().project,
        rooted(&t.path().join("home/repo"))
    );
}

#[test]
fn sky_toml_above_home_is_ignored() {
    let t = TempDir::new("abovehome");
    t.file("sky.toml", ""); // planted above home
    let f = t.file("home/proj/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(discover(&f, &id).unwrap().project, Project::SingleFile);
}

#[test]
fn sky_toml_in_home_itself_counts() {
    let t = TempDir::new("inhome");
    t.file("home/sky.toml", "");
    let f = t.file("home/scratch/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert_eq!(
        discover(&f, &id).unwrap().project,
        rooted(&t.path().join("home"))
    );
}

#[test]
fn a_relative_home_is_ignored_rather_than_resolved_against_the_cwd() {
    let t = TempDir::new("relhome");
    t.dir("repo/.git");
    t.file("repo/sky.toml", "");
    let f = t.file("repo/a/b/main.sky", "");
    // A relative passwd home is dropped (never resolved against the cwd), so
    // the walk continues to the `.git` boundary.
    let id = FakeIdentity {
        euid: my_uid(&t),
        home: Some("a".into()),
    };
    assert_eq!(
        discover(&f, &id).unwrap().project,
        rooted(&t.path().join("repo"))
    );
}

#[test]
fn foreign_owned_sky_toml_is_refused() {
    let t = TempDir::new("foreign");
    let manifest = t.file("home/proj/sky.toml", "");
    let f = t.file("home/proj/main.sky", "");
    let owner = my_uid(&t);
    // Simulate running as someone else: the files are now "foreign".
    let id = FakeIdentity {
        euid: owner.wrapping_add(1),
        home: Some(t.path().join("home")),
    };
    let err = discover(&f, &id).unwrap_err();
    match &err {
        DiscoverError::ForeignOwner {
            manifest: m,
            path,
            owner: o,
            euid,
        } => {
            assert_eq!(m, &manifest);
            assert_eq!(path, &manifest);
            assert_eq!(*o, owner);
            assert_eq!(*euid, owner.wrapping_add(1));
        }
        other => panic!("expected ForeignOwner, got {other:?}"),
    }
    let m = manifest.display();
    let expected = format!(
        "\
{m}:1:1: error: refusing to use project root: {m} is owned by uid {owner}, not by you (uid {me})
  |
1 |
  | ^
  = note: a sky.toml owned by another user could redirect imports to code you did not write
  = help: if you trust this project, have {m} and the directory containing it owned by uid {me}, or copy the project into a directory you own
",
        me = owner.wrapping_add(1)
    );
    assert_eq!(
        err.to_diagnostic().render_to_string(&err.source_file()),
        expected
    );
}

#[test]
fn symlinked_sky_toml_is_refused_even_if_the_target_is_ours() {
    let t = TempDir::new("symlink");
    let real = t.file("home/elsewhere/real.toml", "");
    t.dir("home/proj");
    let manifest = t.path().join("home/proj/sky.toml");
    symlink(&real, &manifest).unwrap();
    let f = t.file("home/proj/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    let err = discover(&f, &id).unwrap_err();
    assert!(matches!(&err, DiscoverError::ManifestSymlink { manifest: m } if m == &manifest));
    let out = err.to_diagnostic().render_to_string(&err.source_file());
    assert!(out.starts_with(&format!(
        "{}:1:1: error: refusing to use {}: it is a symbolic link\n",
        manifest.display(),
        manifest.display()
    )));
    assert!(out.contains("  = help: replace the link with a regular sky.toml file owned by you\n"));
}

#[test]
fn dangling_symlinked_sky_toml_is_refused() {
    let t = TempDir::new("dangling");
    t.dir("home/proj");
    symlink(t.path().join("nope"), t.path().join("home/proj/sky.toml")).unwrap();
    let f = t.file("home/proj/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert!(matches!(
        discover(&f, &id),
        Err(DiscoverError::ManifestSymlink { .. })
    ));
}

#[test]
fn sky_toml_directory_is_refused() {
    let t = TempDir::new("tomldir");
    t.dir("home/proj/sky.toml");
    let f = t.file("home/proj/main.sky", "");
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    assert!(matches!(
        discover(&f, &id),
        Err(DiscoverError::ManifestNotFile { .. })
    ));
}

#[test]
fn missing_file_is_an_io_error() {
    let t = TempDir::new("missing");
    let err = discover(&t.path().join("nope.sky"), &me(&t)).unwrap_err();
    assert!(matches!(err, DiscoverError::Io { .. }));
}

#[test]
fn symlinked_file_is_discovered_from_its_target_directory() {
    let t = TempDir::new("filelink");
    t.file("home/proj/sky.toml", "");
    let target = t.file("home/proj/main.sky", "");
    t.dir("home/other/.git");
    let link = t.path().join("home/other/link.sky");
    symlink(&target, &link).unwrap();
    let id = FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(&t)
    };
    let d = discover(&link, &id).unwrap();
    assert_eq!(d.file, target);
    assert_eq!(d.project, rooted(&t.path().join("home/proj")));
}
