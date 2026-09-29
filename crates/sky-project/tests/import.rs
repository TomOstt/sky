#![cfg(unix)]

mod common;

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use common::{me, my_uid, FakeIdentity, TempDir};
use sky_diag::{SourceFile, Span};
use sky_project::{
    parse_module_path, resolve_import, ImportError, Module, Project, SourceError, MAX_SOURCE_BYTES,
};

const STDLIB: &[&str] = &["math", "linalg", "linalg.cholesky", "units"];

/// The module a successful project import opened: its path and contents.
fn opened(r: Result<Module, ImportError>) -> (PathBuf, String) {
    match r {
        Ok(Module::Project(m)) => {
            let src = m.read_source().unwrap();
            (m.path().to_path_buf(), src)
        }
        other => panic!("expected a project module, got {other:?}"),
    }
}

fn stdlib(r: Result<Module, ImportError>) -> String {
    match r {
        Ok(Module::Stdlib(name)) => name,
        other => panic!("expected a stdlib module, got {other:?}"),
    }
}

fn chmod(p: &Path, mode: u32) {
    fs::set_permissions(p, fs::Permissions::from_mode(mode)).unwrap();
}

fn project(root: &Path) -> Project {
    Project::Rooted {
        root: root.to_path_buf(),
        manifest: root.join("sky.toml"),
    }
}

#[test]
fn dotted_path_opens_a_file_under_the_root() {
    let t = TempDir::new("imp-ok");
    t.file("sky.toml", "");
    let target = t.file("physics/kinematics.sky", "let v = 1\n");
    let deep = t.file("a/b/c/d.sky", "deep");
    let top = t.file("helpers.sky", "top");
    let p = project(t.path());
    assert_eq!(
        opened(resolve_import("physics.kinematics", &p, STDLIB, &me(&t))),
        (target, "let v = 1\n".into())
    );
    assert_eq!(
        opened(resolve_import("a.b.c.d", &p, STDLIB, &me(&t))),
        (deep, "deep".into())
    );
    assert_eq!(
        opened(resolve_import("helpers", &p, STDLIB, &me(&t))),
        (top, "top".into())
    );
}

#[test]
fn symlinks_are_never_followed_even_inside_the_root() {
    let t = TempDir::new("imp-inlink");
    let target = t.file("real/a.sky", "");
    t.dir("alias");
    symlink(&target, t.path().join("alias/a.sky")).unwrap();
    symlink(t.path().join("real"), t.path().join("dirlink")).unwrap();
    for (m, link) in [("alias.a", "alias/a.sky"), ("dirlink.a", "dirlink")] {
        match resolve_import(m, &project(t.path()), STDLIB, &me(&t)) {
            Err(ImportError::Symlink { path, .. }) => assert_eq!(path, t.path().join(link), "{m}"),
            other => panic!("{m}: expected Symlink, got {other:?}"),
        }
    }
}

#[test]
fn symlinked_root_is_refused() {
    let t = TempDir::new("imp-rootlink");
    t.file("real/a.sky", "");
    symlink(t.path().join("real"), t.path().join("link")).unwrap();
    assert!(matches!(
        resolve_import("a", &project(&t.path().join("link")), STDLIB, &me(&t)),
        Err(ImportError::Symlink { .. })
    ));
}

#[test]
fn stdlib_imports_resolve() {
    let t = TempDir::new("imp-std");
    let p = project(t.path());
    assert_eq!(stdlib(resolve_import("math", &p, STDLIB, &me(&t))), "math");
    assert_eq!(
        stdlib(resolve_import("linalg.cholesky", &p, STDLIB, &me(&t))),
        "linalg.cholesky"
    );
}

#[test]
fn unknown_module_under_a_stdlib_name_is_not_looked_up_in_the_project() {
    let t = TempDir::new("imp-unkstd");
    assert!(matches!(
        resolve_import("linalg.qr", &project(t.path()), STDLIB, &me(&t)),
        Err(ImportError::UnknownStdlib { .. })
    ));
}

#[test]
fn project_module_shadowing_stdlib_is_an_error() {
    let t = TempDir::new("imp-shadow");
    let shadow = t.file("math.sky", "");
    let err = resolve_import("math", &project(t.path()), STDLIB, &me(&t)).unwrap_err();
    assert!(
        matches!(&err, ImportError::ShadowsStdlib { name, path } if name == "math" && path == &shadow)
    );

    // A directory named like a stdlib module shadows too, for any import in it.
    t.dir("linalg");
    assert!(matches!(
        resolve_import("linalg.cholesky", &project(t.path()), STDLIB, &me(&t)),
        Err(ImportError::ShadowsStdlib { .. })
    ));

    let src = "import math\n";
    let file = SourceFile::new("main.sky", src);
    let out = err.to_diagnostic(Span::new(7, 11)).render_to_string(&file);
    assert_eq!(
        out,
        format!(
            "\
main.sky:1:8: error: project module {s} shadows the standard library module `math`
  |
1 | import math
  |        ^^^^
  = note: `math` is reserved for the standard library
  = help: rename {s}
",
            s = shadow.display()
        )
    );
}

#[test]
fn parent_traversal_is_rejected() {
    let t = TempDir::new("imp-dotdot");
    t.file("a.sky", "");
    for m in ["..a", "a..b", "..", "...", "a.."] {
        assert!(
            matches!(
                resolve_import(m, &project(t.path()), STDLIB, &me(&t)),
                Err(ImportError::ParentTraversal { .. })
            ),
            "{m}"
        );
    }
}

#[test]
fn empty_path_and_empty_segments_are_rejected() {
    assert!(matches!(parse_module_path(""), Err(ImportError::Empty)));
    for m in [".a", "a.", "a.b."] {
        assert!(
            matches!(parse_module_path(m), Err(ImportError::EmptySegment { .. })),
            "{m}"
        );
    }
}

#[test]
fn absolute_paths_are_rejected() {
    for m in ["/etc/passwd", "\\a", "C:a"] {
        assert!(
            matches!(parse_module_path(m), Err(ImportError::Absolute { .. })),
            "{m}"
        );
    }
}

#[test]
fn non_identifier_segments_are_rejected() {
    for (m, seg) in [
        ("a/b", "a/b"),
        ("a.b-c", "b-c"),
        ("1a", "1a"),
        ("a. b", " b"),
        ("ħ", "ħ"),
        ("a.b\0", "b\0"),
    ] {
        match parse_module_path(m) {
            Err(ImportError::InvalidSegment { segment, .. }) => assert_eq!(segment, seg, "{m}"),
            other => panic!("{m}: expected InvalidSegment, got {other:?}"),
        }
    }
    assert_eq!(parse_module_path("_a.b_2.C").unwrap(), ["_a", "b_2", "C"]);
}

#[test]
fn symlink_escaping_the_root_is_refused() {
    let t = TempDir::new("imp-escape");
    let outside = t.file("outside/secret.sky", "");
    let root = t.dir("proj");
    t.file("proj/pkg/inner.sky", "");
    symlink(&outside, root.join("leak.sky")).unwrap();
    // An intermediate directory that is a link to outside the root, present
    // from the start, at the first level and deeper.
    symlink(t.path().join("outside"), root.join("dir")).unwrap();
    symlink(t.path().join("outside"), root.join("pkg/sub")).unwrap();
    for (m, link) in [
        ("leak", "leak.sky"),
        ("dir.secret", "dir"),
        ("pkg.sub.secret", "pkg/sub"),
    ] {
        match resolve_import(m, &project(&root), STDLIB, &me(&t)) {
            Err(ImportError::Symlink { path, .. }) => assert_eq!(path, root.join(link), "{m}"),
            other => panic!("{m}: expected Symlink, got {other:?}"),
        }
    }
}

#[test]
fn world_writable_directory_without_sticky_bit_is_refused() {
    let t = TempDir::new("imp-ww");
    t.file("pkg/sub/m.sky", "");
    let pkg = t.path().join("pkg");
    chmod(&pkg, 0o777);
    let err = resolve_import("pkg.sub.m", &project(t.path()), STDLIB, &me(&t)).unwrap_err();
    chmod(&pkg, 0o755);
    match &err {
        ImportError::WorldWritable { path, mode, .. } => {
            assert_eq!(path, &pkg);
            assert_eq!(mode & 0o7777, 0o777);
        }
        other => panic!("expected WorldWritable, got {other:?}"),
    }
    let src = "import pkg.sub.m\n";
    let out = err
        .to_diagnostic(Span::new(7, 16))
        .render_to_string(&SourceFile::new("main.sky", src));
    assert_eq!(
        out,
        format!(
            "\
main.sky:1:8: error: refusing to import `pkg.sub.m`: {p} is writable by every user (mode 0777)
  |
1 | import pkg.sub.m
  |        ^^^^^^^^^
  = note: any user can replace files in a world-writable directory that does not have the sticky bit
  = help: remove write permission for others: chmod o-w {p}
",
            p = pkg.display()
        )
    );
}

#[test]
fn world_writable_root_is_refused_for_every_import() {
    let t = TempDir::new("imp-wwroot");
    let root = t.dir("proj");
    t.file("proj/m.sky", "");
    chmod(&root, 0o777);
    let project_import = resolve_import("m", &project(&root), STDLIB, &me(&t));
    // Fail closed: the stdlib shadowing check needs a trusted root too.
    let stdlib_import = resolve_import("math", &project(&root), STDLIB, &me(&t));
    chmod(&root, 0o755);
    for r in [project_import, stdlib_import] {
        assert!(
            matches!(&r, Err(ImportError::WorldWritable { path, .. }) if path == &root),
            "{r:?}"
        );
    }
}

#[test]
fn world_writable_directory_with_sticky_bit_is_allowed() {
    let t = TempDir::new("imp-sticky");
    let m = t.file("shared/m.sky", "sticky");
    let shared = t.path().join("shared");
    chmod(&shared, 0o1777);
    let r = resolve_import("shared.m", &project(t.path()), STDLIB, &me(&t));
    chmod(&shared, 0o755);
    assert_eq!(opened(r), (m, "sticky".into()));
}

#[test]
fn group_writable_directory_is_allowed() {
    // Out of scope for the agreed rule: only world-writable is refused.
    let t = TempDir::new("imp-gw");
    t.file("grp/m.sky", "");
    chmod(&t.path().join("grp"), 0o775);
    assert!(resolve_import("grp.m", &project(t.path()), STDLIB, &me(&t)).is_ok());
}

#[test]
fn root_owned_by_another_uid_is_refused() {
    // Files owned by another uid can't be created without root, so the
    // effective uid is injected instead: every handle's real st_uid is
    // compared with it, and the root is the first handle. The same check
    // on deeper directories and the file is covered by the unit tests of
    // the rule on fstat facts (src/import/tests.rs).
    let t = TempDir::new("imp-owner");
    t.file("pkg/m.sky", "");
    let id = FakeIdentity {
        euid: my_uid(&t) + 1,
        home: None,
    };
    match resolve_import("pkg.m", &project(t.path()), STDLIB, &id) {
        Err(ImportError::ForeignOwner {
            path, owner, euid, ..
        }) => {
            assert_eq!(path, t.path());
            assert_eq!(owner, my_uid(&t));
            assert_eq!(euid, my_uid(&t) + 1);
        }
        other => panic!("expected ForeignOwner, got {other:?}"),
    }
}

#[test]
fn file_where_a_directory_is_expected_is_refused() {
    let t = TempDir::new("imp-notdir");
    t.file("a", "");
    match resolve_import("a.b", &project(t.path()), STDLIB, &me(&t)) {
        Err(ImportError::NotADirectory { path, .. }) => assert_eq!(path, t.path().join("a")),
        other => panic!("expected NotADirectory, got {other:?}"),
    }
}

#[test]
fn fifo_module_is_refused_without_blocking() {
    let t = TempDir::new("imp-fifo");
    let fifo = t.path().join("mod.sky");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    // SAFETY: `c` is a NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
    let root = t.path().to_path_buf();
    let euid = my_uid(&t);
    let (tx, rx) = mpsc::channel();
    // A blocking open would hang this thread forever; the timeout turns
    // that into a test failure instead of a hung CI job.
    std::thread::spawn(move || {
        let id = FakeIdentity { euid, home: None };
        let r = resolve_import("mod", &project(&root), STDLIB, &id);
        let _ = tx.send(format!("{r:?}"));
    });
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("resolve_import blocked on a FIFO");
    assert!(r.starts_with("Err(NotAFile"), "{r}");
}

#[test]
fn oversized_module_is_refused_before_reading() {
    let t = TempDir::new("imp-big");
    let big = t.path().join("big.sky");
    // Sparse: no data blocks are written and nothing is read or allocated,
    // because the fstat size is checked first.
    fs::File::create(&big)
        .unwrap()
        .set_len(MAX_SOURCE_BYTES + 1)
        .unwrap();
    let Ok(Module::Project(m)) = resolve_import("big", &project(t.path()), STDLIB, &me(&t)) else {
        panic!("expected the module to open");
    };
    match m.read_source() {
        Err(SourceError::TooLarge { path, size, limit }) => {
            assert_eq!(path, big);
            assert_eq!(size, Some(MAX_SOURCE_BYTES + 1));
            assert_eq!(limit, MAX_SOURCE_BYTES);
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

#[test]
fn refusal_escapes_control_characters_in_the_path() {
    let t = TempDir::new("imp-esc");
    let root = t.dir("evil\u{1b}[2J\u{202E}proj");
    t.file("evil\u{1b}[2J\u{202E}proj/pkg/m.sky", "");
    chmod(&root.join("pkg"), 0o777);
    let err = resolve_import("pkg.m", &project(&root), STDLIB, &me(&t)).unwrap_err();
    chmod(&root.join("pkg"), 0o755);
    let rendered = err
        .to_diagnostic(Span::new(7, 12))
        .render_to_string(&SourceFile::new("main.sky", "import pkg.m\n"));
    for out in [rendered, err.to_string()] {
        assert!(out.contains("evil\\u{001B}[2J\\u{202E}proj/pkg"), "{out:?}");
        assert!(!out.contains(['\u{1b}', '\u{202E}']), "{out:?}");
    }
}

#[test]
fn missing_module_and_non_file_are_errors() {
    let t = TempDir::new("imp-missing");
    t.dir("pkg.sky");
    assert!(matches!(
        resolve_import("nope", &project(t.path()), STDLIB, &me(&t)),
        Err(ImportError::NotFound { .. })
    ));
    assert!(matches!(
        resolve_import("pkg", &project(t.path()), STDLIB, &me(&t)),
        Err(ImportError::NotAFile { .. })
    ));
}

#[test]
fn single_file_mode_rejects_project_imports_but_allows_stdlib() {
    let t = TempDir::new("imp-single");
    t.file("helpers.sky", ""); // a sibling file must NOT be importable
    let err = resolve_import("helpers", &Project::SingleFile, STDLIB, &me(&t)).unwrap_err();
    assert!(matches!(&err, ImportError::NoProject { module } if module == "helpers"));
    assert_eq!(
        stdlib(resolve_import(
            "units",
            &Project::SingleFile,
            STDLIB,
            &me(&t)
        )),
        "units"
    );
    let src = "import helpers\n";
    let out = err
        .to_diagnostic(Span::new(7, 14))
        .render_to_string(&SourceFile::new("main.sky", src));
    assert_eq!(
        out,
        "\
main.sky:1:8: error: cannot import project module `helpers`: no sky.toml found
  |
1 | import helpers
  |        ^^^^^^^
  = note: without a project root only standard library imports resolve
  = help: create a sky.toml in the project's top directory
"
    );
}
