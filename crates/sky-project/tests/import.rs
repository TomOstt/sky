#![cfg(unix)]

mod common;

use std::os::unix::fs::symlink;
use std::path::Path;

use common::TempDir;
use sky_diag::{SourceFile, Span};
use sky_project::{parse_module_path, resolve_import, ImportError, Module, Project};

const STDLIB: &[&str] = &["math", "linalg", "linalg.cholesky", "units"];

fn project(root: &Path) -> Project {
    Project::Rooted {
        root: root.to_path_buf(),
        manifest: root.join("sky.toml"),
    }
}

#[test]
fn dotted_path_maps_to_a_file_under_the_root() {
    let t = TempDir::new("imp-ok");
    t.file("sky.toml", "");
    let target = t.file("physics/kinematics.sky", "");
    let top = t.file("helpers.sky", "");
    let p = project(t.path());
    assert_eq!(
        resolve_import("physics.kinematics", &p, STDLIB).unwrap(),
        Module::Project(target)
    );
    assert_eq!(
        resolve_import("helpers", &p, STDLIB).unwrap(),
        Module::Project(top)
    );
}

#[test]
fn symlink_that_stays_inside_the_root_is_allowed() {
    let t = TempDir::new("imp-inlink");
    let target = t.file("real/a.sky", "");
    t.dir("alias");
    symlink(&target, t.path().join("alias/a.sky")).unwrap();
    assert_eq!(
        resolve_import("alias.a", &project(t.path()), STDLIB).unwrap(),
        Module::Project(target)
    );
}

#[test]
fn stdlib_imports_resolve() {
    let t = TempDir::new("imp-std");
    let p = project(t.path());
    assert_eq!(
        resolve_import("math", &p, STDLIB).unwrap(),
        Module::Stdlib("math".into())
    );
    assert_eq!(
        resolve_import("linalg.cholesky", &p, STDLIB).unwrap(),
        Module::Stdlib("linalg.cholesky".into())
    );
}

#[test]
fn unknown_module_under_a_stdlib_name_is_not_looked_up_in_the_project() {
    let t = TempDir::new("imp-unkstd");
    assert!(matches!(
        resolve_import("linalg.qr", &project(t.path()), STDLIB),
        Err(ImportError::UnknownStdlib { .. })
    ));
}

#[test]
fn project_module_shadowing_stdlib_is_an_error() {
    let t = TempDir::new("imp-shadow");
    let shadow = t.file("math.sky", "");
    let err = resolve_import("math", &project(t.path()), STDLIB).unwrap_err();
    assert!(
        matches!(&err, ImportError::ShadowsStdlib { name, path } if name == "math" && path == &shadow)
    );

    // A directory named like a stdlib module shadows too, for any import in it.
    t.dir("linalg");
    assert!(matches!(
        resolve_import("linalg.cholesky", &project(t.path()), STDLIB),
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
                resolve_import(m, &project(t.path()), STDLIB),
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
fn symlink_escaping_the_root_is_rejected() {
    let t = TempDir::new("imp-escape");
    let outside = t.file("outside/secret.sky", "");
    let root = t.dir("proj");
    symlink(&outside, root.join("leak.sky")).unwrap();
    symlink(t.path().join("outside"), root.join("dir")).unwrap();
    match resolve_import("leak", &project(&root), STDLIB) {
        Err(ImportError::EscapesRoot { resolved, .. }) => assert_eq!(resolved, outside),
        other => panic!("expected EscapesRoot, got {other:?}"),
    }
    assert!(matches!(
        resolve_import("dir.secret", &project(&root), STDLIB),
        Err(ImportError::EscapesRoot { .. })
    ));
}

#[test]
fn missing_module_and_non_file_are_errors() {
    let t = TempDir::new("imp-missing");
    t.dir("pkg.sky");
    assert!(matches!(
        resolve_import("nope", &project(t.path()), STDLIB),
        Err(ImportError::NotFound { .. })
    ));
    assert!(matches!(
        resolve_import("pkg", &project(t.path()), STDLIB),
        Err(ImportError::NotAFile { .. })
    ));
}

#[test]
fn single_file_mode_rejects_project_imports_but_allows_stdlib() {
    let t = TempDir::new("imp-single");
    t.file("helpers.sky", ""); // a sibling file must NOT be importable
    let err = resolve_import("helpers", &Project::SingleFile, STDLIB).unwrap_err();
    assert!(matches!(&err, ImportError::NoProject { module } if module == "helpers"));
    assert_eq!(
        resolve_import("units", &Project::SingleFile, STDLIB).unwrap(),
        Module::Stdlib("units".into())
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
