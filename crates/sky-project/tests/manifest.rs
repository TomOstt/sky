#![cfg(unix)]
//! Reading `sky.toml` during discovery: the edition, and every way the read
//! can be refused.

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;

use common::{me, my_uid, FakeIdentity, TempDir};
use sky_project::{
    discover, DiscoverError, Edition, ManifestErrorKind, Project, SourceError, MAX_MANIFEST_BYTES,
};

fn id(t: &TempDir) -> FakeIdentity {
    FakeIdentity {
        home: Some(t.path().join("home")),
        ..me(t)
    }
}

#[test]
fn declared_edition_is_read() {
    let t = TempDir::new("m-edition");
    t.file("home/proj/sky.toml", "# my project\nedition = \"2026\"\n");
    let f = t.file("home/proj/main.sky", "");
    let d = discover(&f, &id(&t)).unwrap();
    assert_eq!(d.edition, Edition::E2026);
    assert!(matches!(d.project, Project::Rooted { .. }));
}

#[test]
fn a_manifest_without_an_edition_means_the_first() {
    let t = TempDir::new("m-none");
    t.file("home/proj/sky.toml", "");
    let f = t.file("home/proj/main.sky", "");
    assert_eq!(discover(&f, &id(&t)).unwrap().edition, Edition::FIRST);
}

#[test]
fn a_single_file_uses_the_latest_edition() {
    let t = TempDir::new("m-single");
    let f = t.file("home/proj/main.sky", "");
    let d = discover(&f, &id(&t)).unwrap();
    assert_eq!(d.project, Project::SingleFile);
    assert_eq!(d.edition, Edition::LATEST);
}

#[test]
fn only_the_nearest_manifest_is_read() {
    let t = TempDir::new("m-nearest");
    t.file("home/proj/sky.toml", "not a manifest\n"); // never read
    t.file("home/proj/sub/sky.toml", "edition = \"2026\"\n");
    let f = t.file("home/proj/sub/main.sky", "");
    assert_eq!(discover(&f, &id(&t)).unwrap().edition, Edition::E2026);
}

#[test]
fn a_newer_edition_is_an_error_with_a_span() {
    let t = TempDir::new("m-new");
    let manifest = t.file("home/proj/sky.toml", "# c\nedition = \"2031\"\n");
    let f = t.file("home/proj/main.sky", "");
    let err = discover(&f, &id(&t)).unwrap_err();
    assert!(
        matches!(&err, DiscoverError::ManifestInvalid { error, .. }
            if error.kind == ManifestErrorKind::EditionTooNew { year: 2031 }),
        "{err:?}"
    );
    let m = manifest.display();
    let expected = format!(
        "\
{m}:2:11: error: edition 2031 is newer than this Sky (supports 2026)
  |
2 | edition = \"2031\"
  |           ^^^^^^ newer than this Sky
  = note: a project may not use an edition its Sky does not know, because it could mean something else
  = help: upgrade Sky, or set `edition = \"2026\"` if the project supports it
"
    );
    assert_eq!(
        err.to_diagnostic().render_to_string(&err.source_file()),
        expected
    );
    assert_eq!(
        err.to_string(),
        "edition 2031 is newer than this Sky (supports 2026)"
    );
}

#[test]
fn an_unknown_key_gets_a_did_you_mean() {
    let t = TempDir::new("m-key");
    let manifest = t.file("home/proj/sky.toml", "editon = \"2026\"\n");
    let f = t.file("home/proj/main.sky", "");
    let err = discover(&f, &id(&t)).unwrap_err();
    let m = manifest.display();
    let expected = format!(
        "\
{m}:1:1: error: unknown key `editon`
  |
1 | editon = \"2026\"
  | ^^^^^^ not a sky.toml key
  = note: the only key is `edition`; sky.toml cannot add import paths or change how imports resolve
  = help: did you mean `edition`?
"
    );
    assert_eq!(
        err.to_diagnostic().render_to_string(&err.source_file()),
        expected
    );
}

#[test]
fn a_duplicate_key_is_an_error() {
    let t = TempDir::new("m-dup");
    t.file(
        "home/proj/sky.toml",
        "edition = \"2026\"\nedition = \"2026\"\n",
    );
    let f = t.file("home/proj/main.sky", "");
    assert!(matches!(
        discover(&f, &id(&t)),
        Err(DiscoverError::ManifestInvalid { error, .. })
            if matches!(error.kind, ManifestErrorKind::DuplicateKey { .. })
    ));
}

#[test]
fn a_bidi_control_in_a_comment_is_refused() {
    let t = TempDir::new("m-bidi");
    t.file("home/proj/sky.toml", "# \u{202E}\nedition = \"2026\"\n");
    let f = t.file("home/proj/main.sky", "");
    assert!(matches!(
        discover(&f, &id(&t)),
        Err(DiscoverError::ManifestInvalid { error, .. })
            if matches!(error.kind, ManifestErrorKind::BidiControl { .. })
    ));
}

#[test]
fn a_manifest_over_the_size_cap_is_refused_unread() {
    let t = TempDir::new("m-big");
    let manifest = t.file(
        "home/proj/sky.toml",
        &format!("#{}\n", "a".repeat(MAX_MANIFEST_BYTES)),
    );
    let f = t.file("home/proj/main.sky", "");
    let err = discover(&f, &id(&t)).unwrap_err();
    assert!(
        matches!(
            &err,
            DiscoverError::ManifestUnreadable {
                error: SourceError::TooLarge { limit: 65536, .. },
                ..
            }
        ),
        "{err:?}"
    );
    let shown = err.to_diagnostic().render_to_string(&err.source_file());
    assert!(
        shown.contains("over the limit of 65536 bytes (64 KiB)"),
        "{shown}"
    );
    assert!(
        shown.contains("reads at most 64 KiB from a sky.toml"),
        "{shown}"
    );
    assert!(shown.starts_with(&manifest.display().to_string()));
}

#[test]
fn a_manifest_of_exactly_the_cap_is_read() {
    let t = TempDir::new("m-cap");
    let mut text = String::from("edition = \"2026\"\n#");
    text.push_str(&"a".repeat(MAX_MANIFEST_BYTES - text.len()));
    assert_eq!(text.len(), MAX_MANIFEST_BYTES);
    t.file("home/proj/sky.toml", &text);
    let f = t.file("home/proj/main.sky", "");
    assert_eq!(discover(&f, &id(&t)).unwrap().edition, Edition::E2026);
}

#[test]
fn a_manifest_that_is_not_utf8_is_refused() {
    let t = TempDir::new("m-utf8");
    let manifest = t.path().join("home/proj/sky.toml");
    t.dir("home/proj");
    fs::write(&manifest, b"edition = \"20\xff26\"\n").unwrap();
    let f = t.file("home/proj/main.sky", "");
    assert!(matches!(
        discover(&f, &id(&t)),
        Err(DiscoverError::ManifestUnreadable {
            error: SourceError::NotUtf8 { .. },
            ..
        })
    ));
}

#[test]
fn an_unreadable_manifest_is_an_error_not_a_default() {
    let t = TempDir::new("m-perm");
    let manifest = t.file("home/proj/sky.toml", "edition = \"2026\"\n");
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o000)).unwrap();
    let f = t.file("home/proj/main.sky", "");
    let r = discover(&f, &id(&t));
    // Root can read anything; everyone else must be refused.
    if my_uid(&t) != 0 {
        assert!(matches!(r, Err(DiscoverError::Io { .. })), "{r:?}");
    }
}

#[test]
fn a_foreign_manifest_is_refused_before_it_is_read() {
    let t = TempDir::new("m-foreign");
    // Invalid content: if it were read, the error would be ManifestInvalid.
    t.file("home/proj/sky.toml", "garbage\n");
    let f = t.file("home/proj/main.sky", "");
    let id = FakeIdentity {
        euid: my_uid(&t).wrapping_add(1),
        home: Some(t.path().join("home")),
    };
    assert!(matches!(
        discover(&f, &id),
        Err(DiscoverError::ForeignOwner { .. })
    ));
}

#[test]
fn a_fifo_named_sky_toml_is_refused_without_blocking() {
    let t = TempDir::new("m-fifo");
    t.dir("home/proj");
    let manifest = t.path().join("home/proj/sky.toml");
    let c = std::ffi::CString::new(manifest.to_str().unwrap()).unwrap();
    // SAFETY: `c` is a valid NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    let f = t.file("home/proj/main.sky", "");
    assert!(matches!(
        discover(&f, &id(&t)),
        Err(DiscoverError::ManifestNotFile { .. })
    ));
}
