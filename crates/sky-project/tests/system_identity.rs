//! The real identity must ignore `$HOME`. Its own test binary (one test), so
//! changing the environment cannot race with other tests.
#![cfg(unix)]

use std::os::unix::fs::MetadataExt;

use sky_project::{Identity, SystemIdentity};

#[test]
fn system_identity_ignores_home_env_var() {
    let before = SystemIdentity.home_dir();
    std::env::set_var("HOME", "/nonexistent/sky-evil-home");
    std::env::remove_var("USER");
    std::env::set_var("SKY_ROOT", "/nonexistent/sky-evil-root");
    let after = SystemIdentity.home_dir();
    assert_eq!(before, after);
    assert_ne!(
        after.as_deref(),
        Some(std::path::Path::new("/nonexistent/sky-evil-home"))
    );

    let probe = std::env::temp_dir().join(format!("sky-euid-probe-{}", std::process::id()));
    std::fs::write(&probe, "").unwrap();
    let owner = std::fs::metadata(&probe).unwrap().uid();
    std::fs::remove_file(&probe).unwrap();
    assert_eq!(SystemIdentity.effective_uid(), owner);
}
