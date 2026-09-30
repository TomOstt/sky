//! The process-wide `lookup` reads `SKY_TUNE_DB`. This file holds one test,
//! because it changes the environment of its own process.

use sky_tune::{invalidate_cache, lookup, lookup_among, tuned_note, Params};

#[test]
fn lookup_reads_the_database_named_by_the_environment() {
    let dir = std::env::temp_dir().join(format!("sky-tune-lookup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tune.db");
    std::env::set_var("SKY_TUNE_DB", &path);

    // No file: defaults.
    assert_eq!(lookup("dot_blocked", "2^18"), None);
    assert_eq!(tuned_note("dot_blocked", "2^18"), None);

    let fp = sky_hw::current_fingerprint();
    let text = format!(
        "sky-tune-db 1\n\
         result fp={fp} kernel=dot_blocked size=2^18 params=unroll:8 default=unroll:4 best_ns=100 default_ns=141 when=1790000000\n\
         result fp=ffffffffffffffff kernel=dot_blocked size=2^18 params=unroll:1 default=unroll:4 best_ns=100 default_ns=141 when=1790000000\n\
         result fp={fp} kernel=evil size=2^18 params=unroll:8;rm default=unroll:4 best_ns=1 default_ns=1 when=1\n"
    );
    std::fs::write(&path, text).unwrap();
    invalidate_cache();

    let hit = lookup("dot_blocked", "2^18").expect("a result for this machine");
    assert_eq!(hit, Params::new().with("unroll", 8));
    assert_eq!(lookup("dot_blocked", "2^19"), None);
    assert_eq!(
        lookup("evil", "2^18"),
        None,
        "the malformed entry is ignored"
    );

    let candidates = [
        Params::new().with("unroll", 4),
        Params::new().with("unroll", 8),
    ];
    assert_eq!(lookup_among("dot_blocked", "2^18", &candidates), Some(hit));
    assert_eq!(
        lookup_among("dot_blocked", "2^18", &candidates[..1]),
        None,
        "a stored value outside the kernel's candidates is ignored"
    );

    let note = tuned_note("dot_blocked", "2^18").unwrap();
    assert_eq!(
        note,
        "tuned on this machine (2026-09-21): unroll 8, 1.41x faster than the default 4"
    );

    // A foreign file is data nobody executes: lookups just see no result.
    std::fs::write(&path, "#!/bin/sh\necho pwned > pwned\n").unwrap();
    invalidate_cache();
    assert_eq!(lookup("dot_blocked", "2^18"), None);
    assert!(!dir.join("pwned").exists());

    let _ = std::fs::remove_dir_all(&dir);
}
