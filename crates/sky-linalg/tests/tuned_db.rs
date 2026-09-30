//! The kernels read what `sky tune` stored, and a stored value can only make
//! them faster or slower: even absurd ones (from a stale or hand-edited
//! database) are clamped, and the results stay the reference's, bit for bit.
//!
//! This is its own test binary because the kernels read the database once per
//! process, at their first use.

use sky_linalg::reference;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn random(n: usize, mut s: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

#[test]
fn stored_values_are_used_clamped_and_never_change_results() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("linalg-tuned-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("tune.db");
    let fp = sky_hw::current_fingerprint();
    let row = |kernel: &str, class: &str, params: &str, default: &str| {
        format!(
            "result fp={fp} kernel={kernel} size={class} params={params} default={default} best_ns=100 default_ns=130 when=1790000000\n"
        )
    };
    let mut text = String::from("sky-tune-db 1\n");
    // Absurd values for the class of order 65 to 256 ...
    text += &row(
        "gemm",
        "le256",
        "kc:1099511627776,mc:3,nc:5",
        "kc:256,mc:128,nc:1024",
    );
    text += &row("cholesky", "le256", "nb:9999", "nb:64");
    text += &row("threads", "all", "madds:1", "madds:2097152");
    // ... and a valid one for the smallest.
    text += &row(
        "gemm",
        "le64",
        "kc:128,mc:64,nc:1024",
        "kc:256,mc:128,nc:1024",
    );
    std::fs::write(&db, text).unwrap();
    std::env::set_var("SKY_TUNE_DB", &db);

    let n = 200;
    let (a, b) = (random(n * n, 1), random(n * n, 2));
    let mut spd = random(n * n, 3);
    for i in 0..n {
        for j in 0..i {
            spd[j * n + i] = spd[i * n + j];
        }
        spd[i * n + i] = spd[i * n + i].abs() + n as f64;
    }
    for threads in [1, 3] {
        assert_eq!(
            bits(&sky_linalg::gemm_with(&a, n, n, &b, n, threads)),
            bits(&reference::matmul(&a, n, n, &b, n)),
            "gemm, {threads} threads"
        );
        assert_eq!(
            sky_linalg::cholesky_with(&spd, n, threads).map(|l| bits(&l)),
            reference::cholesky(&spd, n).map(|l| bits(&l)),
            "cholesky, {threads} threads"
        );
        let got = sky_linalg::lu_with(&a, n, threads).unwrap();
        let want = reference::lu(&a, n).unwrap();
        assert_eq!(
            (bits(&got.lu), got.swaps),
            (bits(&want.lu), want.swaps),
            "lu, {threads} threads"
        );
    }
    // A small product takes the valid stored blocks.
    let m = 40;
    assert_eq!(
        bits(&sky_linalg::gemm_with(&a[..m * m], m, m, &b[..m * m], m, 2)),
        bits(&reference::matmul(&a[..m * m], m, m, &b[..m * m], m)),
    );

    // `--explain` cites what was tuned, for the sizes it was tuned for.
    let notes = sky_linalg::tuning::explain("gemm", Some(100));
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        notes[0].starts_with("GEMM blocks for sizes up to 256: tuned on this machine ("),
        "{notes:?}"
    );
    assert!(
        notes[0].contains("1.30x faster than the default"),
        "{notes:?}"
    );
    assert!(sky_linalg::tuning::explain("gemm", Some(500)).is_empty());
    // With no size, every tuned class: the small one and the one after it.
    let all = sky_linalg::tuning::explain("gemm", None);
    assert_eq!(all.len(), 2, "{all:?}");
    assert!(all[0].contains("up to 64") && all[1].contains("up to 256"));
    assert_eq!(sky_linalg::tuning::explain("cholesky", Some(100)).len(), 1);
    assert_eq!(sky_linalg::tuning::explain("threads", None).len(), 1);
    assert!(sky_linalg::tuning::explain("nonsense", None).is_empty());
}
