//! The interpreter reads the thresholds `sky tune` stored (clamped), and a
//! run with them prints what a run with the defaults prints.
//!
//! Its own test binary: the thresholds are read once per process.

use sky_ir::parse;
use sky_ir_interp::{JitOptions, ParConfig, Value};

const PROGRAM: &str = "sir edition 2026

fn @f(%a: num[1; (2000)], %b: num[1; (2000)], %k: num[1; ()]) -> (num[1; (2000)]) effects(fail) {
  %r = fused %a, %b, %k : num[1; (2000)] {
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
      %t = add %x, %y : num[1; ()]
      %u = mul %t, %s : num[1; ()]
      yield %u
  }
  %q = add %r, %b : num[1; (2000)]
  return %q
}
";

fn args() -> Vec<Value> {
    let xs: Vec<f64> = (0..2000).map(|i| i as f64 * 0.25 + 1.0).collect();
    let ys: Vec<f64> = (0..2000).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let dim = sky_units::Dimension::DIMENSIONLESS;
    vec![
        Value::array(vec![2000], xs, dim),
        Value::array(vec![2000], ys, dim),
        Value::scalar(1.5),
    ]
}

#[test]
fn stored_thresholds_are_used_clamped_and_change_nothing() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("interp-tuned-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("tune.db");
    let fp = sky_hw::current_fingerprint();
    let row = |kernel: &str, class: &str, params: &str, default: &str| {
        format!(
            "result fp={fp} kernel={kernel} size={class} params={params} default={default} best_ns=1 default_ns=2 when=1790000000\n"
        )
    };
    let mut text = String::from("sky-tune-db 1\n");
    text += &row(
        "fused_native",
        "cold",
        "min_elements:256",
        "min_elements:512",
    );
    text += &row("par_fused_work", "all", "work:1", "work:4096");
    // An absurd stored value: at least two elements are needed to split.
    text += &row("par_array_elements", "all", "elements:0", "elements:262144");
    std::fs::write(&db, text).unwrap();
    std::env::set_var("SKY_TUNE_DB", &db);

    assert_eq!(sky_jit::tuning::kernel_threshold(), 256);
    assert_eq!(JitOptions::from_env().min_elements, 256);
    assert_eq!(sky_ir::parallel::tuned(), (1, 2));
    let cfg = ParConfig::from_env();
    assert_eq!((cfg.fused_work, cfg.array_elements), (1, 2));
    // `with_threads` stays the measured defaults.
    assert_eq!(ParConfig::with_threads(4).fused_work, 4096);

    let m = parse(PROGRAM).unwrap();
    let show = |cfg: &ParConfig, jit| {
        let mut out = Vec::new();
        let r = sky_ast::with_deep_stack(|| {
            sky_ir_interp::call_with_options(&m, "f", args(), &mut out, cfg, jit)
        })
        .unwrap();
        format!("{r:?}")
    };
    let tuned = show(&ParConfig::tuned(4), JitOptions::from_env());
    assert_eq!(tuned, show(&ParConfig::sequential(), JitOptions::OFF));
    assert_eq!(tuned, show(&ParConfig::with_threads(4), JitOptions::ALWAYS));
}
