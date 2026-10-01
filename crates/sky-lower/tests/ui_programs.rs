//! Every program the checker accepts in the repository's UI suites lowers,
//! or is rejected with a clear `LowerError` for exactly the constructs
//! `sky run` cannot execute either.

mod common;

use std::path::{Path, PathBuf};

use common::*;

fn ok_programs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut out = Vec::new();
    for dir in [
        "tests/ui/check",
        "tests/ui/types",
        "crates/sky-types/tests/ui",
    ] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(dir))
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.starts_with("ok_") && n.ends_with(".sky")
            })
            .collect();
        files.sort();
        out.extend(files);
    }
    out
}

/// Programs the checker accepts that `sky run` cannot follow (an input
/// declared inside a function), or that lowering cannot express yet:
/// rejected, with the reason. Everything else in the suites must lower.
const REJECTED: &[(&str, &str)] = &[
    ("ok_returns.sky", "top level of the file"),
    // SIR has no uncertain value: such programs run on the tree-walker.
    ("ok_uncertainty.sky", "uncertainty (`±`) is not supported"),
    (
        "ok_uncertainty_types.sky",
        "uncertainty (`±`) is not supported",
    ),
    (
        "ok_uncertainty_comparison_warning.sky",
        "uncertainty (`±`) is not supported",
    ),
    // Run on the tree-walker only (see the spec, "Differential equations").
    (
        "ok_equation.sky",
        "differential equations are not supported by SIR yet",
    ),
    (
        "ok_equation_types.sky",
        "differential equations are not supported by SIR yet",
    ),
];

#[test]
fn every_ok_program_lowers() {
    let files = ok_programs();
    assert!(files.len() >= 30, "found {} programs", files.len());
    let mut failures = Vec::new();
    let mut rejected = Vec::new();
    for path in &files {
        let src = std::fs::read_to_string(path).expect("reads");
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        match sky_lower::lower_source(&src) {
            Ok(m) => {
                check_structure(&m);
                assert_eq!(m.functions[0].name, sky_ir::Module::MAIN);
            }
            Err(e) => match REJECTED.iter().find(|(n, _)| *n == name) {
                Some((_, why)) if e.contains(why) => rejected.push(name.to_string()),
                _ => failures.push(format!("{}:\n{e}", path.display())),
            },
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n---\n"));
    assert_eq!(rejected.len(), REJECTED.len(), "{rejected:?}");
}

/// Programs that `sky run` executes fully have no inputs and no array
/// operations: their `<main>` takes no parameters.
#[test]
fn runnable_programs_have_no_inputs() {
    for name in [
        "tests/ui/check/ok_control_flow.sky",
        "tests/ui/check/ok_kinetic.sky",
        "tests/ui/check/ok_constants.sky",
        "tests/ui/check/ok_parses.sky",
        "tests/ui/check/ok_bool_literals.sky",
        "tests/ui/check/ok_shadow_constant.sky",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(name);
        let m = lower_ok(&std::fs::read_to_string(path).expect("reads"));
        assert!(main_fn(&m).params.is_empty(), "{name}");
    }
}

/// `lower_defs` (what `sky run` differentiates) builds exactly the functions
/// full lowering builds, and verifies, for every program in the suites; the
/// programs whose top level SIR cannot express still lower their `def`s.
#[test]
fn defs_only_lowering_builds_the_same_functions() {
    let mut compared = 0;
    for path in ok_programs() {
        let src = std::fs::read_to_string(&path).expect("reads");
        let lexed = sky_syntax::lex(&src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, &src);
        assert!(errors.is_empty(), "{}", path.display());
        let checked = sky_types::check_module(&ast, &src);
        let defs = sky_ast::with_deep_stack(|| sky_lower::lower_defs(&ast, &src, &checked));
        let whole = sky_ast::with_deep_stack(|| sky_lower::lower_with_defs(&ast, &src, &checked));
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let (defs, def_names) = match (defs, &whole) {
            (Ok(d), _) => d,
            // A `def` that cannot lower fails both ways.
            (Err(e), Err(_)) if name == "ok_returns.sky" => {
                assert!(e.message.contains("top level of the file"), "{}", e.message);
                continue;
            }
            (Err(e), _) => panic!("{}: {}", path.display(), e.message),
        };
        if let Err(errs) = sky_ir::verify(&defs) {
            panic!("{}: {}", path.display(), errs[0].message);
        }
        let Ok((whole, whole_names)) = &whole else {
            continue;
        };
        // Full lowering also builds the derivative functions `<main>` asks
        // for; `sky-derive` builds those itself. The `def`s must match.
        assert_eq!(&def_names, whole_names, "{}", path.display());
        for f in def_names.values() {
            let text = |m: &sky_ir::Module| m.function(f).map(sky_ir::print_function);
            assert_eq!(text(&defs), text(whole), "{}: `{f}`", path.display());
        }
        compared += 1;
    }
    assert!(compared >= 30, "compared {compared}");
}
