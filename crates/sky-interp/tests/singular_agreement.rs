//! The checker's "this matrix is singular" warning and `sky run`'s "the
//! matrix is singular" error are one verdict: both come from
//! `linalg::inverse`. This runs both on the same programs and compares.

use sky_ast::parse_module;
use sky_interp::{run_module, RuntimeEnv};
use sky_syntax::lex;

/// (the checker warned "singular", the run failed "singular").
fn verdicts(src: &str) -> (bool, bool) {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?}");
    let checked = sky_types::check_module(&module, &src);
    assert!(!checked.has_errors(), "{:#?}", checked.diagnostics);
    let warned = checked
        .diagnostics
        .iter()
        .any(|d| d.message.contains("singular"));
    let mut out = Vec::new();
    let failed = match run_module(&module, &src, &mut RuntimeEnv::new(), &mut out) {
        Ok(_) => false,
        Err(e) => {
            assert!(e.message.contains("singular"), "{}", e.message);
            true
        }
    };
    (warned, failed)
}

#[test]
fn check_and_run_agree_on_singular_matrices() {
    let cases = [
        "inv([[1, 2], [2, 4]])",
        "inv([[0, 0], [0, 0]])",
        // Numerically singular: elimination leaves a rounding-size pivot.
        "inv([[1, 2, 3], [4, 5, 6], [7, 8, 9]])",
        "A = [[1, 2], [2, 4]]\nA⁻¹ @ [1, 1]",
        "inv([[5]])",
        "inv([[0, 1], [1, 0]])",
        "inv([[1, 1], [1, 1.0000000001]])",
        "inv([[4, 7], [2, 6]])",
    ];
    for src in cases {
        let (warned, failed) = verdicts(src);
        assert_eq!(warned, failed, "checker and run disagree on {src:?}");
    }
    // And the verdicts are not both always false.
    assert_eq!(verdicts("inv([[1, 2], [2, 4]])"), (true, true));
    assert_eq!(verdicts("inv([[4, 7], [2, 6]])"), (false, false));
}
