//! Uncertainty in the checker (spec "Uncertainty"): `±` is typed like `+`,
//! the type is the same with or without an uncertainty, and comparing an
//! uncertain value warns.

use sky_ast::with_deep_stack;
use sky_diag::Severity;
use sky_types::{check_module, Checked, Ty};
use sky_units::Dimension;

fn check(src: &str) -> Checked {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module(&module, src)
    })
}

fn messages(c: &Checked, severity: Severity) -> Vec<&str> {
    c.diagnostics
        .iter()
        .filter(|d| d.severity == severity)
        .map(|d| d.message.as_str())
        .collect()
}

const WARNING: &str = "comparing an uncertain value compares only its nominal value";

#[test]
fn plus_minus_has_the_type_of_its_value() {
    let c = check("m = 5.31 ± 0.02 kg\nt = 1.5 s ± 20 ms\nx = 2 ± 0.1\nv = m / t\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("m"), Some(Ty::scalar(Dimension::MASS)));
    assert_eq!(c.env.get("t"), Some(Ty::scalar(Dimension::TIME)));
    assert_eq!(c.env.get("x"), Some(Ty::scalar(Dimension::DIMENSIONLESS)));
}

#[test]
fn the_uncertainty_must_have_the_dimension_of_the_value() {
    let c = check("a = 5.31 kg ± 20 s\n");
    assert_eq!(
        messages(&c, Severity::Error),
        ["cannot use a time as the uncertainty of a mass"]
    );
    // The shared unit makes a shared dimension: no error.
    assert!(check("a = 5 ± 1 kg\n").diagnostics.is_empty());
    // A parenthesized number does not share it.
    let c = check("a = (5) ± 1 kg\n");
    assert_eq!(
        messages(&c, Severity::Error),
        ["cannot use a mass as the uncertainty of a plain number"]
    );
}

#[test]
fn constant_negative_uncertainty_and_non_numbers() {
    let c = check("a = 5 ± -0.1\nb = 5 ± -(2 m)\n");
    assert_eq!(
        messages(&c, Severity::Error)[0],
        "an uncertainty cannot be negative"
    );
    let c = check("a = true ± 1\n");
    assert_eq!(
        messages(&c, Severity::Error),
        ["`±` needs a number, found Bool"]
    );
    let c = check("a = [1, 2] ± 0.1\n");
    assert_eq!(messages(&c, Severity::Error), ["`±` needs scalar operands"]);
    // Not a constant: no error, it is the run time's.
    assert!(check("d = -1\na = 5 ± d\n").diagnostics.is_empty());
}

#[test]
fn nominal_and_uncertainty_keep_the_dimension() {
    let c =
        check("m = 5 ± 1 kg\nn = nominal(m)\nu = uncertainty(m)\nr = u / n\np = nominal(3 s)\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("n"), Some(Ty::scalar(Dimension::MASS)));
    assert_eq!(c.env.get("u"), Some(Ty::scalar(Dimension::MASS)));
    assert_eq!(c.env.get("r"), Some(Ty::scalar(Dimension::DIMENSIONLESS)));
    assert_eq!(c.env.get("p"), Some(Ty::scalar(Dimension::TIME)));
    let c = check("a = nominal()\n");
    assert_eq!(
        messages(&c, Severity::Error),
        ["`nominal` takes 1 argument, found 0"]
    );
    let c = check("a = uncertainty(true)\n");
    assert_eq!(
        messages(&c, Severity::Error),
        ["`uncertainty` needs a number, found Bool"]
    );
}

#[test]
fn a_program_may_use_the_names_itself() {
    // A user binding wins over the builtin, as for `print` and `inv`.
    let c = check("uncertainty = 0.02 kg\nx = uncertainty + 1 kg\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    let c = check("def nominal(x):\n    return 2 m\nn = nominal(1 s)\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("n"), Some(Ty::scalar(Dimension::LENGTH)));
}

#[test]
fn comparing_an_uncertain_value_warns_once() {
    let c = check("a = 5 ± 1 m\nb = a < 6 m\n");
    assert_eq!(messages(&c, Severity::Warning), [WARNING]);
    assert!(!c.has_errors());
    // Directly, without a name.
    let c = check("b = 5 ± 1 m == 5 m\n");
    assert_eq!(messages(&c, Severity::Warning), [WARNING]);
    // Through arithmetic and calls.
    let c = check("a = 5 ± 1 m\nb = sqrt(a * a) > 2 m\n");
    assert_eq!(messages(&c, Severity::Warning), [WARNING]);
    // Every comparison operator.
    for op in ["==", "!=", "<", "<=", ">", ">="] {
        let c = check(&format!("a = 5 ± 1 m\nb = a {op} 6 m\n"));
        assert_eq!(messages(&c, Severity::Warning), [WARNING], "{op}");
    }
    // A loop body is walked more than once but warns once.
    let c = check("a = 5 ± 1 m\nfor i in range(3):\n    b = a < 6 m\n");
    assert_eq!(messages(&c, Severity::Warning), [WARNING]);
    // Compound assignment makes a name uncertain.
    let c = check("a = 5 m\na += 1 ± 0.1 m\nb = a < 6 m\n");
    assert_eq!(messages(&c, Severity::Warning), [WARNING]);
}

#[test]
fn exact_comparisons_and_the_intended_question_do_not_warn() {
    assert!(check("a = 5 m\nb = a < 6 m\n").diagnostics.is_empty());
    // `nominal` and `uncertainty` are exact.
    let c = check("a = 5 ± 1 m\nb = nominal(a) < 6 m\nu = uncertainty(a) < 2 m\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    let c = check(
        "a = 5 ± 1 m\nb = 5.2 ± 0.3 m\nagree = abs(nominal(a - b)) < 2 * uncertainty(a - b)\n",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // A name rebound to an exact value is exact again.
    let c = check("a = 5 ± 1 m\na = 5 m\nb = a < 6 m\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}
