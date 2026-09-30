//! Gradients and derivatives of programs that use `a * b`, `a / b` and the
//! math functions of arrays, on every engine: the tree-walker (which asks
//! `sky-derive`, so `sky-ad`, for a derivative), the SIR interpreter, the SIR
//! after the default pipeline (which fuses the derivative's kernels) and
//! native code must print the same bytes. The numbers themselves are checked
//! against finite differences in `sky-opt`'s `ad_entrywise` test.

use sky_derive::SirDerivatives;
use sky_interp::{run_module_with_derivatives, RuntimeEnv};

fn run_all(src: &str) -> String {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:?}\n{src}");
    let checked = sky_types::check_module_with(&module, src, sky_types::Env::with_prelude());
    assert!(!checked.has_errors(), "the checker rejects:\n{src}");
    let solves = checked.cholesky_solves.clone();
    let ir = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    sky_ast::with_deep_stack(|| {
        let mut walker = Vec::new();
        let mut env = RuntimeEnv::with_prelude();
        env.add_chain_plans(&checked.chains);
        let mut derivatives = SirDerivatives::new(&module, src);
        run_module_with_derivatives(
            &module,
            src,
            &mut env,
            &mut walker,
            &solves,
            &mut derivatives,
        )
        .unwrap_or_else(|e| panic!("the tree-walker failed: {}\n{src}", e.message));
        let walker = String::from_utf8(walker).expect("utf-8");

        let mut optimized = ir.clone();
        sky_opt::default_pipeline()
            .verify_each(true)
            .run(&mut optimized)
            .expect("the pipeline runs");
        for (which, m) in [("lowered", &ir), ("optimized", &optimized)] {
            let mut out = Vec::new();
            sky_ir_interp::run(m, &mut out)
                .unwrap_or_else(|e| panic!("SIR ({which}) failed: {}\n{src}", e.message));
            assert_eq!(
                String::from_utf8(out).expect("utf-8"),
                walker,
                "SIR ({which})\n{src}"
            );
            let compiled = sky_jit::compile(m)
                .unwrap_or_else(|e| panic!("native code refused ({which}):\n{src}\n{e}"));
            let mut out = Vec::new();
            let (result, _) =
                compiled.run_or_rerun(&mut out, |w| sky_ir_interp::run(m, &mut { w }));
            result.unwrap_or_else(|e| panic!("native ({which}) failed: {}\n{src}", e.message));
            assert_eq!(
                String::from_utf8(out).expect("utf-8"),
                walker,
                "native ({which})\n{src}"
            );
        }
        walker
    })
}

#[test]
fn the_gradient_of_array_arithmetic_and_functions_is_the_same_on_every_engine() {
    let out = run_all(
        "def f(p: Vector[3]) -> Scalar:\n\
         \x20   q = sqrt(p * p + [1, 1, 1]) * [1, 2, 3] + exp(p) / (p * p + [2, 2, 2])\n\
         \x20   return atan2(q, p * p + [1, 1, 1]) @ hypot(q, p)\n\
         x = [0.3, -1, 2]\nprint(f(x))\nprint(grad(f)(x))\n",
    );
    assert_eq!(out.lines().count(), 2, "{out}");
}

#[test]
fn a_gradient_with_units_and_abs_is_the_same_on_every_engine() {
    let out = run_all(
        "def f(p: Vector[3] m) -> N:\n\
         \x20   r = sqrt(p * p + [1 m^2, 1 m^2, 1 m^2])\n\
         \x20   return [1 N/m, 2 N/m, 3 N/m] @ r + [1 N/m^2, 1 N/m^2, 1 N/m^2] @ (abs(p) * p / [1, 2, 4])\n\
         x = [0.3 m, -1 m, 2 m]\nprint(grad(f)(x))\n",
    );
    assert!(out.contains("kg/s^2"), "{out}");
}

#[test]
fn the_derivative_of_an_array_function_of_a_number_is_the_same_on_every_engine() {
    let out = run_all(
        "def r(t: s) -> Vector[3] m:\n\
         \x20   x = [1 m/s, 2 m/s, 3 m/s] * t\n\
         \x20   return sqrt(x * x + [1 m^2, 1 m^2, 1 m^2]) / [1, 2, 4]\n\
         print(derivative(r, t)(0.5 s))\n",
    );
    assert!(out.contains("m/s"), "{out}");
}
