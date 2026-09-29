//! The pass manager, driven by two test-only passes: one that does a real
//! (if trivial) rewrite and explains it, and one that breaks the IR.

mod common;

use common::*;
use sky_diag::{Severity, SourceFile};
use sky_ir::{
    verify, Changed, Function, FunctionBuilder, Module, OpKind, Pass, PassCx, PassError,
    PassManager, Type, ValueDef,
};

/// `neg(neg(x))` becomes `x`: every use of the outer result is redirected
/// to `x`. The dead `neg`s stay behind for a DCE pass to remove.
struct FoldDoubleNeg;

impl Pass for FoldDoubleNeg {
    fn name(&self) -> &'static str {
        "fold-double-neg"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        let mut changed = Changed::No;
        for f in &mut m.functions {
            let entry = f.entry();
            let ops = f.block(entry).ops.clone();
            for o in ops {
                let outer = f.op(o).clone();
                if outer.kind != OpKind::Neg {
                    continue;
                }
                let ValueDef::OpResult { op: inner_op, .. } = f.value(outer.operands[0]).def else {
                    continue;
                };
                let inner = f.op(inner_op).clone();
                if inner.kind != OpKind::Neg {
                    continue;
                }
                let (from, to) = (outer.results[0], inner.operands[0]);
                if !f.ops.iter().any(|op| op.operands.contains(&from)) {
                    continue; // dead: nothing to redirect
                }
                for op in &mut f.ops {
                    for operand in &mut op.operands {
                        if *operand == from {
                            *operand = to;
                        }
                    }
                }
                cx.explain(
                    "TEST-NEGNEG-001",
                    outer.span,
                    "--x",
                    "x",
                    vec!["negation is its own inverse".into()],
                );
                changed = Changed::Yes;
            }
        }
        changed
    }
}

/// Makes the module invalid: the returned value is one nobody defines.
struct Sabotage;

impl Pass for Sabotage {
    fn name(&self) -> &'static str {
        "sabotage"
    }

    fn run(&self, m: &mut Module, _cx: &mut PassCx) -> Changed {
        let f: &mut Function = &mut m.functions[0];
        let last = *f.block(f.entry()).ops.last().unwrap();
        f.ops[last.index()].operands = vec![sky_ir::ValueId(999)];
        Changed::Yes
    }
}

/// Does nothing.
struct Noop;

impl Pass for Noop {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn run(&self, _m: &mut Module, _cx: &mut PassCx) -> Changed {
        Changed::No
    }
}

/// `fn f(x: L) -> (L) { return -(-x) }`, the `--x` at bytes 10..13.
fn double_neg() -> Module {
    let mut b = FunctionBuilder::new("f", vec![len()], vec![len()], sp(0, 20));
    let x = b.params()[0];
    let n1 = b.op1(OpKind::Neg, vec![x], len(), sp(11, 13));
    let n2 = b.op1(OpKind::Neg, vec![n1], len(), sp(10, 13));
    b.op(OpKind::Return, vec![n2], vec![], sp(0, 9));
    module(vec![b.finish()])
}

#[test]
fn a_pipeline_runs_passes_in_order_and_collects_explanations() {
    let mut m = double_neg();
    let report = PassManager::for_tests()
        .with(Noop)
        .with(FoldDoubleNeg)
        .with(FoldDoubleNeg)
        .run(&mut m)
        .unwrap();
    let ran: Vec<_> = report.ran.iter().map(|(n, c)| (*n, c.is_yes())).collect();
    assert_eq!(
        ran,
        [
            ("noop", false),
            ("fold-double-neg", true),
            ("fold-double-neg", false)
        ]
    );
    assert_eq!(report.explanations.len(), 1);
    let e = &report.explanations[0];
    assert_eq!(e.pass, "fold-double-neg");
    assert_eq!(e.rule, "TEST-NEGNEG-001");
    assert_eq!((e.before.as_str(), e.after.as_str()), ("--x", "x"));
    assert_eq!(e.span, sp(10, 13));
    // The return now uses the parameter.
    let f = &m.functions[0];
    let ret = *f.block(f.entry()).ops.last().unwrap();
    assert_eq!(f.op(ret).operands, [f.param_values()[0]]);
    assert_eq!(report.omitted_explanations, 0);
}

#[test]
fn explanations_become_note_diagnostics() {
    let mut m = double_neg();
    let report = PassManager::for_tests()
        .with(FoldDoubleNeg)
        .run(&mut m)
        .unwrap();
    let d = report.explanations[0].to_diagnostic();
    assert_eq!(d.severity, Severity::Note);
    assert_eq!(d.primary.span, sp(10, 13));
    assert_eq!(d.message, "[TEST-NEGNEG-001] `--x` is rewritten to `x`");
    assert_eq!(
        d.notes,
        [
            "applied by the `fold-double-neg` pass",
            "because negation is its own inverse"
        ]
    );
    let src = "return -(-x)\n--x here";
    let out = d.render_to_string(&SourceFile::new("t.sky", src));
    assert!(out.contains("note: [TEST-NEGNEG-001]"), "{out}");
    assert!(
        out.contains("= note: because negation is its own inverse"),
        "{out}"
    );
}

#[test]
fn a_pass_that_breaks_the_ir_is_named() {
    let mut m = double_neg();
    let err = PassManager::new()
        .verify_each(true)
        .with(FoldDoubleNeg)
        .with(Sabotage)
        .with(Noop)
        .run(&mut m)
        .unwrap_err();
    let PassError::InvalidAfter { pass, errors } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(*pass, "sabotage");
    assert!(errors[0].message.contains("out of range"), "{errors:?}");
    let text = err.to_string();
    assert!(
        text.starts_with("pass `sabotage` produced invalid IR"),
        "{text}"
    );
    assert!(
        verify(&m).is_err(),
        "the module is left as the pass made it"
    );
}

#[test]
#[should_panic(expected = "pass `sabotage` produced invalid IR")]
fn for_tests_panics_with_the_pass_name() {
    let mut m = double_neg();
    let _ = PassManager::for_tests().with(Sabotage).run(&mut m);
}

#[test]
fn invalid_input_is_blamed_on_the_input_not_on_a_pass() {
    let mut m = double_neg();
    m.functions[0].ops[0].operands[0] = sky_ir::ValueId(500);
    let err = PassManager::new()
        .verify_each(true)
        .with(Noop)
        .run(&mut m)
        .unwrap_err();
    assert!(matches!(err, PassError::InvalidInput(_)), "{err:?}");
}

#[test]
fn verification_can_be_turned_off() {
    let mut m = double_neg();
    let report = PassManager::new()
        .verify_each(false)
        .with(Sabotage)
        .run(&mut m)
        .unwrap();
    assert_eq!(report.ran.len(), 1);
}

#[test]
fn the_ir_can_be_printed_after_a_chosen_pass() {
    let mut m = double_neg();
    let report = PassManager::for_tests()
        .with(Noop)
        .with(FoldDoubleNeg)
        .print_after("fold-double-neg")
        .run(&mut m)
        .unwrap();
    assert_eq!(report.printed.len(), 1);
    let (name, text) = &report.printed[0];
    assert_eq!(*name, "fold-double-neg");
    assert!(text.contains("return %0"), "{text}");
    assert_eq!(*text, sky_ir::print(&m));

    let mut m = double_neg();
    let all = PassManager::for_tests()
        .with(Noop)
        .with(FoldDoubleNeg)
        .print_after("*")
        .run(&mut m)
        .unwrap();
    assert_eq!(all.printed.len(), 2);
    assert!(
        all.printed[0].1.contains("return %2"),
        "{}",
        all.printed[0].1
    );
}

#[test]
fn explanations_are_capped_but_counted() {
    struct Chatty;
    impl Pass for Chatty {
        fn name(&self) -> &'static str {
            "chatty"
        }
        fn run(&self, _m: &mut Module, cx: &mut PassCx) -> Changed {
            for _ in 0..sky_ir::pass::MAX_EXPLANATIONS + 5 {
                cx.explain("TEST-CHAT-001", sp(0, 0), "a", "b", vec![]);
            }
            assert_eq!(cx.pass_name(), "chatty");
            Changed::Yes
        }
    }
    let mut m = module(vec![{
        let mut b = FunctionBuilder::new("f", vec![], vec![], sp(0, 0));
        b.op(OpKind::Return, vec![], vec![], sp(0, 0));
        b.finish()
    }]);
    let report = PassManager::for_tests().with(Chatty).run(&mut m).unwrap();
    assert_eq!(report.explanations.len(), sky_ir::pass::MAX_EXPLANATIONS);
    assert_eq!(report.omitted_explanations, 5);
}

#[test]
fn changed_combines() {
    assert!(Changed::No.or(Changed::Yes).is_yes());
    assert!(!Changed::No.or(Changed::No).is_yes());
    assert_eq!(Changed::from(true), Changed::Yes);
    let _: Type = Type::Bool;
}
