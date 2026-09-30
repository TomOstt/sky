//! Tests of the specialization pass: cloning, deduplication, limits,
//! recursion, unrolling, the explanations, and the run-time hook. Every test
//! that specializes also runs the program before and after and requires the
//! same output and the same whole diagnostic.

use sky_diag::SourceFile;
use sky_interp::Value;
use sky_ir::{Module, OpKind, PassManager, Type};
use sky_types::ShapeDim;
use sky_units::Dimension;

use super::*;
use crate::pipeline::SpecializePass;

fn lower(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"))
}

/// Stdout and the whole rendered error of running `m`.
fn run(m: &Module, src: &str) -> (String, Option<String>) {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        (
            String::from_utf8(out).expect("UTF-8"),
            r.err().map(|e| {
                e.to_diagnostic()
                    .render_to_string(&SourceFile::new("t.sky", src))
            }),
        )
    })
}

/// Lowers `src`, runs the pass alone (verified), and requires the same
/// behaviour as before. Returns the module and the pass's explanations.
fn specialized(src: &str) -> (Module, Vec<Explanation>) {
    let mut m = lower(src);
    let before = run(&m, src);
    let report = PassManager::for_tests()
        .with(SpecializePass)
        .run(&mut m)
        .expect("valid IR");
    assert_eq!(run(&m, src), before, "behaviour changed for:\n{src}");
    (m, report.explanations)
}

fn names(m: &Module) -> Vec<&str> {
    m.functions.iter().map(|f| f.name.as_str()).collect()
}

fn clones_of<'m>(m: &'m Module, base: &str) -> Vec<&'m str> {
    names(m)
        .into_iter()
        .filter(|n| n.starts_with(&format!("{base}[")))
        .collect()
}

fn count_kind(f: &Function, pred: impl Fn(&OpKind) -> bool) -> usize {
    reachable_ops(f)
        .into_iter()
        .filter(|o| pred(&f.op(*o).kind))
        .count()
}

fn rules(ex: &[Explanation]) -> Vec<&str> {
    ex.iter().map(|e| e.rule).collect()
}

const DOT: &str = "\
def dot(a: Vector[N], b: Vector[N]) -> Scalar:
    total = 0
    i = 0
    while i < len(a):
        total += a[i] * b[i]
        i += 1
    return total

u = [1, 2, 3]
v = [4, 5, 6]
print(dot(u, v))
";

#[test]
fn a_symbolic_size_becomes_concrete_and_the_loop_unrolls() {
    let (m, ex) = specialized(DOT);
    assert_eq!(clones_of(&m, "dot"), ["dot[N=3]"]);
    let clone = m.function("dot[N=3]").unwrap();
    for p in &clone.params {
        let Type::Number(n) = p else { panic!() };
        assert_eq!(n.shape.dims(), [ShapeDim::Known(3)]);
    }
    assert_eq!(count_kind(clone, |k| matches!(k, OpKind::While)), 0);
    assert_eq!(count_kind(clone, |k| matches!(k, OpKind::Len)), 0);
    assert_eq!(count_kind(clone, |k| matches!(k, OpKind::Index)), 6);
    // `<main>` calls the clone, and nothing calls the generic function any
    // more, so it is gone.
    assert!(m.function("dot").is_none());
    let main = m.function(Module::MAIN).unwrap();
    let callees: Vec<&str> = reachable_ops(main)
        .into_iter()
        .filter_map(|o| match &main.op(o).kind {
            OpKind::Call { callee } => Some(callee.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(callees, ["dot[N=3]"]);
    // The last `i += 1` of the unrolled loop is dead, and dead user code is
    // reported (`dce`'s rule, filed under this pass).
    assert_eq!(rules(&ex), [RULE_SPECIALIZED, RULE_UNROLLED, "DCE-001"]);
}

#[test]
fn the_specialization_note_reads_like_the_design() {
    let (_, ex) = specialized(DOT);
    let head = specialize_headline(&ex[0]).unwrap();
    assert_eq!(
        head,
        "[SPEC-001] specialized `dot` for N = 3 at this call: 2 checks removed"
    );
    let head = specialize_headline(&ex[1]).unwrap();
    assert_eq!(
        head,
        "[SPEC-003] this loop runs exactly 3 times in `dot[N=3]`, so it is unrolled"
    );
    assert!(ex[0]
        .because
        .iter()
        .any(|b| b == "the argument for `a` is a Vector[3], so N = 3"));
}

#[test]
fn the_clone_keeps_the_spans_of_the_operations_it_copies() {
    let (m, _) = specialized(DOT);
    let unspecialized = lower(DOT);
    let generic = unspecialized.function("dot").unwrap();
    let clone = m.function("dot[N=3]").unwrap();
    let spans = |f: &Function, pred: fn(&OpKind) -> bool| {
        let mut v: Vec<_> = reachable_ops(f)
            .into_iter()
            .filter(|o| pred(&f.op(*o).kind))
            .map(|o| (f.op(o).span, f.op(o).operand_spans.clone()))
            .collect();
        v.sort_by_key(|(s, _)| (s.start, s.end));
        v.dedup();
        v
    };
    for pred in [
        (|k: &OpKind| matches!(k, OpKind::Mul)) as fn(&OpKind) -> bool,
        |k| matches!(k, OpKind::Index),
    ] {
        assert_eq!(spans(clone, pred), spans(generic, pred));
    }
}

#[test]
fn constants_are_baked_in_and_calls_lose_the_operand() {
    let src = "\
def scale(k: Scalar, x: Scalar) -> Scalar:
    if k > 1:
        return x * k
    return x / (k + 1)

print(scale(2, 3), scale(0, 4))
";
    let (m, ex) = specialized(src);
    assert_eq!(clones_of(&m, "scale"), ["scale[k=2,x=3]", "scale[k=0,x=4]"]);
    let c = m.function("scale[k=2,x=3]").unwrap();
    assert!(c.params.is_empty());
    // 3 * 2 folded to the constant 6: the whole body is a constant.
    assert_eq!(
        c.block(c.entry()).ops.len(),
        2,
        "{}",
        sky_ir::print_function(c)
    );
    let main = m.function(Module::MAIN).unwrap();
    for o in reachable_ops(main) {
        if matches!(main.op(o).kind, OpKind::Call { .. }) {
            assert!(main.op(o).operands.is_empty());
        }
    }
    assert_eq!(rules(&ex), [RULE_SPECIALIZED, RULE_SPECIALIZED]);
}

#[test]
fn a_constant_the_analysis_proves_is_a_fact() {
    let src = "\
def half(x: Scalar) -> Scalar:
    return x / 2

print(half(1 + 2 * 3))
";
    let (m, _) = specialized(src);
    assert_eq!(clones_of(&m, "half"), ["half[x=7]"]);
}

#[test]
fn equal_facts_share_one_clone() {
    let src = "\
def twice(x: Scalar) -> Scalar:
    return x + x

print(twice(2), twice(1 + 1), twice(3))
";
    let (m, ex) = specialized(src);
    assert_eq!(clones_of(&m, "twice"), ["twice[x=2]", "twice[x=3]"]);
    assert_eq!(
        rules(&ex),
        [RULE_SPECIALIZED, RULE_REUSED, RULE_SPECIALIZED]
    );
    assert_eq!(
        specialize_headline(&ex[1]).unwrap(),
        "[SPEC-002] this call uses the existing specialization `twice[x=2]`"
    );
}

#[test]
fn a_fact_about_an_unused_parameter_is_no_fact() {
    let src = "\
def first(x: Scalar, unused: Scalar) -> Scalar:
    return x * 2

print(first(1, 5), first(1, 6))
";
    let (m, _) = specialized(src);
    assert_eq!(clones_of(&m, "first"), ["first[x=1]"]);
}

#[test]
fn a_function_gets_at_most_eight_clones() {
    let mut src = String::from("def k(x: Scalar) -> Scalar:\n    return x * 2 + 1\n\n");
    for i in 1..=10 {
        src.push_str(&format!("print(k({i}))\n"));
    }
    let (m, ex) = specialized(&src);
    assert_eq!(clones_of(&m, "k").len(), MAX_CLONES_PER_FUNCTION);
    // The ninth and tenth calls still call the generic function, and the
    // limit is said once.
    let main = m.function(Module::MAIN).unwrap();
    let generic_calls = reachable_ops(main)
        .into_iter()
        .filter(|o| matches!(&main.op(*o).kind, OpKind::Call { callee } if callee == "k"))
        .count();
    assert_eq!(generic_calls, 2);
    assert_eq!(rules(&ex).iter().filter(|r| **r == RULE_LIMIT).count(), 1);
    let limit = ex.iter().find(|e| e.rule == RULE_LIMIT).unwrap();
    assert_eq!(
        specialize_headline(limit).unwrap(),
        "[SPEC-004] this call of `k` is not specialized: `k` already has 8 specializations, \
         the most one function gets"
    );
}

#[test]
fn a_large_body_is_not_cloned() {
    let mut src = String::from("def big(x: Scalar) -> Scalar:\n    y = x\n");
    for _ in 0..200 {
        src.push_str("    y = y * 1.0001 + 1\n");
    }
    src.push_str("    return y\n\nprint(big(2))\n");
    let (m, ex) = specialized(&src);
    assert!(clones_of(&m, "big").is_empty());
    assert_eq!(rules(&ex), [RULE_LIMIT]);
    assert!(ex[0].because[0].contains("only bodies of at most 300"));
}

#[test]
fn recursion_on_a_constant_stops_at_the_base_case() {
    let src = "\
def fact(n: Scalar) -> Scalar:
    if n <= 1:
        return 1
    return n * fact(n - 1)

print(fact(5))
";
    let (m, _) = specialized(src);
    let c = clones_of(&m, "fact");
    assert_eq!(
        c,
        [
            "fact[n=5]",
            "fact[n=4]",
            "fact[n=3]",
            "fact[n=2]",
            "fact[n=1]"
        ]
    );
    // `fact[n=1]` is the base case: no call left in it.
    let base = m.function("fact[n=1]").unwrap();
    assert_eq!(count_kind(base, |k| matches!(k, OpKind::Call { .. })), 0);
}

#[test]
fn recursion_that_never_ends_is_bounded_by_the_limit() {
    // The chain of constants 20, 19, ... would go on to the base case; the
    // limit stops it after eight, and the last one calls the generic
    // function.
    let src = "\
def fact(n: Scalar) -> Scalar:
    if n <= 1:
        return 1
    return n * fact(n - 1)

print(fact(20))
";
    let (m, _) = specialized(src);
    assert_eq!(clones_of(&m, "fact").len(), MAX_CLONES_PER_FUNCTION);
    let last = m.function("fact[n=13]").unwrap();
    let callees: Vec<String> = reachable_ops(last)
        .into_iter()
        .filter_map(|o| match &last.op(o).kind {
            OpKind::Call { callee } => Some(callee.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(callees, ["fact"]);
}

#[test]
fn unbounded_recursion_still_reports_the_same_error() {
    let src = "\
def r(n: Scalar) -> Scalar:
    return r(n + 1)

print(r(0))
";
    // `specialized` requires the same output and whole diagnostic; the
    // label names `r`, not `r[n=..]`.
    let (m, _) = specialized(src);
    let (_, err) = run(&m, src);
    let err = err.expect("recursion is too deep");
    assert!(err.contains("recursion is too deep"), "{err}");
    assert!(err.contains("calling `r` here"), "{err}");
}

#[test]
fn a_recursive_call_on_the_same_shape_finds_its_own_clone() {
    // g calls itself on the vector it was given, with a bound it cannot
    // know: the clone `g[N=3]` must call itself, not make `g[N=3][N=3]`.
    let text = "\
sir edition 2026

fn @<main>() -> () effects(io, fail, call) {
  %0 = const.float 1 : num[1; ()]
  %1 = const.float 2 : num[1; ()]
  %2 = const.float 3 : num[1; ()]
  %v = array.literal %0, %1, %2 : num[1; (3)]
  %k = input \"k\" : num[1; ()]
  %r = call @g(%v, %k) : num[1; ()]
  print %r
  return
}

fn @g(%v: num[1; (N#0)], %k: num[1; ()]) -> (num[1; ()]) effects(fail, call) {
  %n = len %v : num[1; ()]
  %c = cmp.lt %k, %n : bool
  %r = if %c : num[1; ()] {
    ^1:
      %z = const.float 0 : num[1; ()]
      yield %z
  } {
    ^2:
      %s = call @g(%v, %k) : num[1; ()]
      yield %s
  }
  return %r
}
";
    let mut m = sky_ir::parse(text).expect("parses");
    let report = PassManager::for_tests()
        .with(SpecializePass)
        .run(&mut m)
        .expect("valid");
    assert_eq!(clones_of(&m, "g"), ["g[N=3]"]);
    let g3 = m.function("g[N=3]").unwrap();
    let callees: Vec<String> = reachable_ops(g3)
        .into_iter()
        .filter_map(|o| match &g3.op(o).kind {
            OpKind::Call { callee } => Some(callee.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(callees, ["g[N=3]"]);
    assert_eq!(rules(&report.explanations), [RULE_SPECIALIZED, RULE_REUSED]);
}

#[test]
fn sizes_flow_down_a_chain_of_calls() {
    let src = "\
def norm2(v: Vector[N]) -> Scalar:
    return v @ v

def wrap(w: Vector[M]) -> Scalar:
    return norm2(w) + 1

print(wrap([1, 2, 3]))
";
    let (m, _) = specialized(src);
    assert_eq!(clones_of(&m, "wrap"), ["wrap[M=3]"]);
    assert_eq!(clones_of(&m, "norm2"), ["norm2[N=3]"]);
    let w = m.function("wrap[M=3]").unwrap();
    let callee = reachable_ops(w)
        .into_iter()
        .find_map(|o| match &w.op(o).kind {
            OpKind::Call { callee } => Some(callee.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(callee, "norm2[N=3]");
}

#[test]
fn errors_survive_specialization() {
    // A failing division, an out-of-range index, a failing loop body: the
    // same diagnostic, with the same labels, as before.
    for src in [
        "def f(x: Scalar, d: Scalar) -> Scalar:\n    return x / d\n\nprint(f(1, 0))\n",
        "def at(v: Vector[N], i: Scalar) -> Scalar:\n    return v[i]\n\nprint(at([1, 2, 3], 5))\n",
        "def g(n: Scalar) -> Scalar:\n    s = 10\n    i = 0\n    while i < n:\n        s = s / (2 - i)\n        i += 1\n    return s\n\nprint(g(4))\n",
        "def h(x: Scalar) -> Scalar:\n    return sqrt(x)\n\nprint(1)\nprint(h(-4))\n",
    ] {
        let (m, _) = specialized(src);
        let (out, err) = run(&m, src);
        assert!(err.is_some(), "{src}\n{out}");
    }
}

#[test]
fn a_loop_of_eight_iterations_unrolls_and_nine_does_not() {
    let src = "\
def f(n: Scalar) -> Scalar:
    s = 0
    i = 0
    while i < n:
        s += i * 2
        i += 1
    return s

print(f(8), f(9), f(0))
";
    let (m, ex) = specialized(src);
    let whiles = |name: &str| count_kind(m.function(name).unwrap(), |k| matches!(k, OpKind::While));
    assert_eq!(whiles("f[n=8]"), 0);
    assert_eq!(whiles("f[n=9]"), 1);
    assert_eq!(whiles("f[n=0]"), 0);
    assert_eq!(
        rules(&ex).iter().filter(|r| **r == RULE_UNROLLED).count(),
        2,
        "{ex:#?}"
    );
}

#[test]
fn nested_loops_unroll_together() {
    let src = "\
def tri(n: Scalar) -> Scalar:
    s = 0
    i = 0
    while i < n:
        j = 0
        while j < i:
            s += 1
            j += 1
        i += 1
    return s

print(tri(4))
";
    let (m, _) = specialized(src);
    let f = m.function("tri[n=4]").unwrap();
    assert_eq!(count_kind(f, |k| matches!(k, OpKind::While)), 0);
}

#[test]
fn a_loop_with_an_unknown_bound_is_left_alone() {
    let src = "\
z: Scalar
def f(n: Scalar, m: Scalar) -> Scalar:
    s = 0
    i = 0
    while i < m:
        s += n
        i += 1
    return s

print(f(3, abs(z)))
";
    let (m, _) = specialized(src);
    let f = m.function("f[n=3]").unwrap();
    assert_eq!(count_kind(f, |k| matches!(k, OpKind::While)), 1);
}

#[test]
fn a_known_sign_decides_a_branch_and_names_the_clone() {
    let src = "\
z: Scalar
def g(x: Scalar) -> Scalar:
    if x > 0:
        return 1
    return 2

print(g(abs(z) + 1))
";
    let (m, ex) = specialized(src);
    assert_eq!(clones_of(&m, "g"), ["g[x>0]"]);
    let c = m.function("g[x>0]").unwrap();
    assert_eq!(count_kind(c, |k| matches!(k, OpKind::Cmp(_))), 0);
    assert_eq!(count_kind(c, |k| matches!(k, OpKind::If)), 0);
    assert_eq!(
        specialize_headline(&ex[0]).unwrap(),
        "[SPEC-001] specialized `g` for x > 0 at this call: 0 checks removed"
    );
}

#[test]
fn a_sign_that_decides_nothing_makes_no_clone() {
    let src = "\
z: Scalar
def g(x: Scalar) -> Scalar:
    return x * 2

print(g(abs(z) + 1))
";
    let (m, ex) = specialized(src);
    assert!(clones_of(&m, "g").is_empty());
    assert!(ex.is_empty());
}

#[test]
fn a_negative_sign_and_a_positive_sign_are_two_clones() {
    let src = "\
z: Scalar
def g(x: Scalar) -> Scalar:
    if x > 0:
        return 1
    return 2

print(g(abs(z) + 1))
print(g(-1 - abs(z)))
";
    let (m, _) = specialized(src);
    assert_eq!(clones_of(&m, "g"), ["g[x>0]", "g[x<0]"]);
}

#[test]
fn the_second_run_of_the_pass_finds_nothing_new() {
    let mut m = lower(DOT);
    let again = |m: &mut Module| specialize(m).len();
    assert!(again(&mut m) > 0);
    let functions = m.functions.len();
    assert_eq!(again(&mut m), 0);
    assert_eq!(m.functions.len(), functions);
}

#[test]
fn an_unknown_typed_parameter_gets_no_constant() {
    // `x` has no annotation: its operations check their operands at run
    // time, so a typed constant must not replace it.
    let src = "def f(x):\n    return x + 1\n\nprint(f(2))\n";
    let (m, ex) = specialized(src);
    assert!(clones_of(&m, "f").is_empty());
    assert!(ex.is_empty());
}

#[test]
fn a_literal_vector_entry_is_read_directly() {
    let src = "\
def pick(k: Scalar) -> Scalar:
    a = [10, 20, 30]
    return a[k]

print(pick(1))
";
    let (m, _) = specialized(src);
    let c = m.function("pick[k=1]").unwrap();
    assert_eq!(
        count_kind(c, |k| matches!(k, OpKind::Index)),
        0,
        "{}",
        sky_ir::print_function(c)
    );
}

// ----- the run-time hook -----

fn vector(n: usize) -> Value {
    Value::array(
        vec![n],
        (1..=n).map(|i| i as f64).collect(),
        Dimension::DIMENSIONLESS,
    )
}

fn call(m: &Module, name: &str, args: Vec<Value>) -> String {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        match sky_ir_interp::call(m, name, args, &mut out) {
            Ok(v) => format!("{v:?}"),
            Err(e) => format!("error: {}", e.message),
        }
    })
}

#[test]
fn specialize_for_adds_a_verified_clone_and_leaves_the_rest() {
    let m = lower(DOT);
    let args = [Some(vector(3)), Some(vector(3))];
    let name = specialized_name(&m, "dot", &args);
    assert_eq!(name, "dot[N=3]");
    let m2 = specialize_for(&m, "dot", &args);
    sky_ir::verify(&m2).expect("valid");
    assert_eq!(m2.function("dot"), m.function("dot"));
    let clone = m2.function("dot[N=3]").unwrap();
    assert_eq!(clone.params.len(), 2);
    assert_eq!(count_kind(clone, |k| matches!(k, OpKind::While)), 0);
    // Same results as the generic function, for every 3-vector.
    let of = |xs: [f64; 3]| Value::array(vec![3], xs.to_vec(), Dimension::DIMENSIONLESS);
    for (a, b) in [
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([0.5, -2.0, 1e300], [1e10, 3.0, 0.0]),
        ([1e300, 1.0, 1.0], [1e300, 1.0, 1.0]),
    ] {
        let args = vec![of(a), of(b)];
        assert_eq!(call(&m2, "dot[N=3]", args.clone()), call(&m2, "dot", args));
    }
}

#[test]
fn specialize_for_values_keeps_the_signature() {
    let src = "\
def scale(k: Scalar, x: Scalar) -> Scalar:
    if k > 1:
        return x * k
    return x / (k + 1)

print(scale(2, 3))
";
    let m = lower(src);
    let args = [Some(Value::scalar(2.0)), None];
    assert_eq!(specialized_name(&m, "scale", &args), "scale[k=2]");
    let m2 = specialize_for(&m, "scale", &args);
    sky_ir::verify(&m2).expect("valid");
    let clone = m2.function("scale[k=2]").unwrap();
    assert_eq!(clone.params.len(), 2, "the signature is unchanged");
    for x in [3.0, -7.5, 0.0, 1e300] {
        let args = vec![Value::scalar(2.0), Value::scalar(x)];
        assert_eq!(
            call(&m2, "scale[k=2]", args.clone()),
            call(&m2, "scale", args),
            "x = {x}"
        );
    }
    // Nothing known, nothing done.
    assert_eq!(specialized_name(&m, "scale", &[None, None]), "scale");
    assert_eq!(specialize_for(&m, "scale", &[None, None]), m);
    assert_eq!(specialized_name(&m, "nope", &[]), "nope");
}

#[test]
fn a_cache_specializes_once_per_fact_set() {
    let mut cache = SpecializationCache::new(lower(DOT));
    let args = [Some(vector(3)), None];
    let first = cache.get("dot", &args).expect("specialized");
    let again = cache.get("dot", &args).expect("cached");
    assert!(std::sync::Arc::ptr_eq(&first, &again));
    assert_eq!(cache.stats(), (1, 1));
    let four = cache.get("dot", &[Some(vector(4)), None]).unwrap();
    assert_eq!(four.name, "dot[N=4]");
    assert_eq!(cache.len(), 2);
    assert!(cache.get("dot", &[None, None]).is_none());
}

#[test]
fn a_cache_is_bounded() {
    let src = "def f(x: Scalar) -> Scalar:\n    return x * 3\n";
    let mut cache = SpecializationCache::new(lower(src));
    for i in 0..(SpecializationCache::CAPACITY + 10) {
        cache
            .get("f", &[Some(Value::scalar(i as f64 + 1.0))])
            .unwrap();
    }
    assert_eq!(cache.len(), SpecializationCache::CAPACITY);
}

#[test]
fn a_loop_body_the_folder_changes_while_unrolling_still_unrolls() {
    // `n > 0` is a constant in the clone: the folder inlines that `if` inside
    // the loop's own region while it decides the loop condition, so the
    // region's operation list is not the one it started with. (This was a
    // bug: the unroller copied a stale list, including the dead `if`, and
    // produced invalid IR. Found by the fuzz campaign, seed 18874 of
    // `fuzz_specialize`.)
    let src = "\
def f(n: Scalar) -> Scalar:
    s = 0
    i = 0
    while i < n:
        if n > 0:
            s += 1
        else:
            s += 100
        i += 1
    return s

print(f(3))
";
    let (m, _) = specialized(src);
    let c = m.function("f[n=3]").unwrap();
    assert_eq!(count_kind(c, |k| matches!(k, OpKind::While)), 0);
    assert_eq!(count_kind(c, |k| matches!(k, OpKind::If)), 0);
}

#[test]
fn bounds_checks_on_constant_indices_are_proven_and_out_of_range_ones_stay() {
    let src = "\
def pick(v: Vector[N], i: Scalar) -> Scalar:
    return v[i]

print(pick([1, 2, 3], 1))
print(pick([1, 2, 3], 5))
";
    // The second call fails, in the clone, with the error the generic
    // function gave (`specialized` compares whole diagnostics).
    let (m, _) = specialized(src);
    let proof = |name: &str| {
        let f = m.function(name).unwrap();
        let an = exact::analyze(f);
        reachable_ops(f)
            .into_iter()
            .find(|o| matches!(f.op(*o).kind, OpKind::Index))
            .map(|o| an.proof(o))
    };
    assert_eq!(proof("pick[N=3,i=1]"), Some(Some(exact::Proof::InBounds)));
    assert_eq!(proof("pick[N=3,i=5]"), Some(None));
}
