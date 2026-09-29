//! Blocks and control flow: `if`/`elif`/`else`, `while`, `def`, `return`,
//! `pass`/`break`/`continue`, and the conditional expression. Trees,
//! spans, recovery, and the depth limit.

mod common;

use common::{at, lex, module_errors, module_sexpr, sexpr};
use sky_ast::{parse_module, with_deep_stack, ExprKind, StmtKind, MAX_DEPTH};

fn messages(src: &str) -> Vec<String> {
    module_errors(src).into_iter().map(|e| e.message).collect()
}

// ----- structure -----

#[test]
fn if_elif_else_chain() {
    let src = "if a:\n    x = 1\nelif b:\n    x = 2\nelif c:\n    x = 3\nelse:\n    x = 4\ny = 5\n";
    assert_eq!(
        module_sexpr(src),
        "(if a (block (= x 1)) b (block (= x 2)) c (block (= x 3)) (else (block (= x 4))))\n(= y 5)"
    );
}

#[test]
fn dedent_closes_several_levels_at_once() {
    let src = "while a:\n    if b:\n        while c:\n            pass\ny = 1\n";
    assert_eq!(
        module_sexpr(src),
        "(while a (block (if b (block (while c (block (pass)))))))\n(= y 1)"
    );
}

#[test]
fn a_block_may_end_at_end_of_input_without_a_newline() {
    assert_eq!(module_sexpr("if a:\n    pass"), "(if a (block (pass)))");
    assert_eq!(
        module_sexpr("if a:\n    if b:\n        x = 1"),
        "(if a (block (if b (block (= x 1)))))"
    );
}

#[test]
fn blank_lines_and_comments_do_not_end_blocks() {
    let src = "if a:\n    x = 1\n\n# note\n        # deeper note\n    y = 2\n";
    assert_eq!(module_sexpr(src), "(if a (block (= x 1) (= y 2)))");
}

#[test]
fn tabs_and_crlf_work_like_spaces() {
    assert_eq!(
        module_sexpr("if a:\r\n\tx = 1\r\n\ty = 2\r\nz = 3\r\n"),
        "(if a (block (= x 1) (= y 2)))\n(= z 3)"
    );
}

#[test]
fn def_forms() {
    let src = "def f():\n    pass\ndef g(a, b: Vector[3],) -> Matrix[3, 3]:\n    return a\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params) (block (pass)))\n\
         (def g (params a (: b (type (Vector 3)))) (returns (type (Matrix 3 3))) (block (return a)))"
    );
}

#[test]
fn return_forms() {
    let src = "def f(x):\n    return\ndef g(x):\n    return x + 1\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params x) (block (return)))\n(def g (params x) (block (return (+ x 1))))"
    );
}

#[test]
fn keywords_are_still_attribute_names() {
    assert_eq!(sexpr("a.while"), "(attr a while)");
    assert_eq!(
        sexpr("a.pass.break.continue.elif"),
        "(attr (attr (attr (attr a pass) break) continue) elif)"
    );
}

#[test]
fn a_block_can_hold_annotations_and_compound_assignment() {
    let src = "def f():\n    A: Matrix[3, 3] = B\n    A @= C\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params) (block (: A (type (Matrix 3 3)) B) (@= A C)))"
    );
}

// ----- the conditional expression -----

#[test]
fn conditional_is_looser_than_or_and_right_associative() {
    assert_eq!(sexpr("a if c else b"), "(ifelse c a b)");
    assert_eq!(
        sexpr("a or b if c and d else e"),
        "(ifelse (and c d) (or a b) e)"
    );
    assert_eq!(
        sexpr("a if c else b if d else e"),
        "(ifelse c a (ifelse d b e))"
    );
    assert_eq!(sexpr("not a if c else b"), "(ifelse c (not a) b)");
    assert_eq!(sexpr("a if not c else b"), "(ifelse (not c) a b)");
    assert_eq!(sexpr("a < b if c else d < e"), "(ifelse c (< a b) (< d e))");
    assert_eq!(sexpr("1 + 2 if c else 3 * 4"), "(ifelse c (+ 1 2) (* 3 4))");
    // A conditional operand of a tighter operator needs parentheses.
    assert_eq!(sexpr("(a if c else b) + 1"), "(+ (paren (ifelse c a b)) 1)");
    assert_eq!(sexpr("f(a if c else b, d)"), "(call f (ifelse c a b) d)");
    assert_eq!(sexpr("x[a if c else b]"), "(index x (ifelse c a b))");
}

#[test]
fn conditional_condition_cannot_be_a_bare_conditional() {
    // `a if b if c else d else e`: the condition is an `or` expression.
    let errors = messages("x = a if b if c else d else e\n");
    assert!(!errors.is_empty());
}

#[test]
fn conditional_needs_else() {
    let src = "x = 1 if c\n";
    let errors = module_errors(src);
    assert_eq!(errors.len(), 1);
    assert_eq!(
        errors[0].message,
        "expected `else` in the conditional expression, found end of line"
    );
    assert_eq!(at(src, errors[0].secondary[0].span), "if");
}

#[test]
fn conditional_span_covers_all_three_parts() {
    let src = "y = (a) if c else (b)\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty());
    let StmtKind::Assign { value, .. } = &m.stmts[0].kind else {
        panic!("not an assignment");
    };
    assert!(matches!(value.kind, ExprKind::IfElse { .. }));
    assert_eq!(at(src, value.span), "(a) if c else (b)");
}

// ----- spans -----

#[test]
fn statement_and_block_spans() {
    let src = "if a:\n    x = 1\n    y = 2\nelse:\n    z = 3\nw = 4\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty());
    assert_eq!(m.stmts.len(), 2);
    assert_eq!(
        at(src, m.stmts[0].span),
        "if a:\n    x = 1\n    y = 2\nelse:\n    z = 3"
    );
    let StmtKind::If {
        branches,
        else_block,
    } = &m.stmts[0].kind
    else {
        panic!("not an if");
    };
    assert_eq!(at(src, branches[0].0.span), "a");
    assert_eq!(at(src, branches[0].1.span), "x = 1\n    y = 2");
    assert_eq!(at(src, else_block.as_ref().unwrap().span), "z = 3");
    assert_eq!(at(src, m.stmts[1].span), "w = 4");
}

#[test]
fn def_spans() {
    let src = "def f(a, b: Vector[3]) -> Vector[3]:\n    return b\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty());
    let StmtKind::Def {
        name,
        params,
        ret,
        body,
    } = &m.stmts[0].kind
    else {
        panic!("not a def");
    };
    assert_eq!(at(src, name.span), "f");
    assert_eq!(at(src, params[0].span), "a");
    assert_eq!(at(src, params[1].span), "b: Vector[3]");
    assert_eq!(at(src, params[1].name.span), "b");
    assert_eq!(at(src, params[1].ty.as_ref().unwrap().span), "Vector[3]");
    assert_eq!(at(src, ret.as_ref().unwrap().span), "Vector[3]");
    assert_eq!(at(src, body.span), "return b");
    assert_eq!(at(src, m.stmts[0].span), src.trim_end());
}

#[test]
fn simple_statement_spans() {
    let src = "while a:\n    return 1 + 2\n    pass\n    break\n    continue\n";
    let (m, _) = parse_module(&lex(src), src);
    let StmtKind::While { body, .. } = &m.stmts[0].kind else {
        panic!("not a while");
    };
    let texts: Vec<&str> = body.stmts.iter().map(|s| at(src, s.span)).collect();
    assert_eq!(texts, ["return 1 + 2", "pass", "break", "continue"]);
}

// ----- one-line bodies -----

#[test]
fn one_line_bodies_are_errors_with_a_help_line() {
    let src = "if x: y = 1\nwhile x: pass\ndef f(): return 1\n";
    let errors = module_errors(src);
    assert_eq!(errors.len(), 3, "{errors:#?}");
    for (e, what) in errors.iter().zip(["if", "while", "def"]) {
        assert_eq!(
            e.message,
            format!("the body of `{what}` must be on its own indented line")
        );
        assert_eq!(
            e.help.as_deref(),
            Some("put the body on its own indented line")
        );
    }
    assert_eq!(at(src, errors[0].span), "y");
}

#[test]
fn one_line_body_recovers_without_a_second_error() {
    // The statement after `:` is still parsed, so `else:` after it pairs up.
    let errors = messages("if x: y = 1\nelse:\n    y = 2\nz = 3\n");
    assert_eq!(errors.len(), 1, "{errors:?}");
    // A compound statement on the header line is skipped whole.
    let errors = messages("if a: if b: c\nz = 3\n");
    assert_eq!(errors.len(), 1, "{errors:?}");
}

#[test]
fn empty_blocks_say_use_pass() {
    for src in [
        "if x:\ny = 1\n",
        "def f():\n",
        "while x:",
        "if a:\n    if b:\n    y = 1\n",
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert!(errors[0]
            .message
            .starts_with("expected an indented block after"));
        assert_eq!(
            errors[0].help.as_deref(),
            Some("use `pass` for a block that does nothing")
        );
    }
}

// ----- unsupported and reserved -----

#[test]
fn import_is_not_supported_yet() {
    let errors = module_errors("for i in x:\n    y = i\nz = 1\nimport a.b\nw = 2\n");
    let messages: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
    assert_eq!(messages, ["`import` is not supported yet"]);
}

// ----- for -----

#[test]
fn for_forms() {
    assert_eq!(
        module_sexpr("for i in range(3):\n    x = i\ny = 1\n"),
        "(for i (call range 3) (block (= x i)))\n(= y 1)"
    );
    assert_eq!(
        module_sexpr("for i in range(1, n + 1, 2):\n    pass\n"),
        "(for i (call range 1 (+ n 1) 2) (block (pass)))"
    );
    assert_eq!(
        module_sexpr("for f in forces:\n    for r in A:\n        break\n    continue\n"),
        "(for f forces (block (for r A (block (break))) (continue)))"
    );
}

#[test]
fn for_spans_run_from_the_keyword_to_the_end_of_the_body() {
    let src = "for i in v:\n    x = i\nz = 1\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty(), "{errors:#?}");
    let StmtKind::For { var, iter, body } = &m.stmts[0].kind else {
        panic!("not a for: {:?}", m.stmts[0]);
    };
    assert_eq!(at(src, m.stmts[0].span), "for i in v:\n    x = i");
    assert_eq!(at(src, var.span), "i");
    assert_eq!(at(src, iter.span), "v");
    assert_eq!(at(src, body.span), "x = i");
}

#[test]
fn a_for_header_error_still_reads_the_body() {
    // The body's own error is reported, and the block stays in sync.
    let src = "for i range(3):\n    x = = 1\ny = 2\n";
    assert_eq!(
        messages(src),
        [
            "expected `in` after the loop variable, found name `range`",
            "expected an expression, found `=`"
        ]
    );
}

#[test]
fn a_for_needs_a_plain_name_before_in() {
    assert_eq!(
        messages("for (a, b) in p:\n    pass\n"),
        ["expected a loop variable name after `for`, found `(`"]
    );
    assert_eq!(
        messages("for i, j in p:\n    pass\n"),
        ["expected `in` after the loop variable, found `,`"]
    );
}

#[test]
fn a_for_body_on_the_header_line_is_rejected_like_while() {
    assert_eq!(
        messages("for i in v: x = i\n"),
        ["the body of `for` must be on its own indented line"]
    );
}

#[test]
fn an_empty_for_body_says_use_pass() {
    let errors = module_errors("for i in v:\nx = 1\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(errors[0].message, "expected an indented block after `for`");
}

#[test]
fn reserved_words_are_reserved() {
    for word in ["equation", "solve", "field"] {
        let errors = module_errors(&format!("{word} u:\n    pass\nx = 1\n"));
        assert_eq!(errors.len(), 1, "{word}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("`{word}` is reserved for a future version")
        );
    }
}

#[test]
fn else_without_if() {
    assert_eq!(
        messages("else:\n    pass\nelif a:\n    pass\nx = 1\n"),
        [
            "`else` without a matching `if`",
            "`elif` without a matching `if`"
        ]
    );
    let errors = module_errors("if a:\n    pass\nelse if b:\n    pass\n");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].message, "`else if` is not valid");
    assert_eq!(errors[0].help.as_deref(), Some("write `elif`"));
}

#[test]
fn statement_keywords_in_expression_position() {
    for kw in [
        "return", "while", "def", "pass", "break", "continue", "elif", "else",
    ] {
        let src = format!("x = {kw}\n");
        let errors = module_errors(&src);
        assert_eq!(errors.len(), 1, "{kw}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            format!("expected an expression, found keyword `{kw}`")
        );
    }
}

// ----- recovery -----

#[test]
fn errors_in_nested_blocks_do_not_desynchronize() {
    let src = "\
def f(x):
    if x = 1:
        y = 2 +
    z = 3 +
    while x
        pass
    return z
def g():
    if a:
        b = * 2
    else
        c = 1 2
    d = 2
e = 3 +
f = 4
";
    let (m, errors) = parse_module(&lex(src), src);
    let texts: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
    assert_eq!(
        texts,
        [
            "expected `:` after `if`, found `=`",
            "expected an expression, found end of line",
            "expected an expression, found end of line",
            "expected `:` after `while`, found end of line",
            "expected an expression, found `*`",
            "expected `:` after `else`, found end of line",
            "unexpected integer literal `2` after an expression; expected end of line",
            "expected an expression, found end of line",
        ],
        "{errors:#?}"
    );
    // Both defs and the last line survived: `return z` closes `f`, and the
    // errors inside never ate a `Dedent`.
    let names: Vec<String> = m
        .stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Def { name, .. } => at(src, name.span).to_string(),
            StmtKind::Assign { .. } => "assign".to_string(),
            _ => "?".to_string(),
        })
        .collect();
    assert_eq!(names, ["f", "g", "assign"]);
    let StmtKind::Def { body, .. } = &m.stmts[0].kind else {
        panic!()
    };
    let last = body.stmts.last().expect("f keeps its return");
    assert_eq!(at(src, last.span), "return z");
}

#[test]
fn a_bad_header_still_reports_errors_in_its_body() {
    let errors = messages("if x = 1:\n    y = 2 +\n    z = * 3\nw = 1\n");
    assert_eq!(
        errors,
        [
            "expected `:` after `if`, found `=`",
            "expected an expression, found end of line",
            "expected an expression, found `*`",
        ]
    );
}

#[test]
fn a_bad_elif_keeps_the_else_after_it() {
    let errors = messages("if a:\n    pass\nelif b = 1:\n    pass\nelse:\n    pass\n");
    assert_eq!(errors, ["expected `:` after `elif`, found `=`"]);
}

#[test]
fn stray_indentation_is_one_error_and_stays_in_sync() {
    let src = "x = 1\n    y = 2\n    z = 3\nw = 4\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].message, "unexpected indentation");
    // The indented statements are kept, in order.
    assert_eq!(m.stmts.len(), 4);
}

#[test]
fn stray_indentation_inside_a_block() {
    let src = "if a:\n    x = 1\n        y = 2\n    z = 3\nw = 4\n";
    let (m, errors) = parse_module(&lex(src), src);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(m.stmts.len(), 2);
    let StmtKind::If { branches, .. } = &m.stmts[0].kind else {
        panic!()
    };
    assert_eq!(branches[0].1.stmts.len(), 3);
}

#[test]
fn parameter_errors() {
    assert_eq!(
        messages("def f(x, y=1):\n    pass\n"),
        ["default parameter values are not supported yet"]
    );
    assert_eq!(
        messages("def f(a b):\n    pass\n"),
        ["expected `,` or `)` after the parameter, found name `b`"]
    );
    assert_eq!(
        messages("def f(1):\n    pass\n"),
        ["expected a parameter name, found integer literal `1`"]
    );
    assert_eq!(
        messages("def (x):\n    pass\n"),
        ["expected a function name after `def`, found `(`"]
    );
    assert_eq!(
        messages("def f:\n    pass\n"),
        ["expected `(` after the function name, found `:`"]
    );
}

#[test]
fn simple_keyword_statements_take_no_value() {
    assert_eq!(
        messages("pass x\nbreak 1\n"),
        [
            "expected end of line after `pass`, found name `x`",
            "expected end of line after `break`, found integer literal `1`"
        ]
    );
}

// ----- depth -----

/// `n` nested `if`s, indented with one tab per level (a nest of n levels is n^2/2 bytes of indentation, so a
/// 3,000-level file is already 4.5 MB), the innermost holding `body`.
fn nested_ifs(n: usize, body: &str) -> String {
    let mut src = String::new();
    for i in 0..n {
        src.push_str(&"\t".repeat(i));
        src.push_str("if a:\n");
    }
    src.push_str(&"\t".repeat(n));
    src.push_str(body);
    src.push('\n');
    src
}

fn parse_deep(src: &str) -> (usize, Vec<String>) {
    with_deep_stack(|| {
        let (m, errors) = parse_module(&lex(src), src);
        let _ = m.to_sexpr(src);
        (
            m.stmts.len(),
            errors.into_iter().map(|e| e.message).collect(),
        )
    })
}

#[test]
fn block_nesting_up_to_the_limit_parses() {
    let n = MAX_DEPTH as usize;
    let (stmts, errors) = parse_deep(&nested_ifs(n, "pass"));
    assert_eq!((stmts, errors), (1, vec![]));
}

#[test]
fn block_nesting_past_the_limit_is_a_clean_error() {
    let n = MAX_DEPTH as usize + 1;
    let (stmts, errors) = parse_deep(&nested_ifs(n, "pass"));
    assert_eq!(stmts, 1);
    assert_eq!(
        errors,
        ["blocks are nested too deeply (the limit is 256 levels)"]
    );
    // Far past it, and with statements after the deep part: the rest of the
    // file is still parsed, and only one error is reported.
    let mut src = nested_ifs(3_000, "pass");
    src.push_str("x = 1\n");
    let (stmts, errors) = parse_deep(&src);
    assert_eq!(stmts, 2);
    assert_eq!(errors.len(), 1, "{errors:?}");
}

#[test]
fn blocks_and_expressions_share_the_depth_budget() {
    let n = MAX_DEPTH as usize;
    // 255 blocks leave room for one expression level, not two.
    let (_, errors) = parse_deep(&nested_ifs(n - 1, "x = (1)"));
    assert_eq!(errors, Vec::<String>::new());
    let (_, errors) = parse_deep(&nested_ifs(n - 1, "x = ((1))"));
    assert_eq!(
        errors,
        ["expression is nested too deeply (the limit is 256 levels)"]
    );
}

#[test]
fn deep_defs_and_whiles_count_too() {
    for kw in ["while a", "def f()"] {
        let mut src = String::new();
        let n = MAX_DEPTH as usize + 1;
        for i in 0..n {
            src.push_str(&"\t".repeat(i));
            src.push_str(kw);
            src.push_str(":\n");
        }
        src.push_str(&"\t".repeat(n));
        src.push_str("pass\n");
        let (_, errors) = parse_deep(&src);
        assert_eq!(
            errors,
            ["blocks are nested too deeply (the limit is 256 levels)"],
            "{kw}"
        );
    }
}

#[test]
fn deep_conditional_expression_chains_count() {
    let ok = format!("x = a{}\n", " if c else b".repeat(MAX_DEPTH as usize));
    let too_deep = format!("x = a{}\n", " if c else b".repeat(MAX_DEPTH as usize + 1));
    assert_eq!(parse_deep(&ok).1, Vec::<String>::new());
    assert_eq!(
        parse_deep(&too_deep).1,
        ["expression is nested too deeply (the limit is 256 levels)"]
    );
}

#[test]
fn block_depth_fits_in_a_small_stack() {
    // 256 nested blocks, parsed, printed, and dropped on 1 MiB (half the
    // default spawned-thread stack), then the same past the limit.
    let handle = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            for n in [MAX_DEPTH as usize, MAX_DEPTH as usize + 1, 3_000] {
                let src = nested_ifs(n, "pass");
                let (module, _) = parse_module(&lex(&src), &src);
                let _ = module.to_sexpr(&src);
            }
        })
        .expect("spawn");
    handle.join().expect("no stack overflow");
}

#[test]
fn many_statements_and_clauses_are_not_recursive() {
    let mut src = String::from("if a:\n    pass\n");
    for _ in 0..5_000 {
        src.push_str("elif a:\n    pass\n");
    }
    src.push_str("else:\n    pass\n");
    for _ in 0..5_000 {
        src.push_str("x = 1\n");
    }
    let (stmts, errors) = parse_deep(&src);
    assert_eq!((stmts, errors), (5_001, vec![]));
}
