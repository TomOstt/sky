//! Golden tests for shapes, tags, and the Cholesky rewrite decision.
//! See `tests/golden/README.md`. The case-file parser is test tooling, not
//! Sky syntax, and lives here rather than in the library.

use std::collections::HashMap;
use std::path::Path;

use sky_types::{
    cholesky_rewrite, elementwise, inverse, matmul, transpose, MatmulLhs, Shape, ShapeDim,
    ShapeError, ShapeUnifier, ShapeVar, Tag, TagSet, TensorType,
};

type Res<T> = Result<T, ShapeError>;

#[derive(Default)]
struct Ctx {
    u: ShapeUnifier,
    vars: HashMap<String, ShapeVar>,
    env: HashMap<String, TensorType>,
}

impl Ctx {
    fn var(&mut self, name: &str) -> ShapeVar {
        let u = &mut self.u;
        self.vars
            .entry(name.to_string())
            .or_insert_with(|| u.fresh(name))
            .clone()
    }

    /// `Scalar`, `Vector[d]`, `Matrix[d,d]`, or `Tensor[d,...]` (no spaces inside).
    fn shape(&mut self, tok: &str) -> Res<Shape> {
        if tok == "Scalar" {
            return Ok(Shape::scalar());
        }
        let (head, rest) = tok
            .split_once('[')
            .unwrap_or_else(|| panic!("bad shape {tok:?}"));
        let inner = rest
            .strip_suffix(']')
            .unwrap_or_else(|| panic!("bad shape {tok:?}"));
        let mut dims = Vec::new();
        for d in inner.split(',').filter(|d| !d.is_empty()) {
            dims.push(match d.parse::<i64>() {
                Ok(n) => ShapeDim::try_from_i64(n)?,
                Err(_) => ShapeDim::Var(self.var(d)),
            });
        }
        let want = match head {
            "Vector" => Some(1),
            "Matrix" => Some(2),
            "Tensor" => None,
            _ => panic!("bad shape head {head:?}"),
        };
        if let Some(r) = want {
            assert_eq!(dims.len(), r, "{tok:?} has the wrong rank");
        }
        Ok(Shape::new(dims))
    }

    /// A shape token followed by tag names.
    fn ty(&mut self, src: &str) -> Res<TensorType> {
        let mut words = src.split_whitespace();
        let shape = self.shape(words.next().expect("empty type"))?;
        let tags = TagSet::from_tags(
            words.map(|w| Tag::from_name(w).unwrap_or_else(|| panic!("unknown tag {w:?}"))),
        );
        TensorType::tagged(&mut self.u, shape, tags)
    }

    fn expr(&mut self, src: &str) -> Res<TensorType> {
        let mut p = Expr {
            chars: src.chars().collect(),
            pos: 0,
        };
        let t = p.expr(self)?;
        p.ws();
        assert!(p.pos == p.chars.len(), "trailing input in {src:?}");
        Ok(t)
    }
}

struct Expr {
    chars: Vec<char>,
    pos: usize,
}

impl Expr {
    fn ws(&mut self) {
        while self.chars.get(self.pos).is_some_and(|c| c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, s: &str) -> bool {
        self.ws();
        let want: Vec<char> = s.chars().collect();
        if self.chars[self.pos..].starts_with(&want) {
            self.pos += want.len();
            true
        } else {
            false
        }
    }

    // expr := unary (('@' | '+' | '-') unary)*
    fn expr(&mut self, cx: &mut Ctx) -> Res<TensorType> {
        let mut t = self.unary(cx)?;
        loop {
            if self.eat("@") {
                let r = self.unary(cx)?;
                t = matmul(&mut cx.u, &t, &r)?;
            } else if self.eat("+") || self.eat("-") {
                let r = self.unary(cx)?;
                t = elementwise(&mut cx.u, &t, &r)?;
            } else {
                return Ok(t);
            }
        }
    }

    // unary := primary ('⁻¹' | 'ᵀ')*
    fn unary(&mut self, cx: &mut Ctx) -> Res<TensorType> {
        let mut t = self.primary(cx)?;
        loop {
            if self.eat("⁻¹") {
                t = inverse(&mut cx.u, &t)?;
            } else if self.eat("ᵀ") {
                t = transpose(&t)?;
            } else {
                return Ok(t);
            }
        }
    }

    // primary := 'inv(' expr ')' | 'transpose(' expr ')' | '(' expr ')' | NAME
    fn primary(&mut self, cx: &mut Ctx) -> Res<TensorType> {
        if self.eat("inv(") {
            let t = self.expr(cx)?;
            assert!(self.eat(")"), "expected ')'");
            return inverse(&mut cx.u, &t);
        }
        if self.eat("transpose(") {
            let t = self.expr(cx)?;
            assert!(self.eat(")"), "expected ')'");
            return transpose(&t);
        }
        if self.eat("(") {
            let t = self.expr(cx)?;
            assert!(self.eat(")"), "expected ')'");
            return Ok(t);
        }
        self.ws();
        let start = self.pos;
        while self
            .chars
            .get(self.pos)
            .is_some_and(|c| c.is_alphanumeric() || *c == '_')
        {
            self.pos += 1;
        }
        let name: String = self.chars[start..self.pos].iter().collect();
        assert!(!name.is_empty(), "expected a name at {start}");
        Ok(cx
            .env
            .get(&name)
            .unwrap_or_else(|| panic!("unbound name {name:?}"))
            .clone())
    }
}

/// `explain LHS @ RHS`: LHS is `inv(X)` / `X⁻¹` (explicit inverse), `X^N`
/// (a power), or any other expression.
fn explain(cx: &mut Ctx, rest: &str) -> Res<String> {
    let (lhs, rhs) = rest
        .split_once(" @ ")
        .unwrap_or_else(|| panic!("expected `explain LHS @ RHS`: {rest:?}"));
    let lhs = lhs.trim();
    let rhs_t = cx.expr(rhs)?;
    let inverse_of = lhs
        .strip_prefix("inv(")
        .and_then(|s| s.strip_suffix(')'))
        .or_else(|| lhs.strip_suffix("⁻¹"));
    let plan = if let Some(inner) = inverse_of {
        let a = cx.expr(inner)?;
        cholesky_rewrite(MatmulLhs::Inverse(&a), &rhs_t, &cx.u)
    } else if let Some((base, exp)) = lhs.split_once('^') {
        let base = cx.expr(base)?;
        let exponent = exp.parse().expect("integer exponent");
        cholesky_rewrite(
            MatmulLhs::Power {
                base: &base,
                exponent,
            },
            &rhs_t,
            &cx.u,
        )
    } else {
        let a = cx.expr(lhs)?;
        cholesky_rewrite(MatmulLhs::Value(&a), &rhs_t, &cx.u)
    };
    Ok(match plan {
        Some(r) => format!("explain: {}", r.explanation()),
        None => "explain: no rewrite".to_string(),
    })
}

fn step(cx: &mut Ctx, line: &str) -> Res<Option<String>> {
    if let Some(rest) = line.strip_prefix("let ") {
        let (name, ty) = rest
            .split_once(" = ")
            .unwrap_or_else(|| panic!("expected `let NAME = TYPE`: {line:?}"));
        let t = cx.ty(ty)?;
        cx.env.insert(name.trim().to_string(), t);
        Ok(None)
    } else if let Some(rest) = line.strip_prefix("unify ") {
        let (l, r) = rest
            .split_once(" with ")
            .unwrap_or_else(|| panic!("expected `unify SHAPE with SHAPE`: {line:?}"));
        let (l, r) = (cx.shape(l.trim())?, cx.shape(r.trim())?);
        cx.u.unify(&l, &r)?;
        Ok(None)
    } else if let Some(rest) = line.strip_prefix("show ") {
        let t = cx.expr(rest)?;
        Ok(Some(format!("{} : {}", rest.trim(), cx.u.apply_type(&t))))
    } else if let Some(rest) = line.strip_prefix("explain ") {
        explain(cx, rest).map(Some)
    } else {
        panic!("unknown command: {line:?}");
    }
}

/// Runs a case file; returns the rendered output and whether it ended in an error.
fn run_case(src: &str) -> (String, bool) {
    let mut cx = Ctx::default();
    let mut out = Vec::new();
    for line in src.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match step(&mut cx, line) {
            Ok(Some(s)) => out.push(s),
            Ok(None) => {}
            Err(e) => {
                out.push(format!("error: {e}"));
                return (out.join("\n"), true);
            }
        }
    }
    (out.join("\n"), false)
}

#[test]
fn golden_cases() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bless = std::env::var_os("SKY_BLESS").is_some();
    let mut cases: Vec<_> = std::fs::read_dir(&dir)
        .expect("read tests/golden")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "case"))
        .collect();
    cases.sort();

    let (mut passing, mut failing) = (0, 0);
    let mut problems = Vec::new();
    for case in &cases {
        let stem = case.file_stem().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(case).expect("read case");
        let (out, errored) = run_case(&src);

        if stem.starts_with("pass_") {
            passing += 1;
            if errored {
                problems.push(format!("{stem}: expected success, got:\n{out}"));
            }
        } else if stem.starts_with("fail_") {
            failing += 1;
            if !errored {
                problems.push(format!("{stem}: expected an error, got success"));
            }
        } else {
            problems.push(format!("{stem}: name must start with pass_ or fail_"));
        }

        let expected_path = case.with_extension("expected");
        let rendered = if out.is_empty() {
            String::new()
        } else {
            format!("{out}\n")
        };
        if bless {
            std::fs::write(&expected_path, &rendered).expect("write expected");
            continue;
        }
        match std::fs::read_to_string(&expected_path) {
            Ok(expected) if expected.trim_end() == out.trim_end() => {}
            Ok(expected) => problems.push(format!(
                "{stem}: output differs\n--- expected\n{expected}\n--- actual\n{rendered}"
            )),
            Err(_) => problems.push(format!("{stem}: missing {}", expected_path.display())),
        }
    }

    eprintln!("golden: {passing} passing cases, {failing} failing cases");
    assert!(
        passing >= 5 && failing >= 5,
        "need at least 5 passing and 5 failing cases"
    );
    assert!(problems.is_empty(), "\n{}", problems.join("\n\n"));
}
