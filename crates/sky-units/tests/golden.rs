//! Golden tests for dimension unification. See `tests/golden/README.md`.
//!
//! The case-file parser lives here, not in the library: it is test tooling,
//! not Sky syntax.

use std::collections::HashMap;
use std::path::Path;

use sky_units::{lookup, BaseDim, DimTerm, DimUnifier, DimVar, Dimension, Rational};

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    vars: &'a mut HashMap<String, DimVar>,
    unifier: &'a mut DimUnifier,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn expect(&mut self, want: char) {
        self.ws();
        let got = self.bump();
        assert_eq!(got, Some(want), "expected {want:?} in {:?}", self.src());
    }

    fn src(&self) -> String {
        self.chars.iter().collect()
    }

    fn take_while(&mut self, f: impl Fn(char) -> bool) -> String {
        let start = self.pos;
        while self.peek().is_some_and(&f) {
            self.pos += 1;
        }
        self.chars[start..self.pos].iter().collect()
    }

    fn parse_all(&mut self) -> DimTerm {
        let t = self.expr();
        self.ws();
        assert!(self.peek().is_none(), "trailing input in {:?}", self.src());
        t
    }

    // expr := factor (('*' | '·' | '/') factor)*
    fn expr(&mut self) -> DimTerm {
        let mut t = self.factor();
        loop {
            self.ws();
            match self.peek() {
                Some('*' | '·') => {
                    self.bump();
                    t = t * self.factor();
                }
                Some('/') => {
                    self.bump();
                    t = t / self.factor();
                }
                _ => return t,
            }
        }
    }

    // factor := atom ('^' exponent)?
    fn factor(&mut self) -> DimTerm {
        let base = self.atom();
        self.ws();
        if self.peek() == Some('^') {
            self.bump();
            base.pow_rat(self.exponent())
        } else {
            base
        }
    }

    // exponent := int | '(' int ('/' int)? ')'
    fn exponent(&mut self) -> Rational {
        self.ws();
        if self.peek() == Some('(') {
            self.bump();
            let n = self.int();
            self.ws();
            let d = if self.peek() == Some('/') {
                self.bump();
                self.int()
            } else {
                1
            };
            self.expect(')');
            Rational::new(n, d)
        } else {
            Rational::integer(self.int())
        }
    }

    fn int(&mut self) -> i64 {
        self.ws();
        let neg = self.peek() == Some('-');
        if neg {
            self.bump();
        }
        let digits = self.take_while(|c| c.is_ascii_digit());
        let n: i64 = digits
            .parse()
            .unwrap_or_else(|_| panic!("bad integer in {:?}", self.src()));
        if neg {
            -n
        } else {
            n
        }
    }

    // atom := '(' expr ')' | '1' | '[' unit ']' | '?' ident | L | M | T | I | Θ | N | J
    fn atom(&mut self) -> DimTerm {
        self.ws();
        match self.bump() {
            Some('(') => {
                let t = self.expr();
                self.expect(')');
                t
            }
            Some('1') => DimTerm::known(Dimension::DIMENSIONLESS),
            Some('[') => {
                let sym = self.take_while(|c| c != ']');
                self.expect(']');
                let unit = lookup(&sym).unwrap_or_else(|| panic!("unknown unit [{sym}]"));
                DimTerm::known(unit.dim)
            }
            Some('?') => {
                let name = self.take_while(|c| c.is_alphanumeric() || c == '_');
                assert!(!name.is_empty(), "empty variable name in {:?}", self.src());
                let key = format!("?{name}");
                let unifier = &mut *self.unifier;
                let v = self
                    .vars
                    .entry(key.clone())
                    .or_insert_with(|| unifier.fresh(key));
                DimTerm::var(v.clone())
            }
            Some(c) => {
                let b = BaseDim::from_symbol(&c.to_string())
                    .unwrap_or_else(|| panic!("unexpected {c:?} in {:?}", self.src()));
                DimTerm::known(Dimension::base(b))
            }
            None => panic!("unexpected end of {:?}", self.src()),
        }
    }
}

fn parse(src: &str, vars: &mut HashMap<String, DimVar>, unifier: &mut DimUnifier) -> DimTerm {
    Parser {
        chars: src.chars().collect(),
        pos: 0,
        vars,
        unifier,
    }
    .parse_all()
}

/// Runs a case file; returns the rendered output and whether it ended in an error.
fn run_case(src: &str) -> (String, bool) {
    let mut unifier = DimUnifier::new();
    let mut vars = HashMap::new();
    let mut out = Vec::new();
    for line in src.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("unify ") {
            let (l, r) = rest
                .split_once(" with ")
                .unwrap_or_else(|| panic!("expected `unify A with B`: {line:?}"));
            let l = parse(l, &mut vars, &mut unifier);
            let r = parse(r, &mut vars, &mut unifier);
            if let Err(e) = unifier.unify(&l, &r) {
                out.push(format!("error: {e}"));
                return (out.join("\n"), true);
            }
        } else if let Some(rest) = line.strip_prefix("show ") {
            let t = parse(rest, &mut vars, &mut unifier);
            out.push(format!("{} = {}", rest.trim(), unifier.apply(&t)));
        } else {
            panic!("unknown command: {line:?}");
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
        passing >= 4 && failing >= 4,
        "need at least 4 passing and 4 failing cases"
    );
    assert!(problems.is_empty(), "\n{}", problems.join("\n\n"));
}
