//! The parser must never panic (it is a fuzz target). A deterministic
//! xorshift PRNG drives random token sequences with garbage and well-formed
//! spans, and every prefix/deletion/duplication of real token streams.

mod common;
use sky_ast::{parse_expr, parse_module, Kw, Span, Token, TokenKind};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const KINDS: &[TokenKind] = {
    use TokenKind::*;
    &[
        Int,
        Float,
        Ident,
        Str,
        Newline,
        Indent,
        Dedent,
        Eof,
        LParen,
        RParen,
        LBracket,
        RBracket,
        Comma,
        Colon,
        Assign,
        Plus,
        Minus,
        Star,
        Slash,
        Caret,
        At,
        Inverse,
        Partial,
        Nabla,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        Dot,
        Arrow,
        PlusAssign,
        MinusAssign,
        StarAssign,
        SlashAssign,
        CaretAssign,
        AtAssign,
        Keyword(Kw::Def),
        Keyword(Kw::Return),
        Keyword(Kw::If),
        Keyword(Kw::Elif),
        Keyword(Kw::Else),
        Keyword(Kw::While),
        Keyword(Kw::For),
        Keyword(Kw::In),
        Keyword(Kw::As),
        Keyword(Kw::Import),
        Keyword(Kw::Pass),
        Keyword(Kw::Break),
        Keyword(Kw::Continue),
        Keyword(Kw::Equation),
        Keyword(Kw::Solve),
        Keyword(Kw::Field),
    ]
};

/// Weighted toward the kinds that drive unit literals and nesting.
fn kind(rng: &mut Rng) -> TokenKind {
    use TokenKind::*;
    const HOT: &[TokenKind] = &[
        Int, Ident, Star, Slash, Caret, Minus, LParen, RParen, Newline,
    ];
    if rng.below(2) == 0 {
        HOT[rng.below(HOT.len())]
    } else {
        KINDS[rng.below(KINDS.len())]
    }
}

fn exercise(tokens: &[Token], src: &str) {
    let (expr, errors) = parse_expr(tokens, src);
    assert!(
        expr.is_some() || !errors.is_empty(),
        "None without an error"
    );
    if let Some(e) = expr {
        let _ = e.to_sexpr(src);
    }
    let (module, errors) = parse_module(tokens, src);
    let _ = module.to_sexpr(src);
    for err in errors {
        let _ = err.into_diagnostic();
    }
}

#[test]
fn random_tokens_with_garbage_spans() {
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    const PIECES: &[&str] = &[
        "a",
        "1",
        "é",
        "∂",
        "⁻¹",
        " ",
        "2.5",
        "_",
        "\n",
        "99999999999999999999",
    ];
    for _ in 0..5_000 {
        let src: String = (0..rng.below(16))
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect();
        let max = src.len() as u64 + 4;
        let tokens: Vec<Token> = (0..rng.below(48))
            .map(|_| Token {
                kind: kind(&mut rng),
                // Independent start/end: inverted, out of range, mid-char.
                span: Span {
                    start: (rng.next() % max) as u32,
                    end: (rng.next() % max) as u32,
                },
            })
            .collect();
        exercise(&tokens, &src);
    }
}

fn canonical_text(kind: TokenKind) -> &'static str {
    use TokenKind::*;
    match kind {
        Int => "2",
        Float => "1.5",
        Ident => "m",
        Str => "\"s\"",
        Newline => "\n",
        Indent | Dedent | Eof => "",
        LParen => "(",
        RParen => ")",
        LBracket => "[",
        RBracket => "]",
        Comma => ",",
        Colon => ":",
        Assign => "=",
        Plus => "+",
        Minus => "-",
        Star => "*",
        Slash => "/",
        Caret => "^",
        At => "@",
        Inverse => "⁻¹",
        Partial => "∂",
        Nabla => "∇",
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        Dot => ".",
        Arrow => "->",
        PlusAssign => "+=",
        MinusAssign => "-=",
        StarAssign => "*=",
        SlashAssign => "/=",
        CaretAssign => "^=",
        AtAssign => "@=",
        Keyword(_) => "def",
    }
}

#[test]
fn random_tokens_with_well_formed_spans() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for _ in 0..20_000 {
        let mut src = String::new();
        let mut tokens = Vec::new();
        for _ in 0..rng.below(40) {
            if rng.below(3) == 0 {
                src.push(' '); // sometimes a gap, so both adjacency paths run
            }
            let k = kind(&mut rng);
            let start = src.len() as u32;
            src.push_str(canonical_text(k));
            tokens.push(Token {
                kind: k,
                span: Span::new(start, src.len() as u32),
            });
        }
        if rng.below(2) == 0 {
            let end = src.len() as u32;
            tokens.push(Token {
                kind: TokenKind::Eof,
                span: Span::point(end),
            });
        }
        exercise(&tokens, &src);
    }
}

#[test]
fn truncated_and_mutated_real_programs() {
    let programs = [
        "x = 9.8 m/s^2 * t^2 / 2\nA: Matrix[3, 4] = zeros(3, 4)\ny = A⁻¹ @ b\n",
        "v = 10 km / 30 min\nE = ħ * ω + 1 eV\nq = 1 kg*m^(-3/2)/s^2\n",
        "f(a, g(b)[0].c, -x^-2, ∂u/∂x, ∇φ) <= 3\nz = (((a)))\n",
        "d = 3\ninv(A)\nM[1, 2,]\nr = 1 m^(1/0) + 1 m^(99999999999999999999/2)\n",
    ];
    for src in programs {
        let tokens = common::lex(src);
        let body = &tokens[..tokens.len() - 1]; // without Eof
        for k in 0..=body.len() {
            exercise(&body[..k], src);
        }
        for i in 0..body.len() {
            let mut t = body.to_vec();
            t.remove(i);
            exercise(&t, src);
            let mut t = body.to_vec();
            t.insert(i, body[i]);
            exercise(&t, src);
            if i + 1 < body.len() {
                let mut t = body.to_vec();
                t.swap(i, i + 1);
                exercise(&t, src);
            }
        }
    }
}
