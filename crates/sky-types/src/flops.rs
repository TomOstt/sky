//! Floating-point operation counts as polynomials in shape variables.
//!
//! The algorithm-choosing rewrites (`crate::chain`, `sky-opt`) pick between
//! ways to compute the same thing by counting flops from the operands'
//! shapes, and `--explain` shows the counts. A shape may be symbolic
//! (`Matrix[N,N] @ Vector[N]`), so a count is a polynomial in the shape
//! variables, `4·N² + 2·N`, and two counts are compared for **all** sizes at
//! once: [`Flops::provably_le`] says `a ≤ b` only if it holds for every
//! assignment of sizes of at least 1 (arrays are never empty). When it
//! cannot say, nothing is rewritten; a rewrite never rests on a guess about
//! what `N` will be.
//!
//! Coefficients are counted in thirds of a flop, because an LU
//! factorization costs `2/3·n³`. The model is the textbook one, per
//! operation:
//!
//! | operation | flops |
//! |---|---|
//! | `m×k @ k×n` | `2·m·k·n` |
//! | inverse of `n×n` (Gauss-Jordan, as `sky run` computes it) | `4·n³` |
//! | LU solve, `n×n` with `k` right-hand sides | `2/3·n³ + 2·n²·k` |
//! | Cholesky solve | `1/3·n³ + 2·n²·k` |
//! | Diagonal solve, or `diag.mul` | `n·k` |
//!
//! These are estimates for explaining a choice and ranking alternatives,
//! not measurements.

use std::collections::BTreeMap;
use std::fmt;

use crate::shape::{ShapeDim, ShapeVar};

/// A flop count: a polynomial with integer coefficients (in thirds of a
/// flop) over shape variables. The zero polynomial is no flops.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Flops {
    /// Monomial (variables sorted, repeated for a power) to coefficient in
    /// thirds. No zero coefficients.
    terms: BTreeMap<Vec<ShapeVar>, i128>,
}

impl Flops {
    /// No flops.
    pub fn zero() -> Flops {
        Flops::default()
    }

    /// `thirds/3 · Π dims` flops: a known dimension multiplies the
    /// coefficient, a variable joins the monomial.
    pub fn product(thirds: i128, dims: &[&ShapeDim]) -> Flops {
        let mut coefficient = thirds;
        let mut vars = Vec::new();
        for d in dims {
            match d {
                ShapeDim::Known(n) => coefficient = coefficient.saturating_mul(i128::from(*n)),
                ShapeDim::Var(v) => vars.push(v.clone()),
            }
        }
        vars.sort();
        let mut terms = BTreeMap::new();
        if coefficient != 0 {
            terms.insert(vars, coefficient);
        }
        Flops { terms }
    }

    /// `2·m·k·n`: a matrix product.
    pub fn matmul(m: &ShapeDim, k: &ShapeDim, n: &ShapeDim) -> Flops {
        Flops::product(6, &[m, k, n])
    }

    /// `4·n³`: `inverse` of an `n × n` matrix by Gauss-Jordan.
    pub fn inverse(n: &ShapeDim) -> Flops {
        Flops::product(12, &[n, n, n])
    }

    /// Solving `A x = b` for `k` right-hand sides with an `n × n` `A`, by
    /// `method`.
    pub fn solve(method: crate::algo::SolveMethod, n: &ShapeDim, k: &ShapeDim) -> Flops {
        use crate::algo::SolveMethod::{Cholesky, Diagonal, Lu};
        let factor = match method {
            Cholesky => Flops::product(1, &[n, n, n]),
            Lu => Flops::product(2, &[n, n, n]),
            Diagonal => Flops::zero(),
        };
        let substitutions = match method {
            Cholesky | Lu => Flops::product(6, &[n, n, k]),
            Diagonal => Flops::product(3, &[n, k]),
        };
        factor.plus(&substitutions)
    }

    /// The sum of two counts.
    pub fn plus(&self, other: &Flops) -> Flops {
        let mut terms = self.terms.clone();
        for (m, c) in &other.terms {
            let e = terms.entry(m.clone()).or_insert(0);
            *e = e.saturating_add(*c);
        }
        terms.retain(|_, c| *c != 0);
        Flops { terms }
    }

    /// The count as a number, if it has no variables (rounded to a whole
    /// flop).
    pub fn value(&self) -> Option<u128> {
        match self.terms.len() {
            0 => Some(0),
            1 => {
                let (m, c) = self.terms.iter().next()?;
                m.is_empty().then(|| ((*c).max(0) as u128 + 1) / 3)
            }
            _ => None,
        }
    }

    /// Whether this count is at most `other` for every assignment of sizes
    /// of at least 1. A sufficient test: every term of `other - self` that
    /// is negative is covered by positive terms whose monomial contains it
    /// (a variable is at least 1, so `N³ ≥ N²`, and everything is at least
    /// the constant). `false` means "cannot say", not "greater".
    pub fn provably_le(&self, other: &Flops) -> bool {
        let mut diff = other.terms.clone();
        for (m, c) in &self.terms {
            let e = diff.entry(m.clone()).or_insert(0);
            *e = e.saturating_sub(*c);
        }
        diff.retain(|_, c| *c != 0);
        let mut positive: Vec<(Vec<ShapeVar>, i128)> = diff
            .iter()
            .filter(|(_, c)| **c > 0)
            .map(|(m, c)| (m.clone(), *c))
            .collect();
        let mut negative: Vec<(Vec<ShapeVar>, i128)> = diff
            .iter()
            .filter(|(_, c)| **c < 0)
            .map(|(m, c)| (m.clone(), -*c))
            .collect();
        // The highest-degree negative terms have the fewest terms above
        // them; take from the lowest-degree positive term that covers.
        negative.sort_by_key(|(m, _)| std::cmp::Reverse(m.len()));
        positive.sort_by_key(|(m, _)| m.len());
        for (m, mut need) in negative {
            for (p, available) in &mut positive {
                if need == 0 {
                    break;
                }
                if *available > 0 && contains(p, &m) {
                    let take = need.min(*available);
                    need -= take;
                    *available -= take;
                }
            }
            if need > 0 {
                return false;
            }
        }
        true
    }

    /// Whether this count is at most `other` and not the same polynomial.
    pub fn provably_less(&self, other: &Flops) -> bool {
        self != other && self.provably_le(other)
    }
}

/// Whether the sorted monomial `big` contains every variable of `small`
/// (with multiplicity).
fn contains(big: &[ShapeVar], small: &[ShapeVar]) -> bool {
    let mut it = big.iter();
    small.iter().all(|s| it.any(|b| b == s))
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// `1234567` as `1,234,567`.
pub fn grouped(n: u128) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn superscript(e: usize) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    if e < 2 {
        String::new()
    } else if e < 10 {
        DIGITS[e].to_string()
    } else {
        format!("^{e}")
    }
}

impl fmt::Display for Flops {
    /// `250,500,000` for a number, `2·N³ + 2·N²` (`2/3·N³` for a fraction)
    /// for a polynomial, highest degree first.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(n) = self.value() {
            return f.write_str(&grouped(n));
        }
        let mut terms: Vec<(&Vec<ShapeVar>, &i128)> = self.terms.iter().collect();
        // Highest degree first, then the highest power of one variable
        // (`N³` before `N²·K`), then by name.
        let names = |m: &Vec<ShapeVar>| m.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        let top_power = |m: &Vec<ShapeVar>| {
            m.iter()
                .map(|v| m.iter().filter(|w| *w == v).count())
                .max()
                .unwrap_or(0)
        };
        terms.sort_by(|(ma, _), (mb, _)| {
            mb.len()
                .cmp(&ma.len())
                .then_with(|| top_power(mb).cmp(&top_power(ma)))
                .then_with(|| names(ma).cmp(&names(mb)))
        });
        for (i, (m, c)) in terms.into_iter().enumerate() {
            if i > 0 {
                f.write_str(" + ")?;
            }
            let g = gcd(*c, 3);
            let (num, den) = (*c / g, 3 / g);
            let mut parts = Vec::new();
            let mut names: Vec<(String, usize)> = Vec::new();
            for v in m {
                let name = v.to_string();
                match names.last_mut() {
                    Some((n, e)) if *n == name => *e += 1,
                    _ => names.push((name, 1)),
                }
            }
            for (n, e) in names {
                parts.push(format!("{n}{}", superscript(e)));
            }
            let coefficient = match (num, den, parts.is_empty()) {
                (n, 1, true) => n.to_string(),
                (n, d, true) => format!("{n}/{d}"),
                (1, 1, false) => String::new(),
                (n, 1, false) => format!("{n}·"),
                (n, d, false) => format!("{n}/{d}·"),
            };
            write!(f, "{coefficient}{}", parts.join("·"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ShapeUnifier;

    fn var(u: &mut ShapeUnifier, name: &str) -> ShapeDim {
        ShapeDim::Var(u.fresh(name))
    }

    #[test]
    fn known_sizes_are_numbers() {
        let n = ShapeDim::Known(500);
        let cube = Flops::matmul(&n, &n, &n);
        assert_eq!(cube.value(), Some(250_000_000));
        assert_eq!(cube.to_string(), "250,000,000");
        let lu = Flops::solve(crate::algo::SolveMethod::Lu, &n, &ShapeDim::Known(1));
        // 2/3·500³ + 2·500² = 83,333,333.33 + 500,000
        assert_eq!(lu.value(), Some(83_833_333));
    }

    #[test]
    fn symbolic_counts_print_as_polynomials() {
        let mut u = ShapeUnifier::new();
        let (n, k) = (var(&mut u, "N"), var(&mut u, "K"));
        let f = Flops::matmul(&n, &n, &n).plus(&Flops::matmul(&n, &n, &k));
        assert_eq!(f.to_string(), "2·N³ + 2·N²·K");
        let lu = Flops::solve(crate::algo::SolveMethod::Lu, &n, &k);
        assert_eq!(lu.to_string(), "2/3·N³ + 2·N²·K");
        assert_eq!(Flops::zero().to_string(), "0");
    }

    #[test]
    fn comparison_holds_for_every_size_or_says_nothing() {
        let mut u = ShapeUnifier::new();
        let n = var(&mut u, "N");
        let k = var(&mut u, "K");
        let cubic = Flops::matmul(&n, &n, &n).plus(&Flops::matmul(&n, &n, &ShapeDim::Known(1)));
        let quadratic = Flops::matmul(&n, &n, &ShapeDim::Known(1)).plus(&Flops::matmul(
            &n,
            &n,
            &ShapeDim::Known(1),
        ));
        // 4N² <= 2N³ + 2N² because N >= 1.
        assert!(quadratic.provably_le(&cubic));
        assert!(quadratic.provably_less(&cubic));
        assert!(!cubic.provably_le(&quadratic));
        // N² against K²: no relation is known.
        let a = Flops::matmul(&n, &n, &ShapeDim::Known(1));
        let b = Flops::matmul(&k, &k, &ShapeDim::Known(1));
        assert!(!a.provably_le(&b) && !b.provably_le(&a));
        // Reflexive, and constants compare as numbers.
        assert!(a.provably_le(&a) && !a.provably_less(&a));
        let (c1, c2) = (Flops::product(6, &[]), Flops::product(12, &[]));
        assert!(c1.provably_le(&c2) && !c2.provably_le(&c1));
    }
}
