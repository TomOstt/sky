//! The cross-language workloads: which cases exist, the arguments every
//! implementation receives, and how a Sky template becomes a program.
//!
//! Every implementation (C, Rust, Python, the BLAS/LAPACK one, Sky) computes
//! the same thing from the same deterministic data and prints its result as
//! numbers on the last line of stdout. Sizes are command-line arguments for
//! the compiled and Python programs and are substituted into the Sky
//! templates (`benches/compare/sky/*.sky`, `{{NAME}}` placeholders), because
//! Sky array sizes and `range` bounds must be constants in the source.

/// What a workload mostly exercises; the generated interpretation explains
/// wins and losses per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Function calls (fib).
    Calls,
    /// A scalar loop with a data-dependent exit (Mandelbrot).
    Scalar,
    /// Small fixed-size vectors updated in a loop (N-body, RK4).
    SmallVector,
    /// Elementwise chains over big arrays.
    ArrayChain,
    /// Dense matrix product.
    Matmul,
    /// Cholesky factor-and-solve.
    Cholesky,
}

/// One measured case: a workload at one size.
#[derive(Clone, Debug)]
pub struct Case {
    /// Unique, e.g. `matmul-256`.
    pub id: String,
    pub title: String,
    /// File stem of the C, Rust, Python, NumPy and library programs.
    pub kernel: &'static str,
    /// File stem of the Sky template.
    pub sky: &'static str,
    pub kind: Kind,
    /// Named parameters, in the order the non-Sky programs take them.
    pub params: Vec<(&'static str, u64)>,
    /// The same parameters at the tiny sizes of `--quick`.
    pub quick: Vec<(&'static str, u64)>,
    /// Relative tolerance for "computed the same result".
    pub tolerance: f64,
    /// Whether a BLAS/LAPACK program exists (`benches/compare/lib`).
    pub lib: bool,
    /// Whether a NumPy program exists (`benches/compare/numpy`).
    pub numpy: bool,
}

impl Case {
    fn params(&self, quick: bool) -> &[(&'static str, u64)] {
        if quick {
            &self.quick
        } else {
            &self.params
        }
    }

    /// The command-line arguments of the non-Sky programs.
    pub fn args(&self, quick: bool) -> Vec<String> {
        self.params(quick)
            .iter()
            .map(|(_, v)| v.to_string())
            .collect()
    }

    fn param(&self, name: &str, quick: bool) -> Result<u64, String> {
        self.params(quick)
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("{}: no parameter `{name}`", self.id))
    }

    /// Fills a Sky template's placeholders:
    ///
    /// * `{{NAME}}`: a parameter; `{{LEN}}`: `M * K`;
    /// * `{{COL:s:P}}`, `{{ROW:s:P}}`, `{{VEC:s:P}}`: the data vector of seed
    ///   `s` (an integer, or `ones`) and length parameter `P`, as a
    ///   `Matrix[P, 1]`, a `Matrix[1, P]` or a `Vector[P]` literal;
    /// * `{{KMAT:P}}`: the dense positive definite matrix of the Cholesky
    ///   workload as a `P x P` literal (every entry a number, so the checker
    ///   can prove it positive definite).
    pub fn render_sky(&self, template: &str, quick: bool) -> Result<String, String> {
        let mut out = String::with_capacity(template.len());
        let mut rest = template;
        while let Some(start) = rest.find("{{") {
            out.push_str(&rest[..start]);
            let after = &rest[start + 2..];
            let end = after
                .find("}}")
                .ok_or_else(|| format!("{}: unclosed `{{{{` in the template", self.id))?;
            out.push_str(&self.placeholder(&after[..end], quick)?);
            rest = &after[end + 2..];
        }
        out.push_str(rest);
        Ok(out)
    }

    fn placeholder(&self, token: &str, quick: bool) -> Result<String, String> {
        let parts: Vec<&str> = token.split(':').collect();
        let bad = || format!("{}: bad placeholder `{{{{{token}}}}}`", self.id);
        match parts.as_slice() {
            ["LEN"] => Ok((self.param("M", quick)? * self.param("K", quick)?).to_string()),
            [name] => Ok(self.param(name, quick)?.to_string()),
            [shape @ ("COL" | "ROW" | "VEC"), seed, p] => {
                let seed = parse_seed(seed).ok_or_else(bad)?;
                let n = self.param(p, quick)?;
                let items: Vec<String> = (0..n).map(|i| number(val(seed, i))).collect();
                Ok(match *shape {
                    "COL" => format!(
                        "[{}]",
                        items
                            .iter()
                            .map(|v| format!("[{v}]"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    "ROW" => format!("[[{}]]", items.join(", ")),
                    _ => format!("[{}]", items.join(", ")),
                })
            }
            ["KMAT", p] => {
                let n = self.param(p, quick)?;
                let rows: Vec<String> = (0..n)
                    .map(|i| {
                        let row: Vec<String> = (0..n).map(|j| number(k_entry(n, i, j))).collect();
                        format!("[{}]", row.join(", "))
                    })
                    .collect();
                Ok(format!("[{}]", rows.join(", ")))
            }
            _ => Err(bad()),
        }
    }
}

fn parse_seed(s: &str) -> Option<i64> {
    if s == "ones" {
        Some(-1)
    } else {
        s.parse().ok()
    }
}

/// A number as a Sky literal that reads back as the same `f64`.
fn number(v: f64) -> String {
    format!("{v}")
}

/// The deterministic input data every language builds: a multiple of 1/64 in
/// `[0, 1)`, so products and short sums of these are exact in `f64` and the
/// inputs are bit-identical everywhere. A negative seed gives all ones.
pub fn val(seed: i64, i: u64) -> f64 {
    if seed < 0 {
        return 1.0;
    }
    let i = i as i64;
    ((i * (2 * seed + 5) + 7 * seed + 1) % 64) as f64 / 64.0
}

/// Entry `(i, j)` of the Cholesky workload's matrix `c1 c1' + c2 c2' + n I`.
pub fn k_entry(n: u64, i: u64, j: u64) -> f64 {
    let diag = if i == j { n as f64 } else { 0.0 };
    val(0, i) * val(0, j) + val(1, i) * val(1, j) + diag
}

fn case(
    id: &str,
    title: &str,
    kernel: &'static str,
    kind: Kind,
    params: &[(&'static str, u64)],
    quick: &[(&'static str, u64)],
) -> Case {
    Case {
        id: id.to_string(),
        title: title.to_string(),
        kernel,
        sky: kernel,
        kind,
        params: params.to_vec(),
        quick: quick.to_vec(),
        tolerance: 1e-8,
        lib: false,
        numpy: false,
    }
}

/// Every case, in report order.
pub fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    v.push(Case {
        tolerance: 1e-9,
        ..case(
            "nbody",
            "N-body, 5 bodies, 1e5 steps",
            "nbody",
            Kind::SmallVector,
            &[("STEPS", 100_000)],
            &[("STEPS", 200)],
        )
    });
    for (n, r, qn) in [(64, 100, 16), (256, 8, 32), (512, 2, 48)] {
        v.push(Case {
            lib: true,
            numpy: true,
            ..case(
                &format!("matmul-{n}"),
                &format!("dense matmul {n}x{n}, {r} product(s)"),
                "matmul",
                Kind::Matmul,
                &[("N", n), ("R", r)],
                &[("N", qn), ("R", 2)],
            )
        });
    }
    v.push(Case {
        lib: true,
        numpy: true,
        ..case(
            "cholesky-256",
            "Cholesky solve n=256, 10 solves",
            "cholesky",
            Kind::Cholesky,
            &[("N", 256), ("R", 10)],
            &[("N", 16), ("R", 2)],
        )
    });
    v.push(Case {
        sky: "cholesky_literal",
        lib: true,
        numpy: true,
        ..case(
            "cholesky-32",
            "Cholesky solve n=32 (literal K), 5000 solves",
            "cholesky",
            Kind::Cholesky,
            &[("N", 32), ("R", 5000)],
            &[("N", 8), ("R", 3)],
        )
    });
    for (label, m, k, r, qm, qk) in [
        ("1e6", 1000, 1000, 20, 20, 30),
        ("1e7", 1000, 10_000, 2, 20, 50),
    ] {
        v.push(Case {
            numpy: true,
            ..case(
                &format!("elementwise-{label}"),
                &format!("sqrt(x*x + y*y)*c + d over {label} elements, {r} passes"),
                "elementwise",
                Kind::ArrayChain,
                &[("M", m), ("K", k), ("R", r)],
                &[("M", qm), ("K", qk), ("R", 2)],
            )
        });
        v.push(Case {
            numpy: true,
            ..case(
                &format!("linear_chain-{label}"),
                &format!("(x+y)*c + (x-y)*d over {label} elements, {r} passes"),
                "linear_chain",
                Kind::ArrayChain,
                &[("M", m), ("K", k), ("R", r)],
                &[("M", qm), ("K", qk), ("R", 2)],
            )
        });
    }
    // Escape counts are chaotic near the set's boundary: a fused multiply-add
    // (clang's default contraction) flips a few boundary points, about 0.3%.
    v.push(Case {
        tolerance: 1e-2,
        ..case(
            "mandelbrot",
            "Mandelbrot 300x300, 500 max iterations",
            "mandelbrot",
            Kind::Scalar,
            &[("W", 300), ("MAXIT", 500)],
            &[("W", 24), ("MAXIT", 60)],
        )
    });
    v.push(case(
        "rk4",
        "RK4 damped pendulum, 2e5 steps",
        "rk4",
        Kind::SmallVector,
        &[("STEPS", 200_000)],
        &[("STEPS", 300)],
    ));
    v.push(Case {
        tolerance: 0.0,
        ..case(
            "fib-30",
            "recursive fib(30)",
            "fib",
            Kind::Calls,
            &[("N", 30)],
            &[("N", 15)],
        )
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_ids_are_unique() {
        let mut ids: Vec<_> = cases().into_iter().map(|c| c.id).collect();
        ids.sort();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n);
    }

    #[test]
    fn data_is_exact_multiples_of_a_64th() {
        for seed in 0..4 {
            for i in 0..200 {
                let v = val(seed, i);
                assert!((0.0..1.0).contains(&v));
                assert_eq!(v * 64.0, (v * 64.0).round());
            }
        }
        assert_eq!(val(-1, 7), 1.0);
    }

    #[test]
    fn placeholders_fill() {
        let c = cases().into_iter().find(|c| c.id == "matmul-64").unwrap();
        let out = c
            .render_sky("N = {{N}}\nA = {{COL:0:N}}\nB = {{ROW:ones:N}}\n", true)
            .unwrap();
        assert!(
            out.starts_with("N = 16\nA = [[") && !out.contains("{{"),
            "{out}"
        );
        // A column has one bracket pair per entry, a row has one in all.
        assert_eq!(out.lines().nth(1).unwrap().matches('[').count(), 17);
        assert_eq!(out.lines().nth(2).unwrap().matches('[').count(), 2);
        assert!(c.render_sky("{{NOPE}}", false).is_err());
        assert!(c.render_sky("{{COL:x:N}}", false).is_err());
        assert!(c.render_sky("{{N", false).is_err());
    }

    #[test]
    fn the_k_matrix_is_symmetric_and_diagonally_heavy() {
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(k_entry(8, i, j), k_entry(8, j, i));
            }
            assert!(k_entry(8, i, i) >= 8.0);
        }
    }
}
