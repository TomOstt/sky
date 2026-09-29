//! Symbolic tensor shapes and equality-only shape unification.
//!
//! A [`Shape`] is an ordered list of [`ShapeDim`]s. Each dim is a known size
//! or a shape variable (`Matrix[N, N]`). v0.1 solves equality constraints only.
//! There is no dimension arithmetic (`N+1`, `2*N`), so [`ShapeDim`] has no
//! way to express it.
//!
//! [`ShapeUnifier`] keeps an idempotent substitution: every binding's
//! right-hand side is a known size or an unbound variable. Every unifying
//! operation works on a copy and commits only on success, so a failed call
//! leaves the unifier unchanged. Nothing here panics.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::tags::TagSet;

/// A shape variable such as `N`. Identity is the numeric id; the name is for display.
#[derive(Clone, Debug)]
pub struct ShapeVar {
    id: u64,
    name: Arc<str>,
}

impl ShapeVar {
    /// The variable's id within its [`ShapeUnifier`].
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The display name given at creation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A variable with a chosen id, for readers of stored data (the SIR text
    /// format prints `N#3` and must rebuild exactly that variable). The
    /// checker uses [`ShapeUnifier::fresh`], which keeps ids unique within
    /// one unifier; a caller of this function owns that guarantee.
    pub fn from_parts(id: u64, name: impl Into<Arc<str>>) -> ShapeVar {
        ShapeVar {
            id,
            name: name.into(),
        }
    }
}

impl PartialEq for ShapeVar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShapeVar {}

impl Hash for ShapeVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Ord for ShapeVar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShapeVar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for ShapeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// One dimension of a shape: a known size or a shape variable.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ShapeDim {
    /// A concrete size.
    Known(u64),
    /// A symbolic size, solved by unification.
    Var(ShapeVar),
}

impl ShapeDim {
    /// A known size from a signed source literal. Negative sizes are an error.
    pub fn try_from_i64(n: i64) -> Result<ShapeDim, ShapeError> {
        u64::try_from(n)
            .map(ShapeDim::Known)
            .map_err(|_| ShapeErrorKind::NegativeDim(n).into())
    }
}

impl From<u64> for ShapeDim {
    fn from(n: u64) -> Self {
        ShapeDim::Known(n)
    }
}

impl From<ShapeVar> for ShapeDim {
    fn from(v: ShapeVar) -> Self {
        ShapeDim::Var(v)
    }
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Known(n) => write!(f, "{n}"),
            ShapeDim::Var(v) => v.fmt(f),
        }
    }
}

/// An ordered list of dimensions. Rank 0 is a scalar, rank 1 a vector,
/// and rank 2 a matrix (`Matrix[M, N]`).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Shape {
    dims: Vec<ShapeDim>,
}

impl Shape {
    /// A shape with the given dimensions, outermost first.
    pub fn new(dims: Vec<ShapeDim>) -> Self {
        Shape { dims }
    }

    /// The rank-0 shape.
    pub fn scalar() -> Self {
        Shape { dims: Vec::new() }
    }

    /// `Vector[n]`.
    pub fn vector(n: impl Into<ShapeDim>) -> Self {
        Shape {
            dims: vec![n.into()],
        }
    }

    /// `Matrix[rows, cols]`.
    pub fn matrix(rows: impl Into<ShapeDim>, cols: impl Into<ShapeDim>) -> Self {
        Shape {
            dims: vec![rows.into(), cols.into()],
        }
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The dimensions, outermost first.
    pub fn dims(&self) -> &[ShapeDim] {
        &self.dims
    }
}

impl fmt::Display for Shape {
    /// `Scalar`, `Vector[3]`, `Matrix[M,N]`, or `Tensor[2,3,4]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let head = match self.rank() {
            0 => return f.write_str("Scalar"),
            1 => "Vector",
            2 => "Matrix",
            _ => "Tensor",
        };
        write!(f, "{head}[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

/// A dimension as written and the value it had when a constraint failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DimValue {
    /// The dimension as it appeared in the constraint.
    pub written: ShapeDim,
    /// Its value under the substitution at the point of failure.
    pub value: ShapeDim,
}

impl fmt::Display for DimValue {
    /// `4`, or `N = 3` when a variable had already been solved.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.written == self.value {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} = {}", self.written, self.value)
        }
    }
}

/// Two dimensions that cannot be equal (returned by [`ShapeUnifier::unify_dim`]).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DimConflict {
    /// The expected side.
    pub expected: DimValue,
    /// The found side.
    pub found: DimValue,
}

/// What went wrong in a [`ShapeError`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShapeErrorKind {
    /// The shapes have different ranks.
    RankMismatch {
        /// The expected shape, as written.
        expected: Shape,
        /// The found shape, as written.
        found: Shape,
    },
    /// The shapes differ at one axis (0-based `index`).
    DimMismatch {
        /// The 0-based axis.
        index: usize,
        /// The expected dimension.
        expected: DimValue,
        /// The found dimension.
        found: DimValue,
        /// The expected shape, as written.
        expected_shape: Shape,
        /// The found shape, as written.
        found_shape: Shape,
    },
    /// `lhs @ rhs` where the contracted dimensions differ.
    MatmulInner {
        /// Left operand shape.
        lhs: Shape,
        /// Right operand shape.
        rhs: Shape,
        /// The left operand's contracted dimension.
        left: DimValue,
        /// The right operand's contracted dimension.
        right: DimValue,
    },
    /// `lhs @ rhs` where an operand is not a vector or matrix.
    MatmulRank {
        /// Left operand shape.
        lhs: Shape,
        /// Right operand shape.
        rhs: Shape,
    },
    /// Inverse of a matrix whose row and column counts cannot be equal.
    NotSquare {
        /// The matrix shape.
        shape: Shape,
        /// Its row count.
        rows: DimValue,
        /// Its column count.
        cols: DimValue,
    },
    /// Inverse of a rank-1 shape.
    InvertVector {
        /// The vector shape.
        shape: Shape,
    },
    /// Inverse of a shape with rank 3 or more.
    InvertTensor {
        /// The tensor shape.
        shape: Shape,
    },
    /// Transpose of a shape that is neither a scalar nor a matrix.
    TransposeRank {
        /// The shape.
        shape: Shape,
    },
    /// Matrix-property tags on something that is not a square matrix.
    TagsNeedSquareMatrix {
        /// The shape.
        shape: Shape,
        /// The requested tags.
        tags: TagSet,
    },
    /// A negative size from a source literal.
    NegativeDim(i64),
}

impl fmt::Display for ShapeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeErrorKind::RankMismatch { expected, found } => write!(
                f,
                "rank mismatch: expected rank {} ({expected}), found rank {} ({found})",
                expected.rank(),
                found.rank()
            ),
            ShapeErrorKind::DimMismatch {
                index,
                expected,
                found,
                expected_shape,
                found_shape,
            } => write!(
                f,
                "shape mismatch at axis {index}: expected {expected}, found {found} \
                 ({expected_shape} vs {found_shape})"
            ),
            ShapeErrorKind::MatmulInner {
                lhs,
                rhs,
                left,
                right,
            } => write!(
                f,
                "`@` inner dimensions differ: {left} vs {right} in {lhs} @ {rhs}"
            ),
            ShapeErrorKind::MatmulRank { lhs, rhs } => write!(
                f,
                "`@` needs vector or matrix operands, found {lhs} @ {rhs}"
            ),
            ShapeErrorKind::NotSquare { shape, rows, cols } => write!(
                f,
                "cannot invert a non-square matrix {shape}: {rows} rows, {cols} columns"
            ),
            ShapeErrorKind::InvertVector { .. } => f.write_str("cannot invert a vector"),
            ShapeErrorKind::InvertTensor { shape } => {
                write!(f, "cannot invert a rank-{} tensor", shape.rank())
            }
            ShapeErrorKind::TransposeRank { shape } => {
                write!(f, "cannot transpose {shape}: transpose needs a matrix")
            }
            ShapeErrorKind::TagsNeedSquareMatrix { shape, tags } => write!(
                f,
                "matrix property tags {tags} need a square matrix, found {shape}"
            ),
            ShapeErrorKind::NegativeDim(n) => {
                write!(f, "shape dimension must be non-negative, found {n}")
            }
        }
    }
}

/// A shape-checking error. `Display` gives the diagnostic message.
///
/// The details are boxed (see [`ShapeError::kind`]) so `Result<_, ShapeError>`
/// stays one pointer wide on the checker's hot paths.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShapeError(Box<ShapeErrorKind>);

impl ShapeError {
    /// What went wrong.
    pub fn kind(&self) -> &ShapeErrorKind {
        &self.0
    }

    /// The details, by value.
    pub fn into_kind(self) -> ShapeErrorKind {
        *self.0
    }
}

impl From<ShapeErrorKind> for ShapeError {
    fn from(kind: ShapeErrorKind) -> Self {
        ShapeError(Box::new(kind))
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ShapeError {}

type Subst = BTreeMap<ShapeVar, ShapeDim>;

fn resolve_in(subst: &Subst, d: &ShapeDim) -> ShapeDim {
    // The substitution is idempotent, so one lookup is enough.
    match d {
        ShapeDim::Var(v) => subst.get(v).cloned().unwrap_or_else(|| d.clone()),
        known => known.clone(),
    }
}

fn bind_in(subst: &mut Subst, v: &ShapeVar, to: ShapeDim) {
    let target = ShapeDim::Var(v.clone());
    for rhs in subst.values_mut() {
        if *rhs == target {
            *rhs = to.clone();
        }
    }
    subst.insert(v.clone(), to);
}

fn unify_in(subst: &mut Subst, expected: &ShapeDim, found: &ShapeDim) -> Result<(), DimConflict> {
    let e = resolve_in(subst, expected);
    let f = resolve_in(subst, found);
    match (&e, &f) {
        _ if e == f => Ok(()),
        (ShapeDim::Var(v), _) => {
            bind_in(subst, v, f);
            Ok(())
        }
        (_, ShapeDim::Var(v)) => {
            bind_in(subst, v, e);
            Ok(())
        }
        (ShapeDim::Known(_), ShapeDim::Known(_)) => Err(DimConflict {
            expected: DimValue {
                written: expected.clone(),
                value: e,
            },
            found: DimValue {
                written: found.clone(),
                value: f,
            },
        }),
    }
}

/// Creates shape variables and solves equality constraints between shapes.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ShapeUnifier {
    next_id: u64,
    /// Idempotent: right-hand sides are known sizes or unbound variables.
    subst: Subst,
}

impl ShapeUnifier {
    /// An empty unifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh, unbound shape variable with a display name.
    pub fn fresh(&mut self, name: impl Into<Arc<str>>) -> ShapeVar {
        let id = self.next_id;
        // Saturating rather than panicking; running out of u64 ids would
        // take 2^64 calls.
        self.next_id = self.next_id.saturating_add(1);
        ShapeVar {
            id,
            name: name.into(),
        }
    }

    /// `d` with its variable replaced by its solution, if any.
    pub fn apply_dim(&self, d: &ShapeDim) -> ShapeDim {
        resolve_in(&self.subst, d)
    }

    /// The current solution for `v` (`v` itself if unbound).
    pub fn resolve(&self, v: &ShapeVar) -> ShapeDim {
        self.apply_dim(&ShapeDim::Var(v.clone()))
    }

    /// `s` with every solved variable replaced.
    pub fn apply(&self, s: &Shape) -> Shape {
        Shape {
            dims: s.dims.iter().map(|d| self.apply_dim(d)).collect(),
        }
    }

    /// Makes two dimensions equal. On error the unifier is unchanged.
    pub fn unify_dim(&mut self, expected: &ShapeDim, found: &ShapeDim) -> Result<(), DimConflict> {
        let mut subst = self.subst.clone();
        unify_in(&mut subst, expected, found)?;
        self.subst = subst;
        Ok(())
    }

    /// Makes two shapes equal: same rank, then axis by axis. Reports a
    /// [`ShapeErrorKind::RankMismatch`] or the first [`ShapeErrorKind::DimMismatch`].
    /// On error the unifier is unchanged.
    pub fn unify(&mut self, expected: &Shape, found: &Shape) -> Result<(), ShapeError> {
        if expected.rank() != found.rank() {
            return Err(ShapeErrorKind::RankMismatch {
                expected: expected.clone(),
                found: found.clone(),
            }
            .into());
        }
        let mut subst = self.subst.clone();
        for (index, (e, f)) in expected.dims.iter().zip(&found.dims).enumerate() {
            unify_in(&mut subst, e, f).map_err(|c| -> ShapeError {
                ShapeErrorKind::DimMismatch {
                    index,
                    expected: c.expected,
                    found: c.found,
                    expected_shape: expected.clone(),
                    found_shape: found.clone(),
                }
                .into()
            })?;
        }
        self.subst = subst;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        assert_eq!(Shape::scalar().to_string(), "Scalar");
        assert_eq!(Shape::vector(3).to_string(), "Vector[3]");
        assert_eq!(Shape::matrix(n.clone(), n).to_string(), "Matrix[N,N]");
        let t = Shape::new(vec![2.into(), 3.into(), 4.into()]);
        assert_eq!(t.to_string(), "Tensor[2,3,4]");
    }

    #[test]
    fn solves_square_from_known() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        let sym = Shape::matrix(n.clone(), n.clone());
        u.unify(&sym, &Shape::matrix(5, 5)).unwrap();
        assert_eq!(u.resolve(&n), ShapeDim::Known(5));
        assert_eq!(u.apply(&sym), Shape::matrix(5, 5));
    }

    #[test]
    fn rank_mismatch() {
        let mut u = ShapeUnifier::new();
        let err = u
            .unify(&Shape::matrix(3, 3), &Shape::vector(3))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "rank mismatch: expected rank 2 (Matrix[3,3]), found rank 1 (Vector[3])"
        );
    }

    #[test]
    fn var_bound_to_two_values_fails_and_leaves_state() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        let before = u.clone();
        let err = u
            .unify(&Shape::matrix(n.clone(), n.clone()), &Shape::matrix(3, 4))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "shape mismatch at axis 1: expected N = 3, found 4 (Matrix[N,N] vs Matrix[3,4])"
        );
        // Axis 0 had bound N = 3 on the working copy, but nothing was committed.
        assert_eq!(u, before);
        assert_eq!(u.resolve(&n), ShapeDim::Var(n));
    }

    #[test]
    fn variable_chains_stay_idempotent() {
        let mut u = ShapeUnifier::new();
        let (a, b, c) = (u.fresh("A"), u.fresh("B"), u.fresh("C"));
        u.unify_dim(&a.clone().into(), &b.clone().into()).unwrap();
        u.unify_dim(&b.clone().into(), &c.clone().into()).unwrap();
        u.unify_dim(&c.clone().into(), &7.into()).unwrap();
        for v in [&a, &b, &c] {
            assert_eq!(u.resolve(v), ShapeDim::Known(7));
        }
    }

    #[test]
    fn negative_literal_is_an_error() {
        assert_eq!(ShapeDim::try_from_i64(3), Ok(ShapeDim::Known(3)));
        let err = ShapeDim::try_from_i64(-2).unwrap_err();
        assert_eq!(
            err.to_string(),
            "shape dimension must be non-negative, found -2"
        );
        assert!(ShapeDim::try_from_i64(i64::MIN).is_err());
    }
}
