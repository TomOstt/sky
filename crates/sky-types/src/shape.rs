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
//!
//! # Provenance and explanations
//!
//! While solving, the unifier records where each variable was bound to a
//! size or to another variable: the [`Origin`] the checker installed with
//! [`ShapeUnifier::set_origin`] (a line and a short description such as
//! `` `A @ B` ``). A conflict then carries those bindings in its
//! [`DimValue`]s, so the error can say "`N` was fixed to 4 at line 2". The
//! provenance lives in the same working copy as the substitution, so it
//! commits and rolls back with it: a failed call records nothing.
//! [`ShapeError`] turns the kind into a headline, a label, notes and a help
//! line ([`ShapeError::notes`], [`ShapeError::help`]); the checker attaches
//! them to the diagnostic.

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
    /// How `written` got that value, when it is a variable (see [`Binding`]).
    pub bindings: Vec<Binding>,
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

/// Where a binding came from: the source line of the operation or annotation
/// that forced it and a short description of it (already quoted user text).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Origin {
    /// The 1-based source line, if the checker knew it.
    pub line: Option<u32>,
    /// What bound the variable, e.g. `` `A @ B` ``.
    pub what: Arc<str>,
}

/// One recorded step in a variable's history: it was set equal to `to`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Binding {
    /// A known size, or another variable it was tied to.
    pub to: ShapeDim,
    /// What forced it.
    pub origin: Origin,
}

/// Most bindings kept per variable. A variable is tied to a few others at
/// most before it is fixed, and only the fixing matters most, so the cap keeps
/// the first few ties and always leaves room for a known size (see
/// `record_binding`). It bounds memory however long the program is.
pub const MAX_BINDINGS_PER_VAR: usize = 4;

/// Where a shape error happened, so its wording can name the operation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShapeContext {
    /// An elementwise operator (`+`, `-`, ...), by its symbol.
    Operator(Arc<str>),
    /// A declared shape (`A: Matrix[3, N]`) against the value's shape.
    Declaration,
    /// A call argument against its parameter's shape.
    Argument,
}

/// Which operand of a matrix product, if transposed, would make it valid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transpose {
    /// The left operand.
    Left,
    /// The right operand.
    Right,
    /// Either one.
    Either,
}

/// Two dimensions that cannot be equal (returned, boxed to keep `Result`
/// small, by [`ShapeUnifier::unify_dim`]).
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
        /// The expected shape, with solved variables substituted.
        expected: Shape,
        /// The found shape, with solved variables substituted.
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
        /// The expected shape, with solved variables substituted.
        expected_shape: Shape,
        /// The found shape, with solved variables substituted.
        found_shape: Shape,
        /// Both are matrices and the found shape with its axes swapped equals
        /// the expected one (certainly, after solving), so a transpose fixes it.
        transposed_matches: bool,
    },
    /// `lhs @ rhs` where the contracted dimensions differ.
    MatmulInner {
        /// Left operand shape (solved variables substituted).
        lhs: Shape,
        /// Right operand shape (solved variables substituted).
        rhs: Shape,
        /// The left operand's contracted dimension.
        left: DimValue,
        /// The right operand's contracted dimension.
        right: DimValue,
        /// The operand whose transpose would certainly make the product valid.
        transpose: Option<Transpose>,
    },
    /// `lhs @ rhs` where an operand is not a vector or matrix.
    MatmulRank {
        /// Left operand shape (solved variables substituted).
        lhs: Shape,
        /// Right operand shape (solved variables substituted).
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

/// `a matrix`, `a vector`, `a scalar`, `a rank-4 tensor`.
fn rank_phrase(rank: usize) -> String {
    match rank {
        0 => "a scalar".to_string(),
        1 => "a vector".to_string(),
        2 => "a matrix".to_string(),
        n => format!("a rank-{n} tensor"),
    }
}

/// `[3, N, 4]`; `mark` wraps one axis in `*` (`[3, *N*, 4]`).
fn axes_text(dims: &[ShapeDim], mark: Option<usize>) -> String {
    let parts: Vec<String> = dims
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if Some(i) == mark {
                format!("*{d}*")
            } else {
                d.to_string()
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// `1 axis` / `4 axes`.
fn axes_count(rank: usize) -> String {
    if rank == 1 {
        "1 axis".to_string()
    } else {
        format!("{rank} axes")
    }
}

/// The history of a solved variable as notes: `` `N` was fixed to 4 at line 2
/// (`A: Matrix[3, N]` ...) ``. Empty for a known size or a variable with no
/// recorded history.
fn provenance_notes(d: &DimValue, out: &mut Vec<String>) {
    let ShapeDim::Var(v) = &d.written else {
        return;
    };
    for b in &d.bindings {
        let what = match &b.to {
            ShapeDim::Known(n) => format!("`{v}` was fixed to {n}"),
            ShapeDim::Var(w) => format!("`{v}` was set equal to `{w}`"),
        };
        let note = match b.origin.line {
            Some(line) => format!("{what} at line {line} ({})", b.origin.what),
            None => format!("{what} by {}", b.origin.what),
        };
        // Both sides of a conflict can be the same variable.
        if !out.contains(&note) {
            out.push(note);
        }
    }
}

/// An aligned per-axis table (one note per row), the offending axis marked:
///
/// ```text
/// axis      0  1
/// declared  3  N
/// value     3  4
/// differs      ^
/// ```
fn axis_table(
    names: (&str, &str),
    rows: (&[ShapeDim], &[ShapeDim]),
    index: usize,
    bad: (&DimValue, &DimValue),
) -> Vec<String> {
    let cell = |dims: &[ShapeDim], side: &DimValue, i: usize| match dims.get(i) {
        Some(_) if i == index => side.to_string(),
        Some(d) => d.to_string(),
        None => String::new(),
    };
    let rank = rows.0.len().max(rows.1.len());
    let widths: Vec<usize> = (0..rank)
        .map(|i| {
            [
                i.to_string().len(),
                cell(rows.0, bad.0, i).chars().count(),
                cell(rows.1, bad.1, i).chars().count(),
            ]
            .into_iter()
            .max()
            .unwrap_or(1)
        })
        .collect();
    let label_w = names.0.len().max(names.1.len()).max("differs".len());
    let line = |label: &str, cells: Vec<String>| {
        let body: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!("{c:<w$}"))
            .collect();
        format!("{label:<label_w$}  {}", body.join("  "))
            .trim_end()
            .to_string()
    };
    let header = line("axis", (0..rank).map(|i| i.to_string()).collect());
    let a = line(names.0, (0..rank).map(|i| cell(rows.0, bad.0, i)).collect());
    let b = line(names.1, (0..rank).map(|i| cell(rows.1, bad.1, i)).collect());
    let pad: usize = widths.iter().take(index).map(|w| w + 2).sum();
    let marker = format!("{:<label_w$}  {}^", "differs", " ".repeat(pad));
    vec![header, a, b, marker]
}

/// The headline for `kind`, worded for `ctx` when there is one.
fn headline(kind: &ShapeErrorKind, ctx: Option<&ShapeContext>) -> String {
    match kind {
        ShapeErrorKind::RankMismatch { expected, found } => {
            let (e, f) = (rank_phrase(expected.rank()), rank_phrase(found.rank()));
            match ctx {
                Some(ShapeContext::Operator(op)) if &**op == "+" => {
                    format!("cannot add {f} to {e}")
                }
                Some(ShapeContext::Operator(op)) if &**op == "-" => {
                    format!("cannot subtract {f} from {e}")
                }
                Some(ShapeContext::Operator(op)) => {
                    format!("cannot apply `{op}` to {e} and {f}")
                }
                Some(ShapeContext::Declaration) => {
                    format!("declared as {e}, but the value is {f}")
                }
                Some(ShapeContext::Argument) => {
                    format!("the parameter takes {e}, but the argument is {f}")
                }
                None => format!("rank mismatch: expected {e} ({expected}), found {f} ({found})"),
            }
        }
        ShapeErrorKind::DimMismatch {
            index,
            expected,
            found,
            expected_shape,
            found_shape,
            ..
        } => match ctx {
            Some(ShapeContext::Operator(op)) if &**op == "+" => {
                format!("cannot add {found_shape} to {expected_shape}")
            }
            Some(ShapeContext::Operator(op)) if &**op == "-" => {
                format!("cannot subtract {found_shape} from {expected_shape}")
            }
            Some(ShapeContext::Operator(op)) => {
                format!("cannot apply `{op}` to {expected_shape} and {found_shape}")
            }
            Some(ShapeContext::Declaration) => {
                format!("declared {expected_shape}, but the value is {found_shape}")
            }
            Some(ShapeContext::Argument) => {
                format!("the parameter takes {expected_shape}, but the argument is {found_shape}")
            }
            None => format!(
                "shape mismatch at axis {index}: expected {expected}, found {found} \
                 ({expected_shape} vs {found_shape})"
            ),
        },
        ShapeErrorKind::MatmulInner { lhs, rhs, .. } | ShapeErrorKind::MatmulRank { lhs, rhs } => {
            format!("cannot multiply {lhs} by {rhs}")
        }
        ShapeErrorKind::NotSquare { shape, rows, cols } => {
            format!("cannot invert a non-square matrix {shape}: {rows} rows, {cols} columns")
        }
        ShapeErrorKind::InvertVector { .. } => "cannot invert a vector".to_string(),
        ShapeErrorKind::InvertTensor { shape } => {
            format!("cannot invert a rank-{} tensor", shape.rank())
        }
        ShapeErrorKind::TransposeRank { shape } => {
            format!("cannot transpose {shape}: transpose needs a matrix")
        }
        ShapeErrorKind::TagsNeedSquareMatrix { shape, tags } => {
            format!("matrix property tags {tags} need a square matrix, found {shape}")
        }
        ShapeErrorKind::NegativeDim(n) => {
            format!("shape dimension must be non-negative, found {n}")
        }
    }
}

/// `[m, k] @ [k, n] -> [m, n]` for the operand ranks, or `None` if `@` does not apply.
fn matmul_formula(lhs_rank: usize, rhs_rank: usize) -> Option<&'static str> {
    match (lhs_rank, rhs_rank) {
        (2, 2) => Some("[m, k] @ [k, n] -> [m, n]"),
        (2, 1) => Some("[m, k] @ [k] -> [m]"),
        (1, 2) => Some("[k] @ [k, n] -> [n]"),
        (1, 1) => Some("[k] @ [k] -> [] (a scalar)"),
        _ => None,
    }
}

impl fmt::Display for ShapeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&headline(self, None))
    }
}

/// A shape-checking error. `Display` gives the headline; the label, notes and
/// help explain it (see the module docs).
///
/// The details are boxed (see [`ShapeError::kind`]) so `Result<_, ShapeError>`
/// stays one pointer wide on the checker's hot paths.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShapeError(Box<Detail>);

#[derive(Clone, PartialEq, Eq, Debug)]
struct Detail {
    kind: ShapeErrorKind,
    context: Option<ShapeContext>,
}

impl ShapeError {
    /// What went wrong.
    pub fn kind(&self) -> &ShapeErrorKind {
        &self.0.kind
    }

    /// The details, by value.
    pub fn into_kind(self) -> ShapeErrorKind {
        self.0.kind
    }

    /// The same error, worded for where it happened (an operator, a
    /// declaration or a call argument). The kind is unchanged.
    #[must_use]
    pub fn in_context(mut self, ctx: ShapeContext) -> Self {
        self.0.context = Some(ctx);
        self
    }

    /// The short text for the primary underline, when the primary span is the
    /// operator (`inner dimensions differ: 4 vs 5`). `None` when the caller's
    /// own label (`this is Vector[4]`) says it better.
    pub fn label(&self) -> Option<String> {
        let op = matches!(self.0.context, Some(ShapeContext::Operator(_)));
        match &self.0.kind {
            ShapeErrorKind::MatmulInner { left, right, .. } => {
                Some(format!("inner dimensions differ: {left} vs {right}"))
            }
            ShapeErrorKind::MatmulRank { .. } => Some("`@` needs vectors or matrices".to_string()),
            ShapeErrorKind::DimMismatch {
                index,
                expected,
                found,
                ..
            } if op => Some(format!("axis {index} differs: {expected} vs {found}")),
            ShapeErrorKind::RankMismatch { expected, found } if op => {
                Some(format!("rank {} vs rank {}", expected.rank(), found.rank()))
            }
            _ => None,
        }
    }

    /// `= note:` lines: the contraction or the per-axis comparison, then where
    /// each conflicting size variable got its value.
    pub fn notes(&self) -> Vec<String> {
        let mut out = Vec::new();
        let names = match &self.0.context {
            Some(ShapeContext::Operator(_)) => ("left", "right"),
            Some(ShapeContext::Declaration) => ("declared", "value"),
            Some(ShapeContext::Argument) => ("parameter", "argument"),
            None => ("expected", "found"),
        };
        match &self.0.kind {
            ShapeErrorKind::RankMismatch { expected, found } => {
                out.push(format!(
                    "{}: {} has {}, {}: {} has {}",
                    names.0,
                    axes_text(expected.dims(), None),
                    axes_count(expected.rank()),
                    names.1,
                    axes_text(found.dims(), None),
                    axes_count(found.rank()),
                ));
                out.push(
                    "axes are numbered from 0 and have no names yet; only shapes of the same \
                     rank can be compared axis by axis"
                        .to_string(),
                );
            }
            ShapeErrorKind::DimMismatch {
                index,
                expected,
                found,
                expected_shape,
                found_shape,
                ..
            } => {
                out.extend(axis_table(
                    names,
                    (expected_shape.dims(), found_shape.dims()),
                    *index,
                    (expected, found),
                ));
                if expected_shape.rank() == 2 {
                    out.push("in a matrix, axis 0 is the rows and axis 1 the columns".to_string());
                }
                provenance_notes(expected, &mut out);
                provenance_notes(found, &mut out);
            }
            ShapeErrorKind::MatmulInner {
                lhs,
                rhs,
                left,
                right,
                ..
            } => {
                if let Some(formula) = matmul_formula(lhs.rank(), rhs.rank()) {
                    out.push(format!(
                        "`@` contracts the last axis of the left operand with the first axis of \
                         the right one: {formula}"
                    ));
                }
                out.push(format!(
                    "here {} @ {} needs {} == {}",
                    axes_text(lhs.dims(), lhs.rank().checked_sub(1)),
                    axes_text(rhs.dims(), Some(0)),
                    left.value,
                    right.value,
                ));
                provenance_notes(left, &mut out);
                provenance_notes(right, &mut out);
            }
            ShapeErrorKind::MatmulRank { lhs, rhs } => {
                out.push(format!(
                    "the left operand is {} ({}), the right operand is {} ({})",
                    rank_phrase(lhs.rank()),
                    axes_text(lhs.dims(), None),
                    rank_phrase(rhs.rank()),
                    axes_text(rhs.dims(), None),
                ));
                out.push(
                    "`@` works on vectors and matrices only (no batching or broadcasting yet)"
                        .to_string(),
                );
            }
            ShapeErrorKind::NotSquare { shape, rows, cols } => {
                out.push(format!(
                    "an inverse needs as many rows as columns, but {} has {} rows and {} columns",
                    axes_text(shape.dims(), None),
                    rows.value,
                    cols.value
                ));
                provenance_notes(rows, &mut out);
                provenance_notes(cols, &mut out);
            }
            ShapeErrorKind::InvertVector { shape } => out.push(format!(
                "a vector has one axis ({}); an inverse needs a square matrix",
                axes_text(shape.dims(), None)
            )),
            ShapeErrorKind::InvertTensor { shape } => out.push(format!(
                "a rank-{} tensor has {} ({}); an inverse needs a square matrix",
                shape.rank(),
                axes_count(shape.rank()),
                axes_text(shape.dims(), None)
            )),
            ShapeErrorKind::TransposeRank { shape } => out.push(format!(
                "transpose swaps the two axes of a matrix; this shape has {} ({})",
                axes_count(shape.rank()),
                axes_text(shape.dims(), None)
            )),
            ShapeErrorKind::TagsNeedSquareMatrix { shape, .. } => out.push(format!(
                "properties like Symmetric describe square matrices; this shape has {} ({})",
                axes_count(shape.rank()),
                axes_text(shape.dims(), None)
            )),
            ShapeErrorKind::NegativeDim(_) => {}
        }
        out
    }

    /// A `= help:` line, only when a fix is certain: a transpose that makes
    /// the shapes agree (there is no transpose syntax yet, so it is phrased
    /// as a fact about the shapes), or, for a size-variable conflict, which
    /// variable to look at.
    pub fn help(&self) -> Option<String> {
        let subject = match &self.0.context {
            Some(ShapeContext::Operator(_)) => "the right operand",
            Some(ShapeContext::Declaration) => "the value",
            Some(ShapeContext::Argument) => "the argument",
            None => "the found shape",
        };
        let swapped = |s: &Shape| match s.dims() {
            [a, b] => Shape::matrix(b.clone(), a.clone()).to_string(),
            _ => s.to_string(),
        };
        let var_of = |d: &DimValue| match &d.written {
            ShapeDim::Var(v) => Some(v.to_string()),
            ShapeDim::Known(_) => None,
        };
        let var_help = |a: &DimValue, b: &DimValue| {
            var_of(a).or_else(|| var_of(b)).map(|v| {
                format!(
                    "check the size variable `{v}`: every `{v}` in a module (or, per call, in \
                     a function) stands for one size"
                )
            })
        };
        match &self.0.kind {
            ShapeErrorKind::DimMismatch {
                transposed_matches: true,
                found_shape,
                ..
            } => Some(format!(
                "the shapes would match if {subject} were transposed: {found_shape} would become \
                 {} (there is no transpose syntax yet)",
                swapped(found_shape)
            )),
            ShapeErrorKind::DimMismatch {
                expected, found, ..
            } => var_help(expected, found),
            ShapeErrorKind::MatmulInner {
                lhs,
                rhs,
                transpose: Some(t),
                ..
            } => {
                let left = format!(
                    "the left operand were transposed: {lhs} would become {}",
                    swapped(lhs)
                );
                let right = format!(
                    "the right operand were transposed: {rhs} would become {}",
                    swapped(rhs)
                );
                let which = match t {
                    Transpose::Left => left,
                    Transpose::Right => right,
                    Transpose::Either => format!("{left}, or if {right}"),
                };
                Some(format!(
                    "the shapes would match if {which} (there is no transpose syntax yet)"
                ))
            }
            ShapeErrorKind::MatmulInner { left, right, .. } => var_help(left, right),
            _ => None,
        }
    }
}

impl From<ShapeErrorKind> for ShapeError {
    fn from(kind: ShapeErrorKind) -> Self {
        ShapeError(Box::new(Detail {
            kind,
            context: None,
        }))
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&headline(&self.0.kind, self.0.context.as_ref()))
    }
}

impl std::error::Error for ShapeError {}

/// The substitution and the provenance of its bindings. They are cloned,
/// committed and dropped together, which is what keeps a failed operation from
/// leaving a trace in either.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Solution {
    /// Idempotent: right-hand sides are known sizes or unbound variables.
    map: BTreeMap<ShapeVar, ShapeDim>,
    /// Bindings per variable, at most [`MAX_BINDINGS_PER_VAR`] each.
    prov: BTreeMap<ShapeVar, Vec<Binding>>,
}

fn resolve_in(sol: &Solution, d: &ShapeDim) -> ShapeDim {
    // The substitution is idempotent, so one lookup is enough.
    match d {
        ShapeDim::Var(v) => sol.map.get(v).cloned().unwrap_or_else(|| d.clone()),
        known => known.clone(),
    }
}

fn record_binding(
    prov: &mut BTreeMap<ShapeVar, Vec<Binding>>,
    v: &ShapeVar,
    to: &ShapeDim,
    origin: &Option<Origin>,
) {
    let Some(origin) = origin else {
        return;
    };
    let list = prov.entry(v.clone()).or_default();
    let binding = Binding {
        to: to.clone(),
        origin: origin.clone(),
    };
    if list.len() < MAX_BINDINGS_PER_VAR {
        list.push(binding);
    } else if let (ShapeDim::Known(_), Some(last)) = (to, list.last_mut()) {
        // Full of ties: the size a variable finally got matters most, so it
        // takes the last slot.
        *last = binding;
    }
}

fn bind_in(sol: &mut Solution, v: &ShapeVar, to: ShapeDim, origin: &Option<Origin>) {
    let target = ShapeDim::Var(v.clone());
    let Solution { map, prov } = sol;
    for (k, rhs) in map.iter_mut() {
        if *rhs == target {
            *rhs = to.clone();
            // `k` was tied to `v`; now it has `v`'s solution too.
            record_binding(prov, k, &to, origin);
        }
    }
    record_binding(prov, v, &to, origin);
    map.insert(v.clone(), to);
}

fn unify_in(
    sol: &mut Solution,
    expected: &ShapeDim,
    found: &ShapeDim,
    origin: &Option<Origin>,
) -> Result<(), Box<DimConflict>> {
    let e = resolve_in(sol, expected);
    let f = resolve_in(sol, found);
    match (&e, &f) {
        _ if e == f => Ok(()),
        (ShapeDim::Var(v), _) => {
            bind_in(sol, v, f, origin);
            Ok(())
        }
        (_, ShapeDim::Var(v)) => {
            bind_in(sol, v, e, origin);
            Ok(())
        }
        (ShapeDim::Known(_), ShapeDim::Known(_)) => {
            let side = |written: &ShapeDim, value: ShapeDim| DimValue {
                bindings: match written {
                    ShapeDim::Var(v) => sol.prov.get(v).cloned().unwrap_or_default(),
                    ShapeDim::Known(_) => Vec::new(),
                },
                written: written.clone(),
                value,
            };
            Err(Box::new(DimConflict {
                expected: side(expected, e),
                found: side(found, f),
            }))
        }
    }
}

/// Whether two matrices' shapes agree once the second has its axes swapped,
/// for certain: the solved dimensions are equal (same size, or the same
/// variable), not merely unifiable.
fn transposed_matches(sol: &Solution, expected: &Shape, found: &Shape) -> bool {
    match (expected.dims(), found.dims()) {
        ([a, b], [c, d]) => {
            resolve_in(sol, a) == resolve_in(sol, d) && resolve_in(sol, b) == resolve_in(sol, c)
        }
        _ => false,
    }
}

/// Creates shape variables and solves equality constraints between shapes.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ShapeUnifier {
    next_id: u64,
    solution: Solution,
    /// Stamped on every binding made while set (see [`ShapeUnifier::set_origin`]).
    origin: Option<Origin>,
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

    /// Sets (or clears, with `None`) the [`Origin`] stamped on bindings made
    /// from now on. The checker sets it around each constraint it emits and
    /// clears it after; it does not affect what unifies.
    pub fn set_origin(&mut self, origin: Option<Origin>) {
        self.origin = origin;
    }

    /// The recorded history of `v`: what tied it to other variables and what
    /// fixed its size (at most [`MAX_BINDINGS_PER_VAR`] entries).
    pub fn provenance(&self, v: &ShapeVar) -> &[Binding] {
        self.solution.prov.get(v).map_or(&[], Vec::as_slice)
    }

    /// `d` with its variable replaced by its solution, if any.
    pub fn apply_dim(&self, d: &ShapeDim) -> ShapeDim {
        resolve_in(&self.solution, d)
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
    pub fn unify_dim(
        &mut self,
        expected: &ShapeDim,
        found: &ShapeDim,
    ) -> Result<(), Box<DimConflict>> {
        let mut sol = self.solution.clone();
        unify_in(&mut sol, expected, found, &self.origin)?;
        self.solution = sol;
        Ok(())
    }

    /// Makes two shapes equal: same rank, then axis by axis. Reports a
    /// [`ShapeErrorKind::RankMismatch`] or the first [`ShapeErrorKind::DimMismatch`].
    /// On error the unifier is unchanged.
    pub fn unify(&mut self, expected: &Shape, found: &Shape) -> Result<(), ShapeError> {
        if expected.rank() != found.rank() {
            return Err(ShapeErrorKind::RankMismatch {
                expected: self.apply(expected),
                found: self.apply(found),
            }
            .into());
        }
        let mut sol = self.solution.clone();
        for (index, (e, f)) in expected.dims.iter().zip(&found.dims).enumerate() {
            if let Err(c) = unify_in(&mut sol, e, f, &self.origin) {
                // Judged on the working copy: axes before `index` may have
                // been solved just now, and the swap check should see that.
                let transposed_matches = transposed_matches(&sol, expected, found);
                return Err(ShapeErrorKind::DimMismatch {
                    index,
                    expected: c.expected,
                    found: c.found,
                    // Solved as of before this call: what the operands
                    // were when they arrived, not the working copy's
                    // half-finished bindings.
                    expected_shape: self.apply(expected),
                    found_shape: self.apply(found),
                    transposed_matches,
                }
                .into());
            }
        }
        self.solution = sol;
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
            "rank mismatch: expected a matrix (Matrix[3,3]), found a vector (Vector[3])"
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

    fn origin(line: u32, what: &str) -> Option<Origin> {
        Some(Origin {
            line: Some(line),
            what: Arc::from(what),
        })
    }

    #[test]
    fn a_conflict_reports_where_the_variable_was_fixed() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        u.set_origin(origin(2, "`A @ B`"));
        u.unify_dim(&n.clone().into(), &4.into()).unwrap();
        u.set_origin(origin(3, "`C`"));
        let err = u
            .unify(&Shape::vector(n.clone()), &Shape::vector(5))
            .unwrap_err();
        let notes = err.notes();
        assert!(
            notes.contains(&"`N` was fixed to 4 at line 2 (`A @ B`)".to_string()),
            "{notes:#?}"
        );
        // The failed call left no trace, provenance included.
        assert_eq!(u.provenance(&n).len(), 1);
    }

    #[test]
    fn a_failed_unify_records_no_provenance() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        u.set_origin(origin(1, "`S + T`"));
        let before = u.clone();
        // Axis 0 binds N = 3 on the working copy, axis 1 conflicts.
        u.unify(&Shape::matrix(n.clone(), n.clone()), &Shape::matrix(3, 4))
            .unwrap_err();
        assert_eq!(u, before);
        assert!(u.provenance(&n).is_empty());
    }

    #[test]
    fn provenance_is_capped_and_keeps_the_final_size() {
        let mut u = ShapeUnifier::new();
        let v = u.fresh("V");
        u.set_origin(origin(1, "tie"));
        // A chain of ties: each new variable is set equal to `v`.
        let others: Vec<ShapeVar> = (0..10).map(|i| u.fresh(format!("W{i}"))).collect();
        for w in &others {
            u.unify_dim(&v.clone().into(), &w.clone().into()).unwrap();
        }
        u.set_origin(origin(2, "fix"));
        u.unify_dim(&v.clone().into(), &7.into()).unwrap();
        for x in others.iter().chain([&v]) {
            let history = u.provenance(x);
            assert!(history.len() <= MAX_BINDINGS_PER_VAR, "{history:?}");
            assert_eq!(history.last().map(|b| &b.to), Some(&ShapeDim::Known(7)));
        }
    }

    #[test]
    fn no_origin_means_no_provenance() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        u.unify_dim(&n.clone().into(), &4.into()).unwrap();
        assert!(u.provenance(&n).is_empty());
    }

    #[test]
    fn swapped_matrices_get_a_transpose_help_only_when_certain() {
        let mut u = ShapeUnifier::new();
        let err = u
            .unify(&Shape::matrix(3, 4), &Shape::matrix(4, 3))
            .unwrap_err();
        assert!(
            err.help().unwrap().contains("transposed"),
            "{:?}",
            err.help()
        );
        let err = u
            .unify(&Shape::matrix(3, 4), &Shape::matrix(5, 3))
            .unwrap_err();
        assert_eq!(err.help(), None);
    }

    #[test]
    fn axes_are_aligned_and_the_offending_one_marked() {
        let mut u = ShapeUnifier::new();
        let t = |d: [u64; 4]| Shape::new(d.map(ShapeDim::Known).to_vec());
        let err = u.unify(&t([2, 8, 16, 64]), &t([2, 8, 32, 64])).unwrap_err();
        assert_eq!(
            err.notes(),
            vec![
                "axis      0  1  2   3",
                "expected  2  8  16  64",
                "found     2  8  32  64",
                "differs         ^",
            ]
        );
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
