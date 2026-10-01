//! Arrays in the checker: literals, indexing, the builtins that make arrays
//! (`zeros`, `identity`, `len`), and the proofs of matrix properties for
//! literals.
//!
//! These are the checker's rules for array *values*; the rules for
//! operators over array types (`@`, `+`, `⁻¹`) are in [`crate::rules`], and
//! the walk that calls this module is [`crate::check`]. The spec section
//! "Array literals" is the reference.
//!
//! # Purity
//!
//! Like the rest of the checker, nothing here executes user code. The only
//! values read are constants the source spells out: array sizes
//! (`zeros(3)`), constant indices (`v[0]`), and, for the tag proofs, the
//! entries of a literal made only of numeric literals ([`entry_value`]).

use sky_ast::{Expr, ExprKind, IndexItem};
use sky_diag::{Diagnostic, Span};
use sky_units::Dimension;

use crate::check::{const_exponent, describe, strip_parens, Checker, Ty};
use crate::constant::{entry_value, MAX_ARRAY_ELEMENTS};
use crate::rules::TensorType;
use crate::shape::{Shape, ShapeDim};
use crate::tags::{Tag, TagSet};

/// Most diagnostics of one kind reported for one literal or index: a
/// 10 000-element literal with a wrong unit in every second entry gives a
/// handful of errors and a count, not 5 000 errors.
const MAX_REPORTED: usize = 5;

/// The largest matrix whose PositiveDefinite tag is proven from a literal.
const MAX_PROVEN_SIZE: usize = 32;

/// A Cholesky pivot must exceed this fraction of the largest diagonal entry
/// for the checker to call the matrix positive definite. Far above the
/// rounding error of a factorization of at most [`MAX_PROVEN_SIZE`] rows
/// (about 1e-14 relative), so a proof is never a rounding artifact.
const PIVOT_MARGIN: f64 = 1e-9;

impl Checker<'_> {
    /// `[a, b, c]` or `[[a, b], [c, d]]`. The shape is exact, the element
    /// dimension is that of the entries, and there are no tags.
    pub(crate) fn array_literal(&mut self, elems: &[Expr]) -> Ty {
        let rows = elems
            .iter()
            .filter(|x| matches!(x.kind, ExprKind::Array(_)))
            .count();
        if rows == 0 {
            return self.vector_literal(elems);
        }
        if rows < elems.len() {
            return self.mixed_literal(elems);
        }
        self.matrix_literal(elems)
    }

    fn vector_literal(&mut self, elems: &[Expr]) -> Ty {
        let mut entries = Vec::with_capacity(elems.len());
        let mut ok = self.scalar_elems(elems.iter(), &mut entries);
        ok &= self.check_same_dim(&entries);
        if !ok {
            return Ty::Unknown;
        }
        let dim = entries.first().map_or(Dimension::DIMENSIONLESS, |e| e.dim);
        Ty::value(dim, TensorType::untagged(Shape::vector(elems.len() as u64)))
    }

    /// `[1, [2, 3]]`: some elements are rows and some are not.
    fn mixed_literal(&mut self, elems: &[Expr]) -> Ty {
        let is_row = |x: &Expr| matches!(x.kind, ExprKind::Array(_));
        let row_count = elems.iter().filter(|x| is_row(x)).count();
        // The majority kind is expected; on a tie, the first element's.
        let expect_rows = match (row_count * 2).cmp(&elems.len()) {
            std::cmp::Ordering::Equal => elems.first().is_some_and(is_row),
            other => other.is_gt(),
        };
        let flag_rows = !expect_rows;
        // Type everything once so names inside are still checked.
        let mut scratch = Vec::new();
        self.scalar_elems(elems.iter().filter(|x| !is_row(x)), &mut scratch);
        for row in elems.iter().filter(|x| is_row(x)) {
            if let ExprKind::Array(items) = &row.kind {
                items.iter().for_each(|x| {
                    self.expr(x);
                });
            }
            // Rejected with the literal: unknown, but recorded, so
            // `Checked::expr_types` stays complete.
            self.expr_types.insert(row.span, Ty::Unknown);
        }
        let flagged: Vec<&Expr> = elems.iter().filter(|x| is_row(x) == flag_rows).collect();
        let first = elems.first().map_or(Span::point(0), |x| x.span);
        for x in flagged.iter().take(MAX_REPORTED) {
            let (what, other) = if flag_rows {
                ("a row", "numbers")
            } else {
                ("a number", "rows")
            };
            self.error(
                Diagnostic::error(
                    format!("cannot mix numbers and rows in an array literal: this is {what}"),
                    x.span,
                )
                .with_label(format!("expected {other}, like the other elements"))
                .with_secondary(first, "the first element")
                .with_help(
                    "a vector is `[1, 2, 3]` and a matrix is written as rows, `[[1, 2], [3, 4]]`",
                ),
            );
        }
        Ty::Unknown
    }

    fn matrix_literal(&mut self, rows: &[Expr]) -> Ty {
        let t = self.matrix_literal_untabled(rows);
        // Rows are typed as part of the matrix, not through `expr`, so record
        // them here to keep `Checked::expr_types` complete: each row is a
        // vector of the matrix's element type, or unknown if the literal was
        // rejected.
        for row in rows {
            let rt = match (&t, &row.kind) {
                (Ty::Value(v), ExprKind::Array(items)) if v.ty.shape().rank() == 2 => Ty::value(
                    v.dim,
                    TensorType::untagged(Shape::vector(items.len() as u64)),
                ),
                _ => Ty::Unknown,
            };
            self.expr_types.insert(row.span, rt);
        }
        t
    }

    fn matrix_literal_untabled(&mut self, rows: &[Expr]) -> Ty {
        let mut entries = Vec::new();
        let mut lens = Vec::with_capacity(rows.len());
        let mut ok = true;
        for row in rows {
            let ExprKind::Array(items) = &row.kind else {
                continue;
            };
            lens.push(items.len());
            for item in items {
                if matches!(item.kind, ExprKind::Array(_)) {
                    // Typed (and so recorded, names inside checked), then
                    // rejected: v0.1 literals have at most two levels.
                    self.expr(item);
                    if ok {
                        self.error(
                            Diagnostic::error("array literals have at most two levels", item.span)
                                .with_label("a third level")
                                .with_note("v0.1 has vectors and matrices, and no literal for a rank-3 tensor"),
                        );
                    }
                    ok = false;
                } else {
                    ok &= self.scalar_elems(std::iter::once(item), &mut entries);
                }
            }
        }
        let want_len = majority(&lens);
        let uneven: Vec<usize> = (0..rows.len())
            .filter(|&i| Some(lens[i]) != want_len)
            .collect();
        let reference = (0..rows.len()).find(|&i| Some(lens[i]) == want_len);
        for (shown, &i) in uneven.iter().enumerate().take(MAX_REPORTED) {
            let mut d = Diagnostic::error(
                format!(
                    "matrix rows must have the same length: expected {}, found {}",
                    want_len.unwrap_or(0),
                    lens[i]
                ),
                rows[i].span,
            )
            .with_label(format!(
                "this row has {} element{}",
                lens[i],
                if lens[i] == 1 { "" } else { "s" }
            ));
            if let Some(r) = reference {
                d = d.with_secondary(
                    rows[r].span,
                    format!("this row has {}", want_len.unwrap_or(0)),
                );
            }
            if shown + 1 == MAX_REPORTED && uneven.len() > MAX_REPORTED {
                d = d.with_note(format!(
                    "{} more rows have the wrong length",
                    uneven.len() - MAX_REPORTED
                ));
            }
            self.error(d);
        }
        ok &= uneven.is_empty();
        ok &= self.check_same_dim(&entries);
        if !ok {
            return Ty::Unknown;
        }
        let cols = want_len.unwrap_or(0) as u64;
        let dim = entries.first().map_or(Dimension::DIMENSIONLESS, |e| e.dim);
        Ty::value(
            dim,
            TensorType::untagged(Shape::matrix(rows.len() as u64, cols)),
        )
    }

    /// Types each scalar element (in source order), pushing the ones with a
    /// dimension onto `out`. Reports an element that is not a scalar number
    /// and returns whether every element was fine. An element the checker
    /// cannot type is skipped silently: its own error, if any, was reported.
    fn scalar_elems<'e>(
        &mut self,
        elems: impl Iterator<Item = &'e Expr>,
        out: &mut Vec<Elem<'e>>,
    ) -> bool {
        let mut ok = true;
        for x in elems {
            match self.expr(x) {
                Ty::Value(v) if v.is_scalar() => out.push(Elem {
                    expr: x,
                    dim: v.dim,
                }),
                Ty::Value(v) => {
                    ok = false;
                    let label = self.this_is(&v);
                    self.error(
                        Diagnostic::error(
                            format!(
                                "an array element must be a number, found a {}",
                                crate::check::rank_noun(v.ty.shape().rank())
                            ),
                            x.span,
                        )
                        .with_label(label)
                        .with_help("write a matrix as rows: `[[1, 2], [3, 4]]`"),
                    );
                }
                Ty::Bool => {
                    ok = false;
                    self.error(
                        Diagnostic::error("an array element must be a number, found Bool", x.span)
                            .with_label("this is Bool")
                            .with_note("v0.1 has no truthiness: a Bool is never a number"),
                    );
                }
                Ty::Str => {
                    ok = false;
                    self.error(
                        Diagnostic::error(
                            "an array element must be a number, found a string",
                            x.span,
                        )
                        .with_label("this is a string")
                        .with_note("in v0.1 an array holds numbers only"),
                    );
                }
                Ty::Func(sig) => {
                    ok = false;
                    self.error(
                        Diagnostic::error(
                            format!(
                                "an array element must be a number, found function `{}`",
                                sky_diag::display_capped(sig.name())
                            ),
                            x.span,
                        )
                        .with_label("this is a function"),
                    );
                }
                Ty::Ode(o) => {
                    ok = false;
                    self.error(
                        Diagnostic::error(
                            format!("an array element must be a number, found {o}"),
                            x.span,
                        )
                        .with_label(format!("this is {o}")),
                    );
                }
                Ty::Unknown => ok = false,
            }
        }
        ok
    }

    /// Every element must have the dimension most of them have (the first
    /// one on a tie); the others are errors on their own span. Returns
    /// whether all agreed.
    fn check_same_dim(&mut self, entries: &[Elem<'_>]) -> bool {
        let dims: Vec<Dimension> = entries.iter().map(|e| e.dim).collect();
        let Some(want) = majority(&dims) else {
            return true;
        };
        let reference = entries.iter().find(|e| e.dim == want).map(|e| e.expr.span);
        let odd: Vec<&Elem<'_>> = entries.iter().filter(|e| e.dim != want).collect();
        for (shown, e) in odd.iter().enumerate().take(MAX_REPORTED) {
            let mut d = Diagnostic::error(
                format!(
                    "array elements must have the same dimension: expected {}, found {}",
                    describe(&want),
                    describe(&e.dim)
                ),
                e.expr.span,
            )
            .with_label(format!("this is {}", describe(&e.dim)));
            if let Some(r) = reference {
                d = d.with_secondary(r, format!("this is {}", describe(&want)));
            }
            if shown + 1 == MAX_REPORTED && odd.len() > MAX_REPORTED {
                d = d.with_note(format!(
                    "{} more elements have the wrong dimension",
                    odd.len() - MAX_REPORTED
                ));
            }
            self.error(d);
        }
        odd.is_empty()
    }

    /// `base[items]`: reading an element, a row or column, or a slice.
    ///
    /// An index is a dimensionless integer, and a constant one is checked
    /// against a known axis length here (a variable one at run time). Fewer
    /// items than axes leave the trailing axes whole (`A[0]` is row 0). A
    /// scalar index drops its axis and a slice keeps it. The result has no
    /// tags. Negative indices are not supported.
    pub(crate) fn index_expr(&mut self, base: &Expr, items: &[IndexItem]) -> Ty {
        let b = self.expr(base);
        // Type every item, whatever the base is.
        let mut parts = Vec::with_capacity(items.len());
        let mut ok = true;
        for item in items {
            let part = match item {
                IndexItem::Expr(x) => {
                    let (c, fine) = self.index_bound(x);
                    ok &= fine;
                    Part::At(x.span, c)
                }
                IndexItem::Slice {
                    start, stop, span, ..
                } => {
                    let mut bound = |c: &mut Self, x: &Option<Expr>| match x {
                        Some(x) => {
                            let (v, fine) = c.index_bound(x);
                            ok &= fine;
                            v
                        }
                        None => None,
                    };
                    let (lo, hi) = (bound(self, start), bound(self, stop));
                    Part::Slice(*span, lo, hi, start.is_some(), stop.is_some())
                }
            };
            parts.push(part);
        }
        let v = match b {
            Ty::Value(v) => v,
            Ty::Bool => {
                self.error(
                    Diagnostic::error("cannot index a Bool", base.span).with_label("this is Bool"),
                );
                return Ty::Unknown;
            }
            Ty::Str => {
                self.error(
                    Diagnostic::error("cannot index a string", base.span)
                        .with_label("this is a string")
                        .with_note("in v0.1 a string is only printed; it has no elements"),
                );
                return Ty::Unknown;
            }
            Ty::Func(sig) => {
                self.error(
                    Diagnostic::error(
                        format!(
                            "cannot index function `{}`",
                            sky_diag::display_capped(sig.name())
                        ),
                        base.span,
                    )
                    .with_label("this is a function"),
                );
                return Ty::Unknown;
            }
            Ty::Ode(o) => {
                self.error(
                    Diagnostic::error(format!("cannot index {o}"), base.span)
                        .with_label(format!("this is {o}")),
                );
                return Ty::Unknown;
            }
            Ty::Unknown => return Ty::Unknown,
        };
        if v.is_scalar() {
            let label = self.this_is(&v);
            self.error(
                Diagnostic::error("cannot index a scalar", base.span)
                    .with_label(label)
                    .with_note("only vectors and matrices have elements"),
            );
            return Ty::Unknown;
        }
        let shape = self.env.shapes.apply(v.ty.shape());
        let rank = shape.rank();
        if items.len() > rank {
            let label = self.this_is(&v);
            self.error(
                Diagnostic::error(
                    format!(
                        "too many indices: {} has {rank} axis{}, found {} indices",
                        shape,
                        if rank == 1 { "" } else { "es" },
                        items.len()
                    ),
                    items[rank].span(),
                )
                .with_label("this index has no axis")
                .with_secondary(base.span, label),
            );
            return Ty::Unknown;
        }
        let mut dims = Vec::new();
        for (axis, dim) in shape.dims().iter().enumerate() {
            let known = match dim {
                ShapeDim::Known(n) => Some(*n),
                ShapeDim::Var(_) => None,
            };
            match parts.get(axis) {
                None => dims.push(dim.clone()),
                Some(&Part::At(span, c)) => {
                    if let (Some(i), Some(n)) = (c, known) {
                        if i >= n {
                            self.error(
                                Diagnostic::error(
                                    format!(
                                        "index {i} is out of bounds for axis {axis} of length {n}"
                                    ),
                                    span,
                                )
                                .with_label(format!(
                                    "valid indices are 0 to {}",
                                    n.saturating_sub(1)
                                ))
                                .with_secondary(base.span, self.this_is(&v)),
                            );
                            ok = false;
                        }
                    }
                }
                Some(&Part::Slice(span, lo, hi, has_lo, has_hi)) => {
                    let len = self.slice_len(span, (lo, hi), (has_lo, has_hi), known, axis);
                    match len {
                        Ok(Some(n)) => dims.push(ShapeDim::Known(n)),
                        Ok(None) if !has_lo && !has_hi => dims.push(dim.clone()),
                        Ok(None) => dims.push(ShapeDim::Var(self.env.shapes.fresh("_"))),
                        Err(()) => ok = false,
                    }
                }
            }
        }
        if !ok {
            return Ty::Unknown;
        }
        if dims.is_empty() {
            return Ty::scalar(v.dim);
        }
        Ty::value(v.dim, TensorType::untagged(Shape::new(dims)))
    }

    /// The length of a slice, if it is known: `Ok(None)` when a bound or
    /// the axis is not constant. `Err` after reporting an error.
    fn slice_len(
        &mut self,
        span: Span,
        (lo, hi): (Option<u64>, Option<u64>),
        (has_lo, has_hi): (bool, bool),
        axis_len: Option<u64>,
        axis: usize,
    ) -> Result<Option<u64>, ()> {
        let lo_v = if has_lo { lo } else { Some(0) };
        let hi_v = if has_hi { hi } else { axis_len };
        if let (Some(h), Some(n)) = (hi_v, axis_len) {
            if h > n {
                self.error(
                    Diagnostic::error(
                        format!("slice end {h} is out of bounds for axis {axis} of length {n}"),
                        span,
                    )
                    .with_label(format!("the axis has {n} elements")),
                );
                return Err(());
            }
        }
        if let (Some(l), Some(n)) = (lo_v, axis_len) {
            if l > n {
                self.error(
                    Diagnostic::error(
                        format!("slice start {l} is out of bounds for axis {axis} of length {n}"),
                        span,
                    )
                    .with_label(format!("the axis has {n} elements")),
                );
                return Err(());
            }
        }
        match (lo_v, hi_v) {
            (Some(l), Some(h)) if l >= h => {
                self.error(
                    Diagnostic::error("this slice is empty", span)
                        .with_label(format!("{l} is not less than {h}"))
                        .with_note("arrays are never empty in v0.1"),
                );
                Err(())
            }
            (Some(l), Some(h)) => Ok(Some(h - l)),
            _ => Ok(None),
        }
    }

    /// Types one index expression: it must be a dimensionless scalar, and a
    /// constant one an integer that is not negative. Returns the constant
    /// (if any) and whether the expression was fine.
    fn index_bound(&mut self, x: &Expr) -> (Option<u64>, bool) {
        let t = self.expr(x);
        match &t {
            Ty::Value(v) if !v.is_scalar() => {
                let label = self.this_is(v);
                self.error(
                    Diagnostic::error("an index must be a number, not an array", x.span)
                        .with_label(label),
                );
                return (None, false);
            }
            Ty::Value(v) if !v.dim.is_dimensionless() => {
                self.error(
                    Diagnostic::error("an index must be dimensionless", x.span)
                        .with_label(format!("this is {}", describe(&v.dim))),
                );
                return (None, false);
            }
            Ty::Bool | Ty::Func(_) | Ty::Ode(_) => {
                self.error(
                    Diagnostic::error(format!("an index must be an integer, found {t}"), x.span)
                        .with_label(format!("this is {t}")),
                );
                return (None, false);
            }
            _ => {}
        }
        let Some(c) = const_exponent(self.src, x) else {
            return (None, true);
        };
        if !c.is_integer() {
            self.error(
                Diagnostic::error(
                    format!(
                        "an index must be an integer, found `{}`",
                        self.quoted(x.span)
                    ),
                    x.span,
                )
                .with_label("not an integer"),
            );
            return (None, false);
        }
        match u64::try_from(c.numer()) {
            Ok(i) => (Some(i), true),
            Err(_) => {
                self.error(
                    Diagnostic::error("an index cannot be negative", x.span)
                        .with_label("negative")
                        .with_note("v0.1 counts from 0 and has no negative indices"),
                );
                (None, false)
            }
        }
    }

    /// The builtins that make or measure arrays: `zeros(n)`, `zeros(m, n)`,
    /// `identity(n)`, `len(x)`. `None` if `callee` is not one of them, or the
    /// program has bound the name itself (as with `print` and `inv`).
    pub(crate) fn builtin_call(&mut self, callee: &Expr, args: &[Expr], call: Span) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        let name = self.ident(id);
        if !matches!(name, "zeros" | "identity" | "len") || self.env.get(name).is_some() {
            return None;
        }
        let want: &[usize] = if name == "zeros" { &[1, 2] } else { &[1] };
        if !want.contains(&args.len()) {
            for a in args {
                self.expr(a);
            }
            let takes = if name == "zeros" {
                "1 or 2 arguments"
            } else {
                "1 argument"
            };
            self.error(
                Diagnostic::error(
                    format!("`{name}` takes {takes}, found {}", args.len()),
                    call,
                )
                .with_label("wrong number of arguments"),
            );
            return Some(Ty::Unknown);
        }
        if name == "len" {
            return Some(self.len_call(&args[0]));
        }
        let mut sizes = Vec::with_capacity(args.len());
        for a in args {
            sizes.push(self.size_arg(name, a));
        }
        let Some(sizes) = sizes.into_iter().collect::<Option<Vec<u64>>>() else {
            return Some(Ty::Unknown);
        };
        let total = if name == "identity" {
            sizes[0].checked_mul(sizes[0])
        } else {
            sizes.iter().try_fold(1u64, |acc, &n| acc.checked_mul(n))
        };
        if total.is_none_or(|t| t > MAX_ARRAY_ELEMENTS) {
            self.error(
                Diagnostic::error(format!("the array from `{name}` is too large"), call)
                    .with_label(format!("more than {MAX_ARRAY_ELEMENTS} elements")),
            );
            return Some(Ty::Unknown);
        }
        let ty = match (name, sizes.as_slice()) {
            ("zeros", &[n]) => TensorType::untagged(Shape::vector(n)),
            // An all-zero square matrix is Diagonal (so Symmetric).
            ("zeros", &[m, n]) => self.zeros_matrix(m, n),
            // The identity is all four: `QᵀQ = I`, and `x⁻¹ = x` etc.
            (_, &[n]) => {
                let all = TagSet::from_tags(Tag::ALL);
                TensorType::tagged(&mut self.env.shapes, Shape::matrix(n, n), all)
                    .unwrap_or_else(|_| TensorType::untagged(Shape::matrix(n, n)))
            }
            _ => return Some(Ty::Unknown),
        };
        Some(Ty::value(Dimension::DIMENSIONLESS, ty))
    }

    fn zeros_matrix(&mut self, m: u64, n: u64) -> TensorType {
        let shape = Shape::matrix(m, n);
        if m == n {
            let tags = TagSet::of(Tag::Diagonal);
            if let Ok(t) = TensorType::tagged(&mut self.env.shapes, shape.clone(), tags) {
                return t;
            }
        }
        TensorType::untagged(shape)
    }

    /// `len(x)`: the length of the first axis, as a dimensionless number.
    fn len_call(&mut self, arg: &Expr) -> Ty {
        let t = self.expr(arg);
        if let Ty::Value(v) = &t {
            if v.is_scalar() {
                self.error(
                    Diagnostic::error("`len` needs a vector or a matrix, found a scalar", arg.span)
                        .with_label(self.this_is(v)),
                );
                return Ty::Unknown;
            }
        } else if matches!(t, Ty::Bool | Ty::Str) {
            let found = if t == Ty::Bool { "Bool" } else { "a string" };
            self.error(
                Diagnostic::error(
                    format!("`len` needs a vector or a matrix, found {found}"),
                    arg.span,
                )
                .with_label(format!("this is {found}")),
            );
            return Ty::Unknown;
        }
        Ty::scalar(Dimension::DIMENSIONLESS)
    }

    /// A size argument: a constant positive integer. Array sizes are part
    /// of the type, so they cannot depend on a run-time value.
    fn size_arg(&mut self, name: &str, a: &Expr) -> Option<u64> {
        let t = self.expr(a);
        if let Ty::Value(v) = &t {
            if !v.is_scalar() || !v.dim.is_dimensionless() {
                self.error(
                    Diagnostic::error(
                        format!("`{name}` needs a size, a dimensionless integer"),
                        a.span,
                    )
                    .with_label(self.this_is(v)),
                );
                return None;
            }
        } else if t != Ty::Unknown {
            self.error(
                Diagnostic::error(
                    format!("`{name}` needs a size, a dimensionless integer"),
                    a.span,
                )
                .with_label(format!("this is {t}")),
            );
            return None;
        }
        let quoted = self.quoted(a.span);
        match const_exponent(self.src, a) {
            Some(c) if c.is_integer() && c.numer() >= 1 => u64::try_from(c.numer()).ok(),
            Some(c) if c.is_integer() => {
                self.error(
                    Diagnostic::error(
                        format!("`{name}` needs a positive size, found `{quoted}`"),
                        a.span,
                    )
                    .with_label("not positive")
                    .with_note("arrays are never empty in v0.1"),
                );
                None
            }
            Some(_) => {
                self.error(
                    Diagnostic::error(
                        format!("`{name}` needs an integer size, found `{quoted}`"),
                        a.span,
                    )
                    .with_label("not an integer"),
                );
                None
            }
            None => {
                if t != Ty::Unknown {
                    self.error(
                        Diagnostic::error(
                            format!("`{name}` needs a constant size, found `{quoted}`"),
                            a.span,
                        )
                        .with_label("not a constant")
                        .with_note("an array's size is part of its type, so it is known before the program runs")
                        .with_help(format!("write the number, e.g. `{name}(3)`")),
                    );
                }
                None
            }
        }
    }
}

/// An element of a literal with its dimension.
pub(crate) struct Elem<'e> {
    expr: &'e Expr,
    dim: Dimension,
}

/// One item of an index, after typing: a constant index (if it is one), or
/// a slice with its constant bounds.
#[derive(Clone, Copy)]
enum Part {
    At(Span, Option<u64>),
    /// Span, start, stop, and whether each bound was written.
    Slice(Span, Option<u64>, Option<u64>, bool, bool),
}

/// The value that occurs most often in `items` (the earliest on a tie), in
/// one pass: a literal of a million entries is checked in linear time.
///
/// Almost every literal is uniform, so the scan first compares each entry
/// with the first and stops at the first that differs; only a literal with
/// mixed entries pays for hashing.
fn majority<T: Eq + std::hash::Hash + Copy>(items: &[T]) -> Option<T> {
    let first = *items.first()?;
    let Some(split) = items.iter().position(|&x| x != first) else {
        return Some(first);
    };
    // Counts in order of first appearance, so a tie goes to the value that
    // appears first.
    let mut index: std::collections::HashMap<T, usize> = std::collections::HashMap::new();
    let mut counts: Vec<(T, usize)> = Vec::new();
    counts.push((first, split));
    index.insert(first, 0);
    for &x in &items[split..] {
        let at = *index.entry(x).or_insert_with(|| {
            counts.push((x, 0));
            counts.len() - 1
        });
        counts[at].1 += 1;
    }
    let mut best = counts[0];
    for &(x, n) in &counts[1..] {
        if n > best.1 {
            best = (x, n);
        }
    }
    Some(best.0)
}

/// What the checker can prove about a matrix literal from its entries.
///
/// Built from the expression a binding's value is (parentheses ignored). Only
/// a square matrix literal has entries to prove anything from, and only if
/// every entry is a numeric literal ([`entry_value`]). Everything else is
/// unproven, and the caller falls back to its ordinary "not known to be"
/// error.
pub(crate) struct LiteralProof {
    n: usize,
    /// Row-major: each entry's span and constant value, if it has one.
    entries: Vec<(Span, Option<f64>)>,
}

/// Why a tag could not be proven for a literal: an explanation and the
/// entries to point at.
pub(crate) struct ProofFailure {
    pub(crate) note: String,
    pub(crate) entries: Vec<(Span, String)>,
}

impl LiteralProof {
    /// The proof state for `value`; empty (proving nothing) unless `value`
    /// is a square matrix literal.
    pub(crate) fn of(src: &str, value: &Expr) -> Option<LiteralProof> {
        let ExprKind::Array(rows) = &strip_parens(value).kind else {
            return None;
        };
        let n = rows.len();
        let mut entries = Vec::with_capacity(n * n);
        for row in rows {
            let ExprKind::Array(items) = &row.kind else {
                return None;
            };
            if items.len() != n {
                return None;
            }
            for item in items {
                if matches!(item.kind, ExprKind::Array(_)) {
                    return None;
                }
                entries.push((item.span, entry_value(src, item)));
            }
        }
        Some(LiteralProof { n, entries })
    }

    fn values(&self) -> Option<Vec<f64>> {
        self.entries.iter().map(|&(_, v)| v).collect()
    }

    fn at(&self, i: usize, j: usize) -> (Span, Option<f64>) {
        self.entries[i * self.n + j]
    }

    /// The first pair `i < j` with `aᵢⱼ != aⱼᵢ`.
    fn asymmetry(&self, a: &[f64]) -> Option<(usize, usize)> {
        let n = self.n;
        (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .find(|&(i, j)| a[i * n + j] != a[j * n + i])
    }

    /// Whether `tag` holds for the literal, proven from the constants.
    pub(crate) fn proves(&self, tag: Tag) -> bool {
        let Some(a) = self.values() else {
            return false;
        };
        let n = self.n;
        match tag {
            Tag::Symmetric => self.asymmetry(&a).is_none(),
            Tag::Diagonal => (0..n).all(|i| (0..n).all(|j| i == j || a[i * n + j] == 0.0)),
            Tag::PositiveDefinite => {
                n <= MAX_PROVEN_SIZE
                    && self.asymmetry(&a).is_none()
                    && cholesky_pivot_failure(&a, n).is_none()
            }
            Tag::Orthogonal => false,
        }
    }

    /// Why `tag` is not proven, with the entries responsible.
    pub(crate) fn failure(&self, tag: Tag) -> ProofFailure {
        let n = self.n;
        let Some(a) = self.values() else {
            let bad = self.entries.iter().find(|(_, v)| v.is_none());
            return ProofFailure {
                note: format!(
                    "the checker proves {} of a literal only when every entry is a numeric literal",
                    tag.name()
                ),
                entries: bad
                    .map(|&(s, _)| (s, "this entry is not a numeric literal".to_string()))
                    .into_iter()
                    .collect(),
            };
        };
        let show = |x: f64| sky_diag::display_capped(&format!("{x}")).into_owned();
        let asymmetric = |proof: &LiteralProof| {
            proof.asymmetry(&a).map(|(i, j)| ProofFailure {
                note: format!(
                    "entry [{i}, {j}] is {} but entry [{j}, {i}] is {}",
                    show(a[i * n + j]),
                    show(a[j * n + i])
                ),
                entries: vec![
                    (proof.at(i, j).0, format!("entry [{i}, {j}]")),
                    (proof.at(j, i).0, format!("entry [{j}, {i}]")),
                ],
            })
        };
        match tag {
            Tag::Symmetric => asymmetric(self),
            Tag::Diagonal => (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .find(|&(i, j)| i != j && a[i * n + j] != 0.0)
                .map(|(i, j)| ProofFailure {
                    note: format!("entry [{i}, {j}] is {}, not zero", show(a[i * n + j])),
                    entries: vec![(self.at(i, j).0, format!("entry [{i}, {j}]"))],
                }),
            Tag::PositiveDefinite => asymmetric(self).or_else(|| {
                if n > MAX_PROVEN_SIZE {
                    return Some(ProofFailure {
                        note: format!(
                            "the checker proves PositiveDefinite only for literals up to {MAX_PROVEN_SIZE} by {MAX_PROVEN_SIZE}"
                        ),
                        entries: Vec::new(),
                    });
                }
                cholesky_pivot_failure(&a, n).map(|(k, pivot)| ProofFailure {
                    note: if pivot <= 0.0 {
                        format!(
                            "the leading {}x{} block is not positive definite (its Cholesky pivot is {})",
                            k + 1,
                            k + 1,
                            show(pivot)
                        )
                    } else {
                        format!(
                            "the Cholesky pivot at row {k} is {}, too close to zero to prove: the checker needs every pivot above {PIVOT_MARGIN:e} of the largest diagonal entry",
                            show(pivot)
                        )
                    },
                    entries: vec![(self.at(k, k).0, format!("diagonal entry {k}"))],
                })
            }),
            Tag::Orthogonal => None,
        }
        .unwrap_or_else(|| ProofFailure {
            note: match tag {
                Tag::Orthogonal => "the checker never proves Orthogonal from a literal: exact orthogonality needs exact sums of products, which decimal entries do not give".to_string(),
                _ => format!("{} could not be proven", tag.name()),
            },
            entries: Vec::new(),
        })
    }
}

/// Cholesky `A = L·Lᵀ` of a symmetric matrix (row-major, `n` by `n`),
/// looking only for a pivot that is not comfortably positive. Returns the
/// row and value of the first such pivot, or `None` if every pivot exceeds
/// [`PIVOT_MARGIN`] times the largest diagonal entry.
fn cholesky_pivot_failure(a: &[f64], n: usize) -> Option<(usize, f64)> {
    let scale = (0..n).map(|i| a[i * n + i]).fold(0.0_f64, f64::max);
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 0.0 || d <= PIVOT_MARGIN * scale || !d.is_finite() {
            return Some((j, d));
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    None
}

#[cfg(test)]
mod majority_tests {
    use super::majority;

    #[test]
    fn majority_picks_the_most_common_and_the_earliest_on_a_tie() {
        assert_eq!(majority::<u8>(&[]), None);
        assert_eq!(majority(&[4, 4, 4]), Some(4));
        assert_eq!(majority(&[1, 2, 2]), Some(2));
        assert_eq!(majority(&[1, 2, 1, 2]), Some(1));
        assert_eq!(majority(&[2, 1, 1, 2]), Some(2));
        assert_eq!(majority(&[3, 1, 2]), Some(3));
    }

    #[test]
    fn majority_is_linear_on_a_large_mixed_input() {
        let items: Vec<usize> = (0..200_000).map(|i| i % 1000).collect();
        assert_eq!(majority(&items), Some(0));
    }
}
