//! The verifier's rules for the array operations (`array.literal`,
//! `index`, `slice`, `zeros`, `identity`, `len`); the operations are
//! specified in [`crate::op`].
//!
//! Like the rest of the verifier these rules skip what an `Unknown` type
//! hides: an operand of unknown type passes, and an unknown result is
//! accepted.

use sky_types::{Shape, ShapeDim, MAX_ARRAY_ELEMENTS};
use sky_units::Dimension;

use super::{FnVerifier, Slot};
use crate::ir::{OpData, OpId, Type};
use crate::op::{OpKind, SliceAxis};

impl FnVerifier<'_> {
    /// `array.literal`: scalars of one dimension, arranged as `cols` says
    /// (which the result's type restates).
    pub(super) fn check_array_literal(&mut self, o: OpId, op: &OpData, cols: Option<u64>) {
        let n = op.operands.len();
        if n == 0 {
            self.err_op(o, "`array.literal` needs at least one operand".into());
            return;
        }
        let expected = match cols {
            None => Shape::vector(n as u64),
            Some(c) if c > 0 && (n as u64) % c == 0 => Shape::matrix(n as u64 / c, c),
            Some(c) => {
                self.err_op(
                    o,
                    format!("`array.literal` cannot arrange {n} operand(s) in rows of {c}"),
                );
                return;
            }
        };
        let mut dim: Option<Dimension> = None;
        for i in 0..n {
            match self.operand_num(o, op, i) {
                Some(Slot::Num(a)) if a.shape.rank() == 0 => match dim {
                    None => dim = Some(a.dim),
                    Some(d) if d != a.dim => {
                        self.err_op(
                            o,
                            format!(
                                "entries of `array.literal` have different dimensions: {d} and {}",
                                a.dim
                            ),
                        );
                        return;
                    }
                    Some(_) => {}
                },
                Some(Slot::Num(a)) => {
                    let shape = &a.shape;
                    self.err_op(
                        o,
                        format!(
                            "operand {i} of `array.literal` must be a scalar, found shape {shape}"
                        ),
                    );
                    return;
                }
                Some(Slot::Unknown) => {}
                // Not a number: reported.
                None => return,
            }
        }
        match self.f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(r) => {
                if let Some(d) = dim.filter(|d| *d != r.dim) {
                    self.err_op(
                        o,
                        format!(
                            "the result of `array.literal` should have dimension {d}, found {}",
                            r.dim
                        ),
                    );
                }
                if r.shape != expected {
                    self.err_op(
                        o,
                        format!(
                            "`array.literal` of {n} operand(s) arranged {} has shape {expected}, found {}",
                            match cols {
                                None => "as a vector".to_string(),
                                Some(c) => format!("in rows of {c}"),
                            },
                            r.shape
                        ),
                    );
                }
            }
            other => self.err_op(
                o,
                format!("the result of `array.literal` must be a number, found {other}"),
            ),
        }
    }

    /// `const.array`: finite entries, arranged as `cols` says, in a number
    /// whose shape restates the arrangement.
    pub(super) fn check_const_array(
        &mut self,
        o: OpId,
        op: &OpData,
        cols: Option<u64>,
        bits: &[u64],
    ) {
        let n = bits.len();
        if n == 0 {
            self.err_op(o, "`const.array` needs at least one entry".into());
            return;
        }
        let expected = match cols {
            None => Shape::vector(n as u64),
            Some(c) if c > 0 && (n as u64) % c == 0 => Shape::matrix(n as u64 / c, c),
            Some(c) => {
                self.err_op(
                    o,
                    format!("`const.array` cannot arrange {n} entries in rows of {c}"),
                );
                return;
            }
        };
        if let Some(i) = bits.iter().position(|&b| !f64::from_bits(b).is_finite()) {
            self.err_op(
                o,
                format!(
                    "`const.array` entries must be finite, entry {i} has bits {:#018x}",
                    bits[i]
                ),
            );
        }
        match self.f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(r) => {
                if r.shape != expected {
                    self.err_op(
                        o,
                        format!(
                            "`const.array` of {n} entries has shape {expected}, found {}",
                            r.shape
                        ),
                    );
                }
            }
            other => self.err_op(
                o,
                format!("the result of `const.array` must be a number, found {other}"),
            ),
        }
    }

    /// `index`: a base and one position per leading axis.
    pub(super) fn check_index(&mut self, o: OpId, op: &OpData) {
        let k = op.operands.len().saturating_sub(1);
        if k == 0 {
            self.err_op(o, "`index` needs an array and at least one position".into());
            return;
        }
        self.check_subscript(o, op, &vec![SliceAxis::At; k]);
    }

    /// `slice`: a base and the operands of each item.
    pub(super) fn check_slice(&mut self, o: OpId, op: &OpData, axes: &[SliceAxis]) {
        let want = 1 + axes.iter().map(|a| a.operands()).sum::<usize>();
        if axes.is_empty() {
            self.err_op(o, "`slice` needs at least one item".into());
        } else if op.operands.len() != want {
            self.err_op(
                o,
                format!(
                    "`slice` with {} item(s) takes {want} operand(s), found {}",
                    axes.len(),
                    op.operands.len()
                ),
            );
        } else {
            self.check_subscript(o, op, axes);
        }
    }

    /// The shared rules of `index` and `slice`: `axes` has one item per
    /// subscripted axis, and the operands are the base, then each item's.
    fn check_subscript(&mut self, o: OpId, op: &OpData, axes: &[SliceAxis]) {
        let name = op.kind.mnemonic();
        for i in 1..op.operands.len() {
            match self.operand_num(o, op, i) {
                Some(Slot::Num(p)) if p.shape.rank() != 0 || !p.dim.is_dimensionless() => {
                    self.err_op(
                        o,
                        format!(
                            "operand {i} of `{name}` must be a dimensionless scalar, found {} with dimension {}",
                            p.shape, p.dim
                        ),
                    );
                    return;
                }
                Some(_) => {}
                None => return,
            }
        }
        let a = match self.operand_num(o, op, 0) {
            Some(Slot::Num(a)) => a,
            _ => return self.expect_result_loose(o, op),
        };
        let dims = a.shape.dims();
        if dims.is_empty() {
            self.err_op(
                o,
                format!("`{name}` needs a vector or matrix, found a scalar"),
            );
            return;
        }
        if axes.len() > dims.len() {
            self.err_op(
                o,
                format!(
                    "too many indices: shape {} has {} axis(es), `{name}` subscripts {}",
                    a.shape,
                    dims.len(),
                    axes.len()
                ),
            );
            return;
        }
        // The dimension of each axis the result keeps (`None`: a range with
        // a bound, whose length the operands decide).
        let mut want: Vec<Option<&ShapeDim>> = Vec::new();
        for (axis, d) in dims.iter().enumerate() {
            match axes.get(axis) {
                None
                | Some(SliceAxis::Range {
                    start: false,
                    stop: false,
                }) => want.push(Some(d)),
                Some(SliceAxis::At) => {}
                Some(SliceAxis::Range { .. }) => want.push(None),
            }
        }
        match self.f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(r) => {
                if r.dim != a.dim {
                    self.err_op(
                        o,
                        format!(
                            "the result of `{name}` should have dimension {}, found {}",
                            a.dim, r.dim
                        ),
                    );
                }
                let got = r.shape.dims();
                let same = got.len() == want.len()
                    && got.iter().zip(&want).all(|(g, w)| w.is_none_or(|w| g == w));
                if !same {
                    let shown: Vec<String> = want
                        .iter()
                        .map(|w| w.map_or("_".to_string(), ToString::to_string))
                        .collect();
                    self.err_op(
                        o,
                        format!(
                            "the result of `{name}` should have shape ({}), found {}",
                            shown.join(", "),
                            r.shape
                        ),
                    );
                }
            }
            other => self.err_op(
                o,
                format!("the result of `{name}` must be a number, found {other}"),
            ),
        }
    }

    /// `zeros` (one or two sizes) and `identity` (one).
    pub(super) fn check_zeros_or_identity(&mut self, o: OpId, op: &OpData) {
        let name = op.kind.mnemonic();
        let identity = matches!(op.kind, OpKind::Identity);
        let n = op.operands.len();
        if !(1..=if identity { 1 } else { 2 }).contains(&n) {
            self.err_op(
                o,
                format!(
                    "`{name}` takes {} size operand(s), found {n}",
                    if identity { "1" } else { "1 or 2" }
                ),
            );
            return;
        }
        for i in 0..n {
            match self.operand_num(o, op, i) {
                Some(Slot::Num(s)) if s.shape.rank() != 0 || !s.dim.is_dimensionless() => {
                    self.err_op(
                        o,
                        format!(
                            "size operand {i} of `{name}` must be a dimensionless scalar, found {} with dimension {}",
                            s.shape, s.dim
                        ),
                    );
                    return;
                }
                Some(_) => {}
                None => return,
            }
        }
        match self.f.ty(op.results[0]) {
            Type::Unknown => {}
            Type::Number(r) => {
                if !r.dim.is_dimensionless() {
                    self.err_op(
                        o,
                        format!(
                            "the result of `{name}` should be dimensionless, found dimension {}",
                            r.dim
                        ),
                    );
                }
                let dims = r.shape.dims();
                let rank = if identity { 2 } else { n };
                if dims.len() != rank || (identity && dims[0] != dims[1]) {
                    let want = match (identity, n) {
                        (true, _) => "a square matrix",
                        (false, 1) => "a vector",
                        (false, _) => "a matrix",
                    };
                    self.err_op(
                        o,
                        format!(
                            "the result of `{name}` should be {want}, found shape {}",
                            r.shape
                        ),
                    );
                    return;
                }
                let total = dims.iter().try_fold(1u64, |acc, d| match d {
                    ShapeDim::Known(k) => acc.checked_mul(*k),
                    ShapeDim::Var(_) => Some(acc),
                });
                if total.is_none_or(|t| t > MAX_ARRAY_ELEMENTS) {
                    self.err_op(
                        o,
                        format!(
                            "the array from `{name}` is too large: more than {MAX_ARRAY_ELEMENTS} elements"
                        ),
                    );
                }
            }
            other => self.err_op(
                o,
                format!("the result of `{name}` must be a number, found {other}"),
            ),
        }
    }

    /// `len`: a vector or matrix in, a dimensionless scalar out.
    pub(super) fn check_len(&mut self, o: OpId, op: &OpData) {
        if let Some(Slot::Num(a)) = self.operand_num(o, op, 0) {
            if a.shape.rank() == 0 {
                self.err_op(o, "`len` needs a vector or a matrix, found a scalar".into());
            }
        }
        self.expect_result(o, op, Some(Dimension::DIMENSIONLESS), Some(Shape::scalar()));
    }
}
