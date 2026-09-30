//! Building blocks for array tangents and adjoints: zeros and ones of a
//! given type, elements, rows and columns, outer products, sums.
//!
//! SIR has few array operations (`array.literal`, `index`, `slice`,
//! `zeros`, `identity`, `len`, `matmul`, `transpose`, and `fused` for
//! elementwise work), so the operations differentiation needs that are not
//! among them (a sum of all entries, an outer product, a one-hot vector)
//! are built from those:
//!
//! * `ones(n)` is a `fused` kernel adding 1 to `zeros(n)`, so it works for a
//!   length known only at run time;
//! * the sum of a vector `v` is `v @ ones(n)`, of a matrix `ones(m) @ A @
//!   ones(n)`;
//! * row `i` of `identity(n)` is the one-hot vector `eᵢ`, and
//!   `identity(n)[:, lo:hi]` selects a range of entries as a matrix, so
//!   scattering a slice's adjoint back is a `matmul`;
//! * an outer product `u vᵀ` is `(n×1) @ (1×m)`, and a vector has no
//!   reshape, so the two factors are gathered by `array.literal` from their
//!   entries. That needs the sizes to be known (`Vector[3]`, not `Vector[N]`);
//!   without them the derivative is not supported.
//!
//! Every function here takes and returns values of the function being
//! built (not of the source function).

use sky_diag::Span;
use sky_ir::{NumberType, OpKind, Repr, Type, ValueId};
use sky_types::{Shape, ShapeDim, TagSet};
use sky_units::Dimension;

use crate::emit::{number_type, shaped_type, Diff, Res};

/// The most entries an outer product built from `array.literal` may have:
/// past it the operation count is a burden and reverse mode is the wrong
/// tool (or the shape should be a loop).
const MAX_STATIC_ENTRIES: u64 = 4096;

impl Diff<'_> {
    /// The type of a value of the function being built, as a number.
    pub(crate) fn new_num(&self, v: ValueId) -> Res<NumberType> {
        match self.b.func().ty(v) {
            Type::Number(n) => Ok(n.clone()),
            other => {
                Err(self.malformed(&format!("{v} of the derivative is {other}, not a number")))
            }
        }
    }

    /// A dimensionless scalar constant.
    pub(crate) fn konst_dimless(&mut self, x: f64, repr: Repr, span: Span) -> ValueId {
        self.konst(x, number_type(Dimension::DIMENSIONLESS, repr), span)
    }

    /// The size of axis `axis` of `shape` as a dimensionless scalar: a
    /// constant when it is known, otherwise the run-time length of `like`
    /// (a value of that shape).
    pub(crate) fn size_value(
        &mut self,
        shape: &Shape,
        axis: usize,
        like: Option<ValueId>,
        repr: Repr,
        span: Span,
    ) -> Res<ValueId> {
        let dims = shape.dims();
        match dims.get(axis) {
            Some(ShapeDim::Known(n)) => Ok(self.konst_dimless(*n as f64, repr, span)),
            Some(ShapeDim::Var(v)) => {
                let Some(like) = like else {
                    return Err(
                        self.unsupported("array", &format!("the size {v} is not known here"))
                    );
                };
                let mut source = like;
                if axis > 0 {
                    // The length of a later axis is that of a row.
                    let n = self.new_num(like)?;
                    let zero = self.konst_dimless(0.0, repr, span);
                    let row = NumberType {
                        shape: Shape::new(dims[1..].to_vec()),
                        ..n
                    };
                    source = self.emit(OpKind::Index, vec![like, zero], Type::Number(row), span);
                }
                Ok(self.emit(
                    OpKind::Len,
                    vec![source],
                    number_type(Dimension::DIMENSIONLESS, repr),
                    span,
                ))
            }
            None => Err(self.malformed("an axis out of range")),
        }
    }

    /// The size operands of `zeros`/`identity` for `shape`.
    fn size_values(
        &mut self,
        shape: &Shape,
        like: Option<ValueId>,
        repr: Repr,
        span: Span,
    ) -> Res<Vec<ValueId>> {
        (0..shape.rank())
            .map(|axis| self.size_value(shape, axis, like, repr, span))
            .collect()
    }

    /// A zero of type `ty`: a scalar constant, or a zero array (`zeros`
    /// scaled by a zero constant when the dimension is not 1). `like` is a
    /// value of the same shape, for sizes only known at run time.
    pub(crate) fn zero_of(&mut self, ty: &Type, like: Option<ValueId>, span: Span) -> Res<ValueId> {
        let Type::Number(n) = ty else {
            return Err(self.malformed("a zero of a type that is not a number"));
        };
        if n.shape.rank() == 0 {
            return Ok(self.konst(0.0, ty.clone(), span));
        }
        let sizes = self.size_values(&n.shape, like, n.repr, span)?;
        let z = self.emit(
            OpKind::Zeros,
            sizes,
            shaped_type(Dimension::DIMENSIONLESS, n.shape.clone(), n.repr),
            span,
        );
        if n.dim.is_dimensionless() {
            return Ok(z);
        }
        let zero = self.konst(0.0, number_type(n.dim, n.repr), span);
        Ok(self.emit(OpKind::Mul, vec![z, zero], ty.clone(), span))
    }

    /// The dimensionless array of ones with the shape of `like`.
    pub(crate) fn ones_like(&mut self, like: ValueId, span: Span) -> Res<ValueId> {
        let n = self.new_num(like)?;
        let ty = shaped_type(Dimension::DIMENSIONLESS, n.shape.clone(), n.repr);
        let z = self.zero_of(&ty, Some(like), span)?;
        self.ones_from_zeros(z, span)
    }

    /// `z + 1` elementwise, as a `fused` kernel over the dimensionless
    /// array `z`.
    fn ones_from_zeros(&mut self, z: ValueId, span: Span) -> Res<ValueId> {
        let n = self.new_num(z)?;
        let scalar = number_type(Dimension::DIMENSIONLESS, n.repr);
        let outer = self.b.insertion_block();
        let (op, res) = self
            .b
            .op(OpKind::Fused, vec![z], vec![Type::Number(n)], span);
        let region = self.b.add_region(op);
        let blk = self.b.add_block(region, vec![scalar.clone()]);
        let x = self.b.func().block(blk).args[0];
        self.b.set_insertion_block(blk);
        let one = self.konst(1.0, scalar.clone(), span);
        let y = self.emit(OpKind::Add, vec![x, one], scalar, span);
        self.b.op(OpKind::Yield, vec![y], vec![], span);
        self.b.set_insertion_block(outer);
        Ok(res[0])
    }

    /// The entries of an array multiplied pairwise: a `fused` kernel
    /// `u * w` over two arrays of one shape.
    pub(crate) fn hadamard(&mut self, u: ValueId, w: ValueId, span: Span) -> Res<ValueId> {
        let (nu, nw) = (self.new_num(u)?, self.new_num(w)?);
        let dim = self.dim_mul(nu.dim, nw.dim)?;
        let out = shaped_type(dim, nu.shape.clone(), nu.repr);
        let outer = self.b.insertion_block();
        let (op, res) = self.b.op(OpKind::Fused, vec![u, w], vec![out], span);
        let region = self.b.add_region(op);
        let blk = self.b.add_block(
            region,
            vec![number_type(nu.dim, nu.repr), number_type(nw.dim, nw.repr)],
        );
        let args = self.b.func().block(blk).args.clone();
        self.b.set_insertion_block(blk);
        let y = self.emit(
            OpKind::Mul,
            vec![args[0], args[1]],
            number_type(dim, nu.repr),
            span,
        );
        self.b.op(OpKind::Yield, vec![y], vec![], span);
        self.b.set_insertion_block(outer);
        Ok(res[0])
    }

    /// The sum of all the entries of `v`, a scalar of `v`'s dimension.
    pub(crate) fn sum_all(&mut self, v: ValueId, span: Span) -> Res<ValueId> {
        let n = self.new_num(v)?;
        let scalar = number_type(n.dim, n.repr);
        match n.shape.rank() {
            0 => Ok(v),
            1 => {
                let ones = self.ones_like(v, span)?;
                Ok(self.emit(OpKind::MatMul, vec![v, ones], scalar, span))
            }
            _ => {
                // ones(rows) @ A @ ones(cols)
                let dims = n.shape.dims().to_vec();
                let row_ty = shaped_type(
                    Dimension::DIMENSIONLESS,
                    Shape::vector(dims[0].clone()),
                    n.repr,
                );
                let col_ty = shaped_type(
                    Dimension::DIMENSIONLESS,
                    Shape::vector(dims[1].clone()),
                    n.repr,
                );
                let rows = self.zero_of(&row_ty, Some(v), span)?;
                let rows = self.ones_from_zeros(rows, span)?;
                let cols = self.zero_of(&col_ty, Some(v), span)?;
                let cols = self.ones_from_zeros(cols, span)?;
                let mid = self.emit(
                    OpKind::MatMul,
                    vec![rows, v],
                    shaped_type(n.dim, Shape::vector(dims[1].clone()), n.repr),
                    span,
                );
                Ok(self.emit(OpKind::MatMul, vec![mid, cols], scalar, span))
            }
        }
    }

    /// The sum over all entries of `u ⊙ w` (the pairwise product), a scalar
    /// of dimension `dim(u) dim(w)`: the reduction that a scalar factor
    /// of a scaled array gets as its adjoint.
    pub(crate) fn hadamard_sum(&mut self, u: ValueId, w: ValueId, span: Span) -> Res<ValueId> {
        let (nu, nw) = (self.new_num(u)?, self.new_num(w)?);
        if nu.shape.rank() == 1 {
            let dim = self.dim_mul(nu.dim, nw.dim)?;
            return Ok(self.emit(OpKind::MatMul, vec![u, w], number_type(dim, nu.repr), span));
        }
        let p = self.hadamard(u, w, span)?;
        self.sum_all(p, span)
    }

    /// `v[i]` for a constant position `i`.
    pub(crate) fn element(&mut self, v: ValueId, i: u64, span: Span) -> Res<ValueId> {
        let n = self.new_num(v)?;
        let pos = self.konst_dimless(i as f64, n.repr, span);
        let ty = if n.shape.rank() == 1 {
            number_type(n.dim, n.repr)
        } else {
            shaped_type(n.dim, Shape::new(n.shape.dims()[1..].to_vec()), n.repr)
        };
        Ok(self.emit(OpKind::Index, vec![v, pos], ty, span))
    }

    /// `v[i, j]` for constant positions.
    pub(crate) fn element2(&mut self, v: ValueId, i: u64, j: u64, span: Span) -> Res<ValueId> {
        let n = self.new_num(v)?;
        let pi = self.konst_dimless(i as f64, n.repr, span);
        let pj = self.konst_dimless(j as f64, n.repr, span);
        Ok(self.emit(
            OpKind::Index,
            vec![v, pi, pj],
            number_type(n.dim, n.repr),
            span,
        ))
    }

    /// The known sizes of a vector or matrix, or an error saying the
    /// operation needs them.
    pub(crate) fn known_sizes(&self, v: ValueId, why: &str) -> Res<Vec<u64>> {
        let n = self.new_num(v)?;
        let mut sizes = Vec::new();
        for d in n.shape.dims() {
            match d {
                ShapeDim::Known(k) => sizes.push(*k),
                ShapeDim::Var(name) => {
                    return Err(self.unsupported(
                        "array",
                        &format!("{why} needs sizes known at compile time, and {name} is not"),
                    ))
                }
            }
        }
        Ok(sizes)
    }

    /// The vector `v` (of `m` entries) as an `m×1` column (`cols == 1`) or a
    /// `1×m` row (`cols == m`), gathered from its entries.
    pub(crate) fn as_matrix(&mut self, v: ValueId, column: bool, span: Span) -> Res<ValueId> {
        let sizes = self.known_sizes(v, "reshaping a vector")?;
        let m = sizes[0];
        let n = self.new_num(v)?;
        let entries = (0..m)
            .map(|i| self.element(v, i, span))
            .collect::<Res<Vec<_>>>()?;
        let shape = if column {
            Shape::matrix(m, 1)
        } else {
            Shape::matrix(1, m)
        };
        Ok(self.emit(
            OpKind::ArrayLiteral {
                cols: Some(if column { 1 } else { m }),
            },
            entries,
            shaped_type(n.dim, shape, n.repr),
            span,
        ))
    }

    /// The outer product `u vᵀ`: an `n×m` matrix of dimension
    /// `dim(u) dim(v)`, for vectors of known sizes.
    pub(crate) fn outer(&mut self, u: ValueId, v: ValueId, span: Span) -> Res<ValueId> {
        let (su, sv) = (
            self.known_sizes(u, "an outer product")?,
            self.known_sizes(v, "an outer product")?,
        );
        if su[0].saturating_mul(sv[0]) > MAX_STATIC_ENTRIES {
            return Err(self.unsupported(
                "array",
                "an outer product of this size is too large to build from entries",
            ));
        }
        let (nu, nv) = (self.new_num(u)?, self.new_num(v)?);
        let col = self.as_matrix(u, true, span)?;
        let row = self.as_matrix(v, false, span)?;
        let dim = self.dim_mul(nu.dim, nv.dim)?;
        Ok(self.emit(
            OpKind::MatMul,
            vec![col, row],
            shaped_type(dim, Shape::matrix(su[0], sv[0]), nu.repr),
            span,
        ))
    }

    /// A type with `ty`'s dimension and shape and no tags.
    pub(crate) fn untagged(&self, n: &NumberType) -> Type {
        Type::Number(NumberType {
            tags: TagSet::EMPTY,
            ..n.clone()
        })
    }
}
