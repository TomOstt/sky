//! Small arrays in native code: scalar replacement.
//!
//! A vector or matrix whose shape is known at compile time and has at most
//! [`MAX_SMALL`] entries is held as `N` separate `f64` values ([`Rep::Small`]),
//! not as a handle. Every operation on such arrays that is worth inlining is
//! emitted here as straight-line scalar code; see the module docs of
//! [`super`] for the ABI and for when a small array is materialized.
//!
//! # Exactness
//!
//! Each entry is computed with the operations, and in the order, of the
//! shared kernels (`sky_interp::array`, `sky_interp::vecfn`, `sky_linalg`):
//!
//! * elementwise `+ - *` and `/` are one IEEE operation per entry (`neg` is
//!   `0 - x`, as on arrays); `array * s` is `x * s`, `array / s` is `x / s`;
//! * `dot`, `a @ b` of two vectors and the sum of squares in `norm` add their
//!   products left to right from `+0.0` when there are fewer than 8, and in 8
//!   interleaved partial sums combined by the fixed tree otherwise;
//! * every other product entry is the naive sum `((0 + a₀b₀) + a₁b₁) + …`;
//! * a product is rounded before it is added: Cranelift never fuses.
//!
//! # Errors
//!
//! Operands are finite, so an operation fails exactly when a result entry is
//! not finite (or a divisor is zero, or an index is not a valid position).
//! Each operation checks that once, on all its entries, and on failure jumps
//! to a cold block that hands the *whole operands* to the runtime
//! (`sky_rt_fail_op`), which runs the operation in the interpreters' own code
//! and so produces their diagnostic: message, spans and labels.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, InstBuilder, Value as Cv};
use sky_ir::{OpData, OpId, OpKind, Type, ValueId};
use sky_types::ShapeDim;
use sky_units::Dimension;

use super::{dim_of_type, number_dim, refuse, rep_of, Comp, FnTx, Rep, Res, Rt, Slot};

/// The most entries a small array has. Chosen by measuring `nbody`, an RK4
/// step and a 4-by-4 product with the limit at 8, 16 and 32 (see the commit
/// that introduced this module): a limit of 16 covers 2, 3 and 4-vectors and
/// matrices up to 4 by 4, and a matrix product of two 4 by 4 matrices (64
/// multiplications, 48 additions) is still worth unrolling. Above it the
/// generated code grows faster than the allocation it saves is worth.
pub(super) const MAX_SMALL: usize = 16;

/// The shape of a small array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Sh {
    pub rows: usize,
    /// The columns of a matrix; 0 for a vector.
    pub cols: usize,
}

impl Sh {
    fn vector(n: usize) -> Sh {
        Sh { rows: n, cols: 0 }
    }

    fn matrix(rows: usize, cols: usize) -> Sh {
        Sh { rows, cols }
    }

    fn is_vector(self) -> bool {
        self.cols == 0
    }

    /// The number of entries.
    pub(super) fn len(self) -> usize {
        self.rows * self.cols.max(1)
    }
}

/// The shape `t` has if it is a number type of a known, small shape.
pub(super) fn small_shape(t: &Type) -> Option<Sh> {
    let Type::Number(n) = t else {
        return None;
    };
    let ok = |len: u64| (1..=MAX_SMALL as u64).contains(&len);
    match n.shape.dims() {
        [ShapeDim::Known(a)] if ok(*a) => Some(Sh::vector(*a as usize)),
        [ShapeDim::Known(a), ShapeDim::Known(b)] if a.checked_mul(*b).is_some_and(ok) => {
            Some(Sh::matrix(*a as usize, *b as usize))
        }
        _ => None,
    }
}

/// An operand a small-array operation can work on inline.
enum Opnd {
    /// A scalar.
    S(Cv, Dimension),
    /// A small array's entries.
    A(Vec<Cv>, Sh, Dimension),
}

/// A position of an index: known at compile time, or a run-time `f64`.
enum Pos {
    C(usize),
    D(Cv),
}

impl FnTx<'_, '_> {
    // ----- the runtime side -----

    fn array_args(&mut self, sh: Sh, d: Dimension) -> [Cv; 3] {
        let d = self.dim_id(d);
        [
            self.c32(d),
            self.c32(sh.rows as i32),
            self.c32(sh.cols as i32),
        ]
    }

    /// Pushes the entries `elems` as one array operand for the next runtime
    /// operation.
    pub(super) fn push_small(&mut self, elems: &[Cv], sh: Sh, d: Dimension) {
        for &e in elems {
            self.call(Rt::PushElem, &[e]);
        }
        let args = self.array_args(sh, d);
        self.call(Rt::PushArray, &args);
    }

    /// A handle (owned by the caller) to the array of `elems`.
    pub(super) fn box_small(&mut self, elems: &[Cv], sh: Sh, d: Dimension) -> Cv {
        for &e in elems {
            self.call(Rt::PushElem, &[e]);
        }
        let args = self.array_args(sh, d);
        let h = self.call1(Rt::BoxArray, &args);
        self.check_handle(h);
        h
    }

    /// Prints the array of `elems` as one item of the `print` being built.
    pub(super) fn print_small(&mut self, elems: &[Cv], sh: Sh, d: Dimension) {
        for &e in elems {
            self.call(Rt::PushElem, &[e]);
        }
        let args = self.array_args(sh, d);
        self.call(Rt::PrintArray, &args);
    }

    /// The entries of the array in handle `h`, which must be an array of
    /// shape `sh` and dimension `d` (else the run deoptimizes). With
    /// `release` the handle is released.
    pub(super) fn unpack_small(&mut self, h: Cv, sh: Sh, d: Dimension, release: bool) -> Vec<Cv> {
        let [d, r, c] = self.array_args(sh, d);
        let rel = self.c32(i32::from(release));
        let status = self.call1(Rt::Unpack, &[h, d, r, c, rel]);
        self.status_fail(status);
        (0..sh.len())
            .map(|i| {
                let i = self.c32(i as i32);
                self.call1(Rt::Elem, &[i])
            })
            .collect()
    }

    /// Records that `op`'s result is the constant `x`.
    pub(super) fn note_const(&mut self, op: &OpData, x: f64) {
        if let Some(&r) = op.results.first() {
            self.consts[r.index()] = Some(x);
        }
    }

    /// Continues if `ok` (an integer) is nonzero; else runs `op` in the
    /// runtime on its whole operands, which fails with the interpreters'
    /// error, and leaves.
    fn fail_op_unless(&mut self, ok: Cv, id: OpId, op: &OpData) -> Res<()> {
        let (cold, cont) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(ok, cont, &[], cold, &[]);
        self.b.set_cold_block(cold);
        self.b.switch_to_block(cold);
        self.push_operands(op, 0..op.operands.len())?;
        let site = self.site(id);
        let site = self.c32(site);
        self.call(Rt::FailOp, &[site]);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        self.b.switch_to_block(cont);
        Ok(())
    }

    // ----- emitters -----

    fn f64c(&mut self, x: f64) -> Cv {
        self.b.ins().f64const(x)
    }

    /// An integer, nonzero if every one of `xs` is finite.
    fn all_finite(&mut self, xs: &[Cv]) -> Cv {
        let inf = self.f64c(f64::INFINITY);
        let mut ok: Option<Cv> = None;
        for &x in xs {
            let abs = self.b.ins().fabs(x);
            // False for infinity and for NaN (an ordered comparison).
            let c = self.b.ins().fcmp(FloatCC::LessThan, abs, inf);
            ok = Some(match ok {
                None => c,
                Some(o) => self.b.ins().band(o, c),
            });
        }
        ok.unwrap_or_else(|| self.b.ins().iconst(types::I8, 1))
    }

    /// The sum of `terms` in the fixed order of `sky_linalg::dot`: fewer than
    /// 8 left to right from `+0.0`, else 8 interleaved partial sums combined
    /// by a fixed tree.
    fn ordered_sum(&mut self, terms: &[Cv]) -> Cv {
        const LANES: usize = 8;
        let zero = self.f64c(0.0);
        if terms.len() < LANES {
            return terms.iter().fold(zero, |s, &t| self.b.ins().fadd(s, t));
        }
        let mut lane = [zero; LANES];
        for (i, &t) in terms.iter().enumerate() {
            lane[i % LANES] = self.b.ins().fadd(lane[i % LANES], t);
        }
        let t0 = self.b.ins().fadd(lane[0], lane[4]);
        let t1 = self.b.ins().fadd(lane[1], lane[5]);
        let t2 = self.b.ins().fadd(lane[2], lane[6]);
        let t3 = self.b.ins().fadd(lane[3], lane[7]);
        let (u, v) = (self.b.ins().fadd(t0, t2), self.b.ins().fadd(t1, t3));
        self.b.ins().fadd(u, v)
    }

    /// `list[idx]` for an `idx` (an `f64`) known to be an integer in range.
    fn pick(&mut self, list: &[Cv], idx: Cv) -> Cv {
        let last = list.len() - 1;
        let mut acc = list[last];
        for (k, &x) in list[..last].iter().enumerate().rev() {
            let kc = self.f64c(k as f64);
            let hit = self.b.ins().fcmp(FloatCC::Equal, idx, kc);
            acc = self.b.ins().select(hit, x, acc);
        }
        acc
    }

    // ----- operands -----

    fn opnd(&self, op: &OpData, i: usize) -> Res<Option<Opnd>> {
        let Some(&v) = op.operands.get(i) else {
            return Ok(None);
        };
        let s = self.operand(op, i)?;
        Ok(match s.rep {
            Rep::F64 => dim_of_type(self.f.ty(v)).map(|d| Opnd::S(s.v, d)),
            Rep::Small(sh) => match number_dim(self.f.ty(v)) {
                Some(d) => Some(Opnd::A(self.small_elems(v, op)?, sh, d)),
                None => None,
            },
            _ => None,
        })
    }

    /// Whether `op` reads or makes a small array.
    fn touches_small(&self, op: &OpData) -> bool {
        op.operands.iter().any(|v| {
            matches!(
                self.vals.get(v.index()),
                Some(Some(Slot {
                    rep: Rep::Small(_),
                    ..
                }))
            )
        }) || op
            .results
            .iter()
            .any(|r| matches!(rep_of(self.f.ty(*r)), Rep::Small(_)))
    }

    /// Emits `op` inline if it is a small-array operation this module
    /// handles on these operands. `false`: nothing was defined, the caller
    /// translates it the usual way (through the runtime).
    pub(super) fn small_op(&mut self, id: OpId, op: &OpData) -> Res<bool> {
        if !self.touches_small(op) {
            return Ok(false);
        }
        match &op.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => self.small_arith(id, op),
            OpKind::Neg => self.small_neg(op),
            OpKind::Index => self.small_index(id, op),
            OpKind::Dot => self.small_dot(id, op),
            OpKind::Norm => self.small_norm(id, op),
            OpKind::MatMul => self.small_matmul(id, op, false, false),
            OpKind::MatMulEx { ta, tb, .. } => self.small_matmul(id, op, *ta, *tb),
            OpKind::Transpose => self.small_transpose(op),
            OpKind::ArrayLiteral { cols } => self.small_literal(op, *cols),
            OpKind::Zeros | OpKind::Identity => self.small_fill(op),
            OpKind::Len => self.small_len(op),
            OpKind::ExpectNumber { .. } => self.small_expect(op),
            OpKind::Fused => self.small_fused(op),
            _ => Ok(false),
        }
    }

    // ----- elementwise arithmetic -----

    /// `+ - * /` with a small array operand: array and array (`+ -`), or
    /// array and scalar (`* /`, and `scalar * array`).
    fn small_arith(&mut self, id: OpId, op: &OpData) -> Res<bool> {
        let (Some(a), Some(b)) = (self.opnd(op, 0)?, self.opnd(op, 1)?) else {
            return Ok(false);
        };
        let mut extra_ok = None;
        let (out, sh, dim) = match (&op.kind, a, b) {
            (OpKind::Add | OpKind::Sub, Opnd::A(x, sx, dx), Opnd::A(y, sy, dy)) => {
                if sx != sy || dx != dy {
                    return Ok(false);
                }
                let add = matches!(op.kind, OpKind::Add);
                let out: Vec<Cv> = x
                    .iter()
                    .zip(&y)
                    .map(|(&p, &q)| {
                        if add {
                            self.b.ins().fadd(p, q)
                        } else {
                            self.b.ins().fsub(p, q)
                        }
                    })
                    .collect();
                (out, sx, dx)
            }
            (OpKind::Mul, Opnd::A(x, sx, dx), Opnd::S(s, ds))
            | (OpKind::Mul, Opnd::S(s, ds), Opnd::A(x, sx, dx)) => {
                let Ok(dim) = dx.checked_mul(ds) else {
                    return Ok(false);
                };
                let out = x.iter().map(|&p| self.b.ins().fmul(p, s)).collect();
                (out, sx, dim)
            }
            (OpKind::Div, Opnd::A(x, sx, dx), Opnd::S(s, ds)) => {
                let Ok(dim) = dx.checked_div(ds) else {
                    return Ok(false);
                };
                // A zero divisor fails before any entry is looked at.
                let zero = self.f64c(0.0);
                extra_ok = Some(self.b.ins().fcmp(FloatCC::NotEqual, s, zero));
                let out = x.iter().map(|&p| self.b.ins().fdiv(p, s)).collect();
                (out, sx, dim)
            }
            _ => return Ok(false),
        };
        let mut ok = self.all_finite(&out);
        if let Some(e) = extra_ok {
            ok = self.b.ins().band(ok, e);
        }
        self.fail_op_unless(ok, id, op)?;
        self.define(op, 0, Comp::Small(out, sh, dim))?;
        Ok(true)
    }

    /// `-a` of a small array: `0 - x` for every entry, so a zero stays `+0`.
    fn small_neg(&mut self, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(x, sh, dim)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        let zero = self.f64c(0.0);
        let out = x.iter().map(|&p| self.b.ins().fsub(zero, p)).collect();
        self.define(op, 0, Comp::Small(out, sh, dim))?;
        Ok(true)
    }

    fn small_expect(&mut self, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(x, sh, dim)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        if !matches!(op.results.first().map(|r| rep_of(self.f.ty(*r))), Some(Rep::Small(s)) if s == sh)
        {
            return Ok(false);
        }
        self.define(op, 0, Comp::Small(x, sh, dim))?;
        Ok(true)
    }

    // ----- making arrays -----

    fn small_literal(&mut self, op: &OpData, cols: Option<u64>) -> Res<bool> {
        let Some(&r) = op.results.first() else {
            return Ok(false);
        };
        let Rep::Small(sh) = rep_of(self.f.ty(r)) else {
            return Ok(false);
        };
        let n = op.operands.len();
        let shape_ok = match cols {
            None => sh == Sh::vector(n),
            Some(c) => {
                c > 0 && n % (c as usize) == 0 && sh == Sh::matrix(n / c as usize, c as usize)
            }
        };
        if !shape_ok {
            return Ok(false);
        }
        let mut elems = Vec::with_capacity(n);
        let mut dim = None;
        for i in 0..n {
            let Some(Opnd::S(x, d)) = self.opnd(op, i)? else {
                return Ok(false);
            };
            if *dim.get_or_insert(d) != d {
                return Ok(false);
            }
            elems.push(x);
        }
        let Some(dim) = dim else {
            return Ok(false);
        };
        self.define(op, 0, Comp::Small(elems, sh, dim))?;
        Ok(true)
    }

    /// `zeros(n)`, `zeros(m, n)`, `identity(n)` of constant sizes.
    fn small_fill(&mut self, op: &OpData) -> Res<bool> {
        let Some(&r) = op.results.first() else {
            return Ok(false);
        };
        let Rep::Small(sh) = rep_of(self.f.ty(r)) else {
            return Ok(false);
        };
        let mut sizes = Vec::new();
        for &v in &op.operands {
            match (
                self.consts.get(v.index()).copied().flatten(),
                dim_of_type(self.f.ty(v)),
            ) {
                (Some(c), Some(d)) if d.is_dimensionless() && c.fract() == 0.0 && c >= 1.0 => {
                    sizes.push(c as usize)
                }
                _ => return Ok(false),
            }
        }
        let want = match (&op.kind, sizes[..].as_ref()) {
            (OpKind::Zeros, [n]) => Sh::vector(*n),
            (OpKind::Zeros, [m, n]) => Sh::matrix(*m, *n),
            (OpKind::Identity, [n]) => Sh::matrix(*n, *n),
            _ => return Ok(false),
        };
        if want != sh {
            return Ok(false);
        }
        let identity = matches!(op.kind, OpKind::Identity);
        let elems = (0..sh.len())
            .map(|i| {
                let one = identity && i / sh.cols == i % sh.cols;
                self.f64c(if one { 1.0 } else { 0.0 })
            })
            .collect();
        self.define(op, 0, Comp::Small(elems, sh, Dimension::DIMENSIONLESS))?;
        Ok(true)
    }

    /// `len(a)`: the first axis, known from the shape.
    fn small_len(&mut self, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(_, sh, _)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        let n = self.f64c(sh.rows as f64);
        self.note_const(op, sh.rows as f64);
        self.define(op, 0, Comp::F64(n, Dimension::DIMENSIONLESS))?;
        Ok(true)
    }

    // ----- indexing -----

    /// `a[i]`, `a[i, j]`: a constant position is picked at compile time; a
    /// run-time one is validated (a whole number, in range: the failing case
    /// is run in the runtime for its error) and then selected.
    fn small_index(&mut self, id: OpId, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(x, sh, dim)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        let k = op.operands.len() - 1;
        let lens = if sh.is_vector() {
            vec![sh.rows]
        } else {
            vec![sh.rows, sh.cols]
        };
        if k == 0 || k > lens.len() {
            return Ok(false);
        }
        let mut pos = Vec::with_capacity(k);
        let mut ok: Option<Cv> = None;
        for (j, &len) in lens.iter().enumerate().take(k) {
            let v = op.operands[j + 1];
            let Some(Opnd::S(i, d)) = self.opnd(op, j + 1)? else {
                return Ok(false);
            };
            if !d.is_dimensionless() {
                return Ok(false);
            }
            match self.consts.get(v.index()).copied().flatten() {
                Some(c) => {
                    if c.fract() != 0.0 || c < 0.0 || c >= len as f64 {
                        return Ok(false);
                    }
                    pos.push(Pos::C(c as usize));
                }
                None => {
                    let whole = {
                        let fl = self.b.ins().floor(i);
                        self.b.ins().fcmp(FloatCC::Equal, i, fl)
                    };
                    let zero = self.f64c(0.0);
                    let nonneg = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, i, zero);
                    let n = self.f64c(len as f64);
                    let below = self.b.ins().fcmp(FloatCC::LessThan, i, n);
                    let all = self.b.ins().band(whole, nonneg);
                    let all = self.b.ins().band(all, below);
                    ok = Some(match ok {
                        None => all,
                        Some(o) => self.b.ins().band(o, all),
                    });
                    pos.push(Pos::D(i));
                }
            }
        }
        if let Some(ok) = ok {
            self.fail_op_unless(ok, id, op)?;
        }
        let as_f64 = |tx: &mut Self, p: &Pos| match p {
            Pos::C(c) => tx.f64c(*c as f64),
            Pos::D(v) => *v,
        };
        let comp = if sh.is_vector() {
            let v = match &pos[0] {
                Pos::C(i) => x[*i],
                Pos::D(i) => self.pick(&x, *i),
            };
            Comp::F64(v, dim)
        } else if k == 2 {
            let v = match (&pos[0], &pos[1]) {
                (Pos::C(i), Pos::C(j)) => x[i * sh.cols + j],
                (p, q) => {
                    let (pf, qf) = (as_f64(self, p), as_f64(self, q));
                    let cols = self.f64c(sh.cols as f64);
                    let row = self.b.ins().fmul(pf, cols);
                    let lin = self.b.ins().fadd(row, qf);
                    self.pick(&x, lin)
                }
            };
            Comp::F64(v, dim)
        } else {
            // A row of a matrix.
            let row: Vec<Cv> = (0..sh.cols)
                .map(|c| match &pos[0] {
                    Pos::C(i) => x[i * sh.cols + c],
                    Pos::D(i) => {
                        let column: Vec<Cv> = (0..sh.rows).map(|r| x[r * sh.cols + c]).collect();
                        self.pick(&column, *i)
                    }
                })
                .collect();
            Comp::Small(row, Sh::vector(sh.cols), dim)
        };
        self.define(op, 0, comp)?;
        Ok(true)
    }

    // ----- reductions and products -----

    /// `dot(a, b)` of two small vectors.
    fn small_dot(&mut self, id: OpId, op: &OpData) -> Res<bool> {
        let (Some(Opnd::A(x, sx, dx)), Some(Opnd::A(y, sy, dy))) =
            (self.opnd(op, 0)?, self.opnd(op, 1)?)
        else {
            return Ok(false);
        };
        if !sx.is_vector() || sx != sy {
            return Ok(false);
        }
        let Ok(dim) = dx.checked_mul(dy) else {
            return Ok(false);
        };
        let terms: Vec<Cv> = x
            .iter()
            .zip(&y)
            .map(|(&p, &q)| self.b.ins().fmul(p, q))
            .collect();
        let sum = self.ordered_sum(&terms);
        let ok = self.all_finite(&[sum]);
        self.fail_op_unless(ok, id, op)?;
        self.define(op, 0, Comp::F64(sum, dim))?;
        Ok(true)
    }

    /// `norm(a)` of a small array: the plain sum of squares when the largest
    /// entry is between `1e-100` and `1e100` (or all are zero), which is
    /// what `sky_interp::vecfn::norm` computes then; the scaled form for
    /// extreme entries runs in the runtime.
    fn small_norm(&mut self, id: OpId, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(x, _, dim)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        let zero = self.f64c(0.0);
        let scale = x.iter().fold(zero, |m, &p| {
            let a = self.b.ins().fabs(p);
            self.b.ins().fmax(m, a)
        });
        let squares: Vec<Cv> = x.iter().map(|&p| self.b.ins().fmul(p, p)).collect();
        let sum = self.ordered_sum(&squares);
        let fast = self.b.ins().sqrt(sum);
        let is_zero = self.b.ins().fcmp(FloatCC::Equal, scale, zero);
        let lo = self.f64c(1e-100);
        let hi = self.f64c(1e100);
        let above = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, scale, lo);
        let below = self.b.ins().fcmp(FloatCC::LessThan, scale, hi);
        let mid = self.b.ins().band(above, below);
        let ordinary = self.b.ins().bor(is_zero, mid);

        let (slow, merge) = (self.b.create_block(), self.b.create_block());
        self.b.append_block_param(merge, types::F64);
        self.b
            .ins()
            .brif(ordinary, merge, &[fast.into()], slow, &[]);
        self.b.set_cold_block(slow);
        self.b.switch_to_block(slow);
        self.push_operands(op, 0..1)?;
        let site = self.site(id);
        let site = self.c32(site);
        let h = self.call1(Rt::Op, &[site]);
        self.check_handle(h);
        let d = self.dim_id(dim);
        let d = self.c32(d);
        let r = self.call1(Rt::TakeNum, &[h, d]);
        self.nan_fail(r);
        self.b.ins().jump(merge, &[r.into()]);
        self.b.switch_to_block(merge);
        let result = self.b.block_params(merge)[0];
        self.define(op, 0, Comp::F64(result, dim))?;
        Ok(true)
    }

    /// `a @ b` (and `matmul[ta, tb]`, which is the product of the
    /// transposed operands): the dot product of two vectors in the fixed
    /// order of `sky_linalg::dot`, every other entry the naive sum.
    fn small_matmul(&mut self, id: OpId, op: &OpData, ta: bool, tb: bool) -> Res<bool> {
        let (Some(Opnd::A(mut x, mut sx, dx)), Some(Opnd::A(mut y, mut sy, dy))) =
            (self.opnd(op, 0)?, self.opnd(op, 1)?)
        else {
            return Ok(false);
        };
        for (flag, e, s) in [(ta, &mut x, &mut sx), (tb, &mut y, &mut sy)] {
            if flag {
                if s.is_vector() {
                    return Ok(false);
                }
                *e = transposed(e, *s);
                *s = Sh::matrix(s.cols, s.rows);
            }
        }
        let Ok(dim) = dx.checked_mul(dy) else {
            return Ok(false);
        };
        // A vector is a row on the left and a column on the right.
        let (m, k) = if sx.is_vector() {
            (None, sx.rows)
        } else {
            (Some(sx.rows), sx.cols)
        };
        let (k2, n) = if sy.is_vector() {
            (sy.rows, None)
        } else {
            (sy.rows, Some(sy.cols))
        };
        if k != k2 {
            return Ok(false);
        }
        let comp = if let (None, None) = (m, n) {
            let terms: Vec<Cv> = x
                .iter()
                .zip(&y)
                .map(|(&p, &q)| self.b.ins().fmul(p, q))
                .collect();
            let sum = self.ordered_sum(&terms);
            let ok = self.all_finite(&[sum]);
            self.fail_op_unless(ok, id, op)?;
            Comp::F64(sum, dim)
        } else {
            let (mm, nn) = (m.unwrap_or(1), n.unwrap_or(1));
            let mut out = Vec::with_capacity(mm * nn);
            for i in 0..mm {
                for j in 0..nn {
                    let mut s = self.f64c(0.0);
                    for p in 0..k {
                        let prod = self.b.ins().fmul(x[i * k + p], y[p * nn + j]);
                        s = self.b.ins().fadd(s, prod);
                    }
                    out.push(s);
                }
            }
            let ok = self.all_finite(&out);
            self.fail_op_unless(ok, id, op)?;
            let sh = match (m, n) {
                (Some(m), None) => Sh::vector(m),
                (None, Some(n)) => Sh::vector(n),
                _ => Sh::matrix(mm, nn),
            };
            Comp::Small(out, sh, dim)
        };
        self.define(op, 0, comp)?;
        Ok(true)
    }

    fn small_transpose(&mut self, op: &OpData) -> Res<bool> {
        let Some(Opnd::A(x, sh, dim)) = self.opnd(op, 0)? else {
            return Ok(false);
        };
        if sh.is_vector() {
            return Ok(false);
        }
        let out = transposed(&x, sh);
        self.define(op, 0, Comp::Small(out, Sh::matrix(sh.cols, sh.rows), dim))?;
        Ok(true)
    }

    // ----- fused -----

    /// A `fused` body over small arrays: the body's operations, one after
    /// another, each on every entry (or once, if it reads only scalars).
    ///
    /// The order matters for errors: the interpreter reports the failure of
    /// the earliest operation *in body order* that fails at any entry, and
    /// the first entry among those. Running each operation over all the
    /// entries before the next one, stopping at the first failure, is
    /// exactly that. Each scalar operation is the scalar code of
    /// `module.rs`, whose failure re-runs that one operation on its operands
    /// (the same function the interpreter's loop calls, with the same
    /// spans). `neg` is `0 - x`, as in the interpreter's body evaluation.
    fn small_fused(&mut self, op: &OpData) -> Res<bool> {
        let f = self.f;
        let [region] = op.regions[..] else {
            return Ok(false);
        };
        let Ok(bid) = self.single_block(region, op) else {
            return Ok(false);
        };
        let block = f.block(bid);
        let Some((&last, body)) = block.ops.split_last() else {
            return Ok(false);
        };
        let last = f.op(last);
        let (OpKind::Yield, &[yielded]) = (&last.kind, &last.operands[..]) else {
            return Ok(false);
        };
        let Some(&r) = op.results.first() else {
            return Ok(false);
        };
        let Rep::Small(sh) = rep_of(f.ty(r)) else {
            return Ok(false);
        };
        if block.args.len() != op.operands.len() {
            return Ok(false);
        }
        let supported = body.iter().all(|&o| {
            let b = f.op(o);
            b.results.len() == 1
                && matches!(
                    b.kind,
                    OpKind::ConstFloat { .. }
                        | OpKind::ConstRational(_)
                        | OpKind::Add
                        | OpKind::Sub
                        | OpKind::Mul
                        | OpKind::Div
                        | OpKind::Neg
                )
        });
        if !supported {
            return Ok(false);
        }
        // The value of each body argument: per entry, or one scalar.
        let mut per: HashMap<ValueId, Vec<Cv>> = HashMap::new();
        let mut binds = Vec::new();
        for (i, &arg) in block.args.iter().enumerate() {
            match self.opnd(op, i)? {
                Some(Opnd::A(x, s, _)) if s == sh => {
                    per.insert(arg, x);
                }
                Some(Opnd::S(v, _)) => binds.push((arg, v)),
                _ => return Ok(false),
            }
        }
        for (arg, v) in binds {
            self.vals[arg.index()] = Some(Slot { v, rep: Rep::F64 });
        }
        for &oid in body {
            let b = f.op(oid);
            let result = b.results[0];
            let neg = matches!(b.kind, OpKind::Neg);
            if !b.operands.iter().any(|v| per.contains_key(v)) {
                if neg {
                    let x = self.slot(b.operands[0], b)?.v;
                    let zero = self.f64c(0.0);
                    let v = self.b.ins().fsub(zero, x);
                    self.vals[result.index()] = Some(Slot { v, rep: Rep::F64 });
                } else {
                    self.op(oid)?;
                }
                continue;
            }
            let mut outs = Vec::with_capacity(sh.len());
            for e in 0..sh.len() {
                for v in &b.operands {
                    if let Some(xs) = per.get(v) {
                        self.vals[v.index()] = Some(Slot {
                            v: xs[e],
                            rep: Rep::F64,
                        });
                    }
                }
                if neg {
                    let x = self.slot(b.operands[0], b)?.v;
                    let zero = self.f64c(0.0);
                    outs.push(self.b.ins().fsub(zero, x));
                } else {
                    self.op(oid)?;
                    outs.push(self.slot(result, b)?.v);
                }
            }
            per.insert(result, outs);
        }
        let out = match per.get(&yielded) {
            Some(xs) => xs.clone(),
            None => {
                let v = self.slot(yielded, last)?.v;
                vec![v; sh.len()]
            }
        };
        let Some(dim) = dim_of_type(f.ty(yielded)) else {
            return refuse("`fused` yields a value that is not a scalar", Some(op.span));
        };
        self.define(op, 0, Comp::Small(out, sh, dim))?;
        Ok(true)
    }
}

/// The entries of the transpose of the matrix `x` of shape `sh`.
fn transposed(x: &[Cv], sh: Sh) -> Vec<Cv> {
    let mut out = Vec::with_capacity(x.len());
    for j in 0..sh.cols {
        for i in 0..sh.rows {
            out.push(x[i * sh.cols + j]);
        }
    }
    out
}
