//! Superword-level parallelism: pairs of `f64` operations as `F64X2` ones.
//!
//! Physics code does the same operation on `x`, `y` and `z`, side by side,
//! and the entries of a small array ([`super::small`]) are separate `f64`
//! values. When an array operation has two entries to compute, this module
//! computes them as one two-lane vector operation and hands the results on
//! as the two scalars they are (`extractlane`); a later operation that wants
//! the same two values as a pair finds the vector they came from (the *lane
//! table*, [`FnTx::lanes`]) and uses it, so a chain of array operations
//! stays in vector registers and only the scalars something really reads are
//! extracted. Extractions nobody reads are dead code.
//!
//! # When it pays
//!
//! Packing raises the *throughput* of arithmetic and cuts the branches (one
//! check per operation, not one per entry), but it puts a scalar-to-vector
//! move in front of the first operation of a chain and a vector-to-scalar
//! move behind the last one, and neither shortens a dependence chain. So it
//! wins where the machine is short of issue slots (a long run of independent
//! array operations per iteration) and loses where one iteration waits for
//! the last (a time step `v = v + a * dt`, `x = x + v * dt`, whose result is
//! the next step's input). [`latency_bound`] tells the two apart from the
//! SIR: the longest chain of dependent operations from a loop's carried
//! values to its next ones, in cycles, against the amount of arithmetic in
//! one iteration. [`Slp::Auto`] packs only in loops and functions that are
//! not latency bound; measured on an Apple M2 Pro, a run of independent
//! 4-vector operations takes 37% less time packed, and the recurrences above
//! take 10 to 35% more.
//!
//! # Exactness
//!
//! * A lane of `fadd`, `fsub`, `fmul`, `fdiv` on `F64X2` is the IEEE
//!   operation on that lane: bit for bit the scalar instruction's result
//!   (both machines this runs on round every lane in the current mode,
//!   round-to-nearest-even, with no fusing: Cranelift has no contraction).
//!   Nothing is reassociated: a packed operation is two whole scalar
//!   operations with the same operands, and no operand is a combination of
//!   lanes that the scalar code did not compute. Vector constants hold the
//!   scalars' own bits.
//! * Packing happens only *inside one SIR operation* (the entries of one
//!   array operation, or of one operation of a `fused` body), so what must
//!   be decided is only "does this operation fail, and where". The check is
//!   one for all entries: the sum of the entries' `x - x` (`+0` for a finite
//!   `x`, NaN for infinity and NaN, and a NaN survives every sum) is ordered.
//!   If it is not, control goes to a cold block that does not use the
//!   vectors at all: for an array operation it re-runs the operation on its
//!   whole operands in the runtime, as before (`small::fail_op_unless`); for
//!   an operation of a `fused` body it runs the entries' scalar operations
//!   one by one, each with its own check, the code an unpacked run would
//!   have run, so the first failing entry reports (a zero divisor before a
//!   non-finite quotient) with its own operands. The operations run before it
//!   are pure, so it is the same failure the interpreter's loop reports, at
//!   the same operation, and the output before it is unchanged: nothing is
//!   printed inside an operation.
//! * `-0` and NaN need nothing special: the lanes are the scalar
//!   operations; a NaN result is an error (it is not finite), and `neg` of
//!   an array is `0 - x` (a zero stays `+0`) in the lanes as it is in scalars.

use std::collections::HashMap;
use std::sync::OnceLock;

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{
    types, ConstantData, InstBuilder, InstructionData, Opcode, Value as Cv, ValueDef,
};
use sky_ir::{BlockId, Function, OpData, OpId, OpKind, ValueId};

use super::small::small_shape;
use super::{rep_of, Comp, FnTx, Rep, Res};

/// How much native code vectorizes (`SKY_SLP=off|auto|always` for
/// [`crate::compile`]; [`crate::compile_with`] takes it explicitly).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Slp {
    /// Never: every entry of an array operation is a scalar operation, with
    /// a check of its own.
    Off,
    /// In a loop or function body that is not latency bound (see the module
    /// docs): pairs of entries are packed and an operation's entries share
    /// one check. Elsewhere as `Off`.
    #[default]
    Auto,
    /// Everywhere, whatever the loop does: for tests and measurements.
    Always,
}

impl Slp {
    /// The mode `SKY_SLP` names, else the default.
    pub fn from_env() -> Slp {
        static MODE: OnceLock<Slp> = OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("SKY_SLP").as_deref() {
            Ok("off") => Slp::Off,
            Ok("auto") => Slp::Auto,
            Ok("always") => Slp::Always,
            _ => Slp::default(),
        })
    }
}

/// An elementwise operation on two `f64`s.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Bin {
    Add,
    Sub,
    Mul,
    Div,
}

/// One side of an elementwise operation: an array's entries, or a scalar
/// used with every entry.
#[derive(Clone, Copy)]
pub(super) enum Side<'a> {
    Arr(&'a [Cv]),
    Scalar(Cv),
}

impl Side<'_> {
    fn at(&self, i: usize) -> Cv {
        match self {
            Side::Arr(x) => x[i],
            Side::Scalar(s) => *s,
        }
    }
}

impl<M: cranelift_module::Module> FnTx<'_, '_, M> {
    /// Whether array operations are vectorized here: in this loop or
    /// function body, by the mode.
    pub(super) fn slp_active(&self) -> bool {
        match self.sh.slp {
            Slp::Off => false,
            Slp::Always => true,
            Slp::Auto => self.slp_regions.last().copied().unwrap_or(false),
        }
    }

    /// Runs `body` (the translation of a loop) with the loop's own decision
    /// on vectorizing in force.
    pub(super) fn slp_region<T>(
        &mut self,
        pack: impl FnOnce() -> bool,
        body: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let on = self.sh.slp == Slp::Auto && pack();
        self.slp_regions.push(on);
        let r = body(self);
        self.slp_regions.pop();
        r
    }

    /// The vector `x` is a lane of, and which lane.
    fn lane_of(&self, x: Cv) -> Option<(Cv, u8)> {
        self.lanes.get(&x).copied()
    }

    /// The vector whose lanes are exactly `a` and `b`, if one already exists.
    fn ready_pair(&self, a: Cv, b: Cv) -> Option<Cv> {
        match (self.lane_of(a), self.lane_of(b)) {
            (Some((v, 0)), Some((w, 1))) if v == w => Some(v),
            _ => None,
        }
    }

    /// The value of `x` if it is an `f64const`.
    fn const_of(&self, x: Cv) -> Option<f64> {
        let dfg = &self.b.func.dfg;
        let ValueDef::Result(inst, _) = dfg.value_def(x) else {
            return None;
        };
        match dfg.insts[inst] {
            InstructionData::UnaryIeee64 {
                opcode: Opcode::F64const,
                imm,
            } => Some(f64::from_bits(imm.bits())),
            _ => None,
        }
    }

    /// The vector `[p, q]`, a load from the constant pool.
    fn vconst(&mut self, p: f64, q: f64) -> Cv {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&p.to_le_bytes());
        bytes.extend_from_slice(&q.to_le_bytes());
        let c = self.b.func.dfg.constants.insert(ConstantData::from(bytes));
        self.b.ins().vconst(types::F64X2, c)
    }

    /// `[s, s]`.
    fn splat(&mut self, s: Cv) -> Cv {
        match self.const_of(s) {
            Some(x) => self.vconst(x, x),
            None => self.b.ins().splat(types::F64X2, s),
        }
    }

    /// `side`'s entries `i` and `i + 1` as one vector.
    fn pair_vec(&mut self, side: Side<'_>, i: usize) -> Cv {
        match side {
            Side::Scalar(s) => self.splat(s),
            Side::Arr(x) => {
                let (a, b) = (x[i], x[i + 1]);
                if let Some(v) = self.ready_pair(a, b) {
                    v
                } else if a == b {
                    self.splat(a)
                } else if let (Some(p), Some(q)) = (self.const_of(a), self.const_of(b)) {
                    self.vconst(p, q)
                } else {
                    let v = self.b.ins().scalar_to_vector(types::F64X2, a);
                    self.b.ins().insertlane(v, b, 1u8)
                }
            }
        }
    }

    /// The two lanes of `v` as scalars, remembered as lanes of `v`.
    fn unpack(&mut self, v: Cv) -> [Cv; 2] {
        let a = self.b.ins().extractlane(v, 0u8);
        let b = self.b.ins().extractlane(v, 1u8);
        self.lanes.insert(a, (v, 0));
        self.lanes.insert(b, (v, 1));
        [a, b]
    }

    fn bin(&mut self, k: Bin, a: Cv, b: Cv) -> Cv {
        let ins = self.b.ins();
        match k {
            Bin::Add => ins.fadd(a, b),
            Bin::Sub => ins.fsub(a, b),
            Bin::Mul => ins.fmul(a, b),
            Bin::Div => ins.fdiv(a, b),
        }
    }

    /// `l[i] k r[i]` for every entry `i` of `n`: two lanes at a time where
    /// vectorizing is on ([`FnTx::slp_active`]), one entry when a single one
    /// is left. The results are scalars; a pair's are lanes of one vector,
    /// which [`FnTx::all_finite`] checks together.
    pub(super) fn zip_arith(&mut self, k: Bin, l: Side<'_>, r: Side<'_>, n: usize) -> Vec<Cv> {
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i + 1 < n && self.slp_active() {
                let (a, b) = (self.pair_vec(l, i), self.pair_vec(r, i));
                let v = self.bin(k, a, b);
                out.extend(self.unpack(v));
                i += 2;
            } else {
                let (a, b) = (l.at(i), r.at(i));
                out.push(self.bin(k, a, b));
                i += 1;
            }
        }
        out
    }

    /// An `i8` that is nonzero if every one of `xs` is finite.
    ///
    /// Vectorizing off: `|x| < inf` for each entry. On: `x - x` is `+0` for
    /// a finite `x` and NaN for infinity and NaN, so the sum of the `x - x`
    /// is `+0` when every `x` is finite and NaN otherwise (a NaN stays NaN
    /// in every sum, and `0 + 0` is `0`): one ordered comparison, one
    /// branch. Two consecutive entries that are the lanes of one vector share
    /// one vector subtraction (and the vectors' differences one vector
    /// addition).
    pub(super) fn all_finite(&mut self, xs: &[Cv]) -> Cv {
        if !self.slp_active() {
            return self.all_finite_scalar(xs);
        }
        let mut vsum: Option<Cv> = None;
        let mut sum: Option<Cv> = None;
        let mut i = 0;
        while i < xs.len() {
            if i + 1 < xs.len() {
                if let Some(v) = self.ready_pair(xs[i], xs[i + 1]) {
                    let d = self.b.ins().fsub(v, v);
                    vsum = Some(match vsum {
                        None => d,
                        Some(o) => self.b.ins().fadd(o, d),
                    });
                    i += 2;
                    continue;
                }
            }
            let d = self.b.ins().fsub(xs[i], xs[i]);
            sum = Some(match sum {
                None => d,
                Some(o) => self.b.ins().fadd(o, d),
            });
            i += 1;
        }
        if let Some(v) = vsum {
            let (l0, l1) = (
                self.b.ins().extractlane(v, 0u8),
                self.b.ins().extractlane(v, 1u8),
            );
            let t = self.b.ins().fadd(l0, l1);
            sum = Some(match sum {
                None => t,
                Some(o) => self.b.ins().fadd(o, t),
            });
        }
        match sum {
            Some(s) => self.b.ins().fcmp(FloatCC::Ordered, s, s),
            None => self.b.ins().iconst(types::I8, 1),
        }
    }

    /// [`FnTx::all_finite`] with no vectors: `|x| < inf` for each entry.
    fn all_finite_scalar(&mut self, xs: &[Cv]) -> Cv {
        let inf = self.b.ins().f64const(f64::INFINITY);
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
}

// ----- the latency-bound test -----

/// A loop or function with less arithmetic than this (in scalar
/// operations per iteration) has nothing worth packing.
const MIN_WORK: f64 = 8.0;

/// A loop is latency bound when the cycles its longest recurrence takes per
/// iteration are more than this fraction of its arithmetic in scalar
/// operations. Measured on an Apple M2 Pro (release, min of 5 runs): the
/// 4-vector time step and the 4 by 4 matrix-vector loop, at 0.7 and 0.43,
/// take 30% and 35% longer packed; a run of independent 4-vector operations,
/// at 0.2, takes 30% less.
const LATENCY_RATIO: f64 = 0.35;

/// The most loop-carried values whose recurrences are followed one by one.
const MAX_CARRIED: usize = 48;

/// The estimate of one iteration (or one run of a function body): the
/// longest chain of dependent operations from some values to others, in
/// cycles, and the amount of arithmetic, in scalar operations. The cost of
/// each operation is a round number of the machines this runs on (add 3 to 4
/// cycles, multiply 4, divide and square root about 12).
struct Est<'f> {
    f: &'f Function,
    /// The longest chain from the seeds of the walk to each value they
    /// reach, in cycles.
    at: HashMap<ValueId, f64>,
    work: f64,
}

/// The larger of two optional lengths.
fn longer(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl<'f> Est<'f> {
    fn new(f: &'f Function) -> Self {
        Est {
            f,
            at: HashMap::new(),
            work: 0.0,
        }
    }

    /// The chain to `v`, if the walk's seeds reach it.
    fn at(&self, v: ValueId) -> Option<f64> {
        self.at.get(&v).copied()
    }

    fn reach(&mut self, v: ValueId, d: Option<f64>) {
        if let Some(d) = d {
            self.at.insert(v, d);
        }
    }

    /// The longest chain to any of `vs`.
    fn latest(&self, vs: &[ValueId]) -> Option<f64> {
        vs.iter().fold(None, |m, &v| longer(m, self.at(v)))
    }

    /// How many entries `v` has (1 for a scalar or a value native code holds
    /// as a handle).
    fn entries(&self, v: ValueId) -> f64 {
        small_shape(self.f.ty(v)).map_or(1, |s| s.len()) as f64
    }

    fn block_of(&self, r: sky_ir::RegionId) -> Option<BlockId> {
        match self.f.region(r).blocks[..] {
            [b] => Some(b),
            _ => None,
        }
    }

    /// Walks the operations of `b` but the last (its terminator, which is
    /// returned).
    fn block(&mut self, b: BlockId) -> Option<&'f OpData> {
        let f = self.f;
        let (&last, body) = f.block(b).ops.split_last()?;
        for &o in body {
            self.op(o);
        }
        Some(f.op(last))
    }

    fn op(&mut self, id: OpId) {
        let f = self.f;
        let op = f.op(id);
        let din = self.latest(&op.operands);
        let n = op
            .operands
            .iter()
            .chain(&op.results)
            .map(|&v| self.entries(v))
            .fold(1.0, f64::max);
        let (lat, work) = match &op.kind {
            OpKind::ConstFloat { .. }
            | OpKind::ConstRational(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstStr(_)
            | OpKind::ConstNothing
            | OpKind::LiteralError { .. }
            | OpKind::Input { .. }
            | OpKind::Len => (0.0, 0.0),
            OpKind::Add | OpKind::Sub | OpKind::Neg => (3.0, n),
            OpKind::Mul => (4.0, n),
            OpKind::Div => (12.0, 3.0 * n),
            OpKind::PowConst { .. } | OpKind::Pow | OpKind::Inverse => (10.0, 3.0 * n),
            OpKind::Math(_) => (40.0, 20.0),
            OpKind::Cmp(_) | OpKind::Not => (1.0, 1.0),
            OpKind::ExpectBool { .. } | OpKind::ExpectNumber { .. } => (0.0, 0.0),
            OpKind::Index
            | OpKind::ArrayLiteral { .. }
            | OpKind::ConstArray { .. }
            | OpKind::Transpose => (1.0, 0.5 * n),
            // A sum of `k` products in order: a multiplication and `k - 1`
            // dependent additions.
            OpKind::Dot | OpKind::Norm | OpKind::MatMul | OpKind::MatMulEx { .. } => {
                let e0 = op.operands.first().map_or(1.0, |&v| self.entries(v));
                let k = if matches!(op.kind, OpKind::Dot | OpKind::Norm) {
                    e0
                } else {
                    e0.sqrt().max(1.0)
                };
                let entries = op.results.first().map_or(1.0, |&v| self.entries(v));
                (4.0 + 3.0 * (k - 1.0), 2.0 * entries * k)
            }
            OpKind::Call { .. } => (30.0, 8.0),
            OpKind::Fused => {
                let mult = self.entries(op.results[0]);
                return self.fused(op, mult);
            }
            OpKind::If => return self.branch(op, din),
            // A loop is its own region of the analysis: here it is a long
            // wait and no work.
            OpKind::While => (200.0, 0.0),
            _ => (20.0, 10.0),
        };
        self.work += work;
        for &r in &op.results {
            self.reach(r, din.map(|d| d + lat));
        }
    }

    /// `fused`: the body's chain, and its work on every entry.
    fn fused(&mut self, op: &'f OpData, mult: f64) {
        let Some(b) = op.regions.first().and_then(|&r| self.block_of(r)) else {
            return;
        };
        for (&a, &v) in self.f.block(b).args.iter().zip(&op.operands) {
            let d = self.at(v);
            self.reach(a, d);
        }
        let before = self.work;
        let out = self.block(b).and_then(|t| self.latest(&t.operands));
        self.work = before + (self.work - before) * mult;
        for &r in &op.results {
            self.reach(r, out);
        }
    }

    /// `if`: the slower arm's chain, and its work.
    fn branch(&mut self, op: &'f OpData, din: Option<f64>) {
        let before = self.work;
        let (mut most, mut out) = (0.0f64, din);
        for &r in &op.regions {
            self.work = before;
            if let Some(t) = self.block_of(r).and_then(|b| self.block(b)) {
                out = longer(out, self.latest(&t.operands));
            }
            most = most.max(self.work - before);
        }
        self.work = before + most;
        for &r in &op.results {
            self.reach(r, out);
        }
    }

    /// The test: the recurrence takes `rec` cycles, per iteration.
    fn latency_bound(&self, rec: f64) -> bool {
        self.work < MIN_WORK || rec > LATENCY_RATIO * self.work
    }
}

/// Whether one iteration of the `while` loop `op` is latency bound (see the
/// module docs): the values it carries wait on a long chain of operations
/// that ends in their own next values (or in each other's, two at a time),
/// or it has too little arithmetic to gain from packing.
pub(super) fn loop_latency_bound(f: &Function, op: &OpData) -> bool {
    let [br, ar] = op.regions[..] else {
        return true;
    };
    let probe = Est::new(f);
    let (Some(before), Some(after)) = (probe.block_of(br), probe.block_of(ar)) else {
        return true;
    };
    let carried = &f.block(before).args;
    if carried.len() > MAX_CARRIED {
        return true;
    }
    // `w[j][k]`: the longest chain from the `j`th carried value, at the top
    // of the iteration, to the `k`th at the top of the next.
    let mut w = Vec::with_capacity(carried.len());
    let mut work = 0.0;
    for (j, &seed) in carried.iter().enumerate() {
        let mut e = Est::new(f);
        e.at.insert(seed, 0.0);
        let Some(cond) = e.block(before) else {
            return true;
        };
        for (&a, &v) in f.block(after).args.iter().zip(cond.operands.iter().skip(1)) {
            let d = e.at(v);
            e.reach(a, d);
        }
        let Some(next) = e.block(after) else {
            return true;
        };
        w.push(
            next.operands
                .iter()
                .map(|&v| e.at(v))
                .collect::<Vec<Option<f64>>>(),
        );
        if j == 0 {
            work = e.work;
        }
    }
    if carried.is_empty() {
        return true;
    }
    let mut rec = 0.0f64;
    for j in 0..w.len() {
        if let Some(d) = w[j][j] {
            rec = rec.max(d);
        }
        for k in j + 1..w.len() {
            if let (Some(a), Some(b)) = (w[j][k], w[k][j]) {
                rec = rec.max((a + b) / 2.0);
            }
        }
    }
    let mut e = Est::new(f);
    e.work = work;
    e.latency_bound(rec)
}

/// Whether a run of function `f`'s body is latency bound: the same test on
/// the chain from its parameters to its results.
pub(super) fn body_latency_bound(f: &Function) -> bool {
    let mut e = Est::new(f);
    for &p in f.param_values() {
        e.at.insert(p, 0.0);
    }
    let Some(ret) = e.block(f.entry()) else {
        return true;
    };
    let rec = e.latest(&ret.operands).unwrap_or(0.0);
    e.latency_bound(rec)
}

// ----- pairs of scalar operations -----

impl<M: cranelift_module::Module> FnTx<'_, '_, M> {
    /// Two operations of one kind (`+ - * /`) that come one right after the
    /// other in a block, on scalar operands, the second not reading the
    /// first: computed as the two lanes of one vector operation, with one
    /// check. `false`: not such a pair, nothing was emitted.
    ///
    /// The window is the two operations and nothing between them, so a
    /// failure needs no reordering: the check's cold block runs the two
    /// scalar operations in program order, each with its own check, and the
    /// first that fails reports, exactly as an unpacked run would (a
    /// packed lane and its scalar operation have the same bits, so if a
    /// scalar check would fail, the vector check fails).
    pub(super) fn scalar_pair(&mut self, ia: OpId, ib: OpId) -> Res<bool> {
        if !self.slp_active() {
            return Ok(false);
        }
        let f = self.f;
        let (a, b) = (f.op(ia), f.op(ib));
        let k = match (&a.kind, &b.kind) {
            (OpKind::Add, OpKind::Add) => Bin::Add,
            (OpKind::Sub, OpKind::Sub) => Bin::Sub,
            (OpKind::Mul, OpKind::Mul) => Bin::Mul,
            (OpKind::Div, OpKind::Div) => Bin::Div,
            _ => return Ok(false),
        };
        if b.operands.contains(&a.results[0]) {
            return Ok(false);
        }
        let (mut l, mut r, mut dims) = (Vec::new(), Vec::new(), Vec::new());
        for op in [a, b] {
            let (x, y) = (self.operand(op, 0)?, self.operand(op, 1)?);
            let (Some(dx), Some(dy)) = (self.operand_dim(op, 0), self.operand_dim(op, 1)) else {
                return Ok(false);
            };
            if x.rep != Rep::F64
                || y.rep != Rep::F64
                || op.results.len() != 1
                || rep_of(f.ty(op.results[0])) != Rep::F64
            {
                return Ok(false);
            }
            let dim = match k {
                Bin::Add | Bin::Sub => (dx == dy).then_some(dx),
                Bin::Mul => dx.checked_mul(dy).ok(),
                Bin::Div => dx.checked_div(dy).ok(),
            };
            let Some(dim) = dim else {
                return Ok(false);
            };
            l.push(x.v);
            r.push(y.v);
            dims.push(dim);
        }
        let out = self.zip_arith(k, Side::Arr(&l), Side::Arr(&r), 2);
        if !(self.proven(ia) && self.proven(ib)) {
            let ok = self.all_finite(&out);
            let (cold, cont) = (self.b.create_block(), self.b.create_block());
            self.b.ins().brif(ok, cont, &[], cold, &[]);
            self.b.set_cold_block(cold);
            self.b.switch_to_block(cold);
            self.op(ia)?;
            self.op(ib)?;
            let err = self.err;
            self.b.ins().jump(err, &[]);
            self.b.switch_to_block(cont);
        }
        self.define(a, 0, Comp::F64(out[0], dims[0]))?;
        self.define(b, 0, Comp::F64(out[1], dims[1]))?;
        Ok(true)
    }
}
