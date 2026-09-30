//! Last-use analysis: for every use of a value, does the value die there?
//!
//! Arrays are immutable values in Sky, so an executor that gives every array
//! operation a new buffer allocates and copies for nothing when the operand
//! is never read again. This analysis says where that is: an operand *dies*
//! at an operation if, on every execution path from that operation, the value
//! is not used again. An executor may then move the value out of its slot and
//! reuse its buffer for the result ([`Lifetimes::reuse`]), which is what the
//! SIR interpreter does (`v = v + dt * a` in a time-stepping loop allocates
//! nothing per step), and what `sky check --explain` reports (`MEM-001`).
//!
//! # The rules
//!
//! It is a backward liveness analysis over the structured regions.
//!
//! * **Straight-line code**: walking a block from its end, a value is live
//!   from its definition to its last use. The operand at the last use dies.
//! * **`if`**: each arm is followed by whatever follows the `if`, so a value
//!   used in the `then` arm and nowhere after the `if` dies in that arm, and
//!   is simply not touched on the `else` path. A value that is used after
//!   the `if` does not die in either arm.
//! * **`while`**: the regions run any number of times, so a value defined
//!   *outside* the loop and used inside it never dies inside it (the next
//!   iteration reads it again), and an operand of the `while` itself (an
//!   initial loop-carried value) dies there only if the loop body does not
//!   read it either. A value defined inside a region is a fresh value in
//!   each iteration: its uses are ordered like straight-line code. The
//!   loop-carried values are block arguments, and dying at a `condition` or
//!   `yield` is what hands their buffers to the next iteration.
//! * **Calls**: an argument is an operand like any other. It dies at the call
//!   if the caller does not read it afterwards, and the callee owns its
//!   parameters, so it can reuse them the same way. Nothing else needs to
//!   be known about the callee. A callee that returns its argument hands the
//!   same buffer back, because the buffer is *moved*, never shared.
//! * **The same value twice in one operation** (`a + a`): neither use is
//!   reported as the last one, because an executor that moves the first out
//!   would have nothing left to read for the second. It is conservative,
//!   never wrong.
//! * **`fused`**: the body is scalar and reads only its own arguments, so
//!   there is nothing to analyze inside it. Its operands are uses at the
//!   `fused` operation.
//!
//! The analysis is total. It never panics on a malformed function (an id out
//! of range, regions that contain themselves, nesting deeper than
//! [`MAX_REGION_DEPTH`]): it then reports that no operand dies, the always
//! correct answer.
//!
//! # Reuse in place
//!
//! [`Lifetimes::reuse`] names, for an elementwise operation (`add`, `sub`,
//! `mul` and `div` by a scalar, `neg`, `fused`), the operand whose buffer can
//! hold the result: it dies at the operation and is an array with the
//! result's shape and element representation. The check is on the types
//! only, so an operand of `Unknown` type is never named here; an executor
//! that sees the run-time shapes may reuse more.

use crate::ir::{Function, OpId, RegionId, Type, ValueId};
use crate::op::OpKind;
use crate::verify::MAX_REGION_DEPTH;

/// A set of values, dense (one bit per value of the function).
#[derive(Clone)]
struct Set {
    bits: Vec<u64>,
}

impl Set {
    fn new(values: usize) -> Set {
        Set {
            bits: vec![0; values.div_ceil(64)],
        }
    }

    fn contains(&self, v: ValueId) -> bool {
        self.bits
            .get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    }

    fn insert(&mut self, v: ValueId) {
        if let Some(w) = self.bits.get_mut(v.index() / 64) {
            *w |= 1 << (v.index() % 64);
        }
    }

    fn remove(&mut self, v: ValueId) {
        if let Some(w) = self.bits.get_mut(v.index() / 64) {
            *w &= !(1 << (v.index() % 64));
        }
    }

    fn union_with(&mut self, other: &Set) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= b;
        }
    }
}

/// One operation whose operand's buffer can hold its result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReuseSite {
    pub op: OpId,
    /// Which operand of the operation.
    pub operand: usize,
    /// Whether the operation is inside a `while` (it runs every iteration).
    pub in_loop: bool,
}

/// What [`Lifetimes::analyze`] found for one function.
#[derive(Clone, Debug, Default)]
pub struct Lifetimes {
    /// By operation index: bit `i` says operand `i` dies at the operation.
    /// An operand at index 64 or more never dies.
    dies: Vec<u64>,
    /// By operation index: `1 + i` if operand `i`'s buffer can be reused for
    /// the result, else 0.
    reuse: Vec<u8>,
    /// By operation index: the operation is inside a `while`.
    in_loop: Vec<bool>,
}

impl Lifetimes {
    /// The analysis of `f`. Operations that no block lists are left out (no
    /// operand of theirs dies).
    pub fn analyze(f: &Function) -> Lifetimes {
        // Only arrays (and values of unknown type, which may be arrays) have
        // a buffer worth moving. A function with none of them (scalar code,
        // which is most functions an ODE solver calls thousands of times)
        // gets the empty answer without walking it.
        let may_hold_array = f.values.iter().any(|v| match &v.ty {
            Type::Number(n) => n.shape.rank() > 0,
            Type::Unknown => true,
            _ => false,
        });
        if !may_hold_array {
            return Lifetimes::default();
        }
        let mut a = Analysis {
            f,
            out: Lifetimes {
                dies: vec![0; f.ops.len()],
                reuse: vec![0; f.ops.len()],
                in_loop: vec![false; f.ops.len()],
            },
            free: vec![Free::Todo; f.regions.len()],
        };
        let mut live = Set::new(f.values.len());
        match a.region(f.body, &mut live, false, 0) {
            Some(()) => a.out,
            None => Lifetimes::none(f),
        }
    }

    /// The answer that is always right: nothing dies.
    fn none(f: &Function) -> Lifetimes {
        Lifetimes {
            dies: vec![0; f.ops.len()],
            reuse: vec![0; f.ops.len()],
            in_loop: vec![false; f.ops.len()],
        }
    }

    /// Whether operand `operand` of `op` is not used again on any path from
    /// `op`, and so can be moved out of its slot.
    pub fn dies(&self, op: OpId, operand: usize) -> bool {
        operand < 64
            && self
                .dies
                .get(op.index())
                .is_some_and(|m| m & (1 << operand) != 0)
    }

    /// The operand of `op` whose buffer can hold `op`'s result, see the
    /// module documentation.
    pub fn reuse(&self, op: OpId) -> Option<usize> {
        match self.reuse.get(op.index()) {
            Some(&n) if n > 0 => Some(usize::from(n) - 1),
            _ => None,
        }
    }

    /// Whether `op` is inside a `while`.
    pub fn in_loop(&self, op: OpId) -> bool {
        self.in_loop.get(op.index()).copied().unwrap_or(false)
    }

    /// Every operation with an in-place reuse, in operation order.
    pub fn reuse_sites(&self) -> Vec<ReuseSite> {
        (0..self.reuse.len())
            .filter_map(|i| {
                let op = OpId(i as u32);
                self.reuse(op).map(|operand| ReuseSite {
                    op,
                    operand,
                    in_loop: self.in_loop(op),
                })
            })
            .collect()
    }
}

/// The free values of a region (used in it, defined outside it), computed
/// once per region.
#[derive(Clone)]
enum Free {
    Todo,
    Busy,
    Done(Set),
}

struct Analysis<'f> {
    f: &'f Function,
    out: Lifetimes,
    free: Vec<Free>,
}

impl Analysis<'_> {
    /// Walks the region's block backwards. `live` is what is live after the
    /// region, and becomes what is live before it (its free values, and what
    /// was live after and not defined in it). `None` if the function is
    /// malformed.
    fn region(&mut self, r: RegionId, live: &mut Set, in_loop: bool, depth: usize) -> Option<()> {
        if depth >= MAX_REGION_DEPTH {
            return None;
        }
        let f = self.f;
        let region = f.regions.get(r.index())?;
        let [b] = region.blocks[..] else {
            return None;
        };
        let block = f.blocks.get(b.index())?;
        for &op_id in block.ops.iter().rev() {
            let op = f.ops.get(op_id.index())?;
            for &res in &op.results {
                live.remove(res);
            }
            match op.kind {
                OpKind::If => {
                    // Each arm is followed by what follows the `if`; what is
                    // live before the `if` is the union over the arms.
                    let mut joined: Option<Set> = None;
                    for &arm in &op.regions {
                        let mut l = live.clone();
                        self.region(arm, &mut l, in_loop, depth + 1)?;
                        match &mut joined {
                            Some(j) => j.union_with(&l),
                            None => joined = Some(l),
                        }
                    }
                    if let Some(j) = joined {
                        *live = j;
                    }
                }
                OpKind::While => {
                    // Whatever the loop reads from outside stays live for the
                    // whole loop: the next iteration reads it again.
                    for &part in &op.regions {
                        let free = self.free(part, 0)?;
                        live.union_with(&free);
                    }
                    for &part in &op.regions {
                        let mut l = live.clone();
                        self.region(part, &mut l, true, depth + 1)?;
                    }
                }
                // A `fused` body is scalar and self-contained.
                _ => {}
            }
            // The operands are read before the regions run, so an operand
            // dies here if nothing after this point (the regions included)
            // reads it.
            let mut mask = 0u64;
            for (i, &v) in op.operands.iter().enumerate() {
                let once = op.operands.iter().filter(|&&w| w == v).count() == 1;
                if i < 64 && once && !live.contains(v) {
                    mask |= 1 << i;
                }
            }
            if let Some(slot) = self.out.dies.get_mut(op_id.index()) {
                *slot = mask;
            }
            if let Some(slot) = self.out.in_loop.get_mut(op_id.index()) {
                *slot = in_loop;
            }
            if let Some(i) = self.reusable(op_id, mask) {
                self.out.reuse[op_id.index()] = i as u8 + 1;
            }
            for &v in &op.operands {
                live.insert(v);
            }
        }
        for &arg in &block.args {
            live.remove(arg);
        }
        Some(())
    }

    /// The values `r` reads that it does not define (memoized). `None` on a
    /// cycle or an out-of-range id.
    fn free(&mut self, r: RegionId, depth: usize) -> Option<Set> {
        if depth >= MAX_REGION_DEPTH {
            return None;
        }
        match self.free.get(r.index())? {
            Free::Done(s) => return Some(s.clone()),
            Free::Busy => return None,
            Free::Todo => {}
        }
        self.free[r.index()] = Free::Busy;
        let f = self.f;
        let region = f.regions.get(r.index())?;
        let [b] = region.blocks[..] else {
            return None;
        };
        let block = f.blocks.get(b.index())?;
        let mut defined = Set::new(f.values.len());
        let mut free = Set::new(f.values.len());
        for &arg in &block.args {
            defined.insert(arg);
        }
        for &op_id in &block.ops {
            let op = f.ops.get(op_id.index())?;
            for &v in &op.operands {
                if !defined.contains(v) {
                    free.insert(v);
                }
            }
            if !matches!(op.kind, OpKind::Fused) {
                for &inner in &op.regions {
                    let mut s = self.free(inner, depth + 1)?;
                    for (w, d) in s.bits.iter_mut().zip(&defined.bits) {
                        *w &= !d;
                    }
                    free.union_with(&s);
                }
            }
            for &res in &op.results {
                defined.insert(res);
            }
        }
        self.free[r.index()] = Free::Done(free.clone());
        Some(free)
    }

    /// The operand whose buffer can hold the result of `op`, given which
    /// operands die (see the module documentation).
    fn reusable(&self, op_id: OpId, dies: u64) -> Option<usize> {
        let f = self.f;
        let op = f.ops.get(op_id.index())?;
        let [result] = op.results[..] else {
            return None;
        };
        let Type::Number(out) = &f.values.get(result.index())?.ty else {
            return None;
        };
        if out.shape.rank() == 0 {
            return None;
        }
        let candidates: &[usize] = match &op.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul => &[0, 1],
            // A scalar divided by an array is not an operation.
            OpKind::Div | OpKind::Neg => &[0],
            OpKind::Fused => {
                return (0..op.operands.len().min(64)).find(|&i| self.fits(op, i, dies, out))
            }
            _ => return None,
        };
        candidates
            .iter()
            .copied()
            .find(|&i| i < op.operands.len() && self.fits(op, i, dies, out))
    }

    /// Operand `i` of `op` dies at it and has the result's shape and element
    /// representation.
    fn fits(
        &self,
        op: &crate::ir::OpData,
        i: usize,
        dies: u64,
        out: &crate::ir::NumberType,
    ) -> bool {
        if dies & (1 << i) == 0 {
            return false;
        }
        match op
            .operands
            .get(i)
            .and_then(|v| self.f.values.get(v.index()))
        {
            Some(v) => match &v.ty {
                Type::Number(n) => n.shape.rank() > 0 && n.shape == out.shape && n.repr == out.repr,
                _ => false,
            },
            None => false,
        }
    }
}
