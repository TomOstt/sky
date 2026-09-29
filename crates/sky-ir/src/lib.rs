//! SIR, Sky's semantic intermediate representation (`docs/design/ir.md`).
//!
//! SIR keeps what the checker knows about every value (dimension, symbolic
//! shape, closed matrix-property tags) so that later passes (rewrites,
//! fusion, autodiff, precision, code generation) can use it. It is typed
//! SSA with structured control flow: `if` and `while` are operations with
//! nested regions, never jumps.
//!
//! This crate holds the data model ([`ir`]), the operation set ([`op`]), and
//! a builder ([`builder`]) used by lowering and tests. The verifier, the
//! text format, the pass manager, lowering from the checked AST, and the
//! reference interpreter build on these types.
//!
//! # Levels
//!
//! v1 is level L1 only ("semantic"): every [`OpKind`] here is legal. Later
//! levels (strategy, kernel, backend) add operations and the verifier
//! enforces which are legal where; see the design document.

#![forbid(unsafe_code)]

pub mod builder;
pub mod ir;
pub mod op;

pub use builder::FunctionBuilder;
pub use ir::{
    BlockData, BlockId, Edition, Effect, EffectSet, Function, Module, NumberType, OpData, OpId,
    RegionData, RegionId, Repr, Type, ValueData, ValueDef, ValueId,
};
pub use op::{CmpOp, MathFn, OpKind, OpProperties, SolveMethod};
