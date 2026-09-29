//! SIR, Sky's semantic intermediate representation (`docs/design/ir.md`).
//!
//! SIR keeps what the checker knows about every value (dimension, symbolic
//! shape, closed matrix-property tags) so that later passes (rewrites,
//! fusion, autodiff, precision, code generation) can use it. It is typed
//! SSA with structured control flow: `if` and `while` are operations with
//! nested regions, never jumps.
//!
//! This crate holds the data model ([`ir`]), the operation set ([`op`]), and
//! a builder ([`builder`]) used by lowering and tests, plus the
//! infrastructure every pass relies on: the verifier ([`verify`]), the text
//! format ([`print`] and [`parse`], round-trip exact), and the pass manager
//! with explanations ([`pass`]). Lowering from the checked AST and the
//! reference interpreter build on these.
//!
//! # Levels
//!
//! v1 is level L1 only ("semantic"): every [`OpKind`] here is legal. Later
//! levels (strategy, kernel, backend) add operations and the verifier
//! enforces which are legal where; see the design document.

#![forbid(unsafe_code)]

pub mod builder;
mod canon;
mod display;
pub mod ir;
pub mod op;
pub mod parse;
pub mod pass;
pub mod print;
pub mod verify;

pub use builder::FunctionBuilder;
pub use canon::canonicalize;
pub use ir::{
    BlockData, BlockId, Edition, Effect, EffectSet, Function, Module, NumberType, OpData, OpId,
    RegionData, RegionId, Repr, Type, ValueData, ValueDef, ValueId,
};
pub use op::{CmpOp, MathFn, OpKind, OpProperties, SolveMethod};
pub use parse::{parse, ParseError};
pub use pass::{
    Changed, Explanation, Pass, PassCx, PassError, PassManager, PipelineReport, RuleId,
};
pub use print::{print, print_function};
pub use verify::{assignable, verify, VerifyError, MAX_REGION_DEPTH};
