//! The SIR data model.
//!
//! A [`Module`] holds [`Function`]s. Each function owns flat arenas of
//! values, operations, blocks and regions, addressed by small `Copy` ids, so
//! passes can hold ids across mutations without borrowing trouble, and a
//! whole function clones cheaply for differential testing.
//!
//! Structure: a function's body is a [`RegionData`]; a region holds blocks
//! (exactly one in v1, see "Regions"); a block has arguments and an ordered
//! list of operations; an operation has operands, results, and nested
//! regions (the arms of `if`, the parts of `while`).
//!
//! # SSA
//!
//! Every [`ValueId`] is defined exactly once: as an operation result or as a
//! block argument ([`ValueDef`]). Mutable Sky variables (`x += 1` in a loop)
//! become loop-carried block arguments, as in MLIR's `scf.while`.
//!
//! # Regions
//!
//! Control flow is structured. A region is a list of blocks so the format
//! can grow, but v1 regions have exactly one block, which ends in a
//! terminator (`yield`, `condition` or `return`). The verifier enforces it.

use std::fmt;

use sky_diag::Span;
use sky_types::{Shape, TagSet};
use sky_units::Dimension;

use crate::op::OpKind;

/// The language edition a module was written for (`ROADMAP.md`, idea 75):
/// the meaning of old code never changes silently. Defined in `sky-syntax`,
/// where `sky-project` also reads it from `sky.toml`.
pub use sky_syntax::Edition;

/// How a number is represented. L1 leaves it [`Repr::Unchosen`]; precision
/// passes (roadmap Phase 3) choose. [`Repr::Exact`] marks values that stay
/// rational or symbolic until approximation is needed (idea 9).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Repr {
    #[default]
    Unchosen,
    Exact,
    F32,
    F64,
}

/// The type of a numeric value: what the checker's `ValueTy` knows, plus a
/// representation. Tags are always closed (implications applied).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NumberType {
    pub dim: Dimension,
    pub shape: Shape,
    pub tags: TagSet,
    pub repr: Repr,
}

impl NumberType {
    /// A scalar of dimension `dim`, no tags, representation unchosen.
    pub fn scalar(dim: Dimension) -> Self {
        NumberType {
            dim,
            shape: Shape::scalar(),
            tags: TagSet::EMPTY,
            repr: Repr::Unchosen,
        }
    }
}

/// The type of a SIR value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Type {
    Number(NumberType),
    /// A condition. v1 conditions are scalar Bools.
    Bool,
    Str,
    /// What `print` and a function without a return value give.
    Nothing,
    /// A value the checker could not type (an unannotated parameter, a call
    /// through one). Legal in L1 so lowering never has to invent a type;
    /// passes must treat it conservatively (no rewrite may rely on it), and
    /// the reference interpreter checks such values dynamically.
    Unknown,
}

impl Type {
    pub fn scalar(dim: Dimension) -> Self {
        Type::Number(NumberType::scalar(dim))
    }

    pub fn as_number(&self) -> Option<&NumberType> {
        match self {
            Type::Number(n) => Some(n),
            _ => None,
        }
    }
}

/// A side effect an operation or function may have. Operations with none
/// are pure: passes may reorder, duplicate, remove, cache or parallelise
/// them. Everything else keeps its relative order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Effect {
    /// Writes program output (`print`).
    Io,
    /// Can fail at run time (division by zero, overflow, a non-real power).
    /// Such an operation may be removed only if its result is unused *and*
    /// the program's contract allows dropping the error; v1 passes don't.
    Fail,
    /// Calls a function. The call also has the callee's effects, which
    /// [`crate::effects`] adds to the caller's set; `Call` itself stays as
    /// the mark that a call can fail by exhausting the call depth.
    Call,
}

/// A set of [`Effect`]s. Empty means pure.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct EffectSet {
    bits: u8,
}

impl EffectSet {
    pub const PURE: EffectSet = EffectSet { bits: 0 };

    const fn bit(e: Effect) -> u8 {
        match e {
            Effect::Io => 1,
            Effect::Fail => 2,
            Effect::Call => 4,
        }
    }

    pub const fn of(e: Effect) -> EffectSet {
        EffectSet { bits: Self::bit(e) }
    }

    pub const fn with(self, e: Effect) -> EffectSet {
        EffectSet {
            bits: self.bits | Self::bit(e),
        }
    }

    pub const fn union(self, other: EffectSet) -> EffectSet {
        EffectSet {
            bits: self.bits | other.bits,
        }
    }

    /// The effects in `self` that are not in `other`.
    pub const fn difference(self, other: EffectSet) -> EffectSet {
        EffectSet {
            bits: self.bits & !other.bits,
        }
    }

    pub const fn contains(self, e: Effect) -> bool {
        self.bits & Self::bit(e) != 0
    }

    pub const fn is_pure(self) -> bool {
        self.bits == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Effect> {
        [Effect::Io, Effect::Fail, Effect::Call]
            .into_iter()
            .filter(move |e| self.contains(*e))
    }
}

macro_rules! id {
    ($(#[$m:meta])* $name:ident, $prefix:literal) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

id!(
    /// A value in a function's arena. Printed `%3`.
    ValueId,
    "%"
);
id!(
    /// An operation in a function's arena.
    OpId,
    "op"
);
id!(
    /// A block in a function's arena. Printed `^1`.
    BlockId,
    "^"
);
id!(
    /// A region in a function's arena.
    RegionId,
    "region"
);

/// Where a value is defined.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueDef {
    /// Result number `index` of `op`.
    OpResult { op: OpId, index: u32 },
    /// Argument number `index` of `block` (function parameters are the
    /// arguments of the body's entry block).
    BlockArg { block: BlockId, index: u32 },
}

#[derive(Clone, PartialEq, Debug)]
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
    /// A source name, for printing and diagnostics (`x`, `total`). Not part
    /// of the semantics.
    pub name: Option<String>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct OpData {
    pub kind: OpKind,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub regions: Vec<RegionId>,
    /// The Sky source this operation came from. A pass that merges
    /// operations joins their spans, so diagnostics and `--explain` keep
    /// pointing at the user's code.
    pub span: Span,
    /// The block the operation is in.
    pub parent: BlockId,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct BlockData {
    pub args: Vec<ValueId>,
    pub ops: Vec<OpId>,
    /// The region the block is in.
    pub parent: Option<RegionId>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct RegionData {
    pub blocks: Vec<BlockId>,
    /// The operation the region belongs to; `None` for a function body.
    pub parent: Option<OpId>,
}

/// A function: its signature, its body, and the arenas everything in it
/// lives in.
#[derive(Clone, PartialEq, Debug)]
pub struct Function {
    pub name: String,
    /// Parameter types, in order. The parameters themselves are the entry
    /// block's arguments.
    pub params: Vec<Type>,
    pub results: Vec<Type>,
    pub effects: EffectSet,
    pub body: RegionId,
    pub span: Span,
    pub values: Vec<ValueData>,
    pub ops: Vec<OpData>,
    pub blocks: Vec<BlockData>,
    pub regions: Vec<RegionData>,
}

impl Function {
    pub fn value(&self, v: ValueId) -> &ValueData {
        &self.values[v.index()]
    }

    pub fn op(&self, o: OpId) -> &OpData {
        &self.ops[o.index()]
    }

    pub fn block(&self, b: BlockId) -> &BlockData {
        &self.blocks[b.index()]
    }

    pub fn region(&self, r: RegionId) -> &RegionData {
        &self.regions[r.index()]
    }

    pub fn ty(&self, v: ValueId) -> &Type {
        &self.value(v).ty
    }

    /// The body's entry block, whose arguments are the parameters.
    pub fn entry(&self) -> BlockId {
        self.region(self.body).blocks[0]
    }

    /// The parameter values.
    pub fn param_values(&self) -> &[ValueId] {
        &self.block(self.entry()).args
    }
}

/// A compilation unit: one Sky file. The file's top-level statements are
/// the function named [`Module::MAIN`].
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Module {
    pub edition: Edition,
    pub functions: Vec<Function>,
}

impl Module {
    /// The name of the function holding a file's top-level statements. Not
    /// a valid Sky identifier, so it can't clash with a user function.
    pub const MAIN: &'static str = "<main>";

    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}
