//! Helpers shared by the integration tests: a builder that gives every op a
//! distinct span, a runner on the reference interpreter, and comparisons.
#![allow(dead_code)]

use sky_diag::Span;
use sky_interp::Value;
use sky_ir::{Function, FunctionBuilder, MathFn, Module, NumberType, OpKind, Type, ValueId};
use sky_ir_interp::IrRuntimeError;
use sky_units::{Dimension, Rational};

pub fn num() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

/// A function builder whose every op gets its own span `10k..10k+5`.
pub struct T {
    pub b: FunctionBuilder,
    next: u32,
}

impl T {
    pub fn new(params: Vec<Type>, results: Vec<Type>) -> T {
        T {
            b: FunctionBuilder::new("f", params, results, Span::new(0, 5)),
            next: 0,
        }
    }

    pub fn span(&mut self) -> Span {
        self.next += 1;
        Span::new(self.next * 10, self.next * 10 + 5)
    }

    pub fn params(&self) -> Vec<ValueId> {
        self.b.params()
    }

    pub fn op(&mut self, kind: OpKind, args: &[ValueId], ty: Type) -> ValueId {
        let s = self.span();
        self.b.op1(kind, args.to_vec(), ty, s)
    }

    pub fn rat(&mut self, n: i64, d: i64) -> ValueId {
        self.op(OpKind::ConstRational(Rational::new(n, d)), &[], num())
    }

    pub fn float(&mut self, x: f64) -> ValueId {
        self.op(OpKind::ConstFloat { bits: x.to_bits() }, &[], num())
    }

    pub fn un(&mut self, kind: OpKind, a: ValueId) -> ValueId {
        self.op(kind, &[a], num())
    }

    pub fn bin(&mut self, kind: OpKind, a: ValueId, b: ValueId) -> ValueId {
        self.op(kind, &[a, b], num())
    }

    pub fn math(&mut self, f: MathFn, args: &[ValueId]) -> ValueId {
        self.op(OpKind::Math(f), args, num())
    }

    pub fn pow(&mut self, a: ValueId, n: i64, d: i64) -> ValueId {
        self.un(
            OpKind::PowConst {
                exponent: Rational::new(n, d),
            },
            a,
        )
    }

    pub fn print(&mut self, vs: &[ValueId]) {
        let s = self.span();
        self.b.op(OpKind::Print, vs.to_vec(), vec![], s);
    }

    pub fn ret(mut self, vs: &[ValueId]) -> Function {
        let s = self.span();
        self.b.op(OpKind::Return, vs.to_vec(), vec![], s);
        self.b.finish()
    }
}

pub fn module(f: &Function) -> Module {
    Module {
        functions: vec![f.clone()],
        ..Module::default()
    }
}

/// Runs `f` on `args`: the result values (or error) and the printed text.
pub fn run(f: &Function, args: &[f64]) -> (Result<Vec<Value>, IrRuntimeError>, String) {
    let m = module(f);
    let args: Vec<Value> = args.iter().map(|x| Value::scalar(*x)).collect();
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::call(&m, "f", args, &mut out);
        (r, String::from_utf8_lossy(&out).into_owned())
    })
}

/// Bit-identical values (`-0.0` and `0.0` differ).
pub fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number { value: x, dim: dx }, Value::Number { value: y, dim: dy }) => {
            x.to_bits() == y.to_bits() && dx == dy
        }
        (a, b) => a == b,
    }
}

/// Identical outcomes: same values bit for bit, or the same error (message,
/// span, labels), and the same output.
pub fn same_outcome(
    a: &(Result<Vec<Value>, IrRuntimeError>, String),
    b: &(Result<Vec<Value>, IrRuntimeError>, String),
) -> bool {
    a.1 == b.1
        && match (&a.0, &b.0) {
            (Ok(x), Ok(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_value(p, q)),
            (Err(x), Err(y)) => x == y,
            _ => false,
        }
}

/// Mnemonics of the entry block's ops, in order.
pub fn kinds(f: &Function) -> Vec<&'static str> {
    f.block(f.entry())
        .ops
        .iter()
        .map(|o| f.op(*o).kind.mnemonic())
        .collect()
}

pub fn dim_len() -> Type {
    Type::scalar(Dimension::LENGTH)
}

pub fn matrix(tags: sky_types::TagSet) -> Type {
    Type::Number(NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape: sky_types::Shape::matrix(3u64, 3u64),
        tags: tags.closure(),
        repr: sky_ir::Repr::Unchosen,
    })
}
