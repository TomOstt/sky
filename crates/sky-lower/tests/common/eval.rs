//! A small evaluator for lowered SIR, used only as a test oracle: it runs a
//! module's `<main>` the way `docs/design/ir.md` and `sky_ir::op` define the
//! operations, so the tests can check that lowering preserves the meaning of
//! programs by comparing with the tree-walking interpreter. (The real
//! reference interpreter lives in another crate; this one is deliberately
//! tiny and scalar-only.)

use std::collections::HashMap;

use sky_interp::Value;
use sky_ir::{BlockId, CmpOp, Function, Module, OpKind, RegionId, Type, ValueId};
use sky_units::Dimension;

#[derive(Clone, Debug, PartialEq)]
pub enum V {
    Num(f64, Dimension),
    Bool(bool),
    Str(String),
    Nothing,
}

impl V {
    fn to_value(&self) -> Value {
        match self {
            V::Num(x, d) => Value::number(*x, *d),
            V::Bool(b) => Value::Bool(*b),
            V::Str(s) => Value::Str(s.clone()),
            V::Nothing => Value::None,
        }
    }
}

enum Flow {
    Yield(Vec<V>),
    Condition(bool, Vec<V>),
    Return(Vec<V>),
}

pub struct Eval<'m> {
    m: &'m Module,
    pub out: String,
    steps: u64,
    depth: usize,
}

type R<T> = Result<T, String>;

/// Runs `<main>` with no arguments; returns what it printed and whether it
/// finished (`Err` is a run-time error, as in the interpreter).
pub fn run_main(m: &Module) -> (String, Result<(), String>) {
    let mut ev = Eval {
        m,
        out: String::new(),
        steps: 0,
        depth: 0,
    };
    let main = m.function(Module::MAIN).expect("<main>");
    let r = ev.call(main, Vec::new()).map(|_| ());
    (ev.out, r)
}

const STEP_LIMIT: u64 = 200_000;
const DEPTH_LIMIT: usize = 150;

fn finite(x: f64) -> R<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err("overflow".into())
    }
}

impl Eval<'_> {
    fn call(&mut self, f: &Function, args: Vec<V>) -> R<Vec<V>> {
        if self.depth >= DEPTH_LIMIT {
            return Err("recursion too deep".into());
        }
        if args.len() != f.params.len() {
            return Err(format!("{} takes {} arguments", f.name, f.params.len()));
        }
        self.depth += 1;
        let mut env: HashMap<ValueId, V> = HashMap::new();
        let entry = f.entry();
        for (a, v) in f.block(entry).args.iter().zip(args) {
            env.insert(*a, v);
        }
        let r = self.block(f, entry, &mut env);
        self.depth -= 1;
        match r? {
            Flow::Return(v) => Ok(v),
            _ => Err("a function body must end in return".into()),
        }
    }

    fn get(env: &HashMap<ValueId, V>, v: ValueId) -> R<V> {
        env.get(&v)
            .cloned()
            .ok_or_else(|| format!("{v} is not defined"))
    }

    fn region(
        &mut self,
        f: &Function,
        r: RegionId,
        args: Vec<V>,
        env: &mut HashMap<ValueId, V>,
    ) -> R<Flow> {
        let b: BlockId = f.region(r).blocks[0];
        for (a, v) in f.block(b).args.iter().zip(args) {
            env.insert(*a, v);
        }
        self.block(f, b, env)
    }

    fn block(&mut self, f: &Function, b: BlockId, env: &mut HashMap<ValueId, V>) -> R<Flow> {
        for &o in &f.block(b).ops {
            self.steps += 1;
            if self.steps > STEP_LIMIT {
                return Err("step limit".into());
            }
            let op = f.op(o);
            let args: Vec<V> = op
                .operands
                .iter()
                .map(|v| Self::get(env, *v))
                .collect::<R<_>>()?;
            let bind = |env: &mut HashMap<ValueId, V>, vals: Vec<V>| {
                for (r, v) in op.results.iter().zip(vals) {
                    env.insert(*r, v);
                }
            };
            match &op.kind {
                OpKind::Yield => return Ok(Flow::Yield(args)),
                OpKind::Return => return Ok(Flow::Return(args)),
                OpKind::Condition => {
                    let V::Bool(c) = args[0] else {
                        return Err("condition is not a Bool".into());
                    };
                    return Ok(Flow::Condition(c, args[1..].to_vec()));
                }
                OpKind::If => {
                    let V::Bool(c) = args[0] else {
                        return Err("if needs a Bool".into());
                    };
                    let arm = op.regions[usize::from(!c)];
                    match self.region(f, arm, Vec::new(), env)? {
                        Flow::Yield(vals) => bind(env, vals),
                        _ => return Err("an if arm must yield".into()),
                    }
                }
                OpKind::While => {
                    let mut vals = args;
                    loop {
                        self.steps += 1;
                        if self.steps > STEP_LIMIT {
                            return Err("step limit".into());
                        }
                        let (go, fwd) = match self.region(f, op.regions[0], vals, env)? {
                            Flow::Condition(c, fwd) => (c, fwd),
                            _ => return Err("before must end in condition".into()),
                        };
                        if !go {
                            bind(env, fwd);
                            break;
                        }
                        match self.region(f, op.regions[1], fwd, env)? {
                            Flow::Yield(next) => vals = next,
                            _ => return Err("after must yield".into()),
                        }
                    }
                }
                OpKind::Call { callee } => {
                    let g = self
                        .m
                        .function(callee)
                        .ok_or_else(|| format!("no function {callee}"))?;
                    let r = self.call(g, args)?;
                    bind(env, r);
                }
                OpKind::Print => {
                    let line: Vec<String> =
                        args.iter().map(|v| v.to_value().print_text()).collect();
                    self.out.push_str(&line.join(" "));
                    self.out.push('\n');
                }
                kind => {
                    let ty = op.results.first().map(|r| f.ty(*r).clone());
                    let v = Self::pure_op(kind, &args, ty.as_ref())?;
                    bind(env, vec![v]);
                }
            }
        }
        Err("block without a terminator".into())
    }

    fn pure_op(kind: &OpKind, a: &[V], ty: Option<&Type>) -> R<V> {
        use V::{Bool, Num, Str};
        Ok(match kind {
            OpKind::ConstFloat { bits } => {
                let dim = ty
                    .and_then(|t| t.as_number())
                    .map_or(Dimension::DIMENSIONLESS, |n| n.dim);
                Num(f64::from_bits(*bits), dim)
            }
            OpKind::ConstBool(b) => Bool(*b),
            OpKind::ConstStr(s) => Str(s.clone()),
            OpKind::ConstNothing => V::Nothing,
            OpKind::Add | OpKind::Sub => {
                let (Num(x, dx), Num(y, dy)) = (&a[0], &a[1]) else {
                    return Err("add needs numbers".into());
                };
                if dx != dy {
                    return Err("mismatched dimensions".into());
                }
                let r = if *kind == OpKind::Add { x + y } else { x - y };
                Num(finite(r)?, *dx)
            }
            OpKind::Mul | OpKind::Div => {
                let (Num(x, dx), Num(y, dy)) = (&a[0], &a[1]) else {
                    return Err("mul needs numbers".into());
                };
                if *kind == OpKind::Mul {
                    let d = dx
                        .checked_mul(*dy)
                        .map_err(|_| "dim overflow".to_string())?;
                    Num(finite(x * y)?, d)
                } else {
                    let d = dx
                        .checked_div(*dy)
                        .map_err(|_| "dim overflow".to_string())?;
                    if *y == 0.0 {
                        return Err("division by zero".into());
                    }
                    Num(finite(x / y)?, d)
                }
            }
            OpKind::Neg => {
                let Num(x, d) = &a[0] else {
                    return Err("neg needs a number".into());
                };
                Num(-x, *d)
            }
            OpKind::Inverse => {
                let Num(x, d) = &a[0] else {
                    return Err("inverse needs a number".into());
                };
                if *x == 0.0 {
                    return Err("division by zero".into());
                }
                Num(
                    finite(1.0 / x)?,
                    d.inverse().map_err(|_| "dim overflow".to_string())?,
                )
            }
            OpKind::PowConst { exponent } => {
                let Num(b, d) = &a[0] else {
                    return Err("pow needs a number".into());
                };
                let x = exponent.numer() as f64 / exponent.denom() as f64;
                let dim = if d.is_dimensionless() {
                    Dimension::DIMENSIONLESS
                } else {
                    d.pow_rat(*exponent)
                        .map_err(|_| "dim overflow".to_string())?
                };
                Num(power(*b, x)?, dim)
            }
            OpKind::Pow => {
                let (Num(b, d), Num(x, dx)) = (&a[0], &a[1]) else {
                    return Err("pow needs numbers".into());
                };
                if !dx.is_dimensionless() || !d.is_dimensionless() {
                    return Err("a dynamic exponent needs dimensionless operands".into());
                }
                Num(power(*b, *x)?, Dimension::DIMENSIONLESS)
            }
            OpKind::Cmp(c) => {
                let r = match (&a[0], &a[1]) {
                    (Num(x, dx), Num(y, dy)) => {
                        if dx != dy {
                            return Err("cannot compare".into());
                        }
                        match c {
                            CmpOp::Eq => x == y,
                            CmpOp::Ne => x != y,
                            CmpOp::Lt => x < y,
                            CmpOp::Le => x <= y,
                            CmpOp::Gt => x > y,
                            CmpOp::Ge => x >= y,
                        }
                    }
                    (Bool(x), Bool(y)) if matches!(c, CmpOp::Eq | CmpOp::Ne) => {
                        (x == y) == (*c == CmpOp::Eq)
                    }
                    (Str(x), Str(y)) if matches!(c, CmpOp::Eq | CmpOp::Ne) => {
                        (x == y) == (*c == CmpOp::Eq)
                    }
                    _ => return Err("cannot compare".into()),
                };
                Bool(r)
            }
            OpKind::Not => {
                let Bool(b) = a[0] else {
                    return Err("not needs a Bool".into());
                };
                Bool(!b)
            }
            other => return Err(format!("unsupported {}", other.mnemonic())),
        })
    }
}

/// `b ^ x` as the interpreter computes it.
fn power(b: f64, x: f64) -> R<f64> {
    if b == 0.0 && x < 0.0 {
        return Err("division by zero".into());
    }
    if b < 0.0 && x.fract() != 0.0 {
        return Err("not a real number".into());
    }
    let v = if x.fract() == 0.0 && x.abs() <= 64.0 {
        b.powi(x as i32)
    } else {
        b.powf(x)
    };
    finite(v)
}
