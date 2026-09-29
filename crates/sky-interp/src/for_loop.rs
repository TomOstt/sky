//! `for` loops in the tree walker. A child module of the evaluator, so it
//! shares its `Interp` and `Flow`.
//!
//! `for i in range(...)` counts. The bounds are evaluated once, left to
//! right, and must be dimensionless integers of magnitude at most 2^53
//! ([`MAX_RANGE_BOUND`]) with a step that is not zero; the index is bound
//! to a dimensionless number at the start of each iteration and keeps its
//! last value after the loop, as in Python. `for x in v` walks the first
//! axis of an array, so a vector gives its entries and a matrix its rows,
//! exactly what `v[i]` gives. Both stop on `break`, skip to the next
//! iteration on `continue` and leave the function on `return`, like `while`.
//!
//! An iterable that is not an array (a number, a Bool, a string, a
//! function) is a runtime error: the checker rejects the ones it can type,
//! but a string held in a variable has no static type.

use sky_ast::{Block, Expr, ExprKind, Ident};
use sky_diag::Span;
use sky_types::MAX_RANGE_BOUND;

use super::{Flow, Interp};
use crate::arith::{describe, R};
use crate::{format_number, RuntimeError, Value};

impl Interp<'_> {
    /// `for var in iter: body`.
    pub(super) fn for_stmt(&mut self, var: &Ident, iter: &Expr, body: &Block) -> R<Flow> {
        let name = self.text(var.span);
        if let Some(args) = self.range_args(iter) {
            let (start, stop, step) = self.range_bounds(iter.span, args)?;
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                self.assign(name, Value::scalar(i as f64));
                match self.block(body)? {
                    Flow::Break => break,
                    Flow::Normal(_) | Flow::Continue => {}
                    Flow::Return(v) => return Ok(Flow::Return(v)),
                }
                // Both are at most 2^53 in size, so this cannot overflow.
                i += step;
            }
            return Ok(Flow::Normal(None));
        }
        let v = self.eval(iter)?;
        let Value::Array { dims, data, dim } = v else {
            return Err(
                RuntimeError::new(format!("cannot iterate over {}", v.kind()), iter.span)
                    .with_label(format!("this is {}", v.kind()))
                    .with_help("loop over a vector, a matrix or `range(n)`"),
            );
        };
        let (rows, row_len) = (dims[0], data.len() / dims[0]);
        for k in 0..rows {
            let item = if dims.len() == 1 {
                Value::number(data[k], dim)
            } else {
                let row = data[k * row_len..(k + 1) * row_len].to_vec();
                Value::array(dims[1..].to_vec(), row, dim)
            };
            self.assign(name, item);
            match self.block(body)? {
                Flow::Break => break,
                Flow::Normal(_) | Flow::Continue => {}
                Flow::Return(v) => return Ok(Flow::Return(v)),
            }
        }
        Ok(Flow::Normal(None))
    }

    /// The arguments if `iter` calls the builtin `range`: callee `range`
    /// while the program has not bound the name (the checker's rule).
    fn range_args<'e>(&self, iter: &'e Expr) -> Option<&'e [Expr]> {
        let ExprKind::Call { callee, args } = &iter.kind else {
            return None;
        };
        match &callee.kind {
            ExprKind::Name(id)
                if self.text(id.span) == "range" && self.lookup("range").is_none() =>
            {
                Some(args)
            }
            _ => None,
        }
    }

    /// `(start, stop, step)` of `range(stop)`, `range(start, stop)` or
    /// `range(start, stop, step)`, each evaluated once, in order.
    fn range_bounds(&mut self, call: Span, args: &[Expr]) -> R<(i64, i64, i64)> {
        if !(1..=3).contains(&args.len()) {
            return Err(RuntimeError::new(
                format!("`range` takes 1 to 3 arguments, found {}", args.len()),
                call,
            ));
        }
        let mut n = Vec::with_capacity(3);
        for a in args {
            let v = self.eval(a)?;
            n.push(range_int(&v, a.span)?);
        }
        let (start, stop, step) = match n[..] {
            [stop] => (0, stop, 1),
            [start, stop] => (start, stop, 1),
            [start, stop, step] => (start, stop, step),
            _ => unreachable!("1 to 3 arguments"),
        };
        if step == 0 {
            let at = args.last().map_or(call, |a| a.span);
            return Err(RuntimeError::new("the step of `range` cannot be zero", at)
                .with_label("zero step")
                .with_note("the index would never move, so the loop would never end"));
        }
        Ok((start, stop, step))
    }
}

/// A bound or step of `range`: a dimensionless integer of magnitude at
/// most [`MAX_RANGE_BOUND`].
fn range_int(v: &Value, span: Span) -> R<i64> {
    let Value::Number { value, dim } = v else {
        return Err(RuntimeError::new(
            format!("`range` needs an integer, found {}", v.kind()),
            span,
        )
        .with_label(format!("this is {}", v.kind())));
    };
    if !dim.is_dimensionless() {
        return Err(RuntimeError::new(
            format!(
                "`range` needs a dimensionless integer, found {}",
                describe(dim)
            ),
            span,
        )
        .with_label(format!("this is {}", describe(dim)))
        .with_note("a count of steps has no unit"));
    }
    if value.fract() != 0.0 {
        return Err(RuntimeError::new(
            format!("`range` needs an integer, found {}", format_number(*value)),
            span,
        )
        .with_label("not an integer"));
    }
    if value.abs() > MAX_RANGE_BOUND as f64 {
        return Err(RuntimeError::new(
            format!(
                "the bound {} of `range` is too large",
                format_number(*value)
            ),
            span,
        )
        .with_label("too large")
        .with_note("a bound may be at most 2^53 in size"));
    }
    Ok(*value as i64)
}
