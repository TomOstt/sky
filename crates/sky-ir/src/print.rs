//! The SIR text format, printer half. See [`crate::parse`] for the grammar.
//!
//! ```text
//! sir edition 2026
//!
//! fn @main(%n: num[1; ()]) -> (num[L; ()]) effects(io, fail) {
//!   %0 = const.float 10000 : num[L; ()]
//!   %x = mul %0, %n : num[L; ()] @ 34..52
//!   print %x @ 60..68
//!   return %x
//! }
//! ```
//!
//! Choices, and why:
//! * **Data only.** Every token is a name, number, type or string; nothing
//!   is evaluated when the text is read (spec non-negotiable 3).
//! * **Exact.** Floats print in Rust's shortest round-trip form, or as
//!   `0x` plus 16 hex digits of the bits when they are not finite (which the
//!   verifier rejects, but a stored file must still say what it holds).
//!   Shape variables print with their id (`N#3`), because identity is the
//!   id, not the name.
//! * **Spans are byte ranges** (`@ 34..52`), the only form the parser can
//!   restore without the source file. An empty `0..0` span is omitted and
//!   read back as `0..0`.
//! * **Numbering is canonical** ([`crate::canon`]): unnamed values are
//!   `%<position>`; named values keep their name, uniquified.
//! * The printer never panics, whatever the module holds: ids out of range
//!   print as `%?N`, and nesting stops at [`MAX_REGION_DEPTH`]. Malformed
//!   modules are printed for diagnosis, not for reading back.

use std::fmt::Write as _;

use crate::canon::Layout;
use crate::display::{float_text, write_name, write_string};
use crate::ir::{BlockId, Effect, Function, Module, OpData, RegionId, ValueId};
use crate::op::OpKind;
use crate::verify::MAX_REGION_DEPTH;
use sky_diag::Span;

/// The whole module as text.
pub fn print(m: &Module) -> String {
    // Recursive over regions: see `verify`. `print_function` is called
    // per function inside the one deep-stack thread.
    sky_diag::with_deep_stack(|| print_inner(m))
}

fn print_inner(m: &Module) -> String {
    let mut out = format!("sir edition {}\n", m.edition.year());
    for f in &m.functions {
        out.push('\n');
        out.push_str(&print_function_inner(f));
    }
    out
}

/// One function as text (without the `sir edition` header).
pub fn print_function(f: &Function) -> String {
    sky_diag::with_deep_stack(|| print_function_inner(f))
}

fn print_function_inner(f: &Function) -> String {
    let mut p = Printer {
        f,
        layout: Layout::new(f),
        out: String::new(),
    };
    p.function();
    p.out
}

pub(crate) fn effect_name(e: Effect) -> &'static str {
    match e {
        Effect::Io => "io",
        Effect::Fail => "fail",
        Effect::Call => "call",
    }
}

struct Printer<'a> {
    f: &'a Function,
    layout: Layout,
    out: String,
}

impl Printer<'_> {
    fn v(&self, v: ValueId) -> String {
        format!("%{}", self.layout.value_token(v))
    }

    fn ty(&self, v: ValueId) -> String {
        match self.f.values.get(v.index()) {
            Some(d) => d.ty.to_string(),
            None => "unknown".into(),
        }
    }

    fn list(&self, vs: &[ValueId]) -> String {
        let v: Vec<String> = vs.iter().map(|&a| self.v(a)).collect();
        v.join(", ")
    }

    fn typed_list(&self, vs: &[ValueId]) -> String {
        let v: Vec<String> = vs
            .iter()
            .map(|&a| format!("{}: {}", self.v(a), self.ty(a)))
            .collect();
        v.join(", ")
    }

    fn indent(&mut self, n: usize) {
        for _ in 0..n {
            self.out.push_str("  ");
        }
    }

    fn span(&mut self, s: Span) {
        if s.start != 0 || s.end != 0 {
            let _ = write!(self.out, " @ {}..{}", s.start, s.end);
        }
    }

    fn function(&mut self) {
        let f = self.f;
        self.out.push_str("fn @");
        write_name(&mut self.out, &f.name, true);
        self.out.push('(');
        let entry = f
            .regions
            .get(f.body.index())
            .and_then(|r| r.blocks.first())
            .and_then(|b| f.blocks.get(b.index()));
        match entry {
            Some(b) => {
                let args = self.typed_list(&b.args);
                self.out.push_str(&args);
            }
            None => {
                for (i, t) in f.params.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    let _ = write!(self.out, "%?{i}: {t}");
                }
            }
        }
        self.out.push_str(") -> (");
        for (i, t) in f.results.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let _ = write!(self.out, "{t}");
        }
        self.out.push(')');
        if !f.effects.is_pure() {
            let names: Vec<&str> = f.effects.iter().map(effect_name).collect();
            let _ = write!(self.out, " effects({})", names.join(", "));
        }
        self.span(f.span);
        self.out.push_str(" {\n");
        if let Some(region) = f.regions.get(f.body.index()) {
            for (i, &b) in region.blocks.iter().enumerate() {
                if i == 0 {
                    self.block_ops(b, 1, 0);
                } else {
                    self.block(b, 1, 0);
                }
            }
        }
        self.out.push_str("}\n");
    }

    /// A block with its `^N(%a: T):` header, then its operations.
    fn block(&mut self, b: BlockId, indent: usize, depth: usize) {
        let Some(block) = self.f.blocks.get(b.index()) else {
            return;
        };
        self.indent(indent);
        let pos = self.layout.block_pos.get(b.index()).copied().unwrap_or(0);
        let _ = write!(self.out, "^{pos}");
        if !block.args.is_empty() {
            let args = self.typed_list(&block.args);
            let _ = write!(self.out, "({args})");
        }
        self.out.push_str(":\n");
        self.block_ops(b, indent + 1, depth);
    }

    fn block_ops(&mut self, b: BlockId, indent: usize, depth: usize) {
        let Some(block) = self.f.blocks.get(b.index()) else {
            return;
        };
        for &o in &block.ops {
            match self.f.ops.get(o.index()) {
                Some(op) => self.op(op, indent, depth),
                None => {
                    self.indent(indent);
                    let _ = writeln!(self.out, "<{o} out of range>");
                }
            }
        }
    }

    fn op(&mut self, op: &OpData, indent: usize, depth: usize) {
        self.indent(indent);
        if !op.results.is_empty() {
            let rs = self.list(&op.results);
            let _ = write!(self.out, "{rs} = ");
        }
        self.out.push_str(op.kind.mnemonic());
        let mut operands_done = false;
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                let _ = write!(self.out, " {}", float_text(*bits));
            }
            OpKind::ConstRational(r) => {
                let _ = write!(self.out, " {r}");
            }
            OpKind::ConstBool(b) => {
                let _ = write!(self.out, " {b}");
            }
            OpKind::ConstStr(s) => {
                self.out.push(' ');
                write_string(&mut self.out, s);
            }
            OpKind::Show { unit } => {
                self.out.push(' ');
                write_string(&mut self.out, unit);
            }
            OpKind::PowConst { exponent } => {
                let _ = write!(self.out, "[{exponent}]");
            }
            OpKind::Solve { method } => {
                let _ = write!(self.out, "[{}]", method.mnemonic());
            }
            OpKind::Call { callee } => {
                self.out.push_str(" @");
                write_name(&mut self.out, callee, true);
                let args = self.list(&op.operands);
                let _ = write!(self.out, "({args})");
                operands_done = true;
            }
            _ => {}
        }
        if !operands_done && !op.operands.is_empty() {
            let args = self.list(&op.operands);
            let _ = write!(self.out, " {args}");
        }
        if !op.results.is_empty() {
            let tys: Vec<String> = op.results.iter().map(|&r| self.ty(r)).collect();
            let _ = write!(self.out, " : {}", tys.join(", "));
        }
        for &r in &op.regions {
            self.region(r, indent, depth);
        }
        self.span(op.span);
        self.out.push('\n');
    }

    fn region(&mut self, r: RegionId, indent: usize, depth: usize) {
        let blocks = match self.f.regions.get(r.index()) {
            Some(region) if depth + 1 < MAX_REGION_DEPTH => region.blocks.clone(),
            _ => {
                self.out.push_str(" { ... }");
                return;
            }
        };
        if blocks.is_empty() {
            self.out.push_str(" {}");
            return;
        }
        self.out.push_str(" {\n");
        for b in blocks {
            self.block(b, indent + 1, depth + 1);
        }
        self.indent(indent);
        self.out.push('}');
    }
}
