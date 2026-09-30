//! The smallest end-to-end use of the JIT: compile `a * b + a` for `f64`
//! and call it. It pins down the Cranelift calls every part of the crate
//! builds on (host ISA, module, function builder, finalization, calling
//! compiled code), and fails loudly if the host cannot run JIT code.

// Calling a function pointer produced by the JIT is the one `unsafe`
// operation native code needs; it is confined to this module and to the
// modules that call compiled code, each call justified where it is made.
#![allow(unsafe_code)]

use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

/// Compiles `f(a, b) = a * b + a` and returns `f(x, y)`.
pub fn mul_add(x: f64, y: f64) -> Result<f64, String> {
    let isa = crate::host_isa()?;
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
    let mut ctx = module.make_context();
    let sig = &mut ctx.func.signature;
    sig.params.push(AbiParam::new(types::F64));
    sig.params.push(AbiParam::new(types::F64));
    sig.returns.push(AbiParam::new(types::F64));

    let mut fctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (a, bb) = (b.block_params(entry)[0], b.block_params(entry)[1]);
        let p = b.ins().fmul(a, bb);
        let r = b.ins().fadd(p, a);
        b.ins().return_(&[r]);
        b.finalize();
    }
    let id = module
        .declare_function("mul_add", Linkage::Local, &ctx.func.signature)
        .map_err(|e| e.to_string())?;
    module
        .define_function(id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(id);
    // SAFETY: `code` was compiled above with exactly this signature
    // (f64, f64) -> f64 in the platform's default calling convention, and
    // `module` (which owns the memory) outlives the call.
    let f: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(code) };
    let r = f(x, y);
    // The code memory is freed with the module.
    // SAFETY: no pointer into the module's code is used after this.
    unsafe { module.free_memory() };
    Ok(r)
}

#[cfg(test)]
mod tests {
    #[test]
    fn jit_code_runs_on_the_host() {
        assert_eq!(super::mul_add(3.0, 4.0), Ok(15.0));
    }
}
