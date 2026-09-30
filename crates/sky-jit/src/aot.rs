//! Ahead-of-time compilation: `sky build`.
//!
//! [`compile_object`] translates a SIR module with the very code the JIT
//! uses ([`crate::module::translate_module`], generic over the Cranelift
//! module type) into an object file for a target machine (Mach-O on macOS,
//! ELF on Linux). The object holds
//!
//! * the machine code of every SIR function,
//! * an *image* in its data section: everything the runtime needs that the
//!   JIT keeps in memory (the SIR module as text, the dimension and site
//!   tables the code names by index, and the caller's own bytes, which
//!   `sky build` uses for the source text), and
//! * a C `main` that calls `sky_rt_start(image, image length, <main>)`.
//!
//! The runtime functions the code calls (`sky_rt_*`, see [`crate::runtime`])
//! are imports of the object: the linker resolves them against the runtime
//! library (`crates/sky-rt`, a static library that re-exports this crate's
//! `#[no_mangle]` functions and provides `sky_rt_start`).
//!
//! # The image and deoptimization
//!
//! The runtime must be able to say *why* a run failed, exactly as the JIT
//! does, and it does that by re-running the failing operation from the SIR
//! module (see `runtime.rs`); and the native engine falls back to the SIR
//! interpreter when a value of unknown type is not what a declaration says
//! (deoptimization). Both need the SIR at run time, so the executable carries
//! it, in the SIR text format, and the runtime library carries the SIR
//! interpreter. An executable therefore behaves exactly like `sky run
//! --engine=native`, deoptimization included.
//!
//! The module that is compiled is the one the image's text parses to (the
//! module is printed and parsed back before translation), so the operation
//! indexes native code names are the ones the runtime sees.
//!
//! # Portability
//!
//! Code is generated for the target's baseline CPU, not the host's
//! extensions: an executable built for `x86_64` runs on any x86-64 machine
//! (rounding functions call the C math library instead of `roundsd`; the
//! results are identical). The code is position independent.

#![allow(unsafe_code)]

use std::io::Write;
use std::sync::Arc;

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{default_libcall_names, DataDescription, Linkage, Module as ClModule};
use cranelift_object::{ObjectBuilder, ObjectModule};
use sky_ir::Module as SirModule;
use sky_units::{Dimension, Rational};

use crate::module::{internal, precheck, run_entry, translate_module};
use crate::runtime::{Program, Site};
use crate::{CompileError, RunFailure};

/// The symbol of the runtime's entry point, which the generated `main`
/// calls: `extern "C" fn(image: *const u8, image_len: usize, entry: usize) ->
/// i32` (the process's exit status).
pub const START_SYMBOL: &str = "sky_rt_start";

/// The machine an object file is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetInfo {
    /// The full target triple, `aarch64-apple-darwin`.
    pub triple: String,
    /// `aarch64`, `x86_64`, ...
    pub arch: String,
    /// `darwin` (macOS), `linux`, ...
    pub os: String,
}

/// An object file and the machine it is for.
pub struct Object {
    pub bytes: Vec<u8>,
    pub target: TargetInfo,
}

fn isa_for(triple: Option<&str>) -> Result<OwnedTargetIsa, String> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    // Executables are position independent by default on both platforms.
    flags.set("is_pic", "true").map_err(|e| e.to_string())?;
    // Same as the JIT: functions return status and all results (x86-64).
    flags
        .set("enable_multi_ret_implicit_sret", "true")
        .map_err(|e| e.to_string())?;
    let flags = settings::Flags::new(flags);
    // The host's triple, but no host CPU features (see the module docs).
    let builder = match triple {
        Some(t) => {
            // `lookup_by_name` panics on a malformed triple; parse it here.
            let triple: target_lexicon::Triple = t
                .parse()
                .map_err(|e| format!("unknown target `{t}` ({e})"))?;
            cranelift_codegen::isa::lookup(triple).map_err(|e| e.to_string())?
        }
        None => {
            let host = cranelift_native::builder().map_err(|e| e.to_string())?;
            cranelift_codegen::isa::lookup(host.triple().clone()).map_err(|e| e.to_string())?
        }
    };
    builder.finish(flags).map_err(|e| e.to_string())
}

/// The triple of the machine this process runs on.
pub fn host_target() -> Result<TargetInfo, String> {
    let isa = isa_for(None)?;
    Ok(target_info(&isa))
}

fn target_info(isa: &OwnedTargetIsa) -> TargetInfo {
    let t = isa.triple();
    TargetInfo {
        triple: t.to_string(),
        arch: t.architecture.to_string(),
        os: t.operating_system.to_string(),
    }
}

/// Compiles `module` (which must verify, with a `<main>`) into an object file
/// for `triple` (`None`: this machine). `extra` is stored in the image for
/// the runtime (see [`Image::extra`]).
pub fn compile_object(
    module: &SirModule,
    triple: Option<&str>,
    extra: &[u8],
) -> Result<Object, CompileError> {
    // The text is the module the runtime will see: compile that.
    let text = sky_ir::print(module);
    let module = sky_ir::parse(&text).map_err(|e| CompileError {
        message: format!("the module does not survive printing and parsing ({e})"),
        span: None,
    })?;
    precheck(&module)?;
    let isa = isa_for(triple).map_err(|e| CompileError {
        message: format!("no code generator for the target ({e})"),
        span: None,
    })?;
    let target = target_info(&isa);
    let call_conv = isa.default_call_conv();
    let pointer = isa.pointer_type();
    let builder = ObjectBuilder::new(isa, "sky", default_libcall_names())
        .map_err(|e| internal("the object writer", &e))?;
    let mut obj = ObjectModule::new(builder);

    let done = translate_module(
        &module,
        &mut obj,
        call_conv,
        false,
        false,
        crate::Slp::from_env(),
        // An executable runs on one thread: no windows of calls.
        None,
    )?;
    let main_index = module
        .functions
        .iter()
        .position(|f| f.name == SirModule::MAIN)
        .unwrap_or(0);
    let entry = done.sh.fn_ids[main_index];
    let image = Image::encode(&text, &done.sh.dims, &done.sh.sites, extra);

    // The image, as read-only data.
    let data = obj
        .declare_data("sky_image", Linkage::Local, false, false)
        .map_err(|e| internal("the image", &e))?;
    let mut desc = DataDescription::new();
    desc.define(image.clone().into_boxed_slice());
    desc.set_align(16);
    obj.define_data(data, &desc)
        .map_err(|e| internal("the image", &e))?;

    // int main(int argc, char **argv) { return sky_rt_start(image, len, <main>); }
    let mut start_sig = Signature::new(call_conv);
    start_sig.params.push(AbiParam::new(pointer));
    start_sig.params.push(AbiParam::new(pointer));
    start_sig.params.push(AbiParam::new(pointer));
    start_sig.returns.push(AbiParam::new(types::I32));
    let start = obj
        .declare_function(START_SYMBOL, Linkage::Import, &start_sig)
        .map_err(|e| internal(START_SYMBOL, &e))?;
    let mut main_sig = Signature::new(call_conv);
    main_sig.params.push(AbiParam::new(types::I32));
    main_sig.params.push(AbiParam::new(pointer));
    main_sig.returns.push(AbiParam::new(types::I32));
    let main_id = obj
        .declare_function("main", Linkage::Export, &main_sig)
        .map_err(|e| internal("main", &e))?;
    let mut ctx = obj.make_context();
    ctx.func.signature = main_sig;
    ctx.func.name = UserFuncName::user(0, main_id.as_u32());
    {
        let mut fctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let block = b.create_block();
        b.append_block_params_for_function_params(block);
        b.switch_to_block(block);
        let gv = obj.declare_data_in_func(data, b.func);
        let image_ptr = b.ins().global_value(pointer, gv);
        let len = b.ins().iconst(pointer, image.len() as i64);
        let entry_ref = obj.declare_func_in_func(entry, b.func);
        let entry_addr = b.ins().func_addr(pointer, entry_ref);
        let start_ref = obj.declare_func_in_func(start, b.func);
        let call = b.ins().call(start_ref, &[image_ptr, len, entry_addr]);
        let status = b.inst_results(call)[0];
        b.ins().return_(&[status]);
        b.seal_all_blocks();
        b.finalize();
    }
    obj.define_function(main_id, &mut ctx)
        .map_err(|e| internal("main", &e))?;
    let bytes = obj
        .finish()
        .emit()
        .map_err(|e| internal("the object file", &e))?;
    Ok(Object { bytes, target })
}

/// The header of an image: the format's name and version. The version is
/// bumped when the layout changes, so a runtime library from another build
/// says so instead of misreading.
const MAGIC: &[u8; 8] = b"SKYIMG\x00\x01";

/// What an executable carries, decoded (see the module docs).
pub struct Image {
    program: Arc<Program>,
    /// The caller's bytes given to [`compile_object`].
    pub extra: Vec<u8>,
}

fn put_u64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, n: i64) {
    out.extend_from_slice(&n.to_le_bytes());
}

impl Image {
    fn encode(sir: &str, dims: &[Dimension], sites: &[Site], extra: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        put_u64(&mut out, sir.len() as u64);
        out.extend_from_slice(sir.as_bytes());
        put_u64(&mut out, dims.len() as u64);
        for d in dims {
            for e in d.exponents() {
                put_i64(&mut out, e.numer());
                put_i64(&mut out, e.denom());
            }
        }
        put_u64(&mut out, sites.len() as u64);
        for s in sites {
            out.extend_from_slice(&s.func.to_le_bytes());
            out.extend_from_slice(&s.op.to_le_bytes());
        }
        put_u64(&mut out, extra.len() as u64);
        out.extend_from_slice(extra);
        out
    }

    /// Decodes the bytes of an executable's image. Never panics on any input.
    pub fn decode(bytes: &[u8]) -> Result<Image, String> {
        let bad = || "the program image is damaged or from another version of Sky".to_string();
        let mut r = Reader { bytes, at: 0 };
        if r.take(MAGIC.len()).ok_or_else(bad)? != MAGIC {
            return Err(bad());
        }
        let n = r.u64().ok_or_else(bad)?;
        let sir = std::str::from_utf8(r.take_n(n).ok_or_else(bad)?).map_err(|_| bad())?;
        let module = sky_ir::parse(sir).map_err(|e| format!("the program image: {e}"))?;
        let n = r.u64().ok_or_else(bad)?;
        let mut dims = Vec::new();
        for _ in 0..n {
            let mut exps = [Rational::ZERO; 7];
            for e in &mut exps {
                let (num, den) = (r.i64().ok_or_else(bad)?, r.i64().ok_or_else(bad)?);
                *e = Rational::checked_new(num, den).map_err(|_| bad())?;
            }
            dims.push(Dimension::from_exponents(exps));
        }
        let n = r.u64().ok_or_else(bad)?;
        let mut sites = Vec::new();
        for _ in 0..n {
            let func =
                u32::from_le_bytes(r.take(4).ok_or_else(bad)?.try_into().map_err(|_| bad())?);
            let op = u32::from_le_bytes(r.take(4).ok_or_else(bad)?.try_into().map_err(|_| bad())?);
            sites.push(Site { func, op });
        }
        let n = r.u64().ok_or_else(bad)?;
        let extra = r.take_n(n).ok_or_else(bad)?.to_vec();
        Ok(Image {
            program: Arc::new(Program {
                module,
                dims,
                sites,
                // `sky build` compiles reproducible numerics only.
                relaxed: false,
                // An executable runs on one thread: what it was built to do
                // never depends on the machine it runs on.
                par: sky_ir::parallel::Config::sequential(),
            }),
            extra,
        })
    }

    /// The SIR module the executable was compiled from (the deoptimization
    /// fallback runs it on the SIR interpreter).
    pub fn module(&self) -> &SirModule {
        &self.program.module
    }

    /// Runs the compiled `<main>` at `entry`, writing `print` output to
    /// `out`, exactly as [`crate::Compiled::run`] does. Run it under
    /// `sky_diag::with_deep_stack`.
    ///
    /// # Safety
    ///
    /// `entry` must be the address of the `<main>` that [`compile_object`]
    /// compiled from this image's module, as the executable's `main` passes
    /// it to `sky_rt_start`.
    pub unsafe fn run(&self, entry: usize, out: &mut dyn Write) -> Result<(), RunFailure> {
        // SAFETY: the caller promises what `run_entry` needs.
        unsafe { run_entry(self.program.clone(), entry, out) }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(s)
    }

    fn take_n(&mut self, n: u64) -> Option<&'a [u8]> {
        self.take(usize::try_from(n).ok()?)
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_MAIN: &str = "sir edition 2026\n\nfn @\"<main>\"() -> () {\n  return\n}\n";

    #[test]
    fn an_image_round_trips() {
        let dims = [
            Dimension::from_ints([1, 0, 0, 0, 0, 0, 0]),
            Dimension::from_ints([2, 1, -2, 0, 0, 0, 0]),
        ];
        let sites = [
            Site { func: 0, op: 3 },
            Site {
                func: 1,
                op: u32::MAX,
            },
        ];
        let bytes = Image::encode(EMPTY_MAIN, &dims, &sites, b"extra");
        let image = Image::decode(&bytes).expect("decodes");
        assert_eq!(image.extra, b"extra");
        assert_eq!(image.program.dims, dims);
        assert_eq!(image.program.sites.len(), 2);
        assert_eq!(image.program.sites[1].op, u32::MAX);
        assert_eq!(
            sky_ir::print(image.module()),
            sky_ir::print(&sky_ir::parse(EMPTY_MAIN).unwrap())
        );
        // Every truncation is refused, never a panic.
        for n in 0..bytes.len() {
            assert!(Image::decode(&bytes[..n]).is_err(), "{n}");
        }
        // A runtime library from another version refuses the image.
        let mut other = bytes.clone();
        other[7] ^= 1;
        assert!(Image::decode(&other).is_err());
    }

    #[test]
    fn an_unknown_target_is_an_error_not_a_panic() {
        let m = sky_ir::parse(EMPTY_MAIN).unwrap();
        let e = compile_object(&m, Some("pdp11-none"), b"")
            .err()
            .expect("refused");
        assert!(e.message.contains("pdp11"), "{}", e.message);
    }
}
