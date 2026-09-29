//! Runs lowered SIR with the real reference interpreter (`sky-ir-interp`),
//! the executable specification of SIR. (Before it existed, these tests
//! used a small scalar evaluator of their own.)

use sky_ir::Module;

/// Runs `m`'s `<main>`: what it printed, and the message of the runtime
/// error that stopped it, if any. Runs on the deep stack, as `sky run` does.
pub fn run_main(m: &Module) -> (String, Result<(), String>) {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out).map_err(|e| e.message.clone());
        (String::from_utf8(out).expect("utf-8 output"), r)
    })
}

/// The two resource-limit errors compare as one: which trips first depends
/// on each interpreter's stack use per call, not on the program.
pub fn normalized(r: &Result<(), String>) -> Result<(), String> {
    r.clone().map_err(|e| {
        if e.starts_with("recursion is too deep")
            || e == "the program nests too deeply for the interpreter"
        {
            "<resource limit>".to_string()
        } else {
            e
        }
    })
}
