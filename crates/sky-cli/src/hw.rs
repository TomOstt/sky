//! `sky hw`: what this machine can do, and what Sky will use it for.

use std::process::ExitCode;

/// Prints the hardware report to stdout. Takes no arguments.
pub fn hw() -> ExitCode {
    print!("{}", sky_hw::HardwareInfo::detect().report());
    ExitCode::SUCCESS
}
