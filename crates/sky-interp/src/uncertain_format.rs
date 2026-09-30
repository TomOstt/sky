//! Printing `value ± uncertainty` with a sensible number of digits.
//!
//! The metrology convention (GUM, and the Particle Data Group's variant of
//! it): state the uncertainty to one or two significant digits, then round
//! the value to the same decimal place, so that every digit written is
//! meaningful and none is noise.
//!
//! * The uncertainty σ is written with **two** significant digits if its
//!   three leading digits are 100 to 354 (`0.0234` is `0.023`), and with
//!   **one** if they are 355 to 949 (`0.0567` is `0.06`; the PDG rounds
//!   950 to 999 up to `1.0` and so does this).
//! * A trailing zero of the two-digit form is dropped: an uncertainty of
//!   `0.0200` is written `0.02`, so `5.31 ± 0.02 kg` prints as it was
//!   written.
//! * The value is rounded to the last digit of the uncertainty
//!   (`9.8123 ± 0.0234` is `9.812 ± 0.023`; `1234 ± 56` is `1230 ± 60`).
//!   It may end in zeros the uncertainty says are not significant, and it is
//!   written with all the decimals the uncertainty has (`5.310 ± 0.023`), so
//!   the two line up.
//! * An exact value (`σ = 0`) is `value ± 0`.
//!
//! Numbers are written in plain decimal when that is short and in `e`
//! notation otherwise (`6.67400e-11 ± 1.5e-15`), the same range as
//! [`crate::format_number`]. The text is valid Sky: `x ± dx unit` reads back
//! (as a value with that uncertainty).

use crate::format_number;

/// `x ± sigma` as text, without a unit (the caller appends it: `5.31 ± 0.02
/// kg`). `sigma` must be finite and not negative.
pub fn plus_minus_text(x: f64, sigma: f64) -> String {
    if sigma == 0.0 || !sigma.is_finite() || !x.is_finite() {
        return format!("{} ± {}", format_number(x), format_number(sigma));
    }
    let (k, p) = sigma_digits(sigma);
    let xr = round_to_place(x, p);
    let plain = (xr == 0.0 || (1e-4..1e16).contains(&xr.abs())) && p >= -9;
    let (xt, st) = if plain {
        (plain_value(xr, p), plain_sigma(k, p))
    } else {
        (sci_value(xr, p), sci_sigma(k, p))
    };
    format!("{xt} ± {st}")
}

/// σ as `k × 10^p`: the digits `k` (one or two, not ending in zero) and the
/// decimal place `p` of the last one.
fn sigma_digits(sigma: f64) -> (u32, i32) {
    // The three leading digits decide how many to keep.
    let (lead, _) = split_sci(&format!("{sigma:.2e}"));
    let n: usize = if lead <= 354 { 2 } else { 1 };
    let (mut k, exponent) = split_sci(&format!("{sigma:.*e}", n - 1));
    let mut p = exponent - (n as i32 - 1);
    while k % 10 == 0 && k > 0 {
        k /= 10;
        p += 1;
    }
    (k, p)
}

/// The digits (as an integer, decimal point removed) and the exponent of
/// `d.ddde±x` text.
fn split_sci(text: &str) -> (u32, i32) {
    let (mantissa, exponent) = text.split_once('e').unwrap_or((text, "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    (digits.parse().unwrap_or(0), exponent.parse().unwrap_or(0))
}

/// `x` rounded to a multiple of `10^p`.
fn round_to_place(x: f64, p: i32) -> f64 {
    if p < 0 {
        let decimals = (-p) as usize;
        return format!("{x:.decimals$}").parse().unwrap_or(x) + 0.0;
    }
    let scale = 10f64.powi(p);
    if !scale.is_finite() {
        return 0.0;
    }
    (x / scale).round() * scale + 0.0
}

/// `xr`, already rounded to place `p`, in plain decimal with all the
/// decimals of `p`.
fn plain_value(xr: f64, p: i32) -> String {
    let text = if p < 0 {
        let decimals = (-p) as usize;
        format!("{xr:.decimals$}")
    } else {
        format!("{xr:.0}")
    };
    // `-0.00` is just `0.00`.
    match text.strip_prefix('-') {
        Some(rest) if rest.chars().all(|c| c == '0' || c == '.') => rest.to_string(),
        _ => text,
    }
}

/// `k × 10^p` in plain decimal.
fn plain_sigma(k: u32, p: i32) -> String {
    let digits = k.to_string();
    if p >= 0 {
        return format!("{digits}{}", "0".repeat(p as usize));
    }
    let decimals = (-p) as usize;
    if digits.len() > decimals {
        let (int, frac) = digits.split_at(digits.len() - decimals);
        format!("{int}.{frac}")
    } else {
        format!("0.{}{digits}", "0".repeat(decimals - digits.len()))
    }
}

/// `xr` in `e` notation, with as many mantissa digits as reach place `p`.
fn sci_value(xr: f64, p: i32) -> String {
    if xr == 0.0 {
        return "0".to_string();
    }
    let (_, exponent) = split_sci(&format!("{xr:e}"));
    let decimals = (exponent - p).max(0) as usize;
    format!("{xr:.decimals$e}")
}

/// `k × 10^p` in `e` notation.
fn sci_sigma(k: u32, p: i32) -> String {
    if k < 10 {
        format!("{k}e{p}")
    } else {
        format!("{}.{}e{}", k / 10, k % 10, p + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::plus_minus_text as pm;

    #[test]
    fn the_uncertainty_has_one_or_two_digits() {
        // 100..=354: two digits.
        assert_eq!(pm(9.8123, 0.0234), "9.812 ± 0.023");
        assert_eq!(pm(9.8123, 0.0125), "9.812 ± 0.013");
        assert_eq!(pm(9.8123, 0.0354), "9.812 ± 0.035");
        // 355..=949: one digit.
        assert_eq!(pm(9.8123, 0.0356), "9.81 ± 0.04");
        assert_eq!(pm(9.8123, 0.0567), "9.81 ± 0.06");
        assert_eq!(pm(9.8123, 0.0949), "9.81 ± 0.09");
        // 950..: rounds up to 1.0, which keeps one digit after the zero
        // drops.
        assert_eq!(pm(9.8123, 0.0951), "9.8 ± 0.1");
        assert_eq!(pm(9.8123, 0.0996), "9.8 ± 0.1");
    }

    #[test]
    fn a_trailing_zero_is_dropped() {
        assert_eq!(pm(5.31, 0.02), "5.31 ± 0.02");
        assert_eq!(pm(9.81, 0.05), "9.81 ± 0.05");
        assert_eq!(pm(5.3148, 0.0201), "5.31 ± 0.02");
        assert_eq!(pm(5.3148, 0.0196), "5.31 ± 0.02");
        assert_eq!(pm(2.5, 0.2), "2.5 ± 0.2");
        assert_eq!(pm(2.5, 0.20), "2.5 ± 0.2");
    }

    #[test]
    fn the_value_is_rounded_to_the_uncertainty() {
        assert_eq!(pm(5.3148, 0.0234), "5.315 ± 0.023");
        assert_eq!(pm(1234.0, 56.0), "1230 ± 60");
        assert_eq!(pm(123456.0, 1.0), "123456 ± 1");
        assert_eq!(pm(123456.0, 300.0), "123500 ± 300");
        assert_eq!(pm(0.5, 0.5), "0.5 ± 0.5");
        assert_eq!(pm(3.0, 1.5), "3.0 ± 1.5");
        assert_eq!(pm(3.0, 0.15), "3.00 ± 0.15");
    }

    #[test]
    fn signs_and_zero() {
        assert_eq!(pm(-5.31, 0.02), "-5.31 ± 0.02");
        assert_eq!(pm(0.0, 0.02), "0.00 ± 0.02");
        // Rounds to zero: no sign.
        assert_eq!(pm(-0.001, 0.02), "0.00 ± 0.02");
        assert_eq!(pm(0.004, 0.5), "0.0 ± 0.5");
    }

    #[test]
    fn an_exact_value_has_uncertainty_zero() {
        assert_eq!(pm(5.0, 0.0), "5 ± 0");
        assert_eq!(pm(0.0, 0.0), "0 ± 0");
        assert_eq!(pm(0.25, 0.0), "0.25 ± 0");
    }

    #[test]
    fn large_and_small_numbers_use_e_notation() {
        assert_eq!(pm(6.674e-11, 1.5e-15), "6.67400e-11 ± 1.5e-15");
        assert_eq!(pm(1.5e20, 3e18), "1.50e20 ± 3e18");
        assert_eq!(pm(3.2e-11, 1.5e-12), "3.20e-11 ± 1.5e-12");
        assert_eq!(pm(1e-7, 2e-9), "1.00e-7 ± 2e-9");
        // Value smaller than the last digit of the uncertainty.
        assert_eq!(pm(1e-7, 0.5e-3), "0.0000 ± 0.0005");
    }
}
