//! Measuring candidates: warmup, interleaved repeats, the median, a budget.

use std::fmt;
use std::hint::black_box;
use std::time::{Duration, Instant};

use crate::db::{valid_token, Record};
use crate::params::{Params, MAX_CANDIDATES};

/// How much work one call of the benchmark does, so that a time can be shown
/// as a rate ("3.1 GB/s").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Unknown: times are shown as milliseconds.
    Unknown,
    /// Bytes read and written per call.
    Bytes(u64),
    /// Floating-point operations per call.
    Flops(u64),
}

/// Something worth tuning: one kernel at one problem size, a few candidate
/// parameter sets, and a way to run it.
///
/// The benchmark runs the kernel once with the given parameters and returns
/// a value that depends on the result (a checksum, say), which the tuner
/// hides from the optimizer so the work cannot be removed. It must not
/// depend on the parameters for its *answer*: tuning may change speed,
/// never results (a kernel whose result changed with its block size would
/// not be a tunable, it would be a different function).
pub struct Tunable {
    /// The kernel's name (see [`crate::valid_token`]).
    pub kernel: String,
    /// The problem-size class the measurement applies to, such as `2^18` or
    /// `512x512` (see [`size_class`]).
    pub size_class: String,
    /// The candidates, at most 64. **The first is the default**: it is kept
    /// unless another is clearly faster.
    pub candidates: Vec<Params>,
    /// The work per call, for rates.
    pub work: Work,
    bench: Box<dyn Fn(&Params) -> u64>,
}

impl Tunable {
    /// Builds a tunable. Errors: a kernel name or size class that is not a
    /// valid token, no candidates, more than 64, or a repeated candidate.
    pub fn new(
        kernel: &str,
        size_class: &str,
        candidates: Vec<Params>,
        work: Work,
        bench: impl Fn(&Params) -> u64 + 'static,
    ) -> Result<Tunable, String> {
        if !valid_token(kernel) {
            return Err(format!("invalid kernel name {kernel:?}"));
        }
        if !valid_token(size_class) {
            return Err(format!("invalid size class {size_class:?}"));
        }
        if candidates.is_empty() {
            return Err("no candidates".to_string());
        }
        if candidates.len() > MAX_CANDIDATES {
            return Err(format!("more than {MAX_CANDIDATES} candidates"));
        }
        for (i, c) in candidates.iter().enumerate() {
            if candidates[..i].contains(c) {
                return Err(format!("candidate {c} is repeated"));
            }
        }
        Ok(Tunable {
            kernel: kernel.to_string(),
            size_class: size_class.to_string(),
            candidates,
            work,
            bench: Box::new(bench),
        })
    }

    /// Runs the benchmark once with `params` and returns its value. The
    /// tuner's own timing loop calls this; tests use it to check that every
    /// candidate gives the same answer.
    pub fn run(&self, params: &Params) -> u64 {
        (self.bench)(params)
    }
}

/// A size class for a length: `2^18` for 262144 (the exponent of the largest
/// power of two not above `n`), so that nearby sizes share a result.
pub fn size_class(n: usize) -> String {
    let n = n.max(1);
    format!("2^{}", usize::BITS - 1 - n.leading_zeros())
}

/// How to measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// The time budget for measuring one tunable (warmup included). Default
    /// 500 ms.
    pub budget: Duration,
    /// Every candidate is measured at least this many times even if the
    /// budget runs out (a median of fewer would be noise). Default 5.
    pub min_repeats: usize,
    /// ... and at most this many. Default 31.
    pub max_repeats: usize,
    /// One sample times enough back-to-back calls to last about this long,
    /// so a fast kernel is not lost in timer resolution. Default 200 us.
    pub min_sample: Duration,
    /// A candidate must beat the default's median by this fraction to
    /// replace it (0.03 is 3%), so noise cannot flip the choice. Default
    /// 0.03.
    pub min_gain: f64,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            budget: Duration::from_millis(500),
            min_repeats: 5,
            max_repeats: 31,
            min_sample: Duration::from_micros(200),
            min_gain: 0.03,
        }
    }
}

/// The measurements of one candidate.
#[derive(Clone, Debug)]
pub struct Measurement {
    /// The candidate.
    pub params: Params,
    /// Seconds per call, one entry per repeat.
    pub samples: Vec<f64>,
    /// The median of `samples`.
    pub median: f64,
}

/// The result of measuring one tunable.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The kernel.
    pub kernel: String,
    /// The size class.
    pub size_class: String,
    /// The work per call.
    pub work: Work,
    /// One entry per candidate, in the tunable's order.
    pub measurements: Vec<Measurement>,
    /// The index of the chosen candidate.
    pub winner: usize,
    /// How long measuring took.
    pub elapsed: Duration,
}

impl Outcome {
    /// The chosen parameters.
    pub fn params(&self) -> &Params {
        &self.measurements[self.winner].params
    }

    /// The fastest candidate by median, whether or not it was chosen.
    pub fn fastest(&self) -> usize {
        (0..self.measurements.len())
            .min_by(|&a, &b| {
                self.measurements[a]
                    .median
                    .total_cmp(&self.measurements[b].median)
            })
            .unwrap_or(0)
    }

    /// How many times faster the chosen candidate is than the default.
    pub fn speedup(&self) -> f64 {
        self.measurements[0].median / self.measurements[self.winner].median
    }

    /// A candidate's speed as text: a rate when the work is known, else
    /// milliseconds.
    pub fn rate(&self, index: usize) -> String {
        let secs = self.measurements[index].median;
        match self.work {
            Work::Bytes(b) => format!("{:.2} GB/s", b as f64 / secs / 1e9),
            Work::Flops(f) => format!("{:.2} GFLOP/s", f as f64 / secs / 1e9),
            Work::Unknown => format!("{:.3} ms", secs * 1e3),
        }
    }

    /// Why this candidate was chosen, in one line: `unroll 8: 3.1 GB/s vs
    /// 2.2 GB/s for 4 (1.41x)`.
    pub fn explain(&self) -> String {
        let default = &self.measurements[0];
        let win = &self.measurements[self.winner];
        if self.winner != 0 {
            return format!(
                "{}: {} vs {} for {} ({:.2}x)",
                win.params,
                self.rate(self.winner),
                self.rate(0),
                default.params.short(),
                self.speedup()
            );
        }
        let best = self.fastest();
        if best == 0 {
            format!(
                "{}: {}, the default is fastest",
                default.params,
                self.rate(0)
            )
        } else {
            let gain = default.median / self.measurements[best].median - 1.0;
            format!(
                "{}: {}, kept the default ({} was only {:.1}% faster)",
                default.params,
                self.rate(0),
                self.measurements[best].params,
                gain * 100.0
            )
        }
    }

    /// The database record for this outcome on the machine `fingerprint`.
    pub fn record(&self, fingerprint: &str, when: u64) -> Record {
        let ns = |secs: f64| (secs * 1e9).round().max(1.0) as u64;
        Record {
            fingerprint: fingerprint.to_string(),
            kernel: self.kernel.clone(),
            size_class: self.size_class.clone(),
            params: self.params().clone(),
            default_params: self.measurements[0].params.clone(),
            best_ns: ns(self.measurements[self.winner].median),
            default_ns: ns(self.measurements[0].median),
            when,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] {}",
            self.kernel,
            self.size_class,
            self.explain()
        )
    }
}

/// Measures every candidate of `tunable` and picks one. Does not touch the
/// database (see [`crate::tune`] for that).
///
/// Robustness, in order: each candidate is called once to calibrate how many
/// back-to-back calls make a sample of about `min_sample`, and once more as
/// warmup; then the candidates are timed in rounds, in a rotating order (so
/// a clock ramp or a noisy neighbour hits all of them alike), until every
/// candidate has `max_repeats` samples or the budget is spent (but never
/// fewer than `min_repeats` samples). Each candidate's score is the median
/// of its samples. The winner is the fastest median, unless it beats the
/// default (the first candidate) by less than `min_gain`, in which case the
/// default stays.
pub fn measure(tunable: &Tunable, config: &Config) -> Outcome {
    let start = Instant::now();
    let n = tunable.candidates.len();
    let call = |p: &Params| {
        black_box(tunable.run(black_box(p)));
    };

    // Calibrate and warm up.
    let mut inner = vec![1u32; n];
    for (i, p) in tunable.candidates.iter().enumerate() {
        let t = Instant::now();
        call(p);
        let dt = t.elapsed().as_secs_f64().max(1e-9);
        inner[i] = (config.min_sample.as_secs_f64() / dt)
            .ceil()
            .clamp(1.0, 1024.0) as u32;
        for _ in 0..inner[i] {
            call(p);
        }
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut round = 0usize;
    loop {
        for k in 0..n {
            let i = (k + round) % n;
            let p = &tunable.candidates[i];
            let t = Instant::now();
            for _ in 0..inner[i] {
                call(p);
            }
            samples[i].push(t.elapsed().as_secs_f64().max(1e-9) / f64::from(inner[i]));
        }
        round += 1;
        if round >= config.max_repeats.max(1)
            || (round >= config.min_repeats && start.elapsed() >= config.budget)
        {
            break;
        }
    }

    let measurements: Vec<Measurement> = tunable
        .candidates
        .iter()
        .zip(samples)
        .map(|(p, s)| Measurement {
            params: p.clone(),
            median: median(&s),
            samples: s,
        })
        .collect();
    let best = (0..n)
        .min_by(|&a, &b| measurements[a].median.total_cmp(&measurements[b].median))
        .unwrap_or(0);
    let winner = if best != 0
        && measurements[best].median > measurements[0].median * (1.0 - config.min_gain)
    {
        0
    } else {
        best
    };
    Outcome {
        kernel: tunable.kernel.clone(),
        size_class: tunable.size_class.clone(),
        work: tunable.work,
        measurements,
        winner,
        elapsed: start.elapsed(),
    }
}

/// The median of a non-empty list (the mean of the two middle values when
/// even). Not-a-number sorts last, and an empty list gives infinity.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn spin(us: u64) -> u64 {
        let t = Instant::now();
        let mut x = 0u64;
        while t.elapsed() < Duration::from_micros(us) {
            x = x.wrapping_add(1);
        }
        x
    }

    fn quick() -> Config {
        Config {
            budget: Duration::from_millis(30),
            min_repeats: 3,
            max_repeats: 9,
            min_sample: Duration::from_micros(50),
            min_gain: 0.03,
        }
    }

    fn candidates(values: &[i64]) -> Vec<Params> {
        values
            .iter()
            .map(|&v| Params::new().with("cost", v))
            .collect()
    }

    #[test]
    fn median_of_odd_even_and_empty() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
        assert!(median(&[]).is_infinite());
        // An outlier does not move the median.
        assert_eq!(median(&[1.0, 1.0, 1.0, 1.0, 1000.0]), 1.0);
    }

    #[test]
    fn size_classes() {
        assert_eq!(size_class(1), "2^0");
        assert_eq!(size_class(0), "2^0");
        assert_eq!(size_class(262144), "2^18");
        assert_eq!(size_class(262145), "2^18");
        assert_eq!(size_class(524287), "2^18");
    }

    #[test]
    fn the_tunable_is_validated() {
        let ok = |k: &str, s: &str, c: Vec<Params>| Tunable::new(k, s, c, Work::Unknown, |_| 0);
        assert!(ok("k", "s", candidates(&[1, 2])).is_ok());
        assert!(ok("bad kernel", "s", candidates(&[1])).is_err());
        assert!(ok("k", "", candidates(&[1])).is_err());
        assert!(ok("k", "s", vec![]).is_err());
        assert!(ok("k", "s", candidates(&[1, 1])).is_err());
        let many: Vec<i64> = (0..65).collect();
        assert!(ok("k", "s", candidates(&many)).is_err());
    }

    #[test]
    fn picks_the_clearly_faster_candidate() {
        // Candidate 0 (the default) costs 400 us, candidate 2 costs 100 us.
        let t = Tunable::new(
            "spin",
            "s",
            candidates(&[400, 250, 100]),
            Work::Bytes(1_000_000),
            |p| spin(p.get("cost").unwrap() as u64),
        )
        .unwrap();
        let out = measure(&t, &quick());
        assert_eq!(out.winner, 2, "{out}");
        assert!(out.speedup() > 2.0, "{out}");
        assert!(out.explain().starts_with("cost 100: "), "{}", out.explain());
        assert!(out.explain().contains("GB/s vs "), "{}", out.explain());
        assert!(out.explain().contains("for 400"), "{}", out.explain());
        for m in &out.measurements {
            assert!(m.samples.len() >= 3);
        }
    }

    #[test]
    fn keeps_the_default_when_the_gain_is_noise() {
        // Identical work under different labels: the default must stay.
        let t = Tunable::new(
            "same",
            "s",
            candidates(&[200, 201, 199]),
            Work::Unknown,
            |_| spin(200),
        )
        .unwrap();
        let cfg = Config {
            min_gain: 0.5,
            ..quick()
        };
        let out = measure(&t, &cfg);
        assert_eq!(out.winner, 0, "{out}");
        assert!(out.explain().contains("default"), "{}", out.explain());
        assert!(out.explain().contains(" ms"));
    }

    #[test]
    fn respects_repeat_bounds_and_warms_up() {
        let calls = Rc::new(Cell::new(0u64));
        let seen = Rc::clone(&calls);
        let t = Tunable::new(
            "count",
            "s",
            candidates(&[1, 2]),
            Work::Unknown,
            move |_| {
                seen.set(seen.get() + 1);
                spin(300)
            },
        )
        .unwrap();
        let cfg = Config {
            budget: Duration::ZERO,
            min_repeats: 4,
            max_repeats: 100,
            min_sample: Duration::from_micros(1),
            min_gain: 0.0,
        };
        let out = measure(&t, &cfg);
        // Zero budget still measures min_repeats times per candidate.
        assert!(out.measurements.iter().all(|m| m.samples.len() == 4));
        // Each candidate: 1 calibration call + `inner` warmup calls + 4 samples.
        assert!(calls.get() >= 2 * (1 + 1 + 4), "{}", calls.get());
    }

    #[test]
    fn the_budget_stops_the_measurement() {
        let t = Tunable::new("slow", "s", candidates(&[1, 2]), Work::Unknown, |_| {
            spin(2_000)
        })
        .unwrap();
        let cfg = Config {
            budget: Duration::from_millis(40),
            min_repeats: 2,
            max_repeats: 10_000,
            min_sample: Duration::from_micros(1),
            min_gain: 0.03,
        };
        let out = measure(&t, &cfg);
        assert!(
            out.elapsed < Duration::from_millis(400),
            "{:?}",
            out.elapsed
        );
        assert!(out.measurements[0].samples.len() < 10_000);
    }

    #[test]
    fn the_record_carries_the_outcome() {
        let t = Tunable::new("k", "s", candidates(&[300, 100]), Work::Unknown, |p| {
            spin(p.get("cost").unwrap() as u64)
        })
        .unwrap();
        let out = measure(&t, &quick());
        let r = out.record("0123456789abcdef", 42);
        assert_eq!(r.params, *out.params());
        assert_eq!(r.default_params.get("cost"), Some(300));
        assert!(r.best_ns < r.default_ns);
        assert_eq!(r.when, 42);
    }
}
