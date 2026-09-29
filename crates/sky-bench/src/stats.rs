//! Summary statistics over repeated timings (nanoseconds).

/// The samples of one measurement, in run order, with their summary.
#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub samples: Vec<u64>,
    pub median: u64,
    pub min: u64,
    pub max: u64,
}

impl Stats {
    /// Summarizes `samples` (at least one).
    pub fn new(samples: Vec<u64>) -> Stats {
        assert!(!samples.is_empty(), "a measurement needs a sample");
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            // u128: the sum of two large samples must not overflow.
            ((sorted[n / 2 - 1] as u128 + sorted[n / 2] as u128) / 2) as u64
        };
        Stats {
            median,
            min: sorted[0],
            max: sorted[n - 1],
            samples,
        }
    }

    /// The range of the samples relative to the median, in percent: 0 for
    /// identical samples, large when the machine was noisy. Not a standard
    /// deviation: with a handful of repetitions the range is the honest number.
    pub fn spread_percent(&self) -> f64 {
        if self.median == 0 {
            return 0.0;
        }
        (self.max - self.min) as f64 / self.median as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_and_even_medians() {
        assert_eq!(Stats::new(vec![5, 1, 9]).median, 5);
        assert_eq!(Stats::new(vec![4, 1, 9, 2]).median, 3);
        assert_eq!(Stats::new(vec![7]).median, 7);
    }

    #[test]
    fn spread_is_relative_to_the_median() {
        let s = Stats::new(vec![90, 100, 110]);
        assert_eq!((s.min, s.max), (90, 110));
        assert!((s.spread_percent() - 20.0).abs() < 1e-9);
        assert_eq!(Stats::new(vec![0, 0]).spread_percent(), 0.0);
    }

    #[test]
    fn the_median_of_huge_samples_does_not_overflow() {
        let s = Stats::new(vec![u64::MAX, u64::MAX - 2]);
        assert_eq!(s.median, u64::MAX - 1);
    }
}
