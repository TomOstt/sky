//! Kernel parameters: a few named integers (a block size, an unroll factor).

use std::collections::BTreeMap;
use std::fmt;

/// The most parameters one candidate may have.
pub const MAX_PARAMS: usize = 8;
/// The largest magnitude of a parameter value. Far above any block size, and
/// small enough that a value can never overflow an index computation done in
/// `i64`.
pub const MAX_VALUE: i64 = 1 << 40;
/// The most candidates one tunable may have (a "small discrete set").
pub const MAX_CANDIDATES: usize = 64;

/// A set of named integer parameters, such as `block=64, unroll=4`. Names are
/// lower-case ASCII identifiers; values are integers of magnitude at most
/// [`MAX_VALUE`]. Ordered by name, so equal sets are equal and print alike.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Params(BTreeMap<String, i64>);

impl Params {
    /// No parameters.
    pub fn new() -> Params {
        Params::default()
    }

    /// Adds (or replaces) `name = value`.
    ///
    /// # Panics
    ///
    /// On an invalid name or value, or more than [`MAX_PARAMS`] parameters:
    /// candidates are written in code, so that is a programming error.
    /// (Stored text goes through [`Params::decode`], which never panics.)
    pub fn with(mut self, name: &str, value: i64) -> Params {
        assert!(valid_name(name), "invalid parameter name {name:?}");
        assert!(value.abs() <= MAX_VALUE, "parameter {name} out of range");
        self.0.insert(name.to_string(), value);
        assert!(self.0.len() <= MAX_PARAMS, "too many parameters");
        self
    }

    /// The value of `name`, if set.
    pub fn get(&self, name: &str) -> Option<i64> {
        self.0.get(name).copied()
    }

    /// The value of `name`, or `default` when it is not set.
    pub fn get_or(&self, name: &str, default: i64) -> i64 {
        self.get(name).unwrap_or(default)
    }

    /// The value of `name` clamped into `lo..=hi`, or `default` when it is
    /// not set. A kernel should read stored parameters this way: tuning is a
    /// hint, and a hint outside what the kernel can handle must not matter.
    pub fn get_clamped(&self, name: &str, default: i64, lo: i64, hi: i64) -> i64 {
        self.get_or(name, default).clamp(lo, hi)
    }

    /// The parameters in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, i64)> {
        self.0.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// The number of parameters.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The stored form: `block:64,unroll:4` (no spaces, name order).
    pub fn encode(&self) -> String {
        let parts: Vec<String> = self.0.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        parts.join(",")
    }

    /// Parses the stored form. Every part is validated: a name that is not an
    /// identifier, a value that is not an integer within range, a repeated
    /// name, too many parameters, or an empty text is an error (with the
    /// reason), never a panic.
    pub fn decode(text: &str) -> Result<Params, String> {
        if text.is_empty() {
            return Err("empty parameters".to_string());
        }
        let mut params = BTreeMap::new();
        for part in text.split(',') {
            let (name, value) = part
                .split_once(':')
                .ok_or_else(|| format!("`{}` is not name:value", clip(part)))?;
            if !valid_name(name) {
                return Err(format!("bad parameter name `{}`", clip(name)));
            }
            let value: i64 = value
                .parse()
                .map_err(|_| format!("bad value for {name}: `{}`", clip(value)))?;
            if value.abs() > MAX_VALUE {
                return Err(format!("value of {name} is out of range"));
            }
            if params.insert(name.to_string(), value).is_some() {
                return Err(format!("{name} is given twice"));
            }
            if params.len() > MAX_PARAMS {
                return Err("too many parameters".to_string());
            }
        }
        Ok(Params(params))
    }

    /// The values alone when there is one parameter (`4`), else the full
    /// text: how a comparison reads ("unroll 8 vs ... for 4").
    pub fn short(&self) -> String {
        match self.0.iter().next() {
            Some((_, v)) if self.0.len() == 1 => v.to_string(),
            _ => self.to_string(),
        }
    }
}

/// `unroll 8` or `block 64, unroll 4`.
impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("(no parameters)");
        }
        for (i, (k, v)) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{k} {v}")?;
        }
        Ok(())
    }
}

/// The candidate space of several parameters, each with a few values: every
/// combination, first values first (so the first candidate is the first
/// value of each: put the default there).
///
/// ```
/// use sky_tune::grid;
/// let space = grid(&[("block", &[32, 64]), ("unroll", &[2, 4])]);
/// assert_eq!(space.len(), 4);
/// assert_eq!(space[0].encode(), "block:32,unroll:2");
/// ```
pub fn grid(axes: &[(&str, &[i64])]) -> Vec<Params> {
    let mut out = vec![Params::new()];
    for (name, values) in axes {
        let mut next = Vec::with_capacity(out.len() * values.len());
        for base in &out {
            for &v in *values {
                next.push(base.clone().with(name, v));
            }
        }
        out = next;
    }
    out
}

/// A parameter name: 1 to 32 characters `[a-z][a-z0-9_]*`.
pub(crate) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= 32
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Stored text can be anything: shorten and defang it before it goes into a
/// message.
pub(crate) fn clip(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars().take(24) {
        out.push(if c.is_control() { '?' } else { c });
    }
    if text.chars().count() > 24 {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let p = Params::new().with("unroll", 4).with("block", 64);
        assert_eq!(p.encode(), "block:64,unroll:4");
        assert_eq!(Params::decode(&p.encode()), Ok(p.clone()));
        assert_eq!(p.to_string(), "block 64, unroll 4");
        assert_eq!(p.short(), "block 64, unroll 4");
        assert_eq!(Params::new().with("unroll", 8).short(), "8");
        assert_eq!(p.get("block"), Some(64));
        assert_eq!(p.get_or("tile", 7), 7);
        assert_eq!(p.get_clamped("block", 1, 1, 32), 32);
        assert_eq!(Params::decode("t:-3").unwrap().get("t"), Some(-3));
    }

    #[test]
    fn decode_rejects_everything_odd() {
        for bad in [
            "",
            ":",
            "a",
            "a:",
            ":1",
            "a:1,",
            "a:1,a:2",
            "A:1",
            "1a:1",
            "a b:1",
            "a:x",
            "a:1.5",
            "a:1099511627777",
            "a:-1099511627777",
            "a:99999999999999999999",
            "a:1;rm -rf /",
            "a:1\n",
            "é:1",
            "a:1,b:2,c:3,d:4,e:5,f:6,g:7,h:8,i:9",
        ] {
            assert!(Params::decode(bad).is_err(), "{bad:?} must be rejected");
        }
        let long = format!("{}:1", "a".repeat(33));
        assert!(Params::decode(&long).is_err());
    }

    #[test]
    fn grid_is_the_product() {
        let g = grid(&[("a", &[1, 2, 3]), ("b", &[10, 20])]);
        assert_eq!(g.len(), 6);
        assert_eq!(g[0].encode(), "a:1,b:10");
        assert_eq!(g[1].encode(), "a:1,b:20");
        assert_eq!(g[5].encode(), "a:3,b:20");
        assert_eq!(grid(&[("a", &[5])]).len(), 1);
    }

    #[test]
    #[should_panic(expected = "invalid parameter name")]
    fn with_panics_on_a_bad_name() {
        let _ = Params::new().with("Bad Name", 1);
    }
}
