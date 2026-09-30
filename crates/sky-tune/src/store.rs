//! Where the database lives, and the handle for reading and writing it.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::{is_ours, Db, Record};
use crate::measure::Outcome;
use crate::params::{clip, Params};

/// An error from storing or clearing results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuneError {
    /// No cache directory could be found (`HOME` unset, say) and
    /// `SKY_TUNE_DB` is not set.
    NoLocation,
    /// The database could not be written, or is not ours to overwrite.
    Store(String),
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuneError::NoLocation => f.write_str(
                "cannot find a cache directory for the tuning database (set SKY_TUNE_DB to a file)",
            ),
            TuneError::Store(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for TuneError {}

/// The database file for these environment values: `SKY_TUNE_DB` if it is
/// set and not empty; else `sky/tune.db` under `XDG_CACHE_HOME` if that is an
/// absolute path; else `~/Library/Caches/sky/tune.db` on macOS or
/// `~/.cache/sky/tune.db` elsewhere. `None` when no home directory is known.
/// (Taking the lookup as a function keeps this testable without touching the
/// process environment.)
pub fn db_path_from(env: &dyn Fn(&str) -> Option<OsString>, macos: bool) -> Option<PathBuf> {
    let get = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(explicit) = get("SKY_TUNE_DB") {
        return Some(PathBuf::from(explicit));
    }
    if let Some(xdg) = get("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
    {
        return Some(xdg.join("sky").join("tune.db"));
    }
    let home = PathBuf::from(get("HOME")?);
    if !home.is_absolute() {
        return None;
    }
    let base = if macos {
        home.join("Library").join("Caches")
    } else {
        home.join(".cache")
    };
    Some(base.join("sky").join("tune.db"))
}

/// The database file on this machine, from the real environment.
pub fn db_path() -> Option<PathBuf> {
    db_path_from(&|k| std::env::var_os(k), cfg!(target_os = "macos"))
}

/// The current time as seconds since the Unix epoch (0 if the clock is
/// before it).
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A database file together with the fingerprint results are keyed by.
#[derive(Clone, Debug)]
pub struct Store {
    path: PathBuf,
    fingerprint: String,
}

impl Store {
    /// The store at the default location for this machine, or `None` when no
    /// location can be determined.
    pub fn default_location() -> Option<Store> {
        Some(Store::at(db_path()?, sky_hw::current_fingerprint()))
    }

    /// A store at `path` for the machine `fingerprint`. (Tests use this with
    /// a made-up fingerprint; real code uses [`Store::default_location`].)
    pub fn at(path: PathBuf, fingerprint: &str) -> Store {
        Store {
            path,
            fingerprint: fingerprint.to_string(),
        }
    }

    /// The database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The fingerprint results are keyed by.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Loads the database (missing means empty; a bad file means empty with
    /// warnings).
    pub fn load(&self) -> Db {
        Db::load(&self.path)
    }

    /// The tuned parameters for a kernel and size class on this machine.
    pub fn lookup(&self, kernel: &str, size_class: &str) -> Option<Params> {
        self.load()
            .get(&self.fingerprint, kernel, size_class)
            .map(|r| r.params.clone())
    }

    /// Stores an outcome as this machine's result for its kernel and size
    /// class, replacing an older one, and returns the record. Refuses (and
    /// changes nothing) when the file is not a database of this version.
    pub fn record(&self, outcome: &Outcome) -> Result<Record, TuneError> {
        let record = outcome.record(&self.fingerprint, now());
        let mut db = self.load();
        db.insert(record.clone());
        db.save(&self.path).map_err(TuneError::Store)?;
        Ok(record)
    }

    /// Deletes the database file. Only a file that is a sky-tune database
    /// (or empty) is deleted: `SKY_TUNE_DB` pointing at something else must
    /// not lose that file. Returns whether a file was removed.
    pub fn reset(&self) -> Result<bool, TuneError> {
        if !self.path.exists() {
            return Ok(false);
        }
        if !is_ours(&self.path) {
            return Err(TuneError::Store(format!(
                "not deleting {}: it is not a sky-tune database",
                clip(&self.path.display().to_string())
            )));
        }
        fs::remove_file(&self.path).map_err(|e| {
            TuneError::Store(format!(
                "cannot delete {}: {e}",
                clip(&self.path.display().to_string())
            ))
        })?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn location_follows_the_environment() {
        let p = |pairs: &[(&str, &str)], mac| db_path_from(&env(pairs), mac);
        assert_eq!(
            p(&[("SKY_TUNE_DB", "/x/my.db"), ("HOME", "/h")], true),
            Some(PathBuf::from("/x/my.db"))
        );
        assert_eq!(
            p(&[("XDG_CACHE_HOME", "/xdg"), ("HOME", "/h")], true),
            Some(PathBuf::from("/xdg/sky/tune.db"))
        );
        assert_eq!(
            p(&[("HOME", "/Users/me")], true),
            Some(PathBuf::from("/Users/me/Library/Caches/sky/tune.db"))
        );
        assert_eq!(
            p(&[("HOME", "/home/me")], false),
            Some(PathBuf::from("/home/me/.cache/sky/tune.db"))
        );
        // Empty and relative values do not count.
        assert_eq!(
            p(
                &[
                    ("SKY_TUNE_DB", ""),
                    ("XDG_CACHE_HOME", "rel"),
                    ("HOME", "/h")
                ],
                false
            ),
            Some(PathBuf::from("/h/.cache/sky/tune.db"))
        );
        assert_eq!(p(&[("HOME", "relative")], false), None);
        assert_eq!(p(&[], false), None);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sky-tune-store-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reset_only_deletes_databases() {
        let dir = scratch("reset");
        let store = Store::at(dir.join("tune.db"), "0123456789abcdef");
        assert_eq!(store.reset(), Ok(false), "nothing to delete");
        fs::write(store.path(), "sky-tune-db 1\n").unwrap();
        assert_eq!(store.reset(), Ok(true));
        assert!(!store.path().exists());

        let other = Store::at(dir.join("thesis.txt"), "0123456789abcdef");
        fs::write(other.path(), "chapter 1\n").unwrap();
        assert!(other.reset().is_err());
        assert!(other.path().exists(), "a foreign file must survive");
        let _ = fs::remove_dir_all(&dir);
    }
}
