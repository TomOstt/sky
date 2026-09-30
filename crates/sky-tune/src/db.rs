//! The performance database: a small text file of tuning results.
//!
//! # Format (version 1)
//!
//! ```text
//! # sky-tune performance database. Data only: safe to read, edit or delete.
//! sky-tune-db 1
//! result fp=3f2a9c01d4e5b678 kernel=dot_blocked size=2^18 params=unroll:8 default=unroll:4 best_ns=81234 default_ns=115020 when=1790000000
//! ```
//!
//! One `result` line per (fingerprint, kernel, size class). Every field is a
//! `key=value` token separated by spaces; there is no quoting because no
//! value may contain a space. The first line that is not blank or a `#`
//! comment must be `sky-tune-db <version>`.
//!
//! # Safety
//!
//! The file is *data*. Nothing in it is ever executed, evaluated, used as a
//! path or a command, or passed to a shell; the only things done with its
//! values are comparing them and giving parameter integers to kernels that
//! clamp them. Reading it is bounded: a regular file of at most
//! [`MAX_FILE_BYTES`], lines of at most [`MAX_LINE`] bytes, at most
//! [`MAX_RECORDS`] results. A line that does not validate is skipped with a
//! warning and the rest is used; a file with the wrong header or a newer
//! version is used as empty and is never overwritten (so a future Sky's
//! database, or a file that is not ours, survives).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::params::{clip, Params};

/// The version this crate reads and writes.
pub const FORMAT_VERSION: u32 = 1;
/// The largest database file that is read.
pub const MAX_FILE_BYTES: u64 = 1 << 20;
/// The longest line that is parsed.
pub const MAX_LINE: usize = 1024;
/// The most results kept. Past it, the oldest is dropped.
pub const MAX_RECORDS: usize = 4096;

const HEADER_COMMENT: &str =
    "# sky-tune performance database. Data only: safe to read, edit or delete.\n\
# Written by `sky tune`; results are keyed by a hash of this machine's CPU and caches.\n";

/// One tuning result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// The hardware fingerprint the result was measured on.
    pub fingerprint: String,
    /// The kernel's name.
    pub kernel: String,
    /// The problem-size class.
    pub size_class: String,
    /// The winning parameters.
    pub params: Params,
    /// The default candidate's parameters, for comparison.
    pub default_params: Params,
    /// The winner's median time, in nanoseconds.
    pub best_ns: u64,
    /// The default candidate's median time, in nanoseconds.
    pub default_ns: u64,
    /// When it was measured: seconds since the Unix epoch.
    pub when: u64,
}

impl Record {
    /// How many times faster the winner was than the default (1.0 when the
    /// default won or times are missing).
    pub fn speedup(&self) -> f64 {
        if self.best_ns == 0 {
            1.0
        } else {
            self.default_ns as f64 / self.best_ns as f64
        }
    }

    /// The line this record is stored as.
    fn line(&self) -> String {
        format!(
            "result fp={} kernel={} size={} params={} default={} best_ns={} default_ns={} when={}",
            self.fingerprint,
            self.kernel,
            self.size_class,
            self.params.encode(),
            self.default_params.encode(),
            self.best_ns,
            self.default_ns,
            self.when
        )
    }
}

/// A kernel name or size class: 1 to 64 characters, ASCII letters, digits
/// and `_ . ^ - + : =`. No spaces, so one token per field.
pub fn valid_token(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 64
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_.^-+:=".contains(c))
}

fn valid_fingerprint(text: &str) -> bool {
    text.len() == 16
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A loaded database.
#[derive(Clone, Debug, Default)]
pub struct Db {
    records: Vec<Record>,
    /// What was skipped or ignored while loading, for the user.
    pub warnings: Vec<String>,
    /// Set when the file on disk must not be overwritten (it is not one of
    /// ours, or written by a newer Sky): the reason.
    pub read_only: Option<String>,
}

impl Db {
    /// An empty database.
    pub fn new() -> Db {
        Db::default()
    }

    /// Parses database text. Malformed lines are skipped with a warning in
    /// [`Db::warnings`]; a missing or unsupported header makes the whole text
    /// unusable (empty, with [`Db::read_only`] set).
    pub fn parse(text: &str) -> Db {
        let mut db = Db::new();
        let mut seen_header = false;
        for (i, raw) in text.lines().enumerate() {
            let n = i + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if raw.len() > MAX_LINE {
                db.warnings
                    .push(format!("line {n}: longer than {MAX_LINE} bytes, skipped"));
                continue;
            }
            if !seen_header {
                match line.strip_prefix("sky-tune-db ") {
                    Some(v) if v.trim() == FORMAT_VERSION.to_string() => {
                        seen_header = true;
                        continue;
                    }
                    Some(v) => {
                        let reason = format!(
                            "database version `{}` is not supported (this Sky reads version {FORMAT_VERSION})",
                            clip(v.trim())
                        );
                        return Db::unusable(reason);
                    }
                    None => {
                        return Db::unusable(
                            "not a sky-tune database (no `sky-tune-db` header)".to_string(),
                        );
                    }
                }
            }
            match parse_result(line) {
                Ok(record) => {
                    if db.records.len() >= MAX_RECORDS {
                        db.warnings.push(format!(
                            "line {n}: more than {MAX_RECORDS} results, skipped"
                        ));
                    } else {
                        db.insert(record);
                    }
                }
                Err(why) => db.warnings.push(format!("line {n}: {why}, ignored")),
            }
        }
        db
    }

    fn unusable(reason: String) -> Db {
        Db {
            warnings: vec![reason.clone()],
            read_only: Some(reason),
            ..Db::default()
        }
    }

    /// Reads the database at `path`. A missing file is an empty database
    /// (defaults apply); an unreadable, oversized, non-UTF-8 or foreign file
    /// is an empty database with a warning. Never fails and never panics.
    pub fn load(path: &Path) -> Db {
        let shown = clip(&path.display().to_string());
        match fs::metadata(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Db::new(),
            Err(e) => return Db::unusable(format!("cannot read {shown}: {e}")),
            Ok(m) if !m.is_file() => {
                return Db::unusable(format!("{shown} is not a regular file"));
            }
            Ok(_) => {}
        }
        let file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => return Db::unusable(format!("cannot read {shown}: {e}")),
        };
        // The handle, not the name: what was opened must be a regular file.
        if !file.metadata().is_ok_and(|m| m.is_file()) {
            return Db::unusable(format!("{shown} is not a regular file"));
        }
        let mut bytes = Vec::new();
        if let Err(e) = file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes) {
            return Db::unusable(format!("cannot read {shown}: {e}"));
        }
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Db::unusable(format!("{shown} is larger than {MAX_FILE_BYTES} bytes"));
        }
        let Ok(text) = String::from_utf8(bytes) else {
            return Db::unusable(format!("{shown} is not text"));
        };
        let mut db = Db::parse(&text);
        for w in &mut db.warnings {
            *w = format!("{shown}: {w}");
        }
        db
    }

    /// The results, oldest first.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// The result for this machine, kernel and size class.
    pub fn get(&self, fingerprint: &str, kernel: &str, size_class: &str) -> Option<&Record> {
        self.records.iter().find(|r| {
            r.fingerprint == fingerprint && r.kernel == kernel && r.size_class == size_class
        })
    }

    /// Adds a result, replacing the one with the same key. When the database
    /// is full the oldest result goes.
    pub fn insert(&mut self, record: Record) {
        self.records.retain(|r| {
            !(r.fingerprint == record.fingerprint
                && r.kernel == record.kernel
                && r.size_class == record.size_class)
        });
        if self.records.len() >= MAX_RECORDS {
            if let Some(oldest) = (0..self.records.len()).min_by_key(|&i| self.records[i].when) {
                self.records.remove(oldest);
            }
        }
        self.records.push(record);
    }

    /// The database as text.
    pub fn to_text(&self) -> String {
        let mut text = String::from(HEADER_COMMENT);
        text.push_str(&format!("sky-tune-db {FORMAT_VERSION}\n"));
        for r in &self.records {
            text.push_str(&r.line());
            text.push('\n');
        }
        text
    }

    /// Writes the database to `path` atomically (a temporary file in the
    /// same directory, then a rename), creating the directory. Refuses when
    /// [`Db::read_only`] is set.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(why) = &self.read_only {
            return Err(format!(
                "not overwriting {}: {why}",
                clip(&path.display().to_string())
            ));
        }
        write_atomic(path, &self.to_text())
    }
}

fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let io = |what: &str, e: std::io::Error| {
        format!("{what} {}: {e}", clip(&path.display().to_string()))
    };
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = dir {
        fs::create_dir_all(dir).map_err(|e| io("cannot create the directory of", e))?;
    }
    let name = path
        .file_name()
        .ok_or_else(|| "the database path has no file name".to_string())?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(".tmp{}", std::process::id()));
    let tmp: PathBuf = match dir {
        Some(dir) => dir.join(&tmp_name),
        None => PathBuf::from(&tmp_name),
    };
    // `create_new`: never write through a file or link someone else put there.
    let result = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|mut f| f.write_all(text.as_bytes()).and_then(|()| f.sync_all()))
        .and_then(|()| fs::rename(&tmp, path));
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(io("cannot write", e));
    }
    Ok(())
}

/// Whether the file at `path` is a sky-tune database (or empty/missing), so
/// that deleting it cannot destroy something else.
pub fn is_ours(path: &Path) -> bool {
    Db::load(path).read_only.is_none()
}

fn parse_result(line: &str) -> Result<Record, String> {
    let mut tokens = line.split_whitespace();
    if tokens.next() != Some("result") {
        return Err("unknown line".to_string());
    }
    let (mut fp, mut kernel, mut size, mut params, mut default) = (None, None, None, None, None);
    let (mut best, mut default_ns, mut when) = (None, None, None);
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| format!("`{}` is not key=value", clip(token)))?;
        let slot = match key {
            "fp" => &mut fp,
            "kernel" => &mut kernel,
            "size" => &mut size,
            "params" => &mut params,
            "default" => &mut default,
            "best_ns" => &mut best,
            "default_ns" => &mut default_ns,
            "when" => &mut when,
            // Unknown keys are ignored, so a later minor addition to
            // version 1 does not make older readers drop the result.
            _ => continue,
        };
        if slot.replace(value).is_some() {
            return Err(format!("`{}` given twice", clip(key)));
        }
    }
    fn field<'a>(v: Option<&'a str>, name: &str) -> Result<&'a str, String> {
        v.ok_or_else(|| format!("missing {name}"))
    }
    let fingerprint = field(fp, "fp")?;
    if !valid_fingerprint(fingerprint) {
        return Err("bad fingerprint".to_string());
    }
    let kernel = field(kernel, "kernel")?;
    if !valid_token(kernel) {
        return Err("bad kernel name".to_string());
    }
    let size_class = field(size, "size")?;
    if !valid_token(size_class) {
        return Err("bad size class".to_string());
    }
    let number = |v: Option<&str>, name: &str| -> Result<u64, String> {
        field(v, name)?
            .parse::<u64>()
            .map_err(|_| format!("bad {name}"))
    };
    Ok(Record {
        fingerprint: fingerprint.to_string(),
        kernel: kernel.to_string(),
        size_class: size_class.to_string(),
        params: Params::decode(field(params, "params")?).map_err(|e| format!("params: {e}"))?,
        default_params: Params::decode(field(default, "default")?)
            .map_err(|e| format!("default: {e}"))?,
        best_ns: number(best, "best_ns")?,
        default_ns: number(default_ns, "default_ns")?,
        when: number(when, "when")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: &str = "0123456789abcdef";

    fn record(kernel: &str, size: &str, unroll: i64, when: u64) -> Record {
        Record {
            fingerprint: FP.to_string(),
            kernel: kernel.to_string(),
            size_class: size.to_string(),
            params: Params::new().with("unroll", unroll),
            default_params: Params::new().with("unroll", 4),
            best_ns: 100,
            default_ns: 150,
            when,
        }
    }

    #[test]
    fn round_trips_through_text() {
        let mut db = Db::new();
        db.insert(record("dot", "2^18", 8, 10));
        db.insert(record("transpose", "512x512", 32, 11));
        let text = db.to_text();
        assert!(text.contains("sky-tune-db 1\n"));
        let back = Db::parse(&text);
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        assert_eq!(back.records(), db.records());
        let hit = back.get(FP, "dot", "2^18").unwrap();
        assert_eq!(hit.params.get("unroll"), Some(8));
        assert!((hit.speedup() - 1.5).abs() < 1e-12);
        assert!(back.get("ffffffffffffffff", "dot", "2^18").is_none());
    }

    #[test]
    fn insert_replaces_the_same_key() {
        let mut db = Db::new();
        db.insert(record("dot", "2^18", 8, 10));
        db.insert(record("dot", "2^18", 16, 20));
        assert_eq!(db.records().len(), 1);
        assert_eq!(
            db.get(FP, "dot", "2^18").unwrap().params.get("unroll"),
            Some(16)
        );
    }

    #[test]
    fn a_full_database_drops_the_oldest() {
        let mut db = Db::new();
        for i in 0..MAX_RECORDS {
            db.insert(record(&format!("k{i}"), "s", 1, 100 + i as u64));
        }
        db.insert(record("newest", "s", 1, 1_000_000));
        assert_eq!(db.records().len(), MAX_RECORDS);
        assert!(db.get(FP, "k0", "s").is_none());
        assert!(db.get(FP, "newest", "s").is_some());
    }

    #[test]
    fn malformed_lines_are_skipped_with_a_warning() {
        let good = record("dot", "2^18", 8, 10).line();
        let long = "x".repeat(MAX_LINE);
        let text = format!(
            "sky-tune-db 1\n\
             {good}\n\
             result fp=zz kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=1 when=1\n\
             result fp={FP} kernel=a;b size=b params=x:1 default=x:1 best_ns=1 default_ns=1 when=1\n\
             result fp={FP} kernel=a size=b params=X:1 default=x:1 best_ns=1 default_ns=1 when=1\n\
             result fp={FP} kernel=a size=b params=x:1 default=x:1 best_ns=-1 default_ns=1 when=1\n\
             result fp={FP} kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=1\n\
             result fp={FP} fp={FP} kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=1 when=1\n\
             result garbage\n\
             delete everything\n\
             result {long}\n"
        );
        let db = Db::parse(&text);
        assert_eq!(db.records().len(), 1, "only the good line survives");
        assert_eq!(db.warnings.len(), 9, "{:#?}", db.warnings);
        assert!(db.read_only.is_none());
        assert!(db.warnings.iter().all(|w| w.starts_with("line ")));
    }

    #[test]
    fn unknown_keys_are_ignored_and_comments_allowed() {
        let text = format!(
            "# hello\n\nsky-tune-db 1\n# more\nresult fp={FP} kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=2 when=3 extra=zzz\n"
        );
        let db = Db::parse(&text);
        assert_eq!(db.records().len(), 1);
        assert!(db.warnings.is_empty());
    }

    #[test]
    fn a_wrong_header_or_version_is_unusable_and_read_only() {
        for text in [
            "sky-tune-db 2\n",
            "sky-tune-db x\n",
            "hello world\n",
            "result fp=0123456789abcdef kernel=a size=b params=x:1 default=x:1 best_ns=1 default_ns=1 when=1\n",
        ] {
            let db = Db::parse(text);
            assert!(db.records().is_empty(), "{text}");
            assert!(db.read_only.is_some(), "{text}");
            assert!(db.save(Path::new("/nonexistent-dir/never-written")).is_err());
        }
        let empty = Db::parse("");
        assert!(empty.read_only.is_none() && empty.warnings.is_empty());
        let comments_only = Db::parse("# nothing yet\n");
        assert!(comments_only.read_only.is_none());
    }

    #[test]
    fn tokens_are_validated() {
        assert!(valid_token("dot_blocked"));
        assert!(valid_token("2^18"));
        assert!(valid_token("512x512"));
        assert!(valid_token("n=2^18"));
        for bad in [
            "",
            "a b",
            "a\nb",
            "a;b",
            "a/b",
            "$(x)",
            "é",
            &"a".repeat(65),
        ] {
            assert!(!valid_token(bad), "{bad:?}");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sky-tune-db-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_and_load_on_disk() {
        let dir = scratch("roundtrip");
        let path = dir.join("nested/deeper/tune.db");
        assert!(Db::load(&path).records().is_empty(), "missing means empty");
        assert!(Db::load(&path).warnings.is_empty());
        let mut db = Db::new();
        db.insert(record("dot", "2^18", 8, 10));
        db.save(&path).unwrap();
        let back = Db::load(&path);
        assert_eq!(back.records(), db.records());
        // No temporary file is left behind.
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
        assert!(is_ours(&path));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_files_are_ignored_and_not_overwritten() {
        let dir = scratch("hostile");
        // Not a database.
        let notes = dir.join("notes.txt");
        fs::write(&notes, "my thesis\n").unwrap();
        let db = Db::load(&notes);
        assert!(db.records().is_empty() && db.read_only.is_some());
        assert!(db.save(&notes).is_err());
        assert_eq!(fs::read_to_string(&notes).unwrap(), "my thesis\n");
        assert!(!is_ours(&notes));
        // Binary junk.
        let junk = dir.join("junk.db");
        fs::write(&junk, [0xff, 0xfe, 0x00, 0x80]).unwrap();
        assert!(Db::load(&junk).read_only.is_some());
        // Too large.
        let big = dir.join("big.db");
        fs::write(&big, vec![b'#'; MAX_FILE_BYTES as usize + 1]).unwrap();
        assert!(Db::load(&big).read_only.is_some());
        // A directory.
        assert!(Db::load(&dir).read_only.is_some());
        // Shell-looking content is only ever data.
        let evil = dir.join("evil.db");
        fs::write(
            &evil,
            "sky-tune-db 1\n`touch pwned` $(touch pwned)\nresult fp=$(id)\n",
        )
        .unwrap();
        let db = Db::load(&evil);
        assert!(db.records().is_empty());
        assert_eq!(db.warnings.len(), 2);
        assert!(!dir.join("pwned").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
