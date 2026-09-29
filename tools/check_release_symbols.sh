#!/usr/bin/env bash
# Fails if a test-only name (tools/release-symbol-denylist.txt) reaches a
# release artifact.
#
# Builds sky-project and sky-cli in release mode, then scans every workspace
# artifact that build produced: each rlib, each rlib's embedded lib.rmeta,
# each standalone .rmeta, and the sky binary, with `strings -a` and `nm -a`.
# Any deny-listed name found anywhere is an error. A control check requires
# `resolve_import` in the sky-project rlib (with both tools), so a broken
# tool, empty input, or wrong grep can't pass silently.
#
# Usage: tools/check_release_symbols.sh   (from anywhere in the repo)
set -euo pipefail

cd "$(dirname "$0")/.."
denylist=tools/release-symbol-denylist.txt
control=resolve_import

for tool in strings nm ar cargo; do
    command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 2; }
done
# sed reads all input (head could exit early and SIGPIPE the tool under pipefail).
strings --version | sed -n 1p
nm --version | sed -n 1p

# Deny-listed names: strip comments and whitespace, skip blank lines.
mapfile -t names < <(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$denylist" | grep -v '^$')
if [ "${#names[@]}" -eq 0 ]; then
    echo "error: $denylist lists no names" >&2
    exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Build first, on its own, so a failed build stops the script here (inside a
# process substitution its exit status would be lost).
build_json="$tmp/build.json"
if ! cargo build --release --locked -p sky-project -p sky-cli --message-format=json-render-diagnostics >"$build_json"; then
    echo "error: the release build failed" >&2
    exit 2
fi

# The exact files this build produced (not stale ones from older builds),
# from cargo's JSON messages: workspace crates only (libsky_*) and the binary.
mapfile -t artifacts < <(
    grep '"reason":"compiler-artifact"' "$build_json" |
        grep -o '"filenames":\[[^]]*\]' |
        grep -o '"[^"]*"' | tr -d '"' |
        grep -E '/libsky_[a-z_]*(-[0-9a-f]+)?\.(rlib|rmeta)$|/sky$' |
        sort -u
)
project_rlib=$(printf '%s\n' "${artifacts[@]}" | grep -E '/libsky_project(-[0-9a-f]+)?\.rlib$' | head -n1 || true)
if [ -z "$project_rlib" ] || ! printf '%s\n' "${artifacts[@]}" | grep -q '/sky$'; then
    echo "error: the build did not report the sky-project rlib and the sky binary" >&2
    printf '  %s\n' "${artifacts[@]}" >&2
    exit 2
fi

# Each item to scan: "label<TAB>path". An rlib's embedded lib.rmeta is
# extracted and scanned as its own item.
items=()
for f in "${artifacts[@]}"; do
    items+=("${f#"$PWD"/}"$'\t'"$f")
    if [[ $f == *.rlib ]]; then
        meta="$tmp/$(basename "$f").lib.rmeta"
        ar p "$f" lib.rmeta >"$meta"
        [ -s "$meta" ] || { echo "error: no lib.rmeta in $f" >&2; exit 2; }
        items+=("${f#"$PWD"/} (lib.rmeta)"$'\t'"$meta")
    fi
done

# count TOOL FILE NAME: occurrences of NAME in TOOL's output for FILE, or
# "-" if nm cannot read the file (metadata is not an object file). If strings
# fails, it prints a non-number, which the caller turns into exit 2 (without
# this, an empty output would count as 0 hits).
count() {
    local out
    if [ "$1" = strings ]; then
        out=$(strings -a "$2") || { echo "strings-failed"; return; }
    elif ! out=$(nm -a "$2" 2>/dev/null); then
        echo -
        return
    fi
    grep -cF -- "$3" <<<"$out" || true
}

failed=0
printf '%-8s %-7s %-20s %s\n' COUNT TOOL NAME ARTIFACT
for item in "${items[@]}"; do
    label=${item%%$'\t'*}
    path=${item#*$'\t'}
    for tool in strings nm; do
        for name in "${names[@]}"; do
            c=$(count "$tool" "$path" "$name")
            if ! [[ $c =~ ^([0-9]+|-)$ ]]; then
                echo "error: $tool failed on $label" >&2
                exit 2
            fi
            printf '%-8s %-7s %-20s %s\n' "$c" "$tool" "$name" "$label"
            if [ "$c" != - ] && [ "$c" -gt 0 ]; then
                failed=1
            fi
        done
    done
done

# Control: the scan must be able to see a name that is there.
cs=$(count strings "$project_rlib" "$control")
cn=$(count nm "$project_rlib" "$control")
echo "control: $control in ${project_rlib#"$PWD"/}: strings=$cs nm=$cn"
if ! [[ $cs =~ ^[1-9][0-9]*$ && $cn =~ ^[1-9][0-9]*$ ]]; then
    echo "error: control name $control not found; the scan is broken" >&2
    exit 2
fi

if [ "$failed" -ne 0 ]; then
    echo "error: a deny-listed test-only name reached a release artifact (see nonzero counts above)" >&2
    exit 1
fi
echo "ok: none of ${#names[@]} deny-listed name(s) in ${#items[@]} release artifact(s)"
