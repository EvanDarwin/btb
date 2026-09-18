#!/usr/bin/env bash
# Pre-commit CI gate. First auto-formats the staged sources - `ruff format` for Python, `rustfmt` for Rust: if it
# rewrites any it re-stages them and blocks this one commit with a notice, so a reformat never lands silently
# under you - just re-run the commit. Then blocks the commit while ruff lint or mypy would fail (the checks
# `python build.py check` runs in CI). Working-tree check from the project root. Disable/edit via /hooks.
set -uo pipefail
# only gate git commits: bail fast on any other Bash command (the settings `if` also scopes this)
payload=$(cat 2>/dev/null || true)
case "$payload" in *"git commit"*) ;; *) exit 0 ;; esac
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# a worktree keeps its .venv in the shared main checkout, so look there too, then fall back to PATH
venv=""
for base in "." "$(dirname "$(git rev-parse --git-common-dir 2>/dev/null || echo .)")"; do
  [ -x "$base/.venv/bin/ruff" ] && {
    venv="$base/.venv"
    break
  }
done
if [ -n "$venv" ]; then
  ruff="$venv/bin/ruff"
  mypy="$venv/bin/mypy"
else
  ruff=ruff
  mypy=mypy
fi
if command -v rustfmt >/dev/null 2>&1; then
  rustfmt=rustfmt
elif [ -x "${CARGO_HOME:-$HOME/.cargo}/bin/rustfmt" ]; then
  rustfmt="${CARGO_HOME:-$HOME/.cargo}/bin/rustfmt"
else
  rustfmt=""
fi

# the Rust edition for a file, from the nearest Cargo.toml up the tree (rustfmt needs it to parse 2021 syntax)
rs_edition() {
  d=$(dirname "$1")
  while [ "$d" != "/" ] && [ "$d" != "." ]; do
    if [ -f "$d/Cargo.toml" ]; then
      ed=$(sed -n 's/^[[:space:]]*edition[[:space:]]*=[[:space:]]*"\([0-9]*\)".*/\1/p' "$d/Cargo.toml" | head -1)
      [ -n "$ed" ] && {
        printf '%s' "$ed"
        return
      }
    fi
    d=$(dirname "$d")
  done
  printf '2021'
}

# format the staged sources; re-stage and report any the formatter rewrites, then stop so the reformat is part of
# the commit you re-run rather than a silent change
staged=$(git diff --cached --name-only --diff-filter=ACM)
reformatted=""
if [ -n "$staged" ]; then
  IFS=$'\n'
  for f in $staged; do
    case "$f" in
      *.py) "$ruff" format -q -- "$f" 2>/dev/null || true ;;
      *.rs) [ -n "$rustfmt" ] && "$rustfmt" --edition "$(rs_edition "$f")" "$f" 2>/dev/null || true ;;
      *) continue ;;
    esac
    if ! git diff --quiet -- "$f"; then
      git add -- "$f"
      reformatted="${reformatted}  $f"$'\n'
    fi
  done
  unset IFS
fi
if [ -n "$reformatted" ]; then
  printf '[pre-commit] the formatter rewrote and re-staged these - re-run the commit to include the formatting:\n%s' "$reformatted" >&2
  exit 2
fi

log=$(mktemp); fails=""
"$ruff" check . >"$log" 2>&1 || fails+=$'\nruff check:\n'"$(tail -8 "$log")"
"$mypy" >"$log" 2>&1 || fails+=$'\nmypy:\n'"$(tail -8 "$log")"
rm -f "$log"
if [ -n "$fails" ]; then
  printf '[pre-commit] commit blocked - the CI gate fails; fix these first:%s\n' "$fails" >&2
  exit 2
fi
