#!/usr/bin/env bash
# Pre-commit CI gate: block a git commit while ruff (lint + format) or mypy would fail - the Python checks CI
# runs via `python build.py check`. Working-tree check, from the project root. Disable/edit via /hooks.
set -uo pipefail
# only gate git commits: bail fast on any other Bash command (the settings `if` also scopes this)
payload=$(cat 2>/dev/null || true)
case "$payload" in *"git commit"*) ;; *) exit 0 ;; esac
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
ruff=.venv/bin/ruff; mypy=.venv/bin/mypy
[ -x "$ruff" ] || ruff=ruff
[ -x "$mypy" ] || mypy=mypy
log=$(mktemp); fails=""
"$ruff" check . >"$log" 2>&1 || fails+=$'\nruff check:\n'"$(tail -8 "$log")"
"$ruff" format --check . >"$log" 2>&1 || fails+=$'\nruff format --check:\n'"$(tail -8 "$log")"
"$mypy" >"$log" 2>&1 || fails+=$'\nmypy:\n'"$(tail -8 "$log")"
rm -f "$log"
if [ -n "$fails" ]; then
  printf '[pre-commit] commit blocked - the CI gate fails; fix these first:%s\n' "$fails" >&2
  exit 2
fi
