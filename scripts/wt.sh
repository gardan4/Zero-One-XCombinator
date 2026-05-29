#!/usr/bin/env bash
# Create an isolated git worktree + branch for a parallel experiment.
# Usage: just wt <name>   (or ./scripts/wt.sh <name>)
set -euo pipefail

name="${1:-}"
if [[ -z "$name" ]]; then
  echo "usage: just wt <name>   (or ./scripts/wt.sh <name>)" >&2
  exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slug="$(echo "$name" | tr ' /' '--' | tr '[:upper:]' '[:lower:]')"
branch="exp/$slug"
# Sibling dir without spaces, regardless of how this checkout is named.
dest="$(cd "$root/.." && pwd)/Zero-One-Philyr-$slug"

if [[ -d "$dest" ]]; then
  echo "worktree already exists: $dest" >&2
  exit 1
fi

git -C "$root" worktree add -b "$branch" "$dest"

cat <<EOF

  worktree:  $dest
  branch:    $branch

next:
  cd "$dest"
  just setup        # install deps in the new worktree
  # ...experiment, then:  git push -u origin $branch
EOF
