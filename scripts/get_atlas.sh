#!/usr/bin/env bash
# AI use statement: docs/AI_USE.md
# Fetch the corpus and benchmark this system runs against.
# The data lives in its own repository so there is exactly one copy of it.
set -euo pipefail
DEST="${1:-$(dirname "$(dirname "$(readlink -f "$0")")")/../apollo-anomaly-atlas}"
if [ -d "$DEST/.git" ]; then
  echo "atlas already present at $DEST — pulling"
  git -C "$DEST" pull --ff-only
else
  git clone https://github.com/ARC-lab-University-of-Washington/apollo-anomaly-atlas "$DEST"
fi
echo
echo "Point LODESTAR at it:"
echo "  export LODESTAR_CORPUS=\"$(cd "$DEST" && pwd)/corpus/corpus.json\""
echo "  export LODESTAR_ATLAS=\"$(cd "$DEST" && pwd)/data/atlas_parsed.json\""
