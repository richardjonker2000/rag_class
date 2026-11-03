#!/usr/bin/env bash
# Download the DeckCraft release (CLI included) into tools/; prints the CLI path.
set -euo pipefail
VER="${DECKCRAFT_VERSION:-0.12.0}"
DIR="$(cd "$(dirname "$0")" && pwd)/tools"
mkdir -p "$DIR"
CLI="$DIR/deckcraft-$VER-linux-x86_64/bin/deckcraft-cli"
if [ ! -x "$CLI" ]; then
  curl -sSL -o "$DIR/dc.tar.gz" \
    "https://github.com/storytold/deckcraft/releases/download/v$VER/deckcraft-$VER-linux-x86_64.tar.gz"
  tar xzf "$DIR/dc.tar.gz" -C "$DIR" && rm "$DIR/dc.tar.gz"
fi
echo "$CLI"
