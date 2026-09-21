#!/bin/bash
# Set up Boltz-2 weights in the project's weights/boltz/ directory.
# Run once from the project root: bash scripts/setup_weights.sh
#
# By default links from ~/.boltz. Override with: BOLTZ_CACHE=/your/path bash scripts/setup_weights.sh

set -e
WEIGHTS_DIR="weights/boltz"
BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/.boltz}"

mkdir -p "$WEIGHTS_DIR"
echo "Linking Boltz-2 weights: $BOLTZ_CACHE → $WEIGHTS_DIR"

for f in boltz2_aff.ckpt boltz2_conf.ckpt mols; do
    src="$BOLTZ_CACHE/$f"
    dst="$WEIGHTS_DIR/$f"
    if [ -e "$src" ]; then
        ln -sf "$(realpath "$src")" "$dst"
        echo "  Linked: $f"
    else
        echo "  WARNING: $src not found — download Boltz-2 weights first"
    fi
done

echo "Done."
