#!/usr/bin/env bash
# Regenerate casp17/patches/folding_input_daisy.py from the AF3 docker image.
#
# WHY THIS EXISTS
# The `alphafold3-3.0.3_casp17` image ships a half-patched folding_input.py
# whose two `modelSeeds` rules contradict each other:
#
#   _validate_keys whitelist   'modelSeeds' is commented out
#                              -> passing it   = "Unexpected JSON keys: modelSeeds"
#   loader (a few lines below) raises when the key is absent
#                              -> omitting it  = "must specify at least one rng seed"
#
# So a stock image cannot run any input JSON at all. The fix is to pull that one
# file out of the image, undo both halves, and bind-mount it back over the
# original on every `docker run`.
#
# We do NOT ship the resulting file: it is 1500 lines of DeepMind's AF3 source
# (Apache-2.0, but there is no reason to vendor a copy that only matches one
# image tag). Generate it locally instead — it takes a second and it is
# guaranteed to match whatever image you actually have.
#
# Usage:
#   bash casp17/scripts/make_af3_patch.sh              # default image + output
#   AF3_IMAGE=my-af3:tag bash casp17/scripts/make_af3_patch.sh
#
# The run_af3_*.sh wrappers call this automatically when the patch is missing.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${AF3_IMAGE:-alphafold3-3.0.3_casp17}
SRC=/app/alphafold/src/alphafold3/common/folding_input.py
OUT=${AF3_PATCH:-${PROJECT_ROOT}/casp17/patches/folding_input_daisy.py}

mkdir -p "$(dirname "$OUT")"
echo "[make_af3_patch] image=${IMAGE}"
docker run --rm --entrypoint cat "$IMAGE" "$SRC" > "${OUT}.raw"

python3 - "${OUT}.raw" "$OUT" <<'PY'
import re
import sys

raw, out = sys.argv[1], sys.argv[2]
src = open(raw).read()

# 1. Put 'modelSeeds' back into the alphafold3-dialect key whitelist.
src, n1 = re.subn(r"^(\s*)#\s*'modelSeeds',\s*$", r"\1'modelSeeds',", src, flags=re.M)

# 2. Silence the "must specify at least one rng seed" hard failure. The loader
#    further down already handles a missing key, so commenting out this early
#    guard is enough.
guard = re.compile(
    r"^(\s*)(if 'modelSeeds' not in raw_json or not raw_json\['modelSeeds'\]:\n"
    r"(?:\1  .*\n|\s*\n)*?\1  \)\n)",
    re.M,
)


def comment_out(m):
    # group(1) swallowed the first line's indent, so put it back before
    # splitting or the first line loses len(indent) real characters.
    indent, block = m.group(1), m.group(2)
    return "".join(
        f"{indent}# {line[len(indent):]}\n" if line.strip() else "\n"
        for line in (indent + block).rstrip("\n").split("\n")
    )


src, n2 = guard.subn(comment_out, src)

already = "'modelSeeds',\n" in src and n1 == 0
if not (n1 == 1 or already):
    sys.exit(f"[ERR] whitelist edit matched {n1} times, expected 1 (image changed?)")
if n2 != 1 and "# if 'modelSeeds' not in raw_json" not in src:
    sys.exit(f"[ERR] seed-guard edit matched {n2} times, expected 1 (image changed?)")

open(out, "w").write(src)
print(f"[make_af3_patch] whitelist edits={n1} guard edits={n2}")
PY

rm -f "${OUT}.raw"
echo "[make_af3_patch] wrote ${OUT}"
