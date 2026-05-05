"""Inject MSA from AF3 data pipeline output into all L4000 AF3 input JSONs.

Usage:
    python scripts/af3_inject_msa_l4000.py

Reads MSA from the first completed data pipeline output (L4001),
injects it into all 20 L4000 af3_inputs JSONs, and writes to af3_inputs_msa/.
"""

import json
import sys
from pathlib import Path


def extract_msa_from_data_json(data_json_path: Path) -> dict:
    """Extract protein MSA fields from AF3 *_data.json output."""
    with open(data_json_path) as f:
        data = json.load(f)
    for seq in data["sequences"]:
        if "protein" in seq:
            p = seq["protein"]
            msa_fields = {}
            for key in ("unpairedMsa", "pairedMsa", "templates"):
                if p.get(key) is not None:
                    msa_fields[key] = p[key]
            return msa_fields
    raise ValueError(f"No protein entity found in {data_json_path}")


def main():
    root = Path(__file__).resolve().parent.parent

    # Find the data pipeline output
    msa_output_dir = root / "data" / "test_cases" / "casp16_l4000" / "af3_msa_output"

    # AF3 data pipeline outputs *_data.json files
    data_jsons = sorted(msa_output_dir.rglob("*_data.json"))
    if not data_jsons:
        print(f"ERROR: No *_data.json found in {msa_output_dir}")
        print("Run AF3 data pipeline first: docker run ... --norun_inference")
        sys.exit(1)

    print(f"Found data JSON: {data_jsons[0]}")
    msa_data = extract_msa_from_data_json(data_jsons[0])
    print(f"Extracted MSA fields: {list(msa_data.keys())}")
    for k, v in msa_data.items():
        if isinstance(v, str):
            print(f"  {k}: {len(v)} chars")
        elif isinstance(v, list):
            print(f"  {k}: {len(v)} entries")

    # Input and output directories
    input_dir = root / "data" / "test_cases" / "casp16_l4000" / "af3_inputs"
    output_dir = root / "data" / "test_cases" / "casp16_l4000" / "af3_inputs_msa"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in sorted(input_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        # Inject MSA into all protein chains
        for seq in data["sequences"]:
            if "protein" in seq:
                seq["protein"].update(msa_data)

        out_path = output_dir / json_file.name
        out_path.write_text(json.dumps(data))
        count += 1
        print(f"  Injected: {json_file.name}")

    print(f"\nDone. {count} JSONs written to {output_dir}")
    print(f"\nNext steps:")
    print(f"  bash hellbender/af3_01_sync_inputs.sh casp16_l4000")
    print(f"  bash hellbender/af3_02_submit_l4000.sh casp16_l4000")


if __name__ == "__main__":
    main()
