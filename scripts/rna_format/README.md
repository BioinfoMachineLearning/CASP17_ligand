# RNA / DNA Format Tools (CASP submission helpers)

Vendored from <https://predictioncenter.org/download_area/other/rna-dna/>.
Author: Miao Zhichao (IBMC, CNRS), 2013–2022.

These are CASP-shipped utilities for **submission format checking** of RNA/DNA
predictions. Not used during inference — only at the submission step.

## Files

| File | Purpose | Python | Notes |
|------|---------|--------|-------|
| `rna_str_from_fasta.py` | FASTA → zero-coordinate PDB template (defines required atom set per residue) | **3** | Ported by us from the upstream Py2 source. |
| `rna_ver.py` | Validate submitted PDB atom set against `./TARGETS/{TARGET}.pdb.txt` template | **3** | Upstream-original. Reads `TARGET XXX` line from the submission to look up the template. |
| `dna_str_from_fasta.py` | DNA equivalent of `rna_str_from_fasta.py` | **2** | Upstream-original. Convert to Py3 if/when CASP17 ships a DNA target. |
| `dna_ver.py` | DNA equivalent of `rna_ver.py` | **2** (likely) | Same — convert if needed. |

## Workflow

### 1. Generate the reference template (once per RNA target)

```bash
python3 scripts/rna_format/rna_str_from_fasta.py \
  data/casp17_data/sequences/CASP17_R/R2314.txt \
  > data/casp17_data/struct/CASP17_R_template/R2314.pdb.txt
```

The FASTA header **must** be `>NAME CHAIN_ID LENGTH ...` so the chain is the first
character of the second whitespace-separated field. **Do not** write
`>R2314 Chain A` — the chain would become `C`.

### 2. Pre-submission verification

`rna_ver.py` looks for `./TARGETS/{TARGET}.pdb.txt` relative to its working
directory, where `{TARGET}` is read from a `TARGET XXX` line inside the
submission file. Workflow:

```bash
mkdir -p /tmp/casp17_check/TARGETS
cp data/casp17_data/struct/CASP17_R_template/R2314.pdb.txt /tmp/casp17_check/TARGETS/
cd /tmp/casp17_check
python3 /path/to/scripts/rna_format/rna_ver.py our_submission.pdb
```

Submission must include `TARGET R2314` near the top so the template lookup
resolves. The script checks every model (up to 5) — every atom in the template
must be present in every model, every atom name/residue/chain must match.

## CASP17 status

- R2314 (RNA aptamer, holo with Trp) — first ligand-bound RNA target ever in
  CASP. Server expiry **2026-05-06**.
- Treat the upstream Py2 DNA scripts as legacy until a DNA target appears.
