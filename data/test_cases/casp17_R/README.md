# casp17_R — CASP17 RNA-with-ligand targets

## Schema (RNA-specific, distinct from the protein `ensemble_inputs.csv`)

| column | meaning |
|---|---|
| `target` | Target ID (e.g. `R2314`) |
| `entity_type` | Always `rna` for this dataset; reserved for future hybrids |
| `rna_sequence` | Single-chain RNA sequence (chain A by convention) |
| `ligand_name` | Residue name from official `*.smiles.txt` (e.g. `TRP`) |
| `ligand_smiles` | Ligand SMILES from official `*.smiles.txt` |
| `experimental_affinity` | Optional, usually empty for CASP |
| `notes` | Free-form provenance |

## Targets

| target | RNA len | ligand | release | server expires |
|---|---|---|---|---|
| R2314 | **25 nt** | TRP (Tryptophan) | 2026-05-04 | 2026-05-06 |

> **R2314 sequence note** — Canonical sequence is the **25 nt**
> `CGAGGACCGGUACGGCCGCCACUCG` from
> <https://predictioncenter.org/casp17/target.cgi?target=R2314&view=sequence>
> (user-confirmed 2026-05-04). A second-hand copy we were given carried a
> **97 nt copy-paste error**, so always cross-check against the official
> `view=sequence` endpoint rather than any local mirror.

## Companion data (raw)

- FASTA: `data/casp17_data/sequences/CASP17_R/R2314.txt`
- SMILES TSV (CASP16-format, official from predictioncenter): `data/casp17_data/smiles/CASP17_R/R2314.tsv`
- Submission template (atom set, zero coords): `data/casp17_data/struct/CASP17_R_template/R2314.pdb.txt`

## Submission verification

Before uploading any predicted PDB to CASP17, run the format checker:

```bash
mkdir -p /tmp/casp17_check/TARGETS
cp data/casp17_data/struct/CASP17_R_template/R2314.pdb.txt /tmp/casp17_check/TARGETS/
cd /tmp/casp17_check
python3 $REPO/scripts/rna_format/rna_ver.py our_submission.pdb
```

The submission must contain a `TARGET R2314` header line; `rna_ver.py` reads it
to look up `./TARGETS/R2314.pdb.txt`.
