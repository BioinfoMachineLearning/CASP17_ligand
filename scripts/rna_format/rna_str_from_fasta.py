#!/usr/bin/env python3
"""rna_str_from_fasta.py — generate a CASP-style zero-coordinate RNA PDB template
from a FASTA sequence.

Original source: https://predictioncenter.org/download_area/other/rna-dna/rna_str_from_fasta.py
Copyright (c) 2013, IBMC, CNRS — Author: Miao Zhichao
Ported from Python 2 to Python 3 for use in CASP17_ligand pipeline.

Usage:
    python rna_str_from_fasta.py <fasta_file> [num_models] > <output.pdb>

The generated PDB has every required atom for each residue with coordinates set
to 0.000. Used as the reference atom set for `rna_ver.py` format-checks before
CASP submission.

FASTA header convention expected:
    >NAME CHAIN_ID LENGTH ...
e.g. `>R2314 A 25 RNA Tryptophan binding aptamer`

The chain identifier is taken from the **first character** of the second
whitespace-separated field. Avoid headers like `>R2314 Chain A` — that would
yield chain `C`.
"""

import sys

USAGE = """rna_str_from_fasta.py usage:

input a RNA sequence (fasta format), output the standard PDB format

./rna_str_from_fasta.py fasta.file [number_of_model=1] >output.pdb

fasta.file example:
>RNA1 A length1
UGCGAUGAGAAGAAGAGUAUUAAGGAUUUACUAUGAUUAGCGACUCUAGGAUAGUGAAAG
     CUAGAGGAUAGUAACCUUAAGAAGGCACUUCGAGCA
>RNA2 B length2
GCGGAAGUAGUUCAGUGGUAGAACACCACCUUGCCAAGGUGGGGGUCGCGGGUUCGAAUC
     CCGUCUUCCGCUCCA
"""

A_TEMPLATE = """ATOM  %5d  P     A %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'   A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'   A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'   A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'   A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2'   A %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C1'   A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N9    A %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C8    A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N7    A %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5    A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6    A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N6    A %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  N1    A %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2    A %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N3    A %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4    A %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

G_TEMPLATE = """ATOM  %5d  P     G %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'   G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'   G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'   G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'   G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2'   G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C1'   G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N9    G %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C8    G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N7    G %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5    G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6    G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O6    G %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N1    G %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2    G %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N2    G %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  N3    G %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4    G %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

U_TEMPLATE = """ATOM  %5d  P     U %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'   U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'   U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'   U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'   U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2'   U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C1'   U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N1    U %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2    U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2    U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N3    U %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4    U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4    U %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5    U %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6    U %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""

C_TEMPLATE = """ATOM  %5d  P     C %c%4d       0.000   0.000   0.000  1.00  0.00           P
ATOM  %5d  OP1   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  OP2   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  O5'   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C5'   C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C4'   C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O4'   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C3'   C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O3'   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C2'   C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2'   C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  C1'   C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N1    C %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C2    C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  O2    C %c%4d       0.000   0.000   0.000  1.00  0.00           O
ATOM  %5d  N3    C %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C4    C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  N4    C %c%4d       0.000   0.000   0.000  1.00  0.00           N
ATOM  %5d  C5    C %c%4d       0.000   0.000   0.000  1.00  0.00           C
ATOM  %5d  C6    C %c%4d       0.000   0.000   0.000  1.00  0.00           C
"""


def readfasta(fp):
    """Read a multi-record FASTA. Returns (chains, seqs)."""
    chains = []
    seqs = []
    seq = ""
    with open(fp) as f:
        for line in f:
            if len(line) < 2:
                continue
            if line[0] == "#":
                continue
            if line[0] == ">":
                a = line.strip().split()
                if len(a) < 2:
                    print(USAGE)
                    sys.exit(0)
                chains.append(a[1][0])
                if len(seq) > 0:
                    seqs.append(seq)
                seq = ""
            else:
                seq += line.strip().upper()
    seqs.append(seq)
    return chains, seqs


def prepare_model(chains, seqs):
    n = 1  # atom serial number
    temp_map = {"A": A_TEMPLATE, "U": U_TEMPLATE, "C": C_TEMPLATE, "G": G_TEMPLATE}
    number_map = {"A": 22, "U": 20, "C": 20, "G": 23}
    out = ""
    for chain, seq in zip(chains, seqs):
        rsn = 0  # residue serial number
        last_res = ""
        for k, base in enumerate(seq):
            rsn += 1
            xx = []
            for j in range(number_map.get(base, 0)):
                xx.extend([n, chain, rsn])
                n += 1
            tmpl = temp_map.get(base, None)
            if tmpl is not None:
                out += tmpl % tuple(xx)
                last_res = base
        out += "TER   %5d        %c %c%4d                      \n" % (n, last_res, chain, rsn)
        n += 1
    return out


def format_pdb(fp, num=1):
    """Write a CASP-format PDB template to stdout. `num` = number of MODEL blocks."""
    chains, seqs = readfasta(fp)
    out = ""
    for i in range(num):
        out += "MODEL       %2d                                              \n" % (i + 1)
        out += prepare_model(chains, seqs)
        out += "ENDMDL                                                      \n"
    out += "END                                                        \n"
    sys.stdout.write(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)
    elif len(sys.argv) > 2:
        format_pdb(sys.argv[1], int(sys.argv[2]))
    else:
        format_pdb(sys.argv[1])
