from setuptools import find_packages, setup

# Not listed, on purpose:
#   pymol           conda-forge only (pymol-open-source); no working pip build
#   openmm, openff, openmmforcefields, pdbfixer
#                   used by utils/relax_pose.py and utils/minimal_relax.py, which
#                   run in a separate PoseBench environment, not this one
#   torch, lightning, rtmscore, rf3
#                   optional re-ranking backends, imported inside the functions
#                   that need them
setup(
    name="casp17_ligand",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # orchestration
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "rootutils",
        "pyyaml",
        "requests",
        "tqdm",
        "beartype",
        # structure and chemistry
        "rdkit",
        "numpy",
        "pandas",
        "scipy",
        "biopython",
        "biopandas",
        "prody",
        "gemmi",
        "posebusters",
    ],
)
