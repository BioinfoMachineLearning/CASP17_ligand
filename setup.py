from setuptools import find_packages, setup

setup(
    name="casp17_ligand",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "pandas",
        "rootutils",
    ],
)
