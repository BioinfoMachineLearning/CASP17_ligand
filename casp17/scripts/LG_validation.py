#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:29:23 2022

@author: Aaron Sweeney
@contributor: Xavier Robin
"""
from __future__ import print_function

import os
from rdkit import Chem
from io import BytesIO
from rdkit.Chem import rdFMCS
import sys
from collections import Counter

LOG_PATH = os.environ.get("LG_LOG_PATH", "./logs/")
TARGETS_PATH = os.environ.get("LG_TARGETS_PATH", "./TARGETS/")
if not os.path.isdir(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)



class LG:
    def __init__(self, file):

        with open(file, "r") as f:
            data = f.read().splitlines()

        # get PFRMAT
        if data[0].startswith("PFRMAT"):
            self.PFRMAT = data[0].split()[1]
        else:
            print("# ERROR! PFRMAT record not found !!")
            sys.exit()

        # get TARGET
        if data[1].startswith("TARGET"):
            self.TARGET = data[1].split()[1]
            if len(self.TARGET) == 0:
                print("# ERROR! TARGET not specified !!")
                sys.exit()
        else:
            print("# ERROR! TARGET record not found !!")
            sys.exit()

        # get Author
        if data[2].startswith("AUTHOR"):
            self.AUTHOR = data[2].split()[1]
            if len(self.AUTHOR) == 0:
                print("# ERROR! Author record not found !!")
                sys.exit()

        else:
            print("# ERROR! Author record not found !!")
            sys.exit()

        self.METHOD = ""
        self.MODELS = []

        mdl_idx = []

        for index, line in enumerate(data):
            if line.startswith("METHOD"):
                self.METHOD += line.replace("METHOD", "").strip() + "\n"
            elif line.startswith("MODEL") or line.startswith("END"):
                mdl_idx.append(index)

        if not len(mdl_idx) % 2 == 0:
            print("# ERROR! Missing MODEL/END records !!")
            sys.exit()

        else:
            for i in range(0, len(mdl_idx), 2):
                self.MODELS.append(Model(data[mdl_idx[i] : mdl_idx[i + 1] + 1]))

        self.VALIDATION_REPORT = None
        self.OUT_FILE = os.path.join(LOG_PATH, "{}_{}.log".format(self.AUTHOR, self.TARGET))

    def write_report(self):
        if self.VALIDATION_REPORT is None:
            print("# ERROR! Validation not run !!")

        else:
            with open(self.OUT_FILE, "w") as f:
                f.writelines(self.VALIDATION_REPORT)


class Model:
    def __init__(self, mdl_data):
        self.AFFNTY = None
        self.LIGANDS = []
        ligand_start_idx = []
        ligand_end_idx = []


        for i, line in enumerate(mdl_data):
            if len(ligand_start_idx) != len(ligand_end_idx):
                # We are within the coordinates block, look for a model end
                if line.replace(" ", "") == "MEND":
                    ligand_end_idx.append(i)
            else:
                if line.startswith("AFFNTY"):
                    if i != len(mdl_data) - 2:
                        print('# ERROR! AFFNTY record should occur on the last line before the END of a model !!')
                        sys.exit()
                    try:
                        _, aff_value, aff_type = line.strip().split(None, 2)
                    except Exception:
                        print('# ERROR! Incorrect AFFNTY record format: "%s" !!' % line)
                        sys.exit()
                    else:
                        if aff_type in {"aa", "ra"}:
                            try:
                                self.AFFNTY = float(aff_value)
                            except Exception:
                                print('# ERROR! Incorrect AFFNTY record value: %s !!' % aff_value)
                                sys.exit()
                        elif aff_type == "lr":
                            try:
                                self.AFFNTY = int(aff_value)
                            except Exception:
                                print('# ERROR! Incorrect ligand rank AFFNTY record value: %s !!' % aff_value)
                                sys.exit()
                        else:
                            print('# ERROR! Incorrect AFFNTY record type: %s !!' % aff_type)
                            sys.exit()
                elif line.startswith("LIGAND"):
                    ligand_start_idx.append(i)
                elif line.startswith("LSCORE"):
                    print("# ERROR! LSCORE must appear after LIGAND and before the MDL coordinates block !!")
                    sys.exit()
                elif line.startswith("REMARK"):
                    pass
                elif line.startswith("POSE"):
                    pose_id = int(line.split(None, 1)[1])
                    if pose_id != 1:
                        print(
                            "# ERROR! Invalid record %s: Only 1 pose is allowed in CASP16 !!"
                            % line
                        )
                        sys.exit()
                elif line.startswith("END"):
                    # assume it's the last - no checks
                    pass
                elif line.startswith("MODEL"):
                    # assume it's the first - no checks
                    pass
                # ── PATCH (CASP17 LG block 1: receptor coordinates) ──────
                # The CASP17 LG format spec mandates a CASP-TS-style receptor
                # block (PARENT + ATOM/HETATM + TER) before the LIGAND block.
                # The original LG_validation.py only validates the MDL section
                # (its job: "checking format of MDL sections"); receptor format
                # is delegated to ts_ver_lig. We silently accept those records
                # here so we can run LG_validation.py on the full LG file
                # without first stripping the receptor block.
                elif line.startswith(("ATOM", "HETATM", "PARENT", "TER")):
                    pass
                # ── END PATCH ────────────────────────────────────────────
                elif not line.strip():
                    # blank line — also tolerated (separators between blocks)
                    pass
                else:
                    print("# ERROR! Unknown record %s !!" % line)
                    sys.exit()

        if len(ligand_start_idx) != len(ligand_end_idx):
            print('# ERROR! Found an unequal number of LIGAND and M END records !!')
            sys.exit()

        for start_idx, end_idx in zip(ligand_start_idx, ligand_end_idx):
            self.LIGANDS.append(Ligand(mdl_data[start_idx:end_idx + 1]))


class Ligand:
    def __init__(self, ligand_data):
        self.CONFORMATIONS = []
        self.NUM_CONFORMATIONS = 0
        self.LIG_ID_NUM = None
        self.LIG_ID_CODE = None
        self.LIG_SCORE = None

        if ligand_data[0].startswith("LIGAND"):
            try:
                self.LIG_ID_NUM = ligand_data[0].split(" ")[1]
                self.LIG_ID_CODE = ligand_data[0].split(" ")[2]
            except Exception as e:
                print("# ERROR! Incorrect LIGAND record format !!")
                sys.exit()
        else:
            print("# ERROR! Missing LIGAND records !!")
            sys.exit()

        mol_block = []
        for line in ligand_data[1:]:
            if line.startswith("POSE"):
                pose_id = int(line.split(None, 1)[1])
                if pose_id != 1:
                    print(
                        "# ERROR! Invalid record %s: Only 1 pose is allowed in CASP16 !!"
                        % line
                    )
                    sys.exit()
            elif line.startswith("LSCORE"):
                try:
                    self.LIG_SCORE = float(line.split(None, 1)[1])
                except Exception as e:
                    print('# ERROR! Incorrect LSCORE record: "%s" !!' % line)
                    sys.exit()
                else:
                    if self.LIG_SCORE < 0 or self.LIG_SCORE > 1:
                        print(
                            '# ERROR! LSCORE must be in the interval [0,1], not: "%s" !!'
                            % self.LIG_SCORE
                        )
                        sys.exit()
            elif line.replace(" ", "") == "MEND":
                mol_block.append(line)
                mol = self.get_mol_from_block(mol_block)
                self.CONFORMATIONS.append(mol)
                mol_block = []
            else:
                mol_block.append(line)

        self.NUM_CONFORMATIONS = len(self.CONFORMATIONS)
        self.VALIDATION = [list() for i in range(self.NUM_CONFORMATIONS)]

        if self.NUM_CONFORMATIONS == 0:
            print("# ERROR! Missing M END records for ligand %s!!" % self.LIG_ID_NUM)
            sys.exit()
        elif self.NUM_CONFORMATIONS > 1:
            # This should never happen given the design of this script
            print(
                "# ERROR! Only 1 conformation allowed for ligand %s, found %s!!"
                % (self.LIG_ID_NUM, self.NUM_CONFORMATIONS)
            )
            sys.exit()

    def get_mol_from_block(self, block):

        block = [i for i in block if not i.startswith("REMARK")]
        block.append("$$$$")
        block = "\n".join(block)
        mol = None
        try:
            with BytesIO(block.encode("UTF-8")) as hnd:
                for mol in Chem.ForwardSDMolSupplier(hnd):
                    pass
        except:
            print(
                "# ERROR! reading ligand coords: {} {} !!".format(
                    self.LIG_ID_NUM, self.LIG_ID_CODE
                )
            )
            sys.exit()
        if mol is None:
            print(
                "# ERROR! reading ligand coords: {} {} !!".format(
                    self.LIG_ID_NUM, self.LIG_ID_CODE
                )
            )
            sys.exit()
        mol = Chem.RemoveAllHs(mol)
        return mol


def validate_ligands(LG_obj):

    smiles = find_smiles(LG_obj.TARGET)
    tasks = set(find_task(LG_obj.TARGET).values())
    
    for mdl in LG_obj.MODELS:
        if len(mdl.LIGANDS) == 0 and mdl.AFFNTY is None:
            print("# ERROR! No LIGAND and no AFFNTY records found !!")
            sys.exit()
        elif "A" not in tasks and "PA" not in tasks and len(mdl.LIGANDS) == 0:
            print("# ERROR! No LIGAND found for Pose-only target !!")
            sys.exit()

        for lig in mdl.LIGANDS:
            lig_id = int(lig.LIG_ID_NUM)
            validation_mol = Chem.MolFromSmiles(smiles[lig_id])
            validation_mol = Chem.RemoveAllHs(validation_mol)
            if validation_mol is None:
                print("# ERROR! Validation mol is None !!")
                sys.exit()

            for num, pose in enumerate(lig.CONFORMATIONS):
                valid_atoms = compare_atoms(validation_mol, pose, lig_id)
                lig.VALIDATION[num].append(valid_atoms)

                valid_bonds = compare_bonds(validation_mol, pose, lig_id)
                lig.VALIDATION[num].append(valid_bonds)

                valid_connection = maximum_common_substructure(
                    validation_mol, pose, lig_id
                )
                lig.VALIDATION[num].append(valid_connection)


def maximum_common_substructure(ground_truth_mol, comparison_mol, lig_id):

    """
    finds the maximum common substructure and compares the number of atoms to a given mol
    """

    res = rdFMCS.FindMCS([comparison_mol, ground_truth_mol])

    if ground_truth_mol.GetNumAtoms() == 1 and comparison_mol.GetNumAtoms() == 1:
        return True

    if (
        res.numAtoms == ground_truth_mol.GetNumAtoms()
        and res.numAtoms == comparison_mol.GetNumAtoms()
    ):
        return True
    else:
        print("# ERROR! Topology not valid for ligand %s !!" % lig_id)
        return False


def get_bonds_dic(mol):

    """
    makes a Counter object of the number of bonds in a mol
    """

    all_bonds = []
    for i in mol.GetBonds():
        # may be two restrictive
        bond_type = sorted([i.GetBeginAtom().GetSymbol(), i.GetEndAtom().GetSymbol()])
        bond_type = bond_type[0] + bond_type[1] + str(i.GetBondType())
        all_bonds.append(bond_type)
    return Counter(all_bonds)


def compare_bonds(ground_truth_mol, comparison_mol, lig_id):
    """
    compares the number of atoms between two or more molecules
    """

    gt_bonds = get_bonds_dic(ground_truth_mol)
    mol_bonds = get_bonds_dic(comparison_mol)
    if sorted(gt_bonds.items(), key=lambda x: x[0]) != sorted(
        mol_bonds.items(), key=lambda x: x[0]
    ):
        print("# ERROR! Bonds not valid for ligand %s !!" % lig_id)
        return False
    return True


def compare_atoms(validation_mol, mol, lig_id):

    gt_atoms = Counter([i.GetSymbol() for i in validation_mol.GetAtoms()])
    mol_atoms = Counter([i.GetSymbol() for i in mol.GetAtoms()])
    if sorted(gt_atoms.items(), key=lambda x: x[0]) != sorted(
        mol_atoms.items(), key=lambda x: x[0]
    ):
        print("# ERROR! Atoms not valid for ligand %s !!" % lig_id)
        return False
    return True


def read_smiles(target):
    smiles_fn = os.path.join(TARGETS_PATH, target + ".smiles.txt")
    with open(smiles_fn, "r") as f:
        smiles = f.read().splitlines()
    return smiles


def find_smiles(target):
    try:
        smiles_lines = read_smiles(target)
        smiles = {int(i.split()[0]): i.split()[2] for i in smiles_lines[1:]}
    except Exception as err:
        print("# ERROR! Target data file is invalid (SMILES), get in touch with the Prediction Center !!")
        sys.exit()
    return smiles


def find_task(target):
    try:
        smiles_lines = read_smiles(target)
        tasks = {int(i.split()[0]): i.split()[3] for i in smiles_lines[1:]}
    except Exception as err:
        print("# ERROR! Target data file is invalid (Task), get in touch with the Prediction Center !!")
        sys.exit()
    else:
        for task in tasks.values():
            if task not in {"P", "A", "PA"}:
                print("# ERROR! Target data file Task data is invalid, get in touch with the Prediction Center !!")
                sys.exit()

    return tasks


def make_validation_report(LG_obj):

    report = [
        "----VALIDATION REPORT----\n",
        "AUTHOR : {}\n".format(LG_obj.AUTHOR),
        "TARGET : {}\n".format(LG_obj.TARGET),
        "METHOD : {}\n".format(LG_obj.METHOD),
        "-------------------------\n",
        "NUM_MODELS : {}\n".format(len(LG_obj.MODELS)),
    ]
    if len(LG_obj.MODELS) > 5:
        report.append("# ERROR! NUM_MODELS greater than allowed !!\n")

    for num, mdl in enumerate(LG_obj.MODELS):
        report.append("MODEL :  {}\n".format(num + 1))
        report.append("NUM_LIGANDS :  {}\n".format(len(mdl.LIGANDS)))
        report.append("AFFNTY :  {}\n".format(mdl.AFFNTY))

        for lig in mdl.LIGANDS:
            report.append(
                "\n\tLIGAND : {} {}\n".format(lig.LIG_ID_CODE, lig.LIG_ID_NUM)
            )
            report.append("\tLSCORE : {}\n".format(lig.LIG_SCORE))
            if len(lig.CONFORMATIONS) > 5:
                report.append("\t# ERROR! NUM_POSES greater than allowed !!\n")
            validation_report = lig.VALIDATION
            for num, pose in enumerate(lig.CONFORMATIONS):
                if validation_report[num][0] == True:
                    report.append("\t\tATOMS VALID\n")
                elif validation_report[num][0] == False:
                    report.append("\t\t# ERROR! Invalid number of atoms !!\n")

                if validation_report[num][1] == True:
                    report.append("\t\tBONDS VALID\n")
                elif validation_report[num][1] == False:
                    report.append("\t\t# ERROR! Bonds invalid !!\n")

                if validation_report[num][2] == True:
                    report.append("\t\tTOPOLOGY VALID\n")
                elif validation_report[num][2] == False:
                    report.append("\t\t# ERROR! Invalid topology !!\n")

    LG_obj.VALIDATION_REPORT = report


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage python LG_validation.py <LG file>")
    test_txt = sys.argv[1]
    try:
        a = LG(test_txt)
        validate_ligands(a)
        make_validation_report(a)
        a.write_report()
    except Exception:
        print("# ERROR! The validation script crashed, get in touch with the Prediction Center !!")
        raise
