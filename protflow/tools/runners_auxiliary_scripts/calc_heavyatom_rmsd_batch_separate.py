'''Auxiliary Script to calculate heavy-atom RMSDs based on .pdb inputs.'''
# imports
import json
import math
import os

# dependencies
import Bio.PDB
import Bio.PDB.PDBExceptions
import pandas as pd
from Bio.PDB.Structure import Structure

# customs
from protflow.residues import ResidueSelection
from protflow.utils.biopython_tools import get_atoms, get_atoms_of_motif, load_structure_from_pdbfile

def motif_superimpose_calc_rmsd(mobile: Structure, target: Structure, mobile_atoms: ResidueSelection = None, target_atoms: ResidueSelection = None, atom_list: list[str] = None, include_het_atoms : bool = False, rmsd_mobile_atoms: ResidueSelection = None, rmsd_target_atoms: ResidueSelection = None, rmsd_atom_list: list[str] = None, rmsd_include_het_atoms: bool = False,) -> Structure:
    '''Superimposes :mobile: onto :target: based on provided :mobile_atoms: and :target_atoms:. Calculates RMSD for :rmsd_mobile_atoms: and :rmsd_target_atoms:'''

    # if no motif is specified, superimpose on protein backbones.
    if (mobile_atoms is None and target_atoms is None):
        mobile_atms = get_atoms(mobile, atoms=atom_list, include_het_atoms=include_het_atoms)
        target_atms = get_atoms(target, atoms=atom_list, include_het_atoms=include_het_atoms)
    # collect atoms of motif. If only one of the motifs is specified, use the same motif for both target and mobile
    else:
        # in case heavy-atom superpositioning is desired, pass None for atom_list
        mobile_atms = get_atoms_of_motif(mobile, mobile_atoms or target_atoms, atoms=atom_list, include_het_atoms=include_het_atoms)
        target_atms = get_atoms_of_motif(target, target_atoms or mobile_atoms, atoms=atom_list, include_het_atoms=include_het_atoms)

    # if no motif is specified but separate_superposition_and_rmsd is True, calculate RMSD on protein backbones
    if rmsd_mobile_atoms == None and rmsd_target_atoms == None:
        rmsd_mobile_atoms = get_atoms(mobile, atoms=rmsd_atom_list, include_het_atoms=True)
        rmsd_target_atoms = get_atoms(target, atoms=rmsd_atom_list, include_het_atoms=True)
    # collect atoms of RMSD motif. If only one of the motifs is specified, use the same motif for both target and mobile
    else:
        # in case heavy-atom RMSD calculation is desired, pass None for atom_list
        rmsd_mobile_atoms = get_atoms_of_motif(mobile, rmsd_mobile_atoms or rmsd_target_atoms, atoms=rmsd_atom_list, include_het_atoms=rmsd_include_het_atoms)
        rmsd_target_atoms = get_atoms_of_motif(target, rmsd_target_atoms or rmsd_mobile_atoms, atoms=rmsd_atom_list, include_het_atoms=rmsd_include_het_atoms)

    # superimpose and return RMSD
    super_imposer = Bio.PDB.Superimposer()
    try:
        super_imposer.set_atoms(target_atms, mobile_atms)
    except Bio.PDB.PDBExceptions.PDBException as exc:
        raise ValueError(f"mobile_atoms and target_atoms differ in length. mobile_atoms:\n{mobile_atms}\ntarget_atoms\n{target_atms}") from exc

    super_imposer.rotran # no idea if this is necessary
    super_imposer.apply(rmsd_mobile_atoms)
    rmsd = calculate_rmsd_without_superposition(target=rmsd_target_atoms, reference=rmsd_mobile_atoms)
    return rmsd

def calculate_rmsd_without_superposition(target: list, reference: list):
    '''Calculates RMSD between two list of atoms without performing superposition.'''
    if not len(target) == len(reference): raise ValueError(f"Target and reference atoms for RMSD calculation without superposition differ in length. Target atoms:\n{target}\reference atoms:\n{reference}")
    distances = [target_atm - reference_atm for target_atm, reference_atm in zip(target, reference)]
    rmsd = math.sqrt(sum([dist ** 2 for dist in distances]) / len((target)))
    return round(rmsd, 3)

def parse_input_json(json_path: str) -> dict:
    '''Parses json input for calc_heavyatom_rmsd_batch.py'''
    def check_for_key(key: str, dict_: dict, target: str) -> None:
        if key not in dict_:
            raise KeyError(f"{key} must be specified for target in input_json. target: {target}")

    # define options
    opts = ["ref_pdb", "reference_motif", "target_motif", "rmsd_ref_motif", "rmsd_target_motif"]

    # read
    with open(json_path, 'r', encoding="UTF-8") as f:
        input_dict = json.loads(f.read())

    # check for columns
    for target in input_dict:
        check_for_key("ref_pdb", input_dict[target], target)
        for opt in opts:
            if opt not in input_dict[target]:
                input_dict[target][opt] = None

    return input_dict


def main(args):
    '''Executor'''
    # parse targets from input_json
    target_dict = parse_input_json(args.input_json)
    super_atoms = args.super_atoms.split(",") if args.super_atoms is not None else None
    rmsd_atoms = args.rmsd_atoms.split(",") if args.rmsd_atoms is not None else None

    # calc heavy-atom rmsd for each target
    df_dict = {"description": [], "location": [], "rmsd": []}
    for target in target_dict:
        opts = target_dict[target]
        rms = motif_superimpose_calc_rmsd(
            mobile = load_structure_from_pdbfile(opts["ref_pdb"]),
            target = load_structure_from_pdbfile(target),
            mobile_atoms = ResidueSelection(opts["reference_motif"]),
            target_atoms = ResidueSelection(opts["target_motif"]),
            atom_list = super_atoms,
            include_het_atoms = args.super_include_het_atoms,
            rmsd_mobile_atoms = ResidueSelection(opts["rmsd_ref_motif"]),
            rmsd_target_atoms= ResidueSelection(opts["rmsd_target_motif"]),
            rmsd_atom_list= rmsd_atoms,
            rmsd_include_het_atoms = args.rmsd_include_het_atoms,
        )

        # collect data
        df_dict['description'].append(os.path.splitext(os.path.basename(target))[0])
        df_dict['location'].append(target)
        df_dict['rmsd'].append(rms)

    # store scores in .json DataFrame
    pd.DataFrame(df_dict).to_json(args.output_path)

if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input options
    argparser.add_argument("--input_json", type=str, help=".json formatted file that contains a dictionary with all input information: {'target': {'reference_pdb': 'path', 'target_motif': '[...]'}}")

    # optional args
    argparser.add_argument("--super_atoms", type=str, default=None, help="List of atoms for superposition and RMSD calculation. If nothing is specified, all heavy-atoms are taken.")
    argparser.add_argument("--rmsd_atoms", type=str, default=None, help="Use atoms for superposition, but calculate RMSD only on rmsd_atoms. If nothing is specified, all heavy-atoms are taken. Only works if flag separate_superposition_and_rmsd is passed.")
    argparser.add_argument("--super_include_het_atoms", action="store_true", help="Include hetero atoms (ligands) when selecting atoms for superpositioning & RMSD calculation.")
    argparser.add_argument("--rmsd_include_het_atoms", action="store_true", help="Include hetero atoms (ligands) when selecting atoms for RMSD calculation.")

    # output
    argparser.add_argument("--output_path", type=str, default="heavyatom_rmsd.json")
    arguments = argparser.parse_args()

    # check arguments (either input_json or input_pdb + reference_pdb)
    main(arguments)
