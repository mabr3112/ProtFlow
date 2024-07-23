'''Auxiliary Script to calculate heavy-atom RMSDs based on .pdb inputs.'''
# imports
import json

# dependencies
import Bio.PDB
import Bio.PDB.PDBExceptions
import pandas as pd
from Bio.PDB.Structure import Structure

# customs
from protflow.residues import ResidueSelection
from protflow.utils.biopython_tools import get_atoms, get_atoms_of_motif, load_structure_from_pdbfile

def motif_superimpose_calc_rmsd(mobile: Structure, target: Structure, mobile_atoms: ResidueSelection = None, target_atoms: ResidueSelection = None, atom_list: list[str] = None) -> Structure:
    '''Superimposes :mobile: onto :target: based on provided :mobile_atoms: and :target_atoms: If no atoms are given, superimposition is based on Structure CA.'''
    # if no motif is specified, superimpose on protein backbones.
    if (mobile_atoms is None and target_atoms is None):
        mobile_atms = get_atoms(mobile, atoms=atom_list)
        target_atms = get_atoms(target, atoms=atom_list)

    # collect atoms of motif. If only one of the motifs is specified, use the same motif for both target and mobile
    else:
        # in case heavy-atom superimposition is desired, pass 'all' for atom_list
        atom_list = None if atom_list == "all" else atom_list
        mobile_atms = get_atoms_of_motif(mobile, mobile_atoms or target_atoms, atoms=atom_list)
        target_atms = get_atoms_of_motif(target, target_atoms or mobile_atoms, atoms=atom_list)

    # superimpose and return RMSD
    super_imposer = Bio.PDB.Superimposer()
    try:
        super_imposer.set_atoms(target_atms, mobile_atms)
    except Bio.PDB.PDBExceptions.PDBException as exc:
        try:
            mob = mobile.get_parent().id
        except:
            mob = mobile.id
        try:
            tar = target.get_parent().id
        except:
            tar = target.id
        raise ValueError(f"mobile_atoms of {mob} and target_atoms of {tar} differ in length. mobile_atoms:\n{mobile_atms}\ntarget_atoms\n{target_atms}") from exc

    return super_imposer.rms

def parse_input_json(json_path: str) -> dict:
    '''Parses json input for calc_heavyatom_rmsd_batch.py'''
    def check_for_key(key: str, dict_: dict, target: str) -> None:
        if key not in dict_:
            raise KeyError(f"{key} must be specified for target in input_json. target: {target}")

    # define options
    opts = ["ref_pdb", "reference_motif", "target_motif"]

    # read
    with open(json_path, 'r', encoding="UTF-8") as f:
        input_dict = json.loads(f.read())

    # check for columns
    for target in input_dict:
        check_for_key("ref_pdb", input_dict[target], target)
        for opt in opts:
            if opt not in opts:
                input_dict[target][opt] = None

    return input_dict


def main(args):
    '''Executor'''
    # parse targets from input_json
    target_dict = parse_input_json(args.input_json)
    atoms = args.atoms.split(",") if args.atoms is not None else None

    # calc heavy-atom rmsd for each target
    df_dict = {"description": [], "location": [], "rmsd": []}
    for target in target_dict:
        opts = target_dict[target]
        rms = motif_superimpose_calc_rmsd(
            mobile = load_structure_from_pdbfile(opts["ref_pdb"]),
            target = load_structure_from_pdbfile(target),
            mobile_atoms = ResidueSelection(opts["reference_motif"]),
            target_atoms = ResidueSelection(opts["target_motif"]),
            atom_list = atoms
        )

        # collect data
        df_dict['description'].append(target.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0])
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
    argparser.add_argument("--atoms", type=str, default=None, help="List of atoms to calculate RMSD over. If nothing is specified, all heavy-atoms are taken.")

    # output
    argparser.add_argument("--output_path", type=str, default="heavyatom_rmsd.json")
    arguments = argparser.parse_args()

    # check arguments (either input_json or input_pdb + reference_pdb)
    main(arguments)