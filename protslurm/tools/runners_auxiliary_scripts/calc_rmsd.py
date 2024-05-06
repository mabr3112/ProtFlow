'''Auxiliary Script to calculate RMSDs based on .pdb inputs.'''
# imports
import json

# dependencies
import Bio.PDB
import pandas as pd

# customs
from protslurm.utils.biopython_tools import load_structure_from_pdbfile, get_atoms

def calc_rmsd(target: Bio.PDB.Structure.Structure, reference: Bio.PDB.Structure.Structure, atoms:list[str], chains:list[str]) -> float:
    '''Superimposes and calculates RMSD between target and reference for specified atoms and chains.'''
    # collect atoms
    ref_atoms = get_atoms(reference, atoms=atoms, chains=chains)
    target_atoms = get_atoms(target, atoms=atoms, chains=chains)

    # check if they are the same length.
    if len(ref_atoms) != len(target_atoms):
        raise ValueError(f"Length of collected atom lists is not the same! target_pdb: {len(target_atoms)}, reference_pdb: {len(ref_atoms)}")

    # calculate RMSD
    superimposer = Bio.PDB.Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    return superimposer.rms

def calc_rmsd_pdb(target_pdb: Bio.PDB.Structure.Structure, reference_pdb: Bio.PDB.Structure.Structure, atoms:list[str], chains:list[str]) -> float:
    '''Same as calc_rmsd, but for .pdb files as input.
    Superimposes and calculates RMSD between target and reference for specified atoms and chains.'''
    target = load_structure_from_pdbfile(target_pdb)
    reference = load_structure_from_pdbfile(reference_pdb)
    return calc_rmsd(target, reference, atoms=atoms, chains=chains)

def main(args) -> None:
    '''Calculate RMSD with commandline specifications.'''
    # parse inputs
    if args.input_json:
        with open(args.input_json, 'r', encoding="UTF-8") as f:
            poses_dict = json.loads(f.read())
    elif (args.input_pdb and args.reference_pdb):
        poses_dict = {args.input_pdb, args.reference_pdb}

    # parse atoms and chains:
    atoms = [atom.strip() for atom in args.atoms.split(",") if atom] if args.atoms else None
    chains = [chain.strip() for chain in args.chains.split(",") if chain] if args.chains else None

    # calculate rmsds for every pose in poses_dict.
    out_df_dict = {"description": [], "location": [], "rmsd": []}
    for target_pdb in poses_dict:
        out_df_dict["location"].append(target_pdb)
        out_df_dict["description"].append(target_pdb.split("/")[-1].replace(".pdb", ""))
        out_df_dict["rmsd"].append(calc_rmsd_pdb(target_pdb, poses_dict[target_pdb], atoms=atoms, chains=chains))

    # output to output_path.
    out_df = pd.DataFrame(out_df_dict)
    out_df.to_json(args.output_path)

if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input options
    argparser.add_argument("--input_pdb", type=str, help="")
    argparser.add_argument("--reference_pdb", type=str, help="")
    argparser.add_argument("--input_json", type=str, help=".json formatted file that contains a dictionary pointing to target and reference pdbs in the following way: {'target': 'reference'}")

    # setup optional specifications
    argparser.add_argument("--atoms", type=str, default="CA", help="List of atoms to calculate RMSD over. Only Backbone Atoms are recommended. E.g. --atoms='CA,CO,N'")
    argparser.add_argument("--chains", type=str, default=None, help="Specify which chains to calculate RMSD over. If not specified, RMSD will be calculated over all chains.")

    # output
    argparser.add_argument("--output_path", type=str, default="rmsd.json")
    arguments = argparser.parse_args()

    # check arguments (either input_json or input_pdb + reference_pdb)
    if arguments.input_json and (arguments.input_pdb or arguments.reference_pdb):
        raise ValueError(f"If --input_json is specified, input_pdb and reference_pdb are not allowed!")
    if not (arguments.input_json or arguments.input_pdb or arguments.reference_pdb):
        raise ValueError(f"Both --input_pdb and --reference_pdb MUST be specified for RMSD calculation if --input_json is not given!")

    main(arguments)
