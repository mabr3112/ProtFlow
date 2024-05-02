'''Python Script to remove chains from .pdb files in batches.'''
# imports
import json

# dependencies
from Bio.PDB.Structure import Structure

# customs
from protslurm.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile

def remove_chain_from_pdb(pdb_path: str, chains: list[str]) -> Structure:
    '''Removes chain from protein given as path to a .pdb file. Returns BioPython Structure object.'''
    # load pose
    pose = load_structure_from_pdbfile(pdb_path)

    # remove chains and return
    if isinstance(chains, str):
        chains = [chains]

    for chain in chains:
        if chain not in list(pose.get_chains()):
            raise KeyError(f"Chain {chain} not found in pose. Available Chains: {list(pose.get_chains())}")
        pose.detach_child(chain)

    return pose

def main(args):
    'remove chains from .pdb files'
    # read input_json
    with open(args.input_json, 'r', encoding="UTF-8") as f:
        input_dict = json.loads(f.read())

    # remove chains and save outputs
    for pdb, chains in input_dict.items():
        # remove chain
        pose = remove_chain_from_pdb(
            pdb_path = pdb,
            chains = chains
        )

        # store according to input args:
        if args.inplace:
            new_path = pdb
        elif args.output_dir:
            new_path = f"{args.output_dir}/{pdb.rsplit('/', maxsplit=1)}"

        save_structure_to_pdbfile(pose, save_path=new_path)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # inputs
    argparser.add_argument("--input_json", type=str, required=True, help="Path to json mapping of multiple PDBs (for batch runs). Every target .pdb should be mapped to a list of chains that should be removed: {'target': ['A', 'C'], ...}")

    # outputs
    argparser.add_argument("--inplace", type=str, default="False", help="Edit .pdb files inplace and don't save them at a new location.")
    argparser.add_argument("--output_dir", type=str, help="Directory where to write the output .pdb files to.")
    arguments = argparser.parse_args()

    # check 'inplace' bool:
    arguments.inplace = arguments.inplace.lower() in ["1", "true", "yes"]

    if arguments.output_dir and arguments.inplace:
        raise ValueError(f"Both options --inplace and --output_dir were set which is not allowed. Set either --inplace, or set --output_dir")

    main(arguments)
