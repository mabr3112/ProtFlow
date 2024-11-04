'''Python Script to add chains into proteins as .pdb files.'''
# imports
import json
import os

# dependencies:
from Bio.PDB.Structure import Structure

# customs
from protflow.residues import ResidueSelection
from protflow.utils.biopython_tools import add_chain, get_atoms, get_atoms_of_motif, load_structure_from_pdbfile, save_structure_to_pdbfile, superimpose

def setup_superimpose_atoms(target: Structure, reference: Structure, target_motif: ResidueSelection = None, reference_motif: ResidueSelection = None, target_chains: str = None, reference_chains: str = None, atom_list: list[str] = None) -> tuple[list,list]:
    '''collects atoms for superimposition based on either chain or target input.'''
    def prep_chains(chain_input) -> list:
        if isinstance(chain_input, str):
            return [chain_input]
        if isinstance(chain_input, list):
            return chain_input
        if chain_input is None:
            return None
        raise ValueError(f"Unsopported Value for parameter :chains: - Only str: 'A', or list ['A', 'B'] are allowed. Type: {type(chain_input)}")

    # prep inputs
    atom_list = atom_list or ["N", "CA", "O"]

    # either motif or chains can be specified:
    if (target_motif or reference_motif) and (target_chains or reference_chains):
        raise ValueError(f"Both motif and chain are specified for superimposition. Only specify either chain or motif, but not both!")

    # if nothing is specified, do not superimpose!
    if all((spec is None for spec in [target_motif, reference_motif, target_chains, reference_chains])):
        target_atoms = None
        reference_atoms = None

    # all_atoms is specified, superimpose on all atoms!
    #if all((spec is None for spec in [target_motif, reference_motif, target_chains, reference_chains])):
    #    target_atoms = get_atoms(target, atoms=atom_list)
    #    reference_atoms = get_atoms(reference, atoms=atom_list)

    # parsing motifs
    if (target_motif or reference_motif):
        target_atoms = get_atoms_of_motif(target, target_motif or reference_motif, atoms=atom_list)
        reference_atoms = get_atoms_of_motif(reference, reference_motif or target_motif, atoms=atom_list)

    # parsing chains
    if (target_chains or reference_chains):
        target_chains = prep_chains(target_chains)
        reference_chains = prep_chains(reference_chains)
        if isinstance(target_chains, str):
            target_chains = [target_chains]
        target_atoms = get_atoms(target, atoms=atom_list, chains=target_chains or reference_chains)

    return target_atoms, reference_atoms

def superimpose_add_chain(target: Structure, reference: Structure, copy_chain: str, target_atoms: list = None, reference_atoms: list = None) -> Structure:
    '''Superimposes :copy_chain: from :reference: onto :target: '''
    # if atoms specified, superimpose:
    if reference_atoms and target_atoms:
        target = superimpose(
            mobile = target,
            target = reference,
            mobile_atoms = target_atoms,
            target_atoms = reference_atoms
        )

    # copy chain.
    target_with_chain = add_chain(
        reference = reference,
        target = target,
        copy_chain = copy_chain,
    )

    return target_with_chain

def superimpose_add_chain_pdb(target_pdb: str, reference_pdb: str, copy_chain: str, target_motif: ResidueSelection = None, reference_motif: str = None, target_chains: list[str] = None, reference_chains: list[str] = None, inplace: bool = False, output_dir: str = False, atom_list: list[str] = None) -> str:
    '''Superimposes a chain onto a .pdb file'''
    # safety
    atom_list = atom_list or ["N", "CA", "O"]
    if not (output_dir or inplace):
        raise ValueError(f"Either :output_dir: or :inplace: parameter has to be set.")

    # load .pdbs using BioPython
    target = load_structure_from_pdbfile(target_pdb)
    reference = load_structure_from_pdbfile(reference_pdb)

    # setup superimposition atoms:
    target_atoms, reference_atoms = setup_superimpose_atoms(
        target = target,
        reference = reference,
        target_motif = target_motif,
        reference_motif = reference_motif,
        target_chains = target_chains,
        reference_chains = reference_chains,
        atom_list = atom_list
    )

    # copy chain into target
    target_copied = superimpose_add_chain(
        target=target,
        reference=reference,
        copy_chain=copy_chain,
        target_atoms=target_atoms,
        reference_atoms=reference_atoms,
    )

    # output
    if inplace:
        save_structure_to_pdbfile(target_copied, save_path=target_pdb)
        output = target_pdb

    elif output_dir:
        pdb = target_pdb.rsplit("/", maxsplit=1)[-1]
        output = os.path.join(output_dir, pdb)
        save_structure_to_pdbfile(target_copied, save_path=output)

    return output

def parse_input_json(input_json: str) -> dict:
    '''Reads input json. Returns dict with None as values if input_json is None'''
    def check_for_key(key: str, dict_: dict, target: str) -> None:
        if key not in dict_:
            raise KeyError(f"{key} must be specified for target in input_json. target: {target}")

    opts = [
        "reference_pdb",
        "copy_chain",
        "target_motif",
        "reference_motif",
        "target_chains",
        "reference_chains",
        "atoms"
    ]

    # parse json file
    with open(input_json, 'r', encoding="UTF-8") as f:
        targets = json.loads(f.read())

    for target in targets:
        check_for_key("reference_pdb", targets[target], target) # targets[target] is the dict, target is the key (str)
        check_for_key("copy_chain", targets[target], target)

        # if option is not set in target dict, set it to None (simple parsing so that no KeyError is raised when accessing kwargs later):
        for opt in opts:
            if opt not in targets[target]:
                targets[target][opt] = None

    return targets

def main(args) -> None:
    "YEEEESS"
    # TODO: add logging!

    # setup options:
    poses_dict = parse_input_json(args.input_json)

    # copy chains into poses and save
    for target, opts in poses_dict.items():
        superimpose_add_chain_pdb(
            target_pdb = target,
            reference_pdb = opts["reference_pdb"],
            copy_chain = opts["copy_chain"],
            target_motif = ResidueSelection(opts["target_motif"]) if opts["target_motif"] is not None else None,
            reference_motif = ResidueSelection(opts["reference_motif"]) if opts["reference_motif"] is not None else None,
            target_chains = opts["target_chains"],
            reference_chains = opts["reference_chains"],
            inplace=args.inplace,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    import argparse

    # inputs
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="Path to json mapping of multiple PDBs (for batch runs). Every target can map any option possible for the superimposer. {'target': {'motif_chain': 'A', 'reference_chain': 'C', ...}, ...}")

    # outputs
    argparser.add_argument("--inplace", type=str, default="False")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where to write the output .pdb files to.")
    arguments = argparser.parse_args()

    # check 'inplace' bool:
    arguments.inplace = arguments.inplace.lower() in ["1", "true", "yes"]

    main(arguments)
