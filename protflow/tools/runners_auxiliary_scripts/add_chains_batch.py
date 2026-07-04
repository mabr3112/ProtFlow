'''Python Script to add chains into proteins as .pdb files.'''
# imports
import json
import os
from typing import Any

# dependencies:
from Bio.PDB.Structure import Structure

# customs
from protflow.residues import AtomSelection, ResidueSelection
from protflow.utils.biopython_tools import add_chain, get_atoms, get_atoms_of_atom_selection, get_atoms_of_motif, biopython_load_structure, save_structure_to_file, superimpose

def setup_superimpose_atoms(target: Structure, reference: Structure, target_motif: ResidueSelection|AtomSelection = None, reference_motif: ResidueSelection|AtomSelection = None, target_chains: str = None, reference_chains: str = None, atom_list: list[str] = None) -> tuple[list,list]:
    '''collects atoms for superimposition based on either chain or target input.'''
    def prep_chains(chain_input) -> list:
        if isinstance(chain_input, str):
            return [chain_input]
        if isinstance(chain_input, list):
            return chain_input
        if chain_input is None:
            return None
        raise ValueError(f"Unsopported Value for parameter :chains: - Only str: 'A', or list ['A', 'B'] are allowed. Type: {type(chain_input)}")

    def get_atoms_of_selection(pose: Structure, selection: ResidueSelection|AtomSelection, atoms: list[str]) -> list:
        if isinstance(selection, AtomSelection):
            return get_atoms_of_atom_selection(pose, selection)
        if isinstance(selection, ResidueSelection):
            return get_atoms_of_motif(pose, selection, atoms=atoms)
        raise TypeError(f"Unsupported motif selection type: {type(selection)}")

    # prep inputs
    atom_list = atom_list or ["N", "CA", "O"]

    # either motif or chains can be specified:
    if (target_motif or reference_motif) and (target_chains or reference_chains):
        raise ValueError("Both motif and chain are specified for superimposition. Only specify either chain or motif, but not both!")

    # parsing motifs
    if (target_motif or reference_motif):
        target_atoms = get_atoms_of_selection(target, target_motif or reference_motif, atom_list)
        reference_atoms = get_atoms_of_selection(reference, reference_motif or target_motif, atom_list)

    # parsing chains
    elif (target_chains or reference_chains):
        target_chains = prep_chains(target_chains)
        reference_chains = prep_chains(reference_chains)
        target_atoms = get_atoms(target, atoms=atom_list, chains=target_chains or reference_chains)
        reference_atoms = get_atoms(reference, atoms=atom_list, chains=reference_chains or target_chains)

    # if nothing is specified, do not superimpose!
    elif all((spec is None for spec in [target_motif, reference_motif, target_chains, reference_chains])):
        target_atoms = None
        reference_atoms = None

    else:
        raise ValueError("Impossible parameter combination reached.")

    return target_atoms, reference_atoms

def superimpose_add_chain(target: Structure, reference: Structure, copy_chain: str|list[str], target_atoms: list = None, reference_atoms: list = None, translate_x: float = None, chain_mapping: dict[str, str] = None) -> Structure:
    '''Superimposes :copy_chain: from :reference: onto :target: '''
    # if atoms specified, superimpose:
    if reference_atoms and target_atoms:
        target = superimpose(
            mobile = target,
            target = reference,
            mobile_atoms = target_atoms,
            target_atoms = reference_atoms
        )

    copy_chains = [copy_chain] if isinstance(copy_chain, str) else copy_chain
    if not isinstance(copy_chains, list) or not all(isinstance(chain, str) for chain in copy_chains):
        raise TypeError(f"copy_chain must be a chain ID or list of chain IDs. Type: {type(copy_chain)}")

    chain_mapping = chain_mapping or {}
    if not isinstance(chain_mapping, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in chain_mapping.items()):
        raise TypeError("chain_mapping must be a dictionary mapping source chain IDs to target chain IDs.")

    target_with_chain = target
    for chain in copy_chains:
        target_with_chain = add_chain(
            reference = reference,
            target = target_with_chain,
            copy_chain = chain,
            translate_x = translate_x,
            new_chain_id = chain_mapping.get(chain, chain)
        )

    return target_with_chain

def superimpose_add_chain_pdb(target_pdb: str, reference_pdb: str, copy_chain: str|list[str], target_motif: ResidueSelection|AtomSelection = None, reference_motif: ResidueSelection|AtomSelection = None, target_chains: list[str] = None, reference_chains: list[str] = None, translate_x: float = None, inplace: bool = False, output_dir: str = False, atom_list: list[str] = None, chain_mapping: dict[str, str] = None) -> str:
    '''Superimposes a chain onto a .pdb file'''
    # safety
    atom_list = atom_list or ["N", "CA", "O"]
    if not (output_dir or inplace):
        raise ValueError(f"Either :output_dir: or :inplace: parameter has to be set.")

    # load .pdbs using BioPython
    target = biopython_load_structure(target_pdb)
    reference = biopython_load_structure(reference_pdb)

    # setup superimposition atoms:
    target_atoms, reference_atoms = setup_superimpose_atoms(
        target = target,
        reference = reference,
        target_motif = target_motif,
        reference_motif = reference_motif,
        target_chains = target_chains,
        reference_chains = reference_chains,
        atom_list = atom_list,
    )

    # copy chain into target
    target_copied = superimpose_add_chain(
        target=target,
        reference=reference,
        copy_chain=copy_chain,
        target_atoms=target_atoms,
        reference_atoms=reference_atoms,
        translate_x=translate_x,
        chain_mapping=chain_mapping
    )

    # output
    if inplace:
        save_structure_to_file(target_copied, save_path=target_pdb)
        output = target_pdb

    elif output_dir:
        pdb = target_pdb.rsplit("/", maxsplit=1)[-1]
        output = os.path.join(output_dir, pdb)
        save_structure_to_file(target_copied, save_path=output)

    return output

def parse_motif_spec(motif_spec: Any) -> ResidueSelection|AtomSelection|None:
    '''Parse legacy ResidueSelection specs or AtomSelection JSON specs.'''
    if motif_spec is None:
        return None
    if isinstance(motif_spec, (ResidueSelection, AtomSelection)):
        return motif_spec
    if isinstance(motif_spec, dict):
        selection_type = motif_spec.get("selection_type") or motif_spec.get("type")
        selection = motif_spec.get("selection", motif_spec)
        if selection_type == "AtomSelection" or (isinstance(selection, dict) and "atoms" in selection):
            return AtomSelection(selection)
        if selection_type == "ResidueSelection" or (isinstance(selection, dict) and "residues" in selection):
            return ResidueSelection(selection, from_scorefile=isinstance(selection, dict) and "residues" in selection)
        if "atoms" in motif_spec:
            return AtomSelection(motif_spec)
        if "residues" in motif_spec:
            return ResidueSelection(motif_spec, from_scorefile=True)
        raise ValueError(f"Could not determine motif selection type from dictionary: {motif_spec}")
    if isinstance(motif_spec, (list, tuple)):
        try:
            return AtomSelection(motif_spec)
        except (TypeError, ValueError):
            return ResidueSelection(motif_spec)
    return ResidueSelection(motif_spec)

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
        "atoms",
        "translate_x",
        "chain_mapping"
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
            target_motif = parse_motif_spec(opts["target_motif"]),
            reference_motif = parse_motif_spec(opts["reference_motif"]),
            target_chains = opts["target_chains"],
            reference_chains = opts["reference_chains"],
            translate_x = opts["translate_x"],
            chain_mapping = opts["chain_mapping"],
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
