"""Auxiliary script to calculate atom-level RMSDs from PDB or mmCIF inputs."""
# imports
import json
import math
import os
from typing import Any

# dependencies
import Bio.PDB
import Bio.PDB.PDBExceptions
import pandas as pd
from Bio.PDB.Structure import Structure

# customs
from protflow.poses import description_from_path


def load_structure(path: str, quiet: bool = True) -> Structure:
    """Load a PDB or mmCIF structure as a BioPython Structure object."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Structure file {path} not found!")

    handle = description_from_path(path)
    lower_path = path.lower()
    if lower_path.endswith(".pdb"):
        parser = Bio.PDB.PDBParser(QUIET=quiet)
    elif lower_path.endswith((".cif", ".mmcif")):
        parser = Bio.PDB.MMCIFParser(QUIET=quiet)
    else:
        raise ValueError(f"Unsupported structure file extension for {path}. Supported extensions: .pdb, .cif, .mmcif")

    return parser.get_structure(handle, path)


def save_structure(structure: Structure, save_path: str) -> None:
    """Save a BioPython Structure object as PDB or mmCIF based on file extension."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    lower_path = save_path.lower()
    if lower_path.endswith(".pdb"):
        io = Bio.PDB.PDBIO()
    elif lower_path.endswith((".cif", ".mmcif")):
        io = Bio.PDB.MMCIFIO()
    else:
        raise ValueError(f"Unsupported structure output extension for {save_path}. Supported extensions: .pdb, .cif, .mmcif")

    io.set_structure(structure)
    io.save(save_path)


def normalize_residue_id(residue_id: Any) -> tuple[str, int, str]:
    """Normalize compact residue IDs to BioPython residue IDs."""
    if isinstance(residue_id, (list, tuple)):
        if len(residue_id) != 3:
            raise ValueError(f"BioPython residue IDs must have three elements. Got: {residue_id}")
        hetero_flag, residue_number, insertion_code = residue_id
        return (hetero_flag or " ", int(residue_number), insertion_code or " ")

    try:
        return (" ", int(residue_id), " ")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Compact atom IDs must use an integer residue ID or a BioPython residue ID tuple/list "
            f"(hetero_flag, residue_number, insertion_code). Got: {residue_id}"
        ) from exc


def normalize_atom_id(atom_id: Any) -> tuple[Any, Any]:
    """Normalize BioPython atom IDs and optional altloc identifiers."""
    if isinstance(atom_id, list):
        atom_id = tuple(atom_id)
    if isinstance(atom_id, tuple) and len(atom_id) == 2:
        return atom_id[0], atom_id[1]
    return atom_id, None


def normalize_model_id(model_id: Any) -> Any:
    """Normalize JSON-loaded model IDs where possible."""
    try:
        return int(model_id)
    except (TypeError, ValueError):
        return model_id


def atom_from_spec(structure: Structure, atom_spec: Any, default_model: int = 0) -> Bio.PDB.Atom.Atom:
    """
    Resolve an atom from either a compact atom ID or a BioPython full atom ID.

    Supported JSON-compatible atom IDs:
    - [chain_id, res_id, atom_name]
    - [model_id, chain_id, res_id, atom_name]
    - [structure_id, model_id, chain_id, res_id, atom_name]
    - [structure_id, model_id, chain_id, res_id, atom_name, altloc]

    The structure_id element is accepted for BioPython full IDs but ignored.
    """
    if not isinstance(atom_spec, (list, tuple)):
        raise TypeError(f"Atom specifications must be tuple/list-like. Got {type(atom_spec)}: {atom_spec}")

    atom_spec = list(atom_spec)
    if len(atom_spec) == 3:
        model_id = default_model
        chain_id, residue_id, atom_id = atom_spec
    elif len(atom_spec) == 4:
        model_id, chain_id, residue_id, atom_id = atom_spec
    elif len(atom_spec) == 5:
        _, model_id, chain_id, residue_id, atom_id = atom_spec
    elif len(atom_spec) == 6:
        _, model_id, chain_id, residue_id, atom_name, altloc = atom_spec
        atom_id = (atom_name, altloc)
    else:
        raise ValueError(f"Atom specifications must have 3, 4, 5, or 6 elements. Got {len(atom_spec)}: {atom_spec}")

    residue_id = normalize_residue_id(residue_id)
    atom_name, altloc = normalize_atom_id(atom_id)

    try:
        atom = structure[normalize_model_id(model_id)][chain_id][residue_id][atom_name]
    except KeyError as exc:
        raise KeyError(f"Could not resolve atom specification {atom_spec}") from exc

    if altloc not in (None, "", " ") and hasattr(atom, "disordered_select"):
        atom.disordered_select(altloc)
        atom = atom.selected_child

    return atom


def collect_atoms(structure: Structure, atom_specs: list[Any], target_name: str) -> list[Bio.PDB.Atom.Atom]:
    """Resolve a list of atom specifications for a structure."""
    if not atom_specs:
        raise ValueError(f"{target_name} must contain at least one atom specification.")
    return [atom_from_spec(structure, atom_spec) for atom_spec in atom_specs]


def calc_rmsd_without_superposition(ref_atoms: list[Bio.PDB.Atom.Atom], target_atoms: list[Bio.PDB.Atom.Atom]) -> float:
    """Calculate RMSD between paired atom lists without changing coordinates."""
    if len(ref_atoms) != len(target_atoms):
        raise ValueError(f"RMSD atom lists differ in length. ref_atoms: {len(ref_atoms)}, target_atoms: {len(target_atoms)}")
    if not ref_atoms:
        raise ValueError("Cannot calculate RMSD over zero atoms.")

    distances = [ref_atom - target_atom for ref_atom, target_atom in zip(ref_atoms, target_atoms)]
    return math.sqrt(sum(distance ** 2 for distance in distances) / len(distances))


def apply_superposition(
    target: Structure,
    reference: Structure,
    target_atom_specs: list[Any],
    ref_atom_specs: list[Any],
) -> None:
    """Superimpose target onto reference based on paired atom specifications."""
    target_atoms = collect_atoms(target, target_atom_specs, "target_superimpose_atoms")
    ref_atoms = collect_atoms(reference, ref_atom_specs, "ref_superimpose_atoms")

    if len(ref_atoms) != len(target_atoms):
        raise ValueError(
            "Superimposition atom lists differ in length. "
            f"ref_superimpose_atoms: {len(ref_atoms)}, target_superimpose_atoms: {len(target_atoms)}"
        )

    super_imposer = Bio.PDB.Superimposer()
    try:
        super_imposer.set_atoms(ref_atoms, target_atoms)
    except Bio.PDB.PDBExceptions.PDBException as exc:
        raise ValueError("Could not set superimposition atoms. Check that atom lists are paired and non-empty.") from exc
    super_imposer.apply(target.get_atoms())


def calc_atom_rmsd(
    target: Structure,
    reference: Structure,
    target_atom_specs: list[Any],
    ref_atom_specs: list[Any],
    target_superimpose_atom_specs: list[Any] = None,
    ref_superimpose_atom_specs: list[Any] = None,
) -> float:
    """Optionally superimpose target onto reference, then calculate atom-level RMSD."""
    if target_superimpose_atom_specs is not None or ref_superimpose_atom_specs is not None:
        if target_superimpose_atom_specs is None or ref_superimpose_atom_specs is None:
            raise ValueError("Both target_superimpose_atoms and ref_superimpose_atoms must be specified for superimposition.")
        apply_superposition(
            target=target,
            reference=reference,
            target_atom_specs=target_superimpose_atom_specs,
            ref_atom_specs=ref_superimpose_atom_specs,
        )

    target_atoms = collect_atoms(target, target_atom_specs, "target_atoms")
    ref_atoms = collect_atoms(reference, ref_atom_specs, "ref_atoms")
    return calc_rmsd_without_superposition(ref_atoms=ref_atoms, target_atoms=target_atoms)


def parse_bool(value: Any) -> bool:
    """Parse JSON values commonly used to represent booleans."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    raise ValueError(f"return_superimposed must be a boolean. Got: {value}")


def parse_input_json(json_path: str) -> dict[str, dict[str, Any]]:
    """Parse and validate atom-level RMSD input JSON."""
    with open(json_path, "r", encoding="UTF-8") as f:
        input_dict = json.loads(f.read())

    if not isinstance(input_dict, dict):
        raise TypeError("Input JSON must contain a dictionary mapping target structure paths to option dictionaries.")

    required_options = ["ref_path", "ref_atoms", "target_atoms"]
    optional_options = ["ref_superimpose_atoms", "target_superimpose_atoms", "return_superimposed"]
    for target_path, opts in input_dict.items():
        if not isinstance(opts, dict):
            raise TypeError(f"Options for target {target_path} must be a dictionary.")

        for option in required_options:
            if option not in opts:
                raise KeyError(f"{option} must be specified for target in input_json. target: {target_path}")

        for option in optional_options:
            opts.setdefault(option, None)

        opts["return_superimposed"] = parse_bool(opts["return_superimposed"])

    return input_dict


def superimposed_output_path(target_path: str, superimposed_out_path: str = None) -> str:
    """Return the file path where a superimposed target structure should be written."""
    if superimposed_out_path is None:
        return target_path
    return os.path.join(superimposed_out_path, os.path.basename(target_path))


def main(args) -> None:
    """Calculate atom-level RMSDs for every target in the input JSON."""
    target_dict = parse_input_json(args.input_json)

    df_dict = {"location": [], "description": [], "rmsd": []}
    for target_path, opts in target_dict.items():
        try:
            target = load_structure(target_path)
            reference = load_structure(opts["ref_path"])
            rmsd = calc_atom_rmsd(
                target=target,
                reference=reference,
                target_atom_specs=opts["target_atoms"],
                ref_atom_specs=opts["ref_atoms"],
                target_superimpose_atom_specs=opts["target_superimpose_atoms"],
                ref_superimpose_atom_specs=opts["ref_superimpose_atoms"],
            )

            location = target_path
            if opts["return_superimposed"]:
                location = superimposed_output_path(target_path, args.superimposed_out_path)
                save_structure(target, location)

        except Exception as exc:
            raise RuntimeError(f"Failed to calculate atom-level RMSD for target {target_path}") from exc

        df_dict["location"].append(location)
        df_dict["description"].append(description_from_path(target_path))
        df_dict["rmsd"].append(rmsd)

    pd.DataFrame(df_dict).to_json(args.output_path)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help=(
            ".json file mapping target structure paths to atom-level RMSD options. "
            "Expected format: {'target.pdb': {'ref_path': 'ref.pdb', 'ref_atoms': [...], 'target_atoms': [...]}}"
        ),
    )
    argparser.add_argument(
        "--superimposed_out_path",
        type=str,
        default=None,
        help="Directory for returned superimposed structures. If omitted, target files are overwritten when return_superimposed is true.",
    )
    argparser.add_argument("--output_path", type=str, default="atom_rmsd.json")
    arguments = argparser.parse_args()

    main(arguments)
