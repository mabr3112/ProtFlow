"""
residues
========

The `residues` module is a part of the `protflow` package and is designed to handle residue selection and related operations in protein structures. This module provides functionality to parse, manipulate, and convert residue selections in various formats, making it an essential tool for bioinformatics and computational biology workflows.

The module includes the `ResidueSelection` class for representing and manipulating selections of residues, as well as various functions for parsing and converting residue selections.

Classes
-------

- `ResidueSelection`
    Represents a selection of residues with functionality for parsing, converting, and manipulating selections.
- `AtomSelection`
    Represents an ordered selection of atoms for atom-level operations.

Functions
---------

- `fast_parse_selection`
    Fast parser for selections already in `ResidueSelection` format.
- `parse_selection`
    Parses a selection into `ResidueSelection` formatted selection.
- `parse_residue`
    Parses a single residue identifier into a tuple (chain, residue_index).
- `residue_selection`
    Creates a `ResidueSelection` from a selection of residues.
- `from_dict`
    Creates a `ResidueSelection` object from a dictionary specifying a motif.
- `from_contig`
    Creates a `ResidueSelection` object from a contig string.
- `reduce_to_unique`
    Reduces an input array to its unique elements while preserving order.

Example Usage
-------------

Creating and manipulating `ResidueSelection` objects:

.. code-block:: python
    
    from residues import ResidueSelection, from_dict, from_contig

    # Create a ResidueSelection from a list
    selection = ResidueSelection(["A1", "A2", "B3"])

    # Convert to string
    selection_str = selection.to_string()
    print(selection_str)
    # Output: A1, A2, B3

    # Convert to dictionary
    selection_dict = selection.to_dict()
    print(selection_dict)
    # Output: {'A': [1, 2], 'B': [3]}

    # Create a ResidueSelection from a dictionary
    selection_from_dict = from_dict({"A": [1, 2], "B": [3]})
    print(selection_from_dict.to_string())
    # Output: A1, A2, B3

    # Create a ResidueSelection from a contig string
    selection_from_contig = from_contig("A1-A3, B5")
    print(selection_from_contig.to_string())
    # Output: A1, A2, A3, B5

This module simplifies the process of handling residue selections in bioinformatics workflows, providing a consistent interface for different types of input and output formats.
"""
# imports
from collections import OrderedDict, defaultdict
import os
import re
from typing import Any, TypeAlias

AtomID: TypeAlias = tuple[Any, ...]

RFD3_INPUT_SELECTION_FIELDS = (
    "contig",
    "unindex",
    "select_fixed_atoms",
    "select_unfixed_sequence",
    "fixed_motif_atoms",
    "fixed_motif_atoms_with_ligand",
    "select_buried",
    "select_partially_buried",
    "select_exposed",
    "select_hbond_donor",
    "select_hbond_acceptor",
    "select_hotspots",
)

_RFD3_DERIVED_INPUT_SELECTION_FIELDS = (
    "fixed_motif_atoms",
    "fixed_motif_atoms_with_ligand",
)

_RFD3_BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Foundry/RFD3 shorthand for sidechain "tip" atoms. Kept local so ProtFlow
# does not need Foundry installed just to parse input specifications.
_RFD3_TIP_ATOMS_BY_RESNAME = {
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TYR": ("CZ", "OH"),
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "GLN": ("CG", "CD", "OE1", "NE2"),
    "GLU": ("CG", "CD", "OE1", "OE2"),
    "CYS": ("CB", "SG"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "VAL": ("CG1", "CG2"),
    "ILE": ("CB", "CG2"),
    "MET": ("SD", "CE"),
    "LYS": ("CE", "NZ"),
    "ARG": ("CD", "NE", "CZ", "NH1", "NH2"),
    "PRO": None,
    "ALA": None,
    "GLY": None,
    "UNK": None,
    "MSK": None,
}

_RFD3_RESIDUE_RANGE_RE = re.compile(r"^([A-Za-z])(\d+)(?:-([A-Za-z]?)(\d+))?$")


def _validate_residue_atom_id(residue_id: Any) -> None:
    """Validate compact or BioPython residue ID formats used inside atom IDs."""
    if isinstance(residue_id, int):
        return
    if isinstance(residue_id, str):
        try:
            int(residue_id)
            return
        except ValueError as exc:
            raise ValueError(f"Residue ID strings must be integer-like. Got: {residue_id}") from exc
    if isinstance(residue_id, (list, tuple)) and len(residue_id) == 3:
        try:
            int(residue_id[1])
            return
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BioPython residue IDs must contain an integer-like residue number. Got: {residue_id}") from exc
    raise ValueError(
        "Residue IDs in atom specifications must be integer-like or BioPython residue IDs "
        f"(hetero_flag, residue_number, insertion_code). Got: {residue_id}"
    )


def _validate_atom_name(atom_name: Any) -> None:
    """Validate compact atom names and BioPython atom IDs."""
    if isinstance(atom_name, str) and atom_name:
        return
    if isinstance(atom_name, (list, tuple)) and len(atom_name) == 2 and isinstance(atom_name[0], str) and atom_name[0]:
        return
    raise ValueError(f"Atom names must be non-empty strings or BioPython atom IDs like (atom_name, altloc). Got: {atom_name}")


def _validate_atom_id(atom_id: Any) -> None:
    """Validate compact or BioPython full atom IDs."""
    if not isinstance(atom_id, (list, tuple)):
        raise TypeError(f"Atom IDs must be tuple/list-like. Got {type(atom_id)}: {atom_id}")

    atom_id = list(atom_id)
    if len(atom_id) == 3:
        chain_id, residue_id, atom_name = atom_id
    elif len(atom_id) == 4:
        _, chain_id, residue_id, atom_name = atom_id
    elif len(atom_id) == 5:
        _, _, chain_id, residue_id, atom_name = atom_id
    elif len(atom_id) == 6:
        _, _, chain_id, residue_id, atom_name, _ = atom_id
    else:
        raise ValueError(
            "Atom IDs must have 3 compact elements (chain_id, res_id, atom_name), "
            "4 elements (model_id, chain_id, res_id, atom_name), "
            "5 BioPython full-id elements, or 6 full-id-plus-altloc elements. "
            f"Got {len(atom_id)} elements: {atom_id}"
        )

    if not isinstance(chain_id, str) or not chain_id:
        raise ValueError(f"Atom ID chain identifiers must be non-empty strings. Got: {chain_id}")
    _validate_residue_atom_id(residue_id)
    _validate_atom_name(atom_name)


def _looks_like_single_atom_id(selection: Any) -> bool:
    """Return True if selection itself is one atom ID rather than a sequence of atom IDs."""
    try:
        _validate_atom_id(selection)
    except (TypeError, ValueError):
        return False
    return True


def _as_tuple_recursive(value: Any) -> Any:
    """Convert nested JSON-style lists to tuples while preserving scalar values."""
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuple_recursive(item) for item in value)
    return value


def _as_list_recursive(value: Any) -> Any:
    """Convert nested tuples to JSON-friendly lists while preserving scalar values."""
    if isinstance(value, (list, tuple)):
        return [_as_list_recursive(item) for item in value]
    return value


def _normalize_atom_selection_value(selection: Any, parameter_name: str) -> tuple[AtomID, ...]:
    """Validate and normalize one ordered atom selection."""
    if selection.__class__.__name__ == "AtomSelection" and callable(getattr(selection, "to_list", None)):
        selection = selection.to_list()
    if isinstance(selection, dict) and "atoms" in selection:
        selection = selection["atoms"]

    if not isinstance(selection, (list, tuple)):
        raise TypeError(
            f"{parameter_name} must be an AtomSelection object or an ordered tuple/list of atom IDs. "
            f"Got {type(selection)}: {selection}"
        )

    atom_ids = (selection,) if _looks_like_single_atom_id(selection) else selection

    normalized = []
    for atom_id in atom_ids:
        try:
            _validate_atom_id(atom_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atom ID in {parameter_name}: {atom_id}") from exc
        normalized.append(_as_tuple_recursive(atom_id))
    return tuple(normalized)


def _load_biopython_structure(path: str, quiet: bool = True):
    """Load a PDB or mmCIF structure locally to avoid importing ProtFlow utilities."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Structure file {path} not found.")

    import Bio.PDB

    handle = os.path.splitext(os.path.basename(path))[0]
    lower_path = path.lower()
    if lower_path.endswith(".pdb"):
        parser = Bio.PDB.PDBParser(QUIET=quiet)
    elif lower_path.endswith((".cif", ".mmcif")):
        parser = Bio.PDB.MMCIFParser(QUIET=quiet)
    else:
        raise ValueError(f"Unsupported structure file extension for {path}. Supported extensions: .pdb, .cif, .mmcif")
    return parser.get_structure(handle, path)


def _normalize_biopython_entity(pose: Any, model_id: Any = 0):
    """Return a BioPython entity from a path or an already loaded BioPython entity."""
    if pose is None:
        return None
    if isinstance(pose, os.PathLike):
        pose = os.fspath(pose)
    if isinstance(pose, str):
        return _load_biopython_structure(pose)
    if hasattr(pose, "get_level") and callable(pose.get_level):
        return pose
    raise TypeError(
        "pose must be a path to a PDB/mmCIF file or a BioPython Structure, Model, Chain, or Residue. "
        f"Got {type(pose)}: {pose}"
    )


def _iter_biopython_residues(entity: Any, model_id: Any = 0):
    """Iterate residues from a BioPython Structure, Model, Chain, or Residue."""
    level = entity.get_level()
    if level == "S":
        try:
            model = entity[model_id]
        except KeyError:
            model = next(entity.get_models())
        yield from model.get_residues()
    elif level in {"M", "C"}:
        yield from entity.get_residues()
    elif level == "R":
        yield entity
    else:
        raise ValueError(f"Cannot create AtomSelection from BioPython entity level '{level}'.")


def _residue_chain_id(residue: Any) -> str:
    """Return the chain identifier for a BioPython residue."""
    return residue.get_parent().id


def _format_biopython_residue_id(residue: Any, residue_id_format: str = "auto") -> int | tuple[Any, ...]:
    """Format residue IDs as compact integers or full BioPython residue IDs."""
    residue_id_format = residue_id_format.lower()
    if residue_id_format not in {"auto", "compact", "biopython"}:
        raise ValueError("residue_id_format must be one of 'auto', 'compact', or 'biopython'.")

    hetero_flag, residue_number, insertion_code = residue.id
    if residue_id_format == "compact":
        return int(residue_number)
    if residue_id_format == "auto" and hetero_flag == " " and insertion_code == " ":
        return int(residue_number)
    return residue.id


def _atom_id_from_biopython_atom(atom: Any, residue_id_format: str = "auto") -> AtomID:
    """Convert a BioPython Atom to a compact AtomSelection atom ID."""
    residue = atom.get_parent()
    atom_name = atom.get_id()
    altloc = atom.get_altloc() if hasattr(atom, "get_altloc") else None
    if altloc not in (None, "", " "):
        atom_name = (atom_name, altloc)
    return (_residue_chain_id(residue), _format_biopython_residue_id(residue, residue_id_format), atom_name)


def _parse_rfd3_residue_component(component: str) -> list[tuple[str, int]] | None:
    """Parse an RFD3 residue component such as A1, A1-3, or A1-A3."""
    match = _RFD3_RESIDUE_RANGE_RE.fullmatch(component.strip())
    if not match:
        return None

    chain, start, end_chain, end = match.groups()
    start_idx = int(start)
    end_idx = int(end or start)
    if end_chain and end_chain != chain:
        raise ValueError(f"RFD3 component ranges must stay on one chain. Got: {component}")
    if end_idx < start_idx:
        raise ValueError(f"RFD3 component ranges must be increasing. Got: {component}")
    return [(chain, resi) for resi in range(start_idx, end_idx + 1)]


def _is_rfd3_gap_component(component: str) -> bool:
    """Return True for RFD3 generated-length or chain-break components."""
    component = component.strip()
    if not component:
        return True
    if component.startswith("/"):
        return True
    if re.fullmatch(r"\d+(?:-\d+)?", component):
        return True
    return False


def _rfd3_components_from_string(selection: str) -> list[str]:
    """Expand an RFD3 contig/InputSelection string to residue components and names."""
    components = []
    for part in [part.strip() for part in selection.split(",")]:
        if _is_rfd3_gap_component(part):
            continue
        residue_components = _parse_rfd3_residue_component(part)
        if residue_components is not None:
            components.extend(f"{chain}{resi}" for chain, resi in residue_components)
        else:
            components.append(part)
    return components


def _atom_names_from_rfd3_value(atom_names: Any, residue: Any = None) -> tuple[str, ...] | None:
    """Expand an RFD3 atom-name value to explicit names, or None for ALL."""
    if isinstance(atom_names, (list, tuple)):
        names = tuple(str(name).strip() for name in atom_names)
    elif isinstance(atom_names, str):
        stripped = atom_names.strip()
        upper = stripped.upper()
        if upper == "ALL":
            return None
        if upper == "BKBN":
            return _RFD3_BACKBONE_ATOMS
        if upper == "TIP":
            if residue is None:
                raise ValueError("RFD3 TIP atom selection requires a pose so residue names can be inspected.")
            residue_name = residue.get_resname().strip()
            names = _RFD3_TIP_ATOMS_BY_RESNAME.get(residue_name)
            if names is None:
                raise ValueError(f"Residue {residue_name} does not define RFD3 TIP atoms.")
            return tuple(names)
        if stripped == "":
            return ()
        names = tuple(name.strip() for name in stripped.split(","))
    else:
        raise TypeError(f"RFD3 atom names must be a string or list of strings. Got {type(atom_names)}: {atom_names}")

    if any(not name for name in names):
        raise ValueError(f"Empty atom name found in RFD3 atom selection: {atom_names}")
    if len(set(names)) != len(names):
        raise ValueError(f"Atom names in RFD3 atom selection must be unique. Got: {atom_names}")
    return names


def _select_atoms_from_residue(residue: Any, atom_names: Any) -> list[Any]:
    """Select BioPython atoms from one residue according to RFD3 atom-name syntax."""
    names = _atom_names_from_rfd3_value(atom_names, residue=residue)
    atoms = list(residue.get_atoms())
    if names is None:
        return atoms
    if not names:
        return []

    selected = [atom for atom in atoms if atom.get_id() in names]
    selected_names = [atom.get_id() for atom in selected]
    missing = [name for name in names if name not in selected_names]
    if missing:
        residue_label = f"{_residue_chain_id(residue)}{residue.id[1]}:{residue.get_resname().strip()}"
        raise ValueError(f"Could not find atoms {missing} in residue {residue_label}. Available atoms: {[atom.get_id() for atom in atoms]}")
    return selected


def _matching_residues_for_rfd3_component(component: str, entity: Any, model_id: Any = 0) -> list[Any]:
    """Resolve one RFD3 residue component or ligand/residue name to BioPython residues."""
    component = component.strip()
    residue_components = _parse_rfd3_residue_component(component)
    residues = list(_iter_biopython_residues(entity, model_id=model_id))

    if residue_components is not None:
        wanted = set(residue_components)
        matches = [residue for residue in residues if (_residue_chain_id(residue), int(residue.id[1])) in wanted]
    else:
        matches = [residue for residue in residues if residue.get_resname().strip() == component]

    if not matches:
        raise ValueError(f"Could not resolve RFD3 component '{component}' in pose.")
    return matches


def _atom_ids_from_rfd3_component_with_pose(
    component: str,
    atom_names: Any,
    entity: Any,
    model_id: Any = 0,
    residue_id_format: str = "auto",
) -> list[AtomID]:
    """Resolve one RFD3 component to AtomSelection IDs using a BioPython entity."""
    atom_ids = []
    for residue in _matching_residues_for_rfd3_component(component, entity=entity, model_id=model_id):
        atom_ids.extend(_atom_id_from_biopython_atom(atom, residue_id_format=residue_id_format) for atom in _select_atoms_from_residue(residue, atom_names))
    return atom_ids


def _atom_ids_from_rfd3_component_without_pose(component: str, atom_names: Any) -> list[AtomID]:
    """Resolve explicit-atom RFD3 components without a structure."""
    residue_components = _parse_rfd3_residue_component(component)
    if residue_components is None:
        raise ValueError(f"RFD3 component '{component}' requires a pose because it is not an indexed residue component.")

    names = _atom_names_from_rfd3_value(atom_names)
    if names is None:
        raise ValueError(f"RFD3 component '{component}' uses ALL atoms and requires a pose to inspect atom names.")
    return [(chain, resi, atom_name) for chain, resi in residue_components for atom_name in names]


def _looks_like_chain_atom_dict(input_dict: dict[str, Any]) -> bool:
    """Return True for {chain: {residue_id: atom_names}} dictionaries."""
    return bool(input_dict) and all(isinstance(chain, str) and isinstance(residue_map, dict) for chain, residue_map in input_dict.items())


def _atom_ids_from_chain_atom_dict(
    input_dict: dict[str, dict[Any, Any]],
    pose: Any = None,
    model_id: Any = 0,
    residue_id_format: str = "auto",
) -> list[AtomID]:
    """Create atom IDs from a nested chain/residue/atom-name mapping."""
    entity = _normalize_biopython_entity(pose, model_id=model_id)
    atom_ids = []
    for chain, residue_map in input_dict.items():
        if entity is None:
            for residue_id, atom_names in residue_map.items():
                names = _atom_names_from_rfd3_value(atom_names)
                if names is None:
                    raise ValueError(f"Atom dictionary entry {chain}{residue_id} uses ALL atoms and requires a pose.")
                atom_ids.extend((chain, _as_tuple_recursive(residue_id), atom_name) for atom_name in names)
        else:
            for residue_id, atom_names in residue_map.items():
                component = f"{chain}{residue_id[1] if isinstance(residue_id, (list, tuple)) else residue_id}"
                atom_ids.extend(
                    _atom_ids_from_rfd3_component_with_pose(
                        component,
                        atom_names,
                        entity=entity,
                        model_id=model_id,
                        residue_id_format=residue_id_format,
                    )
                )
    return atom_ids


def _unique_atom_ids(atom_ids: list[AtomID]) -> tuple[AtomID, ...]:
    """Remove duplicate atom IDs while preserving order."""
    return tuple(OrderedDict.fromkeys(_as_tuple_recursive(atom_id) for atom_id in atom_ids))


def _residue_from_atom_id(atom_id: AtomID) -> tuple[str, int]:
    """Collapse one normalized atom ID to the `(chain, residue_number)` pair."""
    _validate_atom_id(atom_id)

    if len(atom_id) == 3:
        chain_id, residue_id, _ = atom_id
    elif len(atom_id) == 4:
        _, chain_id, residue_id, _ = atom_id
    elif len(atom_id) == 5:
        _, _, chain_id, residue_id, _ = atom_id
    else:
        _, _, chain_id, residue_id, _, _ = atom_id

    if isinstance(residue_id, (list, tuple)):
        residue_id = residue_id[1]

    return (chain_id, int(residue_id))


def _atom_name_from_atom_id(atom_id: AtomID) -> str:
    """Return the atom-name component from one normalized atom ID."""
    _validate_atom_id(atom_id)

    if len(atom_id) == 3:
        atom_name = atom_id[2]
    elif len(atom_id) == 4:
        atom_name = atom_id[3]
    elif len(atom_id) == 5:
        atom_name = atom_id[4]
    else:
        atom_name = atom_id[4]

    if isinstance(atom_name, (list, tuple)):
        atom_name = atom_name[0]
    return str(atom_name).strip()


def _sidechain_atom_selection(atom_selection: "AtomSelection") -> "AtomSelection":
    """Return non-backbone atoms from an AtomSelection."""
    return AtomSelection(
        tuple(
            atom
            for atom in atom_selection
            if _atom_name_from_atom_id(atom) not in _RFD3_BACKBONE_ATOMS
        )
    )


def _atom_selection_residues(atom_selection: "AtomSelection") -> set[tuple[str, int]]:
    """Return residue identifiers covered by an AtomSelection."""
    return {_residue_from_atom_id(atom) for atom in atom_selection}


def _rfd3_selection_components(input_selection: Any) -> list[str]:
    """Return component strings referenced by an RFD3 InputSelection-like value."""
    if isinstance(input_selection, str):
        return _rfd3_components_from_string(input_selection)
    if isinstance(input_selection, dict) and "atoms" not in input_selection:
        components = []
        for component_spec in input_selection:
            components.extend(_rfd3_components_from_string(str(component_spec)))
        return components
    return []


def _residues_from_rfd3_component(component: str, entity: Any, model_id: Any = 0) -> set[tuple[str, int]]:
    """Resolve one RFD3 component to residue identifiers."""
    residue_components = _parse_rfd3_residue_component(component)
    if residue_components is not None:
        return {(chain, int(resi)) for chain, resi in residue_components}
    if entity is None:
        raise ValueError(
            f"RFD3 component '{component}' requires a pose because it is not an indexed residue component."
        )
    return {
        (_residue_chain_id(residue), int(residue.id[1]))
        for residue in _matching_residues_for_rfd3_component(
            component,
            entity=entity,
            model_id=model_id,
        )
    }


def _covered_residues_from_rfd3_selection(
    input_selection: Any,
    pose: Any = None,
    model_id: Any = 0,
) -> set[tuple[str, int]]:
    """
    Return residues explicitly covered by an RFD3 InputSelection-like value.

    This preserves the distinction between "selected atoms" and "covered
    residues" needed for RFD3 override fields such as select_fixed_atoms, where
    selecting a subset of atoms in one residue unselects the remaining atoms in
    that same residue but does not affect unrelated residues.
    """
    if input_selection is None:
        return set()
    if (
        isinstance(input_selection, AtomSelection)
        or (isinstance(input_selection, dict) and "atoms" in input_selection)
        or isinstance(input_selection, (list, tuple))
    ):
        return _atom_selection_residues(AtomSelection(input_selection))

    entity = _normalize_biopython_entity(pose, model_id=model_id)
    if isinstance(input_selection, bool):
        if entity is None:
            raise ValueError(f"RFD3 boolean InputSelection={input_selection} requires a pose.")
        return {
            (_residue_chain_id(residue), int(residue.id[1]))
            for residue in _iter_biopython_residues(entity, model_id=model_id)
        }

    residues = set()
    for component in _rfd3_selection_components(input_selection):
        residues.update(
            _residues_from_rfd3_component(component, entity=entity, model_id=model_id)
        )
    return residues


def _apply_rfd3_fixed_atom_overrides(
    atom_selection: "AtomSelection",
    select_fixed_atoms: Any,
    pose: Any = None,
    model_id: Any = 0,
    residue_id_format: str = "auto",
) -> "AtomSelection":
    """Apply RFD3 select_fixed_atoms coordinate-fixing semantics to a selection."""
    if select_fixed_atoms is None:
        return atom_selection
    if isinstance(select_fixed_atoms, bool):
        return atom_selection if select_fixed_atoms else AtomSelection(())

    fixed_atoms = AtomSelection.from_rfd3_input_selection(
        select_fixed_atoms,
        pose=pose,
        model_id=model_id,
        residue_id_format=residue_id_format,
    )
    fixed_atom_set = set(fixed_atoms.atoms)
    covered_residues = _covered_residues_from_rfd3_selection(
        select_fixed_atoms,
        pose=pose,
        model_id=model_id,
    )
    return AtomSelection(
        tuple(
            atom
            for atom in atom_selection
            if _residue_from_atom_id(atom) not in covered_residues or atom in fixed_atom_set
        )
    )


def _fixed_motif_atoms_from_rfd3_input_spec(
    input_spec: dict[str, Any],
    pose: Any = None,
    model_id: Any = 0,
    residue_id_format: str = "auto",
) -> "AtomSelection":
    """Derive coordinate-fixed unindexed motif atoms from an RFD3 input spec."""
    if input_spec.get("unindex") is None:
        return AtomSelection(())

    fixed_motif_atoms = AtomSelection.from_rfd3_input_selection(
        input_spec["unindex"],
        pose=pose,
        model_id=model_id,
        residue_id_format=residue_id_format,
    )
    fixed_motif_atoms = _apply_rfd3_fixed_atom_overrides(
        fixed_motif_atoms,
        input_spec.get("select_fixed_atoms"),
        pose=pose,
        model_id=model_id,
        residue_id_format=residue_id_format,
    )

    if input_spec.get("select_unfixed_sequence") is not None:
        unfixed_sequence_atoms = AtomSelection.from_rfd3_input_selection(
            input_spec["select_unfixed_sequence"],
            pose=pose,
            model_id=model_id,
            residue_id_format=residue_id_format,
        )
        fixed_motif_atoms = fixed_motif_atoms - _sidechain_atom_selection(
            unfixed_sequence_atoms
        )

    return fixed_motif_atoms


def _fixed_ligand_atoms_from_rfd3_input_spec(
    input_spec: dict[str, Any],
    pose: Any = None,
    model_id: Any = 0,
    residue_id_format: str = "auto",
) -> "AtomSelection":
    """Derive coordinate-fixed ligand atoms from an RFD3 input spec."""
    if input_spec.get("ligand") is None:
        return AtomSelection(())

    ligand_atoms = AtomSelection.from_rfd3_ligand(
        input_spec["ligand"],
        pose=pose,
        model_id=model_id,
        residue_id_format=residue_id_format,
    )
    return _apply_rfd3_fixed_atom_overrides(
        ligand_atoms,
        input_spec.get("select_fixed_atoms"),
        pose=pose,
        model_id=model_id,
        residue_id_format=residue_id_format,
    )


class AtomSelection:
    """
    Represent an ordered selection of atoms in a protein structure.

    Atom IDs can be compact IDs ``(chain_id, res_id, atom_name)`` using model 0
    implicitly, or full BioPython-style IDs with model and structure IDs. Atom
    ordering is preserved because RMSD calculation pairs atoms by position.

    Parameters
    ----------
    atoms : AtomSelection, dict, list, or tuple
        Ordered atom selection to normalize. Supported atom ID forms are:

        * ``(chain_id, residue_id, atom_name)``
        * ``(model_id, chain_id, residue_id, atom_name)``
        * ``(structure_id, model_id, chain_id, residue_id, atom_name)``
        * ``(structure_id, model_id, chain_id, residue_id, atom_name, altloc)``

        ``residue_id`` can be a compact integer-like value or a BioPython
        residue ID tuple ``(hetero_flag, residue_number, insertion_code)``.
        ``atom_name`` can be a string or a BioPython disordered atom ID tuple
        ``(atom_name, altloc)``. A scorefile-style dictionary with an
        ``"atoms"`` key is also accepted.

    Attributes
    ----------
    atoms : tuple
        Tuple of normalized atom IDs. Nested lists are converted to tuples so
        selections can be compared and used in set-like operations.

    Raises
    ------
    TypeError
        If *atoms* is not an AtomSelection, scorefile dictionary, or ordered
        sequence of atom IDs.
    ValueError
        If any atom ID has an unsupported shape or invalid chain, residue, or
        atom-name component.

    Notes
    -----
    AtomSelection preserves order deliberately. Many atom-level operations,
    such as RMSD or geometry calculations, pair atoms by position rather than
    treating the selection as an unordered set.

    Examples
    --------
    Create a compact atom selection::

        atoms = AtomSelection([("A", 1, "N"), ("A", 1, "CA")])

    Create the same selection from scorefile-compatible data::

        atoms = AtomSelection({"atoms": [["A", 1, "N"], ["A", 1, "CA"]]})
    """

    def __init__(self, atoms: Any) -> None:
        """Normalize and store an ordered atom selection."""
        self.atoms = _normalize_atom_selection_value(atoms, parameter_name="AtomSelection")

    def __iter__(self):
        """Iterate over normalized atom IDs in selection order."""
        return iter(self.atoms)

    def __len__(self) -> int:
        """Return the number of atom IDs in the selection."""
        return len(self.atoms)

    def __str__(self) -> str:
        """Return a string representation of the tuple-backed selection."""
        return str(self.to_tuple())

    def __add__(self, other):
        """
        Combine two AtomSelections while preserving order and uniqueness.

        Parameters
        ----------
        other : AtomSelection
            Selection to append to ``self``. Atoms already present in ``self``
            are skipped, matching the behavior of
            :meth:`ResidueSelection.__add__`.

        Returns
        -------
        AtomSelection
            New selection containing all atoms from ``self`` followed by atoms
            from ``other`` that were not already present.
        NotImplemented
            Returned when *other* is not an AtomSelection, allowing Python's
            binary operator fallback behavior.

        Examples
        --------
        ::

            a = AtomSelection([("A", 1, "N"), ("A", 1, "CA")])
            b = AtomSelection([("A", 1, "CA"), ("A", 1, "C")])
            (a + b).to_tuple()
            # (("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C"))
        """
        if isinstance(other, AtomSelection):
            # Reuse subtraction so duplicate handling stays identical to the
            # ResidueSelection implementation.
            return AtomSelection(self.atoms + (other - self).atoms)
        return NotImplemented

    def __sub__(self, other):
        """
        Remove atoms in another AtomSelection from this selection.

        Parameters
        ----------
        other : AtomSelection
            Selection whose atoms should be removed from ``self``.

        Returns
        -------
        AtomSelection
            New selection containing atoms from ``self`` whose normalized atom
            IDs are absent from ``other``. Original order is preserved.
        NotImplemented
            Returned when *other* is not an AtomSelection.

        Examples
        --------
        ::

            a = AtomSelection([("A", 1, "N"), ("A", 1, "CA")])
            b = AtomSelection([("A", 1, "CA")])
            (a - b).to_tuple()
            # (("A", 1, "N"),)
        """
        if isinstance(other, AtomSelection):
            other_atoms = set(other.atoms)
            return AtomSelection(tuple(atom for atom in self.atoms if atom not in other_atoms))
        return NotImplemented

    ####################################### INPUT ##############################################
    @staticmethod
    def from_list(atoms: list[Any] | tuple[Any, ...]) -> "AtomSelection":
        """
        Create an AtomSelection from an ordered list or tuple of atom IDs.

        Parameters
        ----------
        atoms : list or tuple
            Ordered atom IDs in any format accepted by :class:`AtomSelection`.
            Passing a single atom ID such as ``("A", 1, "N")`` is also
            supported.

        Returns
        -------
        AtomSelection
            Normalized atom selection preserving the order supplied in
            *atoms*.

        Raises
        ------
        TypeError
            If *atoms* is not sequence-like.
        ValueError
            If any atom ID is malformed.

        Examples
        --------
        ::

            AtomSelection.from_list([("A", 1, "N"), ("A", 1, "CA")])
        """
        return AtomSelection(atoms)

    @staticmethod
    def from_dict(input_dict: dict[str, Any], pose: Any = None, residue_id_format: str = "auto") -> "AtomSelection":
        """
        Create an AtomSelection from a scorefile dict, nested atom dict, or RFD3 dict.

        This is the dictionary-oriented constructor for AtomSelection. It
        supports three dictionary dialects:

        * ``{"atoms": [...]}`` for ProtFlow scorefile-compatible atom
          selections.
        * ``{"A": {1: ["N", "CA"]}}`` for explicit
          chain/residue/atom-name mappings.
        * RFD3 InputSelection dictionaries such as
          ``{"A1-2": "BKBN", "LIG": "C1,O1"}``.

        Parameters
        ----------
        input_dict : dict
            Dictionary describing an atom selection in one of the supported
            forms listed above.
        pose : str, os.PathLike, Bio.PDB entity, optional
            Input structure used to expand RFD3 aliases or residue-name
            selectors. A pose is required when values use ``ALL`` or ``TIP``,
            when keys select ligands/residue names, or when exact atom names
            should be checked against the input structure.
        residue_id_format : {"auto", "compact", "biopython"}, optional
            Controls how residue IDs are written when atoms are read from
            *pose*. ``"auto"`` uses compact integer residue IDs for standard
            residues and BioPython residue IDs for hetero residues. ``"compact"``
            always writes integer residue IDs. ``"biopython"`` always writes
            BioPython residue IDs.

        Returns
        -------
        AtomSelection
            Normalized atom selection described by *input_dict*.

        Raises
        ------
        TypeError
            If *input_dict* is not a dictionary or if atom-name values have an
            unsupported type.
        ValueError
            If the dictionary uses structure-dependent syntax but no *pose* is
            provided, or if requested atoms/components cannot be resolved.

        Examples
        --------
        Parse scorefile-compatible data::

            AtomSelection.from_dict({"atoms": [["A", 1, "N"], ["A", 1, "CA"]]})

        Parse a nested chain/residue mapping::

            AtomSelection.from_dict({"A": {1: ["N", "CA"], 2: "C,O"}})

        Parse an RFD3 InputSelection dictionary against a PDB file::

            AtomSelection.from_dict({"A1-2": "BKBN", "LIG": "C1,O1"}, pose="input.pdb")
        """
        if not isinstance(input_dict, dict):
            raise TypeError(f"input_dict must be a dictionary. Got {type(input_dict)}: {input_dict}")
        # The scorefile representation is already in AtomSelection's native
        # JSON-friendly format.
        if "atoms" in input_dict:
            return AtomSelection(input_dict)
        # A nested chain dictionary mirrors ResidueSelection.from_dict while
        # allowing explicit atom names per residue.
        if _looks_like_chain_atom_dict(input_dict):
            return AtomSelection(_unique_atom_ids(_atom_ids_from_chain_atom_dict(input_dict, pose=pose, residue_id_format=residue_id_format)))
        # Anything else is treated as the RFD3 InputSelection dictionary
        # grammar, where keys are components and values are atom specifiers.
        return AtomSelection.from_rfd3_input_selection(input_dict, pose=pose, residue_id_format=residue_id_format)

    @staticmethod
    def from_rfd3_contig(
        input_contig: str,
        pose: Any = None,
        atom_names: str | list[str] | tuple[str, ...] = "ALL",
        model_id: Any = 0,
        residue_id_format: str = "auto",
    ) -> "AtomSelection":
        """
        Create an AtomSelection from indexed parts of an RFD3 contig string.

        Generated-length components such as ``10``/``10-20`` and chain breaks
        like ``/0`` are skipped. With ``pose`` provided, ``atom_names="ALL"``
        expands to the atoms present in the structure and ligand/residue-name
        components can be resolved. Without a pose, ``atom_names`` must be an
        explicit atom list or an alias that does not require structure context
        such as ``BKBN``.

        Parameters
        ----------
        input_contig : str
            RFD3 contig string. Indexed residue components such as ``"A1"``,
            ``"A1-5"``, and ``"A1-A5"`` are converted to atom IDs. Diffused
            length components and chain breaks are ignored because they do not
            refer to atoms in the input structure.
        pose : str, os.PathLike, Bio.PDB entity, optional
            Input structure used to expand ``ALL`` atoms, validate explicit
            atom names, and resolve ligand/residue-name components. If omitted,
            only indexed residue components with explicit atom-name values can
            be parsed.
        atom_names : str, list, or tuple, optional
            Atom names to select from every indexed component. Supported RFD3
            aliases are ``"ALL"``, ``"BKBN"``, and ``"TIP"``. Explicit names
            can be supplied as comma-separated strings such as ``"N,CA,C,O"``
            or as lists/tuples of strings.
        model_id : int or str, optional
            BioPython model identifier used when *pose* is a Structure object
            or a path to a multi-model file. Defaults to ``0``.
        residue_id_format : {"auto", "compact", "biopython"}, optional
            Controls residue ID formatting for atoms loaded from *pose*.

        Returns
        -------
        AtomSelection
            Atom selection for the indexed input components in *input_contig*.

        Raises
        ------
        TypeError
            If *input_contig* is not a string.
        ValueError
            If a selected component or requested atom cannot be resolved, or
            if structure-dependent syntax is used without *pose*.

        Examples
        --------
        Select backbone atoms from indexed residues without loading a pose::

            AtomSelection.from_rfd3_contig("10,A1-2,/0,B5", atom_names="BKBN")

        Select all atoms present in an input structure::

            AtomSelection.from_rfd3_contig("A1-2,/0,Z9", pose="input.pdb")
        """
        if not isinstance(input_contig, str):
            raise TypeError(f"input_contig must be a string. Got {type(input_contig)}: {input_contig}")

        entity = _normalize_biopython_entity(pose, model_id=model_id)
        atom_ids = []
        for component in _rfd3_components_from_string(input_contig):
            # Without a structure, only explicit atom-name selections can be
            # assembled. With a structure, we can expand ALL/TIP and ligands.
            if entity is None:
                atom_ids.extend(_atom_ids_from_rfd3_component_without_pose(component, atom_names))
            else:
                atom_ids.extend(
                    _atom_ids_from_rfd3_component_with_pose(
                        component,
                        atom_names,
                        entity=entity,
                        model_id=model_id,
                        residue_id_format=residue_id_format,
                    )
                )
        return AtomSelection(_unique_atom_ids(atom_ids))

    @staticmethod
    def from_rfd3_ligand(
        ligand: str,
        pose: Any,
        model_id: Any = 0,
        residue_id_format: str = "auto",
    ) -> "AtomSelection":
        """
        Create an AtomSelection from an RFD3 ligand specification.

        Ligands can be selected by residue name (``"LIG"`` or
        ``"LIG,ACT"``) or by indexed residue components such as ``"Z9"``.

        Parameters
        ----------
        ligand : str
            RFD3 ligand selector. Comma-separated residue names select all
            matching non-protein residues in the input structure. Indexed
            residue components such as ``"Z9"`` can also be used.
        pose : str, os.PathLike, Bio.PDB entity
            Input structure containing the ligand atoms. This argument is
            required because ligand names must be resolved against the actual
            structure.
        model_id : int or str, optional
            BioPython model identifier used for structure-backed parsing.
        residue_id_format : {"auto", "compact", "biopython"}, optional
            Controls residue ID formatting for atoms loaded from *pose*.

        Returns
        -------
        AtomSelection
            Selection containing all atoms selected by the ligand
            specification.

        Raises
        ------
        ValueError
            If *pose* is omitted or if the ligand selector does not match the
            input structure.

        Examples
        --------
        Select all atoms in ligands named ``LIG`` and ``ACT``::

            AtomSelection.from_rfd3_ligand("LIG,ACT", pose="input.pdb")
        """
        if pose is None:
            raise ValueError("Parsing an RFD3 ligand specification requires a pose.")
        return AtomSelection.from_rfd3_input_selection(
            ligand,
            pose=pose,
            model_id=model_id,
            residue_id_format=residue_id_format,
        )

    @staticmethod
    def from_rfd3_input_selection(
        input_selection: Any,
        pose: Any = None,
        model_id: Any = 0,
        residue_id_format: str = "auto",
    ) -> "AtomSelection":
        """
        Create an AtomSelection from an RFD3 InputSelection value.

        Supported RFD3 forms are booleans, contig-style strings, and
        dictionaries whose keys are residue/ligand selections and whose values
        are atom names, ``ALL``, ``BKBN``, ``TIP``, or explicit atom-name lists.
        A pose is required for booleans, ``ALL``, ``TIP``, and ligand/residue
        name selection because those cases need the actual atoms in the input
        structure.

        Parameters
        ----------
        input_selection : None, bool, str, dict, AtomSelection, list, or tuple
            RFD3 InputSelection-like value to parse. Supported forms are:

            ``None``
                Returns an empty AtomSelection.
            ``True`` / ``False``
                Select all atoms in *pose* or no atoms, respectively.
            ``str``
                Parses a contig-style selector such as ``"A1-10,B5"`` or a
                ligand/residue name such as ``"LIG"``. String selections imply
                ``ALL`` atoms for matching components.
            ``dict``
                Parses RFD3 dictionary syntax where keys are components and
                values are atom selectors, e.g. ``{"A1": "BKBN"}``.
            ``AtomSelection`` or atom-ID list/tuple
                Normalizes the existing atom selection directly.
        pose : str, os.PathLike, Bio.PDB entity, optional
            Input structure used for syntax that depends on actual atoms or
            residue names.
        model_id : int or str, optional
            BioPython model identifier used for structure-backed parsing.
        residue_id_format : {"auto", "compact", "biopython"}, optional
            Controls residue ID formatting for atoms loaded from *pose*.

        Returns
        -------
        AtomSelection
            Normalized atom selection represented by *input_selection*.

        Raises
        ------
        TypeError
            If *input_selection* has an unsupported type.
        ValueError
            If the selection requires a structure but *pose* is absent, or if
            selected residues/atoms cannot be found.

        Notes
        -----
        This parser mirrors the user-facing RFD3 InputSelection grammar without
        importing RFD3 or Foundry at runtime. It intentionally returns concrete
        atom IDs rather than RFD3 masks.

        Examples
        --------
        Parse explicit atoms without a pose::

            AtomSelection.from_rfd3_input_selection({"A1-2": "BKBN"})

        Parse ligand atoms and TIP atoms from a structure::

            AtomSelection.from_rfd3_input_selection({"LIG": "ALL", "A20": "TIP"}, pose="input.pdb")
        """
        # Already-normalized selections and scorefile dictionaries are accepted
        # to make repeated parsing idempotent.
        if isinstance(input_selection, AtomSelection):
            return AtomSelection(input_selection)
        if isinstance(input_selection, dict) and "atoms" in input_selection:
            return AtomSelection(input_selection)
        if input_selection is None:
            return AtomSelection(())

        entity = _normalize_biopython_entity(pose, model_id=model_id)
        atom_ids = []

        if isinstance(input_selection, bool):
            if not input_selection:
                return AtomSelection(())
            if entity is None:
                raise ValueError("RFD3 boolean InputSelection=True requires a pose.")
            # Boolean True means "all atoms in the input structure" in RFD3.
            for residue in _iter_biopython_residues(entity, model_id=model_id):
                atom_ids.extend(_atom_id_from_biopython_atom(atom, residue_id_format=residue_id_format) for atom in residue.get_atoms())
            return AtomSelection(_unique_atom_ids(atom_ids))

        if isinstance(input_selection, str):
            if not input_selection.strip():
                return AtomSelection(())
            # String InputSelections select ALL atoms from every resolved
            # component, matching RFD3 canonicalization.
            components_and_atoms = [(component, "ALL") for component in _rfd3_components_from_string(input_selection)]
        elif isinstance(input_selection, dict):
            components_and_atoms = []
            for component_spec, atom_names in input_selection.items():
                # Dictionary keys may contain ranges or comma-separated
                # components; split them before applying the atom-name value.
                for component in _rfd3_components_from_string(str(component_spec)):
                    components_and_atoms.append((component, atom_names))
        elif isinstance(input_selection, (list, tuple)):
            return AtomSelection(input_selection)
        else:
            raise TypeError(
                "RFD3 InputSelection must be None, bool, str, dict, AtomSelection, or an atom-ID list. "
                f"Got {type(input_selection)}: {input_selection}"
            )

        for component, atom_names in components_and_atoms:
            # Resolve each component using the structure when available, or
            # fall back to explicit atom-name assembly for structure-free input.
            if entity is None:
                atom_ids.extend(_atom_ids_from_rfd3_component_without_pose(component, atom_names))
            else:
                atom_ids.extend(
                    _atom_ids_from_rfd3_component_with_pose(
                        component,
                        atom_names,
                        entity=entity,
                        model_id=model_id,
                        residue_id_format=residue_id_format,
                    )
                )

        return AtomSelection(_unique_atom_ids(atom_ids))

    @staticmethod
    def from_rfd3_input_spec(
        input_spec: dict[str, Any],
        pose: Any = None,
        fields: list[str] | tuple[str, ...] | None = None,
        include_ligand: bool = True,
        model_id: Any = 0,
        residue_id_format: str = "auto",
    ) -> dict[str, "AtomSelection"]:
        """
        Parse RFD3 InputSelection fields from one InputSpecification.

        Returns a dictionary mapping each parsed field name to an
        AtomSelection. If ``pose`` is not provided, ``input_spec["input"]`` is
        used when present. The RFD3 ``ligand`` field is included by default even
        though it is not typed as InputSelection in RFD3 itself.

        Parameters
        ----------
        input_spec : dict
            One RFD3 InputSpecification dictionary, for example one value from
            an :class:`~protflow.tools.rfdiffusion3.RFD3Params` object.
        pose : str, os.PathLike, Bio.PDB entity, optional
            Input structure used to resolve InputSelection fields. When omitted,
            ``input_spec["input"]`` is used if present.
        fields : list or tuple of str, optional
            InputSelection field names to parse. Defaults to all RFD3
            InputSelection fields known to ProtFlow: ``contig``, ``unindex``,
            ``select_fixed_atoms``, ``select_unfixed_sequence``,
            ``fixed_motif_atoms``, ``fixed_motif_atoms_with_ligand``,
            ``select_buried``, ``select_partially_buried``,
            ``select_exposed``, ``select_hbond_donor``,
            ``select_hbond_acceptor``, and ``select_hotspots``. The
            ``fixed_motif_atoms`` fields are derived from the RFD3 motif,
            sequence, and coordinate-fixing fields rather than read directly
            from *input_spec*.
        include_ligand : bool, optional
            If ``True`` (default), parse the RFD3 ``ligand`` field into an
            AtomSelection under the key ``"ligand"``.
        model_id : int or str, optional
            BioPython model identifier used for structure-backed parsing.
        residue_id_format : {"auto", "compact", "biopython"}, optional
            Controls residue ID formatting for atoms loaded from *pose*.

        Returns
        -------
        dict[str, AtomSelection]
            Mapping from each parsed input-specification field to the
            corresponding AtomSelection. Fields absent from *input_spec* or set
            to ``None`` are omitted.

        Raises
        ------
        TypeError
            If *input_spec* is not a dictionary.
        ValueError
            If any requested field cannot be resolved to atoms.

        Examples
        --------
        Parse all atom-level selections from an RFD3 spec::

            spec = {
                "input": "input.pdb",
                "contig": "A1-20,/0,50-80",
                "select_fixed_atoms": {"A10": "BKBN", "LIG": "C1,O1"},
                "ligand": "LIG",
            }
            selections = AtomSelection.from_rfd3_input_spec(spec)
            fixed_atoms = selections["select_fixed_atoms"]
        """
        if not isinstance(input_spec, dict):
            raise TypeError(f"input_spec must be a dictionary. Got {type(input_spec)}: {input_spec}")

        # Prefer the explicit pose argument, but follow RFD3 InputSpecification
        # convention by falling back to the "input" path when available.
        pose = pose if pose is not None else input_spec.get("input")
        fields = tuple(fields) if fields is not None else RFD3_INPUT_SELECTION_FIELDS

        selections = {}
        for field in fields:
            if field in _RFD3_DERIVED_INPUT_SELECTION_FIELDS:
                continue
            if field in input_spec and input_spec[field] is not None:
                # Parse each requested InputSelection independently so callers
                # can inspect the semantic source field after conversion.
                selections[field] = AtomSelection.from_rfd3_input_selection(
                    input_spec[field],
                    pose=pose,
                    model_id=model_id,
                    residue_id_format=residue_id_format,
                )

        if "fixed_motif_atoms" in fields and input_spec.get("unindex") is not None:
            selections["fixed_motif_atoms"] = _fixed_motif_atoms_from_rfd3_input_spec(
                input_spec,
                pose=pose,
                model_id=model_id,
                residue_id_format=residue_id_format,
            )

        if "fixed_motif_atoms_with_ligand" in fields and (
            input_spec.get("unindex") is not None
            or (include_ligand and input_spec.get("ligand") is not None)
        ):
            fixed_motif_atoms = selections.get("fixed_motif_atoms")
            if fixed_motif_atoms is None:
                fixed_motif_atoms = _fixed_motif_atoms_from_rfd3_input_spec(
                    input_spec,
                    pose=pose,
                    model_id=model_id,
                    residue_id_format=residue_id_format,
                )
            fixed_ligand_atoms = (
                _fixed_ligand_atoms_from_rfd3_input_spec(
                    input_spec,
                    pose=pose,
                    model_id=model_id,
                    residue_id_format=residue_id_format,
                )
                if include_ligand
                else AtomSelection(())
            )
            selections["fixed_motif_atoms_with_ligand"] = fixed_motif_atoms + fixed_ligand_atoms

        if include_ligand and input_spec.get("ligand") is not None:
            # Ligand is not typed as InputSelection in RFD3, but it resolves to
            # a concrete atom set and is useful for downstream atom metrics.
            selections["ligand"] = AtomSelection.from_rfd3_ligand(
                input_spec["ligand"],
                pose=pose,
                model_id=model_id,
                residue_id_format=residue_id_format,
            )

        return selections

    ####################################### OUTPUT #############################################
    def to_tuple(self) -> tuple[AtomID, ...]:
        """Return the ordered atom selection as tuples."""
        return self.atoms

    def to_list(self) -> list[Any]:
        """Return the ordered atom selection in JSON-friendly list format."""
        return _as_list_recursive(self.atoms)

    def to_dict(self) -> dict[str, list[Any]]:
        """Return a scorefile-friendly dictionary representation."""
        return {"atoms": self.to_list()}

    def to_rfd3_dict(self) -> dict[str, str]:
        """Return a dict formatted like an RFdiffusion3 input contig (e.g. {"A5": "CA,N,O"})."""

        # initialize a defaultdict with list as the default factory
        rfd3_dict = defaultdict(list)

        for chain, num, atm in self.atoms:
            # create the key by concatenating the string and the integer
            key = f"{chain}{num}"
            rfd3_dict[key].append(atm)

        # convert list to comma-separated str
        for res, atms in rfd3_dict.items():
            rfd3_dict[res] = ",".join(atms)

        # Convert back to a regular dict and return
        return dict(rfd3_dict)


AtomSelectionInput: TypeAlias = str | tuple[Any, ...] | list[Any] | dict[str, Any] | AtomSelection | None


class ResidueSelection:
    """
    Represent a selection of residues in a protein structure.

    A selection of residues is represented as a tuple with the hierarchy
    ((chain, residue_idx), ...).

    Parameters
    ----------
    selection : list, optional
        A list of residues in string format, e.g., ["A1", "A2", "B3"]. Default is None.
    delim : str, optional
        The delimiter used to parse the selection string. Default is ",".
    fast : bool, optional
        If True, parses the selection without any type checking. Use when `selection` is already in 
        ResidueSelection format. Default is False.

    Attributes
    ----------
    residues : tuple
        A tuple representing the parsed residues selection.


    Examples
    --------
    >>> from residues import ResidueSelection
    >>> selection = ResidueSelection(["A1", "A2", "B3"])
    >>> print(selection.to_string())
    A1, A2, B3
    >>> print(selection.to_dict())
    {'A': [1, 2], 'B': [3]}
    """
    def __init__(self, selection: list = None, delim: str = ",", fast: bool = False, from_scorefile: bool = False):
        self.residues = parse_selection(selection, delim=delim, fast=fast, from_scorefile=from_scorefile)

    def __len__(self) -> int:
        return len(self.residues)

    def __str__(self) -> str:
        return ", ".join([f"{chain}{str(resi)}" for chain, resi in self])

    def __iter__(self):
        return iter(self.residues)

    def __add__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(self.residues + (other - self).residues, fast=True)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(tuple(res for res in self.residues if res not in set(other.residues)), fast=True)
        return NotImplemented

    ####################################### INPUT ##############################################
    @classmethod
    def from_atomselection(cls, atom_selection: AtomSelection|AtomSelectionInput) -> "ResidueSelection":
        """
        Create a ResidueSelection from an AtomSelection-like input.

        Residues are ordered by the first atom index at which they appear in
        the atom selection. Repeated atoms from the same residue therefore
        collapse to one residue while preserving encounter order.
        """
        atom_selection = AtomSelection(atom_selection)
        residues = reduce_to_unique(tuple(_residue_from_atom_id(atom_id) for atom_id in atom_selection))
        return cls(residues, fast=True)

    def from_selection(self, selection) -> "ResidueSelection":
        """
        Constructs a ResidueSelection instance from the provided selection.

        Parameters
        ----------
        selection : list or str
            The selection of residues to be parsed.

        Returns
        -------
        ResidueSelection
            A new ResidueSelection instance.
        """
        return residue_selection(selection)

    ####################################### OUTPUT #############################################
    def to_string(self, delim: str = ",", ordering: str = None) -> str:
        """
        Converts the ResidueSelection to a string.

        Parameters
        ----------
        delim : str, optional
            The delimiter to use in the resulting string. Default is ",".
        ordering : str, optional
            Specifies the ordering of the residues in the output string. Options are "rosetta" or "pymol".
            Default is None.

        Returns
        -------
        str
            ResidueSelection object formatted as a string, separated by :delim:
            ueSelection.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_string())
        A1, A2, B3
        >>> print(selection.to_string(ordering="rosetta"))
        1A, 2A, 3B
        """
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return delim.join([str(idx) + chain for chain, idx in self])
        if ordering.lower() == "pymol":
            return delim.join([chain + str(idx) for chain, idx in self])
        return delim.join([chain + str(idx) for chain, idx in self])

    def to_list(self, ordering: str = None) -> list[str]:
        """
        Converts the ResidueSelection to a list of strings.

        Parameters
        ----------
        ordering : str, optional
            Specifies the ordering of the residues in the output list. Options are "rosetta" or "pymol".
            Default is None.

        Returns
        -------
        list of str
            The list representation of the ResidueSelection.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_list())
        ['A1', 'A2', 'B3']
        >>> print(selection.to_list(ordering="rosetta"))
        ['1A', '2A', '3B']
        """
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return [str(idx) + chain for chain, idx in self]
        if ordering.lower() == "pymol":
            return [chain + str(idx) for chain, idx in self]
        return [chain+str(idx) for chain, idx in self]

    def to_dict(self) -> dict:
        """
        Converts the ResidueSelection to a dictionary. 

        Note
        ----
        Converting to a dictionary destroys the ordering of specific residues on the same chain in a motif.

        Returns
        -------
        dict
            A dictionary representation of the ResidueSelection with chains as keys and lists of residue 
            indices as values.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_dict())
        {'A': [1, 2], 'B': [3]}
        """
        # collect list of chains and setup chains as dictionary keys
        chains = list(set([x[0] for x in self.residues]))
        out_d = {chain: [] for chain in chains}

        # aggregate all residues to the chains and return
        for (chain, res_id) in self.residues:
            out_d[chain].append(res_id)

        return out_d

    def to_rfdiffusion_contig(self) -> str:
        """
        Parses ResidueSelection object to contig string for RFdiffusion.

        Example:
            If self.residues = (("A", 1), ("A", 2), ("A", 3), ("C", 4), ("C", 6)),
            the output will be "A1-3,C4,C6".
        """
        # Collect residues per chain
        chain_residues = defaultdict(list)
        for chain, resnum in self.residues:
            chain_residues[chain].append(resnum)

        contig_parts = []

        # Process each chain separately
        for chain in sorted(chain_residues.keys()):
            # Sort residue numbers for the chain
            resnums = sorted(chain_residues[chain])

            # Find consecutive ranges
            ranges = []
            start = prev = resnums[0]
            for resnum in resnums[1:]:
                if resnum == prev + 1:
                    # Continue the consecutive range
                    prev = resnum
                else:
                    # End of the current range
                    if start == prev:
                        # Single residue
                        ranges.append(f"{chain}{start}")
                    else:
                        # Range of residues
                        ranges.append(f"{chain}{start}-{prev}")
                    # Start a new range
                    start = prev = resnum
            # Add the last range
            if start == prev:
                ranges.append(f"{chain}{start}")
            else:
                ranges.append(f"{chain}{start}-{prev}")

            # Add ranges to the contig parts
            contig_parts.extend(ranges)

        # Combine all parts into the final contig string
        contig_str = ",".join(contig_parts)
        return contig_str

def fast_parse_selection(input_selection: tuple[tuple[str, int]]) -> tuple[tuple[str, int]]:
    """
    Fast selection parser for pre-formatted selections.

    This function is a fast parser for residue selections that are already in the `ResidueSelection` format.
    It bypasses any additional type checking or parsing to improve performance when the input is guaranteed
    to be correctly formatted.

    Parameters
    ----------
    input_selection : tuple of tuple of (str, int)
        A tuple of tuples where each inner tuple represents a residue with the format (chain, residue_index).

    Returns
    -------
    tuple of tuple of (str, int)
        The input selection, unchanged.

    Examples
    --------
    >>> input_selection = (("A", 1), ("B", 2), ("C", 3))
    >>> fast_parse_selection(input_selection)
    (('A', 1), ('B', 2), ('C', 3))
    """
    return input_selection

def parse_from_scorefile(input_selection: dict) -> tuple[tuple[str, int]]:
    '''Helper to parse ResidueSelection object from ProtFlow scorefile format.'''
    if isinstance(input_selection, dict) and "residues" in input_selection:
        return tuple(tuple(sele) for sele in input_selection["residues"])
    if isinstance(input_selection, ResidueSelection):
        # be lenient to double-parsing. If input_selection is already ResidueSelection, just pass.
        return input_selection.residues # Note: This is not very clean but implemented for backwards compatibility.
    raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. This function is meant to parse ResidueSelections that were written to file. Only dict with 'residues' as key allowed.")

def parse_selection(input_selection, delim: str = ",", fast: bool = False, from_scorefile: bool = False) -> tuple[tuple[str,int]]:
    """
    Parses a selection into ResidueSelection formatted selection.

    This function takes a selection of residues in various formats and parses it into the `ResidueSelection` 
    format, which is a tuple of tuples. Each inner tuple represents a residue with the format (chain, residue_index).

    Parameters
    ----------
    input_selection : str, list, or tuple
        The selection of residues to be parsed. This can be:
        - A string with residues separated by a delimiter.
        - A list or tuple of residue strings.
        - A list or tuple of lists/tuples, where each inner list/tuple represents a residue.
    delim : str, optional
        The delimiter used to split the input string if `input_selection` is a string. Default is ",".
    fast : bool, optional
        If True, uses `fast_parse_selection` to bypass type checking and parsing for performance reasons.
        Use when `input_selection` is already in the correct format. Default is False.
    from_scorefile : bool, optional
        If True, parses a residue selection that was read in from a scorefile (in the form {'residues': [['A', 1], ['B', 3]}).
        Default is False.

    Returns
    -------
    tuple of tuple of (str, int)
        A tuple of tuples where each inner tuple represents a residue in the format (chain, residue_index).

    Raises
    ------
    TypeError
        If `input_selection` is not a supported type (str, list, or tuple).

    Examples
    --------
    >>> parse_selection("A1, B2, C3")
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection(["A1", "B2", "C3"])
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection([["A", 1], ["B", 2], ["C", 3]])
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection([("A", 1), ("B", 2), ("C", 3)], fast=True)
    (('A', 1), ('B', 2), ('C', 3))
    """
    if fast and from_scorefile:
        raise RuntimeError(":fast: and :from_scorefile: are mutually exclusive!")
    if fast:
        return fast_parse_selection(input_selection)
    if from_scorefile:
        return parse_from_scorefile(input_selection)
    if isinstance(input_selection, str):
        return tuple(parse_residue(residue.strip()) for residue in input_selection.split(delim))
    if isinstance(input_selection, (list, tuple)):
        if all(isinstance(residue, str) for residue in input_selection):
            return tuple(parse_residue(residue) for residue in input_selection)
        if all(isinstance(residue, (list, tuple)) for residue in input_selection):
            return tuple(parse_residue("".join([str(r) for r in residue])) for residue in input_selection)
    raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. Only str and list allowed.")

def parse_residue(residue_identifier: str) -> tuple[str,int]:
    """
    Parses a single residue identifier into a tuple (chain, residue_index).

    This function takes a residue identifier string and parses it into a tuple containing the chain identifier
    and the residue index. It currently only supports single-letter chain identifiers.

    Parameters
    ----------
    residue_identifier : str
        A string representing the residue identifier. The format is expected to be either "chain+residue_index" 
        or "residue_index+chain", where "chain" is a single letter and "residue_index" is an integer.

    Returns
    -------
    tuple of (str, int)
        A tuple containing the chain identifier and the residue index.

    Examples
    --------
    >>> parse_residue("A123")
    ('A', 123)

    >>> parse_residue("123A")
    ('A', 123)

    Notes
    -----
    - The function determines whether the chain identifier is at the beginning or the end of the string based 
      on whether the first character is a digit.
    - Only single-letter chain identifiers are supported.

    """
    chain_first = not residue_identifier[0].isdigit()

    # assemble residue tuple
    chain = residue_identifier[0] if chain_first else residue_identifier[-1]
    residue_index = residue_identifier[1:] if chain_first else residue_identifier[:-1]

    # Convert residue_index to int for accurate typing
    return (chain, int(residue_index))

def residue_selection(input_selection, delim: str = ",") -> ResidueSelection:
    """
    Creates a ResidueSelection from a selection of residues.

    This function takes an input selection of residues in various formats and creates a `ResidueSelection` 
    object. The selection can be provided as a string, list, or tuple.

    Parameters
    ----------
    input_selection : str, list, or tuple
        The selection of residues to be parsed. This can be:
            - A string with residues separated by a delimiter.
            - A list or tuple of residue strings.
            - A list or tuple of lists/tuples, where each inner list/tuple represents a residue.
    delim : str, optional
        The delimiter used to split the input string if `input_selection` is a string. Default is ",".

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> residue_selection("A1, B2, C3")
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>

    >>> residue_selection(["A1", "B2", "C3"])
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>

    >>> residue_selection([["A", 1], ["B", 2], ["C", 3]])
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>
    """
    return ResidueSelection(input_selection, delim=delim)

def from_dict(input_dict: dict) -> ResidueSelection:
    """
    Creates a ResidueSelection object from a dictionary.

    This function constructs a `ResidueSelection` instance from a dictionary where the keys represent 
    chain identifiers and the values are lists of residue indices. This format specifies a motif in the 
    following way: {chain: [residues], ...}.

    Parameters
    ----------
    input_dict : dict
        A dictionary specifying the motif. The keys are chain identifiers (str) and the values are lists 
        of residue indices (int).

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> input_dict = {"A": [1, 2], "B": [3, 4]}
    >>> from_dict(input_dict)
    <ResidueSelection object representing ('A', 1), ('A', 2), ('B', 3), ('B', 4)>
    """
    return ResidueSelection([f"{chain}{resi}" for chain, res_l in input_dict.items() for resi in res_l])

def from_contig(input_contig: str) -> ResidueSelection:
    """
    Creates a ResidueSelection object from a contig string.

    This function constructs a `ResidueSelection` instance from a contig string. The contig string can specify 
    ranges of residues using a hyphen (-) to denote the range, with residues separated by commas (,). For example, 
    "A1-A3, B5" specifies residues A1, A2, A3, and B5.

    Parameters
    ----------
    input_contig : str
        A contig string specifying the residues. Ranges can be denoted using hyphens, and residues are separated 
        by commas.

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> from_contig("A1-A3, B5")
    <ResidueSelection object representing ('A', 1), ('A', 2), ('A', 3), ('B', 5)>

    >>> from_contig("C1, C3-C5, D2")
    <ResidueSelection object representing ('C', 1), ('C', 3), ('C', 4), ('C', 5), ('D', 2)>
    """
    sel = []
    elements = [x.strip() for x in input_contig.split(",") if x]
    for element in elements:
        subsplit = element.split("-")
        if len(subsplit) > 1:
            sel += [element[0] + str(i) for i in range(int(subsplit[0][1:]), int(subsplit[-1])+1)]
        else:
            sel.append(element)
    return ResidueSelection(sel)

def reduce_to_unique(input_array: list|tuple) -> list|tuple:
    """
    Reduces an input array to its unique elements while preserving order.

    This function takes a list or tuple and returns a new list or tuple containing only the unique elements 
    from the input, with their original order preserved. The type of the returned collection matches the type 
    of the input.

    Parameters
    ----------
    input_array : list or tuple
        The input array from which to remove duplicate elements. The order of the elements is preserved.

    Returns
    -------
    list or tuple
        A new list or tuple containing only the unique elements from the input array, with the original order 
        preserved.

    Examples
    --------
    >>> reduce_to_unique([1, 2, 2, 3, 1])
    [1, 2, 3]

    >>> reduce_to_unique(("a", "b", "a", "c", "b"))
    ('a', 'b', 'c')

    Notes
    -----
    - The function uses `OrderedDict.fromkeys` to remove duplicates while preserving order.
    - The returned collection is of the same type as the input (list or tuple).
    """
    return type(input_array)(OrderedDict.fromkeys(input_array))
