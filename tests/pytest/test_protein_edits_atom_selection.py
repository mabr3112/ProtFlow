import pandas as pd
from Bio.PDB import PDBParser

from protflow.residues import AtomSelection, ResidueSelection
from protflow.tools.protein_edits import ChainAdder, setup_chain_list
from protflow.tools.runners_auxiliary_scripts.add_chains_batch import parse_motif_spec, setup_superimpose_atoms, superimpose_add_chain
from protflow.utils.biopython_tools import get_atoms_of_atom_selection


class _FakePoses:
    def __init__(self, rows):
        self.rows = [pd.Series(row) for row in rows]
        self.df = pd.DataFrame(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def _pdb_line(record, serial, atom_name, resname, chain, resseq, x, y, z, element):
    return (
        f"{record:<6}{serial:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}           {element:>2}"
    )


def _write_atom_selection_pdb(tmp_path):
    pdb_path = tmp_path / "atom_selection_chain_add.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                _pdb_line("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 3, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0, "C"),
                _pdb_line("ATOM", 4, "O", "ALA", "A", 1, 0.0, 1.0, 0.0, "O"),
                _pdb_line("HETATM", 5, "C1", "LIG", "Z", 9, 2.0, 0.0, 0.0, "C"),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pdb_path


def test_parse_motif_serializes_atomselection_from_object_and_column():
    adder = object.__new__(ChainAdder)
    selection = AtomSelection([("A", 1, "CB"), ("Z", ("H_LIG", 9, " "), "C1")])
    row = pd.Series({"motif_col": selection})

    assert adder.parse_motif(selection, row) == selection.to_dict()
    assert adder.parse_motif(selection.to_dict(), row) == selection.to_dict()
    assert adder.parse_motif("motif_col", row) == selection.to_dict()


def test_parse_motif_keeps_residueselection_legacy_string():
    adder = object.__new__(ChainAdder)
    selection = ResidueSelection(["A1"])

    assert adder.parse_motif(selection, pd.Series({})) == "A1"


def test_worker_parses_atomselection_json_contract():
    selection = AtomSelection([("A", 1, "CB")])

    parsed = parse_motif_spec(selection.to_dict())

    assert isinstance(parsed, AtomSelection)
    assert parsed.to_tuple() == selection.to_tuple()


def test_get_atoms_of_atom_selection_resolves_compact_hetatm_ids(tmp_path):
    model = PDBParser(QUIET=True).get_structure("atoms", _write_atom_selection_pdb(tmp_path))[0]

    atoms = get_atoms_of_atom_selection(model, [("Z", 9, "C1")])

    assert [atom.name for atom in atoms] == ["C1"]


def test_setup_superimpose_atoms_uses_exact_atomselection_atoms(tmp_path):
    model = PDBParser(QUIET=True).get_structure("atoms", _write_atom_selection_pdb(tmp_path))[0]
    target_selection = AtomSelection([("A", 1, "CB"), ("Z", ("H_LIG", 9, " "), "C1")])
    reference_selection = AtomSelection([("A", 1, "O"), ("A", 1, "CA")])

    target_atoms, reference_atoms = setup_superimpose_atoms(
        target=model,
        reference=model,
        target_motif=target_selection,
        reference_motif=reference_selection,
    )

    assert [atom.name for atom in target_atoms] == ["CB", "C1"]
    assert [atom.name for atom in reference_atoms] == ["O", "CA"]


def test_setup_chain_list_accepts_multiple_copy_chains():
    poses = _FakePoses([
        {"copy_chains": ["B", "C"]},
        {"copy_chains": "D"},
    ])

    assert setup_chain_list(["B", "C"], poses) == [["B", "C"], ["B", "C"]]
    assert setup_chain_list("copy_chains", poses) == [["B", "C"], "D"]


def test_add_chain_forwards_translate_x_and_chain_mapping_to_superimpose_add_chain():
    adder = object.__new__(ChainAdder)
    calls = {}

    def fake_superimpose_add_chain(**kwargs):
        calls.update(kwargs)
        return kwargs["poses"]

    adder.superimpose_add_chain = fake_superimpose_add_chain
    poses = object()

    result = adder.add_chain(
        poses=poses,
        prefix="multi",
        ref_col="ref",
        copy_chain=["B", "C"],
        translate_x=12.5,
        chain_mapping={"B": "D", "X": "Y"},
    )

    assert result is poses
    assert calls["copy_chain"] == ["B", "C"]
    assert calls["translate_x"] == 12.5
    assert calls["chain_mapping"] == {"B": "D", "X": "Y"}


def test_setup_superimposition_args_serializes_chain_mapping(tmp_path):
    adder = object.__new__(ChainAdder)
    target_path = str(tmp_path / "target.pdb")
    reference_path = str(tmp_path / "reference.pdb")
    poses = _FakePoses([{"poses": target_path, "ref": reference_path}])

    args = adder._setup_superimposition_args(
        poses=poses,
        ref_col="ref",
        copy_chain=["B", "C"],
        chain_mapping={"B": "D", "X": "Y"},
    )

    assert args[target_path]["copy_chain"] == ["B", "C"]
    assert args[target_path]["chain_mapping"] == {"B": "D", "X": "Y"}


def test_worker_superimpose_add_chain_accepts_copy_chain_list_and_chain_mapping(tmp_path):
    target_path = tmp_path / "target.pdb"
    target_path.write_text(
        "\n".join([
            _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            _pdb_line("ATOM", 2, "CA", "THR", "B", 1, 3.0, 0.0, 0.0, "C"),
            "TER",
            "END",
        ]) + "\n",
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.pdb"
    reference_path.write_text(
        "\n".join([
            _pdb_line("ATOM", 1, "CA", "GLY", "B", 1, 1.0, 0.0, 0.0, "C"),
            _pdb_line("ATOM", 2, "CA", "SER", "C", 1, 2.0, 0.0, 0.0, "C"),
            "TER",
            "END",
        ]) + "\n",
        encoding="utf-8",
    )

    parser = PDBParser(QUIET=True)
    target = parser.get_structure("target", target_path)[0]
    reference = parser.get_structure("reference", reference_path)[0]

    copied = superimpose_add_chain(target, reference, ["B", "C"], chain_mapping={"B": "D", "X": "Y"})

    assert [chain.id for chain in copied.get_chains()] == ["A", "B", "D", "C"]

