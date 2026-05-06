import pytest
from Bio.PDB import PDBParser

from protflow.metrics.biopython_metrics import Distance


def _pdb_line(record, serial, atom_name, resname, chain, resseq, x, y, z, element):
    return (
        f"{record:<6}{serial:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}           {element:>2}"
    )


def _write_ligand_pdb(tmp_path):
    pdb_path = tmp_path / "biopython_metric_ligand.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_line("HETATM", 2, "P09", "LIG", "B", 1, 3.0, 0.0, 0.0, "P"),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pdb_path


def test_distance_resolves_compact_hetatm_residue_id_when_standard_lookup_fails(tmp_path):
    model = PDBParser(QUIET=True).get_structure("ligand", _write_ligand_pdb(tmp_path))[0]

    distance = Distance().calc(model, [("A", 1, "CA"), ("B", 1, "P09")])

    assert distance == pytest.approx(3.0)


def test_distance_resolves_explicit_biopython_hetatm_residue_id(tmp_path):
    model = PDBParser(QUIET=True).get_structure("ligand", _write_ligand_pdb(tmp_path))[0]

    distance = Distance().calc(model, [("A", 1, "CA"), ("B", ("H_LIG", 1, " "), "P09")])

    assert distance == pytest.approx(3.0)
