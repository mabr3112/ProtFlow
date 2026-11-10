import pytest
from Bio.PDB import PDBParser

from protflow.metrics.biopython_metrics import ContactOrder, Distance


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



def _write_contact_order_pdb(tmp_path):
    pdb_path = tmp_path / "contact_order.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 2, "CA", "ALA", "A", 2, 100.0, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 3, "CA", "ALA", "A", 3, 0.0, 3.0, 0.0, "C"),
                _pdb_line("ATOM", 4, "CA", "ALA", "A", 4, 0.0, 0.0, 4.0, "C"),
                _pdb_line("ATOM", 5, "CA", "ALA", "B", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 6, "CA", "ALA", "B", 2, 0.0, 0.0, 3.0, "C"),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pdb_path


def test_contact_order_calculates_relative_sequence_separation(tmp_path):
    model = PDBParser(QUIET=True).get_structure("contact_order", _write_contact_order_pdb(tmp_path))[0]

    contact_order = ContactOrder().calc(model, contact_distance=5.0, chains="A")

    # Chain A contacts are residue pairs 1-3, 1-4, and 3-4.
    # CO = (2 + 3 + 1) / (L=4 * N=3) = 0.5.
    assert contact_order == pytest.approx(0.5)


def test_contact_order_honors_min_sequence_separation(tmp_path):
    model = PDBParser(QUIET=True).get_structure("contact_order", _write_contact_order_pdb(tmp_path))[0]

    contact_order = ContactOrder().calc(model, contact_distance=5.0, min_sequence_separation=2, chains="A")

    # Excluding sequence-neighbor contacts removes the 3-4 contact.
    # CO = (2 + 3) / (L=4 * N=2) = 0.625.
    assert contact_order == pytest.approx(0.625)


def test_contact_order_uses_intrachain_contacts_only(tmp_path):
    model = PDBParser(QUIET=True).get_structure("contact_order", _write_contact_order_pdb(tmp_path))[0]

    contact_order = ContactOrder().calc(model, contact_distance=5.0)

    # Default all-chain calculation includes intrachain contacts from A and B,
    # but excludes spatially close interchain residue pairs.
    assert contact_order == pytest.approx(7 / (6 * 4))


def test_contact_order_raises_without_contacts(tmp_path):
    model = PDBParser(QUIET=True).get_structure("contact_order", _write_contact_order_pdb(tmp_path))[0]

    with pytest.raises(ValueError, match="no residue contacts"):
        ContactOrder().calc(model, contact_distance=1.0, chains="A")
