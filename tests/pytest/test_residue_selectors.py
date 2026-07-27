from protflow.residues import ResidueSelection
from protflow.tools.residue_selectors import DistanceSelector


def _pdb_line(record, serial, atom_name, resname, chain, resseq, x):
    return (
        f"{record:<6}{serial:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{20.0:>6.2f}            C"
    )


def test_distance_selector_excludes_every_center_residue(tmp_path):
    pdb_path = tmp_path / "multiple_centers.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0),
                _pdb_line("HETATM", 2, "C1", "L1", "B", 1, 1.0),
                _pdb_line("HETATM", 3, "C1", "L2", "C", 1, 10.0),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = DistanceSelector().select_single(
        pose_path=str(pdb_path),
        center=ResidueSelection(["B1", "C1"]),
        distance=2.0,
        operator="<=",
    )

    assert selected.to_list() == ["A1"]
