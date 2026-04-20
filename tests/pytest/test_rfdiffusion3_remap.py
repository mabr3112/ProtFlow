import pandas as pd

from protflow.residues import AtomSelection, ResidueSelection
from protflow.tools.rfdiffusion3 import remap_rfd3_motifs


class _Poses:
    def __init__(self, df):
        self.df = df


def test_remap_rfd3_motifs_updates_residue_and_atom_selection_columns():
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{"A1": "B10", "A2": "B11", "Z9": "C12"}],
                "residue_motif": [ResidueSelection(["A1", "A2"])],
                "atom_motif": [AtomSelection([("A", 1, "N"), ("A", 2, "CA"), ("Z", ("H_LIG", 9, " "), "C1")])],
            }
        )
    )

    remap_rfd3_motifs(poses=poses, motifs=["residue_motif", "atom_motif"], prefix="rfd3")

    assert poses.df.at[0, "residue_motif"].residues == (("B", 10), ("B", 11))
    assert poses.df.at[0, "atom_motif"].to_tuple() == (
        ("B", 10, "N"),
        ("B", 11, "CA"),
        ("C", ("H_LIG", 12, " "), "C1"),
    )


def test_remap_rfd3_motifs_preserves_full_atom_id_shape():
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{"A1": "B10", "A2": "B11"}],
                "atom_motif": [
                    AtomSelection(
                        [
                            (0, "A", 1, "N"),
                            ("structure", 0, "A", 2, "CA"),
                            ("structure", 0, "A", 2, "CB", "A"),
                        ]
                    )
                ],
            }
        )
    )

    remap_rfd3_motifs(poses=poses, motifs=["atom_motif"], prefix="rfd3")

    assert poses.df.at[0, "atom_motif"].to_tuple() == (
        (0, "B", 10, "N"),
        ("structure", 0, "B", 11, "CA"),
        ("structure", 0, "B", 11, "CB", "A"),
    )


def test_remap_rfd3_motifs_skips_unmapped_atoms_when_not_strict():
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{"A1": "B10"}],
                "atom_motif": [AtomSelection([("A", 1, "N"), ("A", 2, "CA")])],
            }
        )
    )

    remap_rfd3_motifs(poses=poses, motifs=["atom_motif"], prefix="rfd3", strict=False)

    assert poses.df.at[0, "atom_motif"].to_tuple() == (("B", 10, "N"),)
