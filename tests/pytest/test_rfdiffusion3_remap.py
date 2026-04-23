import logging

import pandas as pd
import pytest

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


def test_remap_rfd3_motifs_preserves_unmapped_atoms_and_warns(caplog):
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{"A1": "B10"}],
                "atom_motif": [AtomSelection([("A", 1, "N"), ("A", 2, "CA")])],
            }
        )
    )

    caplog.set_level(logging.WARNING)
    remap_rfd3_motifs(poses=poses, motifs=["atom_motif"], prefix="rfd3")

    assert poses.df.at[0, "atom_motif"].to_tuple() == (("B", 10, "N"), ("A", 2, "CA"))
    assert "A2" in caplog.text
    assert "keeping their original identifiers unchanged" in caplog.text


def test_remap_rfd3_motifs_preserves_unmapped_residue_selection_members(caplog):
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{"A1": "B10"}],
                "residue_motif": [ResidueSelection(["A1", "Z9"])],
            }
        )
    )

    caplog.set_level(logging.WARNING)
    remap_rfd3_motifs(poses=poses, motifs=["residue_motif"], prefix="rfd3")

    assert poses.df.at[0, "residue_motif"].to_list() == ["B10", "Z9"]
    assert "Z9" in caplog.text


def test_remap_rfd3_motifs_allows_empty_index_map_for_ligand_atoms(caplog):
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [{}],
                "atom_motif": [AtomSelection([("Z", ("H_LIG", 9, " "), "C1")])],
            }
        )
    )

    caplog.set_level(logging.WARNING)
    remap_rfd3_motifs(poses=poses, motifs=["atom_motif"], prefix="rfd3")

    assert poses.df.at[0, "atom_motif"].to_tuple() == (("Z", ("H_LIG", 9, " "), "C1"),)
    assert "Z9" in caplog.text


def test_remap_rfd3_motifs_rejects_missing_index_map_in_strict_mode():
    poses = _Poses(
        pd.DataFrame(
            {
                "rfd3_diffused_index_map": [None],
                "residue_motif": [ResidueSelection(["A1"])],
            }
        )
    )

    with pytest.raises(ValueError, match="no usable diffused_index_map"):
        remap_rfd3_motifs(poses=poses, motifs=["residue_motif"], prefix="rfd3")
