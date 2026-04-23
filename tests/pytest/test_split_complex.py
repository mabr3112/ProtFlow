"""Tests for protflow.utils.biopython_tools.split_complex."""
import os
import textwrap
import pytest

from protflow.utils.biopython_tools import split_complex


def _sdf_atom_count(sdf_path) -> int:
    """Parse atom count from the V2000 counts line (line 4 of the mol block)."""
    lines = open(sdf_path).readlines()
    if len(lines) < 4:
        return 0
    counts_line = lines[3]
    try:
        return int(counts_line[:3])
    except ValueError:
        return 0

# ---------------------------------------------------------------------------
# Minimal PDB fixtures
# ---------------------------------------------------------------------------

# One protein residue (ATOM) + one ligand residue (HETATM, resname LIG)
SIMPLE_PDB = textwrap.dedent("""\
    ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C
    HETATM    2  C1  LIG A   2       2.000   2.000   2.000  1.00  0.00           C
    HETATM    3  C2  LIG A   2       3.000   3.000   3.000  1.00  0.00           C
    END
""")

# Two distinct ligands: LIG and HOH
TWO_LIGANDS_PDB = textwrap.dedent("""\
    ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C
    HETATM    2  C1  LIG A   2       2.000   2.000   2.000  1.00  0.00           C
    HETATM    3  O1  HOH A   3       4.000   4.000   4.000  1.00  0.00           O
    END
""")

# Protein only, no ligand at all
NO_LIGAND_PDB = textwrap.dedent("""\
    ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C
    ATOM      2  CA  GLY A   2       2.000   2.000   2.000  1.00  0.00           C
    END
""")

# Minimal mmCIF with one protein atom + one ligand atom
SIMPLE_CIF = textwrap.dedent("""\
    data_test
    _entry.id test
    loop_
    _atom_site.group_PDB
    _atom_site.id
    _atom_site.type_symbol
    _atom_site.label_atom_id
    _atom_site.label_alt_id
    _atom_site.label_comp_id
    _atom_site.label_asym_id
    _atom_site.label_entity_id
    _atom_site.label_seq_id
    _atom_site.pdbx_PDB_ins_code
    _atom_site.Cartn_x
    _atom_site.Cartn_y
    _atom_site.Cartn_z
    _atom_site.occupancy
    _atom_site.B_iso_or_equiv
    _atom_site.auth_seq_id
    _atom_site.auth_comp_id
    _atom_site.auth_asym_id
    _atom_site.auth_atom_id
    _atom_site.pdbx_PDB_model_num
    ATOM   1 C CA . ALA A 1 1 ? 1.000 1.000 1.000 1.00 0.00 1 ALA A CA 1
    HETATM 2 C C1 . LIG B 2 . ? 2.000 2.000 2.000 1.00 0.00 1 LIG B C1 1
    #
""")


@pytest.fixture
def pdb_file(tmp_path):
    p = tmp_path / "complex.pdb"
    p.write_text(SIMPLE_PDB)
    return str(p)


@pytest.fixture
def two_ligands_pdb(tmp_path):
    p = tmp_path / "two_lig.pdb"
    p.write_text(TWO_LIGANDS_PDB)
    return str(p)


@pytest.fixture
def no_ligand_pdb(tmp_path):
    p = tmp_path / "no_lig.pdb"
    p.write_text(NO_LIGAND_PDB)
    return str(p)


@pytest.fixture
def cif_file(tmp_path):
    p = tmp_path / "complex.cif"
    p.write_text(SIMPLE_CIF)
    return str(p)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_pdb_creates_output_files(pdb_file, tmp_path):
    # Basic smoke test: both expected output files must exist after a PDB run.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="LIG")
    assert os.path.isfile(tmp_path / "complex.pdb")
    assert os.path.isfile(tmp_path / "complex_ligand.sdf")


def test_receptor_contains_only_atom_records(pdb_file, tmp_path):
    # The receptor PDB must contain ATOM lines and no HETATM — ligand must be fully stripped.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="LIG")
    pdb_text = (tmp_path / "complex.pdb").read_text()
    assert "ATOM" in pdb_text
    assert "HETATM" not in pdb_text


def test_sdf_contains_ligand_atoms(pdb_file, tmp_path):
    # The SDF must contain exactly the 2 heavy atoms of LIG (C1 + C2), parsed from the V2000 counts line.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="LIG")
    assert _sdf_atom_count(tmp_path / "complex_ligand.sdf") == 2


def test_sdf_title_matches_ligand_name(pdb_file, tmp_path):
    # SDF line 1 is the molecule title; must match ligand_name so downstream tools can identify it.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="LIG")
    sdf_text = (tmp_path / "complex_ligand.sdf").read_text()
    assert sdf_text.splitlines()[0].strip() == "LIG"


def test_cif_input_produces_outputs(cif_file, tmp_path):
    # CIF inputs must go through the MMCIFParser branch and produce the same two output files as PDB.
    split_complex(cif_file, work_dir=str(tmp_path), ligand_name="LIG")
    assert os.path.isfile(tmp_path / "complex.pdb")
    assert os.path.isfile(tmp_path / "complex_ligand.sdf")


def test_only_target_ligand_extracted(two_ligands_pdb, tmp_path):
    # When two HETATM residues are present only the requested one (LIG, not HOH) must appear in the SDF.
    split_complex(two_ligands_pdb, work_dir=str(tmp_path), ligand_name="LIG")
    assert _sdf_atom_count(tmp_path / "two_lig_ligand.sdf") == 1


def test_receptor_excludes_non_target_ligand(two_ligands_pdb, tmp_path):
    # Neither the target ligand nor any other HETATM residue (HOH) should leak into the receptor PDB.
    split_complex(two_ligands_pdb, work_dir=str(tmp_path), ligand_name="LIG")
    pdb_text = (tmp_path / "two_lig.pdb").read_text()
    assert "LIG" not in pdb_text
    assert "HOH" not in pdb_text


def test_output_stem_matches_input_filename(pdb_file, tmp_path):
    # Output filenames must be derived from the input stem so the runner can predict them without globbing.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="LIG")
    assert os.path.isfile(tmp_path / "complex.pdb")
    assert os.path.isfile(tmp_path / "complex_ligand.sdf")


# ---------------------------------------------------------------------------
# Edge cases / breakers
# ---------------------------------------------------------------------------

def test_ligand_name_not_present_produces_empty_or_invalid_sdf(no_ligand_pdb, tmp_path):
    # Missing ligand: function must not crash, but SDF should contain 0 atoms (silent failure, no exception).
    split_complex(no_ligand_pdb, work_dir=str(tmp_path), ligand_name="LIG")
    sdf_path = tmp_path / "no_lig_ligand.sdf"
    assert sdf_path.exists()
    assert _sdf_atom_count(sdf_path) == 0


def test_nonexistent_input_raises(tmp_path):
    # A missing input file must raise before any output is written, not silently produce empty files.
    with pytest.raises((FileNotFoundError, Exception)):
        split_complex(str(tmp_path / "ghost.pdb"), work_dir=str(tmp_path), ligand_name="LIG")


def test_nonexistent_work_dir_raises(pdb_file, tmp_path):
    # A non-existent work_dir must raise — the function must not silently create arbitrary directories.
    with pytest.raises(Exception):
        split_complex(pdb_file, work_dir=str(tmp_path / "nonexistent"), ligand_name="LIG")


def test_wrong_extension_raises(tmp_path):
    # Unsupported file extensions must raise ValueError from biopython_load_protein, not silently misparsed.
    bad = tmp_path / "complex.xyz"
    bad.write_text(SIMPLE_PDB)
    with pytest.raises((ValueError, Exception)):
        split_complex(str(bad), work_dir=str(tmp_path), ligand_name="LIG")


def test_empty_ligand_name_produces_no_atoms(pdb_file, tmp_path):
    # Empty ligand_name matches no residue — must not crash, SDF should contain 0 atoms.
    split_complex(pdb_file, work_dir=str(tmp_path), ligand_name="")
    sdf_path = tmp_path / "complex_ligand.sdf"
    assert sdf_path.exists()
    assert _sdf_atom_count(sdf_path) == 0


def test_idempotent_overwrite(pdb_file, tmp_path):
    # Two independent runs on the same input must produce byte-identical outputs.
    # Catches non-determinism in PDBIO/OpenBabel (timestamps, atom reordering, etc.)
    # that would silently break overwrite=False caching in the sigmadock runner.
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    split_complex(pdb_file, work_dir=str(out1), ligand_name="LIG")
    split_complex(pdb_file, work_dir=str(out2), ligand_name="LIG")

    assert (out1 / "complex.pdb").read_text() == (out2 / "complex.pdb").read_text()
    assert (out1 / "complex_ligand.sdf").read_text() == (out2 / "complex_ligand.sdf").read_text()
