import json

import pandas as pd

from protflow.poses import Poses
from protflow.residues import AtomSelection
from protflow.tools.rfdiffusion3 import RFD3Params, RFdiffusion3


def _pdb_line(record, serial, atom_name, resname, chain, resseq, x, y, z, element):
    return (
        f"{record:<6}{serial:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}           {element:>2}"
    )


def _write_test_pdb(path, ligand_chain="Z", ligand_resseq=9):
    lines = [
        _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        _pdb_line("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 3, "C", "ALA", "A", 1, 2.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 4, "O", "ALA", "A", 1, 3.0, 0.0, 0.0, "O"),
        _pdb_line("HETATM", 5, "C1", "LIG", ligand_chain, ligand_resseq, 0.0, 2.0, 0.0, "C"),
        _pdb_line("HETATM", 6, "O1", "LIG", ligand_chain, ligand_resseq, 1.0, 2.0, 0.0, "O"),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="UTF-8")
    return path


def _runner_without_config(model_dir):
    runner = object.__new__(RFdiffusion3)
    runner.application_path = "rfd3"
    runner.model_dir = str(model_dir)
    runner.pre_cmd = None
    runner.jobstarter = None
    runner.name = "rfdiffusion3"
    runner.index_layers = 1
    return runner


def test_run_parse_atomic_motifs_adds_input_spec_atom_selection_columns_from_cached_scores(tmp_path):
    input_pdb = _write_test_pdb(tmp_path / "input.pdb")
    output_pdb = _write_test_pdb(tmp_path / "input_0001_0001_0001.pdb", ligand_chain="C", ligand_resseq=12)
    ckpt_path = tmp_path / "rfd3_latest.ckpt"
    ckpt_path.write_text("checkpoint placeholder\n", encoding="UTF-8")

    poses = Poses(poses=[str(input_pdb)], work_dir=str(tmp_path))
    params = RFD3Params(poses=poses)
    params.set_input_specs(contig="A1", ligand="LIG", select_fixed_atoms={"LIG": "C1"})

    work_dir = tmp_path / "rfd3"
    work_dir.mkdir()
    output_dir = work_dir / "outputs"
    output_dir.mkdir()
    (output_dir / f"{output_pdb.stem}.json").write_text(
        json.dumps(
            {
                "diffused_index_map": {"A1": "B10"},
                "specification": {"input": str(input_pdb), "ligand": "LIG"},
            }
        ),
        encoding="UTF-8",
    )
    pd.DataFrame(
        {
            "description": [output_pdb.stem],
            "location": [str(output_pdb)],
            "diffused_index_map": [{"A1": "B10"}],
        }
    ).to_json(work_dir / "rfdiffusion3_scores.json")

    runner = _runner_without_config(model_dir=tmp_path)
    poses = runner.run(
        prefix="rfd3",
        poses=poses,
        params=params,
        ckpt_path=str(ckpt_path),
        parse_atomic_motifs=True,
    )

    assert isinstance(poses.df.at[0, "rfd3_contig"], AtomSelection)
    assert poses.df.at[0, "rfd3_contig"].to_tuple() == (
        ("B", 10, "N"),
        ("B", 10, "CA"),
        ("B", 10, "C"),
        ("B", 10, "O"),
    )
    assert poses.df.at[0, "rfd3_contig_original"].to_tuple() == (
        ("A", 1, "N"),
        ("A", 1, "CA"),
        ("A", 1, "C"),
        ("A", 1, "O"),
    )
    assert poses.df.at[0, "rfd3_ligands"].to_tuple() == (
        ("C", ("H_LIG", 12, " "), "C1"),
        ("C", ("H_LIG", 12, " "), "O1"),
    )
    assert poses.df.at[0, "rfd3_ligands_original"].to_tuple() == (
        ("Z", ("H_LIG", 9, " "), "C1"),
        ("Z", ("H_LIG", 9, " "), "O1"),
    )
    assert poses.df.at[0, "rfd3_ligands_fixed_atoms"].to_tuple() == (
        ("C", ("H_LIG", 12, " "), "C1"),
    )
    assert poses.df.at[0, "rfd3_ligands_fixed_atoms_original"].to_tuple() == (
        ("Z", ("H_LIG", 9, " "), "C1"),
    )
    assert poses.df.at[0, "rfd3_ligand"].to_tuple() == poses.df.at[0, "rfd3_ligands"].to_tuple()
