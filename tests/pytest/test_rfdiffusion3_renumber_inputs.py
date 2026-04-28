import gzip
import json

from protflow.tools.rfdiffusion3 import collect_scores, renumber_rfd3_input_pdb


def _atom_line(record: str, serial: int, atom: str, resname: str, chain: str, resseq: int) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           {atom[0]:>2}\n"
    )


def _ter_line(serial: int, resname: str, chain: str, resseq: int) -> str:
    return f"TER   {serial:5d}      {resname:>3} {chain}{resseq:4d}\n"


def test_renumber_rfd3_input_pdb_copies_whole_file_and_renumbers_indexed_residues(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "renumbered.pdb"
    input_pdb.write_text(
        "".join(
            [
                "HEADER    KEEP THIS HEADER\n",
                _atom_line("ATOM", 1, "N", "ALA", "A", 1),
                _atom_line("ANISOU", 1, "N", "ALA", "A", 1),
                _atom_line("ATOM", 2, "CA", "GLY", "A", 2),
                _ter_line(3, "GLY", "A", 2),
                _atom_line("HETATM", 4, "C1", "LIG", "Z", 9),
                "CONECT    4    1\n",
                "END\n",
            ]
        ),
        encoding="UTF-8",
    )

    result = renumber_rfd3_input_pdb(
        input_pdb=str(input_pdb),
        diffused_index_map={"A1": "B10", "A2": "B11"},
        output_pdb=str(output_pdb),
    )

    assert result == str(output_pdb)
    lines = output_pdb.read_text(encoding="UTF-8").splitlines()
    assert lines[0] == "HEADER    KEEP THIS HEADER"
    assert lines[1][21] == "B"
    assert lines[1][22:26] == "  10"
    assert lines[2][21] == "B"
    assert lines[2][22:26] == "  10"
    assert lines[3][21] == "B"
    assert lines[3][22:26] == "  11"
    assert lines[4][21] == "B"
    assert lines[4][22:26] == "  11"
    assert lines[5][21] == "Z"
    assert lines[5][22:26] == "   9"
    assert lines[6] == "CONECT    4    1"


def test_collect_scores_adds_renumbered_inputs_column(tmp_path):
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text(_atom_line("ATOM", 1, "N", "ALA", "A", 1), encoding="UTF-8")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sidecar = output_dir / "batch0_design_0001_0001.json"
    sidecar.write_text(
        json.dumps(
            {
                "diffused_index_map": {"A1": "C5"},
                "metrics": {"plddt": 0.95},
                "specification": {"input": str(input_pdb)},
            }
        ),
        encoding="UTF-8",
    )
    with gzip.open(output_dir / "batch0_design_0001_0001.cif.gz", "wt", encoding="UTF-8") as handle:
        handle.write("data_design\n")

    scores = collect_scores(
        work_dir=str(tmp_path),
        cif_to_pdb=False,
        run_clean=False,
        renumber_input=True,
    )

    renumbered_input = tmp_path / "renumbered_inputs" / "design_0001_0001.pdb"
    assert scores.at[0, "description"] == "design_0001_0001"
    assert scores.at[0, "renumbered_inputs"] == str(renumbered_input)
    assert renumbered_input.is_file()
    assert renumbered_input.read_text(encoding="UTF-8").splitlines()[0][21] == "C"
    assert renumbered_input.read_text(encoding="UTF-8").splitlines()[0][22:26] == "   5"
