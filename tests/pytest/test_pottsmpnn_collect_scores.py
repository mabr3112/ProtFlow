import pandas as pd

from protflow.tools.pottsmpnn import collect_scores_sample_seqs


def _write_sample_config(tmp_path, out_dir, out_name, input_entries):
    """Write the minimal generated PottsMPNN config used by the collector."""
    input_list = tmp_path / f"{out_name}_input_list.txt"
    input_list.write_text("\n".join(input_entries) + "\n", encoding="utf-8")

    config_dir = tmp_path / "config_files"
    config_dir.mkdir(exist_ok=True)
    config = config_dir / f"{out_name}_config.yaml"
    config.write_text(
        "\n".join([
            f"out_dir: {out_dir}",
            f"out_name: {out_name}",
            f"input_list: {input_list}",
        ]),
        encoding="utf-8",
    )


def test_collect_scores_sample_seqs_adds_pdb_location_for_written_pdb(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    _write_sample_config(
        tmp_path=tmp_path,
        out_dir=out_dir,
        out_name="pose_a",
        input_entries=["pose_a"],
    )

    (out_dir / "pose_a.fasta").write_text(
        ">pose_a_1\nAAAA\n>pose_a_0\nCCCC\n",
        encoding="utf-8",
    )
    pdb_dir = out_dir / "pose_a_pdbs"
    pdb_dir.mkdir()
    pdb_path = pdb_dir / "pose_a.pdb"
    pdb_path.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    scores = collect_scores_sample_seqs(work_dir=str(tmp_path), batched=False)

    pdb_locations = scores.set_index("description")["pdb_location"]
    assert pdb_locations["pose_a_0002"] == str(pdb_path.resolve())
    assert pd.isna(pdb_locations["pose_a_0001"])
