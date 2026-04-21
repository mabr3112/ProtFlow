import logging
import json

from protflow.tools.rosetta import collect_scores


def _write_rosetta_output(tmp_path, raw_description: str, scorefile_name: str) -> None:
    (tmp_path / scorefile_name).write_text(
        json.dumps({"decoy": raw_description, "total_score": -1.23}),
        encoding="utf-8",
    )
    (tmp_path / f"{raw_description}.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")


def test_collect_scores_keeps_descriptions_that_start_with_r(tmp_path):
    raw_description = "r0001_rfd3_motif_0001_0001"
    _write_rosetta_output(tmp_path, raw_description, "r0001_rfd3_motif_0001_score.json")

    scores = collect_scores(work_dir=str(tmp_path))

    assert scores.at[0, "raw_description"] == raw_description
    assert scores.at[0, "description"] == "rfd3_motif_0001_0001"
    assert (tmp_path / "rfd3_motif_0001_0001.pdb").is_file()
    assert not (tmp_path / "motif_0001_fd3.pdb").exists()


def test_collect_scores_does_not_rewrite_already_normalized_numeric_r_prefixes(tmp_path):
    raw_description = "r0001_r1234_foo_0001"
    _write_rosetta_output(tmp_path, raw_description, "r0001_r1234_foo_score.json")

    scores = collect_scores(work_dir=str(tmp_path))

    assert scores.at[0, "description"] == "r1234_foo_0001"
    assert (tmp_path / "r1234_foo_0001.pdb").is_file()
    assert not (tmp_path / "foo_0001_1234.pdb").exists()


def test_collect_scores_skips_missing_pdb_outputs_leniently(tmp_path, monkeypatch, caplog):
    raw_present = "r0001_pose_0001"
    raw_missing = "r0002_pose_0001"
    _write_rosetta_output(tmp_path, raw_present, "r0001_pose_score.json")
    (tmp_path / "r0002_pose_score.json").write_text(
        json.dumps({"decoy": raw_missing, "total_score": -2.0}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "protflow.tools.rosetta._wait_for_expected_rosetta_pdbs",
        lambda expected_pdbs, max_wait_seconds=30.0, poll_interval=1.0: [raw_missing],
    )

    caplog.set_level(logging.WARNING)
    scores = collect_scores(work_dir=str(tmp_path))

    assert scores["raw_description"].to_list() == [raw_present]
    assert scores["description"].to_list() == ["pose_0001"]
    assert (tmp_path / "pose_0001.pdb").is_file()
    assert raw_missing in caplog.text
