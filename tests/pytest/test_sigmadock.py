"""Unit tests for protflow.tools.sigmadock.SigmaDock runner.

These tests cover the pure logic and file-system side effects of the runner
without executing SigmaDock itself. subprocess.run and split_complex are
mocked throughout.
"""
import json
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from protflow.tools.sigmadock import SigmaDock, collect_scores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    # Bypass config loading — supply all paths explicitly.
    with patch("protflow.tools.sigmadock.require_config"), \
         patch("protflow.tools.sigmadock.load_config_path", return_value=""):
        r = SigmaDock(
            application_path="/fake/sample.py",
            python_path="/fake/python",
            ckpt_path="/fake/ckpt",
        )
    return r


def _make_poses(pdb_paths, ligand_paths, ref_ligand_paths=None):
    """Return a minimal Poses-like mock with pre-filled sigmadock columns."""
    poses = MagicMock()
    poses.df = pd.DataFrame({
        "poses": pdb_paths,
        "sigmadock_pdb": pdb_paths,
        "sigmadock_ligands": ligand_paths,
        "sigmadock_ref_ligand": ref_ligand_paths if ref_ligand_paths is not None else [None] * len(pdb_paths),
    })
    poses.poses_list.return_value = pdb_paths
    return poses


# ---------------------------------------------------------------------------
# write_cmd
# ---------------------------------------------------------------------------

def test_write_cmd_contains_python_and_script(runner, tmp_path):
    # write_cmd must embed the configured python and application paths.
    cmd = runner.write_cmd(
        pose_path="/input/complex_A.pdb",
        out_dir=str(tmp_path),
        cli_args="ckpt=/fake/ckpt",
    )
    assert "/fake/python" in cmd
    assert "/fake/sample.py" in cmd


def test_write_cmd_output_dir_derived_from_stem(runner, tmp_path):
    # output_dir must be {out_dir}/{stem} where stem = basename without extension.
    cmd = runner.write_cmd(
        pose_path="/input/complex_A.pdb",
        out_dir=str(tmp_path),
        cli_args="",
    )
    expected_out = os.path.join(str(tmp_path), "complex_A")
    assert f"output_dir={expected_out}" in cmd


def test_write_cmd_cli_args_appended(runner, tmp_path):
    # Extra CLI args must appear verbatim in the command.
    cmd = runner.write_cmd(
        pose_path="/input/x.pdb",
        out_dir=str(tmp_path),
        cli_args="inference.protein_pdb=/a/b.pdb inference.ligand_sdf=/a/c.sdf",
    )
    assert "inference.protein_pdb=/a/b.pdb" in cmd
    assert "inference.ligand_sdf=/a/c.sdf" in cmd


# ---------------------------------------------------------------------------
# _build_sigmadock_cli_opts — redocking
# ---------------------------------------------------------------------------

def test_redocking_cli_opts_contains_pdb_and_sdf(runner, tmp_path):
    # Redocking (ligands is a str) must produce inline protein_pdb + ligand_sdf opts.
    poses = _make_poses(
        ["/input/complex_A.pdb"],
        ["/work/inputs/complex_A_ligand.sdf"],
    )
    opts = runner._build_sigmadock_cli_opts(poses=poses, work_dir=str(tmp_path))
    assert len(opts) == 1
    assert "inference.protein_pdb=" in opts[0]
    assert "inference.ligand_sdf=" in opts[0]
    assert "inference.inference_datafront" not in opts[0]


# ---------------------------------------------------------------------------
# _build_sigmadock_cli_opts — crossdocking
# ---------------------------------------------------------------------------

def test_crossdocking_cli_opts_writes_csv(runner, tmp_path):
    # Crossdocking (ligands is a list) must write an _inference.csv and reference it.
    # inputs/ must exist — normally created by _prepare_sigmadock_inputs.
    (tmp_path / "inputs").mkdir()
    poses = _make_poses(
        ["/input/complex_A.pdb"],
        [["/data/query_ligand.sdf"]],
    )
    opts = runner._build_sigmadock_cli_opts(poses=poses, work_dir=str(tmp_path))
    assert len(opts) == 1
    assert "inference.inference_datafront=" in opts[0]


def test_crossdocking_csv_columns(runner, tmp_path):
    # The written CSV must have PDB, SDF, REF_SDF columns.
    (tmp_path / "inputs").mkdir()
    poses = _make_poses(
        ["/input/complex_A.pdb"],
        [["/data/query_ligand.sdf"]],
    )
    runner._build_sigmadock_cli_opts(poses=poses, work_dir=str(tmp_path))
    csv_path = tmp_path / "inputs" / "complex_A_inference.csv"
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"PDB", "SDF", "REF_SDF"}


def test_crossdocking_query_ligand_path_is_absolute(runner, tmp_path):
    # Relative query ligand paths must be converted to absolute — regression for
    # the doubled-path bug where SigmaDock prepended its work dir to a relative path.
    (tmp_path / "inputs").mkdir()
    poses = _make_poses(
        ["/input/complex_A.pdb"],
        [["relative/path/ligand.sdf"]],
    )
    runner._build_sigmadock_cli_opts(poses=poses, work_dir=str(tmp_path))
    csv_path = tmp_path / "inputs" / "complex_A_inference.csv"
    df = pd.read_csv(csv_path)
    assert os.path.isabs(df["SDF"].iloc[0])


def test_crossdocking_pdb_and_ref_sdf_are_absolute(runner, tmp_path):
    # PDB and REF_SDF paths written to the CSV must always be absolute.
    (tmp_path / "inputs").mkdir()
    poses = _make_poses(
        ["/input/complex_A.pdb"],
        [["/data/query_ligand.sdf"]],
        ref_ligand_paths=["/data/ref_ligand.sdf"],
    )
    runner._build_sigmadock_cli_opts(poses=poses, work_dir=str(tmp_path))
    csv_path = tmp_path / "inputs" / "complex_A_inference.csv"
    df = pd.read_csv(csv_path)
    assert os.path.isabs(df["PDB"].iloc[0])
    assert os.path.isabs(df["REF_SDF"].iloc[0])


# ---------------------------------------------------------------------------
# _cleanup_previous_outputs
# ---------------------------------------------------------------------------

def test_cleanup_removes_inputs_and_outputs(runner, tmp_path):
    # Both inputs/ and outputs/ must be deleted after cleanup.
    (tmp_path / "inputs").mkdir()
    (tmp_path / "outputs").mkdir()
    runner._cleanup_previous_outputs(work_dir=str(tmp_path))
    assert not (tmp_path / "inputs").exists()
    assert not (tmp_path / "outputs").exists()


def test_cleanup_does_not_raise_if_dirs_missing(runner, tmp_path):
    # Calling cleanup on a fresh work_dir with no subdirs must not raise.
    runner._cleanup_previous_outputs(work_dir=str(tmp_path))


def test_cleanup_leaves_other_files_intact(runner, tmp_path):
    # Only inputs/ and outputs/ are removed; sibling files must survive.
    (tmp_path / "inputs").mkdir()
    other = tmp_path / "sigmadock_runner_scores.json"
    other.write_text("{}")
    runner._cleanup_previous_outputs(work_dir=str(tmp_path))
    assert other.exists()


# ---------------------------------------------------------------------------
# _prepare_sigmadock_inputs
# ---------------------------------------------------------------------------

def test_prepare_sets_sigmadock_pdb_column(runner, tmp_path):
    # After preparation, poses.df must have a sigmadock_pdb column with expected paths.
    poses = MagicMock()
    poses.df = pd.DataFrame({"poses": ["/input/complex_A.pdb"]})
    poses.poses_list.return_value = ["/input/complex_A.pdb"]

    with patch("protflow.tools.sigmadock.split_complex"):
        runner._prepare_sigmadock_inputs(
            poses=poses, work_dir=str(tmp_path), ligand_name="LIG"
        )

    expected_pdb = os.path.join(str(tmp_path), "inputs", "complex_A_LIG.pdb")
    assert poses.df["sigmadock_pdb"].iloc[0] == expected_pdb


def test_prepare_redocking_ligands_column_is_string(runner, tmp_path):
    # Redocking: sigmadock_ligands must be a plain string path, not a list.
    poses = MagicMock()
    poses.df = pd.DataFrame({"poses": ["/input/complex_A.pdb"]})
    poses.poses_list.return_value = ["/input/complex_A.pdb"]

    with patch("protflow.tools.sigmadock.split_complex"):
        runner._prepare_sigmadock_inputs(
            poses=poses, work_dir=str(tmp_path), ligand_name="LIG", query_ligands=None
        )

    assert isinstance(poses.df["sigmadock_ligands"].iloc[0], str)


def test_prepare_crossdocking_ligands_column_is_list(runner, tmp_path):
    # Crossdocking: sigmadock_ligands must be the query_ligands list.
    poses = MagicMock()
    poses.df = pd.DataFrame({"poses": ["/input/complex_A.pdb"]})
    poses.poses_list.return_value = ["/input/complex_A.pdb"]
    query = ["/data/lig_1.sdf", "/data/lig_2.sdf"]

    with patch("protflow.tools.sigmadock.split_complex"):
        runner._prepare_sigmadock_inputs(
            poses=poses, work_dir=str(tmp_path), ligand_name="LIG", query_ligands=query
        )

    assert poses.df["sigmadock_ligands"].iloc[0] == query


def test_prepare_skips_split_if_pdb_exists_and_no_overwrite(runner, tmp_path):
    # split_complex must not be called when overwrite=False and the receptor PDB exists.
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "complex_A_LIG.pdb").write_text("existing")

    poses = MagicMock()
    poses.df = pd.DataFrame({"poses": ["/input/complex_A.pdb"]})
    poses.poses_list.return_value = ["/input/complex_A.pdb"]

    with patch("protflow.tools.sigmadock.split_complex") as mock_split:
        runner._prepare_sigmadock_inputs(
            poses=poses, work_dir=str(tmp_path), ligand_name="LIG", overwrite=False
        )
    mock_split.assert_not_called()


def test_prepare_calls_split_when_overwrite_true(runner, tmp_path):
    # split_complex must be called even if the PDB exists when overwrite=True.
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "complex_A_LIG.pdb").write_text("existing")

    poses = MagicMock()
    poses.df = pd.DataFrame({"poses": ["/input/complex_A.pdb"]})
    poses.poses_list.return_value = ["/input/complex_A.pdb"]

    with patch("protflow.tools.sigmadock.split_complex") as mock_split:
        runner._prepare_sigmadock_inputs(
            poses=poses, work_dir=str(tmp_path), ligand_name="LIG", overwrite=True
        )
    mock_split.assert_called_once()


# ---------------------------------------------------------------------------
# collect_scores
# ---------------------------------------------------------------------------

def test_collect_scores_returns_dataframe(tmp_path):
    # collect_scores must return a DataFrame parsed from the JSON the script writes.
    rows = [
        {"description": "pose_A", "location": "/out/pose_A.pdb", "affinity": -6.5},
        {"description": "pose_B", "location": "/out/pose_B.pdb", "affinity": -7.1},
    ]

    def fake_subprocess(cmd, check):
        # Simulate the auxiliary script writing the JSON.
        json_path = os.path.join(cmd[2], "sigmadock_scores.json")
        with open(json_path, "w") as f:
            json.dump(rows, f)

    with patch("subprocess.run", side_effect=fake_subprocess):
        scores = collect_scores(work_dir=str(tmp_path), python_path="/fake/python")

    assert isinstance(scores, pd.DataFrame)
    assert list(scores["description"]) == ["pose_A", "pose_B"]
    assert "affinity" in scores.columns


def test_collect_scores_json_is_removed_after_read(tmp_path):
    # The temporary sigmadock_scores.json must be deleted after parsing.
    rows = [{"description": "x", "location": "/out/x.pdb"}]

    def fake_subprocess(cmd, check):
        json_path = os.path.join(cmd[2], "sigmadock_scores.json")
        with open(json_path, "w") as f:
            json.dump(rows, f)

    with patch("subprocess.run", side_effect=fake_subprocess):
        collect_scores(work_dir=str(tmp_path), python_path="/fake/python")

    assert not (tmp_path / "sigmadock_scores.json").exists()


def test_collect_scores_passes_include_scores_to_script(tmp_path):
    # include_scores must be forwarded to the auxiliary script as --include-scores.
    rows = [{"description": "x", "location": "/out/x.pdb"}]
    captured_cmd = []

    def fake_subprocess(cmd, check):
        captured_cmd.extend(cmd)
        json_path = os.path.join(cmd[2], "sigmadock_scores.json")
        with open(json_path, "w") as f:
            json.dump(rows, f)

    with patch("subprocess.run", side_effect=fake_subprocess):
        collect_scores(
            work_dir=str(tmp_path),
            python_path="/fake/python",
            include_scores=["affinity", "cnn_score"],
        )

    assert "--include-scores" in captured_cmd
    idx = captured_cmd.index("--include-scores")
    assert "affinity" in captured_cmd[idx + 1]
    assert "cnn_score" in captured_cmd[idx + 1]
