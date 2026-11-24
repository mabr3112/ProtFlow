"""Exercise both Boltz output layouts without loading a prediction model."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from protflow.jobstarters import LocalJobStarter
from protflow.poses import Poses
from protflow.tools import Boltz, Boltz2Anthropic
from protflow.tools.boltz import collect_boltz2_anthropic_scores


@pytest.fixture
def runner(tmp_path, monkeypatch):
    launcher = tmp_path / "test kit"
    launcher.write_text("""#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

assert os.environ["KIT_TEST"] == "ready"
parser = argparse.ArgumentParser()
parser.add_argument("verb", choices=["pred"])
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--out_dir", required=True, type=Path)
parser.add_argument("--mode", default="fast")
parser.add_argument("--seeds", default="0")
parser.add_argument("--diffusion_samples", type=int, default=1)
parser.add_argument("--override", action="store_true")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=False)
inputs = args.input.glob("*.yaml") if args.input.is_dir() else [args.input]
for source in inputs:
    name = source.stem
    for seed in args.seeds.split(","):
        if args.mode == "off":
            dest = args.out_dir / f"boltz_results_{args.input.stem}" / "predictions" / name
            suffix = "pdb"
        else:
            dest = args.out_dir / "by_seed" / name / f"s{seed}"
            suffix = "cif"
        dest.mkdir(parents=True)
        (dest / f"affinity_{name}.json").write_text('{"affinity_pred_value": 2.0}')
        for rank in range(args.diffusion_samples):
            model = f"{name}_model_{rank}"
            (dest / f"{model}.{suffix}").write_text(f"{name} {seed} {rank}")
            (dest / f"confidence_{model}.json").write_text('{"ptm": 0.75}')
            (dest / f"plddt_{model}.npz").touch()
""")
    launcher.chmod(0o755)
    config = SimpleNamespace(BOLTZ2_ANTHROPIC_PATH=str(launcher),
                             BOLTZ2_ANTHROPIC_PRE_CMD="export KIT_TEST=ready")
    monkeypatch.setattr("protflow.tools.boltz.require_config", lambda: config)
    return Boltz2Anthropic(jobstarter=LocalJobStarter(max_cores=2))


def test_batched_multiseed_run_cache_and_overwrite(tmp_path, runner):
    paths = []
    for name in ("first_pose", "second", "third"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("sequences: []")
        paths.append(str(path))
    work_dir = str(tmp_path / "work dir")
    result = runner.run(Poses(paths, work_dir=work_dir), "fold", options="--seeds 2,5 --diffusion_samples 2")

    assert len(result) == 12
    assert result.df["poses_description"].is_unique
    assert set(result.df["fold_seed"]) == {2, 5}
    assert set(result.df["fold_model_rank"]) == {0, 1}
    for row in result.df.itertuples():
        name = Path(row.input_poses).stem
        location = Path(row.poses)
        assert location.stem == row.poses_description == f"{name}_s{row.fold_seed}_{row.fold_model_rank}"
        assert location.read_text() == f"{name} {row.fold_seed} {row.fold_model_rank}"
        assert Path(row.fold_plddt_location).is_file()
        assert row.fold_ptm == 0.75 and row.fold_affinity_pred_value == 2.0

    output, = (Path(work_dir) / "fold").glob("prediction_*/outputs")
    assert len(list(output.glob("job_*"))) == 2
    assert len(collect_boltz2_anthropic_scores(output)) == 12
    cached = runner.run(Poses(paths, work_dir=work_dir), "fold")
    assert cached.poses_list() == result.poses_list()
    assert list((Path(work_dir) / "fold").glob("prediction_*/outputs")) == [output]

    poses = Poses(paths[:2], work_dir=work_dir)
    poses.df["kit_options"] = ["--mode off", "--mode fast --seeds 9"]
    rerun = runner.run(poses, "fold", pose_options="kit_options", overwrite=True)
    assert rerun.df["poses_description"].tolist() == ["first_pose_model_0", "second_s9_0"]
    assert Path(rerun.poses_list()[0]).suffix == ".pdb"
    assert all(Path(path).is_file() for path in cached.poses_list())


def test_stock_runner_still_collects_stock_layout(tmp_path):
    dest = tmp_path / "boltz_results_batch" / "predictions" / "first_pose"
    dest.mkdir(parents=True)
    model = dest / "first_pose_model_0.pdb"
    model.touch()
    (dest / "confidence_first_pose_model_0.json").write_text(json.dumps({"ptm": 0.9}))
    row, = Boltz._collect_scores(str(tmp_path)).to_dict("records")
    assert row["description"] == model.stem
    assert row["location"] == str(model)
    assert row["ptm"] == 0.9
    assert row["pae_location"] == str(dest / "pae_first_pose_model_0.npz")
