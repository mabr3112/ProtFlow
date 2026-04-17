"""Runner template for ProtFlow tool integrations.

How to use this template
------------------------
1. Copy this file (or class) to a new module in ``protflow/tools`` or ``protflow/metrics``.
2. Rename ``ExampleRunner`` and ``example_runner`` to your tool name.
3. Replace all ``TODO`` markers.
4. Keep the run lifecycle intact:
   setup workdir -> reuse cached outputs -> prep options -> build commands -> run jobs -> collect scores -> return RunnerOutput.
5. Implement ``collect_scores(...)`` as a **module function**, not a class method.

Design goals
------------
- Keep runner behavior consistent across ProtFlow.
- Make it obvious where tool-specific logic belongs.
- Avoid re-implementing common logic already provided by ``Runner``.
- Keep score parsing callable without constructing a runner instance.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob

import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
import torch
from pathlib import Path

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.runners import (
    Runner,
    RunnerOutput,
    parse_generic_options,
    options_flags_to_string,
    prepend_cmd,
)
from ..utils.openbabel_tools import split_complex

class SigmaDock(Runner):
    """Template class for implementing a new ProtFlow runner.

    Developer instructions
    ----------------------
    - Keep this class focused on one external tool.
    - Put all user-facing run parameters on ``run(...)``.
    - Use config values as defaults in ``__init__``.
    - Keep score parsing in the module-level ``collect_scores(...)`` function.
    - Ensure output parsing returns a dataframe with:
      - ``description``: basename without extension of each produced pose
      - ``location``: absolute path to produced pose file
    - Always return ``RunnerOutput(...).return_poses()``.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        ckpt_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize tool paths and static runner metadata."""

        config = require_config()

        self.application_path = application_path or load_config_path(config, "SIGMADOCK_SCRIPT_PATH")
        self.python_path = python_path or load_config_path(config, "SIGMADOCK_PYTHON_PATH")
        self.ckpt_path = ckpt_path or load_config_path(config, "SIGMADOCK_CKPT_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "SIGMADOCK_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "sigmadock_runner"
        self.index_layers = 0


    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        sample_conformer: bool = True,
        fragmentation_strategy: str = "canonical",
        options: str | None = None,
        pose_options: list[str] | str | None = None,
        include_scores: list[str] | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Execute the full runner lifecycle and merge results into ``poses``."""
        # 1) Generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # 2) Scorefile reuse shortcut.
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)
            return RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # 3a) Split complexes into receptor PDB + ligand SDF; paths stored in poses.df.
        self._prepare_sigmadock_inputs(poses=poses, work_dir=work_dir, overwrite=overwrite)

        # 3b) Build runner defaults from explicit run() parameters.
        runner_defaults = (
            f"ckpt={self.ckpt_path}"
            f"hardware.devices=1 "
            f"graph.sample_conformer={str(sample_conformer).lower()} "
            f"graph.fragmentation_strategy={fragmentation_strategy}"
        )

        # 3c) Merge: defaults < user options < per-pose options < receptor/ligand paths.
        base_opts = options_flags_to_string(*parse_generic_options(runner_defaults, options, sep=" "), sep=" ", no_quotes=True)
        sigmadock_opts = poses.df["sigmadock_inputs"].tolist()
        user_opts = self.prep_pose_options(poses=poses, pose_options=pose_options)
        pose_options_list = [
            options_flags_to_string(*parse_generic_options(base_opts, usr_opt, sep=" "), sep=" ", no_quotes=True)
            for usr_opt in user_opts
        ]
        pose_options_list = [
            options_flags_to_string(*parse_generic_options(po, sd_opt, sep=" "), sep=" ", no_quotes=True)
            for po, sd_opt in zip(pose_options_list, sigmadock_opts)
        ]

        # 4) Build commands.
        cmds = self._build_commands(
            poses=poses,
            work_dir=work_dir,
            pose_options=pose_options_list,
        )

        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # 5) Execute commands.
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # 6) Collect and validate scores (module function, by convention).
        scores = collect_scores(work_dir=work_dir, include_scores=include_scores)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        # 7) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    def _build_commands(
        self,
        poses: Poses,
        work_dir: str,
        pose_options: list[str | None],
    ) -> list[str]:
        """Create one shell command per pose."""
        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for pose_path, cli_args in zip(poses.poses_list(), pose_options):
            cmds.append(self.write_cmd(pose_path=pose_path, out_dir=out_dir, cli_args=cli_args))
        return cmds

    def write_cmd(self, pose_path: str, out_dir: str, cli_args: str) -> str:
        """Return the exact shell command for one pose.

        Developer instructions
        ----------------------
        - Build an executable command string only; do not execute here.
        - Ensure output filename preserves or predictably derives from pose description.
        - Keep quoting robust for paths with spaces.
        """

        description = os.path.splitext(os.path.basename(pose_path))[0]
        out_pose = os.path.join(out_dir, description)

        # TODO: adapt cli_args already contains inference.protein_pdb and inference.ligand_sdf from _prepare_sigmadock_inputs
        return (
            f"{self.python_path} {self.application_path} "
            f"output_dir={out_pose}"
            f"{cli_args}"
        )

    def _prepare_sigmadock_inputs(self, poses: Poses, work_dir: str, overwrite: bool = False) -> None:
        """Split each pose complex into a receptor PDB and ligand SDF for docking.

        Writes per-pose CLI strings into poses.df["sigmadock_inputs"] so they flow
        through the standard pose_options mechanism into _build_commands.
        """
        inputs_dir = os.path.join(work_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)

        input_opts = []
        for pose_path in poses.poses_list():
            stem = os.path.splitext(os.path.basename(pose_path))[0]
            out_pdb = os.path.join(inputs_dir, f"{stem}.pdb") # these are for naming the files and to kn ow where to save them
            out_sdf = os.path.join(inputs_dir, f"{stem}_ligand.sdf")
            if overwrite or not os.path.isfile(out_pdb): #if the file is already there, this is False, skips spliting again, but only if overwrite is also true
                split_complex(pose_path, work_dir=inputs_dir)
            input_opts.append(
                f"inference.protein_pdb={out_pdb} inference.ligand_sdf={out_sdf}"
            )

        poses.df["sigmadock_inputs"] = input_opts # do not think this is a ncie way to handle the inputs, need to check in pose object. 

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete/clear runner-specific output artifacts before rerun.

        Developer instructions
        ----------------------
        - Keep cleanup scoped to this runner's own output directories.
        - Never remove unrelated files outside ``work_dir``.
        - This method is optional but useful for tools that append stale outputs.
        """
        output_dir = os.path.join(work_dir, "outputs")
        if not os.path.isdir(output_dir):
            return

        for file_path in glob(os.path.join(output_dir, "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)

def _is_heavy_value(value: object) -> bool:
    """Heuristic for values that can bloat score tables (2D/per-residue objects)."""
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple, dict)):
            return True
        if len(value) > 200:
            return True

    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return True

    return False


def _extract_score_dict(
    payload: dict,
    include_scores: set[str],
    prefix: str = "",
) -> dict[str, object]:
    """Flatten nested score dictionaries with optional inclusion of heavy values.

    Notes
    -----
    - Do not hardcode score names where possible; parse what is present.
    - By default, this returns scalar values and skips heavy values.
    - Heavy values are included only if their key (or flattened key path) is in
      ``include_scores``.
    """
    out: dict[str, object] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(_extract_score_dict(value, include_scores, prefix=flat_key))
            continue

        if _is_heavy_value(value):
            if key in include_scores or flat_key in include_scores:
                out[flat_key] = json.dumps(value)
            continue

        out[flat_key] = value

    return out

def _mol_with_coords(mol_ref, coords):

    if hasattr(coords, "float"):
        coords = coords.float().cpu().tolist()
    mol = Chem.RWMol(mol_ref)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def _write_complex_pdb(prot_path: str, mol, out_path) -> None:
    
    lig_lines = [ln for ln in Chem.MolToPDBBlock(mol, flavor=4).splitlines() if ln.startswith(("HETATM", "CONECT"))]
    with open(prot_path) as f:
        prot_lines = [ln.rstrip() for ln in f if not ln.startswith(("END", "CONECT"))]
    with open(out_path, "w") as f:
        f.write("\n".join(prot_lines) + "\n")
        f.write("\n".join(lig_lines) + "\n")
        f.write("END\n")


def collect_scores(work_dir: str, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Parse runner outputs and return the canonical scores dataframe.

    Developer instructions
    ----------------------
    - Keep this function at module scope, not inside the runner class.
    - Required output columns:
      - ``description``
      - ``location``
    - Favor score auto-discovery (read keys present in outputs) over hardcoded
      column lists, because external tools frequently rename score terms.
    - Avoid reading heavy per-residue / matrix-like data by default.
      Use ``include_scores`` to opt-in to specific heavy fields.
    - Keep function callable standalone (for debugging/re-parsing old runs).
    """


    output_root = Path(work_dir) / "outputs"
    rows: list[dict[str, object]] = []

    for pose_dir in sorted(output_root.iterdir()):
        if not pose_dir.is_dir():
            continue
        pose_stem = pose_dir.name

        pred_files = sorted(pose_dir.rglob("predictions.pt"))
        if not pred_files:
            logging.warning("No predictions.pt found under %s", pose_dir)
            continue
        seed_dir = pred_files[0].parent

        pred    = torch.load(seed_dir / "predictions.pt", weights_only=False)
        rescore = torch.load(seed_dir / "rescoring.pt",   weights_only=False) if (seed_dir / "rescoring.pt").exists()   else None
        pb      = torch.load(seed_dir / "posebusters.pt", weights_only=False) if (seed_dir / "posebusters.pt").exists() else None

        complex_dir = pose_dir / "complexes"
        complex_dir.mkdir(exist_ok=True)

        for code, samples in pred["results"].items():
            if not samples:
                continue
            s        = samples[0]
            mol_ref  = s.get("lig_ref")
            coords   = s.get("x0_hat")
            pdb_path = s.get("pdb_path")

            if mol_ref is None or coords is None:
                logging.warning("Missing lig_ref or x0_hat for %s — skipping", pose_stem)
                continue

            out_complex = complex_dir / f"{pose_stem}_complex.pdb"
            mol = _mol_with_coords(mol_ref, coords)
            if pdb_path and Path(pdb_path).exists():
                _write_complex_pdb(pdb_path, mol, out_complex)
            else:
                Chem.MolToPDBFile(mol, str(out_complex))

            row: dict[str, object] = {
                "description": pose_stem,
                "location":    str(out_complex.resolve()),
            }

            if rescore is not None:
                pose_scores = rescore["scores"].get(code)
                if pose_scores:
                    sc = pose_scores[0]
                    row["affinity"]              = sc.get("Affinity")
                    row["intramolecular_energy"] = sc.get("Intramolecular energy")
                    row["cnn_score"]             = sc.get("CNNscore")
                    row["cnn_affinity"]          = sc.get("CNNaffinity")
                    row["cnn_variance"]          = sc.get("CNNvariance")

            if pb is not None:
                row["rmsd"]         = pb["rmsds"].get(code)
                row["pb_pass_rate"] = pb["pb_checks"].get(code)
                pb_dict = pb["pb_dicts"].get(code)
                if pb_dict:
                    for k, v in pb_dict.items():
                        row[f"pb_{k}"] = v

            rows.append(row)

    return pd.DataFrame(rows)


# Optional: lightweight checklist for developers implementing a new runner.
IMPLEMENTATION_CHECKLIST: tuple[str, ...] = (
    "Set config variable names in __init__.",
    "Set correct index_layers for your output naming.",
    "Implement write_cmd with real CLI syntax.",
    "Implement module-level collect_scores (not a class method) with description/location columns.",
    "Make collect_scores auto-discover score keys from tool outputs where possible.",
    "Skip heavy per-residue/matrix outputs by default; gate them behind include_scores list.",
    "Ensure scorefile reuse works when overwrite=False.",
    "Confirm RunnerOutput merge updates poses as expected.",
    "Export runner in protflow/tools/__init__.py (submodule import + class import).",
    "Document API and add tool page in docs/source/tools/<tool>.rst (and tools/index.rst toctree if new).",
    "Build docs warning-free: sphinx-build -b html -W docs/source docs/_build/html",
    "Add/extend unit tests for parsing and option handling.",
)
