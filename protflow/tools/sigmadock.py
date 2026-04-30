"""ProtFlow runner for SigmaDock molecular docking.

This module provides the :class:`SigmaDock` runner that:
(1) splits each input complex into a receptor PDB and a reference ligand SDF,
(2) assembles SigmaDock command lines (redocking or crossdocking),
(3) dispatches inference via a :class:`JobStarter`, and
(4) aggregates docking scores (affinity, RMSD, PoseBusters checks) into a
single score table for downstream orchestration.

The typical workflow is:

1. Ensure paths and environment hooks for SigmaDock are configured
   (see Notes on ``SIGMADOCK_SCRIPT_PATH``, ``SIGMADOCK_PYTHON_PATH``,
   ``SIGMADOCK_CKPT_PATH``, ``SIGMADOCK_PRE_CMD``).
2. Provide inputs as a :class:`Poses` collection of PDB or CIF complexes
   that contain both a protein receptor and a bound ligand (HETATM).
   **A ligand must always be present in the input pose**, even for
   crossdocking — the bound ligand is used as the binding-site reference
   (``REF_SDF``) from which SigmaDock infers the pocket location.
3. Call :meth:`SigmaDock.run` with ``ligand_name`` matching the residue name
   of the ligand in the input complex. For crossdocking, pass absolute paths
   to query ligand SDFs via ``query_ligands``.
4. Consume the returned :class:`Poses` object whose ``.df`` is augmented with
   docking scores and the path to the docked complex PDB.

Notes
-----
- Configuration keys
  The runner reads its defaults from ProtFlow's config via:
  ``SIGMADOCK_SCRIPT_PATH`` (path to SigmaDock's ``sample.py``),
  ``SIGMADOCK_PYTHON_PATH`` (interpreter in the SigmaDock environment),
  ``SIGMADOCK_CKPT_PATH`` (model checkpoint directory), and
  ``SIGMADOCK_PRE_CMD`` (optional shell prefix such as conda activation).
  Use ``protflow.config`` utilities to set these once per environment.
- Redocking vs. crossdocking
  When ``query_ligands=None`` (default), the ligand is extracted from the
  input complex and re-docked (redocking). Passing a list of SDF paths to
  ``query_ligands`` switches to crossdocking mode, where each query ligand
  is docked into the receptor extracted from the corresponding input pose.
  Query ligand paths must be absolute or resolvable from the working directory
  at call time — paths inside the runner's ``work_dir`` will be deleted on
  ``overwrite=True`` before they can be read.
- Score collection
  Score parsing is delegated to ``sigmadock_collect_scores.py``, which runs
  in the SigmaDock Python environment where PyTorch and RDKit are available.
  Scores include ``affinity``, ``intramolecular_energy``, ``rmsd`` (if a
  reference is available), and PoseBusters pass rate.
  
Authors
-------
Johannes Peterlechner
 - claude was partly used as a coding assistant
Version
-------
0.1.0

Examples
--------
Redock the native ligand from a set of predicted complexes:

>>> from protflow.poses import Poses
>>> from protflow.tools.sigmadock import SigmaDock
>>> poses = Poses(
...     poses=["complex_A.cif", "complex_B.cif"],
...     work_dir="work/sigmadock_demo"
... )
>>> runner = SigmaDock()  # uses config defaults
>>> poses = runner.run(
...     poses=poses,
...     prefix="redock",
...     ligand_name="LIG",
...     overwrite=False,
... )
>>> poses.df[["redock_affinity", "redock_rmsd", "redock_pb_pass_rate", "intramolecular_energy"...]]

Crossdock a panel of query ligands into receptors from the same complexes:

>>> poses = runner.run(
...     poses=poses,
...     prefix="crossdock",
...     ligand_name="LIG",
...     query_ligands=["/data/ligands/compound_1.sdf", "/data/ligands/compound_2.sdf"],
...     overwrite=True,
... )
"""

from __future__ import annotations

import logging
import os
import shutil

import pandas as pd

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
from ..utils.biopython_tools import split_complex

class SigmaDock(Runner):
    """ProtFlow runner for SigmaDock molecular docking.

    Splits each input complex into a receptor PDB and ligand SDF, invokes
    SigmaDock via the configured Python environment, and merges docking scores
    (affinity, RMSD, PoseBusters checks) back into the poses dataframe.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        ckpt_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize SigmaDock runner.

        All path arguments fall back to ProtFlow config values when omitted.
        ``pre_cmd`` is typically a conda-activation snippet required for
        SigmaDock's subprocess to find its dependencies.
        """
        config = require_config()

        self.application_path = application_path or load_config_path(config, "SIGMADOCK_SCRIPT_PATH")
        self.python_path = python_path or load_config_path(config, "SIGMADOCK_PYTHON_PATH")
        self.ckpt_path = ckpt_path or load_config_path(config, "SIGMADOCK_CKPT_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "SIGMADOCK_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "sigmadock_runner"
        self.index_layers = 1

    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        ligand_name: str = "LIG",
        num_seeds: int = 1,
        seed: int = 0,
        sample_conformer: bool = True,
        fragmentation_strategy: str = "canonical",
        query_ligands: list[str] | None = None,
        receptor_col: str | None = None,
        ligand_col: str | None = None,
        ref_ligand_col: str | None = None,
        options: str | None = None,
        pose_options: list[str] | str | None = None,
        include_scores: list[str] | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Run SigmaDock docking and return poses augmented with docking scores.

        Supports four modes depending on how inputs are supplied:

        - **Redocking from complex** (default): each pose is a protein–ligand
          complex; the native ligand is extracted and re-docked.
        - **Crossdocking from complex**: each pose is a complex; ``query_ligands``
          lists external SDF paths to dock into the extracted receptor.
        - **Redocking, pre-extracted**: ``ligand_col`` points to a column of
          ligand SDF paths; no split_complex call is made.
        - **Crossdocking, pre-extracted**: ``ligand_col`` holds lists of query
          SDF paths; ``ref_ligand_col`` holds the pocket-anchor SDF paths.

        Parameters
        ----------
        ligand_name:
            Residue name of the ligand in the input complex (used by
            split_complex; ignored when ``ligand_col`` is set).
        num_seeds:
            Number of independent stochastic draws per pose (``num_seeds``
            in SigmaDock's Hydra config).  Each seed produces one docked
            pose, written as ``{description}_{i:04d}.pdb``.
        seed:
            Master random seed (``seed`` in SigmaDock's Hydra config).
            SigmaDock derives ``num_seeds`` per-draw seeds from this value,
            so the same ``seed`` + ``num_seeds`` combination always reproduces
            the same set of poses.  Stored as ``{prefix}_master_seed``.
        query_ligands:
            Absolute SDF paths for crossdocking from complex.  One list is
            shared across all poses.
        receptor_col:
            Column name for pre-extracted receptor PDBs (falls back to
            ``poses`` column when omitted).
        ligand_col:
            Column name for pre-extracted ligand SDFs (str) or query ligand
            lists (list[str]).  Setting this skips split_complex entirely.
        ref_ligand_col:
            Column name for the pocket-anchor SDF used in pre-extracted
            crossdocking.
        include_scores:
            Score keys whose values are heavy tensors but should still be
            serialised and included in the output dataframe.
        """

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
        self._prepare_sigmadock_inputs(
            poses=poses, work_dir=work_dir, ligand_name=ligand_name,
            query_ligands=query_ligands, receptor_col=receptor_col,
            ligand_col=ligand_col, ref_ligand_col=ref_ligand_col, overwrite=overwrite,
        )

        # 3b) Build runner defaults from explicit run() parameters.
        runner_defaults = (
            f"ckpt={self.ckpt_path} "
            f"hardware.devices=1 "
            f"seed={seed} "
            f"num_seeds={num_seeds} "
            f"graph.sample_conformer={str(sample_conformer).lower()} "
            f"graph.fragmentation_strategy={fragmentation_strategy}"
        )

        # 3c) Merge: defaults < user options < per-pose options < receptor/ligand paths.
        base_opts = options_flags_to_string(*parse_generic_options(runner_defaults, options, sep=" "), sep=" ", no_quotes=True)
        sigmadock_opts = self._build_sigmadock_cli_opts(poses=poses, work_dir=work_dir)
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
        scores = collect_scores(work_dir=work_dir, python_path=self.python_path, include_scores=include_scores)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")
        scores["master_seed"] = seed
        if len(scores.index) < len(poses) * num_seeds:
            logging.warning("%s: expected %d rows (%d poses × %d seeds), got %d — some runs may have crashed.", self, len(poses) * num_seeds, len(poses), num_seeds, len(scores.index))

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
        """Return one shell command per pose, creating ``outputs/`` if needed."""
        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for pose_path, cli_args in zip(poses.poses_list(), pose_options):
            cmds.append(self.write_cmd(pose_path=pose_path, out_dir=out_dir, cli_args=cli_args))
        return cmds

    def write_cmd(self, pose_path: str, out_dir: str, cli_args: str) -> str:
        """Return the SigmaDock shell command for a single pose.

        The output directory is ``<out_dir>/<pose_stem>`` so each pose gets
        its own sub-directory, matching the layout expected by
        ``sigmadock_collect_scores.py``.
        """
        description = os.path.splitext(os.path.basename(pose_path))[0]
        out_pose = os.path.join(out_dir, description)

        return (
            f"{self.python_path} {self.application_path} "
            f"output_dir={out_pose} "
            f"{cli_args}"
        )

    def _prepare_sigmadock_inputs(self, poses: Poses, work_dir: str, ligand_name: str, query_ligands: list[str] | None = None, receptor_col: str | None = None, ligand_col: str | None = None, ref_ligand_col: str | None = None, overwrite: bool = False) -> None:
        """Populate ``sigmadock_pdb``, ``sigmadock_ligands``, and ``sigmadock_ref_ligand`` columns.

        Two paths through this function:

        **Pre-extracted** (``ligand_col`` is set): user-provided receptor PDBs
        and ligand SDFs are copied into ``inputs/`` so that ``overwrite``
        cleanup never touches the originals.

        **From complex** (default): ``split_complex`` is called on each pose to
        extract the protein (ATOM records) and ligand (HETATM by residue name)
        into ``inputs/``.  The split is skipped when the receptor PDB already
        exists and ``overwrite=False``.  For crossdocking, ``query_ligands``
        replaces the extracted ligand while the extracted SDF is kept as the
        pocket-anchor reference.
        """
        if ligand_col is not None:
            # --- Pre-extracted mode: copy all user files into inputs/ ---
            inputs_dir = os.path.join(work_dir, "inputs")
            os.makedirs(inputs_dir, exist_ok=True)
            pdb_col = receptor_col or "poses"

            def _cp(src, dst):
                shutil.copy2(src, dst)
                return dst

            def _local_stem(src):
                return os.path.splitext(os.path.basename(src))[0]

            def _copy_ligands(stem, ligs):
                # crossdocking: ligs is a list of query SDF paths
                if isinstance(ligs, list):
                    return [_cp(q, os.path.join(inputs_dir, f"{stem}_query_{i}.sdf")) for i, q in enumerate(ligs)]
                # redocking: ligs is a single SDF path
                return _cp(ligs, os.path.join(inputs_dir, f"{stem}_ligand.sdf"))

            rows = [(row[pdb_col], row[ligand_col], row.get(ref_ligand_col)) for _, row in poses.df.iterrows()]
            stems = [_local_stem(pdb) for pdb, _, _ in rows]
            poses.df["sigmadock_pdb"]        = [_cp(pdb, os.path.join(inputs_dir, f"{s}.pdb")) for (pdb, _, _), s in zip(rows, stems)]
            poses.df["sigmadock_ligands"]    = [_copy_ligands(s, ligs) for (_, ligs, _), s in zip(rows, stems)]
            poses.df["sigmadock_ref_ligand"] = [_cp(ref, os.path.join(inputs_dir, f"{s}_ref.sdf")) if ref else None for (_, _, ref), s in zip(rows, stems)]
            return

        # --- From-complex mode: extract receptor + ligand via split_complex ---
        inputs_dir = os.path.join(work_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)

        pdb_paths, ligand_paths, ref_paths = [], [], []
        for pose_path in poses.poses_list():
            stem = os.path.splitext(os.path.basename(pose_path))[0]
            out_pdb = os.path.join(inputs_dir, f"{stem}_{ligand_name}.pdb")
            out_sdf = os.path.join(inputs_dir, f"{stem}_{ligand_name}.sdf")
            if overwrite or not os.path.isfile(out_pdb):
                split_complex(pose_path, work_dir=inputs_dir, ligand_name=ligand_name)
            pdb_paths.append(out_pdb)
            # crossdocking: use external query ligands; keep extracted SDF as pocket anchor
            ligand_paths.append(query_ligands if query_ligands else out_sdf)
            ref_paths.append(out_sdf)

        poses.df["sigmadock_pdb"] = pdb_paths
        poses.df["sigmadock_ligands"] = ligand_paths
        poses.df["sigmadock_ref_ligand"] = ref_paths

    def _build_sigmadock_cli_opts(self, poses: Poses, work_dir: str) -> list[str]:
        """Build per-pose SigmaDock Hydra CLI option strings.

        Redocking (``sigmadock_ligands`` is a str): passes receptor and ligand
        directly via ``inference.protein_pdb`` and ``inference.ligand_sdf``.

        Crossdocking (``sigmadock_ligands`` is a list): writes a CSV with
        columns ``PDB``, ``SDF``, ``REF_SDF`` and passes it via
        ``inference.inference_datafront``.  One CSV row per query ligand.
        """
        inputs_dir = os.path.join(work_dir, "inputs")
        cli_opts = []
        for _, row in poses.df.iterrows():
            pdb = row["sigmadock_pdb"]
            ligands = row["sigmadock_ligands"]
            if isinstance(ligands, list):
                # Write per-pose inference CSV for crossdocking
                stem = os.path.splitext(os.path.basename(pdb))[0]
                ref_sdf = row["sigmadock_ref_ligand"]
                csv_path = os.path.join(inputs_dir, f"{stem}_inference.csv")
                rows = [{"PDB": pdb, "SDF": os.path.abspath(q), "REF_SDF": ref_sdf} for q in ligands]
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                cli_opts.append(f"inference.inference_datafront={csv_path}")
            else:
                cli_opts.append(f"inference.protein_pdb={pdb} inference.ligand_sdf={ligands}")
        return cli_opts

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete ``inputs/`` and ``outputs/`` from a previous run under ``work_dir``."""
        if os.path.isdir(inputs_dir := os.path.join(work_dir, "inputs")):
            shutil.rmtree(inputs_dir)
        if os.path.isdir(outputs_dir := os.path.join(work_dir, "outputs")):
            shutil.rmtree(outputs_dir)

def collect_scores(work_dir: str, python_path: str, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Parse SigmaDock outputs and return a scores dataframe.

    Runs ``sigmadock_collect_scores.py`` as a subprocess (it needs the
    SigmaDock Python environment with PyTorch and RDKit).  The script writes
    ``sigmadock_scores.json`` to ``work_dir``; this function reads it back and
    deletes it.  Can be called standalone to re-parse an old run directory.
    """
    import subprocess

    script = os.path.join(os.path.dirname(__file__), "runners_auxiliary_scripts", "sigmadock_collect_scores.py")
    cmd = [python_path, script, work_dir]
    if include_scores:
        cmd += ["--include-scores", ",".join(include_scores)]
    subprocess.run(cmd, check=True)

    scores_file = os.path.join(work_dir, "sigmadock_scores.json")
    scores = pd.read_json(scores_file, orient="records")
    os.remove(scores_file)
    return scores

