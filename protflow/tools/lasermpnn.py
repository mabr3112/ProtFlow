"""ProtFlow runner for PottsMPNN.

This module integrates the command-line PottsMPNN YAML workflows into
ProtFlow. It supports the two upstream scripts that expose a ``--config``
interface:

- ``sample_seqs.py`` for sequence design from backbone structures.
- ``energy_prediction.py`` for mutation-energy and deep-mutational-scan
  scoring.

The runner writes script-specific YAML files, dispatches one command per
generated config through a :class:`~protflow.jobstarters.JobStarter`, and
collects PottsMPNN FASTA or CSV outputs back into a
:class:`~protflow.poses.Poses` dataframe.

Configuration
-------------
The runner reads default executable paths from the ProtFlow config:

``POTTSMPNN_DIR``
    Path to the local PottsMPNN checkout. Commands are executed from this
    directory so relative checkpoint paths from upstream YAML examples work.
``POTTSMPNN_PYTHON``
    Python interpreter from the PottsMPNN environment.
``POTTSMPNN_PRE_CMD``
    Optional shell prefix used to activate modules or environments before each
    command.

Parameter Objects
-----------------
Use :class:`SampleSequencePottsMPNNParams` with ``sample_seqs.py`` and
:class:`EnergyPredictionPottsMPNNParams` with ``energy_prediction.py``. These
typed dataclasses expose PottsMPNN model and inference fields directly, so IDEs
can autocomplete nested attributes such as ``params.model.check_path`` and
``params.inference.num_samples``.

Pose-specific Values
--------------------
Wrap a dataframe column name in :class:`PoseCol` to fill a parameter from
``Poses.df``. Parameters ending in ``*_custom`` are converted into temporary
JSON files and can still be batched. Other pose-specific parameters require one
config per input pose.

Examples
--------
Design two sequences per backbone:

>>> from protflow.poses import Poses
>>> from protflow.tools import PottsMPNN, SampleSequencePottsMPNNParams
>>> poses = Poses(poses=["backbone_a.pdb", "backbone_b.pdb"], work_dir="work")
>>> params = SampleSequencePottsMPNNParams()
>>> params.inference.num_samples = 2
>>> params.inference.temperature = 0.1
>>> params.inference.optimization_mode = "none"
>>> poses = PottsMPNN().run(poses=poses, prefix="potts_design", params=params)

Score mutations from a CSV file:

>>> from protflow.tools import EnergyPredictionPottsMPNNParams
>>> params = EnergyPredictionPottsMPNNParams(mutant_csv="mutations.csv")
>>> poses = PottsMPNN().run(
...     poses=poses,
...     prefix="potts_energy",
...     script="energy_prediction",
...     params=params,
... )
"""

from __future__ import annotations

import logging
import os
import shutil
from glob import glob
from pathlib import Path
import pandas as pd

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter, split_list
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.runners import (
    Runner,
    RunnerOutput,
    options_flags_to_string,
    parse_generic_options,
    prepend_cmd,
)

from protflow.utils.biopython_tools import biopython_load_structure, save_structure_to_file, select_biopython_residues
from protflow.residues import ResidueSelection

class LASErMPNN(Runner):
    """Run PottsMPNN command-line scripts from ProtFlow.

    Parameters
    ----------
    python_path : str, optional
        Python interpreter used to execute PottsMPNN. If omitted, the value is
        loaded from ``POTTSMPNN_PYTHON`` in the ProtFlow config.
    pottsmpnn_dir : str, optional
        Path to the PottsMPNN checkout. If omitted, the value is loaded from
        ``POTTSMPNN_DIR``.
    pre_cmd : str, optional
        Shell prefix prepended to every command, commonly used to activate a
        conda environment or cluster module. Defaults to ``POTTSMPNN_PRE_CMD``.
    jobstarter : JobStarter, optional
        Default jobstarter used when :meth:`run` is called without one.

    Attributes
    ----------
    name : str
        Runner name used for job names and cached score files.
    index_layers : int
        Default merge index depth. The active value is selected per script in
        :meth:`run` because ``sample_seqs.py`` appends sample indices while
        ``energy_prediction.py`` keeps one row per input pose.
    pottsmpnn_dir : str
        Resolved PottsMPNN checkout path.
    python_path : str
        Resolved PottsMPNN Python interpreter.
    pre_cmd : str
        Resolved shell prefix.

    Notes
    -----
    Only upstream scripts with a ``--config`` YAML interface are supported.
    The runner currently supports ``sample_seqs.py`` and
    ``energy_prediction.py``.
    """

    def __init__(
        self,
        python_path: str | None = None,
        script_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize the runner and resolve PottsMPNN configuration.

        Parameters
        ----------
        python_path : str, optional
            Python interpreter used to run PottsMPNN.
        script_path : str, optional
            Path to LASErMPNN batch inference script.
        pre_cmd : str, optional
            Optional shell prefix for environment activation.
        jobstarter : JobStarter, optional
            Default jobstarter for this runner instance.
        """
        # config required
        config = require_config()

        # setup config paths
        self.script_path = str(script_path or load_config_path(config, "LASERMPNN_SCRIPT_PATH"))
        self.python_path = str(python_path or load_config_path(config, "LASERMPNN_PYTHON_PATH"))
        self.pre_cmd = pre_cmd or load_config_path(config, "LASERMPNN_PRE_CMD", is_pre_cmd=True)

        # setup runner state
        self.jobstarter = jobstarter
        self.name = "lasermpnn"
        self.index_layers = 1

    def __str__(self) -> str:
        """Return the short runner name.

        Returns
        -------
        str
            The literal runner name ``"pottsmpnn"``.
        """
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        options: str | None = None,
        nstruct: int = 1,
        fixed_residues: str | ResidueSelection = None,
        omit_AAs: list | str = None,
        overwrite: bool = False,
    ) -> Poses:
        """Run PottsMPNN and merge collected results into ``poses``.

        Parameters
        ----------
        poses : Poses
            Input structures to pass to PottsMPNN. The ``poses`` column must
            contain PDB paths and ``poses_description`` is used as the upstream
            PottsMPNN structure identifier.
        prefix : str
            Unique run prefix used to create the runner work directory and
            prefixed output score columns.
        jobstarter : JobStarter, optional
            Jobstarter for this call. If omitted, the runner falls back to the
            instance jobstarter and then ``poses.default_jobstarter``.
        script : str, optional
            Script alias or path. Supported aliases are ``"sample_seqs"`` and
            ``"energy_prediction"``.
        params : SampleSequencePottsMPNNParams or EnergyPredictionPottsMPNNParams, optional
            Typed parameter object used to generate YAML configs. If omitted,
            defaults are created for the selected script.
        options : str, optional
            Extra command-line options passed to the upstream script. ``--config``
            is ignored because config files are managed by the runner.
        pose_options : str or list of str, optional
            Unsupported for PottsMPNN. Use :class:`PoseCol` fields in ``params``
            for pose-specific settings.
        include_scores : list of str, optional
            Reserved for API consistency with other runners. PottsMPNN collectors
            currently load the standard output fields.
        overwrite : bool, optional
            If ``True``, remove previous runner-owned outputs and rerun jobs.

        Returns
        -------
        Poses
            The input ``Poses`` object with PottsMPNN score columns merged in.

        Raises
        ------
        ValueError
            If ``pose_options`` are supplied or the params object does not match
            the selected script.
        NotImplementedError
            If ``script`` is not one of the supported config-based scripts.
        RuntimeError
            If PottsMPNN runs but no score rows can be collected.
        """
        # setup run directory and jobstarter
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # scorefile reuse shortcut
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            outputs = RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers
            )
            return outputs.return_poses()

        # cleanup previous outputs
        if overwrite:
            self._cleanup_previous_outputs(work_dir)

        n_jobs = min([len(poses.poses_list()), jobstarter.max_cores])

        # setup for batch mode
        batch_dirs = self._setup_batch_mode(poses=poses, num_batches=n_jobs, work_dir=work_dir, fixed_residues=fixed_residues)

        os.makedirs(out_dir := os.path.join(work_dir, "output"), exist_ok=True)
        # build commands
        cmds = self._write_cmds(
            options=options,
            input_dirs=batch_dirs,
            output_dir=out_dir,
            nstruct=nstruct,
            fixed_residues=fixed_residues,
            omit_AAs=omit_AAs,
        )

        # prepend configured environment command
        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # execute jobs
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir
        )

        # collect and validate scores
        scores = collect_scores(work_dir=work_dir)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output directory: {work_dir}")

        # save scores and merge back into poses
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        outputs = RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers
        )
        return outputs.return_poses()
    
    def _setup_batch_mode(self, poses:Poses, num_batches:int, work_dir:str, fixed_residues:str|ResidueSelection=None):
        # parse fixed residues
        if fixed_residues:
            if isinstance(fixed_residues, str):
                col_in_df(poses.df, fixed_residues)
                fixed_residues = poses.df[fixed_residues].to_list()
            elif isinstance(fixed_residues, ResidueSelection):
                fixed_residues = [fixed_residues for _ in poses]
            else:
                raise KeyError(f"<fixed_residues> must be the name of a poses.df column containing a ResidueSelection or a single ResidueSelection, not {type(fixed_residues)}!")
        else:
            fixed_residues = [None for _ in poses]

        # split poses and fixed residues into batches
        poses_sublists = split_list(input_list=poses.poses_list(), n_sublists=num_batches)
        fixed_res_sublists = split_list(input_list=fixed_residues, n_sublists=num_batches)

        #
        batch_dirs = []
        for i, (pose_batch, fixed_res_batch) in enumerate(zip(poses_sublists, fixed_res_sublists)):
            os.makedirs(batch_dir := os.path.join(work_dir, f"batch_{i}"), exist_ok=True)
            for pose, fixed_res in zip(pose_batch, fixed_res_batch):
                _update_pose_bfactors(pose, batch_dir, fixed_res)
            batch_dirs.append(os.path.abspath(batch_dir))

        return batch_dirs

    def _write_cmds(self, options: str, input_dirs: list, output_dir: str, nstruct:int, fixed_residues:str | ResidueSelection = None, omit_AAs: str | list = None) -> str:
        """Format the shell command for a single PottsMPNN config.

        Parameters
        ----------
        script : str
            Absolute path to the upstream PottsMPNN script.
        config_path : str
            YAML config passed as ``--config``.
        cli_args : str, optional
            Additional parsed command-line arguments.

        Returns
        -------
        str
            Command that runs from the PottsMPNN checkout.
        """

        options, flags = parse_generic_options(options=options, pose_options=None, sep="--")
        if fixed_residues and "fix_beta" not in flags:
            flags.append("fix_beta")
        """
        if "output_fasta" not in flags:
            flags.append("output_fasta")
        """
        if omit_AAs:
            if isinstance(omit_AAs, str) and "," in omit_AAs:
                omit_AAs = omit_AAs.split(",")
            omit_AAs = ",".join(omit_AAs)
            options["disabled_residues"] = omit_AAs
        
        forbidden_opts = set(["output_fasta_only", "designs_per_input", "input_pdb_directory", "output_pdb_directory"])
        
        if any(forbidden_opt in options for forbidden_opt in forbidden_opts) or any(forbidden_opt in flags for forbidden_opt in forbidden_opts):
            raise KeyError(f"Do not set any of these flags as options: {forbidden_opts}!")
        
        opts_flags = options_flags_to_string(options, flags)

        cmds = [
            f"{self.python_path} {self.script_path} {batch_dir} {os.path.join(output_dir, os.path.basename(batch_dir))} {nstruct} {opts_flags}" for batch_dir in input_dirs
        ]
        
        return cmds
    
    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Remove previous runner-owned outputs inside the work directory.

        Parameters
        ----------
        work_dir : str
            Runner work directory created for this prefix.
        """
        # remove only files/directories inside runner work_dir
        if not os.path.isdir(work_dir):
            return
        for path in glob(os.path.join(work_dir, "*")):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)



def _update_pose_bfactors(input_path:str, output_dir:str, fixed_residues: ResidueSelection=None) -> str:
    """LASErMPNN uses bfactor columns to define fixed residues. Bfactors of designable residues are set to 0, fixed positions to 1."""
    pose = biopython_load_structure(input_path)
    for atom in pose.get_atoms():
        atom.bfactor = 0
    if fixed_residues:
        fixed = select_biopython_residues(pose, fixed_residues)
        for res in fixed:
            for atom in res.get_atoms():
                atom.bfactor = 1
    out_path = os.path.join(output_dir, os.path.basename(input_path))
    save_structure_to_file(pose, out_path)
    return out_path

def collect_scores(work_dir: str):
    output_dir = os.path.join(work_dir, "output")
    scores = pd.DataFrame({"original_paths": glob(os.path.join(output_dir, "batch_*", "*", "*.pdb"))})

    # get the input filename
    scores["input_description"] = scores["original_paths"].apply(lambda x: Path(x).parent.name)

    # rename files
    os.makedirs(renamed_dir := os.path.join(work_dir, "renamed_poses"), exist_ok=True)
    renamed_poses = []
    for in_file, df in scores.groupby("input_description", sort=False):
        for i, pose in enumerate(df["original_paths"]):
            shutil.copy(pose, new_path := os.path.join(renamed_dir, f"{in_file}_{str(i).zfill(4)}.pdb"))
            renamed_poses.append(new_path)

    scores["location"] = [os.path.abspath(pose) for pose in renamed_poses]
    scores["description"] = [description_from_path(pose) for pose in scores["location"]]

    # drop temp cols
    scores.drop(["input_description", "original_paths"], axis=1, inplace=True)

    return scores


