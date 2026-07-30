"""LASErMPNN runner for ProtFlow.

This module integrates LASErMPNN batch inference scripts into ProtFlow[cite: 1]. 
The runner dynamically creates input batches, assigns fixed residues by adjusting 
PDB B-factors, dispatches commands through a :class:`~protflow.jobstarters.JobStarter`, 
and collects the resulting structures back into a :class:`~protflow.poses.Poses` dataframe[cite: 1].

Configuration
-------------
The runner reads default executable paths from the ProtFlow config[cite: 1]:

``LASERMPNN_SCRIPT_PATH``
    Path to the LASErMPNN batch inference script[cite: 1].
``LASERMPNN_PYTHON_PATH``
    Python interpreter from the LASErMPNN environment[cite: 1].
``LASERMPNN_PRE_CMD``
    Optional shell prefix used to activate modules or environments before each
    command[cite: 1].

Examples
--------
Design sequences with fixed residues:

>>> from protflow.poses import Poses
>>> from protflow.tools import LASErMPNN
>>> poses = Poses(poses=["backbone_a.pdb", "backbone_b.pdb"], work_dir="work")
>>> runner = LASErMPNN()
>>> poses = runner.run(
...     poses=poses, 
...     prefix="laser_design", 
...     nstruct=2,
...     fixed_residues="fixed_res_column"
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
    """Run LASErMPNN command-line scripts from ProtFlow[cite: 1].

        Parameters
        ----------
        python_path : str, optional
            Python interpreter used to execute LASErMPNN[cite: 1]. If omitted, the value is
            loaded from ``LASERMPNN_PYTHON_PATH`` in the ProtFlow config[cite: 1].
        script_path : str, optional
            Path to the LASErMPNN batch inference script[cite: 1]. If omitted, the value is 
            loaded from ``LASERMPNN_SCRIPT_PATH``[cite: 1].
        pre_cmd : str, optional
            Shell prefix prepended to every command, commonly used to activate a
            conda environment or cluster module[cite: 1]. Defaults to ``LASERMPNN_PRE_CMD``[cite: 1].
        jobstarter : JobStarter, optional
            Default jobstarter used when :meth:`run` is called without one[cite: 1].

        Attributes
        ----------
        name : str
            Runner name used for job names and cached score files (set to ``"lasermpnn"``)[cite: 1].
        index_layers : int
            Default merge index depth, set to 1[cite: 1].
        script_path : str
            Resolved LASErMPNN script path[cite: 1].
        python_path : str
            Resolved LASErMPNN Python interpreter[cite: 1].
        pre_cmd : str
            Resolved shell prefix[cite: 1].
        """
    def __init__(
        self,
        python_path: str | None = None,
        script_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize the runner and resolve LASErMPNN configuration[cite: 1]."""
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
        """Return the short runner name[cite: 1].

        Returns
        -------
        str
            The literal runner name ``"lasermpnn"``[cite: 1].
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
        """Run LASErMPNN and merge collected results into ``poses``[cite: 1].

        Parameters
        ----------
        poses : Poses
            Input structures to pass to LASErMPNN[cite: 1].
        prefix : str
            Unique run prefix used to create the runner work directory and
            prefixed output score columns[cite: 1].
        jobstarter : JobStarter, optional
            Jobstarter for this call[cite: 1]. If omitted, the runner falls back to the
            instance jobstarter and then ``poses.default_jobstarter``[cite: 1].
        options : str, optional
            Extra command-line options passed to the upstream script[cite: 1].
        nstruct : int, default=1
            Number of structures to generate per input pose[cite: 1].
        fixed_residues : str or ResidueSelection, optional
            Residues to hold fixed during design[cite: 1]. If a string is provided, it is 
            treated as a column name in ``poses.df`` containing the selections[cite: 1].
        omit_AAs : list or str, optional
            Amino acids to disable during design[cite: 1]. Can be a comma-separated string 
            or a list of characters[cite: 1].
        overwrite : bool, default=False
            If ``True``, remove previous runner-owned outputs and rerun jobs[cite: 1].

        Returns
        -------
        Poses
            The input ``Poses`` object with LASErMPNN generated PDBs and score columns merged in[cite: 1].

        Raises
        ------
        KeyError
            If restricted parameters are set in options or flags (e.g., ``output_fasta_only``, 
            ``designs_per_input``, ``input_pdb_directory``, ``output_pdb_directory``)[cite: 1].
        RuntimeError
            If LASErMPNN runs but no score rows can be collected[cite: 1].
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
        """Split inputs into batches and format structures for LASErMPNN[cite: 1]."""
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
        """Format the shell commands for the LASErMPNN batches[cite: 1].

        Parameters
        ----------
        options : str
            Command line string of parameters to pass to the script[cite: 1].
        input_dirs : list of str
            List of generated batch directories containing input PDBs[cite: 1].
        output_dir : str
            Master output directory where the script will write outputs[cite: 1].
        nstruct : int
            Number of output structures to generate per input[cite: 1].
        fixed_residues : str or ResidueSelection, optional
            Passed residues to hold fixed, automatically appends the ``--fix_beta`` flag[cite: 1].
        omit_AAs : str or list, optional
            Amino acids to disable[cite: 1]. Translates to the ``--disabled_residues`` argument[cite: 1].

        Returns
        -------
        list of str
            Commands formatted for job execution[cite: 1].
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
        """Remove previous runner-owned outputs inside the work directory[cite: 1].

        Parameters
        ----------
        work_dir : str
            Runner work directory created for this prefix[cite: 1].
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
    """Modify PDB B-factors to define fixed residues for LASErMPNN[cite: 1].
    
    LASErMPNN uses bfactor columns to define fixed residues[cite: 1]. Bfactors of 
    designable residues are set to 0, fixed positions to 1[cite: 1].
    """
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
    """Collect outputs from the LASErMPNN batch directories and format them into a DataFrame[cite: 1]."""
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


