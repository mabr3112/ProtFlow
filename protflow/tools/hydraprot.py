"""ProtFlow runner for Hydraprot water prediction.

This module provides the :class:`Hydraprot` runner that predicts
the position of water molecules in protein structures.

The typical workflow is:

1. Ensure paths and environment hooks for Hydraprot are configured
   (see Notes on ``HYDRAPROT_DIR_PATH``, ``HYDRAPROT_PYTHON_PATH``,
   ``SIGMADOCK_PRE_CMD``).
2. Provide inputs as a :class:`Poses` collection of PDB or CIF complexes.
3. Call :meth:`Hydraprot.run` with possible ``options`` in dict format.
4. Consume the returned :class:`Poses` object whose ``.df`` is augmented with
   confidence scores and the path to the structures including predicted waters.

Authors
-------
Adrian Tripp

Version
-------
0.1.0

Examples
--------
Predict position of waters:

>>> from protflow.poses import Poses
>>> from protflow.tools.hydraprot import Hydraprot
>>> poses = Poses(
...     poses=["complex_A.cif", "complex_B.cif"],
...     work_dir="work/sigmadock_demo"
... )
>>> runner = Hydraprot()  # uses config defaults
>>> poses = runner.run(
...     poses=poses,
...     prefix="hydra",
...     overwrite=False,
... )
"""

import logging
import os
import shutil
import pandas as pd
from glob import glob
import string

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter, split_list
from protflow.poses import Poses, description_from_path
from protflow.utils.biopython_tools import biopython_load_structure, save_structure_to_file
from protflow.runners import Runner, RunnerOutput, prepend_cmd

class Hydraprot(Runner):
    """ProtFlow runner for Hydraprot water prediction.

    Predicts waters and adds them to current poses.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize Hydraprot runner.

        All path arguments fall back to ProtFlow config values when omitted.
        ``pre_cmd`` is typically a conda-activation snippet required for
        Hydraprot's subprocess to find its dependencies.
        """
        config = require_config()

        self.application_path = application_path or load_config_path(config, "HYDRAPROT_DIR_PATH")
        self.python_path = python_path or load_config_path(config, "HYDRAPROT_PYTHON_PATH")
        self.wrapper_path = os.path.join(load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR"), "hydraprot_wrapper.py")

        self.pre_cmd = pre_cmd or load_config_path(config, "HYDRAPROT_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "hydraprot"
        self.index_layers = 0

    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        options: dict | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Run Hydraprot and return poses augmented with predicted waters and their respective confidence scores.

        Options can be provided as a dictionary with keys according to Hydraprot/params/prediction_params.py (config should be omitted!).

        Parameters
        ----------
        options (dict, optional): dict containing keys according to Hydraprot/params/prediction_params.py. Defaults to None.
        """

        # 1) generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # scorefile reuse shortcut.
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

        # convert poses to pdb
        if not poses.determine_pose_type() == [".pdb"]:
            poses.convert_poses(f"{prefix}_conv", "pdb", overwrite=overwrite)

        # define number of batches and split poses accordingly
        n_batches = min(len(poses), jobstarter.max_cores)
        poses_sublist = split_list(input_list=poses.poses_list(), n_sublists=n_batches)

        # create input directories
        in_dirs = []
        for i, sublist in enumerate(poses_sublist):
            in_dirs.append(in_dir := os.path.join(work_dir, f"in_{i}"))
            os.makedirs(in_dir, exist_ok=True)
            for pose in sublist:
                shutil.copy(pose, in_dir)

        # write cmds
        cmds = [self.write_cmd(in_dir=in_dir, out_dir=os.path.join(work_dir, f"out_{i}"), options=options) for i, in_dir in enumerate(in_dirs)]

        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # execute commands.
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # collect and validate scores (module function, by convention).
        scores = collect_scores(work_dir=work_dir)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        # persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    def write_cmd(self, in_dir: str, out_dir: str, options: dict) -> str:
        """
        Return the Hydraprot shell command for a single pose.
        """
        if options:
            options = ",".join([f"{key}={value}" for key, value in options.items()])
        
        cmd = (
            f"{self.python_path} {self.wrapper_path} "
            f"--hydraprot_dir={self.application_path} "
            f"--input_dir={in_dir} "
            f"--output_dir={out_dir}"
        )

        if options:
            cmd = cmd + f" --override {options}"

        return cmd
    
    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete ``inputs/`` and ``outputs/`` from a previous run under ``work_dir``."""
        for in_dir in glob(os.path.join(work_dir, "in_*")):
            shutil.rmtree(in_dir)
        for out_dir in glob(os.path.join(work_dir, "out_*")):
            shutil.rmtree(out_dir)
        if os.path.isdir(outputs_dir := os.path.join(work_dir, "combined")):
            shutil.rmtree(outputs_dir)

def collect_scores(work_dir: str) -> pd.DataFrame:
    """
    Parse Hydraprot outputs and return a scores dataframe.
    """

    def find_first_missing_letter(letter_list):
        # Convert the input list to a set of uppercase letters for fast O(1) lookup
        present_letters = set(char.upper() for char in letter_list)
        
        # Iterate through the alphabet in order ('A' to 'Z')
        for letter in string.ascii_uppercase:
            if letter not in present_letters:
                return letter

    # create output dir
    os.makedirs(out_dir := os.path.join(work_dir, "combined"), exist_ok=True)

    # collect input and output
    input_pdbs = glob(os.path.join(work_dir, "in_*", "*pdb"))
    output_pdbs = glob(os.path.join(work_dir, "out_*", "*pdb"))

    df_in = pd.DataFrame({"original_location": input_pdbs, "description":[description_from_path(pdb) for pdb in input_pdbs]})
    df_out = pd.DataFrame({"water_location": output_pdbs, "description":[description_from_path(pdb).rsplit("_", maxsplit=1)[0] for pdb in output_pdbs]})

    # map input to output
    scores = df_out.merge(df_in, on="description")

    # combine predicted water with input structure
    rows = []
    for _, row in scores.iterrows():
        original = biopython_load_structure(row["original_location"])
        waters = biopython_load_structure(row["water_location"])["A"] # waters are stored in chain A
        water_chain = find_first_missing_letter([chain.id for chain in original.get_chains()])
        waters.id = water_chain
        original.add(waters)
        save_structure_to_file(original, path := os.path.join(out_dir, f"{row['description']}.pdb"))
        # extract confidences
        confidences = [atom.bfactor for atom in waters.get_atoms()]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        row["confidences"] = confidences
        row["mean_confidence"] = mean_confidence
        row["location"] = path
        rows.append(row)

    scores = pd.DataFrame(rows)

    return scores

