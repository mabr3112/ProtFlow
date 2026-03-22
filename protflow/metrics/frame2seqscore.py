"""Frame2Seq Design Runner Module
=================================

This module provides the Frame2SeqScore runner for evaluation of protein sequences
using Frame2Seq within the ProtFlow framework. It orchestrates input
preparation, command construction, job execution, and result collection to
enable high-throughput sequence design against fixed protein backbone frames.

Overview
--------
Frame2Seq is a conditional sequence design model that proposes amino-acid
sequences compatible with an input backbone (“frame”). This module wraps an
auxiliary script (typically run_frame2seq.py) and exposes a Pythonic
interface that integrates with ProtFlow’s :class:~protflow.poses.Poses
containers and :class:~protflow.jobstarters.JobStarter backends. It supports
per-pose options, batching across available resources,
and parsing of per-residue negative pseudo-log-likelihoods.

Key Features
------------
ProtFlow integration: Works with :class:~protflow.poses.Poses,
:class:~protflow.jobstarters.JobStarter, and common runner utilities.

Batching & scheduling: Splits large pose sets into batches and delegates
execution to local or cluster job starters.

Automated parsing: Collects per-residue negative
pseudo-log-likelihoods into tidy :class:pandas.DataFrame structures.

Examples
--------
.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from protflow.runners.frame2seqscore import Frame2SeqScore

    poses = Poses("inputs_with_sequences.json", work_dir="work")
    scorer = Frame2SeqScore()

    evaluated = scorer.run(
        poses=poses,
        prefix="f2s_score_batch_01",
        jobstarter=LocalJobStarter(cpus=8),
        chain="A",
        options={"some_flag": True},
        preserve_original_output=False,
        overwrite=False,
    )

    # Per-sequence scores are available in evaluated.df
    print(evaluated.df)

Authors
-------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""

# general imports
import json
import os
import logging
from glob import glob
import shutil
import re

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path
from protflow.poses import Poses, description_from_path
from protflow.jobstarters import JobStarter, split_list
from protflow.runners import Runner, RunnerOutput, prepend_cmd

class Frame2SeqScore(Runner):
    """
    Frame2SeqScore Class
    ====================

    The :class:`Frame2SeqScore` runner evaluates *existing* protein sequences
    against fixed backbone frames using Frame2Seq’s scoring mode (no sequence
    generation). It prepares inputs, schedules scoring jobs via a
    :class:`~protflow.jobstarters.JobStarter`, and parses per-residue negative
    pseudo-log-likelihoods into a tidy :class:`pandas.DataFrame`.

    Overview
    --------
    This runner wraps the Frame2Seq helper script (typically ``run_frame2seq.py``)
    with ``--method score`` to compute compatibility of provided sequences with
    a structural frame. It integrates with ProtFlow’s :class:`~protflow.poses.Poses`
    container and runner utilities for batch execution, job management, and
    standardized score collection.

    Key Features
    ------------
    - **Pure scoring (no design):** Uses Frame2Seq in scoring mode to compute
      negative pseudo-log-likelihoods (NPLL) for supplied sequences.
    - **Batching & scheduling:** Splits inputs across available cores/nodes and
      dispatches through a :class:`~protflow.jobstarters.JobStarter`.
    - **Clean outputs:** Aggregates mean NPLL per sequence and stores full
      per-residue NPLL vectors for downstream analysis.

    Examples
    --------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import LocalJobStarter
        from protflow.runners.frame2seqscore import Frame2SeqScore

        poses = Poses("inputs_with_sequences.json", work_dir="work")
        scorer = Frame2SeqScore()

        evaluated = scorer.run(
            poses=poses,
            prefix="f2s_score_batch_01",
            jobstarter=LocalJobStarter(cpus=8),
            chain="A",
            options={"some_flag": True},
            preserve_original_output=False,
            overwrite=False,
        )

        # Per-sequence scores are available in evaluated.df
        print(evaluated.df)
    """
    def __init__(self, python_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter = None) -> None:
        """
        __init__ Method
        ===============

        Initialize a :class:`Frame2SeqScore` runner with optional execution settings.

        Parameters
        ----------
        python_path : str or None, optional
            Absolute path to the Python interpreter used to invoke the Frame2Seq
            helper script. If ``None``, read from ProtFlow config
            (``FRAME2SEQ_PYTHON_PATH``).
        pre_cmd : str or None, optional
            Shell snippet to prepend to the command (e.g., environment activation).
            If ``None``, read from ProtFlow config (``FRAME2SEQ_PRE_CMD``).
        jobstarter : :class:`~protflow.jobstarters.JobStarter`, optional
            Default job starter used when none is provided to :meth:`run`.

        Notes
        -----
        The helper script path is resolved from
        ``AUXILIARY_RUNNER_SCRIPTS_DIR/run_frame2seq.py`` in the ProtFlow config.
        """
        # setup config
        config = require_config()
        self.python_path = python_path or load_config_path(config, "FRAME2SEQ_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "FRAME2SEQ_PRE_CMD", is_pre_cmd=True)
        self.script_path = os.path.join(load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR"), "run_frame2seq.py")

        # setup runner
        self.name = "frame2seqscore.py"
        self.index_layers = 0
        self.jobstarter = jobstarter

    def __str__(self):
        return "frame2seqscore.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, chain: str = "A", options: dict = None, pose_options: list|str = None, preserve_original_output: bool = False, overwrite: bool = False) -> Poses:
        """
        run Method
        ==========

        Evaluate existing sequences against backbone frames using Frame2Seq scoring.

        This method prepares batch input JSON files from ``poses``, merges global
        and per-pose options, executes Frame2Seq with ``--method score`` via the
        provided :class:`~protflow.jobstarters.JobStarter`, and aggregates results
        (mean and per-residue negative pseudo-log-likelihoods) into the returned
        :class:`~protflow.poses.Poses` object.

        Parameters
        ----------
        poses : :class:`~protflow.poses.Poses`
            Input poses referencing frames and associated sequences to score.
        prefix : str
            Prefix for the run directory created under ``poses.work_dir``.
        jobstarter : :class:`~protflow.jobstarters.JobStarter`, optional
            Job starter to schedule scoring jobs. If ``None``, falls back to an
            instance-level default or the one attached to ``poses``.
        chain : str, default ``"A"``
            Chain identifier to score (stored as ``chain_id`` in per-pose options).
        options : dict or None, optional
            Global Frame2Seq scoring options applied to all poses (may be empty).
        pose_options : list of dict or None, optional
            Per-pose overrides aligned with ``poses``; each dict can override
            keys from ``options`` for the corresponding pose.
        preserve_original_output : bool, default ``False``
            If ``True``, keep raw Frame2Seq output directories; otherwise, clean
            intermediate files after parsing.
        overwrite : bool, default ``False``
            If ``True``, ignore any existing scorefile and re-run scoring; if
            ``False``, reuse a valid existing scorefile when present.

        Returns
        -------
        :class:`~protflow.poses.Poses`
            The same ``poses`` object with its dataframe augmented by scoring
            results (one row per input pose).

        Raises
        ------
        TypeError
            If ``pose_options`` is not ``None`` or a list of dictionaries.
        KeyError
            If required options are missing for a pose (see Notes).
        RuntimeError
            If the number of parsed outputs is smaller than the number of inputs,
            suggesting failed runs.

        Notes
        -----
        **Required per-pose option**
            ``chain_id`` (str). This is derived from the ``chain`` argument if not
            given explicitly in options.

        **Reserved/managed options**
            The runner removes keys such as ``save_indiv_neg_pll`` and
            ``save_indiv_seqs`` as they are used internally by the helper script.

        Examples
        --------
        .. code-block:: python

            evaluated = scorer.run(
                poses=poses,
                prefix="f2s_score",
                jobstarter=LocalJobStarter(cpus=8),
                chain="A",
                options={"some_flag": True},
                preserve_original_output=False,
                overwrite=False,
            )
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"frame2seq_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # parse pose_options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # make sure pose options are dict
        if not all(pose_opt is None or isinstance(pose_opt, dict) for pose_opt in pose_options):
            raise TypeError("Pose options must be None or a dictionary!")
        
        # merge options and pose_options
        pose_options = self.merge_opts_and_pose_opts(poses, options, pose_options, chain)

        mandatory_opts = ["chain_id"]
        forbidden_opts = ["save_indiv_neg_pll", "save_indiv_seqs"]
        for pose, pose_opt in zip(poses.poses_list(), pose_options):
            if missing := [mandatory for mandatory in mandatory_opts if not mandatory in pose_opt]:
                raise KeyError(f"Mandatory options {missing} are missing (at least) for pose {pose}!")
            # remove opts important for output handling
            for opt in forbidden_opts:
                pose_opt.pop(opt, None)

        # create input dicts
        input_dicts = []
        for pose, pose_opt in zip(poses.poses_list(), pose_options):
            input_dicts.append({os.path.abspath(pose): pose_opt})
        
        # set up batches
        input_sublists = split_list(input_dicts, n_sublists=min([jobstarter.max_cores, len(input_dicts)]))

        # set up input dict
        input_dicts = [{pose: pose_opt for d in sublist for pose, pose_opt in d.items()} for sublist in input_sublists]

        # write input dicts
        in_jsons = []
        for i, input_dict in enumerate(input_dicts):
            in_json = os.path.join(work_dir, f"in_{str(i)}.json")
            with open(in_json, "w") as f:
                json.dump(input_dict, f, indent=4)  
            in_jsons.append(in_json)

        # write cmds:
        cmds = [self.write_cmd(in_json, work_dir) for in_json in in_jsons]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="frame2seqscore",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            preserve_original_output=preserve_original_output,
        )

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    
    def merge_opts_and_pose_opts(self, poses:Poses, options:dict, pose_options:list, chain: str):
        """
        merge_opts_and_pose_opts Method
        ===============================

        Merge global options with per-pose overrides and inject default ``chain_id``.

        Parameters
        ----------
        poses : :class:`~protflow.poses.Poses`
            Input poses providing per-pose context.
        options : dict or None
            Global options applied to all poses (may be empty or ``None``).
        pose_options : list of dict or None
            Per-pose overrides aligned with ``poses``; may include ``None`` entries.
        chain : str
            Default chain identifier; stored as ``chain_id`` if not present.

        Returns
        -------
        list of dict
            A list of merged per-pose option dictionaries ready for JSON input.

        Notes
        -----
        - If a per-pose entry is ``None``, it is promoted to an empty dict before
          merging.
        - ``chain`` is assigned to ``chain_id`` unless already defined per pose.
        """
        if not options:
            options = {}
        if chain:
            options["chain_id"] = chain

        for i, pose_opt in enumerate(pose_options):
            if pose_opt is None:
                pose_opt = {}
            # merge pose_opt into options without overwriting pose_options
            merged = {**options, **pose_opt}
            pose_options[i] = merged

        return pose_options

    def write_cmd(self, input_json: str, work_dir: str):
        """
        write_cmd Method
        ================

        Compose the shell command to invoke Frame2Seq scoring.

        Parameters
        ----------
        input_json : str
            Path to the batch input JSON file.
        work_dir : str
            Output directory for Frame2Seq results.

        Returns
        -------
        str
            Fully assembled shell command invoking the helper script with
            ``--method score``.
        """
        return f"{self.python_path} {self.script_path} --input_json {input_json} --output_dir {work_dir} --method score"

def collect_scores(work_dir: str, preserve_original_output: bool = False) -> pd.DataFrame:
    """
    collect_scores Function
    =======================

    Parse Frame2Seq scoring outputs and aggregate per-sequence metrics.

    This function reads per-pose CSV files from ``frame2seq_outputs/scores``,
    computes the mean negative pseudo-log-likelihood (NPLL) per sequence, and
    gathers the full per-residue NPLL vector. It returns a
    :class:`pandas.DataFrame` with standardized columns for downstream analysis.

    Parameters
    ----------
    work_dir : str
        Working directory containing ``frame2seq_outputs`` created by the runner.
    preserve_original_output : bool, default ``False``
        If ``True``, keep the raw output tree. If ``False``, remove the
        ``frame2seq_outputs`` directory after parsing to save disk space.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per input pose containing at least:
        - ``location`` : str — Absolute path to the scored input file (pose).
        - ``description`` : str — Description derived from the pose path.
        - ``score`` : float — Mean negative pseudo-log-likelihood across residues.
        - ``per_res_neg_log_likelihood`` : list[float] — Per-residue NPLL vector.

    Raises
    ------
    RuntimeError
        If the expected ``frame2seq_outputs`` directory is missing.
    FileNotFoundError
        If required CSV files are not found for an input pose.
    ValueError
        If CSVs are malformed or missing the
        ``Negative pseudo-log-likelihood`` column.

    Notes
    -----
    - CSV files are expected to be named like
      ``<pose_name>_<chain>_seq0.csv`` and to contain a
      ``Negative pseudo-log-likelihood`` column.
    - The returned ``score`` is the arithmetic mean of that column for each pose.

    Examples
    --------
    .. code-block:: python

        df = collect_scores(work_dir="work/f2s_score_batch_01", preserve_original_output=False)
        print(df.head())
    """
    results_dir = os.path.join(work_dir, "frame2seq_outputs")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"Could not find frame2seq_outputs directory at {results_dir}")
    
    in_jsons = glob(os.path.join(work_dir, "in_*.json"))
    poses = []
    chains = []
    for in_json in in_jsons:
        # import input json
        with open(in_json, "r") as jf:
            input_dict = json.load(jf)
        for key in input_dict.keys():
            poses.append(key)
            chains.append(input_dict[key]["chain_id"])

    scores_dir = os.path.join(results_dir, "scores")

    mean_logs = []
    perres_logs = []
    for pose, chain in zip(poses, chains):
        name = description_from_path(pose)
        score = pd.read_csv(os.path.join(scores_dir, f"{os.path.splitext(name)[0]}_{chain}_seq0.csv")) # TODO: cannot get multiple fasta input to work atm, output might look different then
        mean_logs.append(score["Negative pseudo-log-likelihood"].mean())
        perres_logs.append(score["Negative pseudo-log-likelihood"].to_list())
    
    scores = pd.DataFrame({"location": poses, "description": [description_from_path(pose) for pose in poses], "score": mean_logs, "per_res_neg_log_likelihood": perres_logs})

    # delete files
    if not preserve_original_output:
        shutil.rmtree(results_dir)

    return scores
    

    
