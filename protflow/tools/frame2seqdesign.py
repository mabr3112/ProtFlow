"""Frame2Seq Design Runner Module
=================================

This module provides the Frame2SeqDesign runner for protein sequence design
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
per-pose options, batching across available resources, optional residue fixing,
and parsing of per-residue negative pseudo-log-likelihoods.

Key Features
------------
ProtFlow integration: Works with :class:~protflow.poses.Poses,
:class:~protflow.jobstarters.JobStarter, and common runner utilities.

Batching & scheduling: Splits large pose sets into batches and delegates
execution to local or cluster job starters.

Flexible constraints: Supports residue fixing via
:class:~protflow.residues.ResidueSelection and common Frame2Seq options
(e.g., chain selection, sampling temperature, number of samples).

Automated parsing: Collects designed sequences and per-residue negative
pseudo-log-likelihoods into tidy :class:pandas.DataFrame structures.

Example
-------
.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from protflow.runners.frame2seqdesign import Frame2SeqDesign

    poses = Poses("inputs.json", work_dir="work")
    runner = Frame2SeqDesign()

    designed = runner.run(
        poses=poses,
        prefix="f2s_batch_01",
        jobstarter=LocalJobStarter(gpus=0, cpus=8),
        options={"temperature": 0.6, "num_samples": 3, "chain": "A"},
        pose_options=None,
        fixed_res_col=None,
        preserve_original_output=False,
        overwrite=False,
    )

    # Per-sample sequence scores are available in designed.df
    print(designed.df)

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
from protflow.residues import ResidueSelection
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.jobstarters import JobStarter, split_list
from protflow.runners import Runner, RunnerOutput, prepend_cmd

class Frame2SeqDesign(Runner):
    """
    Frame2SeqDesign Class
    =====================

    The :class:`Frame2SeqDesign` runner provides a streamlined interface for running
    Frame2Seq sequence design within the ProtFlow framework. It handles configuration,
    per-pose option merging, input preparation, job scheduling via a
    :class:`~protflow.jobstarters.JobStarter`, and result parsing into a tidy
    :class:`pandas.DataFrame` suitable for downstream analysis.

    Overview
    --------
    Frame2Seq proposes amino-acid sequences conditioned on a fixed protein backbone
    (a *frame*). This runner wraps a helper script (commonly ``run_frame2seq.py``)
    and integrates with ProtFlow’s :class:`~protflow.poses.Poses` and job-starting
    utilities to enable high-throughput or batched sequence design.

    Key Features
    ------------
    - **ProtFlow integration:** Works with :class:`~protflow.poses.Poses`,
      :class:`~protflow.jobstarters.JobStarter`, and Runner utilities.
    - **Batching & scheduling:** Splits large pose sets into batches and dispatches
      them to local or cluster backends.
    - **Flexible constraints:** Supports per-pose overrides and optional residue
      fixing (via a residue selection column on ``poses.df``).
    - **Automated parsing:** Collects designed sequences and per-residue negative
      pseudo-log-likelihoods into dataframes.

    Notes
    -----
    The runner expects a helper script to perform Frame2Seq inference and to emit
    per-sample FASTA sequences and (optionally) per-residue score files. Paths to
    the Python interpreter and any required pre-commands (e.g., environment/module
    activation) can be provided at construction or read from ProtFlow config.

    Examples
    --------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import LocalJobStarter
        from protflow.runners.frame2seqdesign import Frame2SeqDesign

        poses = Poses("inputs.json", work_dir="work")
        runner = Frame2SeqDesign()

        designed = runner.run(
            poses=poses,
            prefix="f2s_batch_01",
            jobstarter=LocalJobStarter(cpus=8),
            options={"chain": "A", "temperature": 0.6, "num_samples": 3},
            pose_options=None,
            fixed_res_col=None,
            preserve_original_output=False,
            overwrite=False,
        )

        # Aggregated results are accessible via designed.df
        print(designed.df)
    """

    def __init__(self, python_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter = None) -> None:
        """
        __init__ Method
        ===============

        Initialize a :class:`Frame2SeqDesign` runner with optional execution settings.

        Parameters
        ----------
        python_path : str or None, optional
            Absolute path to the Python interpreter used to invoke the Frame2Seq
            helper script. If ``None``, the value may be read from ProtFlow config.
        pre_cmd : str or None, optional
            Shell snippet to prepend to the execution command (e.g., environment
            activation, module loads). If ``None``, may be read from ProtFlow config.
        jobstarter : :class:`~protflow.jobstarters.JobStarter`, optional
            Default job starter to use when one is not explicitly supplied to
            :meth:`run`.

        Notes
        -----
        The runner resolves the helper script path and stores bookkeeping fields
        (e.g., ``index_layers``) used when collating batched outputs.
        """
        config = require_config()
        self.python_path = python_path or load_config_path(config, "FRAME2SEQ_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "FRAME2SEQ_PRE_CMD", is_pre_cmd=True)
        self.script_path = os.path.join(load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR"), "run_frame2seq.py")

        # setup runner
        self.name = "frame2seqdesign.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "frame2seqdesign.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, num_samples: int = 1, chain: str = "A", temperature: float = 1, options: dict = None, pose_options: list|str = None, fixed_res_col: str = None, preserve_original_output: bool = False, overwrite: bool = False) -> Poses:
        """
        run Method
        ==========

        Execute Frame2Seq sequence design over a set of input poses.

        This method prepares the working directory, merges global and per-pose
        options, writes batch inputs, delegates execution to ``jobstarter``, and
        aggregates per-sample outputs (sequences and scores) into the returned
        :class:`~protflow.poses.Poses` object.

        Parameters
        ----------
        poses : :class:`~protflow.poses.Poses`
            Input poses (backbone frames) to design against. Each row in
            ``poses.df`` represents one design unit.
        prefix : str
            Prefix for the run directory created beneath ``poses.work_dir``.
        jobstarter : :class:`~protflow.jobstarters.JobStarter`, optional
            Job starter used to schedule/batch the design jobs. If ``None``,
            falls back to an instance-level default or one attached to ``poses``.
        options : dict, optional
            Global Frame2Seq options applied to all poses (e.g., ``{"chain": "A",
            "temperature": 0.6, "num_samples": 3}``). User-supplied values may be
            overridden by pose-specific entries in ``pose_options``.
        pose_options : list of dict or None, optional
            Per-pose overrides, aligned with ``poses`` (length must match).
            Each dict can override keys in ``options`` for that pose.
        num_samples : int, optional
            Default number of samples per pose if not provided in ``options`` or
            ``pose_options``.
        chain : str, optional
            Default chain identifier to design if not provided elsewhere.
        temperature : float, optional
            Default sampling temperature if not provided elsewhere.
        fixed_res_col : str or None, optional
            Name of a column on ``poses.df`` containing residue selections to fix
            during design (e.g., indices, ranges, or selection strings compatible
            with ProtFlow’s residue selection utilities).
        preserve_original_output : bool, default ``False``
            If ``True``, keep the raw helper-script outputs (FASTA/score files).
            If ``False``, intermediate files may be removed after parsing.
        overwrite : bool, default ``False``
            If ``True``, ignore any existing scorefile and re-run design; if
            ``False``, reuse a valid existing scorefile when present.

        Returns
        -------
        :class:`~protflow.poses.Poses`
            The same ``poses`` object with its dataframe augmented by the
            aggregated Frame2Seq outputs (one row per generated sample).

        Raises
        ------
        TypeError
            If ``pose_options`` is not ``None`` or a list of dictionaries.
        ValueError
            If required options are missing for any pose or if forbidden keys are
            supplied (see Notes).
        RuntimeError
            If job execution fails or the number of parsed samples is inconsistent
            with expectations.

        Notes
        -----
        **Required options per pose**
            ``chain`` (str), ``temperature`` (float), and ``num_samples`` (int),
            either provided globally or overridden per pose.

        **Reserved/managed options**
            The runner may manage keys like ``save_indiv_neg_pll`` and
            ``save_indiv_seqs`` internally; users should not set them.

        Examples
        --------
        .. code-block:: python

            designed = runner.run(
                poses=poses,
                prefix="f2s_run",
                jobstarter=LocalJobStarter(cpus=8),
                options={"chain": "A", "temperature": 0.7, "num_samples": 5},
                pose_options=None,
                fixed_res_col=None,
                preserve_original_output=False,
                overwrite=False,
            )
        """

        self.index_layers = 1

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
        pose_options = self.merge_opts_and_pose_opts(poses, options, pose_options, num_samples, chain, temperature, fixed_res_col)

        mandatory_opts = ["chain_id", "temperature", "num_samples"]
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
            jobname="frame2seqdesign",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            preserve_original_output=preserve_original_output,
        )

        if len(scores.index) < len(poses.df.index) * num_samples:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def merge_opts_and_pose_opts(self, poses:Poses, options:dict, pose_options:list, num_samples:int, chain: str, temperature: float, fixed_res_col: str = None):
        """
        merge_opts_and_pose_opts Method
        ===============================

        Merge global options with per-pose overrides and expand derived settings.

        This helper validates and merges dictionaries from ``options`` and
        ``pose_options`` into a final list of per-pose dictionaries, applying default
        values for ``num_samples``, ``chain``, and ``temperature`` where needed.
        If ``fixed_res_col`` is given, per-pose residue fixing is encoded from
        the corresponding column in ``poses.df``.

        Parameters
        ----------
        poses : :class:`~protflow.poses.Poses`
            Input poses providing context (e.g., chain IDs, residue indices).
        options : dict or None
            Global options applied to all poses (may be empty or ``None``).
        pose_options : list of dict or None
            Per-pose overrides aligned with ``poses`` or ``None``.
        num_samples : int
            Default number of samples per pose when not provided elsewhere.
        chain : str
            Default chain identifier when not provided elsewhere.
        temperature : float
            Default sampling temperature when not provided elsewhere.
        fixed_res_col : str or None, optional
            Column name in ``poses.df`` containing residue selections to fix.

        Returns
        -------
        list of dict
            A list of per-pose option dictionaries, one per pose, ready for JSON
            input to the helper script.

        Raises
        ------
        TypeError
            If ``pose_options`` is not ``None`` or a list of dictionaries.
        ValueError
            If a per-pose dictionary is missing required keys after merging.

        Notes
        -----
        The exact encoding of residue selections should match ProtFlow’s residue
        selection utilities so that the helper script can respect fixed residues.
        """
        if not options:
            options = {}
        if chain:
            options["chain_id"] = chain
        if temperature:
            options["temperature"] = temperature
        if num_samples:
            options["num_samples"] = num_samples

        for i, pose_opt in enumerate(pose_options):
            if pose_opt is None:
                pose_opt = {}
            # merge pose_opt into options without overwriting pose_options
            merged = {**options, **pose_opt}
            pose_options[i] = merged

        if fixed_res_col:
            col_in_df(poses.df, fixed_res_col)
            fixed_residues = poses.df[fixed_res_col].to_list()
            for i, (pose_opt, fixed_res) in enumerate(zip(pose_options, fixed_residues)):
                # convert fixed residues to list (without chain information)
                if isinstance(fixed_res, ResidueSelection):
                    fixed_res = fixed_res.to_dict()[pose_opt["chain_id"]]
                if isinstance(fixed_res, dict):
                    fixed_res = fixed_res[pose_opt["chain_id"]]
                if not isinstance(fixed_res, list) or any(isinstance(x, str) for x in fixed_res):
                    raise KeyError("<fixed_res_col> must contain either a ResidueSelection, a dict or a 0-indexed list of residue positions without chain indicators!")
                pose_opt["fixed_positions"] = fixed_res
                pose_options[i] = pose_opt # update pose_options

        return pose_options

    def write_cmd(self, input_json: str, work_dir: str):
        """
        write_cmd Method
        ================

        Compose the shell command used to invoke the Frame2Seq helper script.

        Parameters
        ----------
        input_json : str
            Path to the input JSON file or directory generated for the batch.
        work_dir : str
            Path to the working/output directory for this batch.

        Returns
        -------
        str
            The full shell command string, including the configured Python
            interpreter, script path, and required arguments for Frame2Seq design.

        Notes
        -----
        The command may be prefixed by a user-provided ``pre_cmd`` (e.g., to
        activate an environment). The method does not execute the command.
        """
        return f"{self.python_path} {self.script_path} --input_json {input_json} --output_dir {work_dir} --method design"

def collect_scores(work_dir: str, preserve_original_output: bool = False) -> pd.DataFrame:
    """
    collect_scores Function
    =======================

    Collect and normalize Frame2Seq design outputs into a tidy dataframe.

    This function scans the results directory produced by the helper script,
    parses generated FASTA sequences and (if present) per-residue negative
    pseudo-log-likelihoods, and returns a :class:`pandas.DataFrame` with one row
    per generated sample.

    Parameters
    ----------
    work_dir : str
        Working directory containing the ``frame2seq_outputs`` subdirectories.
    preserve_original_output : bool, default ``False``
        If ``True``, keep the raw output directory tree. If ``False``, intermediate
        files may be deleted after parsing to save disk space.

    Returns
    -------
    pandas.DataFrame
        A dataframe with at least the following columns:

        - ``name`` : str — Sample identifier derived from FASTA headers.
        - ``sequence`` : str — Designed amino-acid sequence.
        - ``recovery`` : float — Fractional recovery parsed from headers (0..1).
        - ``per_res_neg_log_likelihood`` : list[float] — One value per residue,
          if score files were emitted.

        Additional metadata extracted from headers and filenames may be included.

    Raises
    ------
    FileNotFoundError
        If expected output files cannot be found under ``work_dir``.
    ValueError
        If output parsing fails due to malformed FASTA headers or missing fields.
    RuntimeError
        If the expected ``frame2seq_outputs`` directory is absent.

    Notes
    -----
    FASTA headers are expected to contain space-separated ``key=value`` pairs.
    The key ``recovery`` may be given as a percentage (e.g., ``87.5%``) and is
    converted to a fraction. Companion CSVs (if present) are used to assemble
    per-residue negative pseudo-log-likelihood vectors.

    Examples
    --------
    .. code-block:: python

        df = collect_scores(work_dir="work/f2s_batch_01", preserve_original_output=False)
        print(df.head())

    """
    def parse_frame2seq_fasta(fasta:str, out_dir:str):

        def parse_line(s: str):
            result = {}
            for part in s.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key] = value

            result["recovery"] = float(result["recovery"][:-1]) / 100
            return result

        def extract_ints(s: str):
            return [int(x) for x in re.findall(r"\d+", s)][0]

        # import fasta
        with open(fasta, "r") as fa:
            data = fa.readlines()

        header = data[0]
        seq = data[1]

        # parse information from header
        header_dict = parse_line(header)

        # create output file name
        suffix = extract_ints(fasta.split("_")[-1])
        suffix = suffix + 1
        new_name = f"{header_dict[">pdbid"]}_{str(suffix).zfill(4)}"
        filename = os.path.abspath(os.path.join(out_dir, f"{new_name}.fasta"))

        # create fasta file
        data = f">{new_name}\n{seq}"
        with open(filename, "w+") as fa:
            fa.write(data)

        header_dict["location"] = filename
        header_dict["description"] = description_from_path(filename)
        header_dict["seq"] = seq
        header_dict.pop(">pdbid")
        return header_dict


    results_dir = os.path.join(work_dir, "frame2seq_outputs")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"Could not find frame2seq_outputs directory at {results_dir}")

    seqs_dir = os.path.join(results_dir, "seqs")
    scores_dir = os.path.join(results_dir, "scores")

    os.makedirs(updated_seqs := os.path.join(work_dir, "frame2seqs_fasta"), exist_ok=True)

    scores = []
    fastas = glob(os.path.join(seqs_dir, "*.fasta"))
    for fasta in fastas:
        if not fasta.endswith("seqs.fasta"): # ignore collected results
            name = os.path.splitext(os.path.basename(fasta))[0]
            perreslogprobs = pd.read_csv(os.path.join(scores_dir, f"{name}.csv"))
            data_dict = parse_frame2seq_fasta(fasta, updated_seqs)
            data_dict["per_res_neg_log_likelihood"] = perreslogprobs["Negative pseudo-log-likelihood"].to_list()
        scores.append(data_dict)

    scores = pd.DataFrame(scores)

    # delete files
    if not preserve_original_output:
        shutil.rmtree(results_dir)

    return scores
