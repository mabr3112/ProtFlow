"""
ProtFlow runner for Intellifold.

This module provides a high-level `Intellifold` runner that:
(1) prepares Intellifold-compatible JSON inputs from sequences or structures,
(2) composes command lines from global and pose-specific options,
(3) distributes inference across available cores via a `JobStarter`,
and (4) aggregates Intellifold outputs (confidence, affinity, NPZ artifacts) into a
single score table for downstream orchestration.

The typical workflow is:

1. Ensure paths and environment hooks for Intellifold are configured
   (see Notes on `INTELLIFOLD_PATH`, `INTELLIFOLD_PYTHON`, `INTELLIFOLD_PRE_CMD`).
2. Provide inputs as a `Poses` collection (FASTA, PDB/CIF, or already
   Intellifold-formatted JSON). If needed, convert to JSON with
   `convert_poses_to_intellifold_json`.
3. Call `Intellifold.run(...)` with command-line `options` and optional
   `pose_options` to fan-out runs.
4. Consume the returned `Poses` object whose `.df` is augmented with a
   per-model score table and file locations of produced artifacts.

Notes
-----
- Configuration keys
  The runner reads its defaults from ProtFlow’s config via:
  `INTELLIFOLD_PATH` (path to the `intellifold` CLI entry point or module),
  `INTELLIFOLD_PYTHON` (interpreter used to invoke Intellifold), and
  `INTELLIFOLD_PRE_CMD` (shell prefix such as environment activation).
  Use `protflow.config` utilities to set these once per environment.
- MSA handling
  Intellifold can run with an empty MSA or fetch MSAs from a server. The runner
  exposes `msa_setting` to steer JSON content (`"empty"` vs `"server"`),
  while the CLI switch `--use_msa_server` remains the source of truth for
  server fetching. See `Intellifold._parse_msa_setting` and
  `convert_chain_seq_dict_to_json_dict`.

Examples
--------
Run Intellifold on a batch of structures, writing outputs to a fresh work directory
and collecting scores:

>>> from protflow.runners.intellifold import Intellifold
>>> from protflow.poses import Poses
>>> poses = Poses(
...     files=["A.pdb", "B.pdb", "C.pdb"],
...     work_dir="work/intellifold_demo"
... )
>>> runner = Intellifold()  # uses config defaults (INTELLIFOLD_PATH/PYTHON/PRE_CMD)
>>> poses = runner.run(
...     poses=poses,
...     prefix="intellifold_run",
...     options="--num_samples 4 --use_msa_server",
...     overwrite=False,
... )
>>> poses.df.columns[:8]  # score columns will include confidence & file paths
...

"""
# generals
import os
import json
import shutil
import logging
import random
from glob import glob
from pathlib import Path
from collections import defaultdict

# dependencies
import json
import pandas as pd

# custom
from ..poses import Poses, get_format, description_from_path
from ..residues import AtomSelection
from .. import load_config_path, require_config
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, parse_generic_options, options_flags_to_string
from ..utils.biopython_tools import load_sequence_from_fasta, get_sequence_from_pose, biopython_load_structure

class Intellifold(Runner):
    """
    The Intellifold runner prepares inputs (optionally batching by core), assembles Intellifold commands,
    dispatches them via a `JobStarter`, and aggregates results into a unified
    score file stored in the run directory.

    Parameters
    ----------
    intellifold_path : str, optional
        Executable or module path used with `predict` subcommand.
        If not provided, loaded from `INTELLIFOLD_PATH` in the ProtFlow config.
    intellifold_python : str, optional
        Python interpreter used to execute Intellifold. Defaults to `INTELLIFOLD_PYTHON`
        from the ProtFlow config.
    pre_cmd : str, optional
        Shell prefix prepended to each command. Use this to activate
        environments or modules (e.g., `conda activate intellifold`). If omitted,
        taken from `INTELLIFOLD_PRE_CMD` in the ProtFlow config.
    jobstarter : JobStarter, optional
        Default jobstarter to use if none is provided to `run()`.

    Attributes
    ----------
    name : str
        Fixed runner name: `"Intellifold"`.
    index_layers : int
        Number of index layers used when merging outputs (defaults to 2).
    jobstarter : JobStarter or None
        Optional default jobstarter stored on the runner instance.
    intellifold_path : str
        Resolved Intellifold executable/module path.
    intellifold_python : str
        Resolved interpreter path.
    pre_cmd : str
        Resolved shell prefix (may be empty).

    Notes
    -----
    - Score caching
      If a score file already exists for the given `prefix` and format and
      `overwrite` is `False` (and `--override` not present in `options`),
      existing results are returned without re-running Intellifold.
    - Batching behavior
      If `pose_options` are *not* provided, inputs are automatically split
      into at most `jobstarter.max_cores` batches to improve throughput.

    Examples
    --------
    Minimal run with default configuration, batched across cores:

    >>> runner = Intellifold()
    >>> poses = runner.run(
    ...     poses, prefix="demo",
    ...     options="--num_samples 2 --use_msa_server"
    ... )
    """
    def __init__(self, intellifold_path: str = None, intellifold_python: str = None, pre_cmd: str = None, jobstarter: JobStarter = None):
        """
        Initialize the Intellifold runner and resolve configuration.

        Parameters
        ----------
        intellifold_path : str, optional
            Path to the Intellifold program or module (with `predict` subcommand).
            Defaults to `INTELLIFOLD_PATH` from ProtFlow config.
        intellifold_python : str, optional
            Interpreter to call Intellifold with. Defaults to `INTELLIFOLD_PYTHON`.
        pre_cmd : str, optional
            Shell prefix (e.g., environment activation). Defaults to
            `INTELLIFOLD_PRE_CMD`.
        jobstarter : JobStarter, optional
            Default jobstarter to use when `run(jobstarter=None)`.

        Raises
        ------
        KeyError
            If required configuration keys are missing from the ProtFlow config.
        """
        config = require_config()
        self.intellifold_path = intellifold_path or load_config_path(config, "INTELLIFOLD_PATH")
        self.intellifold_python = intellifold_python or load_config_path(config, "INTELLIFOLD_PYTHON")
        self.pre_cmd = pre_cmd or load_config_path(config, "INTELLIFOLD_PRE_CMD", is_pre_cmd=True)

        self.name = "Intellifold"
        self.index_layers = 1 # intellifold can output many samples. We will always add index layers to reduce code complexity
        self.jobstarter = jobstarter

    def __str__(self):
        """
        String representation.

        Returns
        -------
        str
            The literal string ``"Intellifold"``.
        """
        return "Intellifold"

    def _parse_msa_setting(self, options: str, msa_setting: list[str]) -> str:
        """
        Normalize/resolve the MSA strategy used for JSON generation.

        The runner allows two MSA modes in the produced pose JSONs:
        - ``"empty"``: write ``msa: empty`` for each chain.
        - ``"server"``: also write ``msa: empty``, but *expect* the CLI option
          ``--use_msa_server`` to instruct Intellifold to fetch MSAs during runtime.

        Resolution order:
        1) If `msa_setting` is provided, it must be one of
           ``{"server", "empty", None}`` and takes precedence.
        2) Otherwise, if `"--use_msa_server"` appears in `options`, return
           ``"server"``.
        3) Else default to ``"empty"``.

        Parameters
        ----------
        options : str
            Command-line options that will be passed to Intellifold.
        msa_setting : str
            Desired JSON MSA mode or an empty/None value to auto-detect.

        Returns
        -------
        str
            Either ``"server"`` or ``"empty"``.

        Warns
        -----
        UserWarning
            If `msa_setting == "empty"` while `"--use_msa_server"` is present
            in `options`, since those choices conflict and could surprise
            users at execution time.

        Raises
        ------
        ValueError
            If `msa_setting` is neither ``"server"``, ``"empty"``, nor `None`.
        """
        # raise warning!
        if msa_setting == "empty" and "--use_msa_server" in options:
            logging.warning("msa_setting was set to :empty: while --use_msa_server was in options. This will lead to unexpected behavior.")

        # msa_setting has priority
        if msa_setting:
            allowed_settings = {"server", "empty", None}
            if msa_setting not in allowed_settings:
                raise ValueError(f"paramter :msa_setting: can be only one of {allowed_settings}! Your setting: {msa_setting}")
            return msa_setting

        # check in options
        if "--use_msa_server" in options:
            msa_setting = "server"
        else:
            msa_setting = "empty"
        return msa_setting

    def _parse_options(self, poses: Poses, options: str, pose_options: str|list[str], max_cores: int, out_dir: str, overwrite: bool = False) -> list[str]:
        '''Internal helper to parse options for intellifold.
        
        Construct one or more fully-formed option strings for Intellifold.

        If `pose_options` are supplied (string or list of strings), the runner
        expands them per input pose. Otherwise, a single options string is
        replicated across batches (up to `max_cores`) to enable parallel runs.

        In all cases, the output directory (`out_dir`) is injected into the
        parsed options, and the presence of `overwrite=True` appends the flag
        `--override` if it was not already present.

        Parameters
        ----------
        poses : Poses
            Input poses collection (used when mapping pose-level options).
        options : str
            Global CLI options (e.g., ``"--num_samples 4 --use_msa_server"``).
        pose_options : str or list of str
            Pose-specific overrides, templated for a given pose (handled by
            `prep_pose_options`). If provided, batching is disabled.
        max_cores : int
            Maximum number of concurrent batches (via `JobStarter`).
        out_dir : str
            Directory where Intellifold should write outputs for this run.
        overwrite : bool, optional
            If `True`, ensure `--override` is present in the options.

        Returns
        -------
        list of str
            One options string per Intellifold command to be executed.

        Raises
        ------
        ValueError
            If `pose_options` expansion fails or options cannot be parsed.
        '''
        if pose_options:
            # parse pose-specific options
            pose_options = self.prep_pose_options(poses, pose_options)
            parsed_options_raw = [parse_generic_options(options, pose_option) for pose_option in pose_options]

            # add out_dir to opts
            for opts_dict, flags in parsed_options_raw:
                opts_dict["out_dir"] = out_dir
                if overwrite and "override" not in flags:
                    flags.append("override")

            # recompile options strings.
            parsed_options = [options_flags_to_string(opts, flags, sep="--", no_quotes=False) for opts, flags in parsed_options_raw]

        # if no pose_options were given, predictions can be batched for faster inference.
        else:
            # create options for batched inputs
            options_raw = parse_generic_options(options=options, pose_options=None, sep="--") # keep cmd-opts in quotes (if needed)

            # add out_dir to opts
            options_raw[0]["output-dir"] = out_dir
            if overwrite and "override" not in options_raw[1]:
                options_raw[1].append("override")
            options_raw = options_flags_to_string(*options_raw, sep="--", no_quotes=False)

            # one options string per input batch
            parsed_options = [options_raw for _ in range(max_cores)]

        # output
        return parsed_options

    def _parse_poses(self, poses: Poses, pose_options: str|list[str], work_dir: str, max_cores: int) -> list[str]:
        '''helper function to parse poses for batch processing.

        Determine Intellifold input units (per pose vs. per batch subfolder).

        When `pose_options` are provided, Intellifold consumes each pose file directly.
        Otherwise, the runner creates up to `max_cores` batch subdirectories
        under ``{work_dir}/batch_inputs/batch_XXXX/`` and copies a partition of
        pose files into each to improve throughput.

        Parameters
        ----------
        poses : Poses
            The input collection (its `.poses_list()` is consulted).
        pose_options : str or list of str
            Presence disables batching; absence enables batching.
        work_dir : str
            Working directory for this run (batch subfolders are created here).
        max_cores : int
            Number of batch buckets to create at most.

        Returns
        -------
        list of str
            Either a list of individual pose file paths or batch directories.
        '''
        if pose_options:
            # parse poses
            intellifold_inputs = poses.poses_list()

        else:
             # batch input files into number of maximum specified cores:
            logging.info("Pose options not specified. Running in batch mode.")
            poses_sublists = split_list(poses.poses_list(), n_sublists=max_cores)

            # create input dirs and move sublist input files there
            intellifold_inputs = []
            for i, pose_sublist in enumerate(poses_sublists, start=1):
                # create subdir for batched inputs
                subdir_name = os.path.join(work_dir, "batch_inputs", f"batch_{str(i).zfill(4)}")
                os.makedirs(subdir_name, exist_ok=True)

                # copy poses in batch folders
                for pose in pose_sublist:
                    shutil.copy(pose, subdir_name)

                # add to intellifold input_list
                intellifold_inputs.append(subdir_name)
        return intellifold_inputs

    def _write_cmds(self, intellifold_inputs: list[str], parsed_options: list[str], af3_options: str = None) -> list[str]:
        '''
        Compose Intellifold command strings from resolved inputs and options.

        Each command is of the form:

        ``{pre_cmd} {intellifold_python} {intellifold_path} predict {input} {options}``

        Parameters
        ----------
        intellifold_inputs : list of str
            Per-command input path (individual JSON or batch directory).
        parsed_options : list of str
            Per-command options string as produced by `_parse_options`.

        Returns
        -------
        list of str
            Shell commands ready to be dispatched via `JobStarter.start()`.
        '''
        cmd_list = [
            f"{self.pre_cmd} {self.intellifold_python} {self.intellifold_path} predict {input_fn} {parsed_options}".strip()
            for input_fn, parsed_options in zip(intellifold_inputs, parsed_options)
        ]

        if af3_options:
            cmd_list = [f"{cmd} -- {af3_options}" for cmd in cmd_list]
        return cmd_list

    def run(
            self, poses: Poses, prefix: str, jobstarter: JobStarter = None,
            options: str = None, pose_options: str|list[str] = None, af3_options: str = None, nseeds: int = 1, seeds: list[int] = None, params: "IntellifoldParams" = None,
            modifications: list[dict]|str = None, msa_free: bool = False, unpairedMsaPath: str = None,
            pairedMsaPath: str = None, templates: list[dict] = None, poses_cols: list[str] = None, return_top_n_models: int = 1,
            overwrite: bool = False) -> Poses:
        '''
        Execute Intellifold on the given `poses` and collect results.

        The runner prepares inputs (converting to Intellifold JSON if needed),
        resolves MSA behavior, optionally augments pose JSONs using a provided
        `IntellifoldParams` object, dispatches the commands via `JobStarter`, then
        aggregates prediction confidence/affinity scores and artifact paths
        into a DataFrame saved as ``{prefix}/{name}_scores.{storage_format}``.

        Parameters
        ----------
        poses : Poses
            Input poses. Has to be protflow.poses.Poses class with poses in FASTA, 
            PDB/CIF, or Intellifold JSON; if not JSON, they are converted 
            with `convert_poses_to_intellifold_json`.
        prefix : str
            Run prefix / subdirectory under `poses.work_dir`. 
            Intellifold outputs will be stored in {poses.work_dir}/{prefix}/output
        jobstarter : JobStarter, optional
            Overrides the runner’s default jobstarter. If omitted, the runner
            tries, in order: the provided value, the instance default, and
            `poses.default_jobstarter`.
        options : str, optional
            Global CLI options for Intellifold (e.g., ``"--num_samples 8"``,
            ``"--use_msa_server"``).
        pose_options : str or list of str, optional
            Pose-specific option template(s); if provided, disables batching.
        params : IntellifoldParams, optional
            If given, used to *modify* or *extend* per-pose JSONs (e.g.,
            sequences, ligands, constraints, templates, properties) before
            running. Files are emitted under ``{prefix}/intellifold_inputs/``.
        overwrite : bool, optional
            If `True` (or if `--override` is present in `options`), re-run
            even if a scorefile already exists.
        msa_setting : str, optional
            One of ``{"server", "empty", ""}``. Empty/None means auto-resolve
            based on `options` (presence of `--use_msa_server`).

        Returns
        -------
        Poses
            The original `Poses` with results merged and indices layered.
            Artifacts (models, NPZs) are recorded as path columns.

        Raises
        ------
        RuntimeError
            If Intellifold finishes without producing any scores.
        TypeError
            If inputs cannot be converted to Intellifold JSON (unsupported formats).

        Examples
        --------
        Convert PDBs to JSON, add a ligand, and run with 4 samples per pose:

        >>> from protflow.runners.intellifold import Intellifold
        >>> from protflow.runners.intellifold import IntellifoldParams
        >>> params = IntellifoldParams()
        >>> params.add_ligand(ligand="CC(=O)O", id="LIG", ligand_type="smiles")
        >>> runner = Intellifold()
        >>> poses = runner.run(
        ...     poses=poses,
        ...     prefix="intellifold_with_ligand",
        ...     params=params,
        ...     options="--num_samples 4",
        ...     overwrite=True
        ... )

        Notes
        -----
        - Score caching: if a prior score file exists and neither `overwrite`
          nor `--override` is set, the runner returns cached results to save
          time.
        - Batching: when `pose_options` is absent, inputs are partitioned into
          at most `jobstarter.max_cores` batch folders to parallelize runs.
        - Artifacts: columns like ``plddt_location``, ``pae_location``, and
          ``pde_location`` point to NPZ files produced by Intellifold for each model.
        - Override behavior: Intellifold Runner sets overwrite=True if --override is specified in options (does not work for pose_options)!
        '''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        intellifold_out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(intellifold_out_dir, exist_ok=True)

        # sanitize
        options = options or ""

        # check for output
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if os.path.isfile(scorefile) and not (overwrite or "--override" in options):
            scores = get_format(scorefile)(scorefile) # loads scorefile DF with correct loading function
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        seeds = self._determine_seeds(nseeds, seeds)

        # check if poses are in correct format (json) (unless bypass_poses_check)
        convert_poses_to_intellifold_json(
            poses=poses,
            prefix=os.path.join(prefix, "poses_json"),
            modifications=modifications,
            msa_free=msa_free,
            unpairedMsaPath=unpairedMsaPath,
            pairedMsaPath=pairedMsaPath,
            templates=templates,
            poses_cols=poses_cols,
            seeds=seeds)

        # if IntellifoldParams are given, use IntellifoldParams to generate new poses based on params
        if params:
            intellifold_input_dir = os.path.join(work_dir, "intellifold_inputs")
            params.update_json_files(poses, intellifold_input_dir)

        # if pose_options are specified, run as is. Otherwise batch predictions
        intellifold_inputs = self._parse_poses(
            poses=poses,
            pose_options=pose_options,
            work_dir=work_dir,
            max_cores=jobstarter.max_cores
        )

        parsed_options = self._parse_options(
            poses=poses,
            options=options,
            pose_options=pose_options,
            max_cores=jobstarter.max_cores,
            out_dir=intellifold_out_dir,
            overwrite=overwrite
        )

        # compile commands# parse options and pose_options:
        cmds = self._write_cmds(intellifold_inputs, parsed_options, af3_options)

        # run intellifold
        jobstarter.start(
            cmds = cmds,
            jobname = f"{self.name}",
            output_path = work_dir
        )

        # collect scores
        scores = collect_scores(intellifold_out_dir, return_top_n_models)

        # output safety
        if len(scores) == 0:
            raise RuntimeError(f"Intellifold crashed. Check output logs and output directory for error logs: {work_dir}")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # return outputs
        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def _determine_seeds(self, nseeds: int, seeds: list[int] = None):
        """Generate random seeds if no input seed list is provided."""
        if not seeds:
            seeds = [random.randint(1, 10000) for _ in range(nseeds)]
        if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
            raise KeyError(":seeds: must be a list of integers!")
        return seeds

def convert_poses_to_intellifold_json(poses: Poses, prefix: str, seeds: list[int], modifications: list[dict]|str = None, msa_free: bool = False, unpairedMsaPath: str = None,
            pairedMsaPath: str = None, templates: list[dict] = None, poses_cols: list[str] = None, overwrite: bool = True, reset_poses: bool = True) -> None:
    """For now, this only reads the protein sequence, not anything else (no ligand support).

    Convert input poses to Intellifold-compatible JSONs.

    Creates one JSON per pose under ``{poses.work_dir}/{prefix}``, encoding chain
    sequences (and MSA choice) for Intellifold. Optionally updates ``poses.df["poses"]``
    to point to the newly created JSONs.

    Parameters
    ----------
    poses : Poses
        Input poses (protflow.poses.Poses class); poses must be in FASTA/PDB/CIF format poses table.
    prefix : str
        Subdirectory name under ``poses.work_dir`` where JSONs are written.
    msa : str or None
        One of ``"server"``, ``"empty"``, or a path to a custom ``.a3m`` file.
        ``"server"`` writes empty MSA entries and expects Intellifold to fetch MSAs.
    overwrite : bool, optional
        If ``True``, existing JSONs for the same prefix are replaced.
    reset_poses : bool, optional
        If ``True``, replace the ``poses`` column with JSON paths.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the output columns for this prefix already exist in ``poses.df``.
    ValueError
        If ``msa`` is neither ``"server"``, ``"empty"``, a valid path, nor ``None``.

    Examples
    --------
    >>> convert_poses_to_intellifold_json(poses, prefix="intellifold_inputs", msa="empty")
    >>> convert_poses_to_intellifold_json(poses, prefix="intellifold_inputs_srv", msa="server", reset_poses=False)

    Notes
    -----
    - The function is sequence-centric (ligands/templates/properties are handled later via :class:`IntellifoldParams`).
    """
    def _check_prefix(poses, prefix):
        if f"{prefix}_location" in poses.df.columns or f"{prefix}_description" in poses.df.columns:
            raise KeyError(f"Column {prefix} found in Poses DataFrame! Pick different Prefix!")

    def _determine_split_char(seq: str) -> str:
        return ":" if ":" in seq else "/"
    
    def _transpose_chain_list(seq_dict_list: list[dict]):

        chain_dict = defaultdict(list)
        for seq_dict in seq_dict_list:
            for chain, seq in seq_dict.items():
                chain_dict[chain].append(seq)

        all_same_length = len({len(lst) for lst in chain_dict.values()}) <= 1

        if not all_same_length:
            raise ValueError("All input poses must contain the same number of chains with the same chain identifiers!")
        return dict(chain_dict)

    # create output folder
    out_dir = os.path.join(os.path.abspath(poses.work_dir), prefix)
    os.makedirs(out_dir, exist_ok=True)

    # check if outputs already exist:
    out_fn_list = [
        os.path.join(out_dir, os.path.splitext(os.path.basename(pose))[0] + ".json") # replaces file-extension with .json
        for pose in poses.poses_list()
    ] # create new output names

    if all(os.path.isfile(out_fn) for out_fn in out_fn_list) and not overwrite:
        logging.info(f"Intellifold json files exist at {out_dir}. Skipping creation to save time.")

        # set new poses and exit
        if reset_poses:
            poses.df["poses"] = out_fn_list
        return None

    # sanity
    _check_prefix(poses, prefix)

    # get sequence from poses, this differs depending on which type of pose we have (.fasta or .pdb/.cif).
    if all(pose.endswith((".fa", ".fas", ".fasta")) for pose in poses.poses_list()):
        # load raw sequences
        sequences = [str(load_sequence_from_fasta(pose, return_multiple_entries=False).seq) for pose in poses.poses_list()]

        # assign chain IDs for sequences (start with [A -> Z], then [AA -> ZZ]):
        sequence_dict_list = [{idx_to_char(i): chain_seq for i, chain_seq in enumerate(seq.split(_determine_split_char(seq)))} for seq in sequences]

    elif all(pose.endswith((".pdb", "cif")) for pose in poses.poses_list()):
        sequence_dict_list = [get_sequence_from_pose(biopython_load_structure(pose), with_chains=True) for pose in poses.poses_list()]
    else:
        raise TypeError("Intellifold only supports files in .pdb, .cif, or .fa format!")
    
    chain_dict = _transpose_chain_list(sequence_dict_list)

    if len(chain_dict) > 1 and any(modifications, unpairedMsaPath, pairedMsaPath, templates):
        raise KeyError("Input of multi-chain poses with :modifications:, :unpairedMsaPath:, :pairedMsaPath: or :templates: currently not supported. Convert to single-chain poses and add additional poses via IntelliFoldParams.add_protein()!")

    for chain, seqs in chain_dict.items():
        poses.df[f"{prefix}_temp_seq_{chain}"] = seqs  

    if not poses_cols:
        poses_cols = ["sequence"]
    else:
        poses_cols.append("sequence")

    params = IntellifoldParams()
    for chain in chain_dict:
        params.add_protein(sequence=f"{prefix}_temp_seq_{chain}", id=chain, modifications=modifications, msa_free=msa_free, unpairedMsaPath=unpairedMsaPath, pairedMsaPath=pairedMsaPath, templates=templates, poses_cols=poses_cols)
    
    params.create_poses_json_files(poses=poses, out_dir=prefix, seeds=seeds, reset_poses=True)

    return None

def edit_intellifold_json(*args, **kwargs) -> None:
    """
    Placeholder for future JSON editing utilities.

    Raises
    ------
    NotImplementedError
        Always raised; function is a stub.
    """
    raise NotImplementedError

class IntellifoldParams:
    """
    Builder for per-pose Intellifold JSON content.

    Collects entries for proteins, nucleic acids, ligands, constraints,
    templates, and arbitrary properties. Each field value can be provided
    either as a *literal* or as a reference to a column in ``poses.df``.
    Column-referenced values are marked by passing their keys via
    ``poses_cols`` and are resolved at JSON generation time.

    Notes
    -----
    - Each added entity is stored internally and later rendered into
      the final JSON structure via :meth:`generate_json_files`.
    - For sequence modifications, use a list of dicts with at least
      ``{"position": <int>, "ccd": <str>}``.
    """
    def __init__(self):
        """
        Initialize an empty parameter collection.

        The instance accumulates lists:
        ``proteins``, ``dna``, ``rna``, ``ligands``, ``constraints``,
        ``templates``, and ``properties``—all of which are reflected
        into the resulting JSON during :meth:`generate_json_files`.
        """
        self.proteins = []
        self.dna = []
        self.rna = []
        self.ligands = []
        self.bondedAtomPairs = []
        self.properties = []
        self.custom_ccd = []

    def _check_MsaPath_format(self, MsaPath):
        """
        Validate the format of MSA paths.
        """
        if MsaPath is None:
            return None
        
        if not os.path.isfile(MsaPath):
            raise KeyError(f"Could not detect MSA at {MsaPath}")
        
        return MsaPath

    def _check_modifications_format(self, modifications, protein: bool=True) -> list[dict]|None:
        """
        Validate the format of residue modifications.

        Parameters
        ----------
        modifications : list[dict] or None
            A list of dicts with keys like ``"position"`` (int) and ``"ccd"`` (str),
            e.g. ``[{"position": 42, "ccd": "MSE"}]``; or ``None``.
        proteins : bool, optional
            Whether to check for protein or DNA/RNA modification format.


        Returns
        -------
        list[dict] or None
            The validated list (or ``None``) for downstream use.

        Raises
        ------
        ValueError
            If ``modifications`` is not a list of dicts.
        KeyError
            If any dict lacks required keys.
        """
        if modifications is None:
            return None
        if protein:
            if not (isinstance(modifications, list) and all(isinstance(elem, dict) for elem in modifications)):
                raise ValueError(f':modifications: parameter has to be in format [{"basePosition": RES_IDX, "modificationType": CCD}, ...]. modifications: {modifications}')
            for mod in modifications:
                if "ptmPosition" not in mod or "ptmType" not in mod:
                    raise KeyError(f'One of your modifications is missing a "ptmType" or "ptmPosition" key. :modifications: parameter has to be in format: [{"ptmPosition": RES_IDX, "ptmType": CCD}, ...]. culprit: {mod}')
        else:
            if not (isinstance(modifications, list) and all(isinstance(elem, dict) for elem in modifications)):
                raise ValueError(f':modifications: parameter has to be in format [{"basePosition": RES_IDX, "modificationType": CCD}, ...]. modifications: {modifications}')
            for mod in modifications:
                if "basePosition" not in mod or "modificationType" not in mod:
                    raise KeyError(f'One of your modifications is missing a "modificationType" or "basePosition" key. :modifications: parameter has to be in format: [{"basePosition": RES_IDX, "ptmType": CCD}, ...]. culprit: {mod}')

        return modifications
    
    def _check_atom_format(self, atom):
        if isinstance(atom, AtomSelection):
            if not len(atom) == 1:
                raise KeyError(f":atom: must be an AtomSelection containing a single atom, not {atom}!")
            atom = atom.to_boltz_atom()
        if not len(atom) == 3:
            raise KeyError(f":atom: must be a single atom in format [CHAIN_IDX, RES_IDX, ATOM_NAME], not {atom}")
        return atom
    
    def _check_templates_format(self, templates):
        if templates is None:
            return None
        if not (isinstance(templates, list) and all(isinstance(elem, dict) for elem in templates)):
            raise ValueError(f':templates: parameter has to be in format [{"mmcifPath": RES_IDX, "queryIndices": [QUERY_IDX_LIST], "templateIndices": [TEMPLATE_IDX_LIST]}, ...]. modifications: {templates}')
        for temp in templates:
            if "queryIndices" not in temp or "templateIndices" not in temp or not any(mmcif in temp for mmcif in ["mmcif", "mmcifPath"]):
                raise KeyError(f'One of your modifications is missing an essential key. :templates: parameter has to be in format: [{"mmcifPath": RES_IDX, "queryIndices": [QUERY_IDX_LIST], "templateIndices": [TEMPLATE_IDX_LIST]}, ...]. culprit: {temp}')
            if not len(temp["queryIndices"]) == len(temp["templateIndices"]):
                raise KeyError(f"The length of query ({len(temp['queryIndices'])}) and template indices ({len(temp['templateIndices'])}) is not equal for template {temp}.")
        return templates


    def add_protein(self, sequence: str, id: str|list[str], modifications: list[dict]|str = None, msa_free: bool = False, unpairedMsaPath: str = None, pairedMsaPath: str = None, templates: list[dict] = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Intellifold naming convention here, so id overwrite will be ignored in the sake of user experience.
        '''Helper to add protein entry.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; may be a literal or a column name (see Notes).
        id : str or list[str]
            Chain ID(s) to use in the JSON; may be literal or a column name.
        modifications : list[dict] or None, optional
            Per-residue modifications (see :meth:`_check_modifications_format`).
            e.g. [{"ptmType": CCD, "ptmPosition": RES_IDX}, ...] (can also be a string
            pointing to a column in poses.df that contains the modifications dicts)
        poses_cols : list[str], optional
            Keys that should be **read from** ``poses.df`` instead of used literally,
            e.g. ``["sequence", "id", "modifications", "unpairedMsaPath"]``.

        Returns
        -------
        None

        Examples
        --------
        >>> bp.add_protein(sequence="ACDE...", id="A")
        >>> bp.add_protein(sequence="seq_col", id="chain_id_col", poses_cols=["sequence", "id"])

        Notes
        -----
        Any key named in ``poses_cols`` is treated as a reference to a column in
        the current pose row when rendering JSON.
        '''
        # instantiate default value
        poses_cols = poses_cols or []

        if msa_free and (unpairedMsaPath or pairedMsaPath or templates):
            raise KeyError(":msa_free: is incompatible with :unpairedMsaPath:, :pairedMsaPath: and :templates:")

        # compile protein dict in IntellifoldParams representation.
        protein_dict = {
            "id": id,
            "sequence": sequence,
        }
        
        if modifications:
            protein_dict["modifications"] = modifications if "modifications" in poses_cols else self._check_modifications_format(modifications, True),
        if unpairedMsaPath:
            protein_dict["unpairedMsaPath"] = unpairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(unpairedMsaPath),
        if pairedMsaPath:
            protein_dict["pairedMsaPath"] = pairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(pairedMsaPath),
        if templates:
            protein_dict["templates"] = templates if "templates" in poses_cols else self._check_templates_format(templates)

        if msa_free:
            protein_dict["unpairedMsaPath"] = ""
            protein_dict["pairedMsaPath"] = ""

        protein_dict = {key: (val, key in poses_cols) for key, val in protein_dict.items()} # wrap in poses_cols flag!

        # add proteins entry to IntellifoldParams instance.
        self.proteins.append(protein_dict)

    def add_dna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, msa_free: bool = False, unpairedMsaPath: str = None, pairedMsaPath: str = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Intellifold naming convention here, so id overwrite will be ignored in the sake of user experience.
        """
        Add an DNA entry.

        Parameters
        ----------
        sequence : str
            Nucleotide sequence (literal or column name).
        id : str or list[str]
            Identifier(s) for the RNA entry.
        modifications : list[dict] or None, optional
            Residue-level modifications for RNA.
        poses_cols : list[str], optional
            Keys to interpret as column names in ``poses.df``.

        Returns
        -------
        None
        """
        # instantiate default value
        poses_cols = poses_cols or []

        if msa_free and (unpairedMsaPath or pairedMsaPath):
            raise KeyError(":msa_free: is incompatible with :unpairedMsaPath: and :pairedMsaPath:")
        
        # compile dna dict in IntellifoldParams representation
        dna_dict = {
            "id": id,
            "sequence": sequence,
        }

        if modifications:
            dna_dict["modifications"] = modifications if "modifications" in poses_cols else self._check_modifications_format(modifications, True),
        if unpairedMsaPath:
            dna_dict["unpairedMsaPath"] = unpairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(unpairedMsaPath),
        if pairedMsaPath:
            dna_dict["pairedMsaPath"] = pairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(pairedMsaPath),

        if msa_free:
            dna_dict["unpairedMsaPath"] = ""
            dna_dict["pairedMsaPath"] = ""
        
        dna_dict = {key: (val, key in poses_cols) for key, val in dna_dict.items()} # wrap in poses_cols!

        # add rna entry to IntellifoldParams instance
        self.dna.append(dna_dict)

    def add_rna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, msa_free: bool = False, unpairedMsaPath: str = None, pairedMsaPath: str = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Intellifold naming convention here, so id overwrite will be ignored in the sake of user experience.
        """
        Add an RNA entry.

        Parameters
        ----------
        sequence : str
            Nucleotide sequence (literal or column name).
        id : str or list[str]
            Identifier(s) for the RNA entry.
        modifications : list[dict] or None, optional
            Residue-level modifications for RNA.
        poses_cols : list[str], optional
            Keys to interpret as column names in ``poses.df``.

        Returns
        -------
        None
        """
        # instantiate default value
        poses_cols = poses_cols or []

        if msa_free and (unpairedMsaPath or pairedMsaPath):
            raise KeyError(":msa_free: is incompatible with :unpairedMsaPath: and :pairedMsaPath:")
        
        # compile dna dict in IntellifoldParams representation
        rna_dict = {
            "id": id,
            "sequence": sequence,
        }

        if modifications:
            rna_dict["modifications"] = modifications if "modifications" in poses_cols else self._check_modifications_format(modifications, True),
        if unpairedMsaPath:
            rna_dict["unpairedMsaPath"] = unpairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(unpairedMsaPath),
        if pairedMsaPath:
            rna_dict["pairedMsaPath"] = pairedMsaPath if "unpairedMsaPath" in poses_cols else self._check_MsaPath_format(pairedMsaPath),


        if msa_free:
            rna_dict["unpairedMsaPath"] = ""
            rna_dict["pairedMsaPath"] = ""
        
        rna_dict = {key: (val, key in poses_cols) for key, val in rna_dict.items()} # wrap in poses_cols!

        # add rna entry to IntellifoldParams instance
        self.rna.append(rna_dict)

    def add_ligand(self, ligand: str, id: str|list[str], ligand_type: str = "smiles", poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Intellifold naming convention here, so id overwrite will be ignored in the sake of user experience.
        """
        Add a ligand entry.

        Parameters
        ----------
        ligand : str
            The ligand specification. For ``ligand_type="smiles"``, provide a SMILES;
            for ``"ccd"``, provide an RCSB CCD ID.
        id : str or list[str]
            Ligand ID(s) in the output JSON.
        ligand_type : {"smiles", "ccd"}
            How to interpret ``ligand``.
        poses_cols : list[str], optional
            Keys (e.g., ``["ligand", "id"]``) to read from ``poses.df``.
            ``"ligand_type"`` is not supported as a pose-column.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``"ligand_type"`` is included in ``poses_cols``.
        """
        # instantiate default value
        poses_cols = poses_cols or []

        # sanity
        if "ligand_type" in poses_cols:
            raise ValueError("We are sorry, but ligand_type is not yet supported in 'poses_cols'.")

        # verify ligand type
        if ligand_type not in {"smiles", "ccdCodes"}:
            raise ValueError(f"Parameter :ligand_type: can be only one of {{'smiles', 'ccdCodes'}}. ligand_type: {ligand_type}")

        # compile ligand dict in IntellifoldParams representation
        ligand_dict = {
            "id": (id, "id" in poses_cols),
            ligand_type: (ligand, "ligand" in poses_cols),
        }

        # add ligands entry to IntellifoldParams instance
        self.ligands.append(ligand_dict)

    def add_custom_ccd(self, userCCDPath: str, from_pose_col: bool = False):
        self.custom_ccd.append({"userCCDPath": (userCCDPath, from_pose_col)})


    def add_bond(self, atom1: list | AtomSelection, atom2: list | AtomSelection, poses_cols: list[str] = None) -> None:
        """
        Add a geometric or pocket constraint.

        Parameters
        ----------
        constraint_type : str
            One of typical types such as ``"bond"``, ``"angle"``, ``"dihedral"``,
            ``"contact"``, or ``"pocket"`` (see Notes for expected fields).
        poses_cols : list[str], optional
            Keys in ``kwargs`` that should be read from ``poses.df``.
        **kwargs
            Constraint parameters (literal values or column names if listed
            in ``poses_cols``).

        Returns
        -------
        None

        Examples
        --------
        Contact constraint between two tokens:
        >>> bp.add_constraint(
        ...     "contact",
        ...     token1=["A", 42], token2=["B", "CA"], max_distance=6.0
        ... )

        Notes
        -----
        - ``bond/angle/dihedral`` expect standard token lists like
          ``["CHAIN", RES_IDX/ATOM_NAME]``.
        - ``pocket`` typically expects a ``binder`` (chain) and a list of
          pocket ``contacts`` plus an optional ``max_distance``.
        """
        # instantiate default value
        poses_cols = poses_cols or []

        atom1 = atom1 if "atom1" in poses_cols else self._check_atom_format(atom1)
        atom2 = atom2 if "atom2" in poses_cols else self._check_atom_format(atom2)

        # compile ligand dict in IntellifoldParams representation
        bond_list = [
            ({atom1, "atom1" in poses_cols}, {atom2, "atom2" in poses_cols})
        ]

        # add constraint entry to IntellifoldParams instance
        self.bondedAtomPairs.append(bond_list)

    def add_property(self, property_type: str, poses_cols: list[str] = None, **kwargs) -> None:
        """
        Attach arbitrary key–value properties to the JSON.

        Parameters
        ----------
        property_type : str
            A top-level property category (e.g., ``"inference"``).
        poses_cols : list[str], optional
            Keys in ``kwargs`` that should be read from ``poses.df``.
        **kwargs
            Property payload (literal values or column names if listed
            in ``poses_cols``).

        Returns
        -------
        None

        Examples
        --------
        >>> IntellifoldParams.add_property('affinity', binder="binder_chain_col", poses_cols=["binder"])
        >>> IntellifoldParams.add_property('affinity', binder="B")
        """
        supported_properties = {"affinity"}
        if property_type not in supported_properties:
            raise ValueError(f"property {property_type} not supported. Supported properties: {supported_properties}")

        # parse poses cols
        poses_cols = poses_cols or []

        # process property kwargs
        processed_kwargs = {key: (val, key in poses_cols) for key, val in kwargs.items()}

        property_dict = {property_type: processed_kwargs}
        self.properties.append(property_dict)

    def create_poses_json_files(self, poses: Poses, out_dir: str, seeds: list[int], reset_poses: bool = True) -> None:
        '''Converts poses into new .json files at 'prefix' based on current paramters.
        or: render accumulated parameters into per-pose JSON files.

        Resolves all values that were marked as pose-columns against
        ``poses.df`` and writes one JSON per pose into ``out_dir``.
        Optionally updates ``poses.df["poses"]`` to point to the new files.

        Parameters
        ----------
        poses : Poses
            Poses whose table provides column values for pose-bound fields.
        out_dir : str
            Output directory where JSON files are written.
        reset_poses : bool, optional
            If ``True``, replace the ``poses`` column with the new JSON paths.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a requested pose-column is missing from ``poses.df``.
        '''
        def _parse_dict_for_pose(pose: pd.Series, entity_dict: dict) -> dict:
            '''Fills in values from pose.df if values have "pose_col" set to true.'''
            parsed_dict = {
                key: pose[val] if is_pose_col else val # selects value from pose.df if pose_col was specified.
                for key, (val, is_pose_col) in entity_dict.items()
            }
            return parsed_dict

        def _add_key_if_not_there(input_dict, key, value) -> None:
            '''Adds {key: value} into 'input_dict' if 'key' is not yet in 'input_dict'.'''
            if key not in input_dict:
                input_dict[key] = value

        # create output dir
        os.makedirs(out_dir, exist_ok=True)

        # operate per-pose
        # add proteins, dna, rna, and ligands to sequences entry:
        new_poses = []
        for pose in poses:
            # read pose json
            pose_json = {}
            #print(pose_json)

            # add sequences
            for protein_dict in self.proteins:
                pose_json["name"] = pose["poses_description"]
                pose_json["modelSeeds"] = seeds
                _add_key_if_not_there(pose_json, "sequences", [])
                pose_json["sequences"].append({"protein": _parse_dict_for_pose(pose, protein_dict)})
                pose_json["dialect"] = "alphafold3"
                pose_json["version"] = 4

            # write output
            new_pose_fn = os.path.join(out_dir, f"{pose['poses_description']}.json")
            intellifold_json_writer(new_pose_fn, pose_json)

            # add new filename to new_poses list for integration into poses later
            new_poses.append(new_pose_fn)

        # set new poses
        if reset_poses:
            poses.df["poses"] = new_poses
        logging.info(f"Finished converting poses to .json files based on IntellifoldParams.\nAdded {len(self.proteins)} proteins, {len(self.ligands)} ligands, {len(self.dna)} DNA molecules, and {len(self.rna)} RNA molecules.")


    def update_json_files(self, poses: Poses, out_dir: str, reset_poses: bool = True) -> None:
        '''Converts poses into new .json files at 'prefix' based on current paramters.
        or: render accumulated parameters into per-pose JSON files.

        Resolves all values that were marked as pose-columns against
        ``poses.df`` and writes one JSON per pose into ``out_dir``.
        Optionally updates ``poses.df["poses"]`` to point to the new files.

        Parameters
        ----------
        poses : Poses
            Poses whose table provides column values for pose-bound fields.
        out_dir : str
            Output directory where JSON files are written.
        reset_poses : bool, optional
            If ``True``, replace the ``poses`` column with the new JSON paths.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a requested pose-column is missing from ``poses.df``.
        '''
        def _parse_dict_for_pose(pose: pd.Series, entity_dict: dict) -> dict:
            '''Fills in values from pose.df if values have "pose_col" set to true.'''
            parsed_dict = {
                key: pose[val] if is_pose_col else val # selects value from pose.df if pose_col was specified.
                for key, (val, is_pose_col) in entity_dict.items()
            }
            return parsed_dict

        def _add_key_if_not_there(input_dict, key, value) -> None:
            '''Adds {key: value} into 'input_dict' if 'key' is not yet in 'input_dict'.'''
            if key not in input_dict:
                input_dict[key] = value

        # sanity
        if not all(fp.endswith(".json") for fp in poses.poses_list()):
            raise TypeError("Poses must be in intellifold-compatible .json format. Use the function 'protflow.tools.intellifold.convert_poses_to_intellifold_json()' for this!")

        # create output dir
        os.makedirs(out_dir, exist_ok=True)

        # operate per-pose
        # add proteins, dna, rna, and ligands to sequences entry:
        new_poses = []
        for pose in poses:
            # read pose json
            pose_json = intellifold_json_reader(pose["poses"])
            #print(pose_json)

            # add sequences
            for protein_dict in self.proteins:
                _add_key_if_not_there(pose_json, "sequences", [])
                pose_json["sequences"].append({"protein": _parse_dict_for_pose(pose, protein_dict)})

            for dna_dict in self.dna:
                pose_json["sequences"].append({"dna": _parse_dict_for_pose(pose, dna_dict)})

            for rna_dict in self.rna:
                pose_json["sequences"].append({"rna": _parse_dict_for_pose(pose, rna_dict)})

            for ligand_dict in self.ligands:
                pose_json["sequences"].append({"ligand": _parse_dict_for_pose(pose, ligand_dict)})

            if len(self.custom_ccd) > 1:
                raise KeyError("Intellifold currently only supports a single custom ligand as input!")
            
            for userccd in self.custom_ccd:
                _add_key_if_not_there(pose_json, "userCCDPath", [])
                pose_json["userCCDPath"] = userccd

            # add constraints (constraints are in different format than proteins/dna/rna/ligand)
            for bond in self.bondedAtomPairs:
                _add_key_if_not_there(pose_json, "bondedAtomPairs", [])
                pose_json["bondedAtomPairs"].append([_parse_dict_for_pose(pose, atom1_dict), _parse_dict_for_pose(pose, atom2_dict)] for atom1_dict, atom2_dict in bond)
                pose_json["bondedAtomPairs"] = _parse_atomselection_bonds(pose_json["bondedAtomPairs"])

            # add properties
            for property_dict in self.properties:
                _add_key_if_not_there(pose_json, "properties", [])
                pose_json["properties"].append({property_type: _parse_dict_for_pose(pose, property_args) for property_type, property_args in property_dict.items()})

            # write output
            new_pose_fn = os.path.join(out_dir, os.path.basename(pose["poses"]))
            intellifold_json_writer(new_pose_fn, pose_json)

            # add new filename to new_poses list for integration into poses later
            new_poses.append(new_pose_fn)

        # set new poses
        if reset_poses:
            poses.df["poses"] = new_poses
        logging.info(f"Finished converting poses to .json files based on IntellifoldParams.\nAdded {len(self.proteins)} proteins, {len(self.ligands)} ligands, {len(self.dna)} DNA molecules, and {len(self.rna)} RNA molecules.\nAdded {len(self.constraints)} constraints, {len(self.templates)} templates, and {len(self.properties)} properties.")

def _parse_atomselection_bonds(bonds: list):
    """Converts AtomSelections to AF3/Boltz-style bond specifications."""
    bonds_list = []
    for (atom1, atom2) in bonds:
        if isinstance(atom1, AtomSelection):
            atom1 = atom1.to_boltz_atom()
        if isinstance(atom2, AtomSelection):
            atom2 = atom2.to_boltz_atom()
        if not len(atom1) == 3 or not len(atom2) == 3:
            raise KeyError(f"Atoms for bonds must be specified as a list in format [[Chain1, Resnum1, Name1], [Chain2, Resnum2, Name2]], not {[atom1, atom2]}")
        bonds_list.append([atom1, atom2])
    return bonds_list

def convert_chain_seq_dict_to_json_dict(chain_seq_dict: dict[str,str], msa: str = None, ignore_nonexistent_msa_file: bool = False) -> dict[str,str]:
    '''
    Converts dictionary that contains {chain: seq, ...} into intellifold-compatible protein entries {}.
    When msa is set to 'server', the function will set <msa: empty> (use option --use_msa_server!)

    Convert a chain→sequence mapping into Intellifold JSON "protein" entries.

    Parameters
    ----------
    chain_seq_dict : dict[str, str]
        Mapping from chain ID to amino-acid sequence.
    msa : {"server", "empty", "auto"} or str or None, optional
        If ``"server"/"empty"/"auto"/None`` → write ``"msa": "empty"`` per chain.
        If a string path → use it as the MSA file for all chains (exists unless
        ``ignore_nonexistent_msa_file=True``).
    ignore_nonexistent_msa_file : bool, optional
        If ``True``, skip the existence check for the path given in ``msa``.

    Returns
    -------
    list of dict
        One dict per chain with keys ``id``, ``sequence``, and ``msa``.

    Raises
    ------
    FileNotFoundError
        If ``msa`` is a path that does not exist and ``ignore_nonexistent_msa_file`` is ``False``.
    ValueError
        If ``msa`` is not one of the accepted values.

    Examples
    --------
    >>> convert_chain_seq_dict_to_json_dict({"A": "ACDE", "B": "FGHI"}, msa="empty")
    [{'id': 'A', 'sequence': 'ACDE', 'msa': 'empty'}, {'id': 'B', 'sequence': 'FGHI', 'msa': 'empty'}]
    '''
    # parse MSA option
    match msa:
        case "server" | "empty" | "auto" | None:
            msa_val = "empty"
        case str():
            msa_val = msa
            if not os.path.isfile(msa) and not ignore_nonexistent_msa_file:
                raise FileNotFoundError(f"Specified MSA file not found: {msa}")
        case _:
            raise ValueError(f"Not allowed: {msa}. Either provide a path to an existing MSA, None, 'server' (to get msa from msa-server), or 'empty'.")

    # create protein json for each chain.
    protein_json = [
        {
            "id": chain,
            "sequence": seq,
            "msa": msa_val
        }
        for chain, seq in chain_seq_dict.items()
    ]
    return protein_json

def _folders_in_dir(dir_path: str) -> list:
    '''finds and returns all folders in :dir_path: that don't start with a . (hidden folders).'''
    dir_path = Path(dir_path)
    # Note: if this causes issues in the future with random folders, add an additional check for the subdirectory
    # to contain at least a file with f'{parent_folder_name}_model_0.{"cif" or "pdb"}'
    return_dirs = [p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")] # exclude hidden folders
    return return_dirs

def _read_intellifold_confidence_file(fp: str) -> pd.Series:
    '''Reads intellifold confidence output file.'''
    with open(fp, 'r', encoding="UTF-8") as f:
        scores_dict = json.load(f)
    return pd.Series(scores_dict)

def _get_last_dir_name(path: str) -> str:
    '''returns name of last directory in path.'''
    p = Path(path)
    if p.is_dir() or str(path).endswith("/"):
        return p.name
    return p.parent.name

def collect_scores(work_dir: str, return_top_n_models: int = 1) -> pd.DataFrame:
    """
    collect_scores Function
    =======================

    Collects and processes output from AlphaFold3 prediction directories,
    extracting ranking and confidence values while optionally converting CIF models to PDB.

    Detailed Description
    --------------------
    The function navigates through subdirectories of `work_dir`, reads AlphaFold3's
    `ranking_scores.csv` and associated JSON confidence files for each model,
    compiles the data into a Pandas DataFrame, and optionally converts CIF
    files to PDB using Open Babel. Supports limiting output to a specified
    number of top-ranked models.

    Parameters:
        work_dir (str): Root folder containing AF3 output directories for each pose.
        convert_cif_to_pdb_dir (str, optional): If set, converted PDB files will be saved here.
        return_top_n_models (int, optional): Number of top models per pose to include. Default is 1.

    Returns:
        pandas.DataFrame: A DataFrame with columns including:
            - ranking_score, pLDDT, TM-scores, RMSD, etc.
            - location (path to model), description, sequence, etc.

    Raises:
        RuntimeError: If fewer output models are found than expected.
        FileNotFoundError: If essential AF3 files are missing (e.g., ranking_scores.csv).

    Examples
    --------
    .. code-block:: python

        df = collect_scores(
            work_dir="af3_preds",
            convert_cif_to_pdb_dir="af3_pdbs",
            return_top_n_models=1
        )
        print(df.loc[:, ["location", "ranking_score"]])

    Further Details
    ---------------
        - Ignores any folder starting with `mmseq` (MSA generation).
        - Converts only up to `return_top_n_models` CIFs per pose.
        - Converts and updates the `location` column if `convert_cif_to_pdb_dir` is provided.
    """

    def load_all_models(out_dir: str) -> pd.DataFrame:
        os.makedirs(model_dir := os.path.join(out_dir, "models"), exist_ok=True)
        ranks = pd.read_csv(os.path.join(out_dir, f"{os.path.basename(out_dir)}_ranking_scores.csv"))
        ranks.sort_values("ranking_score", ascending=False, inplace=True)
        ranks.reset_index(drop=True, inplace=True)
        in_name = os.path.basename(out_dir)
        data = os.path.join(out_dir, f"{in_name}_data.json")
        with open(data, 'r', encoding="UTF-8") as file:
            data = file.read()
        data = json.loads(data)
        scores = []
        for i, row in ranks.iterrows():
            model_dir = os.path.join(out_dir, f"seed-{int(row['seed'])}_sample-{int(row['sample'])}")
            model_id = in_name + "_" + os.path.basename(model_dir)
            confidences = pd.read_json(os.path.join(model_dir, f"{model_id}_confidences.json"), typ='series', orient='records')
            summary = pd.read_json(os.path.join(model_dir, f"{model_id}_summary_confidences.json"), typ='series', orient='records')
            score = pd.concat([summary, confidences])
            model = os.path.join(model_dir, f"{model_id}_model.cif")
            score["location"] = os.path.abspath(shutil.copy(model, os.path.join(model_dir, f"{data['name']}_{i+1:04d}.cif")))
            score["description"] = description_from_path(score["location"])
            scores.append(score)
        scores = pd.DataFrame(scores)
        scores["sequence"] = data["sequences"][0]["protein"]["sequence"]
        return scores

    # collect all output directories, ignore mmseqs dirs
    out_dirs = [d for d in glob(os.path.join(work_dir, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("mmseq")]

    scores = []
    for out_dir in out_dirs:
        data = load_all_models(out_dir)
        data = data.head(return_top_n_models)
        scores.append(data)
    scores = pd.concat(scores)
    scores.reset_index(drop=True, inplace=True)
    return scores

def idx_to_char(idx: int) -> str:
    """
    Convert a 0-based index to letters like Excel columns.
    0 -> 'A', 25 -> 'Z', 26 -> 'AA', 27 -> 'AB', ...
    """
    if not isinstance(idx, int):
        raise TypeError("idx must be an int")
    if idx < 0:
        raise ValueError("idx must be >= 0")

    n = idx + 1  # shift to 1-based index
    chars = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord('A') + rem))
    return ''.join(reversed(chars))


def intellifold_json_writer(out_path: str, intellifold_json: dict) -> None:
    """
    Write a Intellifold JSON document to disk (pretty, stable layout).

    Parameters
    ----------
    out_path : str
        Output ``.json`` path.
    intellifold_json : dict
        JSON document to write.

    Returns
    -------
    None
    """
    with open(out_path, 'w', encoding='utf-8') as json_file:
        # indent=4 makes the output file easy to read. 
        # You can remove it to save space on large files.
        json.dump(intellifold_json, json_file, indent=4)

def intellifold_json_reader(in_path: str) -> dict:
    """
    Read a Intellifold JSON file into a Python dictionary.

    Parameters
    ----------
    in_path : str
        Path to a ``.json`` file.

    Returns
    -------
    dict
        Parsed JSON document.
    """
    with open(in_path, 'r', encoding="UTF-8") as f:
        return json.load(f)
