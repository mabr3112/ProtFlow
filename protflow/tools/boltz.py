"""
ProtFlow runner for Boltz.

This module provides a high-level `Boltz` runner that:
(1) prepares Boltz-compatible YAML inputs from sequences or structures,
(2) composes command lines from global and pose-specific options,
(3) distributes inference across available cores via a `JobStarter`,
and (4) aggregates Boltz outputs (confidence, affinity, NPZ artifacts) into a
single score table for downstream orchestration.

The typical workflow is:

1. Ensure paths and environment hooks for Boltz are configured
   (see Notes on `BOLTZ_PATH`, `BOLTZ_PYTHON`, `BOLTZ_PRE_CMD`).
2. Provide inputs as a `Poses` collection (FASTA, PDB/CIF, or already
   Boltz-formatted YAML). If needed, convert to YAML with
   `convert_poses_to_boltz_yaml`.
3. Call `Boltz.run(...)` with command-line `options` and optional
   `pose_options` to fan-out runs.
4. Consume the returned `Poses` object whose `.df` is augmented with a
   per-model score table and file locations of produced artifacts.

Notes
-----
- Configuration keys
  The runner reads its defaults from ProtFlow’s config via:
  `BOLTZ_PATH` (path to the `boltz` CLI entry point or module),
  `BOLTZ_PYTHON` (interpreter used to invoke Boltz), and
  `BOLTZ_PRE_CMD` (shell prefix such as environment activation).
  Use `protflow.config` utilities to set these once per environment.
- MSA handling
  Boltz can run with an empty MSA or fetch MSAs from a server. The runner
  exposes `msa_setting` to steer YAML content (`"empty"` vs `"server"`),
  while the CLI switch `--use_msa_server` remains the source of truth for
  server fetching. See `Boltz._parse_msa_setting` and
  `convert_chain_seq_dict_to_yaml_dict`.

Examples
--------
Run Boltz on a batch of structures, writing outputs to a fresh work directory
and collecting scores:

>>> from protflow.runners.boltz import Boltz
>>> from protflow.poses import Poses
>>> poses = Poses(
...     files=["A.pdb", "B.pdb", "C.pdb"],
...     work_dir="work/boltz_demo"
... )
>>> runner = Boltz()  # uses config defaults (BOLTZ_PATH/PYTHON/PRE_CMD)
>>> poses = runner.run(
...     poses=poses,
...     prefix="boltz_run",
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
from glob import glob
from pathlib import Path

# dependencies
import yaml
import pandas as pd

# custom
from ..poses import Poses, get_format
from .. import load_config_path, require_config
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, parse_generic_options, options_flags_to_string
from ..utils.biopython_tools import load_sequence_from_fasta, get_sequence_from_pose, biopython_load_protein

class Boltz(Runner):
    """
    The Boltz runner prepares inputs (optionally batching by core), assembles Boltz commands,
    dispatches them via a `JobStarter`, and aggregates results into a unified
    score file stored in the run directory.

    Parameters
    ----------
    boltz_path : str, optional
        Executable or module path used with `predict` subcommand.
        If not provided, loaded from `BOLTZ_PATH` in the ProtFlow config.
    boltz_python : str, optional
        Python interpreter used to execute Boltz. Defaults to `BOLTZ_PYTHON`
        from the ProtFlow config.
    pre_cmd : str, optional
        Shell prefix prepended to each command. Use this to activate
        environments or modules (e.g., `conda activate boltz`). If omitted,
        taken from `BOLTZ_PRE_CMD` in the ProtFlow config.
    jobstarter : JobStarter, optional
        Default jobstarter to use if none is provided to `run()`.

    Attributes
    ----------
    name : str
        Fixed runner name: `"Boltz"`.
    index_layers : int
        Number of index layers used when merging outputs (defaults to 2).
    jobstarter : JobStarter or None
        Optional default jobstarter stored on the runner instance.
    boltz_path : str
        Resolved Boltz executable/module path.
    boltz_python : str
        Resolved interpreter path.
    pre_cmd : str
        Resolved shell prefix (may be empty).

    Notes
    -----
    - Score caching
      If a score file already exists for the given `prefix` and format and
      `overwrite` is `False` (and `--override` not present in `options`),
      existing results are returned without re-running Boltz.
    - Batching behavior
      If `pose_options` are *not* provided, inputs are automatically split
      into at most `jobstarter.max_cores` batches to improve throughput.

    Examples
    --------
    Minimal run with default configuration, batched across cores:

    >>> runner = Boltz()
    >>> poses = runner.run(
    ...     poses, prefix="demo",
    ...     options="--num_samples 2 --use_msa_server"
    ... )
    """
    def __init__(self, boltz_path: str = None, boltz_python: str = None, pre_cmd: str = None, jobstarter: JobStarter = None):
        """
        Initialize the Boltz runner and resolve configuration.

        Parameters
        ----------
        boltz_path : str, optional
            Path to the Boltz program or module (with `predict` subcommand).
            Defaults to `BOLTZ_PATH` from ProtFlow config.
        boltz_python : str, optional
            Interpreter to call Boltz with. Defaults to `BOLTZ_PYTHON`.
        pre_cmd : str, optional
            Shell prefix (e.g., environment activation). Defaults to
            `BOLTZ_PRE_CMD`.
        jobstarter : JobStarter, optional
            Default jobstarter to use when `run(jobstarter=None)`.

        Raises
        ------
        KeyError
            If required configuration keys are missing from the ProtFlow config.
        """
        config = require_config()
        self.boltz_path = boltz_path or load_config_path(config, "BOLTZ_PATH")
        self.boltz_python = boltz_python or load_config_path(config, "BOLTZ_PYTHON")
        self.pre_cmd = pre_cmd or load_config_path(config, "BOLTZ_PRE_CMD", is_pre_cmd=True)

        self.name = "Boltz"
        self.index_layers = 2 # boltz can output many samples. We will always add index layers to reduce code complexity
        self.jobstarter = jobstarter

    def __str__(self):
        """
        String representation.

        Returns
        -------
        str
            The literal string ``"Boltz"``.
        """
        return "Boltz"

    def _parse_msa_setting(self, options: str, msa_setting: list[str]) -> str:
        """
        Normalize/resolve the MSA strategy used for YAML generation.

        The runner allows two MSA modes in the produced pose YAMLs:
        - ``"empty"``: write ``msa: empty`` for each chain.
        - ``"server"``: also write ``msa: empty``, but *expect* the CLI option
          ``--use_msa_server`` to instruct Boltz to fetch MSAs during runtime.

        Resolution order:
        1) If `msa_setting` is provided, it must be one of
           ``{"server", "empty", None}`` and takes precedence.
        2) Otherwise, if `"--use_msa_server"` appears in `options`, return
           ``"server"``.
        3) Else default to ``"empty"``.

        Parameters
        ----------
        options : str
            Command-line options that will be passed to Boltz.
        msa_setting : str
            Desired YAML MSA mode or an empty/None value to auto-detect.

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
        '''Internal helper to parse options for boltz.
        
        Construct one or more fully-formed option strings for Boltz.

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
            Directory where Boltz should write outputs for this run.
        overwrite : bool, optional
            If `True`, ensure `--override` is present in the options.

        Returns
        -------
        list of str
            One options string per Boltz command to be executed.

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
            options_raw[0]["out_dir"] = out_dir
            if overwrite and "override" not in options_raw[1]:
                options_raw[1].append("override")
            options_raw = options_flags_to_string(*options_raw, sep="--", no_quotes=False)

            # one options string per input batch
            parsed_options = [options_raw for _ in range(max_cores)]

        # output
        return parsed_options

    def _parse_poses(self, poses: Poses, pose_options: str|list[str], work_dir: str, max_cores: int) -> list[str]:
        '''helper function to parse poses for batch processing.

        Determine Boltz input units (per pose vs. per batch subfolder).

        When `pose_options` are provided, Boltz consumes each pose file directly.
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
            boltz_inputs = poses.poses_list()

        else:
             # batch input files into number of maximum specified cores:
            logging.info("Pose options not specified. Running in batch mode.")
            poses_sublists = split_list(poses.poses_list(), n_sublists=max_cores)

            # create input dirs and move sublist input files there
            boltz_inputs = []
            for i, pose_sublist in enumerate(poses_sublists, start=1):
                # create subdir for batched inputs
                subdir_name = os.path.join(work_dir, "batch_inputs", f"batch_{str(i).zfill(4)}")
                os.makedirs(subdir_name, exist_ok=True)

                # copy poses in batch folders
                for pose in pose_sublist:
                    shutil.copy(pose, subdir_name)

                # add to boltz input_list
                boltz_inputs.append(subdir_name)
        return boltz_inputs

    def _write_cmds(self, boltz_inputs: list[str], parsed_options: list[str]) -> list[str]:
        '''
        Compose Boltz command strings from resolved inputs and options.

        Each command is of the form:

        ``{pre_cmd} {boltz_python} {boltz_path} predict {input} {options}``

        Parameters
        ----------
        boltz_inputs : list of str
            Per-command input path (individual YAML or batch directory).
        parsed_options : list of str
            Per-command options string as produced by `_parse_options`.

        Returns
        -------
        list of str
            Shell commands ready to be dispatched via `JobStarter.start()`.
        '''
        cmd_list = [
            f"{self.pre_cmd} {self.boltz_python} {self.boltz_path} predict {input_fn} {parsed_options}".strip()
            for input_fn, parsed_options in zip(boltz_inputs, parsed_options)
        ]
        return cmd_list

    def run(
            self, poses: Poses, prefix: str, jobstarter: JobStarter = None,
            options: str = None, pose_options: str|list[str] = None, params: "BoltzParams" = None,
            overwrite: bool = False, msa_setting: str = ""
        ) -> Poses:
        '''
        Execute Boltz on the given `poses` and collect results.

        The runner prepares inputs (converting to Boltz YAML if needed),
        resolves MSA behavior, optionally augments pose YAMLs using a provided
        `BoltzParams` object, dispatches the commands via `JobStarter`, then
        aggregates prediction confidence/affinity scores and artifact paths
        into a DataFrame saved as ``{prefix}/{name}_scores.{storage_format}``.

        Parameters
        ----------
        poses : Poses
            Input poses. Has to be protflow.poses.Poses class with poses in FASTA, 
            PDB/CIF, or Boltz YAML; if not YAML, they are converted 
            with `convert_poses_to_boltz_yaml`.
        prefix : str
            Run prefix / subdirectory under `poses.work_dir`. 
            Boltz outputs will be stored in {poses.work_dir}/{prefix}/output
        jobstarter : JobStarter, optional
            Overrides the runner’s default jobstarter. If omitted, the runner
            tries, in order: the provided value, the instance default, and
            `poses.default_jobstarter`.
        options : str, optional
            Global CLI options for Boltz (e.g., ``"--num_samples 8"``,
            ``"--use_msa_server"``).
        pose_options : str or list of str, optional
            Pose-specific option template(s); if provided, disables batching.
        params : BoltzParams, optional
            If given, used to *modify* or *extend* per-pose YAMLs (e.g.,
            sequences, ligands, constraints, templates, properties) before
            running. Files are emitted under ``{prefix}/boltz_inputs/``.
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
            If Boltz finishes without producing any scores.
        TypeError
            If inputs cannot be converted to Boltz YAML (unsupported formats).

        Examples
        --------
        Convert PDBs to YAML, add a ligand, and run with 4 samples per pose:

        >>> from protflow.runners.boltz import Boltz
        >>> from protflow.runners.boltz import BoltzParams
        >>> params = BoltzParams()
        >>> params.add_ligand(ligand="CC(=O)O", id="LIG", ligand_type="smiles")
        >>> runner = Boltz()
        >>> poses = runner.run(
        ...     poses=poses,
        ...     prefix="boltz_with_ligand",
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
          ``pde_location`` point to NPZ files produced by Boltz for each model.
        - Override behavior: Boltz Runner sets overwrite=True if --override is specified in options (does not work for pose_options)!
        '''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        boltz_out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(boltz_out_dir, exist_ok=True)

        # sanitize
        options = options or ""

        # check for output
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if os.path.isfile(scorefile) and not (overwrite or "--override" in options):
            scores = get_format(scorefile)(scorefile) # loads scorefile DF with correct loading function
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        #### write boltz inputs
        # parse msa_setting
        msa_setting = self._parse_msa_setting(options, msa_setting)

        # check if poses are in correct format (yaml) (unless bypass_poses_check)
        if not all(fp.endswith(".yaml") for fp in poses.poses_list()):
            convert_poses_to_boltz_yaml(poses, prefix=f"{prefix}/poses_yaml", msa=msa_setting)

        # if BoltzParams are given, use BoltzParams to generate new poses based on params
        if params:
            boltz_input_dir = os.path.join(work_dir, "boltz_inputs")
            params.generate_yaml_files(poses, boltz_input_dir)

        # if pose_options are specified, run as is. Otherwise batch predictions
        boltz_inputs = self._parse_poses(
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
            out_dir=boltz_out_dir,
            overwrite=overwrite
        )

        # compile commands# parse options and pose_options:
        cmds = self._write_cmds(boltz_inputs, parsed_options)

        # run boltz
        jobstarter.start(
            cmds = cmds,
            jobname = f"{self.name}",
            output_path = work_dir
        )

        # collect scores
        scores = collect_boltz_scores(boltz_out_dir)

        # output safety
        if len(scores) == 0:
            raise RuntimeError(f"Boltz crashed. Check output logs and output directory for error logs: {work_dir}")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # return outputs
        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

def convert_poses_to_boltz_yaml(poses: Poses, prefix: str, msa: str = None, overwrite: bool = True, reset_poses: bool = True) -> None:
    """For now, this only reads the protein sequence, not anything else (no ligand support).

    Convert input poses to Boltz-compatible YAMLs.

    Creates one YAML per pose under ``{poses.work_dir}/{prefix}``, encoding chain
    sequences (and MSA choice) for Boltz. Optionally updates ``poses.df["poses"]``
    to point to the newly created YAMLs.

    Parameters
    ----------
    poses : Poses
        Input poses (protflow.poses.Poses class); poses must be in FASTA/PDB/CIF format poses table.
    prefix : str
        Subdirectory name under ``poses.work_dir`` where YAMLs are written.
    msa : str or None
        One of ``"server"``, ``"empty"``, or a path to a custom ``.a3m`` file.
        ``"server"`` writes empty MSA entries and expects Boltz to fetch MSAs.
    overwrite : bool, optional
        If ``True``, existing YAMLs for the same prefix are replaced.
    reset_poses : bool, optional
        If ``True``, replace the ``poses`` column with YAML paths.

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
    >>> convert_poses_to_boltz_yaml(poses, prefix="boltz_inputs", msa="empty")
    >>> convert_poses_to_boltz_yaml(poses, prefix="boltz_inputs_srv", msa="server", reset_poses=False)

    Notes
    -----
    - The function is sequence-centric (ligands/templates/properties are handled later via :class:`BoltzParams`).
    """
    def _check_prefix(poses, prefix):
        if f"{prefix}_location" in poses.df.columns or f"{prefix}_description" in poses.df.columns:
            raise KeyError(f"Column {prefix} found in Poses DataFrame! Pick different Prefix!")

    def _determine_split_char(seq: str) -> str:
        return ":" if ":" in seq else "/"

    # create output folder
    out_dir = os.path.join(os.path.abspath(poses.work_dir), prefix)
    os.makedirs(out_dir, exist_ok=True)

    # check if outputs already exist:
    out_fn_list = [
        os.path.join(out_dir, os.path.splitext(os.path.basename(pose))[0] + ".yaml") # replaces file-extension with .yaml
        for pose in poses.poses_list()
    ] # create new output names

    if all(os.path.isfile(out_fn) for out_fn in out_fn_list) and not overwrite:
        logging.info(f"Boltz yaml files exist at {out_dir}. Skipping creation to save time.")

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
        sequence_dict_list = [get_sequence_from_pose(biopython_load_protein(pose, model_id=0), with_chains=True) for pose in poses.poses_list()]
    else:
        raise TypeError("Boltz only supports files in .pdb, .cif, or .fa format!")

    # now convert pose-level lists to valid boltz yamls. [{chain: seq, ...}, ...] -> [boltz-yaml-formatted-pose, ...]
    pose_yamls_raw = [convert_chain_seq_dict_to_yaml_dict(pose_dict, msa=msa, ignore_nonexistent_msa_file=True) for pose_dict in sequence_dict_list]

    # now create boltz pose_yamls
    boltz_pose_yamls = [
        {"sequences": [{"protein": chain_dict} for chain_dict in pose_yaml]}
        for pose_yaml in pose_yamls_raw
    ]

    # store yamls
    for pose_yaml, out_fn in zip(boltz_pose_yamls, out_fn_list):
        boltz_yaml_writer(out_fn, pose_yaml)

    # set new poses
    if reset_poses:
        poses.df["poses"] = out_fn_list
    return None

def edit_boltz_yaml(*args, **kwargs) -> None:
    """
    Placeholder for future YAML editing utilities.

    Raises
    ------
    NotImplementedError
        Always raised; function is a stub.
    """
    raise NotImplementedError

class BoltzParams:
    """
    Builder for per-pose Boltz YAML content.

    Collects entries for proteins, nucleic acids, ligands, constraints,
    templates, and arbitrary properties. Each field value can be provided
    either as a *literal* or as a reference to a column in ``poses.df``.
    Column-referenced values are marked by passing their keys via
    ``poses_cols`` and are resolved at YAML generation time.

    Notes
    -----
    - Each added entity is stored internally and later rendered into
      the final YAML structure via :meth:`generate_yaml_files`.
    - For sequence modifications, use a list of dicts with at least
      ``{"position": <int>, "ccd": <str>}``.
    """
    def __init__(self):
        """
        Initialize an empty parameter collection.

        The instance accumulates lists:
        ``proteins``, ``dna``, ``rna``, ``ligands``, ``constraints``,
        ``templates``, and ``properties``—all of which are reflected
        into the resulting YAML during :meth:`generate_yaml_files`.
        """
        self.proteins = []
        self.dna = []
        self.rna = []
        self.ligands = []
        self.constraints = []
        self.templates = []
        self.properties = []

    def _check_modifications_format(self, modifications) -> list[dict]|None:
        """
        Validate the format of residue modifications.

        Parameters
        ----------
        modifications : list[dict] or None
            A list of dicts with keys like ``"position"`` (int) and ``"ccd"`` (str),
            e.g. ``[{"position": 42, "ccd": "MSE"}]``; or ``None``.

        Returns
        -------
        list[dict] or None
            The validated list (or ``None``) for downstream use.

        Raises
        ------
        ValueError
            If ``modifications`` is not a list of dicts.
        KeyError
            If any dict lacks required keys such as ``"position"`` or ``"ccd"``.
        """
        if modifications is None:
            return None
        if not (isinstance(modifications, list) and all(isinstance(elem, dict) for elem in modifications)):
            raise ValueError(f':modifications: parameter has to be in format [{"position": RES_IDX, "ccd": CCD}, ...]. modifications: {modifications}')
        for mod in modifications:
            if "position" not in mod or "ccd" not in mod:
                raise KeyError(f'One of your modifications is missing a "ccd" or "position" key. :modifications: parameter has to be in format: [{"position": RES_IDX, "ccd": CCD}, ...]. culprit: {mod}')
        return modifications

    def add_protein(self, sequence: str, id: str|list[str], msa: str|bool = False, modifications: list[dict]|str = None, cyclic: bool = False, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Boltz naming convention here, so id overwrite will be ignored in the sake of user experience.
        '''Helper to add protein entry.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; may be a literal or a column name (see Notes).
        id : str or list[str]
            Chain ID(s) to use in the YAML; may be literal or a column name.
        modifications : list[dict] or None, optional
            Per-residue modifications (see :meth:`_check_modifications_format`).
            e.g. [{"position": RES_IDX, "ccd": CCD}, ...] (can also be a string
            pointing to a column in poses.df that contains the modifications dicts)
        cyclic : bool, optional
            Whether the peptide is cyclic.
        poses_cols : list[str], optional
            Keys that should be **read from** ``poses.df`` instead of used literally,
            e.g. ``["sequence", "id", "modifications"]``.

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
        the current pose row when rendering YAML.
        '''
        # instantiate default value
        poses_cols = poses_cols or []

        # compile protein dict in BoltzParams representation.
        protein_dict = {
            "id": id,
            "sequence": sequence,
            "msa": msa,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
            "cyclic": cyclic
        }
        protein_dict = {key: (val, key in poses_cols) for key, val in protein_dict.items()} # wrap in poses_cols flag!

        # add proteins entry to BoltzParams instance.
        self.proteins.append(protein_dict)

    def add_dna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, cyclic: bool = False, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Boltz naming convention here, so id overwrite will be ignored in the sake of user experience.
        """
        Add a DNA entry.

        Parameters
        ----------
        sequence : str
            Nucleotide sequence (literal or column name).
        id : str or list[str]
            Identifier(s) for the DNA entry.
        modifications : list[dict] or None, optional
            Residue-level modifications for DNA.
        cyclic : bool, optional
            Whether the polymer is cyclic.
        poses_cols : list[str], optional
            Keys to interpret as column names in ``poses.df``.

        Returns
        -------
        None
        """
        # instantiate default value
        poses_cols = poses_cols or []

        # compile dna dict in BoltzParams representation
        dna_dict = {
            "id": id,
            "sequence": sequence,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
            "cyclic": cyclic
        }
        dna_dict = {key: (val, key in poses_cols) for key, val in dna_dict.items()} # wrap in poses_cols!

        # add dna entry to BoltzParams instance
        self.dna.append(dna_dict)

    def add_rna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, cyclic: bool = False, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Boltz naming convention here, so id overwrite will be ignored in the sake of user experience.
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
        cyclic : bool, optional
            Whether the polymer is cyclic.
        poses_cols : list[str], optional
            Keys to interpret as column names in ``poses.df``.

        Returns
        -------
        None
        """
        # instantiate default value
        poses_cols = poses_cols or []

        # compile dna dict in BoltzParams representation
        rna_dict = {
            "id": id,
            "sequence": sequence,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
            "cyclic": cyclic
        }
        rna_dict = {key: (val, key in poses_cols) for key, val in rna_dict.items()} # wrap in poses_cols!

        # add rna entry to BoltzParams instance
        self.rna.append(rna_dict)

    def add_ligand(self, ligand: str, id: str|list[str], ligand_type: str = "smiles", poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to Boltz naming convention here, so id overwrite will be ignored in the sake of user experience.
        """
        Add a ligand entry.

        Parameters
        ----------
        ligand : str
            The ligand specification. For ``ligand_type="smiles"``, provide a SMILES;
            for ``"ccd"``, provide an RCSB CCD ID.
        id : str or list[str]
            Ligand ID(s) in the output YAML.
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
        if ligand_type.lower() not in {"smiles", "ccd"}:
            raise ValueError(f"Parameter :ligand_type: can be only one of {{'smiles', 'ccd'}}. ligand_type: {ligand_type}")

        # compile ligand dict in BoltzParams representation
        ligand_dict = {
            "id": (id, "id" in poses_cols),
            ligand_type.lower(): (ligand, "ligand" in poses_cols),
        }

        # add ligands entry to BoltzParams instance
        self.ligands.append(ligand_dict)

    def add_constraint(self, constraint_type: str, poses_cols: list[str] = None, **kwargs) -> None:
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
        if constraint_type.lower() not in {"bond", "pocket", "contact"}:
            raise ValueError(f"Parameter :constraint_type: has to be one of {'bond', 'pocket', 'contact'}, your constraint_type: {constraint_type}")

        # instantiate default value
        poses_cols = poses_cols or []

        # wrap keys for constraints in poses_cols flags:
        processed_kwargs = {key: (val, key in poses_cols) for key, val in kwargs.items()}

        # create dictionary that stores constraints and their kwargs.
        constraint_dict = {constraint_type.lower(): dict(processed_kwargs)}

        # add constraint entry to BoltzParams instance
        self.constraints.append(constraint_dict)

    def add_template(self, template: str, template_type: str, poses_cols: list[str] = None, **kwargs) -> None:
        '''
        Add a structural template.
        In ``**kwargs``, add the parameters of the given template that you want to use. 


        Parameters
        ----------
        template : str
            Path or identifier of the template (literal or column name).
        template_type : {"pdb", "cif"}
            Template format.
        poses_cols : list[str], optional
            Keys (including any in ``kwargs``) to be read from ``poses.df``.
        **kwargs
            Additional template parameters supported by Boltz (e.g., chain
            selection, residue ranges).

        Returns
        -------
        None

        See the original Boltz documentation for details: https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md      
        '''
        if template_type.lower() not in {"cif", "pdb"}:
            raise ValueError(f"Parameter :template_type: can only be one of {{'cif', 'pdb'}}, your template_type: {template_type}")

        # instantiate default value
        poses_cols = poses_cols or []

        # wrap keys for templates in poses_cols flags
        processed_kwargs = {key: (val, key in poses_cols) for key, val in kwargs.items()}

        # create dictionary that stores constraints and their kwargs:
        templates_dict = {template_type.lower(): (template, "template" in poses_cols), **processed_kwargs}
        self.templates.append(templates_dict)

    def add_property(self, property_type: str, poses_cols: list[str] = None, **kwargs) -> None:
        """
        Attach arbitrary key–value properties to the YAML.

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
        >>> BoltzParams.add_property('affinity', binder="binder_chain_col", poses_cols=["binder"])
        >>> BoltzParams.add_property('affinity', binder="B")
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

    def generate_yaml_files(self, poses: Poses, out_dir: str, reset_poses: bool = True) -> None:
        '''Converts poses into new .yaml files at 'prefix' based on current paramters.
        or: render accumulated parameters into per-pose YAML files.

        Resolves all values that were marked as pose-columns against
        ``poses.df`` and writes one YAML per pose into ``out_dir``.
        Optionally updates ``poses.df["poses"]`` to point to the new files.

        Parameters
        ----------
        poses : Poses
            Poses whose table provides column values for pose-bound fields.
        out_dir : str
            Output directory where YAML files are written.
        reset_poses : bool, optional
            If ``True``, replace the ``poses`` column with the new YAML paths.

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
        if not all(fp.endswith(".yaml") for fp in poses.poses_list()):
            raise TypeError("Poses must be in boltz-compatible .yaml format. Use the function 'protflow.tools.boltz.convert_poses_to_boltz_yaml()' for this!")

        # create output dir
        os.makedirs(out_dir, exist_ok=True)

        # operate per-pose
        # add proteins, dna, rna, and ligands to sequences entry:
        new_poses = []
        for pose in poses:
            # read pose yaml
            pose_yaml = boltz_yaml_reader(pose["poses"])
            #print(pose_yaml)

            # add sequences
            for protein_dict in self.proteins:
                _add_key_if_not_there(pose_yaml, "sequences", [])
                pose_yaml["sequences"].append({"protein": _parse_dict_for_pose(pose, protein_dict)})

            for dna_dict in self.dna:
                pose_yaml["sequences"].append({"dna": _parse_dict_for_pose(pose, dna_dict)})

            for rna_dict in self.rna:
                pose_yaml["sequences"].append({"rna": _parse_dict_for_pose(pose, rna_dict)})

            for ligand_dict in self.ligands:
                pose_yaml["sequences"].append({"ligand": _parse_dict_for_pose(pose, ligand_dict)})

            # add constraints (constraints are in different format than proteins/dna/rna/ligand)
            for constraint_dict in self.constraints:
                _add_key_if_not_there(pose_yaml, "constraints", [])
                pose_yaml["constraints"].append({cst_type: _parse_dict_for_pose(pose, cst_dict) for cst_type, cst_dict in constraint_dict.items()})

            # add templates
            for template_dict in self.templates:
                _add_key_if_not_there(pose_yaml, "templates", [])
                pose_yaml["templates"].append(_parse_dict_for_pose(pose, template_dict))

            # add properties
            for property_dict in self.properties:
                _add_key_if_not_there(pose_yaml, "properties", [])
                pose_yaml["properties"].append({property_type: _parse_dict_for_pose(pose, property_args) for property_type, property_args in property_dict.items()})

            # write output
            new_pose_fn = os.path.join(out_dir, os.path.basename(pose["poses"]))
            boltz_yaml_writer(new_pose_fn, pose_yaml)

            # add new filename to new_poses list for integration into poses later
            new_poses.append(new_pose_fn)

        # set new poses
        if reset_poses:
            poses.df["poses"] = new_poses
        logging.info(f"Finished converting poses to .yaml files based on BoltzParams.\nAdded {len(self.proteins)} proteins, {len(self.ligands)} ligands, {len(self.dna)} DNA molecules, and {len(self.rna)} RNA molecules.\nAdded {len(self.constraints)} constraints, {len(self.templates)} templates, and {len(self.properties)} properties.")

def convert_chain_seq_dict_to_yaml_dict(chain_seq_dict: dict[str,str], msa: str = None, ignore_nonexistent_msa_file: bool = False) -> dict[str,str]:
    '''
    Converts dictionary that contains {chain: seq, ...} into boltz-compatible protein entries {}.
    When msa is set to 'server', the function will set <msa: empty> (use option --use_msa_server!)

    Convert a chain→sequence mapping into Boltz YAML "protein" entries.

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
    >>> convert_chain_seq_dict_to_yaml_dict({"A": "ACDE", "B": "FGHI"}, msa="empty")
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

    # create protein yaml for each chain.
    protein_yaml = [
        {
            "id": chain,
            "sequence": seq,
            "msa": msa_val
        }
        for chain, seq in chain_seq_dict.items()
    ]
    return protein_yaml

def _folders_in_dir(dir_path: str) -> list:
    '''finds and returns all folders in :dir_path: that don't start with a . (hidden folders).'''
    dir_path = Path(dir_path)
    # Note: if this causes issues in the future with random folders, add an additional check for the subdirectory
    # to contain at least a file with f'{parent_folder_name}_model_0.{"cif" or "pdb"}'
    return_dirs = [p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")] # exclude hidden folders
    return return_dirs

def _read_boltz_confidence_file(fp: str) -> pd.Series:
    '''Reads boltz confidence output file.'''
    with open(fp, 'r', encoding="UTF-8") as f:
        scores_dict = json.load(f)
    return pd.Series(scores_dict)

def _get_last_dir_name(path: str) -> str:
    '''returns name of last directory in path.'''
    p = Path(path)
    if p.is_dir() or str(path).endswith("/"):
        return p.name
    return p.parent.name

def collect_boltz_scores(boltz_output_dir: str) -> pd.DataFrame:
    """
    Aggregate per-model Boltz outputs into a Pandas DataFrame.

    Expects the Boltz output layout:
    ``{boltz_output_dir}/{input}/predictions/{pose}/`` containing:
    - structure files: ``{pose}_model_*.cif`` or ``.pdb``
    - confidence JSONs: ``confidence_{pose}_model_{i}.json``
    - optional affinity JSON: ``affinity_{pose}.json``
    - NPZ artifacts per model: ``plddt_*``, ``pae_*``, ``pde_*``

    Parameters
    ----------
    boltz_output_dir : str
        Top-level directory passed to Boltz via ``--out_dir``.

    Returns
    -------
    pandas.DataFrame
        One row per model with at least:
        ``description``, ``location``, and paths for
        ``plddt_location``, ``pae_location``, ``pde_location``; plus all JSON keys.

    Notes
    -----
    The ``description`` column is ``{pose}_model_{rank}`` and ``location`` points
    to the corresponding ``.pdb/.cif`` model file. :contentReference[oaicite:3]{index=3}
    """
    # create list of output files
    out_fl = _folders_in_dir(boltz_output_dir)
    out_fl = [os.path.join(out_f, "predictions") for out_f in out_fl]

    # create output aggregation list
    out_l = []

    # loop over output folders {input_dir/input_file}/{predictions}/{input_file}/{diffusion_samples} -> multiple input files and multiple diffusion samples
    for out_f in out_fl:
        for input_file in _folders_in_dir(out_f): # input_file should be: /path/to/boltz_output_dir/{boltz_input}/predictions/{input_file}/
            # basename of pose
            description = _get_last_dir_name(input_file)

            # loop over output models
            output_models = glob(f"{input_file}/{description}_model_*.cif") + glob(f"{input_file}/{description}_model_*.pdb")
            for pose_fp in output_models:
                # determine rank
                rank = int(os.path.splitext(os.path.basename(pose_fp))[0].rsplit("_", maxsplit=1)[-1]) # takes the rank (1) from /path/to/confidence_{pose_description}_model_1.json

                ## collect scores
                pose_confidence_file = f"{input_file}/confidence_{description}_model_{rank}.json"
                confidence_scores = _read_boltz_confidence_file(pose_confidence_file)

                # parse description
                confidence_scores["description"] = f"{description}_model_{rank}"
                confidence_scores["location"] = pose_fp

                # read affinity scores
                affinity_fn = f"{input_file}/affinity_{description}.json"
                if os.path.isfile(affinity_fn):
                    affinity_scores = _read_boltz_confidence_file(affinity_fn)
                    confidence_scores = pd.concat([confidence_scores, affinity_scores])

                # add .npz file-locations into the scores
                npz_file_headers = ["plddt", "pae", "pde"]
                npz_files = [f"{input_file}/{header}_{description}_model_{rank}.npz" for header in npz_file_headers]
                for npz_file, header in zip(npz_files, npz_file_headers):
                    confidence_scores[f"{header}_location"] = npz_file

                # append model scores to global output list:
                out_l.append(confidence_scores)

    # aggregate scores in DataFrame
    scores = pd.DataFrame(out_l)
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

# --- flow-style helper for specific sequences ---
class FlowSeq(list):
    """
    Marker list that forces YAML *flow style*.

    When dumped with :class:`MyDumper`, lists of this type are emitted as
    ``[a, b, c]`` on one line rather than block style. Used to keep compact
    representations for IDs and token tuples in Boltz YAMLs. :contentReference[oaicite:4]{index=4}
    """
    pass # pylint: disable=W0107

def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class MyDumper(yaml.SafeDumper):
    """YAML dumper enabling flow-style emission for :class:`FlowSeq`."""
    pass # pylint: disable=W0107

# write class that autodetects lists in yaml and converts them into flow stuff
def _process_boltz_yaml_for_output(boltz_yaml: dict) -> dict:
    '''This is now a manually done function which is annoying. Try to convert this with patterns later.'''
    # fix id entries in the same line, e.g.: 'id: [A, B]'
    for sequence_entry in boltz_yaml.get("sequences", []):
        (_, entity_dict), = sequence_entry.items()
        if "id" in entity_dict and isinstance(entity_dict["id"], list):
            entity_dict["id"] = FlowSeq(entity_dict["id"])

    # same for constraint entries
    for constraint_entry in boltz_yaml.get("constraints", []):
        (constraint_type, constraint_dict), = constraint_entry.items()
        if constraint_type == "bond":
            constraint_dict["atom1"] = FlowSeq(constraint_dict["atom1"])
            constraint_dict["atom2"] = FlowSeq(constraint_dict["atom2"])
        if constraint_type == "pocket":
            constraint_dict["contacts"] = FlowSeq(constraint_dict["contacts"])
        if constraint_type == "contact":
            constraint_dict["token1"] = FlowSeq(constraint_dict["token1"])
            constraint_dict["token2"] = FlowSeq(constraint_dict["token2"])

    # same for template entries (specifying ID's usually happens in lists if multiple IDs are specified)
    for template_entry in boltz_yaml.get("constraints", []):
        for template_key in template_entry:
            if isinstance(template_entry[template_key], list):
                template_entry[template_key] = FlowSeq(template_entry[template_key])

    return boltz_yaml

def boltz_yaml_writer(out_path: str, boltz_yaml: dict) -> None:
    """
    Write a Boltz YAML document to disk (pretty, stable layout).

    Parameters
    ----------
    out_path : str
        Output ``.yaml`` path.
    boltz_yaml : dict
        YAML document to write (will be processed for flow-style lists).

    Returns
    -------
    None
    """
    MyDumper.add_representer(FlowSeq, _flow_seq_representer)
    processed_yaml = _process_boltz_yaml_for_output(boltz_yaml)
    with open(out_path, 'w', encoding="UTF-8") as f:
        yaml.dump(
            processed_yaml, f, Dumper=MyDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2, width=10**9,
            allow_unicode=True
        )

def boltz_yaml_reader(in_path: str) -> dict:
    """
    Read a Boltz YAML file into a Python dictionary.

    Parameters
    ----------
    in_path : str
        Path to a ``.yaml`` file.

    Returns
    -------
    dict
        Parsed YAML document.
    """
    with open(in_path, 'r', encoding="UTF-8") as f:
        out_dict = yaml.safe_load(f)
    return out_dict
