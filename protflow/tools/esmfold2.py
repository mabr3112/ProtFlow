"""
ProtFlow runner for ESMFold2.

This module provides a high-level `ESMFold2` runner that:
(1) prepares ESMFold2-compatible YAML inputs from sequences or structures,
(2) composes command lines from global and pose-specific options,
(3) distributes inference across available cores via a `JobStarter`,
and (4) aggregates ESMFold2 outputs (confidence, affinity, NPZ artifacts) into a
single score table for downstream orchestration.

The typical workflow is:

1. Ensure paths and environment hooks for ESMFold2 are configured
   (see Notes on `ESMFOLD2_PATH`, `ESMFOLD2_PYTHON`, `ESMFOLD2_PRE_CMD`).
2. Provide inputs as a `Poses` collection (FASTA, PDB/CIF, or already
   ESMFold2-formatted YAML). If needed, convert to YAML with
   `convert_poses_to_esmfold2_json`.
3. Call `ESMFold2.run(...)` with command-line `options` and optional
   `pose_options` to fan-out runs.
4. Consume the returned `Poses` object whose `.df` is augmented with a
   per-model score table and file locations of produced artifacts.

Notes
-----
- Configuration keys
  The runner reads its defaults from ProtFlow’s config via:
  `ESMFOLD2_PATH` (path to the `esmfold2` CLI entry point or module),
  `ESMFOLD2_PYTHON` (interpreter used to invoke ESMFold2), and
  `ESMFOLD2_PRE_CMD` (shell prefix such as environment activation).
  Use `protflow.config` utilities to set these once per environment.
- MSA handling
  ESMFold2 can run with an empty MSA or fetch MSAs from a server. The runner
  exposes `msa_setting` to steer YAML content (`"empty"` vs `"server"`),
  while the CLI switch `--use_msa_server` remains the source of truth for
  server fetching. See `ESMFold2._parse_msa_setting` and
  `convert_chain_seq_dict_to_json_dict`.

Examples
--------
Run ESMFold2 on a batch of structures, writing outputs to a fresh work directory
and collecting scores:

>>> from protflow.runners.esmfold2 import ESMFold2
>>> from protflow.poses import Poses
>>> poses = Poses(
...     files=["A.pdb", "B.pdb", "C.pdb"],
...     work_dir="work/esmfold2_demo"
... )
>>> runner = ESMFold2()  # uses config defaults (ESMFOLD2_PATH/PYTHON/PRE_CMD)
>>> poses = runner.run(
...     poses=poses,
...     prefix="esmfold2_run",
...     options="--num_samples 4 --use_msa_server",
...     overwrite=False,
... )
>>> poses.df.columns[:8]  # score columns will include confidence & file paths
...

"""
# generals
import os
import json
import logging
from glob import glob

# dependencies
import pandas as pd

# custom
from ..poses import Poses, get_format, col_in_df
from .. import load_config_path, require_config
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..utils.biopython_tools import load_sequence_from_fasta, get_sequence_from_pose, biopython_load_structure

class ESMFold2(Runner):
    """
    The ESMFold2 runner prepares inputs (optionally batching by core), assembles ESMFold2 commands,
    dispatches them via a `JobStarter`, and aggregates results into a unified
    score file stored in the run directory.

    Parameters
    ----------
    esmfold2_path : str, optional
        Executable or module path used with `predict` subcommand.
        If not provided, loaded from `ESMFOLD2_PATH` in the ProtFlow config.
    esmfold2_python : str, optional
        Python interpreter used to execute ESMFold2. Defaults to `ESMFOLD2_PYTHON`
        from the ProtFlow config.
    pre_cmd : str, optional
        Shell prefix prepended to each command. Use this to activate
        environments or modules (e.g., `conda activate esmfold2`). If omitted,
        taken from `ESMFOLD2_PRE_CMD` in the ProtFlow config.
    jobstarter : JobStarter, optional
        Default jobstarter to use if none is provided to `run()`.

    Attributes
    ----------
    name : str
        Fixed runner name: `"ESMFold2"`.
    index_layers : int
        Number of index layers used when merging outputs (defaults to 2).
    jobstarter : JobStarter or None
        Optional default jobstarter stored on the runner instance.
    esmfold2_path : str
        Resolved ESMFold2 executable/module path.
    esmfold2_python : str
        Resolved interpreter path.
    pre_cmd : str
        Resolved shell prefix (may be empty).

    Notes
    -----
    - Score caching
      If a score file already exists for the given `prefix` and format and
      `overwrite` is `False` (and `--override` not present in `options`),
      existing results are returned without re-running ESMFold2.
    - Batching behavior
      If `pose_options` are *not* provided, inputs are automatically split
      into at most `jobstarter.max_cores` batches to improve throughput.

    Examples
    --------
    Minimal run with default configuration, batched across cores:

    >>> runner = ESMFold2()
    >>> poses = runner.run(
    ...     poses, prefix="demo",
    ...     options="--num_samples 2 --use_msa_server"
    ... )
    """
    def __init__(self, esmfold2_python: str = None, pre_cmd: str = None, jobstarter: JobStarter = None):
        """
        Initialize the ESMFold2 runner and resolve configuration.

        Parameters
        ----------
        esmfold2_path : str, optional
            Path to the ESMFold2 program or module (with `predict` subcommand).
            Defaults to `ESMFOLD2_PATH` from ProtFlow config.
        esmfold2_python : str, optional
            Interpreter to call ESMFold2 with. Defaults to `ESMFOLD2_PYTHON`.
        pre_cmd : str, optional
            Shell prefix (e.g., environment activation). Defaults to
            `ESMFOLD2_PRE_CMD`.
        jobstarter : JobStarter, optional
            Default jobstarter to use when `run(jobstarter=None)`.

        Raises
        ------
        KeyError
            If required configuration keys are missing from the ProtFlow config.
        """
        config = require_config()
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        self.esmfold2_path = os.path.join(self.script_dir, "run_esmfold2.py")
        self.esmfold2_python = esmfold2_python or load_config_path(config, "ESMFOLD2_PYTHON")
        self.pre_cmd = pre_cmd or load_config_path(config, "ESMFOLD2_PRE_CMD", is_pre_cmd=True)

        self.name = "ESMFold2"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        """
        String representation.

        Returns
        -------
        str
            The literal string ``"ESMFold2"``.
        """
        return "ESMFold2"

    def _parse_poses(self, poses: Poses, work_dir: str, max_cores: int, options: dict = None, pose_options: str = None) -> list[str]:
        '''helper function to parse poses for batch processing.

        Determine ESMFold2 input units (per pose vs. per batch subfolder).

        When `pose_options` are provided, ESMFold2 consumes each pose file directly.
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

        input_dir = os.path.join(work_dir, "input_jsons")
        os.makedirs(input_dir, exist_ok=True)
        if not options:
            options = {}
        
        if not isinstance(options, dict):
            raise KeyError(":options: must be of type dict!")
        
        if pose_options:
            col_in_df(poses.df, pose_options)
            pose_options = poses.df[pose_options].to_list()
            if not all(isinstance(opt, dict) for opt in pose_options):
                raise KeyError(":pose_options: must be of type dict!")
            
            pose_options = [options.update(pose_opt) for pose_opt in pose_options]
        
        else:
            pose_options = [options for _ in poses]

        poses_opts_list = [{pose["poses_description"]: {"pose_path": pose["poses"], "options": pose_opts}} for pose, pose_opts in zip(poses, pose_options)]

        poses_sublists = split_list(poses_opts_list, n_sublists=max_cores)

        in_jsons = []
        for i, sublist in enumerate(poses_sublists):
            subdict = {k: v for d in sublist for k, v in d.items()} # create a single input dict for each batch
            in_jsons.append(esmfold2_json_dumper(subdict, os.path.join(input_dir, f"batch_{i}.json")))

        return in_jsons


    def _write_cmds(self, esmfold2_inputs: list[str], model: str, output_dir: str) -> list[str]:
        '''
        Compose ESMFold2 command strings from resolved inputs and options.

        Each command is of the form:

        ``{pre_cmd} {esmfold2_python} {esmfold2_path} predict {input} {options}``

        Parameters
        ----------
        esmfold2_inputs : list of str
            Per-command input path (individual YAML or batch directory).
        parsed_options : list of str
            Per-command options string as produced by `_parse_options`.

        Returns
        -------
        list of str
            Shell commands ready to be dispatched via `JobStarter.start()`.
        '''
        cmd_list = [
            f"{self.esmfold2_python} {self.esmfold2_path} --input_json {input_fn} --model {model} --output_dir {output_dir}".strip()
            for input_fn in esmfold2_inputs
        ]
        if self.pre_cmd:
            cmd_list = prepend_cmd(cmds=cmd_list, pre_cmd=self.pre_cmd)

        return cmd_list

    def run(
            self, poses: Poses, prefix: str, jobstarter: JobStarter = None,
            options: dict = None, pose_options: str = None, model: str = "biohub/ESMFold2", params: "ESMFold2Params" = None, overwrite: bool = False) -> Poses:
        '''
        Execute ESMFold2 on the given `poses` and collect results.

        The runner prepares inputs (converting to ESMFold2 YAML if needed),
        resolves MSA behavior, optionally augments pose YAMLs using a provided
        `ESMFold2Params` object, dispatches the commands via `JobStarter`, then
        aggregates prediction confidence/affinity scores and artifact paths
        into a DataFrame saved as ``{prefix}/{name}_scores.{storage_format}``.

        Parameters
        ----------
        poses : Poses
            Input poses. Has to be protflow.poses.Poses class with poses in FASTA, 
            PDB/CIF, or ESMFold2 YAML; if not YAML, they are converted 
            with `convert_poses_to_esmfold2_json`.
        prefix : str
            Run prefix / subdirectory under `poses.work_dir`. 
            ESMFold2 outputs will be stored in {poses.work_dir}/{prefix}/output
        jobstarter : JobStarter, optional
            Overrides the runner’s default jobstarter. If omitted, the runner
            tries, in order: the provided value, the instance default, and
            `poses.default_jobstarter`.
        options : dict, optional
            Global options for the ESMFold2InputBuilder().fold() function (e.g., ``"num_loops``,
            ``"num_sampling_steps"``). Must be provided in dict-format (see Notes).
        pose_options : str, optional
            Pose-specific option dict extracted from a poses.df column (see Notes).
        params : ESMFold2Params, optional
            If given, used to *modify* or *extend* per-pose YAMLs (e.g.,
            sequences, ligands, constraints, templates, properties) before
            running. Files are emitted under ``{prefix}/esmfold2_inputs/``.
        overwrite : bool, optional
            If `True` (or if `--override` is present in `options`), re-run
            even if a scorefile already exists.

        Returns
        -------
        Poses
            The original `Poses` with results merged and indices layered.
            Artifacts (models, NPZs) are recorded as path columns.

        Raises
        ------
        RuntimeError
            If ESMFold2 finishes without producing any scores.
        TypeError
            If inputs cannot be converted to ESMFold2 YAML (unsupported formats).

        Examples
        --------
        Convert PDBs to YAML, add a ligand, and run with 4 samples per pose:

        >>> from protflow.runners.esmfold2 import ESMFold2
        >>> from protflow.runners.esmfold2 import ESMFold2Params
        >>> params = ESMFold2Params()
        >>> params.add_ligand(ligand="CC(=O)O", id="LIG", ligand_type="smiles")
        >>> runner = ESMFold2()
        >>> poses = runner.run(
        ...     poses=poses,
        ...     prefix="esmfold2_with_ligand",
        ...     params=params,
        ...     options="--num_samples 4",
        ...     overwrite=True
        ... )

        Notes
        -----
        - Score caching: if a prior score file exists and neither `overwrite`
          nor `--override` is set, the runner returns cached results to save
          time.
        - Options and pose_options: settings `pose_options` overwrite `options`. Format:
          `{'num_loops': 3, 'num_sampling_steps': 200}`
        '''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # sanitize
        options = options or ""

        # check for output
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if os.path.isfile(scorefile) and not (overwrite or "--override" in options):
            scores = get_format(scorefile)(scorefile) # loads scorefile DF with correct loading function
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # check if poses are in correct format (json) (unless bypass_poses_check)
        if not all(fp.endswith(".json") for fp in poses.poses_list()):
            convert_poses_to_esmfold2_json(poses, prefix=f"{prefix}/poses_json")

        # if ESMFold2Params are given, use ESMFold2Params to generate new poses based on params
        if params:
            esmfold2_input_dir = os.path.join(work_dir, "esmfold2_inputs")
            params.generate_json_files(poses, esmfold2_input_dir)

        # batch poses
        esmfold2_inputs = self._parse_poses(
            poses=poses,
            options=options,
            pose_options=pose_options,
            work_dir=work_dir,
            max_cores=jobstarter.max_cores
        )

        # compile commands
        os.makedirs(output_dir := os.path.join(work_dir, "output"), exist_ok=True)
        cmds = self._write_cmds(esmfold2_inputs, model, output_dir)

        # run esmfold2
        jobstarter.start(
            cmds = cmds,
            jobname = self.name,
            output_path = work_dir
        )

        # collect scores
        scores = collect_esmfold2_scores(work_dir)

        # output safety
        if len(scores) == 0:
            raise RuntimeError(f"ESMFold2 crashed. Check output logs and output directory for error logs: {work_dir}")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # return outputs
        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

def convert_poses_to_esmfold2_json(poses: Poses, prefix: str, overwrite: bool = True, reset_poses: bool = True) -> None:
    """For now, this only reads the protein sequence, not anything else (no ligand support).

    Convert input poses to ESMFold2-compatible YAMLs.

    Creates one YAML per pose under ``{poses.work_dir}/{prefix}``, encoding chain
    sequences (and MSA choice) for ESMFold2. Optionally updates ``poses.df["poses"]``
    to point to the newly created YAMLs.

    Parameters
    ----------
    poses : Poses
        Input poses (protflow.poses.Poses class); poses must be in FASTA/PDB/CIF format poses table.
    prefix : str
        Subdirectory name under ``poses.work_dir`` where YAMLs are written.
    msa : str or None
        One of ``"server"``, ``"empty"``, or a path to a custom ``.a3m`` file.
        ``"server"`` writes empty MSA entries and expects ESMFold2 to fetch MSAs.
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
    >>> convert_poses_to_esmfold2_json(poses, prefix="esmfold2_inputs", msa="empty")
    >>> convert_poses_to_esmfold2_json(poses, prefix="esmfold2_inputs_srv", msa="server", reset_poses=False)

    Notes
    -----
    - The function is sequence-centric (ligands/templates/properties are handled later via :class:`ESMFold2Params`).
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
        os.path.join(out_dir, os.path.splitext(os.path.basename(pose))[0] + ".json") # replaces file-extension with .json
        for pose in poses.poses_list()
    ] # create new output names

    if all(os.path.isfile(out_fn) for out_fn in out_fn_list) and not overwrite:
        logging.info(f"ESMFold2 json files exist at {out_dir}. Skipping creation to save time.")

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
        raise TypeError("ESMFold2 only supports files in .pdb, .cif, or .fa format!")

    # now convert pose-level lists to valid esmfold2 jsons. [{chain: seq, ...}, ...] -> [esmfold2-json-formatted-pose, ...]
    pose_dicts_raw = [convert_chain_seq_dict_to_pose_dict(pose_dict) for pose_dict in sequence_dict_list]

    # now create esmfold2 pose_jsons
    pose_dicts = [
        {"sequences": [{"protein": chain_dict} for chain_dict in pose_json]}
        for pose_json in pose_dicts_raw
    ]
    # store jsons
    for pose_dict, out_fn in zip(pose_dicts, out_fn_list):
        esmfold2_json_dumper(pose_dict, out_fn)

    # set new poses
    if reset_poses:
        poses.df["poses"] = out_fn_list
    return None

class ESMFold2Params:
    """
    Builder for per-pose ESMFold2 YAML content.

    Collects entries for proteins, nucleic acids, ligands, constraints,
    templates, and arbitrary properties. Each field value can be provided
    either as a *literal* or as a reference to a column in ``poses.df``.
    Column-referenced values are marked by passing their keys via
    ``poses_cols`` and are resolved at YAML generation time.

    Notes
    -----
    - Each added entity is stored internally and later rendered into
      the final YAML structure via :meth:`generate_json_files`.
    - For sequence modifications, use a list of dicts with at least
      ``{"position": <int>, "ccd": <str>}``.
    """
    def __init__(self):
        """
        Initialize an empty parameter collection.

        The instance accumulates lists:
        ``proteins``, ``dna``, ``rna``, ``ligands``, ``constraints``,
        ``templates``, and ``properties``—all of which are reflected
        into the resulting YAML during :meth:`generate_json_files`.
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

    def add_protein(self, sequence: str, id: str|list[str], msa: str|bool = False, modifications: list[dict]|str = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to ESMFold2 naming convention here, so id overwrite will be ignored in the sake of user experience.
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

        # compile protein dict in ESMFold2Params representation.
        protein_dict = {
            "id": id,
            "sequence": sequence,
            "msa": msa,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
        }
        protein_dict = {key: (val, key in poses_cols) for key, val in protein_dict.items()} # wrap in poses_cols flag!

        # add proteins entry to ESMFold2Params instance.
        self.proteins.append(protein_dict)

    def add_dna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to ESMFold2 naming convention here, so id overwrite will be ignored in the sake of user experience.
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

        # compile dna dict in ESMFold2Params representation
        dna_dict = {
            "id": id,
            "sequence": sequence,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
        }
        dna_dict = {key: (val, key in poses_cols) for key, val in dna_dict.items()} # wrap in poses_cols!

        # add dna entry to ESMFold2Params instance
        self.dna.append(dna_dict)

    def add_rna(self, sequence: str, id: str|list[str], modifications: list[dict] = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to ESMFold2 naming convention here, so id overwrite will be ignored in the sake of user experience.
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

        # compile dna dict in ESMFold2Params representation
        rna_dict = {
            "id": id,
            "sequence": sequence,
            "modifications": modifications if "modifications" in poses_cols else self._check_modifications_format(modifications),
        }
        rna_dict = {key: (val, key in poses_cols) for key, val in rna_dict.items()} # wrap in poses_cols!

        # add rna entry to ESMFold2Params instance
        self.rna.append(rna_dict)

    def add_ligand(self, ligand: str, id: str|list[str], ligand_type: str = "smiles", poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to ESMFold2 naming convention here, so id overwrite will be ignored in the sake of user experience.
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

        # compile ligand dict in ESMFold2Params representation
        ligand_dict = {
            "id": (id, "id" in poses_cols),
            ligand_type.lower(): ([ligand], "ligand" in poses_cols),
        }

        # add ligands entry to ESMFold2Params instance
        self.ligands.append(ligand_dict)

    def add_constraint(self, constraint_type: str, poses_cols: list[str] = None, **kwargs) -> None:
        """
        Add a geometric or pocket constraint.

        Parameters
        ----------
        constraint_type : str
            One of typical types such as ``"DistogramConditioning"``, ``"PocketConditioning"`` or ``"CovalentBond"``,
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
        ...     "CovalentBond",
        ...     chain_id1="A", res_idx1=42, atom_idx1="512, chain_id2="B", res_idx1=13, atom_idx1=102,
        ... )

        Notes
        -----
        - ``DistogramConditioning`` expects ``chain_id`` (str) and ``distogram`` (np.ndarray)
        - ``PocketConditioning`` expects a ``binder_chain_id`` (chain) and a list of
          pocket ``contacts`` (list[tuple[str, int]]).
        - ``CovalentBond`` expects ``chain_id1`` (str), ``res_idx1`` (int), ``atom_idx1`` (int), 
          ``chain_id2`` (str), ``res_idx2`` (int), ``atom_idx2`` (int).
        """
        csts = {"DistogramConditioning", "PocketConditioning", "CovalentBond"}
        if constraint_type.lower() not in {i.lower() for i in csts}:
            raise ValueError(f"Parameter :constraint_type: has to be one of {csts}, your constraint_type: {constraint_type}")

        # instantiate default value
        poses_cols = poses_cols or []

        # wrap keys for constraints in poses_cols flags:
        processed_kwargs = {key: (val, key in poses_cols) for key, val in kwargs.items()}

        # create dictionary that stores constraints and their kwargs.
        constraint_dict = {constraint_type.lower(): dict(processed_kwargs)}

        # add constraint entry to ESMFold2Params instance
        self.constraints.append(constraint_dict)

    def generate_json_files(self, poses: Poses, out_dir: str, reset_poses: bool = True) -> None:
        '''Converts poses into new .json files at 'prefix' based on current paramters.
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

        # sanity
        if not all(fp.endswith(".json") for fp in poses.poses_list()):
            raise TypeError("Poses must be in esmfold2-compatible .json format. Use the function 'protflow.tools.esmfold2.convert_poses_to_esmfold2_json()' for this!")

        # create output dir
        os.makedirs(out_dir, exist_ok=True)

        # operate per-pose
        # add proteins, dna, rna, and ligands to sequences entry:
        new_poses = []
        for pose in poses:
            pose_dict = esmfold2_json_reader(pose["poses"])

            # add sequences
            for protein_dict in self.proteins:
                pose_dict.setdefault("sequences", [])
                pose_dict["sequences"].append({"protein": _parse_dict_for_pose(pose, protein_dict)})

            for dna_dict in self.dna:
                pose_dict["sequences"].append({"dna": _parse_dict_for_pose(pose, dna_dict)})

            for rna_dict in self.rna:
                pose_dict["sequences"].append({"rna": _parse_dict_for_pose(pose, rna_dict)})

            for ligand_dict in self.ligands:
                pose_dict["sequences"].append({"ligand": _parse_dict_for_pose(pose, ligand_dict)})

            # add constraints (constraints are in different format than proteins/dna/rna/ligand)
            for constraint_dict in self.constraints:
                pose_dict.setdefault("constraints", [])
                pose_dict["constraints"].append({cst_type: _parse_dict_for_pose(pose, cst_dict) for cst_type, cst_dict in constraint_dict.items()})

            # write output
            new_pose_fn = os.path.join(out_dir, os.path.basename(pose["poses"]))
            
            # add new filename to new_poses list for integration into poses later
            new_poses.append(esmfold2_json_dumper(pose_dict, new_pose_fn))

        # set new poses
        if reset_poses:
            poses.df["poses"] = new_poses
        logging.info(f"Finished converting poses to .json files based on ESMFold2Params.\nAdded {len(self.proteins)} proteins, {len(self.ligands)} ligands, {len(self.dna)} DNA molecules, and {len(self.rna)} RNA molecules.\nAdded {len(self.constraints)} constraints, {len(self.templates)} templates, and {len(self.properties)} properties.")

def convert_chain_seq_dict_to_pose_dict(chain_seq_dict: dict[str,str]) -> dict[str,str]:
    '''
    Converts dictionary that contains {chain: seq, ...} into esmfold2-compatible protein entries {}.
    When msa is set to 'server', the function will set <msa: empty> (use option --use_msa_server!)

    Convert a chain→sequence mapping into ESMFold2 YAML "protein" entries.

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
    # create protein json for each chain.
    protein_json = [
        {
            "id": chain,
            "sequence": seq,
            "msa": None
        }
        for chain, seq in chain_seq_dict.items()
    ]
    return protein_json

def collect_esmfold2_scores(work_dir: str) -> pd.DataFrame:
    """
    Aggregate per-model ESMFold2 outputs into a Pandas DataFrame.

    Expects the ESMFold2 output layout:
    ``{esmfold2_output_dir}/{input}/predictions/{pose}/`` containing:
    - structure files: ``{pose}_model_*.cif`` or ``.pdb``
    - confidence JSONs: ``confidence_{pose}_model_{i}.json``
    - optional affinity JSON: ``affinity_{pose}.json``
    - NPZ artifacts per model: ``plddt_*``, ``pae_*``, ``pde_*``

    Parameters
    ----------
    esmfold2_output_dir : str
        Top-level directory passed to ESMFold2 via ``--out_dir``.

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
    out_jsons = glob(os.path.join(work_dir, "output", "*.json"))
    scores = pd.concat([pd.read_json(out_json) for out_json in out_jsons]).reset_index(drop=True)
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

def esmfold2_json_reader(in_path: str) -> dict:
    """Read a json file"""
    with open(in_path, "r", encoding="UTF-8") as p:
        return json.load(p)
    
def esmfold2_json_dumper(data: dict, out_path: str) -> str:
    """Dump a dict to json file"""
    with open(out_path, "w", encoding="UTF-8") as f:
        json.dump(data, f, indent=2)
    return out_path
    