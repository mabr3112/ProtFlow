"""Run local ESMFold2 predictions through ProtFlow.

FASTA and PDB/CIF inputs are converted to JSON sequence descriptions; existing
JSON inputs are passed through. ESMFold2Params can add molecular entities and
constraints. Global and per-pose option dictionaries are forwarded to
ESMFold2InputBuilder.fold(), and predictions and confidence scores are merged
back into Poses. See the ESMFold2 tool tutorial for complete examples."""
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
    """Prepare ESMFold2 inputs, dispatch jobs, and collect predictions.

    Parameters
    ----------
    esmfold2_python : str, optional
        Python interpreter in the ESMFold2 environment. Defaults to ESMFOLD2_PYTHON.
    pre_cmd : str, optional
        Shell setup command. Defaults to ESMFOLD2_PRE_CMD.
    jobstarter : JobStarter, optional
        Default jobstarter. Each job loads its own model on a GPU.

    Notes
    -----
    The worker is run_esmfold2.py in AUXILIARY_RUNNER_SCRIPTS_DIR. Each output
    adds one index layer, for example target_0001. Inputs are batched both with
    and without per-pose options. Cached scores are reused unless overwrite=True."""
    def __init__(self, esmfold2_python: str = None, pre_cmd: str = None, jobstarter: JobStarter = None):
        """Resolve the worker, interpreter, and optional shell setup."""
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
        """Write batch JSON files with independent option dictionaries per pose.

        Per-pose entries override matching global keys without changing either input."""

        input_dir = os.path.join(work_dir, "input_jsons")
        os.makedirs(input_dir, exist_ok=True)
        if options is None:
            options = {}

        if not isinstance(options, dict):
            raise KeyError(":options: must be of type dict!")

        if pose_options:
            col_in_df(poses.df, pose_options)
            pose_options = poses.df[pose_options].to_list()
            if not all(isinstance(opt, dict) for opt in pose_options):
                raise KeyError(":pose_options: must be of type dict!")

            pose_options = [{**options, **pose_opt} for pose_opt in pose_options]

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
        """Build worker commands and prepend the configured shell setup."""
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
        """Predict structures and merge output paths and scores into poses.

        Parameters
        ----------
        poses : Poses
            FASTA, PDB/CIF, or ESMFold2 JSON inputs. Structure conversion extracts
            protein sequences, not coordinates or ligands.
        prefix : str
            Run name beneath poses.work_dir and prefix for result columns.
        jobstarter : JobStarter, optional
            Override the runner's default jobstarter.
        options : dict, optional
            Keyword arguments for ESMFold2InputBuilder.fold(), such as num_loops,
            num_sampling_steps, and num_diffusion_samples. CLI strings are not accepted.
        pose_options : str, optional
            Dataframe column containing one option dictionary per pose. These values
            override matching global options for that pose only.
        model : str, optional
            Hugging Face model ID or local checkpoint directory. Default: biohub/ESMFold2.
        params : ESMFold2Params, optional
            Molecular entities and constraints to append to the input JSON files.
        overwrite : bool, optional
            Recompute instead of reusing the cached score table. For changed input sets
            or batching, use a fresh prefix to avoid collecting stale output files.

        Returns
        -------
        Poses
            Predictions with prefixed location, description, mean_per_res_plddt,
            per_res_plddt, ptm, and iptm columns when supplied by the model."""
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # sanitize
        options = {} if options is None else options

        # check for output
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if os.path.isfile(scorefile) and not overwrite:
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
    """Write protein sequence JSON files from FASTA or PDB/CIF poses.

    FASTA chains are separated by ':' or '/'. Structure inputs supply protein
    sequences only. When reset_poses is True, update poses.df['poses'] to the JSON
    paths. Existing files are reused only when overwrite is False."""
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
    Builder for per-pose ESMFold2 JSON content.

    Collects entries for proteins, nucleic acids, ligands, and constraints. Each field value can be provided
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
        ``proteins``, ``dna``, ``rna``, ``ligands``, and ``constraints`` are reflected
        into the resulting JSON during :meth:`generate_json_files`.
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

    def add_protein(self, sequence: str, id: str|list[str], msa: None = None, modifications: list[dict]|str = None, poses_cols: list[str] = None) -> None: # pylint: disable=W0622 ## we adhere to ESMFold2 naming convention here, so id overwrite will be ignored in the sake of user experience.
        '''Helper to add protein entry.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence; may be a literal or a column name (see Notes).
        id : str or list[str]
            Chain ID(s) to use in the JSON; may be literal or a column name.
        msa : None, optional
            Leave unset for sequence-only prediction.
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
        the current pose row when rendering JSON.
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
        if ligand_type.lower() not in {"smiles", "ccd"}:
            raise ValueError(f"Parameter :ligand_type: can be only one of {{'smiles', 'ccd'}}. ligand_type: {ligand_type}")

        # compile ligand dict in ESMFold2Params representation
        ligand_dict = {
            "id": (id, "id" in poses_cols),
            ligand_type.lower(): (ligand, "ligand" in poses_cols),
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
        ...     chain_id1="A", res_idx1=42, atom_idx1=0, chain_id2="B", res_idx2=13, atom_idx2=0,
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
                entry = _parse_dict_for_pose(pose, ligand_dict)
                if "ccd" in entry:
                    entry["ccd"] = [entry["ccd"]]
                pose_dict["sequences"].append({"ligand": entry})

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
    """Convert a chain-to-sequence dictionary into protein entries with no MSA."""
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
    """Combine worker score tables from work_dir/output/*.json.

    Each row contains a description, CIF location, and available pLDDT, pTM, and
    ipTM values. The worker writes one table per input batch."""
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
