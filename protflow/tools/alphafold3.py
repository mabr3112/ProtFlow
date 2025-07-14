"""
AlphaFold3 Runner Module
========================

This module provides the `AlphaFold3` class for running protein structure predictions using AlphaFold3
within the ProtFlow framework. It facilitates the orchestration of input preparation, command-line
execution, job management, and result collection for AlphaFold3, providing a streamlined interface
for high-throughput or customized modeling workflows.

Overview
--------
AlphaFold3 is a deep learning-based protein structure prediction model. This module wraps its
inference process in a Python interface that works with the ProtFlow ecosystem. It supports various
features such as multi-model output, flexible JSON-based input configuration, pose-specific options,
and automated result parsing.

The primary entry point is the `AlphaFold3` class, which exposes several methods:
    - `run()`: Executes AlphaFold3 predictions for a set of poses.
    - `write_cmd()`: Constructs the shell command to launch the AlphaFold3 inference.
    - `collect_results()`: Parses the output directory and collects scores and paths into a DataFrame.

Typical usage involves preparing a `Poses` object, configuring input parameters, and calling
`AlphaFold3.run()`, optionally controlling execution with a `JobStarter` object.

Dependencies
------------
- ProtFlow (for `Poses`, `JobStarter`, and general framework integration)
- AlphaFold3 (inference script must be available and executable)
- pandas
- json
- os, subprocess, shlex (for system-level command execution)

Example
-------
.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from alphafold3_runner import AlphaFold3

    # Load poses and create an AlphaFold3 instance
    poses = Poses("my_poses.json", work_dir="my_work_dir")
    af3 = AlphaFold3()

    # Run AlphaFold3 predictions
    af3.run(
        poses=poses,
        prefix="af3_batch_01",
        jobstarter=LocalJobStarter(gpus=1),
    )

    print(poses.df)

Further Details
---------------
This module is intended to integrate seamlessly with other ProtFlow components. It abstracts away
command-line complexity while preserving fine-grained control via user-supplied options and
custom JSON configuration.

AlphaFold3 itself is highly configurable via JSON inputs. This module assumes the user is familiar
with AlphaFold3’s requirements (e.g., templates, modifications, MSA inputs) and provides convenient
hooks to supply such input through columns in the `Poses` DataFrame or via dictionaries.

Authors
-------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""
# general imports
import os
import logging
from glob import glob
import shutil
import json
import random
import string
from typing import Union, Any

# dependencies
import pandas as pd

# custom
from .. import config, runners
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..poses import Poses, col_in_df, description_from_path
from ..jobstarters import JobStarter, split_list
from ..utils.biopython_tools import load_sequence_from_fasta
from ..utils.openbabel_tools import openbabel_fileconverter

class AlphaFold3(Runner):
    """
    AlphaFold3 Class
    ================

    The `AlphaFold3` class provides a streamlined interface for running AlphaFold3 structure predictions within the ProtFlow framework. It enables complex structure prediction tasks by incorporating support for paired and unpaired MSAs, templates, ligands, nucleic acids, and user-defined modifications.

    Detailed Description
    --------------------
    This class manages the configuration, input preparation, execution, and result parsing for AlphaFold3 predictions. It is designed to be flexible and extensible, enabling integration into high-throughput workflows with custom options for modeling complex assemblies, such as protein–DNA, protein–ligand, and multimers.

    The class supports standard input through `Poses` objects, and advanced inputs via dataframe columns or dictionaries, allowing fine-grained control over prediction components. It internally handles job management through the `JobStarter` interface and formats the output into Pandas dataframes for downstream analysis.

    Parameters:
        work_dir (str, optional): The working directory where results and intermediate files will be stored. If None, a default temp directory is used.
        executable (str, optional): Path to the AlphaFold3 inference script. Defaults to "inference".
        model (str, optional): AlphaFold3 model to use. Defaults to "multimer".
        default_options (dict, optional): Dictionary of default inference options to apply unless overridden. Defaults to None.

    Attributes:
        work_dir (str): Directory used for running AlphaFold3 jobs.
        executable (str): The command or path to the AlphaFold3 inference script.
        model (str): Selected AlphaFold3 model (e.g., "multimer").
        default_options (dict): Default options passed to the AlphaFold3 runner.

    Methods:
        run: Executes AlphaFold3 prediction jobs based on user input.

    Example
    -------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import SbatchArrayJobStarter
        from alphafold3 import AlphaFold3

        # Initialize Poses and JobStarter
        poses = Poses("my_poses.json", work_dir="my_work_dir")
        jobstarter = SbatchArrayJobStarter(cpus=1, gpus=1)

        # Create an AlphaFold3 instance
        af3 = AlphaFold3()

        # Run prediction
        af3.run(
            poses=poses,
            prefix="af3",
            jobstarter=jobstarter,
            additional_entities={"ligand": {"id": "Z", "smiles": "O=C(CCC1)C(=C1)C(O)c(ccc1[N+]([O-])=O)cc1"}},
            options="--flash_attention_implementation xla --cuda_compute_7x 1",
        )

        # Inspect results
        print(poses.df)
    """
    def __init__(self, script_path: str = config.ALPHAFOLD3_SCRIPT_PATH, python_path: str = config.ALPHAFOLD3_PYTHON_PATH, pre_cmd:str=config.ALPHAFOLD3_PRE_CMD, jobstarter: str = None) -> None:
        """
        __init__ Method
        ===============

        Initializes the `AlphaFold3` class instance for running AlphaFold3 predictions within the ProtFlow framework.

        Detailed Description
        --------------------
        This constructor sets up the `AlphaFold3` class, which provides methods to configure, run, and collect AlphaFold3 predictions on protein structures. It initializes basic configuration needed to interface with the AlphaFold3 inference script and prepares the environment for subsequent method calls. Although it takes no arguments, it acts as the anchor for coordinating inputs, options, and prediction output formatting.

        Parameters:
            None

        Returns:
            AlphaFold3: An instance of the `AlphaFold3` class ready to be used for prediction tasks.

        Raises:
            None

        Examples
        --------
        Here's how to initialize the `AlphaFold3` class:

        .. code-block:: python

            from alphafold3 import AlphaFold3

            # Create an instance of the AlphaFold3 class
            af3 = AlphaFold3()

            # Now ready to call af3.run(), af3.write_cmd(), etc.
            print(type(af3))
            # <class 'alphafold3_runner.AlphaFold3'>

        Further Details
        ---------------
            - The `AlphaFold3` instance acts as a controller for generating input JSONs, building inference commands, launching jobs, and collecting results.
            - Requires the AlphaFold3 inference script to be properly installed and available in the environment path or specified within the `write_cmd` method.
        """
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under COLABFOLD_DIR_PATH.")

        self.python_path = python_path
        self.script_path = script_path
        self.name = "alphafold3.py"
        self.pre_cmd = pre_cmd
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "colabfold.py"

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, json_column: str = None, num_copies: int = 1, msa_paired: str = None, msa_unpaired: str = None, templates: Union[str, list, dict] = None, modifications: Union[str, list, dict] = None, col_as_input: bool = False, single_sequence_mode: bool = False, use_templates: bool = True, additional_entities: Union[str, list, dict] = None, bonded_atom_pairs: Union[str, list] = None, user_ccd: Union[str, list] = None, options: str = None, pose_options: str = None, jobstarter: JobStarter = None, overwrite: bool = False, return_top_n_models: int = 1, convert_cif_to_pdb: bool = True, random_seed: bool = False) -> Poses:
        """
        run Method
        ==========

        The `run` method of the `AlphaFold3` class launches AlphaFold3 structure prediction jobs for protein and complex modeling using the provided inputs. It supports multiple input formats and prediction types, including complex assemblies with optional ligands, templates, and nucleic acids.

        Detailed Description
        --------------------
        This method handles end-to-end execution of AlphaFold3 predictions. It prepares the input files, formats optional arguments like ligands and MSAs, interfaces with a jobstarter to execute jobs locally or remotely, and collects results into a standardized dataframe.

        The method supports:
        - Simple protein input from FASTA sequences.
        - Complex inputs with ligands, DNA/RNA, bonded atoms, templates, and custom modifications.
        - Input column extraction from the Poses dataframe.
        - Paired and unpaired multiple sequence alignments.
        - Generation of multiple structures (nstruct) and multiple input copies (num_copies).

        Parameters:
            poses (Poses): The Poses object containing the input sequences or structure data.
            prefix (str): A string used to tag and organize outputs.
            jobstarter (JobStarter, optional): JobStarter object to handle submission logic. Defaults to poses.jobstarter if not provided.
            json_column (str, optional): Column in the poses dataframe containing AlphaFold3-compatible JSON configuration files.
            num_copies (int, optional): Number of duplicated input poses (chains). Defaults to 1.
            nstruct (int, optional): Number of structures generated per pose. Defaults to 1.
            col_as_input (bool, optional): If True, MSA, template, and entity inputs are fetched from columns of the poses dataframe. Defaults to False.
            additional_entities (str or dict or list, optional): Additional input molecules or entities, such as ligands or nucleic acids.
            msa_paired (str, optional): Paired MSA file path or poses column with paths if `col_as_input` is True.
            msa_unpaired (str, optional): Unpaired MSA file path or poses column with paths if `col_as_input` is True.
            templates (str or list or dict, optional): Structural template information, either directly or from poses columns.
            modifications (str or list or dict, optional): User-defined atom or residue modifications.
            options (str or dict, optional): Additional global inference options as string or dictionary.
            pose_options (str or list, optional): Per-pose options for customization.
            overwrite (bool, optional): Whether to overwrite existing prediction results. Defaults to False.
            return_top_n_models (int, optional): Number of top-ranked models to include in the output. Defaults to 1.

        Returns:
            RunnerOutput: Object containing a DataFrame with prediction paths, scores, and metadata.

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: On invalid input combinations or configurations.
            TypeError: If inputs are not of the expected type.

        Examples
        --------
            .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import SbatchArrayJobStarter
            from alphafold3 import AlphaFold3

            # Initialize Poses and JobStarter
            poses = Poses("my_poses.json", work_dir="my_work_dir")
            jobstarter = SbatchArrayJobStarter(cpus=1, gpus=1)

            # Create an AlphaFold3 instance
            af3 = AlphaFold3()

            # Run prediction
            af3.run(
                poses=poses,
                prefix="af3",
                jobstarter=jobstarter,
                additional_entities={"ligand": {"id": "Z", "smiles": "O=C(CCC1)C(=C1)C(O)c(ccc1[N+]([O-])=O)cc1"}},
                options="--flash_attention_implementation xla --cuda_compute_7x 1",
            )

            # Inspect results
            print(poses.df)


        Further Details
        ---------------
            - **Flexible Input Handling:** Accepts input in both standard and column-driven formats, allowing rich dataset support.
            - **Job Management:** Compatible with local or cluster-based execution through `JobStarter`.
            - **Custom Complexes:** Supports modeling of multimeric or ligand/nucleic acid-bound complexes using AlphaFold3’s new features.
            - **Top-N Models:** Automatically filters and returns the highest-scoring models based on AF3 output metrics.
            - **Reproducibility:** All options and configurations used in prediction are logged for reproducibility.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"af3_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        if overwrite:
            if os.path.isdir(json_dir := os.path.join(work_dir, "input_json")):
                shutil.rmtree(json_dir)
            if os.path.isdir(preds_dir := os.path.join(work_dir, "af3_preds")):
                shutil.rmtree(preds_dir)
            if os.path.isdir(pdb_dir := os.path.join(work_dir, "output_pdbs")):
                shutil.rmtree(pdb_dir)

        # setup af3-specific directories:
        os.makedirs(json_dir := os.path.join(work_dir, "input_json"), exist_ok=True)
        os.makedirs(preds_dir := os.path.join(work_dir, "af3_preds"), exist_ok=True)
        if convert_cif_to_pdb:
            os.makedirs(pdb_dir := os.path.join(work_dir, "output_pdbs"), exist_ok=True)

        # setup input-fastas in batches (to speed up prediction times.), but only if no pose_options are provided!
        num_batches = len(poses.df.index) if pose_options else jobstarter.max_cores

        # create input json files
        if json_column:
            json_dirs = []
            col_in_df(poses.df, json_column)
            json_paths = poses.df[json_column].to_list()
            json_paths = split_list(json_paths, n_sublists=num_batches)
            for i, sublist in enumerate(json_paths):
                os.makedirs(json_in:=os.path.join(json_dir, f"input_{i}"))
                for json_path in sublist:
                    shutil.copy(json_path, os.path.join(json_in, os.path.basename(json_path)))
                json_dirs.append(json_in)
        else:
            json_dirs = create_input_json_dir(json_dir, num_batches, poses, nstruct, num_copies, msa_paired, msa_unpaired, modifications, templates, single_sequence_mode, use_templates, col_as_input, additional_entities, bonded_atom_pairs, user_ccd, random_seed)

        # prepare pose options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # write cmds:
        cmds = []
        for input_dir, pose_opt in zip(json_dirs, pose_options):
            cmds.append(self.write_cmd(input_dir, output_dir=preds_dir, options=options, pose_options=pose_opt))

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        logging.info(f"Starting AF3 predictions of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="alphafold3",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        logging.info("Predictions finished, starting to collect scores.")
        scores = collect_scores(work_dir=preds_dir, convert_cif_to_pdb_dir=pdb_dir if convert_cif_to_pdb else None, return_top_n_models=return_top_n_models)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, input_dir: str, output_dir: str, options: str = None, pose_options: str = None):
        """
        write_cmd Method
        ================

        Builds the shell command to invoke AlphaFold3 using the given
        input/output directories and specified options.

        Detailed Description
        --------------------
        This method transforms provided global and pose-specific options into
        CLI-compliant flags and arguments, links them with the Python executable
        and the AF3 inference script, and returns a fully formatted command string.
        Useful for batching jobs or debugging the exact call being issued.

        Parameters:
            input_dir (str): Directory containing input JSON files for prediction.
            output_dir (str): Destination directory where AF3 will save results.
            options (str, optional): Global options passed to AF3, e.g. `"num_recycles=3"`.
            pose_options (str, optional): Pose-specific options passed individually.

        Returns:
            str: Fully assembled shell command ready for execution.

        Raises:
            None

        Examples
        --------
        .. code-block:: python

            cmd = af3.write_cmd(
                input_dir="json_batch_0",
                output_dir="preds_batch_0",
                options="num_recycles=5",
            )
            print(cmd)
            # "python /path/to/alphafold3.py --input_dir json_batch_0 --output_dir preds_batch_0 --num_recycles 5 --use_templates False"

        Further Details
        ---------------
            - Parses both key=value pairs and boolean flags.
            - Ensures flags are prepended with `--` syntax compatible with AF3 scripts.
            - Intended to be used internally to generate jobstarter commands.
        """
        # parse options
        opts, flags = runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""
        return f"{self.python_path} {self.script_path} --input_dir {input_dir} --output_dir {output_dir} {opts} {flags}"

def collect_scores(work_dir: str, convert_cif_to_pdb_dir: str = None, return_top_n_models: int = 1) -> pd.DataFrame:
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
        ranks = pd.read_csv(os.path.join(out_dir, "ranking_scores.csv"))
        ranks.sort_values("ranking_score", ascending=False, inplace=True)
        ranks.reset_index(drop=True, inplace=True)
        data = os.path.join(out_dir, f"{os.path.basename(out_dir)}_data.json")
        with open(data, 'r', encoding="UTF-8") as file:
            data = file.read()
        data = json.loads(data)
        scores = []
        for i, row in ranks.iterrows():
            model_dir = os.path.join(out_dir, f"seed-{int(row['seed'])}_sample-{int(row['sample'])}")
            confidences = pd.read_json(os.path.join(model_dir, "confidences.json"), typ='series', orient='records')
            summary = pd.read_json(os.path.join(model_dir, "summary_confidences.json"), typ='series', orient='records')
            score = pd.concat([summary, confidences])
            model = os.path.join(model_dir, "model.cif")
            score["location"] = os.path.abspath(shutil.copy(model, os.path.join(model_dir, f"{data['name']}_{i+1:04d}.cif")))
            score["description"] = description_from_path(score["location"])
            scores.append(score)
        scores = pd.DataFrame(scores)
        scores["sequence"] = data["sequences"][0]["protein"]["sequence"]
        return scores

    def convert_cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return output

    # collect all output directories, ignore mmseqs dirs
    out_dirs = [d for d in glob(os.path.join(work_dir, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("mmseq")]

    scores = []
    for out_dir in out_dirs:
        data = load_all_models(out_dir)
        data = data.head(return_top_n_models)
        scores.append(data)
    scores = pd.concat(scores)
    scores.reset_index(drop=True, inplace=True)

    if convert_cif_to_pdb_dir:
        os.makedirs(convert_cif_to_pdb_dir, exist_ok=True)
        scores["location"] = scores.apply(lambda row: convert_cif_to_pdb(input_cif=row["location"], output_format="pdb", output=os.path.abspath(os.path.join(convert_cif_to_pdb_dir, f"{row["description"]}.pdb"))), axis=1)
    return scores


def create_input_json_dir(out_dir, num_batches, poses, nstruct, num_copies, msa_paired, msa_unpaired, modifications, templates, single_sequence_mode, use_templates, col_as_input, additional_entities, bonded_atom_pairs, user_ccd, random_seed: bool) -> list:
    """
    create_input_json_dir Function
    ==============================

    Builds and writes AlphaFold3-compatible JSON input files across multiple batching
    subdirectories, based on the content of a `Poses` object and optional customization.

    Detailed Description
    --------------------
    This function iterates over each pose and:
      - Extracts sequences (FASTA-based) and optional per-pose inputs.
      - Handles column-driven inputs (`col_as_input`): MSAs, templates, modifications, etc.
      - Supports single-sequence mode, disabling MSAs.
      - Loads additional entities (protein, DNA, ligands) and validates formats.
      - Assigns nstruct seeds (sequential or random).
      - Packages all data into proper JSON files named after pose descriptions.
      - Splits inputs into `num_batches` subdirectories (input_0, input_1, ...).

    Parameters:
        out_dir (str): Folder where batch subdirectories and JSON files will be created.
        num_batches (int): How many batches to split the inputs into.
        poses (Poses): `Poses` instance containing the input dataset.
        nstruct (int): Number of models to produce per pose.
        num_copies (int): How many duplicated sequence entries per pose.
        msa_paired (Any): Path, column, list, or dict for paired MSA inputs.
        msa_unpaired (Any): Same, but for unpaired MSA.
        modifications (Any): Structural modification inputs.
        templates (Any): Template definitions for prediction.
        single_sequence_mode (bool): If True, disables MSA-based inference.
        use_templates (bool): If False, disables template usage.
        col_as_input (bool): Use DataFrame columns rather than fixed values.
        additional_entities (Any): Extra entities (ligand, dna, rna) inputs.
        bonded_atom_pairs (Any): Bond list or dict to define inter-entity bonds.
        user_ccd (Any): Custom CCD provided as file path, text, or dict.
        random_seed (bool): Whether to assign random seeds instead of sequential.

    Returns:
        list of str: Paths to each created batch directory (e.g., `["out_dir/input_0", ...]`).

    Raises:
        ValueError: If `additional_entities`, `bonded_atom_pairs`, or `user_ccd` are not correctly formatted.
        FileNotFoundError: If provided `user_ccd` path does not exist.

    Examples
    --------
    .. code-block:: python

        batch_dirs = create_input_json_dir(
            out_dir="batch_inputs",
            num_batches=3,
            poses=poses,
            nstruct=2,
            num_copies=1,
            msa_paired=None,
            msa_unpaired=None,
            modifications=None,
            templates=None,
            single_sequence_mode=True,
            use_templates=False,
            col_as_input=False,
            additional_entities=None,
            bonded_atom_pairs=None,
            user_ccd=None,
            random_seed=True
        )
        print(batch_dirs)
        # ['batch_inputs/input_0', 'batch_inputs/input_1', 'batch_inputs/input_2']

    Further Details
    ---------------
        - Ensures each JSON file is named as `<poses_description>.json`.
        - Wraps entity additions in proper AF3 JSON format.
        - Automatically converts single string or dict inputs into lists for JSON.
        - Seeds are deterministic unless `random_seed` is set to True.
    """

    def _prep_option(option: Any, row: pd.Series, col_as_input: bool) -> Any:
        if option is None:
            return None
        if col_as_input:
            return row[option]
        return option

    def check_entity(entity: dict):
        if not isinstance(entity, dict):
            raise ValueError(f"Additional entities must be provided in dict format, not {type(entity)}! Affected entity is {entity}")
        if not any(key in entity for key in ["protein", "ligand", "dna", "rna"]):
            raise ValueError(f"Input entity must contain about type like 'protein', 'ligand', 'dna' or 'rna'. Affected entity: {entity}")

    def import_custom_ccd(path):
        with open(path, "r", encoding="UTF-8") as f:
            ccd = f.read()
        return {"userCCD": ccd}

    # load sequences
    seqs = [str(load_sequence_from_fasta(pose, return_multiple_entries=False).seq) for pose in poses.poses_list()]

    records = []
    for seq, (_, row) in zip(seqs, poses.df.iterrows()):
        row_msa_paired = _prep_option(msa_paired, row, col_as_input)
        row_msa_unpaired = _prep_option(msa_unpaired, row, col_as_input)
        row_modifications = _prep_option(modifications, row, col_as_input)
        row_templates = _prep_option(templates, row, col_as_input)
        row_additional_entities = _prep_option(additional_entities, row, col_as_input)
        row_bonded_atom_pairs = _prep_option(bonded_atom_pairs, row, col_as_input)
        row_user_ccd = _prep_option(user_ccd, row, col_as_input)

        # prevent MSA generation if set
        if single_sequence_mode:
            row_msa_paired = ""
            row_msa_unpaired = ""

        # prevent usage of templates if not set
        if not use_templates:
            row_templates = []

        # add modifications and templates if provided
        if row_modifications and isinstance(row_modifications, dict):
            row_modifications = [row_modifications]
        if row_templates and isinstance(row_templates, dict):
            row_templates = [row_templates]

        # assign a unique id for each copy
        id_ = list(string.ascii_uppercase)[:num_copies]

        # create record for input pose
        pose_data = {"protein": {
            "id": id_,
            "sequence": seq,
            }
        }

        # AF3 will create MSAs automatically if options are not specified
        if isinstance(row_msa_unpaired, str):
            pose_data["protein"].update({"unpairedMsaPath": row_msa_unpaired})
        if isinstance(row_msa_paired, str):
            pose_data["protein"].update({"pairedMsaPath": row_msa_paired})

        # AF3 will use templates automatically if options are not specified
        if isinstance(row_templates, list):
            pose_data["protein"].update({"templates": row_templates})

        # add modifications
        if row_modifications:
            pose_data["protein"].update({"modifications": row_modifications})

        # add additional data
        sequences = [pose_data]

        # handle additional entities for col as input and direct input
        if row_additional_entities and isinstance(row_additional_entities, list):
            for entity in row_additional_entities:
                check_entity(entity)
            sequences = sequences + row_additional_entities
        elif row_additional_entities and isinstance(row_additional_entities, dict):
            check_entity(row_additional_entities)
            sequences.append(row_additional_entities)

        # create input dict
        seeds = [random.randint(1, 100000) for _ in range(nstruct)] if random_seed else list(range(0, nstruct))
        record = {
            "name": row["poses_description"],
            "modelSeeds": seeds,
            "sequences": sequences,
        }

        # add additional settings
        if row_bonded_atom_pairs and isinstance(row_bonded_atom_pairs, dict):
            record.update(row_bonded_atom_pairs)
        elif row_bonded_atom_pairs and isinstance(row_bonded_atom_pairs, list):
            record.update({"bondedAtomPairs": row_bonded_atom_pairs})
        elif row_bonded_atom_pairs:
            raise ValueError(f"Input to :bonded_atom_pairs: must be a nested list of bonds or a dictionary in the AF3 input format, not {type(user_ccd)}!")

        print(row_user_ccd)
        print(type(row_user_ccd))
        if row_user_ccd and isinstance(row_user_ccd, str) and os.path.isfile(row_user_ccd):
            record.update(import_custom_ccd(row_user_ccd))
        elif row_user_ccd and isinstance(row_user_ccd, str):
            record.update({"userCCD": row_user_ccd})
        elif row_user_ccd and isinstance(row_user_ccd, dict):
            record.update(row_user_ccd)
        elif row_user_ccd:
            raise ValueError(f"Input to :user_ccd: must be the path to a mmcif file, a string containing mmcif data or a dictionary in the AF3 input format, not {type(user_ccd)}!")

        records.append(record)

    # split records into batches
    records = split_list(records, n_sublists=num_batches)

    # create a separate input dir for each batch
    json_dirs = []
    for i, sublist in enumerate(records):
        os.makedirs(json_dir := os.path.join(out_dir, f"input_{i}"), exist_ok=True)
        for record in sublist:
            with open(os.path.join(json_dir, f"{record['name']}.json"), "w", encoding="UTF-8") as file:
                json.dump(record, file, indent=4)
        json_dirs.append(json_dir)

    return json_dirs
