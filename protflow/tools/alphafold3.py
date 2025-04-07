"""
ColabFold Module
================

This module provides functionality to integrate ColabFold within the ProtFlow framework, enabling the execution of AlphaFold2 runs on ColabFold. It includes tools to handle inputs, execute runs, and process outputs in a structured and automated manner.

Detailed Description
--------------------
The `ColabFold` class encapsulates all necessary functionalities to run AlphaFold2 through ColabFold. It manages the configuration of essential scripts and paths, sets up the environment, and handles the execution of prediction processes. The class also includes methods for collecting and processing output data, ensuring the results are organized and accessible for further analysis within the ProtFlow ecosystem.

This module streamlines the integration of ColabFold into larger computational workflows by supporting the automatic setup of job parameters, execution of ColabFold commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ColabFold` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the prediction process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ColabFold` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from colabfold import ColabFold

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = LocalJobStarter(max_cores=4)

    # Initialize the ColabFold class
    colabfold = ColabFold()

    # Run the prediction process
    results = colabfold.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="--msa-mode single-sequence",
        pose_options=None,
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- **Edge Cases:** The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the prediction process.
- **Customizability:** Users can customize the prediction process through multiple parameters, including the number of diffusions, specific options for the ColabFold script, and options for handling pose-specific parameters.
- **Integration:** The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate ColabFold into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Authors
-------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""
# general imports
import re
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
import numpy as np

# custom
import protflow.config
import protflow.jobstarters
import protflow.tools
from protflow.runners import Runner, RunnerOutput, prepend_cmd
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.jobstarters import JobStarter, split_list
from protflow.utils.biopython_tools import load_sequence_from_fasta
from protflow.utils.openbabel_tools import openbabel_fileconverter

class AlphaFold3(Runner):
    """
    ColabFold Class
    ===============

    The `ColabFold` class is a specialized class designed to facilitate the execution of AlphaFold2 within the ColabFold environment as part of the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with AlphaFold2 prediction processes.

    Detailed Description
    --------------------
    The `ColabFold` class manages all aspects of running AlphaFold2 predictions through ColabFold. It handles the configuration of necessary scripts and executables, prepares the environment for the prediction processes, and executes the prediction commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to ColabFold scripts and necessary directories.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of AlphaFold2 prediction commands with support for batch processing.
        - Collecting and processing output data into a pandas DataFrame.
        - Managing input FASTA files and preparing them for prediction.
        - Overwriting previous results if specified.

    Returns
    -------
    An instance of the `ColabFold` class, configured to run AlphaFold2 prediction processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If pose options are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `ColabFold` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from colabfold import ColabFold

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ColabFold class
        colabfold = ColabFold()

        # Run the prediction process
        results = colabfold.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            options="inference.num_designs=10",
            pose_options=["inference.input_pdb='input.pdb'"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the prediction process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ColabFold class is intended for researchers and developers who need to perform AlphaFold2 predictions as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.ALPHAFOLD3_SCRIPT_PATH, python_path: str = protflow.config.ALPHAFOLD3_PYTHON_PATH, pre_cmd:str=protflow.config.ALPHAFOLD3_PRE_CMD, jobstarter: str = None) -> None:
        """
        __init__ Method
        ===============

        The `__init__` method initializes an instance of the `ColabFold` class, setting up necessary configurations for running AlphaFold2 predictions through ColabFold within the ProtFlow framework.

        Detailed Description
        --------------------
        This method sets up the paths to the ColabFold script and initializes default values for various attributes required for running predictions. It also allows for the optional configuration of a job starter.

        Parameters:
            script_path (str, optional): The path to the ColabFold script. Defaults to `protflow.config.COLABFOLD_SCRIPT_PATH`.
            jobstarter (JobStarter, optional): An instance of the `JobStarter` class for managing job execution. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the `script_path` is not provided.

        Examples
        --------
        Here is an example of how to initialize the `ColabFold` class:

        .. code-block:: python

            from colabfold import ColabFold

            # Initialize the ColabFold class
            colabfold = ColabFold(script_path='/path/to/colabfold.py', jobstarter=jobstarter)
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

        The `run` method of the `ColabFold` class executes AlphaFold2 predictions using ColabFold within the ProtFlow framework. It manages the setup, execution, and result collection processes, providing a streamlined way to integrate AlphaFold2 predictions into larger computational workflows.

        Detailed Description
        --------------------
        This method orchestrates the entire prediction process, from preparing input data and configuring the environment to running the prediction commands and collecting the results. The method supports batch processing of input FASTA files and handles various edge cases, such as overwriting existing results and managing job starter options.

        Parameters:
            poses (Poses): The Poses object containing the protein data. Poses have to be single-chain .fasta files!
            prefix (str): A prefix used to name and organize the output files.
            json_column (str, optional): Use the specified column containing paths to json files as input instead of regular poses.
            num_copies (int, optional): How many copies of the input pose sequence should be generated. Default is 1.
            nstruct (int, optional): How many structures should be generated for each pose. Default is 1.
            col_as_input (bool, optional): Given input for :dna:, :rna:, :ligand:, :paired_msa:, :unpaired_msa:, :templates:, :bonded_atom_pairs: is extracted from respective poses dataframe column instead of directly using it as input. Default is False.
            additional_entities (str or dict or list, optional): Can be either a dict specifiying additional protein, ligand, DNA or RNA input for complex prediction, a list of multiple dicts or a poses dataframe column containing dicts or list of dicts (if :col_as_input: is True). Default is None.
            msa_paired (str, optional): Path to .a3m paired alignment file or poses dataframe column containing paths (if :col_as_input: is True). Default is None.
            msa_unpaired (str, optional): Path to .a3m unpaired alignment file or poses dataframe column containing paths (if :col_as_input: is True). Default is None.
            templates (str or list or dict, optional): Can be either a dict specifying template input for AlphaFold3, a list of multiple templates or a poses dataframe column containing dicts or list of dicts (if :col_as_input: is True). Default is None.
            modifications (str or list or dict, optional): Can be either a dict specifying modifcation input for AlphaFold3, a list of multiple templates or a poses dataframe column containing dicts or list of dicts (if :col_as_input: is True). Default is None.
            options (str, optional): Additional options for the AlphaFold2 prediction commands. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters during prediction. Defaults to None.
            overwrite (bool, optional): If True, existing results will be overwritten. Defaults to False.
            return_top_n_models (int, optional): The number of top poses to return based on the prediction scores. Defaults to 1.

        Returns:
            RunnerOutput: An object containing the results of the AlphaFold2 predictions, organized in a pandas DataFrame.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the methods.
            TypeError: If pose options are not of the expected type.

        Examples
        --------
        Here is an example of how to use the `run` method of the `ColabFold` class:

        .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import JobStarter
            from colabfold import ColabFold

            # Create instances of necessary classes
            poses = Poses()
            jobstarter = JobStarter()

            # Initialize the ColabFold class
            colabfold = ColabFold()

            # Run the prediction process
            results = colabfold.run(
                poses=poses,
                prefix="experiment_1",
                jobstarter=jobstarter,
                options="inference.num_designs=10",
                pose_options=["inference.input_pdb='input.pdb'"],
                overwrite=True
            )

            # Access and process the results
            print(results)
        
        Further Details
        ---------------
            - **Batch Processing:** The method can handle large sets of input sequences by batching them into smaller groups, which helps in managing computational resources effectively.
            - **Overwrite Handling:** If `overwrite` is set to True, the method will clean up previous results, ensuring that the new predictions do not get mixed up with old data.
            - **Job Starter Configuration:** The method allows for flexible job management by accepting a `JobStarter` instance. If not provided, it uses the default job starter associated with the poses.
            - **Score Collection:** The method gathers the prediction scores and relevant data into a pandas DataFrame, facilitating easy analysis and integration with other ProtFlow components.
            - **Error Handling:** Robust error handling is incorporated to manage issues such as missing files or incorrect configurations, ensuring that the process can be debugged and verified efficiently.
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
            if os.path.isdir(json_dir := os.path.join(work_dir, "input_json")): shutil.rmtree(json_dir)
            if os.path.isdir(preds_dir := os.path.join(work_dir, "af3_preds")): shutil.rmtree(preds_dir)
            if os.path.isdir(pdb_dir := os.path.join(work_dir, "output_pdbs")): shutil.rmtree(pdb_dir)

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
        logging.info(f"Predictions finished, starting to collect scores.")
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

        The `write_cmd` method constructs the command string necessary to run the ColabFold script with the specified options and input files.

        Detailed Description
        --------------------
        This method generates the command string used to execute the ColabFold script. It incorporates various options and pose-specific parameters provided by the user.

        Parameters:
            pose_path (str): Path to the input FASTA file.
            output_dir (str): Directory where the prediction outputs will be stored.
            options (str, optional): Additional options for the ColabFold script. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters. Defaults to None.

        Returns:
            str: The constructed command string.

        Raises:
            None

        Examples
        --------
        Here is an example of how to use the `write_cmd` method:

        .. code-block:: python

            # Write the command to run ColabFold
            cmd = colabfold.write_cmd(pose_path='/path/to/pose.fa', output_dir='/path/to/output_dir', options='--num_designs=10', pose_options='--input_pdb=input.pdb')
        """
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""
        return f"{self.python_path} {self.script_path} --input_dir {input_dir} --output_dir {output_dir} {opts} {flags}"

def collect_scores(work_dir: str, convert_cif_to_pdb_dir: str = None, return_top_n_models: int = 1) -> pd.DataFrame:
    """
    collect_scores Method
    =====================

    The `collect_scores` method collects and processes the prediction scores from the ColabFold output, organizing them into a pandas DataFrame for further analysis.

    Detailed Description
    --------------------
    This method gathers the prediction scores from the output files generated by ColabFold. It processes these scores and organizes them into a structured DataFrame, which includes various statistical measures.

    Parameters:
        work_dir (str): The working directory where the ColabFold outputs are stored.
        num_return_poses (int, optional): The number of top poses to return based on the prediction scores. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame containing the collected and processed scores.

    Raises:
        FileNotFoundError: If no output files are found in the specified directory.

    Examples
    --------
    Here is an example of how to use the `collect_scores` method:

    .. code-block:: python

        # Collect and process the prediction scores
        scores = collect_scores(work_dir='/path/to/work_dir', num_return_poses=5)     
    
    Further Details
    ---------------
        - **JSON and PDB File Parsing:** The method identifies and parses JSON and PDB files generated by ColabFold, extracting relevant score information from these files.
        - **Statistical Measures:** For each set of predictions, the method calculates various statistics, including mean pLDDT, max PAE, and PTM scores, organizing these measures into the DataFrame.
        - **Rank and Description:** The scores are ranked and annotated with descriptions to help identify the top poses based on prediction quality.
        - **File Handling:** The method includes robust file handling to ensure that only the relevant files are processed, and any existing files are correctly identified or overwritten as needed.
        - **Pose Location:** The final DataFrame includes the file paths to the predicted PDB files, facilitating easy access for further analysis or visualization.
    """
    
    def load_all_models(out_dir: str) -> pd.DataFrame:
        os.makedirs(model_dir := os.path.join(out_dir, "models"), exist_ok=True)
        ranks = pd.read_csv(os.path.join(out_dir, "ranking_scores.csv"))
        ranks.sort_values("ranking_score", ascending=False, inplace=True)
        ranks.reset_index(drop=True, inplace=True)
        data = os.path.join(out_dir, f"{os.path.basename(out_dir)}_data.json")
        with open(data, 'r') as file:
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

    def convert_cif_to_pdb(input: str, format: str, output:str):
        openbabel_fileconverter(input_file=input, output_format=format, output_file=output)
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
    print(scores)

    if convert_cif_to_pdb_dir:
        os.makedirs(convert_cif_to_pdb_dir, exist_ok=True)
        scores["location"] = scores.apply(lambda row: convert_cif_to_pdb(row["location"], format="pdb", output=os.path.abspath(os.path.join(convert_cif_to_pdb_dir, f"{row["description"]}.pdb"))), axis=1)
    return scores


def create_input_json_dir(out_dir, num_batches, poses, nstruct, num_copies, msa_paired, msa_unpaired, modifications, templates, single_sequence_mode, use_templates, col_as_input, additional_entities, bonded_atom_pairs, user_ccd, random_seed: bool) -> list:
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
        id = list(string.ascii_uppercase)[:num_copies]

        # create record for input pose
        pose_data = {"protein": {
            "id": id,
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
