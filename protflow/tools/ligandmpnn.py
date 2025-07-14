"""
LigandMPNN Module
=================

This module provides the functionality to integrate LigandMPNN within the ProtFlow framework. It offers tools to run LigandMPNN, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `LigandMPNN` class encapsulates the functionality necessary to execute LigandMPNN runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of the diffusion processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of LigandMPNN into larger computational workflows. It supports the automatic setup of job parameters, execution of LigandMPNN commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `LigandMPNN` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `LigandMPNN` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from ligandmpnn import LigandMPNN

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the LigandMPNN class
    ligandmpnn = LigandMPNN()

    # Run the diffusion process
    results = ligandmpnn.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        nseq=10,
        model_type="ligand_mpnn",
        options="some_option=some_value",
        pose_options=["pose_option=pose_value"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the process.
- Customizability: Users can customize the process through multiple parameters, including the number of sequences, specific options for the LigandMPNN script, and options for handling pose-specific parameters.
- Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate LigandMPNN into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Author
------
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
from typing import Union

# dependencies
import pandas as pd
import Bio
import Bio.SeqIO

# custom
from protflow import config, jobstarters
from protflow.residues import ResidueSelection
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.runners import Runner, RunnerOutput, regex_expand_options_flags, parse_generic_options, col_in_df, options_flags_to_string, prepend_cmd
from protflow.config import PROTFLOW_ENV

LIGANDMPNN_CHECKPOINT_DICT = {
    "protein_mpnn": "/model_params/proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "/model_params/ligandmpnn_v_32_010_25.pt",
    "per_residue_label_membrane_mpnn": "/model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
    "global_label_membrane_mpnn": "/model_params/global_label_membrane_mpnn_v_48_020.pt",
    "soluble_mpnn": "/model_params/solublempnn_v_48_020.pt"
}

class LigandMPNN(Runner):
    """
    LigandMPNN Class
    ================

    The `LigandMPNN` class provides the necessary methods to execute LigandMPNN runs within the ProtFlow framework. This class is responsible for managing the configuration, execution, and output processing of LigandMPNN tasks.

    Detailed Description
    --------------------
    The `LigandMPNN` class integrates LigandMPNN into the ProtFlow pipeline by setting up the environment, running the diffusion process, and collecting the results. It ensures that the inputs and outputs are handled efficiently, making the data readily available for further analysis.

    Key Features:
    - Manages paths to essential scripts and executables.
    - Configures and executes LigandMPNN processes.
    - Collects and processes output data into a structured DataFrame format.
    - Handles various edge cases and supports custom configurations through multiple parameters.

    Usage
    -----
    To use this class, initialize it with the appropriate script and Python paths, along with an optional job starter. The main functionality is provided through the `run` method, which requires parameters such as poses, prefix, and additional options for customization.

    Example
    -------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from ligandmpnn import LigandMPNN

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the LigandMPNN class
        ligandmpnn = LigandMPNN()

        # Run the diffusion process
        results = ligandmpnn.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            nseq=10,
            model_type="ligand_mpnn",
            options="some_option=some_value",
            pose_options=["pose_option=pose_value"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Notes
    -----
    This class is designed to work within the ProtFlow framework and assumes that the necessary configurations and dependencies are properly set up. It leverages shared data structures and configurations from ProtFlow to provide a seamless integration experience.

    Author
    ------
    Markus Braun, Adrian Tripp

    Version
    -------
    0.1.0
    """
    def __init__(self, script_path:str=config.LIGANDMPNN_SCRIPT_PATH, python_path:str=config.LIGANDMPNN_PYTHON_PATH, pre_cmd:str=config.LIGANDMPNN_PRE_CMD, jobstarter:JobStarter=None) -> None:
        """
        Initializes the LigandMPNN class.

        Parameters:
            script_path (str, optional): The path to the LigandMPNN script. Defaults to the configured script path in ProtFlow.
            python_path (str, optional): The path to the Python executable to run the LigandMPNN script. Defaults to the configured Python path in ProtFlow.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job submissions. If not provided, it will use the default job starter configuration.

        Detailed Description
        --------------------
        The `__init__` method sets up the necessary paths and configurations for running LigandMPNN. It searches for the provided script and Python
        paths to ensure they are correct and sets them as instance attributes. Additionally, it initializes the job starter, which manages the execution
        of jobs in high-performance computing (HPC) environments. This method ensures that all configurations are correctly set up before running any
        LigandMPNN tasks.
        """


        self.script_path = self.search_path(script_path, "LIGANDMPNN_SCRIPT_PATH")
        self.python_path = self.search_path(python_path, "LIGANDMPNN_PYTHON_PATH")
        self.pre_cmd = pre_cmd
        self.name = "ligandmpnn.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "ligandmpnn.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, nseq: int = 1, model_type: str = None, options: str = None, pose_options: object = None, fixed_res_col: str = None, design_res_col: str = None, pose_opt_cols: dict = None, return_seq_threaded_pdbs_as_pose: bool = False, preserve_original_output: bool = False, overwrite: bool = False) -> Poses:
        """
        Execute the LigandMPNN process with given poses and jobstarter configuration.

        This method sets up and runs the LigandMPNN process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            nseq (int, optional): The number of sequences to generate for each input pose. Defaults to 1.
            model_type (str, optional): The type of model to use. Defaults to 'ligand_mpnn'.
            options (str, optional): Additional options for the LigandMPNN script. Defaults to None.
            pose_options (object, optional): Pose-specific options for the LigandMPNN script. Defaults to None.
            fixed_res_col (str, optional): Column name in the poses DataFrame specifying fixed residues. Defaults to None.
            design_res_col (str, optional): Column name in the poses DataFrame specifying residues to be redesigned. Defaults to None.
            pose_opt_cols (dict, optional): Dictionary of pose-specific options for the LigandMPNN script. Defaults to None.
            return_seq_threaded_pdbs_as_pose (bool, optional): If True, return sequence-threaded PDBs as poses. Defaults to False.
            preserve_original_output (bool, optional): If True, preserve the original output files. Defaults to True.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: The updated Poses object containing the results of the LigandMPNN process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from ligandmpnn import LigandMPNN

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the LigandMPNN class
                ligandmpnn = LigandMPNN()

                # Run the diffusion process
                results = ligandmpnn.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    nseq=10,
                    model_type="ligand_mpnn",
                    options="some_option=some_value",
                    pose_options=["pose_option=pose_value"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the process to their specific needs.

        This method is designed to streamline the execution of LigandMPNN processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein design simulations.
        """
        self.index_layers = 1
        # run in batch mode if pose_options are not set:
        pose_opt_cols = pose_opt_cols or {}
        run_batch = self.check_for_batch_run(pose_options, pose_opt_cols)
        if run_batch:
            logging.info("Setting up ligandmpnn for batched design.")

        # check if sidechain packing was specified in options
        pack_sidechains = "pack_side_chains" in options if options else False

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"ligandmpnn_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # integrate redesigned and fixed residue parameters into pose_opt_cols:
        if fixed_res_col is not None:
            pose_opt_cols["fixed_residues"] = fixed_res_col
        if design_res_col is not None:
            pose_opt_cols["redesigned_residues"] = design_res_col

        # parse pose_opt_cols into pose_options format.
        pose_opt_cols_options = self.parse_pose_opt_cols(poses=poses, pose_opt_cols=pose_opt_cols, output_dir=work_dir)

        # parse pose_options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # combine pose_options and pose_opt_cols_options (priority goes to pose_opt_cols_options):
        pose_options = [options_flags_to_string(*parse_generic_options(pose_opt, pose_opt_cols_opt, sep="--"), sep="--") for pose_opt, pose_opt_cols_opt in zip(pose_options, pose_opt_cols_options)]

        # write ligandmpnn cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, model=model_type, nseq=nseq, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df['poses'].to_list(), pose_options)]

        # batch_run setup:
        if run_batch:
            cmds = self.setup_batch_run(cmds, num_batches=jobstarter.max_cores, output_dir=work_dir)

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # create output directories, LigandMPNN crashes sometimes when multiple processes create the same directory simultaneously (frozen os error)
        for folder in ["backbones", "input_json_files", "packed", "seqs"]:
            os.makedirs(os.path.join(work_dir, folder), exist_ok=True)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="ligandmpnn",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose,
            preserve_original_output=preserve_original_output,
            pack_sidechains=pack_sidechains
        )

        if len(scores.index) < len(poses.df.index) * nseq:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def check_for_batch_run(self, pose_options: str, pose_opt_cols):
        """
        Checks if LigandMPNN can be run in batch mode.

        This method determines whether the LigandMPNN process can be executed in batch mode. It does this by checking if pose-specific options are not provided and if only multi-residue columns are specified in the pose options.

        Parameters:
            pose_options (str): Pose-specific options for the LigandMPNN script.
            pose_opt_cols (dict): Dictionary of pose-specific options for the LigandMPNN script.

        Returns:
            bool: True if LigandMPNN can be run in batch mode, False otherwise.

        Examples:
            Here is an example of how to use the `check_for_batch_run` method:

            .. code-block:: python

                # Initialize the LigandMPNN class
                ligandmpnn = LigandMPNN()

                # Check for batch run
                can_batch_run = ligandmpnn.check_for_batch_run(
                    pose_options=None,
                    pose_opt_cols={"fixed_residues": "fixed_res_col"}
                )

                print(can_batch_run)  # Outputs: True or False

        Further Details:
            - **Batch Mode Check:** The method checks if the `pose_options` is None and if the `pose_opt_cols` contains only multi-residue columns, which are necessary for batch processing.
        """
        return pose_options is None and self.multi_cols_only(pose_opt_cols)

    def multi_cols_only(self, pose_opt_cols:dict) -> bool:
        '''checks if only multi_res cols are in pose_opt_cols dict. Only _multi arguments can be used for ligandmpnn_batch runs.'''
        multi_cols = ["omit_AA_per_residue", "bias_AA_per_residue", "redesigned_residues", "fixed_residues"]
        return True if pose_opt_cols is None else all((col in multi_cols for col in pose_opt_cols))

    def setup_batch_run(self, cmds:list[str], num_batches:int, output_dir:str) -> list[str]:
        """
        Concatenates commands for MPNN into batches so that MPNN does not have to be loaded individually for each PDB file.

        This method prepares the LigandMPNN commands for batch execution. It concatenates the commands into batches to optimize the running process by reducing the overhead of loading the MPNN model multiple times.

        Parameters:
            cmds (list[str]): A list of commands to run LigandMPNN.
            num_batches (int): The number of batches to split the commands into.
            output_dir (str): The directory where the batch input JSON files will be saved.

        Returns:
            list[str]: A list of concatenated batch commands.

        Examples:
            Here is an example of how to use the `setup_batch_run` method:

            .. code-block:: python

                # Initialize the LigandMPNN class
                ligandmpnn = LigandMPNN()

                # Example commands
                cmds = [
                    "/path/to/python /path/to/run.py --option1=value1 --pdb_path=path1.pdb",
                    "/path/to/python /path/to/run.py --option2=value2 --pdb_path=path2.pdb",
                    # More commands...
                ]

                # Setup batch run
                batch_cmds = ligandmpnn.setup_batch_run(
                    cmds=cmds,
                    num_batches=2,
                    output_dir="/path/to/output"
                )

                print(batch_cmds)  # Outputs the batch commands

        Further Details:
            - **Batch Command Setup:** The method splits the provided commands into sublists based on the number of batches. It then processes each sublist to handle multi-residue options and generate corresponding JSON files.
            - **JSON Directory:** The method sets up a directory for storing JSON files that contain mappings for multi-residue options.
            - **Command Concatenation:** Each command sublist is processed to extract and convert multi-residue options into JSON files, which are then referenced in the batch commands.
        """
        multi_cols = {
            "omit_AA_per_residue": "omit_AA_per_residue_multi",
            "bias_AA_per_residue": "bias_AA_per_residue_multi", 
            "redesigned_residues": "redesigned_residues_multi",
            "fixed_residues": "fixed_residues_multi", 
            "pdb_path": "pdb_path_multi"
        }
        # setup json directory
        json_dir = f"{output_dir}/input_json_files/"
        if not os.path.isdir(json_dir):
            os.makedirs(json_dir, exist_ok=True)

        # split cmds list into n=num_batches sublists
        cmd_sublists = jobstarters.split_list(cmds, n_sublists=num_batches)

        # concatenate cmds: parse _multi arguments into .json files and keep all other arguments in options.
        batch_cmds = []
        for i, cmd_list in enumerate(cmd_sublists, start=1):
            full_cmd_list = [cmd.split(" ") for cmd in cmd_list]
            opts_flags_list = [regex_expand_options_flags(" ".join(cmd_split[2:])) for cmd_split in full_cmd_list]
            opts_list = [x[0] for x in opts_flags_list] # regex_expand_options_flags() returns (options, flags)

            # take first cmd for general options and flags
            full_opts_flags = opts_flags_list[0]
            cmd_start = " ".join(full_cmd_list[0][:2]) # keep /path/to/python3 /path/to/run.py

            # extract lists for _multi options
            for col, multi_col in multi_cols.items():
                # if col does not exist in options, skip:
                if col not in opts_list[0]:
                    continue

                # extract pdb-file to argument mapping as dictionary:
                col_dict = {opts["pdb_path"]: opts[col] for opts in opts_list}

                # write col_dict to json
                col_json_path = f"{json_dir}/{col}_{i}.json"
                with open(col_json_path, 'w', encoding="UTF-8") as f:
                    json.dump(col_dict, f)

                # remove single option from full_opts_flags and set cmd_json file as _multi option:
                del full_opts_flags[0][col]
                full_opts_flags[0][multi_col] = col_json_path

            # reassemble command and put into batch_cmds
            batch_cmd = f"{cmd_start} {options_flags_to_string(*full_opts_flags, sep='--')}"
            batch_cmds.append(batch_cmd)

        return batch_cmds

    def parse_pose_opt_cols(self, poses: Poses, output_dir: str, pose_opt_cols: dict = None) -> list[dict]:
        """
        Parses pose-specific options columns into pose options formatted strings.

        This method processes the `pose_opt_cols` dictionary and converts its contents into a format that can be used as part of the LigandMPNN pose options. It ensures that the options are properly structured and, if necessary, writes specific arguments into JSON files.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            output_dir (str): The directory where JSON files for multi-residue options will be saved.
            pose_opt_cols (dict, optional): Dictionary of pose-specific options for the LigandMPNN script. Defaults to None.

        Returns:
            list[dict]: A list of dictionaries containing the parsed pose options formatted as strings.

        Raises:
            ValueError: If both fixed_residues and redesigned_residues are defined in pose_opt_cols, or if specified columns do not exist in poses.df.

        Examples:
            Here is an example of how to use the `parse_pose_opt_cols` method:

            .. code-block:: python

                # Initialize the LigandMPNN class
                ligandmpnn = LigandMPNN()

                # Example Poses object and pose_opt_cols
                poses = Poses()
                pose_opt_cols = {
                    "bias_AA_per_residue": "bias_col",
                    "fixed_residues": "fixed_res_col"
                }

                # Parse pose options
                parsed_opts = ligandmpnn.parse_pose_opt_cols(
                    poses=poses,
                    output_dir="/path/to/output",
                    pose_opt_cols=pose_opt_cols
                )

                print(parsed_opts)  # Outputs the parsed pose options

        Further Details:
            - **Option Parsing:** The method converts the `pose_opt_cols` dictionary into a list of strings formatted as pose options. It handles various types of options, including those that need to be written into JSON files and those that can be parsed directly from residue selections.
            - **JSON Directory Setup:** If necessary, the method sets up a directory for storing JSON files that contain mappings for multi-residue options.
            - **Error Handling:** The method includes checks to ensure that incompatible options are not specified simultaneously and that all specified columns exist in the poses DataFrame.
        """
        # return list of empty strings if pose_opts_col is None.
        if pose_opt_cols is None:
            return ["" for _ in poses]

        # setup output_dir for .json files
        if any([key in ["bias_AA_per_residue", "omit_AA_per_residue"] for key in pose_opt_cols]):
            json_dir = f"{output_dir}/input_json_files/"
            if not os.path.isdir(json_dir):
                os.makedirs(json_dir, exist_ok=True)

        # check if fixed_residues and redesigned_residues were set properly (gets checked in LigandMPNN too, so maybe this is redundant.)
        if "fixed_residues" in pose_opt_cols and "redesigned_residues" in pose_opt_cols:
            raise ValueError("Cannot define both <fixed_res_column> and <design_res_column>!")

        # check if all specified columns exist in poses.df:
        for col in list(pose_opt_cols.values()):
            col_in_df(poses.df, col)

        # parse pose_options
        pose_options = []
        for pose in poses:
            opts = []
            for mpnn_arg, mpnn_arg_col in pose_opt_cols.items():
                # arguments that must be written into .json files:
                if mpnn_arg in ["bias_AA_per_residue", "omit_AA_per_residue"]:
                    output_path = f"{json_dir}/{mpnn_arg}_{pose['poses_description']}.json"
                    opts.append(f"--{mpnn_arg}={write_to_json(pose[mpnn_arg_col], output_path)}")

                # arguments that can be parsed as residues (from ResidueSelection objects):
                elif mpnn_arg in ["redesigned_residues", "fixed_residues", "transmembrane_buried", "transmembrane_interface"]:
                    opts.append(f"--{mpnn_arg}={parse_residues(pose[mpnn_arg_col])}")

                # all other arguments:
                else:
                    opts.append(f"--{mpnn_arg}={pose[mpnn_arg_col]}")
            pose_options.append(" ".join(opts))
        return pose_options

    def write_cmd(self, pose_path:str, output_dir:str, model:str, nseq:int, options:str, pose_options:str):
        """
        Writes the command to run ligandmpnn.py.

        This method constructs the command necessary to run the LigandMPNN script, incorporating various options and parameters. It ensures that the command is correctly formatted and includes all required arguments.

        Parameters:
            pose_path (str): The path to the input PDB file for the pose.
            output_dir (str): The directory where the output files will be saved.
            model (str): The type of model to use (e.g., "ligand_mpnn").
            nseq (int): The number of sequences to generate for each input pose. Defaults to 1.
            options (str): Additional options for the LigandMPNN script.
            pose_options (str): Pose-specific options for the LigandMPNN script.

        Returns:
            str: The constructed command string to run LigandMPNN.

        Raises:
            ValueError: If the specified model is not one of the available models.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                # Initialize the LigandMPNN class
                ligandmpnn = LigandMPNN()

                # Write the command
                cmd = ligandmpnn.write_cmd(
                    pose_path="path/to/input.pdb",
                    output_dir="path/to/output",
                    model="ligand_mpnn",
                    nseq=10,
                    options="some_option=some_value",
                    pose_options="pose_option=pose_value"
                )

                print(cmd)  # Outputs the constructed command string

        Further Details:
            - **Model Validation:** The method checks if the specified model is among the available models and raises an error if it is not.
            - **Option Parsing:** The method parses generic options and pose-specific options, ensuring that necessary safety checks and defaults are applied.
            - **Command Construction:** The method assembles the final command string, including paths, model checkpoints, options, and other necessary parameters.
        """
        # parse ligandmpnn_dir:
        ligandmpnn_dir = config.LIGANDMPNN_SCRIPT_PATH.rsplit("/", maxsplit=1)[0]

        # check if specified model is correct.
        available_models = ["protein_mpnn", "ligand_mpnn", "soluble_mpnn", "global_label_membrane_mpnn", "per_residue_label_membrane_mpnn"]
        if model not in available_models:
            raise ValueError(f"{model} must be one of {available_models}!")

        # parse options
        opts, flags = parse_generic_options(options, pose_options)

        # safetychecks:
        if "model_type" not in opts:
            opts["model_type"] = model or "ligand_mpnn"
        if "number_of_batches" not in opts:
            opts["number_of_batches"] = nseq or "1"
        # define model_checkpoint option:
        if f"checkpoint_{model}" not in opts:
            model_checkpoint_options = f"--checkpoint_{model}={ligandmpnn_dir}/{LIGANDMPNN_CHECKPOINT_DICT[model]}"
        else:
            model_checkpoint_options = opts[f"checkpoint_{model}"]

        # safety
        logging.debug("Setting parse_atoms_with_zero_occupancy to 1 to ensure that the run does not crash.")
        if "parse_atoms_with_zero_occupancy" not in opts:
            opts["parse_atoms_with_zero_occupancy"] = "1"
        elif opts["parse_atoms_with_zero_occupancy"] != "1":
            opts["parse_atoms_with_zero_occupancy"] = "1"

        # convert to string
        options = options_flags_to_string(opts, flags, sep="--")

        # write command and return.
        return f"{self.python_path} {self.script_path} {model_checkpoint_options} --out_folder {output_dir}/ --pdb_path {pose_path} {options}"

def collect_scores(work_dir: str, return_seq_threaded_pdbs_as_pose: bool, preserve_original_output: bool = True, pack_sidechains: bool = False) -> pd.DataFrame:
    """
    Collects scores from the LigandMPNN output.

    This method processes the output files generated by LigandMPNN, including multi-sequence FASTA files and PDB files. It reads, renames, and organizes these files into a structured DataFrame.

    Parameters:
        work_dir (str): The directory where LigandMPNN output files are located.
        return_seq_threaded_pdbs_as_pose (bool): If True, replaces FASTA files with sequence-threaded PDB files as poses.
        preserve_original_output (bool, optional): If True, preserves the original output files. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores and relevant data from the LigandMPNN output.

    Raises:
        FileNotFoundError: If required output files are not found in the specified directory.

    Examples:
        Here is an example of how to use the `collect_scores` method:

        .. code-block:: python

            # Initialize the LigandMPNN class
            ligandmpnn = LigandMPNN()

            # Collect scores from the output directory
            scores = ligandmpnn.collect_scores(
                work_dir="/path/to/output",
                return_seq_threaded_pdbs_as_pose=True,
                preserve_original_output=False
            )

            print(scores)  # Outputs the collected scores DataFrame

    Further Details:
        - **Output Processing:** The method reads and parses multi-sequence FASTA files, converts sequences into a structured dictionary, and writes new FASTA files if necessary.
        - **File Management:** Original output files are copied to dedicated directories, and new files are generated and organized for easy access. Optionally, original files can be preserved or deleted based on the `preserve_original_output` parameter.
        - **Error Handling:** The method includes checks to ensure that required output files are present, raising errors if files are missing or paths are incorrect.
    """
    def mpnn_fastaparser(fasta_path):
        '''reads in ligandmpnn multi-sequence fasta, renames sequences and returns them'''
        records = list(Bio.SeqIO.parse(fasta_path, "fasta"))
        #maxlength = len(str(len(records)))

        # Set continuous numerating for the names of mpnn output sequences:
        name = records[0].name.replace(",", "")
        records[0].name = name
        for i, x in enumerate(records[1:]):
            setattr(x, "name", f"{name}_{str(i+1).zfill(4)}")

        return records

    def convert_ligandmpnn_seqs_to_dict(seqs):
        '''
        Takes already parsed list of fastas as input <seqs>. Fastas can be parsed with the function mpnn_fastaparser(file).
        Should be put into list.
        Converts mpnn fastas into a dictionary:
        {
            "col_1": [vals]
                ...
            "col_n": [vals]
        }
        '''
        # Define cols and initiate them as empty lists:
        seqs_dict = {}
        cols = ["mpnn_origin", "seed", "description", "sequence", "T", "id", "seq_rec", "overall_confidence", "ligand_confidence"]
        for col in cols:
            seqs_dict[col] = []

        # Read scores of each sequence in each file and append them to the corresponding columns:
        for seq in seqs:
            for f in seq[1:]:
                seqs_dict["mpnn_origin"].append(seq[0].name)
                seqs_dict["sequence"].append(str(f.seq))
                seqs_dict["description"].append(f.name)
                d = {k: float(v) for k, v in [x.split("=") for x in f.description.split(", ")[1:]]}
                for k, v in d.items():
                    seqs_dict[k].append(v)
        return seqs_dict

    def write_mpnn_fastas(seqs_dict: dict) -> pd.DataFrame:
        seqs_dict["location"] = list()
        for d, s in zip(seqs_dict["description"], seqs_dict["sequence"]):
            seqs_dict["location"].append((fa_file := f"{seq_dir}/{d}.fa"))
            with open(fa_file, 'w', encoding="UTF-8") as f:
                f.write(f">{d}\n{s}")
        return pd.DataFrame(seqs_dict)

    def rename_mpnn_pdb(pdb: str) -> None:
        '''changes single digit file extension to 4 digit file extension'''
        filename, extension = os.path.splitext(pdb)[0].rsplit('_', 1)
        filename = f"{filename}_{extension.zfill(4)}.pdb"
        shutil.move(pdb, filename)

    def rename_packed_pdb(pdb_path: str) -> None:
        '''changes single digit file extension to 4 digit file extension.'''
        filename = os.path.splitext(pdb_path)[0]
        name_split = filename.split("_")
        name_split[-1] = name_split[-1].zfill(4)
        name_split[-2] = name_split[-2].zfill(4)
        name_split.remove("packed")
        filename = f"{'_'.join(name_split)}.pdb"
        shutil.move(pdb_path, filename)
        if filename.endswith("_0001.pdb"):
            shutil.copy(filename, filename.rsplit("_", 1)[0] + ".pdb")

    # read .pdb files
    seq_dir = os.path.join(work_dir, 'seqs')
    pdb_dir = os.path.join(work_dir, 'backbones')
    fl = glob(f"{seq_dir}/*.fa")
    pl = glob(f"{pdb_dir}/*.pdb")
    if not fl:
        raise FileNotFoundError(f"No .fa files were found in the output directory of LigandMPNN {seq_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")
    if not pl:
        raise FileNotFoundError(f"No .pdb files were found in the output directory of LigandMPNN {pdb_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")

    seqs = [mpnn_fastaparser(fasta) for fasta in fl]
    seqs_dict = convert_ligandmpnn_seqs_to_dict(seqs)

    original_seqs_dir = os.path.join(seq_dir, 'original_seqs')
    logging.info(f"Copying original .fa files into directory {original_seqs_dir}")
    os.makedirs(original_seqs_dir, exist_ok=True)
    _ = [shutil.move(fasta, os.path.join(original_seqs_dir, os.path.basename(fasta))) for fasta in fl]

    original_pdbs_dir = os.path.join(pdb_dir, 'original_backbones')
    logging.info(f"Copying original .pdb files into directory {original_pdbs_dir}")
    os.makedirs(original_pdbs_dir, exist_ok=True)
    _ = [shutil.copy(pdb, os.path.join(original_pdbs_dir, os.path.basename(pdb))) for pdb in pl]
    _ = [rename_mpnn_pdb(pdb) for pdb in pl]

    # Write new .fa files by iterating through "description" and "sequence" keys of the seqs_dict
    logging.info(f"Writing new fastafiles at original location {seq_dir}.")
    scores = write_mpnn_fastas(seqs_dict)

    if return_seq_threaded_pdbs_as_pose:
        #replace .fa with sequence threaded pdb files as poses
        scores['location'] = [os.path.join(pdb_dir, f"{os.path.splitext(os.path.basename(series['location']))[0]}.pdb") for _, series in scores.iterrows()]

    if pack_sidechains:
        pack_dir = os.path.join(work_dir, 'packed')
        pack_fl = glob(f"{pack_dir}/*_1.pdb")
        if not pack_fl:
            raise FileNotFoundError(f"No .pdb files were found in the output directory of LigandMPNN {pack_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")
        for pdb in pack_fl:
            rename_packed_pdb(pdb)

        # extract only first replicate
        scores['location'] = [os.path.join(pack_dir, f"{os.path.splitext(os.path.basename(series['location']))[0]}.pdb") for _, series in scores.iterrows()]

    if not preserve_original_output:
        if os.path.isdir(original_seqs_dir):
            logging.info(f"Deleting original .fa files at {original_seqs_dir}!")
            shutil.rmtree(original_seqs_dir)
        if os.path.isdir(original_pdbs_dir):
            logging.info(f"Deleting original .pdb files at {original_pdbs_dir}!")
            shutil.rmtree(original_pdbs_dir)

    return scores

def parse_residues(residues:object) -> str:
    """
    Parses residues from either ResidueSelection object, list, or MPNN-formatted string into MPNN-formatted string.

    This function converts the input residues into a format compatible with MPNN. It supports conversion from ResidueSelection objects, comma-separated strings, and lists of residues.

    Parameters:
        residues (object): The input residues to be parsed. This can be a ResidueSelection object, a comma-separated string, or a list of residues.

    Returns:
        str: The residues formatted as a string compatible with MPNN.

    Raises:
        ValueError: If the input type is not supported (i.e., not a str or ResidueSelection).

    Examples:
        Here is an example of how to use the `parse_residues` function:

        .. code-block:: python

            from protflow.residues import ResidueSelection

            # Example ResidueSelection object
            residues = ResidueSelection(["A:10", "A:20"])

            # Parse residues
            parsed_residues = parse_residues(residues)
            print(parsed_residues)  # Outputs: "A:10 A:20"

            # Example string input
            residues_str = "A:10,A:20"
            parsed_residues = parse_residues(residues_str)
            print(parsed_residues)  # Outputs: "A:10 A:20"

    Further Details:
        - **ResidueSelection Object:** The function calls the `to_string` method of the ResidueSelection object to get the MPNN-formatted string.
        - **String Input:** For comma-separated string inputs, the function splits the string by commas and joins the parts with spaces.
    """
    # ResidueSelection should have to_mpnn function.
    if isinstance(residues, ResidueSelection):
        return residues.to_string(delim=" ")

    # strings:
    if isinstance(residues, str):
        if len(residues.split(",")) > 1:
            return " ".join(residues.split(","))
        return residues
    raise ValueError(f"Residues must be of type str or ResidueSelection. Type: {type(residues)}")

def write_to_json(input_dict: dict, output_path:str) -> str:
    '''Writes json serializable :input_dict: into file and returns path to file. Returns path to json file :output_path:'''
    with open(output_path, 'w', encoding="UTF-8") as f:
        json.dump(input_dict, f)
    return output_path

def create_distance_conservation_bias_cmds(poses: Poses, prefix: str, center: Union[str,ResidueSelection], shell_distances: list = [10, 15, 20, 1000], shell_biases: list = [0, 0.25, 0.5, 1], center_atoms: list[str] = None, noncenter_atoms: list[str] = ["CA"], jobstarter: JobStarter = None, overwrite: bool = False) -> Poses:
    """
    Creates distance-based conservation bias commands for LigandMPNN runs and saves them in a poses DataFrame column.

    This function creates commands for conservation bias based on shells with a distance from a given ResidueSelection.

    Parameters:
        poses (Poses): The Poses object containing the protein structures.
        prefix (str): A prefix used as output folder and column name in the poses DataFrame to save the commands.
        center (str or ResidueSelection): The center of the shells. Can be either a single ResidueSelection or a poses DataFrame column containing ResidueSelections.
        shell_distances (list, optional): The shells for creating conservation bias. The numbers represent the distance from the center. Defaults to [10, 15, 20, 100].
        shell_biases (list, optional): The strength of the bias for each shell. Defaults to [0, 0.25, 0.5, 1].
        center_atoms (list, optional): The atom names of the center ResidueSelection which should be used for shell distance calculations. None means all atoms are selected. Defaults to None.
        noncenter_atoms (list, optional): The atom names of noncenter residues which should be used for shell distance calculations. None means all atoms are selected. Defaults to ["CA"].
        jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
        overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

    Returns:
        Poses: The updated Poses object containing the commands for conservation bias in a poses DataFrame column.

    Raises:
        KeyError: If shell_distances are not sorted in ascending order.

    Examples:
        Here is an example of how to use the `create_distance_conservation_bias_cmds` method:

        .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import LocalJobStarter
            from protflow.residue_selectors import ResidueSelection
            from ligandmpnn import create_distance_conservation_bias_cmds

            # Create instances of necessary classes
            poses = Poses(poses=".", glob_suffix="*.pdb)
            jobstarter = LocalJobStarter()
            central_selection = ResidueSelection("A23")

            # Run the diffusion process
            poses = create_distance_conservation_bias_cmds(
                poses=poses,
                prefix="prefix",
                prefix="conservation_bias_cmd",
                jobstarter=jobstarter,
                center=central_selection,
            )

            # Access and process the results
            print(poses.df["prefix"])

    Further Details:
        - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
        - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
        - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the process to their specific needs.

    This method is designed to streamline the creation of distance-based conservation bias commands  for LigandMPNN within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein design simulations.
    """

    def create_bias_dict(resdict: dict, bias: float):
        bias_dict = {}
        for res, idx in resdict.items():
            bias_dict[res] = {idx: bias}
        return bias_dict

    def combine_dicts(dict_list: list[dict]):
        out_dict = {}
        for in_dict in dict_list:
            out_dict.update(in_dict)
        return out_dict

    from protflow.tools.residue_selectors import DistanceSelector
    from protflow.metrics.selection_identity import SelectionIdentity

    # check input
    if not shell_distances == sorted(shell_distances):
        raise KeyError(f"shell_distances must be in ascending order like {sorted(shell_distances)}, not {shell_distances}!")

    # set python path
    python_path = os.path.join(PROTFLOW_ENV, "python")

    # create output directory
    os.makedirs(working_dir := os.path.abspath(os.path.join(poses.work_dir, prefix)), exist_ok=True)
    original_work_dir = poses.work_dir
    poses.set_work_dir(working_dir)

    # initialize residue selector and id metric
    selector = DistanceSelector(center=center)
    selid = SelectionIdentity(python_path=python_path, jobstarter=jobstarter, overwrite=overwrite)

    # iterate over all shell distances
    for index, (dist, bias) in enumerate(zip(shell_distances, shell_biases)):
        # select residues in shell
        selector.select(prefix=f"{prefix}_selection_{dist}", poses=poses, distance=dist, operator="<=", center_atoms=center_atoms, noncenter_atoms=noncenter_atoms, include_center=False)
        if index == 0:
            poses.df[f"{prefix}_selected_residues"] = poses.df[f"{prefix}_selection_{dist}"]
        else:
            # subtract previous selections in shells that are not the innermost shell
            poses.df[f"{prefix}_selection_{dist}"] = poses.df[f"{prefix}_selection_{dist}"] - poses.df[f"{prefix}_selected_residues"]
            # add current selection to overall selection
            poses.df[f"{prefix}_selected_residues"] = poses.df[f"{prefix}_selected_residues"] + poses.df[f"{prefix}_selection_{dist}"]

        # determine residue ids
        selid.run(poses=poses, prefix=f"{prefix}_selection_{dist}_ids", residue_selection=f"{prefix}_selection_{dist}", onelettercode=True)

        # create bias dictionary
        poses.df[f"{prefix}_{dist}_bias_dicts"] = poses.df.apply(lambda row: create_bias_dict(row[f"{prefix}_selection_{dist}_ids_selection_identities"], bias), axis=1)

    # write bias dict for all shells
    poses.df[f"{prefix}_overall_bias_dict"] = poses.df.apply(lambda row: combine_dicts([row[f"{prefix}_{dist}_bias_dicts"] for dist in shell_distances]), axis=1)

    # write json files for each dict
    os.makedirs(dict_dir := os.path.join(working_dir, "bias_dicts"), exist_ok=True)
    dict_paths = []
    for _, row in poses.df.iterrows():
        with open(dict_path := os.path.join(dict_dir, f"{row['poses_description']}_bias_dict.json"), 'w', encoding="UTF-8") as f:
            json.dump(row[f"{prefix}_overall_bias_dict"], f, indent=4)
        dict_paths.append(dict_path)

    # save paths to json files in poses dataframe
    poses.df[f"{prefix}_overall_bias_json"] = dict_paths

    # save cmds for LigandMPNN in poses dataframe
    poses.df[f"{prefix}"] = [f"--bias_AA_per_residue {dict_path}" for dict_path in dict_paths]

    # clean dataframe
    cols_to_drop = [
        f"{prefix}_selection_{dist}" for dist in shell_distances
    ] + [
        f"{prefix}_selection_{dist}_ids_description" for dist in shell_distances
    ] + [
        f"{prefix}_selection_{dist}_ids_selection_identities" for dist in shell_distances
    ] + [
        f"{prefix}_selection_{dist}_ids_location" for dist in shell_distances
    ] + [
        f"{prefix}_{dist}_bias_dicts" for dist in shell_distances] + [f"{prefix}_selected_residues"]
    poses.df.drop(cols_to_drop, axis=1, inplace=True)

    # revert to original work dir
    poses.set_work_dir(original_work_dir)

    return poses
