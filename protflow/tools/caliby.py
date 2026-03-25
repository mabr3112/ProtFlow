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
import os
import logging
import shutil
import re
from glob import glob

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path
from ..residues import ResidueSelection
from ..poses import Poses, description_from_path
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, col_in_df, options_flags_to_string, prepend_cmd
from ..utils.openbabel_tools import openbabel_fileconverter

class CalibySequenceDesign(Runner):
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
    def __init__(self, caliby_dir: str = None, python_path: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:
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
        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")
        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/seq_des.py")
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_cfg = os.path.join(self.caliby_dir, "caliby/configs/seq_des/atom_mpnn_inference.yaml")

        # setup runner
        self.name = "caliby.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "caliby.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, nseq: int = 1, model: str = "caliby", omit_aas: str|list = None, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None, pos_constraint_csv: str = None, return_seq_threaded_pdbs_as_pose: bool = False, options: str = None, convert_cif_to_pdb: bool = True, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        Execute the LigandMPNN process with given poses and jobstarter configuration.

        This method sets up and runs the LigandMPNN process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            nseq (int, optional): The number of sequences to generate for each input pose. Defaults to 1.
            model_type (str, optional): The type of model to use. Defaults to 'caliby'.
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

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"caliby_seq_des_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # check for pos_constraint_csv file
        if pos_constraint_csv and not os.path.isfile(pos_constraint_csv):
            raise ValueError(f"<pos_constraint_csv> must specify the path to a single csv file. Could not find a file at {pos_constraint_csv}.")

        # convert omit_aas string to list
        if omit_aas and isinstance(omit_aas, str):
            omit_aas = [aa for aa in omit_aas]

        if not os.path.isfile(model) and not os.path.isfile(model_path := os.path.join(self.caliby_dir, "model_params", "caliby", f"{model}.ckpt")):
            raise FileNotFoundError(f"Could not detect a model at {model} or at {model_path}.")

        opt_dict = self._parse_caliby_opts(options)
        opt_dict["sampling_cfg_overrides.num_seqs_per_pdb"] = nseq
        opt_dict["ckpt_name_or_path"] = model if os.path.isfile(model) else model_path
        opt_dict["seq_des_cfg.atom_mpnn.sampling_cfg"] = self.sampling_cfg # TODO: this is a hack so caliby does not crash when running outside of installation dir, there might be better ways to solve this

        if omit_aas:
            opt_dict["omit_aas"] = omit_aas
        if pos_constraint_csv:
            opt_dict["pos_constraint_csv"] = os.path.abspath(pos_constraint_csv)
        else:
            if "pos_constraint_csv" in opt_dict and any([fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col]):
                raise ValueError("Pose-specific constraints cannot be set if a pregenerated pos_constraints_csv is provided!")

            opt_dict["pos_constraint_csv"] = self.create_constraint_csv(poses, work_dir, fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col)
        
        # define number of batches
        if num_batches:
            num_batches = min([len(poses.poses_list()), num_batches])
        else:
            num_batches = min([len(poses.poses_list()), jobstarter.max_cores])

        # setup for batch mode
        batch_opts = self._setup_batch_mode(pose_paths=poses.poses_list(), options=opt_dict, num_batches=num_batches, work_dir=work_dir)

        # write caliby cmds:
        cmds = [self.write_cmd(options=opt_dict) for opt_dict in batch_opts]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="caliby_seqdes",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose,
        )

        if len(scores.index) < len(poses.df.index) * nseq:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def create_constraint_csv(self, poses: Poses, work_dir: str, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None) -> str:
        cst_csv = pd.DataFrame({"pdb_key": [description_from_path(pose) for pose in poses.poses_list()]})
        
        if fixed_pos_seq_col:
            col_in_df(poses.df, fixed_pos_seq_col)
            cst_csv["fixed_pos_seq"] = [sele.to_string() if isinstance(sele, ResidueSelection) else sele for sele in poses.df[fixed_pos_seq_col].to_list()]
        if fixed_pos_scn_col:
            # TODO: add fixed_pos_scn to fixed_pos_seq if missing, but tricky if fixed_pos_seq used description like 'A5-15' and not explicit list
            col_in_df(poses.df, fixed_pos_scn_col)
            cst_csv["fixed_pos_scn"] = [sele.to_string() if isinstance(sele, ResidueSelection) else sele for sele in poses.df[fixed_pos_seq_col].to_list()]
        if fixed_pos_override_seq_col:
            col_in_df(poses.df, fixed_pos_override_seq_col)
            cst_csv["fixed_pos_override_seq"] = poses.df[fixed_pos_override_seq_col]
        if pos_restrict_aatype_col:
            col_in_df(poses.df, pos_restrict_aatype_col)
            cst_csv["pos_restrict_aatype"] = poses.df[pos_restrict_aatype_col]
        if symmetry_pos_col:
            col_in_df(poses.df, symmetry_pos_col)
            cst_csv["symmetry_pos"] = poses.df[symmetry_pos_col]

        cst_csv = cst_csv.fillna('')

        cst_csv.to_csv(out := os.path.join(work_dir, "pos_constraints.csv"), index=False)
        return os.path.abspath(out)

    def _setup_batch_mode(self, pose_paths: list[str], options: dict, num_batches: int, work_dir: str) -> list:

        def same_folder_check(file_paths):
            # Extract the absolute directory path for each file and put them in a set
            directories = {os.path.dirname(os.path.abspath(p)) for p in file_paths}
    
            # If all files are in the same folder, the set will only have 1 unique item
            return directories

        def write_input_list(pose_paths: list, filename: str):
            with open(filename, "w+") as f:
                f.write("\n".join([os.path.basename(pose) for pose in pose_paths]))
            return


        in_folders = same_folder_check(pose_paths)

        # check if input files are all in same folder, otherwise copy to new folder
        if len(in_folders) > 1:
            os.makedirs(input_dir := os.path.join(work_dir, "input"), exist_ok=True)
            updated_paths = []
            for pose in pose_paths:
                shutil.copy(pose, new_path := os.path.join(input_dir, os.path.basename(pose)))
                updated_paths.append(new_path)
            options["input_cfg.pdb_dir"] = input_dir
        else:
            updated_paths = pose_paths
            options["input_cfg.pdb_dir"] = list(in_folders)[0]

        # split poses into batches
        pose_batches = split_list(updated_paths, n_sublists=num_batches)

        os.makedirs(input_list_dir := os.path.join(work_dir, "input_lists"), exist_ok=True)

        batch_opt_list = []
        for i, batch in enumerate(pose_batches):
            list_path = os.path.join(input_list_dir, f"in_{i}.txt")
            write_input_list(batch, list_path)
            batch_opts = options.copy()
            batch_opts["input_cfg.pdb_name_list"] = list_path
            batch_opts["out_dir"] = os.path.join(work_dir, f"batch_{i}")

            batch_opt_list.append(batch_opts)

        return batch_opt_list


    def _parse_caliby_opts(self, options: str = None) -> dict:

        def re_split(command: str) -> list:
            # Return empty list if the string is empty
            if not command.strip():
                return []
            pattern = r'\s+(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
            return re.split(pattern, command)

        if not options:
            return {}

        raw_splits = re_split(options)
        
        parsed_config = {}
        for item in raw_splits:
            if "=" in item:
                key, value = item.split("=", 1)
                parsed_config[key] = value.strip("'\"")
                
        return parsed_config

    def write_cmd(self, options: dict) -> str:
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
        # convert to string
        options = options_flags_to_string(options, None, sep="")

        return f"{self.python_path} {self.script_path} {options}"


def collect_scores(work_dir: str, return_seq_threaded_pdbs_as_pose: bool = False, cif_to_pdb: bool = True) -> pd.DataFrame:
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
    def write_fasta(seq, name, path):
        with open(path, "w+") as f:
            f.write(f">{name}\n{seq}")
        return path

    def convert_cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return output
    
    # read .csv files
    csvs = glob(os.path.join(work_dir, "batch_*/seq_des_outputs.csv"))
    data = pd.concat([pd.read_csv(csv) for csv in csvs])
    data.reset_index(drop=True, inplace=True)


    if not return_seq_threaded_pdbs_as_pose:
        os.makedirs(fasta_dir := os.path.join(work_dir, "fasta"), exist_ok=True)
        data["location"] = data.apply(lambda row: write_fasta(seq=row["seq"], name=description_from_path(row["out_pdb"]), path=os.path.join(fasta_dir, f"{description_from_path(row['out_pdb'])}.fasta")), axis=1)
    
    elif cif_to_pdb:
        os.makedirs(pdb_dir := os.path.join(work_dir, "converted"), exist_ok=True)
        data["location"] = data.apply(lambda row: convert_cif_to_pdb(input_cif=row["out_pdb"], output_format="pdb", output=os.path.join(pdb_dir, f"{description_from_path(row['out_pdb'])}.pdb")), axis=1)
    
    else:
        data["location"] = [os.path.abspath(path) for path in data["out_pdb"].to_list()]
        data.drop(["out_pdb"], axis=1, inplace=True)
    
    data["description"] = [description_from_path(path) for path in data["location"].to_list()]

    return data

