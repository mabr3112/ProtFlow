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
from glob import glob

# dependencies
import pandas as pd

# custom
import protflow.config
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.runners import Runner, RunnerOutput, parse_generic_options, options_flags_to_string
from protflow.tools.protein_edits import ChainRemover
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile

class GNINA(Runner):
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
    def __init__(self, script_path:str=protflow.config.GNINA_PATH, jobstarter:JobStarter=None) -> None:
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


        self.script_path = self.search_path(script_path, "GNINA_PATH")
        self.name = "gnina.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "ligandmpnn.py"

    def run(self, poses: Poses, prefix: str, options: str = None, pose_options: object = None, ligand_chain: str = None, overwrite: bool = False, jobstarter: JobStarter = None, ) -> Poses:
        """
        Execute the LigandMPNN process with given poses and jobstarter configuration.

        This method sets up and runs the LigandMPNN process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            nseq (int, optional): The number of sequences to generate for each input pose. Defaults to None.
            model_type (str, optional): The type of model to use. Defaults to 'ligand_mpnn'.
            options (str, optional): Additional options for the LigandMPNN script. Defaults to None.
            pose_options (object, optional): Pose-specific options for the LigandMPNN script. Defaults to None.
            fixed_res_col (str, optional): Column name in the poses DataFrame specifying fixed residues. Defaults to None.
            design_res_col (str, optional): Column name in the poses DataFrame specifying residues to be redesigned. Defaults to None.
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
        scorefile = os.path.join(work_dir, f"ligandmpnn_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # parse pose_options
        pose_options = self.prep_pose_options(poses, pose_options)

        if ligand_chain:
            self.prepare_ligand_autobox(poses, ligand_chain=ligand_chain, prefix=prefix)
            ligand_chain_opts = [f"--autobox_ligand {ligand} --ligand {ligand}" for ligand in poses.df[f"{prefix}_ligand"]]
        else:
            ligand_chain_opts = [None for _ in poses.poses_list()]

       # combine pose_options and ligand_chain options (priority goes to ligand_chain):
        pose_options = [options_flags_to_string(*parse_generic_options(pose_opt, pose_opt_cols_opt, sep="--"), sep="--") for pose_opt, pose_opt_cols_opt in zip(pose_options, ligand_chain_opts)]

        # define docked ligand output directory
        ligand_out_dir = os.path.join(work_dir, "docked_ligands")
        os.makedirs(ligand_out_dir, exist_ok=True)

        # write ligandmpnn cmds:
        cmds = [self.write_cmd(pose, output_dir=ligand_out_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.poses_list(), pose_options)]

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="gnina",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(work_dir=work_dir)

        separate_ligands_dir = os.path.join(work_dir, "separate_ligands")
        ligands = glob(os.path.join(ligand_out_dir, "*.pdb"))
        descriptions, paths = [], []
        for ligand in ligands:
            lig = load_structure_from_pdbfile(ligand, all_models=True)
            for index, model in enumerate(lig):
                filename = f"{os.path.splitext(os.path.basename(ligand))[0]}_{str(index).zfill(4)}.pdb"
                save_structure_to_pdbfile(model, save_path=os.path.join(separate_ligands_dir, filename))
                paths.append(filename)
                descriptions.append(os.path.splitext(os.path.basename(ligand))[0])

        ligand_df = pd.DataFrame({"ligand_path": paths, "poses_description": descriptions})

        input_df = poses.df[["poses", "poses_description"]]
        input_df = input_df.merge(ligand_df, on="poses_description")

        output_dir = os.path.join(work_dir, "output_pdbs")
        os.makedirs(output_dir, exist_ok=True)





        if len(scores.index) < len(poses.df.index) :
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def prepare_ligand_autobox(self, poses: Poses, ligand_chain, prefix):
        ChainRemover(jobstarter=protflow.jobstarters.LocalJobStarter())
        original_work_dir = poses.work_dir
        new_work_dir = os.path.join(poses.work_dir, prefix)
        poses.set_work_dir(new_work_dir)
        poses.df[f"{prefix}_input_location"] = poses.df["poses"]
        poses = ChainRemover.run(poses=poses, prefix=f"{prefix}_ligand", preserve_chains=ligand_chain)
        poses.df["poses"] = poses.df[f"{prefix}_input_location"]
        poses = ChainRemover.run(poses=poses, prefix=f"{prefix}_noligand", chains=ligand_chain)
        poses.set_work_dir(original_work_dir)
        return poses

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
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
        # parse options
        opts, flags = parse_generic_options(options, pose_options)

        # check if interfering options were set
        forbidden_options = ['r', 'receptor', 'log' 'o', 'out']
        if opts and any(opt in opts for opt in forbidden_options):
            raise KeyError(f"options and pose_options must not contain any of {forbidden_options}")

        # convert to string
        options = options_flags_to_string(opts, flags, sep="--")

        out_file = os.path.join(output_dir, os.path.basename(pose_path))
        out_log = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.log")

        # write command and return.
        return f"{self.script_path} --receptor {pose_path} --out {out_file} --log {out_log} {options}"

def collect_scores(work_dir:str, return_seq_threaded_pdbs_as_pose:bool, preserve_original_output:bool=True) -> pd.DataFrame:
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
    def extract_gnina_table(file_path):

        description = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r', encoding="UTF-8") as file:
            lines = file.readlines()

        # Find the start of the table
        table_start = None
        for i, line in enumerate(lines):
            if line.startswith('mode |'):
                table_start = i + 2
                break
        
        if table_start is None:
            raise ValueError(f"Table not found in {file_path}")
        
        # Read the table lines
        table_lines = lines[table_start:]
        
        # Parse the table lines
        table = {
            'name': [],
            'rank': [],
            'affinity': [],
            'intramol': [],
            'CNN_pose_score': [],
            'CNN_affinity': []
            }
        
        for line in table_lines:
            # Skip empty lines and lines that don't contain data
            if line.strip() == '' or line.startswith('-----'):
                continue
            
            # Split the line into columns
            columns = line.split()
            table['name'].append(f"{description}_{columns[0].zfill(4)}")
            table['rank'].append(int(columns[0]))
            table['affinity'].append(float(columns[1]))
            table['intramol'].append(float(columns[2]))
            table['CNN_pose_score'].append(float(columns[3]))
            table['CNN_affinity'].append(float(columns[4]))

        df = pd.DataFrame(table)
        return df


    # read .pdb files
    scorefiles = glob(f"{work_dir}/*.score")
    if not scorefiles:
        raise FileNotFoundError(f"No .score files were found in the output directory of gnina {work_dir}. Gnina might have crashed (check output log), or path might be wrong!")

    scores = pd.concat([extract_gnina_table(scorefile) for scorefile in scorefiles]).reset_index(drop=True)


    separate_ligands_dir = os.path.join(work_dir, "separate_ligands")
    ligands = glob(os.path.join(ligand_out_dir, "*.pdb"))
    descriptions, paths = [], []
    for ligand in ligands:
        lig = load_structure_from_pdbfile(ligand, all_models=True)
        for index, model in enumerate(lig):
            filename = f"{os.path.splitext(os.path.basename(ligand))[0]}_{str(index+1).zfill(4)}.pdb"
            save_structure_to_pdbfile(model, save_path=os.path.join(separate_ligands_dir, filename))
            paths.append(filename)
            descriptions.append(os.path.splitext(os.path.basename(ligand))[0])
    
    ligand_df = pd.DataFrame({"ligand_path": paths, "poses_description": descriptions})
        
    input_df = poses.df[["poses", "poses_description"]]
    input_df = input_df.merge(ligand_df, on="poses_description")

    output_dir = os.path.join(work_dir, "output_pdbs")
    os.makedirs(output_dir, exist_ok=True)

    return scores