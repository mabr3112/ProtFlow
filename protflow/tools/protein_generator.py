"""
ProteinGenerator Module
=======================

This module provides the functionality to integrate the protein generation process within the ProtFlow framework. It offers tools to run the protein_generator script, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `ProteinGenerator` class encapsulates the functionality necessary to run protein_generator as described in the publication. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of protein generation processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of protein generation into larger computational workflows. It supports the automatic setup of job parameters, execution of protein generator commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ProteinGenerator` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the protein generation process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ProteinGenerator` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from protein_generator import ProteinGenerator

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the ProteinGenerator class
    protein_generator = ProteinGenerator()

    # Run the protein generation process
    results = protein_generator.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="generation.num_proteins=10",
        pose_options=["generation.input_pdb='input.pdb'"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the protein generation process.
    - Customizability: Users can customize the protein generation process through multiple parameters, including the number of generated proteins, specific options for the protein generator script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate protein generation into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import numpy as np

# dependencies
import pandas as pd

# custom
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.runners import Runner, RunnerOutput, parse_generic_options
from .. import config

class ProteinGenerator(Runner):
    """
    ProteinGenerator Class
    ======================

    The `ProteinGenerator` class is a specialized class designed to facilitate the execution of protein generation within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with protein generation processes.

    Detailed Description
    --------------------
    The `ProteinGenerator` class manages all aspects of running protein generation simulations. It handles the configuration of necessary scripts and executables, prepares the environment for protein generation processes, and executes the generation commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to protein generation scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of protein generation commands with support for multiple generation runs.
        - Collecting and processing output data into a pandas DataFrame.
        - Ensuring robust error handling and logging for easier debugging and verification of the generation process.

    Returns
    -------
    An instance of the `ProteinGenerator` class, configured to run protein generation processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If the types of arguments are not as expected.

    Examples
    --------
    Here is an example of how to initialize and use the `ProteinGenerator` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from protein_generator import ProteinGenerator

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ProteinGenerator class
        protein_generator = ProteinGenerator()

        # Run the protein generation process
        results = protein_generator.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            options="generation.num_proteins=10",
            pose_options=["generation.input_pdb='input.pdb'"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the protein generation process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ProteinGenerator class is intended for researchers and developers who need to perform protein generation as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path:str=config.PROTEIN_GENERATOR_SCRIPT_PATH, python_path:str=config.PROTEIN_GENERATOR_PYTHON_PATH, jobstarter:JobStarter=None) -> None:
        """
        Initialize the ProteinGenerator class with paths to the necessary scripts and Python executable.

        This constructor sets up the `ProteinGenerator` class, configuring the paths to the protein generator script and the Python executable. It also sets the jobstarter and initializes essential attributes for the class.

        Parameters:
            script_path (str, optional): The path to the protein generator script. Defaults to the value set in config.PROTEIN_GENERATOR_SCRIPT_PATH.
            python_path (str, optional): The path to the Python executable. Defaults to the value set in config.PROTEIN_GENERATOR_PYTHON_PATH.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Raises:
            ValueError: If no script_path is provided or set in the configuration.

        Examples:
            Here is an example of how to initialize the `ProteinGenerator` class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from protein_generator import ProteinGenerator

                # Initialize the JobStarter class
                jobstarter = JobStarter()

                # Initialize the ProteinGenerator class
                protein_generator = ProteinGenerator(
                    script_path="/path/to/protein_generator.py",
                    python_path="/path/to/python",
                    jobstarter=jobstarter
                )

                print(protein_generator)

        Further Details:
            - **Script Path:** The path to the protein generator script is a critical configuration that needs to be set either through the parameter or the configuration file.
            - **Python Path:** The path to the Python executable is necessary for running the script and should be set accordingly.
            - **JobStarter:** If provided, the JobStarter instance is used to manage job execution, otherwise it can be set later.
        """
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under PROTEIN_GENERATOR_SCRIPT_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "protein_generator.py"
        self.jobstarter = jobstarter
        self.index_layers = 1

    def __str__(self):
        return "protein_generator.py"

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter, options:str=None, pose_options:str=None, overwrite:bool=False) -> RunnerOutput:
        """
        Execute protein_generator with given poses and jobstarter configuration.
        This method sets up and runs protein_generator using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter): An instance of the JobStarter class, which manages job execution.
            options (str, optional): Additional options for the protein generation script. Defaults to None.
            pose_options (list[str], optional): A list of pose-specific options for the protein generation script. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the protein generation process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If pose_options are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from protein_generator import ProteinGenerator

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the ProteinGenerator class
                protein_generator = ProteinGenerator()

                # Run the protein generation process
                results = protein_generator.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    options="generation.num_proteins=10",
                    pose_options=["generation.input_pdb='input.pdb'"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor protein_generator to their specific needs.

        This method is designed to streamline the execution of protein generation processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein generation simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # setup protein_generator specific directories:
        if not os.path.isdir((pdb_dir := f"{work_dir}/output_pdbs/")):
            os.makedirs(pdb_dir, exist_ok=True)

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip protein_generator.
        scorefile = os.path.join(work_dir, f"protein_generator_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # parse_options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # write protein generator cmds:
        cmds = [self.write_cmd(pose, output_dir=pdb_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses, pose_options)]

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="protein_generator",
            wait=True,
            output_path=f"{pdb_dir}/"
        )

        # collect scores
        scores = self.collect_scores(scores_dir=pdb_dir)

        # write scorefile
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def _safecheck_pose_options(self, pose_options: list, poses:list) -> list:
        '''Checks if pose_options are of the same length as poses, if now pose_options are provided, '''
        # safety check (pose_options must have the same length as poses)
        if isinstance(pose_options, list):
            if len(poses) != len(pose_options):
                raise ValueError(f"Arguments <poses> and <pose_options> for RFDiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
            return pose_options
        elif pose_options is None:
            # make sure an empty list is passed as pose_options!
            return ["" for x in poses]
        else:
            raise TypeError(f"Unsupported type for pose_options: {type(pose_options)}. pose_options must be of type [list, None]")

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        """
        Write the command to run the protein_generator.py script with specified options and pose configurations.

        This method constructs the command string necessary to execute the protein generator script, incorporating specified options and pose-specific parameters. The generated command can be used to run the protein generation process in a computational environment.

        Parameters:
            pose_path (str): The file path to the input pose.
            output_dir (str): The directory where output files will be stored.
            options (str): Additional options for the protein generator script.
            pose_options (str): Specific options for the protein generator script related to the pose.

        Returns:
            str: The constructed command string to execute the protein generator script.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from protein_generator import ProteinGenerator

                # Initialize the ProteinGenerator class
                protein_generator = ProteinGenerator(
                    script_path="/path/to/protein_generator.py",
                    python_path="/path/to/python"
                )

                # Construct the command string
                cmd = protein_generator.write_cmd(
                    pose_path="input_poses/pose1.pdb",
                    output_dir="output_directory",
                    options="generation.num_proteins=10",
                    pose_options="generation.input_pdb='input.pdb'"
                )

                print(cmd)

        Further Details:
            - **Command Construction:** The method parses the input pose path to derive a description and combines it with provided options and flags to construct a command string.
            - **Options Parsing:** The options and pose_options parameters are parsed into a format compatible with the protein generator script.
            - **Output Directory:** The output directory is specified to ensure that generated files are stored in the appropriate location.

        """
        # parse description
        desc = pose_path.rsplit("/", maxsplit=1)[-1].lsplit(".", maxsplit=1)[0]

        # parse options
        opts, flags = parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {self.script_path} --out {output_dir}/{desc} {opts} {flags}"

    def collect_scores(self, scores_dir: str) -> pd.DataFrame:
        """
        Collect scores from the protein_generator output directory.

        This method reads the output .pdb files generated by the protein generator and parses the corresponding .trb files into a pandas DataFrame. It consolidates the scores from multiple files into a single DataFrame for further analysis.

        Parameters:
            scores_dir (str): The directory where the output files from the protein generator are stored.

        Returns:
            pd.DataFrame: A DataFrame containing the collected scores from the protein generator output files.

        Raises:
            FileNotFoundError: If no .pdb files are found in the specified output directory, indicating a possible issue with the protein generator execution or an incorrect path.

        Examples:
            Here is an example of how to use the `collect_scores` method:

            .. code-block:: python

                from protein_generator import ProteinGenerator

                # Initialize the ProteinGenerator class
                protein_generator = ProteinGenerator()

                # Collect scores from the output directory
                scores_df = protein_generator.collect_scores(scores_dir="output_directory")

                print(scores_df)

        Further Details:
            - **File Reading:** The method reads all .pdb files from the specified directory. If no .pdb files are found, a FileNotFoundError is raised.
            - **Data Parsing:** The method parses the corresponding .trb files for each .pdb file, converting them into pandas DataFrames and concatenating them into a single DataFrame.
            - **Output Organization:** The resulting DataFrame is organized and returned for further analysis, with all scores consolidated from the multiple output files.

        """
        # read .pdb files
        pl = glob(f"{scores_dir}/*.pdb")
        if not pl:
            raise FileNotFoundError(f"No .pdb files were found in the output directory of protein_generator {scores_dir}. protein_generator might have crashed (check output log), or path might be wrong!")

        # parse .trb-files into DataFrames
        df = pd.concat([self.parse_trbfile(p.replace(".pdb", ".trb")) for p in pl], axis=0).reset_index(drop=True)

        return df

    def parse_trbfile(self, trbfile: str) -> pd.DataFrame:
        """
        Read a protein_generator output .trb file and parse the scores into a pandas DataFrame.

        This method reads the specified .trb file, extracts relevant data, and organizes it into a pandas DataFrame. The data includes scores and various attributes related to the protein generation process.

        Parameters:
            trbfile (str): The file path to the .trb file generated by the protein generator.

        Returns:
            pd.DataFrame: A DataFrame containing parsed scores and attributes from the .trb file.

        Examples:
            Here is an example of how to use the `parse_trbfile` method:

            .. code-block:: python

                from protein_generator import ProteinGenerator

                # Initialize the ProteinGenerator class
                protein_generator = ProteinGenerator()

                # Parse the .trb file
                df = protein_generator.parse_trbfile(trbfile="output_directory/sample.trb")

                print(df)

        Further Details:
            - **File Reading:** The method uses numpy to load the .trb file, which is expected to be in a specific format.
            - **Data Extraction:** The method extracts various pieces of information from the .trb file, including description, location, lddt scores, sequence, and contigs.
            - **Data Formatting:** The extracted data is organized into a dictionary and converted into a pandas DataFrame for ease of use in further analysis.

        """
        trb = np.load(trbfile, allow_pickle=True)

        # expand collected data if needed:
        data_dict = {
            "description": trbfile.split("/")[-1].replace(".trb", ""),
            "location": trbfile.replace("trb", "pdb"),
            "lddt": [sum(trb["lddt"]) / len(trb["lddt"])],
            "perres_lddt": [trb["lddt"]],
            "sequence": trb["args"]["sequence"],
            "contigs": trb["args"]["contigs"],
            "inpaint_str": [trb["inpaint_str"].numpy().tolist()],
            "inpaint_seq": [trb["inpaint_seq"].numpy().tolist()]
        }
        return pd.DataFrame(data_dict)
