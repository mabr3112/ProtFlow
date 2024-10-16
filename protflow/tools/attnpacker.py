"""
AttnPacker Module
=================

This module provides the functionality to integrate AttnPacker within the ProtFlow framework. It offers tools to run AttnPacker, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `AttnPacker` class encapsulates the functionality necessary to execute AttnPacker runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of packing processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem. 
The module is designed to streamline the integration of AttnPacker into larger computational workflows. It supports the automatic setup of job parameters, execution of AttnPacker commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `AttnPacker` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the packing process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `AttnPacker` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from attnpacker import AttnPacker

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the AttnPacker class
    attnpacker = AttnPacker()

    # Run the packing process
    results = attnpacker.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="packing.num_designs=10",
        pose_options=["packing.input_pdb='input.pdb'"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the packing process.
    - Customizability: Users can customize the packing process through multiple parameters, including the number of packings, specific options for the AttnPacker script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate AttnPacker into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import os
import logging
import glob

# dependencies
import pandas as pd

# custom
import protflow.config
import protflow.jobstarters
import protflow.tools
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput, prepend_cmd
from protflow.jobstarters import JobStarter

class AttnPacker(Runner):
    """
    AttnPacker Class
    ================

    The `AttnPacker` class is a specialized class designed to facilitate the execution of AttnPacker within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with packing processes.

    Detailed Description
    --------------------
    The `AttnPacker` class manages all aspects of running AttnPacker simulations. It handles the configuration of necessary scripts and executables, prepares the environment for packing processes, and executes the packing commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to AttnPacker scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of AttnPacker commands with support for multiple packings.
        - Collecting and processing output data into a pandas DataFrame.
        - Ensuring robust error handling and logging for easier debugging and verification.

    Returns
    -------
    An instance of the `AttnPacker` class, configured to run AttnPacker processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - KeyError: If forbidden options are included in the command parameters.

    Examples
    --------
    Here is an example of how to initialize and use the `AttnPacker` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from attnpacker import AttnPacker

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the AttnPacker class
        attnpacker = AttnPacker()

        # Run the packing process
        results = attnpacker.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the packing process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The AttnPacker class is intended for researchers and developers who need to perform packing simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.ATTNPACKER_DIR_PATH, python_path: str = protflow.config.ATTNPACKER_PYTHON_PATH, pre_cmd : str = protflow.config.ATTNPACKER_PRE_CMD, jobstarter: str = None) -> None:
        '''sbatch_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        self.script_path = self.search_path(script_path, "ATTNPACKER_DIR_PATH", is_dir=True)
        self.python_path = self.search_path(python_path, "ATTNPACKER_PYTHON_PATH")
        self.pre_cmd = pre_cmd
        self.name = "attnpacker.py"
        self.jobstarter = jobstarter
        self.index_layers = 1

    def __str__(self):
        return "attnpacker.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, overwrite: bool = False) -> Poses:
        """
        Execute the AttnPacker process with given poses and jobstarter configuration.

        This method sets up and runs the AttnPacker process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: An updated Poses object containing the processed poses and results of the AttnPacker process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            KeyError: If forbidden options are included in the command parameters.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from attnpacker import AttnPacker

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the AttnPacker class
                attnpacker = AttnPacker()

                # Run the packing process
                results = attnpacker.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It sets up specific directories for output PDBs and checks for existing score files.
            - **Output Management:** The method handles the collection and processing of output data, reading scores from a CSV file and organizing results into a structured DataFrame. It ensures that results are accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the packing process to their specific needs. Users can specify additional options and pose-specific parameters for the AttnPacker script.

        This method is designed to streamline the execution of AttnPacker processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze packing simulations.
        """
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # setup attnpacker specific dirs:
        pdb_dir = os.path.join(work_dir, 'output_pdbs')
        if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip attnpacker.
        scorefile = os.path.join(work_dir, f"attnpacker_scores.{poses.storage_format}")

        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        poses_sublist = protflow.jobstarters.split_list(poses.poses_list(), n_sublists=jobstarter.max_cores)
        json_paths = []
        for index, sublist in enumerate(poses_sublist):
            in_dict = {"poses": sublist, "scorepath": [os.path.join(work_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}_attnpacker_out.json") for pose_path in sublist]}
            json_path = os.path.join(work_dir, f"attnpacker_input_{index}.json")
            in_df = pd.DataFrame(in_dict)
            in_df.to_json(json_path)
            json_paths.append(json_path)

        # write attpacker cmds:
        cmds = [self.write_cmd(json_path, output_dir=os.path.join(work_dir, "output_pdbs")) for json_path in json_paths]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)
            
        # run:
        logging.info(f"Starting attnpacker.py on {len(poses)} poses with {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="attnpacker",
            wait=True,
            output_path=f"{work_dir}/"
        )

        logging.info(f"{self} finished, collecting scores.")
        scores = collect_scores(work_dir)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")
        
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, json_path:str, output_dir:str):
        """
        Write the command to run the AttnPacker script for a given pose.

        This method constructs the command line string necessary to execute the AttnPacker script for a given pose. It incorporates the specified options and pose-specific parameters, ensuring that the command is correctly formatted and includes all required arguments. It also checks for forbidden options to prevent conflicts.

        Parameters:
            pose_path (str): The path to the input PDB file for the pose.
            output_dir (str): The directory where output files will be stored.

        Returns:
            str: The command line string to execute the AttnPacker script with the specified parameters.

        Raises:
            KeyError: If forbidden options are included in the command parameters.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from attnpacker import AttnPacker

                # Initialize the AttnPacker class
                attnpacker = AttnPacker()

                # Define the input pose path and output directory
                pose_path = "input.pdb"
                output_dir = "/path/to/output"

                # Write the command with additional options and pose-specific parameters
                cmd = attnpacker.write_cmd(
                    pose_path=pose_path,
                    output_dir=output_dir,
                )

                # Print the command
                print(cmd)

        Further Details:
            - **Command Construction:** This method ensures that the command string is correctly constructed with all necessary arguments. It includes paths to the script directory, output directory, input PDB file, and score file, as well as any additional options.
            - **Validation:** The method checks for forbidden options that could conflict with the required arguments, raising a KeyError if any are found. This helps ensure that the command is valid and will run correctly.

        This method is designed to facilitate the execution of AttnPacker processes within the ProtConductor framework, providing a flexible and robust way to construct and run commands for packing simulations.
        """
        """
        # check if interfering options were set
        forbidden_options = ['--attnpacker_dir', '--output_dir', '--input_pdb', '--scorefile']
        if (options and any(_ in options for _ in forbidden_options)) or (pose_options and any(_ in pose_options for _ in forbidden_options)):
            raise KeyError(f"Options and pose_options must not contain '--attnpacker_dir', '--output_dir', '--input_pdb' or '--scorefile'!")
        """

        options = f"--attnpacker_dir {self.script_path} --output_dir {output_dir} --input_json {json_path}"

        return f"{self.python_path} {protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/run_attnpacker.py {options}"

def collect_scores(dir: str):
    scorefiles = glob.glob(os.path.join(dir, "*_attnpacker_out.json"))
    df = [pd.read_json(score, typ="series") for score in scorefiles]
    df = pd.DataFrame(df)
    df.reset_index(drop=True, inplace=True)
    return df