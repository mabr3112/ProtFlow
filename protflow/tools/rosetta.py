"""
Rosetta Module
==============

This module provides the functionality to integrate Rosetta within the ProtFlow framework. It offers tools to run various Rosetta applications, handle their inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `Rosetta` class encapsulates the functionality necessary to execute Rosetta runs. It manages the configuration of paths to essential scripts and executables, sets up the environment, and handles the execution of Rosetta processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.
The module is designed to streamline the integration of Rosetta into larger computational workflows. It supports the automatic setup of job parameters, execution of Rosetta commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `Rosetta` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the Rosetta process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `Rosetta` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from rosetta import Rosetta

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the Rosetta class
    rosetta = Rosetta()

    # Run the Rosetta process
    results = rosetta.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        rosetta_application="RosettaScripts",
        nstruct=10,
        options="",
        pose_options=["-parser:protocol my_protocol.xml"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as invalid paths for executables and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the Rosetta process.
    - Customizability: Users can customize the Rosetta process through multiple parameters, including the number of structures (nstruct), specific options for the Rosetta application, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate Rosetta into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import time
import logging
from glob import glob
import shutil

# dependencies
import pandas as pd

# custom
import protflow.config
import protflow.jobstarters
from protflow.runners import Runner, RunnerOutput, prepend_cmd
from protflow.poses import Poses
from protflow.jobstarters import JobStarter

class Rosetta(Runner):
    """
    Rosetta Class
    =============

    The `Rosetta` class is a specialized class designed to facilitate the execution of Rosetta applications within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with Rosetta processes.

    Detailed Description
    --------------------
    The `Rosetta` class manages all aspects of running Rosetta simulations. It handles the configuration of necessary scripts and executables, prepares the environment for Rosetta processes, and executes the Rosetta commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to Rosetta scripts and executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of Rosetta commands with support for multiple structures (nstruct).
        - Collecting and processing output data into a pandas DataFrame.
        - Cleaning and renaming PDB files based on Rosetta outputs.
        - Handling score files and converting them into a readable format.

    Returns
    -------
    An instance of the `Rosetta` class, configured to run Rosetta processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - KeyError: If forbidden options are provided to the methods.

    Examples
    --------
    Here is an example of how to initialize and use the `Rosetta` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from rosetta import Rosetta

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the Rosetta class
        rosetta = Rosetta()

        # Run the Rosetta process
        results = rosetta.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            rosetta_application="RosettaScripts",
            nstruct=10,
            options="",
            pose_options=["-parser:protocol my_protocol.xml"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as invalid paths for executables, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the Rosetta process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The Rosetta class is intended for researchers and developers who need to perform Rosetta simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.ROSETTA_BIN_PATH, pre_cmd:str=protflow.config.ROSETTA_PRE_CMD, jobstarter: str = None, fail_on_missing_output_poses: bool = False) -> None:
        """
        Initialize the Rosetta class with the necessary configuration.

        This method sets up the Rosetta class with the provided script path and job starter configuration. It initializes the necessary parameters and prepares the environment for executing Rosetta processes.

        Parameters:
            script_path (str, optional): The path to the Rosetta executable scripts. Defaults to the path specified in `protflow.config.ROSETTA_BIN_PATH`.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Raises:
            ValueError: If no valid script path is provided.

        Examples:
            Here is an example of how to initialize the `Rosetta` class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from rosetta import Rosetta

                # Initialize the Rosetta class with a specific script path
                rosetta = Rosetta(script_path="/path/to/rosetta", jobstarter=JobStarter())

        Further Details:
            - **Configuration:** The method checks if the provided script path is valid and sets it up for further use. If no script path is provided, it defaults to the path specified in the ProtFlow configuration.
            - **Job Starter:** The job starter parameter can be provided to manage the execution of Rosetta jobs. If not provided, it defaults to None, and the default job starter configuration from ProtFlow will be used.

        This method ensures that the Rosetta class is correctly initialized with the necessary configurations to run Rosetta applications within the ProtFlow framework.
        """
        self.script_path = self.search_path(script_path, "ROSETTA_BIN_PATH", is_dir=True)
        self.name = "rosetta.py"
        self.pre_cmd = pre_cmd
        self.index_layers = 1
        self.jobstarter = jobstarter
        self.fail_on_missing_output_poses = fail_on_missing_output_poses

    def __str__(self):
        return "rosetta.py"

    def _setup_executable(self, script_path: str, rosetta_application: str) -> str:
        """
        Sets up the Rosetta executable.

        This method verifies and sets up the path to the Rosetta executable script. It ensures that the provided executable path is valid and can be executed. If the Rosetta application is not provided, it checks the default script path for executables.

        Parameters:
            script_path (str): The path to the Rosetta scripts directory.
            rosetta_application (str, optional): The specific Rosetta application to be executed. Defaults to None.

        Returns:
            str: The full path to the executable Rosetta application.

        Raises:
            ValueError: If the executable is not properly set up or the provided path is not executable.

        Examples:
            Here is an example of how to use the `setup_executable` method:

            .. code-block:: python

                from rosetta import Rosetta

                # Initialize the Rosetta class
                rosetta = Rosetta(script_path="/path/to/rosetta")

                # Set up the Rosetta executable
                executable_path = rosetta.setup_executable(
                    script_path="/path/to/rosetta",
                    rosetta_application="RosettaScripts"
                )

                print(executable_path)

        Further Details:
            - **Executable Verification:** The method checks if the provided path or the Rosetta application is executable. If neither is executable, it raises a ValueError indicating improper setup.
            - **Path Handling:** If the Rosetta application is provided, it verifies if it is executable. If not, it checks the default script path directory for the executable application. The method ensures that a valid executable path is returned for running Rosetta applications.

        This method is designed to ensure that the Rosetta executable is properly set up and can be executed, facilitating the smooth running of Rosetta processes within the ProtFlow framework.
        """
        # if rosetta_application is not provided, check if script_path is executable:
        if not rosetta_application:
            if os.path.isfile(script_path) and os.access(script_path, os.X_OK):
                return script_path
            raise ValueError(f"Rosetta Executable not setup properly. Either provide executable through Runner.script_path or give directly to run(rosetta_application)")

        # if rosetta_application is provided, check if it is executable:
        if os.path.isfile(rosetta_application) and os.access(rosetta_application, os.X_OK):
            return rosetta_application

        # if rosetta_application is not executable, find it at script_dir/rosetta_executable and check if this is executable:
        if os.path.isdir(script_path):
            combined_path = os.path.join(script_path, rosetta_application)
            if os.path.isfile(combined_path) and os.access(combined_path, os.X_OK):
                return combined_path
            raise ValueError(f"Provided rosetta_applicatiaon is not executable: {combined_path}")

        # otherwise raise error for not properly setting up the rosetta script paths.
        raise ValueError(f"No usable Rosetta executable provided. Easiest fix: provide full path to executable with parameter :rosetta_application: in the Rosetta.run() method.")

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, rosetta_application: str = None, nstruct: int = 1, options: str = None, pose_options: list|str = None, overwrite: bool = False, fail_on_missing_output_poses: bool = False) -> Poses:
        """
        Execute the Rosetta process with given poses and jobstarter configuration.

        This method sets up and runs the Rosetta process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            rosetta_application (str, optional): The specific Rosetta application to be executed. Defaults to None.
            nstruct (int, optional): The number of structures to generate for each input pose. Defaults to 1.
            options (str, optional): Additional options for the Rosetta application. Defaults to None.
            pose_options (list[str] | str, optional): A list of pose-specific options for the Rosetta application. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the Rosetta process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            KeyError: If forbidden options are provided to the method.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from rosetta import Rosetta

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the Rosetta class
                rosetta = Rosetta()

                # Run the Rosetta process
                results = rosetta.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    rosetta_application="RosettaScripts",
                    nstruct=10,
                    options="",
                    pose_options=["-parser:protocol my_protocol.xml"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It verifies that the Rosetta application is executable and configures the execution environment accordingly.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis. It manages score files, renames PDB files, and compiles output data into a structured format.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the Rosetta process to their specific needs. This includes setting the number of structures to generate, providing additional options for the Rosetta application, and specifying pose-specific options.

        This method is designed to streamline the execution of Rosetta processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze Rosetta simulations.
        """
        # setup runner:
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check if script_path / rosetta_application have an executable.
        rosetta_exec = self._setup_executable(self.script_path, rosetta_application)

        logging.info(f"Running {self} application {rosetta_exec} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip RosettaScripts.
        scorefile = os.path.join(work_dir, f"{prefix}_rosetta_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        elif overwrite and os.path.isdir(work_dir):
            rosetta_scores = glob(os.path.join(work_dir, "r*_*_score.json"))
            if len(rosetta_scores) > 0:
                for score in rosetta_scores:
                    os.remove(score)

        # parse_options and pose_options:
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)
        pose_options = self.prep_pose_options(poses, pose_options)

        # write rosettascripts cmds:
        cmds = []
        for pose, pose_opts in zip(poses.df['poses'].to_list(), pose_options):
            for i in range(1, nstruct+1):
                cmds.append(self.write_cmd(pose_path=pose, rosetta_application=rosetta_exec, output_dir=work_dir, i=i, overwrite=overwrite, options=options, pose_options=pose_opts))

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="rosetta",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores and rename pdbs.
        time.sleep(10) # Rosetta does not have time to write the last score into the scorefile otherwise?

        # collect scores
        scores = collect_scores(work_dir=work_dir)

        fail_on_missing_output_poses = fail_on_missing_output_poses or self.fail_on_missing_output_poses
        if len(scores.index) < len(poses.df.index) * nstruct and fail_on_missing_output_poses == True:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nstruct. Some runs might have crashed!")
    
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{rosetta_exec} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, rosetta_application: str, pose_path: str, output_dir: str, i: int, overwrite: bool = False, options: str = None, pose_options: str = None):
        """
        Writes the command to run a Rosetta application.

        This method constructs the command string needed to execute a Rosetta application with the specified options and parameters. It ensures that the command includes all necessary arguments and handles the setup for running the application.

        Parameters:
            rosetta_application (str): The path to the Rosetta executable.
            pose_path (str): The path to the input pose file.
            output_dir (str): The directory where output files will be stored.
            i (int): The index of the current structure being processed.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            options (str, optional): Additional options for the Rosetta application. Defaults to None.
            pose_options (str, optional): Pose-specific options for the Rosetta application. Defaults to None.

        Returns:
            str: The command string to execute the Rosetta application.

        Raises:
            KeyError: If forbidden options are included in the provided options or pose_options.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from rosetta import Rosetta

                # Initialize the Rosetta class
                rosetta = Rosetta(script_path="/path/to/rosetta")

                # Write the command to run the Rosetta application
                cmd = rosetta.write_cmd(
                    rosetta_application="/path/to/rosetta/RosettaScripts",
                    pose_path="input.pdb",
                    output_dir="/path/to/output",
                    i=1,
                    overwrite=True,
                    options="-parser:protocol my_protocol.xml",
                    pose_options="-in:file:s input.pdb"
                )

                print(cmd)

        Further Details:
            - **Command Construction:** The method constructs the command string by combining the executable path, input pose path, output directory, and additional options. It ensures that all necessary arguments are included in the command.
            - **Option Parsing:** The method parses the provided options and pose_options, ensuring they are correctly formatted and do not include any forbidden options. Forbidden options include arguments that could interfere with the correct execution of the Rosetta application, such as output path settings.
            - **Overwrite Handling:** If the overwrite parameter is set to True, the command includes the necessary argument to overwrite existing output files. This ensures that the process can be re-run without conflicts.

        This method is designed to facilitate the construction of command strings for running Rosetta applications, making it easier for researchers and developers to execute and manage Rosetta simulations within the ProtFlow framework.
        """
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""

        # check if interfering options were set
        forbidden_options = ['-out:path:all', '-in:file:s', '-out:prefix', '-out:file:scorefile', '-out:file:scorefile_format', ' -s ', '-scorefile_format']
        if (options and any(opt in options for opt in forbidden_options)) or (pose_options and any(pose_opt in pose_options for pose_opt in forbidden_options)):
            raise KeyError(f"options and pose_options must not contain any of {forbidden_options}")

        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""
        overwrite = " -overwrite" if overwrite else ""

        # compile command
        run_string = f"{rosetta_application} -out:path:all {output_dir} -in:file:s {pose_path} -out:prefix r{str(i).zfill(4)}_ -out:file:scorefile r{str(i).zfill(4)}_{os.path.splitext(os.path.basename(pose_path))[0]}_score.json -out:file:scorefile_format json {opts} {flags} {overwrite}"
        
        logging.debug(f"Run command: {run_string}")

        return run_string

def collect_scores(work_dir: str) -> pd.DataFrame:
    """
    Collects scores from Rosetta output files and reindexes PDB files.

    This function collects scores from Rosetta output files, reindexes PDB files based on the scores, and stores the scores in a pandas DataFrame. It also renames PDB files in the working directory to match the reindexed names.

    Parameters:
        work_dir (str): The directory where Rosetta output files are stored.

    Returns:
        pandas.DataFrame: A DataFrame containing the collected scores with reindexed PDB file names.

    Examples:
        Here is an example of how to use the `collect_scores` function:

        .. code-block:: python

            from rosetta import collect_scores

            # Collect scores from the Rosetta output directory
            scores_df = collect_scores(work_dir="/path/to/output_directory")

            print(scores_df)

    Further Details:
        - **Score Collection:** The function collects score files from the specified directory and reads them into a pandas DataFrame.
        - **Reindexing:** The function reindexes the PDB files based on the scores and renames the files in the working directory accordingly.
        - **File Renaming:** The function ensures that all Rosetta output PDB files are renamed to match the reindexed names, and the paths to these files are stored in the DataFrame.
        - **Consistency Check:** The function waits for all Rosetta output files to appear in the output directory, ensuring that the renaming process is consistent and complete.

    This function is designed to streamline the process of collecting and organizing Rosetta output data, making it easier for researchers and developers to analyze the results of Rosetta simulations within the ProtFlow framework.
    """
    scorefiles = glob(os.path.join(work_dir, "r*_*_score.json"))
    scores_l = []
    for scorefile in scorefiles:
        scores_l.append(pd.read_json(scorefile, typ='series'))
    scores_df = pd.DataFrame(scores_l).reset_index(drop=True).rename(columns={"decoy": "raw_description"})
    scores_df.loc[:, "description"] = scores_df["raw_description"].str.split("_").str[1:-1].str.join("_") + "_" + scores_df["raw_description"].str.split("_").str[0].str.replace("r", "")

    # wait for all Rosetta output files to appear in the output directory (for some reason, they are sometimes not there after the runs completed.)
    while len(glob(f"{work_dir}/r*.pdb")) < len(scores_df):
        time.sleep(1)

    # rename .pdb files in work_dir to the reindexed names.
    names_dict = scores_df[["raw_description", "description"]].to_dict()
    logging.info(f"Renaming and reindexing {len(scores_df)} Rosetta output .pdb files")
    for oldname, newname in zip(names_dict["raw_description"].values(), names_dict["description"].values()):
        shutil.move(f"{work_dir}/{oldname}.pdb", (nf := f"{work_dir}/{newname}.pdb"))
        if not os.path.isfile(nf):
            logging.warning(f"WARNING: Could not rename file {oldname} to {nf}\n Retrying renaming.")
            shutil.move(f"{work_dir}/{oldname}.pdb", (nf := f"{work_dir}/{newname}.pdb"))

    # Collect information of path to .pdb files into dataframe under "location" column
    scores_df.loc[:, "location"] = work_dir + "/" + scores_df["description"] + ".pdb"

    # safetycheck rename all remaining files with r*.pdb into proper filename:
    if (remaining_r_pdbfiles := glob(f"{work_dir}/r*.pdb")):
        for pdb_path in remaining_r_pdbfiles:
            pdb_path = pdb_path.split("/")[-1]
            idx = pdb_path.split("_")[0].replace("r", "")
            new_name = "_".join(pdb_path.split("_")[1:-1]).replace(".pdb", "") + "_" + idx + ".pdb"
            shutil.move(f"{work_dir}/{pdb_path}", f"{work_dir}/{new_name}")

    # reset index and write scores to file
    scores_df.reset_index(drop="True", inplace=True)

    return scores_df

def clean_rosetta_scorefile(path_to_file: str, out_path: str) -> str:
    """
    Cleans a faulty Rosetta scorefile.

    This function reads a Rosetta scorefile and removes any lines that do not match the expected format (i.e., lines with a different number of columns than the header). It writes the cleaned scores to a new file.

    Parameters:
        path_to_file (str): The path to the original Rosetta scorefile.
        out_path (str): The path where the cleaned scorefile will be saved.

    Returns:
        str: The path to the cleaned scorefile.

    Examples:
        Here is an example of how to use the `clean_rosetta_scorefile` function:

        .. code-block:: python

            from rosetta import clean_rosetta_scorefile

            # Clean the Rosetta scorefile
            cleaned_file_path = clean_rosetta_scorefile(
                path_to_file="path/to/original_scorefile.sc",
                out_path="path/to/cleaned_scorefile.sc"
            )

            print(f"Cleaned scorefile saved at: {cleaned_file_path}")

    Further Details:
        - **File Reading:** The function reads the scorefile line by line and splits each line into columns.
        - **Format Verification:** It verifies that each line has the same number of columns as the header. Lines with a different number of columns are removed.
        - **File Writing:** The cleaned scores are written to the specified output file. A warning is logged indicating the number of lines removed during the cleaning process.

    This function is useful for ensuring that Rosetta scorefiles are properly formatted and free of inconsistencies, facilitating accurate data analysis.
    """
    # read in file line-by-line:
    with open(path_to_file, 'r', encoding="UTF-8") as f:
        scores = [line.split() for line in list(f.readlines()[1:])]

    # if any line has a different number of scores than the header (columns), that line will be removed.
    scores_cleaned = [line for line in scores if len(line) == len(scores[0])]
    logging.warning(f"{len(scores) - len(scores_cleaned)} scores were removed from Rosetta scorefile at {path_to_file}")

    # write cleaned scores to file:
    with open(out_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join([",".join(line) for line in scores_cleaned]))
    return out_path
