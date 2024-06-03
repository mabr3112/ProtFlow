"""
FPocket Module
==============

This module provides the functionality to integrate FPocket within the ProtFlow framework. It offers tools to run FPocket, handle its inputs and outputs, and process the resulting data in a structured and automated manner. 

Detailed Description
--------------------
The `FPocket` class encapsulates the functionality necessary to execute FPocket runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of FPocket processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.
The module is designed to streamline the integration of FPocket into larger computational workflows. It supports the automatic setup of job parameters, execution of FPocket commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `FPocket` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the FPocket process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `FPocket` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from fpocket import FPocket

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the FPocket class
    fpocket = FPocket()

    # Run the FPocket process
    results = fpocket.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="--some-option value",
        pose_options=["--specific-option value"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the FPocket process.
    - Customizability: Users can customize the FPocket process through multiple parameters, including specific options for the FPocket script and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate FPocket into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
# imports
import glob
import os
import shutil

# dependencies
import pandas as pd

# custom
from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput, options_flags_to_string, parse_generic_options
from protflow.config import FPOCKET_PATH

class FPocket(Runner):
    """
    FPocket Class
    =============

    The `FPocket` class is a specialized class designed to facilitate the execution of FPocket within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with FPocket processes.

    Detailed Description
    --------------------
    The `FPocket` class manages all aspects of running FPocket simulations. It handles the configuration of necessary scripts and executables, prepares the environment for pocket detection processes, and executes the FPocket commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to FPocket scripts and executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of FPocket commands with support for multiple options and pose-specific parameters.
        - Collecting and processing output data into a pandas DataFrame.
        - Ensuring robust error handling and logging for easier debugging and verification of the FPocket process.

    Returns
    -------
    An instance of the `FPocket` class, configured to run FPocket processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If provided options are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `FPocket` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from fpocket import FPocket

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the FPocket class
        fpocket = FPocket()

        # Run the FPocket process
        results = fpocket.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            options="--some-option value",
            pose_options=["--specific-option value"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the FPocket process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The FPocket class is intended for researchers and developers who need to perform FPocket simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    # class attributes
    index_layers = 0

    def __init__(self, fpocket_path: str = FPOCKET_PATH, jobstarter: JobStarter = None):
        """
        Initialize the FPocket class with the specified path and jobstarter configuration.

        This constructor sets up the FPocket instance by configuring the path to the FPocket executable and initializing the jobstarter object. It ensures that the necessary components are in place for running FPocket processes.

        Parameters:
            fpocket_path (str, optional): The path to the FPocket executable. Defaults to the path specified in the ProtFlow configuration (`FPOCKET_PATH`).
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Returns:
            An instance of the FPocket class, ready to run FPocket processes.

        Raises:
            ValueError: If the fpocket_path is not provided or is invalid.

        Examples:
            Here is an example of how to initialize the FPocket class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from fpocket import FPocket

                # Initialize the FPocket class with default settings
                fpocket = FPocket()

                # Initialize the FPocket class with a specific jobstarter
                jobstarter = JobStarter()
                fpocket = FPocket(jobstarter=jobstarter)

        Further Details:
            - **Path Configuration:** Ensures the FPocket executable path is set correctly, raising an error if the path is not provided or invalid.
            - **Job Management:** Initializes the jobstarter object to manage the execution of FPocket commands, allowing for integration with job scheduling systems.
        """
        if not fpocket_path:
            raise ValueError(f"No path was set for {self}. Set the path in the config.py file under FPOCKET_PATH!")
        self.jobstarter = jobstarter
        self.script_path = fpocket_path

    def __str__(self):
        return "fpocket"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str|list = None, pose_options: str|list = None, return_full_scores: bool = False, overwrite: bool = False) -> Poses:
        """
        Execute the FPocket process with given poses and jobstarter configuration.

        This method sets up and runs the FPocket process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            options (str or list[str], optional): Additional options for the FPocket script. Defaults to None.
            pose_options (str or list[str], optional): A list of pose-specific options for the FPocket script. Defaults to None.
            return_full_scores (bool, optional): If True, include detailed scores for each pocket in the output. Defaults to False.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: An updated Poses object containing the processed poses and results of the FPocket process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If options or pose_options are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from fpocket import FPocket

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the FPocket class
                fpocket = FPocket()

                # Run the FPocket process
                results = fpocket.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    options="--some-option value",
                    pose_options=["--specific-option value"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It moves the poses to the working directory and compiles the FPocket commands for execution.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized into a structured DataFrame. It includes the location of each pocket and integrates the results back into the Poses object.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the FPocket process to their specific needs, including the ability to specify additional FPocket options and pose-specific parameters.

        This method is designed to streamline the execution of FPocket processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze pocket detection simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # prep options:
        options_l = self.prep_fpocket_options(poses, options, pose_options)

        # move poses to input_dir (fpocket runs in the directory of the input pdb file)
        work_poses = []
        for pose in poses.poses_list():
            new_p = f"{work_dir}/{pose.split('/')[-1]}"
            if not os.path.isfile(new_p):
                shutil.copy(pose, new_p)
            work_poses.append(new_p)

        # compile cmds
        cmds = [f"{self.script_path} --file {pose} {options}" for pose, options in zip(work_poses, options_l)]

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"fpocket_{prefix}",
            output_path = work_dir
        )

        # collect outputs and write scorefile
        scores = collect_fpocket_scores(work_dir, return_full_scores=return_full_scores)
        scores["location"] = [_get_fpocket_input_location(description, cmds) for description in scores["description"].to_list()]
        scores["pocket_location"] = scores["pocket_location"].fillna(scores["location"])
        self.save_runner_scorefile(scores, scorefile)

        # itegrate and return
        outputs = RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()
        return outputs

    def prep_fpocket_options(self, poses: Poses, options: str, pose_options: str|list[str]) -> list[str]:
        """
        Prepare options for the FPocket process based on given parameters.

        This method processes and prepares the options and pose-specific options for the FPocket run. It filters out forbidden options, merges general options with pose-specific options, and formats them for inclusion in the FPocket commands.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            options (str or list[str], optional): General options for the FPocket script. Defaults to None.
            pose_options (str or list[str], optional): A list of pose-specific options for the FPocket script. Defaults to None.

        Returns:
            list[str]: A list of formatted option strings for each pose, ready to be used in the FPocket commands.

        Raises:
            TypeError: If options or pose_options are not of the expected type.

        Examples:
            Here is an example of how to use the `prep_fpocket_options` method:

            .. code-block:: python

                from protflow.poses import Poses
                from fpocket import FPocket

                # Create instances of necessary classes
                poses = Poses()
                fpocket = FPocket()

                # Prepare FPocket options
                options = "--some-option value"
                pose_options = ["--specific-option value"]
                prepared_options = fpocket.prep_fpocket_options(poses, options, pose_options)

                # Output the prepared options
                print(prepared_options)

        Further Details:
            - **Option Processing:** Merges general and pose-specific options, ensuring that forbidden options are removed and the final option strings are correctly formatted.
            - **Customization:** Allows for extensive customization of the FPocket process through both general and pose-specific options, providing flexibility in configuring FPocket runs.
        """
        forbidden_options = ["file", "pocket_descr_stdout", "write_mode"]
        pose_options = self.prep_pose_options(poses, pose_options)

        # Iterate through pose options, overwrite options and remove options that are not allowed.
        options_l = []
        for pose_opt in pose_options:
            opts, flags = parse_generic_options(options, pose_opt)
            for opt in forbidden_options:
                opts.pop(opt, None)
            options_l.append(options_flags_to_string(opts,flags))

        # merge options and pose_options, with pose_options priority and return
        return options_l

def get_outfile_name(outdir: str) -> str:
    """
    Get the name of the output file from the output directory.

    This function generates the name of the FPocket output file based on the specified output directory.

    Parameters:
        outdir (str): The path to the FPocket output directory.

    Returns:
        str: The name of the output file within the specified directory.

    Examples:
        Here is an example of how to use the `get_outfile_name` function:

        .. code-block:: python

            from fpocket import get_outfile_name

            # Specify the output directory
            outdir = "path/to/output_directory"

            # Get the output file name
            output_file_name = get_outfile_name(outdir)

            # Display the output file name
            print(output_file_name)

    Further Details:
        - **File Naming:** The function constructs the output file name by modifying the output directory name and appending the appropriate suffix.
    """
    f = [x.strip() for x in outdir.split("/") if x][-1].replace("_out", "_info.txt")
    return f"{outdir}/{f}"

def collect_fpocket_scores(output_dir: str, return_full_scores: bool = False) -> pd.DataFrame:
    """
    Collect scores from an FPocket output directory.

    This function collects and processes the scores from FPocket output files located in the specified directory. It aggregates the scores into a pandas DataFrame for further analysis.

    Parameters:
        output_dir (str): The path to the directory containing FPocket output files.
        return_full_scores (bool, optional): If True, include detailed scores for each pocket in the output. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores from the FPocket output files.

    Examples:
        Here is an example of how to use the `collect_fpocket_scores` function:

        .. code-block:: python

            from fpocket import collect_fpocket_scores

            # Specify the output directory
            output_dir = "path/to/output_directory"

            # Collect scores
            scores = collect_fpocket_scores(output_dir, return_full_scores=True)

            # Display the scores
            print(scores)

    Further Details:
        - **Score Aggregation:** The function looks for FPocket output directories, extracts scores from each output file, and combines them into a single DataFrame.
        - **Detailed Scores:** If the return_full_scores parameter is set to True, the function includes detailed scores for each pocket in the DataFrame.
    """
    # collect output_dirs
    output_dirs = glob.glob(f"{output_dir}/*_out")

    # extract individual scores and merge into DF:
    out_df = pd.concat([collect_fpocket_output(get_outfile_name(out_dir), return_full_scores=return_full_scores) for out_dir in output_dirs]).reset_index(drop=True)
    return out_df

def collect_fpocket_output(output_file: str, return_full_scores: bool = False) -> pd.DataFrame:
    """
    Collect output from a single FPocket output file.

    This function processes the output of a single FPocket output file, extracting scores and other relevant information into a pandas DataFrame.

    Parameters:
        output_file (str): The path to the FPocket output file.
        return_full_scores (bool, optional): If True, include detailed scores for each pocket in the output. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the processed output from the FPocket file.

    Examples:
        Here is an example of how to use the `collect_fpocket_output` function:

        .. code-block:: python

            from fpocket import collect_fpocket_output

            # Specify the output file
            output_file = "path/to/output_file"

            # Collect output
            output = collect_fpocket_output(output_file, return_full_scores=True)

            # Display the output
            print(output)

    Further Details:
        - **Output Processing:** The function reads the FPocket output file, extracts relevant scores and information, and formats them into a DataFrame.
        - **Detailed Scores:** If the return_full_scores parameter is set to True, the function includes detailed scores for each pocket in the DataFrame.
    """
    # instantiate output_dict
    file_scores = parse_fpocket_outfile(output_file)

    if file_scores.empty:
        return pd.DataFrame.from_dict({"description": [output_file.split("/")[-1].replace("_info.txt", "")]})

    # integrate all scores if option is set:
    top_df = file_scores.head(1)
    new_cols = ["top_" + col.lower().replace(" ", "_") for col in top_df.columns]
    top_df = top_df.rename(columns=dict(zip(top_df.columns, new_cols)))
    top_df = top_df.reset_index().rename(columns={"index": "pocket"})

    # collect description and integrate into top_df
    top_df["description"] = output_file.split("/")[-1].replace("_info.txt", "")
    if return_full_scores:
        top_df["all_pocket_scores"] = file_scores

    # rename pocket_location column back.
    top_df = top_df.rename(columns={"top_pocket_location": "pocket_location"})

    return top_df.reset_index(drop=True)

def parse_fpocket_outfile(output_file: str) -> pd.DataFrame:
    """
    Parse the FPocket output file to extract scores.

    This function reads and parses the FPocket output file, extracting scores and other relevant information into a pandas DataFrame.

    Parameters:
        output_file (str): The path to the FPocket output file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed scores from the FPocket output file.

    Examples:
        Here is an example of how to use the `parse_fpocket_outfile` function:

        .. code-block:: python

            from fpocket import parse_fpocket_outfile

            # Specify the output file
            output_file = "path/to/output_file"

            # Parse the output file
            scores = parse_fpocket_outfile(output_file)

            # Display the scores
            print(scores)

    Further Details:
        - **File Parsing:** The function reads the FPocket output file, extracts relevant scores and information, and formats them into a DataFrame.
        - **Score Extraction:** The function processes the file line by line, extracting score data and organizing it into a structured format.
    """
    def parse_pocket_line(pocket_line: str) -> tuple[str,float]:
        '''Parses singular line '''
        # split along colon between column: value
        col, val = pocket_line.split(":")
        return col.strip(), float(val[val.index("\t")+1:])

    # read out file and split along "Pocket"
    with open(output_file, 'r', encoding="UTF-8") as f:
        pocket_split = [x.strip() for x in f.read().split("Pocket") if x]

    # create empty pocket dict to populate
    pocket_dict = {}
    for raw_str in pocket_split:
        line_split = [x.strip() for x in raw_str.split("\n") if x]
        pocket_nr = line_split[0].split()[0].strip()
        pocket_dict[f"pocket_{pocket_nr}"] = {col: val for (col, val) in [parse_pocket_line(line) for line in line_split[1:]]}

    out_df = pd.DataFrame.from_dict(pocket_dict).T
    if out_df.empty:
        return out_df
    out_df["pocket_location"] = output_file.replace("info.txt", "out.pdb")
    return out_df.sort_values("Druggability Score", ascending=False)

def _get_fpocket_input_location(description: str, cmds: list[str]) -> str:
    '''Looks ad a pose_description and tries to find the pose in a list of commands that was used as input to generate the description.
    This is an internal function for location mapping'''
    # first get the cmd that contains 'description'
    cmd = [cmd for cmd in cmds if f"/{description}.pdb" in cmd][0]

    # extract location of input pdb:
    return [substr for substr in cmd.split(" ") if f"/{description}.pdb" in substr][0]
