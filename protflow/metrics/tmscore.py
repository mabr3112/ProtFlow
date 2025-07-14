"""
TMscore Module
==============

This module provides the functionality to integrate TMscore calculations within the ProtFlow framework. It offers tools to run TMscore and TMalign, handle their inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `TMalign` and `TMscore` classes encapsulate the functionality necessary to execute TM-align and TM-score runs, respectively. These classes manage the configuration of paths to essential scripts and Python executables, set up the environment, and handle the execution of scoring processes. They include methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of TM-align and TM-score into larger computational workflows. It supports the automatic setup of job parameters, execution of TM-align/TM-score commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `TMalign` or `TMscore` class and invoke their `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the scoring process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `TMalign` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from tmscore import TMalign

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the TMalign class
    tmalign = TMalign()

    # Run the alignment process
    results = tmalign.run(
        poses=poses,
        prefix="experiment_1",
        ref_col="reference_pdb",
        sc_tm_score=True,
        options="-a",
        pose_options=["-b"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Here is an example of how to initialize and use the `TMscore` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from tmscore import TMscore

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the TMscore class
    tmscore = TMscore()

    # Run the scoring process
    results = tmscore.run(
        poses=poses,
        prefix="experiment_2",
        ref_col="reference_pdb",
        options="-c",
        pose_options=["-d"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the scoring process.
    - Customizability: Users can customize the scoring process through multiple parameters, including specific options for the TM-align or TM-score scripts, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate TM-align or TM-score into their protein structure comparison and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
# import general
import os
import glob

# import dependencies
import pandas as pd

# import customs
from .. import jobstarters, runners
from ..poses import Poses
from ..runners import Runner, RunnerOutput, col_in_df
from ..jobstarters import JobStarter
from ..config import PROTFLOW_ENV
from ..utils.metrics import calc_sc_tm

class TMalign(Runner):
    """
    TMalign Class
    =============

    The `TMalign` class is a specialized class designed to facilitate the execution of TMalign within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with TMalign processes.

    Detailed Description
    --------------------
    The `TMalign` class manages all aspects of running TMalign simulations. It handles the configuration of necessary scripts and executables, prepares the environment for alignment processes, and executes the alignment commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to TMalign executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of TMalign commands with support for various alignment options.
        - Collecting and processing output data into a pandas DataFrame.
        - Normalizing TM scores based on the reference structure and calculating self-consistency scores.

    Returns
    -------
    An instance of the `TMalign` class, configured to run TMalign processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If the TMalign executable is not found in the specified environment.
        - ValueError: If invalid arguments are provided to the methods or if required reference columns are missing.
        - RuntimeError: If no TM scores are found in the output files.

    Examples
    --------
    Here is an example of how to initialize and use the `TMalign` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from tmscore import TMalign

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the TMalign class
        tmalign = TMalign()

        # Run the alignment process
        results = tmalign.run(
            poses=poses,
            prefix="experiment_1",
            ref_col="reference_pdb",
            sc_tm_score=True,
            options="-a",
            pose_options=["-b"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the alignment process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    Difference Between TMscore and TMalign
    --------------------------------------
    - **TMscore**: This class calculates the TM-score between protein structures without superimposing them. It is suitable for comparing the overall similarity of protein structures in a sequence-length independent manner. TMscore is used when you need to score the structural similarity directly without modifying the positions of the structures.
    
    - **TMalign**: This class not only calculates the TM-score but also superimposes the structures before scoring. It is used when structural alignment and superimposition are necessary to get a more accurate measure of structural similarity, considering the spatial arrangement of the protein structures.

    The TMalign class is intended for researchers and developers who need to perform TMalign alignments as part of their protein structure comparison and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, jobstarter: JobStarter = None, application: str = None):
        """
        Initialize the TMalign class with optional jobstarter and application path.

        This method sets up the TMalign class by configuring the jobstarter and the path to the TMalign executable. It ensures that the necessary components are ready for executing TMalign processes.

        Parameters:
            jobstarter (JobStarter, optional): An optional jobstarter configuration. Defaults to None.
            application (str, optional): Path to the TMalign executable. If not provided, it defaults to the TMalign executable in the ProtFlow environment.

        Raises:
            ValueError: If the TMalign executable is not found in the specified environment.

        Examples:
            Here is an example of how to initialize the `TMalign` class:

            .. code-block:: python

                from tmscore import TMalign

                # Initialize the TMalign class
                tmalign = TMalign(
                    jobstarter=LocalJobStarter(max_cores=4),
                    application="/path/to/TMalign"
                )

                # Check the instance
                print(tmalign)

        Further Details:
            - **Jobstarter Configuration:** This parameter allows setting up the jobstarter for managing job execution.
            - **Application Path:** This parameter sets the path to the TMalign executable, ensuring the correct executable is used for alignment processes.

        This method is designed to prepare the TMalign class for executing TMalign processes, ensuring that all necessary configurations are in place.
        """
        self.jobstarter = jobstarter
        self.name = "tmscore.py"
        self.index_layers = 0
        self.application = self._check_install(application or os.path.join(PROTFLOW_ENV, "TMalign"))

    def __str__(self):
        return "TMalign"

    def _check_install(self, application_path) -> str:
        '''checks if TMalign is installed in the environment'''
        if not os.path.isfile(application_path):
            raise ValueError(f"Could not find executable for TMalign at {application_path}. Did you set it up in your protflow environment? If not, either install it in your protflow env with 'conda install -c bioconda tmalign' or in any other environment and provide the path to the application with the :application: parameter when initializing a TMalign() runner instance.")
        return application_path


    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str, sc_tm_score: bool = True, options: str = None, pose_options: str = None, overwrite: bool = False, jobstarter: JobStarter = None) -> Poses: # pylint: disable=W0237
        """
        Execute the TMalign process with given poses and jobstarter configuration.

        This method sets up and runs the TMalign process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            ref_col (str|): Column containing paths to PDB files used as reference for TM score calculation. Can also be a path to a singular reference .pdb file.
            sc_tm_score (bool, optional): If True, calculates the self-consistency TM score for each backbone in ref_col and adds it into the column {prefix}_sc_tm. Defaults to True.
            options (str, optional): Additional command-line options for the TMalign script. Defaults to None.
            pose_options (str, optional): Name of poses.df column containing options for TMalign. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the TMalign process.

        Raises:
            FileNotFoundError: If the TMalign executable is not found in the specified environment.
            ValueError: If invalid arguments are provided to the method or if required reference columns are missing.
            RuntimeError: If no TM scores are found in the output files.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from tmscore import TMalign

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the TMalign class
                tmalign = TMalign()

                # Run the alignment process
                results = tmalign.run(
                    poses=poses,
                    prefix="experiment_1",
                    ref_col="reference_pdb",
                    sc_tm_score=True,
                    options="-a",
                    pose_options=["-b"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Reference Preparation:** The method prepares the reference structures for alignment based on the provided reference column or specific PDB file.
            - **Output Management:** The method handles the collection and processing of output data, including merging and normalizing TM scores, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the alignment process to their specific needs.

        This method is designed to streamline the execution of TMalign processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein structure alignments.
        """
        # setup runner and files
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        scorefile = os.path.join(work_dir, f"{prefix}_TM.{poses.storage_format}")

        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
            if sc_tm_score:
                output.df = calc_sc_tm(input_df=output.df, name=f"{prefix}_sc_tm", ref_col=ref_col, tm_col=f"{prefix}_TM_score_ref")
            return output

        # prepare pose options
        pose_options = self.prep_pose_options(poses, pose_options)

        # prepare references:
        ref_l = self.prep_ref(ref=ref_col, poses=poses)

        cmds = []
        for pose, ref, pose_opts in zip(poses.df['poses'].to_list(), ref_l, pose_options):
            cmds.append(self.write_cmd(pose_path=pose, ref_path=ref, output_dir=work_dir, options=options, pose_options=pose_opts))

        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)

        # create batch commands
        cmd_sublists = jobstarters.split_list(cmds, n_sublists=num_cmds)
        cmds = []
        for sublist in cmd_sublists:
            cmds.append("; ".join(sublist))

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "TM",
            output_path = work_dir
        )

        scores = self.collect_scores(output_dir=work_dir)
        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
        if sc_tm_score:
            output.df = calc_sc_tm(input_df=output.df, name=f"{prefix}_sc_tm", ref_col=ref_col, tm_col=f"{prefix}_TM_score_ref")
        return output

    def prep_ref(self, ref: str, poses: Poses) -> list[str]:
        """
        Prepare the reference structures for TMalign.

        This method prepares the reference structures for alignment based on the provided reference column or specific PDB file. It ensures that the references are correctly formatted for the TMalign process.

        Parameters:
            ref (str): The reference structure, either as a path to a PDB file or as a column name in the Poses DataFrame.
            poses (Poses): The Poses object containing the protein structures.

        Returns:
            list[str]: A list of reference paths for each pose.

        Raises:
            ValueError: If the ref parameter is not a string or if the reference column is missing from the Poses DataFrame.

        Examples:
            Here is an example of how to use the `prep_ref` method:

            .. code-block:: python

                from protflow.poses import Poses
                from tmscore import TMalign

                # Create instances of necessary classes
                poses = Poses()
                tmalign = TMalign()

                # Prepare reference structures
                ref_list = tmalign.prep_ref(
                    ref="reference_pdb",
                    poses=poses
                )

                # Print the reference list
                print(ref_list)

        Further Details:
            - **Reference Handling:** The method can handle both a single PDB file and a column name referring to multiple PDB files within the Poses DataFrame.
            - **Validation:** Ensures that the provided reference is valid and exists in the Poses DataFrame if specified as a column name.

        This method is designed to streamline the preparation of reference structures for TMalign processes, ensuring that all references are correctly formatted and validated.
        """
        if not isinstance(ref, str):
            raise ValueError("Parameter :ref: must be string and either refer to a .pdb file or to a column in poses.df!")
        if ref.endswith(".pdb"):
            return [ref for _ in poses]

        # check if reference column exists in poses.df
        col_in_df(poses.df, ref)
        return poses.df[ref].to_list()

    def write_cmd(self, pose_path: str, ref_path: str, output_dir: str, options: str = None, pose_options: str = None) -> str:
        """
        Write the command to run TMalign.

        This method constructs the command to execute TMalign based on the provided parameters. It formats the options and flags correctly and sets up the command to be run in the environment.

        Parameters:
            pose_path (str): The path to the pose file.
            ref_path (str): The path to the reference file.
            output_dir (str): The directory where output files will be saved.
            options (str, optional): Additional command-line options for TMalign. Defaults to None.
            pose_options (str, optional): Pose-specific options for TMalign. Defaults to None.

        Returns:
            str: The constructed command string to run TMalign.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from tmscore import TMalign

                # Initialize the TMalign class
                tmalign = TMalign()

                # Write the command
                cmd = tmalign.write_cmd(
                    pose_path="pose.pdb",
                    ref_path="reference.pdb",
                    output_dir="output/",
                    options="-a",
                    pose_options="-b"
                )

                # Print the command
                print(cmd)

        Further Details:
            - **Command Construction:** The method constructs the command string by parsing and formatting the provided options and pose-specific options.
            - **Output Management:** Ensures that the output files are correctly named and saved in the specified directory.

        This method is designed to streamline the construction of commands for TMalign processes, ensuring that all necessary options are correctly formatted and included.
        """
        # parse options
        opts, flags = runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""

        # parse options
        opts, flags = runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.tmout")

        # compile command
        run_string = f"{self.application} {pose_path} {ref_path} {opts} {flags} > {scorefile}"

        return run_string

    def collect_scores(self, output_dir:str) -> pd.DataFrame:
        """
        Collect scores from TMalign output files.

        This method collects and processes the scores from the output files generated by TMalign. It reads the scores, extracts relevant information, and organizes the data into a structured pandas DataFrame.

        Parameters:
            output_dir (str): The directory where TMalign output files are located.

        Returns:
            pd.DataFrame: A DataFrame containing the collected scores.

        Raises:
            RuntimeError: If no TM scores are found in the output files.

        Examples:
            Here is an example of how to use the `collect_scores` method:

            .. code-block:: python

                from tmscore import TMalign

                # Initialize the TMalign class
                tmalign = TMalign()

                # Collect scores
                scores_df = tmalign.collect_scores(
                    output_dir="output/"
                )

                # Print the scores DataFrame
                print(scores_df)

        Further Details:
            - **Score Extraction:** The method reads the output files, extracts relevant scores, and organizes them into a pandas DataFrame.
            - **Validation:** Ensures that the scores are correctly extracted and that no errors occurred during the process.

        This method is designed to streamline the collection and processing of scores from TMalign output files, ensuring that all relevant data is accurately captured and organized.
        """
        def extract_scores(score_path:str) -> pd.Series:
            '''
            extract TM scores from scorefile, return a Series
            '''

            tm_scores = {}
            with open(score_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    if line.startswith("Aligned length"):
                        tm_scores['num_aligned_res'] = int(line.split()[2].replace(',', ''))
                        tm_scores['RMSD'] = float(line.split()[4].replace(',', ''))
                        tm_scores['n_identical/n_aligned'] = float(line.split()[6])
                        continue
                    elif line.startswith("TM-score") and "Chain_1" in line:
                        # TM score normalized by length of the pose structure
                        tm_scores['TM_score_pose'] = float(line.split()[1])
                        continue
                    elif line.startswith("TM-score") and "Chain_2" in line:
                        # TM score normalized by length of the reference (this is what should be used)
                        tm_scores['TM_score_ref'] = float(line.split()[1])
                        continue
                    elif line.startswith("TM-score") and "average" in line:
                        # if -a flag was provided to TMalign, a TM score normalized by the average length of pose and reference will be calculated
                        tm_scores['TM_score_average'] = float(line.split()[1])

            tm_scores['description'] = os.path.splitext(os.path.basename(score_path))[0]

            # check if scores were present in scorefile
            if not any('TM_score' in key for key in tm_scores):
                raise RuntimeError(f"Could not find any TM scores in {score_path}!")
            return pd.Series(tm_scores)

        # collect scorefiles
        scorefiles = glob.glob(os.path.join(output_dir, "*.tmout"))

        scores = [extract_scores(file) for file in scorefiles]
        scores = pd.DataFrame(scores).reset_index(drop=True)

        return scores

class TMscore(Runner):
    """
    TMscore Class
    =============

    The `TMscore` class is a specialized class designed to facilitate the execution of TMscore within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with TMscore processes.

    Detailed Description
    --------------------
    The `TMscore` class manages all aspects of running TMscore simulations. It handles the configuration of necessary scripts and executables, prepares the environment for scoring processes, and executes the scoring commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to TMscore executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of TMscore commands with support for various scoring options.
        - Collecting and processing output data into a pandas DataFrame.

    Difference Between TMscore and TMalign
    --------------------------------------
    - **TMscore**: This class calculates the TM-score between protein structures without superimposing them. It is suitable for comparing the overall similarity of protein structures in a sequence-length independent manner. TMscore is used when you need to score the structural similarity directly without modifying the positions of the structures.
    
    - **TMalign**: This class not only calculates the TM-score but also superimposes the structures before scoring. It is used when structural alignment and superimposition are necessary to get a more accurate measure of structural similarity, considering the spatial arrangement of the protein structures.

    Returns
    -------
    An instance of the `TMscore` class, configured to run TMscore processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If the TMscore executable is not found in the specified environment.
        - ValueError: If invalid arguments are provided to the methods or if required reference columns are missing.
        - RuntimeError: If no TM scores are found in the output files.

    Examples
    --------
    Here is an example of how to initialize and use the `TMscore` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from tmscore import TMscore

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the TMscore class
        tmscore = TMscore()

        # Run the scoring process
        results = tmscore.run(
            poses=poses,
            prefix="experiment_2",
            ref_col="reference_pdb",
            options="-c",
            pose_options=["-d"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the scoring process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The TMscore class is intended for researchers and developers who need to perform TMscore calculations as part of their protein structure comparison and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, jobstarter: str = None, application: str = None):
        """
        Initialize the TMscore class with optional jobstarter and application path.

        This method sets up the TMscore class by configuring the jobstarter and the path to the TMscore executable. It ensures that the necessary components are ready for executing TMscore processes.

        Parameters:
            jobstarter (str, optional): An optional jobstarter configuration. Defaults to None.
            application (str, optional): Path to the TMscore executable. If not provided, it defaults to the TMscore executable in the ProtFlow environment.

        Examples:
            Here is an example of how to initialize the `TMscore` class:

            .. code-block:: python

                from tmscore import TMscore

                # Initialize the TMscore class
                tmscore = TMscore(
                    jobstarter="local",
                    application="/path/to/TMscore"
                )

                # Check the instance
                print(tmscore)

        Further Details:
            - **Jobstarter Configuration:** This parameter allows setting up the jobstarter for managing job execution.
            - **Application Path:** This parameter sets the path to the TMscore executable, ensuring the correct executable is used for scoring processes.

        This method is designed to prepare the TMscore class for executing TMscore processes, ensuring that all necessary configurations are in place.
        """
        self.jobstarter = jobstarter
        self.name = "tmscore.py"
        self.index_layers = 0
        self.application = application or os.path.join(PROTFLOW_ENV, "TMscore")

    def __str__(self):
        return self.name

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str, options: str = None, pose_options: str = None, overwrite: bool = False, jobstarter: JobStarter = None) -> None: # pylint: disable=W0237
        """
        Execute the TMscore process with given poses and jobstarter configuration.

        This method sets up and runs the TMscore process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            ref_col (str): Column containing paths to PDB files used as reference for TM score calculation.
            options (str, optional): Additional command-line options for the TMscore script. Defaults to None.
            pose_options (str, optional): Name of poses.df column containing options for TMscore. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the TMscore process.

        Raises:
            FileNotFoundError: If the TMscore executable is not found in the specified environment.
            ValueError: If invalid arguments are provided to the method or if required reference columns are missing.
            RuntimeError: If no TM scores are found in the output files.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from tmscore import TMscore

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the TMscore class
                tmscore = TMscore()

                # Run the scoring process
                results = tmscore.run(
                    poses=poses,
                    prefix="experiment_2",
                    ref_col="reference_pdb",
                    options="-c",
                    pose_options=["-d"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Reference Handling:** The method validates the reference column and prepares the reference structures for scoring.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the scoring process to their specific needs.

        This method is designed to streamline the execution of TMscore processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein structure comparisons.
        """
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        scorefile = os.path.join(work_dir, f"{prefix}_TM.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # check if reference column exists in poses.df
        col_in_df(poses.df, ref_col)

        # prepare pose options
        pose_options = self.prep_pose_options(poses, pose_options)

        cmds = []
        for pose, ref, pose_opts in zip(poses.df['poses'].to_list(), poses.df[ref_col].to_list(), pose_options):
            cmds.append(self.write_cmd(pose_path=pose, ref_path=ref, output_dir=work_dir, options=options, pose_options=pose_opts))

        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)

        # create batch commands
        cmd_sublists = jobstarters.split_list(cmds, n_sublists=num_cmds)
        cmds = []
        for sublist in cmd_sublists:
            cmds.append("; ".join(sublist))

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "TM",
            output_path = work_dir
        )

        scores = self.collect_scores(output_dir=work_dir)

        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
        return output.return_poses()

    def write_cmd(self, pose_path: str, ref_path: str, output_dir: str, options: str = None, pose_options: str = None ) -> str:
        """
        Write the command to run TMscore.

        This method constructs the command to execute TMscore based on the provided parameters. It formats the options and flags correctly and sets up the command to be run in the environment.

        Parameters:
            pose_path (str): The path to the pose file.
            ref_path (str): The path to the reference file.
            output_dir (str): The directory where output files will be saved.
            options (str, optional): Additional command-line options for TMscore. Defaults to None.
            pose_options (str, optional): Pose-specific options for TMscore. Defaults to None.

        Returns:
            str: The constructed command string to run TMscore.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from tmscore import TMscore

                # Initialize the TMscore class
                tmscore = TMscore()

                # Write the command
                cmd = tmscore.write_cmd(
                    pose_path="pose.pdb",
                    ref_path="reference.pdb",
                    output_dir="output/",
                    options="-a",
                    pose_options="-b"
                )

                # Print the command
                print(cmd)

        Further Details:
            - **Command Construction:** The method constructs the command string by parsing and formatting the provided options and pose-specific options.
            - **Output Management:** Ensures that the output files are correctly named and saved in the specified directory.

        This method is designed to streamline the construction of commands for TMscore processes, ensuring that all necessary options are correctly formatted and included.
        """
        # parse options
        opts, flags = runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""

        # parse options
        opts, flags = runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.tmout")

        # compile command
        run_string = f"{self.application} {pose_path} {ref_path} {opts} {flags} > {scorefile}"

        return run_string

    def collect_scores(self, output_dir: str) -> pd.DataFrame:
        """
        Collect scores from TMscore output files.

        This method collects and processes the scores from the output files generated TMscore. It reads the scores, extracts relevant information, and organizes the data into a structured pandas DataFrame.

        Parameters:
            output_dir (str): The directory where TMscore output files are located.

        Returns:
            pd.DataFrame: A DataFrame containing the collected scores.

        Raises:
            RuntimeError: If no TM scores are found in the output files.

        Examples:
            Here is an example of how to use the `collect_scores` method:

            .. code-block:: python

                from tmscore import TMscore

                # Initialize the TMscore class
                tmalign = TMscore()

                # Collect scores
                scores_df = tmscore.collect_scores(
                    output_dir="output/"
                )

                # Print the scores DataFrame
                print(scores_df)

        Further Details:
            - **Score Extraction:** The method reads the output files, extracts relevant scores, and organizes them into a pandas DataFrame.
            - **Validation:** Ensures that the scores are correctly extracted and that no errors occurred during the process.

        This method is designed to streamline the collection and processing of scores from TMscore output files, ensuring that all relevant data is accurately captured and organized.
        """
        def extract_scores(score_path:str) -> pd.Series:
            '''
            extract TM scores from scorefile, return a Series
            '''

            tm_scores = {}
            with open(score_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    # extract scores
                    if line.startswith("TM-score"):
                        tm_scores['TM_score_ref'] = float(line.split()[2])
                    elif line.startswith("MaxSub-score"):
                        tm_scores['MaxSub_score'] = float(line.split()[1])
                    elif line.startswith("GDT-TS-score"):
                        tm_scores['GDT-TS_score'] = float(line.split()[1])
                    elif line.startswith("GDT-HA-score"):
                        tm_scores['GDT-HA_score'] = float(line.split()[1])
            tm_scores['description'] = os.path.splitext(os.path.basename(score_path))[0]

            # check if scores were present in scorefile
            if len(list(tm_scores)) < 2:
                raise RuntimeError(f"Could not find any TM scores in {score_path}!")
            return pd.Series(tm_scores)

        # collect scorefiles
        scorefiles = glob.glob(os.path.join(output_dir, "*.tmout"))

        scores = [extract_scores(file) for file in scorefiles]
        scores = pd.DataFrame(scores).reset_index(drop=True)

        return scores
