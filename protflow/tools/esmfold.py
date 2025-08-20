"""
ESMFold Module
==============

This module provides the functionality to integrate ESMFold within the ProtFlow framework. It offers tools to run ESMFold, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `ESMFold` class encapsulates the functionality necessary to execute ESMFold runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of folding processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of ESMFold into larger computational workflows. It supports the automatic setup of job parameters, execution of ESMFold commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ESMFold` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the folding process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ESMFold` class within a ProtFlow pipeline on a SLURM based queueing system:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from esmfold import ESMFold

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = SbatchArrayJobStarter(max_cores=10, gpus=1) # 1 gpu per node, 10 nodes at once.

    # Initialize the ESMFold class
    esmfold = ESMFold()

    # Run the folding process
    results = esmfold.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        num_batches=5,
        options="--additional_option=value",
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the folding process.
    - Customizability: Users can customize the folding process through multiple parameters, including the number of batches, specific options for the ESMFold script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate ESMFold into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
from glob import glob
import shutil
import numpy as np

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path, runners
from protflow.poses import Poses
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..jobstarters import JobStarter

class ESMFold(Runner):
    """
    ESMFold Class
    =============

    The `ESMFold` class is a specialized class designed to facilitate the execution of ESMFold within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with ESMFold processes.

    Detailed Description
    --------------------
    The `ESMFold` class manages all aspects of running ESMFold simulations. It handles the configuration of necessary scripts and executables, prepares the environment for folding processes, and executes the folding commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to ESMFold scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of ESMFold commands with support for multiple batches.
        - Collecting and processing output data into a pandas DataFrame.
        - Preparing input FASTA files for ESMFold predictions and managing output directories.

    Returns
    -------
    An instance of the `ESMFold` class, configured to run ESMFold processes and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        KeyError: If forbidden options are included in the command options.

    Examples
    --------
    Here is an example of how to initialize and use the `ESMFold` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from esmfold import ESMFold

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ESMFold class
        esmfold = ESMFold()

        # Run the folding process
        results = esmfold.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            num_batches=5,
            options="--additional_option=value",
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the folding process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ESMFold class is intended for researchers and developers who need to perform ESMFold simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, python_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter|None = None) -> None:
        """
        Initialize the ESMFold class with necessary configurations.

        This method sets up the ESMFold class, configuring paths to essential scripts and Python executables, and setting up the environment for executing ESMFold processes.

        Parameters:
            python_path (str, optional): The path to the Python executable used for running ESMFold. Defaults to the value specified in `protflow.config.ESMFOLD_PYTHON_PATH`.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Raises:
            ValueError: If no path is set for the ESMFold scripts or Python executable.

        Examples:
            Here is an example of how to initialize the `ESMFold` class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from esmfold import ESMFold

                # Initialize the ESMFold class with default configurations
                esmfold = ESMFold()

                # Initialize the ESMFold class with a custom Python path and jobstarter
                custom_python_path = "/path/to/custom/python"
                jobstarter = JobStarter()
                esmfold = ESMFold(python_path=custom_python_path, jobstarter=jobstarter)

        Further Details:
            - **Configuration:** This method sets the script path for ESMFold inference and the Python path. If these are not correctly set in the configuration, a ValueError is raised.
            - **Initialization:** The method initializes necessary attributes, including the script path, Python path, jobstarter, and other configurations needed for running ESMFold processes.

        This method prepares the ESMFold class for running folding simulations, ensuring that all necessary configurations and paths are correctly set up.
        """
        # setup config
        config = require_config()
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        self.script_path = os.path.join(self.script_dir, "esmfold_inference.py")
        self.python_path = python_path or load_config_path(config, "ESMFOLD_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "ESMFOLD_PRE_CMD", is_pre_cmd=True)

        # runner setup
        self.name = "esmfold.py"
        self.index_layers = 0
        self.jobstarter = jobstarter

    def __str__(self):
        return "esmfold.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str = None, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        Execute the ESMFold process with given poses and jobstarter configuration.

        This method sets up and runs the ESMFold process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            options (str, optional): Additional options for the ESMFold script. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            num_batches (int, optional): The number of batches to split the input poses into for parallel processing. Defaults to None.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the ESMFold process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            KeyError: If forbidden options are included in the command options.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from esmfold import ESMFold

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the ESMFold class
                esmfold = ESMFold()

                # Run the folding process
                results = esmfold.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    num_batches=5,
                    options="--additional_option=value",
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Input Preparation:** The method prepares input FASTA files by splitting the input poses into the specified number of batches, optimizing the use of parallel computing resources.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized into a pandas DataFrame and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the folding process to their specific needs.

        This method is designed to streamline the execution of ESMFold processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze folding simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip ESMFold.
        scorefile = os.path.join(work_dir, f"ESMFold_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # set up esm-specific directories
        os.makedirs((fasta_dir := f"{work_dir}/input_fastas"), exist_ok=True)
        os.makedirs((esm_preds_dir := f"{work_dir}/esm_preds"), exist_ok=True)

        # prep fastas -> split them into number of batches to run
        num_batches = num_batches or jobstarter.max_cores
        pose_fastas = self.prep_fastas_for_prediction(poses=poses.df['poses'].to_list(), fasta_dir=fasta_dir, max_filenum=num_batches)

        # check if interfering options were set
        forbidden_options = ['--fasta', '--output_dir']
        if options and any(opt in options for opt in forbidden_options) :
            raise KeyError("Options must not contain '--fasta' or '--output_dir'!\nThese will be set automatically.")

        # write ESMFold cmds:
        cmds = [self.write_cmd(pose, output_dir=esm_preds_dir, options=options) for pose in pose_fastas]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        logging.info(f"Starting prediction of len {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="ESMFold",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        logging.info("Predictions finished, starting to collect scores.")
        scores = collect_esmfold_scores(work_dir=work_dir)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def prep_fastas_for_prediction(self, poses:list[str], fasta_dir:str, max_filenum:int) -> list[str]:
        """
        Prepare input FASTA files for ESMFold predictions.

        This method splits the input poses into the specified number of batches, prepares the FASTA files, and writes them to the specified directory for ESMFold predictions.

        Parameters:
            poses (list[str]): List of paths to FASTA files.
            fasta_dir (str): Directory to which the new FASTA files should be written.
            max_filenum (int): Maximum number of FASTA files to be written.

        Returns:
            list[str]: List of paths to the prepared FASTA files.

        Examples:
            Here is an example of how to use the `parse_fastas_for_prediction` method:

            .. code-block:: python

                from esmfold import ESMFold

                # Initialize the ESMFold class
                esmfold = ESMFold()

                # Prepare FASTA files for prediction
                fasta_paths = esmfold.parse_fastas_for_prediction(
                    poses=["pose1.fa", "pose2.fa", "pose3.fa"],
                    fasta_dir="/path/to/fasta_dir",
                    max_filenum=2
                )

                # Access the prepared FASTA files
                print(fasta_paths)

        Further Details:
            - **Input Preparation:** The method merges and splits the input FASTA files into the specified number of batches. It ensures that the FASTA files are correctly formatted and written to the specified directory.
            - **Customization:** Users can specify the maximum number of FASTA files to be created, allowing for flexibility in managing input data for parallel processing.
            - **Output Management:** The method returns a list of paths to the newly created FASTA files, which are ready for ESMFold predictions.

        This method is designed to facilitate the preparation of input data for ESMFold, ensuring that the input FASTA files are organized and ready for processing.
        """
        def mergefastas(files:list, path:str, replace:bool=None) -> str:
            '''
            Merges Fastas located in <files> into one single fasta-file called <path>
            '''
            fastas = list()
            for fp in files:
                with open(fp, 'r', encoding="UTF-8") as f:
                    fastas.append(f.read().strip())

            if replace:
                fastas = [x.replace(replace[0], replace[1]) for x in fastas]

            with open(path, 'w', encoding="UTF-8") as f:
                f.write("\n".join(fastas))

            return path

        # determine how to split the poses into <max_gpus> fasta files:
        splitnum = len(poses) if len(poses) < max_filenum else max_filenum
        poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]

        # Write fasta files according to the fasta_split determined above and then return:
        return [mergefastas(files=poses, path=f"{fasta_dir}/fasta_{str(i+1).zfill(4)}.fa", replace=("/",":")) for i, poses in enumerate(poses_split)]

    def write_cmd(self, pose_path:str, output_dir:str, options:str):
        """
        Write the command to run ESMFold with the given parameters.

        This method constructs the command line instruction needed to execute the ESMFold script with the specified pose path, output directory, and additional options.

        Parameters:
            pose_path (str): The path to the input FASTA file.
            output_dir (str): The directory where the ESMFold outputs will be stored.
            options (str, optional): Additional command-line options for the ESMFold script. Defaults to None.

        Returns:
            str: The constructed command line instruction for running ESMFold.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from esmfold import ESMFold

                # Initialize the ESMFold class
                esmfold = ESMFold()

                # Write the ESMFold command
                cmd = esmfold.write_cmd(
                    pose_path="/path/to/pose.fa",
                    output_dir="/path/to/output_dir",
                    options="--additional_option=value"
                )

                # Access the command
                print(cmd)

        Further Details:
            - **Command Construction:** The method parses additional options and constructs the full command line instruction needed to run the ESMFold script. It ensures that the mandatory arguments, such as the FASTA file and output directory, are correctly included.
            - **Customization:** Users can specify additional command-line options to customize the execution of the ESMFold script, allowing for flexibility in configuring the prediction process.

        This method is designed to facilitate the execution of ESMFold by constructing the necessary command line instructions, ensuring that all required parameters and options are included.
        """
        # parse options
        opts, flags = runners.parse_generic_options(options, None)

        return f"{self.python_path} {self.script_path} --fasta {pose_path} --output_dir {output_dir} {runners.options_flags_to_string(opts, flags, sep='--')}"

def collect_esmfold_scores(work_dir:str) -> pd.DataFrame:
    """
    Collect and process the scores from ESMFold output.

    This method collects the JSON and PDB output files from ESMFold predictions, processes the data, and organizes it into a pandas DataFrame.

    Parameters:
        work_dir (str): The working directory where ESMFold output files are stored.
        scorefile (str): The path to the JSON file where the collected scores will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores and corresponding file locations.

    Examples:
        Here is an example of how to use the `collect_esmfold_scores` method:

        .. code-block:: python

            from esmfold import ESMFold

            # Initialize the ESMFold class
            esmfold = ESMFold()

            # Collect scores from ESMFold output
            scores_df = collect_esmfold_scores(
                work_dir="/path/to/work_dir",
                scorefile="/path/to/scorefile.json"
            )

            # Access the collected scores
            print(scores_df)

    Further Details:
        - **Output Collection:** The method scans the working directory for JSON and PDB output files, reads the JSON files into a DataFrame, and merges it with the locations of the PDB files.
        - **Data Organization:** The method organizes the collected data into a structured DataFrame, making it accessible for further analysis and ensuring that the scores are saved to the specified JSON file.
        - **Output Management:** The method also ensures that the temporary directories and files used during the prediction process are cleaned up, maintaining a tidy working environment.

    This method is designed to streamline the collection and processing of ESMFold output data, ensuring that the results are organized and accessible for further analysis.
    """
    # collect all .json files
    pdb_dir = os.path.join(work_dir, "esm_preds")
    fl = glob(f"{pdb_dir}/fasta_*/*.json")
    pl = glob(f"{pdb_dir}/fasta_*/*.pdb")

    output_dir = os.path.join(work_dir, 'output_pdbs')
    os.makedirs(output_dir, exist_ok=True)
    pl = [shutil.copy(pdb, output_dir) for pdb in pl]
    # create dataframe containing new locations
    df_pdb = pd.DataFrame({'location': pl, 'description': [os.path.splitext(os.path.basename(pdb))[0] for pdb in pl]})

    # read the files, add origin column, and concatenate into single DataFrame:
    df = pd.concat([pd.read_json(f) for f in fl]).reset_index(drop=True)

    # merge with df containing locations
    df = df.merge(df_pdb, on='description')
    shutil.rmtree(pdb_dir)

    return df
