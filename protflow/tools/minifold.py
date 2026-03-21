"""
Minifold Module
==============

This module provides the functionality to integrate Minifold within the ProtFlow framework. It offers tools to run Minifold, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `Minifold` class encapsulates the functionality necessary to execute Minifold runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of folding processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of Minifold into larger computational workflows. It supports the automatic setup of job parameters, execution of Minifold commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `Minifold` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the folding process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `Minifold` class within a ProtFlow pipeline on a SLURM based queueing system:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from protflow.tools.minifold import Minifold

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = SbatchArrayJobStarter(max_cores=10, gpus=1) # 1 gpu per node, 10 nodes at once.

    # Initialize the Minifold class
    minifold = Minifold()

    # Run the folding process
    results = minifold.run(
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
    - Customizability: Users can customize the folding process through multiple parameters, including the number of batches, specific options for the Minifold script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate Minifold into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import numpy as np

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path, runners
from protflow.poses import Poses, description_from_path
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..jobstarters import JobStarter

class Minifold(Runner):
    """
    Minifold Class
    =============

    The `Minifold` class is a specialized class designed to facilitate the execution of Minifold within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with Minifold processes.

    Detailed Description
    --------------------
    The `Minifold` class manages all aspects of running Minifold simulations. It handles the configuration of necessary scripts and executables, prepares the environment for folding processes, and executes the folding commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to Minifold scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of Minifold commands with support for multiple batches.
        - Collecting and processing output data into a pandas DataFrame.
        - Preparing input FASTA files for Minifold predictions and managing output directories.

    Returns
    -------
    An instance of the `Minifold` class, configured to run Minifold processes and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        KeyError: If forbidden options are included in the command options.

    Examples
    --------
    Here is an example of how to initialize and use the `Minifold` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from minifold import Minifold

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the Minifold class
        minifold = Minifold()

        # Run the folding process
        results = minifold.run(
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

    The Minifold class is intended for researchers and developers who need to perform Minifold simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, cache_dir: str = None, python_path: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:
        """
        Initialize the Minifold class with necessary configurations.

        This method sets up the Minifold class, configuring paths to essential scripts and Python executables, and setting up the environment for executing Minifold processes.

        Parameters:
            python_path (str, optional): The path to the Python executable used for running Minifold. Defaults to the value specified in `protflow.config.ESMFOLD_PYTHON_PATH`.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Raises:
            ValueError: If no path is set for the Minifold scripts or Python executable.

        Examples:
            Here is an example of how to initialize the `Minifold` class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from minifold import Minifold

                # Initialize the Minifold class with default configurations
                minifold = Minifold()

                # Initialize the Minifold class with a custom Python path and jobstarter
                custom_python_path = "/path/to/custom/python"
                jobstarter = JobStarter()
                minifold = Minifold(python_path=custom_python_path, jobstarter=jobstarter)

        Further Details:
            - **Configuration:** This method sets the script path for Minifold inference and the Python path. If these are not correctly set in the configuration, a ValueError is raised.
            - **Initialization:** The method initializes necessary attributes, including the script path, Python path, jobstarter, and other configurations needed for running Minifold processes.

        This method prepares the Minifold class for running folding simulations, ensuring that all necessary configurations and paths are correctly set up.
        """
        # setup config
        config = require_config()

        self.script_path = load_config_path(config, "MINIFOLD_SCRIPT_PATH")

        self.cache_dir = cache_dir or os.path.join(os.path.dirname(self.script_path), "minifold_cache")

        self.python_path = python_path or load_config_path(config, "MINIFOLD_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "MINIFOLD_PRE_CMD", is_pre_cmd=True)

        # runner setup
        self.name = "minifold.py"
        self.index_layers = 0
        self.jobstarter = jobstarter

    def __str__(self):
        return "minifold.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str = None, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        Execute the Minifold process with given poses and jobstarter configuration.

        This method sets up and runs the Minifold process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            options (str, optional): Additional options for the Minifold script. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            num_batches (int, optional): The number of batches to split the input poses into for parallel processing. Defaults to None.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the Minifold process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            KeyError: If forbidden options are included in the command options.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from minifold import Minifold

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the Minifold class
                minifold = Minifold()

                # Run the folding process
                results = minifold.run(
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

        This method is designed to streamline the execution of Minifold processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze folding simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip Minifold.
        scorefile = os.path.join(work_dir, f"minifold_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # set up esm-specific directories
        os.makedirs((fasta_dir := f"{work_dir}/input_fastas"), exist_ok=True)
        os.makedirs((minifold_preds_dir := f"{work_dir}/minifold_preds"), exist_ok=True)

        # prep fastas -> split them into number of batches to run
        num_batches = num_batches or jobstarter.max_cores
        pose_fastas = self.prep_fastas_for_prediction(poses=poses.df['poses'].to_list(), fasta_dir=fasta_dir, max_filenum=num_batches)

        # check if interfering options were set
        forbidden_options = ['--out_dir', "--cache"]
        if options and any(opt in options for opt in forbidden_options) :
            raise KeyError(f"Options must not contain any of {forbidden_options}\nThese will be set automatically.")

        # write Minifold cmds:
        cmds = [self.write_cmd(pose, output_dir=minifold_preds_dir, options=options) for pose in pose_fastas]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        logging.info(f"Starting prediction of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="Minifold",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        logging.info("Predictions finished, starting to collect scores.")
        scores = collect_scores(work_dir=work_dir)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError(f"Number of output poses ({len(scores.index)}) is smaller than number of input poses {len(poses.df.index)}. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def prep_fastas_for_prediction(self, poses:list[str], fasta_dir:str, max_filenum:int) -> list[str]:
        """
        Prepare input FASTA files for Minifold predictions.

        This method splits the input poses into the specified number of batches, prepares the FASTA files, and writes them to the specified directory for Minifold predictions.

        Parameters:
            poses (list[str]): List of paths to FASTA files.
            fasta_dir (str): Directory to which the new FASTA files should be written.
            max_filenum (int): Maximum number of FASTA files to be written.

        Returns:
            list[str]: List of paths to the prepared FASTA files.

        Examples:
            Here is an example of how to use the `parse_fastas_for_prediction` method:

            .. code-block:: python

                from minifold import Minifold

                # Initialize the Minifold class
                minifold = Minifold()

                # Prepare FASTA files for prediction
                fasta_paths = minifold.parse_fastas_for_prediction(
                    poses=["pose1.fa", "pose2.fa", "pose3.fa"],
                    fasta_dir="/path/to/fasta_dir",
                    max_filenum=2
                )

                # Access the prepared FASTA files
                print(fasta_paths)

        Further Details:
            - **Input Preparation:** The method merges and splits the input FASTA files into the specified number of batches. It ensures that the FASTA files are correctly formatted and written to the specified directory.
            - **Customization:** Users can specify the maximum number of FASTA files to be created, allowing for flexibility in managing input data for parallel processing.
            - **Output Management:** The method returns a list of paths to the newly created FASTA files, which are ready for Minifold predictions.

        This method is designed to facilitate the preparation of input data for Minifold, ensuring that the input FASTA files are organized and ready for processing.
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
        return [mergefastas(files=poses, path=os.path.join(fasta_dir, f"fasta_{str(i+1).zfill(4)}.fa"), replace=("/",":")) for i, poses in enumerate(poses_split)]

    def write_cmd(self, pose_path:str, output_dir:str, options:str):
        """
        Write the command to run Minifold with the given parameters.

        This method constructs the command line instruction needed to execute the Minifold script with the specified pose path, output directory, and additional options.

        Parameters:
            pose_path (str): The path to the input FASTA file.
            output_dir (str): The directory where the Minifold outputs will be stored.
            options (str, optional): Additional command-line options for the Minifold script. Defaults to None.

        Returns:
            str: The constructed command line instruction for running Minifold.

        Examples:
            Here is an example of how to use the `write_cmd` method:

            .. code-block:: python

                from minifold import Minifold

                # Initialize the Minifold class
                minifold = Minifold()

                # Write the Minifold command
                cmd = minifold.write_cmd(
                    pose_path="/path/to/pose.fa",
                    output_dir="/path/to/output_dir",
                    options="--additional_option=value"
                )

                # Access the command
                print(cmd)

        Further Details:
            - **Command Construction:** The method parses additional options and constructs the full command line instruction needed to run the Minifold script. It ensures that the mandatory arguments, such as the FASTA file and output directory, are correctly included.
            - **Customization:** Users can specify additional command-line options to customize the execution of the Minifold script, allowing for flexibility in configuring the prediction process.

        This method is designed to facilitate the execution of Minifold by constructing the necessary command line instructions, ensuring that all required parameters and options are included.
        """
        # parse options
        opts, flags = runners.parse_generic_options(options, None)

        return f"{self.python_path} {self.script_path} {pose_path} --out_dir {output_dir} --cache {self.cache_dir} {runners.options_flags_to_string(opts, flags, sep='--')}"

def collect_scores(work_dir:str) -> pd.DataFrame:
    """
    Collect and process the scores from Minifold output.

    This method collects the PDB output files from Minifold predictions, processes the data, and organizes it into a pandas DataFrame.

    Parameters:
        work_dir (str): The working directory where Minifold output files are stored.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores and corresponding file locations.

    Examples:
        Here is an example of how to use the `collect_minifold_scores` method:

        .. code-block:: python

            from minifold import Minifold

            # Initialize the Minifold class
            minifold = Minifold()

            # Collect scores from Minifold output
            scores_df = collect_minifold_scores(
                work_dir="/path/to/work_dir",
                scorefile="/path/to/scorefile.json"
            )

            # Access the collected scores
            print(scores_df)

    Further Details:
        - **Output Collection:** The method scans the working directory for JSON and PDB output files, reads the JSON files into a DataFrame, and merges it with the locations of the PDB files.
        - **Data Organization:** The method organizes the collected data into a structured DataFrame, making it accessible for further analysis and ensuring that the scores are saved to the specified JSON file.
        - **Output Management:** The method also ensures that the temporary directories and files used during the prediction process are cleaned up, maintaining a tidy working environment.

    This method is designed to streamline the collection and processing of Minifold output data, ensuring that the results are organized and accessible for further analysis.
    """

    def extract_plddt_from_pdb(pdb_path:str, ca_only:bool=True) -> list: 
        headers = ["record", "atom_num", "atom_name", "res_name", "chain", "res_num", "x", "y", "z", "occupancy", "bfactor", "element"]
        df = pd.read_csv(pdb_path, skiprows=1, names=headers, sep=r"\s+")
        df = df[df['record'].isin(['ATOM', 'HETATM'])]
        if ca_only: # minifold assigns per-residue plddts --> extract plddt once per res 
            df = df[df['atom_name'] == "CA"]
        return df["bfactor"].to_list()

    # collect all .pdb files
    pl = glob(os.path.join(work_dir, "minifold_preds", "minifold_results_*", "*.pdb"))

    # extract plddts from pdbs, create score table
    data = []
    for pdb in pl:
        plddt_list = extract_plddt_from_pdb(pdb)
        data.append(pd.Series({"location": os.path.abspath(pdb),
                               "description": description_from_path(pdb),
                               "mean_per_res_plddt": sum(plddt_list) / len(plddt_list),
                               "per_res_plddt": plddt_list}))
        
    data = pd.DataFrame(data)

    return data
