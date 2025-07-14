"""
ColabFold Module
================

This module provides functionality to integrate ColabFold within the ProtFlow framework, enabling the execution of AlphaFold2 runs on ColabFold. It includes tools to handle inputs, execute runs, and process outputs in a structured and automated manner.

Detailed Description
--------------------
The `ColabFold` class encapsulates all necessary functionalities to run AlphaFold2 through ColabFold. It manages the configuration of essential scripts and paths, sets up the environment, and handles the execution of prediction processes. The class also includes methods for collecting and processing output data, ensuring the results are organized and accessible for further analysis within the ProtFlow ecosystem.

This module streamlines the integration of ColabFold into larger computational workflows by supporting the automatic setup of job parameters, execution of ColabFold commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ColabFold` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the prediction process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ColabFold` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from colabfold import ColabFold

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = LocalJobStarter(max_cores=4)

    # Initialize the ColabFold class
    colabfold = ColabFold()

    # Run the prediction process
    results = colabfold.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="--msa-mode single-sequence",
        pose_options=None,
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- **Edge Cases:** The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the prediction process.
- **Customizability:** Users can customize the prediction process through multiple parameters, including the number of diffusions, specific options for the ColabFold script, and options for handling pose-specific parameters.
- **Integration:** The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate ColabFold into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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

# dependencies
import pandas as pd

# custom
import protflow.config
from protflow.runners import Runner, RunnerOutput, prepend_cmd
from protflow.poses import Poses, description_from_path
from protflow.jobstarters import JobStarter, split_list

class PLACER(Runner):
    """
    PLACER Class
    ===============

    The `PLACER` class is a specialized class designed to facilitate the execution of PLACER within the PLACER environment as part of the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with PLACER prediction processes.

    Detailed Description
    --------------------
    The `PLACER` class manages all aspects of running PLACER predictions. It handles the configuration of necessary scripts and executables, prepares the environment for the prediction processes, and executes the prediction commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to PLACER scripts and necessary directories.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of PLACER prediction commands.
        - Collecting and processing output data into a pandas DataFrame.
        - Overwriting previous results if specified.

    Returns
    -------
    An instance of the `PLACER` class, configured to run PLACER prediction processes and handle outputs efficiently.


    Examples
    --------
    Here is an example of how to initialize and use the `ColabFold` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from protflow.tools.placer import PLACER

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ColabFold class
        PLACER = PLACER()

        # Run the prediction process
        results = PLACER.run(
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
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the prediction process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The PLACER class is intended for researchers and developers who need to perform PLACER predictions as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.PLACER_SCRIPT_PATH, python_path: str = protflow.config.PLACER_PYTHON_PATH, pre_cmd:str=protflow.config.PLACER_PRE_CMD, jobstarter: str = None) -> None:
        """
        __init__ Method
        ===============

        The `__init__` method initializes an instance of the `PLACER` class, setting up necessary configurations for running PLACER predictions within the ProtFlow framework.

        Detailed Description
        --------------------
        This method sets up the paths to the PLACER script and initializes default values for various attributes required for running predictions. It also allows for the optional configuration of a job starter.

        Parameters:
            script_path (str, optional): The path to the PLACER script. Defaults to `protflow.config.PLACER_SCRIPT_PATH`.
            jobstarter (JobStarter, optional): An instance of the `JobStarter` class for managing job execution. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the `script_path` is not provided.

        Examples
        --------
        Here is an example of how to initialize the `ColabFold` class:

        .. code-block:: python

            from placer import PLACER

            # Initialize the PLACER class
            PLACER = PLACER(script_path='/path/to/run_PLACER.py', jobstarter=jobstarter)
        """
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under COLABFOLD_DIR_PATH.")

        self.python_path = python_path
        self.script_path = script_path
        self.name = "placer.py"
        self.pre_cmd = pre_cmd
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "placer.py"

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, options: str = None, pose_options: str = None, jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        run Method
        ==========

        The `run` method of the `PLACER` class executes PLACER predictions within the ProtFlow framework. It manages the setup, execution, and result collection processes, providing a streamlined way to integrate PLACER predictions into larger computational workflows.

        Detailed Description
        --------------------
        This method orchestrates the entire prediction process, from preparing input data and configuring the environment to running the prediction commands and collecting the results. The method supports batch processing of input FASTA files and handles various edge cases, such as overwriting existing results and managing job starter options.

        Parameters:
            poses (Poses): The Poses object containing the protein data. Poses have to be single-chain .fasta files!
            prefix (str): A prefix used to name and organize the output files.
            nstruct (int, optional): How many structures should be generated for each pose. Default is 1.
            options (str, optional): Additional options for the AlphaFold2 prediction commands. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters during prediction. Defaults to None.
            overwrite (bool, optional): If True, existing results will be overwritten. Defaults to False.
            num_batches (int, optional): Batch individual jobs. Defaults to None.

        Returns:
            RunnerOutput: An object containing the results of the AlphaFold2 predictions, organized in a pandas DataFrame.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the methods.
            TypeError: If pose options are not of the expected type.

        Examples
        --------
        Here is an example of how to use the `run` method of the `ColabFold` class:

        .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import JobStarter
            from placer import PLACER

            # Create instances of necessary classes
            poses = Poses()
            jobstarter = JobStarter()

            # Initialize the ColabFold class
            PLACER = PLACER()

            # Run the prediction process
            results = PLACER.run(
                poses=poses,
                prefix="experiment_1",
                jobstarter=jobstarter,
                nstruct=100,
                overwrite=True
            )

            # Access and process the results
            print(results)
        
        Further Details
        ---------------
            - **Overwrite Handling:** If `overwrite` is set to True, the method will clean up previous results, ensuring that the new predictions do not get mixed up with old data.
            - **Job Starter Configuration:** The method allows for flexible job management by accepting a `JobStarter` instance. If not provided, it uses the default job starter associated with the poses.
            - **Score Collection:** The method gathers the prediction scores and relevant data into a pandas DataFrame, facilitating easy analysis and integration with other ProtFlow components.
            - **Error Handling:** Robust error handling is incorporated to manage issues such as missing files or incorrect configurations, ensuring that the process can be debugged and verified efficiently.
        """
       
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # define output directory
        os.makedirs(out_dir := os.path.join(work_dir, "output"), exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"placer_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        if overwrite:
            if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        # prepare pose options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # write cmds:
        cmds = [self.write_cmd(pose, output_dir=out_dir, options=options, pose_options=pose_opt, nstruct=nstruct) for pose, pose_opt in zip(poses.poses_list(), pose_options)]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # batch commands 
        if num_batches:
            cmds = split_list(cmds, n_sublists=num_batches)
            cmds = ["; ".join(sub) for sub in cmds]

        # run
        logging.info(f"Starting PLACER predictions of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="placer",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        logging.info(f"Predictions finished, starting to collect scores.")
        scores = collect_scores(work_dir=out_dir)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, pose_path: str, output_dir: str, options: str = None, pose_options: str = None, nstruct: int = 1):
        """
        write_cmd Method
        ================

        The `write_cmd` method constructs the command string necessary to run the PLACER script with the specified options and input files.

        Detailed Description
        --------------------
        This method generates the command string used to execute the PLACER script. It incorporates various options and pose-specific parameters provided by the user.

        Parameters:
            pose_path (str): Path to the input PDB file.
            output_dir (str): Directory where the prediction outputs will be stored.
            options (str, optional): Additional options for the PLACER script. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters. Defaults to None.

        Returns:
            str: The constructed command string.

        Raises:
            None

        Examples
        --------
        Here is an example of how to use the `write_cmd` method:

        .. code-block:: python

            # Write the command to run PLACER
            cmd = PLACER.write_cmd(pose_path='/path/to/pose.pdb', output_dir='/path/to/output_dir', nstruct=100, options='--ignore_ligand_hydrogens', pose_options='--ligand_file /path/to/ligand.mol2')
        """
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""
        return f"{self.python_path} {self.script_path} --ifile {pose_path} --odir {output_dir} --nsamples {nstruct} {opts} {flags}"

def collect_scores(work_dir: str) -> pd.DataFrame:
    """
    collect_scores Method
    =====================

    The `collect_scores` method collects and processes the prediction scores from the PLACER output, organizing them into a pandas DataFrame for further analysis.

    Detailed Description
    --------------------
    This method gathers the prediction scores from the output files generated by PLACER. It processes these scores and organizes them into a structured DataFrame, which includes various statistical measures.

    Parameters:
        work_dir (str): The working directory where the PLACER outputs are stored.

    Returns:
        pd.DataFrame: A DataFrame containing the collected and processed scores.

    Raises:
        FileNotFoundError: If no output files are found in the specified directory.

    Examples
    --------
    Here is an example of how to use the `collect_scores` method:

    .. code-block:: python

        # Collect and process the prediction scores
        scores = collect_scores(work_dir='/path/to/work_dir')     
    
    Further Details
    ---------------
        - **CSV and PDB File Parsing:** The method identifies and parses CSV and PDB files generated by PLACER, extracting relevant score information from these files.
        - **PDB Splitting:** The method splits multimodel output PDBs into single files.
        - **Pose Location:** The final DataFrame includes the file paths to the predicted PDB files, facilitating easy access for further analysis or visualization.
    """
    
    # no idea why, but BioPython can't handle the models produced by PLACER, so this is a workaround
    def split_model_sections(input_file, filename, output_dir):
        """
        Reads a text file, splits the content between all occurrences of "MODEL" and "ENDMDL",
        and writes each section to a separate file in the specified output directory.
        
        Args:
            input_file (str): Path to the input file.
            filename (str): Prefix for output files.
            output_dir (str): Directory where the output files will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        model_counter = 0
        model_data = []
        paths = []
        
        for line in lines:
            model_data.append(line)
            if line.startswith("ENDMDL"):
                output_file = os.path.join(output_dir, f"{filename}_{str(model_counter+1).zfill(4)}.pdb")
                with open(output_file, 'w') as out_f:
                    out_f.writelines(model_data)
                paths.append(os.path.abspath(output_file))
                model_counter += 1
                model_data = []

        return paths
            


    # collect all output directories, ignore mmseqs dirs
    out_csvs = glob(os.path.join(work_dir, "*.csv"))
    if not out_csvs:
        raise FileNotFoundError(f"Could not find any csv files at {work_dir}")
    
    # combine all score files
    out_df = pd.concat([pd.read_csv(csv) for csv in out_csvs])

    # iterate over input, create directories, split pdbs
    scores = []
    for in_pose, df in out_df.groupby("label"):
        df.sort_values("model_idx", ascending=True, inplace=True)
        os.makedirs(pose_dir := os.path.join(work_dir, in_pose), exist_ok=True)
        #out_poses = load_structure_from_pdbfile(os.path.join(work_dir, f"{in_pose}_model.pdb"), all_models=True)
        # no idea why, but BioPython can't handle the models produced by PLACER, so this is a workaround
        df["multimodel_path"] = os.path.abspath(os.path.join(work_dir, f"{in_pose}_model.pdb"))
        df["location"] = split_model_sections(os.path.join(work_dir, f"{in_pose}_model.pdb"), in_pose, pose_dir)
        df["description"] = [description_from_path(path) for path in df["location"].to_list()]
        scores.append(df)

    scores = pd.concat(scores)
    scores.reset_index(drop=True, inplace=True)

    return scores
