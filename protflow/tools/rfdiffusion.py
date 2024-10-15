"""
RFdiffusion Module
==================

This module provides the functionality to integrate RFdiffusion within the ProtFlow framework. It offers tools to run RFdiffusion, handle its inputs and outputs, and process the resulting data in a structured and automated manner. 

Detailed Description
--------------------
The `RFdiffusion` class encapsulates the functionality necessary to execute RFdiffusion runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of diffusion processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.
The module is designed to streamline the integration of RFdiffusion into larger computational workflows. It supports the automatic setup of job parameters, execution of RFdiffusion commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `RFdiffusion` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the diffusion process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `RFdiffusion` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from rfdiffusion import RFdiffusion

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the RFdiffusion class
    rfdiffusion = RFdiffusion()

    # Run the diffusion process
    results = rfdiffusion.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        num_diffusions=3,
        options="inference.num_designs=10",
        pose_options=["inference.input_pdb='input.pdb'"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the diffusion process.
    - Customizability: Users can customize the diffusion process through multiple parameters, including the number of diffusions, specific options for the RFdiffusion script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate RFdiffusion into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import re
from typing import Any
import numpy as np
import random # for delay before rfdiffusion runs

# dependencies
import pandas as pd

# custom
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
import protflow.config
from protflow.residues import ResidueSelection
from protflow.runners import Runner, col_in_df
from protflow.runners import RunnerOutput


class RFdiffusion(Runner):
    """
    RFdiffusion Class
    =================

    The `RFdiffusion` class is a specialized class designed to facilitate the execution of RFdiffusion within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with RFdiffusion processes.

    Detailed Description
    --------------------
    The `RFdiffusion` class manages all aspects of running RFdiffusion simulations. It handles the configuration of necessary scripts and executables, prepares the environment for diffusion processes, and executes the diffusion commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to RFdiffusion scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of RFdiffusion commands with support for multiple diffusions.
        - Collecting and processing output data into a pandas DataFrame.
        - Updating motifs based on RFdiffusion outputs and remapping residue selections.

    Returns
    -------
    An instance of the `RFdiffusion` class, configured to run RFdiffusion processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If motifs are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `RFdiffusion` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from rfdiffusion import RFdiffusion

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the RFdiffusion class
        rfdiffusion = RFdiffusion()

        # Run the diffusion process
        results = rfdiffusion.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            num_diffusions=3,
            options="inference.num_designs=10",
            pose_options=["inference.input_pdb='input.pdb'"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the diffusion process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The RFdiffusion class is intended for researchers and developers who need to perform RFdiffusion simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.RFDIFFUSION_SCRIPT_PATH, python_path: str = protflow.config.RFDIFFUSION_PYTHON_PATH, jobstarter: JobStarter = None) -> None:
        """
        Initialize the RFdiffusion class.

        This constructor sets up the necessary paths to the RFdiffusion script and Python executable, and initializes the job starter. The paths are configured using default values from the ProtFlow configuration, but they can be manually set if required. However, manual setting is generally not recommended due to the potential for misconfiguration.

        Detailed Description
        --------------------
        The `__init__` method initializes the RFdiffusion class by setting up essential paths and configurations. It ensures that the paths to the RFdiffusion script and Python executable are correctly set, and it initializes the job starter object. This setup is crucial for the proper execution of RFdiffusion processes within the ProtFlow framework.

        Parameters
        ----------
        script_path (str, optional): The path to the RFdiffusion script. Defaults to the value specified in the ProtFlow configuration (`protflow.config.RFDIFFUSION_SCRIPT_PATH`).
        python_path (str, optional): The path to the Python executable used to run the RFdiffusion script. Defaults to the value specified in the ProtFlow configuration (`protflow.config.RFDIFFUSION_PYTHON_PATH`).
        jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. If not provided, the default is `None`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided paths are invalid or if there are issues with the configuration.
        
        Examples
        --------
        Here is an example of how to initialize the RFdiffusion class:

        .. code-block:: python
                
            from protflow.jobstarters import JobStarter
            from rfdiffusion import RFdiffusion

            # Initialize the RFdiffusion class
            rfdiffusion = RFdiffusion()

            # Initialize with custom paths
            custom_rfdiffusion = RFdiffusion(
                script_path="/path/to/custom/rfdiffusion_script.py",
                python_path="/path/to/custom/python"
            )

            # Initialize with a job starter
            jobstarter = JobStarter()
            rfdiffusion_with_jobstarter = RFdiffusion(jobstarter=jobstarter)

        Further Details
        ---------------
        - **Path Configuration:** The paths to the RFdiffusion script and Python executable are critical for the correct functioning of the class. It is recommended to use the default paths provided by the ProtFlow configuration unless there is a specific need to customize them.
        - **JobStarter Integration:** The JobStarter object is used to manage job execution, ensuring that RFdiffusion processes are handled efficiently. If a JobStarter is not provided, the class will operate without it, but it is recommended to use one for better job management.

        This method is designed for initializing the RFdiffusion class with the necessary configurations, making it ready for executing RFdiffusion processes within the ProtFlow framework.
        """
        self.script_path = self.search_path(script_path, "RFDIFFUSION_SCRIPT_PATH")
        self.python_path = self.search_path(python_path, "RFDIFFUSION_PYTHON_PATH")
        self.name = "rfdiffusion.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "rfdiffusion.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, num_diffusions: int = 1, options: str = None, pose_options: list[str] = None, overwrite: bool = False, multiplex_poses: int = False, update_motifs: list[str] = None, fail_on_missing_output_poses: bool = False) -> Poses:
        """
        Execute the RFdiffusion process with given poses and jobstarter configuration.

        This method sets up and runs the RFdiffusion process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            num_diffusions (int, optional): The number of diffusions to run for each input pose. Be aware that the number of output poses per input pose is multiplex_poses * num_diffusions! Defaults to 1.
            options (str, optional): Additional options for the RFdiffusion script. Defaults to None.
            pose_options (list[str], optional): A list of pose-specific options for the RFdiffusion script. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            multiplex_poses (int, optional): If specified, create multiple copies of poses to fully utilize parallel computing. Be aware that the number of output poses per input pose is multiplex_poses * num_diffusions! Defaults to False.
            update_motifs (list[str], optional): A list of motifs to update based on the RFdiffusion outputs. Defaults to None.
            fail_on_missing_output_poses (bool, optional): RFdiffusion runs crash sometimes unexpectedly, which might disrupt longer pipelines. Fail if some poses are missing. Defaults to False.


        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the RFdiffusion process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If motifs are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from rfdiffusion import RFdiffusion

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the RFdiffusion class
                rfdiffusion = RFdiffusion()

                # Run the diffusion process
                results = rfdiffusion.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    num_diffusions=3,
                    options="inference.num_designs=10",
                    pose_options=["inference.input_pdb='input.pdb'"],
                    overwrite=True,
                    fail_on_missing_output_poses=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the diffusion process to their specific needs.

        This method is designed to streamline the execution of RFdiffusion processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze diffusion simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # sanity checks
        if multiplex_poses == 1:
            logging.warning(f"Multiplex_poses must be higher than 1 to be effective!")

        # log number of diffusions per backbone
        if multiplex_poses:
            logging.info(f"Total number of diffusions per input pose: {multiplex_poses * num_diffusions}")
            self.index_layers = 2
        else:
            logging.info(f"Total number of diffusions per input pose: {num_diffusions}")

        # setup runner-specific directories
        pdb_dir = os.path.join(work_dir, "output_pdbs")
        if not os.path.isdir(pdb_dir):
            os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip diffusion step.
        scorefile = os.path.join(work_dir, f"rfdiffusion_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            if multiplex_poses:
                self.index_layers += 1
            poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
            if update_motifs:
                self.remap_motifs(
                poses = poses,
                motifs = update_motifs,
                prefix = prefix
            )
            if multiplex_poses:
                poses.reindex_poses(prefix=f"{prefix}_post_multiplex_reindexing", remove_layers=2, force_reindex=True, overwrite=overwrite)
            return poses

        # in case overwrite is set, overwrite previous results.
        if overwrite or not os.path.isfile(scorefile):
            if os.path.isfile(scorefile): os.remove(scorefile)
            for pdb in glob(f"{pdb_dir}/*pdb"):
                if os.path.isfile(trb := pdb.replace(".pdb", ".trb")):
                    os.remove(trb)
                    os.remove(pdb)

        # parse options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # create temporary pose_opts column (makes it easier to match pose_opts when multiplexing input poses)
        poses.df[f"temp_{prefix}_pose_opts"] = pose_options

        # handling of empty poses DataFrame.
        if len(poses) == 0 and pose_options:
            # if no poses are set, but pose_options are provided, create as many jobs as pose_options. output_pdbs must be specified in pose options!
            cmds = [self.write_cmd(pose=None, options=options, pose_opts=pose_option, output_dir=pdb_dir, num_diffusions=num_diffusions) for pose_option in pose_options]
        elif len(poses) == 0 and not pose_options:
            # if neither poses nor pose_options exist: write n=max_cores commands with generic output name.
            cmds = [self.write_cmd(pose=None, options=options, pose_opts="inference.output_prefix=" + os.path.join(pdb_dir, f"diff_{str(i+1).zfill(4)}"), output_dir=pdb_dir, num_diffusions=num_diffusions) for i in range(jobstarter.max_cores)]
        elif multiplex_poses:
            # create multiple copies (specified by multiplex variable) of poses to fully utilize parallel computing:
            poses.duplicate_poses(f"{poses.work_dir}/{prefix}_multiplexed_input_pdbs/", multiplex_poses)
            #self.index_layers += 1
            cmds = [self.write_cmd(pose, options, pose_opts, output_dir=pdb_dir, num_diffusions= num_diffusions) for pose, pose_opts in zip(poses.poses_list(), poses.df[f"temp_{prefix}_pose_opts"].to_list())]
        else:
            # write rfdiffusion cmds
            cmds = [self.write_cmd(pose, options, pose_opts, output_dir=pdb_dir, num_diffusions=num_diffusions) for pose, pose_opts in zip(poses.poses_list(), pose_options)]

        # drop temporary pose_opts col
        poses.df.drop([f"temp_{prefix}_pose_opts"], axis=1, inplace=True)

        # diffuse
        jobstarter.start(
            cmds=cmds,
            jobname="rfdiffusion",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect RFdiffusion outputs
        scores = collect_scores(work_dir=work_dir, rename_pdbs=True)
        if fail_on_missing_output_poses and len(scores.index) < len(poses.df.index) * num_diffusions:
            raise RuntimeError("Number of output poses is smaller than number of input poses * num_diffusions. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # update residue mappings for stored motifs
        poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
        if update_motifs:
            logging.info(f"Updating residue motifs for {update_motifs}.")
            self.remap_motifs(
                poses = poses,
                motifs = update_motifs,
                prefix = prefix
            )

        if multiplex_poses:
            poses.reindex_poses(prefix=f"{prefix}_post_multiplex_reindexing", remove_layers=2, force_reindex=True, overwrite=overwrite)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return poses

    def remap_motifs(self, poses: Poses, motifs: list, prefix: str) -> None:
        """
        Updates ResidueSelection type motifs in poses.df when given prefix of RFdiffusion run.

        This method updates the residue mappings of specified motifs in the poses DataFrame based on the RFdiffusion outputs. It ensures that the motifs are correctly mapped to the new residue indices as generated by the diffusion process.

        Parameters:
            poses (Poses): The Poses object containing the protein structures and associated data.
            motifs (list): A list of motif columns to update in the poses DataFrame.
            prefix (str): The prefix used to identify the relevant columns in the poses DataFrame from the RFdiffusion outputs.

        Returns:
            None

        Raises:
            TypeError: If the motifs are not of the expected type ResidueSelection.

        Further Details:
        - **Motif Update:** The method ensures that the specified motifs in the poses DataFrame are updated with the new residue mappings from the RFdiffusion outputs.
        - **Residue Mapping:** The method uses the reference and halogenated residue indices generated by RFdiffusion to update the motifs.
        - **Integration:** This method integrates seamlessly with the RFdiffusion workflow, ensuring that motifs are correctly remapped after diffusion processes.

        This method is designed to update the residue mappings of motifs in the poses DataFrame, facilitating accurate representation of the protein structures after RFdiffusion processes.
        """
        motifs = prep_motif_input(motifs, poses.df)
        for motif_col in motifs:
            poses.df[motif_col] = update_motif_res_mapping(
                poses.df[motif_col].to_list(),
                poses.df[f"{prefix}_complex_con_ref_pdb_idx"].to_list(),
                poses.df[f"{prefix}_complex_con_hal_pdb_idx"].to_list()
            )

    def write_cmd(self, pose: str, options: str, pose_opts: str, output_dir: str, num_diffusions: int=1) -> str:
        """
        Construct the command to run the RFdiffusion process.

        This method constructs the command string required to execute the RFdiffusion process. It combines the specified options and pose-specific options, ensuring that all necessary parameters are included.

        Parameters:
            pose (str): The path to the input pose file.
            options (str): General options for the RFdiffusion script.
            pose_opts (str): Pose-specific options for the RFdiffusion script.
            output_dir (str): The directory where output files will be saved.
            num_diffusions (int, optional): The number of diffusions to run for each input pose. Defaults to 1.

        Returns:
            str: The constructed command string to execute the RFdiffusion process.

        Raises:
            ValueError: If the provided options or pose_opts are invalid.

        Examples:
            Construct a command for RFdiffusion:
            
            .. code-block:: python
                
                cmd = rfdiffusion.write_cmd("input.pdb", "inference.num_designs=10", "inference.input_pdb='input.pdb'", "/output", 3)

        Further Details:
            - **Option Parsing:** The method parses both general and pose-specific options, ensuring that they are correctly formatted and included in the command string.
            - **Command Construction:** The constructed command string includes the path to the RFdiffusion script, the specified options, and the output directory.
            - **Default Values:** Default values for unspecified options, such as `inference.num_designs`, are included to ensure the command string is complete.

        This method is designed to create a fully-formed command string for running RFdiffusion, making it easier to execute diffusion processes with the desired parameters.
        """
        # parse description:
        if pose:
            desc = os.path.splitext(os.path.basename(pose))[0]

        # parse options:
        start_opts = self.parse_rfdiffusion_opts(options, pose_opts)

        if "inference.input_pdb" not in start_opts and pose is not None: # if no pose present, ignore input_pdb
            start_opts["inference.input_pdb"] = pose
        if "inference.num_designs" not in start_opts:
            start_opts["inference.num_designs"] = num_diffusions
        if "inference.output_prefix" not in start_opts:
            start_opts["inference.output_prefix"] = os.path.join(output_dir, desc)

        opts_str = " ".join([f"{k}={v}" for k, v in start_opts.items()])

        timer = round(random.uniform(0, 5), 4) # delay between 0 and 10 s to prevent jobs crashing because of simultaneious directory creation during parallel computing. TODO: check if this helps!

        # return cmd
        return f"sleep {timer}; {self.python_path} {self.script_path} {opts_str}"

    def parse_rfdiffusion_opts(self, options: str, pose_options: str) -> dict:
        """
        Parse and combine general and pose-specific RFdiffusion options into a dictionary.

        This method splits and processes both general options and pose-specific options, combining them into a single dictionary. Pose-specific options will overwrite general options if there are conflicts.

        Parameters:
            options (str): General options for the RFdiffusion script.
            pose_options (str): Pose-specific options for the RFdiffusion script.

        Returns:
            dict: A dictionary containing the combined options, with pose-specific options taking precedence over general options.

        Examples:
            Here is an example of how to use the `parse_rfdiffusion_opts` method:

            .. code-block:: python

                options = "inference.num_designs=10 inference.use_gpu=True"
                pose_options = "inference.input_pdb='input.pdb' inference.num_designs=5"
                parsed_opts = rfdiffusion.parse_rfdiffusion_opts(options, pose_options)
                # parsed_opts will be:
                # {'inference.num_designs': '5', 'inference.use_gpu': 'True', 'inference.input_pdb': "'input.pdb'"}

        Further Details:
            - **Option Splitting:** The method uses regular expressions to split the options string into individual option entries, ensuring that options within quotes are not split incorrectly.
            - **Option Overwriting:** By adding pose-specific options after general options, the method ensures that pose-specific options can overwrite general options if necessary.

        This method is designed to create a consolidated dictionary of options for the RFdiffusion script, facilitating the construction of command strings with the appropriate parameters.
        """
        def re_split_rfdiffusion_opts(command) -> list:
            if command is None:
                return []
            return re.split(r"\s+(?=(?:[^']*'[^']*')*[^']*$)", command)

        splitstr = [x for x in re_split_rfdiffusion_opts(options) + re_split_rfdiffusion_opts(pose_options) if x] # adding pose_opts after options makes sure that pose_opts overwrites options!
        return {x.split("=")[0]: "=".join(x.split("=")[1:]) for x in splitstr}

def collect_scores(work_dir: str, rename_pdbs: bool = True) -> pd.DataFrame:
    """
    Collect scores from RFdiffusion output files.

    This method collects scores from .trb files generated by RFdiffusion into a single pandas DataFrame. It also optionally renames the output .pdb files based on the diffusion process.

    Parameters:
        work_dir (str): The working directory where RFdiffusion output files are stored.
        rename_pdbs (bool, optional): If True, rename the .pdb files based on the new descriptions. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores from the RFdiffusion output.

    Raises:
        FileNotFoundError: If no .pdb files are found in the specified directory.

    Examples:
        Here is an example of how to use the `collect_scores` method:

        .. code-block:: python

            work_dir = "/path/to/output"
            scores_df = rfdiffusion.collect_scores(work_dir, rename_pdbs=True)
            # scores_df will contain the combined scores from the RFdiffusion output files

    Further Details:
        - **Score Collection:** The method iterates over .pdb files in the specified directory, collecting corresponding .trb files and concatenating their scores into a DataFrame.
        - **File Renaming:** If `rename_pdbs` is set to True, the method renames the .pdb files based on new descriptions to ensure unique identification.
        - **DataFrame Structure:** The resulting DataFrame includes relevant score information, and columns are renamed appropriately if files are renamed.

    This method is designed to streamline the collection and organization of RFdiffusion output scores, facilitating further analysis and processing.
    """
    # collect scores from .trb-files into one pandas DataFrame:
    pdb_dir = os.path.join(work_dir, "output_pdbs")
    pl = glob(f"{pdb_dir}/*.pdb")
    if not pl: raise FileNotFoundError(f"No .pdb files were found in the diffusion output direcotry {pdb_dir}. RFDiffusion might have crashed (check inpainting error-log), or the path might be wrong!")

    # collect rfdiffusion scores into a DataFrame:
    scores = []
    for pdb in pl:
        if os.path.isfile(trb := pdb.replace(".pdb", ".trb")):
            scores.append(parse_diffusion_trbfile(trb))
    scores = pd.concat(scores)

    # rename pdbs if option is set:
    if rename_pdbs is True:
        scores.loc[:, "new_description"] = ["_".join(desc.split("_")[:-1]) + "_" + str(int(desc.split("_")[-1]) + 1).zfill(4) for desc in scores["description"]]
        scores.loc[:, "new_loc"] = [loc.replace(old_desc, new_desc) for loc, old_desc, new_desc in zip(list(scores["location"]), list(scores["description"]), list(scores["new_description"]))]

        # rename all diffusion outputfiles according to new indeces:
        _ = [[os.rename(f, f.replace(old_desc, new_desc)) for f in glob(f"{pdb_dir}/{old_desc}.*")] for old_desc, new_desc in zip(list(scores["description"]), list(scores["new_description"]))]

        # Collect information of path to .pdb files into DataFrame under 'location' column
        scores = scores.drop(columns=["location"]).rename(columns={"new_loc": "location"})
        scores = scores.drop(columns=["description"]).rename(columns={"new_description": "description"})

    scores.reset_index(drop=True, inplace=True)

    return scores

def parse_diffusion_trbfile(path: str) -> pd.DataFrame:
    """
    Parse a .trb file from RFdiffusion and extract relevant scores into a pandas DataFrame.

    This method reads a .trb file generated by RFdiffusion, extracts relevant scoring information, and organizes it into a DataFrame. The extracted information includes pLDDT scores, residue indices, and metadata.

    Parameters:
        path (str): The path to the .trb file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted scores and metadata from the .trb file.

    Raises:
        ValueError: If the provided file path does not end with .trb.

    Examples:
        Here is an example of how to use the `parse_diffusion_trbfile` method:

        .. code-block:: python

            path = "/path/to/output.trb"
            scores_df = rfdiffusion.parse_diffusion_trbfile(path)
            # scores_df will contain the extracted scores and metadata

    Further Details:
        - **File Reading:** The method uses numpy to load the .trb file and allows for pickled objects.
        - **Score Extraction:** Extracted scores include mean pLDDT, per-residue pLDDT, and other relevant metrics.
        - **Metadata Collection:** Metadata such as file location, description, and input PDB are included in the DataFrame.

    This method is designed to parse and organize the data from RFdiffusion .trb files, making it easier to analyze the results.
    """
    # read trbfile:
    if path.endswith(".trb"): data_dict = np.load(path, allow_pickle=True)
    else: raise ValueError(f"only .trb-files can be passed into parse_inpainting_trbfile. <trbfile>: {path}")

    # calc mean_plddt:
    sd = {}
    last_plddts = data_dict["plddt"][-1]
    sd["plddt"] = [sum(last_plddts) / len(last_plddts)]
    sd["perres_plddt"] = [last_plddts]

    # instantiate scoresdict and start collecting:
    scoreterms = ["con_hal_pdb_idx", "con_ref_pdb_idx", "complex_con_hal_pdb_idx", "complex_con_ref_pdb_idx", "sampled_mask"]
    for st in scoreterms:
        sd[st] = [data_dict[st]]

    # collect metadata
    sd["location"] = path.replace(".trb", ".pdb")
    sd["description"] = path.split("/")[-1].replace(".trb", "")
    sd["input_pdb"] = data_dict["config"]["inference"]["input_pdb"]

    return pd.DataFrame(sd)

def prep_motif_input(motif: Any, df: pd.DataFrame) -> list[str]:
    """
    Ensure motif input is a list and validate that motifs are present in the DataFrame.

    This method checks if the given motif is a string or a list, and ensures it is returned as a list. It also validates that the specified motifs are present as columns in the provided DataFrame.

    Parameters:
        motif (Any): The motif or list of motifs to validate and process.
        df (pd.DataFrame): The DataFrame in which to check for the presence of the motifs.

    Returns:
        list[str]: A list of motif column names.

    Raises:
        ValueError: If any of the specified motifs are not present in the DataFrame.

    Examples:
        Here is an example of how to use the `prep_motif_input` function:

        .. code-block:: python

            df = pd.DataFrame({"motif1": [1, 2, 3], "motif2": [4, 5, 6]})
            motif = "motif1"
            motifs_list = rfdiffusion.prep_motif_input(motif, df)
            # motifs_list will be: ["motif1"]

    Further Details:
        - **Motif Handling:** The method ensures that even a single motif string is converted into a list to standardize processing.
        - **Validation:** It checks that each motif in the list is a column in the provided DataFrame, raising an error if any are missing.

    This function is designed to prepare and validate motif inputs for further processing in the RFdiffusion workflow.
    """
    # ambivalence to singular or multiple motif cols
    motifs = [motif] if isinstance(motif, str) else motif

    # clear
    for m in motifs:
        col_in_df(df, m)

    return motifs

def update_motif_res_mapping(motif_l: list[ResidueSelection], con_ref_idx: list, con_hal_idx: list) -> list:
    """
    Update motifs in motif_l based on con_ref_idx and con_hal_idx.

    This method updates the residue mappings of motifs in the provided list based on the reference and halogenated residue indices from RFdiffusion outputs.

    Parameters:
        motif_l (list[ResidueSelection]): A list of ResidueSelection objects representing the motifs to be updated.
        con_ref_idx (list): A list of reference residue indices from the RFdiffusion outputs.
        con_hal_idx (list): A list of halogenated residue indices from the RFdiffusion outputs.

    Returns:
        list: A list of updated ResidueSelection objects with new residue mappings.

    Raises:
        TypeError: If any element in motif_l is not of type ResidueSelection.

    Examples:
        Here is an example of how to use the `update_motif_res_mapping` method:

        .. code-block:: python

            motif_l = [ResidueSelection(["A:10", "A:20"]), ResidueSelection(["B:30", "B:40"])]
            con_ref_idx = [("A", 10), ("A", 20)]
            con_hal_idx = [("A", 11), ("A", 21)]
            updated_motifs = rfdiffusion.update_motif_res_mapping(motif_l, con_ref_idx, con_hal_idx)
            # updated_motifs will contain the updated ResidueSelection objects

    Further Details:
        - **Residue Mapping:** The method sets up a mapping dictionary from reference to halogenated residue indices.
        - **Motif Update:** Each motif in the input list is updated according to the new residue mappings and returned as a new ResidueSelection object.

    This method is designed to update residue selections in motifs based on the outputs from RFdiffusion, facilitating accurate downstream analysis and interpretation.
    """
    output_motif_l = []
    for motif, ref_idx, hal_idx in zip(motif_l, con_ref_idx, con_hal_idx):
        # error handling
        if not isinstance(motif, ResidueSelection):
            raise TypeError(f"Individual motifs must be of type ResidueSelection. Create ResidueSelection objects out of your motifs.")

        # setup mapping from rfdiffusion outputs:
        exchange_dict = get_residue_mapping(ref_idx, hal_idx)

        # exchange and return
        exchanged_motif = [exchange_dict[residue] for residue in motif.residues]
        output_motif_l.append(ResidueSelection(exchanged_motif))
    return output_motif_l

def get_residue_mapping(con_ref_idx: list, con_hal_idx: list) -> dict:
    """
    Create a residue mapping dictionary from RFdiffusion outputs.

    This method creates a mapping dictionary that maps old residue indices (from con_ref_idx) to new residue indices (from con_hal_idx).

    Parameters:
        con_ref_idx (list): A list of reference residue indices from the RFdiffusion outputs, where each element is a tuple of (chain, residue_id).
        con_hal_idx (list): A list of halogenated residue indices from the RFdiffusion outputs, where each element is a tuple of (chain, residue_id).

    Returns:
        dict: A dictionary where keys are tuples of (chain, residue_id) from con_ref_idx and values are tuples of (chain, residue_id) from con_hal_idx.

    Examples:
        Here is an example of how to use the `get_residue_mapping` method:

        .. code-block:: python

            con_ref_idx = [("A", 10), ("A", 20)]
            con_hal_idx = [("A", 11), ("A", 21)]
            residue_mapping = rfdiffusion.get_residue_mapping(con_ref_idx, con_hal_idx)
            # residue_mapping will be: {("A", 10): ("A", 11), ("A", 20): ("A", 21)}

    Further Details:
        - **Mapping Creation:** The method pairs each element in con_ref_idx with the corresponding element in con_hal_idx to create the mapping.
        - **Usage:** This mapping is useful for updating residue selections based on RFdiffusion outputs.

    This method is designed to facilitate the creation of residue mappings for updating motifs or other residue-based selections.
    """
    return {(chain, int(res_id)): hal for (chain, res_id), hal in zip(con_ref_idx, con_hal_idx)}
