# TODO: Generate proper doc strings!!!
"""
Generic Metric module
===========

This module provides the functionality to calculate generic metrics within the ProtFlow framework. It offers tools to run calculations, handle inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `GenericMetric`  class encapsulate the functionality necessary to execute generic metrics. Generic metrics can be any function that accepts the path to a single pose as its input and returns e.g. a score. This class manages the configuration of the module and function, sets up the environment, and handles the execution of calculations. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of calculations into larger computational workflows. It supports the automatic setup of job parameters and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `GenericMetric` class and invoke its `run` methods with appropriate parameters. The module will handle the configuration, execution, and result collection processes.

Examples
--------
Here is an example of how to initialize and use the `BackboneRMSD` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from rmsd import BackboneRMSD

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the BackboneRMSD class
    backbone_rmsd = BackboneRMSD()

    # Run the RMSD calculation
    results = backbone_rmsd.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        ref_col="reference",
        chains=["A", "B"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the RMSD calculation process.
    - Customizability: Users can customize the RMSD calculation process through multiple parameters, including the specific atoms and chains to be used in the calculation, as well as jobstarter configurations.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate RMSD calculations into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import logging
import os
from typing import Union

# import dependencies
import pandas as pd
import numpy as np

# import customs
from protflow.config import PROTFLOW_ENV
from protflow.runners import Runner, RunnerOutput
from protflow.residues import ResidueSelection
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.jobstarters import JobStarter
from protflow.utils.biopython_tools import load_structure_from_pdbfile

class SelectionIdentity(Runner):
    """
    BackboneRMSD Class
    ==================

    The `BackboneRMSD` class is a specialized class designed to facilitate the calculation of backbone RMSD values within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with RMSD calculations.

    Detailed Description
    --------------------
    The `BackboneRMSD` class manages all aspects of calculating RMSD for protein backbones. It handles the configuration of necessary scripts and executables, prepares the environment for RMSD calculations, and executes the commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to RMSD calculation scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of RMSD commands with support for different atoms and chains.
        - Collecting and processing output data into a pandas DataFrame.
        - Managing overwrite options and handling existing score files.

    Returns
    -------
    An instance of the `BackboneRMSD` class, configured to run RMSD calculations and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If atoms or chains are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `BackboneRMSD` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from rmsd import BackboneRMSD

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = LocalJobStarter(max_cores=4)

        # Initialize the BackboneRMSD class
        backbone_rmsd = BackboneRMSD()

        # Run the RMSD calculation
        results = backbone_rmsd.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            ref_col="reference_location",
            chains=["A", "B"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the RMSD calculation process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The BackboneRMSD class is intended for researchers and developers who need to perform backbone RMSD calculations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, residue_selection: Union[str, ResidueSelection] = None, python_path: str = os.path.join(PROTFLOW_ENV, "python3"), jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
        """
        Initialize the BackboneRMSD class.

        This constructor sets up the BackboneRMSD instance with default or provided parameters. It configures the reference column, atoms, chains, jobstarter, and overwrite options for RMSD calculations.

        Parameters:
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            atoms (list[str], optional): The list of atom names to calculate RMSD over. Defaults to ["CA"].
            chains (list[str], optional): The list of chain names to calculate RMSD over. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.
            jobstarter (str, optional): The jobstarter configuration for running the RMSD calculations. Defaults to None.

        Returns:
            None

        Examples:
            Here is an example of how to initialize the BackboneRMSD class:

            .. code-block:: python

                from rmsd import BackboneRMSD

                # Initialize the BackboneRMSD class with default parameters
                backbone_rmsd = BackboneRMSD()

                # Initialize the BackboneRMSD class with custom parameters
                backbone_rmsd = BackboneRMSD(ref_col="reference", atoms=["CA", "CB"], chains=["A", "B"], overwrite=True, jobstarter="custom_starter")

        Further Details:
            - **Default Values:** If no parameters are provided, the class initializes with default values suitable for basic RMSD calculations.
            - **Parameter Storage:** The parameters provided during initialization are stored as instance variables, which are used in subsequent method calls.
            - **Custom Configuration:** Users can customize the RMSD calculation process by providing specific values for the reference column, atoms, chains, and jobstarter.
        """
        self.set_python_path(python_path)
        self.set_residue_selection(residue_selection)

        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################

    def set_python_path(self, python_path: str) -> None:
        self.python_path = python_path

    def set_residue_selection(self, residue_selection: Union[str, ResidueSelection] = None):
        if not residue_selection:
            self.residue_selection = None
        elif isinstance(residue_selection, str) or isinstance(residue_selection, ResidueSelection):
            self.residue_selection = residue_selection
        else:
            raise ValueError("Parameter :residue_selection: must either be a ResidueSelection or a poses dataframe column name containing ResidueSelections!")

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """
        Set the jobstarter configuration.

        This method sets the jobstarter configuration to be used.

        Parameters:
            jobstarter (JobStarter): The jobstarter configuration.

        Returns:
            None

        Raises:
            TypeError: If jobstarter is not of type JobStarter.

        Examples:
            Here is an example of how to use the `set_jobstarter` method:

            .. code-block:: python

                from rmsd import BackboneRMSD

                # Initialize the BackboneRMSD class
                backbone_rmsd = BackboneRMSD()

                # Set the jobstarter configuration
                backbone_rmsd.set_jobstarter("custom_starter")

        Further Details:
            - **Usage:** The jobstarter configuration specifies how the RMSD calculations will be managed and executed, particularly in HPC environments.
            - **Validation:** The method includes validation to ensure that the jobstarter parameter is of the correct type.
            - **Integration:** The jobstarter configuration set by this method is used by other methods in the class to manage the execution of RMSD calculations.
        """
        if isinstance(jobstarter, JobStarter) or jobstarter == None:
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")
        

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, residue_selection: Union[str, ResidueSelection] = None, jobstarter: JobStarter = None, python_path: str = None, overwrite: bool = False) -> Poses:
        """
        Calculate the backbone RMSD for given poses and jobstarter configuration.

        This method sets up and runs the RMSD calculation process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            chains (list[str], optional): A list of chain names to calculate RMSD over. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the RMSD calculation.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If chains are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from rmsd import BackboneRMSD

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = LocalJobStarter(max_cores=4)

                # Initialize the BackboneRMSD class
                backbone_rmsd = BackboneRMSD()

                # Run the RMSD calculation
                results = backbone_rmsd.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    ref_col="reference",
                    chains=["A", "B"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It supports splitting poses into sublists for parallel processing.
            - **Input Handling:** The method prepares input JSON files for each sublist of poses and constructs commands for running RMSD calculations using BioPython.
            - **Output Management:** The method handles the collection and processing of output data from multiple score files, concatenating them into a single DataFrame and saving the results.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the RMSD calculation process to their specific needs, including specifying atoms and chains for RMSD calculations.

        This method is designed to streamline the execution of backbone RMSD calculations within the ProtFlow framework, making it easier for researchers and developers to perform and analyze RMSD calculations.
        """

        # prep variables
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        if not poses.determine_pose_type() == [".pdb"]:
            raise ValueError(f"Poses must be of type .pdb, not {poses.determine_pose_type()}!")

        # use parameters of run function (if available), otherwise fall back to class
        python_path = python_path or self.python_path
        residue_selection = residue_selection or self.residue_selection

        # load residue selections
        if isinstance(residue_selection, str):
            col_in_df(poses.df, residue_selection)
            residue_selections = poses.df[residue_selection].to_list()
            if any(not isinstance(sele, ResidueSelection) for sele in residue_selections):
                raise ValueError(f"Column {residue_selection} in poses DataFrame must only contain ResidueSelections!")
        elif isinstance(residue_selection, ResidueSelection):
            residue_selections = [residue_selection for _ in poses.poses_list()]
        else:
            raise ValueError(f"Parameter :residue_selection: must either be a ResidueSelection or a poses dataframe column containing ResidueSelections!")
        
        logging.info(f"Running metric selection_identity in {work_dir} on {len(poses.df.index)} poses.")

        # define scorefile
        scorefile = os.path.join(work_dir, f"{prefix}_selection_identity.{poses.storage_format}")

        # check if RMSD was calculated if overwrite was not set.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # set number of jobs
        num_json_files = jobstarter.max_cores if len(poses.df.index) >= jobstarter.max_cores else len(poses.df.index)

        # write json files used as input
        in_jsons = []
        out_jsons = []
        input_df = pd.DataFrame({"location": poses.poses_list(), "selections": residue_selections})
        for i, df in enumerate(np.array_split(input_df, num_json_files)):
            name = os.path.join(work_dir, f"input_{i}.json")
            df.to_json(name)
            in_jsons.append(name)
            out_jsons.append(os.path.join(work_dir, f"output_{i}.json"))

        # write cmds
        cmds = [f"{python_path} {__file__} --input_json {in_json} --output_json {out_json}" for in_json, out_json in zip(in_jsons, out_jsons)]

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "selection_identity",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        scores = pd.concat([pd.read_json(output) for output in out_jsons]).reset_index(drop=True)
        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")
        
        logging.info(f"Saving scores of selection identity metric at {scorefile}.")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info(f"selection_identity completed. Returning scores.")
        return output.return_poses()
    

def main(args):

    in_df = pd.read_json(args.input_json)
    selection_resnames = []
    for pose, selection in zip(in_df["location"].to_list(), in_df["selections"].to_list()):
        pose = load_structure_from_pdbfile(pose)
        residues_dict = {}
        for residue in pose.get_residues():
            chain_id = residue.parent.id
            resnum = residue.id[1]
            resname = residue.get_resname()
            if chain_id not in residues_dict:
                residues_dict[chain_id] = {}
            residues_dict[chain_id][resnum] = resname
        selection = ResidueSelection(selection, from_scorefile=True)
        selection_resnames.append({f"{chain_resnum[0]}{chain_resnum[1]}": residues_dict[chain_resnum[0]][chain_resnum[1]] for chain_resnum in selection})

    # create results dataframe
    in_df["selection_identities"] = selection_resnames
    in_df["description"] = [description_from_path(pose) for pose in in_df["location"].to_list()]

    # save output
    in_df[["description", "selection_identities", "location"]].to_json(args.output_json)



if __name__ == "__main__":
    import argparse
    import pandas as pd
    from protflow.residues import ResidueSelection

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_json", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")

    arguments = argparser.parse_args()
    main(arguments)