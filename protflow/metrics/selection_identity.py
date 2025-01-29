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
from protflow.utils.biopython_tools import load_structure_from_pdbfile, three_to_one_AA_code

class SelectionIdentity(Runner):
    """
    A class to calculate selection identity metrics for a set of protein poses.

    This class facilitates the computation of residue identities for a selection of residues 
    in protein structures and can store these identities in a standardized format. 

    Parameters
    ----------
    residue_selection : Union[str, ResidueSelection], optional
        A residue selection or the name of a DataFrame column containing `ResidueSelection` objects. 
        Default is `None`.
    onelettercode : bool, optional
        Whether to use one-letter amino acid codes instead of three-letter codes. Default is `False`.
    python_path : str, optional
        Path to the Python executable to be used for running subprocesses. Default is the Python 
        executable in the `PROTFLOW_ENV`.
    jobstarter : JobStarter, optional
        A `JobStarter` object for managing parallel execution. Default is `None`.
    overwrite : bool, optional
        Whether to overwrite existing results if score files are detected. Default is `False`.

    Methods
    -------
    set_python_path(python_path: str)
        Set the Python executable path for subprocesses.
    set_onelettercode(onelettercode: bool)
        Configure the output format to use one-letter amino acid codes.
    set_residue_selection(residue_selection: Union[str, ResidueSelection])
        Define the residue selection or DataFrame column containing selections.
    set_jobstarter(jobstarter: JobStarter)
        Configure the job starter for parallel execution.
    run(poses: Poses, prefix: str, residue_selection: Union[str, ResidueSelection] = None, 
        onelettercode: bool = False, jobstarter: JobStarter = None, python_path: str = None, 
        overwrite: bool = False) -> Poses
        Run the selection identity calculations on the given poses.
    """
    def __init__(self, residue_selection: Union[str, ResidueSelection] = None, onelettercode: bool = False, python_path: str = os.path.join(PROTFLOW_ENV, "python"), jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
        self.set_python_path(python_path)
        self.set_residue_selection(residue_selection)
        self.set_onelettercode(onelettercode)

        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################

    def set_python_path(self, python_path: str) -> None:
        """
        Set the Python executable path to be used for subprocess execution.

        Parameters
        ----------
        python_path : str
            The path to the Python executable.
        """
        self.python_path = python_path

    def set_onelettercode(self, onelettercode: bool) -> None:
        """
        Configure whether to use one-letter amino acid codes in the output.

        Parameters
        ----------
        onelettercode : bool
            Set to `True` to use one-letter codes; `False` to use three-letter codes.
        """
        self.onelettercode = onelettercode

    def set_residue_selection(self, residue_selection: Union[str, ResidueSelection] = None):
        """
        Set the residue selection criteria or reference column.

        Parameters
        ----------
        residue_selection : Union[str, ResidueSelection], optional
            A `ResidueSelection` object or the name of a DataFrame column containing 
            residue selections. Default is `None`.

        Raises
        ------
        ValueError
            If the provided `residue_selection` is neither a `ResidueSelection` nor a valid column name.
        """
        if not residue_selection:
            self.residue_selection = None
        elif isinstance(residue_selection, str) or isinstance(residue_selection, ResidueSelection):
            self.residue_selection = residue_selection
        else:
            raise ValueError("Parameter :residue_selection: must either be a ResidueSelection or a poses dataframe column name containing ResidueSelections!")

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """
        Set the job starter for managing parallel execution.

        Parameters
        ----------
        jobstarter : JobStarter
            A `JobStarter` object for handling parallel job execution.

        Raises
        ------
        ValueError
            If the provided `jobstarter` is not a `JobStarter` instance or `None`.
        """
        if isinstance(jobstarter, JobStarter) or jobstarter == None:
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")
        

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, residue_selection: Union[str, ResidueSelection] = None, onelettercode: bool = False, jobstarter: JobStarter = None, python_path: str = None, overwrite: bool = False) -> Poses:
        """
        Run the selection identity calculations for the given poses.

        Parameters
        ----------
        poses : Poses
            A `Poses` object containing protein structures to analyze.
        prefix : str
            Prefix for output file names.
        residue_selection : Union[str, ResidueSelection], optional
            Residue selection criteria. Can be a `ResidueSelection` object or a column name in 
            the poses DataFrame. Default is `None`.
        onelettercode : bool, optional
            Use one-letter amino acid codes in the output. Default is `False`.
        jobstarter : JobStarter, optional
            A `JobStarter` object for managing parallel execution. Default is `None`.
        python_path : str, optional
            Path to the Python executable for subprocess execution. Defaults to the class-level 
            `python_path` attribute.
        overwrite : bool, optional
            Whether to overwrite existing score files. Default is `False`.

        Returns
        -------
        Poses
            Updated `Poses` object with calculated residue identities.

        Raises
        ------
        ValueError
            If the input poses are not in `.pdb` format or if invalid residue selections are provided.
        RuntimeError
            If the number of output poses is less than the input poses, indicating possible job failure.
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
        onelettercode = onelettercode or self.onelettercode

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
        cmds = [f"{python_path} {__file__} --input_json {in_json} --output_json {out_json} {'--onelettercode' if onelettercode else ''}" for in_json, out_json in zip(in_jsons, out_jsons)]

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
            if args.onelettercode:
                resname = three_to_one_AA_code(residue.get_resname())
            else:
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
    argparser.add_argument("--onelettercode", action="store_true", help="Return one letter code instead of three-letter code. Fails on noncanonical.")

    arguments = argparser.parse_args()
    main(arguments)