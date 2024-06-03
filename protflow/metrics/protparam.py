"""
ProtParam Module
================

This module provides the functionality to integrate ProtParam calculations within the ProtFlow framework. It offers tools to compute various protein sequence features using the BioPython `Bio.SeqUtils.ProtParam` module, handling inputs and outputs efficiently, and processing the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `ProtParam` class encapsulates the functionality necessary to execute ProtParam calculations. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of parameter calculations. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.
The module is designed to streamline the integration of ProtParam into larger computational workflows. It supports the automatic setup of job parameters, execution of ProtParam commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ProtParam` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the ProtParam process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ProtParam` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from protparam import ProtParam

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the ProtParam class
    protparam = ProtParam()

    # Run the ProtParam calculation process
    results = protparam.run(
        poses=poses,
        prefix="experiment_1",
        seq_col=None,
        pH=7,
        overwrite=True,
        jobstarter=jobstarter
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the ProtParam process.
    - Customizability: Users can customize the ProtParam process through multiple parameters, including the pH for determining protein total charge, specific options for the ProtParam script, and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate ProtParam calculations into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
# import general
import os

# import dependencies
import pandas as pd
import numpy as np

# import customs
from protflow.config import PROTFLOW_ENV
from protflow.config import AUXILIARY_RUNNER_SCRIPTS_DIR as script_dir
from protflow.runners import Runner, RunnerOutput
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.utils.biopython_tools import get_sequence_from_pose, load_sequence_from_fasta, load_structure_from_pdbfile

class ProtParam(Runner):
    '''
    Class handling the calculation of protparams from sequence using the BioPython Bio.SeqUtils.ProtParam module
    '''
    def __init__(self, jobstarter: str = None, default_python = os.path.join(PROTFLOW_ENV, "python3")): # pylint: disable=W0102
        """
        Initialize the ProtParam class.

        This constructor sets up the necessary environment for running ProtParam calculations. It initializes the job starter and sets the path to the Python executable within the ProtFlow environment.

        Parameters
        ----------
        jobstarter : str, optional
            The job starter to be used for executing ProtParam commands. If not provided, it defaults to None.
        default_python : str, optional
            The path to the Python executable within the ProtFlow environment. The default value is constructed using the PROTFLOW_ENV environment variable.

        Attributes
        ----------
        jobstarter : str
            Stores the job starter to be used for executing ProtParam commands.
        python : str
            The path to the Python executable within the ProtFlow environment.

        Raises
        ------
        FileNotFoundError
            If the default Python executable is not found in the specified path.

        Examples
        --------
        Here is an example of how to initialize the `ProtParam` class:

        .. code-block:: python

            from protparam import ProtParam

            # Initialize the ProtParam class with default settings
            protparam = ProtParam()

            # Initialize the ProtParam class with a specific job starter
            custom_jobstarter = "my_custom_jobstarter"
            protparam = ProtParam(jobstarter=custom_jobstarter)

        The `__init__` method ensures that the ProtParam class is ready to perform protein sequence parameter calculations within the ProtFlow framework, setting up the environment and configurations necessary for successful execution.
        """
        self.jobstarter = jobstarter
        self.python = self.search_path(default_python, "PROTFLOW_ENV")

    def __str__(self):
        return "protparam.py"

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, seq_col: str = None, pH: float = 7, overwrite=False, jobstarter: JobStarter = None) -> None:
        """
        ProtParam Class
        ===============

        The `ProtParam` class is a specialized class designed to facilitate the calculation of protein sequence parameters within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with ProtParam calculations.

        Detailed Description
        --------------------
        The `ProtParam` class manages all aspects of running ProtParam calculations. It handles the configuration of necessary scripts and executables, prepares the environment for sequence feature calculations, and executes the ProtParam commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

        Key functionalities include:
            - Setting up paths to ProtParam scripts and Python executables.
            - Configuring job starter options, either automatically or manually.
            - Handling the execution of ProtParam commands with support for various input types.
            - Collecting and processing output data into a pandas DataFrame.
            - Customizing the sequence feature calculations based on user-defined parameters such as pH.

        Returns
        -------
        An instance of the `ProtParam` class, configured to run ProtParam calculations and handle outputs efficiently.

        Raises
        ------
            - FileNotFoundError: If required files or directories are not found during the execution process.
            - ValueError: If invalid arguments are provided to the methods.
            - TypeError: If the input poses are not of the expected type.

        Examples
        --------
        Here is an example of how to initialize and use the `ProtParam` class:

        .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import JobStarter
            from protparam import ProtParam

            # Create instances of necessary classes
            poses = Poses()
            jobstarter = JobStarter()

            # Initialize the ProtParam class
            protparam = ProtParam()

            # Run the ProtParam calculation process
            results = protparam.run(
                poses=poses,
                prefix="experiment_1",
                seq_col=None,
                pH=7,
                overwrite=True,
                jobstarter=jobstarter
            )

            # Access and process the results
            print(results)

        Further Details
        ---------------
            - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
            - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the ProtParam calculations to their specific needs.
            - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

        The ProtParam class is intended for researchers and developers who need to perform ProtParam calculations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
        """
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        scorefile = os.path.join(work_dir, f"{prefix}_protparam.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        if not seq_col:
            # check poses file extension
            pose_type = poses.determine_pose_type()
            if len(pose_type) > 1:
                raise TypeError(f"Poses must be of a single type, not {pose_type}!")
            if not pose_type[0] in [".fa", ".fasta", ".pdb"]:
                raise TypeError(f"Poses must be of type '.fa', '.fasta' or '.pdb', not {pose_type}!")
            elif pose_type[0] in [".fa", ".fasta"]:
                # directly use fasta files as input
                # TODO: this assumes that it is a single entry fasta file (as it should be!)
                seqs = [load_sequence_from_fasta(fasta=pose, return_multiple_entries=False).seq for pose in poses.df['poses'].to_list()]     
            elif pose_type[0] == ".pdb":
                # extract sequences from pdbs
                seqs = [get_sequence_from_pose(load_structure_from_pdbfile(path_to_pdb=pose)) for pose in poses.df['poses'].to_list()]
        else:
            # if not running on poses but on arbitrary sequences, get the sequences from the dataframe
            seqs = poses.df[seq_col].to_list()

        names = poses.df['poses_description'].to_list()

        input_df = pd.DataFrame({"name": names, "sequence": seqs})

        num_json_files = jobstarter.max_cores
        if num_json_files > len(input_df.index):
            num_json_files = len(input_df.index)

        json_files = []
        # create multiple input dataframes to run in parallel
        if num_json_files > 1:
            for i, df in enumerate(np.array_split(input_df, num_json_files)):
                name = os.path.join(work_dir, f"input_{i}.json")
                df.to_json(name)
                json_files.append(name)
        else:
            name = os.path.join(work_dir, f"input_1.json")
            input_df.to_json(name)
            json_files.append(name)

        # write commands
        cmds = []
        for json in json_files:
            cmds.append(f"{self.python} {script_dir}/run_protparam.py --input_json {json} --output_path {os.path.splitext(json)[0]}_out.json --pH {pH}")

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "protparam",
            output_path = work_dir
        )

        # collect scores
        scores = []
        for json in json_files:
            scores.append(pd.read_json(f"{os.path.splitext(json)[0]}_out.json"))

        scores = pd.concat(scores)
        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
        return output.return_poses()
