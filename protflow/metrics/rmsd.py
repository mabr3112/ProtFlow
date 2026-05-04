"""
RMSD Module
===========

This module provides the functionality to calculate Root Mean Square Deviation (RMSD) values for protein structures within the ProtFlow framework. It offers tools to run RMSD calculations, handle inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `BackboneRMSD` and `MotifRMSD` classes encapsulate the functionality necessary to execute RMSD calculations. These classes manage the configuration of paths to essential scripts and Python executables, set up the environment, and handle the execution of RMSD calculations. They also include methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of RMSD calculations into larger computational workflows. It supports the automatic setup of job parameters, execution of RMSD commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `BackboneRMSD` or `MotifRMSD` class and invoke their `run` methods with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the RMSD calculation process is provided through various parameters, allowing for customized runs tailored to specific research needs.

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
import os
import json
import shlex
from typing import Any

# import dependencies
import pandas as pd

# import customs
from ..poses import Poses
from ..residues import AtomSelection, AtomSelectionInput, ResidueSelection
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, col_in_df
from .. import jobstarters, load_config_path, require_config


def _setup_atom_selection_list(
    poses: Poses,
    selection: AtomSelectionInput,
    parameter_name: str,
    allow_none: bool = False,
) -> list[list[Any] | None]:
    """Expand static or dataframe-column atom selections into one JSON-ready list per pose."""
    # None is valid only for optional superimposition selections.
    if selection is None:
        if allow_none:
            return [None for _ in poses]
        raise ValueError(f"{parameter_name} must be specified.")

    # Strings point to dataframe columns with one ordered selection per pose.
    if isinstance(selection, str):
        col_in_df(poses.df, selection)
        out = []
        for idx, row_selection in enumerate(poses.df[selection].to_list()):
            if row_selection is None:
                if allow_none:
                    out.append(None)
                    continue
                raise ValueError(f"{parameter_name} column '{selection}' contains None at row {idx}.")
            out.append(AtomSelection(row_selection).to_list())
        return out

    # Non-column selections are static and reused for every pose.
    atom_selection = AtomSelection(selection).to_list()
    return [atom_selection for _ in poses]


def _setup_reference_paths(poses: Poses, ref_col: str | None, ref_path: str | None) -> list[str]:
    """Resolve mutually exclusive reference path sources to one path per pose."""
    # Users can either pass one static reference path or a column with per-pose paths.
    if ref_col is not None and ref_path is not None:
        raise ValueError("ref_col and ref_path are mutually exclusive. Specify only one reference source.")
    if ref_col is None and ref_path is None:
        raise ValueError("Either ref_col or ref_path must be specified for atom-level RMSD calculation.")

    if ref_col is not None:
        col_in_df(poses.df, ref_col)
        ref_paths = poses.df[ref_col].to_list()
    else:
        if not isinstance(ref_path, str):
            raise TypeError(f"ref_path must be a string path. Got {type(ref_path)}: {ref_path}")
        ref_paths = [ref_path for _ in poses]

    # Validate early so path issues fail before jobs are submitted.
    for path in ref_paths:
        if not isinstance(path, str):
            raise TypeError(f"Reference paths must be strings. Got {type(path)}: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Reference structure file not found: {path}")
    return [os.path.abspath(path) for path in ref_paths]


class AtomRMSD(Runner):
    """Runner for atom-level BioPython RMSD calculations."""

    def __init__(
        self,
        ref_col: str | None = None,
        ref_path: str | None = None,
        ref_atoms: AtomSelectionInput = None,
        ref_superimpose_atoms: AtomSelectionInput = None,
        target_atoms: AtomSelectionInput = None,
        target_superimpose_atoms: AtomSelectionInput = None,
        return_superimposed: bool = False,
        jobstarter: JobStarter | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize an atom-level RMSD runner."""
        # setup config
        config = require_config()
        self.python = os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")

        # runner setup
        self.ref_col = ref_col
        self.ref_path = ref_path
        self.ref_atoms = ref_atoms
        self.ref_superimpose_atoms = ref_superimpose_atoms
        self.target_atoms = target_atoms
        self.target_superimpose_atoms = target_superimpose_atoms
        self.return_superimposed = return_superimposed
        self.jobstarter = jobstarter
        self.overwrite = overwrite
        self.name = "atom_rmsd"

    def __str__(self) -> str:
        return "AtomRMSD"

    def run(
        self, poses: Poses, prefix: str, jobstarter: JobStarter | None = None, overwrite: bool = False,
        ref_col: str | None = None, ref_path: str | None = None, ref_atoms: AtomSelectionInput = None,
        ref_superimpose_atoms: AtomSelectionInput = None, target_atoms: AtomSelectionInput = None, 
        target_superimpose_atoms: AtomSelectionInput = None, return_superimposed: bool | None = None,
    ) -> Poses:
        """Run atom-level RMSD calculation and merge the resulting scores into ``poses``."""
        # prep inputs
        ref_col = ref_col or self.ref_col
        ref_path = ref_path or self.ref_path
        ref_atoms = ref_atoms or self.ref_atoms
        ref_superimpose_atoms = ref_superimpose_atoms or self.ref_superimpose_atoms
        target_atoms = target_atoms or self.target_atoms
        target_superimpose_atoms = target_superimpose_atoms or self.target_superimpose_atoms
        overwrite = overwrite or self.overwrite
        return_superimposed = return_superimposed or self.return_superimposed

        # sanity
        if not isinstance(return_superimposed, bool):
            raise TypeError(f"return_superimposed must be bool. Got {type(return_superimposed)}: {return_superimposed}")

        # setup runner
        script_path = os.path.join(self.script_dir, "calc_atom_rmsd.py")
        if not os.path.isfile(script_path):
            raise ValueError(
                f"Cannot find script 'calc_atom_rmsd.py' at specified directory: '{self.script_dir}'. "
                "Set AUXILIARY_RUNNER_SCRIPTS_DIR in config.py to the protflow/tools/runners_auxiliary_scripts directory."
            )

        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )

        # check if outputs are present
        scorefile = os.path.join(work_dir, f"{prefix}_rmsds.{poses.storage_format}")
        if (rmsd_df := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=rmsd_df, prefix=prefix).return_poses()

        # setup full input dict, batch later
        input_dict = self.setup_input_dict(
            poses=poses,
            ref_col=ref_col,
            ref_path=ref_path,
            ref_atoms=ref_atoms,
            ref_superimpose_atoms=ref_superimpose_atoms,
            target_atoms=target_atoms,
            target_superimpose_atoms=target_superimpose_atoms,
            return_superimposed=return_superimposed,
        )

        if not input_dict:
            raise ValueError("No poses were provided for atom-level RMSD calculation.")

        # setup pose superimposition output directory
        superimposed_dir = os.path.join(work_dir, "superimposed")
        if return_superimposed:
            os.makedirs(superimposed_dir, exist_ok=True)

        # split input_dict into subdicts
        split_sublists = split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores or 1)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write n=max_cores input_json files for calc_atom_rmsd.py
        json_files = []
        output_files = []
        for i, subdict in enumerate(subdicts, start=1):
            input_json = os.path.join(work_dir, f"atom_rmsd_input_{str(i).zfill(4)}.json")
            output_json = os.path.join(work_dir, f"atom_rmsd_output_{str(i).zfill(4)}.json")
            with open(input_json, "w", encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(input_json)
            output_files.append(output_json)

        # setup commands
        cmds = []
        for input_json, output_json in zip(json_files, output_files):
            cmd = (
                f"{shlex.quote(self.python)} {shlex.quote(script_path)} "
                f"--input_json {shlex.quote(input_json)} "
                f"--output_path {shlex.quote(output_json)}"
            )
            if return_superimposed:
                cmd += f" --superimposed_out_path {shlex.quote(superimposed_dir)}"
            cmds.append(cmd)

        # run commands
        jobstarter.start(
            cmds=cmds,
            jobname=prefix,
            wait=True,
            output_path=work_dir,
        )

        # collect outputs
        rmsd_df = pd.concat([pd.read_json(output_path) for output_path in output_files], ignore_index=True)
        self.save_runner_scorefile(scores=rmsd_df, scorefile=scorefile)
        return RunnerOutput(poses=poses, results=rmsd_df, prefix=prefix).return_poses()

    def setup_input_dict(
        self, poses: Poses, ref_col: str | None, ref_path: str | None, ref_atoms: AtomSelectionInput,
        ref_superimpose_atoms: AtomSelectionInput, target_atoms: AtomSelectionInput,
        target_superimpose_atoms: AtomSelectionInput, return_superimposed: bool,
    ) -> dict[str, dict[str, Any]]:
        """Set up the JSON input dictionary for ``calc_atom_rmsd.py``."""

        # setup parameter lists
        ref_paths = _setup_reference_paths(poses=poses, ref_col=ref_col, ref_path=ref_path)
        ref_atoms_list = _setup_atom_selection_list(poses=poses, selection=ref_atoms, parameter_name="ref_atoms")
        target_atoms_list = _setup_atom_selection_list(poses=poses, selection=target_atoms, parameter_name="target_atoms")
        ref_superimpose_atoms_list = _setup_atom_selection_list(
            poses=poses,
            selection=ref_superimpose_atoms,
            parameter_name="ref_superimpose_atoms",
            allow_none=True,
        )
        target_superimpose_atoms_list = _setup_atom_selection_list(
            poses=poses,
            selection=target_superimpose_atoms,
            parameter_name="target_superimpose_atoms",
            allow_none=True,
        )

        # construct rmsd_input_dict
        input_dict = {}
        for i, pose_path in enumerate(poses.poses_list()):
            ref_super = ref_superimpose_atoms_list[i]
            target_super = target_superimpose_atoms_list[i]
            if (ref_super is None) != (target_super is None):
                raise ValueError(
                    "ref_superimpose_atoms and target_superimpose_atoms must either both be specified or both be None "
                    f"for pose {pose_path}."
                )

            input_dict[os.path.abspath(pose_path)] = {
                "ref_path": ref_paths[i],
                "ref_atoms": ref_atoms_list[i],
                "ref_superimpose_atoms": ref_super,
                "target_atoms": target_atoms_list[i],
                "target_superimpose_atoms": target_super,
                "return_superimposed": return_superimposed,
            }
        return input_dict

class BackboneRMSD(Runner):
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
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        TypeError: If atoms or chains are not of the expected type.

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
    def __init__(self, ref_col: str = None, atoms: list = ["CA"], chains: list[str] = None, overwrite: bool = False, jobstarter: str = None): # pylint: disable=W0102
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
        # setup config
        config = require_config()
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        self.python = os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")

        # runner setup
        self.set_ref_col(ref_col)
        self.set_atoms(atoms)
        self.set_chains(chains)
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################
    def set_ref_col(self, ref_col: str) -> None:
        """
        Set the reference column for RMSD calculations.

        This method sets the default reference column to be used in the RMSD calculation process.

        Parameters:
            ref_col (str): The reference column for RMSD calculations.

        Returns:
            None

        Raises:
            TypeError: If ref_col is not of type string.

        Examples:
            Here is an example of how to use the `set_ref_col` method:

            .. code-block:: python

                from rmsd import BackboneRMSD

                # Initialize the BackboneRMSD class
                backbone_rmsd = BackboneRMSD()

                # Set the reference column
                backbone_rmsd.set_ref_col("reference")

        Further Details:
            - **Usage:** The reference column is used to identify which column in the input data contains the reference structures for RMSD calculation.
            - **Validation:** The method includes validation to ensure that the reference column is of the correct type.
            - **Integration:** The reference column set by this method is used by other methods in the class to perform RMSD calculations.
        """
        self.ref_col = ref_col

    def set_atoms(self, atoms:list[str]) -> None:
        """
        Set the atoms for RMSD calculations.

        This method sets the list of atom names to calculate RMSD over. If "all" is provided, all atoms will be considered.

        Parameters:
            atoms (list[str]): The list of atom names to calculate RMSD over.

        Returns:
            None

        Raises:
            TypeError: If atoms is not a list of strings.

        Examples:
            Here is an example of how to use the `set_atoms` method:

            .. code-block:: python

                from rmsd import BackboneRMSD

                # Initialize the BackboneRMSD class
                backbone_rmsd = BackboneRMSD()

                # Set the atoms for RMSD calculation
                backbone_rmsd.set_atoms(["CA", "CB"])

        Further Details:
            - **Usage:** The list of atoms specifies which atoms in the protein backbone will be considered during RMSD calculations.
            - **Validation:** The method includes validation to ensure that the atoms parameter is a list of strings, representing valid atom names.
            - **Flexibility:** Users can specify any set of atoms or choose to include all atoms by setting the parameter to "all".
        """
        if atoms == "all":
            self.atoms = "all"
        if not isinstance(atoms, list) or not all((isinstance(atom, str) for atom in atoms)):
            raise TypeError("Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_chains(self, chains:list[str]) -> None:
        """
        Set the chains for RMSD calculations.

        This method sets the list of chain names to calculate RMSD over. It ensures that the provided chains parameter is a list of strings or a single string representing chain names.

        Parameters:
            chains (list[str] or str): The list of chain names or a single chain name to calculate RMSD over.

        Returns:
            None

        Raises:
            TypeError: If chains is not a list of strings or a single string.

        Examples:
            Here is an example of how to use the `set_chains` method:

            .. code-block:: python

                from rmsd import BackboneRMSD

                # Initialize the BackboneRMSD class
                backbone_rmsd = BackboneRMSD()

                # Set the chains for RMSD calculation
                backbone_rmsd.set_chains(["A", "B"])

                # Alternatively, set a single chain
                backbone_rmsd.set_chains("A")

        Further Details:
            - **Usage:** The chains parameter specifies which chains in the protein structure will be considered during RMSD calculations.
            - **Validation:** The method includes validation to ensure that the chains parameter is either a list of strings or a single string, representing valid chain names.
            - **Flexibility:** Users can specify multiple chains as a list or a single chain as a string, providing flexibility in how the RMSD calculations are configured.
        """
        if chains is None:
            self.chains = None
        elif isinstance(chains, str) and len(chains) == 1:
            self.chains = [chains]
        elif not isinstance(chains, list) or not all((isinstance(chain, str) for chain in chains)):
            raise TypeError("Chains needs to be a list, chain names (list elements) must be string.")
        else:
            self.chains = chains

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """
        Set the jobstarter configuration for the BackboneRMSD runner.

        This method sets the jobstarter configuration to be used in the RMSD calculation process.

        Parameters:
            jobstarter (JobStarter): The jobstarter configuration for running the RMSD calculations.

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
        if isinstance(jobstarter, JobStarter):
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str = None, jobstarter: JobStarter = None, chains: list[str] = None, overwrite: bool = False) -> Poses:
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

        # set up ref_col and check if col contains files:
        ref_col = ref_col or self.ref_col
        if not all([isinstance(x, str) for x in poses.df[ref_col].values]):
            raise KeyError(f"Column {ref_col} contains non-string values. First rows: {poses.df[ref_col].head(5)}")

        scorefile = f"{work_dir}/{prefix}_rmsd.{poses.storage_format}"

        # check if RMSD was calculated if overwrite was not set.
        overwrite = overwrite or self.overwrite
        if (output_df := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=output_df, prefix=prefix)
            return output.return_poses()

        # split poses into number of max_cores lists
        num_json_files = jobstarter.max_cores
        pose_dict = {os.path.abspath(row["poses"]): os.path.abspath(row[ref_col]) for row in poses}
        pose_sublists = jobstarters.split_list(poses.poses_list(), n_sublists=num_json_files)

        # setup inputs to calc_rmsd.py
        json_files = []
        cmds = []
        scorefiles = []
        for i, sublist in enumerate(pose_sublists, start=1):
            # create json dict:
            json_dict = {pose: pose_dict[pose] for pose in sublist}

            # write to file
            json_file = f"{work_dir}/rmsd_input_{str(i)}.json"
            with open(json_file, "w", encoding="UTF-8") as f:
                json.dump(json_dict, f)
            json_files.append(json_file)

            # write scorefile and cmd
            scorefiles.append((sf := f"{work_dir}/rmsd_input_{str(i)}_scores.json"))
            cmds.append(f"{self.python} {self.script_dir}/calc_rmsd.py --input_json {json_file} --output_path {sf}")

        # add options to cmds:
        chains = chains or self.chains
        if self.atoms:
            cmds = [cmd + f" --atoms='{','.join(self.atoms)}'" for cmd in cmds]
        if chains:
            cmds = [cmd + f" --chains='{','.join(chains)}'" for cmd in cmds]

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "backbone_rmsd",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        output_df = pd.concat([pd.read_json(sf) for sf in scorefiles], ignore_index=True).reset_index()
        self.save_runner_scorefile(scores=output_df, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = output_df,
            prefix = prefix,
        )
        return output.return_poses()

    def calc_all_atom_rmsd(self) -> None:
        '''Method to calculate all-atom RMSD between poses'''
        raise NotImplementedError

class MotifRMSD(Runner):
    """
    MotifRMSD Class
    ===============

    The `MotifRMSD` class is a specialized class designed to facilitate the calculation of RMSD values for specific motifs within protein structures in the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with motif-specific RMSD calculations.

    Detailed Description
    --------------------
    The `MotifRMSD` class manages all aspects of calculating RMSD for specified motifs within protein structures. It handles the configuration of necessary scripts and executables, prepares the environment for RMSD calculations, and executes the commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to motif RMSD calculation scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of RMSD commands with support for various motifs and chains.
        - Collecting and processing output data into a pandas DataFrame.
        - Managing overwrite options and handling existing score files.

    Returns
    -------
    An instance of the `MotifRMSD` class, configured to run motif RMSD calculations and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        TypeError: If motifs or chains are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `MotifRMSD` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from rmsd import MotifRMSD

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the MotifRMSD class
        motif_rmsd = MotifRMSD()

        # Run the motif RMSD calculation
        results = motif_rmsd.run(
            poses=poses,
            prefix="experiment_2",
            jobstarter=jobstarter,
            ref_col="reference",
            ref_motif="motif_A",
            target_motif="motif_B",
            atoms=["CA", "CB"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the motif RMSD calculation process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The MotifRMSD class is intended for researchers and developers who need to perform RMSD calculations for specific motifs as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, ref_col: str = None, target_motif: str = None, ref_motif: str = None, atoms: list[str] = None, return_superimposed_poses: bool = False, jobstarter: JobStarter = None, overwrite: bool = False):
        """
        Initialize the MotifRMSD class.

        This constructor sets up the MotifRMSD instance with default or provided parameters. It configures the reference column, target motif, reference motif, target chains, reference chains, jobstarter, and overwrite options for RMSD calculations.

        Parameters:
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            target_motif (str, optional): The target motif for RMSD calculations. Defaults to None.
            ref_motif (str, optional): The reference motif for RMSD calculations. Defaults to None.
            target_chains (list[str], optional): The list of chain names for the target motif. Defaults to None.
            ref_chains (list[str], optional): The list of chain names for the reference motif. Defaults to None.
            jobstarter (JobStarter, optional): The jobstarter configuration for running the RMSD calculations. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            None

        Examples:
            Here is an example of how to initialize the MotifRMSD class:

            .. code-block:: python

                from rmsd import MotifRMSD

                # Initialize the MotifRMSD class with default parameters
                motif_rmsd = MotifRMSD()

                # Initialize the MotifRMSD class with custom parameters
                motif_rmsd = MotifRMSD(
                    ref_col="reference",
                    target_motif="motif_A",
                    ref_motif="motif_B",
                    target_chains=["A"],
                    ref_chains=["B"],
                    jobstarter=JobStarter(),
                    overwrite=True
                )

        Further Details:
            - **Default Values:** If no parameters are provided, the class initializes with default values suitable for basic motif-specific RMSD calculations.
            - **Parameter Storage:** The parameters provided during initialization are stored as instance variables, which are used in subsequent method calls.
            - **Custom Configuration:** Users can customize the motif RMSD calculation process by providing specific values for the reference column, target motif, reference motif, target chains, reference chains, jobstarter, and overwrite option.
        """
        # setup config
        config = require_config()
        self.python = os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")

        #TODO implement MotifRMSD calculation based on Chain input (Should work now with a ChainSelector)!
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

        # motif settings
        self.set_ref_col(ref_col)
        self.set_target_motif(target_motif)
        self.set_ref_motif(ref_motif)
        self.set_atoms(atoms)
        self.set_return_superimposed_poses(return_superimposed_poses)

    def __str__(self):
        return "Heavyatom motif rmsd calculator"

    def set_ref_col(self, col: str) -> None:
        """
        Set the reference column for RMSD calculations.

        Parameters:
            col (str): The reference column name.
        """
        self.ref_col = col

    def set_atoms(self, atoms: list[str] = None) -> None:
        """
        Set the atoms used for superposition and RMSD calculations.

        Parameters:
            atoms (list[str]): The atoms used for superposition.
        """
        self.atoms = atoms

    def set_target_motif(self, motif: str) -> None:
        '''Method to set target motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.target_motif = motif

    def set_ref_motif(self, motif: str) -> None:
        '''Method to set reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.ref_motif = motif

    def set_return_superimposed_poses(self, return_superimposed_poses: bool) -> None:
        '''Method to set if superimposed poses should be returned. :return_superimposed_poses: has to be bool'''
        self.return_superimposed_poses = return_superimposed_poses

    def set_jobstarter(self, jobstarter: str) -> None:
        """
        Set the jobstarter configuration for the MotifRMSD runner.

        Parameters:
            jobstarter (JobStarter): The jobstarter configuration.

        Raises:
            ValueError: If jobstarter is not of type JobStarter.
        """
        if isinstance(jobstarter, JobStarter):
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Unsupported type {type(jobstarter)} for parameter :jobstarter:. Has to be of type JobStarter!")

    ################################################# Calcs ################################################
    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, ref_col: str = None, ref_motif: Any = None, target_motif: Any = None, atoms: list[str] = None, return_superimposed_poses: bool = False, overwrite: bool = False):
        """
        Calculate the motif-specific RMSD for given poses and jobstarter configuration.

        This method sets up and runs the motif-specific RMSD calculation process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            ref_motif (Any, optional): The reference motif for RMSD calculations. Defaults to None.
            target_motif (Any, optional): The target motif for RMSD calculations. Defaults to None.
            atoms (list[str], optional): The list of atom names to calculate RMSD over. Defaults to None.
            return_superimposed_poses (bool, optional): If True, return superimposed poses as new poses.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            RunnerOutput: An instance of the RunnerOutput class, containing the processed poses and results of the RMSD calculation.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If motifs or atoms are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from rmsd import MotifRMSD

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the MotifRMSD class
                motif_rmsd = MotifRMSD()

                # Run the motif RMSD calculation
                results = motif_rmsd.run(
                    poses=poses,
                    prefix="experiment_2",
                    jobstarter=jobstarter,
                    ref_col="reference",
                    ref_motif="motif_A",
                    target_motif="motif_B",
                    atoms=["CA", "CB"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It supports splitting poses into sublists for parallel processing.
            - **Input Handling:** The method prepares input JSON files for each sublist of poses and constructs commands for running motif-specific RMSD calculations.
            - **Output Management:** The method handles the collection and processing of output data from multiple score files, concatenating them into a single DataFrame and saving the results.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the motif RMSD calculation process to their specific needs, including specifying reference and target motifs, as well as atoms for RMSD calculations.

        This method is designed to streamline the execution of motif-specific RMSD calculations within the ProtFlow framework, making it easier for researchers and developers to perform and analyze motif-specific RMSD calculations.
        """
        # prep inputs
        ref_col = ref_col or self.ref_col
        ref_motif = ref_motif or self.ref_motif
        target_motif = target_motif or self.target_motif
        return_superimposed_poses = return_superimposed_poses or self.return_superimposed_poses

        # setup runner
        script_path = f"{self.script_dir}/calc_heavyatom_rmsd_batch.py"
        work_dir, jobstarter = self.generic_run_setup(
            poses = poses,
            prefix = prefix,
            jobstarters = [jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check if script exists
        if not os.path.isfile(script_path):
            raise ValueError(f"Cannot find script 'calc_heavyatom_rmsd_batch.py' at specified directory: '{self.script_dir}'. Set path to '/PATH/protflow/tools/runners_auxiliary_scripts/' for variable AUXILIARY_RUNNER_SCRIPTS_DIR in config.py file.")

        # check if outputs are present
        overwrite = overwrite or self.overwrite
        scorefile = f"{work_dir}/{prefix}_rmsds.{poses.storage_format}"
        if (rmsd_df := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=rmsd_df, prefix=prefix)
            return output.return_poses()

        # setup full input dict, batch later
        input_dict = self.setup_input_dict(
            poses = poses,
            ref_col = ref_col,
            ref_motif = ref_motif,
            target_motif = target_motif
        )

        # split input_dict into subdicts
        split_sublists = split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write n=max_cores input_json files for add_chains_batch.py
        json_files = []
        output_files = []
        for i, subdict in enumerate(subdicts, start=1):
            # setup input_json file for every batch
            opts_json_p = f"{work_dir}/rmsd_input_{str(i).zfill(4)}.json"
            with open(opts_json_p, 'w', encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(opts_json_p)
            output_files.append(f"{work_dir}/rmsd_output_{str(i).zfill(4)}.json")

        # setup atoms option
        atoms = atoms or self.atoms
        atoms_str = "" if atoms is None else f"--atoms '{','.join(atoms)}'"

        # setup pose superimposition
        super_str = "--return_superimposed_poses" if return_superimposed_poses else ""

        # start add_chains_batch.py
        cmds = [f"{self.python} {script_path} --input_json {json_f} --output_path {output_path} {atoms_str} {super_str}" for json_f, output_path in zip(json_files, output_files)]
        jobstarter.start(
            cmds = cmds,
            jobname = prefix,
            wait = True,
            output_path = work_dir
        )

        # collect outputs
        rmsd_df = pd.concat([pd.read_json(output_path) for output_path in output_files]).reset_index()
        self.save_runner_scorefile(scores=rmsd_df, scorefile=scorefile)

        outputs = RunnerOutput(
            poses = poses,
            results = rmsd_df,
            prefix = prefix
        )

        return outputs.return_poses()

    def setup_input_dict(self, poses: Poses, ref_col: str, ref_motif: Any = None, target_motif: Any = None) -> dict:
        """
        Set up the input dictionary for motif RMSD calculations.

        This method prepares a dictionary that can be written to a JSON file and used as input for the motif RMSD calculation script. The dictionary contains mappings of poses to reference PDB files, target motifs, and reference motifs.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            ref_col (str): The reference column for RMSD calculations.
            ref_motif (Any, optional): The reference motif for RMSD calculations. Defaults to None.
            target_motif (Any, optional): The target motif for RMSD calculations. Defaults to None.

        Returns:
            dict: A dictionary structured for input to the motif RMSD calculation script.

        Raises:
            TypeError: If ref_motif or target_motif is not of the expected type.

        Examples:
            Here is an example of how to use the `setup_input_dict` method:

            .. code-block:: python

                from rmsd import MotifRMSD
                from protflow.poses import Poses

                # Initialize the MotifRMSD class
                motif_rmsd = MotifRMSD()

                # Create a Poses object
                poses = Poses()

                # Set up the input dictionary for RMSD calculations
                input_dict = motif_rmsd.setup_input_dict(
                    poses=poses,
                    ref_col="reference",
                    ref_motif="motif_A",
                    target_motif="motif_B"
                )

                # Print the input dictionary
                print(input_dict)

        Further Details:
            - **Dictionary Structure:** The input dictionary maps each pose to its reference PDB file, target motif, and reference motif.
            - **Parameter Handling:** The method handles different types of inputs for motifs, ensuring that they are correctly formatted for the RMSD calculation script.
            - **Integration:** The input dictionary prepared by this method is used by the `run` method to execute motif RMSD calculations.
        """
        def setup_ref_col(ref_col: Any, poses: Poses) -> list:
            col_in_df(poses.df, ref_col)
            return poses.df[ref_col].to_list()

        def setup_motif(motif: Any, poses: Poses) -> list:
            if isinstance(motif, str):
                # if motif points to column in DataFrame, get residues.
                col_in_df(poses.df, motif)
                return [residue_selection.to_string() if isinstance(residue_selection, ResidueSelection) else residue_selection for residue_selection in poses.df[motif].to_list()]
            elif isinstance(motif, ResidueSelection):
                return [motif.to_string() for _ in poses]
            elif motif is None:
                raise ValueError(f"No motif was set for motif {motif}. Either provide a string that points to a column in poses.df containing the motifs, or pass a ResidueSelection object as the motif.")
            raise TypeError(f"Unsupportet parameter type for motif: {type(motif)}. Either provide a string that points to a column in poses.df containing the motifs, or pass a ResidueSelection object.")

        # use class default if parameters were not set and setup parameters:
        ref_l = setup_ref_col(ref_col or self.ref_col, poses)
        ref_motif_l = setup_motif(ref_motif or self.ref_motif, poses)
        target_motif_l = setup_motif(target_motif or self.target_motif, poses)

        # construct rmsd_input_dict:
        rmsd_input_dict = {pose: {} for pose in poses.poses_list()}
        for pose, ref, ref_motif_, target_motif_ in zip(poses.poses_list(), ref_l, ref_motif_l, target_motif_l):
            rmsd_input_dict[pose]["ref_pdb"] = os.path.abspath(ref)
            rmsd_input_dict[pose]["target_motif"] = target_motif_
            rmsd_input_dict[pose]["reference_motif"] = ref_motif_

        return rmsd_input_dict

class MotifSeparateSuperpositionRMSD(Runner):
    """
    MotifSeparateSuperpositionRMSD Class
    ====================================

    The `MotifSeparateSuperpositionRMSD` class is a specialized class designed to facilitate the separate superposition and calculation of RMSD values for specific motifs within protein structures in the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with motif-specific superposition and RMSD calculations.

    Detailed Description
    --------------------
    The `MotifSeparateSuperpositionRMSD` class manages all aspects of superpositioning on one motif and calculating RMSD for another within protein structures. It handles the configuration of necessary scripts and executables, prepares the environment for RMSD calculations, and executes the commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to motif RMSD calculation scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of RMSD commands with support for various motifs and chains.
        - Collecting and processing output data into a pandas DataFrame.
        - Managing overwrite options and handling existing score files.

    Returns
    -------
    An instance of the `MotifSeparateSuperpositionRMSD` class, configured to run motif RMSD calculations and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        TypeError: If motifs or chains are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `MotifRMSD` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from rmsd import MotifSeparateSuperpositionRMSD

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the MotifRMSD class
        motif_rmsd = MotifSeparateSuperpositionRMSD()

        # Run the motif RMSD calculation
        results = motif_rmsd.run(
            poses=poses,
            prefix="experiment_2",
            jobstarter=jobstarter,
            ref_col="reference",
            super_ref_motif="motif_A",
            super_target_motif="motif_B",
            super_atoms=["CA", "CB"],
            rmsd_ref_motif="motif_C",
            rmsd_target_motif=""motif_D",
            rmsd_atoms = ["CA"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the motif RMSD calculation process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The MotifSeparateSuperpositionRMSD class is intended for researchers and developers who need to perform RMSD calculations for specific motifs as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, ref_col: str = None, super_target_motif: str = None, super_ref_motif: str = None, super_atoms: list[str] = None, rmsd_target_motif : str = None, rmsd_ref_motif : str = None, rmsd_atoms: list[str] = None, super_include_het_atoms: bool = False, rmsd_include_het_atoms: bool = False, jobstarter: JobStarter = None, overwrite: bool = False):
        """
        Initialize the MotifSeparateSuperpositionRMSD class.

        This constructor sets up the MotifSeparateSuperpositionRMSD instance with default or provided parameters. It configures the reference column, superposition target motif, superposition reference motif, rmsd target motif, rmsd reference motif, inclusion of hetero atoms, jobstarter, and overwrite options for RMSD calculations.

        Parameters:
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            super_target_motif (str, optional): The target motif for superpositioning. Defaults to None.
            super_ref_motif (str, optional): The reference motif for superpositioning. Defaults to None.
            super_atoms (list, optional): The atom names for superpositioning. Defaults to None.
            super_include_het_atoms (bool, optional): Inclusion of heteroatoms (e.g. from ligands) in superpositioning. Defaults to False. 
            rmsd_target_motif (str, optional): The target motif for RMSD calculations. Defaults to None.
            rmsd_ref_motif (str, optional): The reference motif for RMSD calculations. Defaults to None.
            rmsd_atoms (list, optional): The atom names for RMSD calculations. Defaults to None.
            rmsd_include_het_atoms (bool, optional): Inclusion of heteroatoms (e.g. from ligands) for RMSD calculations. Defaults to False. 
            jobstarter (JobStarter, optional): The jobstarter configuration for running the RMSD calculations. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            None

        Examples:
            Here is an example of how to initialize the MotifRMSD class:

            .. code-block:: python

                from rmsd import MotifSeparateSuperpositionRMSD

                # Initialize the MotifSeparateSuperpositionRMSD class with default parameters
                motif_rmsd = MotifSeparateSuperpositionRMSD()

                # Initialize the MotifSeparateSuperpositionRMSD class with custom parameters
                motif_rmsd = MotifSeparateSuperpositionRMSD(
                    ref_col="reference",
                    super_ref_motif="motif_A",
                    super_target_motif="motif_B",
                    super_atoms=["CA", "CB"],
                    rmsd_ref_motif="motif_C",
                    rmsd_target_motif=""motif_D",
                    rmsd_atoms = None,
                    rmsd_include_het_atoms = True,
                    overwrite=True
                )

        
        Further Details:
            - **Default Values:** If no parameters are provided, the class initializes with default values suitable for basic motif-specific RMSD calculations.
            - **Parameter Storage:** The parameters provided during initialization are stored as instance variables, which are used in subsequent method calls.
            - **Custom Configuration:** Users can customize the motif RMSD calculation process by providing specific values for the reference column, target motif, reference motif, target chains, reference chains, jobstarter, and overwrite option.
        """
        # setup config
        config = require_config()
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        self.python = os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")

        #TODO implement MotifRMSD calculation based on Chain input (Should work now with a ChainSelector)!
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

        # motif settings
        self.set_ref_col(ref_col)
        self.set_super_target_motif(super_target_motif)
        self.set_super_ref_motif(super_ref_motif)
        self.set_super_atoms(super_atoms)
        self.set_rmsd_target_motif(rmsd_target_motif)
        self.set_rmsd_ref_motif(rmsd_ref_motif)
        self.set_rmsd_atoms(rmsd_atoms)
        self.set_super_include_het_atoms(super_include_het_atoms)
        self.set_rmsd_include_het_atoms(rmsd_include_het_atoms)

    def __str__(self):
        return "Heavyatom motif rmsd calculator"

    def set_ref_col(self, col: str) -> None:
        """
        Set the reference column for RMSD calculations.

        Parameters:
            col (str): The reference column name.
        """
        self.ref_col = col

    def set_super_atoms(self, atoms: list[str] = None) -> None:
        """
        Set the atoms used for superposition and RMSD calculations.

        Parameters:
            atoms (list[str]): The atoms used for superposition.
        """
        self.super_atoms = atoms

    def set_super_target_motif(self, motif: str) -> None:
        '''Method to set target motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.super_target_motif = motif

    def set_super_ref_motif(self, motif: str) -> None:
        '''Method to set reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.super_ref_motif = motif

    def set_jobstarter(self, jobstarter: str) -> None:
        """
        Set the jobstarter configuration for the MotifRMSD runner.

        Parameters:
            jobstarter (JobStarter): The jobstarter configuration.

        Raises:
            ValueError: If jobstarter is not of type JobStarter.
        """
        if isinstance(jobstarter, JobStarter):
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Unsupported type {type(jobstarter)} for parameter :jobstarter:. Has to be of type JobStarter!")

    def set_rmsd_ref_motif(self, motif: str) -> None:
        '''Method to set rmsd reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.rmsd_ref_motif = motif

    def set_rmsd_target_motif(self, motif: str) -> None:
        '''Method to set rmsd target motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.rmsd_target_motif = motif

    def set_rmsd_atoms(self, atoms: list[str] = None) -> None:
        """
        Set the atoms used for RMSD calculations.

        Parameters:
            atoms (list[str]): The atoms used for superposition.
        """
        self.rmsd_atoms = atoms

    def set_super_include_het_atoms(self, include_het_atoms: bool) -> None:
        '''Method to set reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.super_include_het_atoms = include_het_atoms

    def set_rmsd_include_het_atoms(self, include_het_atoms: bool) -> None:
        '''Method to set reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.rmsd_include_het_atoms = include_het_atoms

    ################################################# Calcs ################################################
    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, ref_col: str = None, super_ref_motif: Any = None, super_target_motif: Any = None, super_atoms: list[str] = None, rmsd_ref_motif: Any = None, rmsd_target_motif: Any = None, rmsd_atoms: list[str] = None, rmsd_include_het_atoms: bool = False, super_include_het_atoms: bool = False, overwrite: bool = False) -> Poses:
        """
        Superposition on one motif and calculate the RMSD on another for given poses and jobstarter configuration.

        This method sets up and runs the motif-specific superposition and RMSD calculation process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            ref_col (str, optional): The reference column for RMSD calculations. Defaults to None.
            super_target_motif (str, optional): The target motif for superpositioning. Defaults to None.
            super_ref_motif (str, optional): The reference motif for superpositioning. Defaults to None.
            super_atoms (list, optional): The atom names for superpositioning. Defaults to None.
            super_include_het_atoms (bool, optional): Inclusion of heteroatoms (e.g. from ligands) in superpositioning. Defaults to False. 
            rmsd_target_motif (str, optional): The target motif for RMSD calculations. Defaults to None.
            rmsd_ref_motif (str, optional): The reference motif for RMSD calculations. Defaults to None.
            rmsd_atoms (list, optional): The atom names for RMSD calculations. Defaults to None.
            rmsd_include_het_atoms (bool, optional): Inclusion of heteroatoms (e.g. from ligands) for RMSD calculations. Defaults to False. 
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: An instance of the Poses class, containing the processed poses and results of the RMSD calculation.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If motifs or atoms are not of the expected type.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from rmsd import MotifRMSD

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the MotifSeparateSuperpositionRMSD class
                motif_rmsd = MotifSeparateSuperpositionRMSD()

                # Run the motif RMSD calculation
                results = motif_rmsd.run(
                    poses=poses,
                    prefix="experiment_2",
                    jobstarter=jobstarter,
                    ref_col="reference",
                    super_ref_motif="motif_A",
                    super_target_motif="motif_B",
                    super_atoms=["CA", "CB"],
                    rmsd_ref_motif="motif_C",
                    rmsd_target_motif=""motif_D",
                    rmsd_atoms = ["CA"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It supports splitting poses into sublists for parallel processing.
            - **Input Handling:** The method prepares input JSON files for each sublist of poses and constructs commands for running motif-specific RMSD calculations.
            - **Output Management:** The method handles the collection and processing of output data from multiple score files, concatenating them into a single DataFrame and saving the results.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the motif RMSD calculation process to their specific needs, including specifying reference and target motifs, as well as atoms for RMSD calculations.

        This method is designed to streamline the execution of motif-specific RMSD calculations within the ProtFlow framework, making it easier for researchers and developers to perform and analyze motif-specific RMSD calculations.
        """
        # prep inputs
        ref_col = ref_col or self.ref_col
        super_ref_motif = super_ref_motif or self.super_ref_motif
        super_target_motif = super_target_motif or self.super_target_motif
        super_include_het_atoms = super_include_het_atoms or self.super_include_het_atoms
        rmsd_ref_motif = rmsd_ref_motif or self.rmsd_ref_motif
        rmsd_target_motif = rmsd_target_motif or self.rmsd_target_motif
        rmsd_include_het_atoms = rmsd_include_het_atoms or self.rmsd_include_het_atoms

        # setup runner
        script_path = os.path.join(self.script_dir, "calc_heavyatom_rmsd_batch_separate.py")
        work_dir, jobstarter = self.generic_run_setup(
            poses = poses,
            prefix = prefix,
            jobstarters = [jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check if script exists
        if not os.path.isfile(script_path):
            raise ValueError(f"Cannot find script 'calc_heavyatom_rmsd_batch_separate.py' at specified directory: '{self.script_dir}'. Set path to '/PATH/protflow/tools/runners_auxiliary_scripts/' for variable AUXILIARY_RUNNER_SCRIPTS_DIR in config.py file.")

        # check if outputs are present
        overwrite = overwrite or self.overwrite
        scorefile = f"{work_dir}/{prefix}_rmsds.{poses.storage_format}"
        if (rmsd_df := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=rmsd_df, prefix=prefix)
            return output.return_poses()

        # setup full input dict, batch later
        input_dict = self.setup_input_dict(
            poses = poses,
            ref_col = ref_col,
            ref_motif = super_ref_motif,
            target_motif = super_target_motif,
            rmsd_ref_motif = rmsd_ref_motif,
            rmsd_target_motif = rmsd_target_motif,
        )

        # split input_dict into subdicts
        split_sublists = split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write n=max_cores input_json files for add_chains_batch.py
        json_files = []
        output_files = []
        for i, subdict in enumerate(subdicts, start=1):
            # setup input_json file for every batch
            opts_json_p = f"{work_dir}/rmsd_input_{str(i).zfill(4)}.json"
            with open(opts_json_p, 'w', encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(opts_json_p)
            output_files.append(f"{work_dir}/rmsd_output_{str(i).zfill(4)}.json")

        # setup atoms option
        super_atoms = super_atoms or self.super_atoms
        super_atoms_str = "" if super_atoms is None else f"--super_atoms '{','.join(super_atoms)}'"
        rmsd_atoms = rmsd_atoms or self.rmsd_atoms
        rmsd_atoms_str = "" if rmsd_atoms is None else f"--rmsd_atoms '{','.join(rmsd_atoms)}'"

        super_include_het_atoms_str = "--super_include_het_atoms" if super_include_het_atoms == True  or self.super_include_het_atoms == True else ""
        rmsd_include_het_atoms_str = "--rmsd_include_het_atoms" if rmsd_include_het_atoms == True or self.rmsd_include_het_atoms == True else ""

        # start add_chains_batch.py
        cmds = [f"{self.python} {script_path} --input_json {json_f} --output_path {output_path} {super_atoms_str} {rmsd_atoms_str} {super_include_het_atoms_str} {rmsd_include_het_atoms_str}" for json_f, output_path in zip(json_files, output_files)]
        jobstarter.start(
            cmds = cmds,
            jobname = prefix,
            wait = True,
            output_path = work_dir
        )

        # collect outputs
        rmsd_df = pd.concat([pd.read_json(output_path) for output_path in output_files]).reset_index()
        self.save_runner_scorefile(scores=rmsd_df, scorefile=scorefile)

        outputs = RunnerOutput(
            poses = poses,
            results = rmsd_df,
            prefix = prefix
        )

        return outputs.return_poses()

    def setup_input_dict(self, poses: Poses, ref_col: str, ref_motif: Any = None, target_motif: Any = None, rmsd_ref_motif: Any = None, rmsd_target_motif: Any = None) -> dict:
        """
        Set up the input dictionary for motif RMSD calculations.

        This method prepares a dictionary that can be written to a JSON file and used as input for the motif RMSD calculation script. The dictionary contains mappings of poses to reference PDB files, target motifs, and reference motifs.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            ref_col (str): The reference column for RMSD calculations.
            ref_motif (Any, optional): The reference motif for superposition. Defaults to None.
            target_motif (Any, optional): The target motif for superposition. Defaults to None.
            rmsd_ref_motif (Any, optional): The reference motif for RMSD calculations. Defaults to None.
            rmsd_target_motif (Any, optional): The target motif for RMSD calculations. Defaults to None.


        Returns:
            dict: A dictionary structured for input to the motif RMSD calculation script.

        Raises:
            TypeError: If ref_motif or target_motif is not of the expected type.

        Examples:
            Here is an example of how to use the `setup_input_dict` method:

            .. code-block:: python

                from rmsd import MotifRMSD
                from protflow.poses import Poses

                # Initialize the MotifRMSD class
                motif_rmsd = MotifRMSD()

                # Create a Poses object
                poses = Poses()

                # Set up the input dictionary for RMSD calculations
                input_dict = motif_rmsd.setup_input_dict(
                    poses=poses,
                    ref_col="reference",
                    ref_motif="motif_A",
                    target_motif="motif_B"
                )

                # Print the input dictionary
                print(input_dict)

        Further Details:
            - **Dictionary Structure:** The input dictionary maps each pose to its reference PDB file, target motif, and reference motif.
            - **Parameter Handling:** The method handles different types of inputs for motifs, ensuring that they are correctly formatted for the RMSD calculation script.
            - **Integration:** The input dictionary prepared by this method is used by the `run` method to execute motif RMSD calculations.
        """
        def setup_ref_col(ref_col: Any, poses: Poses) -> list:
            col_in_df(poses.df, ref_col)
            return poses.df[ref_col].to_list()

        def setup_motif(motif: Any, poses: Poses) -> list:
            if isinstance(motif, str):
                # if motif points to column in DataFrame, get residues.
                col_in_df(poses.df, motif)
                return [residue_selection.to_string() if isinstance(residue_selection, ResidueSelection) else residue_selection for residue_selection in poses.df[motif].to_list()]
            elif isinstance(motif, ResidueSelection):
                return [motif for _ in poses]
            raise TypeError(f"Unsupportet parameter type for motif: {type(motif)}. Either provide a string that points to a column in poses.df containing the motifs, or pass a ResidueSelection object.")

        # use class default if parameters were not set and setup parameters:
        ref_l = setup_ref_col(ref_col or self.ref_col, poses)
        ref_motif_l = setup_motif(ref_motif or self.super_ref_motif or target_motif or self.super_target_motif, poses)
        target_motif_l = setup_motif(target_motif or self.super_target_motif or ref_motif or self.super_ref_motif, poses)
        rmsd_ref_motif_l = setup_motif(rmsd_ref_motif or self.rmsd_ref_motif or rmsd_target_motif or self.rmsd_target_motif, poses)
        rmsd_target_motif_l = setup_motif(rmsd_target_motif or self.rmsd_target_motif or rmsd_ref_motif or self.rmsd_ref_motif, poses)

        # construct rmsd_input_dict:
        rmsd_input_dict = {pose: {} for pose in poses.poses_list()}
        for pose, ref, ref_motif_, target_motif_, rmsd_ref_motif_, rmsd_target_motif_ in zip(poses.poses_list(), ref_l, ref_motif_l, target_motif_l, rmsd_ref_motif_l, rmsd_target_motif_l):
            rmsd_input_dict[pose]["ref_pdb"] = os.path.abspath(ref)
            rmsd_input_dict[pose]["target_motif"] = target_motif_
            rmsd_input_dict[pose]["reference_motif"] = ref_motif_
            rmsd_input_dict[pose]["rmsd_ref_motif"] = rmsd_ref_motif_
            rmsd_input_dict[pose]["rmsd_target_motif"] = rmsd_target_motif_
        return rmsd_input_dict
