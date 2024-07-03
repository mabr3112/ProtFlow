# TODO: Generate proper doc strings!!!
"""
Ligand Module
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
import logging
import os
from typing import Any

# import dependencies
import pandas as pd
import protflow

# import customs
from protflow.config import PROTFLOW_ENV
from protflow.runners import Runner, RunnerOutput, col_in_df
from protflow.poses import Poses
from protflow.jobstarters import JobStarter, split_list

class LigandClashes(Runner):
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
    def __init__(self, ligand_chain: str = None, factor: float = 1, atoms: list[str] = None, exclude_ligand_elements: list[str] = None, jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
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
        self.set_ligand_chain(ligand_chain)
        self.set_atoms(atoms)
        self.set_factor(factor)
        self.set_exclude_ligand_elements(exclude_ligand_elements)
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################
    def set_ligand_chain(self, ligand_chain: str) -> None:
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
        self.ligand_chain = ligand_chain

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
            raise TypeError(f"Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_factor(self, factor: float) -> None:
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
        self.factor = factor

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
        
    def set_exclude_ligand_elements(self, exclude_ligand_elements: list[str]):
        self.exclude_ligand_elements = exclude_ligand_elements

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ligand_chain: str = None, factor: float = 1, jobstarter: JobStarter = None, atoms: list[str] = None, exclude_ligand_elements: list[str] = None, overwrite: bool = False) -> Poses:
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
        # if self.atoms is all, calculate Allatom RMSD.

        # prep variables
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running ligand clash detection in {work_dir} on {len(poses.df.index)} poses.")

        ligand_chain = ligand_chain or self.ligand_chain
        atoms = atoms or self.atoms
        factor = factor or self.factor
        exclude_ligand_elements = exclude_ligand_elements or self.exclude_ligand_elements
        atoms_str = f"--atoms {','.join(atoms)}" if atoms else ""
        exclude_ligand_elements_str = f"--exclude_elements {','.join(exclude_ligand_elements)}" if exclude_ligand_elements else ""

        scorefile = os.path.join(work_dir, f"{prefix}_ligand_clashes.{poses.storage_format}")

        # check if RMSD was calculated if overwrite was not set.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # split poses into number of max_cores lists, but not more than 100 poses per sublist (otherwise, argument list too long error occurs)
        poses_sublists = split_list(poses.poses_list(), n_sublists=jobstarter.max_cores) if len(poses.df.index) / jobstarter.max_cores < 100 else split_list(poses.poses_list(), element_length=100)
        out_files = [os.path.join(poses.work_dir, prefix, f"out_{index}.json") for index, sublist in enumerate(poses_sublists)]
        cmds = [f"{os.path.join(PROTFLOW_ENV, 'python3')} {__file__} --poses {','.join(poses_sublist)} --out {out_file} --mode clash_vdw --factor {factor} --ligand_chain {ligand_chain} {atoms_str} {exclude_ligand_elements_str}" for out_file, poses_sublist in zip(out_files, poses_sublists)]
        
        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "ligand_clash",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        scores = pd.concat([pd.read_json(output) for output in out_files]).reset_index(drop=True)
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info(f"Ligand clash detection completed. Returning scores.")
        return output.return_poses()
    

class LigandContacts(Runner):
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
    def __init__(self, ligand_chain: str = None, min_dist: float = 0, max_dist: float = 5, atoms: list[str] = None, exclude_elements: list[str] = None, jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
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
        self.set_ligand_chain(ligand_chain)
        self.set_atoms(atoms)
        self.set_min_dist(min_dist)
        self.set_max_dist(max_dist)
        self.set_exclude_elements(exclude_elements)
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################
    def set_ligand_chain(self, ligand_chain: str) -> None:
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
        self.ligand_chain = ligand_chain

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
            raise TypeError(f"Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_min_dist(self, min_dist: float) -> None:
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
        self.min_dist = min_dist

    def set_max_dist(self, max_dist: float) -> None:
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
        self.max_dist = max_dist

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
        
    def set_exclude_elements(self, exclude_elements: list[str]):
        self.exclude_elements = exclude_elements

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ligand_chain: str = None, jobstarter: JobStarter = None, min_dist: float = None, max_dist: float = None, atoms: list[str] = None, exclude_elements: list[str] = None, overwrite: bool = False) -> Poses:
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

        logging.info(f"Running ligand contact detection in {work_dir} on {len(poses.df.index)} poses.")

        ligand_chain = ligand_chain or self.ligand_chain
        min_dist = min_dist or self.min_dist
        max_dist = max_dist or self.max_dist
        if any(attr == None for attr in [ligand_chain, min_dist, max_dist]):
            raise ValueError(f"ligand_chain, min_dist and max_dist must be set, but are {[ligand_chain, min_dist, max_dist]}!")
        atoms = atoms or self.atoms
        exclude_elements = exclude_elements or self.exclude_elements
        
        atoms_str = f"--atoms {','.join(atoms)}" if atoms else ""
        exclude_elements_str = f"--exclude_elements {','.join(exclude_elements)}" if exclude_elements else ""

        scorefile = os.path.join(work_dir, f"{prefix}_ligand_contacts.{poses.storage_format}")

        # check if RMSD was calculated if overwrite was not set.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # split poses into number of max_cores lists, but not more than 100 poses per sublist (otherwise, argument list too long error occurs)
        poses_sublists = split_list(poses.poses_list(), n_sublists=jobstarter.max_cores) if len(poses.df.index) / jobstarter.max_cores < 100 else split_list(poses.poses_list(), element_length=100)
        out_files = [os.path.join(poses.work_dir, prefix, f"out_{index}.json") for index, sublist in enumerate(poses_sublists)]
        cmds = [f"{os.path.join(PROTFLOW_ENV, 'python3')} {__file__} --poses {','.join(poses_sublist)} --out {out_file} --min_dist {min_dist} --max_dist {max_dist} --mode contacts --ligand_chain {ligand_chain} {atoms_str} {exclude_elements_str}" for out_file, poses_sublist in zip(out_files, poses_sublists)]
        
        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "ligand_clash",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        scores = pd.concat([pd.read_json(output) for output in out_files]).reset_index(drop=True)
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info(f"Ligand contact detection completed. Returning scores.")
        return output.return_poses()
    
    
def _calc_ligand_clashes_vdw(pose: str, ligand_chain: str, factor: float = 1, atoms: list[str] = None, exclude_ligand_elements: list[str] = None) -> int:
    """
    Calculate ligand clashes for a PDB file given a ligand chain.

    This method calculates the number of clashes between a specified ligand chain and the rest of the structure in a PDB file or a Bio.PDB Structure object. A clash is defined as any pair of atoms (one from the ligand, one from the rest of the structure) that are within the sum of their Van der Waals radii multiplied by a factor.

    Parameters:
        pose (str | Bio.PDB.Structure.Structure): The pose representing the structure, which can be a path to a PDB file (str) or a Bio.PDB Structure object.
        ligand_chain (str): The chain identifier for the ligand within the structure.
        factor (float, optional): The multiplier for the VdW clash threshold for defining a clash. Lower numbers result in less stringent clash detection. Default is 1.0.
        atoms (list[str], optional): A list of atom names to consider for clash calculations. If None, all atoms are considered. If specified, only these atoms will be included in the clash calculation.
        exclude_ligand_elements (list[str], optional): A list of elements that should not be considered during clash detection (e.g. ['H']). Default is None

    Returns:
        float: The number of clashes found between the ligand and the rest of the structure.

    Examples:
        Here is an example of how to use the `calc_ligand_clashes` method:

        .. code-block:: python

            from Bio.PDB import PDBParser

            # Load structure from a PDB file
            parser = PDBParser()
            structure = parser.get_structure("example", "example.pdb")

            # Calculate clashes
            clashes = calc_ligand_clashes_vdw(structure, ligand_chain="A", factor=0.8, atoms=["N", "CA", "C"], exclude_ligand_atoms=["H"])
            # clashes will be a float representing the number of clashes

    Further Details:
        - **Clash Calculation:** The method calculates the Euclidean distance between all specified atoms of the ligand chain and the rest of the structure. A clash is detected if the distance is less than the sum of their Van der Waals radii multiplied by a set factor.
        - **Usage:** This function is useful for evaluating potential steric clashes in molecular docking studies or for validating the positioning of ligands in structural models.

    This method is designed to facilitate the detection of steric clashes between ligands and the surrounding structure, providing a quantitative measure of potential conflicts.
    """

    # verify inputs
    pose = load_structure_from_pdbfile(pose)
    
    if exclude_ligand_elements:
        if not isinstance(exclude_ligand_elements, list):
            raise ValueError(f"Parameter:exclude_ligand_atoms: has to be a list of str, not {type(exclude_ligand_elements)}!")
        exclude_ligand_elements = [element.lower() for element in exclude_ligand_elements]

    # import VdW radii
    vdw_dict = vdw_radii()
    
    # check for ligand chain
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # get atoms
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
        pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
        pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
    else:
        raise ValueError(f"Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")
    
    # get ligand atoms
    if exclude_ligand_elements:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
        ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
    else:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms()])
        ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms()])

    if np.any(np.isnan(ligand_vdw)):
        raise RuntimeError("Could not find Van der Waals radii for all elements in ligand. Check protflow.utils.vdw_radii and add it, if applicable!")

    # calculate distances between all atoms of ligand and protein
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # calculate distance cutoff for each atom pair, considering VdW radii 
    distance_cutoff = pose_vdw[:, np.newaxis] + ligand_vdw[np.newaxis, :]
    # multiply distance cutoffs with set parameter
    distance_cutoff = distance_cutoff * factor

    # compare distances to distance_cutoff
    check = dgram - distance_cutoff

    # count number of clashes (where distances are below distance cutoff)
    clashes = int(np.sum((check < 0)))

    return clashes


def _calc_ligand_contacts(pose: str, ligand_chain: str, min_dist: float = 3, max_dist: float = 5, atoms: list[str] = None, exclude_elements: list[str] = None) -> float:
    """
    Calculate contacts of a ligand within a structure.

    This method calculates the number of contacts between a specified ligand chain and the rest of the structure within a specified distance range. Contacts are defined as any pair of atoms (one from the ligand, one from the rest of the structure) where the distance falls between the minimum and maximum specified distances.

    Parameters:
        pose (str | Bio.PDB.Structure.Structure): The pose representing the structure, which can be a path to a PDB file (str) or a Bio.PDB Structure object.
        ligand_chain (str): The chain identifier for the ligand within the structure.
        min_dist (float, optional): The minimum distance threshold for defining a contact. Default is 3.0.
        max_dist (float, optional): The maximum distance threshold for defining a contact. Default is 5.0.
        atoms (list[str], optional): A list of atom names to consider for contact calculations. If None, all atoms are considered. If specified, only these atoms will be included in the contact calculation.
        excluded_elements (list[str], optional): A list of element symbols to exclude from the contact calculations. Default is ["H"].

    Returns:
        float: The number of contacts normalized by the number of ligand atoms.

    Examples:
        Here is an example of how to use the `calc_ligand_contacts` method:

        .. code-block:: python

            from Bio.PDB import PDBParser

            # Load structure from a PDB file
            parser = PDBParser()
            structure = parser.get_structure("example", "example.pdb")

            # Calculate contacts
            contacts = calc_ligand_contacts(structure, ligand_chain="A", min_dist=3.0, max_dist=5.0, atoms=["N", "CA", "C"], excluded_elements=["H", "O"])
            # contacts will be a float representing the number of contacts normalized by the number of ligand atoms

    Further Details:
        - **Contact Calculation:** The method calculates the Euclidean distance between all specified atoms of the ligand chain and the rest of the structure. A contact is counted if the distance is within the specified range (min_dist to max_dist).
        - **Usage:** This function is useful for evaluating potential interactions between ligands and the surrounding structure, particularly in drug design and molecular docking studies.

    This method is designed to facilitate the detection of relevant contacts between ligands and the surrounding structure, providing a quantitative measure of potential interactions.
    """

    # verify inputs
    pose = load_structure_from_pdbfile(pose)

    if exclude_elements:
        exclude_elements = [element.lower() for element in exclude_elements]

    # check for ligand chain
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # get pose atoms
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.element.lower() not in exclude_elements])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms and atom.element.lower() not in exclude_elements])
    else:
        raise ValueError(f"Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")
    ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.element.lower() not in exclude_elements])

    # calculate complete dgram
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # return number of contacts
    return round(np.sum((dgram > min_dist) & (dgram < max_dist)) / len(ligand_atoms), 2)

def main(args):

    input_poses = args.poses.split(",")
    if args.atoms: atoms = args.atoms.split(",")
    else: atoms = None
    if args.exclude_elements: exclude_elements = args.exclude_elements
    else: exclude_elements = []

    if args.mode == "clash_vdw":
        clashes = [_calc_ligand_clashes_vdw(pose, args.ligand_chain, args.factor, atoms, exclude_elements) for pose in input_poses]
        out_dict = {"description": [os.path.splitext(os.path.basename(pose))[0] for pose in input_poses], "location": input_poses, "clashes": clashes}
        df = pd.DataFrame(out_dict)
        df.to_json(args.out)

    elif args.mode == "contacts":
        contacts = [_calc_ligand_contacts(pose, args.ligand_chain, args.min_dist, args.max_dist, atoms, exclude_elements) for pose in input_poses]
        out_dict = {"description": [os.path.splitext(os.path.basename(pose))[0] for pose in input_poses], "location": input_poses, "contacts": contacts}
        df = pd.DataFrame(out_dict)
        df.to_json(args.out)


if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd
    from protflow.utils.biopython_tools import load_structure_from_pdbfile
    from protflow.utils.utils import vdw_radii

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--poses", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--out", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--factor", type=float, default=None, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--ligand_chain", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--atoms", type=str, default=None, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--exclude_elements", type=str, default=None, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--mode", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--min_dist", type=float, default=0, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--max_dist", type=float, default=5, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")

    arguments = argparser.parse_args()
    main(arguments)