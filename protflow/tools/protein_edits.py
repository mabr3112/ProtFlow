"""
ProteinEdits Module
===================

This module provides the functionality to handle various protein editing tasks within the ProtFlow framework. It offers tools to add and remove protein chains, add sequences to proteins, and multimerize sequences in a structured and automated manner.

Detailed Description
--------------------
The `protein_edits` module contains classes and methods designed to perform common protein editing operations. The `ChainAdder` class provides methods for adding chains to protein structures, including functionality for superimposing chains based on motifs or existing chains. The `ChainRemover` class allows for the removal of specified chains from protein structures. Additionally, methods for adding sequences to proteins and creating multimers from sequences are included, streamlining the process of preparing protein structures for further analysis.

The module integrates seamlessly with the ProtFlow ecosystem, leveraging shared configurations, job management capabilities, and data structures to provide a cohesive user experience. It supports automatic setup and execution of jobs, handling of input and output files, and robust error handling and logging.

Usage
-----
To use this module, create instances of the `ChainAdder` or `ChainRemover` classes and invoke their respective methods with appropriate parameters. The module handles the configuration, execution, and result collection processes, allowing users to focus on interpreting the results.

Examples
--------
Here is an example of how to initialize and use the `ChainAdder` and `ChainRemover` classes within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from protein_edits import ChainAdder, ChainRemover

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the ChainAdder class
    chain_adder = ChainAdder(jobstarter=jobstarter)

    # Add a chain to the poses
    added_chains = chain_adder.add_chain(
        poses=poses,
        prefix="experiment_1",
        ref_col="reference_column",
        copy_chain="A",
        jobstarter=jobstarter,
        overwrite=True
    )

    # Initialize the ChainRemover class
    chain_remover = ChainRemover(jobstarter=jobstarter)

    # Remove a chain from the poses
    removed_chains = chain_remover.remove_chains(
        poses=poses,
        prefix="experiment_2",
        chains=["A"],
        jobstarter=jobstarter,
        overwrite=True
    )

    # Access and process the results
    print(added_chains)
    print(removed_chains)

Further Details
---------------
- Edge Cases: The module handles various edge cases, such as missing chain specifications and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the process.
- Customizability: Users can customize the processes through multiple parameters, including the chain to add or remove, sequence details for adding sequences, and the number of protomers for multimerization.
- Integration: The module integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate protein editing tasks into their computational workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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

# imports
import json
import os

# dependencies
from numpy import isin
import pandas as pd

# customs
from protflow import jobstarters
from protflow.jobstarters import JobStarter, split_list
from protflow.poses import Poses
from protflow.residues import ResidueSelection
from protflow.runners import Runner, RunnerOutput, col_in_df
from protflow.config import PROTFLOW_ENV
from protflow.config import AUXILIARY_RUNNER_SCRIPTS_DIR
from protflow.utils.plotting import check_for_col_in_df
from protflow.utils.utils import parse_fasta_to_dict
from protflow.utils.biopython_tools import load_structure_from_pdbfile

class ChainAdder(Runner):
    """
    ChainAdder Class
    ================

    The `ChainAdder` class is a specialized class designed to facilitate the addition of chains to protein structures within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with chain addition processes.

    Detailed Description
    --------------------
    The `ChainAdder` class manages all aspects of adding chains to protein structures. It configures necessary scripts and executables, prepares the environment for the addition processes, and executes the required commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to chain addition scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of chain addition commands with support for superimposition on motifs or existing chains.
        - Collecting and processing output data into a structured format.
        - Providing methods for adding sequences to proteins and creating multimers from sequences.

    Returns
    -------
    An instance of the `ChainAdder` class, configured to add chains to protein structures and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If motifs or chains are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `ChainAdder` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from protein_edits import ChainAdder

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ChainAdder class
        chain_adder = ChainAdder(jobstarter=jobstarter)

        # Add a chain to the poses
        added_chains = chain_adder.add_chain(
            poses=poses,
            prefix="experiment_1",
            ref_col="reference_column",
            copy_chain="A",
            jobstarter=jobstarter,
            overwrite=True
        )

        # Access and process the results
        print(added_chains)

    Further Details
    ---------------
        - Edge Cases: The class handles various edge cases, such as missing chain specifications and the need to overwrite previous results.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the chain addition process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ChainAdder class is intended for researchers and developers who need to add chains to protein structures as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, default_python=os.path.join(PROTFLOW_ENV, "python3"), jobstarter: JobStarter = None):
        """
        Initialize the ChainAdder class.

        This method sets up the ChainAdder class by configuring the path to the default Python executable and 
        initializing the job starter. The ChainAdder class is used to add chains to protein structures within 
        the ProtFlow framework.

        Parameters
        ----------
        default_python : str, optional
            The path to the default Python executable, by default `os.path.join(PROTFLOW_ENV, "python3")`.
        jobstarter : JobStarter, optional
            An instance of the JobStarter class to manage job execution, by default None.

        Attributes
        ----------
        python : str
            Path to the Python executable used for running scripts.
        jobstarter : JobStarter
            An instance of the JobStarter class to manage job execution.

        Examples
        --------
        Here is an example of how to initialize the ChainAdder class:

        .. code-block:: python

            from protflow.jobstarters import JobStarter
            from protein_edits import ChainAdder

            # Initialize the ChainAdder class
            jobstarter = JobStarter()
            chain_adder = ChainAdder(jobstarter=jobstarter)

        Notes
        -----
        The ChainAdder class depends on the ProtFlow environment being properly configured. Ensure that the 
        `PROTFLOW_ENV` and necessary scripts are correctly set up before using this class.

        Raises
        ------
        FileNotFoundError
            If the specified Python executable is not found.
        """
        self.python = self.search_path(default_python, "PROTFLOW_ENV")
        self.jobstarter = jobstarter

    def __str__(self):
        return "chain_adder"

    ################ Methods #########################
    def run(self, poses, prefix, jobstarter):
        '''.run() not implemented for ChainAdder class. Use methods like: .add_chain() or .superimpose_add_chain() instead!!!'''
        raise NotImplementedError

    def add_chain(self, poses: Poses, prefix: str, ref_col: str, copy_chain: str, jobstarter: JobStarter = None, overwrite: bool = False) -> Poses:
        """
        Add a chain to the poses.

        This method adds a specified chain to the protein structures in `poses` by using the `superimpose_add_chain` method without any superimposition, effectively copying the chain as-is.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            ref_col (str): The column in the poses DataFrame that references the structures to be used.
            copy_chain (str): The chain identifier to copy.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job execution. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.

        Returns:
            Poses: An updated Poses object with the new chain added.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the methods.
            TypeError: If invalid argument types are provided to the methods.

        Examples:
            Here is an example of how to initialize and use the `add_chain` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from protein_edits import ChainAdder

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the ChainAdder class
                chain_adder = ChainAdder(jobstarter=jobstarter)

                # Add a chain to the poses
                added_chains = chain_adder.add_chain(
                    poses=poses,
                    prefix="experiment_1",
                    ref_col="reference_column",
                    copy_chain="A",
                    jobstarter=jobstarter,
                    overwrite=True
                )

                # Access and process the results
                print(added_chains)

        Further Details
        ---------------
        - **Method Simplicity:** This method uses `superimpose_add_chain` without specifying any superimposition parameters, making it a straightforward way to add chains without the complexity of superimposition.
        - **Path Configuration:** Ensure the paths to the scripts and executables are correctly configured as per ProtFlow setup. Using default paths is recommended unless customization is necessary.
        - **JobStarter Integration:** The JobStarter object is used to manage job execution, ensuring processes are handled efficiently. If a JobStarter is not provided, the method will not operate without it.
        """
        # run superimpose without specifying anything to superimpose on (will not superimpose)
        chains_added = self.superimpose_add_chain(
            poses = poses,
            prefix=prefix,
            ref_col=ref_col,
            copy_chain=copy_chain,
            jobstarter=jobstarter,
            overwrite=overwrite
        )
        return chains_added

    def superimpose_add_chain(self, poses: Poses, prefix: str, ref_col: str, copy_chain: str, jobstarter: JobStarter = None, target_motif: ResidueSelection = None, reference_motif: ResidueSelection = None, target_chains: list = None, reference_chains: list = None, overwrite: bool = False) -> Poses:
        """
        Add a protein chain after superimposition on a motif or chain.

        This method adds a chain to the protein structures in `poses` by superimposing it on a specified motif or chain. 
        It sets up and executes the necessary scripts, handles the environment configuration, and processes the output.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            ref_col (str): The column in the poses DataFrame that references the structures to be used.
            copy_chain (str): The chain identifier to copy.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job execution. Defaults to None.
            target_motif (ResidueSelection, optional): The target motif for superimposition. Defaults to None.
            reference_motif (ResidueSelection, optional): The reference motif for superimposition. Defaults to None.
            target_chains (list, optional): A list of target chains for superimposition. Defaults to None.
            reference_chains (list, optional): A list of reference chains for superimposition. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.

        Returns:
            Poses: An updated Poses object with the new chain added.

        Raises:
            ValueError: If both motifs and chains are specified for superimposition.
            FileNotFoundError: If required files or directories are not found during the execution process.
            TypeError: If invalid argument types are provided to the methods.

        Examples:
            Here is an example of how to initialize and use the `superimpose_add_chain` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from protein_edits import ChainAdder

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the ChainAdder class
                chain_adder = ChainAdder(jobstarter=jobstarter)

                # Add a chain to the poses
                added_chains = chain_adder.superimpose_add_chain(
                    poses=poses,
                    prefix="experiment_1",
                    ref_col="reference_column",
                    copy_chain="A",
                    jobstarter=jobstarter,
                    overwrite=True
                )

                # Access and process the results
                print(added_chains)

        Further Details
        ---------------
        - **Path Configuration:** Ensure the paths to the scripts and executables are correctly configured as per ProtFlow setup. Using default paths is recommended unless customization is necessary.
        - **JobStarter Integration:** The JobStarter object is used to manage job execution, ensuring processes are handled efficiently. If a JobStarter is not provided, the method will not operate without it.
        
        Notes:
            This method ensures robust error handling and logging for easier debugging and verification of the process.
        """
        # sanity (motif and chain superimposition at the same time is not possible)
        def output_exists(work_dir, poses):
            '''checks if output of copying chains exists'''
            return os.path.isdir(work_dir) and all((os.path.isfile(os.path.join(work_dir, pose.rsplit("/", maxsplit=1)[-1])) for pose in poses.poses_list()))

        if (target_motif or reference_motif) and (target_chains or reference_chains):
            raise ValueError(f"Either motif or chains can be specified for superimposition, but never both at the same time! Decide whether to superimpose over a selected chain or a selected motif.")

        # runner setup
        script_path = f"{AUXILIARY_RUNNER_SCRIPTS_DIR}/add_chains_batch.py"
        work_dir, jobstarter = self.generic_run_setup(
            poses = poses,
            prefix = prefix,
            jobstarters = [jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check for outputs
        if output_exists(work_dir, poses) and not overwrite:
            return poses.change_poses_dir(work_dir, copy=False)

        # setup motif args (extra function)
        input_dict = self._setup_superimposition_args(
            poses = poses,
            ref_col = ref_col,
            copy_chain = copy_chain,
            target_motif = target_motif,
            reference_motif = reference_motif,
            target_chains = target_chains,
            reference_chains = reference_chains,
        )

        # split input_dict into subdicts
        split_sublists = jobstarters.split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write n=max_cores input_json files for add_chains_batch.py
        json_files = []
        for i, subdict in enumerate(subdicts, start=1):
            opts_json_p = f"{work_dir}/add_chain_input_{str(i).zfill(4)}.json"
            with open(opts_json_p, 'w', encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(opts_json_p)

        # start add_chains_batch.py
        cmds = [f"{self.python} {script_path} --input_json {json_f} --output_dir {work_dir}" for json_f in json_files]
        jobstarter.start(
            cmds = cmds,
            jobname = f"add_chains_{prefix}",
            wait = True,
            output_path = work_dir
        )

        return poses.change_poses_dir(work_dir, copy=False)

    def _setup_superimposition_args(self, poses: Poses, ref_col: str, copy_chain: str, target_motif: ResidueSelection = None, reference_motif: ResidueSelection = None, target_chains: list = None, reference_chains: list = None) -> dict:
        '''Prepares motif and chain specifications for superimposer setup.
        Returns dictionary (dict) that holds the kwargs for superimposition: {'target_motif': [target_motif_list], ...}'''
        # safety
        if (target_motif or reference_motif) and (target_chains or reference_chains):
            raise ValueError(f"Either motif or chains can be specified for superimposition, but not both!")

        # setup copy_chain and reference_pdb in output:
        col_in_df(poses.df, ref_col)
        copy_chain_l = setup_chain_list(copy_chain, poses)
        out_dict = {pose["poses"]: {"copy_chain": chain, "reference_pdb": os.path.abspath(pose[ref_col])} for pose, chain in zip(poses, copy_chain_l)}
        #out_dict = {'target_motif': None, 'reference_motif': None, 'target_chains': None, 'reference_chains': None}

        # if nothing is specified, return nothing.
        if all ((opt is None for opt in [reference_motif, target_motif, reference_chains, target_chains])):
            return out_dict

        # setup motif definitions
        if (target_motif or reference_motif):
            for pose in poses:
                out_dict[pose["poses"]]['target_motif'] = self.parse_motif(target_motif or reference_motif, pose)
                out_dict[pose["poses"]]['reference_motif'] = self.parse_motif(reference_motif or target_motif, pose)

        # setup chains definitions
        if (target_chains or reference_chains):
            for pose in poses:
                out_dict[pose["poses"]]["target_chains"] = parse_chain(target_chains or reference_chains, pose)
                out_dict[pose["poses"]]["reference_chains"] = parse_chain(reference_chains or target_chains, pose)

        return out_dict

    def parse_motif(self, motif: ResidueSelection|str, pose: pd.Series) -> str:
        """
        Set up motif from target_motif input.

        This method converts a given motif, either a `ResidueSelection` object or a string, into a string format suitable for further processing. 
        If the motif is a string, it checks if it is a column in the `pose` DataFrame and assumes it points to a `ResidueSelection` object.

        Parameters:
            motif (ResidueSelection | str): The motif to be parsed. It can be either a `ResidueSelection` object or a string.
            pose (pd.Series): A row from the poses DataFrame that contains information about the protein structure.

        Returns:
            str: The motif in string format.

        Raises:
            ValueError: If the motif is a string but not a column in the `poses.df` DataFrame.
            TypeError: If the motif is neither a `ResidueSelection` object nor a string.

        Examples:
            Here is an example of how to use the `parse_motif` method:

            .. code-block:: python

                from protflow.residues import ResidueSelection
                from protein_edits import ChainAdder
                import pandas as pd

                # Initialize the ChainAdder class
                chain_adder = ChainAdder()

                # Example pose DataFrame row
                pose = pd.Series({'motif_column': ResidueSelection(...)})

                # Parse a ResidueSelection object
                motif = ResidueSelection(...)
                motif_str = chain_adder.parse_motif(motif, pose)

                # Parse a string that is a column in the pose DataFrame
                motif_str = chain_adder.parse_motif('motif_column', pose)

                # Access the result
                print(motif_str)

        Further Details
        ---------------
        - **ResidueSelection Handling:** The method directly converts a `ResidueSelection` object to its string representation using its `to_string` method.
        - **String Handling:** If a string is provided, the method checks if it is a column in the `pose` DataFrame that points to a `ResidueSelection` object, converting it to a string.
        - **Error Handling:** The method raises appropriate errors if the input is not of the expected type or if the string does not correspond to a valid column in the DataFrame.
        """
        if isinstance(motif, ResidueSelection):
            return motif.to_string()
        if isinstance(motif, str):
            if motif in pose:
                # assumes motif is a column in pose (row in poses.df) that points to a ResidueSelection object
                return pose[motif].to_string()
            raise ValueError(f"If string is passed as motif, it has to be a column of the poses.df DataFrame. Otherwise pass a ResidueSelection object.")
        raise TypeError(f"Unsupportet parameter type for motif: {type(motif)} - Only ResidueSelection or str allowed!")

    def add_sequence(self, prefix: str, poses: Poses, seq: str = None, seq_col: str = None, sep: str = ":") -> None:
        """
        Add a sequence to the poses in .fa format.

        This method appends a specified sequence to the protein sequences in the `poses` object. The sequence can be 
        provided directly or specified through a column in the `poses` DataFrame. The updated sequences are saved 
        in .fa format in a specified directory.

        Parameters:
            prefix (str): A prefix used to name and organize the output files.
            poses (Poses): The Poses object containing the protein structures.
            seq (str, optional): The sequence to be added. If specified, `seq_col` must be None. Defaults to None.
            seq_col (str, optional): The column in the poses DataFrame that contains the sequences to be added. 
                                    If specified, `seq` must be None. Defaults to None.
            sep (str, optional): The separator to be used between the original and new sequences. Defaults to ":".

        Raises:
            ValueError: If poses are not in .fa or .fasta format, if both `seq` and `seq_col` are specified, 
                        or if neither `seq` nor `seq_col` is specified.

        Examples:
            Here is an example of how to use the `add_sequence` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protein_edits import ChainAdder

                # Create instances of necessary classes
                poses = Poses()

                # Initialize the ChainAdder class
                chain_adder = ChainAdder()

                # Add a sequence to the poses
                chain_adder.add_sequence(
                    prefix="experiment_1",
                    poses=poses,
                    seq="ATCGATCGATCG",
                    sep=":"
                )

        Further Details
        ---------------
        - **File Format:** The method checks that all poses are in .fa or .fasta format and raises an error if not.
        - **Sequence Input:** Either `seq` or `seq_col` must be specified to provide the sequence to be added. 
                            The method ensures that both are not specified simultaneously.
        - **Output Directory:** The method creates an output directory if it does not exist and saves the updated 
                                sequences in this directory.
        - **DataFrame Update:** The `poses` DataFrame is updated to reflect the new locations of the modified sequences.
        """
        poses.check_prefix(prefix)
        if not all(pose.endswith(".fa") or pose.endswith(".fasta") for pose in poses.poses_list()):
            raise ValueError(f"Poses must be .fasta files (.fa also fine)!")
        out_dir = f"{poses.work_dir}/prefix/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # prep seq input
        if seq and seq_col:
            raise ValueError(f"Either :seq: or :seq_col: can be passed to specify a sequence, but not both!")
        if seq:
            seqs = [seq for _ in poses]
        elif seq_col:
            col_in_df(poses.df, seq_col)
            seqs = poses.df[seq_col].to_list()
        else:
            raise ValueError(f"One of the parameters :seq: :seq_col: has to be passed to specify the sequence to add.")

        # separator (add sequence, or add protomer?)
        sep = "" if sep is None else sep

        # iterate over poses and add in sequence
        new_poses = []
        for pose, seq_ in zip(poses.poses_list(), seqs):
            # read fasta and add sequence.
            desc, orig_seq = list(parse_fasta_to_dict(pose).items())[0]
            orig_seq += sep + seq_

            # store at new location
            out_path = f"{out_dir}/{desc}.fa"
            with open(out_path, 'w', encoding="UTF-8") as f:
                f.write(f">{desc}\n{orig_seq}")
            new_poses.append(out_path)

        # update poses.df['poses'] to new location
        poses.change_poses_dir(out_dir, copy=False)

    def multimerize(self, prefix: str, poses: Poses, n_protomers: int, sep: str = ":") -> None:
        """
        Create multimers from the sequences in .fa files.

        This method takes .fa files from the `poses` object and creates multimers by repeating the sequence a specified number of times.
        The updated sequences are saved in .fa format in a specified directory.

        Parameters:
            prefix (str): A prefix used to name and organize the output files.
            poses (Poses): The Poses object containing the protein structures.
            n_protomers (int): The number of protomers in the final .fa file.
            sep (str, optional): The separator to be used between the original and new sequences. Defaults to ":".

        Raises:
            ValueError: If poses are not in .fa or .fasta format.

        Examples:
            Here is an example of how to use the `multimerize` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protein_edits import ChainAdder

                # Create instances of necessary classes
                poses = Poses()

                # Initialize the ChainAdder class
                chain_adder = ChainAdder()

                # Multimerize the sequences in the poses
                chain_adder.multimerize(
                    prefix="experiment_1",
                    poses=poses,
                    n_protomers=3,
                    sep=":"
                )

        Further Details
        ---------------
        - **File Format:** The method checks that all poses are in .fa or .fasta format and raises an error if not.
        - **Protomers Specification:** The `n_protomers` parameter specifies the number of times the sequence should be repeated to form a multimer.
        - **Output Directory:** The method creates an output directory if it does not exist and saves the updated sequences in this directory.
        - **DataFrame Update:** The `poses` DataFrame is updated to reflect the new locations of the modified sequences.
        """
        # setup directory and function
        poses.check_prefix(prefix)
        if not all(pose.endswith(".fa") or pose.endswith(".fasta") for pose in poses.poses_list()):
            raise ValueError(f"Poses must be .fasta files (.fa also fine)!")
        out_dir = f"{poses.work_dir}/prefix/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # iterate over poses and add in sequence
        new_poses = []
        for pose in poses.poses_list():
            # read fasta and add sequence.
            desc, orig_seq = list(parse_fasta_to_dict(pose).items())[0]
            orig_seq += f"{sep}{orig_seq}" * (n_protomers - 1)

            # store at new location
            out_path = f"{out_dir}/{desc}.fa"
            with open(out_path, 'w', encoding="UTF-8") as f:
                f.write(f">{desc}\n{orig_seq}")
            new_poses.append(out_path)

        # update poses.df['poses'] to new location
        poses.change_poses_dir(out_dir, copy=False)

def setup_chain_list(chain_arg, poses: Poses) -> list[str]:
    """
    Set up chains for add_chains_batch.py.

    This function configures the list of chains to be used in the `add_chains_batch.py` script based on the provided `chain_arg`.
    It supports specifying a single chain, a column in the `poses` DataFrame, or a list of chains.

    Parameters:
        chain_arg (str or list[str]): The chain specification. It can be a single chain identifier (e.g., 'A'), 
                                      the name of a column in the `poses` DataFrame where the chains are listed, 
                                      or a list of chain identifiers.
        poses (Poses): The Poses object containing the protein structures.

    Returns:
        list[str]: A list of chain identifiers to be used in `add_chains_batch.py`.

    Raises:
        ValueError: If the `chain_arg` value is inappropriate, such as when the specified column does not exist in the DataFrame 
                    or the length of the list does not match the number of poses.

    Examples:
        Here is an example of how to use the `setup_chain_list` function:

        .. code-block:: python

            from protflow.poses import Poses
            from protein_edits import setup_chain_list

            # Create instances of necessary classes
            poses = Poses()

            # Set up a single chain
            chain_list = setup_chain_list('A', poses)
            print(chain_list)

            # Set up chains from a column in the poses DataFrame
            chain_list = setup_chain_list('chain_col', poses)
            print(chain_list)

            # Set up chains from a list
            chain_list = setup_chain_list(['A', 'B', 'C'], poses)
            print(chain_list)

    Further Details
    ---------------
    - **Single Chain Identifier:** If a single chain identifier (e.g., 'A') is provided, it is used for all poses.
    - **DataFrame Column:** If the name of a column in the `poses` DataFrame is provided, the function extracts the chain identifiers 
                           from that column for each pose.
    - **List of Chains:** If a list of chain identifiers is provided, it must match the length of the `poses` DataFrame. 
                           The function raises an error if this condition is not met.
    """
    if isinstance(chain_arg, str):
        if len(chain_arg) == 1:
            return [chain_arg for _ in poses]
        else:
            return [pose[chain_arg] for pose in poses]
    if isinstance(chain_arg, list) and len(chain_arg) == len(poses):
        return chain_arg
    raise ValueError(f"Inappropriate value for parameter :chain_arg:. Specify the chain (e.g. 'A'), the column where the chains are listed (e.g. 'chain_col') or give a list of chains the same length as poses.df (e.g. ['A', ...])")

def parse_chain(chain, pose: pd.Series) -> str:
    '''Sets up chain for add_chains_batch.py'''
    if isinstance(chain, str):
        return chain if len(chain) == 1 else pose[chain]
    raise TypeError(f"Inappropriate parameter type for parameter :chain: {type(chain)}. Only :str: allowed!")

class ChainRemover(Runner):
    """
    ChainRemover Class
    ==================

    The `ChainRemover` class is a specialized class designed to facilitate the removal of chains from protein structures within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with chain removal processes.

    Detailed Description
    --------------------
    The `ChainRemover` class manages all aspects of removing chains from protein structures. It configures necessary scripts and executables, prepares the environment for the removal processes, and executes the required commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to chain removal scripts and Python executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of chain removal commands with support for batch processing.
        - Collecting and processing output data into a structured format.

    Returns
    -------
    An instance of the `ChainRemover` class, configured to remove chains from protein structures and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.

    Examples
    --------
    Here is an example of how to initialize and use the `ChainRemover` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from protein_edits import ChainRemover

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ChainRemover class
        chain_remover = ChainRemover(jobstarter=jobstarter)

        # Remove a chain from the poses
        removed_chains = chain_remover.remove_chains(
            poses=poses,
            prefix="experiment_2",
            chains=["A"],
            jobstarter=jobstarter,
            overwrite=True
        )

        # Access and process the results
        print(removed_chains)

    Further Details
    ---------------
        - Edge Cases: The class handles various edge cases, such as missing chain specifications and the need to overwrite previous results.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the chain removal process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ChainRemover class is intended for researchers and developers who need to remove chains from protein structures as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, default_python=os.path.join(PROTFLOW_ENV, "python3"), jobstarter: JobStarter = None):
        """
        Initialize the ChainRemover class.

        This method sets up the ChainRemover class by configuring the path to the default Python executable and 
        initializing the job starter. The ChainRemover class is used to remove chains from protein structures within 
        the ProtFlow framework.

        Parameters:
            default_python (str, optional): The path to the default Python executable. Defaults to `os.path.join(PROTFLOW_ENV, "python3")`.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job execution. Defaults to None.

        Attributes:
            python (str): Path to the Python executable used for running scripts.
            jobstarter (JobStarter): An instance of the JobStarter class to manage job execution.

        Examples:
            Here is an example of how to initialize the ChainRemover class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from protein_edits import ChainRemover

                # Initialize the ChainRemover class
                jobstarter = JobStarter()
                chain_remover = ChainRemover(jobstarter=jobstarter)

        Notes:
            The ChainRemover class depends on the ProtFlow environment being properly configured. Ensure that the 
            `PROTFLOW_ENV` and necessary scripts are correctly set up before using this class.

        Raises:
            FileNotFoundError: If the specified Python executable is not found.
        """
        self.python = self.search_path(default_python, "PROTFLOW_ENV")
        self.jobstarter = jobstarter

    def __str__(self):
        return "chain_remover"

    def _prep_chain_param(self, chain_param: str|list[str], poses: Poses) -> list[str]:
        '''Internal method to prepare chain parameter for run() function.'''
        if isinstance(chain_param, str):
            if len(chain_param) == 1:
                return [[chain_param] for _ in poses]
        elif isinstance(chain_param, list):
            return [chain_param for _ in poses]

    #################################### METHODS #######################################

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, chains: list = None, preserve_chains: list = None, overwrite: bool = False):
        """
        Remove chains from the poses.

        This method removes specified chains from the protein structures in the `poses` object. It sets up and executes the necessary scripts, 
        handles the environment configuration, and processes the output.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            chains (list, optional): A list of chains to be removed. If specified, each chain in the list will be removed from the poses. Defaults to None.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job execution. Defaults to None.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.

        Returns:
            Poses: An updated Poses object with the specified chains removed.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the methods.

        Examples:
            Here is an example of how to initialize and use the `remove_chains` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from protein_edits import ChainRemover

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the ChainRemover class
                chain_remover = ChainRemover(jobstarter=jobstarter)

                # Remove chains from the poses
                removed_chains = chain_remover.remove_chains(
                    poses=poses,
                    prefix="experiment_2",
                    chains=["A"],
                    jobstarter=jobstarter,
                    overwrite=True
                )

                # Access and process the results
                print(removed_chains)

        Further Details
        ---------------
        - **Output Checking:** The method checks if the output already exists and whether it should be overwritten, ensuring no redundant processing.
        - **Chain Setup:** Chains can be specified as a list, a column in the `poses` DataFrame, or as a single chain identifier for all poses.
        - **Batch Processing:** The method supports batch processing, splitting the inputs into sublists to optimize resource usage during execution.
        - **Path Configuration:** Ensure the paths to the scripts and executables are correctly configured as per ProtFlow setup. Using default paths is recommended unless customization is necessary.
        - **JobStarter Integration:** The JobStarter object is used to manage job execution, ensuring processes are handled efficiently. If a JobStarter is not provided, the method will operate without it, but using one is recommended for better job management.
        """
        def output_exists(work_dir: str, files_list: list[str]) -> bool:
            '''checks if output of copying chains exists'''
            return os.path.isdir(work_dir) and all(os.path.isfile(fn) for fn in files_list)

        if chains and preserve_chains:
            raise ValueError(f":chains: and :preserve_chains: are mutually exclusive!")
        if not chains and not preserve_chains:
            raise ValueError(f"Either :chains: or :preserve_chains: must be set!")

        # setup runner
        script_path = f"{AUXILIARY_RUNNER_SCRIPTS_DIR}/remove_chains_batch.py"
        work_dir, jobstarter = self.generic_run_setup(
            poses = poses,
            prefix = prefix,
            jobstarters = [jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # define location of new poses:
        poses.df[f"{prefix}_location"] = [os.path.join(work_dir, os.path.basename(pose)) for pose in poses.poses_list()]

        # check if output is present
        if output_exists(work_dir, poses.df[f"{prefix}_location"].to_list()) and not overwrite:
            return poses.change_poses_dir(work_dir, copy=False)

        # setup chains
        chain_list = self._prep_chain_param(chains or preserve_chains, poses)

        # setup preserved chains
        if preserve_chains:
            chain_list = [[chain.id for chain in load_structure_from_pdbfile(pose).get_chains() if not chain.id in pres_chains] for pose, pres_chains in zip(poses.poses_list(), chain_list)]

        # batch inputs to max_cores
        input_dict = {pose: chain for pose, chain in zip(poses.poses_list(), chain_list)}
        split_sublists = jobstarters.split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write cmds
        json_files = []
        for i, subdict in enumerate(subdicts, start=1):
            opts_json_p = f"{work_dir}/remove_chain_input_{str(i).zfill(4)}.json"
            with open(opts_json_p, 'w', encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(opts_json_p)

        # start remove_chains_batch.py
        cmds = [f"{self.python} {script_path} --input_json {json_f} --output_dir {work_dir}" for json_f in json_files]
        jobstarter.start(
            cmds = cmds,
            jobname = f"remove_chains_{prefix}",
            wait = True,
            output_path = work_dir
        )

        # reset poses location and return
        return poses.change_poses_dir(work_dir, copy=False)

class SequenceRemover(Runner):
    def __init__(self, chains: list[int] = None, sep: str = None, python: str = PROTFLOW_ENV, jobstarter: JobStarter = None):
        '''
        Parameters:
        chains: list of chain idx to remove.
        '''
        self.chains = chains
        self.sep = sep
        self.jobstarter = jobstarter
        self.python = python

    def _outputs_exist(self, poses: Poses, work_dir: str) -> bool:
        return os.path.isfile(f"{work_dir}/done.txt") and all(os.path.isfile(f"{work_dir}/chains_removed/{description}.fa" for description in poses.df["poses_description"].to_list()))

    def _write_json(self, out_dict: str, fp: str) -> None:
        with open(fp, 'w', encoding="UTF-8") as f:
            json.dump(out_dict, f)

    def _prep_chains(self, chains: list[int]|str, poses: Poses) -> None:
        if isinstance(chains, str):
            col_in_df(poses.df, chains)
            return poses.df[chains]
        if isinstance(chains, list):
            return [chains for _ in poses.poses_list()]
        raise ValueError(f"Unsupported type for paramter 'chains': {type(chains)}. Should be string pointing to column of poses.df or list of integers pointing to the sequence idx that should be removed from the .fa file. For more info, visit the documentation! Current parameter chains: {chains}")

    def _output_df(self, poses: Poses, work_dir: str) -> pd.DataFrame:
        out_df = pd.DataFrame({
            "location": [f"{work_dir}/chains_removed/{poses_description}.fa" for poses_description in list(poses.df["poses_description"])],
            "description": poses.df["poses_description"].to_list()
        })
        return out_df

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, chains: list[int] = None, sep: str = None, overwrite: bool = False) -> Poses:
        '''
        Parameters:
        chains: can either be a list that contains chain idx to drop, or a str that points to the column in poses.df that contains this list for every pose.
        '''
        # sanity
        if not all(fp.endswith(".fa") or fp.endswith(".fasta") for fp in poses.poses_list()):
            raise ValueError(f"Your poses must be .fasta or .fa files. If you would like to remove chains from .pdb files, use the ChainRemover class.")

        # prep parameters
        chains = self._prep_chains(chains or self.chains, poses)

        # setup work_dir
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check if outputs exist
        if self._outputs_exist(poses, work_dir) and not overwrite:
            out_df = self._output_df(poses, work_dir)
            return RunnerOutput(poses, out_df, prefix).return_poses()

        # write json files for jobstarters
        input_dict = dict(zip(poses.poses_list(), chains))
        split_poses = split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores) # splits list into nested sublists
        input_json_list = []
        for i, poses_l in enumerate(split_poses, start=1):
            sublist_dict = {os.path.abspath(pose): input_dict[pose] for pose in poses_l}
            fp = f"{work_dir}/sequence_remover_{str(i).zfill(4)}.json"
            self._write_json(sublist_dict, fp)
            input_json_list.append(fp)

        # write cmd
        script_path = f"{AUXILIARY_RUNNER_SCRIPTS_DIR}/remove_sequence_batch.py"
        cmds = [f"{self.python}/python3 {script_path} --input_json {input_json} --output_dir {work_dir}" for input_json in input_json_list]

        # execute with jobstarter
        jobstarter.start(cmds=cmds, jobname=prefix, output_path=work_dir)

        # integrate into poses (update DataFrame)
        output_df = self._output_df(poses, work_dir)
        return RunnerOutput(poses, output_df, prefix=prefix).return_poses()
