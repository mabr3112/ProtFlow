"""
FPocket Module
==============

This module provides the functionality to integrate FPocket within the ProtFlow framework. It offers tools to run FPocket, handle its inputs and outputs, and process the resulting data in a structured and automated manner. 

Detailed Description
--------------------
The `FPocket` class encapsulates the functionality necessary to execute FPocket runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of FPocket processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.
The module is designed to streamline the integration of FPocket into larger computational workflows. It supports the automatic setup of job parameters, execution of FPocket commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `FPocket` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the FPocket process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `FPocket` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from fpocket import FPocket

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the FPocket class
    fpocket = FPocket()

    # Run the FPocket process
    results = fpocket.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="--some-option value",
        pose_options=["--specific-option value"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the FPocket process.
    - Customizability: Users can customize the FPocket process through multiple parameters, including specific options for the FPocket script and options for handling pose-specific parameters.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate FPocket into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

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
import glob
import os
import json

# dependencies
import pandas as pd
from collections import UserDict
from numpy import array_split

# custom
from ..poses import Poses, col_in_df, description_from_path
from ..residues import ResidueSelection, AtomSelection
from ..jobstarters import JobStarter, split_list
from .. import require_config, load_config_path
from ..runners import Runner, RunnerOutput, options_flags_to_string, parse_generic_options

class HBplus(Runner):
    """
    FPocket Class
    =============

    The `FPocket` class is a specialized class designed to facilitate the execution of FPocket within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with FPocket processes.

    Detailed Description
    --------------------
    The `FPocket` class manages all aspects of running FPocket simulations. It handles the configuration of necessary scripts and executables, prepares the environment for pocket detection processes, and executes the FPocket commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to FPocket scripts and executables.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of FPocket commands with support for multiple options and pose-specific parameters.
        - Collecting and processing output data into a pandas DataFrame.
        - Ensuring robust error handling and logging for easier debugging and verification of the FPocket process.

    Returns
    -------
    An instance of the `FPocket` class, configured to run FPocket processes and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods.
        TypeError: If provided options are not of the expected type.


    Examples
    --------
    Here is an example of how to initialize and use the `FPocket` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from fpocket import FPocket

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the FPocket class
        fpocket = FPocket()

        # Run the FPocket process
        results = fpocket.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            options="--some-option value",
            pose_options=["--specific-option value"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the FPocket process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The FPocket class is intended for researchers and developers who need to perform FPocket simulations as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    # class attributes
    index_layers = 0

    def __init__(self, hbplus_path: str = None, python_path: str = None, jobstarter: JobStarter = None):
        """
        Initialize the FPocket class with the specified path and jobstarter configuration.

        This constructor sets up the FPocket instance by configuring the path to the FPocket executable and initializing the jobstarter object. It ensures that the necessary components are in place for running FPocket processes.

        Parameters:
            fpocket_path (str, optional): The path to the FPocket executable. Defaults to the path specified in the ProtFlow configuration (`FPOCKET_PATH`).
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Returns:
            An instance of the FPocket class, ready to run FPocket processes.

        Raises:
            ValueError: If the fpocket_path is not provided or is invalid.


        Examples:
            Here is an example of how to initialize the FPocket class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from fpocket import FPocket

                # Initialize the FPocket class with default settings
                fpocket = FPocket()

                # Initialize the FPocket class with a specific jobstarter
                jobstarter = JobStarter()
                fpocket = FPocket(jobstarter=jobstarter)

        Further Details:
            - **Path Configuration:** Ensures the FPocket executable path is set correctly, raising an error if the path is not provided or invalid.
            - **Job Management:** Initializes the jobstarter object to manage the execution of FPocket commands, allowing for integration with job scheduling systems.
        """
        # setup config
        config = require_config()
        self.jobstarter = jobstarter
        self.script_path = hbplus_path or load_config_path(config, "HBPLUS_PATH")
        self.python_path = python_path or os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")

    def __str__(self):
        return "hbplus"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, queries: list[HBplus_query] | HBplus_query = None, options: str | list = None, pose_options: str | list = None, overwrite: bool = False) -> Poses:
        """
        Execute the FPocket process with given poses and jobstarter configuration.

        This method sets up and runs the FPocket process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            options (str or list[str], optional): Additional options for the FPocket script. Defaults to None.
            pose_options (str or list[str], optional): A list of pose-specific options for the FPocket script. Defaults to None.
            return_full_scores (bool, optional): If True, include detailed scores for each pocket in the output. Defaults to False.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: An updated Poses object containing the processed poses and results of the FPocket process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.
            TypeError: If options or pose_options are not of the expected type.


        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from fpocket import FPocket

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the FPocket class
                fpocket = FPocket()

                # Run the FPocket process
                results = fpocket.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    options="--some-option value",
                    pose_options=["--specific-option value"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed. It moves the poses to the working directory and compiles the FPocket commands for execution.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized into a structured DataFrame. It includes the location of each pocket and integrates the results back into the Poses object.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the FPocket process to their specific needs, including the ability to specify additional FPocket options and pose-specific parameters.

        This method is designed to streamline the execution of FPocket processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze pocket detection simulations.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # prep options:
        options_l = self._prep_hbplus_options(poses, options, pose_options)

        # compile cmds
        cmds = [f"{self.script_path} {os.path.abspath(pose)} {options}" for pose, options in zip(poses.poses_list(), options_l)]

        # hbplus puts all output in current directory --> change to output dir
        os.makedirs(output_dir := os.path.join(work_dir, "output"), exist_ok=True)
        cwd = os.getcwd()
        os.chdir(output_dir)

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"hbplus_{prefix}",
            output_path = work_dir
        )

        # return to starting dir
        os.chdir(cwd)

        # collect outputs and write scorefile
        scores = collect_scores(work_dir)
        scores["location"] = [_get_hbplus_input_location(description, cmds) for description in scores["description"].to_list()]
        self.save_runner_scorefile(scores, scorefile)

        # itegrate and return
        poses = RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()

        if queries:
            self.query(poses=poses, queries=queries, hbplus_prefix=prefix)
        return poses
    
    def query(self, poses: Poses, queries: list[HBplus_query] | HBplus_query, hbplus_prefix: str, jobstarter: JobStarter = None, full_output: bool = False, overwrite: bool = False):

        score_col = f"{hbplus_prefix}_hb2_scores"
        if not score_col in poses.df.columns:
            raise KeyError(f"Could not find HBplus score column called {score_col} in poses dataframe! Did you run HBplus with the selected prefix?")
        
        if not isinstance(queries, list):
            queries = [queries]

        if not all(isinstance(query, HBplus_query) for query in queries):
            raise ValueError(":queries: must be a HBplus_query or a list of HBplus_queries!")
        
        if not len([query.name for query in queries]) == len(set([query.name for query in queries])):
            raise KeyError("Names of input queries must be unique!")

        prefix = f"{hbplus_prefix}_query"

        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
        
        query_dict = {query.name: query.data for query in queries}

        query_path = _save_dict_to_json(query_dict, os.path.join())
        # fill missing keys ikökljölkjhjlkjh

        n_batches = min([jobstarter.max_cores, len(poses.df.index)])

        subdfs = array_split(poses.df[["poses", "poses_description", f"{hbplus_prefix}_hb2_scores"]], n_batches)

        cmds = []
        output = []
        for i, subdf in enumerate(subdfs):
            json_path = os.path.join(work_dir, f"batch_{i}.json")
            output.append(out_path:= os.path.join(work_dir, f"out_{i}.json"))
            subdf.to_json(json_path)
            cmd = f"{self.python_path} {__file__} --query_path {query_path} --input_poses {json_path} --out_path {out_path}"#
            if full_output:
                cmd = cmd + " --full_output"
            cmds.append(cmd)    
            

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"hbplus_{prefix}",
            output_path = work_dir
        )

        scores = pd.concat([pd.read_json(out) for out in output])

        self.save_runner_scorefile(scores, scorefile)

        # integrate and return
        return RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()







        

    def _prep_hbplus_options(self, poses: Poses, options: str, pose_options: str|list[str]) -> list[str]:
        """
        Prepare options for the FPocket process based on given parameters.

        This method processes and prepares the options and pose-specific options for the FPocket run. It filters out forbidden options, merges general options with pose-specific options, and formats them for inclusion in the FPocket commands.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            options (str or list[str], optional): General options for the FPocket script. Defaults to None.
            pose_options (str or list[str], optional): A list of pose-specific options for the FPocket script. Defaults to None.

        Returns:
            list[str]: A list of formatted option strings for each pose, ready to be used in the FPocket commands.

        Raises:
            TypeError: If options or pose_options are not of the expected type.

        Examples:
            Here is an example of how to use the `prep_fpocket_options` method:

            .. code-block:: python

                from protflow.poses import Poses
                from fpocket import FPocket

                # Create instances of necessary classes
                poses = Poses()
                fpocket = FPocket()

                # Prepare FPocket options
                options = "--some-option value"
                pose_options = ["--specific-option value"]
                prepared_options = fpocket.prep_fpocket_options(poses, options, pose_options)

                # Output the prepared options
                print(prepared_options)

        Further Details:
            - **Option Processing:** Merges general and pose-specific options, ensuring that forbidden options are removed and the final option strings are correctly formatted.
            - **Customization:** Allows for extensive customization of the FPocket process through both general and pose-specific options, providing flexibility in configuring FPocket runs.
        """
        pose_options = self.prep_pose_options(poses, pose_options)

        # Iterate through pose options, overwrite options and remove options that are not allowed.
        options_l = []
        for pose_opt in pose_options:
            opts, flags = parse_generic_options(options, pose_opt)
            options_l.append(options_flags_to_string(opts,flags))

        # merge options and pose_options, with pose_options priority and return
        return options_l

def _detect_networks(unfiltered_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    # searches for hbond networks, only starts networks if target or partner category are sidechain; stops at mainchain (to prevent detecting networks via e.g. helices)

    # create a df with hbonds not present in the filtered subset
    excluded = pd.concat([unfiltered_df, filtered_df]).drop_duplicates("bond_num", keep=False)

    # extract all possible network starting points 
    # (not 100% required as filter for cat SH is applied later, but should speed úp filtering because of fewer input residues)
    target = []
    for cat in ["A", "D"]:
        non_bb = filtered_df[filtered_df[f"{cat}_cat"] != 'M']
        if non_bb.empty:
            # add empty selection
            target.append(AtomSelection(atoms=()))
            continue

        non_bb_atms = AtomSelection.from_list([_convert_hbplus_to_atomselection(res, atm) for res, atm in zip(non_bb[f"{cat}_res"], non_bb[f"{cat}_atom"])])
        target.append(non_bb_atms)

    # combine both selections
    target = target[0] + target[1]

    # return early if no suitable starting points were found
    if not target:
        return pd.DataFrame(columns=unfiltered_df.columns)
    
    # convert to residueselection
    target = ResidueSelection.from_atomselection(target)

    # limit cat to sidechain/hetatm to avoid bridging sidechain to mainchain (e.g. his-n - tyr-oh to tyr-n - ser-oh), which are not real networks
    network = _filter_df_by_selection(df=excluded, target=target, target_category="SH")

    if not network.empty:
        extended_nw = _detect_networks(unfiltered_df=excluded, filtered_df=network)
        network = pd.concat([network, extended_nw])
    
    return network


def collect_scores(work_dir: str) -> pd.DataFrame:
    """
    Collect scores from an FPocket output directory.

    This function collects and processes the scores from FPocket output files located in the specified directory. It aggregates the scores into a pandas DataFrame for further analysis.

    Parameters:
        output_dir (str): The path to the directory containing FPocket output files.
        return_full_scores (bool, optional): If True, include detailed scores for each pocket in the output. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the collected scores from the FPocket output files.

    Examples:
        Here is an example of how to use the `collect_fpocket_scores` function:

        .. code-block:: python

            from fpocket import collect_fpocket_scores

            # Specify the output directory
            output_dir = "path/to/output_directory"

            # Collect scores
            scores = collect_fpocket_scores(output_dir, return_full_scores=True)

            # Display the scores
            print(scores)

    Further Details:
        - **Score Aggregation:** The function looks for FPocket output directories, extracts scores from each output file, and combines them into a single DataFrame.
        - **Detailed Scores:** If the return_full_scores parameter is set to True, the function includes detailed scores for each pocket in the DataFrame.
    """
    # collect scores
    scores = glob.glob(os.path.join(work_dir, "output", "*.hb2"))

    # compile score paths into a dataframe
    out_df = pd.DataFrame({"hb2_scores": [os.path.abspath(score) for score in scores], "description": [description_from_path(score) for score in scores]})
    
    return out_df

def _filter_df_by_selection(
        df: pd.DataFrame,
        target: AtomSelection | ResidueSelection = None,
        target_type: str = None,
        target_category: str | list = None,
        partner: AtomSelection | ResidueSelection = None,
        partner_category: str | list = None) -> pd.DataFrame:
        
    # If no target is provided, match everything (.* is the regex wildcard)
    # If a selection is provided, convert it into {res: atom_regex} dict
    target_dict = _convert_selection_to_hbplus_dict(target) if target else {".*": ".*"}
    
    # If a partner is provided, convert it; otherwise, keep it as None
    partner_dict = _convert_selection_to_hbplus_dict(partner) if partner else None

    # Process categories: join lists into "cat1|cat2" regex or default to "match all"
    t_cat = '|'.join(target_category) if target_category else ".*"
    p_cat = '|'.join(partner_category) if partner_category else ".*"

    # This list will store the result of each specific residue-atom pair search
    all_target_results = []
    
    # Iterate through each Residue:Atom pair in our target selection
    for res, atm in target_dict.items():
        
        # apply filters
        f_df = _apply_filters(df=df, res=res, atm=atm, target_type=target_type, target_category=t_cat, partner_category=p_cat)
        
        # Only check for partners if a partner dict was provided and we actually found targets
        if partner_dict and not f_df.empty:
            f_partner_list = []
            
            # Loop through partner pairs to see if they are on the other side of the interaction
            for p_res, p_atm in partner_dict.items():
                
                # Collect successful partner matches
                f_partner_list.append(_apply_filters(df=f_df, res=p_res, atm=p_atm, target_type=".*", target_category=".*", partner_category=".*"))
            
            # Combine all partner matches
            f_df = pd.concat(f_partner_list)

        # Store the final result for this target residue
        all_target_results.append(f_df)
    
    # Combine all results from the various target residues into one final DataFrame
    return pd.concat(all_target_results).drop_duplicates() # duplicates can occur if hbond to another res in target is formed

def _apply_filters(df: pd.DataFrame, res:str, atm:str, target_type:str, target_category:str, partner_category:str) -> pd.DataFrame:
    # Case A: Target must specifically be the Donor
    if target_type == "donor":
        mask = (df['D_res'].str.contains(res, na=False) &
                df['D_atom'].str.contains(atm, na=False) &
                df['D_cat'].str.contains(target_category, na=False) &
                df['A_cat'].str.contains(partner_category, na=False))

    # Case B: Target must specifically be the Acceptor
    elif target_type == "acceptor":
        mask = (df['A_res'].str.contains(res, na=False) &
                df['A_atom'].str.contains(atm, na=False) &
                df['A_cat'].str.contains(target_category, na=False) &
                df['D_cat'].str.contains(partner_category, na=False))

    # Case C: Target can be either the Donor OR the Acceptor
    else:
        mask = (
            # Option 1: Target is the Donor side
            (df['D_res'].str.contains(res, na=False) &
                df['D_atom'].str.contains(atm, na=False) &
                df['D_cat'].str.contains(target_category, na=False) &
                df['A_cat'].str.contains(partner_category, na=False)) |
            # Option 2: Target is the Acceptor side
            (df['A_res'].str.contains(res, na=False) &
                df['A_atom'].str.contains(atm, na=False) &
                df['A_cat'].str.contains(target_category, na=False) &
                df['D_cat'].str.contains(partner_category, na=False))
        )
        
    # Apply the mask to create a temporary subset for this specific target pair and return
    return df[mask]

    
def _convert_hbplus_to_atomselection(resname:str, atom:str) -> AtomSelection:
    chain_resnum = resname.split("-")[0]
    chain = chain_resnum[0]
    resnum = int(chain_resnum[1:])

    return AtomSelection((chain, resnum, atom))

def _convert_selection_to_hbplus_dict(selection: AtomSelection | ResidueSelection) -> dict:
    if isinstance(selection, AtomSelection):
        selection = selection.to_rfd3_dict()
        for key in selection:
            selection[key] = "|".join(selection[key].split(",")) # join all atoms in residue
            selection[f"{selection[0]}{str(selection[1]).zfill(4)}"] = selection.pop(key) # rename residue to hbplus format
    elif isinstance(selection, ResidueSelection):
        selection = {f"{res[0]}{str(res[1]).zfill(4)}": ".*" for res in selection}
    else:
        raise KeyError("Input must be Atom or ResidueSelection!")
    
    return selection

def _get_hbplus_input_location(description: str, cmds: list[str]) -> str:
    '''Looks at a pose_description and tries to find the pose in a list of commands that was used as input to generate the description.
    This is an internal function for location mapping'''
    # first get the cmd that contains 'description'
    cmd = [cmd for cmd in cmds if f"/{description}.pdb" in cmd][0]

    # extract location of input pdb:
    return [substr for substr in cmd.split(" ") if f"/{description}.pdb" in substr][0]

def parse_hbplus(file_path) -> pd.DataFrame:
    """
    Parses an HBPLUS output file into a pandas DataFrame.
    """
    # Define descriptive column names based on HBPLUS output format
    columns = [
        "D_res", "D_atom", "A_res", "A_atom", 
        "DA_dist", "DA_cat", "res_sep", "CA_CA_dist", 
        "DHA_angle", "HA_dist", "HAAA_angle", "DAAA_angle", "bond_num"
    ]
    
    # Read the file skipping the first 8 lines of header/metadata
    df = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        skiprows=8, 
        names=columns,
        engine='python'
    )

    df["D_cat"] = [cat[0] for cat in df["DA_cat"]]
    df["A_cat"] = [cat[0] for cat in df["DA_cat"]]

    return df


HBPLUS_QUERY_STYLE = {
        "target": (AtomSelection, ResidueSelection), # search for hbonds from these selections
        "target_type": ["donor", "acceptor"], # targets must be one of donor or selector (or None, if any is ok)
        "target_category": ["M", "S", "H"], # hbond must be provided by target main chain, side chain or heteroatom (for ligands)
        "partner": (AtomSelection, ResidueSelection), # search for hbonds from any target to these partners
        "parter_category": ["M", "S", "H"], # search for hbonds from target to main chain, side chain or heteroatoms
        # partner_type is redundant as it is already defined via target_type
    }
                                                                                                                                                                                                                                                                                                                                                                          
class HBplus_query(UserDict):
    
    def __init__(self, name: str, poses: Poses, query: AtomSelection | ResidueSelection | dict = None, per_pose_queries: list | str = None):
        super().__init__()
        self.poses = poses
        self.name = name
        self.reset()
        if query:
            self.add_query(query)
        if per_pose_queries:
            self.add_per_pose_query(per_pose_queries)

    def add_query(self, query: AtomSelection | ResidueSelection | dict | list):
        query = self._parse_query(query)
        for pose in self.data:
            self.data[pose].extend(query)

    def add_per_pose_query(self, queries: list | str):
        if isinstance(queries, list) and not len(queries) == len(self.poses.df):
            raise ValueError("Number of input queries must match number of poses!")
        if isinstance(queries, str):
            col_in_df(self.poses.df, queries)
            queries = self.poses.df[queries].to_list()
        for query, pose in zip(queries, self.data.keys()):
            query = self._parse_query(query)
            self.data[pose].extend(query)
    
    def reset(self):
        query_dict = {}
        for pose in self.poses.df["poses_description"]:
            query_dict[pose] = []

        self.data = query_dict
        return self

    def _parse_query(self, query: AtomSelection | ResidueSelection | dict | list):
        if isinstance(query, ResidueSelection | AtomSelection):
            query = {"target": query}
        self._check_query(query)
        if not isinstance(query, list):
            query = [query]
        return query
    
    def _check_query(self, query: AtomSelection | ResidueSelection | dict | list):
        
        if isinstance(query, AtomSelection | ResidueSelection):
            return
        
        if isinstance(query, dict):
            if not all(key in HBPLUS_QUERY_STYLE for key in query.keys()):
                raise KeyError(f"Only these keys are allowed: {HBPLUS_QUERY_STYLE}")
            
            if "target" in query and not isinstance(query["target"], HBPLUS_QUERY_STYLE["target"]):
                raise ValueError("target must be of type AtomSelection or ResidueSelection")

            #if "separation" in query and not isinstance(query["separation"], HBPLUS_QUERY_STYLE["separation"]):
            #    raise ValueError("separation must be of type int or list")

            for key in ["target", "partner"]:
                if key in query and not isinstance(query[key], HBPLUS_QUERY_STYLE[key]):
                    raise ValueError(f"{key} must be of type AtomSelection or ResidueSelection")
                
            for key in ["target_category", "partner_category"]:
                if key in query and not all(cat in HBPLUS_QUERY_STYLE[key] for cat in query[key]):
                    raise ValueError(f"{key} must be one (or more) of {HBPLUS_QUERY_STYLE[key]}")
            
            for key in ["partner", "target_type"]:
                if key in query and not "target" in query:
                    raise KeyError(f"target is mandatory if setting {key}!")
                
            if "target_type" in query and not query["target_type"] in HBPLUS_QUERY_STYLE["target_type"]:
                raise KeyError(f"target_type must be one of {HBPLUS_QUERY_STYLE["target_type"]}!")
            
        if isinstance(query, list) and all(isinstance(item, AtomSelection | ResidueSelection | dict) for item in query):
            (self._check_query(subquery) for subquery in query)

        else:
            raise ValueError(f"Input must be of type AtomSelection, ResidueSelection, dict, or a list containing these, not {type(query)}!")

def _save_dict_to_json(data, filename):
    """
    Saves a Python dictionary to a JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return filename

def _load_dict_from_json(filename) -> dict:
    """
    Reads a JSON file and returns it as a Python dictionary.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def _atomselection_from_df(df: pd.DataFrame, res_col: str, atom_col: str) -> AtomSelection:
    atms = []

    for _, row in df.iterrows():
        atms.append(AtomSelection(_convert_hbplus_to_atomselection(row[res_col], row[atom_col])))

    return AtomSelection(atms)

def _query_hbplus(path: str, query: dict, full_output: bool = False) -> dict:

    df = parse_hbplus(path)
    results = {}

    f_df = _filter_df_by_selection(df, **query)

    if query["target"]:
        networks = _detect_networks(df, f_df)
        if not networks.empty:
            # add starting points to networks
            networks = pd.concat([f_df, networks]).drop_duplicates()
            networks.reset_index(drop=True, inplace=True)
            results["network_num_hbonds"] = len(networks.index)
            results["network_donor_hbonded_atoms"] = _atomselection_from_df(networks, "D_res", "D_atom")
            results["network_acceptor_hbonded_atoms"] = _atomselection_from_df(networks, "A_res", "A_atom")
            # extract all residues in network that interact via sidechain residues
            sc_donor = _apply_filters(df=networks, res=".*", atm=".*", target_type="donor", target_category="S", partner_category=".*")
            sc_acceptor = _apply_filters(df=networks, res=".*", atm=".*", target_type="acceptor", target_category="S", partner_category=".*")
            # create resisdueselection
            results["network_sc_hbond_residues"] = ResidueSelection.from_atomselection(_atomselection_from_df(sc_donor, "D_res", "D_atom") + _atomselection_from_df(sc_acceptor, "A_res", "A_atom"))
            
            # extract all residues in network that interact via heteroatoms (waters, ligands)
            sc_donor = _apply_filters(df=networks, res=".*", atm=".*", target_type="donor", target_category="H", partner_category=".*")
            sc_acceptor = _apply_filters(df=networks, res=".*", atm=".*", target_type="acceptor", target_category="H", partner_category=".*")
            # create resisdueselection
            results["network_het_hbond_residues"] = ResidueSelection.from_atomselection(_atomselection_from_df(sc_donor, "D_res", "D_atom") + _atomselection_from_df(sc_acceptor, "A_res", "A_atom"))

            if full_output:
                results["network_full_output"] = networks.to_dict("index")
    
    results["query_num_hbonds"] = len(f_df.index) # total number of hbonds according to query criteria

    results["query_donor_hbonded_atoms"] = _atomselection_from_df(f_df, "D_res", "D_atom")
    results["query_acceptor_hbonded_atoms"] = _atomselection_from_df(f_df, "A_res", "A_atom")

    if full_output:
        f_df.reset_index(drop=True, inplace=True)
        results["query_full_output"] = f_df.to_dict("index")

    return results

def main(args):

    queries = _load_dict_from_json(args.query_path)

    poses = pd.read_json(args.input_poses)

    score_path_col = [col for col in poses.columns if col.endswith("_hb2_scores")][0]

    results = {}
    for name, query in queries.items():
        results[name] = []
        for _, row in poses.iterrows():
            pose_query = query[row["poses_description"]]

            for key in HBPLUS_QUERY_STYLE:
                pose_query.setdefault(key, None)

            pose_results = _query_hbplus(row[score_path_col], pose_query, args.full_output)

            # add name prefix
            renamed = {}
            for key, value in pose_results.items():
                renamed[f"{name}_{key}"] = value
            
            renamed["poses_description"] = row["poses_description"]

            results[name].append(row)

        results[name] = pd.DataFrame(results[name])
    
    for _, res in results:
        poses = poses.merge(res, on="poses_description")

    poses.drop([score_path_col], inplace=True, axis=1)
    poses.rename({"poses_description": "description", "poses": "location"})
    
    poses.to_json(args.out_path)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--query_path", type=str, required=True, help="path to input query json")
    argparser.add_argument("--input_poses", type=str, required=True, help="path to input poses json")
    argparser.add_argument("--out_path", type=str, required=True, help="output filename")
    argparser.add_argument("--full_output", action='store_true', help="include full data in output.")

    arguments = argparser.parse_args()
    main(arguments)
