"""
DSSP Module
==============

This module provides functionality to integrate DSSP calculations within the ProtFlow framework. It offers tools to run DSSP, handle the outputs, and process the resulting data in a structured and automated manner.

Classes
-------
DSSP
    A class designed to facilitate the execution of DSSP within the ProtFlow framework. It extends the `Runner` class and manages the setup, execution, and data collection of DSSP processes.

Functions
---------
collect_scores(output_dir: str) -> pd.DataFrame
    Collects and aggregates the DSSP score outputs from a specified directory, returning a pandas DataFrame with the structured results.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package.

Author
------
Markus Braun, Sigrid Kaltenbrunner, Adrian Tripp

Version
-------
0.1.0
"""
# import general
import os
import glob

# import dependencies
import pandas as pd

# import customs
from .. import jobstarters, require_config, load_config_path
from ..poses import Poses
from ..poses import description_from_path
from ..runners import Runner, RunnerOutput
from ..jobstarters import JobStarter

class DSSP(Runner):
    """
    DSSP Class
    ====

    The `DSSP` class is a specialized class within the ProtFlow framework, designed to facilitate the execution of DSSP (Define Secondary Structure of Proteins) calculations. This class extends the `Runner` class, inheriting its core functionality while adding specific methods to handle DSSP-related tasks.

    Attributes
    ----------
    jobstarter : JobStarter, optional
        A `JobStarter` instance to manage job execution. If not provided, the default job starter from `Poses` will be used.
    name : str
        The name of the DSSP runner, set to "dssp.py".
    index_layers : int
        Tracks the indexing of layers, default is 0.
    application : str
        Path to the DSSP executable. Automatically checked for validity during initialization.

    Methods
    -------
    __str__()
        Returns a string representation of the DSSP instance.
    _check_install(application_path: str) -> str
        Verifies the installation of DSSP by checking if the executable exists at the provided path.
    run(poses: Poses, prefix: str, overwrite: bool = False, jobstarter: JobStarter = None) -> Poses
        Executes the DSSP calculations on the provided poses, managing input and output files.
    write_cmd(pose_path: str, output_dir: str) -> str
        Constructs the command line instruction to run DSSP on a given pose.

    Detailed Description
    --------------------
    The `DSSP` class manages all aspects of running DSSP calculations, from setting up the environment and executing the alignment commands to collecting and processing the resulting data. The processed output is organized into a structured format for further analysis, ensuring seamless integration with the rest of the ProtFlow framework.
    """

    def __init__(self, jobstarter: JobStarter = None, application: str|None = None):
        # setup config
        self.application = self._check_install(application or load_config_path(require_config(), "DSSP_PATH"))

        # setup runner
        self.jobstarter = jobstarter
        self.name = "dssp.py"
        self.index_layers = 0

    def __str__(self):
        return "DSSP"

    def _check_install(self, application_path) -> str:
        '''checks if DSSP is installed in the environment'''
        if not os.path.isfile(application_path):
            raise ValueError(
                f"Could not find executable for DSSP at {application_path}. Did you set it up in your protflow environment? If not, either install it in your protflow env with 'conda install -c bioconda tmalign' or in any other environment and provide the path to the application with the :application: parameter when initializing a DSSP() runner instance.")
        return application_path

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, overwrite: bool = False,
            jobstarter: JobStarter = None) -> Poses:  # pylint: disable=W0237
        """
        Execute the DSSP calculation on the given protein poses.

        This method manages the entire DSSP execution process, including setup, command execution, 
        and result collection. It prepares the working directory, generates the necessary commands 
        to run DSSP, and organizes the output into a standardized format.

        Parameters
        ----------
        poses : Poses
            A `Poses` object containing the protein structures to be analyzed.
        prefix : str
            A prefix for naming the output files generated during the DSSP run.
        overwrite : bool, optional
            If True, existing output files will be overwritten. Default is False.
        jobstarter : JobStarter, optional
            A `JobStarter` instance for managing parallel execution of commands. If not provided, 
            the default job starter from `Poses` will be used.

        Returns
        -------
        Poses
            A `Poses` object containing the results of the DSSP calculation.
        """
        # setup runner and files
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        scorefile = os.path.join(work_dir, f"{prefix}_DSSP.{poses.storage_format}")

        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
            return output

        cmds = []
        for pose in poses.poses_list():
            pose = self.add_HEADER(pose_path=pose, output_dir=work_dir)
            cmds.append(self.write_cmd(pose_path=pose, output_dir=work_dir))

        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)

        # create batch commands
        cmd_sublists = jobstarters.split_list(cmds, n_sublists=num_cmds)
        cmds = []
        for sublist in cmd_sublists:
            cmds.append("; ".join(sublist))

        # run command
        jobstarter.start(
            cmds=cmds,
            jobname="DSSP",
            output_path=work_dir
        )

        scores = collect_scores(output_dir=work_dir)
        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description",
                              right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
        return output

    def add_HEADER(self, pose_path: str, output_dir: str) -> str:
        """
        Adds a HEADER line to any PDB files missing it.

        Parameters
        ----------
        pose_path : str
            Path to input pose.
        output_dir : str
            Directory for writing output

        Returns
        -------
        str
            Path to output pose.
        """
        if pose_path.endswith("cif"):
            return pose_path
        elif not pose_path.endswith(".pdb"):
            raise RuntimeError(f"Input must be pdb or cif file, but is {os.path.splitext(pose_path)[1]}!")

        with open(pose_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Flags
        if lines[0].startswith("HEADER"):
            return pose_path
        
        # A minimal HEADER: class/keywords/date/idcode are optional placeholders
        header = ["HEADER    DUMMY STRUCTURE                            01-JAN-00   DUM000\n"]

        os.makedirs(input_dir := os.path.join(output_dir, "input_pdbs"), exist_ok=True)

        with open(pose_path := os.path.join(input_dir, os.path.basename(pose_path)), "w", encoding="utf-8") as f:
            f.writelines(header + lines)
        
        return pose_path

    def write_cmd(self, pose_path: str, output_dir: str) -> str:
        """
        Generate the command line string to run DSSP on a specific pose.

        This method constructs the command that will be executed in the shell to perform the DSSP 
        calculation on the given protein pose. The output will be saved to the specified directory.

        Parameters
        ----------
        pose_path : str
            The file path to the protein structure (pose) to be analyzed by DSSP.
        output_dir : str
            The directory where the DSSP output file will be saved.

        Returns
        -------
        str
            The command string that can be executed in a shell to run DSSP on the specified pose.
        """

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.dsspout")

        # compile command
        run_string = f"{self.application} {pose_path} --output-format dssp > {scorefile}"

        return run_string

def collect_scores(output_dir: str) -> pd.DataFrame:
    """
    Collect and compile DSSP scores from output files in the specified directory.

    This function searches the specified directory for DSSP output files, extracts the 
    secondary structure information, and compiles it into a DataFrame. Each row in the 
    DataFrame corresponds to the results for a single protein pose.

    Parameters
    ----------
    output_dir : str
        The directory containing DSSP output files (`*.dsspout`).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the DSSP scores, with each row corresponding to a protein pose. 
        Columns include secondary structure content (e.g., alpha-helix, beta-sheet) and additional 
        metadata such as the number of amino acids.
    """


    def extract_scores(score_path: str) -> pd.Series:
        """
        Extract secondary structure scores from a DSSP output file.

        This function reads a DSSP output file and extracts the counts of each secondary 
        structure element (e.g., alpha-helix, beta-sheet). These counts are converted into 
        relative content and returned as a Pandas Series.

        Parameters
        ----------
        score_path : str
            The file path to the DSSP output file to be processed.

        Returns
        -------
        pd.Series
            A Series containing the relative content of each secondary structure element and 
            the total number of amino acids.
        """
        # initialize empty lists in which the secondary structure elements are saved in so that the relative
        # content can be determined
        sec_struct_dict = {
            "H": 0,
            "B": 0,
            "E": 0,
            "G": 0,
            "I": 0,
            "T": 0,
            "S": 0,
            "P": 0,
            "L": 0
        }

        # open the DSSP output file, which can be found in score_path, and read in the structure information
        with open(score_path, 'r', encoding="UTF-8") as f:
            start_data_collection = False
            for line in f:
                split = line.split()
                if split[0] == "#":  # the header line of the output data is started with a "#", so once 
                    # this is reached data collection can start in the next iteration
                    start_data_collection = True
                    continue
                if start_data_collection:
                    sec_structure = line[16]  # in this column is the structure information i.e. E, T, etc.
                    if sec_structure in sec_struct_dict:
                        sec_struct_dict[sec_structure] += 1
                    elif sec_structure == " ":
                        sec_struct_dict["L"] += 1
                    else:
                        raise TypeError(
                            f"There seems to be another secondary structure element '{sec_structure}' in the DSSP output!"
                            f" Supported secondary structure elements are H = α-helix, B = residue in isolated β-bridge, "
                            f"E = extended strand, participates in β ladder, G = 3-helix (310 helix), I = 5 helix (π-helix), "
                            f"T = hydrogen bonded turn, S = bend and ' ' = Loops")

        num_aa = sum(sec_struct_dict.values())
        results = {f"{key}_content": value / num_aa for key, value in sec_struct_dict.items()}
        results["num_aa"] = num_aa

        results["description"] = description_from_path(score_path)

        return pd.Series(results)


    # collect scorefiles
    scorefiles = glob.glob(os.path.join(output_dir, "*.dsspout"))

    scores = [extract_scores(file) for file in scorefiles]
    scores = pd.DataFrame(scores).reset_index(drop=True)

    return scores
