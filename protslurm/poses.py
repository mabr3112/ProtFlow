"""
The 'poses' module in the ProtSLURM package is designed for running protein design tools, managing and manipulating protein data. 

It primarily focuses on handling proteins and their associated data represented as Pandas DataFrames. 
This module provides functionalities to parse, store, and manipulate protein data in various file formats, aiding in the management of complex protein study workflows. 

Key Features:
- Parsing protein data from different sources and formats.
- Storing and retrieving protein data in multiple file formats like JSON, CSV, Pickle, Feather, and Parquet.
- Integration with the ProtSLURM package for managing SLURM jobs, facilitating the handling of protein data in distributed computing environments.
- Advanced data manipulation capabilities, including merging and prefixing data from various sources.

Classes:
- Poses: A central class for storing and handling protein data frames. It supports various operations like setting up work directories, parsing protein data, and integrating outputs from different runners.

Dependencies:
- pandas: Used for DataFrame operations.
- protslurm: Required for job management and integrating with SLURM job runners.

Note:
This module is part of the ProtSLURM package and is designed to work in tandem with other components of the package, especially those related to job management in SLURM environments.

Example Usage:
To use the Poses class for managing protein data:
    from poses import Poses
    poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')
    # Further operations using poses_instance

Author: Markus Braun
Version: 0.1.0
"""
import os
from glob import glob
from typing import Union
import shutil
import logging

# dependencies
import pandas as pd

# customs
from protslurm import jobstarters
from protslurm.runners.runners import Runner, RunnerOutput
from protslurm.jobstarters import JobStarter

FORMAT_STORAGE_DICT = {
    "json": "to_json",
    "csv": "to_csv",
    "pickle": "to_pickle",
    "feather": "to_feather",
    "parquet": "to_parquet"
}

class Poses:
    '''Class that stores and handles protein df. '''
    ############################################# SETUP METHODS #########################################
    def __init__(self, poses:list=None, work_dir:str=None, storage_format:str="json", glob_suffix:str=None, jobstarter:JobStarter=jobstarters.SbatchArrayJobstarter()):
        # setup
        self.set_poses(poses, glob_suffix=glob_suffix)
        if work_dir:
            self.set_work_dir(work_dir)

        # setup scorefile for storage
        self.storage_format = storage_format
        scorefile_path = f"{work_dir}/{work_dir.strip('/').rsplit('/', maxsplit=1)[-1]}t" if work_dir else "./poses"
        self.scorefile = f"{scorefile_path}_scores.{self.storage_format}"

        # setup jobstarter
        self.default_jobstarter = jobstarter

    def set_work_dir(self, work_dir:str) -> None:
        '''sets up working_directory for poses. Just creates new work_dir and stores the first instance of Poses DataFrame in there.'''
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            logging.info(f"Creating directory {work_dir}")
        self.dir = work_dir
        return None

    def parse_poses(self, poses:Union[list,str]=None, glob_suffix:str=None) -> list:
        """
        Parses the input 'poses' which can be a directory path, a file path, or a list of file paths.
        
        If 'poses' is a directory path and 'glob_suffix' is provided, it will return a list of file paths
        matching the glob pattern. If 'poses' is a single file path, it returns a list containing just that file path.
        If 'poses' is a list of file paths, it verifies that each file exists and returns the list.

        Parameters:
        - poses (Union[List[str], str, None]): Input poses which could be a list of file paths, a single file path, or None.
        - glob_suffix (str): Optional suffix for glob pattern matching if 'poses' is a directory path.

        Returns:
        - List[str]: A list of file paths.

        Raises:
        - FileNotFoundError: If any of the files or patterns provided do not exist.
        - TypeError: If 'poses' is neither a list, a string, nor None.

        Example usage:
        - parse_poses('/path/to/directory', '*.pdb')  # Returns all '.pdb' files in the directory.
        - parse_poses('/path/to/file.pdb')            # Returns ['/path/to/file.pdb'] if file exists.
        - parse_poses(['/path/to/file1.pdb', '/path/to/file2.pdb'])  # Returns the list if all files exist.
        """
        if isinstance(poses, str) and glob_suffix:
            parsed_poses = glob(f"{poses}/{glob_suffix}")
            if not parsed_poses: raise FileNotFoundError(f"No {glob_suffix} files were found in {poses}. Did you mean to glob? Was the path correct?")
            return parsed_poses
        elif isinstance(poses, str) and not glob_suffix:
            if not os.path.isfile(poses): raise FileNotFoundError(f"File {poses} not found!")
            return [poses]
        elif isinstance(poses, list):
            if not all([os.path.isfile(path) for path in poses]): raise FileNotFoundError(f"Not all files listed in poses were found.")
            return poses
        elif poses is None:
            return []
        else:
            raise TypeError(f"Unrecognized input type {type(poses)} for function parse_poses(). Allowed types: [list, str]")

    def parse_descriptions(self, poses:list=None) -> list:
        '''parses descriptions (names) of poses from a list of pose_paths. Works on already parsed poses'''
        return [pose.strip("/").rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]for pose in poses]

    def set_poses(self, poses:list=None, glob_suffix:str=None) -> None:
        '''Sets up poses from either a list, or a string.'''
        poses = self.parse_poses(poses, glob_suffix=glob_suffix)
        self.df = pd.DataFrame({"input_poses": poses, "poses": poses, "poses_description": self.parse_descriptions(poses)})

    def check_prefix(self, prefix:str) -> None:
        '''checks if prefix is available in poses.df'''
        if prefix in self.df.columns: raise KeyError(f"Prefix {prefix} is already taken in poses.df")

    ############################################ Input Output Methods ######################################
    def save_scores(self, out_path:str=None, out_format=None) -> None:
        '''Saves Poses DataFrame as scorefile.'''
        # setup defaults
        out_path = out_path or self.scorefile
        out_format = out_format or self.storage_format

        # make sure the filename conforms to format
        if not out_path.endswith(f".{out_format}"):
            out_path += f".{out_format}"

        if (save_method_name := FORMAT_STORAGE_DICT.get(out_format.lower())):
            getattr(self.df, save_method_name)(out_path)

    def save_poses(self, out_path:str, poses_col:str="poses_description", overwrite=True) -> None:
        '''Saves current "poses" from poses.df at out_path. Overwrites poses by default.'''
        poses = self.df[poses_col].to_list()
        if not os.path.isdir(out_path): os.makedirs(out_path, exist_ok=True)

        # check if poses are already at out_path, skip if overwrite is set to False
        if all([os.path.isfile(pose) for pose in poses]) and not overwrite:
            return

        # save poses
        logging.info(f"Storing poses at {out_path}")
        for pose in poses:
            shutil.copy(pose, f"{out_path}/{pose.rsplit('/', maxsplit=1)[-1]}")


    ########################################### Runners ##################################################

    def run(self, runner: Runner, prefix:str, options:str=None, pose_options:list=None, jobstarter:JobStarter=None, max_cores:int=10) -> None:
        '''
        Method that runs runners from runners.py
        
        Runners can be any arbitrary scripts. They must send back pandas DataFrames.
        '''
        #check for column <prefix>_description in self.df
        self.check_prefix(prefix=f"{prefix}_description")

        # safety
        if not self.dir: raise AttributeError(f"Attribute 'dir' is not set. Poses.run() requires a working directory. Run Poses.set_work_dir('/path/to/work_dir/')")
        output_dir = f"{self.dir}/{prefix}"

        # start runner
        logging.info(f"Starting Runner {Runner} on {len(self.df)} poses.")
        jobstarter = jobstarter or self.default_jobstarter
        jobstarter.set_max_cores(max_cores)
        runner_out = runner.run(poses=self, prefix=prefix, jobstarter=jobstarter, output_dir=output_dir, options=options, pose_options=pose_options)

        # merge RunnerOutput into Poses
        if isinstance(runner_out, RunnerOutput):
            self.add_runner_output(runner_output=runner_out, prefix=prefix, remove_index_layers=runner.index_layers)
        else:
            raise ValueError(f"Output of runner {runner} (type: {type(runner_out)} is not of type RunnerOutput. Invalid runner!")

    def add_runner_output(self, runner_output:RunnerOutput, prefix:str, remove_index_layers:int, sep:str="_") -> None:
        '''Adds Output of a Runner class formatted in RunnerOutput into Poses.df'''    
        startlen = len(runner_output.df)

        # add prefix before merging
        runner_output.df = runner_output.df.add_prefix(prefix + "_")
        
        # Remove layers if option is set
        if remove_index_layers: runner_output.df["select_col"] = runner_output.df[f"{prefix}_description"].str.split(sep).str[:-1*remove_index_layers].str.join(sep)
        else: runner_output.df["select_col"] = runner_output.df[f"{prefix}_description"]
        # merge DataFrames
        if any(x in list(self.df.columns) for x in list(runner_output.df.columns)): logging.info(f"WARNING: Merging DataFrames that contain column duplicates. Column duplicates will be renamed!")
        self.df = runner_output.df.merge(self.df, left_on="select_col", right_on="poses_description") # pylint: disable=W0201
        self.df.drop(columns="select_col", inplace=True)
        self.df.reset_index(inplace=True)

        # check if merger was successful:
        if len(self.df) == 0: raise ValueError(f"Merging DataFrames failed. This means there was no overlap found between self.df['poses_description'] and runner_output.df[new_df_col]")
        if len(self.df) < startlen: raise ValueError(f"Merging DataFrames failed. Some rows in runner_output.df[new_df_col] were not found in self.df['poses_description']")

        # reset poses and poses_description column
        self.df["poses"] = self.df[f"{prefix}_location"]
        self.df["poses_description"] = self.df[f"{prefix}_description"]

    ########################################## Operations ###############################################
