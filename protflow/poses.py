"""
The 'poses' module in the ProtFlow package is designed for running protein design tools, managing and manipulating protein data. 

It primarily focuses on handling proteins and their associated data represented as Pandas DataFrames. 
This module provides functionalities to parse, store, and manipulate protein data in various file formats, aiding in the management of complex protein study workflows. 

Key Features:
- Parsing protein data from different sources and formats.
- Storing and retrieving protein data in multiple file formats like JSON, CSV, Pickle, Feather, and Parquet.
- Integration with the ProtFlow package for managing SLURM jobs, facilitating the handling of protein data in distributed computing environments.
- Advanced data manipulation capabilities, including merging and prefixing data from various sources.

Classes:
- Poses: A central class for storing and handling protein data frames. It supports various operations like setting up work directories, parsing protein data, and integrating outputs from different runners.

Dependencies:
- pandas: Used for DataFrame operations.
- protflow: Required for job management and integrating with SLURM job runners.

Note:
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in SLURM environments.

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
import re
from typing import Union
import shutil
import logging

# dependencies
import pandas as pd
import Bio.PDB

# customs
from protflow import jobstarters
from protflow.jobstarters import JobStarter
from protflow.residues import ResidueSelection
from protflow.utils.utils import parse_fasta_to_dict
from protflow.utils.biopython_tools import load_structure_from_pdbfile, get_sequence_from_pose
import protflow.utils.plotting as plots

FORMAT_STORAGE_DICT = {
    "json": "to_json",
    "csv": "to_csv",
    "pickle": "to_pickle",
    "feather": "to_feather",
    "parquet": "to_parquet"
}

class Poses:
    '''Class that stores and handles protein df. '''
    ############################################# SETUP #########################################
    def __init__(self, poses: list = None, work_dir: str = None, storage_format: str = "json", glob_suffix: str = None, jobstarter: JobStarter = jobstarters.SbatchArrayJobstarter()):
        # setup attributes poses.df, poses.work_dir;
        self.df = None
        self.set_work_dir(work_dir, set_scorefile=False)
        self.set_poses(poses, glob_suffix=glob_suffix)

        # setup poses.storage_format and poses.scorefile.
        self.set_storage_format(storage_format)
        self.set_scorefile(self.work_dir)

        # setup jobstarter
        self.default_jobstarter = jobstarter

        # set other empty attributes
        self.motifs = []

    def __iter__(self):
        for _, row in self.df.iterrows():
            yield row

    def __len__(self):
        return len(self.df)

    ############################################# SETUP METHODS #########################################
    def set_scorefile(self, work_dir: str) -> None:
        '''Sets scorefile attribute for poses class. This is the path to which the poses.df will be stored.'''
        # if no work_dir is set, store scores in current directory.
        scorefile_path = os.path.join(work_dir, os.path.basename(work_dir)) if work_dir else "./poses"
        self.scorefile = f"{scorefile_path}_scores.{self.storage_format}"

    def set_storage_format(self, storage_format: str) -> None:
        '''Sets storage format for poses.df. poses.df is stored in specified format with poses.save_scores() method.'''
        if storage_format.lower() not in FORMAT_STORAGE_DICT:
            raise KeyError(f"Format {storage_format} not available. Format must be on of {[list(FORMAT_STORAGE_DICT)]}")
        self.storage_format = storage_format # removed .lower() maybe there is a storage format that needs caps letters.

    def set_work_dir(self, work_dir: str, set_scorefile: bool = True) -> None:
        '''sets up working_directory for poses. Just creates new work_dir and stores the first instance of Poses DataFrame in there.'''
        def set_dir(dir_name: str, work_dir: str) -> str:
            '''Creates a directory inside of work_dir that has the name {dir_name}_dir. 
            Also sets an attribute self.{dir_name} that points to the directory.'''
            if work_dir is None:
                return None
            dir_ = os.path.join(work_dir, dir_name)
            os.makedirs(dir_, exist_ok=True)
            return dir_

        # setup and create work_dir if it does not already exist
        if work_dir is not None and not os.path.isdir(work_dir):
            work_dir = os.path.abspath(work_dir)
            os.makedirs(work_dir, exist_ok=True)
            logging.info(f"Creating directory {os.path.abspath(work_dir)}")
        self.work_dir = work_dir

        # setup common directories for workflows:
        self.scores_dir = set_dir("scores", work_dir)
        self.filter_dir = set_dir("filter", work_dir)
        self.plots_dir = set_dir("plots", work_dir)

        # setup scorefile if option is provided (default: True)
        if set_scorefile:
            self.set_scorefile(work_dir)

    def set_jobstarter(self, jobstarter: JobStarter):
        '''Sets JobStarter attribute for Poses class.'''
        self.default_jobstarter = jobstarter

    def change_poses_dir(self, poses_dir: str, copy: bool = False, overwrite: bool = False) -> "Poses":
        '''Changes the location of current poses. (works only if name of poses did not change!!!)'''
        # define new poses:
        new_poses = [os.path.join(poses_dir, os.path.basename(pose)) for pose in self.poses_list()]

        # exchange with check if work_dir is a directory and the poses exist
        if not copy:
            # just change the name of the directory in the poses_df, don't copy the poses anywhere
            if not os.path.isdir(poses_dir):
                raise ValueError(f":work_dir: has to be existing directory!")
            if not all((os.path.isfile(pose) for pose in new_poses)):
                raise ValueError(f"Poses do not exist at specified directory. If you want to copy the poses there, set the parameter :copy: to True!")

        else:
            # actually copy the poses to a new directory (for whatever reason)
            if not os.path.isdir(poses_dir):
                os.makedirs(poses_dir)
            if overwrite:
                for old_path, new_path in zip(self.poses_list(), new_poses):
                    shutil.copy(old_path, new_path)
            else:
                # if overwrite is False, check if the file exists first. This should save read/write speed.
                for old_path, new_path in zip(self.poses_list(), new_poses):
                    if not os.path.isfile(new_path):
                        shutil.copy(old_path, new_path)

        # change path in self.df["poses"] column
        self.df["poses"] = new_poses
        return self

    def parse_poses(self, poses: Union[list,str] = None, glob_suffix: str = None) -> list:
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
            if not parsed_poses:
                raise FileNotFoundError(f"No {glob_suffix} files were found in {poses}. Did you mean to glob? Was the path correct?")
            return parsed_poses
        if isinstance(poses, str) and not glob_suffix:
            if not os.path.isfile(poses):
                raise FileNotFoundError(f"File {poses} not found!")
            return [poses]
        if isinstance(poses, list):
            if not all((os.path.isfile(path) for path in poses)):
                raise FileNotFoundError(f"Not all files listed in poses were found.")
            return poses
        if poses is None:
            return []
        raise TypeError(f"Unrecognized input type {type(poses)} for function parse_poses(). Allowed types: [list, str]")

    def parse_descriptions(self, poses: list = None) -> list:
        '''parses descriptions (names) of poses from a list of pose_paths. Works on already parsed poses'''
        return [pose.strip("/").rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]for pose in poses]

    def set_poses(self, poses: list = None, glob_suffix: str = None) -> None:
        '''Sets up poses from either a list, or a string.'''
        # if DataFrame is passed, load directly.
        if isinstance(poses, pd.DataFrame):
            self.df = self.check_poses_df_integrity(poses)
            return None

        if isinstance(poses, str) and any([poses.endswith(ext) for ext in ['csv', 'json', 'parquet', 'pickle', 'feather']]):
            self.df = get_format(poses)(poses)
            self.df = self.check_poses_df_integrity(self.df)
            return None

        # if Poses are initialized freshly (with input poses as strings:)
        poses = self.parse_poses(poses, glob_suffix=glob_suffix)

        # handle multiline .fa inputs for poses!
        for pose in poses:
            if not pose.endswith(".fa") and not pose.endswith(".fasta"):
                continue
            if len(parse_fasta_to_dict(pose)) > 1:
                poses.remove(pose)
                poses += self.split_multiline_fasta(pose)

        self.df = pd.DataFrame({"input_poses": poses, "poses": poses, "poses_description": self.parse_descriptions(poses)})
        return None

    def check_prefix(self, prefix: str) -> None:
        '''checks if prefix is available in poses.df'''
        if f"{prefix}_location" in self.df.columns or f"{prefix}_description" in self.df.columns:
            raise KeyError(f"Prefix {prefix} is already taken in poses.df")

    def check_poses_df_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        '''checks if mandatory columns are in poses.df'''
        cols = ["input_poses", "poses", "poses_description"]
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Corrupted Format: DataFrame does not contain mandatory Poses column {col}")
        return df

    def split_multiline_fasta(self, path: str, encoding: str = "UTF-8") -> list[str]:
        '''Splits multiline fasta input files.'''
        logging.warning(f"Multiline Fasta detected as input to poses. Splitting up the multiline fasta into multiple poses. Split fastas are stored at work_dir/input_fastas/")
        if not hasattr(self, "work_dir"):
            raise AttributeError(f"Set up a work_dir attribute (Poses.set_work_dir()) for your poses class.")

        # read multilie-fasta file and split into individual poses
        fasta_dict = parse_fasta_to_dict(path, encoding=encoding)

        # prepare descriptions in fasta_dict for writing:
        symbols_to_replace = r"[\/\-\:\ \.\|\,]"
        fasta_dict = {re.sub(symbols_to_replace, "_", description): seq for description, seq in fasta_dict.items()}

        # setup fasta directory self.work_dir/input_fastas_split/
        output_dir = os.path.abspath(f"{self.work_dir}/input_fastas_split/")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # write individual poses in fasta directory:
        out_poses = []
        for description, seq in fasta_dict.items():
            fp = f"{output_dir}/{description}.fa"
            try:
                # check if files are already there. If contents do not match, write the new fasta-file
                subfasta_dict = parse_fasta_to_dict(path, encoding=encoding)
                x_desc = list(subfasta_dict.keys())[0]
                x_seq = list(subfasta_dict.values())[0]
                if description != x_desc or seq != x_seq:
                    raise FileNotFoundError

            except FileNotFoundError:
                with open(fp, 'w', encoding=encoding) as f:
                    f.write(f">{description}\n{seq}")

            # add fasta path to out_poses:
            out_poses.append(fp)

        # return list containing paths to .fa files as poses.
        return out_poses

    def determine_pose_type(self, pose_col: str = None) -> list:
        '''checks the file extensions of poses (or the poses provided in <pose_col>), returns a list of extensions'''
        def extract_extension(file_path):
            _, ext = os.path.splitext(file_path)
            return ext

        pose_col = pose_col or 'poses'

        # extract extensions and create a set containing only unique values
        ext = list(set(self.df[pose_col].apply(extract_extension).to_list()))
        if len(ext) > 1:
            logging.warning(f"Multiple file extensions present in poses: {ext}")
            return ext
        if len(ext) == 1:
            if ext[0] == "":
                logging.warning(f"Could not determine file extension from poses!")
            else:
                logging.info(f"Poses identified as {ext} files")
            return ext
        return []

    ############################################ Input Methods ######################################
    def load_poses(self, poses_path: str) -> "Poses":
        '''Loads Poses class from a stored dataframe.'''
        # read format
        load_function = get_format(poses_path)

        # load df from file:
        self.set_poses(poses=load_function(poses_path))
        return self

    ############################################ Output Methods ######################################
    def save_scores(self, out_path: str = None, out_format: str = None) -> None:
        '''Saves Poses DataFrame as scorefile.'''
        # setup defaults
        out_path = out_path or self.scorefile
        out_format = out_format or self.storage_format

        # make sure the filename conforms to format
        if not out_path.endswith(f".{out_format}"):
            out_path += f".{out_format}"

        if (save_method_name := FORMAT_STORAGE_DICT.get(out_format.lower())):
            getattr(self.df, save_method_name)(out_path)

    def save_poses(self, out_path: str, poses_col: str = "poses", overwrite: bool = True) -> None:
        '''Saves current "poses" from poses.df at out_path. Overwrites poses by default.'''
        poses = self.df[poses_col].to_list()
        new_poses = [os.path.join(out_path, os.path.basename(pose)) for pose in poses]
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        # check if poses are already at out_path, skip if overwrite is set to False
        if all((os.path.isfile(pose) for pose in new_poses)) and not overwrite:
            logging.info(f"Poses already found at {out_path} and overwrite is set to 'False'. Skipping save_poses.")
            return

        # save poses
        logging.info(f"Storing poses from column {poses_col} at {out_path}")
        for pose, new_pose in zip(poses, new_poses):
            shutil.copy(pose, new_pose)

    def poses_list(self):
        '''Simple method to return current poses from DataFrame as a list.'''
        return self.df["poses"].to_list()

    ########################################## Operations ###############################################
    def get_pose(self, pose_description: str) -> Bio.PDB.Structure.Structure:
        '''Loads a singular pose from DataFrame. pose_description has to be current description of pose (stored in poses.df["poses_description"] column)'''
        if not pose_description in self.df["poses_description"]:
            raise KeyError(f"Pose {pose_description} not Found in Poses DataFrame!")
        return load_structure_from_pdbfile(self.df[self.df["poses_description"] == pose_description]["poses"].values[0])

    def duplicate_poses(self, output_dir:str, n_duplicates:int) -> None:
        '''Creates Pose duplicates with added index layers.
        This Function is intended to be used when multiple processing units are needed with distinct inputs.
        '''
        def insert_index_layer(pose, n, sep:str="_") -> str:
            '''inserts index layer.'''
            filepath, filename = pose.rsplit("/", maxsplit=1)
            description, ext = filename.rsplit(".", maxsplit=1)
            return f"{filepath}/{description}{sep}{str(n).zfill(4)}.{ext}"

        # define outputs:
        output_files = [f'{output_dir}/{insert_index_layer(pose, n, "_")}' for pose in self.poses_list() for n in range(n_duplicates)]
        output_dict = {
            "temp_dp_select_col": [file_path.rsplit("/", maxsplit=1)[-1].rsplit("_", maxsplit=1)[0] for file_path in output_files],
            "temp_dp_description": [file_path.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0] for file_path in output_files],
            "temp_dp_location": output_files
        }

        # merge DataFrames:
        self.df.merge(pd.DataFrame(output_dict), left_on="poses_description", right_on="temp_dp_select_col")

        # drop select_col and reset index:
        self.df.drop("temp_dp_select_col", inplace=True, axis=1)
        self.df.reset_index(inplace=True, drop=True)

        # check if outputs exist:
        for pose in self:
            if not os.path.isfile(pose["temp_dp_location"]):
                shutil.copy(pose["pose"], pose["temp_dp_location"])

        # reset poses and poses_description columns
        self.df["poses"] = self.df["temp_dp_location"]
        self.df["poses_description"] = self.df["description"]

    def reset_poses(self, new_poses_col: str='input_poses', reset_df: bool=False):
        '''Resets poses to the poses in <new_poses_col>. Updates the 'poses', 'poses_description' and 'input_poses' columns, if necessary.
        If <reset_df> is True or the number of unique poses in <new_poses_col> is different to the current number of poses, it will delete the poses dataframe (except for the mandatory columns).'''
        
        def unique_ordered_list(original_list):
            seen = set()  # Initialize an empty set to track seen elements
            unique_list = []
            for item in original_list:
                if item not in seen:  # Check membership in the set, which is O(1) (faster lookup in sets than in lists)
                    unique_list.append(item)
                    seen.add(item)  # Add the item to the set
            return unique_list

        col_in_df(self.df, new_poses_col)

        new_poses = self.df[new_poses_col].to_list()
        # handle multiline .fa inputs for poses!
        for pose in new_poses:
            if not pose.endswith(".fa") and not pose.endswith(".fasta"):
                continue
            if len(parse_fasta_to_dict(pose)) > 1:
                new_poses.remove(pose)
                new_poses += self.split_multiline_fasta(pose)

        # create unique poses
        new_poses = unique_ordered_list(new_poses)

        if reset_df:
            if len(new_poses) == len(self.df.index): input_poses = self.df['input_poses'].to_list()
            else:
                logging.warning(f"Different number of new poses ({len(new_poses)}) than number of original poses ({len(self.df.index)}). Input poses will be reset!")
                input_poses = new_poses
            self.df = pd.DataFrame({"input_poses": input_poses, "poses": new_poses, "poses_description": self.parse_descriptions(new_poses)})
        elif not len(new_poses) == len(self.df.index):
            logging.warning(f"Different number of new poses ({len(new_poses)}) than number of original poses ({len(self.df.index)}). Poses dataframe will be reset!")
            self.df = pd.DataFrame({"input_poses": new_poses, "poses": new_poses, "poses_description": self.parse_descriptions(new_poses)})
        else:
            self.df['poses'] = new_poses
            self.df['poses_description'] = self.parse_descriptions(poses=self.df['poses'].to_list())

    def set_motif(self, motif_col: str) -> None:
        '''Sets a motif attribute. This will be accessed by some runners, for example, RFdiffusion, to automatically update residue mappings from contigs to diffused structures.'''
        # check if motif_col exists. check if all entries in motif col are ResidueSelection objects.
        col_in_df(self.df, motif_col)
        if not all([isinstance(motif, ResidueSelection) for motif in self.df[motif_col].to_list()]):
            raise TypeError(f"Setting a motif requires the objects in 'motif_col' to be of type ResidueSelection. Check documentation of protflow.residues module for how to create the object (it's simple).")

        # set motif
        self.motifs.append(motif_col)

    def convert_pdb_to_fasta(self, prefix: str, update_poses: bool = False, chain_sep: str = ":") -> None:
        '''Converts .pdb files to .fasta files in <prefix>_fasta_location. If update_poses is True, fasta files will be set as new poses. Fasta file paths are saved in <prefix>_fasta_location column in poses.df'''
        if not self.determine_pose_type() == ['.pdb']:
            raise RuntimeError(f"Poses must be of type .pdb, not {self.determine_pose_type()}")

        os.makedirs(fasta_dir := os.path.join(self.work_dir, f'{prefix}_fasta_location'), exist_ok=True)
        seqs = [get_sequence_from_pose(load_structure_from_pdbfile(path_to_pdb=pose), chain_sep=chain_sep) for pose in self.df['poses'].to_list()]

        fasta_paths = []
        for name, seq in zip(self.df['poses_description'].to_list(), seqs):
            fasta_path = os.path.join(fasta_dir, f'{name}.fasta')
            fasta_paths.append(fasta_path)
            with open(fasta_path, 'w', encoding="UTF-8") as f:
                f.write(f">{name}\n{seq}")

        self.df[f'{prefix}_fasta_location'] = fasta_paths
        if update_poses:
            self.df['poses'] = fasta_paths

    ########################################## Filtering ###############################################
    def filter_poses_by_rank(self, n: float, score_col: str, remove_layers = None, layer_col = "poses_description", sep = "_", ascending = True, prefix: str = None, plot: bool = False, overwrite: bool = True, storage_format: str = None) -> "Poses":
        '''
        Filters your current poses by a specified scoreterm down to either a fraction (of all poses) or a total number of poses,
        depending on which value was given with <n>.
        
        Args:
            <n>:                   Any positive number. Number between 0 and 1 are interpreted as a fraction to filter,
                                   Numbers >1 are interpreted as absolute number of poses after filtering.
            <score_col>:           Column name of the column that contains the scores by which the poses should be filtered.
            <plot>:                (bool, str, list) Do you want to plot filtered stats? If True, it will plot filtered scoreterm. 
                                   If str, it will look for the argument in self.df and use that scoreterm for plotting.
                                   If list, it will try to plot all scoreterms in the list.
            <ascending>:           Whether to filter ascending (True) or descending (False)
            <prefix>:              Save filter output to <prefix>_filter in filter directory, if given
            <format>:              Save filter output in this format, if None use default pose format
        
        To filter your DataFrame poses based on parent poses, add arguments: 
            <remove_layers>        how many index layers must be removed to reach parent pose? If <remove_layers> = 0, groups 
                                   <layer_col> and filters individual groups (useful e.g. if <layer_col> is not poses_description,
                                   but a group identifier)
            <layer_col>            column name that contains names of the poses, Default="poses_description"
            <sep>                  index layer separator for pose names in layer_col (pose_0001_0003.pdb -> '_')
            
            
            
        Returns poses with filtered DataFrame. Updates pose_df.
        '''

        # define filter output if <prefix> is provided, make sure output directory exists
        if prefix:
            if self.filter_dir is None:
                raise AttributeError(f"Filter directory was not set! Did you set a working directory? work_dir can be set with Poses.set_work_dir() and sets up a filter_dir automatically.")
            os.makedirs(self.filter_dir, exist_ok=True)

            # make sure output format is available
            storage_format = storage_format or self.storage_format
            if storage_format not in FORMAT_STORAGE_DICT:
                raise KeyError(f"Format {storage_format} not available. Format must be on of {list(FORMAT_STORAGE_DICT)}")

            # set filter output name
            output_name = os.path.join(self.filter_dir, f"{prefix}_filter.{storage_format}")

            # load previous filter output if it exists and <overwrite> = False, set poses_df as filtered dataframe and return filtered dataframe
            if not overwrite and os.path.isfile(output_name):
                filter_df = get_format(output_name)(output_name)
                self.df = filter_df
                return filter_df

        # Filter df down to the number of poses specified with <n>
        orig_len = str(len(self.df))
        filter_df = filter_dataframe_by_rank(df=self.df, col=score_col, n=n, remove_layers=remove_layers, layer_col=layer_col, sep=sep, ascending=ascending).reset_index(drop=True)
        logging.info(f"Filtered poses from {orig_len} to {str(len(filter_df))} poses.")

        # save filtered dataframe if prefix is provided
        if prefix:
            logging.info(f"Saving filter output to {output_name}.")
            save_method_name = FORMAT_STORAGE_DICT.get(storage_format)
            getattr(filter_df, save_method_name)(output_name)

        # create filter-plots if specified.
        if plot:
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)

            out_path = os.path.join(self.plots_dir, f"{prefix}_filter.png")
            logging.info(f"Creating filter plot at {out_path}.")
            cols = [score_col]
            plots.violinplot_multiple_cols_dfs(
                dfs=[self.df, filter_df],
                df_names=["Before Filtering", "After Filtering"],
                cols=cols,
                titles=cols,
                y_labels=cols,
                out_path=out_path
            )

        # update object attributs [df]
        self.df = filter_df
        logging.info(f"Filtering completed.")
        return self

    def filter_poses_by_value(self, score_col: str, value, operator: str, prefix: str = None, plot: bool = False, overwrite: bool = True, storage_format: str = None) -> "Poses":
        '''
        Filters your current poses by a specified <score_col> according to the provided <value> and <operator>.
        
        Args:
            <n>:                   Any positive number. Number between 0 and 1 are interpreted as a fraction to filter,
                                   Numbers >1 are interpreted as absolute number of poses after filtering.
            <score_col>:           Column name of the column that contains the scores by which the poses should be filtered.
            <plot>:                (bool, str, list) Do you want to plot filtered stats? If True, it will plot filtered scoreterm. 
                                   If str, it will look for the argument in self.df and use that scoreterm for plotting.
                                   If list, it will try to plot all scoreterms in the list.
            <value>:               Value that is used for comparison.
            <operator>             Type of comparison operator. Can be either '>','>=', '<', '<=', '=' or '!='.")
            <prefix>:              Save filter output to <prefix>_filter.<format> in filter directory, if provided
            <format>:              Save filter output in this format, if None use default pose format
        
            
        Returns poses with filtered DataFrame. Updates pose_df.
        '''

        logging.info(f"Filtering poses according to column {score_col} with operator {operator} and target value {value}")

        # define filter output if <prefix> is provided, make sure output directory exists
        if prefix:
            if self.filter_dir is None:
                raise AttributeError(f"Filter directory was not set! Did you set a working directory?")
            os.makedirs(self.filter_dir, exist_ok=True)

            # make sure output format is available
            storage_format = storage_format or self.storage_format
            if storage_format not in FORMAT_STORAGE_DICT:
                raise KeyError(f"Format {storage_format} not available. Format must be one of {list(FORMAT_STORAGE_DICT)}")

            # set filter output name
            output_name = os.path.join(self.filter_dir, f"{prefix}_filter.{storage_format}")

            # load previous filter output if it exists and <overwrite> = False, set poses_df as filtered dataframe and return filtered dataframe
            if not overwrite and os.path.isfile(output_name):
                filter_df = get_format(output_name)(output_name)
                self.df = filter_df
                return filter_df

        # Filter df down to the number of poses specified with <n>
        orig_len = len(self.df)
        filter_df = filter_dataframe_by_value(df=self.df, col=score_col, value=value, operator=operator).reset_index(drop=True)

        # make sure there are still poses left in the Poses class.
        if len(filter_df) == 0:
            raise ValueError(f"All poses removed from Poses object. No pose fullfills the filtering criterium {operator} {value} for score {score_col}")
        logging.info(f"Filtered poses from {orig_len} to {len(filter_df.index)} poses.")

        # save filtered dataframe if prefix is provided
        if prefix:
            logging.info(f"Saving filter output to {output_name}.")
            save_method_name = FORMAT_STORAGE_DICT.get(storage_format)
            getattr(filter_df, save_method_name)(output_name)

        if plot:
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)
            out_path = os.path.join(self.plots_dir, f"{prefix}_filter.png")
            logging.info(f"Creating filter plot at {out_path}.")
            cols = [score_col]
            plots.violinplot_multiple_cols_dfs(
                dfs=[self.df, filter_df],
                df_names=["Before Filtering", "After Filtering"],
                cols=cols,
                titles=cols,
                y_labels=cols,
                out_path=out_path
            )

        # update object attributs [df]
        self.df = filter_df
        logging.info(f"Filtering completed.")
        return self

    ########################################## Score manipulation ###############################################
    def calculate_composite_score(self, name: str, scoreterms: list[str], weights: list[float], plot: bool = False, scale_output: bool = False) -> "Poses":
        '''
        Combine multiple score columns by weighted addition. Individual scoreterms will be normalized before combination. Score will be scaled from 0 to 1, with 1 indicating the best score.
        Args:
            <name>              name of the column that should contain the composite score
            <scoreterms>        list of score column names
            <weights>           list of weights for each score column. score_col will be multiplied with weight before addition with other scoreterms.
                                if the best value in a score column is the lowest, use negative weights (e.g. -1 for Rosetta total score) and vice versa. 
        '''
        logging.info(f"Creating composite score {name} for scoreterms {scoreterms} with weights {weights}")
        # check if output column already exists in dataframe
        if name in self.df:
            logging.warning(f"Column {name} already exists in poses dataframe! It will be overwritten!")
        # calculate composite score
        self.df[name] = combine_dataframe_score_columns(df=self.df, scoreterms=scoreterms, weights=weights, scale=scale_output)

        if plot:
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)
            out_path = os.path.join(self.plots_dir, f"{name}_comp_score.png")
            logging.info(f"Creating composite score plot at {out_path}.")
            scoreterms.append(name)
            plots.violinplot_multiple_cols(
                df=self.df,
                cols=scoreterms,
                titles=scoreterms,
                y_labels=scoreterms,
                dims=None,
                out_path=out_path
            )

        self.save_scores()
        logging.info("Composite score creation completed.")

        return self

def normalize_series(ser: pd.Series, scale: bool = False) -> pd.Series:
    '''
    Normalizes a pandas series by subtracting the median and dividing by standard deviation.
    If scale = True, the normalized values will be scaled from 0 to 1. Returns a series.
    '''
    ser = ser.copy()
    # calculate median and standard deviation
    median = ser.median()
    std = ser.std()
    # check if all values in <score_col> are the same, return 0 if yes
    if ser.nunique() == 1:
        ser[:] = 0
        return ser
    # normalize score by subtracting median and dividing by standard deviation
    ser = (ser - median) / std
    # scale output to values between 0 and 1
    if scale:
        ser = scale_series(ser)
    return ser

def scale_series(ser: pd.Series) -> pd.Series:
    '''
    Scale a pandas series to values between 0 and 1. Returns a series.
    '''
    ser = ser.copy()
    # check if all values in <score_col> are the same, set all values to 0 if yes as no scaling is possible
    if ser.nunique() == 1:
        ser[:] = 0
        return ser
    # scale series to values between 0 and 1
    factor = ser.max() - ser.min()
    ser = ser / factor
    ser = ser + (1 - ser.max())

    return ser

def combine_dataframe_score_columns(df: pd.DataFrame, scoreterms: list[str], weights: list[float], scale: bool = False) -> pd.Series:
    '''
    Combine multiple score columns by weighted addition. 
    Individual scoreterms will be normalized before combination. 
    If scale is set, returns a series of values scaled from 0 to 1, with 1 indicating the best scoring.

    Args:
        <df>                input dataframe
        <scoreterms>        list of score column names
        <weights>           list of weights for each score column. score_col will be multiplied with weight before addition with other scoreterms.
                            if the best value in a score column is the lowest, use negative weights (e.g. -1 for Rosetta total score) and vice versa. 
    '''
    if not len(scoreterms) == len(weights):
        raise ValueError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")

    df = df.copy()
    for col in scoreterms:
        # check if column contains only floats or integers, raise an error otherwise
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Column {col} must only contain float or integers!")

        # normalize and scale scoreterm
        df[col] = normalize_series(ser=df[col], scale=False)

    # combine weighted scores
    combined_col = sum((df[col]*weight for col, weight in zip(scoreterms, weights)))
    return scale_series(combined_col) if scale else combined_col

def get_format(path: str):
    '''reads in path as str and returns a pandas loading function.'''
    loading_function_dict = {
        "json": pd.read_json,
        "csv": pd.read_csv,
        "pickle": pd.read_pickle,
        "feather": pd.read_feather,
        "parquet": pd.read_parquet
    }
    return loading_function_dict[path.split(".")[-1]]

def load_poses(poses_path: str) -> Poses:
    '''Loads Poses class from a stored dataframe.'''
    return Poses().load_poses(poses_path)

def col_in_df(df: pd.DataFrame, column: str|list[str]) -> None:
    '''Checks if column exists in DataFrame and returns KeyError if not.'''
    if isinstance(column, list):
        for col in column:
            if not col in df.columns:
                raise KeyError(f"Could not find {col} in poses dataframe! Are you sure you provided the right column name?")
    else:
        if not column in df.columns:
            raise KeyError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")

def filter_dataframe_by_rank(df: pd.DataFrame, col: str, n: float|int, remove_layers: int = None, layer_col: str = "poses_description", sep: str = "_", ascending: bool = True) -> pd.DataFrame:
    '''
    remove_layers option allows to filter dataframe based on groupings after removing index layers.
    If the option remove_layers is set (has to be type: int), then n determines how many c
    if <remove_layers> = 0, dataframe will be grouped by values in <layer_col> and the grouped dfs will be filtered for top n rows
    '''

    def determine_filter_n(df: pd.DataFrame, n: float) -> int:
        '''
        determines if n is a fraction or an integer and sets cutoff for dataframe filtering accordingly.
        '''
        filter_n = float(n)
        if filter_n < 1:
            filter_n = round(len(df) * filter_n)
        elif filter_n <= 0:
            raise ValueError(f"ERROR: Argument <n> of filter functions cannot be smaller than 0. It has to be positive number. If n < 1, the top n fraction is taken from the DataFrame. if n > 1, the top n rows are taken from the DataFrame")

        return int(filter_n)

    # make sure <col> exists columns in <df>
    col_in_df(df, col)

    # if remove_layers is set, compile list of unique pose descriptions after removing one index layer:
    if remove_layers:
        if not isinstance(remove_layers, int):
            raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")

        # make sure <layer_col> exists in df
        col_in_df(df, layer_col)

        # create temporary description column with removed index layers
        df["tmp_layer_column"] = df[layer_col].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)

        # group by temporary description column, filter top n rows per group
        filtered = []
        for _, group_df in df.groupby("tmp_layer_column", sort=False):
            filtered.append(group_df.sort_values(by=col, ascending=ascending).head(determine_filter_n(group_df, n)))
        filtered_df = pd.concat(filtered).reset_index(drop=True)

        #drop temporary description column
        filtered_df.drop("tmp_layer_column", axis=1, inplace=True)

    else:
        filtered_df = df.sort_values(by=col, ascending=ascending).head(determine_filter_n(df, n))

    return filtered_df


def filter_dataframe_by_value(df: pd.DataFrame, col: str, value: float|int, operator: str) -> pd.DataFrame:
    '''
    Filters dataframe based on a value and an operator.
    '''
    # make sure <col> exists columns in <df>
    col_in_df(df, col)

    # Define the comparison based on the operator
    if operator == '>': filtered_df = df[df[col] > value]
    elif operator == '>=': filtered_df = df[df[col] >= value]
    elif operator == '<': filtered_df = df[df[col] < value]
    elif operator == '<=': filtered_df = df[df[col] <= value]
    elif operator == '=': filtered_df = df[df[col] == value]
    elif operator == '!=': filtered_df = df[df[col] != value]
    else:
        raise KeyError("Invalid operator. Supported operators are '>','>=', '<', '<=', '=', '!='.")

    return filtered_df
