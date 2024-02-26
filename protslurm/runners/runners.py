'''Module Containing python runners'''
import pandas as pd
from protslurm.jobstarters import JobStarter
import logging
import os



class RunnerOutput:
    '''RunnerOutput class handles how protein data is passed between Runners and Poses classes.'''
    def __init__(self, poses, results:pd.DataFrame, prefix:str, index_layers:int=0, index_sep:str="_"):
        self.results = self.check_data_formatting(results)
        # Remove layers if option is set
        if index_layers: self.results["select_col"] = self.results["description"].str.split(index_sep).str[:-1*index_layers].str.join(index_sep)
        else: self.results["select_col"] = self.results["description"]
        self.results = self.results.add_prefix(f"{prefix}_")
        self.poses = poses
        self.prefix = prefix
    
    def check_data_formatting(self, results:pd.DataFrame):
        def extract_description(path):
            return os.path.splitext(os.path.basename(path))[0]
        
        mandatory_cols = ["description", "location"]
        if any(col not in results.columns for col in mandatory_cols): raise ValueError(f"Input Data to RunnerOutput class MUST contain columns 'description' and 'location'.\nDescription should carry the name of the poses, while 'location' should contain the path (+ filename and suffix).")
        if not (results['description'] == results['location'].apply(extract_description)).all(): raise ValueError(f"'description' column does not match 'location' column in runner output dataframe!")
        return results

    def return_poses(self):
        '''Adds Output of a Runner class formatted in RunnerOutput into Poses.df. Returns Poses class'''    
        startlen = len(self.results.index)
        
        # merge DataFrames
        if any(x in list(self.poses.df.columns) for x in list(self.results.columns)): logging.info(f"WARNING: Merging DataFrames that contain column duplicates. Column duplicates will be renamed!")
        merged_df = self.poses.df.merge(self.results, left_on="poses_description", right_on=f"{self.prefix}_select_col") # pylint: disable=W0201
        merged_df.drop(f"{self.prefix}_select_col", axis=1, inplace=True)
        merged_df.reset_index(inplace=True)

        # check if merger was successful:
        if len(merged_df) == 0: raise ValueError(f"Merging DataFrames failed. This means there was no overlap found between poses.df['poses_description'] and results[new_df_col]")
        if len(merged_df) < startlen: raise ValueError(f"Merging DataFrames failed. Some rows in results[new_df_col] were not found in poses.df['poses_description']")

        # reset poses and poses_description column
        merged_df["poses"] = merged_df[f"{self.prefix}_location"]
        merged_df["poses_description"] = merged_df[f"{self.prefix}_description"]
        
        self.poses.df = merged_df

        return self.poses

class Runner:
    '''Abstract Runner baseclass handling interface between Runners and Poses.'''
    def __str__(self):
        raise NotImplementedError(f"Your Runner needs a name! Set in your Runner class: 'def __str__(self): return \"runner_name\"'")

    def run(self, poses, jobstarter:JobStarter, output_dir:str, options:str=None, pose_options:str=None) -> RunnerOutput:
        '''method that interacts with Poses to run jobs and send Poses the scores.'''
        raise NotImplementedError(f"Runner Method 'run' was not overwritten yet!")

def parse_generic_options(options: str, pose_options: str, sep="--") -> tuple[dict,list]:
    '''Parses options and pose_options together into options (dict {arg: val, ...}) and flags (list: [val, ...])'''
    def tmp_parse_options_flags(options_str: str, sep:str="--") -> tuple[dict, list]:
        '''parses split options '''
        if not options_str: return {}, []
        # split along separator
        firstsplit = [x.strip() for x in options_str.split(sep) if x]

        # parse into options and flags:
        opts = dict()
        flags = list()
        for item in firstsplit:
            if len((x := item.split())) > 1:
                opts[x[0]] = " ".join(x[1:])
            else:
                flags.append(x[0])

        return opts, flags

    # parse into options and flags:
    opts, flags = tmp_parse_options_flags(options, sep=sep)
    pose_opts, pose_flags = tmp_parse_options_flags(pose_options, sep=sep)

    # merge options and pose_options (pose_opts overwrite opts), same for flags
    opts.update(pose_opts)
    flags = list(set(flags) | set(pose_flags))
    return opts, flags
