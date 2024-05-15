'''Module Containing python runners'''
# builtins
import logging
import os
import re
import shlex

# dependencies
import pandas as pd

# custom
from protslurm.poses import Poses, get_format, FORMAT_STORAGE_DICT
from protslurm.jobstarters import JobStarter

class RunnerOutput:
    '''RunnerOutput class handles how protein data is passed between Runners and Poses classes.'''
    def __init__(self, poses: Poses, results:pd.DataFrame, prefix:str, index_layers:int=0, index_sep:str="_"):
        self.results = self.check_data_formatting(results)

        # Remove layers if option is set
        if index_layers:
            self.results["select_col"] = self.results["description"].str.split(index_sep).str[:-1*index_layers].str.join(index_sep)
        else:
            self.results["select_col"] = self.results["description"]

        self.results = self.results.add_prefix(f"{prefix}_")
        self.poses = poses
        self.prefix = prefix

    def check_data_formatting(self, results:pd.DataFrame):
        '''Checks if the input DataFrame has the correct format.
        Needs to contain 'description' and 'location' columns.
        '''
        def extract_description(path):
            return os.path.splitext(os.path.basename(path))[0]

        mandatory_cols = ["description", "location"]
        if any(col not in results.columns for col in mandatory_cols): raise ValueError(f"Input Data to RunnerOutput class MUST contain columns 'description' and 'location'.\nDescription should carry the name of the poses, while 'location' should contain the path (+ filename and suffix).")
        if not (results['description'] == results['location'].apply(extract_description)).all(): raise ValueError(f"'description' column does not match 'location' column in runner output dataframe!")
        return results

    def return_poses(self):
        '''
        This method is intended to be used to integrate the output of a runner into a Poses class.
        Adds Output of a Runner class formatted in RunnerOutput into Poses.df. Returns Poses class.'''
        startlen = len(self.results.index)

        # check for duplicate columns
        if any(x in list(self.poses.df.columns) for x in list(self.results.columns)):
            logging.info(f"WARNING: Merging DataFrames that contain column duplicates. Column duplicates will be renamed!")

        # if poses are empty, concatenate DataFrames:
        if len(self.poses.poses_list()) == 0:
            logging.info(f"Poses.df is empty. This means the existing poses.df will be merged with the new results of {self.prefix}")
            merged_df = pd.concat([self.poses.df, self.results])

        # if poses.df contains scores, merge DataFrames based on poses_description to keep scores continous
        else:
            merged_df = self.poses.df.merge(self.results, left_on="poses_description", right_on=f"{self.prefix}_select_col") # pylint: disable=W0201

        # cleanup after merger
        merged_df.drop(f"{self.prefix}_select_col", axis=1, inplace=True)
        merged_df.reset_index(inplace=True, drop=True)

        # check if merger was successful:
        if len(merged_df) == 0:
            raise ValueError(f"Merging DataFrames failed. This means there was no overlap found between poses.df['poses_description'] and results[new_df_col]")
        if len(merged_df) < startlen:
            raise ValueError(f"Merging DataFrames failed. Some rows in results[new_df_col] were not found in poses.df['poses_description']")

        # reset poses and poses_description column
        merged_df["poses"] = [os.path.abspath(pose) for pose in merged_df[f"{self.prefix}_location"].to_list()]
        merged_df["poses_description"] = merged_df[f"{self.prefix}_description"]

        # integrate new results into Poses object
        self.poses.df = merged_df
        self.poses.save_scores()
        return self.poses

class Runner:
    '''Abstract Runner baseclass handling interface between Runners and Poses.'''
    def __str__(self):
        raise NotImplementedError(f"Your Runner needs a name! Set in your Runner class: 'def __str__(self): return \"runner_name\"'")

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter) -> RunnerOutput:
        '''method that interacts with Poses to run jobs and send Poses the scores.'''
        raise NotImplementedError(f"Runner Method 'run' was not overwritten yet!")

    def search_path(self, input_path: str, path_name: str) -> str:
        '''Checks if a given path exists (is not None) and if it exists on the local filesystem. If so, returns path, otherwise raises Error.'''
        if not input_path:
            raise ValueError(f"Path for {path_name} not set: {input_path}. Set the path uner {path_name} in protflow's config.py file.")
        if not os.path.isfile(input_path):
            raise ValueError(f"Path set for {path_name} does not exist at {input_path}. Check correct filepath!")
        return input_path

    def check_for_prefix(self, prefix: str, poses: "Poses") -> None:
        '''Checks if a column already exists in the poses DataFrame.'''
        if prefix in poses.df.columns:
            raise KeyError(f"Column {prefix} found in Poses DataFrame! Pick different Prefix!")

    def prep_pose_options(self, poses:Poses, pose_options:list[str]=None) -> list:
        '''Checks if pose_options are of the same length as poses, if pose_options are provided, '''
        # if pose_options is str, look up pose_options from poses.df
        if isinstance(pose_options, str):
            col_in_df(poses.df, pose_options)
            pose_options = poses.df[pose_options].poses_list()

        # if pose_options is None (not set) return list with empty dicts.
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            pose_options = [None for _ in poses]

        if len(poses) != len(pose_options) and len(poses) != 0:
            raise ValueError(f"Arguments <poses> and <pose_options> for RFdiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")

        # if pose_options is list and as long as poses, just return list. Has to be list of dicts.
        return pose_options

    def generic_run_setup(self, poses: Poses, prefix:str, jobstarters: list[JobStarter], make_work_dir: bool = True) -> tuple[str, JobStarter]:
        '''Generic setup method to prepare for a .run() method.
        Checks if prefix exists in poses.df, sets up a jobstarter and creates the working_directory.
        Returns path to work_dir.

        Order of jobstarters in :jobstarter: parameter is:
            [Runner.run(jobstarter), Runner.jobstarter, poses.default_jobstarter]'''
        # check for prefix
        self.check_for_prefix(prefix, poses)

        # setup jobstarter
        run_jobstarter, runner_jobstarter, poses_jobstarter = jobstarters
        jobstarter = run_jobstarter or (runner_jobstarter or poses_jobstarter) # select jobstarter, priorities: Runner.run(jobstarter) > Runner.jobstarter > poses.jobstarter
        if not jobstarter or not isinstance(jobstarter, JobStarter):
            raise ValueError(f"No Jobstarter was set either in the Runner, the .run() function or the Poses class.")

        # setup directory
        work_dir = os.path.abspath(f"{poses.work_dir}/{prefix}")
        if not os.path.isdir(work_dir) and make_work_dir:
            os.makedirs(work_dir, exist_ok=True)
        return work_dir, jobstarter

    def check_for_existing_scorefile(self, scorefile: str, overwrite: bool = False) -> pd.DataFrame:
        '''Checks if a scorefile exists and returns DataFrame if overwrite is False. Otherwise returns None'''
        # check if scorefile exists if overwrite is False
        if os.path.isfile(scorefile) and not overwrite:
            # pick method to import scorefile
            scores = get_format(scorefile)(scorefile)
            return scores

    def save_runner_scorefile(self, scores: pd.DataFrame, scorefile: str) -> None:
        '''Stores runner's scorefile based on the format specified at the scorefile extension.'''
        # extract file extension from scorefile
        storage_method = os.path.splitext(scorefile)[1][1:]

        # pick method to save scorefile
        if (save_method_name := FORMAT_STORAGE_DICT.get(storage_method.lower())):
            getattr(scores, save_method_name)(scorefile)
        else:
            raise KeyError(f"Could not find method to save scorefile as {storage_method}. Make sure the score file extension is correct!")

def parse_generic_options(options: str, pose_options: str, sep="--") -> tuple[dict,list]:
    """
    Parses generic options and pose-specific options from two input strings, combining them into a single dictionary of options
    and a list of flags. Pose-specific options overwrite generic options in case of conflicts. Options are expected to be separated
    by a specified separator within each input string, with options and their values separated by spaces.

    Parameters:
    - options (str): A string of generic options, where different options are separated by the specified separator and each option's
      value (if any) is separated by space.
    - pose_options (str): A string of pose-specific options, formatted like the `options` parameter. These options take precedence
      over generic options.
    - sep (str, optional): The separator used to distinguish between different options in both input strings. Defaults to "--".

    Returns:
    - tuple: A 2-element tuple where the first element is a dictionary of merged options (key-value pairs) and the second element
      is a list of unique flags (options without values) from both input strings.

    Example:
    >>> parse_generic_options("--width 800 --height 600", "--color blue --verbose")
    ({'width': '800', 'height': '600', 'color': 'blue'}, ['verbose'])

    This function internally utilizes a helper function `expand_options_flags` to process each input string separately before
    merging the results, ensuring that pose-specific options and flags are appropriately prioritized and duplicates are removed.
    """
    # parse into options and flags:
    opts, flags = regex_expand_options_flags(options, sep=sep)
    pose_opts, pose_flags = regex_expand_options_flags(pose_options, sep=sep)

    # merge options and pose_options (pose_opts overwrite opts), same for flags
    opts.update(pose_opts)
    flags = list(set(flags) | set(pose_flags))
    return opts, flags

def col_in_df(df:pd.DataFrame, column:str):
    '''Checks if column exists in DataFrame and returns KeyError if not.'''
    if not column in df.columns:
        raise KeyError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")

def expand_options_flags(options_str: str, sep:str="--") -> tuple[dict, set]:
    '''parses split options '''
    if not options_str:
        return {}, []

    # split along separator
    firstsplit = [x.strip() for x in options_str.split(sep) if x]

    # parse into options and flags:
    opts = {}
    flags = []
    for item in firstsplit:
        if len((x := item.split())) > 1:
            opts[x[0]] = " ".join(x[1:])
        elif len((x := item.split("="))) > 1:
            opts[x[0]] = " ".join(x[1:])
        else:
            flags.append(x[0])

    return opts, set(flags)

def regex_expand_options_flags(options_str: str, sep: str = "--") -> tuple[dict,set]:
    '''Uses Regex to split stuff'''
    if options_str is None:
        return dict(), set()
    # Regex to split the command line at the separator which is not inside quotes
    split_pattern = rf"(?<!\S){sep}(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)(?=(?:[^\']*\'[^\']*\')*[^\']*$)"
    parts = [x.strip() for x in re.split(split_pattern, options_str) if x]

    opts = {}
    flags = []

    # Setup part_pattern that splits individual parts at first occurrence of whitespace or equals sign.
    part_pattern = r"\s+|\s*=\s*"

    for part in parts:
        split = re.split(part_pattern, part, maxsplit=1)
        if len(split) > 1:
            opts[split[0]] = split[1]
        else:
            flags.append(split[0])

    return opts, set(flags)

def options_flags_to_string(options: dict, flags: list, sep="--") -> str:
    '''Converts options dict and flags list into one string'''
    def value_in_quotes(value) -> str:
        '''Makes sure that split commandline options are passed in quotes: --option='quoted list of args' '''
        if len(str(value).split(" ")) > 1:
            if not value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
                return f"'{value}'"
        return value

    # assemble options
    out_str = " " + " ".join([f"{sep}{key}={value_in_quotes(value)}" for key, value in options.items()]) if options else ""

    # if flags are present, assemble those too and return
    if flags:
        out_str += f" {sep}" + f" {sep}".join(flags)
    return out_str
