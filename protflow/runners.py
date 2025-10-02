'''
runners module
==============

This module provides functionality for handling the interaction between runners and poses in protein data processing workflows.

It includes classes and utility functions to:

- Manage the output from runner processes.
- Define abstract runner interfaces.
- Parse and manage command-line options and flags for runner processes.

Dependencies:
-------------

- builtins: logging, os, re
- pandas
- protflow.poses: Poses, get_format, FORMAT_STORAGE_DICT
- protflow.jobstarters: JobStarter

Overview:
---------

The `runners` module is designed to facilitate the integration of various runner processes with protein pose data, ensuring
consistent data formatting, error handling, and integration of results into the Poses class. Utility functions provided in
this module support the parsing and handling of command-line options and flags, making it easier to configure and execute
runner processes in a flexible manner.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Author
------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
'''
# builtins
import logging
import os
import re
import functools
from multiprocessing import ProcessError

# dependencies
import pandas as pd

# custom
from .poses import Poses, get_format, FORMAT_STORAGE_DICT
from .jobstarters import JobStarter

class RunnerOutput:
    """
    RunnerOutput class
    ==================

    The `RunnerOutput` class handles how protein data is passed between `Runner` and `Poses` classes. It ensures the correct
    formatting of results and facilitates the integration of runner outputs into the Poses data structure.

    Parameters
    ----------
    poses : Poses
        An instance of the Poses class.
    results : pandas.DataFrame
        A DataFrame containing the results to be checked and formatted. The DataFrame must contain 'description' and 'location' columns.
    prefix : str
        A prefix to be added to the results columns.
    index_layers : int, optional
        Number of index layers to remove from the 'description' column (default is 0).
    index_sep : str, optional
        Separator used in the index (default is "_").

    """
    def __init__(self, poses: Poses, results: pd.DataFrame, prefix: str, index_layers: int = 0, index_sep: str = "_"):
        self.results = self.check_data_formatting(results)

        # Remove layers if option is set
        if index_layers:
            self.results["select_col"] = self.results["description"].str.split(index_sep).str[:-1*index_layers].str.join(index_sep)
        else:
            self.results["select_col"] = self.results["description"]

        self.results = self.results.add_prefix(f"{prefix}_")
        self.poses = poses
        self.prefix = prefix

    def check_data_formatting(self, results: pd.DataFrame):
        """
        Checks if the input DataFrame has the correct format.

        Parameters
        ----------
        results : pandas.DataFrame
            The input DataFrame to be checked. It must contain 'description' and 'location' columns.

        Returns
        -------
        pandas.DataFrame
            The validated and formatted DataFrame.

        Raises
        ------
        ValueError
            If the input DataFrame does not contain the required columns or if the 'description' column does not match the 'location' column.
        """
        def extract_description(path):
            return os.path.splitext(os.path.basename(path))[0]

        mandatory_cols = ["description", "location"]
        if any(col not in results.columns for col in mandatory_cols):
            raise ValueError("Input Data to RunnerOutput class MUST contain columns 'description' and 'location'.\nDescription should carry the name of the poses, while 'location' should contain the path (+ filename and suffix).")
        if not (results['description'] == results['location'].apply(extract_description)).all():
            raise ValueError(f"'description' column does not match 'location' column in runner output dataframe!\n{results[['description', 'location']].head(5).values}")
        return results

    def return_poses(self) -> Poses:
        """
        Integrates the output of a runner into a Poses class.

        This method adds the output of a Runner class formatted in RunnerOutput into `Poses.df` and returns the updated Poses instance.

        Returns
        -------
        Poses
            The updated Poses instance with the integrated runner output.

        Raises
        ------
        ValueError
            If merging DataFrames fails due to no overlap between `Poses.df['poses_description']` and `results[new_df_col]` or if some rows in `results[new_df_col]` were not found in `Poses.df['poses_description']`.
        """
        startlen = len(self.results.index)

        # check for duplicate columns
        if any(x in list(self.poses.df.columns) for x in list(self.results.columns)):
            logging.info("WARNING: Merging DataFrames that contain column duplicates. Column duplicates will be renamed!")

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
            print(self.poses.df["poses_description"].to_list()[:3], self.results[f"{self.prefix}_select_col"].to_list()[:3])
            raise ValueError("Merging DataFrames failed. This means there was no overlap found between poses.df['poses_description'] and results[new_df_col]")
        if len(merged_df) < startlen:
            print(self.poses.df["poses_description"].to_list()[:3], self.results[f"{self.prefix}_select_col"].to_list()[:3])
            logging.error(self.poses.df["poses_description"].to_list()[:3], self.results[f"{self.prefix}_select_col"].to_list()[:3])
            raise ValueError("Merging DataFrames failed. Some rows in results[new_df_col] were not found in poses.df['poses_description']")

        # reset poses and poses_description column
        merged_df["poses"] = [os.path.abspath(pose) for pose in merged_df[f"{self.prefix}_location"].to_list()]
        merged_df["poses_description"] = merged_df[f"{self.prefix}_description"]

        # integrate new results into Poses object
        self.poses.df = merged_df
        self.poses.save_scores()
        return self.poses

class Runner:
    """
    Abstract Runner base class
    ==========================

    The `Runner` class provides an abstract base for defining runners that handle the interface between runner processes and the Poses class.
    It includes methods for running jobs, checking paths, verifying prefixes, preparing pose options, and managing job setup and score files.

    Examples
    --------
    To create a custom runner, subclass `Runner` and implement the abstract methods:

    >>> class MyRunner(Runner):
    >>>     def __str__(self):
    >>>         return "MyRunner"
    >>>
    >>>     def run(self, poses: Poses, prefix: str, jobstarter: JobStarter) -> RunnerOutput:
    >>>         # Custom implementation for running jobs
    >>>         pass

    Example usage:

    >>> my_runner = MyRunner()
    >>> poses = Poses()
    >>> jobstarter = JobStarter()
    >>> runner_output = my_runner.run(poses, "example_prefix", jobstarter)
    """

    def __str__(self):
        """
        Abstract method to provide the name of the runner.

        This method should be overridden in subclasses to return the name of the runner.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in the subclass.

        Examples
        --------
        >>> class MyRunner(Runner):
        >>>     def __str__(self):
        >>>         return "MyRunner"
        """
        raise NotImplementedError("Your Runner needs a name! Set in your Runner class: 'def __str__(self): return \"runner_name\"'")

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter) -> Poses:
        """
        Abstract method to run jobs and send scores to Poses.

        This method should be overridden in subclasses to define the job execution logic and integrate the results into the Poses class.

        Parameters
        ----------
        poses : Poses
            An instance of the Poses class to be processed.
        prefix : str
            Prefix to be added to the results columns.
        jobstarter : JobStarter
            An instance of the JobStarter class to handle job execution.

        Returns
        -------
        RunnerOutput
            An instance of the RunnerOutput class containing the processed results.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in the subclass.

        Examples
        --------
        >>> class MyRunner(Runner):
        >>>     def run(self, poses: Poses, prefix: str, jobstarter: JobStarter) -> RunnerOutput:
        >>>         # Custom implementation for running jobs
        >>>         pass
        """
        raise NotImplementedError("Runner Method 'run' was not overwritten yet!")

    def search_path(self, input_path: str, path_name: str, is_dir: bool = False) -> str:
        """
        Checks if a given path exists and is valid.

        Parameters
        ----------
        input_path : str
            The path to be checked.
        path_name : str
            The name associated with the path, used for error messages.

        Returns
        -------
        str
            The validated path.

        Raises
        ------
        ValueError
            If the path is not set or does not exist on the local filesystem.

        Examples
        --------
        >>> runner = MyRunner()
        >>> valid_path = runner.search_path("/path/to/file", "example_path")
        """
        if not input_path:
            raise ValueError(f"Path for {path_name} not set: {input_path}. Set the path uner {path_name} in protflow's config.py file.")
        if is_dir:
            if not os.path.isdir(input_path):
                raise ValueError(f":input_path: is not a directory: {input_path}")
        elif not os.path.isfile(input_path):
            raise ValueError(f"Path set for {path_name} does not exist at {input_path}. Check correct filepath!")
        return input_path

    def check_for_prefix(self, prefix: str, poses: "Poses") -> None:
        """
        Checks if a column with the given prefix already exists in the Poses DataFrame.

        Parameters
        ----------
        prefix : str
            The prefix to be checked.
        poses : Poses
            An instance of the Poses class whose DataFrame will be checked.

        Raises
        ------
        KeyError
            If a column with the given prefix already exists in the Poses DataFrame.

        Examples
        --------
        >>> runner = MyRunner()
        >>> poses = Poses()
        >>> runner.check_for_prefix("example_prefix", poses)
        """
        if f"{prefix}_location" in poses.df.columns or f"{prefix}_description" in poses.df.columns:
            raise KeyError(f"Column {prefix} found in Poses DataFrame! Pick different Prefix!")

    def prep_pose_options(self, poses: Poses, pose_options: list[str] = None) -> list:
        """
        Prepares pose options, ensuring they are of the same length as the poses.

        Parameters
        ----------
        poses : Poses
            An instance of the Poses class.
        pose_options : list[str], optional
            A list of pose options to be prepared. If not provided, an empty list will be used.

        Returns
        -------
        list
            A list of prepared pose options.

        Raises
        ------
        ValueError
            If the length of pose_options does not match the length of poses.

        Examples
        --------
        >>> runner = MyRunner()
        >>> poses = Poses()
        >>> prepared_options = runner.prep_pose_options(poses, ["option1", "option2"])
        """
        # if pose_options is str, look up pose_options from poses.df
        if isinstance(pose_options, str):
            col_in_df(poses.df, pose_options)
            pose_options = poses.df[pose_options].to_list()

        # if pose_options is None (not set) return list with empty dicts.
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            pose_options = [None for _ in poses]

        if len(poses) != len(pose_options) and len(poses) != 0:
            raise ValueError(f"Arguments <poses> and <pose_options> for RFdiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")

        # if pose_options is list and as long as poses, just return list. Has to be list of dicts.
        return pose_options

    def generic_run_setup(self, poses: Poses, prefix:str, jobstarters: list[JobStarter], make_work_dir: bool = True) -> tuple[str, JobStarter]:
        """
        Sets up the runner's working directory and jobstarter.

        Checks if the prefix exists in poses.df, sets up a jobstarter, and creates the working directory if necessary.

        Parameters
        ----------
        poses : Poses
            An instance of the Poses class.
        prefix : str
            The prefix to be used for the setup.
        jobstarters : list[JobStarter]
            A list of JobStarter instances to choose from.
        make_work_dir : bool, optional
            Whether to create the working directory if it does not exist (default is True).

        Note: Order of jobstarters in :jobstarter: parameter is: [Runner.run(jobstarter), Runner.jobstarter, poses.default_jobstarter]
    
        Returns
        -------
        tuple[str, JobStarter]
            A tuple containing the path to the working directory and the selected JobStarter instance.

        Raises
        ------
        ValueError
            If no valid JobStarter is set.

        Examples
        --------
        >>> runner = MyRunner()
        >>> poses = Poses()
        >>> jobstarters = [JobStarter(), JobStarter(), JobStarter()]
        >>> work_dir, jobstarter = runner.generic_run_setup(poses, "example_prefix", jobstarters)

        """

        # check for prefix
        self.check_for_prefix(prefix, poses)

        # setup jobstarter
        run_jobstarter, runner_jobstarter, poses_jobstarter = jobstarters
        jobstarter = run_jobstarter or (runner_jobstarter or poses_jobstarter) # select jobstarter, priorities: Runner.run(jobstarter) > Runner.jobstarter > poses.jobstarter
        if not jobstarter or not isinstance(jobstarter, JobStarter):
            raise ValueError("No Jobstarter was set either in the Runner, the .run() function or the Poses class.")

        # setup directory
        work_dir = os.path.abspath(f"{poses.work_dir}/{prefix}")
        if not os.path.isdir(work_dir) and make_work_dir:
            os.makedirs(work_dir, exist_ok=True)

        self.current_jobstarter = jobstarter # TODO: a bit hacky, might lead to problems if jobstarter is not set to wait until jobs finished and the same runner is called multiple times

        return work_dir, jobstarter

    def check_for_existing_scorefile(self, scorefile: str, overwrite: bool = False) -> pd.DataFrame:
        """
        Checks if a scorefile exists and returns it as a DataFrame if overwrite is False.

        Parameters
        ----------
        scorefile : str
            The path to the scorefile.
        overwrite : bool, optional
            Whether to overwrite the scorefile if it exists (default is False).

        Returns
        -------
        pandas.DataFrame
            The scorefile as a DataFrame if it exists and overwrite is False. None otherwise.

        Examples
        --------
        >>> runner = MyRunner()
        >>> scores_df = runner.check_for_existing_scorefile("/path/to/scorefile.csv")
        """
        # check if scorefile exists if overwrite is False
        if os.path.isfile(scorefile) and not overwrite:
            # pick method to import scorefile
            scores = get_format(scorefile)(scorefile)
            return scores

    def save_runner_scorefile(self, scores: pd.DataFrame, scorefile: str) -> None:
        """
        Saves the runner's scorefile based on the file extension format.

        Parameters
        ----------
        scores : pandas.DataFrame
            The DataFrame containing the scores to be saved.
        scorefile : str
            The path to the scorefile to be saved.

        Raises
        ------
        KeyError
            If the file extension format is not recognized.

        Examples
        --------
        >>> runner = MyRunner()
        >>> scores_df = pd.DataFrame({'score': [1, 2, 3]})
        >>> runner.save_runner_scorefile(scores_df, "/path/to/scorefile.csv")
        """
        # extract file extension from scorefile
        storage_method = os.path.splitext(scorefile)[1][1:]

        # pick method to save scorefile
        if (save_method_name := FORMAT_STORAGE_DICT.get(storage_method.lower())):
            getattr(scores, save_method_name)(scorefile)
        else:
            raise KeyError(f"Could not find method to save scorefile as {storage_method}. Make sure the score file extension is correct!")
        
    class CrashError(RuntimeError):
        """Re-raised error with job stderr context when collect_scores fails."""

    @staticmethod
    def _came_from_collect_scores(exc: BaseException) -> bool:
        "checks if an exception came from the function collect_scores"
        tb = exc.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name == "collect_scores":
                return True
            tb = tb.tb_next
        return False

    @staticmethod
    def _wrap_run_with_stderr_context(fn):
        "catches exceptions in runners and returns the stderr output log of the job"
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                # Only enrich if failure originated in collect_scores(...)
                if Runner._came_from_collect_scores(e) or isinstance(e, ProcessError): # ProcessError for LocalJobstarter
                    js = getattr(self, "current_jobstarter", None)
                    tail = getattr(js, "last_error_message", "") if js else ""
                    msg = f"{self.__class__.__name__}.collect_scores failed: {e!r}"
                    if tail:
                        msg += "\n\n=== JOB ERROR OUTPUT ===\n" + tail
                    err = Runner.CrashError(msg)
                    err.__cause__ = e
                    raise err
                # Otherwise, re-raise the original exception unchanged
                raise
        return wrapped

    def __init_subclass__(cls, **kwargs):
        "overwrites subclasses to check for exceptions"
        super().__init_subclass__(**kwargs)
        # Auto-wrap any subclass that defines/overrides run()
        run_fn = cls.__dict__.get("run")
        if callable(run_fn):
            setattr(cls, "run", Runner._wrap_run_with_stderr_context(run_fn))

def parse_generic_options(options: str, pose_options: str, sep="--") -> tuple[dict,list]:
    """
    Parses generic options and pose-specific options from two input strings, combining them into a single dictionary of options
    and a list of flags. Pose-specific options overwrite generic options in case of conflicts. Options are expected to be separated
    by a specified separator within each input string, with options and their values separated by spaces.

    Parameters:
    -----------
    options : str 
        A string of generic options, where different options are separated by the specified separator and each option's value (if any) is separated by space.
    pose_options : str 
        A string of pose-specific options, formatted like the `options` parameter. These options take precedence over generic options.
    sep : str, optional
        The separator used to distinguish between different options in both input strings. Defaults to "--".

    Returns:
    --------
    tuple 
        A 2-element tuple where the first element is a dictionary of merged options (key-value pairs) and the second element is a list of unique flags (options without values) from both input strings.

    Examples:
    ---------
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

def col_in_df(df: pd.DataFrame, column: str) -> None:
    """
    Checks if a column exists in a DataFrame.

    This function verifies whether a specified column is present in the given DataFrame.
    If the column is not found, it raises a KeyError.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be checked.
    column : str
        The name of the column to be verified.

    Raises
    ------
    KeyError
        If the specified column is not found in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> col_in_df(df, 'A')  # No error raised
    >>> col_in_df(df, 'C')  # Raises KeyError
    Traceback (most recent call last):
        ...
    KeyError: 'Could not find C in poses dataframe! Are you sure you provided the right column name?'
    """
    if not column in df.columns:
        raise KeyError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")

def expand_options_flags(options_str: str, sep:str="--") -> tuple[dict, set]:
    """
    Simple parsing function to parse options and flags from an input string.

    Splits an input string into options and flags only based on a specified separator!
    If your command has more complex patterns in its options, then switch to "regex_expand_options_flags". 
    Options are key-value pairs, while flags are standalone keys without values.

    Parameters
    ----------
    options_str : str
        The input string containing options and flags to be parsed.
    sep : str, optional
        The separator used to distinguish different options and flags (default is "--").

    Returns
    -------
    tuple[dict, set]
        A tuple containing a dictionary of options and a set of flags.

    Examples
    --------
    >>> options_str = "--width 800 --height 600 --verbose"
    >>> opts, flags = expand_options_flags(options_str)
    >>> print(opts)
    {'width': '800', 'height': '600'}
    >>> print(flags)
    {'verbose'}

    >>> options_str = "--color=blue --debug --timeout=30"
    >>> opts, flags = expand_options_flags(options_str)
    >>> print(opts)
    {'color': 'blue', 'timeout': '30'}
    >>> print(flags)
    {'debug'}
    """
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
    """
    Parses options and flags from an input string using regular expressions.

    This function uses regular expressions to split an input string into options and flags. 
    It ensures that separators within quotes are not split.

    Parameters
    ----------
    options_str : str
        The input string containing options and flags to be parsed.
    sep : str, optional
        The separator used to distinguish different options and flags (default is "--").

    Returns
    -------
    tuple[dict, set]
        A tuple containing a dictionary of options and a set of flags.

    Examples
    --------
    >>> options_str = '--width 800 --height 600 --verbose'
    >>> opts, flags = regex_expand_options_flags(options_str)
    >>> print(opts)
    {'width': '800', 'height': '600'}
    >>> print(flags)
    {'verbose'}

    >>> options_str = '--color="dark blue" --debug --timeout=30'
    >>> opts, flags = regex_expand_options_flags(options_str)
    >>> print(opts)
    {'color': 'dark blue', 'timeout': '30'}
    >>> print(flags)
    {'debug'}
    """
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

def options_flags_to_string(options: dict, flags: list, sep="--", no_quotes: bool = False) -> str:
    """
    Converts options dictionary and flags list into a single string.

    This function combines a dictionary of options and a list of flags into a single command-line style string.

    Parameters
    ----------
    options : dict
        A dictionary of options, where keys are option names and values are option values.
    flags : list
        A list of flags (standalone options without values).
    sep : str, optional
        The separator used to distinguish different options and flags (default is "--").
    no_quotes: bool, optional
        (default: False) Setting this option to True will disable the quoting of commandline arguments that are separated by whitespaces.
        For example, if your option is "--my_list='1 4 6 14'" then you'd want your list quoted. 
        setting no_quotes=True would result in "--my_list=1 4 6 14", which can cause errors. 

    Returns
    -------
    str
        A string representation of the combined options and flags.

    Examples
    --------
    >>> options = {'width': '800', 'height': '600'}
    >>> flags = ['verbose', 'debug']
    >>> options_flags_to_string(options, flags)
    " --width=800 --height=600 --verbose --debug"

    >>> options = {'color': 'dark blue', 'timeout': '30'}
    >>> flags = ['force']
    >>> options_flags_to_string(options, flags)
    " --color='dark blue' --timeout=30 --force"
    """
    def value_in_quotes(value) -> str:
        '''Makes sure that split commandline options are passed in quotes: --option='quoted list of args' '''
        if len(str(value).split(" ")) > 1:
            if not ((value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"'))):
                return f"'{value}'"
        return value

    # assemble options
    out_str = " " + " ".join([f"{sep}{key}={value if no_quotes else value_in_quotes(value)}" for key, value in options.items()]) if options else ""

    # if flags are present, assemble those too and return
    if flags and len(flags) >= 1:
        flags = [f" {sep}{flag}" for flag in flags if flag != ""]  # removes any empty strings "" from flags (this causes empty "--" to appear in commands causing crashes)
        out_str += "".join(flags)
    return out_str

def prepend_cmd(cmds: list[str], pre_cmd: str) -> list[str]:
    """
    Prepends a single command to all commands in a list.

    Parameters
    ----------
    cmds : list[str]
        A list of commands, where all elements are strings.
    pre_cmd : str
        A string containing a command, which should be prepended to all commands in the commands list.

    Returns
    -------
    list[str]
        A list of all commands with the additional command prepended to each.

    Examples
    --------
    >>> cmds = [run_inference.sh pose_0001.pdb, run_inference.sh pose_0002.pdb]
    >>> pre_cmd = "conda init"
    >>> prepend_cmd(cmds, pre_cmd)
    "['conda init; run_inference.sh pose_0001.pdb', 'conda init; run_inference.sh pose_0002.pdb']"
    """
    cmds = ["; ".join([pre_cmd, cmd]) for cmd in cmds]
    return cmds
