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
import time
import os
import re
import functools
import subprocess
from datetime import datetime, timedelta
from multiprocessing import ProcessError

# dependencies
import pandas as pd

# custom
from .poses import Poses, get_format, FORMAT_STORAGE_DICT
from .jobstarters import JobStarter, SbatchArrayJobstarter, get_SLURM_stats

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

        Returns absolute path to working directory and the jobstarter that will be used for the runner.

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

class SbatchArrayRunnerTimer(Runner):
    """
    SbatchArrayRunnerTimer Class
    ============================
    
    Instrumentation wrapper that profiles any ProtFlow Runner on SLURM.
 
    :class:`SbatchArrayRunnerTimer` wraps an arbitrary
    :class:`~protflow.runners.Runner` instance and, after each call to
    :meth:`run`, queries SLURM's accounting database via
    :func:`get_SLURM_stats` to collect per-job resource statistics.  All
    timing and statistics records are accumulated in :attr:`history` and can
    be exported at any time via :meth:`report`.
 
    The class inherits from :class:`~protflow.runners.Runner` and uses
    :meth:`__getattr__` to transparently proxy every attribute lookup to the
    wrapped runner, so it can serve as a drop-in replacement in any ProtFlow
    pipeline without modifying the surrounding code.
 
    .. warning::
 
       Profiling relies on :func:`get_SLURM_stats`, which calls ``sacct``
       and therefore requires the process to be running on the **cluster
       login node**.  See :func:`get_SLURM_stats` for details.
 
    Parameters
    ----------
    runner : Runner
        Any instantiated ProtFlow :class:`~protflow.runners.Runner`
        (e.g. :class:`~protflow.runners.caliby.CalibySequenceDesign`,
        :class:`~protflow.runners.ligandmpnn.LigandMPNN`, etc.) whose
        :meth:`run` calls should be timed and profiled.
 
    Attributes
    ----------
    runner : Runner
        The wrapped runner instance.
    history : list of dict
        Accumulated statistics records.  Each entry corresponds to one
        successfully profiled :meth:`run` call and contains all keys
        returned by :func:`get_SLURM_stats` plus the four keys added by
        :meth:`run` (``runner_class``, ``prefix``,
        ``total_python_wall_sec``, ``overhead_plus_queue_sec``).  Empty
        until the first successful profiled run completes.
    job_ids : list of str or None
        SLURM job names recorded for each :meth:`run` call, in call order.
        An entry of ``None`` indicates that
        :attr:`~protflow.jobstarters.SbatchArrayJobstarter.last_job_name`
        could not be retrieved (e.g. because a non-SLURM jobstarter was
        used and the guard did not fire before the append).
    session_start : str
        ISO-8601 timestamp (``YYYY-MM-DDTHH:MM:SS``) set at construction
        time to one minute before instantiation.  Passed as *start_time*
        to every :func:`get_SLURM_stats` call so that only jobs from the
        current session are returned by ``sacct``, preventing false matches
        against stale jobs with the same name from earlier sessions.
 
    Notes
    -----
    * ``__init__`` calls ``super().__init__()`` to satisfy the
      :class:`~protflow.runners.Runner` base-class contract, making all
      base-class utilities (e.g. scorefile helpers) available on ``self``
      in addition to the wrapped ``self.runner``.
    * :attr:`session_start` is backdated by one minute to guard against
      off-by-one errors on clusters with coarse ``sacct`` timestamp
      resolution.
    * :attr:`history` grows unboundedly across :meth:`run` calls within
      the same Python session.  For very long pipelines, consider calling
      :meth:`report` periodically and resetting ``self.history = []`` if
      memory usage is a concern.
 
    Examples
    --------
    Wrap a LigandMPNN runner and time three sequential design rounds::
 
        from protflow.runners.ligandmpnn import LigandMPNN
        from protflow.runners.sbatch_array_runner_timer import SbatchArrayRunnerTimer
 
        timed_runner = SbatchArrayRunnerTimer(LigandMPNN())
 
        for prefix in ["round1", "round2", "round3"]:
            poses = timed_runner.run(poses, prefix=prefix, nseq=20)
 
        summary = timed_runner.report(prefix="full_pipeline")
        print(summary[["prefix", "total_python_wall_sec", "avg_task_runtime_sec"]])
    """


    def __init__(self, runner: Runner):
        super().__init__()
        self.runner = runner
        self.history = []
        self.job_ids = []
        # session_start for sacct filtering
        self.session_start = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, **kwargs) -> Poses:
        """
        Execute the wrapped runner and collect timing and SLURM statistics.
 
        Delegates the actual computation to :attr:`runner` via
        ``self.runner.run(poses, prefix, jobstarter, **kwargs)`` and then,
        if a :class:`~protflow.jobstarters.SbatchArrayJobstarter` was used,
        queries SLURM's accounting database for per-job resource statistics
        using :func:`get_SLURM_stats`.  The combined timing and cluster stats
        record is appended to :attr:`history` and :meth:`report` is called
        automatically to persist an up-to-date CSV and job-ID file.
 
        The method measures time across three consecutive phases:
 
        1. **Phase 1 — wrapper start**: ``time.perf_counter()`` is captured
           immediately before delegating to the wrapped runner.
        2. **Phase 2 — runner execution**: the full body of
           ``self.runner.run()``, which internally performs ProtFlow setup,
           submits the SLURM array job, blocks until all tasks complete
           (``wait=True``), and post-processes the results.
        3. **Phase 3 — wrapper end**: ``time.perf_counter()`` is captured
           immediately after the wrapped runner returns.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection, forwarded verbatim to
            ``self.runner.run``.
        prefix : str
            Column prefix and working-directory identifier forwarded to
            ``self.runner.run`` and used to name the output CSV and job-ID
            files written by :meth:`report`.
        jobstarter : JobStarter, optional
            Job submission backend.  When provided, this value is passed to
            the wrapped runner and is also used to determine whether SLURM
            accounting can be queried.  When omitted, the jobstarter is
            resolved from ``self.runner.jobstarter`` and then from
            ``poses.default_jobstarter`` for the purpose of stat collection.
        **kwargs
            All additional keyword arguments are forwarded unchanged to
            ``self.runner.run``, making the timer fully compatible with any
            runner regardless of its specific signature.
 
        Returns
        -------
        Poses
            The :class:`~protflow.poses.Poses` object returned by the
            wrapped runner, unchanged.  Timing and statistics are stored in
            :attr:`history` and written to disk by :meth:`report`; they do
            not alter the returned poses.
 
        Side Effects
        ------------
        When profiling succeeds (SLURM jobstarter detected and
        ``last_job_name`` is set), the following side effects occur:
 
        * **5-second sleep** inserted via ``time.sleep(5)`` to allow the
          SLURM accounting database to synchronise before ``sacct`` is
          queried.
        * A statistics dictionary is **appended to** :attr:`history`.  The
          dictionary contains all keys from :func:`get_SLURM_stats` (see its
          return-value documentation) plus the following four keys added by
          this method:
 
          ``runner_class`` : str
              ``__class__.__name__`` of the wrapped runner
              (e.g. ``"CalibySequenceDesign"``).
          ``prefix`` : str
              The *prefix* argument passed to this call.
          ``total_python_wall_sec`` : float
              Total elapsed wall-clock time in seconds from Python's
              perspective (Phase 1 → Phase 3), rounded to 2 decimal places.
              Encompasses ProtFlow setup, SLURM queue wait, cluster
              execution, and result post-processing.
          ``overhead_plus_queue_sec`` : float
              ``total_python_wall_sec`` minus ``runtime_sec`` from SLURM,
              rounded to 2 decimal places.  Approximates the combined cost
              of ProtFlow overhead and scheduler queue wait.  May be
              negative in rare cases due to clock skew between the login
              node and compute nodes, or rounding in ``sacct``.
 
        * The SLURM job name is **appended to** :attr:`job_ids`.
        * ``<prefix>_stats.csv`` and ``<prefix>_job_ids.txt`` are written
          (or overwritten) in the current working directory via
          :meth:`report`.
 
        Warns
        -----
        logging.WARNING
            Emitted when the resolved jobstarter is not an instance of
            :class:`~protflow.jobstarters.SbatchArrayJobstarter`.
            Message format: ``"Stats skipped: <type> does not support SLURM
            accounting."``.  Profiling is skipped entirely and the
            unmodified poses are returned immediately.
 
        Notes
        -----
        * The jobstarter resolution priority is:
          *argument* → ``self.runner.jobstarter`` → ``poses.default_jobstarter``.
          This mirrors the fallback chain used by most ProtFlow runners and
          ensures that the correct jobstarter is identified for stat
          collection even when it was set on the runner at construction time.
        * ``total_python_wall_sec`` includes SLURM queue wait time because
          the wrapped runner calls
          :meth:`~protflow.jobstarters.SbatchArrayJobstarter.start` with
          ``wait=True``, blocking until all array tasks complete before
          returning.
        * If ``last_job_name`` is ``None`` (e.g. the jobstarter was never
          used to submit a job), the stats-collection block is skipped
          entirely and :attr:`history` is not updated, even though the
          jobstarter type check passes.
 
        Examples
        --------
        Basic timed run::
 
            timed = SbatchArrayRunnerTimer(CalibySequenceDesign())
            poses = timed.run(
                poses,
                prefix="sd_round1",
                nseq=10,
                jobstarter=SbatchArrayJobstarter(max_cores=50),
            )
            print(timed.history[-1]["total_python_wall_sec"])    # e.g. 312.45
            print(timed.history[-1]["overhead_plus_queue_sec"])  # e.g.  18.72
            print(timed.history[-1]["runner_class"])             # "CalibySequenceDesign"
            print(timed.history[-1]["state"])                    # "COMPLETED"
 
        Passing runner-specific kwargs transparently::
 
            timed = SbatchArrayRunnerTimer(LigandMPNN())
            poses = timed.run(
                poses,
                prefix="mpnn",
                nseq=20,
                model_type="ligand_mpnn",
                fixed_residues_col="binding_site",
            )
 
        Non-SLURM jobstarter (profiling skipped, poses still returned)::
 
            from protflow.jobstarters import LocalJobStarter
            poses = timed.run(poses, prefix="local_test", jobstarter=LocalJobStarter())
            # Logs: WARNING - Stats skipped: <class 'LocalJobStarter'>
            #                 does not support SLURM accounting.
            # timed.history is unchanged.
        """

        # --- PHASE 1: START SETUP ---
        t_start_wrapper = time.perf_counter()

        # Execute the wrapped runner
        # Note: The 'setup' happens inside .run() before the job is submitted.
        # The 'wait' happens inside .run() after submission.
        poses = self.runner.run(poses=poses, prefix=prefix, jobstarter=jobstarter, **kwargs)

        # --- PHASE 2: END POST-PROCESSING ---
        t_end_wrapper = time.perf_counter()
        
        # Calculate Total Wall Time (Python's perspective)
        total_wall_runtime = t_end_wrapper - t_start_wrapper

        # --- PHASE 3: QUERY CLUSTER STATS ---
        used_starter = jobstarter or getattr(self.runner, "jobstarter", None) or getattr(poses, "default_jobstarter", None)

        # Safety Check: Only proceed with timing if it's the correct SLURM starter
        if not isinstance(used_starter, SbatchArrayJobstarter):
            logging.warning(f"Stats skipped: {type(used_starter)} does not support SLURM accounting.")
            return poses
        
        target_job_name = getattr(used_starter, "last_job_name", None)
        self.job_ids.append(target_job_name)

        if target_job_name:
            # Wait for SLURM DB sync
            time.sleep(5)
            stats = get_SLURM_stats(target_job_name, self.session_start)
            
            # Decompose the time
            # Note: total_wall_runtime includes (Setup + Queue Wait + Cluster Run + Post-processing)
            # real_runtime_sec is just (Cluster Run)
            
            stats.update({
                "runner_class": self.runner.__class__.__name__,
                "prefix": prefix,
                "total_python_wall_sec": round(total_wall_runtime, 2),
                "overhead_plus_queue_sec": round(total_wall_runtime - stats.get("runtime_sec", 0), 2)
            })
            self.history.append(stats)
            self.report(prefix=prefix)
        
        return poses

    def report(self, prefix:str=None):
        """Export accumulated timing and SLURM statistics to disk and return as a DataFrame.
 
        Converts :attr:`history` to a :class:`~pandas.DataFrame` and, when
        *prefix* is provided, writes two files to the current working
        directory:
 
        * ``<prefix>_stats.csv`` — the full statistics table, one row per
          profiled :meth:`run` call, written with
          :meth:`~pandas.DataFrame.to_csv` (index column included).
        * ``<prefix>_job_ids.txt`` — a newline-delimited list of all SLURM
          job names from :attr:`job_ids`, in the order the runs were
          performed.
 
        Parameters
        ----------
        prefix : str, optional
            Filename stem for the output files.  When ``None``, no files are
            written and only the in-memory DataFrame is returned.  When
            provided, both output files are created or overwritten in the
            current working directory.
 
        Returns
        -------
        pandas.DataFrame
            DataFrame built from :attr:`history`, with one row per profiled
            :meth:`run` call.  Columns are the union of all keys present in
            :attr:`history` entries.  Guaranteed columns (when at least one
            profiled run has completed) include:
 
            ``runner_class`` : str
                Class name of the wrapped runner for that run.
            ``prefix`` : str
                The *prefix* used in that :meth:`run` call.
            ``total_python_wall_sec`` : float
                Total Python wall-clock time for that run (seconds).
            ``overhead_plus_queue_sec`` : float
                Estimated overhead + queue-wait time (seconds).
            ``job_name`` : str
                SLURM job name queried by :func:`get_SLURM_stats`.
            ``total_cpu_sec`` : int
                Total CPU-core-seconds reserved across all tasks.
            ``avg_task_runtime_sec`` : float
                Mean per-task wall-clock elapsed time (seconds).
            ``max_task_runtime_sec`` : int
                Longest per-task wall-clock elapsed time (seconds).
            ``min_task_runtime_sec`` : int
                Shortest per-task wall-clock elapsed time (seconds).
            ``num_tasks`` : int
                Number of SLURM array tasks.
            ``total_cpus_reserved`` : int
                Total CPU cores allocated across all tasks.
            ``state`` : str
                Aggregated job-array completion state.
            ``queried_after`` : str or None
                The ``sacct`` start-time filter used for that query.
 
            Returns an **empty** :class:`~pandas.DataFrame` when
            :attr:`history` is empty (i.e. before any profiled run has
            completed, or when all runs used a non-SLURM jobstarter).
 
        Notes
        -----
        * :meth:`report` is called automatically at the end of every
          successful profiled :meth:`run` call using that run's *prefix*,
          so the CSV and job-ID files are always up to date after each run.
          Manual calls to :meth:`report` are useful for retrieving an in-
          memory summary or writing a consolidated report under a different
          prefix after multiple runs.
        * The job-ID file is written from :attr:`job_ids` (not from the
          ``job_name`` column of :attr:`history`), which means it includes
          entries from runs where ``last_job_name`` was ``None`` or where
          the non-SLURM guard fired before the append.  ``None`` values will
          appear as the literal string ``"None"`` in the file.
        * Output files are written with UTF-8 encoding and will overwrite
          existing files of the same name without prompting.
 
        Examples
        --------
        Inspect stats after two runs and write a combined report::
 
            timed = SbatchArrayRunnerTimer(CalibySequenceDesign())
            poses = timed.run(poses, prefix="round1", nseq=5)
            poses = timed.run(poses, prefix="round2", nseq=10)
 
            df = timed.report(prefix="pipeline_summary")
            # Writes:
            #   pipeline_summary_stats.csv
            #   pipeline_summary_job_ids.txt
            print(df[["prefix", "total_python_wall_sec", "avg_task_runtime_sec"]])
            #      prefix  total_python_wall_sec  avg_task_runtime_sec
            # 0    round1                 245.12                228.40
            # 1    round2                 510.87                491.33
 
        In-memory summary without writing files::
 
            df = timed.report()   # prefix=None — no files written
            print(df[["state", "num_tasks", "total_cpu_sec"]].to_string())
        """

        jobs = self.job_ids
        df = pd.DataFrame(self.history)
        if prefix:
            with open(f"{prefix}_job_ids.txt", "w", encoding="UTF-8") as f:
                f.write("\n".join(jobs))
            df.to_csv(f"{prefix}_stats.csv")
        return df