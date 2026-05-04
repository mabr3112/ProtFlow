"""
Generic metric runner for ProtFlow.

This module exposes :class:`GenericMetric`, a lightweight :class:`protflow.runners.Runner`
that executes any importable Python function over the poses stored in a
:class:`protflow.poses.Poses` object. The target function must accept a single
pose path as its first positional argument and return a JSON-serializable value.
Additional keyword arguments can be forwarded through the runner's ``options``
dictionary.

How it works
------------
``GenericMetric.run()`` resolves the working directory and jobstarter, splits
``poses.poses_list()`` into manageable chunks, and starts one worker command
per chunk. Each worker re-enters this module as a small CLI program, imports
the requested module and function dynamically, evaluates the function on every
pose path in its chunk, and stores the results as JSON. The parent process then
concatenates the worker outputs and merges them back into ``poses.df`` through
``RunnerOutput``.

Walkthrough
-----------
The example below calculates the radius of gyration for every pose by reusing
``protflow.utils.metrics.calc_rog_of_pdb``:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.metrics.generic_metric_runner import GenericMetric

    poses = Poses(
        poses=["/data/designs/design_0001.pdb", "/data/designs/design_0002.pdb"],
        work_dir="/data/protflow_runs"
    )
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)

    rog = GenericMetric(
        module="protflow.utils.metrics",
        function="calc_rog_of_pdb",
        options={"chain": "A"},
        jobstarter=cpu_jobstarter,
    )

    poses = rog.run(poses=poses, prefix="rog")

    # GenericMetric stores the returned value in <prefix>_data.
    print(poses.df[["poses_description", "rog_data"]])

In that run, ``GenericMetric`` will:

1. Build ``/data/protflow_runs/rog`` as its working directory.
2. Split the input pose paths into chunks based on ``max_cores`` and a hard
   limit of 100 poses per command.
3. Launch worker commands that call ``calc_rog_of_pdb(pose_path, chain="A")``.
4. Save intermediate JSON files such as ``out_0.json``.
5. Merge the combined results back into ``poses.df`` as
   ``rog_data``, ``rog_description``, and ``rog_location``.

This module is intended for simple, embarrassingly parallel per-pose metrics.
If your function needs multiple inputs, non-JSON output, or a richer output
schema than a single ``data`` column, a dedicated runner is usually a better
fit.
"""

# import general
import os
import json
import logging
import importlib

# import dependencies
import pandas as pd

# import customs
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput
from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter, split_list

class GenericMetric(Runner):
    """
    Run a simple Python metric function over every pose in a :class:`Poses`.

    ``GenericMetric`` is the most lightweight metric runner in ProtFlow. You
    point it at an importable module and a function name, optionally provide a
    shared ``options`` dictionary, and the runner takes care of chunking the
    pose list, dispatching jobs through a :class:`JobStarter`, collecting the
    JSON outputs, and merging the results back into ``poses.df``.

    The target function contract is intentionally small:

    - The first positional argument must be the pose path.
    - Optional keyword arguments can be supplied via ``options``.
    - The return value must be serializable to JSON.

    The resulting metric value is stored in ``<prefix>_data`` after the run is
    merged back into ``poses.df``.
    """
    def __init__(self, python_path: str|None = None, module: str = None, function: str = None, options: dict = None, jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
        """
        Initialize a generic per-pose metric runner.

        Parameters
        ----------
        python_path : str | None, optional
            Python interpreter used to launch worker commands. If omitted, the
            interpreter from the configured ``PROTFLOW_ENV`` is used.
        module : str | None, optional
            Importable module path that contains the target metric function.
        function : str | None, optional
            Name of the function to call inside ``module``.
        options : dict | None, optional
            Keyword arguments forwarded to the target function for every pose.
        jobstarter : JobStarter | None, optional
            Default jobstarter used when ``run()`` is called without one.
        overwrite : bool, optional
            Whether existing runner scorefiles should be recomputed by default.
        """
        # setup config
        config = require_config()
        self.set_python_path(python_path or os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python"))

        # setup runner
        self.set_module(module)
        self.set_function(function)
        self.set_jobstarter(jobstarter)
        self.set_options(options)
        self.overwrite = overwrite

    def __str__(self):
        return "GenericMetric"

    ########################## Input ################################################
    def set_module(self, module: str) -> None:
        """
        Set the importable module path that contains the metric function.

        Parameters
        ----------
        module : str
            Importable module path, for example ``"protflow.utils.metrics"``.
        """
        self.module = module

    def set_python_path(self, python_path: str) -> None:
        """Set the Python interpreter used for worker execution."""
        self.python_path = python_path

    def set_function(self, function: str) -> None:
        """
        Set the function name to import from ``self.module``.

        Parameters
        ----------
        function : str
            Attribute name of the target metric function.
        """
        self.function = function

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """
        Set the default jobstarter for this runner instance.

        Parameters
        ----------
        jobstarter : JobStarter | None
            Jobstarter used when ``run()`` does not receive one explicitly.

        Raises
        ------
        ValueError
            If ``jobstarter`` is neither ``None`` nor a :class:`JobStarter`.
        """
        if isinstance(jobstarter, JobStarter) or jobstarter is None:
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")

    def set_options(self, options: dict) -> None:
        """
        Set shared keyword arguments for the metric function.

        Parameters
        ----------
        options : dict | None
            Keyword arguments forwarded as ``function(pose, **options)``.

        Raises
        ------
        ValueError
            If ``options`` is neither ``None`` nor a dictionary.
        """
        if isinstance(options, dict) or options is None:
            self.options = options
        else:
            raise ValueError(f"Parameter :options: must be of type dict. type(options= = {type(options)})")

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, python_path: str = None, module: str = None, function: str = None, options: dict = None, jobstarter: JobStarter = None, overwrite: bool = False) -> Poses:
        """
        Execute the configured metric function across all poses.

        Parameters
        ----------
        poses : Poses
            Input poses. ``GenericMetric`` reads the pose file paths from
            ``poses.df["poses"]``.
        prefix : str
            Prefix used for the runner work directory, cached scorefile, and
            merged result columns.
        python_path : str | None, optional
            Python interpreter used for worker commands. Defaults to the value
            configured on the runner instance.
        module : str | None, optional
            Importable module path for the metric function. Defaults to the
            value configured on the runner instance.
        function : str | None, optional
            Function name inside ``module``. Defaults to the value configured on
            the runner instance.
        options : dict | None, optional
            Shared keyword arguments forwarded to the metric function. Defaults
            to the value configured on the runner instance.
        jobstarter : JobStarter | None, optional
            Jobstarter used for this invocation. Resolution priority is
            ``run(jobstarter)`` -> ``self.jobstarter`` ->
            ``poses.default_jobstarter``.
        overwrite : bool, optional
            If ``True``, recompute the metric even when the cached scorefile
            already exists.

        Returns
        -------
        Poses
            The input ``Poses`` instance with additional columns such as
            ``<prefix>_data``, ``<prefix>_description``, and
            ``<prefix>_location`` merged into ``poses.df``.

        Raises
        ------
        ValueError
            If ``options`` is not a dictionary or if no usable jobstarter is
            available.
        RuntimeError
            If fewer output rows are collected than input poses, which usually
            indicates failed worker jobs.

        Examples
        --------
        .. code-block:: python

            from protflow.metrics.generic_metric_runner import GenericMetric

            rog = GenericMetric(
                module="protflow.utils.metrics",
                function="calc_rog_of_pdb",
                options={"chain": "A"},
            )

            poses = rog.run(poses=poses, prefix="rog", jobstarter=cpu_jobstarter)

        Notes
        -----
        Internally, ``run()`` launches this module as a worker script for each
        pose chunk. Each worker writes a JSON file with the columns ``data``,
        ``description``, and ``location``. The parent process concatenates
        those files and lets :class:`RunnerOutput` merge the final table back
        into ``poses.df``.
        """
        # if self.atoms is all, calculate Allatom RMSD.

        # prep variables
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        python_path = python_path or self.python_path
        module = module or self.module
        function = function or self.function
        options = options or self.options
        if not (isinstance(options, dict) or options is None):
            raise ValueError(f"Parameter :options: must be of type dict. type(options= = {type(options)})")

        logging.info(f"Running metric {function} of module {module} in {work_dir} on {len(poses.df.index)} poses.")

        scorefile = os.path.join(work_dir, f"{prefix}_{function}_generic_metric.{poses.storage_format}")

        # check if RMSD was calculated if overwrite was not set.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # split poses into number of max_cores lists, but not more than 100 poses per sublist (otherwise, argument list too long error occurs)
        poses_sublists = split_list(poses.poses_list(), n_sublists=jobstarter.max_cores) if len(poses.df.index) / jobstarter.max_cores < 100 else split_list(poses.poses_list(), element_length=100)
        out_files = [os.path.join(poses.work_dir, prefix, f"out_{index}.json") for index, sublist in enumerate(poses_sublists)]
        cmds = [f"{python_path} {__file__} --poses {','.join(poses_sublist)} --out {out_file} --module {module} --function {function}" for out_file, poses_sublist in zip(out_files, poses_sublists)]
        if options:
            options_path = os.path.join(poses.work_dir, prefix, f"{prefix}_options.json")
            with open(options_path, "w", encoding="UTF-8") as f:
                json.dump(options, f)
            cmds = [f"{cmd} --options {options_path}" for cmd in cmds]

        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = f"{function}_generic_metric",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        scores = pd.concat([pd.read_json(output) for output in out_files]).reset_index(drop=True)
        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of generic metric runner with function {function} at {scorefile}.")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info(f"{function} completed. Returning scores.")
        return output.return_poses()


def main(args):
    """Worker entrypoint used by :meth:`GenericMetric.run`.

    The parent runner starts this module as a CLI script, passes a comma-
    separated list of pose paths plus the import target, and expects a JSON file
    containing ``data``, ``description``, and ``location`` columns.
    """
    input_poses = args.poses.split(",")

    # import function
    module_ = importlib.import_module(args.module)
    function = getattr(module_, args.function)

    # calculate data
    if args.options:
        with open(args.options, "r", encoding="UTF-8") as f:
            options = json.load(f)
        data = [function(pose, **options) for pose in input_poses]
    else:
        data = [function(pose) for pose in input_poses]
    description = [os.path.splitext(os.path.basename(pose))[0] for pose in input_poses]
    location = list(input_poses)

    # create results dataframe
    results = pd.DataFrame({"data": data, "description": description, "location": location})

    # save output
    results.to_json(args.out)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--poses", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--out", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--module", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--function", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--options", type=str, default=None, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")

    arguments = argparser.parse_args()
    main(arguments)
