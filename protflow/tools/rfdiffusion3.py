"""
This module provides the ProtFlow runner implementation for RFDiffusion3.
It enables structured, automated execution of RFDiffusion3 within the
ProtFlow framework, including input specification, job execution,
result collection, and optional motif remapping.

Detailed Description
--------------------
The `RFdiffusion3` class extends the generic ProtFlow `Runner` base class
to support the RFDiffusion3 application. It manages:

- Automatic construction of required input JSON files
- Command-line interface (CLI) argument assembly
- Job submission via ProtFlow's JobStarter abstraction
- Collection and flattening of output scores
- Optional residue motif remapping based on diffused index maps
- Safe reuse of cached scorefiles
- Multiplexing of poses for parallel GPU utilization
- De novo design by prodividing an empty input pose and output length 
  as well as motif scaffolding by providing an input .pdb file and
  unindexed residues according to the RFD3 documentation.

RFDiffusion3 output structures are written as `.cif.gz` files with
accompanying sidecar `.json` files containing metrics, specification
data, and diffused index maps. These are parsed and flattened into
a structured pandas DataFrame for integration into ProtFlow.

Usage
-----
To use this module, instantiate the `RFdiffusion3` runner and call
its `run()` method with appropriate arguments.

Examples
--------
Example usage within a ProtFlow workflow:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from rfdiffusion3 import RFdiffusion3

    poses = Poses()
    jobstarter = JobStarter()

    runner = RFdiffusion3()

    results = runner.run(
        poses=poses,
        prefix="rfd3_experiment",
        settings_group_name="rfd3",
        n_batches=2,
        diffusion_batch_size=8,
        multiplex_poses=4,
        overwrite=True
    )

    print(results)

Further Details
---------------
- JSON Construction: The input JSON file required by RFDiffusion3
  is always built internally from the arguments supplied to `run()`.
  Providing a pre-existing JSON file is not supported. Supported
  arguments can be parsed with the same parameter names as described 
  in the RFdiffusion3 documentation.
- Dynamic Index Handling: The number of index layers used during
  pose merging is inferred automatically from `settings_group_name`.
- Score Flattening: Nested metrics in sidecar JSON files are
  flattened recursively.
- Heavy Data Filtering: Large per-residue arrays are excluded
  by default unless explicitly requested via `include_scores`.
- Multiplexing: Input poses can be duplicated to maximize
  GPU utilization. Index layers are collapsed after completion.
- Robustness: Optional failure detection ensures missing
  outputs are caught early.

Notes
-----
This module is part of the ProtFlow package. The code was built 
based on previously created ProtFlow code and a runner-template 
provided by Markus Braun. ProtFlow is designed
to work in HPC environments with job schedulers.

Authors
-------
Sigrid Kaltenbrunner

Version
-------
0.1.0
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob
import gzip
import shutil
from collections import UserDict
from pathlib import Path
import re

import pandas as pd

from protflow import load_config_path, require_config
from ..jobstarters import JobStarter, split_list
from ..poses import Poses, col_in_df, description_from_path
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..utils.openbabel_tools import openbabel_fileconverter
from ..residues import ResidueSelection, parse_residue

class RFD3InputSpecification(UserDict):
    def __init__(self, poses: Poses, spec_from_json: str = None, spec_from_dict : dict | RFD3InputSpecification = None):
        self.poses = poses

        if spec_from_json and spec_from_dict:
            raise ValueError("<spec_from_json> and <spec_from_dict> are mutually exclusive!")
        
        if spec_from_json:
            self.spec_from_json(spec_from_json)

        elif spec_from_dict:
            self._check_specs(spec_from_dict)
            self.data = spec_from_dict
        else:
            self.data = self._create_pose_dict(poses)

    @property
    def input_specs(self):
        return self.data

    def set_RFD3_input_specs(self,
            contig: str | dict = None,
            unindex: str | dict = None,
            length: str = None,
            ligand: str = None,
            cif_parser_args: dict = None,
            extra: dict = None,
            dialect: int = None,
            select_fixed_atoms: str | dict = None,
            select_unfixed_sequence: str | dict = None,
            select_buried: str | dict = None,
            select_partially_buried: str | dict = None,
            select_exposed: str | dict = None,
            select_hbond_donor: str | dict = None,
            select_hbond_acceptor: str | dict = None,
            select_hotspots: str | dict = None,
            redesign_motif_sidechains: bool = None,
            symmetry = None,
            ori_token: list[float] = None,
            infer_ori_strategy: str = None,
            plddt_enhanced: bool = None,
            is_non_loopy: bool | None = None,
            partial_t: float = None,
            **kwargs
            )  -> RFD3InputSpecification:
                
        # 1. Capture all local variables (including arguments and kwargs)
        params = locals().copy()
        
        exclude = {'self', 'kwargs'}
        
        # 3. Build the dictionary: exclude the blacklist and filter out None
        spec_dict = {
            k: v for k, v in params.items() 
            if k not in exclude and v is not None
        }

        if kwargs:
            spec_dict.update(kwargs)

        for pose in self.input_specs:
            self.data[pose].update(spec_dict)
        
        return self
    
    def set_per_pose_RFD3_input_specs(self,
            contig: str | list = None,
            unindex: str | list = None,
            length: str | list = None,
            ligand: str | list = None,
            cif_parser_args: str | list = None,
            extra: str | list = None,
            dialect: str | list = None,
            select_fixed_atoms: str | list = None,
            select_unfixed_sequence: str | list = None,
            select_buried: str | list = None,
            select_partially_buried: str | list = None,
            select_exposed: str | list = None,
            select_hbond_donor: str | list = None,
            select_hbond_acceptor: str | list = None,
            select_hotspots: str | list = None,
            redesign_motif_sidechains: str | list = None,
            symmetry: str | list = None,
            ori_token: str | list = None,
            infer_ori_strategy: str | list = None,
            plddt_enhanced: str | list = None,
            is_non_loopy: str | list = None,
            partial_t: str | list = None,
            **kwargs
            ) -> RFD3InputSpecification:
        
        if not self.poses:
            raise ValueError("Per-pose input specifications cannot be set on empty poses!")

        # 1. Capture all local variables (including arguments and kwargs)
        params = locals().copy()
        
        exclude = {'self', 'kwargs'}
        
        # 3. Build the dictionary: exclude the blacklist and filter out None
        spec_dict = {
            k: v for k, v in params.items() 
            if k not in exclude and v is not None
        }

        if kwargs:
            spec_dict.update(kwargs)

        # extract specs from poses df
        for key, val in spec_dict.items():
            if isinstance(val, str):
                col_in_df(self.poses.df, val)
                pose_specs = self.poses.df[val]
            elif isinstance(val, list):
                if not len(val) == len(self.poses):
                    raise ValueError(f"Length of input specifications for {val} ({len(val)}) does not match number of poses {len(self.poses)}!")
                pose_specs = val
            else:
                raise TypeError(f"Input must be a str indicating a poses dataframe column or a list, not {type(val)}!")

            for pose, spec in zip(self.poses.df["poses_description"], pose_specs):
                if pd.notna(spec): # only update if spec is specified for this pose
                    self.data[pose].update({key: spec})

        return self    
    
    def spec_from_dict(self, spec_dict: dict | RFD3InputSpecification) -> RFD3InputSpecification:
        self._check_specs(spec_dict)
        self.data = spec_dict
        return self
    
    def spec_from_json(self, json_path: str) -> RFD3InputSpecification:
        if not os.path.isfile(json_path):
            raise ValueError(f"Could not detect json file at {json_path}!")
        spec = read_json(json_path)
        self._check_specs(spec)
        self.data = spec
        return self
        
    def reset_pose_specs(self, poses: Poses) -> RFD3InputSpecification:
        self.poses = poses
        self.data = self._create_pose_dict(self.poses)
        return self

    def add_specs(self, additional_specs: RFD3InputSpecification | dict) -> RFD3InputSpecification:
        if self.poses:
            raise ValueError("Additional pose-specific input specifications can ony be added if no poses are present (unconditional diffusion)!")
        self.data.update(additional_specs)
        return self

    def modify_specs(self, new_specs: RFD3InputSpecification | dict) -> RFD3InputSpecification:
        # does not check for poses in case of multi-specs for unconditional diffusion
        if not all(pose in self.data for pose in new_specs) or not len(self.data) == len(new_specs):
            raise KeyError("Poses in <new_specs> do not match existing poses!")
        for pose in new_specs:
            self.data[pose].update(new_specs[pose])
        return self
    
    def _check_specs(self, specs: RFD3InputSpecification | dict):
        if self.poses and not all(pose in self.data for pose in specs) or not len(self.data) == len(specs):
            raise ValueError("Specs do not fit existing poses!")

    def _create_pose_dict(self, poses: Poses) -> dict:
        if poses:
            return {name: {"input": path} for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"])}
        else:
            return {"denovo": {}}

    
class RFdiffusion3(Runner):
    """    RFdiffusion3 Runner Class
    =========================

    The `RFdiffusion3` class provides a full ProtFlow runner
    implementation for RFDiffusion3.

    Detailed Description
    --------------------
    This class manages the complete lifecycle of an RFDiffusion3 run:

        - Automatic construction of per-pose input JSON files
        - Command assembly for the RFDiffusion3 CLI
        - Execution through a JobStarter
        - Parsing and flattening of output sidecar JSON files
        - Integration of results into the Poses DataFrame
        - Optional residue motif remapping
        - Optional multiplexing of poses for GPU scaling

    Important Implementation Details
    ---------------------------------
    1. JSON File Construction
       The input `.json` file required by RFDiffusion3 is always
       generated internally using `_write_input_json()`. Users
       cannot supply an already existing JSON file. All specification
       parameters must be passed directly to `run()`.

    2. Dynamic index_layers
       The attribute `self.index_layers` is inferred dynamically
       inside `run()` based on `settings_group_name`. The number
       of underscore-separated components in the group name affects
       how many trailing naming layers must be stripped to recover
       the original pose description.

    3. Output Format
       RFDiffusion3 outputs:

           <json_name>_<settings_group>_<batch_number>_model_<n>.cif.gz

       Each structure has a corresponding sidecar `.json` file
       containing:

           - metrics
           - specification
           - diffused_index_map entries

    4. Score Handling
       Heavy per-residue or multidimensional values are excluded
       by default to prevent bloated DataFrames. They can be
       selectively included via `include_scores`.

    5. Multiplexing
       If `multiplex_poses` is set (>1), poses are duplicated
       prior to execution and reindexed afterward to collapse
       duplication layers.

    Raises
    ------
    RuntimeError
        If no outputs are collected or expected outputs are missing.
    ValueError
        If motif remapping is requested but required index maps
        are missing.
        If de novo design is requested (no input .pdb file is provided)
        but length is not defined.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        config = require_config()

        self.application_path = application_path or load_config_path(config, "RFDIFFUSION3_SCRIPT_PATH")
        self.python_path = python_path or load_config_path(config, "RFDIFFUSION3_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "RFDIFFUSION3_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "rfdiffusion3"
        self.index_layers = 1

    def __str__(self) -> str:
        return self.name

    def run(
        self,
        prefix: str,
        poses: Poses,
        input_specification: RFD3InputSpecification,
        # parameters that go into input json files, see https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields
        # can be either a path to a json file, a dictionary, or a str indicating a dataframe column containing json paths or dicts
        # --- RFD3 CLI arguments ---
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        dump_trajectories: bool = False,
        # --- general ProtFlow parameters ---
        options: str = None,
        update_motifs: list[str] = None,
        multiplex_poses: int = None,
        jobstarter: JobStarter = None,
        convert_cif_to_pdb: bool = True,
        run_clean: bool = True,
        fail_on_missing_output_poses: bool = False,
        overwrite: bool = False,
    ) -> Poses:
        """Execute the full runner lifecycle and merge results into poses.

        Parameters
        ----------
        multiplex_poses : int | None
            If set, create this many copies of each input pose before running
            diffusion. Useful to fully utilize parallel GPUs when you have
            fewer input poses than GPUs. After the run, poses are reindexed
            back to remove the duplication layer. Must be > 1 to have any
            effect.

            Example: 1 input pose + multiplex_poses=8 + diffusion_batch_size=8
            gives 8 parallel jobs each producing 8 outputs = 64 total outputs.
        fail_on_missing_output_poses : bool
            If True, raise a RuntimeError when the number of collected output
            poses is less than the expected number (n_poses * n_batches *
            diffusion_batch_size). RFDiffusion3 runs occasionally crash silently,
            and enabling this flag ensures such failures are caught early rather
            than propagating through a longer pipeline. Defaults to False.
        """

        if poses and not all(name in input_specification for name in poses.df["poses_description"]) or not len(poses) == len(input_specification):
            raise ValueError("Input <poses> do not match <input_specification>")
        
        # Warn if multiplex_poses=1 since it has no effect.
        if multiplex_poses == 1:
            logging.warning("multiplex_poses=1 has no effect. Set to None or an integer > 1.")
        
        if not multiplex_poses:
            multiplex_poses = 1

        index_layers = self.index_layers + 2

        # 1) Generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )

        total_designs = n_batches * diffusion_batch_size * multiplex_poses
        logging.info(
            f"Total designs per input pose: {total_designs}\n({n_batches} batches x {diffusion_batch_size} per batch)"
        )
        if multiplex_poses > 1:
            logging.info(f"and multiplexing input poses {multiplex_poses} times.")
            index_layers += 1
            suffixes = [f"_{str(i).zfill(4)}" for i in range(1, multiplex_poses +1)]

            # multiplex and add an index layer to each input so that filenames are unique
            pose_specs = [
                {f"{pose}{sfx}": spec for pose, spec in input_specification.items()} 
                for sfx in suffixes
            ]

        else:
            # list of pose dicts
            pose_specs = [{pose: spec} for pose, spec in input_specification.items()]

        expected_outputs = n_batches * diffusion_batch_size * len(pose_specs)
        logging.info(f"Expected number of output poses: {expected_outputs}")

        # 2) Scorefile reuse shortcut.
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)

            if not poses:
                poses.df = scores.copy()
                poses.df["input_poses"] = None
                logging.info("De novo cached reuse: populated poses.df from scorefile.")
            else:
                poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=index_layers).return_poses()

                if update_motifs:
                    logging.info(f"Remapping residue motifs {update_motifs} from cached scorefile.")
                    self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)

                if update_motifs:
                    logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
                    self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)

            poses.reindex_poses(f"{prefix}_rfd3_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)
            return poses

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        n_jobs = min(len(pose_specs), jobstarter.max_cores)

        os.makedirs(output_dir := os.path.join(work_dir, "outputs"), exist_ok=True)
        os.makedirs(input_dir := os.path.join(work_dir, "inputs"), exist_ok=True)

        cmds = self.setup_run(
            pose_specs=pose_specs,
            input_dir=input_dir,
            output_dir=output_dir,
            n_jobs = n_jobs,
            options=options,
            n_batches=n_batches,
            diffusion_batch_size=diffusion_batch_size,
            dump_trajectories = dump_trajectories,
        )

        # 5) Prepend pre-cmd if set.
        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # 6) Execute commands.
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # 7) Collect and validate scores.
        scores = collect_scores(work_dir=work_dir, cif_to_pdb=convert_cif_to_pdb, run_clean=run_clean)

        if len(scores.index) == 0:
            raise RuntimeError(
                f"{self}: collect_scores returned no rows. "
                f"Check runner output logs and runner output directory ({work_dir})"
            )

        if fail_on_missing_output_poses and len(expected_outputs) < len(scores.index):
            raise RuntimeError(f"Number of output poses ({len(scores.index)}) is smaller than expected number of output poses {expected_outputs}. Some runs might have crashed!")

        # 8) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        if not poses:
            poses.df = scores.copy()
            poses.df["input_poses"] = None
            logging.info(
                f"De novo mode: populated poses.df directly from scores {len(poses.df.index)} rows).")
        else:
            poses = RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

            if update_motifs:
                logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
                self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)
        
        poses.reindex_poses(f"{prefix}_rfd3_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)

        logging.info(f"{self} finished. Returning {len(poses.df.index)} poses.")
        return poses
    
    def setup_run(self, pose_specs: list[dict], input_dir:str, output_dir: str, n_jobs=int, options:str=None, n_batches: int = 1, diffusion_batch_size: int = 8, 
                  dump_trajectories: bool = False) -> list:

        batched_pose_specs = split_list(pose_specs, n_sublists=n_jobs)

        # write input json files for each batch
        json_paths = []
        for i, batch in enumerate(batched_pose_specs):
            batch_dict = {}
            for d in batch:
                batch_dict.update(d)
            json_paths.append(write_json(batch_dict, os.path.join(input_dir, f"batch{i}.json")))

        cmds = [
            self.write_cmd(
                in_json=in_json,
                out_dir=output_dir, 
                options=options, 
                n_batches=n_batches, 
                diffusion_batch_size=diffusion_batch_size, 
                dump_trajectories=dump_trajectories)
            for in_json in json_paths
            ]
        
        return cmds


    def write_cmd(self,
        in_json: str,
        out_dir: str,
        options: str = None,
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        dump_trajectories: bool = False) -> str:

        if not options:
            options = ""
            
        return f"{self.python_path} {self.application_path} design inputs={in_json} out_dir={out_dir} " \
            f"n_batches={n_batches} diffusion_batch_size={diffusion_batch_size} dump_trajectories={dump_trajectories} {options}"

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete all files in the outputs directory before a rerun."""
        output_dir = os.path.join(work_dir, "outputs")
        if not os.path.isdir(output_dir):
            return
        for file_path in glob(os.path.join(output_dir, "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)

    def remap_motifs(self, poses: Poses, motifs: list[str], prefix: str) -> None:
        """Remap ResidueSelection motifs in poses.df using diffused_index_map columns.

        Uses the {prefix}_diffused_index_map_* columns added by collect_scores
        to translate input residue positions to their new positions in the
        diffused output structures. Overwrites motif columns in place,
        consistent with RFD1 behavior.

        Parameters
        ----------
        poses : Poses
            The Poses object after the RFD3 run (must have index_map columns).
        motifs : list[str]
            Column names in poses.df containing ResidueSelection objects to remap.
        prefix : str
            The prefix used in the RFD3 run, used to find diffused_index_map columns.
        """

        map_prefix = f"{prefix}_diffused_index_map_"
        map_cols = [c for c in poses.df.columns if c.startswith(map_prefix)]

        logging.info(f"[remap_motifs] Found {len(map_cols)} diffused_index_map columns: {map_cols}")

        if not map_cols:
            raise ValueError(
                f"No diffused_index_map columns found for prefix '{prefix}'. "
                f"Make sure collect_scores ran successfully and the sidecar JSONs exist."
            )

        logging.info(f"[remap_motifs] Motifs to remap: {motifs}")

        for motif_col in motifs:
            logging.info(f"[remap_motifs] Processing motif column '{motif_col}'")

            if motif_col not in poses.df.columns:
                raise ValueError(
                    f"[remap_motifs] Motif column '{motif_col}' not found in poses.df!"
                )

            output_motif_l = []
            for idx, row in poses.df.iterrows():
                logging.info(
                    f"[remap_motifs] Row {idx}: description='{row.get('poses_description', 'N/A')}'"
                )

                # reconstruct exchange_dict: {("A", 77): ("A", 96), ...}
                exchange_dict = {}
                for col in map_cols:
                    input_res_str = col.replace(map_prefix, "")  # e.g. "A77"
                    output_res_str = row[col]                     # e.g. "A96"
                    if pd.notna(output_res_str):
                        exchange_dict[parse_residue(input_res_str)] = parse_residue(output_res_str)
                    else:
                        logging.warning(
                            f"[remap_motifs] Row {idx}: NaN value for column '{col}', skipping."
                        )

                logging.info(
                    f"[remap_motifs] Row {idx}: reconstructed exchange_dict: {exchange_dict}"
                )

                motif = row[motif_col]
                exchanged_motif = [exchange_dict[residue] for residue in motif.residues]
                logging.info(f"[remap_motifs] Row {idx}: exchanged_motif = {exchanged_motif}")
                output_motif_l.append(ResidueSelection(exchanged_motif))

            # overwrite in place, consistent with RFD1 behavior
            poses.df[motif_col] = output_motif_l
            logging.info(
                f"[remap_motifs] Finished remapping '{motif_col}' for all {len(poses.df)} rows."
            )

        logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")


def collect_scores(work_dir: str, cif_to_pdb: bool = True, run_clean: bool=True) -> pd.DataFrame:
    """Parse runner outputs and return the canonical scores dataframe.

    Reads all .cif.gz output files from the outputs directory, decompresses
    them, and parses their accompanying sidecar .json files for scores
    including metrics, specification, and diffused_index_map entries.

    Parameters
    ----------
    work_dir : str
        The runner working directory (parent of the outputs/ subdirectory).
    include_scores : list[str] | None
        Optional list of heavy score field names to include. Heavy fields
        (per-residue arrays, nested lists) are excluded by default.

    Returns
    -------
    pd.DataFrame
        One row per output structure with columns:
        - description: basename without extension
        - location: absolute path to decompressed .cif file
        - all scalar fields from sidecar JSON, flattened and prefixed
    """
    def decompress_cif_gz(path: str, out_path: str = None) -> str:
        """Decompress a .cif.gz file and return path to the decompressed .cif file."""
        if not out_path:
            out_path = path.replace(".cif.gz", ".cif")
        if not os.path.isfile(out_path):
            with gzip.open(path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return out_path
    
    def convert_cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return os.path.abspath(output)
    
    output_dir = os.path.join(work_dir, "outputs")

    directory = Path(output_dir)
    pattern = r"batch.*?_"

    # rename paths and jsons to remove prefix derived from json input
    for file_path in directory.iterdir():
        if file_path.is_file():
            new_name = re.sub(pattern, "", file_path.name)
            file_path.rename(file_path.with_name(new_name))

    output_jsons = glob(os.path.join(output_dir, "*.json"))

    data = []
    # iterate over jsons because additional cif files might be there if dump_trajectories is true
    for j in output_jsons:
        p_data = read_json(j)
        p_data.update(p_data["metrics"]) # flatten metrics
        p_data["compressed_cif_location"] = re.sub(r"\.json$", ".cif.gz", j)
        # delete specifications, 
        for key in ["specification", "metrics"]:
            p_data.pop(key)
        data.append(pd.Series(p_data))
    
    data = pd.DataFrame(data)
    # unpack and rename to remove index layers
    data["cif_location"] = data.apply(
        lambda row: decompress_cif_gz(path=row["compressed_cif_location"]), axis=1)

    if cif_to_pdb:
        data["location"] = data.apply(lambda row: convert_cif_to_pdb(row["cif_location"], "pdb", re.sub(r"\.cif$", ".pdb", row["cif_location"])), axis=1)
    else:
        data["location"] = data["cif_location"]

    data["description"] = [description_from_path(p) for p in data["location"]]

    if run_clean:
        _ = [os.remove(comp_cif) for comp_cif in data["compressed_cif_location"]]
        data.drop(["compressed_cif_location"], axis=1, inplace=True)
        if cif_to_pdb:
            _ = [os.remove(cif) for cif in data["cif_location"]]
            data.drop(["cif_location"], axis=1, inplace=True)

    return data


def read_json(path) -> dict:
    with open(path, 'r', encoding="UTF-8") as j:
        data = json.load(j)

    return data

def write_json(data, path) -> str:
    with open(path, 'w', encoding="UTF-8") as j:
        json.dump(data, j, indent=2)
    return path