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
from ..residues import ResidueSelection

class RFD3Params(UserDict):
    """

    RFD3InputSpecification class
    ============================

    Helper class to specify input for RFD3. It manages per-pose assignment of input specifications for RFD3 in dict-like format
    (see https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields
    for more information on input specification format). <input> fields are infered automatically from poses.

    """
    def __init__(self, poses: Poses, spec_from_json: str = None, spec_from_dict : dict | RFD3Params = None):
        super().__init__()
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

    def set_input_specs(self, **kwargs)  -> RFD3Params:
        
        """
        Set input specifications for all poses (e.g.: contig='5-10,A4-20,5-10'). Setting <input> field manually not recommended, 
        it is deduced automatically from poses. 
        For possible keyword arguments, see https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields
        """
        
        # check for existing spec_dict:
        if not self.data:
            self.data = {"denovo": {}}

        # Capture all local variables (including arguments and kwargs)
        params = locals().copy()
        
        exclude = {'self', 'kwargs'}
        
        # Build the dictionary: exclude the blacklist and filter out None
        spec_dict = {
            k: v for k, v in params.items() 
            if k not in exclude and v is not None
        }

        # update with generic inputs
        if kwargs:
            spec_dict.update(kwargs)

        if "input" in spec_dict:
            logging.warning("Defining <input> manually not recommended, it is deduced automatically from poses. Make sure to use absolute paths!")

        # update input_specs for each pose
        for pose in self.data:
            self.data[pose].update(spec_dict)
        
        return self
    
    def set_per_pose_input_specs(self, **kwargs) -> RFD3Params:
        """
        Set pose-specific input specifications from poses.df columns or from lists containing an input spec value per pose.
        For possible keyword arguments, see https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields
        """
        
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
            if key == "input":
                logging.warning("Defining <input> manually not recommended, it is deduced automatically from poses. Make sure to use absolute paths!")
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
    
    def spec_from_dict(self, spec_dict: dict | RFD3Params) -> RFD3Params:
        """
        Set input specifications from an existing dict
        """
        self._check_specs(spec_dict)
        self.data = spec_dict
        self._update_input()
        return self
    
    def spec_from_json(self, json_path: str) -> RFD3Params:
        """
        Set input specifications from an existing json file
        """
        if not os.path.isfile(json_path):
            raise ValueError(f"Could not detect json file at {json_path}!")
        spec = read_json(json_path)
        self._check_specs(spec)
        self.data = spec
        self._update_input()
        return self
        
    def reset_pose_specs(self, poses: Poses) -> RFD3Params:
        """
        Delete existing input specifications, create new blank input
        """
        self.poses = poses
        self.data = self._create_pose_dict(self.poses)
        return self

    def add_specs(self, additional_specs: RFD3Params | dict) -> RFD3Params:
        """
        Function to add new specs, only if no input poses are set --> to create multiple unconditional diffusions with different settings
        """
        if self.poses:
            raise ValueError("Additional pose-specific input specifications can ony be added if no poses are present (unconditional diffusion)!")
        self._check_specs(additional_specs)
        self.data.update(additional_specs)
        return self

    def modify_specs(self, new_specs: RFD3Params | dict) -> RFD3Params:
        """
        Function to modify EXISTING input specifications, will update them with values from new_specs
        """
        if not all(pose in self.data for pose in new_specs) or not len(self.data) == len(new_specs):
            raise KeyError("Poses in <new_specs> do not match existing poses!")
        
        # update each pose
        for pose in new_specs:
            self.data[pose].update(new_specs[pose])
        self._update_input()
        return self

    def _update_input(self):
        """
        Update input field for each pose. Do nothing if no input poses were set (unconditional diffusion)
        """
        if self.poses and self.data:
            for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"]):
                self.data[name].update({"input": os.path.abspath(path)})
    
    def _check_specs(self, specs: RFD3Params | dict):
        """
        Check if input_specs format is correct
        """
        # check if dict is nested, raise error otherwise
        if not all(isinstance(v, dict) for v in specs.values()):
            dict_example = {"pose_1": {"spec_1": 1, "spec_2": 2}, "pose_2": {"spec_3": 3}}
            raise ValueError(f"Input specifications must be supplied in the format {dict_example}")
        
        # check if new specs fit to existing poses
        if self.poses and not all(pose in specs for pose in self.poses["poses_description"]) or not len(self.poses) == len(specs):
            raise ValueError("Specs do not fit existing poses!")


    def _create_pose_dict(self, poses: Poses) -> dict:
        """
        Create a new input specification dict for each pose, return an empty dict for empty poses
        """
        if poses:
            return {name: {"input": os.path.abspath(path)} for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"])}
        else:
            return {}

    
class RFdiffusion3(Runner):
    def __init__(
        self,
        application_path: str | None = None,
        model_dir: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        config = require_config()

        self.application_path = application_path or load_config_path(config, "RFDIFFUSION3_BIN_PATH")
        self.model_dir = model_dir or load_config_path(config, "RFDIFFUSION3_MODEL_DIR")
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
        params: RFD3Params,
        # --- RFD3 CLI arguments ---
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        ckpt_path: str = "rfd3_latest", # can be either full path or just name (without extension) of checkpoint file in checkpoint dir
        # --- general ProtFlow parameters ---
        options: str = None,
        update_motifs: list[str] = None, 
        multiplex_poses: int = None,
        jobstarter: JobStarter = None,
        convert_cif_to_pdb: bool = True,
        strict_remap: bool = True, # if true, fail if residues in motifs are not preserved post-diffusion
        run_clean: bool = True, # delete additional outputs like pre-conversion files
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
        fail_on_missing_output_poses : bool
            If True, raise a RuntimeError when the number of collected output
            poses is less than the expected number (n_poses * n_batches *
            diffusion_batch_size * multiplex_poses). RFDiffusion3 runs occasionally crash silently,
            and enabling this flag ensures such failures are caught early rather
            than propagating through a longer pipeline. Defaults to False.
        """

        def identify_checkpoint(model):
            if os.path.isfile(model):
                return model
            elif os.path.isfile(model := os.path.join(self.model_dir, model)):
                return model
            elif os.path.isfile(model := os.path.join(self.model_dir, f"{model}.ckpt")):
                return model
            else:
                raise ValueError(f"Could not detect model at {model} or at {self.model_dir}.")


        # check if input_specification fits to input poses
        if poses and not all(name in params for name in poses.df["poses_description"]) or not len(poses) == len(params):
            raise ValueError("Input <poses> do not match <input_specification>")
        
        ckpt_path = identify_checkpoint(ckpt_path)
        
        # Warn if multiplex_poses=1 since it has no effect.
        if multiplex_poses == 1:
            logging.warning("multiplex_poses=1 has no effect. Set to None or an integer > 1.")
        
        if not multiplex_poses:
            multiplex_poses = 1

        # update index layers as RFD3 adds 3 layers (later removed via reindexing)
        index_layers = self.index_layers + 2

        # Generic setup shared by all runners
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )

        # calculate total number of diffusions
        total_designs = n_batches * diffusion_batch_size * multiplex_poses
        logging.info(f"Total designs per input pose: {total_designs}\n({n_batches} batches x {diffusion_batch_size} per batch)")

        if multiplex_poses > 1:
            logging.info(f"and multiplexing input poses {multiplex_poses} times.")
            index_layers += 1
            suffixes = [f"_{str(i).zfill(4)}" for i in range(1, multiplex_poses + 1)]

            # multiplex and add an index layer to each input so that filenames are unique
            pose_specs = [
                {f"{pose}{sfx}": spec} for pose, spec in params.items()
                for sfx in suffixes
            ]

        else:
            # list of pose dicts
            pose_specs = [{pose: spec} for pose, spec in params.items()]

        expected_outputs = n_batches * diffusion_batch_size * len(pose_specs)
        logging.info(f"Expected number of output poses: {expected_outputs}")

        # scorefile reuse shortcut
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Reusing existing scorefile: {scorefile}")

            if not poses:
                poses.df = scores.copy()
                poses.df["input_poses"] = None
                logging.info("Populated poses.df from scorefile.")
            else:
                poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=index_layers).return_poses()

                if update_motifs:
                    logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
                    remap_rfd3_motifs(poses=poses, motifs=update_motifs, prefix=prefix, strict=strict_remap)

            # forced to reindex by layers because of possible empty input poses
            poses.reindex_poses(f"{prefix}_rfd3_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)
            return poses

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # define number of jobs
        n_jobs = min(len(pose_specs), jobstarter.max_cores)

        # create directories for in and output
        os.makedirs(output_dir := os.path.join(work_dir, "outputs"), exist_ok=True)
        os.makedirs(input_dir := os.path.join(work_dir, "inputs"), exist_ok=True)

        # split pose_specs into batches and write cmds
        cmds = self.setup_run(
            pose_specs=pose_specs,
            input_dir=input_dir,
            output_dir=output_dir,
            n_jobs = n_jobs,
            options=options,
            n_batches=n_batches,
            diffusion_batch_size=diffusion_batch_size,
            ckpt_path=ckpt_path
        )

        # prepend pre-cmd if set
        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # execute commands
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # collect and validate scores
        scores = collect_scores(work_dir=work_dir, cif_to_pdb=convert_cif_to_pdb, run_clean=run_clean)

        n_out_poses = len(scores.index)
        if n_out_poses == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        if fail_on_missing_output_poses and expected_outputs < n_out_poses:
            raise RuntimeError(f"Number of output poses ({n_out_poses}) is smaller than expected number of output poses {expected_outputs}. Some runs might have crashed!")

        # save scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # merge back into poses
        if not poses:
            poses.df = scores.copy()
            poses.df["input_poses"] = None
            logging.info(
                f"Populated poses.df directly from scores {len(poses.df.index)} rows).")
        else:
            poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=index_layers).return_poses()

            if update_motifs:
                logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
                remap_rfd3_motifs(poses=poses, motifs=update_motifs, prefix=prefix, strict=strict_remap)
        
        poses.reindex_poses(f"{prefix}_rfd3_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)

        logging.info(f"{self} finished. Returning {len(poses.df.index)} poses.")
        return poses
    
    def setup_run(self, pose_specs: list[dict], input_dir:str, output_dir: str, n_jobs: int, ckpt_path: str, options:str=None, 
                  n_batches: int = 1, diffusion_batch_size: int = 8) -> list:

        # split per-pose specs into several batches
        batched_pose_specs = split_list(pose_specs, n_sublists=n_jobs)

        # write input json files for each batch
        json_paths = []
        for i, batch in enumerate(batched_pose_specs):
            batch_dict = {}
            for d in batch:
                batch_dict.update(d)
            json_paths.append(write_json(batch_dict, os.path.join(input_dir, f"batch{i}.json")))

        # write cmds
        cmds = [
            self.write_cmd(
                in_json=in_json,
                out_dir=output_dir,
                ckpt_path=ckpt_path,
                options=options, 
                n_batches=n_batches, 
                diffusion_batch_size=diffusion_batch_size)
            for in_json in json_paths
            ]
        
        return cmds


    def write_cmd(self,
        in_json: str,
        out_dir: str,
        ckpt_path: str,
        options: str = None,
        n_batches: int = 1,
        diffusion_batch_size: int = 8
        ) -> str:

        if not options:
            options = ""

        # check for forbidden options        
        forbidden_options = ["inputs", "out_dir", "ckpt_path", "n_batches", "diffusion_batch_size"]
        if any(f" {f_opt}=" in options for f_opt in forbidden_options):
            raise ValueError(f"<options> must not contain any of {forbidden_options}, set them via .run arguments instead!")
        
        # return cmd string
        return f"{self.application_path} design inputs={in_json} out_dir={out_dir} ckpt_path={ckpt_path} " \
            f"n_batches={n_batches} diffusion_batch_size={diffusion_batch_size} {options}"

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete all files in the outputs directory before a rerun."""
        output_dir = os.path.join(work_dir, "outputs")
        input_dir = os.path.join(work_dir, "inputs")
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        if os.path.isdir(input_dir):
            shutil.rmtree(input_dir)

def remap_rfd3_motifs(poses: Poses, motifs: list[str], prefix: str, strict: bool = True) -> None:
    """Remap ResidueSelection motifs in poses.df using diffused_index_map columns.

    Uses the {prefix}_diffused_index_map column to translate input residue positions to their new positions in the
    diffused output structures.
    """
    diff_index_map_name = f"{prefix}_diffused_index_map"
    col_in_df(poses.df, diff_index_map_name)
    diffused_index_maps = poses.df[diff_index_map_name].to_list()

    logging.info(f"[remap_motifs] Motifs to remap: {motifs}")

    for motif_col in motifs:

        logging.info(f"[remap_motifs] Processing motif column '{motif_col}'")
        col_in_df(poses.df, motif_col)
        ref_motifs = poses.df[motif_col].to_list()
        
        # check if all motifs are ResidueSelections
        if not all(isinstance(motif, ResidueSelection) for motif in ref_motifs):
            raise ValueError(f"Not all motifs in column {motif_col} are of type ResidueSelection!")
        
        # create a list out of each ResidueSelection
        ref_motifs = [motif.to_list() for motif in ref_motifs] 
        updated_motifs = []
        for diff_idx_map, ref_motif in zip(diffused_index_maps, ref_motifs):
            # check if every residue is present in the diffused index map if strict matching is required
            if strict and not all(res in diff_idx_map for res in ref_motif): 
                raise ValueError(f"Could not find all original residues in diffused index map for selection {motif_col}. Are you sure they were preserved?")
            # check if diff_idx_map is present for all poses
            if strict and not diff_idx_map:
                raise ValueError("Not all poses feature diffused_index_map property. Are you sure preserved motifs exist for all input poses?")
            updated_motifs.append([diff_idx_map[res] for res in ref_motif if res in diff_idx_map]) # skip residues not in diff_idx_map
        # update motif col with new residue selections
        poses.df[motif_col] = [ResidueSelection(updated_motif) for updated_motif in updated_motifs]
        
    logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")


def collect_scores(work_dir: str, cif_to_pdb: bool = True, run_clean: bool=True) -> pd.DataFrame:
    
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

    # unpack
    data["cif_location"] = data.apply(
        lambda row: decompress_cif_gz(path=row["compressed_cif_location"]), axis=1)

    # convert cif to pdb, set new location
    if cif_to_pdb:
        data["location"] = data.apply(lambda row: convert_cif_to_pdb(row["cif_location"], "pdb", re.sub(r"\.cif$", ".pdb", row["cif_location"])), axis=1)
    else:
        data["location"] = data["cif_location"]

    data["description"] = [description_from_path(p) for p in data["location"]]

    # delete obsolete output
    if run_clean:
        _ = [os.remove(comp_cif) for comp_cif in data["compressed_cif_location"]]
        data.drop(["compressed_cif_location"], axis=1, inplace=True)
        if cif_to_pdb:
            _ = [os.remove(cif) for cif in data["cif_location"]]
            data.drop(["cif_location"], axis=1, inplace=True)

    data.reset_index(drop=True, inplace=True)

    return data


def read_json(path) -> dict:
    with open(path, 'r', encoding="UTF-8") as j:
        data = json.load(j)

    return data

def write_json(data, path) -> str:
    with open(path, 'w', encoding="UTF-8") as j:
        json.dump(data, j, indent=2)
    return path