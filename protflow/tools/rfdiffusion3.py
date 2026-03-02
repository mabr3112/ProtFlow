"""
RFdiffusion3 Module
===================

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

This implementation does **not**
accept a pre-existing JSON specification file. The input JSON file
is always constructed internally from the parameters provided to
the `run()` method. Users must supply specification arguments
(e.g. contig, symmetry, selection options, etc.), and the runner
generates the JSON file automatically for each pose.

Additionally, the `index_layers` parameter is **not manually configurable**.
It is dynamically inferred from the `settings_group_name` argument
via `_retrieve_underscores_from_settings_group()`. This ensures
correct pose reindexing based on the RFDiffusion3 output naming scheme.

RFDiffusion3 output structures are written as `.cif.gz` files with
accompanying sidecar `.json` files containing metrics, specification
data, and diffused index maps. These are parsed and flattened into
a structured pandas DataFrame for integration into ProtFlow.

Usage
-----
To use this module, instantiate the `RFdiffusion3` runner and call
its `run()` method with appropriate arguments.

The runner will:

1. Infer index layers automatically.
2. Construct per-pose input JSON files.
3. Generate shell commands.
4. Execute jobs via JobStarter.
5. Parse outputs and merge results back into the Poses object.

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
        contig="A1-100",
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
  Providing a pre-existing JSON file is not supported.
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
This module is part of the ProtFlow package and is designed
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

import pandas as pd

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.runners import (
    Runner,
    RunnerOutput,
    parse_generic_options,
    options_flags_to_string,
    prepend_cmd,
)

class RFdiffusion3(Runner):
    """
    RFdiffusion3 Runner Class
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
        # self.index_layers is set dynamically in run() via _retrieve_underscores_from_settings_group()
        # because it depends on settings_group_name which is only known at run time

    def __str__(self) -> str:
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        # --- parameters to build a JSON file for RFD3 automatically ---
        settings_group_name: str = "rfd3",
        input: str | None = None,
        contig: str | None = None,
        unindex: str | None = None,
        length: str | None = None,
        ligand: str | None = None,
        select_fixed_atoms: dict | str | bool | None = None,
        select_unfixed_sequence: dict | str | bool | None = None,
        select_hotspots: dict | str | bool | None = None,
        select_buried: dict | str | bool | None = None,
        select_partially_buried: dict | str | bool | None = None,
        select_exposed: dict | str | bool | None = None,
        select_hbond_donor: dict | None = None,
        select_hbond_acceptor: dict | None = None,
        redesign_motif_sidechains: bool | None = None,
        partial_t: float | None = None,
        plddt_enhanced: bool | None = None,
        is_non_loopy: bool | None = None,
        symmetry: dict | None = None,
        ori_token: list | None = None,
        infer_ori_strategy: str | None = None,
        cif_parser_args: dict | None = None,
        dialect: int | None = None,
        extra: dict | None = None,
        # --- RFD3 CLI arguments ---
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        # --- general ProtFlow parameters ---
        options: str | None = None,
        pose_options: list[str] | str | None = None,
        include_scores: list[str] | None = None,
        update_motifs: list[str] | None = None,
        multiplex_poses: int | None = None,
        fail_on_missing_output_poses: bool = False,
        overwrite: bool = False,
    ) -> Poses:
        """
        Execute the full RFDiffusion3 runner lifecycle and merge results into poses.

        This method performs the complete workflow:

            1. Infer index_layers dynamically from settings_group_name.
            2. Perform generic runner setup.
            3. Optionally reuse cached scorefile.
            4. Optionally duplicate poses for multiplexing.
            5. Automatically generate per-pose input JSON files.
            6. Build and execute CLI commands.
            7. Parse outputs and flatten sidecar JSON scores.
            8. Merge results into the Poses object.
            9. Optionally remap residue motifs.
            10. Optionally collapse multiplex index layers.

        Important Notes
        ---------------
        - The input JSON file is ALWAYS constructed internally.
        It is not possible to provide a pre-existing JSON file.
        - index_layers is inferred automatically and cannot be
        manually specified.

        Parameters
        ----------
        settings_group_name : str
            Name of the JSON settings group. Determines output naming
            and affects index_layers inference.

        multiplex_poses : int | None
            Duplicate each input pose this many times before execution.
            Must be >1 to have an effect.

        fail_on_missing_output_poses : bool
            Raise RuntimeError if fewer outputs than expected are collected.

        include_scores : list[str] | None
            Heavy score field names to include explicitly.

        overwrite : bool
            If True, delete previous outputs and rerun.

        Returns
        -------
        Poses
            Updated Poses object containing merged RFD3 results.
        """

        # -1) Determine index_layers from settings_group_name.
        self.index_layers = _retrieve_underscores_from_settings_group(
            settings_group_name=settings_group_name
        )

        # Warn if multiplex_poses=1 since it has no effect.
        if multiplex_poses == 1:
            logging.warning(
                "multiplex_poses=1 has no effect. Set to None or an integer > 1."
            )

        # 1) Generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))
        total_designs = n_batches * diffusion_batch_size
        logging.info(
            f"Total designs per input pose: {total_designs} "
            f"({n_batches} batches x {diffusion_batch_size} per batch)"
        )

        # 2) Scorefile reuse shortcut.
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)

            # If multiplexing, duplicate poses before merge so index layers match.
            if multiplex_poses:
                poses.duplicate_poses(
                    f"{poses.work_dir}/{prefix}_multiplexed_input_pdbs/",
                    multiplex_poses,
                )

            poses = RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

            if update_motifs:
                logging.info(f"Remapping residue motifs {update_motifs} from cached scorefile.")
                self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)

            # remove_layers = 1 (from duplicate_poses) + self.index_layers (from RFD3 output naming)
            if multiplex_poses:
                poses.reindex_poses(
                    prefix=f"{prefix}_post_multiplex_reindexing",
                    remove_layers=1 + self.index_layers,
                    force_reindex=True,
                    overwrite=overwrite,
                )

            return poses

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # 3) If multiplexing, duplicate poses now so each copy gets its own job.
        if multiplex_poses:
            poses.duplicate_poses(
                f"{poses.work_dir}/{prefix}_multiplexed_input_pdbs/",
                multiplex_poses,
            )
            logging.info(
                f"Multiplexed input poses to {multiplex_poses} copies: "
                f"{len(poses)} poses total."
            )

        # 4) Prepare pose-level options.
        pose_options_list = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # 5) Build commands.
        cmds = self._build_commands(
            poses=poses,
            work_dir=work_dir,
            options=options,
            pose_options=pose_options_list,
            settings_group_name=settings_group_name,
            input=input,
            contig=contig,
            unindex=unindex,
            length=length,
            ligand=ligand,
            select_fixed_atoms=select_fixed_atoms,
            select_unfixed_sequence=select_unfixed_sequence,
            select_hotspots=select_hotspots,
            select_buried=select_buried,
            select_partially_buried=select_partially_buried,
            select_exposed=select_exposed,
            select_hbond_donor=select_hbond_donor,
            select_hbond_acceptor=select_hbond_acceptor,
            redesign_motif_sidechains=redesign_motif_sidechains,
            partial_t=partial_t,
            plddt_enhanced=plddt_enhanced,
            is_non_loopy=is_non_loopy,
            symmetry=symmetry,
            ori_token=ori_token,
            infer_ori_strategy=infer_ori_strategy,
            cif_parser_args=cif_parser_args,
            dialect=dialect,
            extra=extra,
            n_batches=n_batches,
            diffusion_batch_size=diffusion_batch_size,
        )

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
        scores = collect_scores(work_dir=work_dir, include_scores=include_scores)

        if len(scores.index) == 0:
            raise RuntimeError(
                f"{self}: collect_scores returned no rows. "
                f"Check runner output logs and runner output directory ({work_dir})"
            )

        n_input_poses = len(poses.df.index)
        expected_outputs = n_input_poses * n_batches * diffusion_batch_size
        logging.info(f"expected outputs {expected_outputs} = n_input_poses * n_batches * diffusion_batch_size with len(poses.df.index) {len(poses.df.index)} and n_batches {n_batches} and diffusion_batch_size {diffusion_batch_size} and len(scores.index) {len(scores.index)}.")
        if fail_on_missing_output_poses and len(scores.index) < expected_outputs:
            raise RuntimeError(
                f"{self}: Expected {expected_outputs} output poses "
                f"({n_input_poses} input poses x {n_batches} batches x {diffusion_batch_size} per batch) "
                f"but only collected {len(scores.index)}."
            )

        # 8) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        poses = RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

        # 9) Optionally remap motifs using diffused_index_map from sidecar JSONs.
        if update_motifs:
            logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
            self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)

        # 10) If multiplexing, reindex poses to collapse duplication layer.
        #     remove_layers = 1 (from duplicate_poses) + self.index_layers (from RFD3 output naming)
        if multiplex_poses:
            poses.reindex_poses(
                prefix=f"{prefix}_post_multiplex_reindexing",
                remove_layers=1 + self.index_layers,
                force_reindex=True,
                overwrite=overwrite,
            )
            logging.info(
                f"Reindexed multiplexed poses: {len(poses)} poses after collapsing "
                f"{1 + self.index_layers} index layers."
            )

        logging.info(f"{self} finished. Returning {len(poses.df.index)} poses.")
        return poses


    def _build_commands(
        self,
        poses: Poses,
        work_dir: str,
        options: str | None,
        pose_options: list[str | None],
        settings_group_name: str | None = None,
        input: str | None = None,
        contig: str | None = None,
        unindex: str | None = None,
        length: str | None = None,
        ligand: str | None = None,
        select_fixed_atoms: dict | str | bool | None = None,
        select_unfixed_sequence: dict | str | bool | None = None,
        select_hotspots: dict | str | bool | None = None,
        select_buried: dict | str | bool | None = None,
        select_partially_buried: dict | str | bool | None = None,
        select_exposed: dict | str | bool | None = None,
        select_hbond_donor: dict | None = None,
        select_hbond_acceptor: dict | None = None,
        redesign_motif_sidechains: bool | None = None,
        partial_t: float | None = None,
        plddt_enhanced: bool | None = None,
        is_non_loopy: bool | None = None,
        symmetry: dict | None = None,
        ori_token: list | None = None,
        infer_ori_strategy: str | None = None,
        cif_parser_args: dict | None = None,
        dialect: int | None = None,
        extra: dict | None = None,
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
    ) -> list[str]:
        """Create one shell command per pose."""
        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for pose_path, pose_opt in zip(poses.poses_list(), pose_options):
            merged_opts, merged_flags = parse_generic_options(options, pose_opt, sep="--")
            cli_args = options_flags_to_string(merged_opts, list(merged_flags), sep="--")

            cmds.append(self.write_cmd(
                pose_path=pose_path,
                out_dir=out_dir,
                cli_args=cli_args,
                settings_group_name=settings_group_name,
                input=input,
                contig=contig,
                unindex=unindex,
                length=length,
                ligand=ligand,
                select_fixed_atoms=select_fixed_atoms,
                select_unfixed_sequence=select_unfixed_sequence,
                select_hotspots=select_hotspots,
                select_buried=select_buried,
                select_partially_buried=select_partially_buried,
                select_exposed=select_exposed,
                select_hbond_donor=select_hbond_donor,
                select_hbond_acceptor=select_hbond_acceptor,
                redesign_motif_sidechains=redesign_motif_sidechains,
                partial_t=partial_t,
                plddt_enhanced=plddt_enhanced,
                is_non_loopy=is_non_loopy,
                symmetry=symmetry,
                ori_token=ori_token,
                infer_ori_strategy=infer_ori_strategy,
                cif_parser_args=cif_parser_args,
                dialect=dialect,
                extra=extra,
                n_batches=n_batches,
                diffusion_batch_size=diffusion_batch_size,
            ))
        return cmds

    def _write_input_json(
        self,
        pose_path: str,
        out_dir: str,
        settings_group_name: str | None = None,
        input: str | None = None,
        contig: str | None = None,
        unindex: str | None = None,
        length: str | None = None,
        ligand: str | None = None,
        select_fixed_atoms: dict | str | bool | None = None,
        select_unfixed_sequence: dict | str | bool | None = None,
        select_hotspots: dict | str | bool | None = None,
        select_buried: dict | str | bool | None = None,
        select_partially_buried: dict | str | bool | None = None,
        select_exposed: dict | str | bool | None = None,
        select_hbond_donor: dict | None = None,
        select_hbond_acceptor: dict | None = None,
        redesign_motif_sidechains: bool | None = None,
        partial_t: float | None = None,
        plddt_enhanced: bool | None = None,
        is_non_loopy: bool | None = None,
        symmetry: dict | None = None,
        ori_token: list | None = None,
        infer_ori_strategy: str | None = None,
        cif_parser_args: dict | None = None,
        dialect: int | None = None,
        extra: dict | None = None,
    ) -> str:
        """Generate a RFDiffusion3 input JSON file for a single pose."""
        desc = os.path.splitext(os.path.basename(pose_path))[0]
        group_name = settings_group_name or desc

        spec = {}
        spec["input"] = input or pose_path

        optional_fields = {
            "contig": contig,
            "unindex": unindex,
            "length": length,
            "ligand": ligand,
            "select_fixed_atoms": select_fixed_atoms,
            "select_unfixed_sequence": select_unfixed_sequence,
            "select_hotspots": select_hotspots,
            "select_buried": select_buried,
            "select_partially_buried": select_partially_buried,
            "select_exposed": select_exposed,
            "select_hbond_donor": select_hbond_donor,
            "select_hbond_acceptor": select_hbond_acceptor,
            "redesign_motif_sidechains": redesign_motif_sidechains,
            "partial_t": partial_t,
            "plddt_enhanced": plddt_enhanced,
            "is_non_loopy": is_non_loopy,
            "symmetry": symmetry,
            "ori_token": ori_token,
            "infer_ori_strategy": infer_ori_strategy,
            "cif_parser_args": cif_parser_args,
            "dialect": dialect,
            "extra": extra,
        }

        for key, value in optional_fields.items():
            if value is not None:
                spec[key] = value

        content = {group_name: spec}

        json_path = os.path.join(out_dir, f"{desc}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=4)

        logging.info(f"Written input JSON for {desc} to {json_path}")

        return os.path.abspath(json_path)

    def write_cmd(
        self,
        pose_path: str,
        out_dir: str,
        cli_args: str,
        settings_group_name: str | None = None,
        input: str | None = None,
        contig: str | None = None,
        unindex: str | None = None,
        length: str | None = None,
        ligand: str | None = None,
        select_fixed_atoms: dict | str | bool | None = None,
        select_unfixed_sequence: dict | str | bool | None = None,
        select_hotspots: dict | str | bool | None = None,
        select_buried: dict | str | bool | None = None,
        select_partially_buried: dict | str | bool | None = None,
        select_exposed: dict | str | bool | None = None,
        select_hbond_donor: dict | None = None,
        select_hbond_acceptor: dict | None = None,
        redesign_motif_sidechains: bool | None = None,
        partial_t: float | None = None,
        plddt_enhanced: bool | None = None,
        is_non_loopy: bool | None = None,
        symmetry: dict | None = None,
        ori_token: list | None = None,
        infer_ori_strategy: str | None = None,
        cif_parser_args: dict | None = None,
        dialect: int | None = None,
        extra: dict | None = None,
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
    ) -> str:
        """Construct the shell command to run RFDiffusion3 for one pose."""
        json_path = self._write_input_json(
            pose_path=pose_path,
            out_dir=out_dir,
            settings_group_name=settings_group_name,
            input=input,
            contig=contig,
            unindex=unindex,
            length=length,
            ligand=ligand,
            select_fixed_atoms=select_fixed_atoms,
            select_unfixed_sequence=select_unfixed_sequence,
            select_hotspots=select_hotspots,
            select_buried=select_buried,
            select_partially_buried=select_partially_buried,
            select_exposed=select_exposed,
            select_hbond_donor=select_hbond_donor,
            select_hbond_acceptor=select_hbond_acceptor,
            redesign_motif_sidechains=redesign_motif_sidechains,
            partial_t=partial_t,
            plddt_enhanced=plddt_enhanced,
            is_non_loopy=is_non_loopy,
            symmetry=symmetry,
            ori_token=ori_token,
            infer_ori_strategy=infer_ori_strategy,
            cif_parser_args=cif_parser_args,
            dialect=dialect,
            extra=extra,
        )

        return (
            f"{self.python_path} {self.application_path} design "
            f"inputs='{json_path}' "
            f"out_dir='{out_dir}' "
            f"n_batches={n_batches} "
            f"diffusion_batch_size={diffusion_batch_size} "
            f"{cli_args}"
        )

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
        from protflow.residues import ResidueSelection, parse_residue

        map_prefix = f"{prefix}_diffused_index_map_"
        map_cols = [c for c in poses.df.columns if c.startswith(map_prefix)]

        #logging.info(f"[remap_motifs] Found {len(map_cols)} diffused_index_map columns: {map_cols}")

        if not map_cols:
            raise ValueError(
                f"No diffused_index_map columns found for prefix '{prefix}'. "
                f"Make sure collect_scores ran successfully and the sidecar JSONs exist."
            )

        #logging.info(f"[remap_motifs] Motifs to remap: {motifs}")

        for motif_col in motifs:
            logging.info(f"[remap_motifs] Processing motif column '{motif_col}'")

            if motif_col not in poses.df.columns:
                raise ValueError(
                    f"[remap_motifs] Motif column '{motif_col}' not found in poses.df!"
                )

            output_motif_l = []
            for idx, row in poses.df.iterrows():
                #logging.info(f"[remap_motifs] Row {idx}: description='{row.get('poses_description', 'N/A')}'")

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

                #logging.info(f"[remap_motifs] Row {idx}: reconstructed exchange_dict: {exchange_dict}")

                motif = row[motif_col]
                exchanged_motif = [exchange_dict[residue] for residue in motif.residues]
                #logging.info(f"[remap_motifs] Row {idx}: exchanged_motif = {exchanged_motif}")
                output_motif_l.append(ResidueSelection(exchanged_motif))

            # overwrite in place, consistent with RFD1 behavior
            poses.df[motif_col] = output_motif_l
            #logging.info(f"[remap_motifs] Finished remapping '{motif_col}' for all {len(poses.df)} rows.")

        #logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")


def _retrieve_underscores_from_settings_group(settings_group_name: str) -> int:
    """Calculate index_layers from settings_group_name.

    RFD3 output format: <json_name>_<settings_group>_<batch_number>_model_<n>
    Stripping index_layers from the back recovers the original pose description.
    Base of 4 accounts for: <settings_group>(1+) + <batch_number>(1) + model(1) + <n>(1).
    Additional layers are added for each underscore in settings_group_name.
    """
    underscore_count = settings_group_name.count("_")
    index_layers = 4 + underscore_count
    #logging.info(f"settings_group_name='{settings_group_name}' contains {underscore_count} underscores -> index_layers={index_layers}")
    return index_layers


def _is_heavy_value(value: object) -> bool:
    """Heuristic for values that can bloat score tables (2D/per-residue objects)."""
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple, dict)):
            return True
        if len(value) > 200:
            return True
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return True
    return False


def _extract_score_dict(
    payload: dict,
    include_scores: set[str],
    prefix: str = "",
) -> dict[str, object]:
    """Flatten nested score dictionaries with optional inclusion of heavy values."""
    out: dict[str, object] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(_extract_score_dict(value, include_scores, prefix=flat_key))
            continue

        if _is_heavy_value(value):
            if key in include_scores or flat_key in include_scores:
                out[flat_key] = json.dumps(value)
            continue

        out[flat_key] = value

    return out


def _decompress_cif_gz(path: str) -> str:
    """Decompress a .cif.gz file and return path to the decompressed .cif file."""
    out_path = path.replace(".cif.gz", ".cif")
    if not os.path.isfile(out_path):
        with gzip.open(path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return out_path


def collect_scores(work_dir: str, include_scores: list[str] | None = None) -> pd.DataFrame:
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
    include_set = set(include_scores or [])
    output_dir = os.path.join(work_dir, "outputs")
    output_paths = sorted(glob(os.path.join(output_dir, "*.cif.gz")))

    output_paths = [_decompress_cif_gz(path) for path in output_paths]

    rows: list[dict[str, object]] = []
    for path in output_paths:
        desc = os.path.splitext(os.path.basename(path))[0]
        row: dict[str, object] = {
            "description": desc,
            "location": os.path.abspath(path),
        }

        sidecar = os.path.join(output_dir, f"{desc}.json")
        if os.path.isfile(sidecar):
            with open(sidecar, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                row.update(_extract_score_dict(parsed, include_set))

        rows.append(row)

    return pd.DataFrame(rows)