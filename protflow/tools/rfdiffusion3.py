"""Runner template for ProtFlow tool integrations.

How to use this template
------------------------
1. Copy this file (or class) to a new module in ``protflow/tools`` or ``protflow/metrics``.
2. Rename ``ExampleRunner`` and ``example_runner`` to your tool name.
3. Replace all ``TODO`` markers.
4. Keep the run lifecycle intact:
   setup workdir -> reuse cached outputs -> prep options -> build commands -> run jobs -> collect scores -> return RunnerOutput.
5. Implement ``collect_scores(...)`` as a **module function**, not a class method.

Design goals
------------
- Keep runner behavior consistent across ProtFlow.
- Make it obvious where tool-specific logic belongs.
- Avoid re-implementing common logic already provided by ``Runner``.
- Keep score parsing callable without constructing a runner instance.
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
    """Template class for implementing a new ProtFlow runner.

    Developer instructions
    ----------------------
    - Keep this class focused on one external tool.
    - Put all user-facing run parameters on ``run(...)``.
    - Use config values as defaults in ``__init__``.
    - Keep score parsing in the module-level ``collect_scores(...)`` function.
    - Ensure output parsing returns a dataframe with:
      - ``description``: basename without extension of each produced pose
      - ``location``: absolute path to produced pose file
    - Always return ``RunnerOutput(...).return_poses()``.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize tool paths and static runner metadata.

        Developer instructions
        ----------------------
        - Load paths from config by default.
        - Keep constructor lightweight; do not run jobs here.
        - Define ``self.index_layers`` according to output naming:
          - ``0`` if output descriptions match input pose descriptions.
          - ``>0`` if your tool appends index layers like ``_0001``.
        """

        config = require_config()

        self.application_path = application_path or load_config_path(config, "RFDIFFUSION3_SCRIPT_PATH")
        self.python_path = python_path or load_config_path(config, "RFDIFFUSION3_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "RFDIFFUSION3_PRE_CMD", is_pre_cmd=True)



        self.jobstarter = jobstarter
        self.name = "rfdiffusion3"
        #self.index_layers = 4 # since index layers is dependent on the <settings_group> name only later defined in run() it cannot be defined here

    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        # --- provide parameters to build a JSON file for RFD3 automatically ---
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
        # --- RFD3 CLI arguments worth exposing explicitly ---
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        # --- general ProtFlow parameters ---
        options: str | None = None,
        pose_options: list[str] | str | None = None,
        include_scores: list[str] | None = None,
        update_motifs: list[str] | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Execute the full runner lifecycle and merge results into ``poses``.

        Developer instructions
        ----------------------
        The canonical order is:
        1. Generic setup (prefix check, jobstarter resolution, workdir creation).
        2. Reuse cached scorefile when available and ``overwrite=False``.
        3. Prepare per-pose options.
        4. Build command list.
        5. Execute via selected jobstarter.
        6. Collect scores into required dataframe format.
        7. Save runner scorefile and merge with ``RunnerOutput``.

        Notes on ``include_scores``
        ---------------------------
        ``include_scores`` is passed through to module-level ``collect_scores``.
        Use it to opt into heavy optional score fields (e.g., per-residue vectors
        or 2D matrices) that should not be loaded by default.
        """
        
        # -1)
        self.index_layers = _retrieve_underscores_from_settings_group(settings_group_name=settings_group_name)   
        # RFD3 output format: <name>_<settings_group>_<batch_number>_model_n.<suffix>. So typically index_layers needed to strip would be 4
        # but if there are underscores in <settings_group> additional layers need to be stripped, that's why the helper function is included 


      
        
        # 1) Generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))
        # log total number of designs
        total_designs = n_batches * diffusion_batch_size
        logging.info(
            f"Total designs per input pose: {total_designs} "
            f"({n_batches} batches x {diffusion_batch_size} per batch)"
        )

        # 2) Scorefile reuse shortcut.
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)
            return RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # 3) Prepare pose-level options.
        pose_options_list = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # 4) Build commands.
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

        # 5) Execute commands.
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # 6) Collect and validate scores (module function, by convention).
        scores = collect_scores(work_dir=work_dir, include_scores=include_scores)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        # 7) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # 8) Optionally remap motifs using diffused_index_map from sidecar JSONs.
        if update_motifs:
            logging.info(f"Remapping motifs {update_motifs} after RFD3 run.")
            self.remap_motifs(poses=poses, motifs=update_motifs, prefix=prefix)

        # 9) add here optional multiplex_poses


        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
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
        """Create one shell command per pose.

        Parameters
        ----------
        poses : Poses
            The current poses object.
        work_dir : str
            Working directory for this run.
        options : str | None
            Global options string passed by the user.
        pose_options : list[str | None]
            Per-pose options list, one entry per pose.
        All other parameters are passed through to write_cmd.

        Returns
        -------
        list[str]
            List of shell commands, one per pose.
        """
        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for pose_path, pose_opt in zip(poses.poses_list(), pose_options):
            # merge global and pose-specific options
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
        """Generate a RFDiffusion3 input JSON file for a single pose.

        Parameters
        ----------
        pose_path : str
            Path to the input pose file. Used to derive the settings group
            name if none is provided, and set as 'input' in the JSON if
            no explicit input is given.
        out_dir : str
            Directory where the generated JSON file will be saved.
        settings_group_name : str | None
            Name of the settings group in the JSON. If None, the pose
            description (filename without extension) is used. Note that
            this name appears in output filenames.
        All other parameters map directly to RFDiffusion3 InputSpecification
        fields. Only non-None values are written to the JSON.

        Returns
        -------
        str
            Absolute path to the generated JSON file.
        """
        # derive description from pose path
        desc = os.path.splitext(os.path.basename(pose_path))[0]
        group_name = settings_group_name or desc

        # build the settings dict, only including non-None values
        spec = {}

        # input defaults to the pose_path if not explicitly provided
        spec["input"] = input or pose_path

        # add all other fields only if provided
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

        # wrap in the settings group
        content = {group_name: spec}

        # save to out_dir
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
        """Construct the shell command to run RFDiffusion3 for one pose.

        Parameters
        ----------
        pose_path : str
            Path to the input pose file.
        out_dir : str
            Directory where RFD3 outputs will be saved.
        cli_args : str
            Additional CLI arguments assembled by _build_commands.
        All other parameters map to RFDiffusion3 InputSpecification fields
        and are passed to _write_input_json.

        Returns
        -------
        str
            The complete shell command string for one pose.
        """


        # generate JSON from provided parameters
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

        # assemble and return the command
        return (
            f"{self.python_path} {self.application_path} design "
            f"inputs='{json_path}' "
            f"out_dir='{out_dir}' "
            f"n_batches={n_batches} "
            f"diffusion_batch_size={diffusion_batch_size} "
            f"{cli_args}"
        )

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete all files in the outputs directory before a rerun.
        
        Removes all files including .cif.gz structure files, decompressed
        .cif files, .json score files, and generated input JSON files.
        """
        output_dir = os.path.join(work_dir, "outputs")
        if not os.path.isdir(output_dir):
            return

        for file_path in glob(os.path.join(output_dir, "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)

    def remap_motifs(self, poses: Poses, motifs: list, prefix: str) -> None:
        """
        Update ResidueSelection motifs in poses.df after an RFD3 run.

        Uses the diffused_index_map columns written by collect_scores to remap
        input residue positions to their new positions in the diffused output.
        The motif columns are updated in place — old positions are overwritten.

        Parameters
        ----------
        poses : Poses
            The Poses object after the RFD3 run.
        motifs : list[str]
            Column names in poses.df containing ResidueSelection objects to remap.
        prefix : str
            The prefix used in the RFD3 run, used to find diffused_index_map columns.
        """
        from protflow.residues import ResidueSelection, parse_residue

        logging.info(f"[remap_motifs] Starting motif remapping for prefix='{prefix}'")
        logging.info(f"[remap_motifs] Motifs to remap: {motifs}")
        logging.info(f"[remap_motifs] poses.df has {len(poses.df)} rows and the following columns: {poses.df.columns.tolist()}")

        # find all diffused_index_map columns for this prefix
        map_prefix = f"{prefix}_diffused_index_map_"
        map_cols = [c for c in poses.df.columns if c.startswith(map_prefix)]

        logging.info(f"[remap_motifs] Found {len(map_cols)} diffused_index_map columns: {map_cols}")

        if not map_cols:
            raise ValueError(
                f"No diffused_index_map columns found for prefix '{prefix}'. "
                f"Make sure collect_scores ran successfully and the sidecar JSONs exist."
            )

        for motif_col in motifs:
            logging.info(f"[remap_motifs] Processing motif column '{motif_col}'")

            if motif_col not in poses.df.columns:
                raise ValueError(f"[remap_motifs] Motif column '{motif_col}' not found in poses.df!")

            updated_motifs = []
            for idx, row in poses.df.iterrows():
                logging.info(f"[remap_motifs] Row {idx}: description='{row.get('poses_description', 'N/A')}'")

                # reconstruct mapping dict from columns: {("A", 77): ("A", 96), ...}
                mapping = {}
                for col in map_cols:
                    input_res_str = col.replace(map_prefix, "")  # e.g. "A77"
                    output_res_str = row[col]                     # e.g. "A96"
                    if pd.notna(output_res_str):
                        mapping[parse_residue(input_res_str)] = parse_residue(output_res_str)
                    else:
                        logging.warning(
                            f"[remap_motifs] Row {idx}: NaN value for column '{col}', skipping."
                        )

                logging.info(f"[remap_motifs] Row {idx}: reconstructed mapping: {mapping}")

                # remap the motif
                motif = row[motif_col]
                logging.info(f"[remap_motifs] Row {idx}: original motif '{motif_col}': {motif}")

                if not isinstance(motif, ResidueSelection):
                    raise TypeError(
                        f"Column '{motif_col}' must contain ResidueSelection objects. "
                        f"Got {type(motif)} instead."
                    )

                remapped = tuple(
                    mapping.get(res, res)  # fall back to original if not in map
                    for res in motif.residues
                )

                logging.info(
                    f"[remap_motifs] Row {idx}: remapped motif '{motif_col}': "
                    f"{[f'{c}{r}' for c, r in remapped]}"
                )

                # warn if any residues were not found in the mapping
                not_remapped = [res for res in motif.residues if res not in mapping]
                if not_remapped:
                    logging.warning(
                        f"[remap_motifs] Row {idx}: {len(not_remapped)} residues in '{motif_col}' "
                        f"were not found in diffused_index_map and kept at original positions: "
                        f"{[f'{c}{r}' for c, r in not_remapped]}"
                    )

                updated_motifs.append(ResidueSelection(remapped, fast=True))

            poses.df["residues_postdiffusion"] = updated_motifs
            logging.info(
                f"[remap_motifs] Finished remapping '{motif_col}' for all {len(poses.df)} rows."
            )

        logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")



import logging
from typing import Optional


def _retrieve_underscores_from_settings_group(settings_group_name: str) -> int:
    """
    Count underscores in settings_group_name and return 4 + underscore count.
    
    RFD3 output format: <json_name>_<settings_group>_<batch_number>_model_<n>
    Stripping index_layers from the back recovers the original pose description.
    Base of 4 accounts for: <settings_group>(1+) + <batch_number>(1) + model(1) + <n>(1).
    Additional layers are added for each underscore in settings_group_name.
    """
    underscore_count = settings_group_name.count("_")
    index_layers = 4 + underscore_count
    logging.info(
        f"settings_group_name='{settings_group_name}' contains {underscore_count} "
        f"underscores -> index_layers={index_layers}"
    )
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
    """Flatten nested score dictionaries with optional inclusion of heavy values.

    Notes
    -----
    - Do not hardcode score names where possible; parse what is present.
    - By default, this returns scalar values and skips heavy values.
    - Heavy values are included only if their key (or flattened key path) is in
      ``include_scores``.
    """
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


# decompress .cif.gz to .cif
def _decompress_cif_gz(path: str) -> str:
    """Decompress a .cif.gz file and return path to the decompressed .cif file."""
    out_path = path.replace(".cif.gz", ".cif")
    if not os.path.isfile(out_path):  # don't decompress if already done
        with gzip.open(path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return out_path


def collect_scores(work_dir: str, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Parse runner outputs and return the canonical scores dataframe.

    Developer instructions
    ----------------------
    - Keep this function at module scope, not inside the runner class.
    - Required output columns:
      - ``description``
      - ``location``
    - Favor score auto-discovery (read keys present in outputs) over hardcoded
      column lists, because external tools frequently rename score terms.
    - Avoid reading heavy per-residue / matrix-like data by default.
      Use ``include_scores`` to opt-in to specific heavy fields.
    - Keep function callable standalone (for debugging/re-parsing old runs).
    """
    include_set = set(include_scores or [])
    output_dir = os.path.join(work_dir, "outputs")
    output_paths = sorted(glob(os.path.join(output_dir, "*.cif.gz")))

    # decompress all files first
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


# Optional: lightweight checklist for developers implementing a new runner.
IMPLEMENTATION_CHECKLIST: tuple[str, ...] = (
    "Set config variable names in __init__.",
    "Set correct index_layers for your output naming.",
    "Implement write_cmd with real CLI syntax.",
    "Implement module-level collect_scores (not a class method) with description/location columns.",
    "Make collect_scores auto-discover score keys from tool outputs where possible.",
    "Skip heavy per-residue/matrix outputs by default; gate them behind include_scores list.",
    "Ensure scorefile reuse works when overwrite=False.",
    "Confirm RunnerOutput merge updates poses as expected.",
    "Export runner in protflow/tools/__init__.py (submodule import + class import).",
    "Document API and add tool page in docs/source/tools/<tool>.rst (and tools/index.rst toctree if new).",
    "Build docs warning-free: sphinx-build -b html -W docs/source docs/_build/html",
    "Add/extend unit tests for parsing and option handling.",
)
