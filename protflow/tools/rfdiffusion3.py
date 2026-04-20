"""
RFdiffusion3 Module
=========================================
 
.. module:: protflow.runners.rfdiffusion3
   :synopsis: ProtFlow runner interface for RFDiffusion3, a deep-learning
              model for de novo protein backbone diffusion and motif
              scaffolding.
 
This module provides two public classes and several module-level utilities
for executing RFDiffusion3 within the ProtFlow framework:
 
* :class:`RFD3Params` — a :class:`~collections.UserDict` subclass that
  manages per-pose input specifications for RFDiffusion3, handling both
  de novo and motif-scaffolding modes.
* :class:`RFdiffusion3` — the main ProtFlow :class:`~protflow.runners.Runner`
  subclass that handles JSON construction, CLI assembly, job submission,
  output collection, motif remapping, and scorefile caching.
 
Supported design modes
----------------------
De novo design
    Provide an empty :class:`~protflow.poses.Poses` object together with
    an output-length specification in the :class:`RFD3Params`.  No input
    PDB is required.
 
Motif scaffolding
    Provide a :class:`~protflow.poses.Poses` collection of input PDB
    files.  Residue positions are specified in :class:`RFD3Params` using
    unindexed contigs following the RFDiffusion3 documentation.
 
 
Authors
-------
Sigrid Kaltenbrunner, Adrian Tripp
 
Version
-------
0.1.0

 
Examples
--------
Motif-scaffolding run::
 
    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.runners.rfdiffusion3 import RFdiffusion3, RFD3Params
 
    poses = Poses("scaffolds/", prefix="scaffold")
    jobstarter = SbatchArrayJobstarter(max_cores=8)
 
    params = RFD3Params(poses=poses)
    params.set_input_specs(contig="5-10,A4-20,5-10")
 
    runner = RFdiffusion3()
    poses = runner.run(
        poses=poses,
        prefix="rfd3_motif",
        params=params,
        n_batches=2,
        diffusion_batch_size=8,
        multiplex_poses=4,
    )
 
De novo design::
 
    from protflow.poses import Poses
    from protflow.runners.rfdiffusion3 import RFdiffusion3, RFD3Params
 
    poses = Poses()  # empty — no input structures
    params = RFD3Params(poses=poses)
    params.set_input_specs(length="100-150")
 
    runner = RFdiffusion3()
    poses = runner.run(
        poses=poses,
        prefix="rfd3_denovo",
        params=params,
        n_batches=4,
        diffusion_batch_size=16,
    )
"""
 
 
# ---------------------------------------------------------------------------


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
from ..residues import AtomSelection, ResidueSelection, parse_residue

class RFD3Params(UserDict):
    """
    RFD3InputSpecification class
    ============================

    :class:`RFD3Params` is a helper class to specify input for RFD3. It manages per-pose assignment of input specifications for RFD3 
    in dict-like format
    (see https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields
    for more information on input specification format). <input> fields are infered automatically from poses.
    Each pose is represented by one key in the underlying dictionary.  The
    key is the pose description (filename stem) and the value is a nested
    ``dict`` of field names to specification values.  The mandatory
    ``"input"`` field (the absolute path to the input PDB) is always inferred
    automatically from the :class:`~protflow.poses.Poses` object and should
    not be set manually.
 
    Two construction paths are supported:
 
    * **From poses** (default) — pass a :class:`~protflow.poses.Poses`
      object; a skeleton specification dict with only the ``"input"`` field
      is created for each pose automatically.
    * **From an existing specification** — pass either *spec_from_json*
      (path to a JSON file) or *spec_from_dict* (a nested ``dict`` or
      another :class:`RFD3Params`) to initialise from pre-existing data.
 
    Parameters
    ----------
    poses : Poses
        The current pose collection.  Used to derive the pose descriptions
        (keys) and to auto-populate the ``"input"`` field.  Pass an empty
        :class:`~protflow.poses.Poses` for unconditional (de novo) diffusion
        runs where no input PDB is required.
    spec_from_json : str, optional
        Path to an existing JSON file whose contents should be used as the
        initial specification.  Mutually exclusive with *spec_from_dict*.
    spec_from_dict : dict or RFD3Params, optional
        An existing nested specification dictionary or another
        :class:`RFD3Params` instance.  Mutually exclusive with
        *spec_from_json*.
 
    Attributes
    ----------
    poses : Poses
        Reference to the current pose collection.
    data : dict
        The underlying specification dictionary.  Keys are pose descriptions;
        values are ``dict`` objects mapping field names to values.
 
    Raises
    ------
    ValueError
        If both *spec_from_json* and *spec_from_dict* are provided.
 
    Notes
    -----
    * Only one of *spec_from_json* and *spec_from_dict* may be provided; if
      neither is given, the specification is initialised from *poses*.
    * When initialising from an existing specification (JSON or dict), the
      ``"input"`` fields are updated to reflect the current ``poses.df``
      paths via :meth:`_update_input`.
    * The ``data`` property mirrors :attr:`~collections.UserDict.data` and
      is the canonical access point for the raw specification dict.
 
    Examples
    --------
    Build from poses with a shared contig::
 
        params = RFD3Params(poses=poses)
        params.set_input_specs(contig="5-10,A4-20,5-10")
 
    Build from a JSON file::
 
        params = RFD3Params(poses=poses, spec_from_json="/data/my_spec.json")
 
    Build for de novo diffusion (no input PDB)::
 
        params = RFD3Params(poses=Poses())
        params.set_input_specs(length="100-150") 
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
    def input_specs(self) -> dict:
        """Return the raw specification dictionary.
 
        Returns
        -------
        dict
            The underlying ``data`` dictionary mapping pose descriptions to
            their per-pose specification dicts.
        """
        return self.data

    def set_input_specs(self, **kwargs)  -> RFD3Params:
        
        """
        Apply the same input specification fields to all poses.
 
        Updates the specification for every pose in :attr:`data` with the
        supplied keyword arguments.  This is the preferred method for
        setting fields that are identical across all input poses (e.g. a
        shared contig string, a temperature value, or a noise schedule).
 
        Parameters
        ----------
        **kwargs
            Arbitrary RFDiffusion3 input specification fields passed as
            keyword arguments.  Field names correspond exactly to those
            described in the
            `RFDiffusion3 input documentation
            <https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md#inputspecification-fields>`_.
            Common examples include:
 
            ``contig`` : str
                Contig string specifying fixed and free residue segments
                (e.g. ``"5-10,A4-20,5-10"``).
            ``length`` : str
                Total output length range for de novo design
                (e.g. ``"100-150"``).
            ``hotspot_res`` : list of str
                Hotspot residues that should be near the diffused region.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Warns
        -----
        logging.WARNING
            Emitted if ``"input"`` is included in *kwargs*, since the
            ``input`` field is reserved for automatic population from poses
            and manual overrides require absolute paths.
 
        Notes
        -----
        * If :attr:`data` is empty when this method is called (e.g. after
          constructing with an empty :class:`~protflow.poses.Poses`), a
          minimal ``{"denovo": {}}`` skeleton is created automatically
          before applying the kwargs.
        * Setting ``"input"`` manually is not recommended.  When poses are
          present, :meth:`_update_input` will overwrite it with the correct
          absolute paths anyway.
 
        Examples
        --------
        Shared contig for all poses::
 
            params = RFD3Params(poses=poses)
            params.set_input_specs(contig="10,A5-30,10", noise_scale=0.5)
 
        De novo design with a length range::
 
            params = RFD3Params(poses=Poses())
            params.set_input_specs(length="80-120")
        
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
        Apply pose-specific input specification fields from DataFrame columns or lists.
 
        For each keyword argument, the value can be either the name of a
        column in ``poses.df`` or a list of per-pose values.  Each pose is
        updated with its corresponding value; ``NaN`` / ``None`` entries
        are silently skipped so that poses without a specification for a
        given field are left unchanged.
 
        Parameters
        ----------
        **kwargs
            Per-pose specification fields.  Each value must be one of:
 
            ``str``
                The name of a column in ``poses.df``.  Each row's value is
                assigned to the corresponding pose.
            ``list``
                A list of values with length equal to ``len(poses)``.  Each
                element is assigned to the pose at the same index.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Raises
        ------
        ValueError
            If :attr:`poses` is empty (unconditional diffusion has no per-
            pose concept).
        ValueError
            If a list value has a different length than the number of poses.
        TypeError
            If a value is neither a ``str`` (column name) nor a ``list``.
 
        Warns
        -----
        logging.WARNING
            Emitted if ``"input"`` is among the kwargs.
 
        Notes
        -----
        * Only non-null values update the specification; poses for which a
          field is ``NaN`` or ``None`` retain their previous (or default)
          specification.
        * Column lookup is validated by :func:`~protflow.poses.col_in_df`
          before any updates are applied.
 
        Examples
        --------
        Different contig strings per pose, stored in a DataFrame column::
 
            params = RFD3Params(poses=poses)
            params.set_per_pose_input_specs(contig="contig_col")
 
        Per-pose hotspot residues supplied as a list::
 
            hotspots = [["A12", "A15"], ["A20"], ["A7", "A8", "A9"]]
            params.set_per_pose_input_specs(hotspot_res=hotspots)
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
        Replace the current specification with data from a nested dict.
 
        Validates the format of *spec_dict*, replaces :attr:`data`, and
        refreshes the ``"input"`` fields for any poses currently in
        :attr:`poses`.
 
        Parameters
        ----------
        spec_dict : dict or RFD3Params
            A nested specification dictionary in the format
            ``{pose_description: {field: value, ...}, ...}``.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Raises
        ------
        ValueError
            If *spec_dict* is not a properly nested dictionary (validated
            by :meth:`_check_specs`).
        ValueError
            If the pose descriptions in *spec_dict* do not match those in
            :attr:`poses` (when poses are present).
 
        Examples
        --------
        ::
 
            my_dict = {"scaffold_001": {"contig": "5,A4-20,5"}}
            params.spec_from_dict(my_dict)
        """

        self._check_specs(spec_dict)
        self.data = spec_dict
        self._update_input()
        return self
    
    def spec_from_json(self, json_path: str) -> RFD3Params:
        """
        Replace the current specification by loading a JSON file from disk.
 
        Reads *json_path*, validates the resulting dictionary, replaces
        :attr:`data`, and refreshes the ``"input"`` fields.
 
        Parameters
        ----------
        json_path : str
            Absolute or resolvable path to a JSON file whose top-level
            structure is a nested specification dictionary.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Raises
        ------
        ValueError
            If no file exists at *json_path*.
        ValueError
            If the parsed dictionary is not correctly nested
            (validated by :meth:`_check_specs`).
 
        Examples
        --------
        ::
 
            params.spec_from_json("/data/my_experiment_spec.json")
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
        Clear all existing specifications and reinitialise from a new pose collection.
 
        Replaces :attr:`poses` with *poses*, discards all current entries in
        :attr:`data`, and rebuilds the skeleton specification dictionary with
        only the ``"input"`` fields populated.
 
        Parameters
        ----------
        poses : Poses
            The new pose collection to use as the basis for the
            specification.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Notes
        -----
        All previously set specification fields (contigs, hotspots, etc.)
        are lost.  Call :meth:`set_input_specs` or
        :meth:`set_per_pose_input_specs` after this method to repopulate.
 
        Examples
        --------
        ::
 
            params.reset_pose_specs(new_poses)
            params.set_input_specs(contig="5,A4-20,5")
        """
        self.poses = poses
        self.data = self._create_pose_dict(self.poses)
        return self

    def add_specs(self, additional_specs: RFD3Params | dict) -> RFD3Params:
        """
        Add new pose entries to an unconditional (de novo) specification.
 
        Merges the entries from *additional_specs* into :attr:`data`.  This
        method is restricted to instances created with an empty
        :class:`~protflow.poses.Poses` (unconditional diffusion), where
        multiple independent specification blocks can be accumulated before
        a single run.
 
        Parameters
        ----------
        additional_specs : RFD3Params or dict
            Additional specification entries to merge into :attr:`data`.
            Must be a properly nested dictionary.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Raises
        ------
        ValueError
            If :attr:`poses` is non-empty (this method is only permitted
            for unconditional diffusion without input poses).
        ValueError
            If *additional_specs* is not a properly nested dictionary
            (validated by :meth:`_check_specs`).
 
        Notes
        -----
        Because ``dict.update`` is used, duplicate keys in
        *additional_specs* will silently overwrite existing entries.
 
        Examples
        --------
        ::
 
            params = RFD3Params(poses=Poses())
            params.set_input_specs(length="100-120")
 
            extra = {"denovo_long": {"length": "150-200"}}
            params.add_specs(extra)
        """

        if self.poses:
            raise ValueError("Additional pose-specific input specifications can ony be added if no poses are present (unconditional diffusion)!")
        self._check_specs(additional_specs)
        self.data.update(additional_specs)
        return self

    def modify_specs(self, new_specs: RFD3Params | dict) -> RFD3Params:
        """
        Update fields in existing pose specifications without adding or removing poses.
 
        For each pose key present in both :attr:`data` and *new_specs*, the
        corresponding specification dictionary is updated with the values
        from *new_specs*.  The set of poses must match exactly: no new poses
        may be introduced and no existing poses may be omitted.
 
        Parameters
        ----------
        new_specs : RFD3Params or dict
            A nested specification dictionary with the same pose keys as
            :attr:`data`.  Only the fields listed in each per-pose dict are
            updated; all other existing fields are preserved.
 
        Returns
        -------
        RFD3Params
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If the pose keys in *new_specs* do not exactly match those in
            :attr:`data` (checked by key membership and length equality).
 
        Notes
        -----
        After updating, :meth:`_update_input` is called to ensure that the
        ``"input"`` field remains consistent with the current pose paths.
 
        Examples
        --------
        ::
 
            corrections = {"scaffold_001": {"contig": "8,A4-20,8"}}
            params.modify_specs(corrections)
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
        Refresh the ``"input"`` field for every pose from the current pose paths.
 
        Iterates over ``poses.df`` and sets ``data[description]["input"]``
        to the absolute path of each pose file.  Does nothing when
        :attr:`poses` is empty or :attr:`data` is empty (unconditional
        diffusion).
 
        Notes
        -----
        This method is called internally after :meth:`spec_from_dict`,
        :meth:`spec_from_json`, and :meth:`modify_specs` to keep the
        ``"input"`` field consistent with the actual pose paths on disk.
        Direct manipulation of the ``"input"`` field is strongly
        discouraged; always let this method maintain it.
        """
        if self.poses and self.data:
            for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"]):
                self.data[name].update({"input": os.path.abspath(path)})
    
    def _check_specs(self, specs: RFD3Params | dict):
        """
        Validate that *specs* conforms to the required nested-dict format.
 
        Parameters
        ----------
        specs : RFD3Params or dict
            The specification dictionary to validate.
 
        Raises
        ------
        ValueError
            If any top-level value in *specs* is not itself a ``dict``
            (i.e. the structure is not properly nested).  An illustrative
            example of the required format is included in the error message.
        ValueError
            If :attr:`poses` is non-empty and the pose descriptions in
            *specs* do not match those in ``poses.df["poses_description"]``,
            or if the lengths differ.
 
        Notes
        -----
        This method is called internally before any assignment to
        :attr:`data` to prevent silent corruption of the specification.
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
        Build a skeleton specification dict from a pose collection.
 
        Creates one entry per pose with only the ``"input"`` field
        (absolute path) populated.  Returns an empty dict for an empty
        :class:`~protflow.poses.Poses` (de novo mode).
 
        Parameters
        ----------
        poses : Poses
            The pose collection to build from.
 
        Returns
        -------
        dict
            A nested dict ``{pose_description: {"input": abs_path}, ...}``
            for each pose, or ``{}`` if *poses* is empty.
        """
        if poses:
            return {name: {"input": os.path.abspath(path)} for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"])}
        else:
            return {}

    
class RFdiffusion3(Runner):
    """
    RFdiffusion3 Class
    ==================
    ProtFlow runner for RFDiffusion3 backbone diffusion.
 
    :class:`RFdiffusion3` inherits from :class:`~protflow.runners.Runner`
    and wraps RFDiffusion3's command-line interface to support both de novo
    backbone generation and motif-scaffolding runs inside ProtFlow pipelines.
 
    The runner manages the full lifecycle of a diffusion run:
 
    1. **Input JSON construction** — per-pose specifications from an
       :class:`RFD3Params` object are serialised to per-batch JSON files.
    2. **CLI assembly** — commands are composed from the JSON path, output
       directory, checkpoint, and user-supplied options.
    3. **Job submission** — commands are submitted via the ProtFlow
       :class:`~protflow.jobstarters.JobStarter` abstraction.
    4. **Output collection** — ``.cif.gz`` / sidecar ``.json`` files are
       decompressed, optionally converted to PDB, and flattened into a
       :class:`~pandas.DataFrame`.
    5. **Motif remapping** — when requested, residue positions in designated
       :class:`~protflow.residues.ResidueSelection` columns are translated
       from input to diffused-output numbering using the ``diffused_index_map``
       fields in the sidecar JSONs.
    6. **Scorefile caching** — results are cached to disk and reloaded on
       subsequent calls unless *overwrite* is ``True``.
 
    Parameters
    ----------
    application_path : str, optional
        Absolute path to the RFDiffusion3 binary (``rfd3`` or equivalent).
        Resolved from the ProtFlow config key ``RFDIFFUSION3_BIN_PATH`` when
        omitted.
    model_dir : str, optional
        Directory containing RFDiffusion3 checkpoint files (``*.ckpt``).
        Resolved from ``RFDIFFUSION3_MODEL_DIR`` when omitted.
    pre_cmd : str, optional
        Shell preamble prepended to every generated command (e.g.
        ``"conda activate rfd3_env &&"``).  Resolved from
        ``RFDIFFUSION3_PRE_CMD`` when omitted.
    jobstarter : JobStarter, optional
        Default :class:`~protflow.jobstarters.JobStarter` instance used when
        :meth:`run` is called without an explicit *jobstarter* argument.
 
    Attributes
    ----------
    application_path : str
        Resolved path to the RFDiffusion3 binary.
    model_dir : str
        Resolved path to the checkpoint directory.
    pre_cmd : str or None
        Shell preamble, or ``None`` when not set.
    jobstarter : JobStarter or None
        Default job submission backend.
    name : str
        Runner identifier (``"rfdiffusion3"``).
    index_layers : int
        Base number of index layers added per pose (``1``).  The effective
        value is increased internally when multiplex mode is used.
 
    Notes
    -----
    * Pre-existing JSON input files are **not** supported; the input JSON
      is always constructed internally from the :class:`RFD3Params` object.
 
    Examples
    --------
    ::
 
        from protflow.runners.rfdiffusion3 import RFdiffusion3, RFD3Params
 
        runner = RFdiffusion3()
        params = RFD3Params(poses=poses)
        params.set_input_specs(contig="5-10,A4-20,5-10")
 
        poses = runner.run(poses=poses, prefix="rfd3", params=params,
                           n_batches=2, diffusion_batch_size=8)
    """

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
        """
        Execute the full RFDiffusion3 runner lifecycle and return updated poses.
 
        Orchestrates the complete pipeline from input preparation through
        job execution to result integration:
 
        1. Validates that *poses* and *params* are consistent.
        2. Resolves the checkpoint path.
        3. Optionally multiplexes input poses to saturate available GPUs.
        4. Checks for a cached scorefile and returns early if found.
        5. Splits pose specifications into batches and writes input JSONs.
        6. Assembles and submits CLI commands via the jobstarter.
        7. Collects output scores from ``.cif.gz`` / ``.json`` files.
        8. Optionally remaps residue motifs using the diffused index map.
        9. Reindexes poses to remove extra index layers and returns.
 
        Parameters
        ----------
        prefix : str
            Column prefix used to namespace all new columns added to
            ``poses.df`` and to name the working directory
            (``<poses.work_dir>/<prefix>/``).
        poses : Poses
            Input pose collection.  May be empty for de novo diffusion;
            in that case ``poses.df`` is populated directly from the
            collected scores.
        params : RFD3Params
            Per-pose input specification object.  Must contain exactly one
            entry per pose in *poses* (or be empty/denovo).  The ``"input"``
            field is managed automatically; all other fields are forwarded
            verbatim to RFDiffusion3.
        n_batches : int, optional
            Number of diffusion batches RFDiffusion3 runs per input JSON
            file.  Total designs per pose =
            ``n_batches x diffusion_batch_size``.  Default is ``1``.
        diffusion_batch_size : int, optional
            Number of structures generated per batch inside RFDiffusion3.
            Default is ``8``.
        ckpt_path : str, optional
            Checkpoint to use for diffusion.  Accepts either:
 
            * A named alias looked up as
              ``<model_dir>/<name>.ckpt``, **or**
            * An absolute path to a ``.ckpt`` file.
 
            Default is ``"rfd3_latest"``.
        options : str, optional
            Additional RFDiffusion3 CLI arguments in ``key=value`` format,
            forwarded verbatim after the mandatory arguments.  Must **not**
            contain any of ``inputs``, ``out_dir``, ``ckpt_path``,
            ``n_batches``, or ``diffusion_batch_size`` (those are set
            programmatically).
        update_motifs : list of str, optional
            Names of :class:`~protflow.residues.ResidueSelection` columns
            in ``poses.df`` whose residue positions should be remapped from
            input to diffused-output numbering after the run.  Requires that
            the sidecar JSON files contain ``diffused_index_map`` entries.
            When ``None`` (default), no remapping is performed.
        multiplex_poses : int, optional
            If set to an integer > 1, each input pose is duplicated this
            many times before submission to maximise GPU utilisation when
            the number of input poses is smaller than the number of
            available GPUs.  After the run, the extra index layer introduced
            by multiplexing is collapsed during reindexing.  A value of
            ``1`` has no effect and triggers a warning.
        jobstarter : JobStarter, optional
            Job submission backend for this run.  Resolved via the standard
            ProtFlow fallback chain: argument → ``self.jobstarter`` →
            ``poses.default_jobstarter``.
        convert_cif_to_pdb : bool, optional
            When ``True`` (default), decompressed ``.cif`` files are
            converted to ``.pdb`` format via OpenBabel.  When ``False``,
            the ``location`` column in the returned poses points to
            ``.cif`` files.
        strict_remap : bool, optional
            When ``True`` (default), :func:`remap_rfd3_motifs` raises a
            :exc:`ValueError` if any residue in a motif column is absent
            from the ``diffused_index_map``, or if the map itself is missing
            for any pose.  Set to ``False`` to silently skip unmapped
            residues.
        run_clean : bool, optional
            When ``True`` (default), intermediate files (compressed ``.cif.gz``
            and, when *convert_cif_to_pdb* is ``True``, the intermediate
            ``.cif`` files) are deleted after successful output collection to
            minimise disk usage.
        fail_on_missing_output_poses : bool, optional
            When ``True``, a :exc:`RuntimeError` is raised if the number of
            collected output poses is less than
            ``n_batches x diffusion_batch_size x len(pose_specs)``.
            Useful for catching silent RFDiffusion3 job failures early in
            long pipelines.  Default is ``False``.
        overwrite : bool, optional
            When ``True``, any existing scorefile and output files in the
            working directory are deleted before re-running.  When ``False``
            (default), an existing scorefile causes the run to be skipped
            and cached results to be returned immediately.
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns
            prefixed by *prefix*, including:
 
            ``<prefix>_location``
                Absolute path to the output PDB or CIF file for each
                diffused structure.
            ``<prefix>_description``
                Unique identifier derived from the output filename stem.
            ``<prefix>_diffused_index_map``
                Mapping from input residue identifiers to their new
                positions in the diffused structure (required for
                :func:`remap_rfd3_motifs`).
            All numeric metrics from the sidecar ``metrics`` block,
            flattened into individual columns (e.g.
            ``<prefix>_plddt``, ``<prefix>_ptm``, etc.).
 
        Raises
        ------
        ValueError
            If *poses* and *params* have different lengths or mismatched
            pose descriptions.
        ValueError
            If *ckpt_path* cannot be resolved to an existing file in
            *model_dir* or as an absolute path.
        ValueError
            If *options* contains any of the reserved argument names
            (``inputs``, ``out_dir``, ``ckpt_path``, ``n_batches``,
            ``diffusion_batch_size``).
        RuntimeError
            If :func:`collect_scores` returns an empty DataFrame (all jobs
            crashed or produced no output).
        RuntimeError
            If *fail_on_missing_output_poses* is ``True`` and the number of
            collected outputs is less than the expected count.
 
        Warns
        -----
        logging.WARNING
            If *multiplex_poses* is set to ``1`` (no-op).
 
        Notes
        -----
        * The total expected number of output structures per input pose is
          ``n_batches x diffusion_batch_size``.  When *multiplex_poses* > 1,
          the total across all jobs is further multiplied.
        * When *poses* is empty (de novo mode), ``poses.df`` is populated
          directly from the score DataFrame; ``input_poses`` is set to
          ``None`` for all rows.
        * The effective *index_layers* value used for
          :class:`~protflow.runners.RunnerOutput` is:
 
          - ``self.index_layers + 2`` (= 3) for standard runs.
          - ``self.index_layers + 3`` (= 4) for multiplex runs.
 
          These are collapsed by :meth:`~protflow.poses.Poses.reindex_poses`
          at the end of the method.
 
        Examples
        --------
        Standard motif-scaffolding run with 64 designs per input::
 
            poses = runner.run(
                prefix="rfd3_scaffold",
                poses=poses,
                params=params,
                n_batches=4,
                diffusion_batch_size=16,
                update_motifs=["binding_site"],
            )
 
        De novo design with GPU multiplexing::
 
            poses = runner.run(
                prefix="rfd3_denovo",
                poses=Poses(),
                params=params,
                n_batches=2,
                diffusion_batch_size=8,
                multiplex_poses=4,
                convert_cif_to_pdb=True,
            )
 
        Reuse cached results from a previous run::
 
            poses = runner.run(
                prefix="rfd3_scaffold",
                poses=poses,
                params=params,
                overwrite=False,  # default — returns cached scorefile
            )
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
        """
        Partition pose specifications into batches, write input JSONs, and assemble commands.
 
        Splits *pose_specs* into *n_jobs* sublists using
        :func:`~protflow.jobstarters.split_list`, serialises each sublist to
        a JSON file in *input_dir*, and returns a list of shell command
        strings — one per JSON / job — ready for submission.
 
        Parameters
        ----------
        pose_specs : list of dict
            A list of single-pose specification dicts, each in the format
            ``{pose_description: {field: value, ...}}``, as constructed in
            :meth:`run` from the :class:`RFD3Params` object.
        input_dir : str
            Directory into which per-batch input JSON files
            (``batch0.json``, ``batch1.json``, …) are written.
        output_dir : str
            Directory passed to RFDiffusion3 as ``out_dir``; all output
            files are written here.
        n_jobs : int
            Number of batches / parallel jobs to create.  Must be
            ≤ ``len(pose_specs)`` and ≤ ``jobstarter.max_cores``.
        ckpt_path : str
            Absolute path to the resolved checkpoint file.
        options : str, optional
            Additional CLI arguments forwarded to :meth:`write_cmd`.
        n_batches : int, optional
            RFDiffusion3 internal batch count, forwarded to each command.
            Default is ``1``.
        diffusion_batch_size : int, optional
            Structures per internal RFDiffusion3 batch, forwarded to each
            command.  Default is ``8``.
 
        Returns
        -------
        list of str
            Shell command strings, one per batch JSON file.
 
        Notes
        -----
        Each batch JSON file consolidates the specification dicts from its
        sublist into a single flat top-level dict (``{pose_desc: spec, ...}``)
        before serialisation.
 
        Examples
        --------
        ::
 
            cmds = runner.setup_run(
                pose_specs=pose_specs,
                input_dir="/scratch/run/inputs",
                output_dir="/scratch/run/outputs",
                n_jobs=8,
                ckpt_path="/models/rfd3_latest.ckpt",
                n_batches=2,
                diffusion_batch_size=16,
            )
        """

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
        """
        Compose a single RFDiffusion3 shell command string.
 
        Combines :attr:`application_path` with the mandatory CLI arguments
        and any user-supplied options into a single executable string.
 
        Parameters
        ----------
        in_json : str
            Absolute path to the input specification JSON file for this
            batch.
        out_dir : str
            Output directory path passed as ``out_dir=`` to RFDiffusion3.
        ckpt_path : str
            Absolute path to the checkpoint file passed as ``ckpt_path=``.
        options : str, optional
            Additional ``key=value`` CLI arguments appended verbatim after
            the mandatory arguments.  Must not contain any of the reserved
            argument names listed below.
        n_batches : int, optional
            Number of diffusion batches.  Default is ``1``.
        diffusion_batch_size : int, optional
            Structures per batch.  Default is ``8``.
 
        Returns
        -------
        str
            A complete shell command string, e.g.::
 
                /opt/rfd3/rfd3 design inputs=/run/inputs/batch0.json \
                    out_dir=/run/outputs ckpt_path=/models/rfd3.ckpt \
                    n_batches=2 diffusion_batch_size=16
 
        Raises
        ------
        ValueError
            If *options* contains any of the reserved argument names:
            ``inputs``, ``out_dir``, ``ckpt_path``, ``n_batches``, or
            ``diffusion_batch_size``.  These must always be set via the
            explicit method parameters.
 
        Examples
        --------
        ::
 
            cmd = runner.write_cmd(
                in_json="/scratch/run/inputs/batch0.json",
                out_dir="/scratch/run/outputs",
                ckpt_path="/models/rfd3_latest.ckpt",
                n_batches=2,
                diffusion_batch_size=8,
            )
        """

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
        """
        Delete the ``outputs/`` and ``inputs/`` sub-directories before a rerun.
 
        Called internally by :meth:`run` when *overwrite* is ``True``.
        Removes ``<work_dir>/outputs/`` and ``<work_dir>/inputs/``
        recursively using :func:`shutil.rmtree` if they exist, ensuring
        that stale files from a previous run do not interfere with
        output collection.
 
        Parameters
        ----------
        work_dir : str
            Root working directory of the run (the ``<poses.work_dir>/<prefix>/``
            path created by :meth:`~protflow.runners.Runner.generic_run_setup`).
 
        Notes
        -----
        Only the ``outputs/`` and ``inputs/`` sub-directories are removed.
        The scorefile (``rfdiffusion3_scores.<format>``) is handled
        separately by the base-class :meth:`~protflow.runners.Runner.check_for_existing_scorefile`
        logic.
        """
        output_dir = os.path.join(work_dir, "outputs")
        input_dir = os.path.join(work_dir, "inputs")
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        if os.path.isdir(input_dir):
            shutil.rmtree(input_dir)

def _ensure_rfd3_index_map(diff_idx_map: dict | None, motif_col: str, strict: bool) -> dict:
    """
    Validate one RFD3 ``diffused_index_map`` before motif remapping.

    Parameters
    ----------
    diff_idx_map : dict or None
        Per-pose mapping from original residue identifiers, for example
        ``"A5"``, to output residue identifiers, for example ``"B12"``.
    motif_col : str
        Name of the motif column being remapped. This is used only to make
        error messages specific to the failing column.
    strict : bool
        If ``True``, missing or empty maps are treated as errors. If
        ``False``, missing maps are converted to an empty dictionary and
        unmapped residues or atoms are skipped by the caller.

    Returns
    -------
    dict
        Validated index map. The returned dictionary may be empty only when
        ``strict`` is ``False``.

    Raises
    ------
    ValueError
        If ``strict`` is ``True`` and *diff_idx_map* is missing, empty, or not
        a dictionary.
    """
    if isinstance(diff_idx_map, dict):
        if strict and not diff_idx_map:
            raise ValueError("Not all poses feature diffused_index_map property. Are you sure preserved motifs exist for all input poses?")
        return diff_idx_map
    if strict:
        raise ValueError(f"Pose has no usable diffused_index_map while remapping selection {motif_col}. Got: {diff_idx_map}")
    return {}


def _remap_residue_selection(selection: ResidueSelection, diff_idx_map: dict, motif_col: str, strict: bool) -> ResidueSelection:
    """
    Remap one ResidueSelection through an RFD3 ``diffused_index_map``.

    Parameters
    ----------
    selection : ResidueSelection
        Residue selection in input-structure numbering.
    diff_idx_map : dict
        Mapping from input residue identifiers to output residue identifiers.
    motif_col : str
        Name of the column being remapped. Used for error messages.
    strict : bool
        If ``True``, every selected residue must be present in
        *diff_idx_map*. If ``False``, residues absent from the map are skipped.

    Returns
    -------
    ResidueSelection
        New residue selection in output-structure numbering.

    Raises
    ------
    ValueError
        If ``strict`` is ``True`` and any selected residue is absent from the
        map.
    """
    residues = selection.to_list()
    if strict and not all(residue in diff_idx_map for residue in residues):
        raise ValueError(f"Could not find all original residues in diffused index map for selection {motif_col}. Are you sure they were preserved?")
    return ResidueSelection([diff_idx_map[residue] for residue in residues if residue in diff_idx_map])


def _atom_residue_number(residue_id) -> int:
    """
    Extract the numeric residue component from compact or BioPython atom IDs.

    Parameters
    ----------
    residue_id : int, str, list, or tuple
        Compact residue ID or BioPython residue ID tuple
        ``(hetero_flag, residue_number, insertion_code)``.

    Returns
    -------
    int
        Numeric residue number used to look up ``diffused_index_map`` entries.
    """
    if isinstance(residue_id, (list, tuple)) and len(residue_id) == 3:
        return int(residue_id[1])
    return int(residue_id)


def _format_remapped_atom_residue_id(original_residue_id, remapped_residue_number: int):
    """
    Format a remapped residue ID to match the original atom ID style.

    Parameters
    ----------
    original_residue_id : int, str, list, or tuple
        Residue ID component from the original AtomSelection atom ID.
    remapped_residue_number : int
        Residue number parsed from the RFD3 output residue identifier.

    Returns
    -------
    int, str, or tuple
        Residue ID with the same broad representation as
        *original_residue_id*. BioPython hetero flags are preserved. Insertion
        codes are reset to blank because RFD3 ``diffused_index_map`` values do
        not encode insertion codes.

    Raises
    ------
    ValueError
        If *original_residue_id* is not a supported residue ID form.
    """
    if isinstance(original_residue_id, int):
        return remapped_residue_number
    if isinstance(original_residue_id, str):
        return str(remapped_residue_number)
    if isinstance(original_residue_id, (list, tuple)) and len(original_residue_id) == 3:
        hetero_flag = original_residue_id[0] or " "
        return (hetero_flag, remapped_residue_number, " ")
    raise ValueError(f"Cannot remap unsupported atom residue ID: {original_residue_id}")


def _remap_atom_id(atom_id, diff_idx_map: dict, motif_col: str, strict: bool):
    """
    Remap the chain/residue part of one AtomSelection atom ID.

    Parameters
    ----------
    atom_id : tuple or list
        Atom ID in one of the compact or BioPython-style formats supported by
        :class:`protflow.residues.AtomSelection`.
    diff_idx_map : dict
        Mapping from input residue identifiers to output residue identifiers.
    motif_col : str
        Name of the column being remapped. Used for error messages.
    strict : bool
        If ``True``, the atom's source residue must be present in
        *diff_idx_map*. If ``False``, missing residues return ``None`` so the
        caller can skip that atom.

    Returns
    -------
    tuple or None
        Remapped atom ID with the original atom-ID shape preserved. ``None`` is
        returned only when ``strict`` is ``False`` and the atom's residue is not
        present in *diff_idx_map*.

    Raises
    ------
    ValueError
        If *atom_id* has an unsupported shape, or if ``strict`` is ``True`` and
        the atom's residue is absent from the map.
    """
    atom_id = tuple(atom_id)
    # Decompose each supported atom ID shape into prefix, chain/residue, and
    # suffix components so the final atom ID can be reconstructed losslessly.
    if len(atom_id) == 3:
        chain_id, residue_id, atom_name = atom_id
        prefix_values = ()
        suffix_values = (atom_name,)
    elif len(atom_id) == 4:
        model_id, chain_id, residue_id, atom_name = atom_id
        prefix_values = (model_id,)
        suffix_values = (atom_name,)
    elif len(atom_id) == 5:
        structure_id, model_id, chain_id, residue_id, atom_name = atom_id
        prefix_values = (structure_id, model_id)
        suffix_values = (atom_name,)
    elif len(atom_id) == 6:
        structure_id, model_id, chain_id, residue_id, atom_name, altloc = atom_id
        prefix_values = (structure_id, model_id)
        suffix_values = (atom_name, altloc)
    else:
        raise ValueError(f"Atom IDs must have 3, 4, 5, or 6 elements while remapping {motif_col}. Got: {atom_id}")

    # RFD3 maps residue identifiers without atom names, so atom-level remapping
    # first collapses each atom ID to the corresponding residue key.
    source_residue = f"{chain_id}{_atom_residue_number(residue_id)}"
    if source_residue not in diff_idx_map:
        if strict:
            raise ValueError(f"Could not find original atom residue {source_residue} in diffused index map for selection {motif_col}. Are you sure it was preserved?")
        return None

    remapped_chain, remapped_residue_number = parse_residue(diff_idx_map[source_residue])
    remapped_residue_id = _format_remapped_atom_residue_id(residue_id, remapped_residue_number)
    # Preserve atom names, altlocs, model IDs, and structure IDs exactly; only
    # the chain and residue index come from RFD3.
    return (*prefix_values, remapped_chain, remapped_residue_id, *suffix_values)


def _remap_atom_selection(selection: AtomSelection, diff_idx_map: dict, motif_col: str, strict: bool) -> AtomSelection:
    """
    Remap one AtomSelection through an RFD3 ``diffused_index_map``.

    Parameters
    ----------
    selection : AtomSelection
        Atom selection in input-structure numbering.
    diff_idx_map : dict
        Mapping from input residue identifiers to output residue identifiers.
    motif_col : str
        Name of the column being remapped. Used for error messages.
    strict : bool
        If ``True``, every atom's residue must be present in *diff_idx_map*.
        If ``False``, atoms whose residues are absent from the map are skipped.

    Returns
    -------
    AtomSelection
        New atom selection in output-structure numbering. Atom order and atom
        names are preserved.
    """
    remapped_atoms = []
    for atom_id in selection.to_tuple():
        remapped_atom_id = _remap_atom_id(atom_id, diff_idx_map=diff_idx_map, motif_col=motif_col, strict=strict)
        if remapped_atom_id is not None:
            remapped_atoms.append(remapped_atom_id)
    return AtomSelection(remapped_atoms)


def remap_rfd3_motifs(poses: Poses, motifs: list[str], prefix: str, strict: bool = True) -> None:
    """Remap ResidueSelection or AtomSelection motifs to diffused-output numbering.
 
    Translates residue positions stored in designated
    :class:`~protflow.residues.ResidueSelection` or
    :class:`~protflow.residues.AtomSelection` columns from their original
    input-structure numbering to the corresponding positions in the diffused
    output structures, using the per-pose ``diffused_index_map`` entries
    produced by RFDiffusion3 and stored in ``poses.df`` by :func:`collect_scores`.
 
    The motif columns are updated **in place**, matching the convention
    established by the RFDiffusion 1 runner in ProtFlow.
 
    Parameters
    ----------
    poses : Poses
        Pose collection returned by :meth:`RFdiffusion3.run`, or any object
        exposing a ``df`` attribute compatible with :class:`pandas.DataFrame`.
        ``poses.df`` must contain a column named
        ``"<prefix>_diffused_index_map"``. Each value in that column should be
        a dictionary mapping input residue identifiers, such as ``"A5"``, to
        output residue identifiers, such as ``"B12"``.
    motifs : list of str
        Names of columns in ``poses.df`` to remap in place. Each named column
        must be homogeneous: all values in one column must be either
        :class:`~protflow.residues.ResidueSelection` objects or
        :class:`~protflow.residues.AtomSelection` objects. Different columns
        may use different selection types.
    prefix : str
        The run prefix used to locate the ``<prefix>_diffused_index_map``
        column. For example, ``prefix="rfd3"`` reads the column
        ``"rfd3_diffused_index_map"``.
    strict : bool, optional
        When ``True`` (default):
 
        * Raises a :exc:`ValueError` if any residue in a motif selection
          is absent from the ``diffused_index_map`` for its pose. For
          AtomSelections this check is performed at the residue level for each
          atom ID.
        * Raises a :exc:`ValueError` if the ``diffused_index_map`` is
          empty (``None`` / ``{}``) for any pose, indicating that no
          motif was preserved during diffusion.
 
        When ``False``, residues not found in the map are silently skipped
        and the remapped selection will contain only the successfully
        mapped subset.
 
    Returns
    -------
    None
        This function modifies ``poses.df`` in place. Remapped
        ResidueSelection columns remain ResidueSelection columns, and remapped
        AtomSelection columns remain AtomSelection columns.
 
    Raises
    ------
    KeyError
        If any column name in *motifs* or ``<prefix>_diffused_index_map``
        is absent from ``poses.df`` (raised by
        :func:`~protflow.poses.col_in_df`).
    ValueError
        If a motif column is empty or contains mixed/unsupported object types.
        Supported types are :class:`~protflow.residues.ResidueSelection` and
        :class:`~protflow.residues.AtomSelection`.
    ValueError
        If *strict* is ``True`` and a residue from a ResidueSelection, or an
        atom's residue from an AtomSelection, is absent from the corresponding
        ``diffused_index_map``.
    ValueError
        If *strict* is ``True`` and the ``diffused_index_map`` is empty
        for any pose (motifs were not preserved for all inputs).
 
    Notes
    -----
    * ResidueSelection values are remapped residue-wise using their
      :meth:`~protflow.residues.ResidueSelection.to_list` representation.
    * AtomSelection values keep atom order, atom names, model IDs, structure
      IDs, altlocs, and BioPython hetero flags unchanged. Only the chain and
      numeric residue index are replaced according to ``diffused_index_map``.
    * BioPython insertion codes in AtomSelection residue IDs are reset to blank
      after remapping, because RFD3 ``diffused_index_map`` values do not carry
      insertion-code information.
    * This function is called automatically from :meth:`RFdiffusion3.run`
      when *update_motifs* is provided.  It can also be called manually
      on previously computed poses.
    * The ``diffused_index_map`` is a ``dict`` keyed by input residue
      identifiers (e.g. ``"A5"``) with values being the corresponding
      identifiers in the diffused structure.
 
    Examples
    --------
    ::
 
        from protflow.residues import AtomSelection, ResidueSelection
        from protflow.tools.rfdiffusion3 import remap_rfd3_motifs
 
        poses.df["rfd3_diffused_index_map"] = [{"A1": "B10", "A2": "B11"}]
        poses.df["active_site"] = [ResidueSelection(["A1", "A2"])]
        poses.df["active_site_atoms"] = [AtomSelection([("A", 1, "N"), ("A", 2, "CA")])]

        remap_rfd3_motifs(
            poses=poses,
            motifs=["active_site", "active_site_atoms"],
            prefix="rfd3",
            strict=True,
        )

        poses.df["active_site"].iloc[0].to_list()
        # ["B10", "B11"]
        poses.df["active_site_atoms"].iloc[0].to_tuple()
        # (("B", 10, "N"), ("B", 11, "CA"))
 
    With non-strict remapping (partial motif preservation allowed)::
 
        remap_rfd3_motifs(
            poses=poses,
            motifs=["flexible_loop"],
            prefix="rfd3",
            strict=False,
        )
    """

    diff_index_map_name = f"{prefix}_diffused_index_map"
    col_in_df(poses.df, diff_index_map_name)
    diffused_index_maps = poses.df[diff_index_map_name].to_list()

    logging.info(f"[remap_motifs] Motifs to remap: {motifs}")

    for motif_col in motifs:

        logging.info(f"[remap_motifs] Processing motif column '{motif_col}'")
        col_in_df(poses.df, motif_col)
        ref_motifs = poses.df[motif_col].to_list()
        if not ref_motifs:
            raise ValueError(f"Motif column {motif_col} is empty and cannot be remapped.")

        # Each DataFrame column must contain one concrete selection type so
        # the output column remains predictable for downstream runners.
        if all(isinstance(motif, ResidueSelection) for motif in ref_motifs):
            remap_selection = _remap_residue_selection
        elif all(isinstance(motif, AtomSelection) for motif in ref_motifs):
            remap_selection = _remap_atom_selection
        else:
            raise ValueError(f"All motifs in column {motif_col} must be either ResidueSelections or AtomSelections!")

        updated_motifs = []
        for diff_idx_map, ref_motif in zip(diffused_index_maps, ref_motifs):
            # Validate the per-pose map once, then dispatch to the selection
            # type-specific remapper selected above.
            diff_idx_map = _ensure_rfd3_index_map(diff_idx_map, motif_col=motif_col, strict=strict)
            updated_motifs.append(remap_selection(ref_motif, diff_idx_map=diff_idx_map, motif_col=motif_col, strict=strict))
        poses.df[motif_col] = updated_motifs
        
    logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")


def collect_scores(work_dir: str, cif_to_pdb: bool = True, run_clean: bool=True) -> pd.DataFrame:
    """
    Collect, decompress, convert, and flatten RFDiffusion3 output files.
 
    Scans ``<work_dir>/outputs/`` for the ``.json`` sidecar files produced
    by a completed RFDiffusion3 run, reads their contents, flattens the
    nested ``metrics`` block, decompresses the paired ``.cif.gz`` structure
    files, optionally converts them to PDB, and returns a consolidated
    :class:`~pandas.DataFrame`.
 
    The batch-index prefix added to output filenames by RFDiffusion3 is
    stripped during collection so that descriptions match the original
    pose names.
 
    Parameters
    ----------
    work_dir : str
        Root working directory of the run.  Must contain an ``outputs/``
        sub-directory populated by RFDiffusion3.
    cif_to_pdb : bool, optional
        When ``True`` (default), decompressed ``.cif`` files are converted
        to ``.pdb`` format using OpenBabel via
        :func:`~protflow.utils.openbabel_tools.openbabel_fileconverter`.
        The ``location`` column in the returned DataFrame will point to
        the ``.pdb`` files.  When ``False``, ``location`` points to
        ``.cif`` files.
    run_clean : bool, optional
        When ``True`` (default), intermediate files are removed after
        successful collection:
 
        * Compressed ``.cif.gz`` files are always deleted.
        * Intermediate ``.cif`` files are additionally deleted when
          *cif_to_pdb* is ``True``.
 
        When ``False``, all intermediate files are retained.
 
    Returns
    -------
    pandas.DataFrame
        One row per output structure with the following guaranteed columns:
 
        ``location`` : str
            Absolute path to the primary output file (PDB or CIF,
            depending on *cif_to_pdb*).
        ``description`` : str
            Filename stem of the output file, used as the pose identifier
            in ProtFlow.
        ``diffused_index_map`` : dict
            Mapping from input residue identifiers to their new positions
            in the diffused output, as reported by RFDiffusion3.
        All fields from the top-level sidecar JSON (excluding
        ``"specification"`` and ``"metrics"``), plus all fields from the
        ``"metrics"`` sub-dict, flattened to top-level columns.
 
    Notes
    -----
    * Filename normalisation (stripping the batch prefix) is performed by
      iterating over ``<output_dir>`` and applying a regex substitution
      ``r"batch.*?_"`` → ``""`` **in place** — the files are renamed on
      disk via :meth:`pathlib.Path.rename`.
    * The function iterates over ``.json`` files rather than ``.cif.gz``
      files to avoid collecting trajectory dumps or other auxiliary CIF
      files that RFDiffusion3 may write when ``dump_trajectories`` is
      enabled.
    * The ``"specification"`` and ``"metrics"`` keys are popped from the
      per-pose dict before building the DataFrame; ``"metrics"`` values
      are merged into the top level first.
 
    Raises
    ------
    This function does not raise explicitly, but will propagate exceptions
    from :func:`gzip.open`, :func:`~protflow.utils.openbabel_tools.openbabel_fileconverter`,
    or file I/O if outputs are malformed or missing.
 
    Examples
    --------
    Collect with PDB conversion (default)::
 
        scores = collect_scores("/scratch/rfd3_run")
        print(scores[["description", "location", "plddt"]].head())
 
    Collect CIF files without conversion or cleanup::
 
        scores = collect_scores(
            "/scratch/rfd3_run",
            cif_to_pdb=False,
            run_clean=False,
        )
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
