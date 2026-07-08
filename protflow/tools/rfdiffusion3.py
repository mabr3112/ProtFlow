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
import shlex

import pandas as pd

from protflow import load_config_path, require_config
from ..jobstarters import JobStarter, split_list
from ..poses import Poses, col_in_df, description_from_path
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..utils.openbabel_tools import openbabel_fileconverter
from ..utils.biopython_tools import load_structure_from_ciffile, load_structure_from_pdbfile
from ..residues import AtomSelection, ResidueSelection, atom_id_to_residue, parse_residue

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
            self.spec_from_dict(spec_from_dict)

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
            k: convert_selection_to_contig(v) for k, v in params.items()
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
                    self.data[pose].update({key: convert_selection_to_contig(spec)})

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
        self.selections_to_contigs()
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
        self.selections_to_contigs()
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
        self.selections_to_contigs()
        return self

    def selections_to_contigs(self):
        """Convert all Residue- or AtomSelections to RFD3 contigs/dicts"""
        for spec in self.data.values():
            for key, val in spec.items():
                spec[key] = convert_selection_to_contig(val)

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
        if self.poses and not all(pose in specs for pose in self.poses.df["poses_description"]) or not len(self.poses) == len(specs):
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
        self.python_path = os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")

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
        renumber_input: bool = False,
        parse_atomic_motifs: bool = False,
        strict_remap: bool = True, # if true, fail when a required remapping map is missing
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
        renumber_input : bool, optional
            When ``True``, write one renumbered copy of the RFDiffusion3
            input PDB per output structure under
            ``<work_dir>/renumbered_inputs``.  Residues present in the
            sidecar ``diffused_index_map`` are renumbered to their diffused
            chain/residue identifiers. If the input specification contains a
            ligand, ProtFlow also compares the input and output structures to
            infer ligand chain/residue moves omitted by RFD3 and applies those
            moves to the renumbered copy. The scorefile gains
            ``renumbered_inputs``, ``ligand_renumbering_map``, and
            ``ligand_renumbering_changed`` columns. Default is ``False``.
        parse_atomic_motifs : bool, optional
            When ``True``, parse each RFDiffusion3 input specification with
            :meth:`protflow.residues.AtomSelection.from_rfd3_input_spec` and
            add the resulting :class:`~protflow.residues.AtomSelection`
            objects to ``poses.df``. The main columns are remapped through
            each output's ``diffused_index_map`` plus any available or
            inferred ligand renumbering map; matching ``*_original``
            columns preserve the input-spec numbering before remapping.
            Columns are named ``<prefix>_<selection_key>`` and
            ``<prefix>_<selection_key>_original`` and are created only for
            keys present or derivable in at least one input spec. Expected
            base column names are ``<prefix>_contig``, ``<prefix>_unindex``,
            ``<prefix>_select_fixed_atoms``,
            ``<prefix>_select_unfixed_sequence``,
            ``<prefix>_fixed_motif_atoms``,
            ``<prefix>_fixed_motif_atoms_with_ligand``,
            ``<prefix>_select_buried``,
            ``<prefix>_select_partially_buried``,
            ``<prefix>_select_exposed``,
            ``<prefix>_select_hbond_donor``,
            ``<prefix>_select_hbond_acceptor``,
            ``<prefix>_select_hotspots``, ``<prefix>_ligand``,
            ``<prefix>_ligands``, and ``<prefix>_ligands_fixed_atoms``.
            Each base column can have a corresponding ``*_original`` column.
            Missing per-pose values in created columns are represented as
            empty ``AtomSelection`` objects. Default is ``False``.
        strict_remap : bool, optional
            When ``True`` (default), :func:`remap_rfd3_motifs` raises a
            :exc:`ValueError` if the remapping map itself is missing for any
            pose. Residues absent from the effective RFD3 remapping map are
            preserved unchanged with a warning. Set to ``False`` to also
            tolerate missing maps.
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
            ``<prefix>_renumbered_inputs``
                Present when *renumber_input* is ``True``.  Absolute path to
                the input PDB copy renumbered according to the output's
                ``diffused_index_map`` plus any inferred ligand move.
            ``<prefix>_ligand_renumbering_map``
                Present when *renumber_input* is ``True``.  Mapping from input
                ligand residue identifiers to their inferred output
                identifiers, containing only ligands that moved chain or
                residue number.
            ``<prefix>_ligand_renumbering_changed``
                Present when *renumber_input* is ``True``.  Boolean indicating
                whether any ligand move was inferred for that output.
            ``<prefix>_<selection_key>``
                Present when *parse_atomic_motifs* is ``True`` for each
                parsed input-spec atom selection key described above, remapped
                to output numbering via ``diffused_index_map`` plus any ligand remapping.
            ``<prefix>_<selection_key>_original``
                Present when *parse_atomic_motifs* is ``True``; contains the
                corresponding unremapped input-spec atom selection.
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
            '''helper to resolve model path, lenient towards {model_dir}/model{.ckpt}'''
            if os.path.isfile(model):
                return model
            elif os.path.isfile(model := os.path.join(self.model_dir, model)):
                return model
            elif os.path.isfile(model := os.path.join(self.model_dir, f"{model}.ckpt")):
                return model
            else:
                raise ValueError(f"Could not detect model at {model} or at {self.model_dir}.")

        # check if input_specification fits to input poses
        if poses and (not all(name in params for name in poses.df["poses_description"]) or not len(poses) == len(params)):
            raise ValueError("Input <poses> do not match <input_specification>")

        ckpt_path = identify_checkpoint(ckpt_path)

        # Warn if multiplex_poses=1 since it has no effect.
        if multiplex_poses == 1:
            logging.warning("multiplex_poses=1 has no effect. Set to None or an integer > 1.")

        if not multiplex_poses:
            multiplex_poses = 1
    
        if update_motifs:
            if not isinstance(update_motifs, list):
                raise ValueError(f"Parameter update_motifs must contain list of pose_cols! update_motifs: {update_motifs}")
            if not all(isinstance(item, str) for item in update_motifs):
                raise ValueError(f"Parameter update_motifs must contain list of pose_cols! update_motifs: {update_motifs}")

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

        # convert params from selections to strings again in case params were manually manipulated
        params.selections_to_contigs()

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
            if renumber_input:
                scores = add_renumbered_inputs_to_scores(
                    scores=scores,
                    work_dir=work_dir,
                    jobstarter=jobstarter,
                    python_path=self.python_path,
                    script_path=self._rfd3_renumber_worker_script(),
                )
                self.save_runner_scorefile(scores=scores, scorefile=scorefile)

            poses = self._merge_scores_into_poses(
                poses=poses,
                scores=scores,
                params=params,
                prefix=prefix,
                work_dir=work_dir,
                index_layers=index_layers,
                parse_atomic_motifs=parse_atomic_motifs,
                strict_remap=strict_remap,
                update_motifs=update_motifs,
            )
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
        scores = collect_scores(
            work_dir=work_dir,
            cif_to_pdb=convert_cif_to_pdb,
            run_clean=run_clean,
            renumber_input=renumber_input,
            jobstarter=jobstarter,
            python_path=self.python_path,
            script_path=self._rfd3_renumber_worker_script(),
        )

        n_out_poses = len(scores.index)
        if n_out_poses == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        if fail_on_missing_output_poses and expected_outputs < n_out_poses:
            raise RuntimeError(f"Number of output poses ({n_out_poses}) is smaller than expected number of output poses {expected_outputs}. Some runs might have crashed!")

        # save scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # merge back into poses
        poses = self._merge_scores_into_poses(
            poses=poses,
            scores=scores,
            params=params,
            prefix=prefix,
            work_dir=work_dir,
            index_layers=index_layers,
            parse_atomic_motifs=parse_atomic_motifs,
            strict_remap=strict_remap,
            update_motifs=update_motifs,
        )
        poses.reindex_poses(f"{prefix}_rfd3_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)

        logging.info(f"{self} finished. Returning {len(poses.df.index)} poses.")
        return poses

    def _rfd3_renumber_worker_script(self) -> str:
        """Return the configured RFD3 renumbering worker script path."""
        return os.path.join(self.script_dir, "rfd3_renumber_inputs.py")

    def _merge_scores_into_poses(
        self,
        poses: Poses,
        scores: pd.DataFrame,
        params: RFD3Params,
        prefix: str,
        work_dir: str,
        index_layers: int,
        parse_atomic_motifs: bool,
        strict_remap: bool,
        update_motifs: list[str] | None,
    ) -> Poses:
        """Merge collected RFD3 scores back into the active Poses object."""
        if not poses:
            poses.df = scores.copy()
            poses.df["input_poses"] = None
            poses.df["poses"] = poses.df["location"].apply(os.path.abspath)
            poses.df["poses_description"] = poses.df["description"]
            logging.info(f"Populated poses.df directly from scores {len(poses.df.index)} rows).")
            return poses

        if parse_atomic_motifs:
            scores, added_columns = _add_rfd3_atomic_motif_columns_to_scores(
                scores=scores,
                params=params,
                prefix=prefix,
                index_layers=index_layers,
                work_dir=work_dir,
                existing_pose_columns=set(poses.df.columns),
                strict=strict_remap,
            )
            if added_columns:
                logging.info(f"Parsed RFD3 atomic motif columns: {added_columns}")

        poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=index_layers).return_poses()
        if update_motifs:
            logging.info(f"Remapping residue motifs {update_motifs} after RFD3 run.")
            remap_rfd3_motifs(poses=poses, motifs=update_motifs, prefix=prefix, strict=strict_remap)
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
        Delete generated sub-directories before a rerun.

        Called internally by :meth:`run` when *overwrite* is ``True``.
        Removes ``<work_dir>/outputs/``, ``<work_dir>/inputs/``, and
        ``<work_dir>/renumbered_inputs/`` recursively using
        :func:`shutil.rmtree` if they exist, ensuring that stale files from a
        previous run do not interfere with output collection.

        Parameters
        ----------
        work_dir : str
            Root working directory of the run (the ``<poses.work_dir>/<prefix>/``
            path created by :meth:`~protflow.runners.Runner.generic_run_setup`).

        Notes
        -----
        Only the generated output/input sub-directories are removed.
        The scorefile (``rfdiffusion3_scores.<format>``) is handled
        separately by the base-class :meth:`~protflow.runners.Runner.check_for_existing_scorefile`
        logic.
        """
        for subdir in ["outputs", "inputs", "renumbered_inputs"]:
            path = os.path.join(work_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path)


def _rfd3_source_description_from_output(description: str, index_layers: int, index_sep: str = "_") -> str:
    """
    Recover the input-spec key represented by one RFDiffusion3 output name.
    """
    if not index_layers:
        return description
    return index_sep.join(description.split(index_sep)[:-1 * index_layers])


def _add_rfd3_atomic_motif_columns_to_scores(
    scores: pd.DataFrame,
    params: RFD3Params | dict,
    prefix: str,
    index_layers: int,
    work_dir: str | None = None,
    existing_pose_columns: set[str] | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add remapped and original AtomSelection columns to RFDiffusion3 scores.

    Raw column names are added before :class:`RunnerOutput` prefixes the score
    columns, so ``contig`` and ``contig_original`` become
    ``<prefix>_contig`` and ``<prefix>_contig_original`` in ``poses.df``.
    """
    if scores.empty:
        return scores, []

    parsed_by_pose = {}
    field_names = []
    for pose_name, input_spec in params.items():
        selections = AtomSelection.from_rfd3_input_spec(input_spec)
        parsed_by_pose[pose_name] = selections
        for field_name in selections:
            if field_name not in field_names:
                field_names.append(field_name)

    if not field_names:
        logging.info("parse_atomic_motifs=True found no atom selections in the RFDiffusion3 input specs.")
        return scores, []

    raw_column_names = [
        column_name
        for field_name in field_names
        for column_name in (field_name, f"{field_name}_original")
    ]
    existing_columns = [column_name for column_name in raw_column_names if column_name in scores.columns]
    if existing_columns:
        raise ValueError(f"Cannot parse RFD3 atomic motifs because columns already exist in RFD3 scores: {existing_columns}")

    added_columns = [f"{prefix}_{column_name}" for column_name in raw_column_names]
    if existing_pose_columns:
        existing_columns = [column_name for column_name in added_columns if column_name in existing_pose_columns]
        if existing_columns:
            raise ValueError(f"Cannot parse RFD3 atomic motifs because columns already exist in poses.df: {existing_columns}")

    col_in_df(scores, "description")
    col_in_df(scores, "diffused_index_map")

    scores = scores.copy()
    source_descriptions = [
        _rfd3_source_description_from_output(description, index_layers=index_layers)
        for description in scores["description"].to_list()
    ]
    diffused_index_maps = scores["diffused_index_map"].to_list()
    ligand_renumbering_maps = _ligand_renumbering_maps_for_scores(
        scores=scores,
        params=params,
        work_dir=work_dir,
    )

    for field_name in field_names:
        original_selections = []
        remapped_selections = []
        for source_description, diffused_index_map, ligand_renumbering_map in zip(source_descriptions, diffused_index_maps, ligand_renumbering_maps):
            original_selection = parsed_by_pose.get(source_description, {}).get(field_name, AtomSelection(()))
            effective_index_map = _effective_rfd3_index_map(
                diffused_index_map=diffused_index_map,
                ligand_renumbering_map=ligand_renumbering_map,
                motif_col=field_name,
                strict=strict,
            )
            original_selections.append(original_selection)
            remapped_selections.append(
                _remap_atom_selection(
                    original_selection,
                    diff_idx_map=effective_index_map,
                    motif_col=field_name,
                    strict=strict,
                )
            )

        scores[field_name] = remapped_selections
        scores[f"{field_name}_original"] = original_selections

    return scores, added_columns


def _rfd3_params_include_ligands(params: RFD3Params | dict) -> bool:
    """Return True when any RFD3 input spec names a ligand."""
    return any(
        isinstance(input_spec, dict) and input_spec.get("ligand") is not None
        for input_spec in params.values()
    )


def _is_missing_rfd3_value(value) -> bool:
    """Return True for common missing-value sentinels without treating containers as missing."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_ligand_renumbering_map(value, description: str | None = None) -> dict[str, str]:
    """Validate one optional ligand renumbering map from an RFD3 score row."""
    if isinstance(value, dict):
        return value
    if _is_missing_rfd3_value(value):
        return {}
    row_label = f" for {description}" if description else ""
    raise ValueError(f"Expected ligand_renumbering_map{row_label} to be a dictionary. Got {type(value)}: {value}")


def _infer_ligand_renumbering_maps_from_scores(scores: pd.DataFrame, work_dir: str) -> list[dict[str, str]]:
    """Infer ligand renumbering maps for collected score rows without writing files."""
    records = _renumbered_input_records_from_scores(scores=scores, work_dir=work_dir, overwrite=False)
    maps_by_description = {}
    for record in records:
        sidecar_json = record["sidecar_json"]
        sidecar_data = read_json(sidecar_json)
        specification = sidecar_data.get("specification") or {}
        ligand_renumbering_map = {}
        if specification.get("ligand") and record.get("output_structure"):
            ligand_renumbering_map = _infer_ligand_renumbering_map(
                input_pdb=_input_pdb_from_rfd3_sidecar(sidecar_data, sidecar_json=sidecar_json),
                output_structure=record["output_structure"],
                ligand=specification.get("ligand"),
            )
        maps_by_description[record["description"]] = ligand_renumbering_map

    return [maps_by_description.get(description, {}) for description in scores["description"].to_list()]


def _ligand_renumbering_maps_for_scores(
    scores: pd.DataFrame,
    params: RFD3Params | dict,
    work_dir: str | None = None,
) -> list[dict[str, str]]:
    """Return per-score ligand maps, preferring collected score columns when present."""
    if "ligand_renumbering_map" in scores.columns:
        return [
            _normalize_ligand_renumbering_map(value, description=description)
            for description, value in zip(scores["description"].to_list(), scores["ligand_renumbering_map"].to_list())
        ]

    empty_maps = [{} for _ in scores.index]
    if not _rfd3_params_include_ligands(params):
        return empty_maps
    if work_dir is None:
        logging.warning(
            "Could not infer RFDiffusion3 ligand renumbering maps for parsed atomic motifs without a work_dir; "
            "ligand atoms will keep input numbering."
        )
        return empty_maps

    try:
        return _infer_ligand_renumbering_maps_from_scores(scores=scores, work_dir=work_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.warning(
            "Could not infer RFDiffusion3 ligand renumbering maps for parsed atomic motifs; "
            "ligand atoms will keep input numbering. Error: %s",
            exc,
        )
        return empty_maps


def _effective_rfd3_index_map(
    diffused_index_map: dict | None,
    ligand_renumbering_map: dict | None,
    motif_col: str,
    strict: bool,
) -> dict:
    """Merge RFD3 residue remapping with inferred ligand-only remapping."""
    diffused_index_map = _ensure_rfd3_index_map(diffused_index_map, motif_col=motif_col, strict=strict)
    ligand_renumbering_map = _normalize_ligand_renumbering_map(ligand_renumbering_map)
    return _merged_rfd3_renumbering_map(
        diffused_index_map=diffused_index_map,
        ligand_renumbering_map=ligand_renumbering_map,
    )

def convert_selection_to_contig(selection):
    """Converts input selections to rfd3 contig strings. Returns original if not a Residue- or AtomSelection."""
    if isinstance(selection, ResidueSelection):
        return selection.to_rfdiffusion_contig()
    if isinstance(selection, AtomSelection):
        return selection.to_rfd3_dict()
    return selection

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
        If ``True``, missing or invalid maps are treated as errors. If
        ``False``, missing maps are converted to an empty dictionary and
        downstream remapping preserves the original selection identifiers.

    Returns
    -------
    dict
        Validated index map. The returned dictionary may be empty when RFD3
        omitted residue remapping information, for example for ligand-only
        motifs.

    Raises
    ------
    ValueError
        If ``strict`` is ``True`` and *diff_idx_map* is missing or not a
        dictionary.
    """
    if isinstance(diff_idx_map, dict):
        return diff_idx_map
    if strict:
        raise ValueError(f"Pose has no usable diffused_index_map while remapping selection {motif_col}. Got: {diff_idx_map}")
    return {}


def _warn_preserved_unmapped_residues(missing_residues: list[str], motif_col: str, selection_type: str) -> None:
    """
    Emit one warning for residues that were preserved without remapping.

    Parameters
    ----------
    missing_residues : list of str
        Residue identifiers that were requested during remapping but were not
        present in the pose-specific ``diffused_index_map``.
    motif_col : str
        Name of the motif column being remapped.
    selection_type : str
        Human-readable selection type used to make the warning message more
        specific, for example ``"ResidueSelection"`` or ``"AtomSelection"``.
    """
    if not missing_residues:
        return

    unique_missing = list(dict.fromkeys(missing_residues))
    logging.warning(
        "[remap_motifs] %s column '%s' contains residues absent from the effective RFD3 remapping map; "
        "keeping their original identifiers unchanged: %s. This is expected for ligands "
        "or other residues that keep their original numbering.",
        selection_type,
        motif_col,
        ", ".join(unique_missing),
    )


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
        Included for API symmetry with atom remapping. Missing residues are
        preserved unchanged regardless of this flag because RFD3 omits ligands
        and other non-remapped residues from ``diffused_index_map``.

    Returns
    -------
    ResidueSelection
        New residue selection in output-structure numbering. Residues absent
        from *diff_idx_map* are preserved unchanged.
    """
    residues = selection.to_list()
    missing_residues = [residue for residue in residues if residue not in diff_idx_map]
    _warn_preserved_unmapped_residues(missing_residues, motif_col=motif_col, selection_type="ResidueSelection")
    return ResidueSelection([diff_idx_map.get(residue, residue) for residue in residues])


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


def _remap_atom_id(atom_id, diff_idx_map: dict, motif_col: str, strict: bool) -> tuple[tuple, bool, str]:
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
        Included for API symmetry with residue remapping. Missing residues are
        preserved unchanged regardless of this flag because RFD3 omits ligands
        and other non-remapped residues from ``diffused_index_map``.

    Returns
    -------
    tuple
        Three values packed into one tuple: the remapped atom ID (or original
        atom ID when no map entry exists), a boolean indicating whether
        remapping occurred, and the atom's source residue identifier.

    Raises
    ------
    ValueError
        If *atom_id* has an unsupported shape.
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
        return atom_id, False, source_residue

    remapped_chain, remapped_residue_number = parse_residue(diff_idx_map[source_residue])
    remapped_residue_id = _format_remapped_atom_residue_id(residue_id, remapped_residue_number)
    # Preserve atom names, altlocs, model IDs, and structure IDs exactly; only
    # the chain and residue index come from RFD3.
    return (*prefix_values, remapped_chain, remapped_residue_id, *suffix_values), True, source_residue


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
        Included for API symmetry with residue remapping. Atoms whose residues
        are absent from *diff_idx_map* are preserved unchanged regardless of
        this flag.

    Returns
    -------
    AtomSelection
        New atom selection in output-structure numbering. Atom order and atom
        names are preserved. Atoms whose residues are absent from
        *diff_idx_map* keep their original IDs.
    """
    remapped_atoms = []
    missing_residues = []
    for atom_id in selection.to_tuple():
        remapped_atom_id, was_remapped, source_residue = _remap_atom_id(
            atom_id,
            diff_idx_map=diff_idx_map,
            motif_col=motif_col,
            strict=strict,
        )
        if not was_remapped:
            missing_residues.append(source_residue)
        remapped_atoms.append(remapped_atom_id)
    _warn_preserved_unmapped_residues(missing_residues, motif_col=motif_col, selection_type="AtomSelection")
    return AtomSelection(remapped_atoms)


def remap_rfd3_motifs(poses: Poses, motifs: list[str], prefix: str, strict: bool = True) -> None:
    """Remap ResidueSelection or AtomSelection motifs to diffused-output numbering.

    Translates residue positions stored in designated
    :class:`~protflow.residues.ResidueSelection` or
    :class:`~protflow.residues.AtomSelection` columns from their original
    input-structure numbering to the corresponding positions in the diffused
    output structures, using the per-pose ``diffused_index_map`` entries
    produced by RFDiffusion3 plus any optional ligand remapping stored in
    ``<prefix>_ligand_renumbering_map``.

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

        * Raises a :exc:`ValueError` if the pose has no usable
          ``diffused_index_map`` object at all, for example ``None`` or a
          non-dictionary value.
        * Preserves residues absent from the effective remapping map
          unchanged and emits a warning. This covers ligands when no inferred
          ligand map is available and residues that keep their original numbering.

        When ``False``, missing or invalid maps are treated like empty maps.
        Residues absent from the effective map are still preserved unchanged
        and warned about.

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
        If *strict* is ``True`` and a pose has no usable
        ``diffused_index_map`` object.

    Notes
    -----
    * ResidueSelection values are remapped residue-wise using their
      :meth:`~protflow.residues.ResidueSelection.to_list` representation.
    * AtomSelection values keep atom order, atom names, model IDs, structure
      IDs, altlocs, and BioPython hetero flags unchanged. Only the chain and
      numeric residue index are replaced according to ``diffused_index_map``
      and, when present, ``<prefix>_ligand_renumbering_map``.
    * Residues absent from both maps are preserved unchanged and trigger a
      warning. This covers ligands when no inferred ligand map is available.
    * BioPython insertion codes in AtomSelection residue IDs are reset to blank
      after remapping, because RFD3 ``diffused_index_map`` values do not carry
      insertion-code information.
    * This function is called automatically from :meth:`RFdiffusion3.run`
      when *update_motifs* is provided.  It can also be called manually
      on previously computed poses.
    * The ``diffused_index_map`` and optional ``<prefix>_ligand_renumbering_map``
      are ``dict`` objects keyed by input residue identifiers (e.g. ``"A5"``)
      with values being the corresponding identifiers in the diffused structure.

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
    ligand_map_name = f"{prefix}_ligand_renumbering_map"
    col_in_df(poses.df, diff_index_map_name)
    diffused_index_maps = poses.df[diff_index_map_name].to_list()
    ligand_renumbering_maps = (
        poses.df[ligand_map_name].to_list()
        if ligand_map_name in poses.df.columns
        else [{} for _ in poses.df.index]
    )

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
        for diff_idx_map, ligand_renumbering_map, ref_motif in zip(diffused_index_maps, ligand_renumbering_maps, ref_motifs):
            # Validate the per-pose map once, merge any inferred ligand moves,
            # then dispatch to the selection type-specific remapper selected above.
            effective_index_map = _effective_rfd3_index_map(
                diffused_index_map=diff_idx_map,
                ligand_renumbering_map=ligand_renumbering_map,
                motif_col=motif_col,
                strict=strict,
            )
            updated_motifs.append(remap_selection(ref_motif, diff_idx_map=effective_index_map, motif_col=motif_col, strict=strict))
        poses.df[motif_col] = updated_motifs

    logging.info(f"[remap_motifs] All motifs remapped successfully for prefix='{prefix}'.")


def _normalize_rfd3_output_names(output_dir: str) -> None:
    """
    Strip RFDiffusion3 batch prefixes from output file names in place.
    """
    directory = Path(output_dir)
    pattern = r"batch.*?_"
    for file_path in directory.iterdir():
        if file_path.is_file():
            new_name = re.sub(pattern, "", file_path.name)
            new_path = file_path.with_name(new_name)
            if new_path != file_path:
                file_path.rename(new_path)


def _rfd3_output_jsons(work_dir: str) -> list[str]:
    """
    Return normalized RFDiffusion3 output JSON sidecars for *work_dir*.
    """
    output_dir = os.path.join(work_dir, "outputs")
    if not os.path.isdir(output_dir):
        return []
    _normalize_rfd3_output_names(output_dir)
    return sorted(glob(os.path.join(output_dir, "*.json")))


def _resolve_rfd3_spec_input_path(input_pdb: str, sidecar_json: str) -> str:
    """
    Resolve an input PDB path recorded in an RFDiffusion3 output sidecar.
    """
    if os.path.isabs(input_pdb):
        return input_pdb

    candidate_bases = [
        os.getcwd(),
        os.path.dirname(sidecar_json),
        os.path.dirname(os.path.dirname(sidecar_json)),
    ]
    candidates = [os.path.abspath(os.path.join(base, input_pdb)) for base in candidate_bases]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _input_pdb_from_rfd3_sidecar(sidecar_data: dict, sidecar_json: str) -> str:
    """
    Extract the input PDB path from an RFDiffusion3 sidecar ``specification``.
    """
    specification = sidecar_data.get("specification")
    if not isinstance(specification, dict):
        raise ValueError(f"RFDiffusion3 sidecar {sidecar_json} does not contain a specification dictionary.")

    input_pdb = specification.get("input")
    if not input_pdb:
        raise ValueError(f"RFDiffusion3 sidecar {sidecar_json} does not contain specification.input.")
    return _resolve_rfd3_spec_input_path(str(input_pdb), sidecar_json=sidecar_json)


def _rfd3_index_map_to_residue_mapping(diffused_index_map: dict) -> dict[tuple[str, int], tuple[str, int]]:
    """
    Convert an RFD3 ``diffused_index_map`` into tuple-key residue mapping.
    """
    if not isinstance(diffused_index_map, dict):
        raise ValueError(f"Expected diffused_index_map to be a dictionary, got {type(diffused_index_map)}.")

    residue_mapping = {}
    for source, target in diffused_index_map.items():
        source_chain, source_residue = parse_residue(str(source).strip())
        target_chain, target_residue = parse_residue(str(target).strip())
        residue_mapping[(source_chain, source_residue)] = (target_chain, target_residue)
    return residue_mapping


def _load_rfd3_renumber_model(path: str):
    """Load the first model from a PDB or mmCIF file for ligand matching."""
    lower_path = path.lower()
    if lower_path.endswith(".pdb"):
        return load_structure_from_pdbfile(path, all_models=False, model=0)
    if lower_path.endswith((".cif", ".mmcif")):
        return load_structure_from_ciffile(path, all_models=False, model=0)
    raise ValueError(f"Unsupported structure extension for ligand renumbering: {path}")


def _residue_chain_id(residue) -> str:
    """Return the single-character chain ID for a BioPython residue."""
    chain_id = str(residue.get_parent().id)
    return chain_id[:1] if chain_id else " "


def _residue_key(residue) -> tuple[str, int]:
    """Return ``(chain, residue_number)`` for a BioPython residue."""
    return (_residue_chain_id(residue), int(residue.id[1]))


def _format_rfd3_residue_id(residue_key: tuple[str, int]) -> str:
    """Format a tuple residue key as an RFD3-style residue identifier."""
    chain_id, residue_number = residue_key
    return f"{chain_id}{residue_number}"


def _residue_records_from_structure(path: str) -> list[dict]:
    """Extract residue records used for ligand matching."""
    model = _load_rfd3_renumber_model(path)
    records = []
    for residue in model.get_residues():
        atom_coords = {
            atom.get_name().strip(): atom.coord
            for atom in residue.get_atoms()
        }
        if not atom_coords:
            continue
        key = _residue_key(residue)
        records.append(
            {
                "key": key,
                "identifier": _format_rfd3_residue_id(key),
                "resname": residue.get_resname().strip(),
                "atom_names": frozenset(atom_coords),
                "atom_coords": atom_coords,
            }
        )
    return records

def _input_ligand_residue_records(input_pdb: str, ligand: str | None) -> list[dict]:
    """
    Resolve the RFD3 ligand selector against the input PDB.
    """
    if not ligand:
        return []

    ligand_atoms = AtomSelection.from_rfd3_ligand(ligand, pose=input_pdb)
    ligand_keys = list(dict.fromkeys(atom_id_to_residue(atom_id) for atom_id in ligand_atoms.to_tuple()))
    if not ligand_keys:
        return []

    input_records = {
        record["key"]: record
        for record in _residue_records_from_structure(input_pdb)
    }
    missing = [key for key in ligand_keys if key not in input_records]
    if missing:
        raise ValueError(
            "Could not resolve all RFDiffusion3 ligand residues in input structure "
            f"{input_pdb}: {missing}"
        )
    return [input_records[key] for key in ligand_keys]


def _ligand_match_score(input_record: dict, output_record: dict) -> float:
    """
    Score one possible input/output ligand residue match.
    """
    atom_name_penalty = len(input_record["atom_names"] ^ output_record["atom_names"]) * 1000000.0
    common_atom_names = sorted(input_record["atom_names"] & output_record["atom_names"])
    if not common_atom_names:
        return float("inf")

    coord_score = sum(
        float(((input_record["atom_coords"][atom_name] - output_record["atom_coords"][atom_name]) ** 2).sum())
        for atom_name in common_atom_names
    ) / len(common_atom_names)
    return atom_name_penalty + coord_score


def _infer_ligand_renumbering_map(input_pdb: str, output_structure: str, ligand: str | None) -> dict[str, str]:
    """
    Infer moved ligand residue identifiers from input and output structures.
    """
    input_ligands = _input_ligand_residue_records(input_pdb=input_pdb, ligand=ligand)
    if not input_ligands:
        return {}

    output_records = _residue_records_from_structure(output_structure)
    candidate_pairs = []
    for input_index, input_record in enumerate(input_ligands):
        for output_index, output_record in enumerate(output_records):
            if output_record["resname"] != input_record["resname"]:
                continue
            score = _ligand_match_score(input_record, output_record)
            if score != float("inf"):
                candidate_pairs.append((score, input_index, output_index))

    moved_ligand_map = {}
    used_input_indices = set()
    used_output_indices = set()
    for _, input_index, output_index in sorted(candidate_pairs, key=lambda item: item[0]):
        if input_index in used_input_indices or output_index in used_output_indices:
            continue
        input_record = input_ligands[input_index]
        output_record = output_records[output_index]
        source_id = input_record["identifier"]
        target_id = output_record["identifier"]
        if source_id != target_id:
            moved_ligand_map[source_id] = target_id
        used_input_indices.add(input_index)
        used_output_indices.add(output_index)

    unmatched = [
        input_ligands[index]["identifier"]
        for index in range(len(input_ligands))
        if index not in used_input_indices
    ]
    if unmatched:
        logging.warning(
            "Could not infer RFDiffusion3 ligand renumbering for input residues %s from output %s.",
            ", ".join(unmatched),
            output_structure,
        )

    return moved_ligand_map


def _merged_rfd3_renumbering_map(diffused_index_map: dict | None, ligand_renumbering_map: dict[str, str]) -> dict:
    """
    Merge RFD3's residue map with ProtFlow-inferred ligand moves.
    """
    if diffused_index_map is None:
        renumbering_map = {}
    elif isinstance(diffused_index_map, dict):
        renumbering_map = dict(diffused_index_map)
    else:
        raise ValueError(f"Expected diffused_index_map to be a dictionary, got {type(diffused_index_map)}.")

    for source_id, target_id in ligand_renumbering_map.items():
        existing_target = renumbering_map.get(source_id)
        if existing_target is not None and existing_target != target_id:
            logging.warning(
                "RFDiffusion3 diffused_index_map already contains %s -> %s; "
                "keeping the RFD3 value instead of inferred ligand target %s.",
                source_id,
                existing_target,
                target_id,
            )
            continue
        renumbering_map[source_id] = target_id
    return renumbering_map

def _pdb_line_residue_key(line: str) -> tuple[str, int] | None:
    """
    Return ``(chain, resseq)`` for PDB records with residue fields.
    """
    if line[:6] not in {"ATOM  ", "HETATM", "ANISOU", "TER   "}:
        return None
    if len(line) < 26:
        return None

    residue_number = line[22:26].strip()
    if not residue_number:
        return None
    try:
        return line[21], int(residue_number)
    except ValueError:
        return None


def _split_line_ending(line: str) -> tuple[str, str]:
    """
    Split a text line into content and original line ending.
    """
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _format_pdb_residue_line(line: str, chain_id: str, residue_number: int) -> str:
    """
    Replace chain, residue number, and insertion code fields in one PDB line.
    """
    if len(chain_id) != 1:
        raise ValueError(f"PDB chain IDs must be a single character, got {chain_id!r}.")
    if residue_number < -999 or residue_number > 9999:
        raise ValueError(f"PDB residue number {residue_number} cannot be represented in a 4-character resSeq field.")

    content, line_ending = _split_line_ending(line)
    if len(content) < 27:
        content = content.ljust(27)
    return f"{content[:21]}{chain_id}{residue_number:>4} {content[27:]}{line_ending}"


def renumber_rfd3_input_pdb(
    input_pdb: str,
    diffused_index_map: dict,
    output_pdb: str,
    overwrite: bool = False,
) -> str:
    """
    Copy an RFDiffusion3 input PDB and renumber residues in ``diffused_index_map``.

    The function preserves every line from *input_pdb*.  For PDB records with
    residue fields (``ATOM``, ``HETATM``, ``ANISOU``, and ``TER``), only residues
    present in *diffused_index_map* have their chain/residue fields replaced by
    the mapped RFDiffusion3 output identifiers.  Unmapped residues, ligands, and
    non-coordinate records are copied unchanged.
    """
    input_pdb = os.path.abspath(input_pdb)
    output_pdb = os.path.abspath(output_pdb)

    if not os.path.isfile(input_pdb):
        raise FileNotFoundError(f"Cannot renumber missing RFDiffusion3 input PDB: {input_pdb}")
    if os.path.isfile(output_pdb) and not overwrite:
        return output_pdb

    residue_mapping = _rfd3_index_map_to_residue_mapping(diffused_index_map)
    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)

    with open(input_pdb, "r", encoding="UTF-8") as input_handle, open(output_pdb, "w", encoding="UTF-8") as output_handle:
        for line in input_handle:
            residue_key = _pdb_line_residue_key(line)
            if residue_key in residue_mapping:
                chain_id, residue_number = residue_mapping[residue_key]
                line = _format_pdb_residue_line(line, chain_id=chain_id, residue_number=residue_number)
            output_handle.write(line)

    return output_pdb


RFD3_RENUMBER_RESULT_COLUMNS = (
    "renumbered_inputs",
    "ligand_renumbering_map",
    "ligand_renumbering_changed",
)


def _rfd3_renumber_worker_script(script_path: str | None = None) -> str:
    """Return the configured auxiliary worker script for RFD3 input renumbering."""
    if script_path:
        return script_path
    script_dir = load_config_path(require_config(), "AUXILIARY_RUNNER_SCRIPTS_DIR")
    return os.path.join(script_dir, "rfd3_renumber_inputs.py")


def _rfd3_renumber_input_record(record: dict) -> dict:
    """
    Create one renumbered RFD3 input PDB and return scorefile columns.
    """
    sidecar_json = record["sidecar_json"]
    description = record.get("description") or description_from_path(sidecar_json)
    output_structure = record.get("output_structure")
    output_pdb = record["output_pdb"]
    overwrite = bool(record.get("overwrite", False))

    sidecar_data = read_json(sidecar_json)
    specification = sidecar_data.get("specification") or {}
    input_pdb = _input_pdb_from_rfd3_sidecar(sidecar_data, sidecar_json=sidecar_json)
    ligand_renumbering_map = {}

    if specification.get("ligand") and output_structure:
        ligand_renumbering_map = _infer_ligand_renumbering_map(
            input_pdb=input_pdb,
            output_structure=output_structure,
            ligand=specification.get("ligand"),
        )

    renumbering_map = _merged_rfd3_renumbering_map(
        diffused_index_map=sidecar_data.get("diffused_index_map"),
        ligand_renumbering_map=ligand_renumbering_map,
    )
    renumbered_input = renumber_rfd3_input_pdb(
        input_pdb=input_pdb,
        diffused_index_map=renumbering_map,
        output_pdb=output_pdb,
        overwrite=overwrite,
    )
    return {
        "description": description,
        "renumbered_inputs": renumbered_input,
        "ligand_renumbering_map": ligand_renumbering_map,
        "ligand_renumbering_changed": bool(ligand_renumbering_map),
    }


def _renumbered_input_records_from_scores(scores: pd.DataFrame, work_dir: str, overwrite: bool = False) -> list[dict]:
    """
    Build per-output renumbering worker input records from collected scores.
    """
    col_in_df(scores, "description")
    col_in_df(scores, "location")
    sidecar_by_description = {
        description_from_path(sidecar_json): sidecar_json
        for sidecar_json in _rfd3_output_jsons(work_dir)
    }

    missing = [
        description
        for description in scores["description"].to_list()
        if description not in sidecar_by_description
    ]
    if missing:
        raise RuntimeError(
            "Could not create renumbered RFDiffusion3 inputs for all score rows. "
            f"Missing output sidecars for descriptions: {missing}"
        )

    renumbered_input_dir = os.path.join(work_dir, "renumbered_inputs")
    return [
        {
            "description": row.description,
            "sidecar_json": sidecar_by_description[row.description],
            "output_structure": os.path.abspath(row.location),
            "output_pdb": os.path.join(renumbered_input_dir, f"{row.description}.pdb"),
            "overwrite": overwrite,
        }
        for row in scores[["description", "location"]].itertuples(index=False)
    ]


def _run_rfd3_renumber_records_with_jobstarter(
    records: list[dict],
    work_dir: str,
    jobstarter: JobStarter,
    python_path: str | None = None,
    script_path: str | None = None,
) -> list[dict]:
    """
    Dispatch RFD3 input renumbering records through a ProtFlow jobstarter.
    """
    if not records:
        return []

    worker_script = _rfd3_renumber_worker_script(script_path=script_path)
    if not os.path.isfile(worker_script):
        raise FileNotFoundError(f"Cannot find RFDiffusion3 renumbering worker script: {worker_script}")

    python_path = python_path or os.path.join(load_config_path(require_config(), "PROTFLOW_ENV"), "python")
    worker_dir = os.path.join(work_dir, "renumbered_inputs", "worker_json")
    os.makedirs(worker_dir, exist_ok=True)

    n_workers = min(len(records), max(1, jobstarter.max_cores or 1))
    record_sublists = split_list(records, n_sublists=n_workers)
    input_jsons = []
    output_jsons = []
    for index, sublist in enumerate(record_sublists, start=1):
        input_json = os.path.join(worker_dir, f"rfd3_renumber_input_{index:04}.json")
        output_json = os.path.join(worker_dir, f"rfd3_renumber_output_{index:04}.json")
        with open(input_json, "w", encoding="UTF-8") as handle:
            json.dump(sublist, handle)
        input_jsons.append(input_json)
        output_jsons.append(output_json)

    cmds = [
        f"{shlex.quote(str(python_path))} {shlex.quote(worker_script)} "
        f"--input_json {shlex.quote(input_json)} "
        f"--output_json {shlex.quote(output_json)}"
        for input_json, output_json in zip(input_jsons, output_jsons)
    ]
    jobstarter.start(
        cmds=cmds,
        jobname="rfd3_renumber_inputs",
        wait=True,
        output_path=worker_dir,
    )

    missing_outputs = [output_json for output_json in output_jsons if not os.path.isfile(output_json)]
    if missing_outputs:
        raise FileNotFoundError(f"RFDiffusion3 renumbering worker output files were not created: {missing_outputs}")

    results = []
    for output_json in output_jsons:
        with open(output_json, "r", encoding="UTF-8") as handle:
            results.extend(json.load(handle))
    return results


def add_renumbered_inputs_to_scores(
    scores: pd.DataFrame,
    work_dir: str,
    overwrite: bool = False,
    jobstarter: JobStarter | None = None,
    python_path: str | None = None,
    script_path: str | None = None,
) -> pd.DataFrame:
    """
    Ensure an RFDiffusion3 scores DataFrame contains renumbered-input columns.
    """
    scores = scores.copy()
    has_required_columns = all(column in scores.columns for column in RFD3_RENUMBER_RESULT_COLUMNS)
    if has_required_columns and not overwrite:
        paths = scores["renumbered_inputs"].dropna().to_list()
        if len(paths) == len(scores.index) and all(os.path.isfile(path) for path in paths):
            return scores

    if jobstarter is None:
        raise ValueError("RFDiffusion3 input renumbering requires a jobstarter.")

    force_file_refresh = overwrite or not has_required_columns
    records = _renumbered_input_records_from_scores(
        scores=scores,
        work_dir=work_dir,
        overwrite=force_file_refresh,
    )
    results = _run_rfd3_renumber_records_with_jobstarter(
        records=records,
        work_dir=work_dir,
        jobstarter=jobstarter,
        python_path=python_path,
        script_path=script_path,
    )

    result_by_description = {result["description"]: result for result in results}
    missing = [description for description in scores["description"].to_list() if description not in result_by_description]
    if missing:
        raise RuntimeError(
            "Could not create renumbered RFDiffusion3 inputs for all score rows. "
            f"Missing worker results for descriptions: {missing}"
        )

    for column in RFD3_RENUMBER_RESULT_COLUMNS:
        scores[column] = scores["description"].map(lambda description: result_by_description[description][column])
    return scores


def collect_scores(
    work_dir: str,
    cif_to_pdb: bool = True,
    run_clean: bool = True,
    renumber_input: bool = False,
    jobstarter: JobStarter | None = None,
    python_path: str | None = None,
    script_path: str | None = None,
) -> pd.DataFrame:
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
    renumber_input : bool, optional
        When ``True``, create one renumbered copy of each sidecar's
        ``specification.input`` PDB under ``<work_dir>/renumbered_inputs`` and
        add renumbering columns to the returned DataFrame. Ligand chain/residue
        moves are inferred from input/output structures when a ligand is
        present in the RFD3 input specification.
    jobstarter : JobStarter, optional
        Jobstarter used to parallelize renumbering and ligand-move inference.
        Required when *renumber_input* is ``True``.
    python_path : str, optional
        Python executable used for renumbering worker jobs. Defaults to
        ``<PROTFLOW_ENV>/python`` from the ProtFlow config.
    script_path : str, optional
        Explicit renumbering worker script path. Defaults to
        ``<AUXILIARY_RUNNER_SCRIPTS_DIR>/rfd3_renumber_inputs.py`` from the
        ProtFlow config.

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
        ``renumbered_inputs`` : str
            Present when *renumber_input* is ``True``.  Absolute path to the
            corresponding renumbered input PDB.
        ``ligand_renumbering_map`` : dict
            Present when *renumber_input* is ``True``.  Inferred ligand-only
            residue mapping used in addition to ``diffused_index_map``.
        ``ligand_renumbering_changed`` : bool
            Present when *renumber_input* is ``True``.  Whether the inferred
            ligand map contains at least one moved ligand residue.
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

    output_jsons = _rfd3_output_jsons(work_dir)

    data = []
    # iterate over jsons because additional cif files might be there if dump_trajectories is true
    for j in output_jsons:
        p_data = read_json(j)
        p_data.update(p_data["metrics"]) # flatten metrics
        p_data["compressed_cif_location"] = re.sub(r"\.json$", ".cif.gz", j)
        # delete specifications
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

    if renumber_input:
        data = add_renumbered_inputs_to_scores(
            scores=data,
            work_dir=work_dir,
            overwrite=True,
            jobstarter=jobstarter,
            python_path=python_path,
            script_path=script_path,
        )

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
