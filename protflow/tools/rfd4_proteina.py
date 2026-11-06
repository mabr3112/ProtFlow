"""
RFD4-Proteina Module
=========================================

.. module:: protflow.tools.rfd4_proteina
   :synopsis: ProtFlow runner interface for RFD4-Proteina, the Baker Lab x
              NVIDIA flow-matching protein design model.

This module provides two public classes and several module-level utilities for
executing RFD4-Proteina within the ProtFlow framework:

* :class:`RFD4Params` — a :class:`~collections.UserDict` subclass that manages
  per-pose *condition specifications* (the RFD4 YAML/JSON input format) and
  translates ProtFlow :class:`~protflow.residues.ResidueSelection` and
  :class:`~protflow.residues.AtomSelection` objects into AtomWorks selection
  queries and ``contig`` tokens.
* :class:`RFD4Proteina` — the main ProtFlow :class:`~protflow.runners.Runner`
  subclass that writes the condition-spec files, assembles the Hydra command
  line for ``rfd4-generate``, submits jobs, collects outputs and inline
  metrics, and caches the runner scorefile.

The condition-spec format
-------------------------
RFD4-Proteina does not take contig strings on the command line. Every design is
one top-level key in a YAML (or JSON) *condition spec*::

    my_design:
      input: /abs/path/to/complex.cif
      contig: "A1-120, /0, B1-80"
      conditions:
        C_CRD: {true: [{select: "sel('B') and is_protein_backbone"}]}
        C_HOT: {true: [{select: "sel('A/*/[45, 46, 82]')"}]}

:class:`RFD4Params` builds exactly this structure, one key per pose, with
``input`` inferred automatically from the :class:`~protflow.poses.Poses`
object. The three stages of a spec are applied by RFD4 in order:

1. **load** the ``input`` structure(s),
2. **``contig``** subsets the input to preserved segments and inserts
   expandable (variable-length) segments,
3. **``conditions``** sets conditioning annotations, resolved against the
   structure that ``contig`` produced.

``res_id`` in both ``contig`` and ``conditions`` refers to the **original**
input numbering (the mmCIF ``label_seq_id``), before ``contig`` renumbers the
output.

Selections
----------
Every ``select:`` value is one AtomWorks query string. :class:`RFD4Params`
accepts a raw query string, a :class:`~protflow.residues.ResidueSelection`, an
:class:`~protflow.residues.AtomSelection`, or a list of these. The conversion
itself lives on the selection classes, as
:meth:`~protflow.residues.ResidueSelection.to_atomworks_query` and
:meth:`~protflow.residues.AtomSelection.to_atomworks_query`;
:func:`selection_to_query` only dispatches to them and ``or``-combines lists:

* ``ResidueSelection(["A1", "A2", "A3", "B7"])`` becomes
  ``"(sel('A/*/[1-3]') or sel('B/*/[7]'))"``
* ``AtomSelection([("A", 5, "CA"), ("A", 5, "N")])`` becomes
  ``"sel('A/*/[5]/[CA,N]')"``

``contig`` values accept the same selection objects and are converted with
:func:`selection_to_contig`, which emits preserved-segment tokens
(``"A1-3, B7"``).

Supported design modes
----------------------
Motif scaffolding / binder design / enzyme design / redesign
    Provide a :class:`~protflow.poses.Poses` collection of input structures
    (``.cif``, ``.cif.gz``, ``.bcif`` — RFD4 does not read PDB) and describe
    the design problem with ``contig`` and ``conditions``.

Unconditional generation
    A condition spec always needs at least one preserved segment, so pure de
    novo generation cannot be expressed through a spec built from poses.  Pass
    an empty :class:`~protflow.poses.Poses` and point the spec at one of the
    wholly expandable benchmark inputs shipped with RFD4-Proteina
    (``data/benchmarks/monomer/*.cif.zst``) via :meth:`RFD4Params.add_specs`.

Authors
-------
Adrian Tripp

Version
-------
0.1.0

Examples
--------
Binder design against a fixed target::

    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.tools.rfd4_proteina import RFD4Proteina, RFD4Params

    poses = Poses("targets/", glob_suffix="*.cif")
    jobstarter = SbatchArrayJobstarter(max_cores=8, gpus=1)

    params = RFD4Params(poses=poses)
    params.set_input_specs(contig="50-120, /0, A17-131")
    params.set_coordinate_group(select="sel('A') and is_protein_backbone", group_id=1)
    params.set_sequence_condition(select="sel('A')")
    params.set_hotspots(select=ResidueSelection(["A45", "A46", "A82"]))

    runner = RFD4Proteina(jobstarter=jobstarter)
    poses = runner.run(
        poses=poses,
        prefix="rfd4_binders",
        params=params,
        num_replicates=40,
        batch_size=8,
    )

Motif scaffolding with per-pose motifs stored in ``poses.df``::

    params = RFD4Params(poses=poses)
    params.set_per_pose_contig(contig="motif_contig_col")
    params.set_per_pose_condition("C_CRD", value=True, select="motif_col")

    poses = runner.run(poses=poses, prefix="rfd4_motif", params=params,
                       num_replicates=16, batch_size=4,
                       metrics_tags=["design", "motif_scaffold_refolding"])
"""


# ---------------------------------------------------------------------------


from __future__ import annotations

import copy
import gzip
import json
import logging
import os
import re
import shlex
import shutil
from collections import defaultdict, UserDict
from glob import glob

import pandas as pd
import yaml

from protflow import load_config_path, require_config, get_config
from ..jobstarters import JobStarter, split_list
from ..poses import Poses, col_in_df
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..residues import (
    AtomSelection,
    ResidueSelection,
    remap_atom_selection,
    remap_residue_selection,
)

# --------------------------------------------------------------------------- #
#                         condition-spec vocabulary                            #
# --------------------------------------------------------------------------- #

#: Mapping of RFD4 condition *full names* to their ``C_*`` shorthand aliases.
#: Both forms are accepted by RFD4, but ProtFlow always writes the ``C_*`` form,
#: which is the form documented as safe in the RFD4 YAML/JSON reference.
RFD4_CONDITION_ALIASES: dict[str, str] = {
    "coordinate": "C_CRD",
    "sequence": "C_SEQ",
    "hotspot": "C_HOT",
    "rasa-binned": "C_RAS",
    "rasa_binned": "C_RAS",
    "hbond": "C_HB",
    "pistack": "C_PISTACK",
    "pication": "C_PICATION",
    "cath-label": "C_CATH",
    "cath_label": "C_CATH",
    "chaincentroid": "C_CENTROID",
    "index": "C_IDX",
    "coordinategroup": "C_CRDGRP",
}

#: Conditions whose value is a boolean mask.
RFD4_BOOLEAN_CONDITIONS: tuple[str, ...] = ("C_CRD", "C_SEQ", "C_HOT", "C_PISTACK")

#: Allowed enum values per condition.
RFD4_CONDITION_ENUMS: dict[str, tuple[str, ...]] = {
    "C_RAS": ("BURIED", "INTERMEDIATE", "EXPOSED", "UNSPECIFIED"),
    "C_HB": ("DONOR", "ACCEPTOR", "BOTH"),
    "C_PICATION": ("AROMATIC", "CATION", "BOTH"),
}

#: Conditions that require a ``group:`` label beside every ``select:``.
RFD4_GROUP_CONDITIONS: tuple[str, ...] = ("C_IDX",)

#: Canonical polymer tokens that can carry a ``C_SEQ_PROB_<TOKEN>`` condition.
RFD4_SEQ_PROB_TOKENS: tuple[str, ...] = (
    # 20 amino acids
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # 4 RNA bases
    "A", "C", "G", "U",
    # 4 DNA bases
    "DA", "DC", "DG", "DT",
)

#: Every condition ProtFlow knows how to validate (``C_SEQ_PROB_*`` excluded,
#: those are checked against :data:`RFD4_SEQ_PROB_TOKENS`).
RFD4_CONDITIONS: tuple[str, ...] = (
    "C_CRD", "C_SEQ", "C_HOT", "C_RAS", "C_HB", "C_PISTACK", "C_PICATION",
    "C_CATH", "C_CENTROID", "C_IDX", "C_CRDGRP",
)

#: Top-level fields accepted under a design key in a condition spec.
RFD4_SPEC_FIELDS: tuple[str, ...] = ("input", "input_as_annotated_cif", "contig", "conditions")

#: Hydra keys the runner sets itself; they must not appear in ``options``.
RFD4_RESERVED_OPTIONS: tuple[str, ...] = (
    "inputs", "out_dir", "num_replicates", "batch_size", "metrics_tags", "seed",
    "skip_existing", "gen_njobs", "job_id", "inference_experiment",
    "recycle_strategy", "compress_outputs", "subset_to_examples",
)


class _RFD4Dumper(yaml.SafeDumper):
    """YAML dumper that never emits anchors/aliases and indents lists."""

    def ignore_aliases(self, data) -> bool:  # noqa: D102 - yaml API
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: D102 - yaml API
        return super().increase_indent(flow=flow, indentless=False)


# --------------------------------------------------------------------------- #
#                           selection conversion                               #
# --------------------------------------------------------------------------- #

def selection_to_query(selection) -> str:
    """
    Convert a ProtFlow selection into an AtomWorks ``select:`` query string.

    Parameters
    ----------
    selection : str, ResidueSelection, AtomSelection, list, or tuple
        The selection to convert:

        ``str``
            Returned unchanged — already an AtomWorks query (path syntax such
            as ``"sel('A/*/[10-20]')"`` or query syntax such as
            ``"is_ligand and not is_protein_backbone"``).
        :class:`~protflow.residues.ResidueSelection`
            Converted by
            :meth:`~protflow.residues.ResidueSelection.to_atomworks_query`.
        :class:`~protflow.residues.AtomSelection`
            Converted by
            :meth:`~protflow.residues.AtomSelection.to_atomworks_query`.
        ``list`` or ``tuple``
            Each element converted and combined with ``or``.

    Returns
    -------
    str
        One AtomWorks query string.

    Raises
    ------
    TypeError
        If *selection* is not one of the supported types.
    ValueError
        If *selection* is an empty selection object or empty list.

    Notes
    -----
    This function only dispatches: the conversion itself lives on the
    selection classes as ``to_atomworks_query()``, next to the other
    tool-specific serializers (``to_rfdiffusion_contig``, ``to_rfd3_dict``,
    ``to_boltz_atom``).  What this adds is passthrough of raw query strings and
    ``or``-combination of mixed lists.

    AtomWorks parses a query as a Python boolean expression, so combining terms
    with ``or`` / ``and`` / ``not`` is always valid.  Multi-term conversions are
    parenthesised so they can be further combined by the caller, e.g.
    ``f"{selection_to_query(sel)} and is_protein_backbone"``.

    Examples
    --------
    ::

        >>> selection_to_query(ResidueSelection(["A1", "A2", "A3", "B7"]))
        "(sel('A/*/[1-3]') or sel('B/*/[7]'))"
        >>> selection_to_query(AtomSelection([("A", 5, "CA"), ("A", 5, "N")]))
        "sel('A/*/[5]/[CA,N]')"
    """
    if isinstance(selection, str):
        return selection

    if isinstance(selection, (ResidueSelection, AtomSelection)):
        return selection.to_atomworks_query()

    if isinstance(selection, (list, tuple)):
        if not selection:
            raise ValueError("Cannot convert an empty list of selections into a select query!")
        terms = [selection_to_query(item) for item in selection]
        return terms[0] if len(terms) == 1 else "(" + " or ".join(terms) + ")"

    raise TypeError(
        f"<select> must be a query string, ResidueSelection, AtomSelection or a list thereof, not {type(selection)}!"
    )


def selection_to_contig(selection) -> str:
    """
    Convert a ProtFlow selection into RFD4 ``contig`` preserved-segment tokens.

    Parameters
    ----------
    selection : str, ResidueSelection, AtomSelection, list, or tuple
        ``str`` values are returned unchanged (already a contig string).
        :class:`~protflow.residues.ResidueSelection` objects are converted to
        comma-separated preserved-segment tokens; an
        :class:`~protflow.residues.AtomSelection` is first collapsed to its
        residues.  Lists are converted element-wise and joined with ``", "``,
        which is how chainbreak (``/0``) and expandable-segment tokens are
        mixed with selection objects.

    Returns
    -------
    str
        A contig string such as ``"A1-3, B7"``.

    Notes
    -----
    * Only *preserved* segments can be derived from a selection.  Expandable
      segments (``"50-120"``, ``"E1:2-20"``) and chainbreaks (``"/0"``) have no
      selection equivalent and must be supplied as strings.
    * A contig must contain at least one preserved segment; RFD4 rejects a
      contig built entirely from expandable segments.

    Examples
    --------
    ::

        >>> selection_to_contig(ResidueSelection(["A1", "A2", "A3", "C6"]))
        'A1-3,C6'
        >>> selection_to_contig(["50-120", "/0", ResidueSelection(["A17", "A18"])])
        '50-120, /0, A17-18'
    """
    if isinstance(selection, str):
        return selection
    if isinstance(selection, AtomSelection):
        selection = ResidueSelection.from_atomselection(selection)
    if isinstance(selection, ResidueSelection):
        if not len(selection):
            raise ValueError("Cannot convert an empty ResidueSelection into a contig!")
        return selection.to_rfdiffusion_contig()
    if isinstance(selection, (list, tuple)):
        if not selection:
            raise ValueError("Cannot convert an empty list of selections into a contig!")
        return ", ".join(selection_to_contig(item) for item in selection)
    raise TypeError(
        f"<contig> must be a string, ResidueSelection, AtomSelection or a list thereof, not {type(selection)}!"
    )


def normalize_condition_name(annotation: str) -> str:
    """
    Normalize a condition name to its ``C_*`` shorthand and validate it.

    Parameters
    ----------
    annotation : str
        Shorthand (``"C_CRD"``) or full name (``"coordinate"``) of an RFD4
        condition, or a ``C_SEQ_PROB_<TOKEN>`` sequence-probability condition.

    Returns
    -------
    str
        The canonical ``C_*`` name.

    Raises
    ------
    ValueError
        If *annotation* is not a known RFD4 condition, or is a
        ``C_SEQ_PROB_*`` condition for a token outside the 28-token canonical
        polymer vocabulary.

    Notes
    -----
    ``C_INTCRD`` is rejected: internal-coordinate targets are set through
    ``C_CRDGRP`` (see :meth:`RFD4Params.set_coordinate_group`), never written
    directly.
    """
    if not isinstance(annotation, str):
        raise ValueError(f"Condition names must be strings, not {type(annotation)}!")

    name = RFD4_CONDITION_ALIASES.get(annotation.strip().lower(), annotation.strip())

    if name == "C_INTCRD":
        raise ValueError(
            "Direct C_INTCRD entries are rejected by RFD4. Set internal-coordinate conditioning "
            "through C_CRDGRP instead (RFD4Params.set_coordinate_group)."
        )

    if name.startswith("C_SEQ_PROB_"):
        token = name[len("C_SEQ_PROB_"):]
        if token not in RFD4_SEQ_PROB_TOKENS:
            raise ValueError(
                f"'{token}' is not a canonical polymer token. C_SEQ_PROB_<TOKEN> is defined for "
                f"{RFD4_SEQ_PROB_TOKENS}."
            )
        return name

    if name not in RFD4_CONDITIONS:
        raise ValueError(
            f"Unknown RFD4 condition '{annotation}'. Known conditions: {RFD4_CONDITIONS} "
            "(plus C_SEQ_PROB_<TOKEN>). Full names such as 'coordinate' or 'hotspot' are also accepted."
        )
    return name


def _validate_condition_value(annotation: str, value, group) -> None:
    """Validate the value (and group presence) of one condition entry."""
    if annotation in RFD4_BOOLEAN_CONDITIONS:
        if not isinstance(value, bool):
            raise ValueError(f"{annotation} takes a boolean value (True/False), not {value!r}.")

    elif annotation in RFD4_CONDITION_ENUMS:
        allowed = RFD4_CONDITION_ENUMS[annotation]
        if annotation == "C_RAS" and isinstance(value, int) and not isinstance(value, bool):
            pass  # RASA bins may be given as the integer bin index
        elif not isinstance(value, str) or value.upper() not in allowed:
            raise ValueError(f"{annotation} takes one of {allowed}, not {value!r}.")

    elif annotation == "C_IDX":
        if value is not False:
            raise ValueError(f"C_IDX only takes the value False, not {value!r}.")

    elif annotation == "C_CRDGRP":
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(
                f"C_CRDGRP takes a positive integer group ID or None, not {value!r}. "
                "IDs 0 and -1 are reserved and cannot be assigned."
            )

    elif annotation == "C_CENTROID":
        if not isinstance(value, str) or len(value.split(",")) != 3:
            raise ValueError(f"C_CENTROID takes a literal 'x,y,z' string, not {value!r}.")

    elif annotation == "C_CATH":
        if not isinstance(value, str) or len(value.split(".")) != 3:
            raise ValueError(
                f"C_CATH takes a quoted C.A.T label such as '2.60.40' (with 'x' for unspecified "
                f"trailing levels), not {value!r}."
            )

    elif annotation.startswith("C_SEQ_PROB_"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{annotation} takes a float in [0, 1], not {value!r}.")

    # group handling
    if annotation in RFD4_GROUP_CONDITIONS and group is None:
        raise ValueError(f"{annotation} requires a <group> label beside every selection.")
    if annotation not in RFD4_GROUP_CONDITIONS and group is not None:
        raise ValueError(
            f"{annotation} entries must contain only 'select'; a <group> label is accepted for "
            f"{RFD4_GROUP_CONDITIONS} only. For C_CRDGRP the group ID is the condition value."
        )


# --------------------------------------------------------------------------- #
#                                 RFD4Params                                   #
# --------------------------------------------------------------------------- #

class RFD4Params(UserDict):
    """
    RFD4Params class
    ================

    :class:`RFD4Params` is a helper class to specify input for RFD4-Proteina.
    It manages per-pose assignment of *condition specifications* (the RFD4
    YAML/JSON input format, see the RFD4-Proteina documentation page
    ``docs/reference/yaml-json.md``) in dict-like format.

    Each pose is represented by one key in the underlying dictionary.  The key
    is the pose description (filename stem) and the value is a nested ``dict``
    holding the condition-spec fields for that design.  The mandatory
    ``"input"`` field (the absolute path to the input structure) is always
    inferred automatically from the :class:`~protflow.poses.Poses` object and
    should not be set manually.

    Accepted fields per design key are the RFD4 top-level spec fields:

    ``input`` : str or list of str
        Starting structure(s); set automatically from poses.  RFD4 reads
        ``.cif``, ``.cif.gz`` and ``.bcif`` only — **not** PDB.
    ``input_as_annotated_cif`` : bool
        ``True`` loads ``input`` as is, preserving existing condition
        annotations.  ``False`` (default) treats ``input`` as a raw structure:
        heavy atoms are added, ``MSE`` is converted to ``MET``, and existing
        conditioning is ignored.
    ``contig`` : str
        Comma-separated contig tokens; subsets the input to preserved segments
        and inserts expandable segments.
    ``conditions`` : dict
        Mapping of condition annotation to ``{value: [{select: ...}, ...]}``.

    Three construction paths are supported:

    * **From poses** (default) — pass a :class:`~protflow.poses.Poses` object;
      a skeleton specification dict with only the ``"input"`` field is created
      for each pose automatically.
    * **From an existing specification** — pass *spec_from_yaml*,
      *spec_from_json* (paths to a condition-spec file) or *spec_from_dict* (a
      nested ``dict`` or another :class:`RFD4Params`).

    Parameters
    ----------
    poses : Poses
        The current pose collection.  Used to derive the pose descriptions
        (keys) and to auto-populate the ``"input"`` field.  Pass an empty
        :class:`~protflow.poses.Poses` for unconditional generation, where the
        spec is supplied through :meth:`add_specs` instead.
    spec_from_yaml : str, optional
        Path to an existing YAML condition spec.  Mutually exclusive with the
        other ``spec_from_*`` arguments.
    spec_from_json : str, optional
        Path to an existing JSON condition spec.  Mutually exclusive with the
        other ``spec_from_*`` arguments.
    spec_from_dict : dict or RFD4Params, optional
        An existing nested specification dictionary.  Mutually exclusive with
        the other ``spec_from_*`` arguments.

    Attributes
    ----------
    poses : Poses
        Reference to the current pose collection.
    data : dict
        The underlying specification dictionary.  Keys are pose descriptions;
        values are the per-design condition-spec dicts.
    global_args : dict
        Fields folded into *every* design in the written spec file, emitted as
        the reserved ``global_args`` top-level key.  A value set on a design
        key itself overrides the global default.

    Raises
    ------
    ValueError
        If more than one ``spec_from_*`` argument is provided.

    Examples
    --------
    Binder design, same spec for every pose::

        params = RFD4Params(poses=poses)
        params.set_input_specs(contig="50-120, /0, A17-131")
        params.set_coordinate_group(select="sel('A') and is_protein_backbone", group_id=1)
        params.set_sequence_condition(select="sel('A')")
        params.set_hotspots(select=ResidueSelection(["A45", "A46", "A82"]))

    Per-pose motifs from ``poses.df`` columns::

        params = RFD4Params(poses=poses)
        params.set_per_pose_contig(contig="contig_col")
        params.set_per_pose_condition("C_CRD", value=True, select="motif_col")

    Unconditional monomers from a benchmark input::

        params = RFD4Params(poses=Poses())
        params.add_specs({"monomer_100": {
            "input": "/opt/RFD4-Proteina/data/benchmarks/monomer/monomer_100.cif.zst",
            "input_as_annotated_cif": True,
        }})
    """

    def __init__(
        self,
        poses: Poses,
        spec_from_yaml: str = None,
        spec_from_json: str = None,
        spec_from_dict: dict | RFD4Params = None,
    ):
        super().__init__()
        self.poses = poses
        self.global_args: dict = {}

        provided = [arg for arg in (spec_from_yaml, spec_from_json, spec_from_dict) if arg]
        if len(provided) > 1:
            raise ValueError("<spec_from_yaml>, <spec_from_json> and <spec_from_dict> are mutually exclusive!")

        if spec_from_yaml:
            self.spec_from_yaml(spec_from_yaml)
        elif spec_from_json:
            self.spec_from_json(spec_from_json)
        elif spec_from_dict:
            self.spec_from_dict(spec_from_dict)
        else:
            self.data = self._create_pose_dict(poses)

    # ------------------------------------------------------------------ #
    #                               access                               #
    # ------------------------------------------------------------------ #

    @property
    def input_specs(self) -> dict:
        """Return the raw specification dictionary.

        Returns
        -------
        dict
            The underlying ``data`` dictionary mapping pose descriptions to
            their per-design condition-spec dicts.
        """
        return self.data

    def set_global_args(self, **kwargs) -> RFD4Params:
        """
        Set condition-spec fields shared by every design in the spec file.

        These are written to the reserved ``global_args`` top-level key.  A
        field set on an individual design key overrides the global default.

        Parameters
        ----------
        **kwargs
            Any of the RFD4 top-level spec fields
            (:data:`RFD4_SPEC_FIELDS`).  ``input`` is rejected here because it
            is managed per pose.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If an unknown field or ``input`` is supplied.

        Examples
        --------
        ::

            params.set_global_args(input_as_annotated_cif=True)
        """
        for key in kwargs:
            if key == "input":
                raise ValueError("<input> is managed per pose and cannot be set as a global_arg!")
            if key not in RFD4_SPEC_FIELDS:
                raise ValueError(f"Unknown condition-spec field '{key}'. Accepted fields: {RFD4_SPEC_FIELDS}")
        self.global_args.update(_to_native(copy.deepcopy(kwargs)))
        return self

    # ------------------------------------------------------------------ #
    #                           top-level fields                         #
    # ------------------------------------------------------------------ #

    def set_input_specs(self, **kwargs) -> RFD4Params:
        """
        Apply the same condition-spec fields to all poses.

        Updates the specification for every pose in :attr:`data` with the
        supplied keyword arguments.  This is the preferred method for setting
        fields that are identical across all input poses (e.g. a shared
        ``contig``).

        Parameters
        ----------
        **kwargs
            RFD4 top-level condition-spec fields
            (:data:`RFD4_SPEC_FIELDS`):

            ``contig`` : str, ResidueSelection, AtomSelection or list
                Contig tokens.  Selection objects are converted to preserved
                segments by :func:`selection_to_contig`.
            ``input_as_annotated_cif`` : bool
                Whether ``input`` already carries condition annotations.
            ``conditions`` : dict
                A complete ``conditions`` block.  Prefer the ``set_*``
                condition methods, which build and validate this for you.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If an unknown condition-spec field is supplied.

        Warns
        -----
        logging.WARNING
            Emitted if ``"input"`` is included in *kwargs*, since the ``input``
            field is reserved for automatic population from poses.

        Notes
        -----
        Values are deep-copied per pose, so later per-pose edits never leak
        into other poses.

        Examples
        --------
        ::

            params.set_input_specs(contig="50-120, /0, A17-131",
                                   input_as_annotated_cif=False)
        """
        if not self.data:
            raise ValueError(
                "Cannot set input specs on an empty RFD4Params. Add design keys with add_specs() first "
                "(unconditional generation) or construct RFD4Params from a non-empty Poses object."
            )

        for key in kwargs:
            if key == "input":
                logging.warning(
                    "Defining <input> manually not recommended, it is deduced automatically from poses. "
                    "Make sure to use absolute paths!"
                )
            elif key not in RFD4_SPEC_FIELDS:
                raise ValueError(f"Unknown condition-spec field '{key}'. Accepted fields: {RFD4_SPEC_FIELDS}")

        for pose in self.data:
            self.data[pose].update(_to_native(copy.deepcopy(kwargs)))

        self.selections_to_queries()
        return self

    def set_per_pose_input_specs(self, **kwargs) -> RFD4Params:
        """
        Apply pose-specific condition-spec fields from DataFrame columns or lists.

        For each keyword argument, the value can be either the name of a column
        in ``poses.df`` or a list of per-pose values.  Each pose is updated with
        its corresponding value; ``NaN`` / ``None`` entries are silently skipped
        so that poses without a specification for a given field are left
        unchanged.

        Parameters
        ----------
        **kwargs
            Per-pose condition-spec fields.  Each value must be one of:

            ``str``
                The name of a column in ``poses.df``.
            ``list``
                A list of values with length equal to ``len(poses)``.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If :attr:`poses` is empty, if an unknown condition-spec field is
            supplied, or if a list value has a different length than the number
            of poses.
        TypeError
            If a value is neither a ``str`` (column name) nor a ``list``.

        Examples
        --------
        Different contig strings per pose, stored in a DataFrame column::

            params.set_per_pose_input_specs(contig="contig_col")
        """
        for key in kwargs:
            if key == "input":
                logging.warning(
                    "Defining <input> manually not recommended, it is deduced automatically from poses. "
                    "Make sure to use absolute paths!"
                )
            elif key not in RFD4_SPEC_FIELDS:
                raise ValueError(f"Unknown condition-spec field '{key}'. Accepted fields: {RFD4_SPEC_FIELDS}")

        for key, val in kwargs.items():
            for pose, spec in zip(self.poses.df["poses_description"], self._per_pose_values(val, parameter_name=key)):
                if _is_missing(spec):
                    continue
                self.data[pose][key] = _to_native(copy.deepcopy(spec))

        self.selections_to_queries()
        return self

    def set_contig(self, contig) -> RFD4Params:
        """
        Set one ``contig`` string for every pose.

        Parameters
        ----------
        contig : str, ResidueSelection, AtomSelection, or list
            Contig specification.  Selection objects are converted to
            preserved-segment tokens; mix them with expandable segments and
            chainbreaks by passing a list, e.g.
            ``["50-120", "/0", motif_selection]``.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Examples
        --------
        ::

            params.set_contig(["50-120", "/0", ResidueSelection(["A17", "A18"])])
        """
        return self.set_input_specs(contig=selection_to_contig(contig))

    def set_per_pose_contig(self, contig: str | list) -> RFD4Params:
        """
        Set a pose-specific ``contig`` from a ``poses.df`` column or a list.

        Parameters
        ----------
        contig : str or list
            Name of a ``poses.df`` column holding per-pose contigs/selections,
            or a list of per-pose values with length ``len(poses)``.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Examples
        --------
        ::

            params.set_per_pose_contig("rfd4_contig")
        """
        return self.set_per_pose_input_specs(contig=contig)

    # ------------------------------------------------------------------ #
    #                              conditions                            #
    # ------------------------------------------------------------------ #

    def set_condition(self, annotation: str, value, select, group=None, overwrite: bool = False) -> RFD4Params:
        """
        Add one condition entry to every pose's ``conditions`` block.

        This is the generic entry point behind all ``set_*`` condition
        convenience methods.  It validates the annotation name and value,
        converts *select* into an AtomWorks query, and appends the entry.

        Parameters
        ----------
        annotation : str
            Condition name, either the ``C_*`` shorthand (``"C_CRD"``) or the
            full name (``"coordinate"``).  Normalized by
            :func:`normalize_condition_name`.
        value : bool, str, int, float, or None
            The annotation value this selection should receive.  ``C_CRD`` and
            ``C_SEQ`` take booleans; the other conditions take enum, string,
            integer or float values (see the RFD4 condition table).
        select : str, ResidueSelection, AtomSelection, or list
            Atoms/residues the value applies to.  Converted by
            :func:`selection_to_query`.
        group : str, optional
            Group label, required for ``C_IDX`` and rejected for every other
            condition.  Selections sharing a label receive one fresh
            index-group ID, preserving their relative residue offsets.
        overwrite : bool, optional
            When ``True``, any existing entries for this annotation are
            discarded before the new entry is added.  Default is ``False``,
            which appends — RFD4 resolves overlapping entries for the same
            annotation **last write wins**, in declaration order.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If the annotation is unknown, the value is outside the annotation's
            domain, or a required/forbidden ``group`` label is missing/present.

        Notes
        -----
        Anything you do not mention is not conditioned.  Setting only ``C_HOT``
        gets you a hotspot and nothing else: no fixed coordinates and no known
        sequence.

        Examples
        --------
        ::

            params.set_condition("C_HB", value="DONOR", select="sel('B/*/*/N')")
            params.set_condition("C_SEQ_PROB_CYS", value=0.0, select="sel('B/*/[10-20]')")
        """
        annotation = normalize_condition_name(annotation)
        value, group = _to_native(value), _to_native(group)
        _validate_condition_value(annotation, value, group)
        query = selection_to_query(select)

        entry = {"select": query}
        if group is not None:
            entry["group"] = group

        for spec in self.data.values():
            conditions = spec.setdefault("conditions", {})
            if overwrite:
                conditions.pop(annotation, None)
            conditions.setdefault(annotation, {}).setdefault(value, []).append(dict(entry))

        return self

    def set_per_pose_condition(self, annotation: str, value, select, group=None, overwrite: bool = False) -> RFD4Params:
        """
        Add one condition entry per pose, with pose-specific selections.

        Parameters
        ----------
        annotation : str
            Condition name; see :meth:`set_condition`.
        value : object, str, or list
            Either a single value applied to every pose, the name of a
            ``poses.df`` column holding per-pose values, or a list of per-pose
            values.  A string is interpreted as a column name **only** when a
            column of that name exists; otherwise it is treated as a literal
            value (e.g. the enum ``"DONOR"``).
        select : str or list
            Name of a ``poses.df`` column holding per-pose selections (query
            strings, :class:`~protflow.residues.ResidueSelection` or
            :class:`~protflow.residues.AtomSelection` objects), or a list of
            per-pose selections with length ``len(poses)``.
        group : str, list, or None, optional
            Group label, or a column name / list of per-pose labels.  Required
            for ``C_IDX``, rejected otherwise.
        overwrite : bool, optional
            When ``True``, existing entries for this annotation are discarded
            first.  Default is ``False`` (append).

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If :attr:`poses` is empty, or if a per-pose list length does not
            match the number of poses.

        Notes
        -----
        Poses whose *select* value is ``NaN``/``None``/an empty selection are
        skipped, so a motif column that is only populated for some poses does
        not create empty conditions for the rest.

        Examples
        --------
        Fix per-pose motif coordinates stored as ResidueSelections::

            params.set_per_pose_condition("C_CRD", value=True, select="motif")
        """
        annotation = normalize_condition_name(annotation)

        selects = self._per_pose_values(select, parameter_name="select")
        values = self._per_pose_values(value, parameter_name="value", allow_scalar=True)
        groups = self._per_pose_values(group, parameter_name="group", allow_scalar=True)

        for pose, pose_select, pose_value, pose_group in zip(
            self.poses.df["poses_description"], selects, values, groups
        ):
            if _is_missing(pose_select) or (hasattr(pose_select, "__len__") and len(pose_select) == 0):
                continue
            if _is_missing(pose_group):
                pose_group = None

            pose_value, pose_group = _to_native(pose_value), _to_native(pose_group)
            _validate_condition_value(annotation, pose_value, pose_group)
            entry = {"select": selection_to_query(pose_select)}
            if pose_group is not None:
                entry["group"] = pose_group

            conditions = self.data[pose].setdefault("conditions", {})
            if overwrite:
                conditions.pop(annotation, None)
            conditions.setdefault(annotation, {}).setdefault(pose_value, []).append(entry)

        return self

    def clear_conditions(self, annotation: str = None) -> RFD4Params:
        """
        Remove conditions from every pose.

        Parameters
        ----------
        annotation : str, optional
            Condition to remove.  When ``None`` (default), the whole
            ``conditions`` block is removed from every design.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        if annotation is not None:
            annotation = normalize_condition_name(annotation)
        for spec in self.data.values():
            if annotation is None:
                spec.pop("conditions", None)
            elif "conditions" in spec:
                spec["conditions"].pop(annotation, None)
        return self

    # --- condition convenience wrappers -------------------------------- #

    def set_coordinate_condition(self, select, value: bool = True, **kwargs) -> RFD4Params:
        """
        Set ``C_CRD`` (absolute coordinate conditioning) on a selection.

        Reveals each selected atom's real coordinate to the model.  Combine
        with ``is_protein_backbone`` for backbone-only conditioning that does
        not leak sequence, e.g.
        ``select="sel('B') and is_protein_backbone"``.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms whose coordinates are revealed.
        value : bool, optional
            ``True`` (default) fixes the coordinates; ``False`` clears absolute
            conditioning, which is required before assigning those atoms to an
            internal-coordinate group.
        **kwargs
            Forwarded to :meth:`set_condition` (e.g. ``overwrite``).

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        return self.set_condition("C_CRD", value=value, select=select, **kwargs)

    def set_sequence_condition(self, select, value: bool = True, **kwargs) -> RFD4Params:
        """
        Set ``C_SEQ`` (sequence conditioning) on a selection.

        Stores each selected residue's real ``res_name``.  The selection is
        resolved at residue level.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Residues whose identity is revealed.
        value : bool, optional
            ``True`` (default) reveals the identity; ``False`` leaves it free.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        return self.set_condition("C_SEQ", value=value, select=select, **kwargs)

    def set_hotspots(self, select, value: bool = True, **kwargs) -> RFD4Params:
        """
        Set ``C_HOT`` (hotspots) on a selection.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to mark as hotspots.
        value : bool, optional
            Default ``True``.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        Keep hotspot conditioning sparse.  The RFD4 conditioning guide warns
        that saturating a condition — marking every exposed atom a hotspot, for
        example — pushes the model off its training distribution.
        """
        return self.set_condition("C_HOT", value=value, select=select, **kwargs)

    def set_rasa_condition(self, select, value: str = "BURIED", **kwargs) -> RFD4Params:
        """
        Set ``C_RAS`` (binned relative solvent accessibility) on a selection.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to annotate.
        value : str or int, optional
            One of ``"BURIED"``, ``"INTERMEDIATE"``, ``"EXPOSED"``,
            ``"UNSPECIFIED"`` (or the corresponding integer bin).  Default is
            ``"BURIED"``.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        RASA was trained with sparse conditioning; annotate the few atoms whose
        burial actually matters rather than the whole structure.
        """
        return self.set_condition("C_RAS", value=value, select=select, **kwargs)

    def set_hbond_condition(self, select, value: str = "DONOR", **kwargs) -> RFD4Params:
        """
        Set ``C_HB`` (hydrogen-bond interaction role) on a selection.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to annotate.
        value : str, optional
            ``"DONOR"``, ``"ACCEPTOR"`` or ``"BOTH"``.  ``"BOTH"`` requests
            donation *and* acceptance in different bonds; separate ``DONOR``
            and ``ACCEPTOR`` entries do not combine automatically.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        return self.set_condition("C_HB", value=value, select=select, **kwargs)

    def set_pistack_condition(self, select, value: bool = True, **kwargs) -> RFD4Params:
        """
        Set ``C_PISTACK`` (pi-stacking) on a selection of complete aromatic rings.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Must cover complete aromatic rings.
        value : bool, optional
            Default ``True``.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        return self.set_condition("C_PISTACK", value=value, select=select, **kwargs)

    def set_pication_condition(self, select, value: str = "AROMATIC", **kwargs) -> RFD4Params:
        """
        Set ``C_PICATION`` (pi-cation interaction) on a selection.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to annotate.  The ``AROMATIC`` side must select complete
            aromatic rings.
        value : str, optional
            ``"AROMATIC"``, ``"CATION"`` or ``"BOTH"``.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        return self.set_condition("C_PICATION", value=value, select=select, **kwargs)

    def set_cath_condition(self, select, value: str, **kwargs) -> RFD4Params:
        """
        Set ``C_CATH`` (CATH class.architecture.topology label) on a selection.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Whole residues, usually a chain or a named expandable segment.
        value : str
            A C.A.T label such as ``"2.60.40"``, with ``x`` for unspecified
            trailing levels (``"1.x.x"``; ``"x.x.x"`` clears it).
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        The condition addresses the CATH hierarchy only down to topology, so
        homologous-superfamily folds cannot be requested directly.
        """
        return self.set_condition("C_CATH", value=value, select=select, **kwargs)

    def set_centroid_condition(self, select, value: str, **kwargs) -> RFD4Params:
        """
        Set ``C_CENTROID`` (chain centroid) on a complete chain.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Must cover the *entire* chain.
        value : str
            A literal ``"x,y,z"`` coordinate string.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        The condition is per chain and all-or-nothing: either every chain in the
        design gets a centroid or none does, and a partially covered chain is
        rejected.
        """
        return self.set_condition("C_CENTROID", value=value, select=select, **kwargs)

    def set_sequence_probability(self, select, token: str, probability: float, **kwargs) -> RFD4Params:
        """
        Set ``C_SEQ_PROB_<TOKEN>`` (per-token sequence probability) on residues.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Residues to annotate.
        token : str
            One canonical polymer token: a three-letter amino acid
            (``"CYS"``), an RNA base (``"U"``) or a DNA base (``"DA"``).  See
            :data:`RFD4_SEQ_PROB_TOKENS`.
        probability : float
            Probability in ``[0, 1]``.  ``0.0`` is an explicit exclusion; an
            omitted identity remains flexible and shares the remaining
            probability mass.  Specified values on one residue may not sum
            above 1.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Examples
        --------
        Forbid cysteine in a designed stretch::

            params.set_sequence_probability(select="sel('B/*/[10-20]')",
                                            token="CYS", probability=0.0)
        """
        return self.set_condition(f"C_SEQ_PROB_{token.strip().upper()}", value=float(probability), select=select, **kwargs)

    def set_index_group(self, select, group: str, **kwargs) -> RFD4Params:
        """
        Set ``C_IDX: false`` (unindexing) on a selection, under a group label.

        Unindexed atoms lose their absolute residue index; selections sharing a
        *group* label receive one fresh index-group ID that preserves their
        relative residue offsets.  This is how unindexed motif scaffolding is
        specified.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to unindex.
        group : str
            Group label.  Call this method several times with the same label to
            tie multiple segments into one index group.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Examples
        --------
        ::

            params.set_index_group(select="sel('A/*/[10-12]')", group="linker_a")
            params.set_index_group(select="sel('B/*/[40-42]')", group="linker_a")
        """
        return self.set_condition("C_IDX", value=False, select=select, group=group, **kwargs)

    def set_coordinate_group(self, select, group_id: int | None, **kwargs) -> RFD4Params:
        """
        Set ``C_CRDGRP`` (internal-coordinate group membership) on a selection.

        Atoms in one positive-integer group have their *relative* geometry
        conditioned rather than absolute coordinates — the recommended way to
        condition on a rigid target in PPI design.

        Parameters
        ----------
        select : str, ResidueSelection, AtomSelection, or list
            Atoms to assign.
        group_id : int or None
            Positive integer group ID, or ``None`` to remove group membership.
            IDs ``0`` and ``-1`` are reserved and cannot be assigned.
        **kwargs
            Forwarded to :meth:`set_condition`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        Before assigning an internal-coordinate group to atoms that already
        carry absolute conditioning, clear it with a ``C_CRD: false`` entry
        **declared first** (``set_coordinate_condition(select, value=False)``),
        because entries resolve in declaration order.  Setting ``C_CRD: true``
        removes atoms from their internal-coordinate group.

        Examples
        --------
        Backbone-only internal-coordinate conditioning on a rigid target::

            params.set_coordinate_group(select="sel('A') and is_protein_backbone", group_id=1)
            params.set_sequence_condition(select="sel('A')")
        """
        return self.set_condition("C_CRDGRP", value=group_id, select=select, **kwargs)

    # ------------------------------------------------------------------ #
    #                      construction / bulk editing                   #
    # ------------------------------------------------------------------ #

    def spec_from_dict(self, spec_dict: dict | RFD4Params) -> RFD4Params:
        """
        Replace the current specification with data from a nested dict.

        Validates the format of *spec_dict*, replaces :attr:`data`, pulls out a
        ``global_args`` key if present, and refreshes the ``"input"`` fields for
        any poses currently in :attr:`poses`.

        Parameters
        ----------
        spec_dict : dict or RFD4Params
            A nested specification dictionary in the format
            ``{design_key: {field: value, ...}, ...}``.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If *spec_dict* is not properly nested, contains unknown
            condition-spec fields, or does not match the current poses.

        Examples
        --------
        ::

            params.spec_from_dict({"scaffold_001": {"contig": "A4-20, 5-10"}})
        """
        spec_dict = copy.deepcopy(dict(spec_dict))
        self.global_args.update(spec_dict.pop("global_args", {}) or {})
        self._check_specs(spec_dict)
        self.data = spec_dict
        self._update_input()
        self.selections_to_queries()
        return self

    def spec_from_yaml(self, yaml_path: str) -> RFD4Params:
        """
        Replace the current specification by loading a YAML condition spec.

        Parameters
        ----------
        yaml_path : str
            Path to a YAML condition-spec file.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If no file exists at *yaml_path* or the parsed spec is invalid.

        Examples
        --------
        ::

            params.spec_from_yaml("/data/binder_spec.yaml")
        """
        if not os.path.isfile(yaml_path):
            raise ValueError(f"Could not detect condition-spec file at {yaml_path}!")
        with open(yaml_path, "r", encoding="UTF-8") as handle:
            spec = yaml.safe_load(handle)
        return self.spec_from_dict(spec)

    def spec_from_json(self, json_path: str) -> RFD4Params:
        """
        Replace the current specification by loading a JSON condition spec.

        JSON condition specs quote every key, including boolean condition
        values (``"true"``, ``"false"``, ``"null"``); those keys are converted
        back to Python ``True`` / ``False`` / ``None`` on load.

        Parameters
        ----------
        json_path : str
            Path to a JSON condition-spec file.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If no file exists at *json_path* or the parsed spec is invalid.
        """
        if not os.path.isfile(json_path):
            raise ValueError(f"Could not detect condition-spec file at {json_path}!")
        with open(json_path, "r", encoding="UTF-8") as handle:
            spec = json.load(handle)
        return self.spec_from_dict(_json_condition_keys_to_python(spec))

    def reset_pose_specs(self, poses: Poses) -> RFD4Params:
        """
        Clear all existing specifications and reinitialise from a new pose collection.

        Parameters
        ----------
        poses : Poses
            The new pose collection to use as the basis for the specification.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Notes
        -----
        All previously set spec fields (contigs, conditions) are lost.  Call
        :meth:`set_input_specs` or the ``set_*`` condition methods afterwards to
        repopulate.  :attr:`global_args` is preserved.
        """
        self.poses = poses
        self.data = self._create_pose_dict(poses)
        return self

    def add_specs(self, additional_specs: RFD4Params | dict) -> RFD4Params:
        """
        Add new design keys to a specification built without poses.

        This method is restricted to instances created with an empty
        :class:`~protflow.poses.Poses` (unconditional generation), where
        multiple independent design blocks are accumulated before a single run.

        Parameters
        ----------
        additional_specs : RFD4Params or dict
            Additional specification entries to merge into :attr:`data`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        ValueError
            If :attr:`poses` is non-empty, or if *additional_specs* is not a
            properly nested dictionary.

        Examples
        --------
        ::

            params = RFD4Params(poses=Poses())
            params.add_specs({"monomer_100": {
                "input": "/opt/RFD4-Proteina/data/benchmarks/monomer/monomer_100.cif.zst",
                "input_as_annotated_cif": True,
            }})
        """
        if self.poses:
            raise ValueError(
                "Additional design keys can only be added if no poses are present (unconditional generation)!"
            )
        additional_specs = copy.deepcopy(dict(additional_specs))
        self._check_specs(additional_specs)
        self.data.update(additional_specs)
        self.selections_to_queries()
        return self

    def modify_specs(self, new_specs: RFD4Params | dict) -> RFD4Params:
        """
        Update fields in existing design specifications without adding or removing keys.

        Parameters
        ----------
        new_specs : RFD4Params or dict
            A nested specification dictionary with the same design keys as
            :attr:`data`.  Only the fields listed per key are updated; all other
            existing fields are preserved.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.

        Raises
        ------
        KeyError
            If the design keys in *new_specs* do not exactly match those in
            :attr:`data`.
        """
        if not all(key in self.data for key in new_specs) or not len(self.data) == len(new_specs):
            raise KeyError("Design keys in <new_specs> do not match existing design keys!")

        for key in new_specs:
            self.data[key].update(copy.deepcopy(new_specs[key]))
        self._update_input()
        self.selections_to_queries()
        return self

    def selections_to_queries(self) -> RFD4Params:
        """
        Convert every selection object in the spec to its string form.

        Walks all design keys and converts ``contig`` values with
        :func:`selection_to_contig` and every ``select:`` value inside
        ``conditions`` with :func:`selection_to_query`.  Values that are already
        strings are left unchanged, so the method is idempotent and safe to call
        after manual edits to :attr:`data`.

        Returns
        -------
        RFD4Params
            ``self``, to allow method chaining.
        """
        for spec in self.data.values():
            if "contig" in spec and spec["contig"] is not None:
                spec["contig"] = selection_to_contig(spec["contig"])
            for annotation, values in (spec.get("conditions") or {}).items():
                for entries in values.values():
                    for entry in entries:
                        if "select" in entry:
                            entry["select"] = selection_to_query(entry["select"])

        if "contig" in self.global_args and self.global_args["contig"] is not None:
            self.global_args["contig"] = selection_to_contig(self.global_args["contig"])
        return self

    # ------------------------------------------------------------------ #
    #                                output                              #
    # ------------------------------------------------------------------ #

    def to_spec_dict(self, keys: list[str] = None) -> dict:
        """
        Return the condition spec as a plain dictionary, ready to serialise.

        Parameters
        ----------
        keys : list of str, optional
            Restrict the returned spec to these design keys.  When ``None``
            (default), every key is included.

        Returns
        -------
        dict
            ``{design_key: spec, ...}``, with a ``global_args`` entry prepended
            when :attr:`global_args` is non-empty.

        Raises
        ------
        ValueError
            If a requested key is not present, or if any design lacks the
            required ``input`` field (and no global ``input`` is set).
        """
        self.selections_to_queries()
        keys = list(self.data) if keys is None else list(keys)

        missing = [key for key in keys if key not in self.data]
        if missing:
            raise ValueError(f"Design keys {missing} are not present in this RFD4Params object!")

        spec: dict = {}
        if self.global_args:
            spec["global_args"] = copy.deepcopy(self.global_args)
        for key in keys:
            design = copy.deepcopy(self.data[key])
            if not design.get("input") and "input" not in self.global_args:
                raise ValueError(
                    f"Design '{key}' has no <input>. RFD4 requires a starting structure for every design "
                    "(.cif, .cif.gz or .bcif)."
                )
            spec[key] = design
        return spec

    def write_spec(self, path: str, keys: list[str] = None) -> str:
        """
        Write the condition spec to a YAML or JSON file.

        Parameters
        ----------
        path : str
            Output path.  A ``.json`` suffix writes JSON (with condition keys
            stringified as RFD4 expects); anything else writes YAML.
        keys : list of str, optional
            Restrict the written spec to these design keys.

        Returns
        -------
        str
            Absolute path to the written file.

        Notes
        -----
        Key order is preserved on write.  This matters: RFD4 resolves
        overlapping entries for one annotation **last write wins**, in
        declaration order.

        Examples
        --------
        ::

            params.write_spec("/scratch/run/inputs/designs.yaml")
        """
        spec = self.to_spec_dict(keys=keys)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        with open(path, "w", encoding="UTF-8") as handle:
            if path.endswith(".json"):
                json.dump(_python_condition_keys_to_json(spec), handle, indent=2)
            else:
                yaml.dump(spec, handle, Dumper=_RFD4Dumper, sort_keys=False, default_flow_style=False)
        return os.path.abspath(path)

    # ------------------------------------------------------------------ #
    #                               internals                            #
    # ------------------------------------------------------------------ #

    def _per_pose_values(self, value, parameter_name: str, allow_scalar: bool = False) -> list:
        """
        Resolve a per-pose argument into one value per pose.

        Parameters
        ----------
        value : str, list, or object
            A ``poses.df`` column name, a list with one entry per pose, or (if
            *allow_scalar*) a scalar broadcast to all poses.
        parameter_name : str
            Name used in error messages.
        allow_scalar : bool, optional
            When ``True``, a value that is neither a known column name nor a
            list is broadcast to every pose.

        Returns
        -------
        list
            One value per pose, in ``poses.df`` order.
        """
        if not self.poses:
            raise ValueError("Per-pose input specifications cannot be set on empty poses!")

        if isinstance(value, str) and value in self.poses.df.columns:
            return list(self.poses.df[value])

        if isinstance(value, str):
            if allow_scalar:
                return [value] * len(self.poses)
            col_in_df(self.poses.df, value)  # raises with the standard ProtFlow message

        if isinstance(value, list):
            if not len(value) == len(self.poses):
                raise ValueError(
                    f"Length of <{parameter_name}> ({len(value)}) does not match number of poses ({len(self.poses)})!"
                )
            return value

        if allow_scalar:
            return [value] * len(self.poses)

        raise TypeError(
            f"<{parameter_name}> must be a str indicating a poses dataframe column or a list, not {type(value)}!"
        )

    def _update_input(self) -> None:
        """
        Refresh the ``"input"`` field for every pose from the current pose paths.

        Iterates over ``poses.df`` and sets ``data[description]["input"]`` to
        the absolute path of each pose file.  Does nothing when :attr:`poses` or
        :attr:`data` is empty.
        """
        if self.poses and self.data:
            for name, path in zip(self.poses.df["poses_description"], self.poses.df["poses"]):
                if name in self.data:
                    self.data[name]["input"] = os.path.abspath(path)

    def _check_specs(self, specs: RFD4Params | dict) -> None:
        """
        Validate that *specs* conforms to the condition-spec format.

        Raises
        ------
        ValueError
            If any top-level value is not a ``dict``, if a design carries an
            unknown field, if a ``conditions`` block is malformed, or if the
            design keys do not match the current poses.
        """
        if not all(isinstance(v, dict) for v in specs.values()):
            dict_example = {"design_1": {"contig": "A1-20, 10-30", "conditions": {"C_CRD": {True: [{"select": "sel('A')"}]}}}}
            raise ValueError(f"Condition specifications must be supplied in the format {dict_example}")

        for key, design in specs.items():
            unknown = [field for field in design if field not in RFD4_SPEC_FIELDS]
            if unknown:
                raise ValueError(
                    f"Design '{key}' carries unknown condition-spec field(s) {unknown}. "
                    f"Accepted fields: {RFD4_SPEC_FIELDS}"
                )
            _check_conditions_block(design.get("conditions"), design_key=key)

        if self.poses and (
            not all(pose in specs for pose in self.poses.df["poses_description"]) or not len(self.poses) == len(specs)
        ):
            raise ValueError("Design keys in the condition spec do not fit existing poses!")

    def _create_pose_dict(self, poses: Poses) -> dict:
        """
        Build a skeleton specification dict from a pose collection.

        Returns
        -------
        dict
            ``{pose_description: {"input": abs_path}, ...}`` for each pose, or
            ``{}`` if *poses* is empty.
        """
        if not poses:
            return {}

        for path in poses.df["poses"]:
            if not str(path).endswith((".cif", ".cif.gz", ".bcif", ".cif.zst")):
                logging.warning(
                    "RFD4-Proteina reads .cif, .cif.gz, .bcif (and .cif.zst) inputs only; pose '%s' will likely "
                    "fail to load. Convert your poses to CIF before running.", path
                )
        return {
            str(name): {"input": os.path.abspath(path)}
            for name, path in zip(poses.df["poses_description"], poses.df["poses"])
        }


def _check_conditions_block(conditions, design_key: str = "") -> None:
    """Validate the structure of one ``conditions`` block."""
    if conditions is None:
        return
    if not isinstance(conditions, dict):
        raise ValueError(f"Design '{design_key}': <conditions> must be a mapping, not {type(conditions)}.")

    for annotation, values in conditions.items():
        name = normalize_condition_name(annotation)
        if not isinstance(values, dict):
            raise ValueError(
                f"Design '{design_key}', condition {name}: expected a mapping of value -> list of selections, "
                f"not {type(values)}."
            )
        for value, entries in values.items():
            if not isinstance(entries, list):
                raise ValueError(
                    f"Design '{design_key}', condition {name}: a value maps to a *list* of selections, even for "
                    f"one entry (got {type(entries)})."
                )
            for entry in entries:
                if not isinstance(entry, dict) or "select" not in entry:
                    raise ValueError(
                        f"Design '{design_key}', condition {name}: every entry must be a mapping with a "
                        f"'select' key (got {entry!r})."
                    )
                _validate_condition_value(name, value, entry.get("group"))


def _json_condition_keys_to_python(spec: dict) -> dict:
    """Convert quoted JSON condition-value keys back to Python objects."""
    lookup = {"true": True, "false": False, "null": None}
    for key, design in spec.items():
        if key == "global_args" or not isinstance(design, dict):
            continue
        conditions = design.get("conditions")
        if not isinstance(conditions, dict):
            continue
        for annotation, values in conditions.items():
            converted = {}
            for value, entries in values.items():
                if isinstance(value, str) and value.lower() in lookup:
                    value = lookup[value.lower()]
                converted[value] = entries
            conditions[annotation] = converted
    return spec


def _python_condition_keys_to_json(spec: dict) -> dict:
    """Stringify condition-value keys for JSON serialisation."""
    spec = copy.deepcopy(spec)
    for key, design in spec.items():
        if key == "global_args" or not isinstance(design, dict):
            continue
        conditions = design.get("conditions")
        if not isinstance(conditions, dict):
            continue
        for annotation, values in conditions.items():
            conditions[annotation] = {
                ("true" if value is True else "false" if value is False else "null" if value is None else str(value)): entries
                for value, entries in values.items()
            }
    return spec


def _is_missing(value) -> bool:
    """Return True for None/NaN-like per-pose values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_native(value):
    """
    Convert numpy/pandas scalars to plain Python types.

    Values pulled from ``poses.df`` columns are numpy scalars
    (``numpy.int64``, ``numpy.bool_``, ...), which ``yaml.SafeDumper`` refuses
    to represent.  Condition values, group labels and spec fields are passed
    through this helper so a spec built from DataFrame columns always
    serialises.

    Parameters
    ----------
    value : object
        Any value destined for the condition spec.

    Returns
    -------
    object
        ``value.item()`` for numpy scalars, element-wise conversion for lists,
        tuples and dicts, otherwise *value* unchanged.  Selection objects are
        passed through untouched, to be converted later by
        :meth:`RFD4Params.selections_to_queries`.
    """
    if isinstance(value, (ResidueSelection, AtomSelection, str, type(None))):
        return value
    if type(value) in (bool, int, float):  # pylint: disable=C0123 - exact type, not numpy subclasses
        return value
    if callable(getattr(value, "item", None)) and getattr(value, "shape", ()) == ():
        return value.item()
    if isinstance(value, dict):
        return {_to_native(key): _to_native(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
#                                  Runner                                      #
# --------------------------------------------------------------------------- #

class RFD4Proteina(Runner):
    """
    RFD4Proteina Class
    ==================
    ProtFlow runner for RFD4-Proteina generation.

    :class:`RFD4Proteina` inherits from :class:`~protflow.runners.Runner` and
    wraps the ``rfd4-generate`` Hydra entry point.  One run writes one condition
    spec per job, submits ``rfd4-generate`` with ``inputs=<spec>``, and collects
    the designed CIFs plus any inline metrics into the ProtFlow score table.

    Parallelism is handled by splitting *design keys* across jobs: each job gets
    its own condition-spec file containing a disjoint subset of the designs, all
    writing into a shared output directory.  RFD4's own ``gen_njobs``/``job_id``
    sharding is deliberately not used, because it restarts the replicate counter
    per job and would collide example ids.

    Parameters
    ----------
    application_path : str, optional
        Path to the ``rfd4-generate`` executable.  Defaults to
        ``RFD4_PROTEINA_BIN_PATH`` from the ProtFlow config.
    data_path : str, optional
        Value exported as ``DATA_PATH`` before each command, needed when native
        (non-HuggingFace) checkpoints are used, since ``configs/inference.yaml``
        resolves ``ckpt_path`` and ``autoencoder_ckpt_path`` from it.  Defaults
        to ``RFD4_PROTEINA_DATA_PATH`` from the ProtFlow config when set.
    pre_cmd : str, optional
        Command prepended to every job (environment activation, ``export
        CLUSTER=...``, module loads).  Defaults to ``RFD4_PROTEINA_PRE_CMD``.
    jobstarter : JobStarter, optional
        Default jobstarter for this runner instance.

    Attributes
    ----------
    name : str
        ``"rfd4_proteina"``; used for the scorefile name and job names.
    index_layers : int
        ``1``.  RFD4's condition-spec dataset always appends a replicate index
        (``<design_key>_<i>``) to the example id, which :meth:`run` collapses
        again via :meth:`~protflow.poses.Poses.reindex_poses`.

    Notes
    -----
    * ``rfd4-generate`` is a Hydra application: every option is ``key=value``,
      never ``--key value``.
    * RFD4 writes annotated CIFs.  With *convert_cif_to_pdb* the runner also
      writes PDB copies for downstream ProtFlow tools, but those copies lose the
      per-atom provenance annotations; the CIF path is always kept in the
      ``<prefix>_cif_location`` column.

    Examples
    --------
    ::

        runner = RFD4Proteina(jobstarter=SbatchArrayJobstarter(max_cores=8, gpus=1))
        poses = runner.run(poses=poses, prefix="rfd4", params=params,
                           num_replicates=40, batch_size=8)
    """

    def __init__(
        self,
        application_path: str | None = None,
        data_path: str | None = None,
        pre_cmd: str | None = None,
        rfd4_python: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        config = require_config()

        self.application_path = application_path or load_config_path(config, "RFD4_PROTEINA_BIN_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "RFD4_PROTEINA_PRE_CMD", is_pre_cmd=True)
        self.data_path = data_path or getattr(get_config(), "RFD4_PROTEINA_DATA_PATH", "") or None
        self.script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        # the provenance worker needs atomworks, which lives in the RFD4-Proteina
        # environment rather than the ProtFlow one; it sits beside rfd4-generate
        self.rfd4_python = rfd4_python or os.path.join(os.path.dirname(str(self.application_path)), "python")

        self.jobstarter = jobstarter
        self.name = "rfd4_proteina"
        self.index_layers = 1

    def __str__(self) -> str:
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        params: RFD4Params,
        # --- rfd4-generate arguments ---
        num_replicates: int = 1,
        batch_size: int = 1,
        metrics_tags: list[str] | str | None = None,
        inference_experiment: str | None = None,
        recycle_strategy: str | None = None,
        seed: int | None = None,
        skip_existing: bool = False,
        # --- general ProtFlow parameters ---
        options: str | None = None,
        update_motifs: list[str] | None = None,
        renumber_input: bool = False,
        multiplex_poses: int | None = None,
        jobstarter: JobStarter | None = None,
        renumber_jobstarter: JobStarter | None = None,
        strict_remap: bool = True,
        convert_cif_to_pdb: bool = True,
        include_scores: list[str] | None = None,
        run_clean: bool = True,
        fail_on_missing_output_poses: bool = False,
        overwrite: bool = False,
    ) -> Poses:
        """
        Execute the full RFD4-Proteina runner lifecycle and return updated poses.

        Orchestrates the complete pipeline from condition-spec construction
        through job execution to result integration:

        1. Validates that *poses* and *params* are consistent.
        2. Optionally multiplexes input poses to saturate available GPUs.
        3. Checks for a cached scorefile and returns early if found.
        4. Splits design keys into batches and writes one condition spec per job.
        5. Assembles and submits the ``rfd4-generate`` commands.
        6. Collects designed CIFs and inline metrics.
        7. Reindexes poses to remove the replicate index layer and returns.

        Parameters
        ----------
        poses : Poses
            Input pose collection.  May be empty for unconditional generation;
            in that case ``poses.df`` is populated directly from the collected
            scores.
        prefix : str
            Column prefix used to namespace all new columns added to
            ``poses.df`` and to name the working directory
            (``<poses.work_dir>/<prefix>/``).
        params : RFD4Params
            Per-pose condition specification.  Must contain exactly one design
            key per pose in *poses* (or be poses-free for unconditional runs).
            The ``input`` field is managed automatically.
        num_replicates : int, optional
            Designs generated per design key.  Default is ``1`` (a smoke test);
            production runs typically use 8-80.
        batch_size : int, optional
            Designs processed in parallel on the GPU.  Default is ``1``; confirm
            that the longest input fits at ``1`` before raising it.
        metrics_tags : list of str or str, optional
            Tags selecting which metrics run inline after generation; results
            land in ``per_example_metrics.csv`` and are merged into the score
            table.  Every ``*_refolding`` tag invokes the AF3 oracle, which
            needs its own container — plain generation does not.
        inference_experiment : str, optional
            Hydra group selector for the top-level experiment config composing
            generation, search and metrics (e.g.
            ``"sm_binder_hbond_search"`` for reward-guided search).  When
            ``None`` (default), RFD4's own default
            (``inference_on_user_inputs``) applies.
        recycle_strategy : str, optional
            Hydra group selector for the 60-step trunk schedule
            (``"<low|medium|high>/<flat|midpeak|front_loaded>"``).  When
            ``None``, RFD4's default (``thinking_medium``) applies.
        seed : int, optional
            Base random seed; job ``i`` is submitted with ``seed + i`` so that
            parallel jobs do not sample identical trajectories.  When ``None``
            (default), RFD4 draws a random seed per run.
        skip_existing : bool, optional
            Forwarded to ``skip_existing``.  ``False`` (default) regenerates
            every design; set ``True`` together with ``overwrite=False`` to
            resume an interrupted run from the outputs already on disk.
        options : str, optional
            Additional ``rfd4-generate`` arguments in Hydra ``key=value``
            format, forwarded verbatim.  Must not contain any of
            :data:`RFD4_RESERVED_OPTIONS`, which are set programmatically.
        update_motifs : list of str, optional
            Names of :class:`~protflow.residues.ResidueSelection` or
            :class:`~protflow.residues.AtomSelection` columns in ``poses.df``
            whose positions should be remapped from input numbering to the
            numbering of each generated structure.  The map is read from the
            ``SA_SRC`` standard annotation that RFD4-Proteina writes into every
            output CIF, extracted in parallel by an auxiliary worker.  When
            ``None`` (default), no remapping is performed and no worker runs.
        renumber_input : bool, optional
            When ``True``, write one copy of each design's input structure,
            renumbered into that design's own numbering, to
            ``<work_dir>/renumbered_inputs/<description>.cif``, and add a
            ``<prefix>_renumbered_input`` column.  Useful for superimposing an
            input motif onto the design without reconstructing the mapping by
            hand.  Produces an identical copy when the design preserved the
            input numbering.  Uses the same worker and *renumber_jobstarter* as
            *update_motifs*, and can be enabled independently of it.
        multiplex_poses : int, optional
            If set to an integer > 1, each design key is duplicated this many
            times before submission to maximise GPU utilisation when the number
            of designs is smaller than the number of available jobs.  The extra
            index layer introduced by multiplexing is collapsed during
            reindexing.  A value of ``1`` has no effect and triggers a warning.
        jobstarter : JobStarter, optional
            Job submission backend for this run.  Resolved via the standard
            ProtFlow fallback chain: argument -> ``self.jobstarter`` ->
            ``poses.default_jobstarter``.
        renumber_jobstarter : JobStarter, optional
            Jobstarter used **only** for the motif-remapping worker.  Extracting
            provenance is CPU-only and embarrassingly parallel, so a large
            CPU-only jobstarter can be used here while generation itself runs on
            a small GPU jobstarter.  Defaults to the jobstarter resolved for the
            run.
        strict_remap : bool, optional
            When ``True`` (default), :func:`remap_rfd4_motifs` raises a
            :exc:`ValueError` if a pose has no usable source index map.
            Residues absent from an otherwise valid map are preserved unchanged
            with a warning either way.
        convert_cif_to_pdb : bool, optional
            When ``True`` (default), designed CIFs are additionally converted to
            PDB via OpenBabel and ``location`` points at the PDB files.  When
            ``False``, ``location`` points at the annotated CIFs, which preserve
            RFD4's per-atom provenance columns.
        include_scores : list of str, optional
            Opt into score fields skipped by default.  Pass ``"generation"`` to
            keep the fully resolved ``generation.*`` config columns that RFD4
            embeds in every metrics row.
        run_clean : bool, optional
            When ``True`` (default), intermediate files (decompressed copies of
            compressed outputs) are removed after successful collection.
        fail_on_missing_output_poses : bool, optional
            When ``True``, a :exc:`RuntimeError` is raised if fewer output poses
            were collected than ``num_replicates x <number of design keys>``.
            Default is ``False``.
        overwrite : bool, optional
            When ``True``, the existing scorefile and output files in the
            working directory are deleted before re-running.  When ``False``
            (default), an existing scorefile causes the run to be skipped and
            cached results to be returned immediately.

        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns prefixed by
            *prefix*, including:

            ``<prefix>_location``
                Absolute path to the output PDB (or CIF) for each design.
            ``<prefix>_description``
                Unique identifier derived from the RFD4 ``example_id``.
            ``<prefix>_cif_location``
                Absolute path to the designed annotated CIF.
            ``<prefix>_renumbered_input``
                Present when *renumber_input* is ``True``.  Absolute path to the
                input structure rewritten into this design's numbering.
            ``<prefix>_task``
                The design key the output came from (RFD4's ``task`` column).
            All inline metric columns from ``per_example_metrics.csv``, with
            ``.`` replaced by ``_`` (e.g.
            ``<prefix>_clash_metrics_frac_atoms_clashing_1p5A``).

        Raises
        ------
        ValueError
            If *poses* and *params* have different lengths or mismatched design
            keys, or if *options* contains a reserved Hydra key.
        RuntimeError
            If :func:`collect_scores` returns an empty DataFrame (all jobs
            crashed or produced no output), or if
            *fail_on_missing_output_poses* is ``True`` and fewer outputs than
            expected were collected.

        Notes
        -----
        * RFD4 names each output ``<design_key>_<replicate_index>``, so one
          index layer is always added and then collapsed by
          :meth:`~protflow.poses.Poses.reindex_poses`.
        * When *poses* is empty, ``poses.df`` is populated directly from the
          score DataFrame and ``input_poses`` is set to ``None``.

        Examples
        --------
        Binder design with inline interface metrics::

            poses = runner.run(
                poses=poses,
                prefix="rfd4_binders",
                params=params,
                num_replicates=40,
                batch_size=8,
                metrics_tags=["design", "ppi_refolding"],
            )

        Reward-guided search through a shipped experiment config::

            poses = runner.run(
                poses=poses,
                prefix="rfd4_search",
                params=params,
                num_replicates=8,
                inference_experiment="sm_binder_hbond_search",
            )
        """
        # check that the condition spec fits the input poses
        if poses and (
            not all(name in params for name in poses.df["poses_description"]) or not len(poses) == len(params)
        ):
            raise ValueError("Input <poses> do not match the design keys in <params>!")

        if not len(params):
            raise ValueError(
                "<params> contains no design keys. Build an RFD4Params from poses, or add design keys with "
                "add_specs() for unconditional generation."
            )

        if update_motifs is not None:
            if not isinstance(update_motifs, list) or not all(isinstance(item, str) for item in update_motifs):
                raise ValueError(f"Parameter update_motifs must be a list of poses_df column names! Got: {update_motifs}")

        if multiplex_poses == 1:
            logging.warning("multiplex_poses=1 has no effect. Set to None or an integer > 1.")
        if not multiplex_poses:
            multiplex_poses = 1

        # RFD4 appends '_<replicate>' to every example id; multiplexing adds one more layer
        index_layers = self.index_layers + (1 if multiplex_poses > 1 else 0)

        # generic setup shared by all runners
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d design(s)", self, work_dir, len(params))

        # make sure selections are strings, in case params were manipulated manually
        params.selections_to_queries()

        # build the list of design keys for this run
        if multiplex_poses > 1:
            logging.info("Multiplexing design keys %d times.", multiplex_poses)
            design_keys = [
                (f"{key}_{str(i).zfill(4)}", key)
                for key in params
                for i in range(1, multiplex_poses + 1)
            ]
        else:
            design_keys = [(key, key) for key in params]

        # design key -> the structure that design was generated from, so the worker
        # can write a renumbered copy of it
        input_by_task = {
            str(output_key): os.path.abspath(str(params[str(params_key)].get("input", "")))
            for output_key, params_key in design_keys
            if params[str(params_key)].get("input")
        }

        total_designs = num_replicates * multiplex_poses
        expected_outputs = num_replicates * len(design_keys)
        logging.info(
            "Total designs per input pose: %d (%d replicates x %d multiplexed copies). "
            "Expected number of output poses: %d",
            total_designs, num_replicates, multiplex_poses, expected_outputs,
        )

        # scorefile reuse shortcut
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)
            if update_motifs or renumber_input:
                scores = add_source_maps_to_scores(
                    scores=scores,
                    work_dir=work_dir,
                    jobstarter=renumber_jobstarter or jobstarter,
                    python_path=self.rfd4_python,
                    script_path=self._rfd4_remap_worker_script(),
                    renumber_input=renumber_input,
                    input_by_task=input_by_task,
                )
                self.save_runner_scorefile(scores=scores, scorefile=scorefile)

            poses = self._merge_scores_into_poses(poses=poses, scores=scores, prefix=prefix, index_layers=index_layers)
            if update_motifs:
                remap_rfd4_motifs(poses=poses, motifs=update_motifs, prefix=prefix, strict=strict_remap)
            poses.reindex_poses(f"{prefix}_rfd4_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)
            return poses

        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # number of parallel jobs
        n_jobs = min(len(design_keys), jobstarter.max_cores)

        os.makedirs(output_dir := os.path.join(work_dir, "outputs"), exist_ok=True)
        os.makedirs(input_dir := os.path.join(work_dir, "inputs"), exist_ok=True)

        # write one condition spec per job and assemble commands
        cmds = self.setup_run(
            params=params,
            design_keys=design_keys,
            input_dir=input_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            n_jobs=n_jobs,
            num_replicates=num_replicates,
            batch_size=batch_size,
            metrics_tags=metrics_tags,
            inference_experiment=inference_experiment,
            recycle_strategy=recycle_strategy,
            seed=seed,
            skip_existing=skip_existing,
            options=options,
        )

        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

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
            include_scores=include_scores,
        )

        n_out_poses = len(scores.index)
        if n_out_poses == 0:
            raise RuntimeError(
                f"{self}: collect_scores returned no rows. Check runner output logs and runner output "
                f"directory ({output_dir})"
            )

        if fail_on_missing_output_poses and n_out_poses < expected_outputs:
            raise RuntimeError(
                f"Number of output poses ({n_out_poses}) is smaller than the expected number of output poses "
                f"({expected_outputs}). Some runs might have crashed!"
            )

        if update_motifs or renumber_input:
            scores = add_source_maps_to_scores(
                scores=scores,
                work_dir=work_dir,
                jobstarter=renumber_jobstarter or jobstarter,
                python_path=self.rfd4_python,
                script_path=self._rfd4_remap_worker_script(),
                renumber_input=renumber_input,
                input_by_task=input_by_task,
            )

        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        poses = self._merge_scores_into_poses(poses=poses, scores=scores, prefix=prefix, index_layers=index_layers)
        if update_motifs:
            logging.info("Remapping residue motifs %s after RFD4 run.", update_motifs)
            remap_rfd4_motifs(poses=poses, motifs=update_motifs, prefix=prefix, strict=strict_remap)
        poses.reindex_poses(f"{prefix}_rfd4_reindex", remove_layers=index_layers, force_reindex=True, overwrite=overwrite)

        logging.info("%s finished. Returning %d poses.", self, len(poses.df.index))
        return poses

    def _rfd4_remap_worker_script(self) -> str:
        """Return the configured RFD4 provenance worker script path."""
        return os.path.join(str(self.script_dir), "rfd4_remap_motifs.py")

    def _merge_scores_into_poses(self, poses: Poses, scores: pd.DataFrame, prefix: str, index_layers: int) -> Poses:
        """Merge collected RFD4 scores back into the active Poses object."""
        if not poses:
            poses.df = scores.copy()
            poses.df["input_poses"] = None
            poses.df["poses"] = poses.df["location"].apply(os.path.abspath)
            poses.df["poses_description"] = poses.df["description"]
            logging.info("Populated poses.df directly from scores (%d rows).", len(poses.df.index))
            return poses

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=index_layers).return_poses()

    def setup_run(
        self,
        params: RFD4Params,
        design_keys: list[tuple[str, str]],
        input_dir: str,
        output_dir: str,
        work_dir: str,
        n_jobs: int,
        num_replicates: int = 1,
        batch_size: int = 1,
        metrics_tags: list[str] | str | None = None,
        inference_experiment: str | None = None,
        recycle_strategy: str | None = None,
        seed: int | None = None,
        skip_existing: bool = False,
        options: str | None = None,
    ) -> list[str]:
        """
        Partition design keys into batches, write condition specs, and assemble commands.

        Splits *design_keys* into *n_jobs* sublists using
        :func:`~protflow.jobstarters.split_list`, writes one YAML condition spec
        per sublist into *input_dir*, and returns one shell command per spec.

        Parameters
        ----------
        params : RFD4Params
            The condition specification to draw designs from.
        design_keys : list of tuple
            ``(output_key, params_key)`` pairs.  *output_key* is the design key
            written into the spec (it carries the multiplex suffix when
            multiplexing); *params_key* is the key to copy the spec from.
        input_dir : str
            Directory for the per-batch condition specs (``batch0.yaml``, ...).
        output_dir : str
            Directory passed to ``rfd4-generate`` as ``out_dir``.
        work_dir : str
            Runner working directory; per-job Hydra run directories are created
            beneath it so concurrent jobs do not clobber each other's config
            snapshots.
        n_jobs : int
            Number of batches / parallel jobs to create.
        num_replicates, batch_size, metrics_tags, inference_experiment, recycle_strategy, skip_existing
            Forwarded to :meth:`write_cmd`.
        seed : int, optional
            Base seed.  Job ``i`` receives ``seed + i``, mirroring RFD4's own
            ``base_seed + job_id`` convention: every ProtFlow job runs with
            ``job_id`` unset, so a shared seed would make multiplexed copies of
            one design key reproduce each other exactly.
        options : str, optional
            Additional Hydra arguments forwarded to :meth:`write_cmd`.

        Returns
        -------
        list of str
            Shell command strings, one per batch condition spec.

        Notes
        -----
        Design keys are disjoint across batches, so a shared *output_dir* cannot
        produce colliding example ids.
        """
        batches = split_list(design_keys, n_sublists=n_jobs)

        cmds: list[str] = []
        for i, batch in enumerate(batches):
            if not batch:
                continue

            batch_params = RFD4Params(poses=Poses())
            batch_params.global_args = dict(params.global_args)
            batch_params.data = {
                str(output_key): copy.deepcopy(params[str(params_key)]) for output_key, params_key in batch
            }
            spec_path = batch_params.write_spec(os.path.join(input_dir, f"batch{i}.yaml"))

            cmds.append(
                self.write_cmd(
                    in_spec=spec_path,
                    out_dir=output_dir,
                    hydra_run_dir=os.path.join(work_dir, "hydra_outputs", f"batch{i}"),
                    num_replicates=num_replicates,
                    batch_size=batch_size,
                    metrics_tags=metrics_tags,
                    inference_experiment=inference_experiment,
                    recycle_strategy=recycle_strategy,
                    # offset the seed per job, mirroring RFD4's own `base_seed + job_id`:
                    # all jobs run with job_id unset, so a shared seed would make multiplexed
                    # copies of one design key reproduce each other exactly.
                    seed=None if seed is None else seed + i,
                    skip_existing=skip_existing,
                    options=options,
                )
            )
        return cmds

    def write_cmd(
        self,
        in_spec: str,
        out_dir: str,
        hydra_run_dir: str | None = None,
        num_replicates: int = 1,
        batch_size: int = 1,
        metrics_tags: list[str] | str | None = None,
        inference_experiment: str | None = None,
        recycle_strategy: str | None = None,
        seed: int | None = None,
        skip_existing: bool = False,
        options: str | None = None,
    ) -> str:
        """
        Compose a single ``rfd4-generate`` shell command string.

        Parameters
        ----------
        in_spec : str
            Absolute path to the condition-spec file for this batch, passed as
            ``inputs=``.
        out_dir : str
            Output directory passed as ``out_dir=``.
        hydra_run_dir : str, optional
            Directory passed as ``hydra.run.dir=``.  Give concurrent jobs
            distinct directories; otherwise Hydra's config snapshots collide.
        num_replicates : int, optional
            Designs per design key.  Default ``1``.
        batch_size : int, optional
            Designs processed in parallel on the GPU.  Default ``1``.
        metrics_tags : list of str or str, optional
            Inline metric tags, rendered as a quoted Hydra list.
        inference_experiment : str, optional
            ``inference_experiment=`` group selector.
        recycle_strategy : str, optional
            ``recycle_strategy=`` group selector.
        seed : int, optional
            ``seed=`` value.
        skip_existing : bool, optional
            ``skip_existing=`` value.  Default ``False``.
        options : str, optional
            Additional Hydra ``key=value`` arguments appended verbatim.

        Returns
        -------
        str
            A complete shell command, e.g.::

                DATA_PATH=/data rfd4-generate inputs=/run/inputs/batch0.yaml \\
                    out_dir=/run/outputs num_replicates=40 batch_size=8 \\
                    skip_existing=false metrics_tags='[design]'

        Raises
        ------
        ValueError
            If *options* contains any of :data:`RFD4_RESERVED_OPTIONS`, or uses
            ``--key value`` instead of Hydra's ``key=value`` syntax.
        """
        options = options or ""

        if "--" in options:
            raise ValueError(
                "rfd4-generate is a Hydra application: every option is 'key=value', never '--key value'. "
                f"Got options: {options}"
            )

        for reserved in RFD4_RESERVED_OPTIONS:
            if re.search(rf"(^|\s){re.escape(reserved)}=", options):
                raise ValueError(
                    f"<options> must not contain any of {RFD4_RESERVED_OPTIONS}, set them via .run() arguments instead!"
                )

        args = [
            f"inputs={in_spec}",
            f"out_dir={out_dir}",
            f"num_replicates={num_replicates}",
            f"batch_size={batch_size}",
            f"skip_existing={str(bool(skip_existing)).lower()}",
        ]

        if hydra_run_dir:
            args.append(f"hydra.run.dir={hydra_run_dir}")
        if inference_experiment:
            args.append(f"inference_experiment={inference_experiment}")
        if recycle_strategy:
            args.append(f"recycle_strategy={recycle_strategy}")
        if seed is not None:
            args.append(f"seed={seed}")
        if metrics_tags:
            tags = [metrics_tags] if isinstance(metrics_tags, str) else list(metrics_tags)
            args.append("metrics_tags='[" + ",".join(str(tag) for tag in tags) + "]'")
        if options:
            args.append(options.strip())

        env_prefix = f"DATA_PATH={self.data_path} " if self.data_path else ""
        return f"{env_prefix}{self.application_path} " + " ".join(args)

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """
        Delete generated sub-directories before a rerun.

        Removes ``<work_dir>/outputs/``, ``<work_dir>/inputs/`` and
        ``<work_dir>/hydra_outputs/`` recursively, so stale designs from a
        previous run are not collected.

        Parameters
        ----------
        work_dir : str
            Root working directory of the run.
        """
        for subdir in ["outputs", "inputs", "hydra_outputs", "renumbered_inputs", "motif_remap"]:
            path = os.path.join(work_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path)


# --------------------------------------------------------------------------- #
#                              motif remapping                                 #
# --------------------------------------------------------------------------- #

#: Columns that :func:`add_source_maps_to_scores` adds to the scores table.
RFD4_REMAP_RESULT_COLUMNS: tuple[str, ...] = (
    "source_index_map",
    "ambiguous_source_labels",
    "motif_residues",
    "unindexed_residues",
    "index_group_map",
)


def _rfd4_remap_worker_script(script_path: str | None = None) -> str:
    """Resolve the provenance worker script path, falling back to the config dir."""
    if script_path:
        return script_path
    return os.path.join(str(load_config_path(require_config(), "AUXILIARY_RUNNER_SCRIPTS_DIR")), "rfd4_remap_motifs.py")


def _run_rfd4_remap_records_with_jobstarter(
    records: list[dict],
    work_dir: str,
    jobstarter: JobStarter,
    python_path: str | None = None,
    script_path: str | None = None,
) -> list[dict]:
    """
    Dispatch RFD4 provenance-extraction records through a ProtFlow jobstarter.

    Parsing one annotated CIF is cheap, but a design campaign produces thousands
    of them, so the work is split across *jobstarter* the same way RFDiffusion3
    parallelises input renumbering.

    Parameters
    ----------
    records : list of dict
        ``[{"description": str, "cif_location": str}, ...]``.
    work_dir : str
        Runner working directory; worker JSONs are written beneath it.
    jobstarter : JobStarter
        Jobstarter used for the workers. This is CPU-only work, so it can be a
        different (larger) jobstarter than the one used for generation.
    python_path : str, optional
        Interpreter used to run the worker. Must be the RFD4-Proteina
        environment's python, since the worker imports ``atomworks`` and
        ``rfproteina``.
    script_path : str, optional
        Explicit worker script path. Defaults to
        ``<AUXILIARY_RUNNER_SCRIPTS_DIR>/rfd4_remap_motifs.py``.

    Returns
    -------
    list of dict
        One provenance record per input record.

    Raises
    ------
    FileNotFoundError
        If the worker script is missing, or a worker produced no output file.
    """
    if not records:
        return []

    worker_script = _rfd4_remap_worker_script(script_path=script_path)
    if not os.path.isfile(worker_script):
        raise FileNotFoundError(f"Cannot find RFD4-Proteina provenance worker script: {worker_script}")

    if not python_path:
        raise ValueError(
            "RFD4 motif remapping needs the RFD4-Proteina environment's python (it imports atomworks). "
            "Pass python_path, or set RFD4_PROTEINA_BIN_PATH so it can be derived."
        )

    worker_dir = os.path.join(work_dir, "motif_remap", "worker_json")
    os.makedirs(worker_dir, exist_ok=True)

    n_workers = min(len(records), max(1, jobstarter.max_cores or 1))
    record_sublists = split_list(records, n_sublists=n_workers)

    input_jsons, output_jsons = [], []
    for index, sublist in enumerate(record_sublists, start=1):
        if not len(sublist):
            continue
        input_json = os.path.join(worker_dir, f"rfd4_remap_input_{index:04}.json")
        output_json = os.path.join(worker_dir, f"rfd4_remap_output_{index:04}.json")
        with open(input_json, "w", encoding="UTF-8") as handle:
            json.dump(list(sublist), handle)
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
        jobname="rfd4_remap_motifs",
        wait=True,
        output_path=worker_dir,
    )

    missing_outputs = [output_json for output_json in output_jsons if not os.path.isfile(output_json)]
    if missing_outputs:
        raise FileNotFoundError(f"RFD4-Proteina provenance worker output files were not created: {missing_outputs}")

    results = []
    for output_json in output_jsons:
        with open(output_json, "r", encoding="UTF-8") as handle:
            results.extend(json.load(handle))
    return results


def add_source_maps_to_scores(
    scores: pd.DataFrame,
    work_dir: str,
    jobstarter: JobStarter | None = None,
    python_path: str | None = None,
    script_path: str | None = None,
    overwrite: bool = False,
    renumber_input: bool = False,
    input_by_task: dict | None = None,
) -> pd.DataFrame:
    """
    Add RFD4 provenance columns to a scores DataFrame.

    Reads the ``SA_SRC`` / ``SA_CRDSRC`` / ``SA_IDXGRPSRC`` / ``SA_UII``
    standard annotations from each designed CIF and attaches the derived maps,
    dispatching the work across *jobstarter*.

    Parameters
    ----------
    scores : pandas.DataFrame
        Scores table from :func:`collect_scores`; must carry ``description``
        and ``cif_location`` columns.
    work_dir : str
        Runner working directory.
    jobstarter : JobStarter
        Jobstarter used for the workers. Required.
    python_path : str, optional
        RFD4-Proteina environment python.
    script_path : str, optional
        Explicit worker script path.
    overwrite : bool, optional
        Recompute even when the provenance columns are already present.
    renumber_input : bool, optional
        Also write one renumbered copy of each design's input structure under
        ``<work_dir>/renumbered_inputs/``, in that design's own numbering, and
        add a ``renumbered_input`` column. Requires *input_by_task*.
    input_by_task : dict, optional
        ``{design_key: input_structure_path}``, used to find the structure each
        design came from. Rows whose ``task`` is absent are skipped with a
        warning rather than failing the run.

    Returns
    -------
    pandas.DataFrame
        Copy of *scores* with the :data:`RFD4_REMAP_RESULT_COLUMNS` added, plus
        ``renumbered_input`` when *renumber_input* is set.

    Raises
    ------
    ValueError
        If no jobstarter is supplied, or ``cif_location`` is missing.
    RuntimeError
        If a worker returned no record for some score row, or reported an error
        for one of them.
    """
    scores = scores.copy()
    wanted_columns = RFD4_REMAP_RESULT_COLUMNS + (("renumbered_input",) if renumber_input else ())
    if all(column in scores.columns for column in wanted_columns) and not overwrite:
        return scores

    if jobstarter is None:
        raise ValueError("RFD4 motif remapping requires a jobstarter.")
    for column in ("description", "cif_location"):
        if column not in scores.columns:
            raise ValueError(f"Scores table needs a '{column}' column to extract RFD4 provenance.")
    if renumber_input and not input_by_task:
        raise ValueError("renumber_input needs input_by_task to locate each design's input structure.")

    renumbered_dir = os.path.join(work_dir, "renumbered_inputs")
    tasks = scores["task"] if "task" in scores.columns else [None] * len(scores.index)

    records = []
    unknown_tasks = []
    for description, cif_location, task in zip(scores["description"], scores["cif_location"], tasks):
        record = {"description": description, "cif_location": cif_location}
        if renumber_input:
            input_location = (input_by_task or {}).get(task)
            if input_location is None:
                unknown_tasks.append(task)
            else:
                record["input_location"] = input_location
                record["renumbered_input_path"] = os.path.join(renumbered_dir, f"{description}.cif")
        records.append(record)

    if unknown_tasks:
        logging.warning(
            "No input structure known for design key(s) %s; their inputs will not be renumbered.",
            sorted({str(task) for task in unknown_tasks}),
        )
    results = _run_rfd4_remap_records_with_jobstarter(
        records=records,
        work_dir=work_dir,
        jobstarter=jobstarter,
        python_path=python_path,
        script_path=script_path,
    )

    result_by_description = {result["description"]: result for result in results}
    missing = [d for d in scores["description"].to_list() if d not in result_by_description]
    if missing:
        raise RuntimeError(f"RFD4 provenance worker returned no record for descriptions: {missing}")

    failed = {d: r["error"] for d, r in result_by_description.items() if r.get("error")}
    if failed:
        raise RuntimeError(f"RFD4 provenance extraction failed for {len(failed)} design(s): {failed}")

    for column in RFD4_REMAP_RESULT_COLUMNS:
        scores[column] = scores["description"].map(lambda d, col=column: result_by_description[d][col])
    if renumber_input:
        scores["renumbered_input"] = scores["description"].map(
            lambda d: result_by_description[d].get("renumbered_input")
        )
    return scores


def remap_rfd4_motifs(poses: Poses, motifs: list[str], prefix: str, strict: bool = True) -> None:
    """
    Remap ResidueSelection or AtomSelection motifs to RFD4 output numbering.

    Translates residue positions stored in designated
    :class:`~protflow.residues.ResidueSelection` or
    :class:`~protflow.residues.AtomSelection` columns from their original
    input-structure numbering to the corresponding positions in each generated
    structure, using the per-pose ``<prefix>_source_index_map`` built from
    RFD4's ``SA_SRC`` annotation.

    The motif columns are updated **in place**, matching the convention of the
    RFdiffusion and RFdiffusion3 runners.

    Parameters
    ----------
    poses : Poses
        Pose collection returned by :meth:`RFD4Proteina.run` with
        ``update_motifs`` set, so ``poses.df`` carries a
        ``<prefix>_source_index_map`` column.
    motifs : list of str
        Names of columns in ``poses.df`` to remap in place. Each column must be
        homogeneous: all values either
        :class:`~protflow.residues.ResidueSelection` or all
        :class:`~protflow.residues.AtomSelection`.
    prefix : str
        Run prefix used to locate the ``<prefix>_source_index_map`` column.
    strict : bool, optional
        When ``True`` (default), raise :exc:`ValueError` if a pose has no usable
        source index map. Residues absent from an otherwise valid map are
        preserved unchanged with a warning either way — expected for ligands and
        for atoms the model generated rather than carried over from the input.

    Raises
    ------
    ValueError
        If the source-map column is missing, a motif column is empty or mixes
        selection types, or (when *strict*) a pose has no usable map.

    Notes
    -----
    Atoms the model generated from scratch carry ``SA_SRC == "."`` and so never
    appear in the map; a motif residue that vanishes from the output therefore
    keeps its input identifier rather than being silently dropped.

    Examples
    --------
    ::

        poses = runner.run(poses=poses, prefix="rfd4", params=params,
                           update_motifs=["binding_site"])
        # poses.df["binding_site"] now holds output numbering
    """
    source_map_name = f"{prefix}_source_index_map"
    col_in_df(poses.df, source_map_name)
    source_index_maps = poses.df[source_map_name].to_list()

    logging.info("[remap_motifs] Motifs to remap: %s", motifs)

    for motif_col in motifs:
        logging.info("[remap_motifs] Processing motif column '%s'", motif_col)
        col_in_df(poses.df, motif_col)
        ref_motifs = poses.df[motif_col].to_list()
        if not ref_motifs:
            raise ValueError(f"Motif column {motif_col} is empty and cannot be remapped.")

        if all(isinstance(motif, ResidueSelection) for motif in ref_motifs):
            remap_selection = remap_residue_selection
        elif all(isinstance(motif, AtomSelection) for motif in ref_motifs):
            remap_selection = remap_atom_selection
        else:
            raise ValueError(f"All motifs in column {motif_col} must be either ResidueSelections or AtomSelections!")

        updated_motifs = []
        for source_index_map, ref_motif in zip(source_index_maps, ref_motifs):
            effective_map = _ensure_rfd4_source_map(source_index_map, motif_col=motif_col, strict=strict)
            updated_motifs.append(
                remap_selection(ref_motif, index_map=effective_map, motif_col=motif_col, strict=strict, tool_name="RFD4")
            )
        poses.df[motif_col] = updated_motifs

    logging.info("[remap_motifs] All motifs remapped successfully for prefix='%s'.", prefix)


def _ensure_rfd4_source_map(source_index_map, motif_col: str, strict: bool) -> dict:
    """
    Validate one pose's source index map before remapping.

    Parameters
    ----------
    source_index_map : dict or None
        Value from the ``<prefix>_source_index_map`` column.
    motif_col : str
        Motif column being remapped, used in the error message.
    strict : bool
        When ``True``, an unusable map raises; otherwise an empty map is used,
        which preserves every residue unchanged.

    Returns
    -------
    dict
        A usable mapping, possibly empty when *strict* is ``False``.
    """
    if isinstance(source_index_map, dict):
        return source_index_map
    if strict:
        raise ValueError(
            f"Cannot remap motif column '{motif_col}': this pose has no usable RFD4 source index map "
            f"(got {type(source_index_map).__name__}). Run with update_motifs set, or pass strict_remap=False "
            "to preserve the original numbering instead."
        )
    logging.warning(
        "[remap_motifs] No usable RFD4 source index map for one pose while remapping '%s'; "
        "preserving original numbering.", motif_col
    )
    return {}


# --------------------------------------------------------------------------- #
#                               score collection                               #
# --------------------------------------------------------------------------- #

#: Metrics-CSV columns that duplicate ProtFlow bookkeeping and are dropped.
_RFD4_METRICS_DROP_COLUMNS: tuple[str, ...] = ("output_base_dir", "metrics_tags")


def _decompress_structure(path: str) -> str:
    """
    Decompress a ``.cif.gz`` / ``.cif.zst`` structure file next to the original.

    Parameters
    ----------
    path : str
        Path to the compressed structure.

    Returns
    -------
    str
        Path to the decompressed ``.cif`` file.  Returns *path* unchanged if it
        is not compressed.

    Raises
    ------
    ImportError
        If a ``.cif.zst`` file is encountered but ``zstandard`` is not
        installed.
    """
    if path.endswith(".cif.gz"):
        out_path = path[: -len(".gz")]
        if not os.path.isfile(out_path):
            with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out_path

    if path.endswith(".cif.zst"):
        out_path = path[: -len(".zst")]
        if not os.path.isfile(out_path):
            try:
                import zstandard  # pylint: disable=C0415
            except ImportError as exc:
                raise ImportError(
                    "Reading .cif.zst outputs requires the 'zstandard' package. Install it, or run "
                    "rfd4-generate with compress_outputs=false."
                ) from exc
            with open(path, "rb") as f_in, open(out_path, "wb") as f_out:
                zstandard.ZstdDecompressor().copy_stream(f_in, f_out)
        return out_path

    return path


def _read_rfd4_metrics(metrics_dir: str, include_scores: set[str]) -> pd.DataFrame:
    """
    Read and concatenate RFD4 per-example metrics CSVs.

    Parameters
    ----------
    metrics_dir : str
        ``<out_dir>/inference_metrics`` directory.
    include_scores : set of str
        Opt-in score groups.  ``"generation"`` keeps the fully resolved
        ``generation.*`` config columns RFD4 embeds in every row.

    Returns
    -------
    pandas.DataFrame
        One row per ``example_id``, with ``example_id`` renamed to
        ``description`` and ``.`` replaced by ``_`` in every metric column name.
        Empty when no metrics were written (``metrics_tags`` unset).
    """
    if not os.path.isdir(metrics_dir):
        return pd.DataFrame()

    csv_paths = sorted(
        glob(os.path.join(metrics_dir, "per_example_metrics.csv"))
        + glob(os.path.join(metrics_dir, "per_example_metrics_job*.csv"))
        + glob(os.path.join(metrics_dir, "individual_job_results", "per_example_metrics_job*.csv"))
    )
    if not csv_paths:
        return pd.DataFrame()

    frames = [pd.read_csv(path) for path in csv_paths]
    metrics = pd.concat(frames, ignore_index=True)

    if "example_id" not in metrics.columns:
        logging.warning("RFD4 metrics CSVs in %s carry no 'example_id' column; skipping metrics merge.", metrics_dir)
        return pd.DataFrame()

    metrics = metrics.drop_duplicates(subset="example_id", keep="last")

    drop = [col for col in _RFD4_METRICS_DROP_COLUMNS if col in metrics.columns]
    if "generation" not in include_scores:
        drop += [col for col in metrics.columns if col.startswith("generation.")]
    metrics = metrics.drop(columns=drop)

    metrics = metrics.rename(columns={"example_id": "description"})
    metrics.columns = [col.replace(".", "_") for col in metrics.columns]
    return metrics


def collect_scores(
    work_dir: str,
    cif_to_pdb: bool = True,
    run_clean: bool = True,
    include_scores: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse RFD4-Proteina outputs and return the canonical scores dataframe.

    Scans ``<work_dir>/outputs/inference_samples/`` for designed structures,
    decompresses any compressed outputs, optionally converts them to PDB, and
    merges the inline metrics written to
    ``<work_dir>/outputs/inference_metrics/`` on the RFD4 ``example_id``.

    Parameters
    ----------
    work_dir : str
        Root working directory of the run.  Must contain an ``outputs/``
        sub-directory populated by ``rfd4-generate``.
    cif_to_pdb : bool, optional
        When ``True`` (default), designed CIFs are converted to PDB with
        OpenBabel and ``location`` points at the PDB files.  When ``False``,
        ``location`` points at the annotated CIFs.
    run_clean : bool, optional
        When ``True`` (default), decompressed intermediates of compressed
        outputs are removed after conversion.
    include_scores : list of str, optional
        Opt-in score groups.  Pass ``"generation"`` to keep the resolved
        ``generation.*`` config columns from the metrics CSV, which are dropped
        by default because RFD4 embeds the whole generation config in every row.

    Returns
    -------
    pandas.DataFrame
        One row per designed structure with the guaranteed columns:

        ``description`` : str
            RFD4 ``example_id`` (``<design_key>_<replicate_index>``), used as
            the pose identifier in ProtFlow.
        ``location`` : str
            Absolute path to the primary output pose (PDB or CIF, depending on
            *cif_to_pdb*), written to ``<work_dir>/outputs/poses/`` under the
            name ``<description>.<ext>``.
        ``cif_location`` : str
            Absolute path to RFD4's own designed annotated CIF under
            ``inference_samples/``.
        ``task`` : str
            The design key this output came from.

        plus every inline metric column present in
        ``per_example_metrics.csv``, with ``.`` replaced by ``_``.

    Notes
    -----
    * Only ``*_generated.cif`` files are collected, so input sidecars
      (``*_input.cif``), chain-of-interest exports (``*_chain.cif``), refolds
      (``*_refolded.cif``) and trajectory dumps are never mistaken for designs.
    * ProtFlow requires ``description`` to equal the basename stem of
      ``location``, while RFD4 names its files ``<example_id>_generated.cif``.
      Poses are therefore written as normalised copies under
      ``<work_dir>/outputs/poses/``; RFD4's own output tree is left untouched,
      so ``skip_existing=true`` can still resume an interrupted run.
    * PDB conversion loses RFD4's per-atom provenance annotations
      (``SA_SRC``, ``SA_CRDSRC``, ``SA_IDXGRPSRC``, ``SA_UII``); read them from
      the path in ``cif_location``.

    Examples
    --------
    Re-parse a finished run without re-running it::

        scores = collect_scores("/scratch/rfd4_run", cif_to_pdb=False)
        print(scores[["description", "location"]].head())
    """
    from ..utils.openbabel_tools import openbabel_fileconverter  # pylint: disable=C0415

    include_set = set(include_scores or [])
    output_dir = os.path.join(work_dir, "outputs")
    samples_dir = os.path.join(output_dir, "inference_samples")
    poses_dir = os.path.join(output_dir, "poses")

    structure_paths = sorted(
        glob(os.path.join(samples_dir, "*", "*_generated.cif"))
        + glob(os.path.join(samples_dir, "*", "*_generated.cif.gz"))
        + glob(os.path.join(samples_dir, "*", "*_generated.cif.zst"))
    )
    if not structure_paths:
        return pd.DataFrame()

    os.makedirs(poses_dir, exist_ok=True)

    rows: list[dict] = []
    intermediates: list[str] = []
    for path in structure_paths:
        cif_path = _decompress_structure(path)
        if cif_path != path:
            intermediates.append(cif_path)

        description = re.sub(r"_generated$", "", os.path.basename(cif_path).removesuffix(".cif"))

        if cif_to_pdb:
            pose_path = os.path.join(poses_dir, f"{description}.pdb")
            openbabel_fileconverter(input_file=cif_path, output_format="pdb", output_file=pose_path)
        else:
            pose_path = os.path.join(poses_dir, f"{description}.cif")
            shutil.copy2(cif_path, pose_path)

        rows.append({
            "description": description,
            "location": os.path.abspath(pose_path),
            "cif_location": os.path.abspath(path),
        })

    scores = pd.DataFrame(rows)

    metrics = _read_rfd4_metrics(os.path.join(output_dir, "inference_metrics"), include_set)
    if not metrics.empty:
        scores = scores.merge(metrics, how="left", on="description")

    # 'task' is the design key; RFD4 reports it in the metrics CSV, otherwise derive it
    derived_task = scores["description"].apply(lambda desc: "_".join(desc.split("_")[:-1]))
    if "task" not in scores.columns:
        scores["task"] = derived_task
    else:
        scores["task"] = scores["task"].fillna(derived_task)

    # drop decompressed copies of compressed RFD4 outputs; the compressed originals stay
    if run_clean:
        for path in intermediates:
            if os.path.isfile(path):
                os.remove(path)

    return scores
