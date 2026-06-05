"""
hbplus.py — ProtFlow Metric Module
=====================================
 
.. module:: protflow.metrics.hbplus
   :synopsis: ProtFlow runner and query interface for HBplus, a tool for
              computing hydrogen bonds in protein structures.
 
This module provides two public classes and a set of module-level utilities
for running HBplus within ProtFlow and querying its output:
 
* :class:`HBplus_query` — a :class:`~collections.UserDict` subclass that
  builds per-pose H-bond filter specifications (target atoms/residues,
  donor/acceptor type, backbone/sidechain category, and partner selections).
* :class:`HBplus` — the ProtFlow :class:`~protflow.runners.Runner` subclass
  that executes HBplus on every pose, collects the resulting ``.hb2`` files,
  and optionally applies one or more :class:`HBplus_query` filters via
  :meth:`~HBplus.query`.
 
Workflow overview
-----------------
A typical usage pattern involves two steps:
 
1. **Run HBplus** via :meth:`HBplus.run` to generate ``.hb2`` output files
   for every input pose.
2. **Query the results** via :meth:`HBplus.query` — either in the same call
   by passing *queries* to :meth:`~HBplus.run`, or afterwards as a separate
   step.  Each query returns atom- and residue-level H-bond network metrics
   for the specified selection.
 
Query style reference
---------------------
The module-level constant :data:`HBPLUS_QUERY_STYLE` defines the allowed
keys and value types for H-bond queries:
 
.. code-block:: python
 
    HBPLUS_QUERY_STYLE = {
        "target":           (AtomSelection, ResidueSelection),
        "target_type":      ["donor", "acceptor"],
        "target_category":  ["M", "S", "H"],   # main-chain, side-chain, heteroatom
        "partner":          (AtomSelection, ResidueSelection),
        "partner_category": ["M", "S", "H"],
    }
 
``"partner_type"`` is deliberately absent: it is implicitly defined by
``"target_type"`` (the opposite role).
 
For a full description of HBplus output fields and category codes see the
`HBplus manual <https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/manual.html>`_.
 
Configuration
-------------
The runner reads its environment from the ProtFlow configuration file
(``~/.config/protflow/config.py`` by default).  The following keys are
relevant:
 
``HBPLUS_PATH``
    Absolute path to the HBplus binary.
 
``PROTFLOW_ENV``
    Path to the ProtFlow conda environment root; the Python interpreter is
    resolved as ``<PROTFLOW_ENV>/python``.
 
Dependencies
------------
* :mod:`glob`, :mod:`os`, :mod:`json` (standard library)
* `pandas <https://pandas.pydata.org/>`_
* `numpy <https://numpy.org/>`_ (:func:`~numpy.array_split`,
  :func:`~numpy.sort`)
* :mod:`collections.UserDict`
* :mod:`protflow.poses` (:class:`~protflow.poses.Poses`)
* :mod:`protflow.residues` (:class:`~protflow.residues.ResidueSelection`,
  :class:`~protflow.residues.AtomSelection`)
* :mod:`protflow.jobstarters` (:class:`~protflow.jobstarters.JobStarter`)
* :mod:`protflow.runners` (:class:`~protflow.runners.Runner`,
  :class:`~protflow.runners.RunnerOutput`)
 
Examples
--------
Run HBplus and immediately query for active-site H-bonds::
 
    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.runners.hbplus import HBplus, HBplus_query
    from protflow.residues import ResidueSelection
 
    poses = Poses("structures/", prefix="design")
    jobstarter = SbatchArrayJobstarter(max_cores=20)
 
    # Build a query: all H-bonds donated by active-site residues (side-chain only)
    query = HBplus_query(name="active_site", poses=poses)
    query.set_target(target_res=ResidueSelection("A12,A57,A102"))
    query.set_target_type("donor")
    query.set_target_category("S")
 
    runner = HBplus()
    poses = runner.run(
        poses=poses,
        prefix="hb",
        queries=[query],
        jobstarter=jobstarter,
    )
 
Run HBplus first, then query separately::
 
    runner = HBplus()
    poses = runner.run(poses=poses, prefix="hb")
 
    query = HBplus_query(name="ligand_contacts", poses=poses)
    query.set_target(target_res=ResidueSelection("A200"))
    query.set_target_category(["S", "H"])
 
    poses = runner.query(poses=poses, queries=[query], hbplus_prefix="hb")
"""

# imports
import glob
import os
import json

# dependencies
import pandas as pd
from collections import UserDict
from numpy import array_split, sort

# custom
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.residues import ResidueSelection, AtomSelection
from protflow.jobstarters import JobStarter
from protflow import require_config, load_config_path
from protflow.runners import Runner, RunnerOutput, options_flags_to_string, parse_generic_options

HBPLUS_QUERY_STYLE = {
        "target": (AtomSelection, ResidueSelection), # search for hbonds from these selections
        "target_type": ["donor", "acceptor"], # targets must be one of donor or selector (or None, if any is ok)
        "target_category": ["M", "S", "H"], # hbond must be provided by target main chain, side chain or heteroatom (for ligands)
        "partner": (AtomSelection, ResidueSelection), # search for hbonds from any target to these partners
        "partner_category": ["M", "S", "H"], # search for hbonds from target to main chain, side chain or heteroatoms
        # partner_type is redundant as it is already defined via target_type
    }
                                                                                                                                                                                                                                                                                                                                                                          
class HBplus_query(UserDict):
    """
    Builder for H-bond filter queries on HBplus output.
 
    :class:`HBplus_query` is a :class:`~collections.UserDict` subclass
    whose underlying dictionary maps each pose description to a per-pose
    query specification dict.  The query dict controls which H-bonds are
    selected when :meth:`HBplus.query` processes the ``.hb2`` output files
    produced by :meth:`HBplus.run`.
 
    Filters can target specific atoms or residues (via
    :class:`~protflow.residues.AtomSelection` or
    :class:`~protflow.residues.ResidueSelection`), restrict the role of
    those atoms (donor or acceptor), limit by bond category (main-chain
    ``"M"``, side-chain ``"S"``, or heteroatom ``"H"``), and optionally
    require that the other side of the bond also belongs to a specified
    partner selection.
 
    Each ``set_*`` method supports either a single value applied uniformly
    to all poses or a ``poses.df`` column name from which a per-pose value
    is extracted (controlled by the corresponding ``from_pose_col`` /
    ``atms_from_pose_col`` / ``res_from_pose_col`` flag).
 
    Parameters
    ----------
    name : str
        A unique, human-readable identifier for this query.  Used as the
        column-name prefix when results are merged back into ``poses.df``
        by :meth:`HBplus.query`.  All query names within a single
        :meth:`~HBplus.query` call must be distinct.
    poses : Poses
        The current pose collection.  Used to initialise one empty
        specification entry per pose and to resolve column-name lookups
        in ``poses.df``.
 
    Attributes
    ----------
    name : str
        The query identifier.
    data : dict
        The underlying query dictionary.  Keys are pose descriptions;
        values are dicts mapping filter names to their values.  Populated
        by the ``set_*`` methods and reset by :meth:`reset`.
 
    Notes
    -----
    * :class:`HBplus_query` can be passed directly to :meth:`HBplus.run`
      (via the *queries* argument) to run the query in the same pipeline
      step, or to :meth:`HBplus.query` as a standalone post-processing
      step on previously computed ``.hb2`` files.
    * When both *target_atms* and *target_res* are provided to
      :meth:`set_target` or :meth:`set_partner`, their atom selections are
      merged into a single combined :class:`~protflow.residues.AtomSelection`.
      For residue selections, a wildcard atom pattern (``".*"``) is used
      so that any atom of those residues is matched.
    * The ``"partner"`` filter requires that ``"target"`` is also set;
      querying for partners alone is not supported.
    * Category values can be combined: passing ``["S", "H"]`` to
      :meth:`set_target_category` matches both side-chain and heteroatom
      H-bonds.
 
    Examples
    --------
    Query for all donor H-bonds from a binding-site residue selection,
    restricted to side-chain atoms, targeting any partner::
 
        query = HBplus_query(name="binding_site_donors")
        query.set_target(target_res=ResidueSelection("A12,A57"))
        query.set_target_type("donor")
        query.set_target_category("S")
 
    Query using per-pose residue selections stored in a DataFrame column::
 
        query = HBplus_query(name="per_pose_target")
        query.set_target(target_res="active_site_col", res_from_pose_col=True)
        query.set_target_category(["S", "H"])
 
    Query for H-bonds between a target residue and a ligand (heteroatom)::
 
        query = HBplus_query(name="target_ligand_hbonds")
        query.set_target(target_res=ResidueSelection("A57"))
        query.set_partner(partner_res=ResidueSelection("A200"))
        query.set_partner_category("H")
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.target_atms = None
        self.target_res = None
        self.partner_atms = None
        self.partner_res = None
        self.target_type = None
        self.target_category = None
        self.partner_category = None

    def set_target(self, target_atms: AtomSelection = None, target_res: ResidueSelection = None, atms_from_pose_col: bool = False, res_from_pose_col: bool = False):
        """Set the target atom or residue selection for H-bond detection.
 
        Defines the atoms or residues that must appear on either side of a
        detected H-bond.  At least one of *target_atms* or *target_res*
        must be provided.  When both are given, their atom patterns are
        merged into a single :class:`~protflow.residues.AtomSelection`.
 
        Parameters
        ----------
        target_atms : AtomSelection, optional
            Explicit atom-level selection for the target.  When
            *atms_from_pose_col* is ``True``, this should be the name of
            a ``poses.df`` column containing :class:`~protflow.residues.AtomSelection`
            objects instead.
        target_res : ResidueSelection, optional
            Residue-level selection for the target.  Any atom of the
            selected residues will be matched (wildcard atom ``".*"``).
            When *res_from_pose_col* is ``True``, this should be the name
            of a ``poses.df`` column containing
            :class:`~protflow.residues.ResidueSelection` objects.
        atms_from_pose_col : bool, optional
            When ``True``, *target_atms* is interpreted as a
            ``poses.df`` column name; one :class:`~protflow.residues.AtomSelection`
            per pose is extracted from that column.  Default is ``False``.
        res_from_pose_col : bool, optional
            When ``True``, *target_res* is interpreted as a ``poses.df``
            column name; one :class:`~protflow.residues.ResidueSelection`
            per pose is extracted from that column.  Default is ``False``.
 
        Returns
        -------
        HBplus_query
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If neither *target_atms* nor *target_res* is provided.
        KeyError
            If a column name passed with ``*_from_pose_col=True`` is absent
            from ``poses.df`` (raised by :func:`~protflow.poses.col_in_df`).
 
        Examples
        --------
        Atom-level target::
 
            from protflow.residues import AtomSelection
            query.set_target(target_atms=AtomSelection([("A", 57, "NE2")]))
 
        Residue-level target from a DataFrame column::
 
            query.set_target(target_res="binding_site_col", res_from_pose_col=True)
        """
        self.target_atms = (target_atms, atms_from_pose_col)
        self.target_res = (target_res, res_from_pose_col)
        return self

    def set_partner(self, partner_atms: AtomSelection = None, partner_res: ResidueSelection = None, atms_from_pose_col: bool = False, res_from_pose_col: bool = False):
        """Filter for H-bonds between the target and a specific partner selection.
 
        Restricts results to H-bonds where the atom on the *other* side of the
        bond (i.e. not the target) belongs to the partner selection.  Requires
        that :meth:`set_target` has been called first.  Accepts the same
        argument formats as :meth:`set_target`.
 
        Parameters
        ----------
        partner_atms : AtomSelection, optional
            Explicit atom-level partner selection, or a ``poses.df`` column
            name when *atms_from_pose_col* is ``True``.
        partner_res : ResidueSelection, optional
            Residue-level partner selection (wildcard atom), or a
            ``poses.df`` column name when *res_from_pose_col* is ``True``.
        atms_from_pose_col : bool, optional
            Interpret *partner_atms* as a column name.  Default ``False``.
        res_from_pose_col : bool, optional
            Interpret *partner_res* as a column name.  Default ``False``.
 
        Returns
        -------
        HBplus_query
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If neither *partner_atms* nor *partner_res* is provided.
        KeyError
            If a column name passed with ``*_from_pose_col=True`` is absent
            from ``poses.df``.
 
        Examples
        --------
        ::
 
            query.set_target(target_res=ResidueSelection("A57"))
            query.set_partner(partner_res=ResidueSelection("A200"))
        """
        self.partner_atms = (partner_atms, atms_from_pose_col)
        self.partner_res = (partner_res, res_from_pose_col)
        return self

    def set_target_type(self, target_type: str, from_pose_col: bool = False):
        """Restrict the target to a specific H-bond role: donor or acceptor.
 
        Parameters
        ----------
        target_type : str
            Must be one of ``"donor"`` or ``"acceptor"``.  When
            *from_pose_col* is ``True``, this should be a ``poses.df``
            column name whose values are ``"donor"`` or ``"acceptor"``.
        from_pose_col : bool, optional
            Interpret *target_type* as a ``poses.df`` column name.
            Default is ``False``.
 
        Returns
        -------
        HBplus_query
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If *target_type* (or any per-pose value from the column) is not
            one of ``["donor", "acceptor"]``.
        KeyError
            If *from_pose_col* is ``True`` and the column is absent from
            ``poses.df``.
 
        Examples
        --------
        ::
 
            query.set_target_type("donor")
 
            # Per-pose donor/acceptor role from a DataFrame column:
            query.set_target_type("role_col", from_pose_col=True)
        """
        self.target_type = (target_type, from_pose_col)
        return
    
    def set_target_category(self, target_cat: str | list, from_pose_col: bool = False):
        """Filter for H-bonds originating from main-chain, side-chain, or heteroatoms.
 
        Parameters
        ----------
        target_cat : str or list of str
            One or more category codes: ``"M"`` (main-chain),
            ``"S"`` (side-chain), ``"H"`` (heteroatom / ligand / water).
            A list selects multiple categories simultaneously (OR logic).
            When *from_pose_col* is ``True``, this should be a
            ``poses.df`` column name whose values are strings or lists of
            the same codes.
        from_pose_col : bool, optional
            Interpret *target_cat* as a ``poses.df`` column name.
            Default is ``False``.
 
        Returns
        -------
        HBplus_query
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If any supplied category code is not in ``["M", "S", "H"]``.
 
        Examples
        --------
        ::
 
            query.set_target_category("S")          # side-chain only
            query.set_target_category(["S", "H"])   # side-chain or heteroatom
        """
        self.target_category = (target_cat, from_pose_col)
        return self

    def set_partner_category(self, partner_cat: str | list, from_pose_col: bool = False):
        """Filter for H-bonds where the partner belongs to a specific category.
 
        Mirrors :meth:`set_target_category` but applies the category
        restriction to the *partner* (i.e. the non-target side of the bond).
 
        Parameters
        ----------
        partner_cat : str or list of str
            One or more of ``"M"``, ``"S"``, ``"H"``.  When *from_pose_col*
            is ``True``, a ``poses.df`` column name.
        from_pose_col : bool, optional
            Interpret *partner_cat* as a ``poses.df`` column name.
            Default is ``False``.
 
        Returns
        -------
        HBplus_query
            ``self``, to allow method chaining.
 
        Raises
        ------
        KeyError
            If any category code is not in ``["M", "S", "H"]``.
 
        Examples
        --------
        ::
 
            query.set_partner_category("H")   # partner must be a heteroatom
        """
        self.partner_category = (partner_cat, from_pose_col)

        return self
           
    def parse_query(self, poses: Poses) -> dict:
        query_dict = {}
        for pose in poses.df["poses_description"]:
            query_dict[pose] = {}
        
        if self.target_atms or self.target_res:
            query_dict = self._set_selection(query_dict, "target", poses, self.target_atms[0], self.target_res[0], self.target_atms[1], self.target_res[1])

        if self.partner_atms or self.partner_res:
            query_dict = self._set_selection(query_dict, "partner", poses, self.partner_atms[0], self.partner_res[0], self.partner_atms[1], self.partner_res[1])

        if self.target_type:
            if self.target_type[1]:
                col_in_df(poses.df, self.target_type[0])
                target_type = poses.df[target_type[0]].to_list()
            else:
                target_type = [target_type for _ in poses.poses_list()]

            if not all(t_type in HBPLUS_QUERY_STYLE["target_type"] for t_type in target_type):
                raise KeyError(f":target_type: must be one of {HBPLUS_QUERY_STYLE['target_type']}!")
            
            query_dict = self._set_filter(query_dict, "target_type", target_type)
        
        if self.target_category:
            query_dict = self._set_category(query_dict, "target_category", poses, self.target_category[0], self.target_category[1])

        if self.partner_category:
            query_dict = self._set_category(query_dict, "partner_category", poses, self.partner_category[0], self.partner_category[1])

        return query_dict

    def _set_filter(self, query_dict: dict, name: str, values) -> dict:
        """Assign a per-pose filter value to every pose in :attr:`data`."""
        for pose, value in zip(query_dict.keys(), values):
            query_dict[pose][name] = value

        return query_dict

    def _set_selection(self, query_dict: dict, name, poses: Poses, sel_atms: AtomSelection = None, sel_res: ResidueSelection = None, atms_from_pose_col: bool = False, res_from_pose_col: bool = False):
        """Resolve atom/residue selections (optionally from DataFrame columns) and
        store them as a per-pose :class:`~protflow.residues.AtomSelection` list
        via :meth:`_set_filter`."""
        if not sel_atms and not sel_res:
            raise KeyError(f"Either :{name}_atms: or :{name}_res: must be set!")

        if sel_atms:
            if atms_from_pose_col:
                col_in_df(poses.df, sel_atms)
                sel_atms = poses.df[sel_atms].to_list()
            else:
                sel_atms = [sel_atms for _ in poses.poses_list()]

        if sel_res:
            if res_from_pose_col:
                col_in_df(poses.df, sel_res)
                sel_res = poses.df[sel_res].to_list()

            else:
                sel_res = [sel_res for _ in poses.poses_list()]

            # use any atm for resselections (ALL works as well, but then all poses have to be loaded)
            sel_res_atms = [AtomSelection.from_residueselection(res, ".*") for res in sel_res]
        
        if sel_atms and sel_res:
            sel_atms = [atms +  res_atms for atms, res_atms in zip(sel_atms, sel_res_atms)]
        elif sel_res:
            sel_atms = sel_res_atms

        return self._set_filter(query_dict, name, sel_atms)
    
    def _set_category(self, query_dict: dict, name: str, poses: Poses, cat: str | list, from_pose_col: bool = False):
        """Validate and assign a category filter (``target_category`` or
        ``partner_category``) for each pose via :meth:`_set_filter`."""
        if from_pose_col:
            col_in_df(poses.df, cat)
            cat = poses.df[cat].to_list()
        else:
            cat = [cat for _ in poses.poses_list()]

        for c in cat:
            if not all(i in HBPLUS_QUERY_STYLE[f"{name}_category"] for i in c):
                raise KeyError(f":{name}_cat: must be one of {HBPLUS_QUERY_STYLE[f'{name}_category']}!")
        
        return self._set_filter(query_dict, f"{name}_category", cat)


class HBplus(Runner):
    """
    ProtFlow runner for HBplus hydrogen-bond detection.
 
    :class:`HBplus` inherits from :class:`~protflow.runners.Runner` and
    wraps the HBplus binary to detect hydrogen bonds in all input PDB
    structures.  It manages the full execution lifecycle: option parsing,
    per-pose command construction, job submission, ``.hb2`` output
    collection, and optional downstream H-bond network analysis via
    :meth:`query`.
 
    Unlike most ProtFlow runners, :class:`HBplus` does not add any index
    layers to poses (``index_layers = 0``).  The runner adds one column to
    ``poses.df``: ``<prefix>_hb2_scores``, which contains the absolute path
    to the ``.hb2`` file for each pose.  All quantitative H-bond metrics are
    produced downstream by :meth:`query`.
 
    Because HBplus writes all output to the *current working directory*,
    the runner temporarily changes the working directory to a dedicated
    ``output/`` sub-directory before starting jobs and restores it
    afterwards.
 
    Parameters
    ----------
    hbplus_path : str, optional
        Absolute path to the HBplus binary.  Resolved from the ProtFlow
        config key ``HBPLUS_PATH`` when omitted.
    python_path : str, optional
        Python interpreter used to invoke the :func:`main` query script in
        distributed mode.  Defaults to ``<PROTFLOW_ENV>/python`` from the
        ProtFlow config.
    jobstarter : JobStarter, optional
        Default :class:`~protflow.jobstarters.JobStarter` used when
        :meth:`run` or :meth:`query` is called without an explicit
        *jobstarter* argument.
 
    Attributes
    ----------
    index_layers : int
        Class attribute; always ``0`` (HBplus does not expand the pose index).
    script_path : str
        Resolved path to the HBplus binary.
    python_path : str
        Resolved path to the Python interpreter.
    jobstarter : JobStarter or None
        Default job submission backend.
 
    Notes
    -----
    * HBplus must be run on PDB files.  CIF input is not supported.
    * The ``__str__`` method returns ``"hbplus"`` for use in log messages
      and scorefile naming.
    * :meth:`query` always overwrites its own scorefile (``overwrite=True``
      is hard-coded internally) because query results are cheap to recompute
      and depend on the query definition which may change between calls.
 
    Examples
    --------
    ::
 
        from protflow.runners.hbplus import HBplus, HBplus_query
 
        runner = HBplus()
        poses = runner.run(poses=poses, prefix="hb")
 
        query = HBplus_query(name="sc_hbonds", poses=poses)
        query.set_target_category("S")
        poses = runner.query(poses=poses, queries=[query], hbplus_prefix="hb")
    """
    index_layers = 0

    def __init__(self, hbplus_path: str = None, python_path: str = None, jobstarter: JobStarter = None):
        # setup config
        config = require_config()
        self.jobstarter = jobstarter
        self.script_path = hbplus_path or load_config_path(config, "HBPLUS_PATH")
        self.python_path = python_path or os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")

    def __str__(self):
        """
        Return the runner's display name.
 
        Returns
        -------
        str
            Always ``"hbplus"``.
        """

        return "hbplus"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, queries: list[HBplus_query] | HBplus_query = None, options: str | list = None, pose_options: str | list = None, overwrite: bool = False) -> Poses:
        """Run HBplus on all poses and optionally query the results.
 
        For each input PDB file, constructs and submits an HBplus command,
        collects the resulting ``.hb2`` output files, and records their
        paths in ``poses.df``.  If one or more :class:`HBplus_query` objects
        are provided, :meth:`query` is called automatically on the collected
        results before returning.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  All poses must be PDB files.
        prefix : str
            Column prefix used to namespace new columns in ``poses.df`` and
            to name the working directory
            (``<poses.work_dir>/<prefix>/``).  The primary output column is
            named ``<prefix>_hb2_scores``.
        jobstarter : JobStarter, optional
            Job submission backend for this run.  Resolved via the standard
            ProtFlow fallback chain: argument → ``self.jobstarter`` →
            ``poses.default_jobstarter``.
        queries : HBplus_query or list of HBplus_query, optional
            One or more query objects to apply to the HBplus output
            immediately after the run.  When provided, :meth:`query` is
            called automatically and its results are merged into
            ``poses.df`` before returning.  When ``None`` (default), only
            the ``.hb2`` file paths are recorded.
        options : str or list, optional
            Global HBplus CLI options applied to every pose, in the format
            accepted by :func:`~protflow.runners.parse_generic_options`.
        pose_options : str or list, optional
            Per-pose HBplus CLI options.  When provided, pose-level values
            take priority over *options* for the same flag.  Must be a
            string (column name) or a list of the same length as the number
            of poses.
        overwrite : bool, optional
            When ``True``, any existing scorefile is ignored and HBplus is
            re-run.  When ``False`` (default), an existing scorefile causes
            immediate return of cached results (with *queries* still
            applied if provided).
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with at minimum the
            column:
 
            ``<prefix>_hb2_scores``
                Absolute path to the HBplus ``.hb2`` output file for each
                pose.
 
            When *queries* are provided, additional columns from
            :meth:`query` are also present (see :meth:`query` → *Returns*
            for the full column listing).
 
        Raises
        ------
        RuntimeError
            Propagated from the jobstarter if one or more HBplus jobs fail.
 
        Notes
        -----
        * HBplus writes output to the *current working directory*.  The
          runner changes ``os.getcwd()`` to ``<work_dir>/output/`` before
          starting jobs and restores it immediately after, regardless of
          job success or failure.
        * One HBplus command is generated per pose; no batching is applied
          at the :meth:`run` level (batching for query post-processing is
          handled inside :meth:`query`).
        * The ``location`` column in the returned poses points to the
          original input PDB (recovered from the submitted commands), not
          to the ``.hb2`` output file.
 
        Examples
        --------
        Run HBplus only (collect paths, defer querying)::
 
            poses = runner.run(poses=poses, prefix="hb")
 
        Run HBplus and immediately apply a donor query::
 
            query = HBplus_query(name="donors", poses=poses)
            query.set_target_type("donor")
            query.set_target_category("S")
 
            poses = runner.run(
                poses=poses,
                prefix="hb",
                queries=[query],
                jobstarter=SbatchArrayJobstarter(max_cores=20),
            )
        """
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            poses = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
            if queries:
                poses = self.query(poses=poses, queries=queries, hbplus_prefix=prefix)

            return poses

        # hbplus only works on pdbs
        ext = poses.determine_pose_type("poses")
        if not ext == [".pdb"]:
            poses.convert_poses(f"{prefix}_conversion", "pdb", jobstarter=jobstarter, overwrite=overwrite)

        # prep options:
        options_l = self._prep_hbplus_options(poses, options, pose_options)

        # compile cmds
        cmds = [f"{self.script_path} {os.path.abspath(pose)} {options}" for pose, options in zip(poses.poses_list(), options_l)]

        # hbplus puts all output in current directory --> change to output dir
        os.makedirs(output_dir := os.path.join(work_dir, "output"), exist_ok=True)
        cwd = os.getcwd()
        os.chdir(output_dir)

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"hbplus_{prefix}",
            output_path = work_dir
        )

        # return to starting dir
        os.chdir(cwd)

        # collect outputs and write scorefile
        scores = collect_scores(work_dir)
        scores["location"] = [_get_hbplus_input_location(description, cmds) for description in scores["description"].to_list()]
        self.save_runner_scorefile(scores, scorefile)

        # itegrate and return
        poses = RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()

        if queries:
            poses = self.query(poses=poses, queries=queries, hbplus_prefix=prefix, jobstarter=jobstarter, overwrite=overwrite)
        return poses
    
    def query(self, poses: Poses, queries: list[HBplus_query] | HBplus_query, hbplus_prefix: str, jobstarter: JobStarter = None, full_output: bool = False, overwrite: bool = False):
        """Apply one or more H-bond queries to previously computed HBplus output.
 
        Distributes the query computation across available cores by splitting
        poses into batches, serialising each batch and the query definitions
        to JSON, and invoking this module's :func:`main` entry point in
        parallel as a sub-process.  Results from all batches are concatenated
        and merged back into ``poses.df``.
 
        Each query produces a set of columns prefixed by
        ``<hbplus_prefix>_query_<query.name>_``, reporting both direct
        H-bond metrics (the bonds matching the query filter) and extended
        network metrics (bonds reachable via side-chain / heteroatom bridges
        from the initial matches).
 
        Parameters
        ----------
        poses : Poses
            The pose collection on which to run the queries.  Must contain
            a column named ``<hbplus_prefix>_hb2_scores`` populated by a
            prior call to :meth:`run`.
        queries : HBplus_query or list of HBplus_query
            One or more query objects.  All query names must be unique within
            a single call; duplicate names raise a :exc:`KeyError`.
        hbplus_prefix : str
            The *prefix* value used in the :meth:`run` call that produced the
            ``.hb2`` files.  Used to locate the ``<hbplus_prefix>_hb2_scores``
            column and to construct the working directory path for query
            outputs (``<poses.work_dir>/<hbplus_prefix>_query/``).
        jobstarter : JobStarter, optional
            Job submission backend.  Resolved via the standard ProtFlow
            fallback chain.
        full_output : bool, optional
            When ``True``, the full filtered H-bond DataFrame (``query_full_output``)
            and the full network DataFrame (``network_full_output``) are
            included as additional columns in ``poses.df``, stored as
            dicts of dicts.  Useful for detailed inspection but increases
            memory usage.  Default is ``False``.
        overwrite : bool, optional
            Currently unused; the query scorefile is always recomputed.
            Retained for API consistency with other runners.
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns prefixed
            by ``<hbplus_prefix>_query_<query.name>_`` for each query.  The
            following sub-columns are always present:
 
            ``query_num_hbonds`` : int
                Number of unique H-bonds directly matching the query filter.
            ``query_donor_hbonded_atoms`` : AtomSelection
                Atoms acting as donors in the filtered H-bonds.
            ``query_acceptor_hbonded_atoms`` : AtomSelection
                Atoms acting as acceptors in the filtered H-bonds.
            ``query_hbonded_atoms`` : AtomSelection
                Union of donor and acceptor atoms in the filtered H-bonds.
            ``network_num_hbonds`` : int
                Total unique H-bonds in the extended H-bond network seeded
                by the query matches (propagated through side-chain and
                heteroatom bridges).
            ``network_donor_hbonded_atoms`` : AtomSelection
                All donor atoms in the extended network.
            ``network_acceptor_hbonded_atoms`` : AtomSelection
                All acceptor atoms in the extended network.
            ``network_hbonded_atoms`` : AtomSelection
                Union of all donor and acceptor atoms in the extended network.
            ``network_sc_hbond_residues`` : ResidueSelection
                Residues in the network that participate via side-chain H-bonds.
            ``network_het_hbond_residues`` : ResidueSelection
                Residues in the network that participate via heteroatom
                (water, ligand) H-bonds.
 
            When *full_output* is ``True``, two additional columns are present:
 
            ``query_full_output`` : dict
                Row-indexed dict representation of the filtered H-bond
                DataFrame.
            ``network_full_output`` : dict
                Row-indexed dict representation of the full network DataFrame.
 
        Raises
        ------
        KeyError
            If ``<hbplus_prefix>_hb2_scores`` is not present in
            ``poses.df`` (i.e. :meth:`run` has not been called with the
            specified prefix).
        ValueError
            If *queries* contains objects that are not
            :class:`HBplus_query` instances.
        KeyError
            If any two queries share the same :attr:`~HBplus_query.name`.
 
        Notes
        -----
        * The query working directory is always named
          ``<hbplus_prefix>_query`` and is created inside
          ``poses.work_dir``.  Each call produces batch JSON files
          (``batch_<i>.json``) and output JSON files (``out_<i>.json``)
          there.
        * Network detection (:func:`_detect_networks`) propagates H-bond
          networks recursively through side-chain (``"S"``) and heteroatom
          (``"H"``) bridges, stopping at main-chain (``"M"``) atoms and
          preventing propagation through water–water H-bonds to avoid
          unlimited network expansion.
        * The number of parallel batches is
          ``min(len(poses), jobstarter.max_cores)``.
 
        Examples
        --------
        Query for all H-bonds donated by active-site side-chains, plus
        the extended network they seed::
 
            query = HBplus_query(name="active_site", poses=poses)
            query.set_target(target_res=ResidueSelection("A12,A57,A102"))
            query.set_target_type("donor")
            query.set_target_category("S")
 
            poses = runner.query(
                poses=poses,
                queries=[query],
                hbplus_prefix="hb",
                full_output=True,
            )
            # poses.df["hb_query_active_site_query_num_hbonds"]  -> int per pose
            # poses.df["hb_query_active_site_network_sc_hbond_residues"]  -> ResidueSelection per pose
 
        Multiple queries in one call::
 
            q1 = HBplus_query(name="donors", poses=poses)
            q1.set_target_type("donor")
 
            q2 = HBplus_query(name="ligand_contacts", poses=poses)
            q2.set_target(target_res=ResidueSelection("A200"))
            q2.set_target_category("H")
 
            poses = runner.query(poses=poses, queries=[q1, q2], hbplus_prefix="hb")
        """

        score_col = f"{hbplus_prefix}_hb2_scores"
        if score_col not in poses.df.columns:
            raise KeyError(f"Could not find HBplus score column called {score_col} in poses dataframe! Did you run HBplus with the selected prefix?")
        
        if not isinstance(queries, list):
            queries = [queries]

        if not all(isinstance(query, HBplus_query) for query in queries):
            raise ValueError(":queries: must be a HBplus_query or a list of HBplus_queries!")
        
        if not len([query.name for query in queries]) == len(set([query.name for query in queries])):
            raise KeyError("Names of input queries must be unique!")

        prefix = f"{hbplus_prefix}_query"

        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
        
        query_dict = {}
        for query in queries:
            query_dict[query.name] = query.parse_query(poses)

        query_path = _save_dict_to_json(query_dict, os.path.join(work_dir, "queries.json"))

        n_batches = min([jobstarter.max_cores, len(poses.df.index)])

        subset = poses.df[["poses", "poses_description", f"{hbplus_prefix}_hb2_scores"]]

        # 2. Split the INDEX (integers), not the DataFrame itself
        index_chunks = array_split(subset.index, n_batches)

        cmds = []
        output = []

        for i, indices in enumerate(index_chunks):
            # 3. Use .loc to grab the rows for this chunk
            # This keeps it a DataFrame 100% of the time
            subdf = subset.loc[indices].reset_index(drop=True)
            
            json_path = os.path.join(work_dir, f"batch_{i}.json")
            out_path = os.path.join(work_dir, f"out_{i}.json")

            output.append(out_path)
            subdf.to_json(json_path)

            cmd = f"{self.python_path} {__file__} --query_path {query_path} --input_poses {json_path} --out_path {out_path}"#
            if full_output:
                cmd = cmd + " --full_output"
            cmds.append(cmd)
            

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"hbplus_{prefix}",
            output_path = work_dir
        )

        scores = pd.concat([pd.read_json(out) for out in output])
        scores.reset_index(drop=True, inplace=True)

        self.save_runner_scorefile(scores, scorefile)

        # integrate and return
        return RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()

    def _prep_hbplus_options(self, poses: Poses, options: str, pose_options: str|list[str]) -> list[str]:
        """Merge global *options* and per-pose *pose_options* into a per-pose
        options string list, with pose-level values taking priority."""

        pose_options = self.prep_pose_options(poses, pose_options)

        # Iterate through pose options, overwrite options and remove options that are not allowed.
        options_l = []
        for pose_opt in pose_options:
            opts, flags = parse_generic_options(options, pose_opt)
            options_l.append(options_flags_to_string(opts,flags))

        # merge options and pose_options, with pose_options priority and return
        return options_l

def _detect_networks(unfiltered_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Recursively extend a set of seed H-bonds into a full H-bond network.
 
    Starting from the non-main-chain atoms in *filtered_df*, searches
    *unfiltered_df* for additional H-bonds reachable via side-chain
    (``"S"``) or heteroatom (``"H"``) bridges, recursing until no new
    bonds are found.  Water–water H-bonds are excluded from network
    starting points to prevent unlimited propagation through solvent.
    Networks do not propagate through more than one consecutive water
    molecule.
 
    Parameters
    ----------
    unfiltered_df : pandas.DataFrame
        The complete set of H-bonds for this pose, as returned by
        :func:`parse_hbplus`.
    filtered_df : pandas.DataFrame
        The seed H-bonds matching a query filter, from which the network
        search begins.
 
    Returns
    -------
    pandas.DataFrame
        All H-bonds reachable from the seeds via side-chain / heteroatom
        bridges, *excluding* the seed bonds themselves (those are
        concatenated by the caller in :func:`_query_hbplus`).  Returns an
        empty DataFrame with the same columns as *unfiltered_df* when no
        network extensions are found.
 
    Notes
    -----
    The recursion terminates when ``_filter_df_by_selection`` returns no
    new bonds not already in *filtered_df*.  Main-chain atoms (``"M"``)
    act as network terminators to prevent false connectivity via
    backbone hydrogen bonds (e.g. within helices).
    """

    # create a df with hbonds not present in the filtered subset
    excluded = pd.concat([unfiltered_df, filtered_df]).drop_duplicates("bond_num", keep=False)

    # extract all possible network starting points 
    # (not 100% required as filter for cat SH is applied later, but should speed úp filtering because of fewer input residues)
    target = []
    for cat in ["A", "D"]:
        non_bb = filtered_df[filtered_df[f"{cat}_cat"] != 'M']

        # remove water-water hbonds from targets
        non_bb = non_bb[~((non_bb["A_resname"] == "HOH") & (non_bb["D_resname"] == "HOH"))]

        if non_bb.empty:
            # add empty selection
            target.append(AtomSelection(atoms=()))
            
        non_bb.to_csv("test.csv")

        non_bb_atms = AtomSelection.from_list([_convert_hbplus_to_tuple(res, atm) for res, atm in zip(non_bb[f"{cat}_res"], non_bb[f"{cat}_atom"])])
        target.append(non_bb_atms)

    # combine both selections
    target = target[0] + target[1]

    # return early if no suitable starting points were found
    if not target:
        return pd.DataFrame(columns=unfiltered_df.columns)
    
    # convert to residueselection
    target = ResidueSelection.from_atomselection(target)

    # limit cat to sidechain/hetatm to avoid bridging sidechain to mainchain (e.g. his-n - tyr-oh to tyr-n - ser-oh), which are not real networks
    network = _filter_df_by_selection(df=excluded, target=target, target_category="SH")
    if not network.empty:
        extended_nw = _detect_networks(unfiltered_df=excluded, filtered_df=network)
        network = pd.concat([network, extended_nw]).drop_duplicates()
    
    return network


def collect_scores(work_dir: str) -> pd.DataFrame:
    """Collect paths to all HBplus ``.hb2`` output files into a DataFrame.
 
    Scans ``<work_dir>/output/`` for ``.hb2`` files produced by a completed
    HBplus run and returns a DataFrame mapping pose descriptions to their
    ``.hb2`` file paths.
 
    Parameters
    ----------
    work_dir : str
        Root working directory of the HBplus run.  Must contain an
        ``output/`` sub-directory populated by the runner.
 
    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with:
 
        ``hb2_scores`` : str
            Absolute path to the ``.hb2`` file for each pose.
        ``description`` : str
            Filename stem of the ``.hb2`` file (i.e. the pose description,
            without extension), derived via
            :func:`~protflow.poses.description_from_path`.
 
    Notes
    -----
    Only paths are collected at this stage; the ``.hb2`` files are parsed
    later during :meth:`HBplus.query` via :func:`parse_hbplus`.  This
    deferred parsing avoids loading all H-bond data into memory when only
    a subset of poses will be queried.
 
    Examples
    --------
    ::
 
        scores = collect_scores("/scratch/hbplus_run")
        print(scores[["description", "hb2_scores"]].head())
    """
    # collect scores
    scores = glob.glob(os.path.join(work_dir, "output", "*.hb2"))

    # compile score paths into a dataframe
    out_df = pd.DataFrame({"hb2_scores": [os.path.abspath(score) for score in scores], "description": [description_from_path(score) for score in scores]})
    
    return out_df

def _filter_df_by_selection(
        df: pd.DataFrame,
        target: AtomSelection | ResidueSelection = None,
        target_type: str = None,
        target_category: str | list = None,
        partner: AtomSelection | ResidueSelection = None,
        partner_category: str | list = None) -> pd.DataFrame:
    """Filter an H-bond DataFrame by target/partner selections, role, and category.
 
    Converts selections to HBplus-format residue–atom regex dicts, joins
    category lists into ``"|"``-delimited regex strings, and delegates to
    :func:`_apply_filters` for each target residue–atom pair.  Partner
    filtering is applied as a second-pass filter on the target-filtered
    result.  Duplicate rows (possible when a bond matches multiple target
    residues) are dropped.
    """
    # If no target is provided, match everything (.* is the regex wildcard)
    # If a selection is provided, convert it into {res: atom_regex} dict
    target_dict = _convert_selection_to_hbplus_dict(target) if target else {".*": ".*"}
    
    # If a partner is provided, convert it; otherwise, keep it as None
    partner_dict = _convert_selection_to_hbplus_dict(partner) if partner else None

    # Process categories: join lists into "cat1|cat2" regex or default to "match all"
    t_cat = '|'.join(target_category) if target_category else ".*"
    p_cat = '|'.join(partner_category) if partner_category else ".*"

    # This list will store the result of each specific residue-atom pair search
    all_target_results = []
    
    # Iterate through each Residue:Atom pair in our target selection
    for res, atm in target_dict.items():
        
        # apply filters
        f_df = _apply_filters(df=df, res=res, atm=atm, target_type=target_type, target_category=t_cat, partner_category=p_cat)
        
        # Only check for partners if a partner dict was provided and we actually found targets
        if partner_dict and not f_df.empty:
            f_partner_list = []
            
            # Loop through partner pairs to see if they are on the other side of the interaction
            for p_res, p_atm in partner_dict.items():
                
                # Collect successful partner matches
                f_partner_list.append(_apply_filters(df=f_df, res=p_res, atm=p_atm, target_type=".*", target_category=".*", partner_category=".*"))
            
            # Combine all partner matches
            f_df = pd.concat(f_partner_list)

        # Store the final result for this target residue
        all_target_results.append(f_df)
    
    # Combine all results from the various target residues into one final DataFrame
    return pd.concat(all_target_results).drop_duplicates() # duplicates can occur if hbond to another res in target is formed

def _apply_filters(df: pd.DataFrame, res:str, atm:str, target_type:str, target_category:str, partner_category:str) -> pd.DataFrame:
    """Apply regex-based donor/acceptor role and category masks to an H-bond DataFrame.
 
    Handles three cases: target must be the donor (``target_type="donor"``),
    target must be the acceptor (``target_type="acceptor"``), or either role
    is acceptable (any other value).  Returns the filtered subset.
    """
    # Case A: Target must specifically be the Donor
    if target_type == "donor":
        mask = (df['D_resnum'].str.contains(res, na=False) &
                df['D_atom'].str.contains(atm, na=False) &
                df['D_cat'].str.contains(target_category, na=False) &
                df['A_cat'].str.contains(partner_category, na=False))

    # Case B: Target must specifically be the Acceptor
    elif target_type == "acceptor":
        mask = (df['A_resnum'].str.contains(res, na=False) &
                df['A_atom'].str.contains(atm, na=False) &
                df['A_cat'].str.contains(target_category, na=False) &
                df['D_cat'].str.contains(partner_category, na=False))

    # Case C: Target can be either the Donor OR the Acceptor
    else:
        mask = (
            # Option 1: Target is the Donor side
               (df['D_resnum'].str.contains(res, na=False) &
                df['D_atom'].str.contains(atm, na=False) &
                df['D_cat'].str.contains(target_category, na=False) &
                df['A_cat'].str.contains(partner_category, na=False)) |
            # Option 2: Target is the Acceptor side
               (df['A_resnum'].str.contains(res, na=False) &
                df['A_atom'].str.contains(atm, na=False) &
                df['A_cat'].str.contains(target_category, na=False) &
                df['D_cat'].str.contains(partner_category, na=False))
        )
        
    # Apply the mask to create a temporary subset for this specific target pair and return
    return df[mask]

def _determine_n_unique_hbonds(df: pd.DataFrame):
    """Count unique H-bonds in *df*, collapsing mirrored donor/acceptor duplicate
    rows that HBplus occasionally emits when the donor/acceptor assignment is
    ambiguous."""

    if df.empty:
        return 0
    
    df.reset_index(drop=True, inplace=True)
    df["A_temp"] = df["A_resnum"] + df["A_atom"]
    df["D_temp"] = df["D_resnum"] + df["D_atom"]

    nonunique = len(df)

    # 1. Create a sorted version of the two columns
    # This puts both (A, B) and (B, A) into the same order: (A, B)
    temp_df = pd.DataFrame(sort(df[['A_temp', 'D_temp']], axis=1))

    # 2. Identify duplicates in the sorted data
    mirrored_mask = temp_df.duplicated(keep=False)

    # 3. View the mirrored rows
    mirrored_rows = df[mirrored_mask]
    n_mirrored = len(mirrored_rows)

    return int(nonunique - (n_mirrored / 2))
    
def _convert_hbplus_to_tuple(resname:str, atom:str) -> AtomSelection:
    """Parse an HBplus residue string (e.g. ``"A0057-HIS"``) and an atom name
    into an ``(chain, resnum, atom)`` tuple suitable for
    :class:`~protflow.residues.AtomSelection`."""
    chain_resnum = resname.split("-")[0]
    chain = chain_resnum[0]
    resnum = int(chain_resnum[1:])

    return (chain, resnum, atom)

def _convert_selection_to_hbplus_dict(selection: AtomSelection | ResidueSelection) -> dict:
    """Convert an :class:`~protflow.residues.AtomSelection` or
    :class:`~protflow.residues.ResidueSelection` to a dict keyed by HBplus
    residue IDs (e.g. ``"A0057"``) with atom-name regex strings as values."""
    hbplus_dict = {}
    if isinstance(selection, AtomSelection):
        selection = selection.to_rfd3_dict()
        for key in selection:
            hbplus_dict[f"{key[0]}{str(key[1:]).zfill(4)}"] = "|".join(selection[key].split(",")) # rename residue to hbplus format, join all selection atoms
    elif isinstance(selection, ResidueSelection):
        hbplus_dict = {f"{res[0]}{str(res[1:]).zfill(4)}": ".*" for res in selection.to_list()}
    else:
        raise KeyError("Input must be Atom or ResidueSelection!")
    
    return hbplus_dict

def _get_hbplus_input_location(description: str, cmds: list[str]) -> str:
    """Recover the absolute input PDB path for a given pose description by
    scanning the list of submitted HBplus commands for the matching filename."""

    # first get the cmd that contains 'description'
    cmd = [cmd for cmd in cmds if f"/{description}.pdb" in cmd][0]

    # extract location of input pdb:
    return [substr for substr in cmd.split(" ") if f"/{description}.pdb" in substr][0]

def parse_hbplus(file_path) -> pd.DataFrame:
    """Parse an HBplus ``.hb2`` output file into a structured DataFrame.
 
    Reads the fixed-width-format ``.hb2`` file produced by HBplus,
    applies the exact column-position specifications defined in the
    `HBplus manual <https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/manual.html>`_,
    and returns a tidy DataFrame with one row per detected H-bond.
 
    Parameters
    ----------
    file_path : str
        Absolute or resolvable path to a ``.hb2`` HBplus output file.
 
    Returns
    -------
    pandas.DataFrame
        One row per H-bond entry with the following columns:
 
        ``D_res`` : str
            Full donor residue string in HBplus format (e.g. ``"A0057-HIS"``).
        ``D_resnum`` : str
            Donor chain + residue number (e.g. ``"A0057"``), split from
            ``D_res`` at the first ``"-"``.
        ``D_resname`` : str
            Donor residue name (e.g. ``"HIS"``), split from ``D_res``.
        ``D_atom`` : str
            Donor atom name (e.g. ``"NE2"``).
        ``D_cat`` : str
            Donor category: ``"M"`` (main-chain), ``"S"`` (side-chain),
            or ``"H"`` (heteroatom); first character of ``DA_cat``.
        ``A_res`` : str
            Full acceptor residue string.
        ``A_resnum`` : str
            Acceptor chain + residue number.
        ``A_resname`` : str
            Acceptor residue name.
        ``A_atom`` : str
            Acceptor atom name.
        ``A_cat`` : str
            Acceptor category; second character of ``DA_cat``.
        ``DA_dist`` : float
            Donor-to-acceptor distance (Å).
        ``DA_cat`` : str
            Two-character category string (donor category + acceptor category).
        ``res_sep`` : float
            Sequence separation between donor and acceptor residues.
        ``CA_CA_dist`` : float
            Cα–Cα distance between donor and acceptor residues (Å).
        ``DHA_angle`` : float
            D–H···A angle (degrees).
        ``HA_dist`` : float
            H-to-acceptor distance (Å).
        ``HAAA_angle`` : float
            H···A–AA angle (degrees).
        ``DAAA_angle`` : float
            D···A–AA angle (degrees).
        ``bond_num`` : float
            HBplus internal bond index number.
 
    Notes
    -----
    * The file is read with :func:`pandas.read_fwf` using hard-coded
      character-position column specifications (``colspecs``), skipping the
      first 8 header lines.  All columns are initially read as strings to
      prevent dtype-inference errors on one- and two-letter ligand residue
      names.
    * All string columns are stripped of surrounding whitespace after
      parsing.
    * Numeric columns are cast to ``float`` after stripping.
    * HBplus uses one- or two-letter residue name codes for small-molecule
      ligands, which prevents whitespace-based parsing; hence fixed-width
      format is mandatory.
 
    Examples
    --------
    ::
 
        df = parse_hbplus("/scratch/hbplus_run/output/pose_001.hb2")
        print(df[["D_resnum", "D_atom", "A_resnum", "A_atom", "DA_dist"]].head())
    """
    # Define exact character column ranges based on HBPLUS .hb2 standard
    # These indices represent (start, end) character positions
    col_specs = [
        (0, 9),   # D_res (9 chars)
        (10, 13),  # D_atom (3 chars)
        (14, 23), # A_res (9 chars)
        (24, 27), # A_atom (3 chars)
        (28, 32), # DA_dist (3 chars)
        (33, 35), # DA_cat (2 chars)
        (36, 39), # res_sep (3 chars)
        (40, 45), # CA_CA_dist (5 chars)
        (46, 51), # DHA_angle (5 chars)
        (52, 57), # HA_dist (5 chars)
        (58, 63), # HAAA_angle (5 chars)
        (64, 69), # DAAA_angle (5 chars)
        (70, 75)  # bond_num (5 chars)
    ]
    
    columns = [
        "D_res", "D_atom", "A_res", "A_atom", 
        "DA_dist", "DA_cat", "res_sep", "CA_CA_dist", 
        "DHA_angle", "HA_dist", "HAAA_angle", "DAAA_angle", "bond_num"
    ]

    # Read using Fixed-Width Format (cannot read with whitespace separation because of one or two letter code names for ligands)
    df = pd.read_fwf(
        file_path,
        colspecs=col_specs,
        names=columns,
        skiprows=8,
        dtype=str
    )

    # Split D_res into two temporary columns on the first dash
    d_split = df["D_res"].str.split("-", n=1, expand=True)
    df["D_resnum"] = d_split[0]
    df["D_resname"] = d_split[1]

    # Split A_res into two temporary columns on the first dash
    a_split = df["A_res"].str.split("-", n=1, expand=True)
    df["A_resnum"] = a_split[0]
    df["A_resname"] = a_split[1]

    # split into separate cols
    df["D_cat"] = df["DA_cat"].str[0]
    df["A_cat"] = df["DA_cat"].str[1]

    # Clean up whitespace left by FWF
    for col in df:
        df[col] = df[col].str.strip()

    # convert cols to float
    for col in ["DA_dist", "res_sep", "CA_CA_dist", "DHA_angle", "HA_dist", "HAAA_angle", "DAAA_angle", "bond_num"]:
        df[col] = df[col].astype(float)

    return df


def _save_dict_to_json(data, filename):
    """Serialise *data* to a JSON file at *filename* with ``indent=4`` and return the path."""

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=list)
    return filename

def _load_dict_from_json(filename) -> dict:
    """Read a JSON file and return its contents as a Python dictionary."""

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def _atomselection_from_df(df: pd.DataFrame, res_col: str, atom_col: str) -> AtomSelection:
    """Build a deduplicated :class:`~protflow.residues.AtomSelection` from two
    DataFrame columns containing HBplus residue IDs and atom names."""
    atms = []

    for _, row in df.iterrows():
        atms.append(_convert_hbplus_to_tuple(row[res_col], row[atom_col]))

    return AtomSelection.from_list(atms).deduplicate()

def _query_hbplus(path: str, query: dict, full_output: bool = False) -> dict:
    """Run a single cleaned query dict against one ``.hb2`` file.
 
    Calls :func:`parse_hbplus`, applies :func:`_filter_df_by_selection`,
    runs :func:`_detect_networks` to extend matches into a full network,
    and aggregates the results into a dictionary of atom selections,
    residue selections, and H-bond counts — covering both the direct query
    matches and the extended network.
    """
    df = parse_hbplus(path)
    results = {}

    f_df = _filter_df_by_selection(df, **query).drop_duplicates(["bond_num"])
    #f_df.reset_index(drop=True, inplace=True)

    results["query_num_hbonds"] = _determine_n_unique_hbonds(f_df) # total number of hbonds according to query criteria
    results["query_donor_hbonded_atoms"] = _atomselection_from_df(f_df, "D_res", "D_atom")
    results["query_acceptor_hbonded_atoms"] = _atomselection_from_df(f_df, "A_res", "A_atom")
    results["query_hbonded_atoms"] = (results["query_donor_hbonded_atoms"] + results["query_acceptor_hbonded_atoms"]).deduplicate()

    networks = _detect_networks(df, f_df)
    # add starting points to networks
    networks = pd.concat([f_df, networks]).drop_duplicates(["bond_num"])
    networks.reset_index(drop=True, inplace=True)

    results["network_num_hbonds"] = _determine_n_unique_hbonds(networks)
    results["network_donor_hbonded_atoms"] = _atomselection_from_df(networks, "D_res", "D_atom")
    results["network_acceptor_hbonded_atoms"] = _atomselection_from_df(networks, "A_res", "A_atom")
    results["network_hbonded_atoms"] = (results["network_donor_hbonded_atoms"] + results["network_acceptor_hbonded_atoms"]).deduplicate()

    # extract all residues in network that interact via sidechain residues
    sc_donor = _apply_filters(df=networks, res=".*", atm=".*", target_type="donor", target_category="S", partner_category=".*")
    sc_acceptor = _apply_filters(df=networks, res=".*", atm=".*", target_type="acceptor", target_category="S", partner_category=".*")
    # create resisdueselection
    results["network_sc_hbond_residues"] = ResidueSelection.from_atomselection(_atomselection_from_df(sc_donor, "D_res", "D_atom") + _atomselection_from_df(sc_acceptor, "A_res", "A_atom"))
    
    # extract all residues in network that interact via heteroatoms (waters, ligands)
    sc_donor = _apply_filters(df=networks, res=".*", atm=".*", target_type="donor", target_category="H", partner_category=".*")
    sc_acceptor = _apply_filters(df=networks, res=".*", atm=".*", target_type="acceptor", target_category="H", partner_category=".*")
    # create resisdueselection
    results["network_het_hbond_residues"] = ResidueSelection.from_atomselection(_atomselection_from_df(sc_donor, "D_res", "D_atom") + _atomselection_from_df(sc_acceptor, "A_res", "A_atom"))

    if full_output:
        results["network_full_output"] = networks.to_dict("index")
        results["query_full_output"] = f_df.to_dict("index")

    return results

def _clean_per_pose_query(query: dict) -> dict:
    """Reconstruct :class:`~protflow.residues.AtomSelection` objects from serialised
    query dicts loaded from JSON, validate the result with :func:`_check_query`,
    and fill any missing optional keys with ``None``."""

    if "target" in query:
        query["target"] = AtomSelection(query["target"])

    if "partner" in query:
        query["partner"] = AtomSelection(query["partner"])

    _check_query(query)

    for key in HBPLUS_QUERY_STYLE:
        query.setdefault(key, None)
    
    return query

def _check_query(query: dict):
    """Validate a query dictionary against :data:`HBPLUS_QUERY_STYLE`, raising
    descriptive errors for unknown keys, wrong types, missing mandatory fields,
    and invalid category or type values."""    
    if isinstance(query, dict):
        if not all(key in HBPLUS_QUERY_STYLE for key in query.keys()):
            raise KeyError(f"Only these keys are allowed: {HBPLUS_QUERY_STYLE}")
        
        #if "separation" in query and not isinstance(query["separation"], HBPLUS_QUERY_STYLE["separation"]):
        #    raise ValueError("separation must be of type int or list")

        for key in ["target", "partner"]:
            if key in query and not isinstance(query[key], HBPLUS_QUERY_STYLE[key]):
                raise ValueError(f"{key} must be of type AtomSelection!")
            
        for key in ["target_category", "partner_category"]:
            if key in query and not all(cat in HBPLUS_QUERY_STYLE[key] for cat in query[key]):
                raise ValueError(f"{key} must be one (or more) of {HBPLUS_QUERY_STYLE[key]}")
        
        if "partner" in query and "target" not in query:
            raise KeyError(f"target is mandatory if setting {key}!")
            
        if "target_type" in query and query["target_type"] not in HBPLUS_QUERY_STYLE["target_type"]:
            raise KeyError(f"target_type must be one of {HBPLUS_QUERY_STYLE['target_type']}!")

    else:
        raise ValueError(f"Input must be of type dict, not {type(query)}!")

def _get_pose_metrics(row: pd.Series, query_dict: dict, score_path_col: str, name:str, full_output:bool=False):
    """Extract the per-pose query dict, run :func:`_query_hbplus`, and return
    results as a flat dict with keys prefixed by *name*."""
    desc = row["poses_description"]
    # Fetch and clean the specific query for this pose
    pose_query = _clean_per_pose_query(query_dict[desc])
    # Run the heavy calculation
    pose_results = _query_hbplus(row[score_path_col], pose_query, full_output)
    
    # Return a dictionary with prefixed keys
    return {f"{name}_{k}": v for k, v in pose_results.items()}

def main(args):
    """Entry point for distributed HBplus query execution.
 
    Loads the query definitions and a batch of poses from JSON files,
    applies each query via :func:`_get_pose_metrics`, and writes the
    combined results to an output JSON file.  Called as a sub-process by
    :meth:`HBplus.query` for each parallel batch.
 
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with the following attributes:
 
        ``query_path`` : str
            Path to the query JSON file written by :meth:`HBplus.query`
            (contains the serialised :class:`HBplus_query` data dict).
        ``input_poses`` : str
            Path to a batch JSON file (a serialised subset of ``poses.df``,
            containing at minimum ``poses``, ``poses_description``, and the
            ``<prefix>_hb2_scores`` column).
        ``out_path`` : str
            Destination path for the output JSON file (the poses subset
            augmented with per-query result columns).
        ``full_output`` : bool
            When ``True``, full H-bond DataFrames are included in the
            output (see :meth:`HBplus.query` → *full_output*).
 
    Returns
    -------
    None
        Results are written to ``args.out_path`` as a JSON file via
        :meth:`~pandas.DataFrame.to_json`.
 
    Notes
    -----
    * The ``<prefix>_hb2_scores`` column is identified dynamically by
      finding the first column whose name ends with ``"_hb2_scores"``.
    * This function is designed to be invoked via the command line using
      the ``__main__`` block at the bottom of the module; it is not
      typically called directly from Python.
    * Output columns are renamed for compatibility with
      :class:`~protflow.runners.RunnerOutput`: ``poses_description`` →
      ``description``, ``poses`` → ``location``.
    """
    queries = _load_dict_from_json(args.query_path)

    poses = pd.read_json(args.input_poses)

    score_path_col = [col for col in poses.columns if col.endswith("_hb2_scores")][0]

    all_results = []

    for name, query_dict in queries.items():
    # Pass the loop-dependent variables through the 'args' tuple
        query_df = poses.apply(
            _get_pose_metrics,
            axis=1,
            args=(query_dict, score_path_col, name, args.full_output)
        )
        # convert result to Series to expand dictionary into columns
        all_results.append(pd.DataFrame(query_df.tolist(), index=query_df.index))

    # combine everything side-by-side in one shot
    poses = pd.concat([poses] + all_results, axis=1)

    # clean up names and columns
    poses = poses.drop(columns=[score_path_col]).rename(columns={
        "poses_description": "description",
        "poses": "location"
    })

    poses.to_json(args.out_path)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--query_path", type=str, required=True, help="path to input query json")
    argparser.add_argument("--input_poses", type=str, required=True, help="path to input poses json")
    argparser.add_argument("--out_path", type=str, required=True, help="output filename")
    argparser.add_argument("--full_output", action='store_true', help="include full data in output.")

    arguments = argparser.parse_args()
    main(arguments)
