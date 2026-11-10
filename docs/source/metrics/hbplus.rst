.. _metrics.hbplus:

HBPlus
======

Overview
--------

`HBPlus <https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/>`_ detects
hydrogen bonds in protein structures. ProtFlow's
:py:class:`~protflow.metrics.hbplus.HBplus` runner separates the workflow
into two steps:

1. Run HBPlus for every pose and collect the generated ``.hb2`` files.
2. Apply named :py:class:`~protflow.metrics.hbplus.HBplus_query` objects to
   select H-bonds and summarize their extended H-bond networks.

This separation lets multiple queries reuse the same HBPlus calculation.
Queries can use a fixed :py:class:`~protflow.residues.ResidueSelection` or
:py:class:`~protflow.residues.AtomSelection`, or read a different selection
from each row of ``Poses.df``.

Installation and configuration
------------------------------

Install HBPlus following its upstream instructions, then set the binary path
in your ProtFlow configuration file:

.. code-block:: python
   :name: config-excerpt-hbplus

   # path to the HBPlus executable
   HBPLUS_PATH = "/path/to/hbplus/hbplus"

The query step runs with the Python interpreter at ``<PROTFLOW_ENV>/python``.
Both ``HBPLUS_PATH`` and ``PROTFLOW_ENV`` must therefore be configured.

To display the active ProtFlow configuration file, run:

.. code-block:: bash

   protflow-check-config

Quick start
-----------

The following example finds H-bonds donated by either of two active-site
residues:

.. code-block:: python

   from protflow.jobstarters import LocalJobStarter
   from protflow.metrics import HBplus, HBplus_query
   from protflow.poses import Poses
   from protflow.residues import ResidueSelection

   jobstarter = LocalJobStarter(max_cores=4)

   poses = Poses(
       poses="/path/to/input_pdbs/",
       glob_suffix="*.pdb",
       work_dir="/path/to/output_dir/",
       jobstarter=jobstarter,
   )

   query = HBplus_query(name="active_site_donors")
   query.set_target(target_res=ResidueSelection(["A57", "A102"]))
   query.set_target_type("donor")

   runner = HBplus()
   poses = runner.run(
       poses=poses,
       prefix="hb",
       queries=query,
   )

   print(poses.df[[
       "poses_description",
       "hb_hb2_scores",
       "hb_query_active_site_donors_query_num_hbonds",
   ]])

Running and querying separately
--------------------------------

Passing a query to :meth:`HBplus.run
<protflow.metrics.hbplus.HBplus.run>` is convenient for a single analysis.
For multiple analyses, run HBPlus first and query the collected ``.hb2`` files
afterwards:

.. code-block:: python

   poses = runner.run(poses=poses, prefix="hb")

   query = HBplus_query(name="active_site")
   query.set_target(target_res=ResidueSelection(["A57", "A102"]))

   poses = runner.query(
       poses=poses,
       queries=query,
       hbplus_prefix="hb",
       full_output=True,
   )

``hbplus_prefix`` must match the prefix of the earlier ``run()`` call. A
single ``query()`` call can accept multiple queries, but their names must be
unique.

Defining selections
-------------------

Residue selections
^^^^^^^^^^^^^^^^^^

Use ``target_res`` when any atom in the selected residues may participate in
the H-bond:

.. code-block:: python

   query = HBplus_query(name="site_hbonds")
   query.set_target(target_res=ResidueSelection(["A57", "A102"]))

This finds H-bonds for which at least one endpoint belongs to the selection.

Atom selections
^^^^^^^^^^^^^^^

Use ``target_atms`` to restrict matches to particular atoms:

.. code-block:: python

   from protflow.residues import AtomSelection

   catalytic_atoms = AtomSelection([
       ("A", 57, "NE2"),
       ("A", 102, "OD1"),
   ])

   query = HBplus_query(name="catalytic_atoms")
   query.set_target(target_atms=catalytic_atoms)

HBPlus queries currently expect compact atom identifiers of the form
``(chain, integer_residue_number, atom_name)``.

Pose-specific selections
^^^^^^^^^^^^^^^^^^^^^^^^

If each pose has a different active site, store one selection object per row
and pass the column name:

.. code-block:: python

   poses.df["active_site"] = [
       ResidueSelection(["A57", "A102"]),
       ResidueSelection(["A61", "A106"]),
   ]

   query = HBplus_query(name="per_pose_site")
   query.set_target(
       target_res="active_site",
       res_from_pose_col=True,
   )

For atom selections, use ``target_atms="column_name"`` together with
``atms_from_pose_col=True``.

Donor and acceptor roles
^^^^^^^^^^^^^^^^^^^^^^^^

By default, a target may occur on either side of an H-bond. Use
``set_target_type()`` to require that it acts as the donor or acceptor:

.. code-block:: python

   query.set_target_type("donor")
   # or
   query.set_target_type("acceptor")

The role can also come from a pose column:

.. code-block:: python

   query.set_target_type("target_role", from_pose_col=True)

Target and partner selections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a partner selection to find H-bonds between two distinct selections:

.. code-block:: python

   query = HBplus_query(name="site_to_ligand")
   query.set_target(target_res=ResidueSelection(["A57", "A102"]))
   query.set_partner(partner_res=ResidueSelection(["Z1"]))

.. warning::

   The current partner filter verifies that both selections occur in the
   matched bond, but it does not track which endpoint originally matched the
   target. This works for disjoint target and partner selections. Do not use
   the same or overlapping selection for both arguments to calculate strictly
   internal H-bonds.

Category filters
^^^^^^^^^^^^^^^^

HBPlus classifies each endpoint as main chain (``M``), side chain (``S``), or
heteroatom (``H``). ``HBplus_query`` exposes ``set_target_category()`` and
``set_partner_category()`` for these fields.

.. warning::

   Category-filter parsing is currently unavailable in this runner. Calls to
   the category setters can be stored on a query but fail when that query is
   parsed. Use the full output described below for manual category filtering
   until category query support is repaired.

Internal H-bonds within one selection
-------------------------------------

Until overlapping target and partner selections are supported, parse the
``.hb2`` output and require both endpoints to belong to the selection:

.. code-block:: python

   from protflow.metrics.hbplus import parse_hbplus

   site = ResidueSelection(["A57", "A102"])
   selected = {
       f"{chain}{residue_number:04d}"
       for chain, residue_number in site
   }

   hbonds = parse_hbplus(poses.df.loc[0, "hb_hb2_scores"])
   internal_hbonds = hbonds[
       hbonds["D_resnum"].isin(selected)
       & hbonds["A_resnum"].isin(selected)
   ]

For an ``AtomSelection``, apply the same logic to the ``D_resnum`` /
``D_atom`` and ``A_resnum`` / ``A_atom`` pairs.

Outputs
-------

Runner outputs
^^^^^^^^^^^^^^

``HBplus.run(poses, prefix="hb")`` adds these columns:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``hb_hb2_scores``
     - Absolute path to the pose's HBPlus ``.hb2`` file.
   * - ``hb_location``
     - Input PDB passed to HBPlus.
   * - ``hb_description``
     - Description used to merge the result into ``Poses.df``.

The active pose remains the input PDB; the ``.hb2`` file is metadata rather
than a replacement structure.

Query outputs
^^^^^^^^^^^^^

Each query adds columns beginning with
``{hbplus_prefix}_query_{query_name}_``:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Suffix
     - Description
   * - ``query_num_hbonds``
     - Number of unique H-bonds directly matching the query.
   * - ``query_donor_hbonded_atoms``
     - Donor atoms in direct matches.
   * - ``query_acceptor_hbonded_atoms``
     - Acceptor atoms in direct matches.
   * - ``query_hbonded_atoms``
     - Union of direct donor and acceptor atoms.
   * - ``network_num_hbonds``
     - Number of bonds in the direct and extended network.
   * - ``network_donor_hbonded_atoms``
     - Donor atoms in the full network.
   * - ``network_acceptor_hbonded_atoms``
     - Acceptor atoms in the full network.
   * - ``network_hbonded_atoms``
     - Union of donor and acceptor atoms in the network.
   * - ``network_sc_hbond_residues``
     - Network residues participating through side-chain atoms.
   * - ``network_het_hbond_residues``
     - Network residues participating as heteroatoms.

Atom selections are stored in scorefile form as ``{"atoms": [...]}``, and
residue selections as ``{"residues": [...]}``. Reconstruct objects when
needed:

.. code-block:: python

   atoms = AtomSelection(poses.df.loc[0, atom_output_column])
   residues = ResidueSelection(
       poses.df.loc[0, residue_output_column],
       from_scorefile=True,
   )

Full H-bond records
^^^^^^^^^^^^^^^^^^^

Passing ``full_output=True`` to ``query()`` adds ``query_full_output`` and
``network_full_output`` columns. Each value is a row-indexed dictionary with
one record per H-bond. Records contain:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Fields
     - Meaning
   * - ``D_res``, ``D_resnum``, ``D_resname``, ``D_atom``, ``D_cat``
     - Donor residue, atom, and category.
   * - ``A_res``, ``A_resnum``, ``A_resname``, ``A_atom``, ``A_cat``
     - Acceptor residue, atom, and category.
   * - ``DA_dist``, ``HA_dist``, ``CA_CA_dist``
     - Donor-acceptor, hydrogen-acceptor, and C-alpha distances in Angstrom.
   * - ``DHA_angle``, ``HAAA_angle``, ``DAAA_angle``
     - HBPlus geometry angles in degrees.
   * - ``DA_cat``, ``res_sep``, ``bond_num``
     - Combined categories, sequence separation, and HBPlus bond number.

H-bond networks
---------------

The direct query output contains only bonds satisfying the query filters. The
network output additionally follows connections through side-chain and
heteroatom endpoints. Main-chain endpoints terminate network expansion, and
water-water bonds are not used as expansion starting points. The network count
includes the direct query bonds.

Caching and files
-----------------

HBPlus writes run files below ``<poses.work_dir>/<prefix>/`` and query files
below ``<poses.work_dir>/<hbplus_prefix>_query/``. With ``overwrite=False``,
both ``run()`` and ``query()`` reuse an existing scorefile.

.. important::

   The query cache is shared by all query definitions using the same
   ``hbplus_prefix``. Pass ``overwrite=True`` after changing query names,
   selections, roles, or other filters.

API reference
-------------

.. autoclass:: protflow.metrics.hbplus.HBplus
   :members: run, query
   :no-index:

.. autoclass:: protflow.metrics.hbplus.HBplus_query
   :members:
   :no-index:

.. autofunction:: protflow.metrics.hbplus.parse_hbplus
   :no-index:
