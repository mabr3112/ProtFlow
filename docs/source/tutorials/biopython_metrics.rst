.. _biopython_metrics_tutorial:

Calculate BioPython Geometry Metrics
====================================

Use :class:`protflow.metrics.biopython_metrics.BiopythonMetricRunner` when you
want to calculate simple atom-level geometry metrics from BioPython structures.
The runner can calculate several metrics for every pose in one ``run()`` call,
including distances, angles, dihedrals, and plane angles.

This is useful when you want to track geometry such as:

- the distance between two atoms
- the distance from one atom to a backbone axis
- the distance from one atom to a plane
- a bond angle or vector angle
- a dihedral angle
- the angle between two atom-defined planes

The basic idea
--------------

Each metric object describes one score column. For example,
``Distance(name="n_ca_distance", atoms=...)`` will create a score named
``<prefix>_n_ca_distance`` after the runner merges results back into
``poses.df``.

Atom selections are ordered. This is important because metrics interpret atoms
by position. A compact atom ID has the format:

.. code-block:: python

   ("A", 10, "CA")

This means chain ``A``, residue ``10``, atom ``CA``. You can also use full
BioPython-style atom IDs when you need model IDs, hetero residue IDs, or altloc
selection.

Setup
-----

.. code-block:: python

   from protflow.poses import Poses
   from protflow.jobstarters import LocalJobStarter
   from protflow.residues import AtomSelection
   from protflow.metrics.biopython_metrics import (
       Angle,
       BiopythonMetricRunner,
       Dihedral,
       Distance,
       PlaneAngle,
   )

   jobstarter = LocalJobStarter(max_cores=4)

   poses = Poses(
       poses="data/input_pdbs",
       glob_suffix="*.pdb",
       work_dir="biopython_metrics_example",
       jobstarter=jobstarter,
   )

The runner uses the Python interpreter from ``PROTFLOW_ENV`` and the auxiliary
script directory from ``AUXILIARY_RUNNER_SCRIPTS_DIR`` in the ProtFlow config.

Store atom selections in ``poses.df``
-------------------------------------

You can pass a fixed atom selection directly to a metric, or you can store a
different atom selection for each pose in ``poses.df`` and pass the column name
to the metric.

The example below creates several atom-selection columns. The selections are
the same for every pose here, but in real workflows these lists can be
different for every row.

.. code-block:: python

   # Two atoms: point-to-point distance.
   poses.df["n_ca_atoms"] = [
       AtomSelection((("A", 1, "N"), ("A", 1, "CA")))
       for _ in poses
   ]

   # Three atoms: distance from the first atom to the line through atoms 2 and 3.
   poses.df["point_to_axis_atoms"] = [
       AtomSelection((("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C")))
       for _ in poses
   ]

   # Four atoms: distance from atom 1 to the plane through atoms 2, 3, and 4.
   poses.df["point_to_plane_atoms"] = [
       AtomSelection((("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C"), ("A", 1, "O")))
       for _ in poses
   ]

   # Four atoms: angle or dihedral example.
   poses.df["n_ca_c_o_atoms"] = [
       AtomSelection((("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C"), ("A", 1, "O")))
       for _ in poses
   ]

   # Six atoms: two planes, each defined by three atoms.
   poses.df["plane_angle_atoms"] = [
       AtomSelection((
           ("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C"),
           ("A", 2, "N"), ("A", 2, "CA"), ("A", 2, "C"),
       ))
       for _ in poses
   ]

When a metric receives ``atoms="n_ca_atoms"``, the runner looks up the
``n_ca_atoms`` column for each pose row and sends that row-specific selection
to the worker script.

Calculate several metrics in one run
------------------------------------

The next block calculates multiple scores in a single ``run()`` call. It
includes three different distance metrics, plus angle, dihedral, and plane-angle
metrics.

.. code-block:: python

   metrics = [
       Distance(
           name="n_ca_distance",
           atoms="n_ca_atoms",
       ),
       Distance(
           name="n_to_ca_c_axis_distance",
           atoms="point_to_axis_atoms",
       ),
       Distance(
           name="n_to_ca_c_o_plane_distance",
           atoms="point_to_plane_atoms",
           distance_type="point_plane",
       ),
       Angle(
           name="n_ca_c_angle",
           atoms=(("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C")),
       ),
       Dihedral(
           name="n_ca_c_o_dihedral",
           atoms="n_ca_c_o_atoms",
       ),
       PlaneAngle(
           name="residue_1_2_plane_angle",
           atoms="plane_angle_atoms",
       ),
   ]

   biopython_metrics = BiopythonMetricRunner()

   poses = biopython_metrics.run(
       poses=poses,
       prefix="bio_geom",
       jobstarter=jobstarter,
       metrics=metrics,
       overwrite=True,
   )

   print(
       poses.df[
           [
               "poses_description",
               "bio_geom_n_ca_distance",
               "bio_geom_n_to_ca_c_axis_distance",
               "bio_geom_n_to_ca_c_o_plane_distance",
               "bio_geom_n_ca_c_angle",
               "bio_geom_n_ca_c_o_dihedral",
               "bio_geom_residue_1_2_plane_angle",
           ]
       ]
   )

What the metrics mean
---------------------

``Distance`` supports several atom counts:

- 2 atoms: point-to-point distance
- 3 atoms: distance from atom 1 to the line through atoms 2 and 3
- 4 atoms with ``distance_type="vector_vector"``: distance between two lines
- 4 atoms with ``distance_type="point_plane"``: distance from atom 1 to a plane

Four-atom distances are ambiguous, so specify ``distance_type`` explicitly.

``Angle`` supports:

- 3 atoms: angle formed by atoms 1-2-3
- 4 atoms: angle between vectors atom 1 -> atom 2 and atom 3 -> atom 4

``Dihedral`` expects 4 atoms and returns a signed dihedral angle.

``PlaneAngle`` expects 6 atoms. Atoms 1-3 define the first plane, and atoms 4-6
define the second plane. By default, the metric returns the acute angle between
the two planes.

Results and caching
-------------------

The runner writes a scorefile into ``<poses.work_dir>/<prefix>`` and merges the
results into ``poses.df``. Every score column is prefixed with the run prefix.

For the example above, the output columns include:

.. code-block:: text

   bio_geom_n_ca_distance
   bio_geom_n_to_ca_c_axis_distance
   bio_geom_n_to_ca_c_o_plane_distance
   bio_geom_n_ca_c_angle
   bio_geom_n_ca_c_o_dihedral
   bio_geom_residue_1_2_plane_angle

If you run the same prefix again with ``overwrite=False``, ProtFlow reuses the
cached scorefile instead of recalculating the BioPython metrics.
