.. _generic_metric_tutorial:

Write a Generic Metric
======================

Use :class:`protflow.metrics.generic_metric_runner.GenericMetric` when you need
one function call per pose and the output can be represented as a single
JSON-serializable value per pose.

Typical use-cases:

- quick structural sanity checks that are not worth a dedicated runner
- custom metrics used only in one project
- wrapping an existing helper function so it can run through a ProtFlow
  ``JobStarter``

Do not use ``GenericMetric`` when the metric:

- needs multiple poses at once
- produces non-JSON output
- should return several columns instead of one ``<prefix>_data`` column
- needs different options for different poses in one run

What the function must look like
--------------------------------

The worker process imports the target with ``importlib.import_module(module)``
and then calls ``function(pose_path, **options)`` for each pose. That means:

- the first argument must be the pose path as a string
- additional parameters must be regular keyword arguments
- the return value must be JSON-serializable
- the function must live in an importable Python module

Notebook-only functions, lambdas, and functions defined only in an interactive
session will not work, because worker jobs import the function again in a new
Python process.

Make the module importable
--------------------------

The ``module=...`` argument is a Python module path, not a shell ``PATH``
lookup.

Your metric function must be importable by the Python interpreter used by
``GenericMetric``:

- the interpreter given through ``python_path``, or
- the default ProtFlow environment interpreter from ``PROTFLOW_ENV``

Valid ways to make the module importable:

- put the function into ``protflow`` itself
- put it into another package that is installed in the same environment
- install your own package into that environment
- export ``PYTHONPATH`` before launching ProtFlow so the module directory is on
  Python's import path

If you launch jobs through SLURM, the compute nodes also need that same Python
environment or ``PYTHONPATH`` setup.

Example file layout
-------------------

If you store your function in:

.. code-block:: text

   my_project_metrics/
   |-- __init__.py
   `-- custom_metrics.py

then the corresponding ``module`` string is:

.. code-block:: python

   module="my_project_metrics.custom_metrics"

Example metric with options
---------------------------

The example below counts C-alpha atoms on a selected chain whose B-factor is at
least ``min_bfactor``. For AlphaFold-style structures, this can be used as a
quick proxy for "confident residues on chain A" if confidence values are stored
in the B-factor column.

.. code-block:: python

   # file: my_project_metrics/custom_metrics.py
   from Bio.PDB import PDBParser


   def count_confident_ca_atoms(pose: str, chain: str, min_bfactor: float = 0.0) -> int:
       """Return the number of CA atoms on one chain above a B-factor threshold."""
       structure = PDBParser(QUIET=True).get_structure("pose", pose)
       model = next(structure.get_models())

       if chain not in model:
           raise KeyError(f"Chain {chain!r} not found in {pose}")

       return sum(
           1
           for atom in model[chain].get_atoms()
           if atom.id == "CA" and atom.bfactor >= min_bfactor
       )

This function satisfies the ``GenericMetric`` contract:

- ``pose`` is the first argument
- ``chain`` and ``min_bfactor`` are regular keyword arguments
- the return value is an integer, which is JSON-serializable

Run the metric in ProtFlow
--------------------------

.. code-block:: python

   from protflow.poses import Poses
   from protflow.jobstarters import LocalJobStarter
   from protflow.metrics.generic_metric_runner import GenericMetric

   jobstarter = LocalJobStarter(max_cores=8)

   poses = Poses(
       poses="/data/input_pdbs",
       glob_suffix="*.pdb",
       work_dir="generic_metric_example",
       jobstarter=jobstarter,
   )

   confident_len = GenericMetric(
       module="my_project_metrics.custom_metrics",
       function="count_confident_ca_atoms",
       options={"chain": "A", "min_bfactor": 70.0},
       jobstarter=jobstarter,
   )

   poses = confident_len.run(poses=poses, prefix="chain_a_confident_len")
   print(poses.df[["poses_description", "chain_a_confident_len_data"]])

What happens during ``run()``:

1. ProtFlow creates ``<work_dir>/<prefix>``.
2. It splits the pose list into chunks.
3. It starts one worker command per chunk through the selected ``JobStarter``.
4. Each worker imports ``my_project_metrics.custom_metrics`` and calls
   ``count_confident_ca_atoms(pose, chain="A", min_bfactor=70.0)`` for every
   pose in its chunk.
5. The results are merged back into ``poses.df``.

Results and files
-----------------

After the run:

- the metric value is in ``chain_a_confident_len_data``
- ProtFlow also stores ``chain_a_confident_len_description`` and
  ``chain_a_confident_len_location``
- the cached runner scorefile is written inside the run directory

If you run the same prefix again and do not set ``overwrite=True``, ProtFlow
reuses the cached scorefile instead of recomputing the metric.

Important limits
----------------

``GenericMetric`` applies one shared ``options`` dictionary to all poses in a
single run. It does not support pose-specific options columns.

If you need per-pose options, use one of these approaches:

- split the poses into subsets and run ``GenericMetric`` multiple times
- write a dedicated runner for that metric

Returning a scalar is usually the cleanest option. Returning a list or dict is
allowed as long as it is JSON-serializable, but it will still be stored inside a
single ``<prefix>_data`` column.
