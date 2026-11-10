.. _esmfold2:

ESMFold2
========

Overview
--------

The ESMFold2 runner predicts structures from protein sequences and collects
the resulting CIF files and confidence scores in your :class:`~protflow.poses.Poses`
dataframe. It runs the model locally through the Biohub ESM input builder and
Hugging Face Transformers.

For model setup and capabilities, see the
`Biohub ESM repository <https://github.com/Biohub/esm>`_ and the
`ESMFold2 model page <https://huggingface.co/biohub/ESMFold2>`_.

Installation and configuration
------------------------------

Install ESMFold2 in its own environment following the upstream instructions.
The environment must provide these imports used by ProtFlow's worker:

.. code-block:: python

    from esm.models.esmfold2 import ESMFold2InputBuilder, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

The worker also uses pandas, NumPy, and Biotite. It loads the model onto CUDA;
allocate a GPU for each concurrent job. ProtFlow's own environment does not
need to import the model packages.

Set these entries in your ProtFlow configuration:

.. code-block:: python

    ESMFOLD2_PYTHON = "/path/to/esmfold2/environment/bin/python"
    ESMFOLD2_PRE_CMD = ""  # Optional shell setup, e.g. loading a cluster module.

The runner locates the bundled ``run_esmfold2.py`` worker through
``AUXILIARY_RUNNER_SCRIPTS_DIR``. There is no separate ``ESMFOLD2_PATH`` setting.
See :doc:`../quickstart/configuration` for locating and editing your configuration.

Predicting protein structures
-----------------------------

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from protflow.tools import ESMFold2

    poses = Poses(
        poses="/path/to/fastas",
        glob_suffix="*.fasta",
        work_dir="/path/to/results",
    )
    runner = ESMFold2(jobstarter=LocalJobStarter(max_cores=1))
    poses = runner.run(
        poses=poses,
        prefix="esmfold2",
        model="biohub/ESMFold2",
        options={"num_loops": 3, "num_sampling_steps": 200},
    )
    print(poses.df[["esmfold2_location", "esmfold2_mean_per_res_plddt"]])

``options`` is a Python dictionary of keyword arguments passed to
``ESMFold2InputBuilder.fold()``. It is not a command-line string. The sampling
settings above are an example; choose settings appropriate for your task.
Use ``num_diffusion_samples`` to request multiple predictions per input.

Inputs can be FASTA (``.fa``, ``.fas``, ``.fasta``), PDB, CIF, or pre-written
ESMFold2 JSON files. For FASTA inputs, put the chains in one sequence separated
by ``:`` or ``/``; ProtFlow assigns chain IDs A, B, and so on. PDB/CIF conversion
extracts protein sequences and retains chain IDs. Coordinates and ligands in
those structures are not automatically transferred into the prediction input.

The automatically generated protein entries use no MSA. This runner does not
launch an MSA server. Pre-written JSON inputs should use the same entity format
as the generated files; for example:

.. code-block:: json

    {"sequences": [{"protein": {"id": "A", "sequence": "ACDEFGHIK", "msa": null}}]}

Pose-specific prediction options
--------------------------------

Store one dictionary per input pose in a dataframe column and name that column
in ``pose_options``. A pose-specific key overrides the corresponding global
key. Other global options are retained. Neither input dictionary is modified,
and overrides do not carry over to other poses.

.. code-block:: python

    # Start with two input poses, before running predictions.
    poses.df["fold_options"] = [
        {"num_loops": 5, "num_diffusion_samples": 2},
        {},  # Use the global options for this pose.
    ]
    poses = runner.run(
        poses=poses,
        prefix="esmfold2_per_pose",
        options={"num_loops": 3, "num_sampling_steps": 200},
        pose_options="fold_options",
    )

Inputs are split into at most ``jobstarter.max_cores`` batches, including when
using per-pose options. Each worker processes its batch sequentially with one
loaded model. On a single GPU, use ``LocalJobStarter(max_cores=1)``. On SLURM,
use :class:`~protflow.jobstarters.SbatchArrayJobstarter` with ``gpus=1`` and
your cluster's partition and resource options.

Adding molecular entities with ESMFold2Params
---------------------------------------------

:class:`~protflow.tools.esmfold2.ESMFold2Params` appends proteins, DNA, RNA,
ligands, and constraints to the input JSON. Choose chain IDs that are not
already present in the input.

.. code-block:: python

    from protflow.tools import ESMFold2Params

    params = ESMFold2Params()
    params.add_protein(id="B", sequence="ACDEFGHIKLMNPQRSTVWY")
    params.add_ligand(id="Z", ligand="CCO", ligand_type="smiles")

    # Start from your original input poses, with protein chain A.
    poses = runner.run(
        poses=poses,
        prefix="esmfold2_complex",
        params=params,
        options={"num_diffusion_samples": 2},
    )

For a CCD ligand, pass a single code with ``ligand_type="ccd"``. For an entity
that varies between poses, use ``poses_cols`` to identify which arguments
refer to dataframe columns:

.. code-block:: python

    # One SMILES string per input pose.
    poses.df["ligand_smiles"] = ["CCO", "CC(=O)O"]
    params = ESMFold2Params()
    params.add_ligand(
        id="Z",
        ligand="ligand_smiles",
        ligand_type="smiles",
        poses_cols=["ligand"],
    )

Pass this ``params`` object to ``runner.run()`` as above. ``poses_cols`` names
method arguments (here ``ligand``); their values name dataframe columns.
This changes the molecular input, whereas ``pose_options`` changes prediction
settings. The worker also accepts ``PocketConditioning``, ``CovalentBond``, and
``DistogramConditioning`` entries added with ``params.add_constraint()``; their
arguments must match the installed ESM input classes and be JSON-serializable.

Outputs and reruns
------------------

Each prediction adds one index layer, such as ``target_0001``. Output CIFs and
per-batch JSON score tables are written beneath ``<work_dir>/<prefix>/output``.
The combined cache is ``<work_dir>/<prefix>/ESMFold2_scores.json`` when using
the default JSON storage format.

Result columns are prefixed with the run name:

* ``<prefix>_location`` and ``<prefix>_description`` identify each prediction.
* ``<prefix>_mean_per_res_plddt`` and ``<prefix>_per_res_plddt`` contain mean and
  per-residue confidence, when returned by the model.
* ``<prefix>_ptm`` and ``<prefix>_iptm`` are included when returned by the model.

With ``overwrite=False``, an existing score table is reused. Changing options
does not invalidate that cache. Use ``overwrite=True`` to recompute the same
inputs. When changing the input set or batching, use a fresh prefix: the
collector reads all batch score tables in the output directory.

API reference
-------------

.. autoclass:: protflow.tools.esmfold2.ESMFold2
    :members: run
    :no-index:

.. autoclass:: protflow.tools.esmfold2.ESMFold2Params
    :no-index:
