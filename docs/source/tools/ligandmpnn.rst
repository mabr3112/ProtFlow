.. _ligandmpnn:

LigandMPNN
==========

Overview
--------

LigandMPNN is a protein sequence design tool that can design sequences for protein backbones,
including binding interfaces to ligands or partners. This tutorial shows how to run the
``LigandMPNN`` runner in ProtFlow and collect results back into your :py:class:`~protflow.poses.Poses`
object.

For tool details, see:

- `GitHub <https://github.com/dauparas/LigandMPNN>`_
- `Paper <https://www.science.org/doi/10.1126/science.adg2022>`_

Installation
------------

Follow the LigandMPNN installation instructions and then register paths in
your ProtFlow config file:

.. code-block:: python
    :name: config-excerpt-ligandmpnn

    # LigandMPNN runner
    LIGANDMPNN_SCRIPT_PATH = ""  # e.g. "/path/to/LigandMPNN/run.py"
    LIGANDMPNN_PYTHON_PATH = ""  # e.g. "/path/to/conda/envs/ligandmpnn/bin/python3"
    LIGANDMPNN_PRE_CMD = ""      # optional, e.g. "module load cuda/11.8"

.. note::
    To check which config file ProtFlow is using, run:

    .. code-block:: bash

        protflow-check-config

Quickstart: run LigandMPNN
--------------------------

This example loads input PDBs, runs LigandMPNN locally, and inspects the scores.
Replace the file paths with your own.

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from protflow.tools.ligandmpnn import LigandMPNN

    # Use a local jobstarter for small tests
    local_jobstarter = LocalJobStarter(max_cores=1)

    # Load input structures as poses
    poses = Poses(
        poses="path/to/input_pdbs/",
        glob_suffix="*.pdb",
        work_dir="ligandmpnn_example",
        storage_format="csv",
        jobstarter=local_jobstarter,
    )

    # Initialize LigandMPNN runner
    ligandmpnn = LigandMPNN()

    # Run LigandMPNN
    poses = ligandmpnn.run(
        poses=poses,
        prefix="ligmpnn",
        nseq=2,
        model_type="protein_mpnn",
    )

    # Inspect results (columns are prefixed with "ligmpnn_")
    print(poses.df[["poses_description", "ligmpnn_sequence", "ligmpnn_overall_confidence"]])

Common options
--------------

LigandMPNN supports many command line options. Pass them via ``options`` exactly
as you would on the command line (excluding input/output paths, which ProtFlow manages).

.. code-block:: python

    ligandmpnn_opts = "--temperature 0.1 --seed 1"
    poses = ligandmpnn.run(
        poses=poses,
        prefix="ligmpnn_opts",
        nseq=2,
        model_type="protein_mpnn",
        options=ligandmpnn_opts,
    )

Pose-specific options
---------------------

Sometimes each pose needs different settings. Use ``pose_options`` to pass an
option string per pose (one entry per row in ``Poses.df``).

.. code-block:: python

    poses = Poses(
        poses="path/to/input_pdbs/",
        glob_suffix="*.pdb",
        work_dir="ligandmpnn_pose_opts",
        jobstarter=local_jobstarter,
    )

    # One options string per pose (None means "no extra options")
    poses.df["ligandmpnn_pose_opts"] = [
        "--fixed_residues 'A34 A173'",
        "--fixed_residues 'A36 A134'",
        None,
    ]

    poses = ligandmpnn.run(
        poses=poses,
        prefix="ligmpnn_pose_opts",
        nseq=1,
        model_type="protein_mpnn",
        pose_options="ligandmpnn_pose_opts",
    )

Fixed or redesigned residues from columns
-----------------------------------------

If you already store residue selections in columns, you can map them to
LigandMPNN options with ``fixed_res_col`` and ``design_res_col``. Each value
should be a whitespace-separated list like ``"A34 A173"``.

.. code-block:: python

    poses = Poses(
        poses="path/to/input_pdbs/",
        glob_suffix="*.pdb",
        work_dir="ligandmpnn_fixed_design",
        jobstarter=local_jobstarter,
    )

    poses.df["fixed_residues"] = ["A34 A173", "A36 A134", ""]
    poses.df["design_residues"] = ["A1 A2", "A5 A6", "A7 A8"]

    poses = ligandmpnn.run(
        poses=poses,
        prefix="ligmpnn_fixed_design",
        nseq=1,
        model_type="protein_mpnn",
        fixed_res_col="fixed_residues",
        design_res_col="design_residues",
    )

Outputs
-------

LigandMPNN writes its output into the run directory inside your ``work_dir``:

- ``backbones/``: input backbones
- ``seqs/``: designed sequences (FASTA)
- ``packed/``: optionally packed structures when using sidechain packing
- ``ligandmpnn_scores.<format>``: collected scores

The :py:class:`~protflow.poses.Poses` dataframe is updated with new columns
prefixed by the run name (e.g., ``ligmpnn_sequence``, ``ligmpnn_overall_confidence``,
``ligmpnn_ligand_confidence``, ``ligmpnn_location``).
