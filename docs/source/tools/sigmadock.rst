.. _sigmadock:

SigmaDock
=========

Overview
--------

SigmaDock is a deep-learning molecular docking tool based on a diffusion generative model.
It predicts the binding pose of a ligand to a protein by learning the joint distribution
of protein–ligand complexes.  For full details see:

- `GitHub <https://github.com/alvaroprat97/sigmadock>`_
- `SigmaDock preprint <https://arxiv.org/abs/2511.04854>`_

With ProtFlow's SigmaDock runner you can integrate molecular docking into automated
protein-design pipelines.  The runner supports:

- **Redocking** — re-dock the native ligand extracted from an input complex to
  validate binding-pose prediction.
- **Crossdocking** — dock one or more external query ligands into the protein
  extracted from an input complex.
- **Pre-extracted inputs** — supply protein PDBs and ligand SDFs directly via
  dataframe columns, bypassing the automatic complex-splitting step.

Like any other ProtFlow runner, SigmaDock collects docking scores (affinity,
RMSD, PoseBusters checks) and integrates them back into your
:py:class:`~protflow.poses.Poses` instance.

.. note::

    **A ligand must always be present in the input complex** even for crossdocking —
    the bound ligand is used as a binding-site reference from which SigmaDock infers
    the pocket location.  Pass the residue name of this ligand via ``ligand_name``.

Installation
------------

Follow the installation instructions in the official SigmaDock repository:
`https://github.com/alvaroprat97/sigmadock <https://github.com/alvaroprat97/sigmadock>`_.

Once installed, add your SigmaDock environment paths to the ProtFlow configuration file:

.. code-block:: python
    :name: config-excerpt-sigmadock

    # path to SigmaDock's sample.py inference script
    SIGMADOCK_SCRIPT_PATH = ""  # e.g. "/your/path/to/sigmadock/sample.py"

    # path to the Python interpreter inside the SigmaDock environment
    SIGMADOCK_PYTHON_PATH = ""  # e.g. "/your/path/to/conda/envs/sigmadock/bin/python"

    # path to the SigmaDock model checkpoint directory
    SIGMADOCK_CKPT_PATH = ""  # e.g. "/your/path/to/sigmadock/checkpoints"

    # optional shell prefix to activate the SigmaDock conda environment
    # required if SigmaDock's subprocess cannot find its dependencies otherwise
    SIGMADOCK_PRE_CMD = ""  # e.g. "conda run -n sigmadock"

.. note::

    If you are having trouble finding your ProtFlow config file, run:

    .. code-block:: bash

        protflow-check-config

Redocking
---------

Redocking extracts the native ligand from each input complex and docks it back into
the same protein.  This is useful for validating predicted structures or benchmarking
pose-prediction accuracy.

.. code-block:: python

    from protflow.poses import Poses
    from protflow.tools.sigmadock import SigmaDock
    from protflow.jobstarters import LocalJobStarter

    jst = LocalJobStarter(max_cores=1)

    # load protein–ligand complexes (PDB or CIF)
    my_poses = Poses(
        poses="/path/to/complexes/",
        glob_suffix="*.cif",
        work_dir="/path/to/output_dir/"
    )

    runner = SigmaDock(jobstarter=jst)

    my_poses = runner.run(
        poses=my_poses,
        prefix="redock",
        ligand_name="LIG",   # residue name of the ligand in the input complex
        overwrite=False,
    )

    # scores are available as prefixed columns
    display(my_poses.df[["redock_affinity", "redock_rmsd", "redock_pb_pass_rate"]])

Crossdocking
------------

Crossdocking docks one or more external query ligands into the protein extracted from
each input complex.  Pass a list of absolute SDF paths via ``query_ligands``.

.. code-block:: python

    my_poses = runner.run(
        poses=my_poses,
        prefix="crossdock",
        ligand_name="LIG",
        query_ligands=[
            "/data/ligands/compound_1.sdf",
            "/data/ligands/compound_2.sdf",
        ],
        overwrite=True,
    )

    display(my_poses.df[["crossdock_affinity", "crossdock_pb_pass_rate"]])

.. note::

    Each query ligand is docked into **every** protein in the poses collection.
    Store query ligand SDF files outside the runner's ``work_dir`` — the ``overwrite``
    cleanup step removes the ``inputs/`` and ``outputs/`` sub-directories.

Pre-extracted inputs
--------------------

If you already have split protein PDBs and ligand SDFs (e.g. from a previous
ProtFlow step), pass them via dataframe columns to skip the automatic
complex-splitting step.

**Redocking with pre-extracted files**

.. code-block:: python

    my_poses.df["ligand_sdf"] = ["/data/extracted/compound_A.sdf", ...]

    my_poses = runner.run(
        poses=my_poses,
        prefix="redock_preextracted",
        ligand_col="ligand_sdf",   # column holding the ligand SDF paths
        overwrite=True,
    )

**Crossdocking with pre-extracted files**

.. code-block:: python

    my_poses.df["query_ligands"] = [
        ["/data/ligands/compound_1.sdf", "/data/ligands/compound_2.sdf"],
        ...
    ]
    my_poses.df["ref_ligand"] = ["/data/extracted/ref_A.sdf", ...]

    my_poses = runner.run(
        poses=my_poses,
        prefix="crossdock_preextracted",
        ligand_col="query_ligands",   # column holding lists of query SDF paths
        ref_ligand_col="ref_ligand",  # column holding the pocket-anchor SDF paths
        overwrite=True,
    )

.. note::

    When ``ligand_col`` is set, the poses column (``Poses.df["poses"]``) is used as
    the protein PDB.  Pass ``receptor_col`` to override this with a different column.

Output scores
-------------

After each run the following columns are added to ``Poses.df`` under the given
``prefix``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``{prefix}_location``
     - Absolute path to the docked protein–ligand complex PDB.
   * - ``{prefix}_affinity``
     - Predicted binding affinity from SigmaDock's rescoring model (lower is better).
   * - ``{prefix}_intramolecular_energy``
     - Intramolecular strain energy of the docked ligand pose.
   * - ``{prefix}_rmsd``
     - Ligand RMSD to the reference pose (available in redocking when a reference SDF exists).
   * - ``{prefix}_pb_pass_rate``
     - Fraction of `PoseBusters <https://github.com/maabuu/posebusters>`_ geometry checks passed.
