.. _pottsmpnn:

PottsMPNN
=========

Overview
--------

PottsMPNN is a protein sequence-design and mutation-energy prediction tool based
on ProteinMPNN-style structure conditioning with learned Potts energies. For
tool details, see:

- `GitHub <https://github.com/KeatingLab/PottsMPNN>`_
- `Zenodo DOI <https://doi.org/10.5281/zenodo.18274667>`_

With ProtFlow's PottsMPNN runner you can run the upstream YAML-based command-line
workflows from a :py:class:`~protflow.poses.Poses` object:

- ``sample_seqs.py`` designs sequences for input PDB backbones.
- ``energy_prediction.py`` scores mutations or deep mutational scans.

The runner writes PottsMPNN YAML configs, starts jobs through a ProtFlow
jobstarter, collects FASTA or CSV outputs, and merges the results back into the
poses dataframe.

Installation
------------

Follow the installation instructions in the official PottsMPNN repository:
`https://github.com/KeatingLab/PottsMPNN <https://github.com/KeatingLab/PottsMPNN>`_.

Once installed, add the PottsMPNN paths to your ProtFlow config file:

.. code-block:: python
    :name: config-excerpt-pottsmpnn

    # path to the PottsMPNN repository checkout
    POTTSMPNN_DIR = ""  # e.g. "/path/to/PottsMPNN"

    # path to the Python interpreter inside the PottsMPNN environment
    POTTSMPNN_PYTHON = ""  # e.g. "/path/to/conda/envs/PottsMPNN/bin/python"

    # optional shell prefix to activate modules or environments
    POTTSMPNN_PRE_CMD = ""  # e.g. "conda run -n PottsMPNN"

.. note::

    To check which config file ProtFlow is using, run:

    .. code-block:: bash

        protflow-check-config

Sequence Design
---------------

Use :py:class:`~protflow.tools.pottsmpnn.SampleSequencePottsMPNNParams` for
``sample_seqs.py``. The params object mirrors the upstream YAML structure and
exposes nested ``model`` and ``inference`` attributes for IDE autocomplete.

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    from protflow.tools import PottsMPNN, SampleSequencePottsMPNNParams

    jobstarter = LocalJobStarter(max_cores=2)

    poses = Poses(
        poses="/path/to/input_pdbs/",
        glob_suffix="*.pdb",
        work_dir="/path/to/output_dir/",
        jobstarter=jobstarter,
    )

    params = SampleSequencePottsMPNNParams()
    params.inference.num_samples = 4
    params.inference.temperature = 0.1
    params.model.check_path = "vanilla_model_weights/pottsmpnn_msa_20.pt"

    runner = PottsMPNN()
    poses = runner.run(
        poses=poses,
        prefix="potts_design",
        params=params,
    )

    print(poses.df[["poses_description", "potts_design_sequence", "potts_design_location"]])

The ``location`` column points to per-sequence FASTA files written by ProtFlow.
If PottsMPNN also writes optimized sequences, those are collected in
``{prefix}_optimized_potts_sequence`` columns.

Pose-Specific Settings
----------------------

Use :py:class:`~protflow.tools.pottsmpnn.PoseCol` when a PottsMPNN parameter
should be filled from a ``Poses.df`` column. The ``*_custom`` fields are
ProtFlow helpers: they are converted into temporary JSON files and passed to the
matching upstream ``*_json`` config key.

.. code-block:: python

    from protflow.tools import PoseCol, SampleSequencePottsMPNNParams

    poses.df["fixed_positions"] = [
        {"A": [10, 11, 12]},
        {"A": [25, 26]},
    ]

    params = SampleSequencePottsMPNNParams()
    params.inference.fixed_positions_custom = PoseCol("fixed_positions")

    poses = runner.run(
        poses=poses,
        prefix="potts_fixed",
        params=params,
    )

When all pose-specific values are stored in batch-compatible ``*_custom`` fields,
the runner can batch multiple poses per config. Pose-specific scalar fields, such
as ``params.inference.temperature = PoseCol("temperature")``, are written as one
config per pose.

Chain Design JSON
-----------------

PottsMPNN accepts chain-design information either through ``input_list`` entries
or ``chain_dict_json``. ProtFlow manages ``input_list`` automatically. To pass
pose-specific chain dictionaries, use ``chain_dict_custom``:

.. code-block:: python

    poses.df["chain_design"] = [
        [["A"], ["B"]],
        [["A", "B"], []],
    ]

    params = SampleSequencePottsMPNNParams()
    params.chain_dict_custom = PoseCol("chain_design")

    poses = runner.run(
        poses=poses,
        prefix="potts_chain_design",
        params=params,
    )

Energy Prediction
-----------------

Use :py:class:`~protflow.tools.pottsmpnn.EnergyPredictionPottsMPNNParams` with
``script="energy_prediction"``. You can provide either ``mutant_csv`` or
``mutant_fasta``. If both are ``None``, upstream PottsMPNN performs a deep
mutational scan.

.. code-block:: python

    from protflow.tools import EnergyPredictionPottsMPNNParams

    params = EnergyPredictionPottsMPNNParams()
    params.mutant_csv = "/path/to/mutations.csv"
    params.inference.ddG = True
    params.inference.mean_norm = False

    poses = runner.run(
        poses=poses,
        prefix="potts_energy",
        script="energy_prediction",
        params=params,
    )

    print(poses.df[[
        "poses_description",
        "potts_energy_energy_prediction_scorefile",
        "potts_energy_energy_prediction_n_mutations",
    ]])

The returned scorefile column points to one JSON sidecar per input pose. Each
sidecar contains the full table read from PottsMPNN's ``*_scores.csv`` output.

API Reference
-------------

See :mod:`protflow.tools.pottsmpnn` for the full autodoc API, including all
parameter dataclasses and score-collection helpers.
