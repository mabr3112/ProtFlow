"""ProtFlow runner for PottsMPNN.

This module integrates the command-line PottsMPNN YAML workflows into
ProtFlow. It supports the two upstream scripts that expose a ``--config``
interface:

- ``sample_seqs.py`` for sequence design from backbone structures.
- ``energy_prediction.py`` for mutation-energy and deep-mutational-scan
  scoring.

The runner writes script-specific YAML files, dispatches one command per
generated config through a :class:`~protflow.jobstarters.JobStarter`, and
collects PottsMPNN FASTA or CSV outputs back into a
:class:`~protflow.poses.Poses` dataframe.

Configuration
-------------
The runner reads default executable paths from the ProtFlow config:

``POTTSMPNN_DIR``
    Path to the local PottsMPNN checkout. Commands are executed from this
    directory so relative checkpoint paths from upstream YAML examples work.
``POTTSMPNN_PYTHON``
    Python interpreter from the PottsMPNN environment.
``POTTSMPNN_PRE_CMD``
    Optional shell prefix used to activate modules or environments before each
    command.

Parameter Objects
-----------------
Use :class:`SampleSequencePottsMPNNParams` with ``sample_seqs.py`` and
:class:`EnergyPredictionPottsMPNNParams` with ``energy_prediction.py``. These
typed dataclasses expose PottsMPNN model and inference fields directly, so IDEs
can autocomplete nested attributes such as ``params.model.check_path`` and
``params.inference.num_samples``.

Pose-specific Values
--------------------
Wrap a dataframe column name in :class:`PoseCol` to fill a parameter from
``Poses.df``. Parameters ending in ``*_custom`` are converted into temporary
JSON files and can still be batched. Other pose-specific parameters require one
config per input pose.

Examples
--------
Design two sequences per backbone:

>>> from protflow.poses import Poses
>>> from protflow.tools import PottsMPNN, SampleSequencePottsMPNNParams
>>> poses = Poses(poses=["backbone_a.pdb", "backbone_b.pdb"], work_dir="work")
>>> params = SampleSequencePottsMPNNParams()
>>> params.inference.num_samples = 2
>>> params.inference.temperature = 0.1
>>> params.inference.optimization_mode = "none"
>>> poses = PottsMPNN().run(poses=poses, prefix="potts_design", params=params)

Score mutations from a CSV file:

>>> from protflow.tools import EnergyPredictionPottsMPNNParams
>>> params = EnergyPredictionPottsMPNNParams(mutant_csv="mutations.csv")
>>> poses = PottsMPNN().run(
...     poses=poses,
...     prefix="potts_energy",
...     script="energy_prediction",
...     params=params,
... )
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shlex
import shutil
from dataclasses import dataclass, field, fields, is_dataclass
from glob import glob
from typing import Any, ClassVar

import pandas as pd
import yaml

from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter, split_list
from protflow.poses import Poses
from protflow.runners import (
    Runner,
    RunnerOutput,
    options_flags_to_string,
    parse_generic_options,
    prepend_cmd,
)

# scripts with upstream --config YAML entrypoints.
SUPPORTED_CONFIG_SCRIPTS = {"sample_seqs", "energy_prediction"}

class PottsMPNN(Runner):
    """Run PottsMPNN command-line scripts from ProtFlow.

    Parameters
    ----------
    python_path : str, optional
        Python interpreter used to execute PottsMPNN. If omitted, the value is
        loaded from ``POTTSMPNN_PYTHON`` in the ProtFlow config.
    pottsmpnn_dir : str, optional
        Path to the PottsMPNN checkout. If omitted, the value is loaded from
        ``POTTSMPNN_DIR``.
    pre_cmd : str, optional
        Shell prefix prepended to every command, commonly used to activate a
        conda environment or cluster module. Defaults to ``POTTSMPNN_PRE_CMD``.
    jobstarter : JobStarter, optional
        Default jobstarter used when :meth:`run` is called without one.

    Attributes
    ----------
    name : str
        Runner name used for job names and cached score files.
    index_layers : int
        Default merge index depth. The active value is selected per script in
        :meth:`run` because ``sample_seqs.py`` appends sample indices while
        ``energy_prediction.py`` keeps one row per input pose.
    pottsmpnn_dir : str
        Resolved PottsMPNN checkout path.
    python_path : str
        Resolved PottsMPNN Python interpreter.
    pre_cmd : str
        Resolved shell prefix.

    Notes
    -----
    Only upstream scripts with a ``--config`` YAML interface are supported.
    The runner currently supports ``sample_seqs.py`` and
    ``energy_prediction.py``.
    """

    def __init__(
        self,
        python_path: str | None = None,
        pottsmpnn_dir: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize the runner and resolve PottsMPNN configuration.

        Parameters
        ----------
        python_path : str, optional
            Python interpreter used to run PottsMPNN.
        pottsmpnn_dir : str, optional
            Local PottsMPNN checkout.
        pre_cmd : str, optional
            Optional shell prefix for environment activation.
        jobstarter : JobStarter, optional
            Default jobstarter for this runner instance.
        """
        # config required
        config = require_config()

        # setup config paths
        self.pottsmpnn_dir = str(pottsmpnn_dir or load_config_path(config, "POTTSMPNN_DIR"))
        self.python_path = str(python_path or load_config_path(config, "POTTSMPNN_PYTHON"))
        self.pre_cmd = pre_cmd or load_config_path(config, "POTTSMPNN_PRE_CMD", is_pre_cmd=True)

        # setup runner state
        self.jobstarter = jobstarter
        self.name = "pottsmpnn"
        self.index_layers = 0

    def __str__(self) -> str:
        """Return the short runner name.

        Returns
        -------
        str
            The literal runner name ``"pottsmpnn"``.
        """
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        script: str | None = "sample_seqs",
        params: SampleSequencePottsMPNNParams | EnergyPredictionPottsMPNNParams | None = None,
        options: str | None = None,
        pose_options: str | list[str] | None = None,
        include_scores: list[str] | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Run PottsMPNN and merge collected results into ``poses``.

        Parameters
        ----------
        poses : Poses
            Input structures to pass to PottsMPNN. The ``poses`` column must
            contain PDB paths and ``poses_description`` is used as the upstream
            PottsMPNN structure identifier.
        prefix : str
            Unique run prefix used to create the runner work directory and
            prefixed output score columns.
        jobstarter : JobStarter, optional
            Jobstarter for this call. If omitted, the runner falls back to the
            instance jobstarter and then ``poses.default_jobstarter``.
        script : str, optional
            Script alias or path. Supported aliases are ``"sample_seqs"`` and
            ``"energy_prediction"``.
        params : SampleSequencePottsMPNNParams or EnergyPredictionPottsMPNNParams, optional
            Typed parameter object used to generate YAML configs. If omitted,
            defaults are created for the selected script.
        options : str, optional
            Extra command-line options passed to the upstream script. ``--config``
            is ignored because config files are managed by the runner.
        pose_options : str or list of str, optional
            Unsupported for PottsMPNN. Use :class:`PoseCol` fields in ``params``
            for pose-specific settings.
        include_scores : list of str, optional
            Reserved for API consistency with other runners. PottsMPNN collectors
            currently load the standard output fields.
        overwrite : bool, optional
            If ``True``, remove previous runner-owned outputs and rerun jobs.

        Returns
        -------
        Poses
            The input ``Poses`` object with PottsMPNN score columns merged in.

        Raises
        ------
        ValueError
            If ``pose_options`` are supplied or the params object does not match
            the selected script.
        NotImplementedError
            If ``script`` is not one of the supported config-based scripts.
        RuntimeError
            If PottsMPNN runs but no score rows can be collected.
        """
        # sanity
        if pose_options is not None:
            raise ValueError("PottsMPNN uses YAML configs; use PoseCol params instead of pose_options.")

        # sanitize script_path and params:
        script_path, script_key = self._resolve_script(script)
        index_layers = 1 if script_key == "sample_seqs" else 0
        if params is None:
            params = (
                SampleSequencePottsMPNNParams()
                if script_key == "sample_seqs"
                else EnergyPredictionPottsMPNNParams()
            )
        if params.script != script_key:
            raise ValueError(f"Params for '{params.script}' cannot be used with script '{script_key}'.")
        if script_key == "sample_seqs":
            _check_sample_descriptions(poses)

        # setup run directory and jobstarter
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # scorefile reuse shortcut
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            outputs = RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=index_layers
            )
            return outputs.return_poses()

        # cleanup previous outputs
        if overwrite:
            self._cleanup_previous_outputs(work_dir)

        # prepare config files
        batched, config_files = params_to_config(
            poses=poses,
            n_batches=jobstarter.max_cores,
            work_dir=work_dir,
            params=params,
        )
        # build commands
        cmds = self._build_commands(
            script=script_path,
            config_files=config_files,
            options=options
        )

        # prepend configured environment command
        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # execute jobs
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir
        )

        # collect and validate scores
        scores = collect_scores(
            work_dir=work_dir,
            script=script_key,
            batched=batched,
            include_scores=include_scores
        )
        scores = _fill_missing_locations(scores=scores, poses=poses, index_layers=index_layers)
        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output directory: {work_dir}")

        # save scores and merge back into poses
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        outputs = RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=index_layers
        )
        return outputs.return_poses()

    def _resolve_script(self, script: str | None) -> tuple[str, str]:
        """Resolve a script alias or path to an executable PottsMPNN script.

        Parameters
        ----------
        script : str, optional
            Script alias, script filename, or absolute path.

        Returns
        -------
        tuple of str
            Absolute script path and normalized script key.

        Raises
        ------
        NotImplementedError
            If the script does not provide a supported ``--config`` interface.
        FileNotFoundError
            If the script cannot be found directly or inside ``pottsmpnn_dir``.
        """
        # normalize script alias
        script = script or "sample_seqs"
        script_key = _script_key(script)
        # restrict to config-based scripts
        if script_key not in SUPPORTED_CONFIG_SCRIPTS:
            raise NotImplementedError(
                "Only PottsMPNN scripts with a '--config' YAML interface are supported: "
                f"{sorted(SUPPORTED_CONFIG_SCRIPTS)}"
            )

        # search direct path and checkout-relative path
        candidates = [str(script)]
        if not str(script).endswith(".py"):
            candidates.append(f"{script}.py")
        candidates.extend(
            os.path.join(self.pottsmpnn_dir, candidate)
            for candidate in list(candidates)
        )

        # return first valid script path.
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate), script_key
        raise FileNotFoundError(f"Could not find PottsMPNN script '{script}' in {self.pottsmpnn_dir}.")

    def _prep_pottsmpnn_opts(self, raw_opts: str | None) -> str:
        """Normalize extra command-line options for PottsMPNN.

        Parameters
        ----------
        raw_opts : str, optional
            User-supplied options string.

        Returns
        -------
        str
            Parsed option string with runner-managed ``--config`` removed.
        """
        # parse generic CLI options
        if not raw_opts:
            return ""
        opts, flags = parse_generic_options(raw_opts, "", sep="--")

        # config files are managed by the runner
        if "config" in opts:
            logging.warning("Ignoring user-specified PottsMPNN --config option: %s", opts["config"])
            del opts["config"]
        return options_flags_to_string(opts, flags, sep="--")

    def _build_commands(self, script: str, config_files: list[str], options: str | None) -> list[str]:
        """Build one PottsMPNN command per generated config file.

        Parameters
        ----------
        script : str
            Absolute path to the upstream PottsMPNN script.
        config_files : list of str
            YAML config files generated for this run.
        options : str, optional
            Extra command-line options shared by all configs.

        Returns
        -------
        list of str
            Shell commands ready for a :class:`~protflow.jobstarters.JobStarter`.
        """
        # options are shared across generated configs
        cli_args = self._prep_pottsmpnn_opts(options)
        cmds = [
            self.write_cmd(script=script, config_path=config_file, cli_args=cli_args)
            for config_file in config_files
        ]
        return cmds

    def write_cmd(self, script: str, config_path: str, cli_args: str = "") -> str:
        """Format the shell command for a single PottsMPNN config.

        Parameters
        ----------
        script : str
            Absolute path to the upstream PottsMPNN script.
        config_path : str
            YAML config passed as ``--config``.
        cli_args : str, optional
            Additional parsed command-line arguments.

        Returns
        -------
        str
            Command that runs from the PottsMPNN checkout.
        """
        # run from PottsMPNN checkout so relative model paths work
        cmd = (
            f"cd {shlex.quote(self.pottsmpnn_dir)}; "
            f"{shlex.quote(self.python_path)} {shlex.quote(script)} --config {shlex.quote(config_path)}"
        )
        return f"{cmd} {cli_args}" if cli_args else cmd

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Remove previous runner-owned outputs inside the work directory.

        Parameters
        ----------
        work_dir : str
            Runner work directory created for this prefix.
        """
        # remove only files/directories inside runner work_dir
        if not os.path.isdir(work_dir):
            return
        for path in glob(os.path.join(work_dir, "*")):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


class PoseCol(str):
    """Mark a PottsMPNN parameter as coming from ``Poses.df``.

    ``PoseCol`` behaves like a string at runtime, but it signals the config
    writer to replace the value with data from the named dataframe column.

    Parameters
    ----------
    col_name : str
        Name of the column in ``poses.df``.

    Examples
    --------
    Use a dataframe column to write a per-pose fixed-position JSON file:

    >>> from protflow.tools import PoseCol, SampleSequencePottsMPNNParams
    >>> params = SampleSequencePottsMPNNParams()
    >>> params.inference.fixed_positions_custom = PoseCol("fixed_positions")
    """

    def __new__(cls, col_name: str) -> "PoseCol":
        """Create a string subclass that preserves ``PoseCol`` type checks.

        Parameters
        ----------
        col_name : str
            Referenced ``poses.df`` column.

        Returns
        -------
        PoseCol
            Marker value used by config generation.
        """
        return super().__new__(cls, col_name)

    @property
    def col_name(self) -> str:
        """Return the referenced poses dataframe column name.

        Returns
        -------
        str
            Column name stored by this marker.
        """
        return str(self)


@dataclass
class PottsMPNNModelParams:
    """Store PottsMPNN model configuration fields.

    Attributes
    ----------
    check_path : str or PoseCol
        Path to the PottsMPNN checkpoint. Relative paths are resolved from the
        PottsMPNN checkout because commands run in ``POTTSMPNN_DIR``.
    hidden_dim : int or PoseCol
        Hidden dimension used to construct the model.
    edge_features : int or PoseCol
        Number of edge features.
    potts_dim : int or PoseCol
        Potts representation dimension.
    num_layers : int or PoseCol
        Number of encoder and decoder layers.
    num_edges : int or PoseCol
        Number of structural neighbors.
    vocab : int or PoseCol
        Vocabulary size expected by the checkpoint.
    """
    check_path: str | PoseCol = "vanilla_model_weights/pottsmpnn_20.pt"
    hidden_dim: int | PoseCol = 128
    edge_features: int | PoseCol = 128
    potts_dim: int | PoseCol = 400
    num_layers: int | PoseCol = 3
    num_edges: int | PoseCol = 48
    vocab: int | PoseCol = 21


@dataclass
class SampleSequenceInferenceParams:
    """Store ``sample_seqs.py`` inference configuration fields.

    Attributes
    ----------
    num_samples : int or PoseCol
        Number of sequences sampled per input structure.
    temperature : float or PoseCol
        Autoregressive sampling temperature.
    noise : float or PoseCol
        Coordinate noise added during inference.
    skip_gaps : bool or PoseCol
        Whether upstream parsing should skip structural gaps.
    fix_decoding_order : bool or PoseCol
        Whether to use a fixed decoding order.
    decoding_order_offset : int or PoseCol
        Offset applied to the fixed decoding order.
    optimization_mode : str or PoseCol
        Optimization mode, typically ``"none"``, ``"potts"``, or ``"nodes"``.
    optimization_temperature : float or PoseCol
        Temperature used during sequence optimization.
    binding_energy_optimization : str or PoseCol
        Binding-energy optimization mode, typically ``"none"``, ``"both"``,
        or ``"only"``.
    binding_energy_json : str or None or PoseCol
        Path to upstream binding-energy partition JSON.
    binding_energy_cutoff : float or PoseCol
        Interface cutoff in Angstrom used for binding-energy optimization.
    optimize_pdb : bool or PoseCol
        Optimize sequences read from the input PDB files.
    optimize_fasta : str or PoseCol
        FASTA file whose sequences should be optimized.
    write_pdb : bool or PoseCol
        Whether PottsMPNN should write redesigned PDB files.
    fixed_positions_json, pssm_json, omit_AA_json, bias_AA_json, tied_positions_json, bias_by_res_json : str or PoseCol
        Paths to upstream ProteinMPNN-style constraint and bias JSON files.
    fixed_positions_custom, pssm_custom, omit_AA_custom, bias_AA_custom, tied_positions_custom, bias_by_res_custom : str or PoseCol
        ProtFlow helpers that can be populated from ``PoseCol`` values and are
        written to temporary JSON files before running PottsMPNN.
    omit_AAs : list of str or PoseCol
        Amino acids globally omitted from sampling.
    pssm_threshold : float or PoseCol
        PSSM threshold passed to PottsMPNN.
    pssm_multi : float or PoseCol
        PSSM mixing weight.
    pssm_log_odds_flag : bool or PoseCol
        Whether to use PSSM log odds.
    pssm_bias_flag : bool or PoseCol
        Whether to use PSSM biasing.
    """
    num_samples: int | PoseCol = 1
    temperature: float | PoseCol = 0.1
    noise: float | PoseCol = 0.0
    skip_gaps: bool | PoseCol = False
    fix_decoding_order: bool | PoseCol = True
    decoding_order_offset: int | PoseCol = 0
    optimization_mode: str | PoseCol = "potts"
    optimization_temperature: float | PoseCol = 0.0
    binding_energy_optimization: str | PoseCol = "none"
    binding_energy_json: str | None | PoseCol = None
    binding_energy_cutoff: float | PoseCol = 8
    optimize_pdb: bool | PoseCol = False
    optimize_fasta: str | PoseCol = ""
    write_pdb: bool | PoseCol = True
    fixed_positions_json: str | PoseCol = ""
    pssm_json: str | PoseCol = ""
    omit_AA_json: str | PoseCol = ""
    bias_AA_json: str | PoseCol = ""
    tied_positions_json: str | PoseCol = ""
    tied_epistasis: bool | PoseCol = False
    bias_by_res_json: str | PoseCol = ""
    fixed_positions_custom: str | PoseCol = ""
    pssm_custom: str | PoseCol = ""
    omit_AA_custom: str | PoseCol = ""
    bias_AA_custom: str | PoseCol = ""
    tied_positions_custom: str | PoseCol = ""
    bias_by_res_custom: str | PoseCol = ""
    omit_AAs: list[str] | PoseCol = field(default_factory=list)
    pssm_threshold: float | PoseCol = 0.0
    pssm_multi: float | PoseCol = 0.0
    pssm_log_odds_flag: bool | PoseCol = False
    pssm_bias_flag: bool | PoseCol = False

    batchable_params = [
        "fixed_positions_custom",
        "pssm_custom",
        "omit_AA_custom",
        "bias_AA_custom",
        "tied_positions_custom",
        "bias_by_res_custom",
    ]


@dataclass
class SampleSequenceParams:
    """Store top-level ``sample_seqs.py`` YAML configuration fields.

    Attributes
    ----------
    dev : str or PoseCol
        Device string passed to PottsMPNN, usually ``"cuda"`` or ``"cpu"``.
    out_dir : str or PoseCol
        Output directory written by the runner.
    out_name : str or PoseCol
        Output basename written by the runner.
    input_list : str or PoseCol
        Path to the generated PottsMPNN input list.
    input_dir : str or PoseCol
        Directory containing staged input PDB files.
    chain_dict_json : str or None or PoseCol
        Optional upstream chain-design JSON path.
    chain_dict_custom : str or PoseCol
        ProtFlow helper for pose-specific chain dictionaries written to JSON.
    model : PottsMPNNModelParams
        Model checkpoint and architecture settings.
    inference : SampleSequenceInferenceParams
        Sequence-sampling and optimization settings.
    """
    dev: str | PoseCol = "cuda"
    out_dir: str | PoseCol = ""
    out_name: str | PoseCol = ""
    input_list: str | PoseCol = ""
    input_dir: str | PoseCol = ""
    chain_dict_json: str | None | PoseCol = None
    chain_dict_custom: str | PoseCol = ""
    model: PottsMPNNModelParams = field(default_factory=PottsMPNNModelParams)
    inference: SampleSequenceInferenceParams = field(default_factory=SampleSequenceInferenceParams)

    batchable_params = ["chain_dict_custom"]


@dataclass
class EnergyPredictionInferenceParams:
    """Store ``energy_prediction.py`` inference configuration fields.

    Attributes
    ----------
    ddG : bool or PoseCol
        If ``True``, output mutant minus wild-type energies.
    mean_norm : bool or PoseCol
        Whether to mean-center predicted mutation energies.
    max_tokens : int or PoseCol
        Token budget used by upstream batching.
    filter : bool or PoseCol
        Whether to return only mutants with experimental energies.
    binding_energy_json : str or None or PoseCol
        Path to upstream binding-energy partition JSON.
    binding_energy_custom : str or PoseCol
        ProtFlow helper for pose-specific binding-energy JSON payloads.
    binding_energy_cutoff : float or PoseCol
        Interface cutoff in Angstrom.
    skip_gaps : bool or PoseCol
        Whether upstream parsing should skip structural gaps.
    noise : float or PoseCol
        Coordinate noise added during inference.
    chain_dict : str or None or PoseCol
        Optional upstream chain dictionary setting.
    chain_ranges : str or None or PoseCol
        Optional JSON path used by upstream heatmap plotting.
    exclude_chains : list of str or None or PoseCol
        Chains excluded from mutation-energy scoring.
    """
    ddG: bool | PoseCol = True
    mean_norm: bool | PoseCol = False
    max_tokens: int | PoseCol = 20000
    filter: bool | PoseCol = False
    binding_energy_json: str | None | PoseCol = None
    binding_energy_custom: str | PoseCol = ""
    binding_energy_cutoff: float | PoseCol = 8
    skip_gaps: bool | PoseCol = False
    noise: float | PoseCol = 0.0
    chain_dict: str | None | PoseCol = None
    chain_ranges: str | None | PoseCol = None
    exclude_chains: list[str] | None | PoseCol = None

    batchable_params = ["binding_energy_custom"]


@dataclass
class EnergyPredictionParams:
    """Store top-level ``energy_prediction.py`` YAML configuration fields.

    Attributes
    ----------
    dev : str or PoseCol
        Device string passed to PottsMPNN, usually ``"cuda"`` or ``"cpu"``.
    out_dir : str or PoseCol
        Output directory written by the runner.
    out_name : str or PoseCol
        Output basename written by the runner.
    input_list : str or PoseCol
        Path to the generated PottsMPNN input list.
    input_dir : str or PoseCol
        Directory containing staged input PDB files.
    mutant_fasta : str or None or PoseCol
        FASTA file defining mutants to score. If both mutant inputs are
        ``None``, upstream PottsMPNN performs a deep mutational scan.
    mutant_csv : str or None or PoseCol
        CSV file defining mutants to score.
    model : PottsMPNNModelParams
        Model checkpoint and architecture settings.
    inference : EnergyPredictionInferenceParams
        Mutation-energy prediction settings.
    """
    dev: str | PoseCol = "cuda"
    out_dir: str | PoseCol = ""
    out_name: str | PoseCol = ""
    input_list: str | PoseCol = ""
    input_dir: str | PoseCol = ""
    mutant_fasta: str | None | PoseCol = None
    mutant_csv: str | None | PoseCol = None
    model: PottsMPNNModelParams = field(default_factory=PottsMPNNModelParams)
    inference: EnergyPredictionInferenceParams = field(default_factory=EnergyPredictionInferenceParams)


class PottsMPNNParamsBase:
    """Share YAML config helpers across typed PottsMPNN parameter classes.

    This mixin is inherited by :class:`SampleSequencePottsMPNNParams` and
    :class:`EnergyPredictionPottsMPNNParams`. It is not intended to be
    instantiated directly.

    Attributes
    ----------
    script : str
        Normalized upstream script key expected by :class:`PottsMPNN`.
    """
    script: ClassVar[str]

    def _compile_attrs_dict(self, flat: bool = False) -> dict[str, Any]:
        """Return parameter values as a nested or flattened dictionary.

        Parameters
        ----------
        flat : bool, optional
            If ``True``, return dot-separated keys for nested fields.

        Returns
        -------
        dict
            Parameter values including ProtFlow ``*_custom`` helper fields.
        """
        if flat:
            return {".".join(path): value for path, value, _ in _iter_param_values(self)}
        return _params_to_dict(self, include_custom=True)

    def _non_batchable_attrs(self) -> list[Any]:
        """Return parameter values that prevent batched execution.

        Returns
        -------
        list
            Values from fields that are not declared as batch-compatible.
        """
        return [value for _, value, is_batchable in _iter_param_values(self) if not is_batchable]

    def _params_are_batchable(self) -> bool:
        """Return whether all ``PoseCol`` values can be materialized per batch.

        Returns
        -------
        bool
            ``True`` if all pose-specific values are stored in batch-compatible
            fields.
        """
        return not any(isinstance(value, PoseCol) for value in self._non_batchable_attrs())

    def resolve_pose_cols_batched(self, poses: Poses, n_batches: int, work_dir: str) -> list[str]:
        """Write batched configs while materializing batch-compatible ``PoseCol`` values.

        Parameters
        ----------
        poses : Poses
            Input poses whose dataframe columns may be referenced by
            :class:`PoseCol`.
        n_batches : int
            Maximum number of config batches to create.
        work_dir : str
            Runner work directory.

        Returns
        -------
        list of str
            Paths to generated YAML config files.
        """
        # validate PoseCol references
        self._check_pose_cols(poses)

        # split poses into job-sized batches
        batches = _split_pose_dataframe(poses=poses, n_batches=n_batches)

        config_files = []
        for i, pose_batch in enumerate(batches, start=1):
            # stage batch input PDBs and metadata
            batch_params = copy.deepcopy(self)
            batch_dir = os.path.abspath(os.path.join(work_dir, f"batch_{i}"))
            batch_input_dir = os.path.join(batch_dir, "input_pdbs")
            json_dir = os.path.join(batch_dir, "json_files")
            os.makedirs(batch_input_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            for pose_path in pose_batch["poses"].to_list():
                shutil.copy(pose_path, os.path.join(batch_input_dir, os.path.basename(pose_path)))

            input_list_fn = os.path.join(batch_dir, "input_list.txt")
            pose_descriptions = pose_batch["poses_description"].to_list()
            _write_lines(input_list_fn, pose_descriptions)

            batch_params.out_dir = os.path.join(batch_dir, "outputs") #pylint: disable=w0201
            batch_params.out_name = f"batch_{i}" #pylint: disable=w0201
            batch_params.input_list = input_list_fn #pylint: disable=w0201
            batch_params.input_dir = batch_input_dir #pylint: disable=w0201

            # materialize PoseCol-backed JSON files
            for path, value, is_batchable in _iter_param_values(self):
                if not isinstance(value, PoseCol):
                    continue
                if not is_batchable:
                    raise ValueError(f"Internal error: non-batchable PoseCol reached batched setup: {'.'.join(path)}")
                json_path = os.path.join(json_dir, f"batch_{i}_{'_'.join(path)}.json")
                _write_json(json_path, {row["poses_description"]: row[str(value)] for _, row in pose_batch.iterrows()})
                _set_nested_attr(batch_params, _custom_path_to_json_path(path), json_path)

            # write final YAML
            config_path = os.path.join(batch_dir, "config.yaml")
            batch_params.to_yaml(config_path)
            config_files.append(config_path)

        return config_files

    def resolve_pose_cols(self, poses: Poses, n_batches: int, work_dir: str) -> tuple[bool, list[str]]:
        """Return batch mode and generated config paths.

        Parameters
        ----------
        poses : Poses
            Input poses used for config generation.
        n_batches : int
            Maximum number of batch configs to write.
        work_dir : str
            Runner work directory.

        Returns
        -------
        tuple
            ``(batched, config_files)`` where ``batched`` records whether the
            configs represent pose batches.
        """
        # choose batched mode only when every PoseCol can be encoded per batch
        if self._params_are_batchable():
            return True, self.resolve_pose_cols_batched(poses=poses, n_batches=n_batches, work_dir=work_dir)
        return False, self.resolve_pose_cols_unbatched(poses=poses, work_dir=work_dir)

    def resolve_pose_cols_unbatched(self, poses: Poses, work_dir: str) -> list[str]:
        """Write one config per pose for non-batchable ``PoseCol`` values.

        Parameters
        ----------
        poses : Poses
            Input poses used for config generation.
        work_dir : str
            Runner work directory.

        Returns
        -------
        list of str
            Paths to generated YAML config files.
        """
        # validate PoseCol references
        self._check_pose_cols(poses)

        # setup output directories
        config_dir = os.path.join(work_dir, "config_files")
        input_list_dir = os.path.join(work_dir, "input_lists")
        json_dir = os.path.join(work_dir, "json_files")
        output_dir = os.path.join(work_dir, "outputs")
        for path in (config_dir, input_list_dir, json_dir, output_dir):
            os.makedirs(path, exist_ok=True)

        # write one config per pose
        config_files = []
        for pose in poses:
            pose_params = copy.deepcopy(self)
            desc = pose["poses_description"]

            input_list = os.path.join(input_list_dir, f"{desc}_input_list.txt")
            _write_lines(input_list, [desc])

            pose_params.out_dir = output_dir #pylint: disable=w0201
            pose_params.out_name = desc #pylint: disable=w0201
            pose_params.input_list = input_list #pylint: disable=w0201
            pose_params.input_dir = os.path.dirname(pose["poses"]) #pylint: disable=w0201

            # resolve PoseCol values for this pose
            for path, value, _ in _iter_param_values(self):
                if not isinstance(value, PoseCol):
                    continue
                if path[-1].endswith("_custom"):
                    json_path = os.path.join(json_dir, f"{desc}_{'_'.join(path)}.json")
                    _write_json(json_path, {desc: pose[str(value)]})
                    _set_nested_attr(pose_params, _custom_path_to_json_path(path), json_path)
                else:
                    _set_nested_attr(pose_params, path, pose[str(value)])

            config_path = os.path.join(config_dir, f"{desc}_config.yaml")
            pose_params.to_yaml(config_path)
            config_files.append(config_path)

        return config_files

    def to_yaml(self, out_path: str) -> None:
        """Write this parameter set as a PottsMPNN YAML config.

        Parameters
        ----------
        out_path : str
            Destination YAML path.
        """
        # exclude *_custom helpers from upstream YAML
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="UTF-8") as handle:
            yaml.safe_dump(_params_to_dict(self, include_custom=False), handle, sort_keys=False)

    def _check_pose_cols(self, poses: Poses) -> None:
        """Validate that all ``PoseCol`` references exist in ``poses.df``.

        Parameters
        ----------
        poses : Poses
            Input poses whose dataframe is checked.

        Raises
        ------
        KeyError
            If any referenced dataframe column is missing.
        """
        # collect missing dataframe columns once for a clear error
        missing = sorted({str(value) for _, value, _ in _iter_param_values(self) if isinstance(value, PoseCol)} - set(poses.df.columns))
        if missing:
            raise KeyError(f"PoseCol column(s) not found in poses.df: {missing}")


@dataclass
class SampleSequencePottsMPNNParams(SampleSequenceParams, PottsMPNNParamsBase):
    """Typed params for ``sample_seqs.py``.

    Use this class when running :meth:`PottsMPNN.run` with
    ``script="sample_seqs"`` or the default script.

    Examples
    --------
    >>> params = SampleSequencePottsMPNNParams()
    >>> params.inference.num_samples = 8
    >>> params.inference.temperature = 0.2
    """
    script: ClassVar[str] = "sample_seqs"


@dataclass
class EnergyPredictionPottsMPNNParams(EnergyPredictionParams, PottsMPNNParamsBase):
    """Typed params for ``energy_prediction.py``.

    Use this class when running :meth:`PottsMPNN.run` with
    ``script="energy_prediction"``.

    Examples
    --------
    >>> params = EnergyPredictionPottsMPNNParams(mutant_csv="mutations.csv")
    >>> params.inference.ddG = True
    """
    script: ClassVar[str] = "energy_prediction"


def params_to_config(
    poses: Poses,
    n_batches: int,
    work_dir: str,
    params: SampleSequencePottsMPNNParams | EnergyPredictionPottsMPNNParams,
) -> tuple[bool, list[str]]:
    """Generate PottsMPNN config files and report whether they are batched.

    Parameters
    ----------
    poses : Poses
        Input poses for this run.
    n_batches : int
        Maximum number of batch configs to write.
    work_dir : str
        Runner work directory.
    params : SampleSequencePottsMPNNParams or EnergyPredictionPottsMPNNParams
        Typed params object for the selected upstream script.

    Returns
    -------
    tuple
        ``(batched, config_files)`` where ``config_files`` contains YAML paths.

    Raises
    ------
    ValueError
        If ``params`` is ``None``.
    """
    # params are required so script-specific defaults are explicit
    if params is None:
        raise ValueError("PottsMPNN params must not be None.")
    return params.resolve_pose_cols(poses=poses, n_batches=n_batches, work_dir=work_dir)


def fasta_to_df(fasta_file: str, desc_col_name: str = "description", seq_col_name: str = "sequence") -> pd.DataFrame:
    """Parse a FASTA file into description and sequence columns.

    Parameters
    ----------
    fasta_file : str
        FASTA file to parse.
    desc_col_name : str, optional
        Name of the output description column.
    seq_col_name : str, optional
        Name of the output sequence column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per FASTA record.
    """
    # parse records manually to avoid adding another parser dependency
    records = []
    description = None
    seq_chunks: list[str] = []
    with open(fasta_file, "r", encoding="UTF-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if description is not None:
                    records.append({desc_col_name: description, seq_col_name: "".join(seq_chunks)})
                description = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if description is not None:
        records.append({desc_col_name: description, seq_col_name: "".join(seq_chunks)})
    return pd.DataFrame(records, columns=[desc_col_name, seq_col_name])


def collect_scores_sample_seqs(work_dir: str, batched: bool, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Collect ``sample_seqs.py`` FASTA and loss outputs into score rows.

    The collector reads raw sampled FASTA files, optional optimized FASTA
    files, and ``*_av_loss.csv`` metrics. It writes one FASTA file per
    returned row under ``output_fastas`` so :class:`RunnerOutput` can treat
    sampled sequences as pose locations.

    Parameters
    ----------
    work_dir : str
        Runner work directory containing generated configs and PottsMPNN
        outputs.
    batched : bool
        Whether the run used batched configs.
    include_scores : list of str, optional
        Reserved for API consistency; currently ignored.

    Returns
    -------
    pandas.DataFrame
        Score rows with normalized ``description`` values, sequences, optional
        PottsMPNN loss metrics, and per-sequence FASTA ``location`` paths.

    Raises
    ------
    FileNotFoundError
        If no sample FASTA outputs are found.
    """
    # include_scores is unused because sample outputs are scalar and FASTA-based
    del include_scores

    # discover output directories from generated configs
    configs = _load_run_configs(work_dir)
    out_dirs = _output_dirs_from_configs(configs, batched=batched)
    av_loss_files = _glob_output_files(out_dirs, "*_av_loss.csv")
    raw_seq_files = [
        path for path in _glob_output_files(out_dirs, "*.fasta")
        if "_optimized_" not in os.path.splitext(os.path.basename(path))[0]
    ]
    optimized_seq_files = _glob_output_files(out_dirs, "*_optimized_*.fasta")

    if not raw_seq_files and not optimized_seq_files:
        raise FileNotFoundError(f"No PottsMPNN FASTA outputs found under {work_dir}.")

    # parse raw sequences, optimized sequences, and loss metrics
    raw_df = _read_sample_fastas(raw_seq_files, configs, "sequence")
    optimized_df = _read_sample_fastas(optimized_seq_files, configs, "optimized_potts_sequence")
    av_loss_df = _read_av_loss_files(av_loss_files, configs)

    # prefer optimized sequences as output poses when present
    if not optimized_df.empty and not raw_df.empty:
        scores = raw_df.merge(optimized_df, on=["raw_description", "description", "sample_idx"], how="outer")
    elif not optimized_df.empty:
        scores = optimized_df
    else:
        scores = raw_df

    if not av_loss_df.empty:
        scores = scores.merge(av_loss_df, on=["raw_description", "description", "sample_idx"], how="left")

    # write per-sequence FASTA files for RunnerOutput locations
    fasta_output_dir = os.path.join(work_dir, "output_fastas")
    os.makedirs(fasta_output_dir, exist_ok=True)
    seq_col = "optimized_potts_sequence" if "optimized_potts_sequence" in scores.columns else "sequence"
    locations = []
    for _, row in scores.iterrows():
        fasta_path = os.path.abspath(os.path.join(fasta_output_dir, f"{row['description']}.fa"))
        with open(fasta_path, "w", encoding="UTF-8") as handle:
            handle.write(f">{row['description']}\n{row[seq_col]}\n")
        locations.append(fasta_path)
    scores["location"] = locations
    return scores


def collect_scores_energy_prediction(work_dir: str, batched: bool, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Collect ``energy_prediction.py`` CSV outputs into score rows.

    PottsMPNN can emit many mutation rows per input pose. The collector stores
    the full per-pose mutation table as JSON sidecars under ``output_scores``
    and returns one ProtFlow score row per input structure.

    Parameters
    ----------
    work_dir : str
        Runner work directory containing generated configs and PottsMPNN
        outputs.
    batched : bool
        Whether the run used batched configs.
    include_scores : list of str, optional
        Reserved for API consistency; currently ignored.

    Returns
    -------
    pandas.DataFrame
        One row per input pose with the JSON sidecar path and number of scored
        mutations.

    Raises
    ------
    FileNotFoundError
        If no ``*_scores.csv`` files are found.
    """
    # include_scores is unused because full per-pose CSV rows are stored sidecar-style
    del include_scores
    configs = _load_run_configs(work_dir)
    out_dirs = _output_dirs_from_configs(configs, batched=batched)
    score_files = _glob_output_files(out_dirs, "*_scores.csv")
    if not score_files:
        raise FileNotFoundError(f"No PottsMPNN energy prediction score files found under {work_dir}.")

    # split combined score CSVs into per-pose JSON sidecars
    output_dir = os.path.join(work_dir, "output_scores")
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for score_file in score_files:
        score_df = pd.read_csv(score_file)
        stats_file = score_file.replace("_scores.csv", "_stats.csv")
        for desc in score_df["pdb"].unique():
            pose_df = score_df[score_df["pdb"] == desc]
            pose_scorefile = os.path.abspath(os.path.join(output_dir, f"{desc}.json"))
            pose_df.to_json(pose_scorefile, orient="records")
            row = {
                "description": desc,
                "energy_prediction_scorefile": pose_scorefile,
                "energy_prediction_n_mutations": len(pose_df.index),
            }
            if os.path.isfile(stats_file):
                stats_df = pd.read_csv(stats_file)
                pose_stats = stats_df[stats_df["pdb"] == desc]
                if not pose_stats.empty:
                    row["energy_prediction_pearson_r"] = pose_stats.iloc[0].get("Pearson r")
            rows.append(row)

    return pd.DataFrame(rows)


def collect_scores(work_dir: str, script: str, batched: bool, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Dispatch score collection for the selected PottsMPNN script.

    Parameters
    ----------
    work_dir : str
        Runner work directory.
    script : str
        Script alias used for the run.
    batched : bool
        Whether the run used batched configs.
    include_scores : list of str, optional
        Optional collector-specific score selection.

    Returns
    -------
    pandas.DataFrame
        Score rows produced by the script-specific collector.

    Raises
    ------
    NotImplementedError
        If score collection is not implemented for ``script``.
    """
    # map script aliases to collector functions
    script_key = _script_key(script)
    collectors = {
        "sample_seqs": collect_scores_sample_seqs,
        "energy_prediction": collect_scores_energy_prediction,
    }
    if script_key not in collectors:
        raise NotImplementedError(f"Score collection is not implemented for PottsMPNN script: {script}")
    return collectors[script_key](work_dir=work_dir, batched=batched, include_scores=include_scores)


def _script_key(script: str | None) -> str:
    """Normalize a script path or alias to its basename key."""
    if not script:
        return "sample_seqs"
    return os.path.splitext(os.path.basename(str(script)))[0]


def _check_sample_descriptions(poses: Poses) -> None:
    """Reject input names that collide with sample output suffixes."""
    # optimized output suffixes are reserved by upstream PottsMPNN
    bad_suffixes = ("_optimized_potts", "_optimized_nodes")
    bad = [desc for desc in poses.df["poses_description"].to_list() if str(desc).endswith(bad_suffixes)]
    if bad:
        raise ValueError(
            "PottsMPNN sample output parsing reserves descriptions ending in "
            f"{bad_suffixes}. Rename these poses first: {bad}"
        )


def _fill_missing_locations(scores: pd.DataFrame, poses: Poses, index_layers: int) -> pd.DataFrame:
    """Map locationless score rows back to the input pose paths."""
    # score collectors may omit locations when the input pose remains active
    if "location" in scores.columns:
        return scores
    scores = scores.copy()
    select_col = scores["description"].astype(str)
    if index_layers:
        select_col = select_col.str.split("_").str[:-index_layers].str.join("_")
    pose_locations = poses.df.set_index("poses_description")["poses"]
    scores["location"] = select_col.map(pose_locations)
    if scores["location"].isna().any():
        missing = scores.loc[scores["location"].isna(), "description"].to_list()
        raise ValueError(f"Could not map PottsMPNN score rows back to input poses: {missing}")
    return scores


def _split_pose_dataframe(poses: Poses, n_batches: int) -> list[pd.DataFrame]:
    """Split poses.df into up to n_batches dataframe chunks."""
    # empty poses produce no config batches
    if len(poses) == 0:
        return []
    batches = split_list(list(poses.df.index), n_sublists=max(1, n_batches or 1))
    return [poses.df.loc[batch].reset_index(drop=True) for batch in batches]


def _iter_param_values(obj: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any, bool]]:
    """Yield nested parameter paths, values, and batchability flags."""
    # recurse through dataclass parameter containers
    out = []
    if is_dataclass(obj):
        names = [param_field.name for param_field in fields(obj)]
        batchable = set(getattr(obj, "batchable_params", []))
    else:
        return out

    for name in names:
        value = getattr(obj, name)
        next_path = path + (name,)
        if is_dataclass(value):
            out.extend(_iter_param_values(value, next_path))
        else:
            out.append((next_path, value, name in batchable))
    return out


def _params_to_dict(obj: Any, include_custom: bool) -> dict[str, Any]:
    """Convert parameter objects into YAML-serializable dictionaries."""
    # unwrap dataclass fields
    source = {param_field.name: getattr(obj, param_field.name) for param_field in fields(obj)}
    out = {}
    for key, value in source.items():
        if not include_custom and key.endswith("_custom"):
            continue
        if is_dataclass(value):
            out[key] = _params_to_dict(value, include_custom=include_custom)
        elif isinstance(value, PoseCol):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _set_nested_attr(obj: Any, path: tuple[str, ...], value: Any) -> None:
    """Set a nested attribute on a parameter object."""
    # walk to the owning nested object before assignment
    target = obj
    for name in path[:-1]:
        target = getattr(target, name)
    setattr(target, path[-1], value)


def _custom_path_to_json_path(path: tuple[str, ...]) -> tuple[str, ...]:
    """Convert a *_custom parameter path to its paired *_json path."""
    # upstream expects JSON path fields, not ProtFlow custom helper fields
    leaf = path[-1]
    if not leaf.endswith("_custom"):
        raise ValueError(f"Expected custom parameter path, got {'.'.join(path)}")
    return path[:-1] + (f"{leaf[:-len('_custom')]}_json",)


def _write_lines(path: str, lines: list[str]) -> None:
    """Write text lines to a file, creating parent directories."""
    # ensure parent directory exists before writing
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="UTF-8") as handle:
        handle.write("\n".join(lines))


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON payload to a file, creating parent directories."""
    # ensure parent directory exists before writing
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="UTF-8") as handle:
        json.dump(payload, handle)


def _load_run_configs(work_dir: str) -> list[dict[str, Any]]:
    """Load generated PottsMPNN YAML configs from a work directory."""
    # configs live in either unbatched config_files or batch_* directories
    config_files = sorted(glob(os.path.join(work_dir, "config_files", "*.yaml")))
    config_files += sorted(glob(os.path.join(work_dir, "batch_*", "config.yaml")))
    configs = []
    for config_file in config_files:
        with open(config_file, "r", encoding="UTF-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        cfg["_config_path"] = config_file
        cfg["_input_descriptions"] = _read_input_descriptions(cfg.get("input_list"))
        cfg["out_dir"] = os.path.abspath(cfg["out_dir"])
        configs.append(cfg)
    return configs


def _read_input_descriptions(input_list: str | None) -> list[str]:
    """Read base pose descriptions from a PottsMPNN input list."""
    if not input_list:
        return []
    with open(input_list, "r", encoding="UTF-8") as handle:
        return [line.strip().split("|", maxsplit=1)[0] for line in handle if line.strip()]


def _output_dirs_from_configs(configs: list[dict[str, Any]], batched: bool) -> list[str]:
    """Return unique output directories declared by generated configs."""
    # output directories are encoded in YAML for both batched and unbatched runs
    del batched
    return sorted({cfg["out_dir"] for cfg in configs})


def _glob_output_files(out_dirs: list[str], pattern: str) -> list[str]:
    """Find output files matching a pattern across output directories."""
    return sorted(path for out_dir in out_dirs for path in glob(os.path.join(out_dir, pattern)))


def _config_for_output_file(path: str, configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the generated config that produced an output file."""
    # match by output directory and configured out_name prefix
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.abspath(os.path.dirname(path))
    matches = []
    for cfg in configs:
        if os.path.abspath(cfg["out_dir"]) != out_dir:
            continue
        out_name = str(cfg["out_name"])
        if stem == out_name or stem == f"{out_name}_av_loss" or stem.startswith(f"{out_name}_optimized_") or stem == f"{out_name}_scores":
            matches.append(cfg)
    if not matches:
        raise ValueError(f"Could not match PottsMPNN output file to a generated config: {path}")
    return max(matches, key=lambda cfg: len(str(cfg["out_name"])))


def _canonical_sample_description(raw_description: str, input_descriptions: list[str]) -> tuple[str, int]:
    """Convert PottsMPNN sample names to ProtFlow merge descriptions."""
    # prefer the longest input description to handle names containing underscores
    input_descriptions = sorted(input_descriptions, key=len, reverse=True)
    for input_desc in input_descriptions:
        if raw_description == input_desc:
            return f"{input_desc}_0001", 1
        prefix = f"{input_desc}_"
        if raw_description.startswith(prefix):
            suffix = raw_description[len(prefix):]
            if suffix.isdigit():
                sample_idx = int(suffix) + 1
                return f"{input_desc}_{sample_idx:04d}", sample_idx
    if raw_description.rsplit("_", maxsplit=1)[-1].isdigit():
        base, idx = raw_description.rsplit("_", maxsplit=1)
        return f"{base}_{int(idx) + 1:04d}", int(idx) + 1
    return f"{raw_description}_0001", 1


def _read_sample_fastas(files: list[str], configs: list[dict[str, Any]], seq_col: str) -> pd.DataFrame:
    """Read sample or optimized FASTA outputs into normalized rows."""
    # normalize upstream sample names before merging into poses
    rows = []
    for fasta_file in files:
        cfg = _config_for_output_file(fasta_file, configs)
        df = fasta_to_df(fasta_file, desc_col_name="raw_description", seq_col_name=seq_col)
        for _, row in df.iterrows():
            description, sample_idx = _canonical_sample_description(row["raw_description"], cfg["_input_descriptions"])
            rows.append({
                "raw_description": row["raw_description"],
                "description": description,
                "sample_idx": sample_idx,
                seq_col: row[seq_col],
            })
    return pd.DataFrame(rows)


def _read_av_loss_files(files: list[str], configs: list[dict[str, Any]]) -> pd.DataFrame:
    """Read average-loss CSV files into normalized score rows."""
    # keep upstream metrics while replacing pdb with normalized descriptions
    rows = []
    for av_loss_file in files:
        cfg = _config_for_output_file(av_loss_file, configs)
        df = pd.read_csv(av_loss_file)
        for _, row in df.iterrows():
            raw_description = row["pdb"]
            description, sample_idx = _canonical_sample_description(raw_description, cfg["_input_descriptions"])
            out = row.to_dict()
            out["raw_description"] = raw_description
            out["description"] = description
            out["sample_idx"] = sample_idx
            rows.append(out)
    return pd.DataFrame(rows)
