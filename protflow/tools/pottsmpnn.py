"""Runner template for ProtFlow tool integrations.

How to use this template
------------------------
1. Copy this file (or class) to a new module in ``protflow/tools`` or ``protflow/metrics``.
2. Rename ``ExampleRunner`` and ``example_runner`` to your tool name.
3. Replace all ``TODO`` markers.
4. Keep the run lifecycle intact:
   setup workdir -> reuse cached outputs -> prep options -> build commands -> run jobs -> collect scores -> return RunnerOutput.
5. Implement ``collect_scores(...)`` as a **module function**.

Design goals
------------
- Keep runner behavior consistent across ProtFlow.
- Make it obvious where tool-specific logic belongs.
- Avoid re-implementing common logic already provided by ``Runner``.
- Keep score parsing callable without constructing a runner instance.
"""

# imports
from __future__ import annotations
import json
import logging
import os
from glob import glob
from dataclasses import dataclass
import shutil
import copy

# dependencies
import pandas as pd
import yaml

# customs
from protflow import load_config_path, require_config
from protflow.jobstarters import JobStarter
from protflow.poses import Poses
from protflow.runners import (
    Runner,
    RunnerOutput,
    parse_generic_options,
    options_flags_to_string,
    prepend_cmd,
)

class PottsMPNN(Runner):
    """Template class for implementing a new ProtFlow runner.

    Developer instructions
    ----------------------
    - Keep this class focused on one external tool.
    - Put all user-facing run parameters on ``run(...)``.
    - Use config values as defaults in ``__init__``.
    - Keep score parsing in the module-level ``collect_scores(...)`` function.
    - Ensure output parsing returns a dataframe with:
      - ``description``: basename without extension of each produced pose
      - ``location``: absolute path to produced pose file
    - Always return ``RunnerOutput(...).return_poses()``.
    """

    def __init__(
        self,
        application_path: str | None = None,
        python_path: str | None = None,
        pre_cmd: str | None = None,
        jobstarter: JobStarter | None = None,
    ) -> None:
        """Initialize tool paths and static runner metadata.

        Developer instructions
        ----------------------
        - Load paths from config by default.
        - Keep constructor lightweight; do not run jobs here.
        - Define ``self.index_layers`` according to output naming:
          - ``0`` if output descriptions match input pose descriptions.
          - ``>0`` if your tool appends index layers like ``_0001``.
        """
        # config required
        config = require_config()

        # setup config.
        self.pottsmpnn_dir = pottsmpnn_dir or load_config_path(config, "POTSSMPNN_DIR")
        self.python_path = python_path or load_config_path(config, "POTTSMPNN_PYTHON")
        self.pre_cmd = pre_cmd or load_config_path(config, "POTTSMPNN_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "pottsmpnn"
        self.index_layers = 1

    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        script: str = None,
        params: PottsMPNNParams = None,
        options: str | None = None,
        pose_options: str|list[str] = None,
        include_scores: list[str] | None = None,
        overwrite: bool = False,
    ) -> Poses:
        """Execute the full runner lifecycle and merge results into ``poses``.

        Developer instructions
        ----------------------
        The canonical order is:
        1. Generic setup (prefix check, jobstarter resolution, workdir creation).
        2. Reuse cached scorefile when available and ``overwrite=False``.
        3. Prepare per-pose options.
        4. Build command list.
        5. Execute via selected jobstarter.
        6. Collect scores into required dataframe format.
        7. Save runner scorefile and merge with ``RunnerOutput``.

        Notes on ``include_scores``
        ---------------------------
        ``include_scores`` is passed through to module-level ``collect_scores``.
        Use it to opt into heavy optional score fields (e.g., per-residue vectors
        or 2D matrices) that should not be loaded by default.
        """
        # sanity
        script = self._resolve_script(script)
        
        # 1) Generic setup shared by all runners.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses", self, work_dir, len(poses))

        # 2) Scorefile reuse shortcut.
        scorefile = os.path.join(work_dir, f"{self.name}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Reusing existing scorefile: %s", scorefile)
            return RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

        # Optional cleanup when overwrite is requested.
        if overwrite:
            self._cleanup_previous_outputs(work_dir=work_dir)

        # 3) Prep config files from specified params and pose options
        pose_options_list = self.prep_pose_options(poses=poses, pose_options=pose_options)
        config_files_list = params_to_config(poses=poses, prefix=prefix, params=params, work_dir=work_dir)

        # 4) Build commands.
        cmds = self._build_commands(
            script=script,
            config_files=config_files_list,
            work_dir=work_dir,
            options=options,
        )

        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        # 5) Execute commands.
        jobstarter.start(
            cmds=cmds,
            jobname=self.name,
            wait=True,
            output_path=work_dir,
        )

        # 6) Collect and validate scores (module function, by convention).
        scores = collect_scores(work_dir=work_dir, include_scores=include_scores)

        if len(scores.index) == 0:
            raise RuntimeError(f"{self}: collect_scores returned no rows. Check runner output logs and runner output directory ({work_dir})")

        # 7) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    def _resolve_script(self, script: str) -> str:
        '''quick helper to resolve the script path for pottsmpnn scripts.'''
        if os.path.isfile(script):
            return script
        if os.path.isfile(os.path.join(self.pottsmpnn_dir, script)):
            return os.path.join(self.pottsmpnn_dir, script)
        raise FileNotFoundError(f"File not found by itself or in pottsmpnn_dir: {script} {os.path.join(self.pottsmpnn_dir, script)}")

    def _build_commands(
        self,
        script: str,
        config_files: list[str],
        work_dir: str,
        options: str | None,
    ) -> list[str]:
        """Create one shell command per pose.

        Developer instructions
        ----------------------
        - Convert a global options string + per-pose options into final CLI options.
        - Keep all command assembly in one place to simplify debugging.
        - Return a list with deterministic order matching ``poses`` rows.
        """
        # sanity
        options = options or ""

        # setup output directory
        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for config_file in config_files.poses_list():
            cmds.append(self.write_cmd(script=script, config_path=config_file, cli_args=options))
        return cmds

    def write_cmd(self, script: str, config_path: str, cli_args: str) -> str:
        """Return the exact shell command for one pose.

        Developer instructions
        ----------------------
        - Build an executable command string only; do not execute here.
        - Ensure output filename preserves or predictably derives from pose description.
        - Keep quoting robust for paths with spaces.
        """
        cmd = (
            f"{self.python_path} {script} "
            f"--config '{config_path}' {cli_args}"
        )
        # TODO: replace with your tool's real command structure.
        return cmd

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete/clear runner-specific output artifacts before rerun.

        Developer instructions
        ----------------------
        - Keep cleanup scoped to this runner's own output directories.
        - Never remove unrelated files outside ``work_dir``.
        - This method is optional but useful for tools that append stale outputs.
        """
        if not os.path.isdir(work_dir):
            return

        for file_path in glob(os.path.join(work_dir, "*")):
            os.remove(file_path)

def _is_heavy_value(value: object) -> bool:
    """Heuristic for values that can bloat score tables (2D/per-residue objects)."""
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple, dict)):
            return True
        if len(value) > 200:
            return True

    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return True

    return False


def _extract_score_dict(
    payload: dict,
    include_scores: set[str],
    prefix: str = "",
) -> dict[str, object]:
    """Flatten nested score dictionaries with optional inclusion of heavy values.

    Notes
    -----
    - Do not hardcode score names where possible; parse what is present.
    - By default, this returns scalar values and skips heavy values.
    - Heavy values are included only if their key (or flattened key path) is in
      ``include_scores``.
    """
    out: dict[str, object] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(_extract_score_dict(value, include_scores, prefix=flat_key))
            continue

        if _is_heavy_value(value):
            if key in include_scores or flat_key in include_scores:
                out[flat_key] = json.dumps(value)
            continue

        out[flat_key] = value

    return out

def collect_scores(work_dir: str, include_scores: list[str] | None = None) -> pd.DataFrame:
    """Parse runner outputs and return the canonical scores dataframe.

    Developer instructions
    ----------------------
    - Keep this function at module scope, not inside the runner class.
    - Required output columns:
      - ``description``
      - ``location``
    - Favor score auto-discovery (read keys present in outputs) over hardcoded
      column lists, because external tools frequently rename score terms.
    - Avoid reading heavy per-residue / matrix-like data by default.
      Use ``include_scores`` to opt-in to specific heavy fields.
    - Keep function callable standalone (for debugging/re-parsing old runs).
    """
    include_set = set(include_scores or [])
    output_dir = os.path.join(work_dir, "outputs")
    output_paths = sorted(
        glob(os.path.join(output_dir, "*.pdb")) +
        glob(os.path.join(output_dir, "*.cif"))
    )

    rows: list[dict[str, object]] = []
    for path in output_paths:
        desc = os.path.splitext(os.path.basename(path))[0]
        row: dict[str, object] = {
            "description": desc,
            "location": os.path.abspath(path),
        }

        # TODO: Adjust sidecar discovery for your tool's naming convention.
        sidecars = sorted(glob(os.path.join(output_dir, f"{desc}*.json")))
        for sidecar in sidecars:
            with open(sidecar, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)

            if isinstance(parsed, dict):
                row.update(_extract_score_dict(parsed, include_set))

        rows.append(row)

    return pd.DataFrame(rows)


# TODO @Codex I intend to designate PoseCol as a class. This is to allow PottsMPNN params to check directly for the type of parameter it received. Depending on that, either the string or the poses.df[string] will be set for the config file.
class PoseCol(str):
    def __init__(self, col_name: str):
        '''wrapper class'''
        super.__init__()
        self.col_name = col_name

    def __str__(self):
        return self.col_name
    #TODO @Codex how do I specify that passing the PoseCol instance will automatically pass PoseCol.col_name instead of the PoseCol object ID (for everything)? Basically, this class should behave exactly like a string but just be of type PoseCol. Is the code I wrote for self.pose_col and __str__ even necessary?

@dataclass
class PottsMPNNModelParams:
    check_path: str = ""
    hidden_dim: int = 128
    edge_features: int = 128
    potts_dim: int = 400
    num_layers: int = 3
    num_edges: int = 48
    vocab: int = 21
        

@dataclass
class SampleSequenceInferenceParams:
    # sample_sequence params
    num_samples: int|PoseCol = 1
    temperature: int|PoseCol = 0.1
    noise: int|PoseCol = 0.1
    skip_gaps: bool|PoseCol = False
    fix_decoding_order: bool|PoseCol = True
    decoding_order_offset: int|PoseCol = 0
    optimization_mode: str|PoseCol = "potts"
    optimization_temperature: float|PoseCol = 0.0
    binding_energy_optimization: bool|PoseCol = None
    binding_energy_json: str|PoseCol = "null"
    binding_energy_cutoff: int|PoseCol = 8
    optimize_pdb: bool|PoseCol = False
    optimize_fasta: str|PoseCol = ''
    write_pdb: bool|PoseCol = True
    fixed_positions_json: str|PoseCol = ''
    pssm_json: str|PoseCol = ''
    omit_AA_json: str|PoseCol = ''
    bias_AA_json: str|PoseCol = ''
    tied_positions_json: str|PoseCol = ''
    bias_by_res_json: str|PoseCol = ''
    fixed_positions_custom: str|PoseCol = ''
    pssm_custom: str|PoseCol = ''
    omit_AA_custom: str|PoseCol = ''
    bias_AA_custom: str|PoseCol = ''
    tied_positions_custom: str|PoseCol = ''
    bias_by_res_custom: str|PoseCol = ''
    omit_AAs: list|PoseCol = []
    pssm_threshold: float|PoseCol = 0.0
    pssm_multi: float|PoseCol = 0.0
    pssm_log_odds_flag: bool|PoseCol = False
    pssm_bias_flag: bool|PoseCol = False

    # general param
    batchable_params = [
        "fixed_positions_custom", "pssm_custom", "omit_AA_custom",
        "bias_AA_custom", "tied_positions_custom", "bias_by_res_custom"
    ]

@dataclass
class SampleSequenceParams:
    dev: str | PoseCol = "cuda"
    #TODO @Codex: The params out_dir, out_name, and input_list should not be exposed to the user. Is there a way to hide them? (These are set by ProtFlow automatically)
    out_dir: str | PoseCol = ""
    out_name: str | PoseCol = ""
    input_list: str | PoseCol = ""
    chain_dict_json: str | PoseCol = "null"
    chain_dict_custom: str | PoseCol
    model: PottsMPNNModelParams = PottsMPNNModelParams()
    inference: SampleSequenceInferenceParams = SampleSequenceInferenceParams()

    # batchable
    batchable_params = ["chain_dict_custom"]

@dataclass
class EnergyPredictionInferenceParams:
    ddG: bool | PoseCol = True
    mean_norm: bool | PoseCol = False
    max_tokens: int | PoseCol = 20000
    filter: bool | PoseCol = False
    binding_energy_json: str | PoseCol = "null"
    binding_energy_custom: str | PoseCol = ""
    binding_energy_cutoff: int | PoseCol = 8
    skip_gaps: bool | PoseCol = False
    noise: float | PoseCol = 0.0
    chain_dict: str | PoseCol = "null"
    chain_ranges: str | PoseCol = "null"

    # batchable
    batchable_params = ["binding_energy_custom"]

@dataclass
class EnergyPredictionParams:
    dev: str | PoseCol = "cuda"
    out_dir: str | PoseCol = ""
    out_name: str | PoseCol = ""
    input_list: str | PoseCol = ""
    input_dir: str | PoseCol = ""
    mutant_fasta: str | PoseCol = "null"
    mutant_csv: str | PoseCol = "null"
    model: PottsMPNNModelParams = PottsMPNNModelParams()
    inference: EnergyPredictionInferenceParams = EnergyPredictionInferenceParams()

class PottsMPNNParams:
    PARAMS_DICT = {
        "sample_sequence": SampleSequenceParams,
        "energy_prediction": PottsMPNNModelParams
    }
    def __init__(self, script: str) -> None:
        '''
        Sets up PottsMPNNParams.
        type: {sample_sequence, energy_prediction} type of params to generate.
        '''
        self._set_attrs(script)
        self.script = script

    def _set_attrs(self, script: str):
        #TODO @Codex set attributes of this class -> combined dataclasses (-> should be directly parseable to .yaml)
        self.PARAMS_DICT[script]()


    def _compile_attrs_dict(self, flat: bool = False) -> dict:
        '''Function that creates a dictionary of the attributes of this class. Class nesting is carried over into dict nesting if flat=False.'''
        return attrs_dict #TODO @Codex: Please code this out.

    def _non_batchable_attrs(self) -> list:
        '''Returns list of attributes that are batchable'''
        # collect all non-batchable attrs recursively (check against list of batchable attributes)
        return non_batchable_attrs  #@Codex please fill code

    def _params_are_batchable(self) -> bool:
        '''Helper that checks whether this set of Parameters can be batched. (batching = combining config files of multiple poses)'''
        # iterate through attributes listed in 'is_batchable' and check if any of them is of type PoseCol
        return not any((isinstance(attr, PoseCol) for attr in self._non_batchable_attrs())) # all non-batchables must be non PoseCol type

    def resolve_pose_cols_batched(self, poses: Poses, n_batches: int, work_dir: str) -> list[str]:
        '''Converts PoseCol types in PottsMPNNParams into actual values. Returns list of .config files. Batches config files.'''
        # split poses in batches
        batch_df_list: list[pd.DataFrame] = _split_into_batches(poses) #TODO @Codex fill in whatever is the canonical way to do this

        # iterate over batches and convert PoseCol attributes into values denoted in poses.
        batch_config_files = []
        for i, pose_batch in enumerate(batch_df_list, start=1):
            # initialize new batch params
            batch_params = copy.deepcopy(self)

            # create batch input dir
            batch_dir = os.path.abspath(os.path.join(work_dir, f"batch_{i}"))
            batch_pdb_dir = os.path.join(batch_dir, "input_pdbs")
            list_of_poses = pose_batch["poses"].to_list()
            list_of_pose_descriptions = pose_batch["poses_description"].to_list()

            # move batch poses to batch_dir
            os.makedirs(batch_pdb_dir, exist_ok=True)
            for pose in list_of_poses:
                shutil.copy(pose, batch_pdb_dir)

            # write input_list.txt
            input_list_fn = os.path.join(batch_dir, "input_list.txt")
            with open(input_list_fn, 'w', encoding="UTF-8") as f:
                f.write("\n".join(list_of_pose_descriptions))

            # configure i/o params
            batch_params.out_dir = os.path.join(batch_dir, "outputs")
            batch_params.out_name = f"batch_{i}"
            batch_params.input_list = input_list_fn

            # now go through each attribute in params and convert any PoseCol into an input_json file.
            for attr, val in self._compile_attrs_dict(flat=True):
                # skip values that are already  
                if not isinstance(val, PoseCol):
                    continue
                # assign value from pose_col:

                # values for each pose must be written into a .json file. The .json file becomes the new attribute.
                batch_attr_dict = {pose["poses"]: pose[val] for _, pose in pose_batch.iterrows()}

                # write out batch json
                batch_attr_json_name = f"batch_{i}_{attr}.json"
                with open(batch_attr_json_name, 'w', encoding="UTF-8") as f:
                    json.dump(batch_attr_dict, f)

                # add to batch_params
                setattr(batch_params, attr.replace("custom", "json"), batch_attr_json_name) # <--- attribute points to json file that contains the info from pose_cols #TODO @Codex make sure to only replace the ending here? (regex or something?)

            # write batch config.yaml
            batch_config_fn = os.path.join(batch_dir, "config.yaml")
            #TODO @Codex please write the .yaml file here (I'm unfamiliar with yaml library)
            batch_config_files.append(batch_config_fn)
            # next batch
        return batch_config_files


    def resolve_pose_cols(self, poses: Poses, n_batches: int, work_dir: str) -> list[str]:
        '''Converts PoseCol types in PottsMPNNParams into actual values. Returns list of .config files. Batches config files if possible.'''
        # first, check if current set of params are batchable
        if self._params_are_batchable():
            return self.resolve_pose_cols_batched(self, poses=poses, n_batches=n_batches, work_dir=work_dir)

        # otherwise run in non-batch mode: iterate over poses.
        # create i/o directories
        work_dir = os.path.abspath(work_dir)
        config_files_dir = os.path.join(work_dir, "config_files")
        json_files_dir = os.path.join(work_dir, "json_files")
        input_list_dir = os.path.join(work_dir, "input_lists")
        output_dir = os.path.join(work_dir, "outputs")
        os.makedirs(config_files_dir, exist_ok=True)
        os.makedirs(input_list_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(json_files_dir, exist_ok=True)

        # iterate over every pose and create config file
        config_files = []
        for pose in poses:
            pose_params = copy.deepcopy(self)

            # write input_list file:
            input_list_fn = os.path.join(input_list_dir, f"{pose['description']}_input_list.txt")
            with open(input_list_fn, 'w', encoding="UTF-8") as f:
                f.write(pose["description"])

            # set i/o params
            pose_params.out_dir = output_dir
            pose_params.out_name = pose["description"]
            pose_params.input_list = input_list_fn
            pose_params.input_dir = os.path.dirname(pose["poses"])

            for attr, val in self._compile_attrs_dict(flat=True):
                # skip params that aren't PoseCol type (they're fine already)
                if not isinstance(val, PoseCol):
                    continue

                # for _custom attributes, write PoseCol val into json file and pass .json file as param.
                if attr.endswith("custom"):
                    # convert PoseCol into actual value
                    attr_json_fn = os.path.join(json_files_dir, f"{pose['poses_description']}_{attr}.json")
                    attr_dict = {pose["description"]: pose[val]}

                    # write actual value into .json file and pass .json filepath to attr.
                    with open(attr_json_fn, 'w', encoding="UTF-8") as f:
                        json.dump(attr_dict, f)

                    setattr(pose_params, attr.replace("_custom", "_json"), attr_json_fn) #TODO @Codex make sure to only replace the ending here? (regex or something?)

                # for any other attribute, write PoseCol val directly from PoseCol
                else:
                    setattr(pose_params, attr, pose["val"])

            # set params_name
            config_fn = os.path.join(config_files_dir, str(pose['description']) + "_config.yaml")
            self.to_yaml(config_fn)
            config_files.append(config_fn)
        return config_files

    def to_yaml(self, out_path: str) -> None:
        '''Write PottsMPNNModelParams out as a config file at 'out_path'. '''

        #TODO @Codex please write this function. Make sure to exclude any parameters that end with "_custom". Those are the ones we can set custom with PoseCols.


def params_to_config(poses: Poses, prefix: str, work_dir: str, params: PottsMPNNParams) -> list[str]:
    '''Generates physical config files for PottsMPNN and stores paths under 'prefix'.'''
    # setup prefix for config files
    config_files_dir = os.path.join(work_dir, "config_files")
    os.makedirs(config_files_dir, exist_ok=True)
    config_files_prefix = f"{prefix}_config_files_location"
    poses.check_prefix(config_files_prefix)

    # check for batching
    config_files = params.resolve_pose_cols()
    return config_files


# Optional: lightweight checklist for developers implementing a new runner.
IMPLEMENTATION_CHECKLIST: tuple[str, ...] = (
    "Set config variable names in __init__.",
    "Set correct index_layers for your output naming.",
    "Implement write_cmd with real CLI syntax.",
    "Implement module-level collect_scores (not a class method) with description/location columns.",
    "Make collect_scores auto-discover score keys from tool outputs where possible.",
    "Skip heavy per-residue/matrix outputs by default; gate them behind include_scores list.",
    "Ensure scorefile reuse works when overwrite=False.",
    "Confirm RunnerOutput merge updates poses as expected.",
    "Export runner in protflow/tools/__init__.py (submodule import + class import).",
    "Document API and add tool page in docs/source/tools/<tool>.rst (and tools/index.rst toctree if new).",
    "Build docs warning-free: sphinx-build -b html -W docs/source docs/_build/html",
    "Add/extend unit tests for parsing and option handling.",
)
