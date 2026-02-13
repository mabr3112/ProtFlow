"""Runner template for ProtFlow tool integrations.

How to use this template
------------------------
1. Copy this file (or class) to a new module in ``protflow/tools`` or ``protflow/metrics``.
2. Rename ``ExampleRunner`` and ``example_runner`` to your tool name.
3. Replace all ``TODO`` markers.
4. Keep the run lifecycle intact:
   setup workdir -> reuse cached outputs -> prep options -> build commands -> run jobs -> collect scores -> return RunnerOutput.
5. Implement ``collect_scores(...)`` as a **module function**, not a class method.

Design goals
------------
- Keep runner behavior consistent across ProtFlow.
- Make it obvious where tool-specific logic belongs.
- Avoid re-implementing common logic already provided by ``Runner``.
- Keep score parsing callable without constructing a runner instance.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob

import pandas as pd

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


class ExampleRunner(Runner):
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

        config = require_config()

        # TODO: replace config variable names with your tool's entries.
        self.application_path = application_path or load_config_path(config, "YOUR_TOOL_PATH")
        self.python_path = python_path or load_config_path(config, "YOUR_TOOL_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "YOUR_TOOL_PRE_CMD", is_pre_cmd=True)

        self.jobstarter = jobstarter
        self.name = "example_runner"
        self.index_layers = 0

    def __str__(self) -> str:
        """Return a short runner name used in logs."""
        return self.name

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        options: str | None = None,
        pose_options: list[str] | str | None = None,
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

        # 3) Prepare pose-level options.
        pose_options_list = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # 4) Build commands.
        cmds = self._build_commands(
            poses=poses,
            work_dir=work_dir,
            options=options,
            pose_options=pose_options_list,
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
            raise RuntimeError(f"{self}: collect_scores returned no rows.")

        # 7) Persist and merge back into poses.
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    def _build_commands(
        self,
        poses: Poses,
        work_dir: str,
        options: str | None,
        pose_options: list[str | None],
    ) -> list[str]:
        """Create one shell command per pose.

        Developer instructions
        ----------------------
        - Convert a global options string + per-pose options into final CLI options.
        - Keep all command assembly in one place to simplify debugging.
        - Return a list with deterministic order matching ``poses`` rows.
        """

        out_dir = os.path.join(work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        cmds: list[str] = []
        for pose_path, pose_opt in zip(poses.df["poses"].to_list(), pose_options):
            # Merge global and pose-specific options; pose options override global values.
            merged_opts, merged_flags = parse_generic_options(options, pose_opt, sep="--")
            cli_args = options_flags_to_string(merged_opts, list(merged_flags), sep="--")
            cmds.append(self.write_cmd(pose_path=pose_path, out_dir=out_dir, cli_args=cli_args))
        return cmds

    def write_cmd(self, pose_path: str, out_dir: str, cli_args: str) -> str:
        """Return the exact shell command for one pose.

        Developer instructions
        ----------------------
        - Build an executable command string only; do not execute here.
        - Ensure output filename preserves or predictably derives from pose description.
        - Keep quoting robust for paths with spaces.
        """

        description = os.path.splitext(os.path.basename(pose_path))[0]
        out_pose = os.path.join(out_dir, f"{description}.pdb")

        # TODO: replace with your tool's real command structure.
        return (
            f"{self.python_path} {self.application_path} "
            f"--input '{pose_path}' --output '{out_pose}'{cli_args}"
        )

    def _cleanup_previous_outputs(self, work_dir: str) -> None:
        """Delete/clear runner-specific output artifacts before rerun.

        Developer instructions
        ----------------------
        - Keep cleanup scoped to this runner's own output directories.
        - Never remove unrelated files outside ``work_dir``.
        - This method is optional but useful for tools that append stale outputs.
        """

        output_dir = os.path.join(work_dir, "outputs")
        if not os.path.isdir(output_dir):
            return

        for file_path in glob(os.path.join(output_dir, "*")):
            if os.path.isfile(file_path):
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
