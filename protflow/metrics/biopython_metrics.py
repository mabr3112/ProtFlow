'''Runner utilities for calculating BioPython-based metrics through an auxiliary worker script.'''
from __future__ import annotations

# imports
import json
import logging
import math
import os
import shlex
from typing import Any

# dependencies
import numpy as np
import pandas as pd
from Bio.PDB import Entity
from Bio.PDB.Atom import Atom
from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral

# customs
from protflow import load_config_path, require_config
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput
from protflow.jobstarters import JobStarter, split_list
from protflow.residues import AtomSelectionInput, AtomSelection


def _json_ready(value: Any) -> Any:
    '''Convert common ProtFlow objects into JSON-ready values.'''
    # Convert ProtFlow atom selections into plain nested lists for JSON.
    if isinstance(value, AtomSelection):
        return value.to_list()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    # JSON can store lists, but not tuples, so convert tuples recursively.
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    return value


def _normalize_metrics(metrics: BiopythonMetric | list[BiopythonMetric] | tuple[BiopythonMetric, ...] | None) -> list[BiopythonMetric]:
    '''Normalize one or multiple BiopythonMetric objects to a list.'''
    # Accept a single metric for convenience, but use a list internally.
    if metrics is None:
        raise ValueError("No Biopython metrics were provided. Pass at least one BiopythonMetric object to run().")
    if isinstance(metrics, BiopythonMetric):
        metrics = [metrics]
    elif isinstance(metrics, tuple):
        metrics = list(metrics)

    if not isinstance(metrics, list) or not metrics:
        raise TypeError(f"metrics must be a BiopythonMetric object or a non-empty list/tuple of metrics. Got: {metrics}")

    # Fail early if the caller passes objects that are not ProtFlow metrics.
    for metric in metrics:
        if not isinstance(metric, BiopythonMetric):
            raise TypeError(f"All metrics must be BiopythonMetric objects. Got {type(metric)}: {metric}")
        if not callable(getattr(metric, "parse", None)):
            raise TypeError(f"Metric {metric} does not provide a callable parse() method.")
    return metrics


def _validate_metric_spec(metric_spec: dict[str, Any], metric: BiopythonMetric, pose_path: str) -> dict[str, Any]:
    '''Validate one parsed metric specification before writing it to worker JSON.'''
    # Each metric.parse() call must describe one worker-side metric call.
    if not isinstance(metric_spec, dict):
        raise TypeError(f"Metric {metric} returned a non-dictionary spec for pose {pose_path}: {metric_spec}")

    # Convert selections and tuples before the spec is written into JSON.
    metric_spec = _json_ready(metric_spec)
    # The worker uses this signature to import the metric class.
    if "metric" not in metric_spec:
        raise KeyError(
            f"Metric {metric} returned a spec without required key 'metric' for pose {pose_path}: {metric_spec}. "
            "This likely means your metric.parse() doesn't work properly."
        )
    if not isinstance(metric_spec["metric"], str) or not metric_spec["metric"]:
        raise TypeError(f"Metric import signature must be a non-empty string for pose {pose_path}: {metric_spec}")

    # These values are forwarded to metric.calc(entity, *args, **kwargs).
    metric_spec["args"] = metric_spec.get("args", []) or []
    metric_spec["kwargs"] = metric_spec.get("kwargs", {}) or {}
    if not isinstance(metric_spec["args"], list):
        raise TypeError(f"Metric args must be a list after parsing for pose {pose_path}: {metric_spec}")
    if not isinstance(metric_spec["kwargs"], dict):
        raise TypeError(f"Metric kwargs must be a dictionary after parsing for pose {pose_path}: {metric_spec}")

    # The metric name becomes the raw score column in the worker output.
    metric_spec["name"] = metric_spec.get("name") or metric_spec["metric"].replace(":", ".").split(".")[-1]
    if not isinstance(metric_spec["name"], str) or not metric_spec["name"]:
        raise TypeError(f"Metric name must be a non-empty string for pose {pose_path}: {metric_spec}")
    return metric_spec


def collect_scores(output_files: list[str], expected_rows: int) -> pd.DataFrame:
    '''Collect worker output JSON files into one scores dataframe.'''
    # Missing files usually mean a worker job crashed before writing results.
    missing_outputs = [output_file for output_file in output_files if not os.path.isfile(output_file)]
    if missing_outputs:
        raise FileNotFoundError(f"Biopython metric worker output files were not created: {missing_outputs}")

    # Each worker writes one JSON table, and here we rebuild the full table.
    scores = pd.concat([pd.read_json(output_file) for output_file in output_files], ignore_index=True)
    # A row-count mismatch means at least one pose did not produce scores.
    if len(scores.index) < expected_rows:
        raise RuntimeError(
            "Number of output poses is smaller than number of input poses. "
            "Some Biopython metric worker jobs might have crashed."
        )
    return scores


class BiopythonMetricRunner(Runner):
    '''Runner that dispatches one or more BiopythonMetric objects over every pose.'''
    def __init__(
        self,
        python_path: str | None = None,
        script_path: str | None = None,
        metrics: BiopythonMetric | list[BiopythonMetric] | tuple[BiopythonMetric, ...] | None = None,
        biopython_level: str = "model",
        jobstarter: JobStarter | None = None,
        overwrite: bool = False,
    ) -> None:
        '''Initialize paths, default metrics, and runner defaults.'''
        # Load script and Python paths from the user's ProtFlow config.
        config = require_config()
        script_dir = load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR")
        self.python_path = python_path or os.path.join(load_config_path(config, "PROTFLOW_ENV"), "python")
        self.script_path = script_path or os.path.join(script_dir, "run_biopython_metrics.py")

        # Store defaults that can be overridden in run().
        self.metrics = metrics
        self.biopython_level = biopython_level
        self.jobstarter = jobstarter
        self.overwrite = overwrite
        self.name = "biopython_metrics"
        self.index_layers = 0

    def __str__(self):
        '''name of runner'''
        return "BiopythonMetricRunner"

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter | None = None,
        metrics: BiopythonMetric | list[BiopythonMetric] | tuple[BiopythonMetric, ...] | None = None,
        overwrite: bool = False,
        biopython_level: str | None = None,
    ) -> Poses:
        '''Run Biopython metric calculations and merge the resulting score table into poses.'''
        # Prefer run-level inputs, otherwise fall back to runner defaults.
        metrics = metrics or self.metrics
        biopython_level = biopython_level or self.biopython_level
        overwrite = overwrite or self.overwrite

        # Create the work directory and choose the jobstarter.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running %s in %s on %d poses.", self, work_dir, len(poses))

        # Reuse cached scores unless overwrite is requested.
        scorefile = os.path.join(work_dir, f"{prefix}_{self.name}.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Found existing Biopython metrics scorefile at %s. Returning cached scores.", scorefile)
            return RunnerOutput(
                poses=poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers,
            ).return_poses()

        # validate run inputs after scorefile reuse, so cached results can be loaded without metric objects.
        metrics = _normalize_metrics(metrics)
        # Check the worker script before submitting jobs.
        if not os.path.isfile(self.script_path):
            raise FileNotFoundError(
                f"Cannot find script 'run_biopython_metrics.py' at {self.script_path}. "
                "Set AUXILIARY_RUNNER_SCRIPTS_DIR in config.py to the protflow/tools/runners_auxiliary_scripts directory."
            )
        if len(poses) == 0:
            raise ValueError("No poses were provided for Biopython metric calculation.")

        # Build the pose-to-metric-spec dictionary consumed by the worker.
        input_dict = self.setup_input_dict(poses=poses, metrics=metrics)

        # Split the large input dictionary into one JSON file per worker.
        input_jsons, output_jsons = self.write_input_jsons(input_dict=input_dict, work_dir=work_dir, jobstarter=jobstarter)

        # Create one shell command per worker JSON file.
        cmds = self.write_cmds(
            input_jsons=input_jsons,
            output_jsons=output_jsons,
            biopython_level=biopython_level,
        )

        # Start all worker jobs and wait until they finish.
        jobstarter.start(
            cmds=cmds,
            jobname=f"{prefix}_{self.name}",
            wait=True,
            output_path=work_dir,
        )

        # Merge worker outputs, save a scorefile, and update poses.df.
        scores = collect_scores(output_files=output_jsons, expected_rows=len(poses))
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(
            poses=poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()

    def setup_input_dict(self, poses: Poses, metrics: list[BiopythonMetric]) -> dict[str, list[dict[str, Any]]]:
        '''Set up the worker input dictionary for run_biopython_metrics.py.'''
        input_dict = {}
        for pose in poses:
            # Use absolute paths because worker jobs may run from another directory.
            pose_path = os.path.abspath(pose["poses"])
            metric_specs = []
            metric_names = set()
            for metric in metrics:
                # parse() can read per-pose values from the poses dataframe row.
                metric_spec = _validate_metric_spec(metric.parse(pose), metric=metric, pose_path=pose_path)
                # Metric names must be unique because they become score columns.
                if metric_spec["name"] in metric_names:
                    raise ValueError(f"Duplicate metric name '{metric_spec['name']}' for pose {pose_path}. Metric names must be unique per pose.")
                metric_names.add(metric_spec["name"])
                metric_specs.append(metric_spec)
            input_dict[pose_path] = metric_specs
        return input_dict

    def write_input_jsons(
        self,
        input_dict: dict[str, list[dict[str, Any]]],
        work_dir: str,
        jobstarter: JobStarter,
    ) -> tuple[list[str], list[str]]:
        '''Write worker input JSON files and return paired input/output JSON paths.'''
        pose_paths = list(input_dict.keys())
        # Use the jobstarter core count as the number of worker chunks.
        pose_sublists = split_list(pose_paths, n_sublists=jobstarter.max_cores or 1)

        input_jsons = []
        output_jsons = []
        for index, pose_sublist in enumerate(pose_sublists, start=1):
            # Each worker receives only its assigned subset of poses.
            subdict = {pose_path: input_dict[pose_path] for pose_path in pose_sublist}
            input_json = os.path.join(work_dir, f"biopython_metrics_input_{index:04}.json")
            output_json = os.path.join(work_dir, f"biopython_metrics_output_{index:04}.json")
            with open(input_json, "w", encoding="UTF-8") as f:
                json.dump(subdict, f)
            input_jsons.append(input_json)
            output_jsons.append(output_json)
        return input_jsons, output_jsons

    def write_cmds(self, input_jsons: list[str], output_jsons: list[str], biopython_level: str) -> list[str]:
        '''Write shell commands for the Biopython metric worker script.'''
        cmds = []
        for input_json, output_json in zip(input_jsons, output_jsons):
            # Quote paths so spaces or shell characters do not break commands.
            cmds.append(
                f"{shlex.quote(self.python_path)} {shlex.quote(self.script_path)} "
                f"--input_json {shlex.quote(input_json)} "
                f"--output_path {shlex.quote(output_json)} "
                f"--biopython_level {shlex.quote(biopython_level)}"
            )
        return cmds


class BiopythonMetric:
    '''Abstract baseclass for a metric that operates on a loaded biopython structure.'''
    def __init__(self, name: str | None = None, *args, **kwargs): #pylint: disable=W1113
        '''Store a JSON-serializable metric call definition.'''
        # The worker imports the metric class using this signature.
        self.signature = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # This name becomes the raw score column before ProtFlow prefixes it.
        self.name = name or self.__class__.__name__.lower()

        # Stored args are resolved by parse() before worker input is written.
        self.args = list(args)
        self.kwargs = kwargs

    def __str__(self) -> str:
        '''Return the metric name.'''
        return self.name

    def parse(self, pose: pd.Series|dict) -> dict:
        '''Parses metric. Basic implementation. Complex implementation could look for columns in pose df.'''
        def _arg_is_pose_col(pose: pd.Series|dict, arg) -> bool:
            # A string is treated as a dataframe column if the row has that key.
            return isinstance(arg, str) and arg in pose

        # TODO: raise warning if arg appears both as a column and a value in pose!

        # Resolve positional args, including dataframe column references.
        args = getattr(self, "args", []) or []
        args_parsed = [
            _json_ready(pose[arg] if _arg_is_pose_col(pose, arg) else arg)
            for arg in args
        ]

        # Resolve keyword args with the same dataframe column logic.
        kwargs = getattr(self, "kwargs", {}) or {}
        kwargs_parsed = {
            key: _json_ready(pose[val] if _arg_is_pose_col(pose, val) else val)
            for key, val in kwargs.items()
        }

        # This dictionary is the JSON contract consumed by the worker script.
        parsed_metric_dict = {
            "metric": getattr(self, "signature", f"{self.__class__.__module__}.{self.__class__.__name__}"),
            "name": getattr(self, "name", None) or self.__class__.__name__.lower(),
            "args": args_parsed,
            "kwargs": kwargs_parsed
        }
        return parsed_metric_dict

    def calc(self, biomolecule: Entity, *args, **kwargs):
        '''Uniform calculation function.'''
        raise NotImplementedError

    @staticmethod
    def _normalize_residue_id(residue_id: Any) -> tuple[str, int, str]:
        '''Normalize compact residue IDs to BioPython residue IDs.'''
        if isinstance(residue_id, (list, tuple)):
            if len(residue_id) != 3:
                raise ValueError(f"BioPython residue IDs must have three elements. Got: {residue_id}")
            hetero_flag, residue_number, insertion_code = residue_id
            return (hetero_flag or " ", int(residue_number), insertion_code or " ")
        return (" ", int(residue_id), " ")

    @staticmethod
    def _normalize_atom_id(atom_id: Any) -> tuple[Any, Any]:
        '''Normalize BioPython atom IDs and optional altloc identifiers.'''
        if isinstance(atom_id, list):
            atom_id = tuple(atom_id)
        if isinstance(atom_id, tuple) and len(atom_id) == 2:
            return atom_id[0], atom_id[1]
        return atom_id, None

    @staticmethod
    def _normalize_model_id(model_id: Any) -> Any:
        '''Normalize JSON-loaded model IDs where possible.'''
        try:
            return int(model_id)
        except (TypeError, ValueError):
            return model_id

    def _atom_from_spec(self, biomolecule: Entity, atom_spec: Any, default_model: int = 0) -> Atom:
        '''Resolve one atom from a compact or full BioPython atom specification.'''
        if not isinstance(atom_spec, (list, tuple)):
            raise TypeError(f"Atom specifications must be tuple/list-like. Got {type(atom_spec)}: {atom_spec}")

        atom_spec = list(atom_spec)
        if len(atom_spec) == 3:
            model_id = default_model
            chain_id, residue_id, atom_id = atom_spec
        elif len(atom_spec) == 4:
            model_id, chain_id, residue_id, atom_id = atom_spec
        elif len(atom_spec) == 5:
            _, model_id, chain_id, residue_id, atom_id = atom_spec
        elif len(atom_spec) == 6:
            _, model_id, chain_id, residue_id, atom_name, altloc = atom_spec
            atom_id = (atom_name, altloc)
        else:
            raise ValueError(f"Atom specifications must have 3, 4, 5, or 6 elements. Got {len(atom_spec)}: {atom_spec}")

        residue_id = self._normalize_residue_id(residue_id)
        atom_name, altloc = self._normalize_atom_id(atom_id)

        try:
            atom = self._resolve_atom_from_entity(biomolecule, model_id, chain_id, residue_id, atom_name)
        except KeyError as exc:
            raise KeyError(f"Could not resolve atom specification {atom_spec}") from exc

        if altloc not in (None, "", " ") and hasattr(atom, "disordered_select"):
            atom.disordered_select(altloc)
            atom = atom.selected_child
        return atom

    def _resolve_atom_from_entity(self, biomolecule: Entity, model_id: Any, chain_id: str, residue_id: tuple[str, int, str], atom_name: str) -> Atom:
        '''Resolve an atom from a BioPython Structure, Model, Chain, or Residue.'''
        level = biomolecule.get_level()
        if level == "S":
            return biomolecule[self._normalize_model_id(model_id)][chain_id][residue_id][atom_name]
        if level == "M":
            return biomolecule[chain_id][residue_id][atom_name]
        if level == "C":
            return biomolecule[residue_id][atom_name]
        if level == "R":
            return biomolecule[atom_name]
        raise ValueError(f"Cannot resolve atom specs from BioPython entity level '{level}'.")

    def _parse_atoms(self, biomolecule: Entity, atoms: AtomSelectionInput, expected_counts: tuple[int, ...], parameter_name: str = "atoms") -> list[Atom]:
        '''Resolve an ordered atom selection from a BioPython entity.'''
        try:
            atom_specs = AtomSelection(atoms).to_list()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{parameter_name} must be an ordered AtomSelection or atom ID list. Got: {atoms}") from exc

        if len(atom_specs) not in expected_counts:
            raise ValueError(f"{parameter_name} must contain {expected_counts} atoms. Got {len(atom_specs)}: {atoms}")
        return [self._atom_from_spec(biomolecule, atom_spec) for atom_spec in atom_specs]

    @staticmethod
    def _coord(atom: Atom) -> np.ndarray:
        '''Return atom coordinates as a float numpy array.'''
        return np.asarray(atom.get_coord(), dtype=float)

    @staticmethod
    def _vector(atom_a: Atom, atom_b: Atom) -> np.ndarray:
        '''Return the vector from atom_a to atom_b.'''
        return BiopythonMetric._coord(atom_b) - BiopythonMetric._coord(atom_a)

    @staticmethod
    def _safe_norm(vector: np.ndarray, context: str) -> float:
        '''Return vector length and fail on zero-length vectors.'''
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError(f"Cannot calculate {context} with a zero-length vector.")
        return norm

    @staticmethod
    def _angle_between_vectors(vector_a: np.ndarray, vector_b: np.ndarray, degrees: bool = True, acute: bool = False) -> float:
        '''Calculate the angle between two vectors.'''
        norm_a = BiopythonMetric._safe_norm(vector_a, "angle")
        norm_b = BiopythonMetric._safe_norm(vector_b, "angle")
        cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        cosine = abs(cosine) if acute else cosine
        angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        return math.degrees(angle) if degrees else angle

    @staticmethod
    def _plane_normal(atom_a: Atom, atom_b: Atom, atom_c: Atom) -> np.ndarray:
        '''Return the normal vector of the plane through three atoms.'''
        normal = np.cross(BiopythonMetric._vector(atom_a, atom_b), BiopythonMetric._vector(atom_a, atom_c))
        BiopythonMetric._safe_norm(normal, "plane normal")
        return normal

class Distance(BiopythonMetric):
    '''Calculate distances between points, vectors, and planes.'''
    def __init__(self, name: str | None = None, atoms: AtomSelectionInput|str = None, distance_type: str = "auto") -> None:
        '''Initialize a distance metric.'''
        super().__init__(name=name, atoms=atoms, distance_type=distance_type)

    @staticmethod
    def _normalize_distance_type(distance_type: str, atom_count: int) -> str:
        '''Resolve distance type aliases and auto-detection.'''
        aliases = {
            "point_point": "point_point",
            "points": "point_point",
            "point-point": "point_point",
            "point_vector": "point_vector",
            "point_line": "point_vector",
            "point-line": "point_vector",
            "point_vector_line": "point_vector",
            "vector_vector": "vector_vector",
            "line_line": "vector_vector",
            "line-line": "vector_vector",
            "vector-vector": "vector_vector",
            "point_plane": "point_plane",
            "point-plane": "point_plane",
        }

        if distance_type is None or distance_type == "auto":
            if atom_count == 2:
                return "point_point"
            if atom_count == 3:
                return "point_vector"
            raise ValueError("Four-atom distances are ambiguous; set distance_type to 'vector_vector' or 'point_plane'.")

        if not isinstance(distance_type, str):
            raise TypeError(f"distance_type must be a string. Got {type(distance_type)}: {distance_type}")

        try:
            return aliases[distance_type.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown distance_type '{distance_type}'. Use auto, point_point, point_vector, vector_vector, or point_plane.") from exc

    def calc(self, biomolecule: Entity, atoms: AtomSelectionInput, distance_type: str = "auto") -> float: #pylint: disable=W0221
        '''Calculate the requested distance in Angstrom.'''
        atom_list = self._parse_atoms(biomolecule, atoms, expected_counts=(2, 3, 4))
        distance_type = self._normalize_distance_type(distance_type, atom_count=len(atom_list))

        if distance_type == "point_point":
            if len(atom_list) != 2:
                raise ValueError("point_point distance requires exactly 2 atoms.")
            return float(atom_list[0] - atom_list[1])

        if distance_type == "point_vector":
            if len(atom_list) != 3:
                raise ValueError("point_vector distance requires exactly 3 atoms.")
            point, line_a, line_b = atom_list
            line = self._vector(line_a, line_b)
            return float(np.linalg.norm(np.cross(self._coord(point) - self._coord(line_a), line)) / self._safe_norm(line, "point_vector distance"))

        if distance_type == "vector_vector":
            if len(atom_list) != 4:
                raise ValueError("vector_vector distance requires exactly 4 atoms.")
            vec_a = self._vector(atom_list[0], atom_list[1])
            vec_b = self._vector(atom_list[2], atom_list[3])
            self._safe_norm(vec_a, "vector_vector distance")
            self._safe_norm(vec_b, "vector_vector distance")
            cross = np.cross(vec_a, vec_b)
            cross_norm = float(np.linalg.norm(cross))
            if cross_norm <= 1e-12:
                return float(np.linalg.norm(np.cross(self._coord(atom_list[2]) - self._coord(atom_list[0]), vec_a)) / self._safe_norm(vec_a, "parallel vector_vector distance"))
            return float(abs(np.dot(self._coord(atom_list[2]) - self._coord(atom_list[0]), cross)) / cross_norm)

        if distance_type == "point_plane":
            if len(atom_list) != 4:
                raise ValueError("point_plane distance requires exactly 4 atoms.")
            point, plane_a, plane_b, plane_c = atom_list
            normal = self._plane_normal(plane_a, plane_b, plane_c)
            return float(abs(np.dot(self._coord(point) - self._coord(plane_a), normal)) / self._safe_norm(normal, "point_plane distance"))

        raise ValueError(f"Unsupported distance_type '{distance_type}'.")

class Angle(BiopythonMetric):
    '''Calculate an angle from three atoms or from two atom-defined vectors.'''
    def __init__(self, name: str | None = None, atoms: AtomSelectionInput|str = None, degrees: bool = True) -> None:
        '''Initialize an angle metric.'''
        super().__init__(name=name, atoms=atoms, degrees=degrees)

    def calc(self, biomolecule: Entity, atoms: AtomSelectionInput, degrees: bool = True) -> float: #pylint: disable=W0221
        '''Calculate an atom angle in degrees by default.'''
        atom_list = self._parse_atoms(biomolecule, atoms, expected_counts=(3, 4))
        if len(atom_list) == 3:
            angle = calc_angle(*(Vector(atom.get_coord()) for atom in atom_list))
            return float(math.degrees(angle) if degrees else angle)
        return self._angle_between_vectors(
            self._vector(atom_list[0], atom_list[1]),
            self._vector(atom_list[2], atom_list[3]),
            degrees=degrees,
        )


class Dihedral(BiopythonMetric):
    '''Calculate a signed dihedral angle from four atoms.'''
    def __init__(self, name: str | None = None, atoms: AtomSelectionInput|str = None, degrees: bool = True) -> None:
        '''Initialize a dihedral metric.'''
        super().__init__(name=name, atoms=atoms, degrees=degrees)

    def calc(self, biomolecule: Entity, atoms: AtomSelectionInput, degrees: bool = True) -> float: #pylint: disable=W0221
        '''Calculate a signed dihedral angle in degrees by default.'''
        atom_list = self._parse_atoms(biomolecule, atoms, expected_counts=(4,))
        angle = calc_dihedral(*(Vector(atom.get_coord()) for atom in atom_list))
        return float(math.degrees(angle) if degrees else angle)


class PlaneAngle(BiopythonMetric):
    '''Calculate the angle between two planes defined by six atoms.'''
    def __init__(self, name: str | None = None, atoms: AtomSelectionInput|str = None, degrees: bool = True, acute: bool = True) -> None:
        '''Initialize a plane-angle metric.'''
        super().__init__(name=name, atoms=atoms, degrees=degrees, acute=acute)

    def calc(self, biomolecule: Entity, atoms: AtomSelectionInput, degrees: bool = True, acute: bool = True) -> float: #pylint: disable=W0221
        '''Calculate the angle between two atom-defined planes.'''
        atom_list = self._parse_atoms(biomolecule, atoms, expected_counts=(6,))
        normal_a = self._plane_normal(atom_list[0], atom_list[1], atom_list[2])
        normal_b = self._plane_normal(atom_list[3], atom_list[4], atom_list[5])
        return self._angle_between_vectors(normal_a, normal_b, degrees=degrees, acute=acute)
