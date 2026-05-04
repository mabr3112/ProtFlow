'''Module to run biopython metrics.'''

from __future__ import annotations

# imports
from typing import Any
import importlib
import inspect
import json
import os

# dependencies
import Bio.PDB
import pandas as pd

# customs
from protflow.poses import description_from_path


def load_biopython_structure(path: str, quiet: bool = True) -> Bio.PDB.Structure.Structure:
    '''Load a PDB or mmCIF structure as a BioPython Structure object.'''
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Structure file {path} not found!")

    handle = description_from_path(path)
    lower_path = path.lower()
    if lower_path.endswith(".pdb"):
        parser = Bio.PDB.PDBParser(QUIET=quiet)
    elif lower_path.endswith((".cif", ".mmcif")):
        parser = Bio.PDB.MMCIFParser(QUIET=quiet)
    else:
        raise ValueError(f"Unsupported structure file extension for {path}. Supported extensions: .pdb, .cif, .mmcif")

    return parser.get_structure(handle, path)


def load_biopython_entity(path: str, biopython_level: str = "model") -> Bio.PDB.Entity.Entity:
    '''Load a BioPython structure and return it at the requested entity level.'''
    structure = load_biopython_structure(path)
    biopython_level = biopython_level.lower()

    if biopython_level == "structure" or biopython_level == "entity":
        return structure
    if biopython_level == "model":
        try:
            return structure[0]
        except KeyError:
            return next(structure.get_models())

    raise ValueError(f"Unsupported biopython_level '{biopython_level}'. Supported levels: structure, model, entity")


def import_metric(metric_signature: str) -> Any:
    '''Import and instantiate one metric from its import signature.'''
    if not isinstance(metric_signature, str) or not metric_signature:
        raise TypeError(f"Metric import signature must be a non-empty string. Got: {metric_signature}")

    if ":" in metric_signature:
        module_name, attribute_name = metric_signature.rsplit(":", 1)
        separator = ":"
    else:
        module_name, separator, attribute_name = metric_signature.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"Metric import signature '{metric_signature}' must include a module and attribute name, "
            "for example 'protflow.metrics.biopython_metrics.Distance'."
        )

    metric_obj = getattr(importlib.import_module(module_name), attribute_name)
    if inspect.isclass(metric_obj):
        return metric_obj()
    return metric_obj


def metric_callable(metric: Any) -> Any:
    '''Return the callable used to calculate a metric.'''
    if hasattr(metric, "calc") and callable(metric.calc):
        return metric.calc
    if callable(metric):
        return metric
    raise TypeError(f"Imported metric {metric} is neither callable nor an object with a callable calc() method.")


def validate_metric_spec(metric_spec: dict[str, Any], target: str) -> dict[str, Any]:
    '''Validate a metric specification and fill optional fields.'''
    if not isinstance(metric_spec, dict):
        raise TypeError(f"Metric specs for target {target} must be dictionaries. Got: {metric_spec}")
    if "metric" not in metric_spec:
        raise KeyError(f"Metric spec for target {target} is missing required key 'metric'. Spec: {metric_spec}")

    metric_spec = metric_spec.copy()
    metric_spec["args"] = metric_spec.get("args", []) or []
    metric_spec["kwargs"] = metric_spec.get("kwargs", {}) or {}

    if not isinstance(metric_spec["args"], list):
        raise TypeError(f"Metric spec args for target {target} must be a list. Spec: {metric_spec}")
    if not isinstance(metric_spec["kwargs"], dict):
        raise TypeError(f"Metric spec kwargs for target {target} must be a dictionary. Spec: {metric_spec}")

    metric_name = metric_spec["metric"].replace(":", ".").split(".")[-1] # compile name from metric if none specified (should never happen)
    metric_spec["name"] = metric_spec.get("name") or metric_name
    if not isinstance(metric_spec["name"], str) or not metric_spec["name"]:
        raise TypeError(f"Metric spec name for target {target} must be a non-empty string. Spec: {metric_spec}")

    return metric_spec


def parse_input_json(json_path: str) -> dict[str, list[dict[str, Any]]]:
    '''Parse and validate the Biopython metrics input JSON.'''
    with open(json_path, "r", encoding="UTF-8") as f:
        input_dict = json.loads(f.read())

    if not isinstance(input_dict, dict):
        raise TypeError("Input JSON must contain a dictionary mapping target structure paths to metric spec lists.")

    for target, metric_specs in input_dict.items():
        if not isinstance(target, str) or not target:
            raise TypeError(f"Input JSON target keys must be non-empty strings. Got: {target}")
        if not isinstance(metric_specs, list):
            raise TypeError(f"Input JSON target {target} must map to a list of metric specs. Got: {metric_specs}")
        input_dict[target] = [validate_metric_spec(metric_spec, target=target) for metric_spec in metric_specs]

    return input_dict


def unique_metric_signatures(input_dict: dict[str, list[dict[str, Any]]]) -> list[str]:
    '''Return all unique metric import signatures used in an input dictionary.'''
    return sorted({metric_spec["metric"] for metric_specs in input_dict.values() for metric_spec in metric_specs})


def calculate_metric(metric: Any, entity: Bio.PDB.Entity.Entity, metric_spec: dict[str, Any]) -> Any:
    '''Calculate one metric for a loaded BioPython entity.'''
    return metric_callable(metric)(entity, *metric_spec["args"], **metric_spec["kwargs"])


def main(args):
    '''Main function.'''
    # load input json
    input_json = parse_input_json(args.input_json)

    # import all unique metrics from metric specs
    metric_dict = {metric_signature: import_metric(metric_signature) for metric_signature in unique_metric_signatures(input_json)}

    # loop over pdbs and metrics
    pose_data = []
    for pdb, metrics in input_json.items():
        ent = load_biopython_entity(pdb, biopython_level=args.biopython_level)
        ent_data_d = {
            'location': pdb, 
            'description': description_from_path(pdb)
        }

        for metric_spec in metrics:
            # score name is parsed by the runner and used for score storage
            score_name = metric_spec["name"]
            if score_name in ent_data_d:
                raise ValueError(f"Duplicate metric score name '{score_name}' for target {pdb}. Names must be unique per target.")

            # run metric
            ent_data_d[score_name] = calculate_metric(metric_dict[metric_spec["metric"]], ent, metric_spec)

        pose_data.append(pd.Series(ent_data_d))

    # combine
    out_scores = pd.DataFrame(pose_data)

    # save out_scores
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_scores.to_json(args.output_path)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help=(
            ".json file mapping target structure paths BiopythonMetrics and parameters"
            "Expected format: "
            "{'target.pdb': [{'metric': 'metric.import.signature', 'name': 'custom_metric_name', 'args': [...], 'kwargs': {...} ] }"
        ),
    )

    argparser.add_argument(
        "--biopython_level",
        type=str,
        default="model",
        help="{Entity, Structure, Model} Level at which BioPython entities should be loaded. Default is model."
    )

    argparser.add_argument("--output_path", type=str, default="biopython_metrics.json")
    arguments = argparser.parse_args()

    main(arguments)
