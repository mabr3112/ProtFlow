"""Unit tests for the RFD4-Proteina runner module."""

import ast
import logging
import os
import shlex
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from protflow.poses import Poses
from protflow.residues import AtomSelection, ResidueSelection
from protflow.tools.rfd4_proteina import (
    RFD4_REMAP_RESULT_COLUMNS,
    _run_rfd4_remap_records_with_jobstarter,
    RFD4Params,
    RFD4Proteina,
    add_source_maps_to_scores,
    collect_scores,
    remap_rfd4_motifs,
    normalize_condition_name,
    selection_to_contig,
    selection_to_query,
)

from protflow.tools.runners_auxiliary_scripts.rfd4_remap_motifs import build_provenance_record


# --------------------------------------------------------------------------- #
#                           selection conversion                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("selection", [
    ResidueSelection(["A1", "A2", "A3", "A10"]),
    AtomSelection([("A", 5, "CA"), ("A", 5, "N")]),
])
def test_selection_objects_delegate_to_their_own_converter(selection):
    """The conversion lives on the selection classes; this only dispatches."""
    assert selection_to_query(selection) == selection.to_atomworks_query()


def test_query_strings_are_passed_through_unchanged():
    query = "is_ligand and not is_protein_backbone"
    assert selection_to_query(query) == query


def test_selection_lists_are_combined_with_or():
    query = selection_to_query([ResidueSelection(["A1"]), "is_ligand"])
    assert query == "(sel('A/*/[1]') or is_ligand)"


@pytest.mark.parametrize("selection", [
    [ResidueSelection(["A1"]), "is_ligand"],
    [ResidueSelection(["A1", "B7"]), AtomSelection([("A", 5, "CA")])],
])
def test_combined_queries_parse_as_python_expressions(selection):
    """AtomWorks evaluates a selection with ast.parse(query, mode='eval')."""
    ast.parse(selection_to_query(selection), mode="eval")


def test_selection_to_contig_emits_preserved_segments():
    assert selection_to_contig(ResidueSelection(["A1", "A2", "A3", "C6"])) == "A1-3,C6"


def test_selection_to_contig_mixes_strings_and_selections():
    contig = selection_to_contig(["50-120", "/0", ResidueSelection(["A17", "A18"])])
    assert contig == "50-120, /0, A17-18"


def test_empty_selections_are_rejected():
    with pytest.raises(ValueError):
        selection_to_query(ResidueSelection([]))


# --------------------------------------------------------------------------- #
#                           condition validation                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("given,expected", [
    ("C_CRD", "C_CRD"),
    ("coordinate", "C_CRD"),
    ("hotspot", "C_HOT"),
    ("C_SEQ_PROB_CYS", "C_SEQ_PROB_CYS"),
    ("C_SEQ_PROB_DA", "C_SEQ_PROB_DA"),
])
def test_condition_names_normalize_to_shorthand(given, expected):
    assert normalize_condition_name(given) == expected


@pytest.mark.parametrize("name", ["C_NOPE", "C_SEQ_PROB_XYZ", "C_INTCRD"])
def test_unknown_conditions_are_rejected(name):
    with pytest.raises(ValueError):
        normalize_condition_name(name)


def _poses_fixture(tmp_path, n=2):
    return _poses(tmp_path, n)


def _poses(tmp_path, n=2):
    paths = []
    for i in range(n):
        path = tmp_path / f"target_{i}.cif"
        path.write_text("data_test\n", encoding="utf-8")
        paths.append(str(path))
    return Poses(poses=paths, work_dir=str(tmp_path))


def test_boolean_conditions_reject_non_boolean_values(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_condition("C_CRD", value="yes", select="sel('A')")


def test_enum_conditions_reject_values_outside_their_domain(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_hbond_condition(select="sel('A')", value="GIVER")


def test_index_conditions_require_a_group_label(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_condition("C_IDX", value=False, select="sel('A')")


def test_group_labels_are_rejected_for_other_conditions(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_condition("C_CRD", value=True, select="sel('A')", group="x")


def test_reserved_coordinate_group_ids_are_rejected(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_coordinate_group(select="sel('A')", group_id=0)


def test_sequence_probabilities_must_lie_in_the_unit_interval(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_sequence_probability(select="sel('A')", token="CYS", probability=1.5)


def test_unknown_spec_fields_are_rejected(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    with pytest.raises(ValueError):
        params.set_input_specs(contigs="A1-10")


# --------------------------------------------------------------------------- #
#                             spec construction                                #
# --------------------------------------------------------------------------- #

def test_input_is_inferred_from_poses(tmp_path):
    poses = _poses(tmp_path)
    params = RFD4Params(poses=poses)
    for description, pose in zip(poses.df["poses_description"], poses.df["poses"]):
        assert params[description]["input"] == os.path.abspath(pose)


def test_conditions_append_in_declaration_order(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path, n=1))
    params.set_coordinate_condition(select="sel('A')", value=False)
    params.set_coordinate_group(select="sel('A') and is_protein_backbone", group_id=1)

    design = params["target_0"]
    assert list(design["conditions"]) == ["C_CRD", "C_CRDGRP"]
    assert design["conditions"]["C_CRD"][False] == [{"select": "sel('A')"}]
    assert design["conditions"]["C_CRDGRP"][1] == [{"select": "sel('A') and is_protein_backbone"}]


def test_overwrite_replaces_existing_entries_for_one_annotation(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path, n=1))
    params.set_hotspots(select="sel('A/*/[1]')")
    params.set_hotspots(select="sel('A/*/[2]')", overwrite=True)
    assert params["target_0"]["conditions"]["C_HOT"][True] == [{"select": "sel('A/*/[2]')"}]


def test_specs_are_deep_copied_per_pose(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path))
    params.set_hotspots(select="sel('A/*/[1]')")
    params["target_0"]["conditions"]["C_HOT"][True].append({"select": "sel('B')"})
    assert len(params["target_1"]["conditions"]["C_HOT"][True]) == 1


def test_index_groups_share_one_label_across_selections(tmp_path):
    params = RFD4Params(poses=_poses(tmp_path, n=1))
    params.set_index_group(select="sel('A/*/[10-12]')", group="linker_a")
    params.set_index_group(select="sel('B/*/[40-42]')", group="linker_a")

    entries = params["target_0"]["conditions"]["C_IDX"][False]
    assert [entry["group"] for entry in entries] == ["linker_a", "linker_a"]


def test_per_pose_conditions_read_selections_from_a_dataframe_column(tmp_path):
    poses = _poses(tmp_path)
    poses.df["motif"] = [ResidueSelection(["A45", "A46"]), ResidueSelection(["A12"])]

    params = RFD4Params(poses=poses)
    params.set_per_pose_condition("C_CRD", value=True, select="motif")

    assert params["target_0"]["conditions"]["C_CRD"][True] == [{"select": "sel('A/*/[45-46]')"}]
    assert params["target_1"]["conditions"]["C_CRD"][True] == [{"select": "sel('A/*/[12]')"}]


def test_per_pose_conditions_skip_poses_without_a_selection(tmp_path):
    poses = _poses(tmp_path)
    poses.df["hotspots"] = [ResidueSelection(["A45"]), None]

    params = RFD4Params(poses=poses)
    params.set_per_pose_condition("C_HOT", value=True, select="hotspots")

    assert "C_HOT" in params["target_0"].get("conditions", {})
    assert "C_HOT" not in params["target_1"].get("conditions", {})


def test_per_pose_values_from_a_numpy_column_become_native_python(tmp_path):
    """yaml.SafeDumper cannot represent numpy scalars, so they must be coerced."""
    poses = _poses(tmp_path)
    poses.df["group_id"] = pd.Series([1, 2], dtype="int64")
    poses.df["motif"] = [ResidueSelection(["A45"]), ResidueSelection(["A12"])]

    params = RFD4Params(poses=poses)
    params.set_per_pose_condition("C_CRDGRP", value="group_id", select="motif")

    group_ids = list(params["target_0"]["conditions"]["C_CRDGRP"])
    assert group_ids == [1] and type(group_ids[0]) is int  # noqa: E721 - exact type matters here


def test_per_pose_contig_reads_a_dataframe_column(tmp_path):
    poses = _poses(tmp_path)
    poses.df["contig"] = ["A1-10, 20-30", ResidueSelection(["A5", "A6"])]

    params = RFD4Params(poses=poses)
    params.set_per_pose_contig("contig")

    assert params["target_0"]["contig"] == "A1-10, 20-30"
    assert params["target_1"]["contig"] == "A5-6"


# --------------------------------------------------------------------------- #
#                            spec serialisation                                #
# --------------------------------------------------------------------------- #

def test_write_spec_yaml_round_trips_condition_values(tmp_path):
    poses = _poses(tmp_path, n=1)
    params = RFD4Params(poses=poses)
    params.set_global_args(input_as_annotated_cif=True)
    params.set_contig(["50-120", "/0", "A17-131"])
    params.set_sequence_condition(select="sel('A')")
    params.set_sequence_probability(select="sel('B/*/[10-20]')", token="CYS", probability=0.0)
    params.set_index_group(select=ResidueSelection(["A10", "A11"]), group="linker_a")

    spec_path = params.write_spec(str(tmp_path / "spec.yaml"))
    loaded = yaml.safe_load(open(spec_path, "r", encoding="utf-8"))

    assert loaded["global_args"] == {"input_as_annotated_cif": True}
    design = loaded["target_0"]
    assert design["contig"] == "50-120, /0, A17-131"
    # booleans/floats survive as YAML scalar keys, not strings
    assert design["conditions"]["C_SEQ"][True] == [{"select": "sel('A')"}]
    assert design["conditions"]["C_SEQ_PROB_CYS"][0.0] == [{"select": "sel('B/*/[10-20]')"}]
    assert design["conditions"]["C_IDX"][False][0]["group"] == "linker_a"


def test_write_spec_yaml_emits_no_anchors(tmp_path):
    """Shared sub-dicts must not be emitted as YAML anchors/aliases."""
    params = RFD4Params(poses=_poses(tmp_path, n=3))
    params.set_hotspots(select="sel('A/*/[1]')")
    text = open(params.write_spec(str(tmp_path / "spec.yaml")), "r", encoding="utf-8").read()
    assert "&id" not in text and "*id" not in text


def test_write_spec_json_stringifies_condition_keys(tmp_path):
    import json

    params = RFD4Params(poses=_poses(tmp_path, n=1))
    params.set_sequence_condition(select="sel('A')")
    spec = json.load(open(params.write_spec(str(tmp_path / "spec.json")), "r", encoding="utf-8"))
    assert spec["target_0"]["conditions"]["C_SEQ"] == {"true": [{"select": "sel('A')"}]}


def test_spec_from_yaml_restores_an_equivalent_params_object(tmp_path):
    poses = _poses(tmp_path, n=1)
    params = RFD4Params(poses=poses)
    params.set_contig("A1-20, 10-30")
    params.set_hotspots(select="sel('A/*/[5]')")
    spec_path = params.write_spec(str(tmp_path / "spec.yaml"))

    reloaded = RFD4Params(poses=poses, spec_from_yaml=spec_path)
    assert reloaded["target_0"]["contig"] == "A1-20, 10-30"
    assert reloaded["target_0"]["conditions"]["C_HOT"][True] == [{"select": "sel('A/*/[5]')"}]


def test_spec_from_json_converts_quoted_condition_keys_back(tmp_path):
    poses = _poses(tmp_path, n=1)
    params = RFD4Params(poses=poses)
    params.set_sequence_condition(select="sel('A')")
    spec_path = params.write_spec(str(tmp_path / "spec.json"))

    reloaded = RFD4Params(poses=poses, spec_from_json=spec_path)
    assert reloaded["target_0"]["conditions"]["C_SEQ"][True] == [{"select": "sel('A')"}]


def test_designs_without_an_input_are_rejected(tmp_path):
    params = RFD4Params(poses=Poses(work_dir=str(tmp_path)))
    params.add_specs({"denovo": {"contig": "A1-10"}})
    with pytest.raises(ValueError):
        params.to_spec_dict()


# --------------------------------------------------------------------------- #
#                             command assembly                                 #
# --------------------------------------------------------------------------- #

def _runner():
    runner = RFD4Proteina.__new__(RFD4Proteina)
    runner.application_path = "rfd4-generate"
    runner.pre_cmd = ""
    runner.data_path = None
    runner.jobstarter = None
    runner.name = "rfd4_proteina"
    runner.index_layers = 1
    return runner


def test_write_cmd_uses_hydra_key_value_syntax():
    cmd = _runner().write_cmd(
        in_spec="/run/inputs/batch0.yaml",
        out_dir="/run/outputs",
        num_replicates=40,
        batch_size=8,
        metrics_tags=["design", "ppi_refolding"],
        seed=1234,
    )
    assert "inputs=/run/inputs/batch0.yaml" in cmd
    assert "out_dir=/run/outputs" in cmd
    assert "num_replicates=40" in cmd and "batch_size=8" in cmd
    assert "metrics_tags='[design,ppi_refolding]'" in cmd
    assert "seed=1234" in cmd
    assert "--" not in cmd


def test_write_cmd_exports_data_path_when_configured():
    runner = _runner()
    runner.data_path = "/shared/rfd4_data"
    cmd = runner.write_cmd(in_spec="/s.yaml", out_dir="/o")
    assert cmd.startswith("DATA_PATH=/shared/rfd4_data rfd4-generate ")


def test_write_cmd_omits_unset_optional_arguments():
    cmd = _runner().write_cmd(in_spec="/s.yaml", out_dir="/o")
    assert "seed=" not in cmd
    assert "metrics_tags=" not in cmd
    assert "inference_experiment=" not in cmd


def test_write_cmd_rejects_reserved_options():
    with pytest.raises(ValueError):
        _runner().write_cmd(in_spec="/s.yaml", out_dir="/o", options="num_replicates=5")


def test_write_cmd_passes_hydra_config_flags_and_overrides_verbatim():
    options = "--config-path '/path with spaces/ts1' --config-name run hydra.verbose=true"
    cmd = _runner().write_cmd(in_spec="/s.yaml", out_dir="/o", options=options)

    assert cmd.startswith(f"rfd4-generate {options} inputs=")
    assert shlex.split(cmd)[1:6] == [
        "--config-path", "/path with spaces/ts1", "--config-name", "run", "hydra.verbose=true",
    ]


def test_setup_run_writes_one_spec_per_job_with_disjoint_design_keys(tmp_path):
    poses = _poses(tmp_path, n=4)
    params = RFD4Params(poses=poses)
    params.set_contig("A1-20, 10-30")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    cmds = _runner().setup_run(
        params=params,
        design_keys=[(key, key) for key in params],
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "outputs"),
        work_dir=str(tmp_path),
        n_jobs=2,
        num_replicates=8,
    )

    assert len(cmds) == 2
    written = sorted(input_dir.glob("batch*.yaml"))
    assert len(written) == 2

    keys = []
    for path in written:
        spec = yaml.safe_load(open(path, "r", encoding="utf-8"))
        keys.extend(key for key in spec if key != "global_args")
    assert sorted(keys) == sorted(params)  # disjoint and complete

    # concurrent jobs must not share a hydra run directory
    run_dirs = [cmd.split("hydra.run.dir=")[1].split()[0] for cmd in cmds]
    assert len(set(run_dirs)) == 2

    # RFD4's metric filenames are fixed, so its output directories must differ too.
    output_dirs = [next(arg.split("=", 1)[1] for arg in shlex.split(cmd)
                        if arg.startswith("out_dir=")) for cmd in cmds]
    assert output_dirs == [str(tmp_path / "outputs" / f"batch{i}") for i in range(2)]

    # design keys must be plain strings; split_list routes through numpy.array_split
    # and yaml.SafeDumper cannot represent numpy.str_
    assert all(type(key) is str for key in keys)  # noqa: E721 - exact type matters here


def test_setup_run_offsets_the_seed_per_job(tmp_path):
    """All jobs run with job_id unset, so a shared seed would duplicate trajectories."""
    poses = _poses(tmp_path, n=4)
    params = RFD4Params(poses=poses)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    cmds = _runner().setup_run(
        params=params,
        design_keys=[(key, key) for key in params],
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "outputs"),
        work_dir=str(tmp_path),
        n_jobs=2,
        seed=1234,
    )
    seeds = sorted(int(cmd.split("seed=")[1].split()[0]) for cmd in cmds)
    assert seeds == [1234, 1235]


# --------------------------------------------------------------------------- #
#                             score collection                                 #
# --------------------------------------------------------------------------- #

def _write_generated_cif(samples_dir, example_id):
    design_dir = samples_dir / example_id
    design_dir.mkdir(parents=True)
    cif = design_dir / f"{example_id}_generated.cif"
    cif.write_text("data_design\n#\n", encoding="utf-8")
    # decoys the collector must ignore
    (design_dir / f"{example_id}_input.cif").write_text("data_input\n", encoding="utf-8")
    (design_dir / f"{example_id}_af3_refolded.cif").write_text("data_refold\n", encoding="utf-8")
    return cif


def test_collect_scores_picks_up_designs_and_normalises_descriptions(tmp_path):
    samples = tmp_path / "outputs" / "batch0" / "inference_samples"
    for example_id in ("target_0_0", "target_0_1"):
        _write_generated_cif(samples, example_id)

    scores = collect_scores(str(tmp_path), cif_to_pdb=False)

    assert sorted(scores["description"]) == ["target_0_0", "target_0_1"]
    # ProtFlow requires description == basename stem of location
    stems = [os.path.splitext(os.path.basename(path))[0] for path in scores["location"]]
    assert stems == list(scores["description"])
    assert all(os.path.isfile(path) for path in scores["location"])
    assert list(scores["task"]) == ["target_0", "target_0"]


def test_collect_scores_merges_inline_metrics(tmp_path):
    samples = tmp_path / "outputs" / "batch0" / "inference_samples"
    _write_generated_cif(samples, "design_0")

    metrics_dir = tmp_path / "outputs" / "batch0" / "inference_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame([{
        "example_id": "design_0",
        "task": "design",
        "clash_metrics.frac_atoms_clashing_1p5A": 0.01,
        "generation.args.nsteps": 60,
        "output_base_dir": "/somewhere",
    }]).to_csv(metrics_dir / "per_example_metrics.csv", index=False)

    scores = collect_scores(str(tmp_path), cif_to_pdb=False)

    assert scores.loc[0, "clash_metrics_frac_atoms_clashing_1p5A"] == 0.01
    assert scores.loc[0, "task"] == "design"
    # config echo and bookkeeping columns are dropped by default
    assert "generation_args_nsteps" not in scores.columns
    assert "output_base_dir" not in scores.columns


def test_collect_scores_keeps_generation_columns_on_request(tmp_path):
    samples = tmp_path / "outputs" / "batch0" / "inference_samples"
    _write_generated_cif(samples, "design_0")

    metrics_dir = tmp_path / "outputs" / "batch0" / "inference_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame([{"example_id": "design_0", "generation.args.nsteps": 60}]).to_csv(
        metrics_dir / "per_example_metrics.csv", index=False
    )

    scores = collect_scores(str(tmp_path), cif_to_pdb=False, include_scores=["generation"])
    assert scores.loc[0, "generation_args_nsteps"] == 60


def test_collect_scores_merges_metrics_from_separate_batches(tmp_path):
    for i in range(2):
        directory = tmp_path / "outputs" / f"batch{i}"
        _write_generated_cif(directory / "inference_samples", f"design_{i}_0")
        metrics_dir = directory / "inference_metrics"
        metrics_dir.mkdir()
        pd.DataFrame([{
            "example_id": f"design_{i}_0",
            "sse_distribution.loop_fraction": 0.2 + i * 0.1,
        }]).to_csv(metrics_dir / "per_example_metrics.csv", index=False)

    scores = collect_scores(str(tmp_path), cif_to_pdb=False).set_index("description")

    assert sorted(scores.index) == ["design_0_0", "design_1_0"]
    assert scores["sse_distribution_loop_fraction"].tolist() == pytest.approx([0.2, 0.3])
    assert all(os.path.isfile(path) for path in scores["location"])
    assert all(os.path.isfile(path) for path in scores["cif_location"])


def test_collect_scores_returns_empty_frame_without_outputs(tmp_path):
    assert collect_scores(str(tmp_path), cif_to_pdb=False).empty


# --------------------------------------------------------------------------- #
#                              motif remapping                                 #
# --------------------------------------------------------------------------- #

class _FakePoses:
    """Minimal stand-in exposing the .df attribute remap_rfd4_motifs needs."""

    def __init__(self, df):
        self.df = df


def _poses_with_map(index_map, **columns):
    data = {"poses_description": ["d0"], "rfd4_source_index_map": [index_map]}
    data.update({name: [value] for name, value in columns.items()})
    return _FakePoses(pd.DataFrame(data))


def test_remap_moves_residue_selection_to_output_numbering():
    poses = _poses_with_map({"A5": "B12", "A6": "B13"}, motif=ResidueSelection(["A5", "A6"]))
    remap_rfd4_motifs(poses=poses, motifs=["motif"], prefix="rfd4")
    assert poses.df["motif"][0].to_list() == ["B12", "B13"]


def test_remap_moves_atom_selection_and_preserves_atom_names_and_order():
    selection = AtomSelection([("A", 5, "CA"), ("A", 5, "N"), ("A", 6, "CB")])
    poses = _poses_with_map({"A5": "B12", "A6": "B13"}, motif=selection)
    remap_rfd4_motifs(poses=poses, motifs=["motif"], prefix="rfd4")
    assert poses.df["motif"][0].to_tuple() == (("B", 12, "CA"), ("B", 12, "N"), ("B", 13, "CB"))


def test_remap_preserves_residues_absent_from_the_map(caplog):
    """Generated atoms carry SA_SRC == '.', so they never appear in the map."""
    poses = _poses_with_map({"A5": "B12"}, motif=ResidueSelection(["A5", "Z9"]))
    with caplog.at_level(logging.WARNING):
        remap_rfd4_motifs(poses=poses, motifs=["motif"], prefix="rfd4")
    assert poses.df["motif"][0].to_list() == ["B12", "Z9"]
    assert "Z9" in caplog.text and "RFD4" in caplog.text


def test_remap_raises_on_an_unusable_map_when_strict():
    poses = _poses_with_map(None, motif=ResidueSelection(["A5"]))
    with pytest.raises(ValueError):
        remap_rfd4_motifs(poses=poses, motifs=["motif"], prefix="rfd4", strict=True)


def test_remap_preserves_numbering_on_an_unusable_map_when_not_strict():
    poses = _poses_with_map(None, motif=ResidueSelection(["A5"]))
    remap_rfd4_motifs(poses=poses, motifs=["motif"], prefix="rfd4", strict=False)
    assert poses.df["motif"][0].to_list() == ["A5"]


def test_remap_rejects_a_column_mixing_selection_types():
    df = pd.DataFrame({
        "poses_description": ["d0", "d1"],
        "rfd4_source_index_map": [{"A5": "B12"}, {"A5": "B12"}],
        "motif": [ResidueSelection(["A5"]), AtomSelection([("A", 5, "CA")])],
    })
    with pytest.raises(ValueError):
        remap_rfd4_motifs(poses=_FakePoses(df), motifs=["motif"], prefix="rfd4")


def test_remap_requires_the_source_map_column():
    df = pd.DataFrame({"poses_description": ["d0"], "motif": [ResidueSelection(["A5"])]})
    with pytest.raises(KeyError):
        remap_rfd4_motifs(poses=_FakePoses(df), motifs=["motif"], prefix="rfd4")


def test_add_source_maps_requires_a_jobstarter(tmp_path):
    scores = pd.DataFrame([{"description": "d0", "cif_location": "x.cif"}])
    with pytest.raises(ValueError):
        add_source_maps_to_scores(scores=scores, work_dir=str(tmp_path), jobstarter=None)


def test_add_source_maps_requires_cif_location(tmp_path):
    scores = pd.DataFrame([{"description": "d0"}])
    with pytest.raises(ValueError):
        add_source_maps_to_scores(scores=scores, work_dir=str(tmp_path), jobstarter=object())


def test_add_source_maps_is_a_noop_when_columns_already_present(tmp_path):
    """No jobstarter needed when the provenance columns are already there."""
    row = {"description": "d0", "cif_location": "x.cif"}
    row.update({column: {} for column in RFD4_REMAP_RESULT_COLUMNS})
    scores = add_source_maps_to_scores(scores=pd.DataFrame([row]), work_dir=str(tmp_path), jobstarter=None)
    assert len(scores) == 1


@pytest.mark.parametrize("bad", ["motif_col", [1, 2], ["ok", 3]])
def test_run_rejects_malformed_update_motifs(tmp_path, bad):
    poses = _poses_fixture(tmp_path)
    params = RFD4Params(poses=poses)
    with pytest.raises(ValueError):
        _runner().run(poses=poses, prefix="x", params=params, update_motifs=bad)


def test_add_source_maps_requires_input_map_when_renumbering(tmp_path):
    scores = pd.DataFrame([{"description": "d0", "cif_location": "x.cif", "task": "d"}])
    with pytest.raises(ValueError):
        add_source_maps_to_scores(
            scores=scores, work_dir=str(tmp_path), jobstarter=object(), renumber_input=True
        )


def test_add_source_maps_noop_check_accounts_for_the_renumber_column(tmp_path):
    """Provenance columns alone must not short-circuit a renumbering request."""
    row = {"description": "d0", "cif_location": "x.cif", "task": "d"}
    row.update({column: {} for column in RFD4_REMAP_RESULT_COLUMNS})
    with pytest.raises(ValueError):          # reaches the jobstarter check, i.e. not a no-op
        add_source_maps_to_scores(
            scores=pd.DataFrame([row]), work_dir=str(tmp_path), jobstarter=None, renumber_input=True
        )
    row["renumbered_input"] = "r.cif"        # now complete -> no-op, no jobstarter needed
    out = add_source_maps_to_scores(
        scores=pd.DataFrame([row]), work_dir=str(tmp_path), jobstarter=None, renumber_input=True
    )
    assert out.loc[0, "renumbered_input"] == "r.cif"


def test_cleanup_removes_renumbered_and_remap_directories(tmp_path):
    names = ("outputs", "inputs", "hydra_outputs", "renumbered_inputs", "motif_remap")
    for name in names:
        (tmp_path / name).mkdir()
    _runner()._cleanup_previous_outputs(work_dir=str(tmp_path))
    assert not any((tmp_path / name).exists() for name in names)


class ProvenanceTests(TestCase):
    def record(self, sources, conditioned, chains, residues, input_index_map=None):
        array = SimpleNamespace(chain_id=chains, res_id=residues)
        annotations = ModuleType("rfproteina.utils.standard_annotations")
        for name, value in {"SA_SRC": sources, "SA_CRDSRC": conditioned,
                            "SA_IDXGRPSRC": None, "SA_UII": None}.items():
            setattr(annotations, name, SimpleNamespace(annotation=lambda _, value=value: value))
        parser = ModuleType("atomworks.io.parser")
        parser.parse = lambda *_, **kwargs: {"assemblies": {"1": [array]}}
        presets = ModuleType("rfproteina.utils.parse_presets")
        presets.STANDARD_ANNOTATIONS_ONLY = object()
        with patch.dict(sys.modules, {
            "rfproteina": ModuleType("rfproteina"),
            "rfproteina.conditions": ModuleType("rfproteina.conditions"),
            "rfproteina.utils": ModuleType("rfproteina.utils"),
            "rfproteina.utils.standard_annotations": annotations,
            "rfproteina.utils.parse_presets": presets,
            "atomworks": ModuleType("atomworks"),
            "atomworks.io": ModuleType("atomworks.io"),
            "atomworks.io.parser": parser,
        }):
            return build_provenance_record("design", "output.cif", input_index_map=input_index_map)

    def test_unindexed_motif_is_mapped_when_component_sources_only_identify_ligand(self):
        record = self.record([".", ".", "Z0"], ["A2", "A2", "Z0"], ["A", "A", "Z"], [40, 40, 0])
        self.assertEqual(record["source_index_map"], {"A2": "A40", "Z0": "Z0"})
        self.assertEqual(record["n_source_atoms"], 3)

    def test_composes_both_annotation_sources_without_prepared_id_fallback(self):
        record = self.record(
            ["A1", ".", "Z0"], ["A1", "A2", "Z0"], ["B", "B", "Z"], [12, 13, 0],
            {"A41": {"A1"}, "A42": {"A2"}, "A43": {"A3"}, "Z0": {"Z0"}},
        )
        self.assertEqual(record["source_index_map"], {"A41": "B12", "A42": "B13", "Z0": "Z0"})
        poses = _poses_with_map(
            record["source_index_map"],
            residue_motif=ResidueSelection(["A42"]),
            atom_motif=AtomSelection([("A", 42, "CA")]),
        )
        remap_rfd4_motifs(poses, ["residue_motif", "atom_motif"], "rfd4", strict=True)
        self.assertEqual(poses.df.loc[0, "residue_motif"].to_list(), ["B13"])
        self.assertEqual(poses.df.loc[0, "atom_motif"].to_tuple(), (("B", 13, "CA"),))

    def test_duplicated_original_residue_is_ambiguous_after_composition(self):
        record = self.record(
            ["A1", "A2"], None, ["B", "B"], [12, 13], {"A41": {"A1", "A2"}},
        )
        self.assertEqual(record["source_index_map"], {})
        self.assertEqual(record["ambiguous_source_labels"], ["A41"])


# --------------------------------------------------------------------------- #
#                   provenance worker environment (pre_cmd)                    #
# --------------------------------------------------------------------------- #

class _RecordingJobStarter:
    """Jobstarter stand-in: records the commands and fakes the worker output."""

    def __init__(self, max_cores: int = 2):
        self.max_cores = max_cores
        self.cmds: list[str] = []

    def start(self, cmds, jobname, wait, output_path):  # noqa: ARG002 - signature parity
        import json
        import shlex

        self.cmds = list(cmds)
        for cmd in cmds:
            tokens = shlex.split(cmd.split(";")[-1])
            with open(tokens[tokens.index("--input_json") + 1], encoding="UTF-8") as handle:
                records = json.load(handle)
            payload = [{**r, **{column: {} for column in RFD4_REMAP_RESULT_COLUMNS}} for r in records]
            with open(tokens[tokens.index("--output_json") + 1], "w", encoding="UTF-8") as handle:
                json.dump(payload, handle)


def _dispatch_with(tmp_path, pre_cmd):
    """Run the provenance dispatcher against a recording jobstarter."""
    script = tmp_path / "rfd4_remap_motifs.py"
    script.write_text("")
    jobstarter = _RecordingJobStarter()
    records = [{"description": f"d{i}", "cif_location": f"d{i}.cif"} for i in range(2)]
    results = _run_rfd4_remap_records_with_jobstarter(
        records=records,
        work_dir=str(tmp_path),
        jobstarter=jobstarter,
        python_path="/venv/bin/python",
        script_path=str(script),
        pre_cmd=pre_cmd,
    )
    assert len(results) == 2
    return jobstarter.cmds


def test_pre_cmd_is_prepended_to_every_provenance_worker_command(tmp_path):
    """The worker imports atomworks, so it needs the same environment as generation."""
    pre_cmd = "export CCD_MIRROR_PATH=/mirror"
    cmds = _dispatch_with(tmp_path, pre_cmd)
    assert cmds and all(cmd.startswith(f"{pre_cmd}; /venv/bin/python ") for cmd in cmds)


def test_provenance_worker_commands_are_unchanged_without_a_pre_cmd(tmp_path):
    cmds = _dispatch_with(tmp_path, None)
    assert cmds and all(cmd.startswith("/venv/bin/python ") for cmd in cmds)


def test_add_source_maps_forwards_pre_cmd_to_the_worker_dispatcher(tmp_path, monkeypatch):
    """RFD4Proteina.run() relies on this hand-off to reach the workers."""
    seen = {}

    def fake_dispatch(records, work_dir, jobstarter, python_path=None, script_path=None, pre_cmd=None):
        seen["pre_cmd"] = pre_cmd
        return [
            {**record, **{column: {} for column in RFD4_REMAP_RESULT_COLUMNS}}
            for record in records
        ]

    monkeypatch.setattr(
        "protflow.tools.rfd4_proteina._run_rfd4_remap_records_with_jobstarter", fake_dispatch
    )
    scores = pd.DataFrame([{"description": "d0", "cif_location": "d0.cif"}])
    add_source_maps_to_scores(
        scores=scores,
        work_dir=str(tmp_path),
        jobstarter=object(),
        pre_cmd="export CCD_MIRROR_PATH=/mirror",
    )
    assert seen["pre_cmd"] == "export CCD_MIRROR_PATH=/mirror"
