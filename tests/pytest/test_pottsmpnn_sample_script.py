import ast
from pathlib import Path


SAMPLER_PATH = (
    Path(__file__).parents[2]
    / "protflow"
    / "tools"
    / "runners_auxiliary_scripts"
    / "pottsmpnn_sample_seqs.py"
)


def _load_sampler_helpers(*helper_names):
    """Load pure helpers without importing the external PottsMPNN checkout."""
    source = SAMPLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == set(helper_names)

    namespace = {}
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    exec(compile(helper_module, SAMPLER_PATH, "exec"), namespace)  # pylint: disable=exec-used
    return [namespace[name] for name in helper_names]


def test_external_sequence_keys_use_the_most_specific_pdb_owner():
    index_external_sequence_keys, = _load_sampler_helpers("index_external_sequence_keys")
    short_name = "petase_motif_1123"
    long_name = "petase_motif_11235"

    ownership = index_external_sequence_keys(
        ["foo", "foo_0", "foo_123", "foo_123_0", f"{long_name}_0"],
        ["foo", "foo_123", short_name, long_name],
    )

    assert ownership["foo"] == ["foo", "foo_0"]
    assert ownership["foo_123"] == ["foo_123", "foo_123_0"]
    assert ownership[short_name] == []
    assert ownership[long_name] == [f"{long_name}_0"]


def test_decoding_orders_use_full_pdb_and_normalized_sample_keys():
    get_decoding_order, store_decoding_order = _load_sampler_helpers(
        "get_decoding_order",
        "store_decoding_order",
    )
    decoding_orders = {}

    store_decoding_order(decoding_orders, "pose|A|B", "0", [2, 0, 1])
    store_decoding_order(decoding_orders, "pose|A|B", "1", [1, 2, 0])
    store_decoding_order(decoding_orders, "pose|B|A", "0", [0, 1, 2])
    store_decoding_order(decoding_orders, "single_pose", None, [1, 0])

    assert get_decoding_order(decoding_orders, "pose|A|B", "0") == [2, 0, 1]
    assert get_decoding_order(decoding_orders, "pose|A|B", "1") == [1, 2, 0]
    assert get_decoding_order(decoding_orders, "pose|B|A", "0") == [0, 1, 2]
    assert get_decoding_order(decoding_orders, "single_pose", None) == [1, 0]
    assert get_decoding_order(decoding_orders, "single_pose", "0") is None


def test_optimize_fasta_reads_the_configured_input_path():
    source = SAMPLER_PATH.read_text(encoding="utf-8")

    assert "with open(cfg.inference.optimize_fasta, 'r') as f:" in source
    assert "with open(filename, 'r') as f:" not in source
