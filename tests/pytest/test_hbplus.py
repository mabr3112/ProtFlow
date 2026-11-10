from types import SimpleNamespace

import pandas as pd

from protflow.metrics.hbplus import HBplus_query


def _poses(target_types=None):
    """Create the minimal pose interface needed to parse an HBplus query."""
    data = {"poses_description": ["pose_1", "pose_2"]}
    if target_types is not None:
        data["target_types"] = target_types
    return SimpleNamespace(
        df=pd.DataFrame(data),
        poses_list=lambda: ["pose_1.pdb", "pose_2.pdb"],
    )


def test_hbplus_query_parses_fixed_target_type_and_supports_chaining():
    query = HBplus_query("donors")

    assert query.set_target_type("donor") is query
    assert query.parse_query(_poses()) == {
        "pose_1": {"target_type": "donor"},
        "pose_2": {"target_type": "donor"},
    }


def test_hbplus_query_parses_target_type_from_pose_column():
    query = HBplus_query("roles")
    query.set_target_type("target_types", from_pose_col=True)

    assert query.parse_query(_poses(["donor", "acceptor"])) == {
        "pose_1": {"target_type": "donor"},
        "pose_2": {"target_type": "acceptor"},
    }
