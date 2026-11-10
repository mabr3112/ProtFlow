import pandas as pd

from protflow.tools.boltz import (
    BoltzParams,
    boltz_yaml_reader,
    boltz_yaml_writer,
    convert_chain_seq_dict_to_yaml_dict,
)


class _Poses:
    def __init__(self, pose_path):
        self.df = pd.DataFrame({"poses": [str(pose_path)]})

    def poses_list(self):
        return self.df["poses"].to_list()

    def __iter__(self):
        yield from (row for _, row in self.df.iterrows())


def test_convert_chain_seq_dict_omits_msa_for_server_mode():
    protein_entries = convert_chain_seq_dict_to_yaml_dict({"A": "ACDE"}, msa="server")

    assert protein_entries == [{"id": "A", "sequence": "ACDE"}]


def test_convert_chain_seq_dict_keeps_explicit_empty_msa():
    protein_entries = convert_chain_seq_dict_to_yaml_dict({"A": "ACDE"}, msa="empty")

    assert protein_entries == [{"id": "A", "sequence": "ACDE", "msa": "empty"}]


def test_boltz_params_default_msa_uses_server_mode_without_writing_msa(tmp_path):
    pose_path = tmp_path / "pose.yaml"
    boltz_yaml_writer(str(pose_path), {"sequences": []})
    poses = _Poses(pose_path)
    params = BoltzParams()
    params.add_protein(sequence="ACDE", id="A")

    params.generate_yaml_files(poses, str(tmp_path / "out"), default_msa="server")

    generated_yaml = boltz_yaml_reader(poses.poses_list()[0])
    protein = generated_yaml["sequences"][0]["protein"]
    assert protein == {"id": "A", "sequence": "ACDE", "cyclic": False}


def test_boltz_params_explicit_empty_msa_overrides_server_default(tmp_path):
    pose_path = tmp_path / "pose.yaml"
    boltz_yaml_writer(str(pose_path), {"sequences": []})
    poses = _Poses(pose_path)
    params = BoltzParams()
    params.add_protein(sequence="ACDE", id="A", msa="empty")

    params.generate_yaml_files(poses, str(tmp_path / "out"), default_msa="server")

    generated_yaml = boltz_yaml_reader(poses.poses_list()[0])
    protein = generated_yaml["sequences"][0]["protein"]
    assert protein["msa"] == "empty"
