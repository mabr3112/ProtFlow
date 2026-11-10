import json
from pathlib import Path
from types import SimpleNamespace

from protflow.poses import Poses
from protflow.utils.biopython_tools import biopython_fileconverter
from protflow.tools.runners_auxiliary_scripts.biopython_fileconverter import main as biopython_fileconverter_main


def _write_ligand_cif(tmp_path: Path) -> Path:
    cif = tmp_path / "chain_ligand.cif"
    cif.write_text(
        """data_chain_ligand
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM   1 C CA . ALA A 1 1 0.000 0.000 0.000 A 1 ? 1.00 10.00 1
ATOM   2 C C  . ALA A 1 1 1.500 0.000 0.000 A 1 ? 1.00 10.00 1
HETATM 3 C C1 . LIG B 2 . 4.000 0.000 0.000 Z 1 ? 1.00 20.00 1
HETATM 4 O O1 . LIG B 2 . 5.200 0.000 0.000 Z 1 ? 1.00 20.00 1
#
""",
        encoding="UTF-8",
    )
    return cif


def _chain_residue_records(pdb_path: str) -> set[tuple[str, str, int]]:
    records = set()
    for line in Path(pdb_path).read_text(encoding="UTF-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            records.add((line[21].strip(), line[17:20].strip(), int(line[22:26])))
    return records


def test_biopython_fileconverter_preserves_author_chain_and_residue_id(tmp_path):
    cif = _write_ligand_cif(tmp_path)
    pdb = biopython_fileconverter(str(cif), "pdb", str(tmp_path / "converted.pdb"))

    records = _chain_residue_records(pdb)

    assert ("A", "ALA", 1) in records
    assert ("Z", "LIG", 1) in records


def test_biopython_fileconverter_auxiliary_script_preserves_ligand_metadata(tmp_path):
    cif = _write_ligand_cif(tmp_path)
    out_dir = tmp_path / "batch"
    out_dir.mkdir()
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"input_poses": [str(cif)], "out_dir": str(out_dir), "out_format": "pdb", "overwrite": True}),
        encoding="UTF-8",
    )

    biopython_fileconverter_main(SimpleNamespace(input_json=str(input_json)))

    records = _chain_residue_records(str(out_dir / "chain_ligand.pdb"))
    assert ("A", "ALA", 1) in records
    assert ("Z", "LIG", 1) in records


def test_convert_poses_defaults_to_biopython_and_preserves_ligand_metadata(tmp_path):
    cif = _write_ligand_cif(tmp_path)
    poses = Poses(poses=[str(cif)], work_dir=str(tmp_path / "work"))

    poses.convert_poses("converted", "pdb", overwrite=True)

    records = _chain_residue_records(poses.df.loc[0, "poses"])
    assert ("A", "ALA", 1) in records
    assert ("Z", "LIG", 1) in records


def test_convert_poses_jobstarter_uses_configured_biopython_auxiliary_script(tmp_path):
    class DummyJobStarter:
        max_cores = 1

        def start(self, cmds, jobname, wait, output_path):
            self.cmds = cmds
            for cmd in cmds:
                input_json = cmd.rsplit("--input_json ", maxsplit=1)[-1]
                data = json.loads(Path(input_json).read_text(encoding="UTF-8"))
                for pose in data["input_poses"]:
                    out_path = Path(data["out_dir"]) / f"{Path(pose).stem}.{data['out_format']}"
                    biopython_fileconverter(pose, data["out_format"], str(out_path))

    cif = _write_ligand_cif(tmp_path)
    poses = Poses(poses=[str(cif)], work_dir=str(tmp_path / "work"))
    jobstarter = DummyJobStarter()

    poses.convert_poses("converted", "pdb", jobstarter=jobstarter, overwrite=True)

    assert "runners_auxiliary_scripts/biopython_fileconverter.py" in jobstarter.cmds[0]
    records = _chain_residue_records(poses.df.loc[0, "poses"])
    assert ("Z", "LIG", 1) in records


def test_convert_poses_can_dispatch_to_openbabel(tmp_path, monkeypatch):
    cif = _write_ligand_cif(tmp_path)
    poses = Poses(poses=[str(cif)], work_dir=str(tmp_path / "work"))
    called = {}

    def fake_openbabel_fileconverter(input_file, output_format, output_file):
        called["args"] = (input_file, output_format, output_file)
        Path(output_file).write_text("END\n", encoding="UTF-8")
        return output_file

    monkeypatch.setattr("protflow.poses.openbabel_fileconverter", fake_openbabel_fileconverter)

    poses.convert_poses("converted", "pdb", overwrite=True, conversion_engine="openbabel")

    assert called["args"] == (str(cif), "pdb", poses.df.loc[0, "poses"])
    assert Path(poses.df.loc[0, "poses"]).is_file()


def test_convert_poses_rejects_unknown_conversion_engine(tmp_path):
    cif = _write_ligand_cif(tmp_path)
    poses = Poses(poses=[str(cif)], work_dir=str(tmp_path / "work"))

    try:
        poses.convert_poses("converted", "pdb", conversion_engine="unknown")
    except ValueError as exc:
        assert "conversion_engine" in str(exc)
    else:
        raise AssertionError("convert_poses accepted an unknown conversion engine")
