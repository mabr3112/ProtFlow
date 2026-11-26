"""Regression tests for coupled atom assignments, partial selections and runner scoring."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from protflow.metrics import AtomRMSD, BiopythonMetricRunner, Dihedral, Distance, SymmetryAwareMetric
from protflow.metrics.rmsd import _setup_atom_selection_list
from protflow.residues import AtomSelection, ResidueSelection
from protflow.tools.runners_auxiliary_scripts.calc_atom_rmsd import calc_atom_rmsd, save_structure
from protflow.tools.runners_auxiliary_scripts import run_biopython_metrics as worker


@pytest.fixture
def structure():
    model = Model(0)
    structure = Structure("motif")
    structure.add(model)
    chain = Chain("A")
    model.add(chain)
    rng = np.random.default_rng(7)
    for i, (name, atoms) in enumerate([
        ("ASP", "N CA C O OD1 OD2"), ("HIS", "N CA C O ND1 CD2 CE1 NE2"),
        ("PHE", "N CA C O CD1 CD2 CE1 CE2"), ("ARG", "N CA C O NH1 NH2"),
        ("ASN", "N CA C O OD1 ND2"), ("GLU", "N CA C O OE1 OE2"),
    ], 1):
        residue = Residue((" ", i, " "), name, "")
        chain.add(residue)
        for a in atoms.split():
            residue.add(Atom(a, rng.normal(size=3) + i, 10, 1, " ", a, 1, a[0]))
    return structure


def flip(residue, a, b):
    residue[a].coord, residue[b].coord = residue[b].coord.copy(), residue[a].coord.copy()


@pytest.mark.parametrize("mode", ["exact", "backbone"])
def test_rmsd_fits_and_returns_consistent_transformation(structure, mode):
    target = deepcopy(structure)
    chain = target[0]["A"]
    for i, a, b in [(1, "OD1", "OD2"), (2, "ND1", "CD2"), (2, "CE1", "NE2"),
                    (3, "CD1", "CD2"), (3, "CE1", "CE2"), (4, "NH1", "NH2"),
                    (5, "OD1", "ND2"), (6, "OE1", "OE2")]:
        flip(chain[i], a, b)
    target.transform(np.array([[0,1,0],[-1,0,0],[0,0,1]]), np.array([3,6,4]))
    selection = {"residues": [["A", i] for i in range(1, 7)]}
    assert calc_atom_rmsd(target, structure, selection, selection, symmetry=mode, apply_transform=False) > 0.1
    rmsd = calc_atom_rmsd(target, structure, selection, selection, symmetry=mode,
                         his_ring_flip=True, amide_flip=True)
    assert rmsd < 1e-6
    np.testing.assert_allclose(chain[1]["OD2"].coord, structure[0]["A"][1]["OD1"].coord, atol=1e-6)


def test_partial_selection_and_inconsistent_ring_swaps(structure):
    target = deepcopy(structure)
    flip(target[0]["A"][1], "OD1", "OD2")
    atoms = [["A", 1, "OD1"]]
    assert calc_atom_rmsd(target, structure, atoms, atoms, symmetry="backbone") < 1e-6
    flip(target[0]["A"][3], "CD1", "CD2")
    ring = [["A", 3, a] for a in ("CD1", "CD2", "CE1", "CE2")]
    assert calc_atom_rmsd(target, structure, ring, ring, symmetry="backbone") > 0.1


def test_ordered_static_and_saved_residue_selections():
    residues = ResidueSelection(["B17", "A9"])
    df = pd.DataFrame({"motif": [residues, {"residues": [["B", 17], ["A", 9]]}]})
    poses = type("Rows", (list,), {})(range(2))
    poses.df = df
    for selection in (residues, "motif"):
        out = _setup_atom_selection_list(poses, selection, "motif")
        assert json.loads(json.dumps(out)) == [{"residues": [["B", 17], ["A", 9]]}]*2


def test_independent_his_nitrogens_and_potential_ranking(structure):
    metric = Distance(name="contact", atoms=AtomSelection([("A", 1, "OD1"), ("A", 2, "ND1")]))
    wrapped = SymmetryAwareMetric(metric, lambda v: abs(v - 6), expected_value=6)
    candidates = worker.calculate_metric(metric, structure[0], wrapped.parse({}))
    assert len(candidates) == 4
    values = [v for v, _ in candidates]
    extreme = max(values)
    wrapped = SymmetryAwareMetric(metric, lambda v: -v, expected_value=min(values))
    value, potential, atoms = wrapped.choose(candidates)
    assert value == extreme and potential == -extreme
    assert [atom[-1][0] for atom in atoms] == next([a[-1][0] for a in ids] for v, ids in candidates if v == value)
    no_his = SymmetryAwareMetric(metric, abs, 0, his_nitrogens=False)
    assert len(worker.calculate_metric(metric, structure[0], no_his.parse({}))) == 2


def test_periodic_dihedral_potential_tie():
    metric = SymmetryAwareMetric(Dihedral(), lambda _: 0, expected_value=179)
    assert metric.choose([(150, ["literal"]), (-179, ["alternate"])]) == (-179, 0, ["alternate"])


def test_runner_round_trip_and_cached_potential_rescore(tmp_path, structure, monkeypatch):
    from protflow.poses import Poses
    from protflow.jobstarters import LocalJobStarter
    path = str(tmp_path/"input.pdb")
    save_structure(structure, path)
    poses = Poses(poses=[path], work_dir=str(tmp_path/"work"))
    jobs = LocalJobStarter(max_cores=1)
    import sys
    from pathlib import Path
    scripts = Path(worker.__file__).parent
    runner = BiopythonMetricRunner(jobstarter=jobs, script_path=str(scripts/"run_biopython_metrics.py"),
                                  python_path=sys.executable)
    distance = Distance(name="contact", atoms=[("A", 1, "OD2"), ("A", 2, "ND1")])
    score = lambda sign: SymmetryAwareMetric(distance, lambda v: sign*v, expected_value=3)
    runner.run(poses, "geom", metrics=[score(1)])
    short = poses.df.geom_contact.iloc[0]
    poses = Poses(poses=[path], work_dir=str(tmp_path/"work"))
    with monkeypatch.context() as patch:
        patch.setattr(jobs, "start", lambda **_: pytest.fail("cached coordinates should be reused"))
        runner.run(poses, "geom", metrics=[score(-1)])
    assert poses.df.geom_contact.iloc[0] > short
    assert poses.df.geom_contact_potential.iloc[0] == -poses.df.geom_contact.iloc[0]
    assert len(poses.df.geom_contact_atoms.iloc[0]) == 2
    rmsd = AtomRMSD(ref_path=path, ref_atoms=ResidueSelection(["A1","A2"]),
                    target_atoms=ResidueSelection(["A1","A2"]), symmetry="backbone", jobstarter=jobs)
    rmsd.python, rmsd.script_dir = sys.executable, str(scripts)
    rmsd.run(poses, "backbone_fit")
    assert poses.df.backbone_fit_rmsd.iloc[0] < 1e-6


def test_acid_proton_and_incomplete_alternative(structure):
    metric = Distance(name="contact", atoms=[("A", 1, "OD1"), ("A", 3, "CA")])
    wrapped = SymmetryAwareMetric(metric, abs, 0)
    aspartate = structure[0]["A"][1]
    aspartate.add(Atom("HD2", np.zeros(3), 10, 1, " ", "HD2", 9, "H"))
    assert len(worker.calculate_metric(metric, structure[0], wrapped.parse({}))) == 1
    aspartate.detach_child("HD2")
    aspartate.detach_child("OD2")
    assert len(worker.calculate_metric(metric, structure[0], wrapped.parse({}))) == 1


def test_explicit_conformer_does_not_change_next_metric(structure):
    from Bio.PDB.Atom import DisorderedAtom
    residue = structure[0]["A"][1]
    for name in ("OD1", "OD2"):
        atom = residue[name]
        residue.detach_child(name)
        wrapper = DisorderedAtom(name)
        residue.add(wrapper)
        for label, occupancy in (("A", 0.7), ("B", 0.3)):
            child = deepcopy(atom)
            child.set_altloc(label)
            child.set_occupancy(occupancy)
            wrapper.disordered_add(child)
    metric = Distance(name="contact", atoms=[("A", 1, ("OD1", "B")), ("A", 3, "CA")])
    spec = SymmetryAwareMetric(metric, abs, 0).parse({})
    alternatives = worker.calculate_metric(metric, structure[0], spec)
    assert [ids[0][-1] for _, ids in alternatives] == [("OD1", "B"), ("OD2", "B")]
    assert residue["OD1"].get_altloc() == residue["OD2"].get_altloc() == "A"
