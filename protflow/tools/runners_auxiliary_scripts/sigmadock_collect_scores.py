"""Auxiliary script for parsing SigmaDock inference outputs.

This script runs inside the SigmaDock Python environment (requires PyTorch and
RDKit) and is invoked as a subprocess by :func:`protflow.tools.sigmadock.collect_scores`.
It should never be imported directly from within the ProtFlow environment.

For each pose directory under ``<work_dir>/outputs/``, the script:

1. Loads ``predictions.pt``, ``rescoring.pt`` (if present), and
   ``posebusters.pt`` (if present) from the first ``predictions.pt``
   seed directory found via recursive glob.
2. Reconstructs the docked ligand molecule by placing the predicted
   ``x0_hat`` coordinates onto the reference RDKit mol (``lig_ref``).
3. Writes a protein–ligand complex PDB to
   ``<work_dir>/outputs/<pose_stem>/complexes/<pose_stem>.pdb`` by
   concatenating the receptor ATOM lines with HETATM lines from the
   reconstructed ligand.
4. Collects scalar scores from ``rescoring.pt`` (e.g. ``affinity``,
   ``intramolecular_energy``) and from ``posebusters.pt`` (``rmsd``,
   ``pb_pass_rate``).  Heavy tensor/list values are skipped unless their
   key appears in ``include_scores``.
5. Writes all rows as a JSON array to
   ``<work_dir>/sigmadock_scores.json`` and exits.  The calling
   :func:`~protflow.tools.sigmadock.collect_scores` reads this file and
   deletes it immediately after parsing.

Parameters
----------
work_dir : str
    Root directory of a single SigmaDock runner invocation (the directory
    that contains the ``outputs/`` sub-tree).
--include-scores : str, optional
    Comma-separated list of score keys whose values are heavy (tensors or
    long lists) but should be serialised as JSON strings and included in
    the output anyway.  Example: ``--include-scores torsion_angles,feat``.

Output columns (always present when the corresponding file exists)
------------------------------------------------------------------
description : str
    Stem of the pose directory name (matches ``poses.df["description"]``).
location : str
    Absolute path to the written complex PDB.
affinity : float
    Predicted binding affinity from ``rescoring.pt`` (lower is better).
intramolecular_energy : float
    Intramolecular strain energy from ``rescoring.pt``.
rmsd : float
    Ligand RMSD to the reference pose from ``posebusters.pt``.  Only
    present when a reference SDF was supplied (redocking) or when
    PoseBusters computed it.
pb_pass_rate : float
    Fraction of PoseBusters checks that passed for this pose.

Usage
-----
    python sigmadock_collect_scores.py <work_dir> [--include-scores key1,key2]
"""

import json
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D


def _is_heavy(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple, dict)):
            return True
        if len(value) > 200:
            return True
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return True
    return False


def _set_residue_name(mol, name: str):
    """Set PDB residue name on every atom (controls HETATM resName in the complex PDB)."""
    padded = name[:3].ljust(3)
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is not None:
            info.SetResidueName(padded)
        else:
            new_info = Chem.AtomPDBResidueInfo()
            new_info.SetResidueName(padded)
            new_info.SetIsHeteroAtom(True)  # ensures HETATM not ATOM in MolToPDBBlock
            atom.SetMonomerInfo(new_info)
    return mol


def _mol_with_coords(mol_ref, coords):
    if hasattr(coords, "float"):
        coords = coords.float().cpu().tolist()
    mol = Chem.RWMol(mol_ref)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def _write_complex_pdb(prot_path: str, mol, out_path: Path) -> None:
    lig_lines = [ln for ln in Chem.MolToPDBBlock(mol, flavor=4).splitlines() if ln.startswith(("HETATM", "CONECT"))]
    with open(prot_path) as f:
        prot_lines = [ln.rstrip() for ln in f if not ln.startswith(("END", "CONECT"))]
    with open(out_path, "w") as f:
        f.write("\n".join(prot_lines) + "\n")
        f.write("\n".join(lig_lines) + "\n")
        f.write("END\n")


def collect(work_dir: str, include_scores: set[str] | None = None) -> list[dict]:
    include_scores = include_scores or set()
    output_root = Path(work_dir) / "outputs"
    rows = []

    for pose_dir in sorted(output_root.iterdir()):
        if not pose_dir.is_dir():
            continue
        pose_stem = pose_dir.name

        pred_files = sorted(pose_dir.rglob("predictions.pt"))
        if not pred_files:
            print(f"[WARN] No predictions.pt under {pose_dir}", file=sys.stderr)
            continue
        seed_dir = pred_files[0].parent

        pred    = torch.load(seed_dir / "predictions.pt", weights_only=False)
        rescore = torch.load(seed_dir / "rescoring.pt",   weights_only=False) if (seed_dir / "rescoring.pt").exists()   else None
        pb      = torch.load(seed_dir / "posebusters.pt", weights_only=False) if (seed_dir / "posebusters.pt").exists() else None

        complex_dir = pose_dir / "complexes"
        complex_dir.mkdir(exist_ok=True)

        for code, samples in pred["results"].items():
            if not samples:
                continue
            s        = samples[0]
            mol_ref  = s.get("lig_ref")
            coords   = s.get("x0_hat")
            pdb_path = s.get("pdb_path")
            ligand_path = s.get("ligand_path")
            

            if mol_ref is None or coords is None:
                print(f"[WARN] Missing lig_ref or x0_hat for {pose_stem} — skipping", file=sys.stderr)
                continue

            out_complex = complex_dir / f"{pose_stem}.pdb"
            mol = _mol_with_coords(mol_ref, coords)
            if ligand_path and Path(ligand_path).exists():
                lig_name = Path(ligand_path).read_text().splitlines()[0].strip() or "LIG"
                mol = _set_residue_name(mol, lig_name)
            if pdb_path and Path(pdb_path).exists():
                _write_complex_pdb(pdb_path, mol, out_complex)
            else:
                Chem.MolToPDBFile(mol, str(out_complex))

            row = {
                "description": pose_stem,
                "location":    str(out_complex.resolve()),
            }

            if rescore is not None:
                pose_scores = rescore["scores"].get(code)
                if pose_scores:
                    seed_idx = s.get("seed", 0)
                    sc = pose_scores[seed_idx] if len(pose_scores) > seed_idx else pose_scores[0]
                    for k, v in sc.items():
                        flat_key = k.lower().replace(" ", "_")
                        if _is_heavy(v):
                            if flat_key in include_scores:
                                row[flat_key] = json.dumps(v)
                        else:
                            row[flat_key] = v
            else:
                print(f"[DEBUG] No rescoring.pt found for {pose_stem}", file=sys.stderr)

            if pb is not None:
                rmsd = pb["rmsds"].get(code) # use these hardcode calls since pb gives a lot of unrelevant boolen scores
                pb_pass = pb["pb_checks"].get(code)
                if rmsd is not None:
                    row["rmsd"] = rmsd
                else:
                    print(f"[WARN] No RMSD found for {pose_stem} / {code} in posebusters.pt", file=sys.stderr)
                if pb_pass is not None:
                    row["pb_pass_rate"] = pb_pass
                else:
                    print(f"[WARN] No pb_pass_rate found for {pose_stem} / {code} in posebusters.pt", file=sys.stderr)

            rows.append(row)

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir")
    parser.add_argument("--include-scores", default="")
    args = parser.parse_args()

    include_scores = set(args.include_scores.split(",")) if args.include_scores else set()
    rows = collect(args.work_dir, include_scores=include_scores)
    out_path = Path(args.work_dir) / "sigmadock_scores.json"
    with open(out_path, "w") as f:
        json.dump(rows, f)
