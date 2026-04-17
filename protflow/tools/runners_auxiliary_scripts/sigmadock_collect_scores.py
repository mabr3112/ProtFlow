"""Auxiliary script — runs in the sigmadock Python environment.

Loads .pt outputs, exports complex PDBs, writes scores to JSON.

Usage:
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

            if mol_ref is None or coords is None:
                print(f"[WARN] Missing lig_ref or x0_hat for {pose_stem} — skipping", file=sys.stderr)
                continue

            out_complex = complex_dir / f"{pose_stem}.pdb"
            mol = _mol_with_coords(mol_ref, coords)
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
                    for k, v in pose_scores[0].items():
                        flat_key = k.lower().replace(" ", "_")
                        if _is_heavy(v):
                            if flat_key in include_scores:
                                row[flat_key] = json.dumps(v)
                        else:
                            row[flat_key] = v

            if pb is not None:
                rmsd = pb["rmsds"].get(code)
                pb_pass = pb["pb_checks"].get(code)
                if rmsd is not None:
                    row["rmsd"] = rmsd
                if pb_pass is not None:
                    row["pb_pass_rate"] = pb_pass
                pb_dict = pb["pb_dicts"].get(code)
                if pb_dict:
                    for k, v in pb_dict.items():
                        flat_key = f"pb_{k}"
                        if _is_heavy(v):
                            if flat_key in include_scores:
                                row[flat_key] = json.dumps(v)
                        else:
                            row[flat_key] = v

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
    print(f"Wrote {len(rows)} rows to {out_path}")
