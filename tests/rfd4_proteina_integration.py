#!/usr/bin/env python
"""End-to-end integration test for the RFD4-Proteina runner.

Runs a small real generation through the full ProtFlow lifecycle and checks the
parts that unit tests cannot reach: that ``rfd4-generate`` accepts the command
the runner assembles, that the condition specs are semantically valid, that
output names survive index-layer collapse and reindexing, and that the scorefile
cache is reused on a second call.

This needs a GPU and the RFD4-Proteina environment, so it is a script rather
than a pytest case. Run it on the machine that has both::

    python tests/rfd4_proteina_integration.py                 # local GPU
    python tests/rfd4_proteina_integration.py --slurm         # submit via sbatch
    python tests/rfd4_proteina_integration.py --dry-run       # build inputs only, no GPU

Configuration comes from ``protflow/config.py``
(``RFD4_PROTEINA_BIN_PATH``, ``RFD4_PROTEINA_DATA_PATH``, ``RFD4_PROTEINA_PRE_CMD``).

Defaults are sized as a smoke test: 2 inputs x 2 replicates = 4 designs.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from glob import glob

from protflow.jobstarters import LocalJobStarter, SbatchArrayJobstarter
from protflow.poses import Poses
from protflow.residues import ResidueSelection
from protflow.tools.rfd4_proteina import RFD4Params, RFD4Proteina

# Benchmark binder inputs that ship with RFD4-Proteina; already annotated, so they
# run as-is and isolate runner behaviour from input-construction mistakes.
DEFAULT_INPUT_GLOB = "/home/adrian/software/RFD4-Proteina/data/benchmarks/ppi/ppi_bb_only/*.cif*"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default=DEFAULT_INPUT_GLOB, help="glob of RFD4 input structures")
    parser.add_argument("--n-inputs", type=int, default=2, help="how many inputs from the glob to use")
    parser.add_argument("--work-dir", default="./rfd4_integration", help="ProtFlow working directory")
    parser.add_argument("--num-replicates", type=int, default=2, help="designs per input")
    parser.add_argument("--batch-size", type=int, default=1, help="designs per GPU batch")
    parser.add_argument("--max-cores", type=int, default=1, help="parallel jobs (GPUs) to use")
    parser.add_argument("--metrics-tags", default="design", help="comma-separated inline metric tags, '' for none")
    parser.add_argument("--nsteps", type=int, default=None, help="override generation.args.nsteps to go faster")
    parser.add_argument("--keep-cif", action="store_true", help="keep CIF poses instead of converting to PDB")
    parser.add_argument("--remap-motifs", action="store_true",
                        help="also exercise motif remapping (adds a motif column and passes update_motifs)")
    parser.add_argument("--renumber-input", action="store_true",
                        help="also write a renumbered copy of each design's input structure")
    parser.add_argument("--remap-cores", type=int, default=None,
                        help="cores for the CPU-only remapping jobstarter (default: same jobstarter as the run)")
    parser.add_argument("--slurm", action="store_true", help="submit via sbatch instead of running locally")
    parser.add_argument("--dry-run", action="store_true", help="build poses, params and commands, but do not run")
    parser.add_argument("--overwrite", action="store_true", help="discard previous outputs in --work-dir")
    return parser.parse_args()


def rfd4_venv_python() -> str | None:
    """Return the python interpreter of the configured RFD4-Proteina environment."""
    from protflow import get_config

    binary = getattr(get_config(), "RFD4_PROTEINA_BIN_PATH", "")
    candidate = os.path.join(os.path.dirname(str(binary)), "python")
    return candidate if binary and os.path.isfile(candidate) else None


def ensure_plain_cif(path: str, out_dir: str) -> str:
    """
    Decompress a ``.cif.zst`` / ``.cif.gz`` input into *out_dir*.

    RFD4 reads compressed inputs, but a condition spec's ``input`` field is
    documented for ``.cif``, ``.cif.gz`` and ``.bcif``; decompressing keeps the
    test independent of that. Plain ``.cif`` inputs are copied unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(path)

    if path.endswith(".cif"):
        target = os.path.join(out_dir, name)
        shutil.copy2(path, target)
        return os.path.abspath(target)

    if path.endswith(".cif.gz"):
        import gzip

        target = os.path.join(out_dir, name[: -len(".gz")])
        with gzip.open(path, "rb") as f_in, open(target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return os.path.abspath(target)

    if path.endswith(".cif.zst"):
        target = os.path.join(out_dir, name[: -len(".zst")])
        try:
            import zstandard  # noqa: F401 - availability probe

            with open(path, "rb") as f_in, open(target, "wb") as f_out:
                zstandard.ZstdDecompressor().copy_stream(f_in, f_out)
            return os.path.abspath(target)
        except ImportError:
            pass

        # zstandard lives in the RFD4 environment, not necessarily in the ProtFlow one
        if (venv_python := rfd4_venv_python()) is not None:
            subprocess.run(
                [venv_python, "-c",
                 "import sys, zstandard; "
                 "zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1],'rb'), open(sys.argv[2],'wb'))",
                 path, target],
                check=True,
            )
            return os.path.abspath(target)

        if shutil.which("zstd"):
            subprocess.run(["zstd", "-d", "-f", path, "-o", target], check=True)
            return os.path.abspath(target)

        raise RuntimeError(f"Cannot decompress {path}: install zstandard, or provide plain .cif inputs via --inputs.")

    raise ValueError(f"RFD4-Proteina reads .cif, .cif.gz and .bcif inputs; got {path}")


def report(poses: Poses, prefix: str, expected: int) -> None:
    """Print what the run produced and check the runner's own guarantees."""
    df = poses.df
    print(f"\n{'=' * 70}\nRESULT\n{'=' * 70}")
    print(f"poses returned      : {len(df)} (expected {expected})")
    print(f"descriptions        : {list(df['poses_description'])[:6]}")

    missing = [path for path in df["poses"] if not os.path.isfile(path)]
    print(f"all pose files exist: {not missing}")
    if missing:
        print(f"  MISSING: {missing[:3]}")

    stems_match = all(
        os.path.splitext(os.path.basename(path))[0] == description
        for path, description in zip(df["poses"], df["poses_description"])
    )
    print(f"description == stem : {stems_match}")
    print(f"unique descriptions : {df['poses_description'].nunique()} / {len(df)}")

    cif_col = f"{prefix}_cif_location"
    if cif_col in df.columns:
        print(f"annotated CIFs kept : {all(os.path.isfile(p) for p in df[cif_col])}")

    metric_cols = [
        c for c in df.columns
        if c.startswith(f"{prefix}_") and c not in
        (f"{prefix}_location", f"{prefix}_description", f"{prefix}_cif_location", f"{prefix}_task")
    ]
    print(f"metric columns      : {len(metric_cols)}")
    if metric_cols:
        print(f"  e.g. {metric_cols[:4]}")

    if len(df) != expected:
        print(f"\nWARNING: expected {expected} designs, got {len(df)}. Check the job logs in the work directory.")


def main() -> int:
    """Build inputs, run the runner twice (second call must hit the cache) and report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # --- inputs
    sources = sorted(glob(args.inputs))[: args.n_inputs]
    if not sources:
        print(f"No inputs matched {args.inputs!r}", file=sys.stderr)
        return 1
    input_cifs = [ensure_plain_cif(path, os.path.join(work_dir, "input_cifs")) for path in sources]
    print(f"inputs: {[os.path.basename(p) for p in input_cifs]}")

    # --- poses + condition spec
    poses = Poses(poses=input_cifs, work_dir=work_dir, storage_format="json")

    params = RFD4Params(poses=poses)
    # the benchmark CIFs already carry their conditioning, so load them as annotated
    # and let the shipped conditions drive the design
    params.set_input_specs(input_as_annotated_cif=True)

    options = f"generation.args.nsteps={args.nsteps}" if args.nsteps else None
    metrics_tags = [tag for tag in args.metrics_tags.split(",") if tag] or None

    jobstarter = (
        SbatchArrayJobstarter(max_cores=args.max_cores, gpus=1)
        if args.slurm else LocalJobStarter(max_cores=args.max_cores)
    )

    runner = RFD4Proteina(jobstarter=jobstarter)

    # motif remapping is CPU-only, so it can use a wider jobstarter than generation
    renumber_jobstarter = LocalJobStarter(max_cores=args.remap_cores) if args.remap_cores else None
    update_motifs = None
    if args.remap_motifs:
        # a stand-in motif on the conditioned target chain; after the run these
        # positions should be expressed in each design's own numbering
        poses.df["motif"] = [ResidueSelection(["B45", "B46", "B82"]) for _ in range(len(poses.df))]
        update_motifs = ["motif"]

    if args.dry_run:
        spec = params.write_spec(os.path.join(work_dir, "dry_run_spec.yaml"))
        print(f"\n--- condition spec ({spec}) ---")
        print(open(spec, "r", encoding="utf-8").read())
        print("--- command that would be submitted ---")
        print(runner.write_cmd(
            in_spec=spec,
            out_dir=os.path.join(work_dir, "outputs"),
            hydra_run_dir=os.path.join(work_dir, "hydra_outputs", "batch0"),
            num_replicates=args.num_replicates,
            batch_size=args.batch_size,
            metrics_tags=metrics_tags,
            options=options,
        ))
        print("\ndry run only - nothing was executed.")
        return 0

    # --- run
    prefix = "rfd4_test"
    poses = runner.run(
        poses=poses,
        prefix=prefix,
        params=params,
        num_replicates=args.num_replicates,
        batch_size=args.batch_size,
        metrics_tags=metrics_tags,
        options=options,
        update_motifs=update_motifs,
        renumber_input=args.renumber_input,
        renumber_jobstarter=renumber_jobstarter,
        convert_cif_to_pdb=not args.keep_cif,
        overwrite=args.overwrite,
    )
    report(poses, prefix=prefix, expected=len(input_cifs) * args.num_replicates)

    if update_motifs:
        print(f"\n{'=' * 70}\nMOTIF REMAPPING\n{'=' * 70}")
        map_col = f"{prefix}_source_index_map"
        sizes = [len(m) if isinstance(m, dict) else 0 for m in poses.df[map_col]]
        print(f"source index map sizes : {sizes}")
        print(f"motif residues/design  : {[len(m) for m in poses.df[f'{prefix}_motif_residues']]}")
        print(f"remapped motif (row 0) : {poses.df['motif'][0]}")
        print("  (input was B45, B46, B82 - identical output means this input kept its numbering)")

    if args.renumber_input:
        col = f"{prefix}_renumbered_input"
        paths = [path for path in poses.df[col] if path]
        print(f"\n{'=' * 70}\nRENUMBERED INPUTS\n{'=' * 70}")
        print(f"written : {len(paths)} / {len(poses.df)}")
        print(f"exist   : {all(os.path.isfile(path) for path in paths)}")
        if paths:
            print(f"e.g.    : {paths[0]}")

    # --- second call must reuse the cached scorefile rather than regenerate
    print(f"\n{'=' * 70}\nCACHE REUSE (second run, overwrite=False)\n{'=' * 70}")
    poses_again = Poses(poses=input_cifs, work_dir=work_dir, storage_format="json")
    if update_motifs:
        poses_again.df["motif"] = [ResidueSelection(["B45", "B46", "B82"]) for _ in range(len(poses_again.df))]
    params_again = RFD4Params(poses=poses_again)
    params_again.set_input_specs(input_as_annotated_cif=True)
    poses_again = runner.run(
        poses=poses_again, prefix=prefix, params=params_again,
        num_replicates=args.num_replicates, batch_size=args.batch_size,
        metrics_tags=metrics_tags, options=options,
        update_motifs=update_motifs, renumber_input=args.renumber_input,
        renumber_jobstarter=renumber_jobstarter,
        convert_cif_to_pdb=not args.keep_cif, overwrite=False,
    )
    print(f"cached run returned : {len(poses_again.df)} poses (should equal the first run)")

    print(f"\nwork directory: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
