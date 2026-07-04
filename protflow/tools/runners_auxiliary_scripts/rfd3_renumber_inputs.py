"""Worker for parallel RFDiffusion3 input renumbering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protflow.tools.rfdiffusion3 import _rfd3_renumber_input_record


def main(args):
    """Renumber all records in one worker input JSON."""
    with open(args.input_json, "r", encoding="UTF-8") as handle:
        records = json.load(handle)

    results = [_rfd3_renumber_input_record(record) for record in records]

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="UTF-8") as handle:
        json.dump(results, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_json", required=True, help="Worker input JSON containing renumbering records.")
    parser.add_argument("--output_json", required=True, help="Worker output JSON path.")
    main(parser.parse_args())
