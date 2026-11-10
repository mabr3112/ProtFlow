"""Batch BioPython structure file converter."""

# imports
import json
import os

# customs
from protflow.utils.biopython_tools import biopython_fileconverter

def replace_extension(path: str, ext: str) -> str:
    """Replace a file extension with ``ext``."""
    return f"{os.path.splitext(path)[0]}.{ext}"


def main(args) -> None:
    """Batch convert PDB/CIF structure files."""
    with open(args.input_json, "r", encoding="UTF-8") as j:
        in_dict = json.load(j)

    in_poses = in_dict["input_poses"]
    overwrite = in_dict["overwrite"]
    out_dir = in_dict["out_dir"]
    out_format = in_dict["out_format"]

    for pose in in_poses:
        out_path = os.path.join(out_dir, replace_extension(os.path.basename(pose), out_format))
        if not os.path.isfile(out_path) or overwrite is True:
            biopython_fileconverter(pose, out_format, out_path)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="JSON file containing input poses and conversion settings.")

    arguments = argparser.parse_args()
    main(arguments)
