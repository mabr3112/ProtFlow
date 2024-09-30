'''Python Script to remove sequences from .fa files in bulk.'''
# imports
import json
import os
import time

# custom
from protflow.utils.utils import parse_fasta_to_dict, sequence_dict_to_fasta

def find_sep(input_str: str) -> str:
    '''Finds separator in protein sequence.'''
    return "/" if "/" in input_str else ":"

def add_sequence_str(seq: str, insert_seq: str, insert_idx: int = -1, sep: str = ":") -> str:
    '''Adds 'insert_seq' into seq after splitting by 'sep' at 'insert_idx'.'''
    # split sequence along separator
    parts = seq.split(sep)

    # convert negative indeces to positive ones (.insert method inserts before 'insert_idx', so -1 would insert before the last element as an example)
    insert_idx = insert_idx if insert_idx > 0 else len(parts) + insert_idx + 1

    # insert and return
    parts.insert(insert_idx, insert_seq)
    return sep.join(parts)

def add_sequence(fasta_path: str, insert_seq: str, sep: str = ":", insert_idx: int = -1, out_path: str = None) -> None:
    '''If no 'out_path' is specified, the function will operate on the .fa file in-place, be aware!'''
    # load fasta from file
    protein_fasta_dict = parse_fasta_to_dict(fasta_path) # {name_of_pose: seq_of_pose, ...}

    # parse description and sequence from fasta
    description, sequence = list(protein_fasta_dict.items())[0]

    # prep separator
    sep = sep or find_sep(sequence)

    # add chains specified in input_json
    seq = add_sequence_str(sequence, insert_seq, insert_idx, sep=sep)

    # write back to .fa file
    sequence_dict_to_fasta({description: seq}, out_path=out_path)

def main(args) -> None:
    "removes sequences from .fa files in bulk."
    # parse inputs
    with open(args.input_json, 'r', encoding="UTF-8") as f:
        protein_dict = json.loads(f.read())

    # prep output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # parse fastas and add 'chains'
    for protein_fasta, kwargs in list(protein_dict.items()):
        if "seq" not in kwargs:
            raise ValueError(f"No Sequence specified that should be added. If you want no sequence, don't add it.")
        if "insert_idx" not in kwargs:
            kwargs["insert_idx"] = -1

        add_sequence(
            fasta_path=protein_fasta,
            insert_seq=kwargs["seq"],
            sep=args.sep,
            insert_idx=kwargs["insert_idx"],
            out_path=f"{args.output_dir}/sequence_added/"
        )

    # write file to indicate we are done.
    with open(f"{args.output_dir}/done.txt", 'w', encoding="UTF-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " Done.\n")

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # inputs
    argparser.add_argument("--input_json", required=True, type=str, help="Path to json file with .pdb and chain indeces to remove: {'pdb_path': {insert_idx: [chain_idx], seq: 'SEQVENCE'}}")

    # outputs
    argparser.add_argument("--inplace", action="store_true", help="Specify whether to modify the files inplace.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where to write the output .pdb files to.")
    argparser.add_argument("--sep", type=str, help="specify the charactor along which your sequences should be separated.")
    arguments = argparser.parse_args()

    # arguments safetycheck:
    if not os.path.isfile(arguments.input_json):
        raise FileNotFoundError(f"File specified for --input_json could not be found: {arguments.input_json}")

    main(arguments)
