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

def remove_sequence(seq: str, chains: list[int], sep: str = ":") -> str:
    '''removes all idx of seq.split(sep).'''
    return sep.join([subseq for i, subseq in enumerate(seq.split(sep)) if i not in chains])

def main(args) -> None:
    "removes sequences from .fa files in bulk."
    # parse inputs
    with open(args.input_json, 'r', encoding="UTF-8") as f:
        protein_dict = json.loads(f.read())

    # prep output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # parse fastas and remove 'chains'
    for protein_fasta, chains in list(protein_dict.items()):
        # load fasta from file
        protein_fasta_dict = parse_fasta_to_dict(protein_fasta) # {name_of_pose: seq_of_pose, ...}

        # parse description and sequence from fasta
        description, sequence = list(protein_fasta_dict.items())[0]

        # prep separator
        sep = args.sep or find_sep(sequence)

        # remove chains specified in input_json
        seq = remove_sequence(sequence, chains, sep=sep)

        # write back to .fa file
        sequence_dict_to_fasta({description: seq}, out_path=f"{args.output_dir}/chains_removed/")

    # write file to indicate we are done.
    with open(f"{args.output_dir}/done.txt", 'w', encoding="UTF-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " Done.\n")

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # inputs
    argparser.add_argument("--input_json", required=True, type=str, help="Path to json file with .pdb and chain indeces to remove: {{'pdb_path': [chain_idx], ...}}")

    # outputs
    argparser.add_argument("--inplace", action="store_true", help="Specify whether to modify the files inplace.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where to write the output .pdb files to.")
    argparser.add_argument("--sep", type=str, help="specify the charactor along which your sequences should be separated.")
    arguments = argparser.parse_args()

    # arguments safetycheck:
    if not os.path.isfile(arguments.input_json):
        raise FileNotFoundError(f"File specified for --input_json could not be found: {arguments.input_json}")

    main(arguments)
