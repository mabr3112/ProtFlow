"""
Script that takes cleavage positions as input and predicts cleavage products with AlphaFold2 as output.
"""
# Imports
import logging
from itertools import combinations
import os

# dependencies
from protslurm import jobstarters
from protslurm.poses import Poses
from protslurm.residues import ResidueSelection
from protslurm.tools.alphafold2 import Alphafold2
from protslurm.utils.biopython_tools import get_sequence_from_pose, load_structure_from_pdbfile

def read_fasta(input_str: str, sep: str) -> dict:
    '''Reads fasta file with singular fasta inside and returns as dict {chain: seq(chain), ...}'''
    # read sequence (supports multiline sequences)
    with open(input_str, 'r', encoding="UTF-8") as f:
        split_seq = "".join([x for x in f.read().split("\n")[1:] if x]).split(sep)

    # assemble seq_dict and return. chr(65) := "A"
    return {chr(65 + i): seq for i, seq in enumerate(split_seq)}

def read_pdb(input_pdb: str, sep: str) -> dict:
    '''Collects sequence from .pdb file and returns sequence as dict {chain: seq(chain), ...}'''
    # load pose:
    pose = load_structure_from_pdbfile(input_pdb)

    # get sequence
    seq = get_sequence_from_pose(pose, sep)

    # get chains
    chains = pose.get_chains()

    # assemble and return
    return dict(zip(chains, seq))

def parse_cleavage_sites(cleavage_sites: list[str]) -> list[tuple[int,str]]:
    """Parse cleavage site positions from the cleavage_sites list, maintaining their labels."""
    return sorted([(int(site[:-1]), site) for site in cleavage_sites], key = lambda x: x[0])

def cleave_sequence_to_dict(seq: str, cleavage_sites: list[str]) -> dict:
    """Cleave the sequence at specified cleavage sites and return a dictionary with detailed keys."""
    fragments_dict = {}
    parsed_sites = parse_cleavage_sites(cleavage_sites)

    # Add fragments for each individual cleavage site
    for pos, label in parsed_sites:
        fragments_dict[f"nterm_{label}"] = seq[:pos]
        fragments_dict[f"{label}_cterm"] = seq[pos:]

    # Add fragments for combinations of cleavage sites
    for i in range(2, len(parsed_sites) + 1):
        for combo in combinations(parsed_sites, i):
            combo_labels = [label for _, label in combo]
            key = "_".join(combo_labels)
            combo_positions = [pos for pos, _ in combo]
            start_pos = combo_positions[0]
            end_pos = combo_positions[-1]
            fragments_dict[key] = seq[start_pos:end_pos]

    # Ensure unique fragments across all keys
    unique_fragments = set(fragments_dict.values())
    if len(unique_fragments) < len(fragments_dict):
        # Filter dictionary to remove duplicate fragments, keeping the shortest key for each fragment
        fragments_dict = {k: v for k, v in sorted(fragments_dict.items(), key=lambda item: (len(item[1]), len(item[0]))) if v in unique_fragments and unique_fragments.remove(v) is None}

    return fragments_dict

def main(args) -> None:
    '''Predicts cleavage products'''
    # setup jobstarter:
    if args.jobstarter == "slurm":
        jobstarter = jobstarters.SbatchArrayJobstarter(max_cores=args.max_cores, gpus=1)
    elif args.jobstarter == "local":
        jobstarter = jobstarters.LocalJobStarter(max_cores=args.max_cores)
    else:
        raise ValueError(f"Parameter 'jobstarter': {args.jobstarter} not allowed. Has to be one of {{slurm, local}}.")

    # setup input (if poses are .pdb, convert to sequence)
    prot = Poses(
        poses = args.input,
        work_dir = args.output_dir,
    )

    # excract chain-sequence:
    pose = prot.poses_list()[0]
    if pose.endswith(".fa"):
        seq_dict = read_fasta(pose, args.chain_separator)
    if pose.endswith(".pdb"):
        seq_dict = read_pdb(pose, args.chain_separator)
    else:
        raise ValueError(f"Unsupported file type for this protocol: {pose.rsplit('/', maxsplit=1)[-1]}. Only .pdb or .fa files are allowed.")

    # create cleavage variants ####!!!!! Only support for single chain so far.
    cleavage_sites = ResidueSelection(args.cleavage_sites).to_list(ordering="rosetta")
    cleaved_sequences_dict = cleave_sequence_to_dict(seq = seq_dict["A"], cleavage_sites = cleavage_sites)

    # create output dir for split fastas:
    if not os.path.isdir((fasta_dict := f"{prot.work_dir}/cleaved_fastas")):
        os.makedirs(fasta_dict, exist_ok=True)

    # write sequences to fastas and set as poses.
    fasta_list = []
    for cleavage, seq in cleaved_sequences_dict.items():
        desc = prot.df.iloc[0, :]["description"] + "_" + cleavage
        fn = os.path.join(fasta_dict, desc + ".fa")
        with open(fn, 'w', encoding="UTF-8") as f:
            f.write(f">{desc}\n{seq}")
        fasta_list.append(fn)

    # read in as new poses and predict.
    cleaved_seqs = Poses(
        poses = fasta_dict,
        work_dir = prot.work_dir,
        glob_suffix = "*.fa",
        jobstarter = jobstarter
    )

    # setup AF2
    af2 = Alphafold2(jobstarter = jobstarter)
    af2.run(
        poses = cleaved_seqs,
        prefix = "af2"
    )

    # postprocessing


if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required options
    argparser.add_argument("--input", type=str, required=True, help=".pdb or .fa file for which to predict the cleavage variants from.")
    argparser.add_argument("--cleavage_sites", type=str, required=True, help="Comma-separated list of cleavage sites. has to contain chain info as well. Example: --cleavage_sites=A13,A144,A123")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")

    # optional
    argparser.add_argument("--sequential", type=str, required=True, help="Path to output directory.")
    argparser.add_argument("--jobstarter", default="slurm", help="Options: {slurm, local}. Select JobStarter to run your script.")
    argparser.add_argument("--max_cores", type=int, default=10, help="Select default number of cores to run predictions on.")
    argparser.add_argument("--chain_separator", type=str, default=":", help="Specify how chains are separated in the fasta file.")

    arguments = argparser.parse_args()

    # check arguments (either input_json or input_pdb + reference_pdb)
    main(arguments)
