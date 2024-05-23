'''
Script to write a .pml pymol script that visualizes mutations given two input pdb structures - one reference (wild-type) and one variant. 
'''

# Imports
import os
import logging
import argparse

# dependencies
import protflow.utils.biopython_tools as bio_tools
from protflow.utils import metrics
from protflow.utils import pymol_tools

def str_to_bool(input_str: str) -> bool:
    '''turns str to bool'''
    if input_str in ["True", "true", "1", 1]:
        return True
    elif input_str in ["False", "false", "0", 0]:
        return False
    else:
        raise ValueError(f"Argument {input_str} can not be interpreted as boolean. Try 'True' or 'False'.")

def main(args):
    '''main function. How to document?'''
    # sanity
    use_absolute_path = str_to_bool(args.use_pdb_absolute_path)

    # load structures
    logging.info(f"Loading reference and variant structures from:\n{args.reference_pdb}\n{args.variant_pdb}")
    wt = bio_tools.load_structure_from_pdbfile(args.reference_pdb)
    var = bio_tools.load_structure_from_pdbfile(args.variant_pdb)

    # get sequences
    wt_seq = bio_tools.get_sequence_from_pose(wt)
    var_seq = bio_tools.get_sequence_from_pose(var)

    # get mutation indeces
    mutations = metrics.get_mutation_indeces(wt_seq, var_seq)

    # configure output path and variant/reference pdb paths:
    out_path = args.output_path or f"{args.variant_pdb.rsplit('/', maxsplit=1)[-1].replace('.pdb','')}_mutations.pml"
    reference_pdb = os.path.abspath(args.reference_pdb) if use_absolute_path else args.reference_pdb.rsplit("/", maxsplit=1)[-1]
    variant_pdb = os.path.abspath(args.variant_pdb) if use_absolute_path else args.variant_pdb.rsplit("/", maxsplit=1)[-1]

    # write pymol script that colors mutations
    pymol_tools.mutations_pymol_scriptwriter(out_path, reference_pdb, variant_pdb, mutations)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required
    argparser.add_argument("--reference_pdb", type=str, required=True, help="Input directory containing .pdb files")
    argparser.add_argument("--variant_pdb", type=str, required=True, help="path to output directory")
    argparser.add_argument("--output_path", type=str, default=None, help="Path to your output file. Defaults to: variant_pdb_mutations.pml \t Example: --output_path=outputs/variant_0002_mutations.pml")

    # optional
    argparser.add_argument("--use_pdb_absolute_path", type=str, default=False, help="Should the pymol script be written for absolute paths, or for relative paths?")
    arguments = argparser.parse_args()

    #TODO: setup logging!

    main(arguments)
