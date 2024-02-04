'''Script to test various runners
'''
import logging

# customs
from protslurm.poses import Poses
from protslurm.runners.protein_generator import ProteinGenerator
from protslurm.runners.ligandmpnn import LigandMPNN

def main(args):
    '''.'''
    # instantiate Poses class and fill it with input_dir
    proteins = Poses(
        poses=args.input_dir,
        glob_suffix="*.pdb",
        work_dir=args.output_dir,
        storage_format="feather"
    )

    # start ligand_mpnn
    proteins.run(
            #poses=proteins.df['poses'].to_list(),
            runner=LigandMPNN(),
            prefix="ligandmpnn",
    )

    print(proteins.df)

    # start protein_generator
    proteins.run(
        runner=ProteinGenerator(),
        prefix="protein_generator",
        options="--seq XXXXXXXXXXXMYXXXSEQVENCEXXXXXXXXXXXXXXX"
    )

if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .pdb files")
    argparser.add_argument("--output_dir", type=str, required=True, help="path to output directory")
    arguments = argparser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{arguments.output_dir}/log.txt")

    #run main
    main(arguments)
