'''Script to test various runners
'''
import logging

# customs
from protslurm.poses import Poses
from protslurm.runners.protein_generator import ProteinGenerator
from protslurm.runners.ligandmpnn import LigandMPNN
from protslurm.runners.rosettascripts import RosettaScripts


def main(args):
    '''.'''
    # instantiate Poses class and fill it with input_dir
    proteins = Poses(
        poses="input_pdbs/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=args.output_dir,
        storage_format="feather"
    )

    proteins = RosettaScripts().run(poses=proteins, output_dir='scripts', prefix="rosettatest", nstruct=1, xml_path="empty.xml", options="-beta", overwrite=True)
    print(proteins.df)
    proteins.df['fixed_residues'] = ['A3,B3,C3,D3']
    # start ligand_mpnn

    proteins = LigandMPNN().run(poses=proteins, output_dir='ligandmpnn', prefix="test", model="ligand_mpnn", nseq=5, fixed_res_column='fixed_residues')
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
