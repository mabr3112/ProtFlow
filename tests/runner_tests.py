'''Script to test various runners
'''
import logging

# customs
from protslurm.poses import Poses
#from protslurm.runners.protein_generator import ProteinGenerator
from protslurm.runners.ligandmpnn import LigandMPNN
from protslurm.runners.rosetta import Rosetta
from protslurm.runners.rfdiffusion import RFdiffusion
from protslurm.runners.attnpacker import AttnPacker
from protslurm.runners.esmfold import ESMFold

#TODO: Automatically execute tests only for runners that are set up in the config.py file!
#TODO: Print Test output: Which Runners are Implemented, Which runners succeeded (all if the test runs through).
#TODO: Write Tutorials as Jupyter Notebooks in 'examples' Folder!

def main():
    '''.'''
####################### ESMFold #######################

    out_dir = "output_esmfold"

    proteins = Poses(
        poses="input_files/esmfold/",
        glob_suffix="*.fasta",
        work_dir=out_dir,
        storage_format="json"
    )

    proteins = ESMFold().run(poses=proteins, output_dir=out_dir, prefix="test", overwrite=True)



####################### AttnPacker #######################

    out_dir = "output_attnpacker"

    proteins = Poses(
        poses="input_files/attnpacker/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="json"
    )

    proteins = AttnPacker().run(poses=proteins, output_dir=out_dir, prefix="test", overwrite=False)



####################### ROSETTA #######################

    out_dir = "output_rosetta"

    # instantiate Poses class and fill it with input_dir
    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="pickle"
    )
    options = "-parser:protocol input_files/rosettascripts/empty.xml -beta"
    proteins = Rosetta().run(poses=proteins, output_dir=out_dir, prefix="test", rosetta_application="rosetta_scripts.linuxgccrelease", nstruct=1, options=options, overwrite=True)


####################### LIGANDMPNN #######################

    out_dir = "output_ligandmpnn"

    proteins = Poses(
        poses="input_files/ligandmpnn/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="feather"
    )

    #set fixed residues
    proteins.df['fixed_residues'] = ['A3,B3,C3,D3']
    # start ligand_mpnn
    proteins = LigandMPNN().run(poses=proteins, output_dir=out_dir, prefix="test", model="ligand_mpnn", nseq=5, fixed_res_column='fixed_residues')



####################### RFDIFFUSION #######################

    out_dir = "output_rfdiffusion"

    proteins = Poses(
        poses="input_files/rfdiffusion/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="json"
    )

    options = "diffuser.T=50 potentials.guide_scale=5 'contigmap.contigs=[Q1-21/0 20/A1-5/10-50/B1-5/10-50/C1-5/10-50/D1-5/20]' contigmap.length=200-200 'contigmap.inpaint_seq=[A1/A2/A4/A5/B1/B2/B4/B5/C1/C2/C4/C5/D1/D2/D4/D5]' potentials.substrate=LIG"
    proteins = RFdiffusion().run(poses=proteins, output_dir=out_dir, prefix="test", num_diffusions=1, options=options, overwrite=True)





if __name__ == "__main__":


    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="log.txt")

    #run main
    main()
