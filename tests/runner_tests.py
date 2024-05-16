'''Script to test various runners
'''
import logging
import os
import shutil
from protslurm.jobstarters import SbatchArrayJobstarter
from protslurm.jobstarters import LocalJobStarter

import protslurm.config
# customs
from protslurm.poses import Poses
#from protslurm.runners.protein_generator import ProteinGenerator
from protslurm.tools.ligandmpnn import LigandMPNN
from protslurm.tools.rosetta import Rosetta
from protslurm.tools.rfdiffusion import RFdiffusion
from protslurm.tools.attnpacker import AttnPacker
from protslurm.tools.esmfold import ESMFold
from protslurm.tools.metrics.protparam import ProtParam
from protslurm.tools.metrics.tmscore import TMalign
from protslurm.tools.metrics.tmscore import TMscore
from protslurm.utils.plotting import sequence_logo



#TODO: Print Test output: Which Runners are Implemented, Which runners succeeded (all if the test runs through).
#TODO: @Adrian: For Attnpacker, ligandmpnn and AF please write Tutorials as Jupyter Notebooks in 'examples' Folder.
#TODO: @Adrian: Please write tests for LigandMPNN running +10 sequences with/ and without pose_options (to test non_batch and batch run)

def main(args):

    '''.'''
    js_dict = {
        "slurm_gpu_jobstarter": SbatchArrayJobstarter(max_cores=10, gpus=1, options="-c1"),
        "local_jobstarter": LocalJobStarter()
    }

    runner_dict = {
        "ESMFold": "not tested",
        "AttnPacker": "not tested",
        "TMalign": "not tested",
        "TMscore": "not tested",
    }


    if args.jobstarter:
        jobstarter = js_dict[args.jobstarter]
    else:
        jobstarter = None


    
####################### ESMFold #######################
    
    if os.path.isfile(protslurm.config.ESMFOLD_SCRIPT_PATH):
        logging.info("Running test for ESMFold...")
        out_dir = "output_esmfold"
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        proteins = Poses(
            poses="input_files/esmfold/",
            glob_suffix="*.fasta",
            work_dir=out_dir,
            storage_format="json"
        )

        esm_runner = ESMFold()
        proteins = esm_runner.run(poses=proteins, prefix="test", overwrite=True, jobstarter=jobstarter)

        logging.info("ESMFold passed!")

####################### Sequence Logo #######################
    

    logging.info("Running test for sequence logo generation...")
    out_dir = "output_seqlogo"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/esmfold/",
        glob_suffix="*.fasta",
        work_dir=out_dir,
        storage_format="json"
    )

    sequence_logo(dataframe=proteins.df, input_col="poses", save_path=os.path.join(out_dir, "seq.logo"), refseq="AS", title=None, resnums=[1,2], units="probability")

    logging.info("ESMFold passed!")

####################### AttnPacker #######################

    if os.path.isdir(protslurm.config.ATTNPACKER_DIR_PATH):
        logging.info("Running test for AttnPacker...")
        out_dir = "output_attnpacker"
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        proteins = Poses(
            poses="input_files/attnpacker/",
            glob_suffix="*.pdb",
            work_dir=out_dir,
            storage_format="json"
        )

        proteins = AttnPacker().run(poses=proteins, prefix="test", overwrite=False, jobstarter=jobstarter)

        logging.info("AttnPacker passed!")

####################### TMalign #######################

    logging.info("Running test for TMalign...")
    out_dir = "output_TMalign"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="csv",
        jobstarter=LocalJobStarter()
    )

    proteins = TMalign().run(poses=proteins, prefix="test", ref_col="poses", overwrite=True, jobstarter=jobstarter)


    logging.info("TMalign passed!")


####################### TMscore #######################

    logging.info("Running test for TMscore...")
    out_dir = "output_TMscore"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="csv",
        jobstarter=LocalJobStarter()
    )

    proteins = TMscore().run(poses=proteins, prefix="test", ref_col="poses", overwrite=True, jobstarter=jobstarter)


    logging.info("TMalign passed!")


####################### ProtParam #######################

    logging.info("Running test for ProtParam...")
    out_dir = "output_protparam"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="csv",
        jobstarter=LocalJobStarter()
    )

    proteins = ProtParam().run(poses=proteins, prefix="test", pH=6, overwrite=True, jobstarter=jobstarter)


    logging.info("ProtParam passed!")

####################### ROSETTA #######################

    if os.path.isdir(protslurm.config.ROSETTA_BIN_PATH):
        logging.info("Running test for Rosetta...")
        out_dir = "output_rosetta"
        #if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        proteins = Poses(
            poses="input_files/rosettascripts/",
            glob_suffix="*.pdb",
            work_dir=out_dir,
            storage_format="json"
        )

        options = "-parser:protocol input_files/rosettascripts/empty.xml -beta"
        proteins = Rosetta().run(poses=proteins, prefix="test", rosetta_application="rosetta_scripts.linuxgccrelease", nstruct=2, options=options, overwrite=True, jobstarter=jobstarter)
        proteins.calculate_composite_score(name="comp_score", scoreterms=["test_total_score", "test_fa_dun_rot", "test_fa_elec"], weights=[-1, 1, -1], plot=True)
        proteins.filter_poses_by_rank(n=1, score_col="test_total_score", remove_layers=1, prefix="test_rank", plot=True)
        proteins.filter_poses_by_value(value=3000, score_col="test_total_score", operator="<=", prefix="test_value", plot=True)

        logging.info("Rosetta passed!")

####################### LIGANDMPNN #######################

    if os.path.isfile(protslurm.config.LIGANDMPNN_SCRIPT_PATH):
        logging.info("Running test for LigandMPNN...")
        out_dir = "output_ligandmpnn"
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        proteins = Poses(
            poses="input_files/ligandmpnn/",
            glob_suffix="*.pdb",
            work_dir=out_dir,
            storage_format="feather"
        )

        #set fixed residues
        proteins.df['fixed_residues'] = ['A3,B3,C3,D3']
        # start ligand_mpnn
        proteins = LigandMPNN().run(poses=proteins, prefix="test", model_type="ligand_mpnn", nseq=5, fixed_res_col='fixed_residues', jobstarter=jobstarter)

        logging.info("LigandMPNN passed!")


####################### RFDIFFUSION #######################


    if os.path.isfile(protslurm.config.RFDIFFUSION_SCRIPT_PATH):
        logging.info("Running test for RFdiffusion...")
        out_dir = "output_rfdiffusion"
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)

        proteins = Poses(
            poses="input_files/rfdiffusion/",
            glob_suffix="*.pdb",
            work_dir=out_dir,
            storage_format="json"
        )

        options = "diffuser.T=50 potentials.guide_scale=5 'contigmap.contigs=[Q1-21/0 20/A1-5/10-50/B1-5/10-50/C1-5/10-50/D1-5/20]' contigmap.length=200-200 'contigmap.inpaint_seq=[A1/A2/A4/A5/B1/B2/B4/B5/C1/C2/C4/C5/D1/D2/D4/D5]' potentials.substrate=LIG"
        proteins = RFdiffusion().run(poses=proteins, prefix="test", num_diffusions=1, options=options, overwrite=True, jobstarter=jobstarter)

        logging.info("RFdiffusion passed!")




if __name__ == "__main__":


    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="log.txt")
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--jobstarter", type=str, default=None, help="define jobstarter. can be either slurm_gpu_jobstarter or local_jobstarter, if None default slurm is used.")
    args = argparser.parse_args()
    #run main
    main(args)