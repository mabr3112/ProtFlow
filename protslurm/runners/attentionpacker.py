'''Module to handle AttnPacker within ProtSLURM'''
# general imports
import os
import logging
from glob import glob
import shutil

# dependencies
import pandas as pd
import Bio
from Bio import SeqIO

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.runners
from .runners import Runner
from .runners import RunnerOutput


class AttnPacker(Runner):
    '''Class to run AttnPacker and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.ATTNPACKER_SCRIPT_PATH, python_path:str=protslurm.config.ATTNPACKER_PYTHON_PATH, sbatch_options:str=None) -> None:
        '''sbatch_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under LIGANDMPNN_SCRIPT_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under LIGANDMPNN_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "ligandmpnn.py"
        self.sbatch_options = sbatch_options

    def __str__(self):
        return "ligandmpnn.py"

    def run(self, poses:list, jobstarter:protslurm.jobstarters.JobStarter, output_dir:str, options:str=None, pose_options:str=None) -> RunnerOutput:
        '''Runs ligandmpnn.py on acluster'''

        available_models = ["protein_mpnn", "ligand_mpnn", "soluble_mpnn", "global_label_membrane_mpnn", "per_residue_label_membrane_mpnn"]
        if not model in available_models: raise ValueError(f"{model} must be one of {available_models}!")

        # setup output_dir
        work_dir = os.path.abspath(output_dir)
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip LigandMPNN.
        scorefile = "ligandmpnn_scores.json"
        if os.path.isfile((scorefilepath := f"{work_dir}/{scorefile}")):
            return RunnerOutput(pd.read_json(scorefilepath))

        # parse_options and pose_options:
        pose_options = self.safecheck_pose_options(pose_options, poses)

        # write protein generator cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, model=model, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses, pose_options)]

        # run
        sbatch_options = self.sbatch_options or f"--gpus-per-node 1 -c1 -e {work_dir}/ligandmpnn_err.log -o {work_dir}/ligandmpnn_out.log"
        jobstarter.start(cmds=cmds,
                         options=sbatch_options,
                         jobname="ligandmpnn",
                         wait=True,
                         cmdfile_dir=f"{output_dir}/"
        )

        # collect scores
        return RunnerOutput(self.collect_scores(work_dir=work_dir, scorefile=scorefile, return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose, preserve_original_output=preserve_original_output))

    def safecheck_pose_options(self, pose_options: list, poses:list) -> list:
        '''Checks if pose_options are of the same length as poses, if now pose_options are provided, '''
        # safety check (pose_options must have the same length as poses)
        if isinstance(pose_options, list):
            if len(poses) != len(pose_options):
                raise ValueError(f"Arguments <poses> and <pose_options> for LigandMPNN must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
            return pose_options
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            return ["" for x in poses]
        raise TypeError(f"Unsupported type for pose_options: {type(pose_options)}. pose_options must be of type [list, None]")

    def write_cmd(self, pose_path:str, output_dir:str, model:str, options:str, pose_options:str):
        '''Writes Command to run ligandmpnn.py'''

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {self.script_path} --model_type {model} --out_folder {output_dir} --pdb_path {pose_path} {opts} {flags}"

    def collect_scores(self, work_dir: str, scorefile: str, return_seq_threaded_pdbs_as_pose:bool, preserve_original_output:bool=True) -> pd.DataFrame:
        '''collects scores from ligandmpnn output'''

        def mpnn_fastaparser(fasta_path):
            '''reads in ligandmpnn multi-sequence fasta, renames sequences and returns them'''
            records = list(Bio.SeqIO.parse(fasta_path, "fasta"))
            #maxlength = len(str(len(records)))
            
            # Set continuous numerating for the names of mpnn output sequences:
            name = records[0].name.replace(",", "")
            records[0].name = name
            for i, x in enumerate(records[1:]):
                setattr(x, "name", f"{name}_{str(i+1).zfill(4)}")
            
            return records
    
        def convert_ligandmpnn_seqs_to_dict(seqs):
            '''
            Takes already parsed list of fastas as input <seqs>. Fastas can be parsed with the function mpnn_fastaparser(file).
            Should be put into list.
            Converts mpnn fastas into a dictionary:
            {
                "col_1": [vals]
                    ...
                "col_n": [vals]
            }
            '''
            # Define cols and initiate them as empty lists:
            seqs_dict = dict()
            cols = ["mpnn_origin", "seed", "description", "sequence", "T", "id", "seq_rec", "overall_confidence", "ligand_confidence"]
            for col in cols:
                seqs_dict[col] = []

            # Read scores of each sequence in each file and append them to the corresponding columns:
            for seq in seqs:
                for f in seq[1:]:
                    seqs_dict["mpnn_origin"].append(seq[0].name)
                    seqs_dict["sequence"].append(str(f.seq))
                    seqs_dict["description"].append(f.name)
                    d = {k: float(v) for k, v in [x.split("=") for x in f.description.split(", ")[1:]]}
                    for k, v in d.items():
                        seqs_dict[k].append(v)
            return seqs_dict
        
        def write_mpnn_fastas(seqs_dict:dict) -> pd.DataFrame:
            seqs_dict["location"] = list()
            for d, s in zip(seqs_dict["description"], seqs_dict["sequence"]):
                seqs_dict["location"].append((fa_file := f"{seq_dir}/{d}.fa"))
                with open(fa_file, 'w') as f:
                    f.write(f">{d}\n{s}")
            return pd.DataFrame(seqs_dict)
        
        def rename_mpnn_pdb(pdb):
            '''changes single digit file extension to 4 digit file extension'''
            filename, extension = os.path.splitext(pdb)[0].rsplit('_', 1)
            filename = f"{filename}_{extension.zfill(4)}.pdb"
            shutil.move(pdb, filename)
            return


        # read .pdb files
        seq_dir = os.path.join(work_dir, 'seqs')
        pdb_dir = os.path.join(work_dir, 'backbones')
        fl = glob(seq_dir)
        pl = glob(pdb_dir)
        if not fl: raise FileNotFoundError(f"No .fa files were found in the output directory of LigandMPNN {seq_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")
        if not pl: raise FileNotFoundError(f"No .pdb files were found in the output directory of LigandMPNN {pdb_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")

        seqs = [mpnn_fastaparser(fasta) for fasta in fl]
        seqs_dict = convert_ligandmpnn_seqs_to_dict(seqs)

        original_seqs_dir = os.path.join(seq_dir, 'original_seqs')
        logging.info(f"Copying original .fa files into directory {original_seqs_dir}")
        os.makedirs(original_seqs_dir, exist_ok=True)
        _ = [shutil.move(fasta, os.path.join(original_seqs_dir, os.path.basename(fasta))) for fasta in fl]

        original_pdbs_dir = os.path.join(pdb_dir, 'original_backbones')
        logging.info(f"Copying original .pdb files into directory {original_pdbs_dir}")
        os.makedirs(original_pdbs_dir, exist_ok=True)
        _ = [shutil.copy(pdb, os.path.join(original_pdbs_dir, os.path.basename(pdb))) for pdb in pl]
        _ = [rename_mpnn_pdb(pdb) for pdb in pl]

        # Write new .fa files by iterating through "description" and "sequence" keys of the seqs_dict
        logging.info(f"Writing new fastafiles at original location {seq_dir}.")

        #write mpnn fasta files to output directory
        scores = write_mpnn_fastas(seqs_dict)

        if return_seq_threaded_pdbs_as_pose == True:
            #replace .fa with sequence threaded pdb files as poses
            scores['location'] = [os.path.join(pdb_dir, f"{os.path.splitext(os.path.basename(series['location']))[0]}.pdb") for _, series in scores.iterrows()]

        logging.info(f"Saving scores of {self} at {scorefile}")
        scores.to_json(scorefile)

        if preserve_original_output == False:
            if os.path.isdir(original_seqs_dir):
                logging.info(f"Deleting original .fa files at {original_seqs_dir}!")
                shutil.rmtree(original_seqs_dir)
            if os.path.isdir(original_pdbs_dir):
                logging.info(f"Deleting original .pdb files at {original_pdbs_dir}!")
                shutil.rmtree(original_pdbs_dir)

        return scores

