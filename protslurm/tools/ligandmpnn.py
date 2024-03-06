'''Module to handle LigandMPNN within ProtSLURM'''
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
import protslurm.tools
from protslurm.runners import Runner
from protslurm.runners import RunnerOutput


class LigandMPNN(Runner):
    '''Class to run LigandMPNN and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.LIGANDMPNN_SCRIPT_PATH, python_path:str=protslurm.config.LIGANDMPNN_PYTHON_PATH, jobstarter_options:str=None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under LIGANDMPNN_SCRIPT_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under LIGANDMPNN_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "ligandmpnn.py"
        self.index_layers = 1
        self.jobstarter_options = jobstarter_options

    def __str__(self):
        return "ligandmpnn.py"

    def run(self, poses:protslurm.poses.Poses, output_dir:str, prefix:str, nseq:int=1, model:str="ligand_mpnn", options:str=None, pose_options:list or str=None, fixed_res_column:str=None, design_res_column:str=None, return_seq_threaded_pdbs_as_pose:bool=False, preserve_original_output:bool=True, overwrite:bool=False, jobstarter:protslurm.jobstarters.JobStarter=None) -> RunnerOutput:
        '''Runs ligandmpnn.py on acluster'''
        #TODO: reorder .run() arguments according to Runner.run() abstract mehtod in base class.

        available_models = ["protein_mpnn", "ligand_mpnn", "soluble_mpnn", "global_label_membrane_mpnn", "per_residue_label_membrane_mpnn"]
        if not model in available_models: raise ValueError(f"{model} must be one of {available_models}!")

        # setup output_dir
        work_dir = os.path.abspath(output_dir)
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip LigandMPNN.
        scorefile = "ligandmpnn_scores.json"
        scorefilepath = os.path.join(work_dir, scorefile)
        if overwrite == False and os.path.isfile(scorefilepath):
            return RunnerOutput(poses=poses, results=pd.read_json(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()

        # parse_options and pose_options:
        pose_options = self.create_pose_options(poses.df, pose_options, nseq, fixed_res_column, design_res_column)

        # write ligandmpnn cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, model=model, nseq=nseq, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df['poses'].to_list(), pose_options)]

        # run
        jobstarter = jobstarter or poses.default_jobstarter
        jobstarter_options = self.jobstarter_options or f"--gpus-per-node 1 -c1 -e {work_dir}/ligandmpnn_err.log -o {work_dir}/ligandmpnn_out.log"
        jobstarter.start(cmds=cmds,
                         options=jobstarter_options,
                         jobname="ligandmpnn",
                         wait=True,
                         output_path=f"{output_dir}/"
        )

        # collect scores
        scores = self.collect_scores(work_dir=work_dir, scorefile=scorefilepath, return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose, preserve_original_output=preserve_original_output)
        
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def create_pose_options(self, df:pd.DataFrame, pose_options:list or str=None, nseq:int=1, fixed_res_column:str=None, design_res_column:str=None) -> list:
        '''Checks if pose_options are of the same length as poses, if pose_options are provided, '''

        def check_if_column_in_poses_df(df:pd.DataFrame, column:str):
            if not column in [col for col in df.columns]: raise ValueError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")
            return
        def parse_residues(df:pd.DataFrame, column:str) -> list:
            check_if_column_in_poses_df(df, column)
            return [' '.join([res for res in series[column].split(',')]) for _, series in df.iterrows()]

        poses = df['poses'].to_list()


        if isinstance(pose_options, str):
            check_if_column_in_poses_df(df, pose_options)
            pose_options = df[pose_options].to_list()
        # safety check (pose_options must have the same length as poses)
        if fixed_res_column and design_res_column:
            raise ValueError(f"Cannot define both <fixed_res_column> and <design_res_column>!")
        if fixed_res_column:
            fixed_residues = parse_residues(df, fixed_res_column)
            if pose_options: pose_options = [" ".join([pose_opt, f"fixed_residues '{res}'"]) for pose_opt, res in zip(pose_options, fixed_residues)]
            else: pose_options = [f"fixed_residues='{res}'" for res in fixed_residues]
        if design_res_column:
            design_residues = parse_residues(df, design_res_column)
            if pose_options: pose_options = [' '.join([pose_opt, f"redesigned_residues '{res}'"]) for pose_opt, res in zip(pose_options, design_residues)]
            else: pose_options = [f"redesigned_residues='{res}'" for res in fixed_residues]
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            pose_options = ["" for x in poses]

        if len(poses) != len(pose_options):
            raise ValueError(f"Arguments <poses> and <pose_options> for LigandMPNN must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
        return pose_options

    def write_cmd(self, pose_path:str, output_dir:str, model:str, nseq:int, options:str, pose_options:str):
        '''Writes Command to run ligandmpnn.py'''

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {self.script_path} --model_type {model} --number_of_batches={nseq} --out_folder {output_dir} --pdb_path {pose_path} {opts} {flags}"


    def collect_scores(self, work_dir:str, scorefile:str, return_seq_threaded_pdbs_as_pose:bool, preserve_original_output:bool=True) -> pd.DataFrame:
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
        fl = glob(f"{seq_dir}/*.fa")
        pl = glob(f"{pdb_dir}/*.pdb")
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
