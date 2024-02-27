'''Module to handle ESMFold within ProtSLURM'''
# general imports
import os
import logging
from glob import glob
import shutil
import numpy as np

# dependencies
import pandas as pd

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.runners
from .runners import Runner
from .runners import RunnerOutput


#TODO: write script that only requires path to esmfold dir, not the path to Markus' esmfold_inference.py


class ESMFold(Runner):
    '''Class to run ESMFold and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.ESMFOLD_SCRIPT_PATH, python_path:str=protslurm.config.ESMFOLD_PYTHON_PATH, jobstarter_options:str=None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under ESMFOLD_SCRIPT_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under ESMFOLD_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "esmfold.py"
        self.index_layers = 0
        self.jobstarter_options = jobstarter_options

    def __str__(self):
        return "esmfold.py"

    def run(self, poses:protslurm.poses.Poses, output_dir:str, prefix:str, options:str=None, overwrite:bool=False, num_batches:int=10, jobstarter:protslurm.jobstarters.JobStarter=None) -> RunnerOutput:
        '''Runs ESMFold.py on acluster'''

        # setup output_dir
        work_dir = os.path.abspath(output_dir)
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip ESMFold.
        scorefile = "ESMFold_scores.json"
        scorefilepath = os.path.join(work_dir, scorefile)
        if overwrite == False and os.path.isfile(scorefilepath):
            return RunnerOutput(poses=poses, results=pd.read_json(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()
    
        os.makedirs((fasta_dir := f"{work_dir}/input_fastas"), exist_ok=True)
        os.makedirs((esm_preds_dir := f"{work_dir}/esm_preds"), exist_ok=True)

        pose_fastas = self.prep_fastas_for_prediction(poses=poses.df['poses'].to_list(), fasta_dir=fasta_dir, max_filenum=num_batches or self.num_batches)


        # write ESMFold cmds:
        cmds = [self.write_cmd(pose, output_dir=esm_preds_dir, options=options) for pose in pose_fastas]

        # run
        jobstarter = jobstarter or poses.default_jobstarter
        jobstarter_options = self.jobstarter_options or f"--gpus-per-node 1 -c1 -e {work_dir}/ESMFold_err.log -o {work_dir}/ESMFold_out.log"
        jobstarter.start(cmds=cmds,
                         options=jobstarter_options,
                         jobname="ESMFold",
                         wait=True,
                         output_path=f"{work_dir}/"
        )

        # collect scores
        scores = self.collect_scores(work_dir=work_dir, scorefile=scorefilepath)
        
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def prep_fastas_for_prediction(self, poses: list[str], fasta_dir: str, max_filenum: int) -> list[str]:
        '''
        Args:
            <poses>             List of paths to *.fa files
            <fasta_dir>         Directory to which the new fastas should be written into
            <max_filenum>          Maximum number of *.fa files that should be written
        '''
        def mergefastas(files: list, path: str, replace=None) -> str:
            '''
            Merges Fastas located in <files> into one single fasta-file called <path>
            '''
            fastas = list()
            for fp in files:
                with open(fp, 'r', encoding="UTF-8") as f:
                    fastas.append(f.read().strip())

            if replace: fastas = [x.replace(replace[0], replace[1]) for x in fastas]

            with open(path, 'w', encoding="UTF-8") as f:
                f.write("\n".join(fastas))

            return path

        # determine how to split the poses into <max_gpus> fasta files:
        splitnum = len(poses) if len(poses) < max_filenum else max_filenum
        poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]

        # Write fasta files according to the fasta_split determined above and then return:
        return [mergefastas(files=poses, path=f"{fasta_dir}/fasta_{str(i+1).zfill(4)}.fa", replace=("/",":")) for i, poses in enumerate(poses_split)]


    def write_cmd(self, pose_path:str, output_dir:str, options:str):
        '''Writes Command to run ESMFold.py'''

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, "")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/esmfold_inference.py --fasta {pose_path} --output_dir {output_dir} {opts} {flags}"

    def collect_scores(self, work_dir:str, scorefile:str) -> pd.DataFrame:
        '''collects scores from ESMFold output'''
    
        # collect all .json files
        pdb_dir = os.path.join(work_dir, "esm_preds")
        fl = glob(f"{pdb_dir}/fasta_*/*.json")
        pl = glob(f"{pdb_dir}/fasta_*/*.pdb")

        output_dir = os.path.join(work_dir, 'output_pdbs')
        os.makedirs(output_dir, exist_ok=True)
        pl = [shutil.copy(pdb, output_dir) for pdb in pl]
        # create dataframe containing new locations
        df_pdb = pd.DataFrame({'location': pl, 'description': [os.path.splitext(os.path.basename(pdb))[0] for pdb in pl]})

        # read the files, add origin column, and concatenate into single DataFrame:
        df = pd.concat([pd.read_json(f) for f in fl]).reset_index(drop=True)
        # merge with df containing locations
        df = df.merge(df_pdb, on='description')
        df.to_json(scorefile)
        shutil.rmtree(pdb_dir)

        return df
