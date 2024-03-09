'''Module to handle Alphafold2 within ProtSLURM'''
# general imports
import re
import os
import logging
from glob import glob
import shutil

# dependencies
import pandas as pd
import numpy as np

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.tools
from protslurm.runners import Runner, RunnerOutput
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter

# TODO @Adrian: Please write AF2 run_singular() method that does not batch fastas together, but predicts each fasta individually. We need this to supply pose_options to af2 prediction runs (e.g. custom-templates that are unique for each pose.)
class Alphafold2(Runner):
    '''Class to run Alphafold2 and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.AF2_DIR_PATH, python_path:str=protslurm.config.AF2_PYTHON_PATH, jobstarter:str=None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under Alphafold2_SCRIPT_PATH.")
        if not python_path:
            raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under Alphafold2_PYTHON_PATH.")

        self.script_path = script_path
        self.python_path = python_path
        self.name = "alphafold2.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "alphafold2.py"

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter, options:str=None, overwrite:bool=False, num_batches:int=None, return_top_n_poses:int=1) -> RunnerOutput:
        '''Runs alphafold2.py on acluster'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for output-file in pdb-dir. If output is present and correct, then skip Alphafold2.
        scorefile = "Alphafold2_scores.json"
        scorefilepath = os.path.join(work_dir, scorefile)
        if not overwrite and os.path.isfile(scorefilepath):
            return RunnerOutput(poses=poses, results=pd.read_json(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()
        elif overwrite:
            if os.path.isdir(fasta_dir := os.path.join(work_dir, "input_fastas")):
                shutil.rmtree(fasta_dir)
            if os.path.isdir(af2_preds_dir := os.path.join(work_dir, "af2_preds")):
                shutil.rmtree(af2_preds_dir)
            if os.path.isdir(af2_pdb_dir := os.path.join(work_dir, "output_pdbs")):
                shutil.rmtree(af2_pdb_dir)

        # setup af2-specific directories:
        os.makedirs(fasta_dir := os.path.join(work_dir, "input_fastas"), exist_ok=True)
        os.makedirs(af2_preds_dir := os.path.join(work_dir, "af2_preds"), exist_ok=True)
        os.makedirs(af2_pdb_dir := os.path.join(work_dir, "output_pdbs"), exist_ok=True)

        # setup input-fastas in batches (to speed up prediction times.)
        num_batches = num_batches or jobstarter.max_cores
        pose_fastas = self.prep_fastas_for_prediction(poses=poses.df['poses'].to_list(), fasta_dir=fasta_dir, max_filenum=num_batches)

        # write Alphafold2 cmds:
        cmds = [self.write_cmd(pose, output_dir=af2_preds_dir, options=options) for pose in pose_fastas]

        # run
        logging.info(f"Starting AF2 predictions of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="alphafold2",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        logging.info(f"Predictions finished, starting to collect scores.")
        scores = self.collect_scores(work_dir=work_dir, scorefile=scorefilepath, num_return_poses=return_top_n_poses)
        scores.to_json(scorefilepath)

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()


    def prep_fastas_for_prediction(self, poses:list[str], fasta_dir:str, max_filenum:int) -> list[str]:
        '''
        Args:
            <poses>             List of paths to *.fa files
            <fasta_dir>         Directory to which the new fastas should be written into
            <max_filenum>          Maximum number of *.fa files that should be written
        '''
        def mergefastas(files:list, path:str, replace=None) -> str:
            '''
            Merges Fastas located in <files> into one single fasta-file called <path>
            '''
            fastas = list()
            for fp in files:
                with open(fp, 'r', encoding="UTF-8") as f:
                    fastas.append(f.read().strip())

            if replace:
                fastas = [x.replace(replace[0], replace[1]) for x in fastas]

            with open(path, 'w', encoding="UTF-8") as f:
                f.write("\n".join(fastas))

            return path

        # determine how to split the poses into <max_gpus> fasta files:
        splitnum = len(poses) if len(poses) < max_filenum else max_filenum
        poses_split = [list(x) for x in np.array_split(poses, int(splitnum))]

        # Write fasta files according to the fasta_split determined above and then return:
        return [mergefastas(files=poses, path=f"{fasta_dir}/fasta_{str(i+1).zfill(4)}.fa", replace=("/",":")) for i, poses in enumerate(poses_split)]


    def write_cmd(self, pose_path:str, output_dir:str, options:str):
        '''Writes Command to run Alphafold2.py'''

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, "")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {self.script_path}/colabfold_batch {opts} {flags} {pose_path} {output_dir} "

    def collect_scores(self, work_dir:str, scorefile:str, num_return_poses:int=1) -> pd.DataFrame:
        '''collects scores from Alphafold2 output'''

        def get_json_files_of_description(description:str, input_dir:str) -> str:
            return sorted([filepath for filepath in glob(f"{input_dir}/{description}*rank*.json") if re.search(f"{description}_scores_rank_..._.*_model_._seed_...\.json", filepath)]) # pylint: disable=W1401

        def get_pdb_files_of_description(description:str, input_dir:str) -> str:
            return sorted([filepath for filepath in glob(f"{input_dir}/{description}*rank*.pdb") if re.search(f"{description}_.?.?relaxed_rank_..._.*_model_._seed_...\.pdb", filepath)]) # pylint: disable=W1401

        def get_json_pdb_tuples_from_description(description:str, input_dir:str) -> "list[tuple[str,str]]":
            '''Collects af2-output scores.json and .pdb file for a given 'description' as corresponding tuples (by sorting).'''
            return [(jsonf, pdbf) for jsonf, pdbf in zip(get_json_files_of_description(description, input_dir), get_pdb_files_of_description(description, dir))]

        def calc_statistics_over_af2_models(index, input_tuple_list:"list[tuple[str,str]]") -> list:
            '''
            takes list of .json files from af2_predictions and collects scores (mean_plddt, max_plddt, etc.)
            '''
            df = pd.concat([summarize_af2_json(af2_tuple[0], af2_tuple[1]) for af2_tuple in input_tuple_list], ignore_index=True)
            df = df.sort_values("json_file").reset_index(drop=True)
            df["rank"] = [i for i in range(1, len(df.index) + 1)]
            df["description"] = [f"{index}_{str(i).zfill(4)}" for i in range(1, len(df.index) + 1)]
            df.to_csv('df.csv')
            for col in ['plddt', 'max_pae', 'ptm']:
                df[f"mean_{col}"] = df[col].mean()
                df[f"std_{col}"] = df[col].std()
                df[f"top_{col}"] = df[col].max()
            return df

        def summarize_af2_json(json_path:str, input_pdb:str) -> pd.DataFrame:
            '''
            Takes raw AF2_scores.json file and calculates mean pLDDT over the entire structure, also puts perresidue pLDDTs and paes in list.
            
            Returns pd.DataFrame
            '''
            df = pd.read_json(json_path)
            means = df.mean(numeric_only=True).to_frame().T # pylint: disable=E1101
            means["plddt_list"] = [df["plddt"]]
            means["pae_list"] = [df["pae"]]
            means["json_file"] = json_path
            means["pdb_file"] = input_pdb
            return means

        # create pdb_dir
        pdb_dir = os.path.join(work_dir, "output_pdbs")
        preds_dir = os.path.join(work_dir, "af2_preds")

        # collect all unique 'descriptions' leading to predictions
        descriptions = [x.split("/")[-1].replace(".done.txt", "") for x in glob(f"{preds_dir}/*.done.txt")]
        if not descriptions:
            raise FileNotFoundError(f"ERROR: No AF2 prediction output found at {preds_dir} Are you sure it was the correct path?")

        # Collect all .json and corresponding .pdb files of each 'description' into a dictionary. (This collects scores from all 5 models)
        scores_dict = {description: get_json_pdb_tuples_from_description(description, preds_dir) for description in descriptions}
        if not scores_dict:
            raise FileNotFoundError("No .json files were matched to the AF2 output regex. Check AF2 run logs. Either AF2 crashed or the AF2 regex is outdated (check at function 'collect_af2_scores()'")

        # Calculate statistics over prediction scores for each of the five models.
        scores_df = pd.concat([calc_statistics_over_af2_models(description, af2_output_tuple_list) for description, af2_output_tuple_list in scores_dict.items()]).reset_index(drop=True)

        # Return only top n poses
        scores_df = scores_df[scores_df['rank'] <= num_return_poses].reset_index(drop=True)

        # Copy poses to pdb_dir and store location in DataFrame
        scores_df.loc[:, "location"] = [shutil.copy(row['pdb_file'], os.path.join(pdb_dir, f"{row['description']}.pdb")) for _, row in scores_df.iterrows()]
        scores_df.drop(['pdb_file', 'json_file'], axis=1, inplace=True)

        # Write output df
        scores_df.to_json(scorefile)

        return scores_df
    