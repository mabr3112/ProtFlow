'''Module to handle the Alphafold2 implementation in Colabfold within ProtFlow'''
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
import protflow.config
import protflow.jobstarters
import protflow.tools
from protflow.runners import Runner, RunnerOutput
from protflow.poses import Poses
from protflow.jobstarters import JobStarter

class Colabfold(Runner):
    '''Class to run Alphafold2 within Colabfold and collect its outputs into a DataFrame'''
    def __init__(self, script_path: str = protflow.config.COLABFOLD_SCRIPT_PATH, jobstarter: str = None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under COLABFOLD_DIR_PATH.")

        self.script_path = script_path
        self.name = "colabfold.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "colabfold.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str = None, pose_options: str = None, overwrite: bool = False, return_top_n_poses: int = 1) -> RunnerOutput:
        '''Runs colabfold.py on acluster'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"colabfold_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        if overwrite:
            if os.path.isdir(fasta_dir := os.path.join(work_dir, "input_fastas")): shutil.rmtree(fasta_dir)
            if os.path.isdir(af2_preds_dir := os.path.join(work_dir, "af2_preds")): shutil.rmtree(af2_preds_dir)
            if os.path.isdir(af2_pdb_dir := os.path.join(work_dir, "output_pdbs")): shutil.rmtree(af2_pdb_dir)

        # setup af2-specific directories:
        os.makedirs(fasta_dir := os.path.join(work_dir, "input_fastas"), exist_ok=True)
        os.makedirs(af2_preds_dir := os.path.join(work_dir, "af2_preds"), exist_ok=True)
        os.makedirs(af2_pdb_dir := os.path.join(work_dir, "output_pdbs"), exist_ok=True)

        # setup input-fastas in batches (to speed up prediction times.), but only if no pose_options are provided!
        num_batches = len(poses.df.index) if pose_options else jobstarter.max_cores
        pose_fastas = self.prep_fastas_for_prediction(poses=poses.df['poses'].to_list(), fasta_dir=fasta_dir, max_filenum=num_batches)

        # prepare pose options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # write colabfold cmds:
        cmds = []
        for pose, pose_opt in zip(pose_fastas, pose_options):
            cmds.append(self.write_cmd(pose, output_dir=af2_preds_dir, options=options, pose_options=pose_opt))

        # run
        logging.info(f"Starting AF2 predictions of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="colabfold",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        logging.info(f"Predictions finished, starting to collect scores.")
        scores = self.collect_scores(work_dir=work_dir, num_return_poses=return_top_n_poses)
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        
        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()


    def prep_fastas_for_prediction(self, poses: list[str], fasta_dir: str, max_filenum: int) -> list[str]:
        '''
        Args:
            <poses>             List of paths to *.fa files
            <fasta_dir>         Directory to which the new fastas should be written into
            <max_filenum>          Maximum number of *.fa files that should be written
        '''
        def mergefastas(files: list, path: str, replace: bool = None) -> str:
            '''
            Merges Fastas located in <files> into one single fasta-file called <path>
            '''
            fastas = []
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


    def write_cmd(self, pose_path: str, output_dir: str, options: str = None, pose_options: str = None):
        '''Writes Command to run colabfold.py'''

        # parse options
        opts, flags = protflow.runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""

        return f"{self.script_path} {opts} {flags} {pose_path} {output_dir} "

    def collect_scores(self, work_dir: str, num_return_poses: int =1 ) -> pd.DataFrame:
        '''collects scores from colabfold output'''

        def get_json_files_of_description(description: str, input_dir: str) -> str:
            return sorted([filepath for filepath in glob(f"{input_dir}/{description}*rank*.json") if re.search(f"{description}_scores_rank_..._.*_model_._seed_...\.json", filepath)]) # pylint: disable=W1401

        def get_pdb_files_of_description(description: str, input_dir: str) -> str:
            return sorted([filepath for filepath in glob(f"{input_dir}/{description}*rank*.pdb") if re.search(f"{description}_.?.?relaxed_rank_..._.*_model_._seed_...\.pdb", filepath)]) # pylint: disable=W1401

        def get_json_pdb_tuples_from_description(description: str, input_dir: str) -> list[tuple[str,str]]:
            '''Collects af2-output scores.json and .pdb file for a given 'description' as corresponding tuples (by sorting).'''
            return list(zip(get_json_files_of_description(description, input_dir), get_pdb_files_of_description(description, input_dir)))

        def calc_statistics_over_af2_models(index: str, input_tuple_list: list[tuple[str,str]]) -> pd.DataFrame:
            '''
            index: "description" (name) of the pose.
            takes list of .json files from af2_predictions and collects scores (mean_plddt, max_plddt, etc.)
            '''
            # no statistics to calculate if only one model was used:
            print(input_tuple_list)
            print(len(input_tuple_list))
            if len(input_tuple_list) == 1:
                json_path, input_pdb = input_tuple_list[0]
                df = summarize_af2_json(json_path, input_pdb)
                df["description"] = [f"{index}_{str(i).zfill(4)}" for i in range(1, len(df.index) + 1)]
                df["rank"] = [1]
                return df

            # otherwise collect scores from individual .json files of models for each input fasta into one DF
            df = pd.concat([summarize_af2_json(json_path, input_pdb) for (json_path, input_pdb) in input_tuple_list], ignore_index=True)
            df = df.sort_values("json_file").reset_index(drop=True)

            # assign rank (for tracking) and extract 'description'
            df["rank"] = list(range(1, len(df.index) + 1))
            df["description"] = [f"{index}_{str(i).zfill(4)}" for i in range(1, len(df.index) + 1)]

            # calculate statistics
            for col in ['plddt', 'max_pae', 'ptm']:
                df[f"mean_{col}"] = df[col].mean()
                df[f"std_{col}"] = df[col].std()
                df[f"top_{col}"] = df[col].max()
            return df

        def summarize_af2_json(json_path: str, input_pdb: str) -> pd.DataFrame:
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

        return scores_df
    
