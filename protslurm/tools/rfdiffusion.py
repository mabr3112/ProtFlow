'''Module to handle RFdiffusion within ProtSLURM'''
# general imports
import os
import logging
from glob import glob
import re
from typing import Any
import numpy as np

# dependencies
import pandas as pd

# custom
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter
import protslurm.config
from protslurm.residues import ResidueSelection
from protslurm.runners import Runner, col_in_df
from protslurm.runners import RunnerOutput


class RFdiffusion(Runner):
    '''Class to run RFdiffusion and collect its outputs into a DataFrame'''
    def __init__(self, script_path: str = protslurm.config.RFDIFFUSION_SCRIPT_PATH, python_path: str = protslurm.config.RFDIFFUSION_PYTHON_PATH, jobstarter: None = JobStarter, jobstarter_options: str = None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under RFDIFFUSION_SCRIPT_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under RFDIFFUSION_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "rfdiffusion.py"
        self.index_layers = 1
        self.jobstarter_options = jobstarter_options
        self.jobstarter = jobstarter

    def __str__(self):
        return "rfdiffusion.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, num_diffusions: int = 1, options: str = None, pose_options: list[str] = None, overwrite: bool = False, multiplex_poses: int = None, update_motifs: list[str] = None) -> RunnerOutput:
        '''running function for RFDiffusion given poses and a jobstarter object.'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # setup runner-specific directories
        pdb_dir = os.path.join(work_dir, "output_pdbs")
        if not os.path.isdir(pdb_dir):
            os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip diffusion step.
        scorefile="rfdiffusion_scores.json"
        scorefilepath = os.path.join(work_dir, scorefile)
        if overwrite is False and os.path.isfile(scorefilepath):
            return RunnerOutput(poses=poses, results=pd.read_json(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()

        # in case overwrite is set, overwrite previous results.
        elif overwrite is True or not os.path.isfile(scorefilepath):
            if os.path.isfile(scorefilepath): os.remove(scorefilepath)
            for pdb in glob(f"{pdb_dir}/*pdb"):
                if os.path.isfile(trb := pdb.replace(".pdb", ".trb")):
                    os.remove(trb)
                    os.remove(pdb)

        # parse options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # handling of empty poses DataFrame.
        if len(poses) == 0 and pose_options:
            # if no poses are set, but pose_options are provided, create as many jobs as pose_options. output_pdbs must be specified in pose options!
            cmds = [self.write_cmd(pose=None, options=options, pose_opts=pose_option, output_dir=pdb_dir, num_diffusions=num_diffusions) for pose_option in pose_options]
        elif len(poses) == 0 and not pose_options:
            # if neither poses nor pose_options exist: write n=max_cores commands with generic output name.
            cmds = [self.write_cmd(pose=None, options=options, pose_opts="inference.output_prefix=" + os.path.join(pdb_dir, f"diff_{str(i+1).zfill(4)}"), output_dir=pdb_dir, num_diffusions=num_diffusions) for i in range(jobstarter.max_cores)]
        elif multiplex_poses:
            # create multiple copies (specified by multiplex variable) of poses to fully utilize parallel computing:
            poses.duplicate_poses(f"{poses.work_dir}/{prefix}_input_pdbs/", jobstarter.max_cores)
            self.index_layers += 1
            cmds = [self.write_cmd(pose, options, pose_opts, output_dir=pdb_dir, num_diffusions=num_diffusions) for pose, pose_opts in zip(poses.poses_list(), pose_options)]
        else:
            # write rfdiffusion cmds
            cmds = [self.write_cmd(pose, options, pose_opts, output_dir=pdb_dir, num_diffusions=num_diffusions) for pose, pose_opts in zip(poses.poses_list(), pose_options)]

        # diffuse
        jobstarter.start(
            cmds=cmds,
            jobname="rfdiffusion",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect RFdiffusion outputs
        scores = self.collect_scores(work_dir=work_dir, scorefile=scorefilepath, rename_pdbs=True).reset_index(drop=True)

        # update residue mappings for stored motifs
        if update_motifs:
            motifs = prep_motif_input(update_motifs, poses.df)
            for motif_col in motifs:
                poses.df[motif_col] = update_motif_res_mapping(poses.df[motif_col].to_list(), poses.df[f"{prefix}_con_ref_pdb_idx"].to_list(), poses.df[f"{prefix}_con_hal_idx"])

        # Reintegrate into poses and return
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, pose: str, options: str, pose_opts: str, output_dir: str, num_diffusions: int=1) -> str:
        '''AAA'''
        # parse description:
        if pose:
            desc = os.path.splitext(os.path.basename(pose))[0]

        # parse options:
        start_opts = self.parse_rfdiffusion_opts(options, pose_opts)

        if "inference.input_pdb" not in start_opts and pose is not None: # if no pose present, ignore input_pdb
            start_opts["inference.input_pdb"] = pose
        if "inference.num_designs" not in start_opts:
            start_opts["inference.num_designs"] = num_diffusions
        if "inference.output_prefix" not in start_opts:
            start_opts["inference.output_prefix"] = os.path.join(output_dir, desc)

        opts_str = " ".join([f"{k}={v}" for k, v in start_opts.items()])

        # return cmd
        return f"{self.python_path} {self.script_path} {opts_str}"

    def parse_rfdiffusion_opts(self, options: str, pose_options: str) -> dict:
        '''AAA'''
        def re_split_rfdiffusion_opts(command) -> list:
            if command is None:
                return []
            return re.split(r"\s+(?=(?:[^']*'[^']*')*[^']*$)", command)

        splitstr = [x for x in re_split_rfdiffusion_opts(options) + re_split_rfdiffusion_opts(pose_options) if x] # adding pose_opts after options makes sure that pose_opts overwrites options!
        return {x.split("=")[0]: "=".join(x.split("=")[1:]) for x in splitstr}

    def collect_scores(self, work_dir: str, scorefile: str, rename_pdbs: bool = True) -> pd.DataFrame:
        '''collects scores from RFdiffusion output'''
        # collect scores from .trb-files into one pandas DataFrame:
        pdb_dir = os.path.join(work_dir, "output_pdbs")
        pl = glob(f"{pdb_dir}/*.pdb")
        if not pl: raise FileNotFoundError(f"No .pdb files were found in the diffusion output direcotry {pdb_dir}. RFDiffusion might have crashed (check inpainting error-log), or the path might be wrong!")

        # collect rfdiffusion scores into a DataFrame:
        scores = []
        for pdb in pl:
            if os.path.isfile(trb := pdb.replace(".pdb", ".trb")):
                scores.append(parse_diffusion_trbfile(trb))
        scores = pd.concat(scores)

        # rename pdbs if option is set:
        if rename_pdbs is True:
            scores.loc[:, "new_description"] = ["_".join(desc.split("_")[:-1]) + "_" + str(int(desc.split("_")[-1]) + 1).zfill(4) for desc in scores["description"]]
            scores.loc[:, "new_loc"] = [loc.replace(old_desc, new_desc) for loc, old_desc, new_desc in zip(list(scores["location"]), list(scores["description"]), list(scores["new_description"]))]

            # rename all diffusion outputfiles according to new indeces:
            _ = [[os.rename(f, f.replace(old_desc, new_desc)) for f in glob(f"{pdb_dir}/{old_desc}.*")] for old_desc, new_desc in zip(list(scores["description"]), list(scores["new_description"]))]

            # Collect information of path to .pdb files into DataFrame under 'location' column
            scores = scores.drop(columns=["location"]).rename(columns={"new_loc": "location"})
            scores = scores.drop(columns=["description"]).rename(columns={"new_description": "description"})

        scores.reset_index(drop=True, inplace=True)

        logging.info(f"Saving scores of {self} at {scorefile}")
        scores.to_json(scorefile)

        return scores

def parse_diffusion_trbfile(path: str) -> pd.DataFrame:
    '''AAA'''
    # read trbfile:
    if path.endswith(".trb"): data_dict = np.load(path, allow_pickle=True)
    else: raise ValueError(f"only .trb-files can be passed into parse_inpainting_trbfile. <trbfile>: {path}")

    # calc mean_plddt:
    sd = {}
    last_plddts = data_dict["plddt"][-1]
    sd["plddt"] = [sum(last_plddts) / len(last_plddts)]
    sd["perres_plddt"] = [last_plddts]

    # instantiate scoresdict and start collecting:
    scoreterms = ["con_hal_pdb_idx", "con_ref_pdb_idx", "sampled_mask"]
    for st in scoreterms:
        sd[st] = [data_dict[st]]

    # collect metadata
    sd["location"] = path.replace(".trb", ".pdb")
    sd["description"] = path.split("/")[-1].replace(".trb", "")
    sd["input_pdb"] = data_dict["config"]["inference"]["input_pdb"]

    return pd.DataFrame(sd)

def prep_motif_input(motif: Any, df: pd.DataFrame) -> list[str]:
    '''Makes sure motif is list (even when string given) and that motifs are present in df.'''
    # ambivalence to singular or multiple motif cols
    motifs = [motif] if isinstance(motif, str) else motif

    # clear
    for m in motifs:
        col_in_df(df, m)

    return motifs

def update_motif_res_mapping(motif_l: list[ResidueSelection], con_ref_idx: list, con_hal_idx: list) -> list:
    '''Updates motifs in motif_l based on con_ref_idx and con_hal_idx'''
    output_motif_l = []
    for motif, ref_idx, hal_idx in zip(motif_l, con_ref_idx, con_hal_idx):
        # error handling
        if not isinstance(motif, ResidueSelection):
            raise TypeError(f"Individual motifs must be of type ResidueSelection. Create ResidueSelection objects out of your motifs.")

        # setup mapping from rfdiffusion outputs:
        exchange_dict = get_residue_mapping(ref_idx, hal_idx)

        # exchange and return
        exchanged_motif = ResidueSelection([exchange_dict[residue] for residue in motif.residues])
        output_motif_l.append(exchanged_motif)
    return output_motif_l

def get_residue_mapping(con_ref_idx: list, con_hal_idx: list) -> dict:
    '''Creates a residue mapping dictionary {old: new} from rfdiffusion outputs.'''
    return {(chain, int(res_id)): hal for (chain, res_id), hal in zip(con_ref_idx, con_hal_idx)}
