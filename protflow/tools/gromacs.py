'''Module to run gromacs on clean, predicted .pdb files within ProtFlow.'''
# generals
import os
import logging
import re

# dependencies
import pandas as pd

# customs
import protflow
from protflow.runners import Runner
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.config import PROTFLOW_DIR, PROTFLOW_ENV
from protflow.utils import biopython_tools as bpt

GROMACS_PARAMS_DIR = os.path.join(PROTFLOW_DIR, "protflow/utils/gromacs/params")

class Gromacs(Runner):
    '''Class Docs'''
    def __init__(self, gromacs_path: str = protflow.config.GROMACS_PATH, jobstarter: JobStarter = None, pre_cmd: str = None, md_params: "MDParams" = None):
        '''Init Docs'''
        self.gromacs_path = self.search_path(os.path.join(gromacs_path, "gmx"), "GROMACS_PATH")
        self.gromacs_dir = self.search_path(gromacs_path, "GROMACS_PATH", is_dir=True)
        self.pre_cmd = pre_cmd
        self.name = "esmfold.py"
        self.index_layers = 0
        self.jobstarter = jobstarter

        self.md_params = md_params or MDParams()
        self.overwrite_prep = False
        self.overwrite_equilibration = False
        self.overwrite_md = False
        self.overwrite_extract = False

    def set_md_params(self, md_params: "MDParams") -> None:
        '''utility method to set md_params.'''
        if not isinstance(md_params, MDParams):
            raise ValueError(f"md_params must be of type MDParams. type(md_params): {type(md_params)}")
        self.md_params = md_params

    def __str__(self):
        return "gromacs.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, n: int = 1, t_ns: int = None):
        '''Run function docstring. Overwriting is configured at the runner level.'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # check if poses are .pdb files
        if not all(pose.endswith(".pdb") for pose in poses.poses_list()):
            raise ValueError(f"All poses must be .pdb files to run MD simulation! If you have sequences, then run any structure prediction method beforehand (see runners esmfold, colabfold, alphafold3 or boltz).\nYour current poses: {poses.poses_list()}")

        # setup pose dirs
        pose_dirs = []
        for pose in poses.poses_list():
            pose_dir = os.path.join(work_dir, protflow.poses.description_from_path(pose))
            os.makedirs(pose_dir, exist_ok=True)
            pose_dirs.append(pose_dir)

        # multiplex poses for any given n:
        poses.duplicate_poses(
            output_dir = os.path.join(prefix, "input_pdbs"),
            n_duplicates = n,
            overwrite = self.overwrite_prep
        )

        # prepare inputs
        self.prep_md_inputs(
            work_dir = work_dir,
            poses = poses,
            prefix = prefix,
            jobstarter = jobstarter,
        )

        # equilibrate system
        self.equilibrate_system(
            work_dir = work_dir,
            poses = poses,
            prefix = prefix,
            jobstarter = jobstarter,
        )

        # run simulation
        self.run_md(
            work_dir = work_dir,
            poses = poses,
            prefix = prefix,
            jobstarter = jobstarter,
        )

        # extract simulation
        self.extract_md(work_dir = work_dir,
            poses = poses,
            prefix = prefix,
            jobstarter = jobstarter,
        )

        # return unchanged poses (gromacs files are stored under pefixes!)
        return poses

    def prep_md_inputs(self, work_dir: str, poses: Poses, prefix: str, jobstarter: JobStarter):
        '''Setup a system for equilibration.'''
        # clean up input models
        cleaned_poses = []
        pose_dirs = []
        for pose in poses.poses_list():
            # create prep dir
            pose_dir = os.path.join(work_dir, protflow.poses.description_from_path(pose))
            prep_dir = os.path.join(pose_dir, "prep")
            os.makedirs(prep_dir, exist_ok=True)
            pose_dirs.append(prep_dir)

            # clean .pdb file (any molecules will cause gromacs to crash)
            cleaned_fn = os.path.join(prep_dir, os.path.basename(pose))
            if not os.path.isfile(cleaned_fn) or self.overwrite_prep:
                # load model using biopython
                model = bpt.load_structure_from_pdbfile(pose)

                # load and save protein-only file using BioPython (remove non-protein residues)
                only_residue_model = bpt.remove_non_residue_residues(model)
                bpt.save_structure_to_pdbfile(only_residue_model, save_path=cleaned_fn)
            else:
                logging.info(f"File {cleaned_fn} exists and overwrite is set to {self.overwrite_prep}. Skipping pdb_cleanup.")
            cleaned_poses.append(cleaned_fn)
        poses.df[f"{prefix}_cleaned_poses"] = cleaned_poses

        # compile commands to convert pdb to gmx file
        cmds = []
        processed_poses = []
        topol_fn_list = []
        for pose, pose_dir in zip(cleaned_poses, pose_dirs):
            processed_fn = os.path.join(pose_dir, protflow.poses.description_from_path(pose) + "_processed.gro")
            pdb2gmx_cmd = f"cd {pose_dir}; {self.gromacs_path} pdb2gmx -f {pose} -o {processed_fn} -ter -ignh -water {self.md_params.water_model} -ff {self.md_params.force_field}"
            cmds.append(pdb2gmx_cmd)
            processed_poses.append(processed_fn)
            topol_fn_list.append(os.path.join(pose_dir, "topol.top"))
        poses.df[f"{prefix}_processed_poses"] = processed_poses
        poses.df[f"{prefix}_topol_poses"] = topol_fn_list

        # convert cleaned poses to gmx (.gro) files
        # check if all outputs exist:
        if not all(os.path.isfile(processed_fn) for processed_fn in processed_poses) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_prep",
                wait = True,
                output_path = work_dir
            )
        else:
            logging.info(f"Outputs of gmx processing found at {work_dir} and overwrite is set to {self.overwrite_prep}. Skipping pdb2gmx.")

        ###### input prep #######
        # create periodic boundary (editconf)
        pbc_fn_list = []
        cmds = []
        for pose, pose_dir in zip(poses, pose_dirs):
            pbc_fn = os.path.join(pose_dir, pose["poses_description"] + "_pbc.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} editconf -f {pose[f'{prefix}_processed_poses']} -o {pbc_fn} -bt dodecahedron -d 1.0" #TODO implement this as md_params parameter!
            cmds.append(cmd)
            pbc_fn_list.append(pbc_fn)
        poses.df[f"{prefix}_pbc_poses"] = pbc_fn_list

        if not all(os.path.isfile(pbc_fn) for pbc_fn in pbc_fn_list) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_pbc_setup",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"PBCs created at for poses at {work_dir}.")
        else:
            logging.info(f"PBC files exist at {work_dir} and overwrite is set to {self.overwrite_prep}. Skipping editconf (pbc generation).")

        # solvate protein in periodic boundary (solvate)
        cmds = []
        solv_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            solv_fn = os.path.join(pose_dir, pose["poses_description"] + "_solv.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} solvate -cp {pose[f'{prefix}_pbc_poses']} -cs spc216.gro -p {pose[f'{prefix}_topol_poses']} -o {solv_fn}"
            solv_fn_list.append(solv_fn)
            cmds.append(cmd)
        poses.df[f"{prefix}_solv_poses"] = solv_fn_list

        if not all(os.path.isfile(solv_fn) for solv_fn in solv_fn_list) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_solv",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Proteins solvated at {work_dir}.")
        else:
            logging.info(f"Solvated files exist in {work_dir} and overwrite is set to {self.overwrite_prep}. Skipping solvate.")

        # add ions to solvated protein (grompp .tpr file creation and then genion)
        cmds = []
        ions_tpr_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            ions_tpr_fn = os.path.join(pose_dir, pose["poses_description"] + "_ions.tpr")
            grompp_cmd = f"cd {pose_dir}; {self.gromacs_path} grompp -f {self.md_params.ions} -c {pose[f'{prefix}_solv_poses']} -p {pose[f'{prefix}_topol_poses']} -o {ions_tpr_fn}"
            ions_tpr_fn_list.append(ions_tpr_fn)
            cmds.append(grompp_cmd)
        poses.df[f"{prefix}_ions_tpr_poses"] = ions_tpr_fn_list

        if not all(os.path.isfile(ions_tpr_fn) for ions_tpr_fn in ions_tpr_fn_list) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_ions_tpr_setup",
                wait = True,
                output_path = work_dir
            )
        else:
            logging.info(f"File {ions_tpr_fn} exists and overwrite is set to {self.overwrite_prep}. Skipping generation of ion_tpr file.")

        # add ions to protein solvation shells.
        cmds = []
        ions_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            ions_fn = os.path.join(pose_dir, pose["poses_description"] + "_ions.gro")
            genion_cmd = f"cd {pose_dir}; {self.gromacs_path} genion -s {pose[f'{prefix}_ions_tpr_poses']} -o {ions_fn} -p {pose[f'{prefix}_topol_poses']} -pname NA -nname CL -conc 0.15 -neutral <<< SOL" #TODO: implement ions and concentration as parameters in md_params
            cmds.append(genion_cmd)
            ions_fn_list.append(ions_fn)
        poses.df[f"{prefix}_ions_poses"] = ions_fn_list

        if not all(os.path.isfile(ions_fn) for ions_fn in ions_fn_list) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_ions_setup",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Added ions to system at {work_dir}")
        else:
            logging.info(f"Files for ionization existn at {work_dir} and overwrite is set to {self.overwrite_prep}. Skipping genion step.")

        # create index files
        cmds = []
        index_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            index_fn = os.path.join(pose_dir, "index.ndx")
            index_cmd = f"cd {pose_dir}; {self.gromacs_path} make_ndx -f {pose[f'{prefix}_ions_poses']} -o {index_fn} <<< $'1\\nq'"
            cmds.append(index_cmd)
            index_fn_list.append(index_fn)
        poses.df[f"{prefix}_index_poses"] = index_fn_list

        if not all(os.path.isfile(index_fn) for index_fn in index_fn_list) or self.overwrite_prep:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_index_setup",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Created index files at {work_dir}.")
        else:
            logging.info(f"File {index_fn} exists and overwrite is set to {self.overwrite_prep}. Skipping creation of index file (make_ndx)")

    def equilibrate_system(self, work_dir: str, poses: Poses, prefix: str, jobstarter: JobStarter):
        '''Helper function to equilibrate and energy-minimize system for md simulation.'''
        #### energy minimization ####

        # setup individual equilibration dirs
        pose_dirs = []
        for pose in poses.poses_list():
            # setup equilibration root dir
            pose_dir = os.path.join(work_dir, protflow.poses.description_from_path(pose) + "/equilibration")
            os.makedirs(pose_dir, exist_ok=True)
            pose_dirs.append(pose_dir)

        # write commands for system energy minimization
        em_tpr_fn_list = []
        em_fn_list = []
        cmds = []
        for pose, pose_dir in zip(poses, pose_dirs):
            em_tpr_fn = os.path.join(pose_dir, "em.tpr")
            em_fn = os.path.join(pose_dir, "em.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} grompp -f {self.md_params.em} -c {pose[f'{prefix}_ions_poses']} -p {pose[f'{prefix}_topol_poses']} -o {em_tpr_fn} && {self.gromacs_path} mdrun -v -deffnm em"
            em_tpr_fn_list.append(em_tpr_fn)
            em_fn_list.append(em_fn)
            cmds.append(cmd)
        poses.df[f"{prefix}_em_tpr_poses"] = em_tpr_fn_list
        poses.df[f"{prefix}_em_poses"] = em_fn_list

        # start energy minimization
        if not all(os.path.isfile(em_file) for em_file in em_tpr_fn_list + em_fn_list) or self.overwrite_equilibration:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_em",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Starting energy minimization at {work_dir}")
        else:
            logging.info(f"Energy minimization files exist at {work_dir} and overwrite is set to {self.overwrite_equilibration}. Skipping energy minimization")

        # compile nvt equilibration commands
        cmds = []
        nvt_tpr_fn_list = []
        nvt_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            nvt_tpr_fn = os.path.join(pose_dir, "nvt.tpr")
            nvt_fn = os.path.join(pose_dir, "nvt.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} grompp -f {self.md_params.nvt} -c {pose[f'{prefix}_em_poses']} -r {pose[f'{prefix}_em_poses']} -p {pose[f'{prefix}_topol_poses']} -n {pose[f'{prefix}_index_poses']} -o {nvt_tpr_fn} && {self.gromacs_path} mdrun -v -deffnm nvt"
            cmds.append(cmd)
            nvt_tpr_fn_list.append(nvt_tpr_fn)
            nvt_fn_list.append(nvt_fn)
        poses.df[f"{prefix}_nvt_tpr_poses"] = nvt_tpr_fn_list
        poses.df[f"{prefix}_nvt_poses"] = nvt_fn_list

        # execute nvt equilibration (check if outputs are present)
        if not all(os.path.isfile(nvt_file) for nvt_file in nvt_tpr_fn_list + nvt_fn_list) or self.overwrite_equilibration:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_nvt",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Starting nvt equilibration at {work_dir}")
        else:
            logging.info(f"Equilibration files exist at {work_dir} and overwrite is set to {self.overwrite_equilibration}. Skipping nvt equilibration")

        # compile npt-equilibration commands
        cmds = []
        npt_tpr_fn_list = []
        npt_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            npt_tpr_fn = os.path.join(pose_dir, "npt.tpr")
            npt_fn = os.path.join(pose_dir, "npt.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} grompp -f {self.md_params.npt} -c {pose[f'{prefix}_nvt_poses']} -r {pose[f'{prefix}_nvt_poses']} -p {pose[f'{prefix}_topol_poses']} -n {pose[f'{prefix}_index_poses']} -o {npt_tpr_fn} && {self.gromacs_path} mdrun -v -deffnm npt"
            cmds.append(cmd)
            npt_tpr_fn_list.append(npt_tpr_fn)
            npt_fn_list.append(npt_fn)
        poses.df[f"{prefix}_npt_tpr_poses"] = npt_tpr_fn_list
        poses.df[f"{prefix}_npt_poses"] = npt_fn_list

        # execute npt-equilibration commands
        if not all(os.path.isfile(npt_file) for npt_file in npt_tpr_fn_list + npt_fn_list) or self.overwrite_equilibration:
            jobstarter.start(
                cmds = cmds,
                jobname = "md_npt",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Starting npt equilibration at {work_dir}")
        else:
            logging.info(f"Files {npt_tpr_fn} and {npt_fn} exist and overwrite is set to {self.overwrite_equilibration}. Skipping nvt equilibration")

    def run_md(self, work_dir: str, poses: Poses, prefix: str, jobstarter: JobStarter):
        '''Runs MD simulation after preparation and equilibration'''
        #### production md ####
        # setup individual md dirs
        pose_dirs = []
        for pose in poses.poses_list():
            # setup equilibration root dir
            pose_dir = os.path.join(work_dir, protflow.poses.description_from_path(pose) + "/md")
            os.makedirs(pose_dir, exist_ok=True)
            pose_dirs.append(pose_dir)

        # compile commands for md run
        cmds = []
        md_tpr_fn_list = []
        md_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            md_tpr_fn = os.path.join(pose_dir, "md.tpr")
            md_fn = os.path.join(pose_dir, "md.xtc")
            cmd = f"cd {pose_dir}; {self.gromacs_path} grompp -f {self.md_params.md} -c {pose[f'{prefix}_npt_poses']} -p {pose[f'{prefix}_topol_poses']} -n {pose[f'{prefix}_index_poses']} -o {md_tpr_fn} && {self.gromacs_path} mdrun -v -deffnm md -nb auto -pme auto -bonded cpu -update auto"
            cmds.append(cmd)
            md_tpr_fn_list.append(md_tpr_fn)
            md_fn_list.append(md_fn)
        poses.df[f"{prefix}_md_tpr_poses"] = md_tpr_fn_list
        poses.df[f"{prefix}_md_poses"] = md_fn_list

        if not all(os.path.isfile(sim_file) for sim_file in md_tpr_fn_list + md_fn_list) or self.overwrite_md:
            jobstarter.start(
                cmds = cmds,
                jobname = "md",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Starting production MD runs at {work_dir}")
        else:
            logging.info(f"Production MD output files found at {work_dir} and overwrite is set to {self.overwrite_md}. Skipping MD simulation!")

    def extract_md(self, work_dir: str, poses: Poses, prefix: str, jobstarter: JobStarter):
        '''MD production postprocessing function.'''
        # setup individual md dirs
        pose_dirs = []
        for pose in poses.poses_list():
            # setup equilibration root dir
            pose_dir = os.path.join(work_dir, protflow.poses.description_from_path(pose) + "/postprocessing")
            os.makedirs(pose_dir, exist_ok=True)
            pose_dirs.append(pose_dir)

        # compile command to rewrap trajectories around the PBC
        cmds = []
        pbc_remove_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            sys_pbc_remove_fn = os.path.join(pose_dir, f"{pose['poses_description']}_noPBC.xtc")
            cmd = f"cd {pose_dir}; {self.gromacs_path} trjconv -s {pose[f'{prefix}_md_tpr_poses']} -f {pose[f'{prefix}_md_poses']} -o {sys_pbc_remove_fn} -pbc mol -ur compact -center <<< $'C-alpha\\nSystem'"
            pbc_remove_fn_list.append(sys_pbc_remove_fn)
            cmds.append(cmd)
        poses.df[f"{prefix}_pbc_remove_poses"] = pbc_remove_fn_list

        # execute pbc rewrapping:
        if not all(os.path.isfile(outfile) for outfile in pbc_remove_fn_list) or self.overwrite_extract:
            jobstarter.start(
                cmds = cmds,
                jobname = "md",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Rewrapping MD run around periodic boundary at {work_dir}")
        else:
            logging.info(f"Rewrapped files exist at {work_dir} and overwrite is set to {self.overwrite_extract}. Skipping removal of PBC.")

        # compile fitting command for trajectory
        cmds = []
        fit_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            fit_fn = os.path.join(pose_dir, f"{pose['poses_description']}_noHOH.xtc")
            cmd = f"cd {pose_dir}; {self.gromacs_path} trjconv -s {pose[f'{prefix}_md_tpr_poses']} -f {pose[f'{prefix}_pbc_remove_poses']} -o {fit_fn} -n {pose[f'{prefix}_index_poses']} -fit rot+trans <<< $'C-alpha\\n1'"
            fit_fn_list.append(fit_fn)
            cmds.append(cmd)
        poses.df[f'{prefix}_fit_poses'] = fit_fn_list

        # execute fitting of trajectory to starting coords
        if not all(os.path.isfile(fit_fn) for fit_fn in fit_fn_list) or self.overwrite_extract:
            jobstarter.start(
                cmds = cmds,
                jobname = "md",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Fitting Protein MD trajectory to starting coords for visualization. Working in: {work_dir}")
        else:
            logging.info(f"Output files of MD fitting were found at {work_dir} and overwrite is set to {self.overwrite_extract}. Skipping fit and extraction of Protein from System.")

        # repeat re-wrap and fitting for protein starting conformation
        # compile rewrap command
        cmds = []
        t0_extract_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            t0_extract_fn = os.path.join(pose_dir, f"{pose['poses_description']}_t0.gro")
            cmd = f"cd {pose_dir}; {self.gromacs_path} trjconv -s {pose[f'{prefix}_md_tpr_poses']} -f {pose[f'{prefix}_npt_poses']} -o {t0_extract_fn} -n {pose[f'{prefix}_index_poses']} -center -pbc mol <<< $'System\\n1'"
            t0_extract_fn_list.append(t0_extract_fn)
            cmds.append(cmd)
        poses.df[f'{prefix}_t0_extract_poses'] = t0_extract_fn_list

        # execute overwrite
        if not all(os.path.isfile(outf) for outf in t0_extract_fn_list) or self.overwrite_extract:
            jobstarter.start(
                cmds = cmds,
                jobname = "md",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Rewrapping and fitting starting conformation of md traj. Working in: {work_dir}")
        else:
            logging.info(f"Outputs for rewrapping and fitting of starting conformation already exist at {work_dir}. Overwrite is set to {self.overwrite_extract}. Skipping extraction of Protein at t0.")

        # create protein-only .tpr file (for visualization)
        cmds = []
        tpr_extract_fn_list = []
        for pose, pose_dir in zip(poses, pose_dirs):
            tpr_extract_fn = os.path.join(pose_dir, f"{pose['poses_description']}_protein_only.tpr")
            cmd = f"cd {pose_dir}; {self.gromacs_path} convert-tpr -s {pose[f'{prefix}_md_tpr_poses']} -o {tpr_extract_fn} <<< $'1'"
            tpr_extract_fn_list.append(tpr_extract_fn)
            cmds.append(cmd)

        if not all(os.path.isfile(tpr_extract_fn) for tpr_extract_fn in tpr_extract_fn_list) or self.overwrite_extract:
            jobstarter.start(
                cmds = cmds,
                jobname = "md",
                wait = True,
                output_path = work_dir
            )
            logging.info(f"Creating matching tpr files for viz. Working in: {work_dir}")
        else:
            logging.info(f"Protein-only .tpr files already exist at {work_dir} and overwrite is set to {self.overwrite_extract}. Skipping creation of matching tpr file.")

class MDParams:
    '''Dataclass that links MD parameter fiels (.mdp).'''
    def __init__(
            self,
            ions = f"{GROMACS_PARAMS_DIR}/default_ions.mdp",
            em = f"{GROMACS_PARAMS_DIR}/default_em.mdp",
            nvt = f"{GROMACS_PARAMS_DIR}/default_nvt.mdp",
            npt = f"{GROMACS_PARAMS_DIR}/default_npt.mdp",
            md = f"{GROMACS_PARAMS_DIR}/default_md_1ns.mdp",
            water_model = "tip3p",
            force_field = "amber99sb-ildn"
        ) -> None:

        self.ions = ions
        self.em = em
        self.nvt = nvt
        self.npt = npt
        self.md = md
        self.water_model = water_model
        self.force_field = force_field

    def set_params(self, ions: str = None, em: str = None, nvt: str = None, npt: str = None, md: str = None, water_model: str = None, force_field: str = None) -> None:
        """document"""
        if ions is not None:
            self.ions = ions
        if em is not None:
            self.em = em
        if nvt is not None:
            self.nvt = nvt
        if npt is not None:
            self.npt = npt
        if md is not None:
            self.md = md
        if water_model is not None:
            self.water_model = water_model
        if force_field is not None:
            self.force_field = force_field

# define allowed pattern
def is_valid_variable(var: str) -> None:
    '''Checking function for security purposes'''
    pattern = r'^[A-Za-z0-9_/\-]+$'
    if not bool(re.fullmatch(pattern, var)):
        raise ValueError(f'Variable contains illegal characters. Only letters, "_", "-", or "/" allowed. variable: {var}')

class MDAnalysis(Runner):
    '''MDAnalysis class docs'''
    def __init__(self, python: str = f"{PROTFLOW_ENV}/python", script_path: str = None, jobstarter: JobStarter = None):
        self.python = python
        self.set_script(script_path)
        self.index_layers = 0
        self.jobstarter = jobstarter

        # options and flags
        self.options = ""
        self.pose_options = ""
        self.pose_flags = ""

    def __str__(self):
        return "MDAnalysis"

    def set_script(self, script_path: str) -> None:
        '''Set the script that should be run by MDAnalysis.run() method. 
        The script has to output a pandas DataFrame containing at least "location" and "description" columns that can be integrated by RunnerOutput methods.'''
        self.script_path = script_path

    def set_options(self, options: str) -> None:
        '''Set options that should be added to every execution of your md-analysis script.'''
        self.options = options

    def set_pose_options(self, pose_options: dict) -> None:
        '''Set pose_options for MDAnalysis self.script. 
        'pose_options' has to be a dictionary. 
        The keys of the dictionary are the 'flags' of the command-line options to be invoked.
        The values specified for the keys should be of type 'str' and should represent columns in the poses.df DataFrame of the Poses that you want to analyze.
        The command-writer will assign every pose its corresponding value of the column specified.
        This way, every pose can have its own value for any given parameter (i.e. residue index for which to calculate RMSF).'''
        self.pose_options = pose_options

    def set_pose_flags(self, pose_flags: str) -> None:
        '''
        pose_flags parameter should point to a column name in the poses.df applied to MDAnalysis.run() that specifies pose-specific flags to be added to script execution.
        Values in poses.df column 'pose_flags' should be one single string that contains all flags as you would put them in a commandline.
        TODO: write example.
        '''
        self.pose_flags = pose_flags

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, overwrite: bool = False) -> Poses:
        '''Run your MDAnalysis script.'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check for output
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            self.integrate_scores(poses, scores, prefix)
            return poses

        # write commands
        cmds = self.write_cmds(poses)

        # create input folders
        for pose in poses:
            os.makedirs(os.path.join(work_dir, pose["poses_description"]), exist_ok=True)

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = "md_analysis",
            wait = True,
            output_path = work_dir
        )

        # collect scores
        scores = self.collect_scores(work_dir, poses)

        # integrate and return
        self.integrate_scores(poses, scores, prefix)
        return poses

    def write_cmds(self, poses: Poses) -> list[str]:
        '''Automated cmd-file writer. Takes class attributes self.python, self.script_path, self.options and self.pose_options to write command for .run() method.'''
        # check if pose_options specified columns are present in poses.df
        if self.pose_options:
            if not isinstance(self.pose_options, dict):
                raise ValueError(f"MDAnalysis attribute .pose_options must be a dictionary holding {{'option': 'poses.df column name'}}\nCurrent .pose_options: {self.pose_options}")
            protflow.poses.col_in_df(poses.df, list(self.pose_options.values()))
        if self.pose_flags:
            if not isinstance(self.pose_flags, str):
                raise ValueError(f"MDAnalysis attribute .pose_flags must be of type(str). Current self.pose_flags: {self.pose_flags}")
            protflow.poses.col_in_df(poses.df, self.pose_flags)

        # prepare options and flags from class attributes for every pose.
        options_list = []
        for pose in poses:
            pose_options_str = ""
            if self.pose_options:
                # parse individual pose_opts from pose-level pose_options
                pose_options_str += " ".join([f"--{opt}={pose[opt_col]}" for opt, opt_col in self.pose_options.items()])

            # add flags
            if self.pose_flags:
                # parse pose-level flags
                pose_options_str += f" {pose[self.pose_flags]}"

            # ensure that pose_options overwrite options. Same with flags
            parsed_opts, parsed_flags = protflow.runners.parse_generic_options(self.options, pose_options_str, sep="--")

            # parse options and flags into combined string and add to options_list
            parsed_cmd = protflow.runners.options_flags_to_string(parsed_opts, parsed_flags)
            options_list.append(parsed_cmd)

        # be aware that this compiling of commands requires the user to specify the input in pose_options!
        cmds = [f"{self.python} {self.script_path} {options}" for options in options_list]
        return cmds

    def integrate_scores(self, poses: Poses, scores: pd.DataFrame, prefix: str) -> None:
        '''Merges 'scores' from collect_scores() call into poses.df'''
        startlen = len(poses)
        dn = poses.df["poses_description"].head(5)

        # add prefix to scores
        scores.add_prefix(prefix)

        # merge
        poses.df.merge(scores, left_on="poses_description", right_on=f"{prefix}_description")

        # check if merge was successful
        if len(poses.df < startlen):
            raise ValueError(f"Merging DataFrames failed. Some rows in results[new_df_col] were not found in poses.df['poses_description']\nposes_description: {dn}\nmerge_col {prefix}_description: {scores[f'{prefix}_descrption'].head(5)}")

        return None


    def collect_scores(self, work_dir: str, poses: Poses) -> pd.DataFrame:
        '''Collect scores of MDAnalysis scripts.'''
        # Every command writes its scores into its own directories.
        scores_list = []
        for pose in poses:
            print(pose.index)
            pose_scores_fn = f"{work_dir}/{pose['poses_description']}/mdanalysis_scores.json"
            scores_list.append(pd.read_json(pose_scores_fn))
        scores_df = pd.concat(scores_list, ignore_index=True).reset_index(drop=True)
        return scores_df
