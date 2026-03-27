"""
LigandMPNN Module
=================

This module provides the functionality to integrate LigandMPNN within the ProtFlow framework. It offers tools to run LigandMPNN, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `LigandMPNN` class encapsulates the functionality necessary to execute LigandMPNN runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of the diffusion processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of LigandMPNN into larger computational workflows. It supports the automatic setup of job parameters, execution of LigandMPNN commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `LigandMPNN` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `LigandMPNN` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from ligandmpnn import LigandMPNN

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the LigandMPNN class
    ligandmpnn = LigandMPNN()

    # Run the diffusion process
    results = ligandmpnn.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        nseq=10,
        model_type="ligand_mpnn",
        options="some_option=some_value",
        pose_options=["pose_option=pose_value"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the process.
- Customizability: Users can customize the process through multiple parameters, including the number of sequences, specific options for the LigandMPNN script, and options for handling pose-specific parameters.
- Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate LigandMPNN into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Author
------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""
# general imports
import os
import logging
import shutil
import re
from glob import glob
from pathlib import Path
from copy import deepcopy

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path
from ..residues import ResidueSelection
from ..poses import Poses, description_from_path
from ..jobstarters import JobStarter, split_list
from ..runners import Runner, RunnerOutput, col_in_df, options_flags_to_string, prepend_cmd
from ..utils.openbabel_tools import openbabel_fileconverter

class _CalibyRunner(Runner):
    def __init__(self, caliby_dir: str = None, python_path: str = None, pre_cmd: str = None, model_dir: str = None, script_path: str = None) -> None:

        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")
        self.model_dir = model_dir or os.path.join(self.caliby_dir, "model_params")
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)
        self.script_path = script_path


        # setup runner
        self.name = "caliby.py"

    def __str__(self):
        return "caliby.py"

    def create_constraint_csv(self, poses: Poses, work_dir: str, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None) -> str:
        cst_csv = pd.DataFrame({"pdb_key": [description_from_path(pose) for pose in poses.poses_list()]})

        if fixed_pos_seq_col:
            col_in_df(poses.df, fixed_pos_seq_col)
            cst_csv["fixed_pos_seq"] = [sele.to_string() if isinstance(sele, ResidueSelection) else sele for sele in poses.df[fixed_pos_seq_col].to_list()]
        if fixed_pos_scn_col:
            # TODO: add fixed_pos_scn to fixed_pos_seq if missing, but tricky if fixed_pos_seq used description like 'A5-15' and not explicit list
            col_in_df(poses.df, fixed_pos_scn_col)
            cst_csv["fixed_pos_scn"] = [sele.to_string() if isinstance(sele, ResidueSelection) else sele for sele in poses.df[fixed_pos_seq_col].to_list()]
        if fixed_pos_override_seq_col:
            col_in_df(poses.df, fixed_pos_override_seq_col)
            cst_csv["fixed_pos_override_seq"] = poses.df[fixed_pos_override_seq_col]
        if pos_restrict_aatype_col:
            col_in_df(poses.df, pos_restrict_aatype_col)
            cst_csv["pos_restrict_aatype"] = poses.df[pos_restrict_aatype_col]
        if symmetry_pos_col:
            col_in_df(poses.df, symmetry_pos_col)
            cst_csv["symmetry_pos"] = poses.df[symmetry_pos_col]

        cst_csv = cst_csv.fillna('')

        cst_csv.to_csv(out := os.path.join(work_dir, "pos_constraints.csv"), index=False)
        return os.path.abspath(out)

    def setup_batch_mode(self, pose_paths: list[str], options: dict, num_batches: int, work_dir: str, mode="single") -> list:

        def same_folder_check(file_paths):
            # Extract the absolute directory path for each file and put them in a set
            directories = {os.path.dirname(os.path.abspath(p)) for p in file_paths}

            # If all files are in the same folder, the set will only have 1 unique item
            return directories

        def write_input_list(pose_paths: list, filename: str):
            with open(filename, "w+", encoding="UTF-8") as f:
                f.write("\n".join([os.path.basename(pose) for pose in pose_paths]))

        in_folders = same_folder_check(pose_paths)

        # check if input files are all in same folder, otherwise copy to new folder
        if len(in_folders) > 1:
            os.makedirs(input_dir := os.path.join(work_dir, "input"), exist_ok=True)
            updated_paths = []
            for pose in pose_paths:
                shutil.copy(pose, new_path := os.path.join(input_dir, os.path.basename(pose)))
                updated_paths.append(new_path)
            if mode == "single":
                options["input_cfg.pdb_dir"] = input_dir
        else:
            updated_paths = pose_paths
            if mode == "single":
                options["input_cfg.pdb_dir"] = list(in_folders)[0]

        # split poses into batches
        pose_batches = split_list(updated_paths, n_sublists=num_batches)

        os.makedirs(input_list_dir := os.path.join(work_dir, "input_lists"), exist_ok=True)

        batch_opt_list = []
        for i, batch in enumerate(pose_batches):
            list_path = os.path.join(input_list_dir, f"in_{i}.txt")
            write_input_list(batch, list_path)
            batch_opts = options.copy()
            batch_opts["input_cfg.pdb_name_list"] = list_path
            batch_opts["out_dir"] = os.path.join(work_dir, f"batch_{i}")

            batch_opt_list.append(batch_opts)

        return batch_opt_list


    def parse_caliby_opts(self, options: str = None) -> dict:

        def re_split(command: str) -> list:
            # Return empty list if the string is empty
            if not command.strip():
                return []
            pattern = r'\s+(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
            return re.split(pattern, command)

        if not options:
            return {}

        raw_splits = re_split(options)
        
        parsed_config = {}
        for item in raw_splits:
            if "=" in item:
                key, value = item.split("=", 1)
                parsed_config[key] = value.strip("'\"")
                
        return parsed_config

    def write_cmd(self, options: dict) -> str:
        # convert to string
        options = options_flags_to_string(options, None, sep="")
        return f"{self.python_path} {self.script_path} {options}"

class CalibySequenceDesign(_CalibyRunner):
    """
    LigandMPNN Class
    ================

    The `LigandMPNN` class provides the necessary methods to execute LigandMPNN runs within the ProtFlow framework. This class is responsible for managing the configuration, execution, and output processing of LigandMPNN tasks.

    Detailed Description
    --------------------
    The `LigandMPNN` class integrates LigandMPNN into the ProtFlow pipeline by setting up the environment, running the diffusion process, and collecting the results. It ensures that the inputs and outputs are handled efficiently, making the data readily available for further analysis.

    Key Features:
    - Manages paths to essential scripts and executables.
    - Configures and executes LigandMPNN processes.
    - Collects and processes output data into a structured DataFrame format.
    - Handles various edge cases and supports custom configurations through multiple parameters.

    Usage
    -----
    To use this class, initialize it with the appropriate script and Python paths, along with an optional job starter. The main functionality is provided through the `run` method, which requires parameters such as poses, prefix, and additional options for customization.

    Example
    -------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from ligandmpnn import LigandMPNN

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the LigandMPNN class
        ligandmpnn = LigandMPNN()

        # Run the diffusion process
        results = ligandmpnn.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            nseq=10,
            model_type="ligand_mpnn",
            options="some_option=some_value",
            pose_options=["pose_option=pose_value"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Notes
    -----
    This class is designed to work within the ProtFlow framework and assumes that the necessary configurations and dependencies are properly set up. It leverages shared data structures and configurations from ProtFlow to provide a seamless integration experience.

    Author
    ------
    Markus Braun, Adrian Tripp

    Version
    -------
    0.1.0
    """
    def __init__(self, caliby_dir: str = None, python_path: str = None, model_dir: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:

        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")
        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/seq_des.py")
        self.model_dir = model_dir or os.path.join(self.caliby_dir, "model_params") 
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_cfg = os.path.join(self.caliby_dir, "caliby/configs/seq_des/atom_mpnn_inference.yaml")

        # setup runner
        self.name = "caliby.py"
        self.index_layers = 1
        self.jobstarter = jobstarter


    def run(self, poses: Poses, prefix: str, nseq: int = 1, model: str = "caliby", omit_aas: str|list = None, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None, pos_constraint_csv: str = None, return_seq_threaded_pdbs_as_pose: bool = False, options: str = None, cif_to_pdb: bool = True, jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None) -> Poses:

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"caliby_seq_des_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # check for pos_constraint_csv file
        if pos_constraint_csv and not os.path.isfile(pos_constraint_csv):
            raise ValueError(f"<pos_constraint_csv> must specify the path to a single csv file. Could not find a file at {pos_constraint_csv}.")

        if not os.path.isfile(model) and not os.path.isfile(model_path := os.path.join(self.model_dir, "caliby", f"{model}.ckpt")):
            raise FileNotFoundError(f"Could not detect a model at {model} or at {model_path}.")

        opt_dict = self.parse_caliby_opts(options)
        opt_dict["sampling_cfg_overrides.num_seqs_per_pdb"] = nseq
        opt_dict["ckpt_name_or_path"] = model if os.path.isfile(model) else model_path
        opt_dict["seq_des_cfg.atom_mpnn.sampling_cfg"] = self.sampling_cfg # TODO: this is a hack so caliby does not crash when running outside of installation dir, there might be better ways to solve this

        # convert omit_aas string to list, then to str that looks like a list
        if omit_aas and isinstance(omit_aas, str):
            omit_aas = [aa for aa in omit_aas]
        if omit_aas and isinstance(omit_aas, list):
            omit_aas = str(omit_aas)
            opt_dict["sampling_cfg_overrides.omit_aas"] = omit_aas

        if pos_constraint_csv:
            opt_dict["pos_constraint_csv"] = os.path.abspath(pos_constraint_csv)
        
        # check for conflicting options (pos_constraint_csv might have been defined via options!)
        if "pos_constraint_csv" in opt_dict and any([fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col]):
            raise ValueError("Pose-specific constraints cannot be set if a pregenerated pos_constraints_csv is provided!")
        
        # create new pos_constraint_csv from inputs, do not overwrite existing one
        opt_dict.setdefault("pos_constraint_csv", self.create_constraint_csv(poses, work_dir, fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col))
        
        # define number of batches
        if num_batches:
            num_batches = min([len(poses.poses_list()), num_batches])
        else:
            num_batches = min([len(poses.poses_list()), jobstarter.max_cores])

        # setup for batch mode
        batch_opts = self.setup_batch_mode(pose_paths=poses.poses_list(), options=opt_dict, num_batches=num_batches, work_dir=work_dir, mode="single")

        # write caliby cmds:
        cmds = [self.write_cmd(options=opt_dict) for opt_dict in batch_opts]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="caliby_seqdes",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose,
            cif_to_pdb=cif_to_pdb
        )

        if len(scores.index) < len(poses.df.index) * nseq:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()


class CalibyEnsembleGenerator(_CalibyRunner):
    def __init__(self, caliby_dir: str = None, model_dir: str = None, python_path: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:

        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")

        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/generate_ensembles.py")
        self.model_dir = model_dir or os.path.join(self.caliby_dir, "model_params") 
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_yaml_path = os.path.join(self.caliby_dir, "caliby/configs/protpardelle-1c/multichain_backbone_partial_diffusion.yaml")

        # setup runner
        self.name = "caliby.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, options: str = None, cif_to_pdb: bool = True, model_dir: str = None, jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None) -> Poses:

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"caliby_seq_des_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        opt_dict = self.parse_caliby_opts(options)
        opt_dict["num_samples_per_pdb"] = nstruct
        opt_dict["model_params_path"] = model_dir or self.model_dir
        opt_dict["sampling_yaml_path"] = self.sampling_yaml_path # TODO: this is a hack so caliby does not crash when running outside of installation dir, there might be better ways to solve this

        # define number of batches
        if num_batches:
            num_batches = min([len(poses.poses_list()), num_batches])
        else:
            num_batches = min([len(poses.poses_list()), jobstarter.max_cores])

        # setup for batch mode
        batch_opts = self.setup_batch_mode(pose_paths=poses.poses_list(), options=opt_dict, num_batches=num_batches, work_dir=work_dir, mode="single")

        # write caliby cmds:
        cmds = [self.write_cmd(options=opt_dict) for opt_dict in batch_opts]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="caliby_ensgen",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            mode="ens_gen",
            return_seq_threaded_pdbs_as_pose=False,
            cif_to_pdb=cif_to_pdb
        )

        if len(scores.index) < len(poses.df.index) * nstruct:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()



class CalibyEnsembleSeqDesign(_CalibyRunner):
    def __init__(self, caliby_dir: str = None, python_path: str = None, pre_cmd: str = None, model_dir: str = None, jobstarter: JobStarter = None) -> None:

        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")
        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/seq_des_ensemble.py")
        self.model_dir = model_dir or os.path.join(self.caliby_dir, "model_params")
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_cfg = os.path.join(self.caliby_dir, "caliby/configs/seq_des/atom_mpnn_inference.yaml")

        # setup runner
        self.name = "caliby.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "caliby.py"

    def run(self, poses: Poses, prefix: str, generate_ensembles: bool = True, gen_num_ensembles: int = 16, gen_ens_options: str = None,
            conformer_col: str = None, nseq: int = 1, model: str = "caliby", omit_aas: str|list = None, fixed_pos_seq_col: str = None, 
            fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None,
            pos_constraint_csv: str = None, return_seq_threaded_pdbs_as_pose: bool = False, options: str = None, cif_to_pdb: bool = True,
            jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None, run_clean: bool = True) -> Poses:

        """
        all ensembles that belong together have to be grouped by ensemble_group_col, ideally this is the path to the primary conformer (group_is_primary), otherwise primary will be chosen at random
        input poses can also be the primary conf, then ensemble_list_col must the path to a dir containing conformers or a list of conformer paths
        """

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"caliby_seq_des_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        
        if generate_ensembles:

            if conformer_col:
                logging.warning("<conformer_col> was set but <generate_ensembles> was set to True. <conformer_col> will be ignored.")

            ens_gen_prefix = f"{prefix}_auto_ensgen"
            poses = self.run_protpardelle_ensemble_generation(
                poses=poses,
                prefix=ens_gen_prefix, 
                nstruct=gen_num_ensembles, 
                options=gen_ens_options, 
                num_batches=num_batches, 
                jobstarter=jobstarter or self.jobstarter
            )
            
            conformer_col = f"{ens_gen_prefix}_conformer_dir"

        if not conformer_col:
            raise ValueError("Either <conformer_col> must be provided or <generate_ensembles> must be set to True!")

        col_in_df(poses.df, conformer_col)

        os.makedirs(ens_dir := os.path.join(work_dir, "conformers"), exist_ok=True)
        for _, row in poses.df.iterrows():
            os.makedirs(conf_dir := os.path.join(ens_dir, row["poses_description"]), exist_ok=True)
            confs = row[conformer_col]
            if isinstance(confs, str) and os.path.isdir(confs):
                files = Path(confs).glob("*.pdb")
            elif isinstance(confs, list):
                files = confs
            else:
                raise ValueError(f"<conformer_col> must be a path to a directory containing conformers or a list of conformer paths, not {confs}")
            
            # copy conformer files to new dir
            for conf in files:
                shutil.copy(conf, conf_dir)

        # check for pos_constraint_csv file
        if pos_constraint_csv and not os.path.isfile(pos_constraint_csv):
            raise ValueError(f"<pos_constraint_csv> must specify the path to a single csv file. Could not find a file at {pos_constraint_csv}.")

        if not os.path.isfile(model) and not os.path.isfile(model_path := os.path.join(self.model_dir, "caliby", f"{model}.ckpt")):
            raise FileNotFoundError(f"Could not detect a model at {model} or at {model_path}.")

        opt_dict = self.parse_caliby_opts(options)
        opt_dict["sampling_cfg_overrides.num_seqs_per_pdb"] = nseq
        opt_dict["ckpt_name_or_path"] = model if os.path.isfile(model) else model_path
        opt_dict["seq_des_cfg.atom_mpnn.sampling_cfg"] = self.sampling_cfg # TODO: this is a hack so caliby does not crash when running outside of installation dir, there might be better ways to solve this
        opt_dict["input_cfg.conformer_dir"] = ens_dir

        # convert omit_aas string to list, then to str that looks like a list
        if omit_aas and isinstance(omit_aas, str):
            omit_aas = [aa for aa in omit_aas]
        if omit_aas and isinstance(omit_aas, list):
            omit_aas = str(omit_aas)
            opt_dict["+sampling_cfg_overrides.omit_aas"] = omit_aas

        if pos_constraint_csv:
            opt_dict["pos_constraint_csv"] = os.path.abspath(pos_constraint_csv)
        
        # check for conflicting options (pos_constraint_csv might have been defined via options!)
        if "pos_constraint_csv" in opt_dict and any([fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col]):
            raise ValueError("Pose-specific constraints cannot be set if a pregenerated pos_constraints_csv is provided!")
        
        # create new pos_constraint_csv from inputs, do not overwrite existing one
        opt_dict.setdefault("pos_constraint_csv", self.create_constraint_csv(poses, work_dir, fixed_pos_seq_col, fixed_pos_scn_col, fixed_pos_override_seq_col, pos_restrict_aatype_col, symmetry_pos_col))
        
        # define number of batches
        if num_batches:
            num_batches = min([len(poses.poses_list()), num_batches])
        else:
            num_batches = min([len(poses.poses_list()), jobstarter.max_cores])

        # setup for batch mode
        batch_opts = self.setup_batch_mode(pose_paths=poses.poses_list(), options=opt_dict, num_batches=num_batches, work_dir=work_dir, mode="ensemble")

        # write caliby cmds:
        cmds = [self.write_cmd(options=opt_dict) for opt_dict in batch_opts]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="caliby_seqdes",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            mode="seq_des",
            return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose,
            cif_to_pdb=cif_to_pdb
        )

        if len(scores.index) < len(poses.df.index) * nseq:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # delete conformer dir
        if run_clean:
            shutil.rmtree(ens_dir)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def run_protpardelle_ensemble_generation(self, poses: Poses, prefix: str , nstruct: int, options: str, num_batches: int, jobstarter: JobStarter):
        # save current poses for later
        poses.df[f"{prefix}_temp_poses"] = poses.df["poses"]
        temp_poses = deepcopy(poses)

        ensgenerator = CalibyEnsembleGenerator(
            caliby_dir=self.caliby_dir,
            python_path=self.python_path,
            pre_cmd=self.pre_cmd,
            jobstarter=jobstarter)
        
        ensgenerator.run(poses=poses, prefix=prefix, nstruct=nstruct, options=options, num_batches=num_batches)
        poses.df = poses.df[[col for col in poses.df.columns if col.startswith(prefix)]]
        poses.df.drop_duplicates(subset=[f"{prefix}_conformer_dir"], inplace=True)
        poses.df = temp_poses.df.merge(poses.df, on=f"{prefix}_temp_poses")

        return poses


def collect_scores(work_dir: str, mode: str = "seq_des", return_seq_threaded_pdbs_as_pose: bool = False, cif_to_pdb: bool = True) -> pd.DataFrame:
    def write_fasta(seq, name, path):
        with open(path, "w+", encoding="UTF-8") as f:
            f.write(f">{name}\n{seq}")
        return os.path.abspath(path)

    def convert_cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return os.path.abspath(output)
    
    modes = ["seq_des", "ens_gen"]
    if mode not in  modes:
        raise ValueError(f"<mode> must be one of {modes}, depending on which scores (sequence design or ensemble generation) should be parsed.")

    if mode == "seq_des":
        # read .csv files
        csvs = glob(os.path.join(work_dir, "batch_*", "seq_des_outputs.csv"))
        data = pd.concat([pd.read_csv(csv) for csv in csvs])
        data.reset_index(drop=True, inplace=True)


        if not return_seq_threaded_pdbs_as_pose:
            os.makedirs(fasta_dir := os.path.join(work_dir, "fasta"), exist_ok=True)
            data["location"] = data.apply(
                lambda row: write_fasta(
                    seq=row["seq"],
                    name=description_from_path(row["out_pdb"]),
                    path=os.path.join(fasta_dir, f"{description_from_path(row['out_pdb'])}.fasta")),
                axis=1)

        if cif_to_pdb:
            os.makedirs(pdb_dir := os.path.join(work_dir, "converted"), exist_ok=True)
            data["temp_out"] = data["out_pdb"]
            data["location" if return_seq_threaded_pdbs_as_pose else "out_pdb"] = data.apply(
                lambda row: convert_cif_to_pdb(
                    input_cif=row["temp_out"],
                    output_format="pdb",
                    output=os.path.join(pdb_dir, f"{description_from_path(row['out_pdb'])}.pdb")),
                axis=1)

            data.drop(["temp_out"], axis=1, inplace=True)

        else:
            data["location"] = data.apply(lambda row: os.path.abspath(row["out_pdb"]), axis=1)
            data.drop(["out_pdb"], axis=1, inplace=True)

        data["description"] = data.apply(lambda row: description_from_path(row['location']), axis=1)

    elif mode == "ens_gen":

        records = []
        # gather all pdbs
        for path in Path(work_dir).rglob("batch_*/*/*/*.pdb"):
            ens_dir = path.parent
            input_description = description_from_path(str(ens_dir))

            # Remove "sample_" prefix
            new_name = path.name.replace("sample_", "", 1)
            new_path = path.with_name(new_name)
            path.rename(new_path)
            path = new_path

            # exclude input structure
            primary_conf = f"{input_description}.pdb"
            if path.name != primary_conf:
                records.append({
                    "location": str(path.absolute()),
                    "input_description": input_description,
                    "input_path": os.path.join(ens_dir, primary_conf),
                    "conformer_dir": str(path.parent),
                    "description": description_from_path(str(path))
                })

        data = pd.DataFrame(records)

    return data

