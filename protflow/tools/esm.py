"""Module to run ESM on protein sequences."""
# imports
import os
import logging
from glob import glob

# dependencies
import pandas as pd
import numpy as np

# customs
from .. import config, runners
from ..jobstarters import split_list
from ..runners import Runner, RunnerOutput
from ..jobstarters import JobStarter
from ..poses import Poses

ESM_ALPHABET = [
    '<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 
    'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 
    'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>'
]

class ESM(Runner):
    '''ESM runner baseclass'''
    def __init__(
            self,
            python_path: str = config.ESM_PYTHON_PATH,
            pre_cmd: str = config.ESM_PRE_CMD,
            jobstarter: JobStarter = None
        ) -> None:
        '''ESM baseclass docstring.'''
        self.python = self.search_path(python_path, "ESM_PYTHON_PATH")
        self.pre_cmd = pre_cmd
        self.index_layers = 0
        self.jobstarter = jobstarter

    def __str__(self):
        return "esm.py"

    def run(
            self,
            poses: Poses,
            prefix: str,
            jobstarter: JobStarter = None,
            include: list[str] = None,
            model: str = "esm2_t33_650M_UR50D",
            options: str = None,
            overwrite: bool = False
        ) -> Poses:
        '''
        Documentation of esm.run() method.

        include : list[str]
            Supply list of scores that should be calculated and returned. Defaults to ["perres_entropy"].
            Options: {"mean", "per_tok", "bos", "contacts", "logits", "logprobs", "perres_probabilities", "perres_entropy", "mean_entropy"}
        '''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        logging.info(f"Running ESM in {work_dir} on {len(poses.df.index)} poses")
        scorefile = os.path.join(work_dir, f"{prefix}_esm_scores.json")
        os.makedirs((input_fasta_dir := os.path.join(work_dir, "input_fastas")), exist_ok=True)
        os.makedirs((esm_output_dir := os.path.join(work_dir, "esm_output")), exist_ok=True)

        # skip running esm if output exists
        if self.output_exists(scorefile) and not overwrite:
            scores = pd.read_json(scorefile)
            output_poses = RunnerOutput(
                poses,
                results=scores,
                prefix=prefix,
                index_layers=self.index_layers
            ).return_poses()
            return output_poses

        # check if poses are sequences (end with .fa or .fasta)
        if all(pose.endswith(".pdb") for pose in poses.poses_list()):
            poses.convert_pdb_to_fasta(prefix=prefix, update_poses=False, chain_sep=":")
            fastas = poses.df[f"{prefix}_fasta_location"]
        elif all(pose.endswith((".fa", ".fasta", ".fas")) for pose in poses.poses_list()):
            fastas = poses.poses_list()
        else:
            raise TypeError("Non-fasta file poses detected. To run ESM, your poses must be fasta files, ending with .fa, .fas, or .fasta. .pdb files are also allowed, but all files must be uniform (only .pdb or only .fa)")

        # batch input fastas
        n_jobs = max(jobstarter.max_cores, len(fastas))
        prediction_inputs = self.prep_fastas_for_prediction(fastas, input_fasta_dir, max_filenum=n_jobs)

        # parse options
        options = self._parse_options(options, include)

        # write commands
        cmds = self._write_cmds(prediction_inputs, options, model, esm_output_dir)

        # start jobs
        logging.info(f"Running esm on {len(poses)} sequences")
        jobstarter.start(
            cmds=cmds,
            jobname="esm",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores and return
        scores = collect_esm_scores(output_dir=esm_output_dir)

        # add location col (original poses location)
        location_mapping = dict(zip(poses.df["poses_description"].to_list(), poses.df["poses"].to_list()))
        scores["location"] = scores["description"].map(location_mapping)

        # save scores
        scores.to_json(scorefile)

        # return poses
        output_poses = RunnerOutput(
            poses,
            results=scores,
            prefix=prefix,
            index_layers=self.index_layers,
        ).return_poses()
        return output_poses

    def _parse_options(self, options: str, include: list[str]) -> dict:
        '''Internal method to parse options for running run_esm.py'''
        options = runners.parse_generic_options(options=options, pose_options=None, sep="--")[0]

        # handle include:
        if include:
            if "include" in options:
                options["include"] += include
            else:
                options["include"] = include
        if "include" in options:
            options["include"] = ",".join(options["include"])

        # safetycheck include parameter
        allowed_include_opts = {"mean", "per_tok", "bos", "contacts", "logits", "logprobs", "perres_probabilities", "perres_entropy", "mean_entropy"}
        if set(include) - allowed_include_opts:
            raise ValueError(f"Unsupported option for parameter 'include': {set(include) - allowed_include_opts}")
        return options

    def _write_cmds(self, prediction_inputs: list[str], options: dict, model: str, output_dir: str) -> list[str]:
        '''Write commands to run run_esm.py'''
        # setup options
        opts_str = runners.options_flags_to_string(options=options, flags=[], sep="--")

        # compile path to script
        script_path = os.path.join(config.AUXILIARY_RUNNER_SCRIPTS_DIR, "run_esm.py")
        return [f"{self.pre_cmd}; {self.python} {script_path} {opts_str} {model} {input_fa} {output_dir}" for input_fa in prediction_inputs]

    def output_exists(self, scorefilepath: str) -> bool:
        '''Simple check for collected scores of esm runner.'''
        return os.path.isfile(scorefilepath)

    def prep_fastas_for_prediction(self, fastas: list[str], fasta_dir: str, max_filenum: int) -> list[str]:
        """
        Prepare input FASTA files for ESMFold predictions.

        This method splits the input poses into the specified number of batches, prepares the FASTA files, and writes them to the specified directory for ESMFold predictions.

        Parameters:
            poses : Poses
                List of paths to FASTA files.
            fasta_dir : str
                Directory to which the new FASTA files should be written.
            max_filenum : int
                Maximum number of FASTA files to be written.

        Returns:
            list[str]
                List of paths to the prepared FASTA files.

        Examples:
            Here is an example of how to use the `parse_fastas_for_prediction` method:

            .. code-block:: python

                from esmfold import ESMFold

                # Initialize the ESMFold class
                esmfold = ESMFold()

                # Prepare FASTA files for prediction
                fasta_paths = esmfold.parse_fastas_for_prediction(
                    poses=["pose1.fa", "pose2.fa", "pose3.fa"],
                    fasta_dir="/path/to/fasta_dir",
                    max_filenum=2
                )

                # Access the prepared FASTA files
                print(fasta_paths)

        Further Details:
            - **Input Preparation:** The method merges and splits the input FASTA files into the specified number of batches. It ensures that the FASTA files are correctly formatted and written to the specified directory.
            - **Customization:** Users can specify the maximum number of FASTA files to be created, allowing for flexibility in managing input data for parallel processing.
            - **Output Management:** The method returns a list of paths to the newly created FASTA files, which are ready for ESMFold predictions.

        This method is designed to facilitate the preparation of input data for ESMFold, ensuring that the input FASTA files are organized and ready for processing.
        """
        def mergefastas(files: list[str], out_path: str, exchange_char: tuple[str,str] = None) -> str:
            '''
            Merges Fastas located in <files> into one single fasta-file called <path>
            '''
            fastas = []
            for fp in files:
                with open(fp, 'r', encoding="UTF-8") as f:
                    fastas.append(f.read().strip())

            if exchange_char:
                fastas = [x.replace(exchange_char[0], exchange_char[1]) for x in fastas]

            with open(out_path, 'w', encoding="UTF-8") as f:
                f.write("\n".join(fastas))

            return out_path

        # determine how to split the poses into <max_gpus> fasta files:
        splitnum = max(len(fastas), max_filenum)
        fastas_list = split_list(fastas, splitnum)

        # Write fasta files according to the fasta_split determined above and then return:
        return [mergefastas(files=fastas, out_path=f"{fasta_dir}/fasta_{str(i+1).zfill(4)}.fa", exchange_char=("/",":")) for i, fastas in enumerate(fastas_list)]

def collect_esm_scores(output_dir: str) -> pd.DataFrame:
    """Function to collect ESM scores from esm run.py output folder."""
    # collect all output files
    raw_scorefiles = glob(f"{output_dir}/*.npy")
    if not raw_scorefiles:
        raise FileNotFoundError(f"No output files found of run_esm.py at directory {output_dir}\nEither run_esm.py crashed or the directory is wrong. Check any output logs of run_esm.py in and around the directory {output_dir}")

    # collect scores
    ser_list = []
    for raw_scorefile in raw_scorefiles:
        # load file
        raw_scores = np.load(raw_scorefile, allow_pickle=True).item()

        # collect metadata
        scores_dict = {
            "description": raw_scores['label'],
            "scores_location": raw_scorefile,
        }

        # collect scores
        for label in list(raw_scores.keys()):
            if label == "description":
                continue
            scores_dict[label] = raw_scores[label]

        # convert to Series and append to aggregation list
        ser_list.append(pd.Series(scores_dict))

    # merge into singular dataframe
    out_df = pd.DataFrame(ser_list)
    print(out_df.head())
    print(out_df.columns)
    return out_df
