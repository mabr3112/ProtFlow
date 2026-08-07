"""
MMseqs Module
==============

This module provides functionality to integrate batched MMseqs2 MSA generation within the ProtFlow framework. It searches multiple sequences simultaneously against a custom MMseqs database to drastically reduce I/O overhead and leverage SIMD vectorization.

Classes
-------
MMseqs
    A class designed to facilitate the execution of batched MMseqs2 searches within the ProtFlow framework. It extends the `Runner` class and manages the setup, execution, and data collection of alignments.

Functions
---------
collect_scores(output_dir: str) -> pd.DataFrame
    Collects the A3M MSA outputs from a specified directory, returning a pandas DataFrame with the structured results containing the paths to the alignments.

Notes
-----
This module is part of the ProtFlow package and relies on the Sphinx Napoleon extension to parse NumPy-style docstrings into Read the Docs formatting.
"""
# import general
import os
import glob
import shutil

# import dependencies
import pandas as pd

# import customs
from .. import jobstarters, require_config, load_config_path
from ..poses import Poses, description_from_path
from ..runners import Runner, RunnerOutput
from ..jobstarters import JobStarter


class MMseqs(Runner):
    """
    MMseqs Class
    ============

    The `MMseqs` class is a specialized class within the ProtFlow framework, designed to facilitate the rapid execution of batched MMseqs2 MSA generation. This class extends the `Runner` class, inheriting its core functionality while adding specific methods to handle batched sequence concatenation, MMseqs execution, and MSA extraction.

    Attributes
    ----------
    jobstarter : JobStarter, optional
        A `JobStarter` instance to manage job execution. If not provided, the default job starter from `Poses` will be used.
    name : str
        The name of the MMseqs runner, set to "mmseqs.py".
    index_layers : int
        Tracks the indexing of layers, default is 0.
    application : str
        Path to the MMseqs2 executable. Automatically checked for validity during initialization.

    Methods
    -------
    __str__()
        Returns a string representation of the MMseqs instance.
    _check_install(application_path: str) -> str
        Verifies the installation of MMseqs2 by checking if the executable exists at the provided path.
    run(poses: Poses, prefix: str, target_db: str, overwrite: bool = False, jobstarter: JobStarter = None) -> Poses
        Executes the batched MMseqs searches on the provided poses against a target database.
    create_batch_fasta(fasta_paths: list[str], out_path: str) -> None
        Concatenates multiple single-sequence FASTA files into one multi-sequence FASTA for batched processing.
    write_cmd(batch_fasta: str, target_db: str, tmp_dir: str) -> str
        Constructs the command line instruction to run MMseqs on a given batch FASTA.
    """

    def __init__(self, jobstarter: JobStarter = None, application: str | None = None):
        """
        Initializes the MMseqs runner.

        Parameters
        ----------
        jobstarter : JobStarter, optional
            An instance of the JobStarter class used to manage job submissions. 
        application : str, optional
            The path to the MMseqs2 executable. If not provided, it is loaded from the ProtFlow configuration.
        """
        # setup config
        self.application = self._check_install(application or load_config_path(require_config(), "MMSEQS_PATH"))

        # setup runner
        self.jobstarter = jobstarter
        self.name = "mmseqs.py"
        self.index_layers = 0

    def __str__(self):
        return "MMseqs"

    def _check_install(self, application_path) -> str:
        """checks if MMseqs2 is installed in the environment"""
        if not shutil.which(application_path) and not os.path.isfile(application_path):
            raise ValueError(
                f"Could not find executable for MMseqs2 at {application_path}. "
                f"Did you set it up in your protflow environment? Provide the path to the application "
                f"with the :application: parameter when initializing an MMseqs() runner instance.")
        return application_path

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, target_db: str, overwrite: bool = False, # pylint: disable=W0237
            jobstarter: JobStarter = None) -> Poses:
        """
        Execute the batched MMseqs2 calculation on the given protein poses.

        This method manages the entire MMseqs execution process. It evaluates the input poses, 
        converts them to FASTA if necessary, batches them according to the maximum cores of the 
        JobStarter, executes the search, and extracts the resulting A3M alignments into individual files.

        Parameters
        ----------
        poses : Poses
            A `Poses` object containing the protein structures or sequences to be aligned.
        prefix : str
            A prefix for naming the output files generated during the run.
        target_db : str
            The path to the pre-compiled MMseqs target database (e.g. `ache_db_rep`).
        overwrite : bool, optional
            If True, existing output files will be overwritten. Default is False.
        jobstarter : JobStarter, optional
            A `JobStarter` instance for managing parallel execution of commands. If not provided, 
            the default job starter from `Poses` will be used.

        Returns
        -------
        Poses
            A `Poses` object containing the original poses with paths to the new A3M files in the DataFrame.
            The structural `location` paths remain unaltered to maintain compatibility with downstream runners.
        """
        # setup runner and files
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        scorefile = os.path.join(work_dir, f"{prefix}_MMseqs.{poses.storage_format}")

        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
            return output

        # Determine pose formats and convert to isolated fastas using native ProtFlow tools if needed
        pose_types = poses.determine_pose_type()
        
        if pose_types == ['.pdb']:
            poses.convert_pdb_to_fasta(prefix=prefix, update_poses=False, overwrite=overwrite)
            fasta_col = f"{prefix}_fasta_location"
        elif all(ext in ['.fa', '.fasta', '.fas'] for ext in pose_types):
            fasta_col = "poses"
        else:
            raise ValueError(f"Unsupported or mixed input types for MMseqs: {pose_types}. Expected exclusively .pdb or .fasta")

        fasta_list = poses.df[fasta_col].to_list()
        
        # Determine number of batches based on available cores vs total sequences
        num_batches = min(jobstarter.max_cores, len(fasta_list))
        fasta_sublists = jobstarters.split_list(fasta_list, n_sublists=num_batches)

        batch_dir = os.path.join(work_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)

        # Compile commands for each batch
        cmds = []
        for i, sublist in enumerate(fasta_sublists):
            batch_name = f"batch_{i+1}"
            batch_fasta = os.path.join(batch_dir, f"{batch_name}.fasta")
            
            # create temporary directory for this batch natively in python
            tmp_dir = os.path.join(work_dir, f"{batch_name}_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Concatenate fastas into a single batched query file
            self.create_batch_fasta(fasta_paths=sublist, out_path=batch_fasta)
            
            # Add command pipeline for this batch
            cmds.append(self.write_cmd(batch_fasta=batch_fasta, target_db=target_db, tmp_dir=tmp_dir))

        # Run batched commands in parallel
        jobstarter.start(
            cmds=cmds,
            jobname="MMseqs",
            output_path=work_dir
        )

        # collect_scores natively disentangles the tmp_dirs using python
        scores = collect_scores(work_dir=work_dir)
        
        # Merge scores back into poses df to grab the ORIGINAL poses paths
        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description",
                              right_on="poses_description").drop('poses_description', axis=1)

        # Rename 'poses' to 'location' to satisfy RunnerOutput.
        # Because we are feeding the original pose paths into 'location', 
        # ProtFlow will retain the original .pdb files as the primary poses 
        # for downstream tools, while appending the new 'a3m_path' column!
        scores.rename(columns={"poses": "location"}, inplace=True)

        # Write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # Create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
        return output

    def create_batch_fasta(self, fasta_paths: list[str], out_path: str) -> None:
        """
        Concatenates multiple single-sequence FASTA files into one multi-sequence FASTA.

        Parameters
        ----------
        fasta_paths : list[str]
            A list of file paths to the individual input FASTA files.
        out_path : str
            The output path for the concatenated multi-sequence FASTA file.

        Returns
        -------
        None
        """
        with open(out_path, 'w', encoding="utf-8") as outfile:
            for fasta_file in fasta_paths:
                with open(fasta_file, 'r', encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)
                    if not content.endswith("\n"):
                        outfile.write("\n")

    def write_cmd(self, batch_fasta: str, target_db: str, tmp_dir: str) -> str:
            """
            Generate the command line string to run MMseqs on a batch FASTA.

            Constructs a chained bash command that creates a temporary MMseqs database for 
            the batch, searches the target database, converts the alignments to an A3M 
            database, and unpacks them. (Folder creation and cleanup are handled in Python).
            """
            run_string = (
                f"{self.application} createdb {batch_fasta} {tmp_dir}/qdb && "
                f"{self.application} search {tmp_dir}/qdb {target_db} {tmp_dir}/res {tmp_dir}/tmp && "
                f"{self.application} result2msa {tmp_dir}/qdb {target_db} {tmp_dir}/res {tmp_dir}/msa --msa-format-mode 2 && "
                f"cp {tmp_dir}/qdb.lookup {tmp_dir}/msa.lookup && "
                f"{self.application} unpackdb {tmp_dir}/msa {tmp_dir}/unpack"
            )

            return run_string


def collect_scores(work_dir: str) -> pd.DataFrame:
    """
    Collect and disentangle A3M output files from the specified directory.

    This function scans for temporary batch directories created by the JobStarter. 
    It parses the internal MMseqs `qdb.lookup` mapping to identify the output files, 
    renames them to match their original FASTA headers, and moves them to the 
    root `output_dir`. It then cleans up the temporary directories and compiles the 
    paths into a DataFrame.
    """

    os.makedirs(a3m_dir := os.path.join(work_dir, "a3m_files"), exist_ok=True)

    # 1. Process any temporary batch directories to disentangle the A3Ms in Python
    tmp_dirs = glob.glob(os.path.join(work_dir, "*_tmp"))
    
    for tmp_dir in tmp_dirs:
        lookup_file = os.path.join(tmp_dir, "qdb.lookup")
        unpack_dir = os.path.join(tmp_dir, "unpack")
        
        if os.path.isfile(lookup_file) and os.path.isdir(unpack_dir):
            with open(lookup_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    # qdb.lookup format: numeric_id \t fasta_header \t file_offset
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        num_id = parts[0].strip()
                        description = parts[1].strip()
                        
                        # Fallback check depending on whether the local MMseqs 
                        # version extracts by numerical ID or string description
                        unpacked_num = os.path.join(unpack_dir, num_id)
                        unpacked_desc = os.path.join(unpack_dir, description)
                        final_a3m = os.path.join(a3m_dir, f"{description}.a3m")
                        
                        # Move and rename the flat file if it was successfully generated
                        if os.path.isfile(unpacked_num):
                            shutil.move(unpacked_num, final_a3m)
                        elif os.path.isfile(unpacked_desc):
                            shutil.move(unpacked_desc, final_a3m)
                            
        # Clean up the temporary batch directory using Python
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 2. Collect all A3M files now present in the output directory
    a3m_files = glob.glob(os.path.join(a3m_dir, "*.a3m"))

    results = []
    for file_path in a3m_files:
        results.append({
            "description": description_from_path(file_path),
            "a3m_path": file_path
        })

    scores = pd.DataFrame(results)
    
    # Ensure empty df behaves gracefully
    if scores.empty:
        return pd.DataFrame(columns=["description", "a3m_path"])

    return scores