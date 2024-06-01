"""
ColabFold Module
================

This module provides functionality to integrate ColabFold within the ProtFlow framework, enabling the execution of AlphaFold2 runs on ColabFold. It includes tools to handle inputs, execute runs, and process outputs in a structured and automated manner.

Detailed Description
--------------------
The `ColabFold` class encapsulates all necessary functionalities to run AlphaFold2 through ColabFold. It manages the configuration of essential scripts and paths, sets up the environment, and handles the execution of prediction processes. The class also includes methods for collecting and processing output data, ensuring the results are organized and accessible for further analysis within the ProtFlow ecosystem.

This module streamlines the integration of ColabFold into larger computational workflows by supporting the automatic setup of job parameters, execution of ColabFold commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `ColabFold` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the prediction process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `ColabFold` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from colabfold import ColabFold

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = LocalJobStarter(max_cores=4)

    # Initialize the ColabFold class
    colabfold = ColabFold()

    # Run the prediction process
    results = colabfold.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        options="--msa-mode single-sequence",
        pose_options=None,
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- **Edge Cases:** The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the prediction process.
- **Customizability:** Users can customize the prediction process through multiple parameters, including the number of diffusions, specific options for the ColabFold script, and options for handling pose-specific parameters.
- **Integration:** The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate ColabFold into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Authors
------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""
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
    """
    ColabFold Class
    ===============

    The `ColabFold` class is a specialized class designed to facilitate the execution of AlphaFold2 within the ColabFold environment as part of the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with AlphaFold2 prediction processes.

    Detailed Description
    --------------------
    The `ColabFold` class manages all aspects of running AlphaFold2 predictions through ColabFold. It handles the configuration of necessary scripts and executables, prepares the environment for the prediction processes, and executes the prediction commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to ColabFold scripts and necessary directories.
        - Configuring job starter options, either automatically or manually.
        - Handling the execution of AlphaFold2 prediction commands with support for batch processing.
        - Collecting and processing output data into a pandas DataFrame.
        - Managing input FASTA files and preparing them for prediction.
        - Overwriting previous results if specified.

    Returns
    -------
    An instance of the `ColabFold` class, configured to run AlphaFold2 prediction processes and handle outputs efficiently.

    Raises
    ------
        - FileNotFoundError: If required files or directories are not found during the execution process.
        - ValueError: If invalid arguments are provided to the methods.
        - TypeError: If pose options are not of the expected type.

    Examples
    --------
    Here is an example of how to initialize and use the `ColabFold` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from colabfold import ColabFold

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the ColabFold class
        colabfold = ColabFold()

        # Run the prediction process
        results = colabfold.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            options="inference.num_designs=10",
            pose_options=["inference.input_pdb='input.pdb'"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as empty pose lists, the need to overwrite previous results, and the presence of existing score files.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the prediction process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The ColabFold class is intended for researchers and developers who need to perform AlphaFold2 predictions as part of their protein design and analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    def __init__(self, script_path: str = protflow.config.COLABFOLD_SCRIPT_PATH, jobstarter: str = None) -> None:
        """
        __init__ Method
        ===============

        The `__init__` method initializes an instance of the `ColabFold` class, setting up necessary configurations for running AlphaFold2 predictions through ColabFold within the ProtFlow framework.

        Detailed Description
        --------------------
        This method sets up the paths to the ColabFold script and initializes default values for various attributes required for running predictions. It also allows for the optional configuration of a job starter.

        Parameters:
            script_path (str, optional): The path to the ColabFold script. Defaults to `protflow.config.COLABFOLD_SCRIPT_PATH`.
            jobstarter (JobStarter, optional): An instance of the `JobStarter` class for managing job execution. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the `script_path` is not provided.

        Examples
        --------
        Here is an example of how to initialize the `ColabFold` class:

        .. code-block:: python

            from colabfold import ColabFold

            # Initialize the ColabFold class
            colabfold = ColabFold(script_path='/path/to/colabfold.py', jobstarter=jobstarter)
        """
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under COLABFOLD_DIR_PATH.")

        self.script_path = script_path
        self.name = "colabfold.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "colabfold.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str = None, pose_options: str = None, overwrite: bool = False, return_top_n_poses: int = 1) -> RunnerOutput:
        """
        run Method
        ==========

        The `run` method of the `ColabFold` class executes AlphaFold2 predictions using ColabFold within the ProtFlow framework. It manages the setup, execution, and result collection processes, providing a streamlined way to integrate AlphaFold2 predictions into larger computational workflows.

        Detailed Description
        --------------------
        This method orchestrates the entire prediction process, from preparing input data and configuring the environment to running the prediction commands and collecting the results. The method supports batch processing of input FASTA files and handles various edge cases, such as overwriting existing results and managing job starter options.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            options (str, optional): Additional options for the AlphaFold2 prediction commands. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters during prediction. Defaults to None.
            overwrite (bool, optional): If True, existing results will be overwritten. Defaults to False.
            return_top_n_poses (int, optional): The number of top poses to return based on the prediction scores. Defaults to 1.

        Returns:
            RunnerOutput: An object containing the results of the AlphaFold2 predictions, organized in a pandas DataFrame.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the methods.
            TypeError: If pose options are not of the expected type.

        Examples
        --------
        Here is an example of how to use the `run` method of the `ColabFold` class:

        .. code-block:: python

            from protflow.poses import Poses
            from protflow.jobstarters import JobStarter
            from colabfold import ColabFold

            # Create instances of necessary classes
            poses = Poses()
            jobstarter = JobStarter()

            # Initialize the ColabFold class
            colabfold = ColabFold()

            # Run the prediction process
            results = colabfold.run(
                poses=poses,
                prefix="experiment_1",
                jobstarter=jobstarter,
                options="inference.num_designs=10",
                pose_options=["inference.input_pdb='input.pdb'"],
                overwrite=True
            )

            # Access and process the results
            print(results)
        
        Further Details
        ---------------
            - **Batch Processing:** The method can handle large sets of input sequences by batching them into smaller groups, which helps in managing computational resources effectively.
            - **Overwrite Handling:** If `overwrite` is set to True, the method will clean up previous results, ensuring that the new predictions do not get mixed up with old data.
            - **Job Starter Configuration:** The method allows for flexible job management by accepting a `JobStarter` instance. If not provided, it uses the default job starter associated with the poses.
            - **Score Collection:** The method gathers the prediction scores and relevant data into a pandas DataFrame, facilitating easy analysis and integration with other ProtFlow components.
            - **Error Handling:** Robust error handling is incorporated to manage issues such as missing files or incorrect configurations, ensuring that the process can be debugged and verified efficiently.
        """
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"colabfold_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
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

        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()


    def prep_fastas_for_prediction(self, poses: list[str], fasta_dir: str, max_filenum: int) -> list[str]:
        """
        prep_fastas_for_prediction Method
        =================================

        The `prep_fastas_for_prediction` method prepares input FASTA files for AlphaFold2 predictions by splitting the input sequences into batches and writing them to files.

        Detailed Description
        --------------------
        This method divides the input protein sequences into batches, which can help in managing computational resources effectively. It writes the batches into FASTA files stored in the specified directory.

        Parameters:
            poses (list[str]): List of paths to input FASTA files.
            fasta_dir (str): Directory where the new FASTA files will be written.
            max_filenum (int): Maximum number of FASTA files to write.

        Returns:
            list[str]: List of paths to the prepared FASTA files.

        Raises:
            None

        Examples
        --------
        Here is an example of how to use the `prep_fastas_for_prediction` method:

        .. code-block:: python

            # Prepare input FASTA files for prediction
            fastas = colabfold.prep_fastas_for_prediction(poses=poses_list, fasta_dir='/path/to/fasta_dir', max_filenum=10)
        """
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
        """
        write_cmd Method
        ================

        The `write_cmd` method constructs the command string necessary to run the ColabFold script with the specified options and input files.

        Detailed Description
        --------------------
        This method generates the command string used to execute the ColabFold script. It incorporates various options and pose-specific parameters provided by the user.

        Parameters:
            pose_path (str): Path to the input FASTA file.
            output_dir (str): Directory where the prediction outputs will be stored.
            options (str, optional): Additional options for the ColabFold script. Defaults to None.
            pose_options (str, optional): Specific options for handling pose-related parameters. Defaults to None.

        Returns:
            str: The constructed command string.

        Raises:
            None

        Examples
        --------
        Here is an example of how to use the `write_cmd` method:

        .. code-block:: python

            # Write the command to run ColabFold
            cmd = colabfold.write_cmd(pose_path='/path/to/pose.fa', output_dir='/path/to/output_dir', options='--num_designs=10', pose_options='--input_pdb=input.pdb')
        """
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""

        return f"{self.script_path} {opts} {flags} {pose_path} {output_dir} "

    def collect_scores(self, work_dir: str, num_return_poses: int = 1) -> pd.DataFrame:
        """
        collect_scores Method
        =====================

        The `collect_scores` method collects and processes the prediction scores from the ColabFold output, organizing them into a pandas DataFrame for further analysis.

        Detailed Description
        --------------------
        This method gathers the prediction scores from the output files generated by ColabFold. It processes these scores and organizes them into a structured DataFrame, which includes various statistical measures.

        Parameters:
            work_dir (str): The working directory where the ColabFold outputs are stored.
            num_return_poses (int, optional): The number of top poses to return based on the prediction scores. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the collected and processed scores.

        Raises:
            FileNotFoundError: If no output files are found in the specified directory.

        Examples
        --------
        Here is an example of how to use the `collect_scores` method:

        .. code-block:: python

            # Collect and process the prediction scores
            scores = colabfold.collect_scores(work_dir='/path/to/work_dir', num_return_poses=5)     
        
        Further Details
        ---------------
            - **JSON and PDB File Parsing:** The method identifies and parses JSON and PDB files generated by ColabFold, extracting relevant score information from these files.
            - **Statistical Measures:** For each set of predictions, the method calculates various statistics, including mean pLDDT, max PAE, and PTM scores, organizing these measures into the DataFrame.
            - **Rank and Description:** The scores are ranked and annotated with descriptions to help identify the top poses based on prediction quality.
            - **File Handling:** The method includes robust file handling to ensure that only the relevant files are processed, and any existing files are correctly identified or overwritten as needed.
            - **Pose Location:** The final DataFrame includes the file paths to the predicted PDB files, facilitating easy access for further analysis or visualization.
        """

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
