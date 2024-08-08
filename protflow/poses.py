"""
poses Module
============

This module provides functionalities for handling and manipulating protein data within the ProtFlow framework. It focuses on managing protein data represented as Pandas DataFrames, allowing for efficient parsing, storage, and manipulation of protein data across various file formats. The module facilitates complex protein study workflows and integrates seamlessly with other components of the ProtFlow package.

Detailed Description
--------------------
The `poses` module offers a robust class, `Poses`, designed to encapsulate the functionality necessary to manage protein data. It supports various operations such as setting up work directories, parsing protein data, and integrating outputs from different computational processes. The module ensures that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

Key Features
------------
- **Parsing Protein Data**: Supports reading protein data from various file formats like JSON, CSV, Pickle, Feather, and Parquet.
- **Data Storage and Retrieval**: Allows storing and retrieving protein data in multiple formats, facilitating easy data management.
- **Integration with ProtFlow**: Seamlessly integrates with ProtFlow's job management components, enhancing its utility in distributed computing environments.
- **Advanced Data Manipulation**: Provides functionalities to merge and prefix data from various sources, making it easier to handle complex datasets.
- **Flexible and Customizable**: Users can customize the data handling processes through various parameters, enabling tailored data management solutions.

Usage
-----
To use this module, create an instance of the `Poses` class and utilize its methods to manage protein data. Here is an example demonstrating its usage within a ProtFlow pipeline:

.. code-block:: python

    from poses import Poses

    # Initialize the Poses class with protein data and a working directory
    poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')
    
    # Further operations using poses_instance
    poses_instance.save_scores('path/to/save/scores')
    poses_instance.filter_poses_by_rank(n=10, score_col='score', prefix='filtered_poses')

Examples
--------
Here is an example of how to initialize and use the `Poses` class for managing protein data:

.. code-block:: python

    from poses import Poses

    # Create an instance of the Poses class
    poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')

    # Perform various operations using the instance
    poses_instance.set_work_dir('new/work/dir')
    poses_instance.save_scores('path/to/save/scores', out_format='csv')
    filtered_poses = poses_instance.filter_poses_by_value(score_col='score', value=0.5, operator='>')

Further Details
---------------
    - **Edge Cases**: The module handles various edge cases such as empty pose lists and the need to overwrite previous results. It includes robust error handling and logging for easier debugging and verification.
    - **Customizability**: Users can customize the data handling process through multiple parameters, including storage formats, pose-specific parameters, and job management settings.
    - **Integration**: The module integrates seamlessly with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to manage protein data within their computational workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Authors
-------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""

import os
from glob import glob
import re
from typing import Union
import shutil
import logging

# dependencies
import pandas as pd
import Bio.PDB

# customs
from protflow import jobstarters
from protflow.jobstarters import JobStarter
from protflow.residues import ResidueSelection
from protflow.utils.utils import parse_fasta_to_dict
from protflow.utils.biopython_tools import load_structure_from_pdbfile, get_sequence_from_pose
import protflow.utils.plotting as plots

FORMAT_STORAGE_DICT = {
    "json": "to_json",
    "csv": "to_csv",
    "pickle": "to_pickle",
    "feather": "to_feather",
    "parquet": "to_parquet"
}

class Poses:
    """
    Poses Class
    ===========

    The `Poses` class within the ProtFlow package is designed for handling protein data, enabling the parsing, storage, and manipulation of protein data represented as Pandas DataFrames. This class facilitates the management of complex protein study workflows and integrates seamlessly with other components of the ProtFlow framework.

    Detailed Description
    --------------------
    The `Poses` class encapsulates the functionality necessary for comprehensive management of protein data. It supports various operations, including setting up work directories, parsing protein data from different sources, integrating outputs from different runners, and handling protein data in multiple file formats. This class is essential for users looking to streamline their protein data management within computational workflows.

    Key Features
    ------------
    - **Work Directory Setup**: Easily sets up and manages work directories for storing intermediate and final results.
    - **Data Parsing**: Parses protein data from various sources and formats, including JSON, CSV, Pickle, Feather, and Parquet.
    - **Data Storage and Retrieval**: Stores and retrieves protein data in multiple file formats, ensuring flexibility in data management.
    - **Job Management Integration**: Integrates with ProtFlow's job management components, facilitating the handling of protein data in distributed computing environments.
    - **Advanced Data Manipulation**: Supports operations like merging, prefixing, and duplicating data, providing robust data manipulation capabilities.
    - **Filtering and Scoring**: Offers methods to filter protein data based on various criteria and calculate composite scores for better data analysis.
    - **Pose Handling**: Manages protein poses, including loading, saving, and converting between different formats (e.g., PDB to FASTA).

    Usage
    -----
    To use this class, create an instance of the `Poses` class and utilize its methods to manage protein data. Here is an example demonstrating its usage within a ProtFlow pipeline:

    .. code-block:: python

        from poses import Poses

        # Initialize the Poses class with protein data and a working directory
        poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')

        # Set up the work directory
        poses_instance.set_work_dir('path/to/new_work_dir')

        # Parse and manipulate poses
        poses_instance.set_poses(poses=my_protein_data)
        poses_instance.save_scores('path/to/save/scores', out_format='csv')

        # Filter poses
        filtered_poses = poses_instance.filter_poses_by_rank(n=10, score_col='score', prefix='filtered_poses')

        # Calculate a composite score
        poses_instance.calculate_composite_score(name='composite_score', scoreterms=['score1', 'score2'], weights=[0.5, 0.5], plot=True)

    Further Details
    ---------------
        - **Edge Cases**: The class handles various edge cases, such as empty pose lists, the need to overwrite previous results, and handling multiline FASTA inputs.
        - **Customizability**: Users can customize the data handling process through multiple parameters, including storage formats, pose-specific parameters, and job management settings.
        - **Integration**: The class integrates seamlessly with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.
        - **Error Handling**: Includes robust error handling and logging for easier debugging and verification of data processing steps.

    Attributes
    ----------
    - `df` : pd.DataFrame
        A DataFrame to store protein data.
    - `work_dir` : str
        The working directory for storing data and results.
    - `storage_format` : str
        The format for storing protein data (e.g., 'json', 'csv').
    - `default_jobstarter` : JobStarter
        The default job starter for managing jobs.

    Notes
    -----
    This class is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

    Author
    ------
    Markus Braun, Adrian Tripp

    Version
    -------
    0.1.0
    """
    ############################################# SETUP #########################################
    def __init__(self, poses: list = None, work_dir: str = None, storage_format: str = "json", glob_suffix: str = None, jobstarter: JobStarter = jobstarters.SbatchArrayJobstarter()):
        """
        Initializes the Poses class with optional parameters for poses, working directory, storage format, glob suffix, and job starter.

        Parameters
        ----------
        poses : list, optional
            A list of paths to the protein data files to be managed. If not provided, an empty DataFrame is initialized.
        work_dir : str, optional
            The working directory where intermediate and final results will be stored. If not provided, the current directory is used.
        storage_format : str, optional
            The format used for storing protein data (default is 'json'). Supported formats include 'json', 'csv', 'pickle', 'feather', and 'parquet'.
        glob_suffix : str, optional
            A suffix used for globbing multiple files. This allows for batch processing of files matching the given pattern.
        jobstarter : JobStarter, optional
            An instance of the JobStarter class used to manage job submissions. The default is an instance of SbatchArrayJobstarter from the jobstarters module.

        Attributes
        ----------
        df : pd.DataFrame
            A DataFrame to store protein data.
        work_dir : str
            The working directory for storing data and results.
        storage_format : str
            The format for storing protein data.
        default_jobstarter : JobStarter
            The default job starter for managing jobs.

        Notes
        -----
        This method initializes the Poses class and sets up various attributes required for managing protein data. It prepares the environment for subsequent data manipulation and analysis operations.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with protein data and a working directory
            poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')

        """
        self.df = None
        self.set_work_dir(work_dir, set_scorefile=False)
        self.set_poses(poses, glob_suffix=glob_suffix)

        # setup poses.storage_format and poses.scorefile.
        self.set_storage_format(storage_format)
        self.set_scorefile(self.work_dir)

        # setup jobstarter
        self.default_jobstarter = jobstarter

        # set other empty attributes
        self.motifs = []

    def __iter__(self):
        for _, row in self.df.iterrows():
            yield row

    def __len__(self):
        return len(self.df)

    ############################################# SETUP METHODS #########################################
    def set_scorefile(self, work_dir: str) -> None:
        """
        Sets the scorefile path for storing protein scores.

        Parameters
        ----------
        work_dir : str
            The working directory where the scorefile will be stored. If the work directory is not set, the scorefile is stored in the current directory.

        Attributes
        ----------
        scorefile : str
            The path to the scorefile where protein scores are stored.

        Notes
        -----
        This method configures the path for the scorefile based on the provided working directory. If no working directory is specified, the scorefile is stored in the current directory.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Set the scorefile path
            poses_instance.set_scorefile(work_dir='path/to/work_dir')

        """
        # if no work_dir is set, store scores in current directory.
        scorefile_path = os.path.join(work_dir, os.path.basename(work_dir)) if work_dir else "./poses"
        self.scorefile = f"{scorefile_path}_scores.{self.storage_format}"

    def set_storage_format(self, storage_format: str) -> None:
        """
        Sets the storage format for storing protein data.

        Parameters
        ----------
        storage_format : str
            The format used for storing protein data. Supported formats include 'json', 'csv', 'pickle', 'feather', and 'parquet'.

        Raises
        ------
        KeyError
            If the provided storage format is not supported.

        Attributes
        ----------
        storage_format : str
            The format for storing protein data.

        Notes
        -----
        This method configures the storage format for protein data. It ensures that the format is one of the supported formats and raises an error if the format is invalid.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Set the storage format to 'csv'
            poses_instance.set_storage_format('csv')

        """
        if storage_format.lower() not in FORMAT_STORAGE_DICT:
            raise KeyError(f"Format {storage_format} not available. Format must be on of {[list(FORMAT_STORAGE_DICT)]}")
        self.storage_format = storage_format # removed .lower() maybe there is a storage format that needs caps letters.

    def set_work_dir(self, work_dir: str, set_scorefile: bool = True) -> None:
        """
        Sets up and configures the working directory for storing data and results.

        Parameters
        ----------
        work_dir : str
            The working directory where data and results will be stored. If the directory does not exist, it will be created.
        set_scorefile : bool, optional
            If True, also sets the path for the scorefile in the specified working directory (default is True).

        Attributes
        ----------
        work_dir : str
            The working directory for storing data and results.
        scores_dir : str
            Directory for storing score files.
        filter_dir : str
            Directory for storing filtered results.
        plots_dir : str
            Directory for storing plot files.

        Further Details
        ---------------
        This method creates the necessary subdirectories within the specified working directory to organize score files, filter results, and plots. It ensures that the required directory structure is in place for subsequent data management operations.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Set the working directory
            poses_instance.set_work_dir('path/to/new_work_dir')

        Notes
        -----
        - The method will log the creation of directories if they do not already exist.
        - If `set_scorefile` is set to True, the scorefile path will be configured within the working directory.

        """
        def set_dir(dir_name: str, work_dir: str) -> str:
            '''Creates a directory inside of work_dir that has the name {dir_name}_dir. 
            Also sets an attribute self.{dir_name} that points to the directory.'''
            if work_dir is None:
                return None
            dir_ = os.path.join(work_dir, dir_name)
            os.makedirs(dir_, exist_ok=True)
            return dir_

        # setup and create work_dir if it does not already exist
        if work_dir:
            work_dir = os.path.abspath(work_dir)
            os.makedirs(work_dir, exist_ok=True)
            logging.info(f"Creating directory {os.path.abspath(work_dir)}")
            self.work_dir = os.path.abspath(work_dir)
        else:
            self.work_dir = None

        # setup common directories for workflows:
        self.scores_dir = set_dir("scores", work_dir)
        self.filter_dir = set_dir("filter", work_dir)
        self.plots_dir = set_dir("plots", work_dir)

        # setup scorefile if option is provided (default: True)
        if set_scorefile:
            self.set_scorefile(work_dir)

    def set_logger(self) -> None:
        """
        Configures the logger for the Poses class.

        Further Details
        ---------------
        This method sets up the logging configuration for the Poses class. It creates a logger that outputs log messages to both the console and a log file in the working directory (if set). This aids in debugging and tracking the progress of data processing operations.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses(work_dir='path/to/work_dir')

            # Set up the logger
            poses_instance.set_logger()

        Notes
        -----
        - The log file is named after the working directory and stored within it.
        - The logging level is set to INFO, and log messages include timestamps, logger names, log levels, and messages.

        """
        # Create a logger
        if self.work_dir:
            logfile_path = os.path.join(self.work_dir, f"{os.path.basename(self.work_dir)}.log")
        else:
            logfile_path = None

        # Configure the basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

        if logfile_path:
            logging.getLogger().addHandler(logging.FileHandler(logfile_path))

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """
        Configures the job starter for managing job submissions.

        Parameters
        ----------
        jobstarter : JobStarter
            An instance of the JobStarter class used to manage job submissions.

        Attributes
        ----------
        default_jobstarter : JobStarter
            The default job starter for managing jobs.

        Further Details
        ---------------
        This method sets the job starter for the Poses class, which is used to manage job submissions in distributed computing environments. It allows the user to specify a custom job starter for handling computational tasks.

        Example
        -------
        .. code-block:: python

            from poses import Poses
            from protflow.jobstarters import CustomJobStarter

            # Initialize the Poses class
            poses_instance = Poses()

            # Set a custom job starter
            custom_jobstarter = CustomJobStarter()
            poses_instance.set_jobstarter(custom_jobstarter)

        Notes
        -----
        - The job starter must be an instance of the JobStarter class or a subclass thereof.
        - This method enables customization of job management to suit specific computational workflows.
        """

        self.default_jobstarter = jobstarter

    def change_poses_dir(self, poses_dir: str, copy: bool = False, overwrite: bool = False) -> "Poses":
        """
        Changes the directory of the stored poses, with options to copy or overwrite existing poses.

        Parameters
        ----------
        poses_dir : str
            The new directory where the poses will be located.
        copy : bool, optional
            If True, the poses will be copied to the new directory (default is False).
        overwrite : bool, optional
            If True, existing files in the new directory will be overwritten (default is False).

        Returns
        -------
        Poses
            The updated Poses instance with poses located in the new directory.

        Further Details
        ---------------
        This method updates the paths of the stored poses to a new directory. If the `copy` parameter is set to True, the poses are copied to the new directory. The `overwrite` parameter controls whether existing files in the new directory are overwritten.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses(poses=my_protein_data, work_dir='path/to/work_dir')

            # Change the directory of the poses
            poses_instance.change_poses_dir('path/to/new_poses_dir', copy=True, overwrite=True)

        Notes
        -----
        - If `copy` is set to False, the method only updates the paths in the DataFrame without moving the files.
        - Raises a ValueError if the new directory does not exist or if the poses do not exist in the specified directory (when `copy` is False).
        - Ensures the integrity of the poses by verifying their existence in the new directory.

        """
        # define new poses:
        new_poses = [os.path.join(poses_dir, os.path.basename(pose)) for pose in self.poses_list()]

        # exchange with check if work_dir is a directory and the poses exist
        if not copy:
            # just change the name of the directory in the poses_df, don't copy the poses anywhere
            if not os.path.isdir(poses_dir):
                raise ValueError(f":work_dir: has to be existing directory!")
            if not all((os.path.isfile(pose) for pose in new_poses)):
                raise ValueError(f"Poses do not exist at specified directory. If you want to copy the poses there, set the parameter :copy: to True!")

        else:
            # actually copy the poses to a new directory (for whatever reason)
            if not os.path.isdir(poses_dir):
                os.makedirs(poses_dir)
            if overwrite:
                for old_path, new_path in zip(self.poses_list(), new_poses):
                    shutil.copy(old_path, new_path)
            else:
                # if overwrite is False, check if the file exists first. This should save read/write speed.
                for old_path, new_path in zip(self.poses_list(), new_poses):
                    if not os.path.isfile(new_path):
                        shutil.copy(old_path, new_path)

        # change path in self.df["poses"] column
        self.df["poses"] = new_poses
        return self

    def parse_poses(self, poses: Union[list,str] = None, glob_suffix: str = None) -> list:
        """
        Parses the input poses, which can be provided as a list or a directory with a glob suffix.

        Parameters
        ----------
        poses : Union[list, str], optional
            A list of file paths or a directory containing the protein data files. If not provided, an empty list is returned.
        glob_suffix : str, optional
            A suffix used for globbing multiple files in the specified directory.

        Returns
        -------
        list
            A list of parsed pose file paths.

        Further Details
        ---------------
        This method handles various input types for parsing poses. It can parse a list of file paths directly or glob files in a specified directory using a suffix. The method ensures that all specified files exist and raises appropriate errors if they do not.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Parse poses from a directory with a glob suffix
            parsed_poses = poses_instance.parse_poses(poses='path/to/pose_dir', glob_suffix='*.pdb')

        Notes
        -----
        - Raises FileNotFoundError if any specified files do not exist.
        - Supports both single file and multiple file (via globbing) inputs.
        - Ensures that the returned list contains valid file paths.

        """
        if isinstance(poses, str) and glob_suffix:
            parsed_poses = glob(f"{poses}/{glob_suffix}")
            if not parsed_poses:
                raise FileNotFoundError(f"No {glob_suffix} files were found in {poses}. Did you mean to glob? Was the path correct?")
            return parsed_poses
        if isinstance(poses, str) and not glob_suffix:
            if not os.path.isfile(poses):
                raise FileNotFoundError(f"File {poses} not found!")
            return [poses]
        if isinstance(poses, list):
            if not all((os.path.isfile(path) for path in poses)):
                raise FileNotFoundError(f"Not all files listed in poses were found.")
            return poses
        if poses is None:
            return []
        raise TypeError(f"Unrecognized input type {type(poses)} for function parse_poses(). Allowed types: [list, str]")

    def parse_descriptions(self, poses: list = None) -> list:
        """
        Parses descriptions from the provided pose file paths.

        Parameters
        ----------
        poses : list, optional
            A list of pose file paths from which descriptions are extracted.

        Returns
        -------
        list
            A list of descriptions parsed from the pose file paths.

        Further Details
        ---------------
        This method extracts descriptions from the provided list of pose file paths. Descriptions are derived from the file names by stripping the directory path and file extension.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Parse descriptions from pose file paths
            descriptions = poses_instance.parse_descriptions(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

        Notes
        -----
        - This method is useful for generating a list of concise descriptions based on file names.
        - Ensures that descriptions are derived in a consistent format, suitable for use in data management and analysis.

        """
        return [description_from_path(pose) for pose in poses]

    def set_poses(self, poses: Union[list,str,pd.DataFrame] = None, glob_suffix: str = None) -> None:
        """
        Sets the poses for the Poses instance, parsing the input if necessary.

        Parameters
        ----------
        poses : Union[list, str, pd.DataFrame], optional
            A list of file paths, a directory containing the protein data files, or a DataFrame containing the poses. If not provided, an empty DataFrame is initialized.
        glob_suffix : str, optional
            A suffix used for globbing multiple files in the specified directory.

        Further Details
        ---------------
        This method initializes the poses for the Poses instance. It can accept various input types, including a list of file paths, a directory for globbing files, or a DataFrame. The method ensures that the poses are correctly parsed and set up for further processing.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Set poses from a directory with a glob suffix
            poses_instance.set_poses(poses='path/to/pose_dir', glob_suffix='*.pdb')

            # Set poses from a list of file paths
            poses_instance.set_poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

        Notes
        -----
        - If a DataFrame is provided, it is directly used as the poses DataFrame after integrity checks.
        - The method supports parsing multiline FASTA inputs and handles them appropriately.
        - Ensures that the poses DataFrame contains necessary columns for subsequent operations.

        """
        # if DataFrame is passed, load directly.
        if isinstance(poses, pd.DataFrame):
            self.df = self.check_poses_df_integrity(poses)
            return None

        if isinstance(poses, str) and any([poses.endswith(ext) for ext in ['csv', 'json', 'parquet', 'pickle', 'feather']]):
            self.df = get_format(poses)(poses)
            # importing .csv files results in the index column being read in as Unnamed: 0, it can be dropped
            if 'Unnamed: 0' in self.df.columns: self.df.drop('Unnamed: 0', axis=1, inplace=True)
            self.df = self.check_poses_df_integrity(self.df)
            return None

        # if Poses are initialized freshly (with input poses as strings:)
        poses = self.parse_poses(poses, glob_suffix=glob_suffix)

        # handle multiline .fa inputs for poses!
        for pose in poses:
            if not pose.endswith(".fa") and not pose.endswith(".fasta"):
                continue
            if len(parse_fasta_to_dict(pose)) > 1:
                poses.remove(pose)
                poses += self.split_multiline_fasta(pose)

        self.df = pd.DataFrame({"input_poses": poses, "poses": poses, "poses_description": self.parse_descriptions(poses)})
        return None

    def check_prefix(self, prefix: str) -> None:
        """
        Checks if the given prefix is already used in the poses DataFrame.

        Parameters
        ----------
        prefix : str
            The prefix to be checked in the poses DataFrame.

        Raises
        ------
        KeyError
            If the prefix is already used in the poses DataFrame.

        Further Details
        ---------------
        This method verifies whether the specified prefix is already in use within the poses DataFrame. It is useful for ensuring that new prefixes do not conflict with existing ones, maintaining data integrity.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Check if a prefix is already used
            poses_instance.check_prefix('new_prefix')

        Notes
        -----
        - The method raises a KeyError if the prefix is found in the DataFrame, indicating a conflict.
        - Ensures that new prefixes are unique and can be safely used for new columns or attributes.

        """
        if f"{prefix}_location" in self.df.columns or f"{prefix}_description" in self.df.columns:
            raise KeyError(f"Prefix {prefix} is already taken in poses.df")
        if "/" in prefix:
            raise ValueError(f"Prefix must not contain a slash '/' as this will raise problems with runner directories")

    def check_poses_df_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks the integrity of the poses DataFrame, ensuring it contains necessary columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be checked for integrity.

        Returns
        -------
        pd.DataFrame
            The validated poses DataFrame.

        Raises
        ------
        KeyError
            If the DataFrame does not contain the mandatory columns 'input_poses', 'poses', and 'poses_description'.

        Further Details
        ---------------
        This method verifies that the poses DataFrame contains the necessary columns required for proper functioning. It ensures that the DataFrame has 'input_poses', 'poses', and 'poses_description' columns, which are essential for various operations.

        Example
        -------
        .. code-block:: python

            from poses import Poses
            import pandas as pd

            # Initialize the Poses class
            poses_instance = Poses()

            # Create a sample DataFrame
            sample_df = pd.DataFrame({
                'input_poses': ['path/to/pose1.pdb'],
                'poses': ['path/to/pose1.pdb'],
                'poses_description': ['pose1']
            })

            # Check the integrity of the DataFrame
            validated_df = poses_instance.check_poses_df_integrity(sample_df)

        Notes
        -----
        - The method raises a KeyError if any of the mandatory columns are missing.
        - Ensures that the DataFrame is properly structured for further data manipulation and analysis.

        """
        cols = ["input_poses", "poses", "poses_description"]
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Corrupted Format: DataFrame does not contain mandatory Poses column {col}")
        return df

    def split_multiline_fasta(self, path: str, encoding: str = "UTF-8") -> list[str]:
        """
        Splits a multiline FASTA file into individual FASTA files, each containing a single sequence.

        Parameters
        ----------
        path : str
            The path to the multiline FASTA file.
        encoding : str, optional
            The encoding of the FASTA file (default is "UTF-8").

        Returns
        -------
        list[str]
            A list of file paths to the individual FASTA files.

        Further Details
        ---------------
        This method reads a multiline FASTA file and splits it into individual FASTA files, each containing a single sequence. The individual FASTA files are stored in a subdirectory named 'input_fastas_split' within the working directory.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with a working directory
            poses_instance = Poses(work_dir='path/to/work_dir')

            # Split a multiline FASTA file
            individual_fasta_paths = poses_instance.split_multiline_fasta('path/to/multiline.fasta')

        Notes
        -----
        - The method creates a subdirectory named 'input_fastas_split' within the working directory to store the individual FASTA files.
        - The descriptions in the FASTA file are sanitized to replace special characters with underscores.
        - Raises an AttributeError if the working directory is not set.

        """
        logging.warning(f"Multiline Fasta detected as input to poses. Splitting up the multiline fasta into multiple poses. Split fastas are stored at work_dir/input_fastas/")
        if not hasattr(self, "work_dir"):
            raise AttributeError(f"Set up a work_dir attribute (Poses.set_work_dir()) for your poses class.")

        # read multilie-fasta file and split into individual poses
        fasta_dict = parse_fasta_to_dict(path, encoding=encoding)

        # prepare descriptions in fasta_dict for writing:
        symbols_to_replace = r"[\/\-\:\ \.\|\,]"
        fasta_dict = {re.sub(symbols_to_replace, "_", description): seq for description, seq in fasta_dict.items()}

        # setup fasta directory self.work_dir/input_fastas_split/
        output_dir = os.path.abspath(f"{self.work_dir}/input_fastas_split/")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # write individual poses in fasta directory:
        out_poses = []
        for description, seq in fasta_dict.items():
            fp = f"{output_dir}/{description}.fa"
            try:
                # check if files are already there. If contents do not match, write the new fasta-file
                subfasta_dict = parse_fasta_to_dict(path, encoding=encoding)
                x_desc = list(subfasta_dict.keys())[0]
                x_seq = list(subfasta_dict.values())[0]
                if description != x_desc or seq != x_seq:
                    raise FileNotFoundError

            except FileNotFoundError:
                with open(fp, 'w', encoding=encoding) as f:
                    f.write(f">{description}\n{seq}")

            # add fasta path to out_poses:
            out_poses.append(fp)

        # return list containing paths to .fa files as poses.
        return out_poses

    def determine_pose_type(self, pose_col: str = None) -> list:
        """
        Determines the file types of the poses based on their extensions.

        Parameters
        ----------
        pose_col : str, optional
            The column in the DataFrame containing the pose file paths (default is 'poses').

        Returns
        -------
        list
            A list of unique file extensions found in the pose file paths.

        Further Details
        ---------------
        This method extracts and identifies the file extensions of the pose file paths in the specified column. It returns a list of unique file extensions, which helps in understanding the types of files being managed.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Determine the pose file types
            pose_types = poses_instance.determine_pose_type()

        Notes
        -----
        - The method logs a warning if multiple file extensions are found.
        - If no file extensions are found, it logs a warning indicating the inability to determine file types.
        - Ensures that the returned list contains only unique file extensions.

        """
        def extract_extension(file_path):
            _, ext = os.path.splitext(file_path)
            return ext
        
        pose_col = pose_col or 'poses'

        # extract extensions and create a set containing only unique values
        ext = list(set(self.df[pose_col].apply(extract_extension).to_list()))
        if len(ext) > 1:
            logging.warning(f"Multiple file extensions present in poses: {ext}")
            return ext
        if len(ext) == 1:
            if ext[0] == "":
                logging.warning(f"Could not determine file extension from poses!")
            else:
                logging.info(f"Poses identified as {ext} files")
            return ext
        return []

    ############################################ Input Methods ######################################
    def load_poses(self, poses_path: str) -> "Poses":
        """
        Loads poses from a specified file and updates the Poses instance.

        Parameters
        ----------
        poses_path : str
            The path to the file containing the poses to be loaded.

        Returns
        -------
        Poses
            The updated Poses instance with poses loaded from the specified file.

        Further Details
        ---------------
        This method reads a file containing poses and updates the Poses instance with the data. The file format is automatically detected based on the file extension, and the corresponding loading function is used to read the data into a DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class
            poses_instance = Poses()

            # Load poses from a file
            poses_instance.load_poses('path/to/poses.json')

        Notes
        -----
        - The method supports various file formats, including JSON, CSV, Pickle, Feather, and Parquet.
        - Ensures that the loaded DataFrame contains the necessary columns and updates the Poses instance accordingly.

        """
        # read format
        load_function = get_format(poses_path)

        # load df from file:
        self.set_poses(poses=load_function(poses_path))
        return self

    ############################################ Output Methods ######################################
    def save_scores(self, out_path: str = None, out_format: str = None) -> None:
        """
        Saves the scores DataFrame to a specified file path in the desired format.

        Parameters
        ----------
        out_path : str, optional
            The file path where the scores will be saved. If not provided, the default scorefile path is used.
        out_format : str, optional
            The format in which to save the scores. If not provided, the default storage format is used.

        Further Details
        ---------------
        This method saves the scores DataFrame to the specified file path in the desired format. It ensures that the file name conforms to the specified format by appending the correct file extension if necessary.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Save scores to a specific path in CSV format
            poses_instance.save_scores(out_path='path/to/scores.csv', out_format='csv')

        Notes
        -----
        - Supports various file formats, including JSON, CSV, Pickle, Feather, and Parquet.
        - The method automatically appends the correct file extension if it is not already present in the out_path.
        - Ensures that the scores are saved in a format suitable for further analysis and processing.
        """
        # setup defaults
        out_path = out_path or self.scorefile
        out_format = out_format or self.storage_format

        # make sure the filename conforms to format
        if not out_path.endswith(f".{out_format}"):
            out_path += f".{out_format}"

        if (save_method_name := FORMAT_STORAGE_DICT.get(out_format.lower())):
            getattr(self.df, save_method_name)(out_path)

    def save_poses(self, out_path: str, poses_col: str = "poses", overwrite: bool = True) -> None:
        """
        Saves the poses to a specified directory, with an option to overwrite existing files.

        Parameters
        ----------
        out_path : str
            The directory where the poses will be saved.
        poses_col : str, optional
            The column in the DataFrame containing the pose file paths (default is 'poses').
        overwrite : bool, optional
            If True, existing files in the target directory will be overwritten (default is True).

        Further Details
        ---------------
        This method saves the pose files to the specified directory. It copies the pose files from their current locations to the new directory, ensuring that the directory structure is maintained. The `overwrite` parameter controls whether existing files in the target directory are overwritten.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Save poses to a new directory
            poses_instance.save_poses(out_path='path/to/new_poses_dir', overwrite=False)

        Notes
        -----
        - The method ensures that the target directory exists, creating it if necessary.
        - If `overwrite` is set to False, the method skips saving poses that already exist in the target directory.
        - Logs the saving process, including any skipped files due to the overwrite setting.

        """
        poses = self.df[poses_col].to_list()
        new_poses = [os.path.join(out_path, os.path.basename(pose)) for pose in poses]
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        # check if poses are already at out_path, skip if overwrite is set to False
        if all((os.path.isfile(pose) for pose in new_poses)) and not overwrite:
            logging.info(f"Poses already found at {out_path} and overwrite is set to 'False'. Skipping save_poses.")
            return

        # save poses
        logging.info(f"Storing poses from column {poses_col} at {out_path}")
        for pose, new_pose in zip(poses, new_poses):
            shutil.copy(pose, new_pose)

    def poses_list(self) -> list[str]:
        """
        Returns a list of pose file paths from the DataFrame.

        Returns
        -------
        list
            A list of pose file paths.

        Further Details
        ---------------
        This method extracts the pose file paths from the 'poses' column of the DataFrame and returns them as a list. It provides a convenient way to access the stored pose file paths.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Get the list of pose file paths
            pose_paths = poses_instance.poses_list()

        Notes
        -----
        - The method assumes that the 'poses' column exists in the DataFrame.
        - Provides a simple way to retrieve all pose file paths managed by the Poses instance.

        """
        return self.df["poses"].to_list()

    ########################################## Operations ###############################################
    def get_pose(self, pose_description: str) -> Bio.PDB.Structure.Structure:
        """
        Retrieves a pose structure based on its description.

        Parameters
        ----------
        pose_description : str
            The description of the pose to be retrieved.

        Returns
        -------
        Bio.PDB.Structure.Structure
            The Bio.PDB Structure object corresponding to the specified pose description.

        Raises
        ------
        KeyError
            If the pose description is not found in the poses DataFrame.

        Further Details
        ---------------
        This method locates the pose file based on its description and loads it as a Bio.PDB Structure object. It is useful for accessing specific pose structures for further analysis or manipulation.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Retrieve a specific pose structure
            pose_structure = poses_instance.get_pose('pose1')

        Notes
        -----
        - The method uses the 'poses_description' column to locate the specified pose.
        - Ensures that the returned pose is loaded as a Bio.PDB Structure object for further processing.

        """
        if pose_description not in self.df["poses_description"]:
            raise KeyError(f"Pose {pose_description} not Found in Poses DataFrame!")
        return load_structure_from_pdbfile(self.df[self.df["poses_description"] == pose_description]["poses"].values[0])
    
    def reindex_poses(self, prefix:str, remove_layers:int=1, force_reindex:bool=False, sep:str="_", overwrite:bool=False) -> None:
        """
        Removes index layers from poses. Saves reindexed poses to an output directory.

        Parameters
        ----------
        prefix : str
            The directory where the duplicated poses will be saved and the prefix for the DataFrame columns containing the original paths and descriptions.
        remove_layers : int, optional
            The number of index layers to remove. 
        force_reindex : bool, optional
            Add a new index layer to all poses.
        sep : str, optional
            The separator used to split the description column into layers.
            
        Further Details
        ---------------
        This method removes index layers from poses (_0001, 0002, etc). Subtracts the set number of layers from the description column and groups the poses accordingly.
        If force_reindex is True, adds one index layer to all poses. 

        Notes
        -----
        - The method creates the output directory if it does not exist.
        - Raises a RuntimeError if multiple poses with identical description after index layer removal are found and force_reindex is False..

        """

        out_dir = os.path.join(self.work_dir, prefix)
        os.makedirs(out_dir, exist_ok=True)
        self.df[f"{prefix}_pre_reindexing_poses_description"] = self.df['poses_description']
        self.df[f"{prefix}_pre_reindexing_poses"] = self.df['poses']

        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            self.df["tmp_layer_column"] = self.df['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        self.df.sort_values(["tmp_layer_column", "poses_description"], inplace=True) # sort to make sure that all poses are in the same order after grouping

        # group by temporary description column, reindex
        out = []
        if any([len(group_df.index) > 1 for name, group_df in self.df.groupby("tmp_layer_column", sort=False)]) and not force_reindex:
            raise RuntimeError(f'Multiple files with identical description found after removing index layers. Set <force_reindex> to True if new index layers should be added.')

        for name, group_df in self.df.groupby("tmp_layer_column", sort=False):
            group_df.reset_index(drop=True, inplace=True) # resetting index, otherwise index of original poses df would be used
                # adding new index layer since multiple
            for i, ser in group_df.iterrows():
                ext = os.path.splitext(ser['poses'])[1]
                if force_reindex: description = f"{name}{sep}{str(i+1).zfill(4)}"
                else: description = name
                path = os.path.join(out_dir, f"{description}{ext}")
                if overwrite == True or not os.path.isfile(path):
                    shutil.copy(ser['poses'], path)
                ser['poses'] = path
                ser['poses_description'] = description
                out.append(ser)
        
        self.df = pd.DataFrame(out)
        self.df.reset_index(inplace=True, drop=True)
        # drop temporary description column
        self.df.drop("tmp_layer_column", inplace=True, axis=1)

    def duplicate_poses(self, output_dir:str, n_duplicates:int, overwrite:bool=False) -> None:
        """
        Duplicates poses a specified number of times and saves them to an output directory.

        Parameters
        ----------
        output_dir : str
            The directory where the duplicated poses will be saved.
        n_duplicates : int
            The number of duplicates to create for each pose.

        Further Details
        ---------------
        This method creates multiple copies of each pose file and saves them to the specified output directory. The duplicated files are named with an incremented index to distinguish them.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Duplicate the poses
            poses_instance.duplicate_poses(output_dir='path/to/duplicates', n_duplicates=3)

        Notes
        -----
        - The method creates the output directory if it does not exist.
        - Ensures that the duplicated files have unique names by appending an index.
        - Logs the duplication process and verifies the creation of duplicate files.

        """
        def insert_index_layer(dir:str, input_path:str, n:int, sep:str="_") -> str:
            '''inserts index layer.'''
            in_file = os.path.basename(input_path)
            description, extension = os.path.splitext(in_file)
            out_path = os.path.join(dir, f"{description}{sep}{str(n).zfill(4)}{extension}")
            return out_path
        
        # create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # iterate over poses and copy them to new location with one additional index layer
        duplicates = []
        for n in range(1, n_duplicates+1):
            new_df = self.df.copy(deep=True)
            new_paths = [insert_index_layer(output_dir, pose, n, "_") for pose in new_df["poses"].to_list()]
            new_descriptions = [description_from_path(path) for path in new_paths]
            for old_pose, new_pose in zip(new_df["poses"].to_list(), new_paths):
                if overwrite == True or not os.path.isfile(new_pose):
                    shutil.copy(old_pose, new_pose)
            new_df["poses"] = new_paths
            new_df["poses_description"] = new_descriptions
            duplicates.append(new_df)
        
        self.df = pd.concat(duplicates)
        self.df.reset_index(drop=True, inplace=True)

    def reset_poses(self, new_poses_col: str='input_poses', force_reset_df: bool=False):
        """
        Resets the poses DataFrame to the original input poses, with an option to force reset.

        Parameters
        ----------
        new_poses_col : str, optional
            The column in the DataFrame containing the new pose file paths (default is 'input_poses').
        force_reset_df : bool, optional
            If True, forces a reset of the DataFrame even if the number of new poses does not match the original (default is False).

        Further Details
        ---------------
        This method resets the poses DataFrame to use the original input poses. It handles multiline FASTA inputs and ensures that the DataFrame structure is preserved or reset based on the force_reset_df parameter.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Reset the poses to the original input poses
            poses_instance.reset_poses()

        Notes
        -----
        - The method ensures that the new poses are unique and properly formatted.
        - Raises a RuntimeError if the number of new poses does not match the original and force_reset_df is False.
        - Logs warnings and information about the reset process, ensuring data integrity.

        """
        def unique_ordered_list(original_list):
            seen = set()  # Initialize an empty set to track seen elements
            unique_list = []
            for item in original_list:
                if item not in seen:  # Check membership in the set, which is O(1) (faster lookup in sets than in lists)
                    unique_list.append(item)
                    seen.add(item)  # Add the item to the set
            return unique_list

        col_in_df(self.df, new_poses_col)

        new_poses = self.df[new_poses_col].to_list()
        # handle multiline .fa inputs for poses!
        for pose in new_poses:
            if not pose.endswith(".fa") and not pose.endswith(".fasta"):
                continue
            if len(parse_fasta_to_dict(pose)) > 1:
                new_poses.remove(pose)
                new_poses += self.split_multiline_fasta(pose)

        # create unique poses
        new_poses = unique_ordered_list(new_poses)

        if not len(new_poses) == len(self.df.index):
            logging.warning(f"Different number of new poses ({len(new_poses)}) than number of original poses ({len(self.df.index)})!")
            if force_reset_df:
                logging.warning(f"Resetting poses dataframe. Be aware of the consequences like possibly reading in false outputs when reusing prefixes!")
                self.df = pd.DataFrame({"input_poses": new_poses, "poses": new_poses, "poses_description": self.parse_descriptions(new_poses)})
            else: raise RuntimeError(f"Could not preserve original dataframe. You can set <force_reset_df> if you want to delete it, but be aware of the consequences like possibly reading in false outputs when reusing prefixes!")
        else:
            self.df['poses'] = new_poses
            self.df['poses_description'] = self.parse_descriptions(poses=self.df['poses'].to_list())

    def set_motif(self, motif_col: str) -> None:
        """
        Sets a motif column in the poses DataFrame for further analysis.

        Parameters
        ----------
        motif_col : str
            The column in the DataFrame containing the motifs to be set.

        Raises
        ------
        KeyError
            If the specified motif column is not found in the poses DataFrame.
        TypeError
            If the objects in the specified motif column are not of type ResidueSelection.

        Further Details
        ---------------
        This method sets a column in the poses DataFrame to be used as motifs for further analysis. The motifs must be instances of the ResidueSelection class.

        Example
        -------
        .. code-block:: python

            from poses import Poses
            from protflow.residues import ResidueSelection

            # Initialize the Poses class with some poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Assume we have a column 'motifs' with ResidueSelection objects
            poses_instance.set_motif('motifs')

        Notes
        -----
        - The method ensures that the specified column exists and contains ResidueSelection objects.
        - Logs any errors encountered during the process for easier debugging and verification.

        """
        # check if motif_col exists. check if all entries in motif col are ResidueSelection objects.
        col_in_df(self.df, motif_col)
        if not all([isinstance(motif, ResidueSelection) for motif in self.df[motif_col].to_list()]):
            raise TypeError(f"Setting a motif requires the objects in 'motif_col' to be of type ResidueSelection. Check documentation of protflow.residues module for how to create the object (it's simple).")

        # set motif
        self.motifs.append(motif_col)

    def convert_pdb_to_fasta(self, prefix: str, update_poses: bool = False, chain_sep: str = ":") -> None:
        """
        Converts PDB pose files to FASTA format and optionally updates the poses.

        Parameters
        ----------
        prefix : str
            The prefix used for naming the output FASTA files.
        update_poses : bool, optional
            If True, updates the poses DataFrame to use the new FASTA files (default is False).
        chain_sep : str, optional
            The separator used for chain identifiers in the FASTA file (default is ":").

        Raises
        ------
        RuntimeError
            If the poses are not of type PDB.

        Further Details
        ---------------
        This method converts PDB pose files to FASTA format and stores them in a directory named with the given prefix. It can also update the poses DataFrame to use the new FASTA files if specified.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some PDB poses
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Convert the PDB files to FASTA format
            poses_instance.convert_pdb_to_fasta(prefix='converted', update_poses=True)

        Notes
        -----
        - The method checks that the poses are of type PDB before conversion.
        - Creates a new directory within the working directory to store the FASTA files.
        - Logs the conversion process and verifies the creation of FASTA files.

        """
        if not self.determine_pose_type() == ['.pdb']:
            raise RuntimeError(f"Poses must be of type .pdb, not {self.determine_pose_type()}")

        os.makedirs(fasta_dir := os.path.join(self.work_dir, f'{prefix}_fasta_location'), exist_ok=True)
        seqs = [get_sequence_from_pose(load_structure_from_pdbfile(path_to_pdb=pose), chain_sep=chain_sep) for pose in self.df['poses'].to_list()]

        fasta_paths = []
        for name, seq in zip(self.df['poses_description'].to_list(), seqs):
            fasta_path = os.path.join(fasta_dir, f'{name}.fasta')
            fasta_paths.append(fasta_path)
            with open(fasta_path, 'w', encoding="UTF-8") as f:
                f.write(f">{name}\n{seq}")

        self.df[f'{prefix}_fasta_location'] = fasta_paths
        if update_poses:
            self.df['poses'] = fasta_paths

    ########################################## Filtering ###############################################
    def filter_poses_by_rank(self, n: float, score_col: str, remove_layers = None, layer_col = "poses_description", sep = "_", ascending = True, prefix: str = None, plot: bool = False, overwrite: bool = True, storage_format: str = None) -> "Poses":
        """
        Filters poses based on their rank in a specified score column, with options to handle layers and generate plots.

        Parameters
        ----------
        n : float
            The number of top-ranked poses to keep. If n < 1, it represents a fraction of the total poses.
        score_col : str
            The column in the DataFrame containing the scores used for ranking.
        remove_layers : int, optional
            The number of layers to remove from the pose descriptions before ranking. This helps in grouping similar poses.
        layer_col : str, optional
            The column used for layer-based grouping of poses (default is "poses_description").
        sep : str, optional
            The separator used in the layer descriptions (default is "_").
        ascending : bool, optional
            If True, ranks poses in ascending order of scores; otherwise, in descending order (default is True).
        prefix : str, optional
            The prefix used for naming the output filtered poses file and plot.
        plot : bool, optional
            If True, generates a plot comparing scores before and after filtering (default is False).
        overwrite : bool, optional
            If True, overwrites existing filtered poses files (default is True).
        storage_format : str, optional
            The format used for storing the filtered poses (default is None, which uses the existing storage format).

        Returns
        -------
        Poses
            The updated Poses instance with filtered poses.

        Further Details
        ---------------
        This method filters the poses DataFrame to retain only the top-ranked poses based on their scores. It supports fractional ranking, layer-based grouping, and optional plot generation for visualizing the filtering process. The filtered poses can be saved to a file with a specified prefix and storage format.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Filter poses by rank
            poses_instance.filter_poses_by_rank(n=10, score_col='score', prefix='top_poses', plot=True)

        Notes
        -----
        - The method creates a filtered poses file and an optional plot in the specified working directory.
        - Ensures that the DataFrame is properly sorted and filtered based on the provided parameters.
        - Logs the filtering process, including any errors or warnings related to the ranking criteria.

        """
        # define filter output if <prefix> is provided, make sure output directory exists
        if prefix:
            if self.filter_dir is None:
                raise AttributeError(f"Filter directory was not set! Did you set a working directory? work_dir can be set with Poses.set_work_dir() and sets up a filter_dir automatically.")
            os.makedirs(self.filter_dir, exist_ok=True)

            # make sure output format is available
            storage_format = storage_format or self.storage_format
            if storage_format not in FORMAT_STORAGE_DICT:
                raise KeyError(f"Format {storage_format} not available. Format must be on of {list(FORMAT_STORAGE_DICT)}")

            # set filter output name
            output_name = os.path.join(self.filter_dir, f"{prefix}_filter.{storage_format}")

            # load previous filter output if it exists and <overwrite> = False, set poses_df as filtered dataframe and return filtered dataframe
            if not overwrite and os.path.isfile(output_name):
                filter_df = get_format(output_name)(output_name)
                self.df = filter_df
                return filter_df

        # Filter df down to the number of poses specified with <n>
        orig_len = str(len(self.df))
        filter_df = filter_dataframe_by_rank(df=self.df, col=score_col, n=n, remove_layers=remove_layers, layer_col=layer_col, sep=sep, ascending=ascending).reset_index(drop=True)
        logging.info(f"Filtered poses from {orig_len} to {str(len(filter_df))} poses according to {score_col}.")

        # save filtered dataframe if prefix is provided
        if prefix:
            logging.info(f"Saving filter output to {output_name}.")
            save_method_name = FORMAT_STORAGE_DICT.get(storage_format)
            getattr(filter_df, save_method_name)(output_name)

        # create filter-plots if specified.
        if plot:
            if not prefix:
                raise RuntimeError(f"<prefix> was not set, but is mandatory for plotting!")
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)

            out_path = os.path.join(self.plots_dir, f"{prefix}_filter.png")
            logging.info(f"Creating filter plot at {out_path}.")
            cols = [score_col]
            plots.violinplot_multiple_cols_dfs(
                dfs=[self.df, filter_df],
                df_names=["Before Filtering", "After Filtering"],
                cols=cols,
                y_labels=cols,
                out_path=out_path,
                show_fig=False
            )

        # update object attributs [df]
        self.df = filter_df
        logging.info(f"Filtering completed.")
        return self

    def filter_poses_by_value(self, score_col: str, value, operator: str, prefix: str = None, plot: bool = False, overwrite: bool = True, storage_format: str = None) -> "Poses":
        """
        Filters poses based on a specified value in a score column, with options to generate plots.

        Parameters
        ----------
        score_col : str
            The column in the DataFrame containing the scores used for filtering.
        value : float or int
            The value used as the threshold for filtering poses.
        operator : str
            The comparison operator used for filtering ('>', '>=', '<', '<=', '=', '!=').
        prefix : str, optional
            The prefix used for naming the output filtered poses file and plot.
        plot : bool, optional
            If True, generates a plot comparing scores before and after filtering (default is False).
        overwrite : bool, optional
            If True, overwrites existing filtered poses files (default is True).
        storage_format : str, optional
            The format used for storing the filtered poses (default is None, which uses the existing storage format).

        Returns
        -------
        Poses
            The updated Poses instance with filtered poses.

        Raises
        ------
        ValueError
            If all poses are removed based on the filtering criteria.

        Further Details
        ---------------
        This method filters the poses DataFrame based on a specified value in a score column, using the provided comparison operator. It supports optional plot generation for visualizing the filtering process and allows saving the filtered poses to a file with a specified prefix and storage format.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses(poses=['path/to/pose1.pdb', 'path/to/pose2.pdb'])

            # Filter poses by value
            poses_instance.filter_poses_by_value(score_col='score', value=0.5, operator='>', prefix='filtered_poses', plot=True)

        Notes
        -----
        - The method creates a filtered poses file and an optional plot in the specified working directory.
        - Ensures that the DataFrame is properly filtered based on the provided criteria.
        - Logs the filtering process, including any errors or warnings related to the filtering criteria.
        - Raises a ValueError if the filtering criteria remove all poses, ensuring that the Poses instance retains valid data.

        """
        logging.info(f"Filtering poses according to column {score_col} with operator {operator} and target value {value}")

        # define filter output if <prefix> is provided, make sure output directory exists
        if prefix:
            if self.filter_dir is None:
                raise AttributeError(f"Filter directory was not set! Did you set a working directory?")
            os.makedirs(self.filter_dir, exist_ok=True)

            # make sure output format is available
            storage_format = storage_format or self.storage_format
            if storage_format not in FORMAT_STORAGE_DICT:
                raise KeyError(f"Format {storage_format} not available. Format must be one of {list(FORMAT_STORAGE_DICT)}")

            # set filter output name
            output_name = os.path.join(self.filter_dir, f"{prefix}_filter.{storage_format}")

            # load previous filter output if it exists and <overwrite> = False, set poses_df as filtered dataframe and return filtered dataframe
            if not overwrite and os.path.isfile(output_name):
                filter_df = get_format(output_name)(output_name)
                self.df = filter_df
                return filter_df

        # Filter df down to the number of poses specified with <n>
        orig_len = len(self.df)
        filter_df = filter_dataframe_by_value(df=self.df, col=score_col, value=value, operator=operator).reset_index(drop=True)

        # make sure there are still poses left in the Poses class.
        if len(filter_df) == 0:
            logging.warning(f"All poses removed from Poses object. No pose fullfills the filtering criterium {operator} {value} for score {score_col}")
        logging.info(f"Filtered poses from {orig_len} to {len(filter_df.index)} poses.")

        # save filtered dataframe if prefix is provided
        if prefix:
            logging.info(f"Saving filter output to {output_name}.")
            save_method_name = FORMAT_STORAGE_DICT.get(storage_format)
            getattr(filter_df, save_method_name)(output_name)

        if plot:
            if not prefix:
                raise RuntimeError(f"<prefix> was not set, but is mandatory for plotting!")
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)
            out_path = os.path.join(self.plots_dir, f"{prefix}_filter.png")
            logging.info(f"Creating filter plot at {out_path}.")
            cols = [score_col]
            plots.violinplot_multiple_cols_dfs(
                dfs=[self.df, filter_df],
                df_names=["Before Filtering", "After Filtering"],
                cols=cols,
                y_labels=cols,
                out_path=out_path,
                show_fig=False
            )

        # update object attributs [df]
        self.df = filter_df
        logging.info(f"Filtering completed.")
        return self

    ########################################## Score manipulation ###############################################
    def calculate_composite_score(self, name: str, scoreterms: list[str], weights: list[float], plot: bool = False, scale_output: bool = False) -> "Poses":
        """
        Calculates a composite score from specified score columns, applying weights and normalization, and optionally generates a plot.

        Parameters
        ----------
        name : str
            The name of the new composite score column to be created.
        scoreterms : list[str]
            The list of score columns to be included in the composite score.
        weights : list[float]
            The list of weights corresponding to each score column.
        plot : bool, optional
            If True, generates a plot of the composite score and the individual score terms (default is False).
        scale_output : bool, optional
            If True, scales the composite score to a range between 0 and 1 (default is False).

        Returns
        -------
        Poses
            The updated Poses instance with the new composite score column.

        Raises
        ------
        ValueError
            If the number of scoreterms and weights do not match.
        TypeError
            If any score column contains non-numeric values.

        Further Details
        ---------------
        This method calculates a composite score from multiple score columns by applying the specified weights and normalizing the columns. The normalization process involves subtracting the median and dividing by the standard deviation for each score column. Optionally, the composite score can be scaled to a range between 0 and 1.

        The method ensures that each score column contains numeric values and applies the normalization process as follows:
        1. Calculate the median and standard deviation of each score column.
        2. Normalize the column by subtracting the median and dividing by the standard deviation.
        3. Optionally scale the normalized values to a range between 0 and 1.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate a composite score
            poses_instance.calculate_composite_score(
                name='composite_score',
                scoreterms=['score1', 'score2'],
                weights=[0.5, 0.5],
                plot=True,
                scale_output=True
            )

        Notes
        -----
        - The method ensures that the number of scoreterms and weights match.
        - Normalization helps in making the scores comparable by removing scale differences.
        - Generates a violin plot if the plot parameter is set to True, showing the distribution of the composite score and individual score terms.

        """
        logging.info(f"Creating composite score {name} for scoreterms {scoreterms} with weights {weights}")
        # check if output column already exists in dataframe
        if name in self.df:
            logging.warning(f"Column {name} already exists in poses dataframe! It will be overwritten!")
        # calculate composite score
        self.df[name] = combine_dataframe_score_columns(df=self.df, scoreterms=scoreterms, weights=weights, scale=scale_output)

        if plot:
            if self.plots_dir is None:
                raise AttributeError(f"Plots directory was not set! Did you set a working directory?")
            os.makedirs(self.plots_dir, exist_ok=True)
            out_path = os.path.join(self.plots_dir, f"{name}_comp_score.png")
            logging.info(f"Creating composite score plot at {out_path}.")
            scoreterms.append(name)
            plots.violinplot_multiple_cols(
                dataframe=self.df,
                cols=scoreterms,
                titles=scoreterms,
                y_labels=scoreterms,
                dims=None,
                out_path=out_path,
                show_fig=False
            )

        self.save_scores()
        logging.info("Composite score creation completed.")

        return self
    
    def calculate_mean_score(self, name: str, score_col: str, skipna: bool = False, remove_layers: int = None, sep: str = "_"):
        """
        Calculate the mean score of the selected score column. If remove_layers is set, calculates mean scores over poses grouped by the description column with the set number of index layers removed.

        Parameters
        ----------
        name : str
            The name of the new column where the mean scores will be stored.
        score_col : str
            The name of the column from which to calculate the mean scores.
        skipna : bool, optional
            Whether to skip NA/null values. Default is False.
        remove_layers : int, optional
            The number of layers to remove from the index for grouping. If None, no layers are removed. Default is None.
        sep : str, optional
            The separator used in the 'poses_description' column for splitting and joining layers. Default is "_".

        Returns
        -------
        self
            The instance of the class with the mean scores added to the DataFrame.

        Raises
        ------
        TypeError
            If `remove_layers` is not an integer.
        ValueError
            If `score_col` does not exist in the DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate the mean score
            poses_instance.calculate_mean_score(
                name='mean_score1',
                score_col='score1',
                skipna=True,
                remove_layers=1,
            )
        """
        col_in_df(self.df, score_col)
        df_layers = self.df.copy()
        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            df_layers["tmp_layer_column"] = df_layers['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        df = []
        for _, group_df in df_layers.groupby("tmp_layer_column", sort=False):
            group_df[name] = group_df[score_col].mean(skipna=skipna)
            df.append(group_df)

        df = pd.concat(df).reset_index(drop=True)
        df = df[['poses_description', name]]
        
        # drop temporary description column
        self.df = self.df.merge(df, on='poses_description')
        return self

    def calculate_median_score(self, name: str, score_col: str, skipna: bool = False, remove_layers: int = None, sep: str = "_"):
        """
        Calculate the median score of the selected score column. If remove_layers is set, calculates median scores over poses grouped by the description column with the set number of index layers removed.

        Parameters
        ----------
        name : str
            The name of the new column where the mean scores will be stored.
        score_col : str
            The name of the column from which to calculate the median scores.
        skipna : bool, optional
            Whether to skip NA/null values. Default is False.
        remove_layers : int, optional
            The number of layers to remove from the index for grouping. If None, no layers are removed. Default is None.
        sep : str, optional
            The separator used in the 'poses_description' column for splitting and joining layers. Default is "_".

        Returns
        -------
        self
            The instance of the class with the mean scores added to the DataFrame.

        Raises
        ------
        TypeError
            If `remove_layers` is not an integer.
        ValueError
            If `score_col` does not exist in the DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate the median score
            poses_instance.calculate_median_score(
                name='median_score1',
                score_col='score1',
                skipna=True,
                remove_layers=1,
            )
        """
        col_in_df(self.df, score_col)
        df_layers = self.df.copy()
        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            df_layers["tmp_layer_column"] = df_layers['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        df = []
        for _, group_df in df_layers.groupby("tmp_layer_column", sort=False):
            group_df[name] = group_df[score_col].median(skipna=skipna)
            df.append(group_df)

        df = pd.concat(df).reset_index(drop=True)
        df = df[['poses_description', name]]
        
        # drop temporary description column
        self.df = self.df.merge(df, on='poses_description')
        return self
    
    def calculate_std_score(self, name: str, score_col: str, skipna: bool = False, remove_layers: int = None, sep: str = "_"):
        """
        Calculate the standard deviation of the selected score column. If remove_layers is set, calculates standard deviations over poses grouped by the description column with the set number of index layers removed.

        Parameters
        ----------
        name : str
            The name of the new column where the mean scores will be stored.
        score_col : str
            The name of the column from which to calculate the standard deviation.
        skipna : bool, optional
            Whether to skip NA/null values. Default is False.
        remove_layers : int, optional
            The number of layers to remove from the index for grouping. If None, no layers are removed. Default is None.
        sep : str, optional
            The separator used in the 'poses_description' column for splitting and joining layers. Default is "_".

        Returns
        -------
        self
            The instance of the class with the mean scores added to the DataFrame.

        Raises
        ------
        TypeError
            If `remove_layers` is not an integer.
        ValueError
            If `score_col` does not exist in the DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate the standard deviation
            poses_instance.calculate_std_score(
                name='mean_score1',
                score_col='score1',
                skipna=True,
                remove_layers=1,
            )
        """
        col_in_df(self.df, score_col)
        df_layers = self.df.copy()
        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            df_layers["tmp_layer_column"] = df_layers['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        df = []
        for _, group_df in df_layers.groupby("tmp_layer_column", sort=False):
            group_df[name] = group_df[score_col].std(skipna=skipna)
            df.append(group_df)

        df = pd.concat(df).reset_index(drop=True)
        df = df[['poses_description', name]]
        
        # drop temporary description column
        self.df = self.df.merge(df, on='poses_description')
        return self
    
    def calculate_max_score(self, name: str, score_col: str, skipna: bool = False, remove_layers: int = None, sep: str = "_"):
        """
        Calculate the maximum value of the selected score column. If remove_layers is set, calculates the maximum value over poses grouped by the description column with the set number of index layers removed.

        Parameters
        ----------
        name : str
            The name of the new column where the maximum values will be stored.
        score_col : str
            The name of the column from which to calculate the maximum value.
        skipna : bool, optional
            Whether to skip NA/null values. Default is False.
        remove_layers : int, optional
            The number of layers to remove from the index for grouping. If None, no layers are removed. Default is None.
        sep : str, optional
            The separator used in the 'poses_description' column for splitting and joining layers. Default is "_".

        Returns
        -------
        self
            The instance of the class with the maximum values added to the DataFrame.

        Raises
        ------
        TypeError
            If `remove_layers` is not an integer.
        ValueError
            If `score_col` does not exist in the DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate the maximum values
            poses_instance.calculate_max_score(
                name='max_score1',
                score_col='score1',
                skipna=True,
                remove_layers=1,
            )
        """

        col_in_df(self.df, score_col)
        df_layers = self.df.copy()
        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            df_layers["tmp_layer_column"] = df_layers['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        df = []
        for _, group_df in df_layers.groupby("tmp_layer_column", sort=False):
            group_df[name] = group_df[score_col].max(skipna=skipna)
            df.append(group_df)

        df = pd.concat(df).reset_index(drop=True)
        df = df[['poses_description', name]]
        
        # drop temporary description column
        self.df = self.df.merge(df, on='poses_description')
        return self

    def calculate_min_score(self, name: str, score_col: str, skipna: bool = False, remove_layers: int = None, sep: str = "_"):
        """
        Calculate the minimum value of the selected score column. If remove_layers is set, calculates the maximum value over poses grouped by the description column with the set number of index layers removed.

        Parameters
        ----------
        name : str
            The name of the new column where the minimum values will be stored.
        score_col : str
            The name of the column from which to calculate the minimum value.
        skipna : bool, optional
            Whether to skip NA/null values. Default is False.
        remove_layers : int, optional
            The number of layers to remove from the index for grouping. If None, no layers are removed. Default is None.
        sep : str, optional
            The separator used in the 'poses_description' column for splitting and joining layers. Default is "_".

        Returns
        -------
        self
            The instance of the class with the minimum values added to the DataFrame.

        Raises
        ------
        TypeError
            If `remove_layers` is not an integer.
        ValueError
            If `score_col` does not exist in the DataFrame.

        Example
        -------
        .. code-block:: python

            from poses import Poses

            # Initialize the Poses class with some scores
            poses_instance = Poses()

            # Calculate the minimum values
            poses_instance.calculate_min_score(
                name='min_score1',
                score_col='score1',
                skipna=True,
                remove_layers=1,
            )
        """
        col_in_df(self.df, score_col)
        df_layers = self.df.copy()
        if remove_layers == 0: remove_layers = None
        # create temporary description column with removed index layers
        if remove_layers:
            if not isinstance(remove_layers, int): raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")
            df_layers["tmp_layer_column"] = df_layers['poses_description'].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)
        else: self.df["tmp_layer_column"] = self.df['poses_description']

        df = []
        for _, group_df in df_layers.groupby("tmp_layer_column", sort=False):
            group_df[name] = group_df[score_col].min(skipna=skipna)
            df.append(group_df)

        df = pd.concat(df).reset_index(drop=True)
        df = df[['poses_description', name]]
        
        # drop temporary description column
        self.df = self.df.merge(on='poses_description')
        return self


def normalize_series(ser: pd.Series, scale: bool = False) -> pd.Series:
    """
    Normalizes a pandas Series by subtracting the median and dividing by the standard deviation, with an option to scale the values.

    Parameters
    ----------
    ser : pd.Series
        The pandas Series to be normalized.
    scale : bool, optional
        If True, scales the normalized values to a range between 0 and 1 (default is False).

    Returns
    -------
    pd.Series
        The normalized (and optionally scaled) Series.

    Further Details
    ---------------
    This function normalizes a pandas Series by first subtracting the median and then dividing by the standard deviation. If the `scale` parameter is set to True, the normalized values are further scaled to a range between 0 and 1. This normalization process centers the data around zero and adjusts for variability, making the values comparable.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import normalize_series

        # Create a sample pandas Series
        sample_series = pd.Series([10, 20, 30, 40, 50])

        # Normalize the Series
        normalized_series = normalize_series(sample_series, scale=True)

    Notes
    -----
    - If all values in the Series are the same, the function returns a Series of zeros.
    - The optional scaling step ensures that the values are adjusted to a standardized range.

    """
    ser = ser.copy()
    # calculate median and standard deviation
    median = ser.median()
    std = ser.std()
    # check if all values in <score_col> are the same, return 0 if yes
    if ser.nunique() == 1:
        ser[:] = 0
        return ser
    # normalize score by subtracting median and dividing by standard deviation
    ser = (ser - median) / std
    # scale output to values between 0 and 1
    if scale:
        ser = scale_series(ser)
    return ser

def scale_series(ser: pd.Series) -> pd.Series:
    """
    Scales a pandas Series to a range between 0 and 1.

    Parameters
    ----------
    ser : pd.Series
        The pandas Series to be scaled.

    Returns
    -------
    pd.Series
        The scaled Series with values between 0 and 1.

    Further Details
    ---------------
    This function scales a pandas Series to a range between 0 and 1. It ensures that the minimum value in the Series becomes 0 and the maximum value becomes 1, with all other values adjusted proportionately.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import scale_series

        # Create a sample pandas Series
        sample_series = pd.Series([10, 20, 30, 40, 50])

        # Scale the Series
        scaled_series = scale_series(sample_series)

    Notes
    -----
    - If all values in the Series are the same, the function returns a Series of zeros.
    - The scaling process adjusts the values to fit within a standardized range, making them comparable.

    """
    ser = ser.copy()
    # check if all values in <score_col> are the same, set all values to 0 if yes as no scaling is possible
    if ser.nunique() == 1:
        ser[:] = 0
        return ser
    # scale series to values between 0 and 1
    factor = ser.max() - ser.min()
    ser = ser / factor
    ser = ser + (1 - ser.max())

    return ser

def combine_dataframe_score_columns(df: pd.DataFrame, scoreterms: list[str], weights: list[float], scale: bool = False) -> pd.Series:
    """
    Combines multiple score columns in a DataFrame into a single composite score, applying weights and normalization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the score columns.
    scoreterms : list[str]
        The list of score columns to be combined.
    weights : list[float]
        The list of weights corresponding to each score column.
    scale : bool, optional
        If True, scales the composite score to a range between 0 and 1 (default is False).

    Returns
    -------
    pd.Series
        The composite score as a pandas Series.

    Raises
    ------
    ValueError
        If the number of scoreterms and weights do not match.
    TypeError
        If any score column contains non-numeric values.

    Further Details
    ---------------
    This function combines multiple score columns in a DataFrame into a single composite score. Each score column is normalized by subtracting the median and dividing by the standard deviation. The normalized scores are then weighted according to the specified weights and summed to create the composite score. Optionally, the composite score can be scaled to a range between 0 and 1.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import combine_dataframe_score_columns

        # Create a sample DataFrame
        data = {
            'score1': [10, 20, 30, 40, 50],
            'score2': [15, 25, 35, 45, 55]
        }
        df = pd.DataFrame(data)

        # Combine score columns into a composite score
        composite_score = combine_dataframe_score_columns(df, scoreterms=['score1', 'score2'], weights=[0.5, 0.5], scale=True)

    Notes
    -----
    - The method ensures that the number of scoreterms and weights match.
    - Normalization helps in making the scores comparable by removing scale differences.
    - Raises a ValueError if the number of scoreterms and weights do not match, ensuring correct input.
    - The optional scaling step ensures that the composite score remains within a standardized range.

    """
    if not len(scoreterms) == len(weights):
        raise ValueError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")

    df = df.copy()
    for col in scoreterms:
        # check if column contains only floats or integers, raise an error otherwise
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Column {col} must only contain float or integers!")

        # normalize scoreterm
        df[col] = normalize_series(ser=df[col], scale=False)

    # combine weighted scores
    combined_col = sum((df[col]*weight for col, weight in zip(scoreterms, weights)))
    return scale_series(combined_col) if scale else combined_col

def get_format(path: str):
    """
    Returns the appropriate pandas function to load a file based on its extension.

    Parameters
    ----------
    path : str
        The path to the file whose format needs to be determined.

    Returns
    -------
    function
        The pandas function corresponding to the file format (e.g., pd.read_json, pd.read_csv).

    Further Details
    ---------------
    This function determines the appropriate pandas function to use for loading a file based on its extension. It supports various file formats, including JSON, CSV, Pickle, Feather, and Parquet.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import get_format

        # Determine the format function for a JSON file
        load_function = get_format('path/to/data.json')

        # Use the function to load the data
        df = load_function('path/to/data.json')

    Notes
    -----
    - Raises a KeyError if the file format is not supported.
    - Ensures that the appropriate pandas function is returned based on the file extension.

    """
    loading_function_dict = {
        "json": pd.read_json,
        "csv": pd.read_csv,
        "pickle": pd.read_pickle,
        "feather": pd.read_feather,
        "parquet": pd.read_parquet
    }
    return loading_function_dict[path.split(".")[-1]]

def load_poses(poses_path: str) -> Poses:
    """
    Loads poses from a specified file and returns a Poses instance.

    Parameters
    ----------
    poses_path : str
        The path to the file containing the poses to be loaded.

    Returns
    -------
    Poses
        A Poses instance with poses loaded from the specified file.

    Further Details
    ---------------
    This function reads a file containing poses and returns a Poses instance with the data. The file format is automatically detected based on the file extension, and the corresponding loading function is used to read the data into a DataFrame.

    Example
    -------
    .. code-block:: python

        from poses import Poses, load_poses

        # Load poses from a file
        poses_instance = load_poses('path/to/poses.json')

    Notes
    -----
    - The function supports various file formats, including JSON, CSV, Pickle, Feather, and Parquet.
    - Ensures that the loaded DataFrame contains the necessary columns and updates the Poses instance accordingly.

    """
    return Poses().load_poses(poses_path)

def col_in_df(df: pd.DataFrame, column: str|list[str]) -> None:
    """
    Checks if the specified column(s) exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked.
    column : str or list[str]
        The column name or list of column names to check for existence in the DataFrame.

    Raises
    ------
    KeyError
        If any of the specified columns are not found in the DataFrame.

    Further Details
    ---------------
    This function checks whether the specified column or list of columns exist in the given DataFrame. It is useful for ensuring that the DataFrame contains the necessary columns before performing further operations.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import col_in_df

        # Create a sample DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        # Check if a column exists
        col_in_df(df, 'col1')

        # Check if multiple columns exist
        col_in_df(df, ['col1', 'col2'])

    Notes
    -----
    - The function raises a KeyError if any of the specified columns are not found in the DataFrame.
    - Ensures that the DataFrame contains the necessary columns for subsequent operations.

    """
    if isinstance(column, list):
        for col in column:
            if not col in df.columns:
                raise KeyError(f"Could not find {col} in poses dataframe! Are you sure you provided the right column name?")
    else:
        if not column in df.columns:
            raise KeyError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")

def filter_dataframe_by_rank(df: pd.DataFrame, col: str, n: float|int, remove_layers: int = None, layer_col: str = "poses_description", sep: str = "_", ascending: bool = True) -> pd.DataFrame:
    """
    Filters the DataFrame to retain only the top-ranked rows based on a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be filtered.
    col : str
        The column in the DataFrame used for ranking.
    n : Union[float, int]
        The number of top-ranked rows to retain. If n < 1, it represents a fraction of the total rows.
    remove_layers : int, optional
        The number of layers to remove from the column values before ranking. This helps in grouping similar rows.
    layer_col : str, optional
        The column used for layer-based grouping of rows (default is "poses_description").
    sep : str, optional
        The separator used in the layer descriptions (default is "_").
    ascending : bool, optional
        If True, ranks rows in ascending order; otherwise, in descending order (default is True).

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only the top-ranked rows.

    Further Details
    ---------------
    This function filters the DataFrame to retain only the top-ranked rows based on the values in a specified column. It supports fractional ranking, layer-based grouping, and sorting in ascending or descending order. The function also allows for removing layers from column values before ranking to handle grouped data.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import filter_dataframe_by_rank

        # Create a sample DataFrame
        data = {
            'poses_description': ['pose1', 'pose2', 'pose3', 'pose4', 'pose5'],
            'score': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)

        # Filter the DataFrame to retain the top 3 rows based on the score column
        filtered_df = filter_dataframe_by_rank(df, col='score', n=3)

    Notes
    -----
    - The function raises a KeyError if the specified column is not found in the DataFrame.
    - Ensures that the DataFrame is properly sorted and filtered based on the provided parameters.

    """
    def determine_filter_n(df: pd.DataFrame, n: float) -> int:
        '''
        determines if n is a fraction or an integer and sets cutoff for dataframe filtering accordingly.
        '''
        filter_n = float(n)
        if filter_n < 1:
            filter_n = round(len(df) * filter_n)
        elif filter_n <= 0:
            raise ValueError(f"ERROR: Argument <n> of filter functions cannot be smaller than 0. It has to be positive number. If n < 1, the top n fraction is taken from the DataFrame. if n > 1, the top n rows are taken from the DataFrame")

        return int(filter_n)

    # make sure <col> exists columns in <df>
    col_in_df(df, col)

    # if remove_layers is set, compile list of unique pose descriptions after removing one index layer:
    if remove_layers:
        if not isinstance(remove_layers, int):
            raise TypeError(f"ERROR: only value of type 'int' allowed for remove_layers. You set it to {type(remove_layers)}")

        # make sure <layer_col> exists in df
        col_in_df(df, layer_col)

        # create temporary description column with removed index layers
        df["tmp_layer_column"] = df[layer_col].str.split(sep).str[:-1*int(remove_layers)].str.join(sep)

        # group by temporary description column, filter top n rows per group
        filtered = []
        for _, group_df in df.groupby("tmp_layer_column", sort=False):
            filtered.append(group_df.sort_values(by=col, ascending=ascending).head(determine_filter_n(group_df, n)))
        filtered_df = pd.concat(filtered).reset_index(drop=True)

        #drop temporary description column
        filtered_df.drop("tmp_layer_column", axis=1, inplace=True)

    else:
        filtered_df = df.sort_values(by=col, ascending=ascending).head(determine_filter_n(df, n))

    return filtered_df


def filter_dataframe_by_value(df: pd.DataFrame, col: str, value: float|int, operator: str) -> pd.DataFrame:
    """
    Filters the DataFrame based on a specified value in a column using the provided comparison operator.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be filtered.
    col : str
        The column in the DataFrame used for filtering.
    value : Union[float, int]
        The value used as the threshold for filtering rows.
    operator : str
        The comparison operator used for filtering ('>', '>=', '<', '<=', '=', '!=').

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only the rows that meet the filtering criteria.

    Further Details
    ---------------
    This function filters the DataFrame based on a specified value in a column, using the provided comparison operator. It supports various comparison operators such as greater than, less than, equal to, and not equal to.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from poses import filter_dataframe_by_value

        # Create a sample DataFrame
        data = {
            'poses_description': ['pose1', 'pose2', 'pose3', 'pose4', 'pose5'],
            'score': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)

        # Filter the DataFrame to retain rows where the score is greater than 30
        filtered_df = filter_dataframe_by_value(df, col='score', value=30, operator='>')

    Notes
    -----
    - The function raises a KeyError if the specified column is not found in the DataFrame.
    - Ensures that the DataFrame is properly filtered based on the provided criteria.

    """
    # make sure <col> exists columns in <df>
    col_in_df(df, col)

    # Define the comparison based on the operator
    if operator == '>': filtered_df = df[df[col] > value]
    elif operator == '>=': filtered_df = df[df[col] >= value]
    elif operator == '<': filtered_df = df[df[col] < value]
    elif operator == '<=': filtered_df = df[df[col] <= value]
    elif operator == '=': filtered_df = df[df[col] == value]
    elif operator == '!=': filtered_df = df[df[col] != value]
    else:
        raise KeyError("Invalid operator. Supported operators are '>','>=', '<', '<=', '=', '!='.")

    return filtered_df

def description_from_path(path:str):
    description = os.path.splitext(os.path.basename(path))[0]
    return description