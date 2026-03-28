"""
Caliby Runner Module
====================
 
.. module:: protflow.runners.caliby
   :synopsis: ProtFlow runner interfaces for the Caliby protein sequence design
              and conformational ensemble generation toolkit.
 
This module provides ProtFlow-compatible :class:`Runner` subclasses that wrap
the Caliby suite of deep-learning models for structure-conditioned sequence 
design and backbone ensemble generation.  Three public runners are exposed:
 
* :class:`CalibySequenceDesign` — single-structure sequence design via Caliby's
  AtomMPNN-based pipeline.
* :class:`CalibyEnsembleGenerator` — backbone ensemble generation via the
  ProtPardelle partial-diffusion model bundled inside Caliby.
* :class:`CalibyEnsembleSeqDesign` — end-to-end pipeline that first generates a
  conformational ensemble (or uses pregenerated ones) and then runs ensemble-aware 
  sequence design on it.
 
All three classes inherit from the private base class :class:`_CalibyRunner`,
which provides shared utilities for option parsing, batch setup, and constraint
CSV creation.
 
Configuration
-------------
The runners read their environment from the ProtFlow configuration file
(``ProtFlow/protflow/config.py`` by default).  The following keys are
relevant:
 
``CALIBY_DIR_PATH``
    Absolute path to the root of the Caliby installation directory.
 
``CALIBY_PYTHON_PATH``
    Absolute path to the Python interpreter inside the Caliby conda
    environment.
 
``CALIBY_PRE_CMD``
    Optional shell preamble executed before every Caliby command (e.g. a
    ``source setup_env.sh`` or ``module load`` statement).
 
Examples
--------
Minimal sequence design run::
 
    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.runners.caliby import CalibySequenceDesign
 
    poses = Poses("pdbs/", prefix="design")
    jobstarter = SbatchArrayJobstarter(max_cores=50)
 
    runner = CalibySequenceDesign()
    poses = runner.run(poses, prefix="caliby_sd", nseq=5, model="caliby")
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
    """
    _CalibyRunner Class
    ===================

    Private base class shared by all Caliby runners.
    :class:`_CalibyRunner` inherits from :class:`~protflow.runners.Runner`
    and provides the common infrastructure needed to launch Caliby
    sub-scripts: configuration resolution, option parsing, batch job
    preparation, positional-constraint CSV creation, and command
    construction.  It is not intended to be instantiated directly.
 
    Parameters
    ----------
    caliby_dir : str, optional
        Absolute path to the root of the Caliby installation.  When omitted
        the value is read from the ProtFlow config key ``CALIBY_DIR_PATH``.
    python_path : str, optional
        Path to the Python interpreter to use when calling Caliby scripts.
        Defaults to the config key ``CALIBY_PYTHON_PATH``.
    pre_cmd : str, optional
        Shell preamble prepended to every generated command (e.g.
        ``"conda activate caliby_env &&"``).  Defaults to
        ``CALIBY_PRE_CMD`` from the ProtFlow config, or ``None`` if
        that key is absent.
    model_dir : str, optional
        Directory that contains Caliby model checkpoint files
        (``*.ckpt``).  Defaults to ``<caliby_dir>/model_params``.
    jobstarter : str, optional
        Default :class:`~protflow.jobstarters.JobStarter` to use when
        :meth:`run` is called without an explicit *jobstarter* argument.
 
    Notes
    -----
    Configuration resolution uses a "first wins" priority:
    explicitly-passed constructor arguments > ProtFlow config values.
    Missing mandatory config values (``CALIBY_DIR_PATH``,
    ``CALIBY_PYTHON_PATH``) raise an error at instantiation time via
    :func:`~protflow.require_config` and
    :func:`~protflow.load_config_path`.
    """

    def __init__(self, caliby_dir: str = None, python_path: str = None, pre_cmd: str = None, model_dir: str = None, jobstarter: JobStarter = None) -> None:
        # setup config
        config = require_config()
        self.caliby_dir = caliby_dir or load_config_path(config, "CALIBY_DIR_PATH")
        self.model_dir = model_dir or os.path.join(self.caliby_dir, "model_params")
        self.python_path = python_path or load_config_path(config, "CALIBY_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "CALIBY_PRE_CMD", is_pre_cmd=True)
        self.jobstarter = jobstarter
        self.script_path = None

        # setup runner
        self.name = "caliby.py"

    def __str__(self):
        return "caliby.py"

    def create_constraint_csv(self, poses: Poses, work_dir: str, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None) -> str:
        """
        create_constraint_csv Method
        ============================

        Build a per-pose positional-constraint CSV and write it to disk.
        Assembles a ``pos_constraints.csv`` file inside *work_dir* that
        maps every pose description to its design constraints.  Only the
        columns for which a corresponding DataFrame column name is supplied
        are included in the output CSV.  Empty cells are filled with an
        empty string so that Caliby can parse the file without errors.
 
        Parameters
        ----------
        poses : Poses
            The current collection of poses.  The pose descriptions are
            derived from :func:`~protflow.poses.description_from_path`
            applied to each element of
            :meth:`~protflow.poses.Poses.poses_list`.
        work_dir : str
            Directory into which ``pos_constraints.csv`` is written.  The
            directory must already exist.
        fixed_pos_seq_col : str, optional
            Name of the column in ``poses.df`` containing residue positions
            whose *sequence* should be held fixed.  Values may be
            :class:`~protflow.residues.ResidueSelection` objects or plain
            strings in Caliby's residue selection syntax (e.g.
            ``"A5-15,B3"``).
        fixed_pos_scn_col : str, optional
            Name of the column containing residue positions for which
            *side-chain* coordinates should be fixed (the sequence at
            these positions is also implicitly fixed).  Values follow the
            same format as *fixed_pos_seq_col*.
        fixed_pos_override_seq_col : str, optional
            Name of the column containing per-pose sequence overrides.
            These residues are forced to a specific amino-acid identity
            regardless of the model's prediction.
        pos_restrict_aatype_col : str, optional
            Name of the column that restricts which amino-acid types are
            allowed at specific positions (e.g. ``"A5:[HKR],A10:[DE]"``).
        symmetry_pos_col : str, optional
            Name of the column encoding symmetric position groups that
            should receive identical sequence assignments.
 
        Returns
        -------
        str
            Absolute path to the written ``pos_constraints.csv`` file.
 
        Raises
        ------
        KeyError
            If any of the supplied column names are absent from
            ``poses.df`` (raised internally by
            :func:`~protflow.runners.col_in_df`).
 
        Notes
        -----
        * :class:`~protflow.residues.ResidueSelection` objects are
          converted to their string representation via
          :meth:`~protflow.residues.ResidueSelection.to_string` before
          being written.
        * The ``pdb_key`` column always contains only the *stem* of the
          pose filename (no directory prefix, no extension), matching the
          convention used by Caliby's input reader.
        * When ``fixed_pos_scn_col`` is provided, the implementation
          currently reads from ``fixed_pos_seq_col`` for the
          ``fixed_pos_scn`` column as well (see source comment for the
          pending TODO).
 
        Examples
        --------
        ::
 
            csv_path = runner.create_constraint_csv(
                poses=poses,
                work_dir="/scratch/my_run",
                fixed_pos_seq_col="fixed_residues",
                pos_restrict_aatype_col="aa_restrictions",
            )
            # csv_path == "/scratch/my_run/pos_constraints.csv"
        """

        # create df from input poses
        cst_csv = pd.DataFrame({"pdb_key": [description_from_path(pose) for pose in poses.poses_list()]})

        # add constraints to df
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

        # fill all nans/None with empty string
        cst_csv = cst_csv.fillna('')

        # save csv
        cst_csv.to_csv(out := os.path.join(work_dir, "pos_constraints.csv"), index=False)
        return os.path.abspath(out)

    def setup_batch_mode(self, pose_paths: list[str], options: dict, num_batches: int, work_dir: str, mode="single") -> list:
        """
        setup_batch_mode Method
        =======================

        Partition poses into batches and prepare per-batch option dicts.
        Splits *pose_paths* into *num_batches* sublists, writes a plain-text
        input-list file for each batch, and returns a list of option
        dictionaries—one per batch—with the batch-specific keys
        ``input_cfg.pdb_name_list`` and ``out_dir`` already set.
 
        If the input poses reside in more than one directory they are first
        copied into a shared ``<work_dir>/input/`` directory so that Caliby
        can locate them all via a single ``pdb_dir`` path.
 
        Parameters
        ----------
        pose_paths : list of str
            Absolute (or resolvable) paths to all input PDB files.
        options : dict
            Base option dictionary that is copied and augmented for each
            batch.  Keys use Caliby's dotted-path syntax (e.g.
            ``"input_cfg.pdb_dir"``).
        num_batches : int
            Number of batches to create.  Must be ≥ 1 and ≤
            ``len(pose_paths)``.
        work_dir : str
            Root working directory.  Batch output directories
            (``batch_0/``, ``batch_1/``, …) and the
            ``input_lists/`` sub-directory are created here.
        mode : {"single", "ensemble"}, optional
            Controls how the ``pdb_dir`` key is populated:
 
            * ``"single"`` — sets ``input_cfg.pdb_dir`` to the common
              input directory.
            * ``"ensemble"`` — omits ``input_cfg.pdb_dir`` (the caller
              is expected to set ``input_cfg.conformer_dir`` instead).
 
            Default is ``"single"``.
 
        Returns
        -------
        list of dict
            One option dictionary per batch.  Each dictionary is an
            independent copy of *options* augmented with:
 
            ``input_cfg.pdb_name_list``
                Path to the text file listing the basenames of poses in
                this batch.
            ``out_dir``
                Batch-specific output directory path
                (``<work_dir>/batch_<i>/``).
 
        Notes
        -----
        * Input-list files contain *basenames only* (no directory prefix)
          because Caliby resolves filenames relative to ``pdb_dir``.
        * Pose-to-batch assignment uses
          :func:`~protflow.jobstarters.split_list`, which distributes
          poses as evenly as possible.
        * Files are copied with :func:`shutil.copy`; no renaming occurs, so
          filename collisions across source directories will silently
          overwrite each other in the shared input directory.
 
        Examples
        --------
        ::
 
            batch_opts = runner.setup_batch_mode(
                pose_paths=poses.poses_list(),
                options={"sampling_cfg_overrides.num_seqs_per_pdb": 5},
                num_batches=4,
                work_dir="/scratch/my_run",
                mode="single",
            )
            # len(batch_opts) == 4
            # batch_opts[0]["out_dir"] == "/scratch/my_run/batch_0"
        """

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
        """
        parse_caliby_opts Method
        ========================

        Parse a Caliby options string into a key-value dictionary.
        Splits a whitespace-delimited string of ``key=value`` tokens
        (respecting single- and double-quoted values) into a Python
        dictionary suitable for further manipulation before being passed
        to :meth:`write_cmd`.
 
        Parameters
        ----------
        options : str, optional
            Raw Caliby option string in the form
            ``"key1=value1 key2='value with spaces' key3=value3"``.
            Tokens that do not contain ``=`` are silently ignored.
            Passing ``None`` or an empty string returns an empty dict.
 
        Returns
        -------
        dict
            Mapping of option keys to their string values.  Surrounding
            single or double quotes are stripped from values.
 
        Notes
        -----
        * The regex used for splitting respects quoted substrings, so
          values containing spaces can be passed as
          ``key='value with spaces'``.
        * Only the *first* ``=`` in a token is used as the separator,
          allowing values that themselves contain ``=`` signs.
        * The returned dictionary uses the same dotted-path keys that
          Caliby's Hydra-based CLI expects (e.g.
          ``"sampling_cfg_overrides.num_seqs_per_pdb"``).
 
        Examples
        --------
        ::
 
            opts = runner.parse_caliby_opts(
                "sampling_cfg_overrides.num_seqs_per_pdb=10 "
                "ckpt_name_or_path=/models/caliby.ckpt"
            )
            # opts == {
            #     "sampling_cfg_overrides.num_seqs_per_pdb": "10",
            #     "ckpt_name_or_path": "/models/caliby.ckpt",
            # }
 
            runner.parse_caliby_opts(None)
            # {}
        """

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
        """
        write_cmd Method
        ================
        
        Compose the full shell command string for a single Caliby invocation.
        Combines :attr:`python_path`, :attr:`script_path`, and the
        provided option dictionary into a single executable command string.
 
        Parameters
        ----------
        options : dict
            Option key-value pairs in Caliby's dotted-path format.  Values
            are converted to strings and joined without extra separators
            between key and value (``key=value`` syntax).
 
        Returns
        -------
        str
            A shell-ready command such as::
 
                /env/caliby/bin/python /opt/caliby/caliby/eval/sampling/seq_des.py \
                    sampling_cfg_overrides.num_seqs_per_pdb=5 \
                    ckpt_name_or_path=/models/caliby.ckpt
 
        Notes
        -----
        The conversion of *options* to a string is delegated to
        :func:`~protflow.runners.options_flags_to_string` with
        ``sep=""`` (no whitespace between key and ``=`` and value).
 
        Examples
        --------
        ::
 
            cmd = runner.write_cmd({
                "sampling_cfg_overrides.num_seqs_per_pdb": 5,
                "ckpt_name_or_path": "/models/caliby.ckpt",
            })
        """

        # convert to string
        options = options_flags_to_string(options, None, sep="")
        return f"{self.python_path} {self.script_path} {options}"

class CalibySequenceDesign(_CalibyRunner):
    """
    CalibySequenceDesign Class
    ==========================
    
    Runner for single-structure sequence design using Caliby's AtomMPNN pipeline.
    Wraps ``caliby/eval/sampling/seq_des.py`` and exposes ProtFlow's standard
    :meth:`run` interface.  Given a collection of input PDB structures,
    :class:`CalibySequenceDesign` produces *nseq* designed sequences per
    structure (optionally threaded back onto the backbone as CIF/PDB files)
    and returns an updated :class:`~protflow.poses.Poses` object augmented
    with Caliby's scoring columns.
 
    Parameters
    ----------
    caliby_dir : str, optional
        Root of the Caliby installation.  Resolved from the ProtFlow config
        key ``CALIBY_DIR_PATH`` when omitted.
    python_path : str, optional
        Python interpreter for the Caliby environment.  Resolved from
        ``CALIBY_PYTHON_PATH`` when omitted.
    model_dir : str, optional
        Directory containing model checkpoint files.  Defaults to
        ``<caliby_dir>/model_params``.
    pre_cmd : str, optional
        Optional shell preamble (e.g. environment activation) prepended to
        every command.  Resolved from ``CALIBY_PRE_CMD`` when omitted.
    jobstarter : JobStarter, optional
        Default :class:`~protflow.jobstarters.JobStarter` to use when
        :meth:`run` is called without an explicit *jobstarter* argument.
 
    Attributes
    ----------
    script_path : str
        Absolute path to ``caliby/eval/sampling/seq_des.py`` inside the
        Caliby installation.
    sampling_cfg : str
        Absolute path to the AtomMPNN inference YAML config
        (``caliby/configs/seq_des/atom_mpnn_inference.yaml``).  Injected
        into every run to ensure Caliby resolves its config relative to the
        correct installation directory.
    index_layers : int
        Number of index layers added to pose descriptions by this runner
        (``1``).
    name : str
        Runner identifier (``"caliby.py"``).
 
    Notes
    -----
    * Output poses are indexed with one additional layer (e.g.
      ``design_0001_caliby_sd_0001``).
    * The runner caches results in a scorefile
      ``caliby_seq_des_scores.<format>`` inside *work_dir*.  If this file
      already exists and *overwrite* is ``False``, the previous results are
      returned immediately without re-running Caliby.
    * When *return_seq_threaded_pdbs_as_pose* is ``False`` (default), the
      ``location`` column of the returned :class:`~protflow.poses.Poses`
      points to a ``.fasta`` file; otherwise it points to a PDB/CIF
      structure file.
 
    Examples
    --------
    Basic usage::
 
        from protflow.poses import Poses
        from protflow.jobstarters import SbatchArrayJobstarter
        from protflow.runners.caliby import CalibySequenceDesign
 
        poses = Poses("input_pdbs/", prefix="scaffold")
        jobstarter = SbatchArrayJobstarter(max_cores=50)
        runner = CalibySequenceDesign()
 
        poses = runner.run(
            poses,
            prefix="sd",
            nseq=10,
            omit_aas="CM",
            fixed_pos_seq_col="fixed_residues",
            jobstarter=jobstarter,
        )
 
    With a custom model checkpoint::
 
        runner = CalibySequenceDesign(model_dir="/path/to/custom_models")
        poses = runner.run(
            poses,
            prefix="custom_sd",
            nseq=5,
            model="/absolute/path/to/my_model.ckpt",
        )
    """
    def __init__(self, caliby_dir: str = None, python_path: str = None, model_dir: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:

        super().__init__(
            caliby_dir=caliby_dir, 
            python_path=python_path, 
            pre_cmd=pre_cmd, 
            model_dir=model_dir,
            jobstarter=jobstarter
        )
        
        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/seq_des.py")
        self.sampling_cfg = os.path.join(self.caliby_dir, "caliby/configs/seq_des/atom_mpnn_inference.yaml")
        
        self.index_layers = 1


    def run(self, poses: Poses, prefix: str, nseq: int = 1, model: str = "caliby", omit_aas: str|list = None, fixed_pos_seq_col: str = None, fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None, pos_constraint_csv: str = None, return_seq_threaded_pdbs_as_pose: bool = False, options: str = None, cif_to_pdb: bool = True, jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        run Method
        ==========

        Run Caliby sequence design on a collection of poses.
        For each input structure, generates *nseq* designed amino-acid
        sequences using Caliby's AtomMPNN-based model.  Sequences can be
        returned as FASTA files (default) or as sequence-threaded PDB
        structures.  The method handles batching, job submission, output
        collection, and scorefile caching automatically.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  Each pose must be a valid PDB file.
        prefix : str
            Column prefix used to namespace all new columns added to
            ``poses.df`` and to name the working directory
            (``<poses.work_dir>/<prefix>/``).
        nseq : int, optional
            Number of sequences to design per input structure.
            Default is ``1``.
        model : str, optional
            Either a named model alias (looked up as
            ``<model_dir>/caliby/<model>.ckpt``) or an absolute path to a
            ``.ckpt`` checkpoint file.  Default is ``"caliby"``.
        omit_aas : str or list of str, optional
            Amino-acid types (one-letter codes) to exclude from sampling.
            Can be provided as a concatenated string (``"CM"``), which is
            split into individual characters, or as an explicit list
            (``["C", "M"]``).
        fixed_pos_seq_col : str, optional
            ``poses.df`` column with residue positions whose sequence is
            fixed.  Accepts :class:`~protflow.residues.ResidueSelection`
            objects or Caliby residue-selection strings.
        fixed_pos_scn_col : str, optional
            ``poses.df`` column with positions whose side-chain
            coordinates are fixed (implies sequence fixation as well).
        fixed_pos_override_seq_col : str, optional
            ``poses.df`` column with per-pose amino-acid overrides forced
            regardless of model predictions (e.g. ``"A26:A,A27:L"``).
        pos_restrict_aatype_col : str, optional 
            ``poses.df`` column specifying per-position amino-acid type
            restrictions in Caliby's bracket notation (e.g.
            ``"A26:AVG,A27:VG"``).
        symmetry_pos_col : str, optional
            ``poses.df`` column encoding symmetric residue groups that
            must share a sequence identity in the format ``"A10,B10,C10|A11,B11,C11"``.
        pos_constraint_csv : str, optional
            Absolute path to a pre-generated constraints CSV file.  When
            provided, all per-column constraint parameters
            (*fixed_pos_seq_col*, etc.) must be ``None`` to avoid
            ambiguity.
        return_seq_threaded_pdbs_as_pose : bool, optional
            If ``True``, the returned poses point to sequence-threaded PDB
            files instead of FASTA files.  Default is ``False``.
        options : str, optional
            Raw Caliby option string (``"key=value key2=value2"``).
            Parsed by :meth:`parse_caliby_opts` and merged with
            programmatically set options; explicitly set parameters
            (``nseq``, ``model``, etc.) take priority.
        cif_to_pdb : bool, optional
            When ``True`` (default), Caliby's CIF output files are
            converted to PDB format via OpenBabel before being recorded
            as output poses.
        jobstarter : JobStarter, optional
            Job submission backend for this run.  Falls back to
            ``self.jobstarter`` and then to
            ``poses.default_jobstarter`` in that order.
        overwrite : bool, optional
            If ``False`` (default), an existing scorefile causes the run
            to be skipped and the cached results to be returned.  Set to
            ``True`` to force re-computation.
        num_batches : int, optional
            Override the number of parallel batches.  Defaults to
            ``min(len(poses), jobstarter.max_cores)``. Not identical with
            caliby batch setting!
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns
            prefixed by *prefix*, including:
 
            ``<prefix>_location``
                Path to the output FASTA or PDB file for each designed
                sequence.
            ``<prefix>_description``
                Unique identifier for each output sequence.
            ``<prefix>_seq``
                Designed amino-acid sequence (one-letter code).
            Additional Caliby scoring columns as produced by
            ``seq_des_outputs.csv``.
 
        Raises
        ------
        ValueError
            If *pos_constraint_csv* does not point to an existing file.
        ValueError
            If *pos_constraint_csv* is set (either directly or via
            *options*) simultaneously with any per-column constraint
            parameter.
        FileNotFoundError
            If *model* is neither a recognisable alias in *model_dir* nor
            a path to an existing file.
        RuntimeError
            If the number of collected output poses is smaller than
            ``len(poses) * nseq``, indicating that one or more Caliby
            jobs crashed.
 
        Notes
        -----
        * The number of jobs submitted equals *num_batches* (or
          ``min(len(poses), jobstarter.max_cores)`` by default).  Increasing
          *num_batches* reduces wall-clock time on large clusters.
        * Setting *cif_to_pdb* to ``False`` keeps Caliby's native CIF
          output and avoids an OpenBabel dependency, but downstream
          ProtFlow runners may not accept CIF files.
        * The ``+`` prefix on the ``omit_aas`` Hydra override key is
          required when the list is appended rather than overriding an
          existing config node; this is handled automatically.
 
        Examples
        --------
        Design 20 sequences per structure, excluding cysteine and
        methionine, fixing the active-site residues::
 
            poses = runner.run(
                poses=poses,
                prefix="round1",
                nseq=20,
                omit_aas="CM",
                fixed_pos_seq_col="active_site",
                jobstarter=SbatchArrayJobstarter(max_cores=100),
            )
 
        Return sequence-threaded PDB files instead of FASTAs::
 
            poses = runner.run(
                poses=poses,
                prefix="threaded",
                nseq=5,
                return_seq_threaded_pdbs_as_pose=True,
            )
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

        # check for pos_constraint_csv file
        if pos_constraint_csv and not os.path.isfile(pos_constraint_csv):
            raise ValueError(f"<pos_constraint_csv> must specify the path to a single csv file. Could not find a file at {pos_constraint_csv}.")

        # check for model at model dir or at path
        if not os.path.isfile(model) and not os.path.isfile(model_path := os.path.join(self.model_dir, "caliby", f"{model}.ckpt")):
            raise FileNotFoundError(f"Could not detect a model at {model} or at {model_path}.")

        # parse options, set inputs
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

        # set position-specific constraint csv
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
    """
    CalibyEnsembleGenerator Class
    =============================

    Runner for backbone conformational ensemble generation via ProtPardelle.
    Wraps ``caliby/eval/sampling/generate_ensembles.py``, which uses the
    ProtPardelle partial-diffusion model distributed with Caliby to
    generate structurally diverse backbone conformers starting from a
    single input structure.
 
    This runner is primarily used as an internal component of
    :class:`CalibyEnsembleSeqDesign`, but can also be called independently
    when backbone diversity sampling is the end goal.
 
    Parameters
    ----------
    caliby_dir : str, optional
        Root of the Caliby installation.  Resolved from ``CALIBY_DIR_PATH``
        when omitted.
    model_dir : str, optional
        Directory containing model checkpoints.  Defaults to
        ``<caliby_dir>/model_params``.
    python_path : str, optional
        Python interpreter path.  Resolved from ``CALIBY_PYTHON_PATH`` when
        omitted.
    pre_cmd : str, optional
        Shell preamble prepended to commands.  Resolved from
        ``CALIBY_PRE_CMD`` when omitted.
    jobstarter : JobStarter, optional
        Default job submission backend.
 
    Notes
    -----
    Output conformers are written as individual PDB files inside a per-pose
    subdirectory.  The ``sample_`` prefix added by ProtPardelle is stripped
    from filenames during collection; the original input structure (which
    is also written by Caliby) is excluded from the output collection.
 
    Examples
    --------
    ::
 
        from protflow.runners.caliby import CalibyEnsembleGenerator
 
        runner = CalibyEnsembleGenerator()
        poses = runner.run(poses, prefix="ens", nstruct=16)
        # poses.df["ens_conformer_dir"] contains the conformer directory
        # for each input structure.
    """

    def __init__(self, caliby_dir: str = None, model_dir: str = None, python_path: str = None, pre_cmd: str = None, jobstarter: JobStarter = None) -> None:

        super().__init__(
            caliby_dir=caliby_dir, 
            python_path=python_path, 
            pre_cmd=pre_cmd, 
            model_dir=model_dir,
            jobstarter=jobstarter
        )

        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/generate_ensembles.py")

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_yaml_path = os.path.join(self.caliby_dir, "caliby/configs/protpardelle-1c/multichain_backbone_partial_diffusion.yaml")

        # setup runner
        self.index_layers = 1

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, options: str = None, cif_to_pdb: bool = True, model_dir: str = None, jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None) -> Poses:
        """
        run Method
        ==========

        Generate a backbone conformational ensemble for each input pose.
        Runs Caliby's ProtPardelle-based partial-diffusion script to
        produce *nstruct* structurally diverse backbone conformers per input
        structure.  Results are stored in per-pose subdirectories and
        recorded in the returned :class:`~protflow.poses.Poses` object.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  Each pose must be a PDB file.
        prefix : str
            Column prefix and working-directory identifier.
        nstruct : int, optional
            Number of backbone conformers to generate per input structure.
            Default is ``1``.
        options : str, optional
            Raw Caliby option string passed verbatim to
            :meth:`parse_caliby_opts` and merged with programmatic
            options.
        cif_to_pdb : bool, optional
            Convert any CIF output to PDB via OpenBabel.  Default is
            ``True``.
        model_dir : str, optional
            Override the model directory for this run only.  Falls back to
            ``self.model_dir`` when omitted.
        jobstarter : JobStarter, optional
            Job submission backend.  Resolved via the standard fallback
            chain (argument → ``self.jobstarter`` → ``poses.default_jobstarter``).
        overwrite : bool, optional
            Re-run even if a scorefile already exists.  Default is ``False``.
        num_batches : int, optional
            Number of parallel batches.  Defaults to
            ``min(len(poses), jobstarter.max_cores)``.
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns
            prefixed by *prefix*:
 
            ``<prefix>_location``
                Path to each generated conformer PDB file.
            ``<prefix>_description``
                Conformer identifier derived from the filename stem.
            ``<prefix>_input_description``
                Description of the parent (input) pose.
            ``<prefix>_input_path``
                Path to the primary conformer (the original input
                structure written by Caliby alongside the samples).
            ``<prefix>_conformer_dir``
                Directory containing all conformers for a given input
                pose.
 
        Raises
        ------
        RuntimeError
            If the number of collected conformers is less than
            ``len(poses) * nstruct``, indicating crashed jobs.
 
        Notes
        -----
        * The primary input structure that ProtPardelle writes into the
          output directory (named ``<input_description>.pdb``) is
          **excluded** from the collected output poses to avoid
          conflating the input with generated samples.
        * ``sample_`` prefixes are automatically removed from conformer
          filenames during collection.
        * Because the scorefile key is ``caliby_seq_des_scores`` (shared
          with :class:`CalibySequenceDesign`), running both runners with
          the same *prefix* in the same working directory may cause
          scorefile collisions.  Use distinct prefixes.
 
        Examples
        --------
        Generate 16 conformers per pose::
 
            runner = CalibyEnsembleGenerator()
            poses = runner.run(
                poses,
                prefix="backbone_ens",
                nstruct=16,
                jobstarter=SbatchArrayJobstarter(max_cores=50),
            )
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
    """
    CalibyEnsembleSeqDesign Class
    =============================

    End-to-end runner combining ensemble generation with ensemble-aware sequence design.
    Wraps ``caliby/eval/sampling/seq_des_ensemble.py`` and orchestrates a
    two-stage pipeline:
 
    1. **Ensemble generation** (optional) — runs
       :class:`CalibyEnsembleGenerator` internally via
       :meth:`run_protpardelle_ensemble_generation` to produce backbone
       conformers for each input structure. Otherwise, pregenerated conformers
       can be specified.
    2. **Ensemble sequence design** — runs Caliby's ensemble-aware
       AtomMPNN variant that conditions sequence predictions on the full
       conformational ensemble rather than a single structure.
 
    This runner is the recommended entry point when structural flexibility
    should inform the designed sequence.
 
    Parameters
    ----------
    caliby_dir : str, optional
        Root of the Caliby installation.  Resolved from ``CALIBY_DIR_PATH``
        when omitted.
    python_path : str, optional
        Python interpreter.  Resolved from ``CALIBY_PYTHON_PATH`` when
        omitted.
    pre_cmd : str, optional
        Shell preamble for commands.  Resolved from ``CALIBY_PRE_CMD`` when
        omitted.
    model_dir : str, optional
        Model checkpoint directory.  Defaults to
        ``<caliby_dir>/model_params``.
    jobstarter : JobStarter, optional
        Default job submission backend.
 
    Notes
    -----
    When *generate_ensembles* is ``True`` (default), an internal
    :class:`CalibyEnsembleGenerator` run is performed with the prefix
    ``<prefix>_auto_ensgen``.  The resulting ``conformer_dir`` column is
    then used automatically; any explicitly provided *conformer_col* is
    ignored (with a logged warning).
 
    Examples
    --------
    Fully automatic pipeline (ensemble generation + design)::
 
        from protflow.runners.caliby import CalibyEnsembleSeqDesign
 
        runner = CalibyEnsembleSeqDesign()
        poses = runner.run(
            poses,
            prefix="ens_sd",
            nseq=10,
            gen_num_ensembles=16,
        )
 
    Using pre-computed conformers::
 
        runner = CalibyEnsembleSeqDesign()
        poses = runner.run(
            poses,
            prefix="ens_sd",
            generate_ensembles=False,
            conformer_col="precomputed_conformer_dir",
            nseq=5,
        )
    """
 

    def __init__(self, caliby_dir: str = None, python_path: str = None, pre_cmd: str = None, model_dir: str = None, jobstarter: JobStarter = None) -> None:

        super().__init__(
            caliby_dir=caliby_dir, 
            python_path=python_path, 
            pre_cmd=pre_cmd, 
            model_dir=model_dir,
            jobstarter=jobstarter
        )

        self.script_path = os.path.join(self.caliby_dir, "caliby/eval/sampling/seq_des_ensemble.py")

        # TODO: find a better way, but otherwise caliby will look in wrong directory because of relative paths
        self.sampling_cfg = os.path.join(self.caliby_dir, "caliby/configs/seq_des/atom_mpnn_inference.yaml")

        self.index_layers = 1

    def __str__(self):
        return "caliby.py"

    def run(self, poses: Poses, prefix: str, generate_ensembles: bool = True, gen_num_ensembles: int = 16, gen_ens_options: str = None,
            conformer_col: str = None, nseq: int = 1, model: str = "caliby", omit_aas: str|list = None, fixed_pos_seq_col: str = None, 
            fixed_pos_scn_col: str = None, fixed_pos_override_seq_col: str = None, pos_restrict_aatype_col: str = None, symmetry_pos_col: str = None,
            pos_constraint_csv: str = None, return_seq_threaded_pdbs_as_pose: bool = False, options: str = None, cif_to_pdb: bool = True,
            jobstarter: JobStarter = None, overwrite: bool = False, num_batches: int = None, run_clean: bool = True) -> Poses:

        """Run ensemble-conditioned sequence design, optionally generating ensembles first.
 
        Executes a two-stage pipeline: (1) optional backbone ensemble generation
        using Caliby's ProtPardelle partial-diffusion model, and (2)
        ensemble-aware sequence design using Caliby's AtomMPNN variant
        that explicitly accounts for conformational flexibility when scoring
        and sampling sequences. Input poses are used as primary conformers.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  Each pose must be a PDB file
            representing the primary (reference) conformer.
        prefix : str
            Column prefix and working-directory identifier for this run.
        generate_ensembles : bool, optional
            If ``True`` (default), automatically run
            :class:`CalibyEnsembleGenerator` to produce backbone
            conformers before sequence design.  If ``False``, a pre-
            computed conformer directory must be provided via
            *conformer_col*.
        gen_num_ensembles : int, optional
            Number of backbone conformers to generate per pose when
            *generate_ensembles* is ``True``.  Default is ``16``. 
            Ignored when *generate_ensembles* is ``False``.
        gen_ens_options : str, optional
            Raw Caliby option string forwarded to the internal
            :class:`CalibyEnsembleGenerator` run. Ignored when
            *generate_ensembles* is ``False``.
        conformer_col : str, optional
            Name of the ``poses.df`` column containing pre-computed
            conformer data for each pose.  Each cell must be either:
 
            * A path (``str``) to a directory containing ``*.pdb``
              conformer files, **or**
            * A ``list`` of absolute paths to individual conformer PDB
              files.
 
            Required when *generate_ensembles* is ``False``; ignored
            (with a warning) when *generate_ensembles* is ``True``.
        nseq : int, optional
            Number of sequences to design per input pose.  Default is ``1``.
        model : str, optional
            Named model alias or absolute path to a ``.ckpt`` checkpoint.
            Default is ``"caliby"``.
        omit_aas : str or list of str, optional
            One-letter amino-acid codes to exclude from sampling.  A plain
            string is split character-by-character; a list is used as-is.
        fixed_pos_seq_col : str, optional
            ``poses.df`` column with positions whose sequence is fixed.
        fixed_pos_scn_col : str, optional
            ``poses.df`` column with positions whose side chains are fixed.
        fixed_pos_override_seq_col : str, optional
            ``poses.df`` column with forced per-position amino-acid
            identities.
        pos_restrict_aatype_col : str, optional
            ``poses.df`` column restricting allowed amino-acid types per
            position.
        symmetry_pos_col : str, optional
            ``poses.df`` column encoding symmetric residue groups.
        pos_constraint_csv : str, optional
            Path to a pre-generated constraint CSV.  Mutually exclusive
            with all ``*_col`` constraint parameters.
        return_seq_threaded_pdbs_as_pose : bool, optional
            If ``True``, output poses point to sequence-threaded PDB
            files rather than FASTA files.  Default is ``False``.
        options : str, optional
            Raw Caliby option string for the sequence-design stage.
        cif_to_pdb : bool, optional
            Convert CIF outputs to PDB via OpenBabel.  Default is ``True``.
        jobstarter : JobStarter, optional
            Job backend for both the ensemble-generation and
            sequence-design stages.  Resolved via the standard fallback
            chain.
        overwrite : bool, optional
            Force re-computation even if a cached scorefile exists.
            Default is ``False``.
        num_batches : int, optional
            Number of parallel batches for both stages.  Defaults to
            ``min(len(poses), jobstarter.max_cores)``. Not identical 
            with caliby batch option!
        run_clean : bool, optional
            If ``True`` (default), the intermediate conformer directory
            (``<work_dir>/conformers/``) is deleted after sequence design
            completes to free disk space.
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with columns prefixed
            by *prefix*, including all columns from
            :meth:`CalibySequenceDesign.run` (sequences, scores, output
            paths) and any auto-generated ensemble columns when
            *generate_ensembles* is ``True``.
 
        Raises
        ------
        ValueError
            If *generate_ensembles* is ``False`` and *conformer_col* is
            not provided.
        ValueError
            If *conformer_col* values are neither a valid directory path
            nor a list of file paths.
        ValueError
            If *pos_constraint_csv* is set simultaneously with any
            per-column constraint parameter.
        ValueError
            If *pos_constraint_csv* does not point to an existing file.
        FileNotFoundError
            If *model* cannot be resolved to an existing checkpoint file.
        RuntimeError
            If the number of collected outputs is less than
            ``len(poses) * nseq``.
 
        Notes
        -----
        * When *generate_ensembles* is ``True``, the internal ensemble-
          generation run uses the prefix ``<prefix>_auto_ensgen``.  The
          corresponding columns (e.g. ``<prefix>_auto_ensgen_conformer_dir``)
          are merged back into ``poses.df`` before sequence design begins.
        * Conformers are copied into ``<work_dir>/conformers/<pose_desc>/``
          before being passed to the Caliby ensemble script; the originals
          are not modified.
        * Disk usage can be significant when *gen_num_ensembles* is large.
          Use *run_clean* to remove the conformer directory automatically.
 
        Examples
        --------
        Full pipeline with 16 conformers and 10 sequences per pose,
        excluding cysteine::
 
            runner = CalibyEnsembleSeqDesign()
            poses = runner.run(
                poses=poses,
                prefix="ens_design",
                generate_ensembles=True,
                gen_num_ensembles=16,
                nseq=10,
                omit_aas="C",
                fixed_pos_seq_col="binding_site_residues",
                jobstarter=SbatchArrayJobstarter(max_cores=100),
            )
 
        Using pre-computed conformers stored as directory paths::
 
            # poses.df["conf_dir"] holds paths like "/data/conformers/pose_001/"
            poses = runner.run(
                poses=poses,
                prefix="ens_design",
                generate_ensembles=False,
                conformer_col="conf_dir",
                nseq=5,
            )
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
        """
        run_protpardelle_ensemble_generation Method
        ===========================================

        Run backbone ensemble generation and merge results back into the pose collection.
        A thin orchestration wrapper around :class:`CalibyEnsembleGenerator`
        that preserves the original ``poses`` column values by saving and
        restoring them across the ensemble-generation sub-run.  After
        generation, the resulting conformer-directory column is merged back
        into the original ``poses.df`` via a join on the temporary pose path
        column.
 
        This method is called automatically by :meth:`run` when
        *generate_ensembles* is ``True``.  It is exposed as a public method
        to allow fine-grained re-use in custom pipelines.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  This object is modified **in-place**
            during the ensemble-generation sub-run; a deep copy of the
            DataFrame is used internally to restore the original state.
        prefix : str
            Prefix passed to :class:`CalibyEnsembleGenerator`.  The
            conformer-directory column produced will be named
            ``<prefix>_conformer_dir``.
        nstruct : int
            Number of conformers to generate per pose, forwarded to
            :meth:`CalibyEnsembleGenerator.run`.
        options : str
            Raw Caliby option string for the ensemble-generation stage.
        num_batches : int
            Number of parallel batches.
        jobstarter : JobStarter
            Job submission backend for the ensemble-generation step.
 
        Returns
        -------
        Poses
            The input *poses* object with ``poses.df`` updated to include
            the ``<prefix>_conformer_dir`` column (and all other columns
            produced by the ensemble runner) merged alongside the original
            columns.
 
        Notes
        -----
        * A temporary column ``<prefix>_temp_poses`` is added to
          ``poses.df`` before the sub-run to facilitate the merge; this
          column is retained in the returned DataFrame.
        * After ensemble generation, only columns whose names start with
          *prefix* are kept from the sub-run result before the merge,
          avoiding column-name collisions.
        * Duplicate entries (same ``<prefix>_conformer_dir``) are dropped
          before the merge to produce a one-to-one join.
 
        Examples
        --------
        ::
 
            runner = CalibyEnsembleSeqDesign()
            poses = runner.run_protpardelle_ensemble_generation(
                poses=poses,
                prefix="auto_ens",
                nstruct=8,
                options="",
                num_batches=4,
                jobstarter=my_jobstarter,
            )
            # poses.df["auto_ens_conformer_dir"] is now populated
        """        
        # save current poses for later
        poses.df[f"{prefix}_temp_poses"] = poses.df["poses"]
        temp_poses = deepcopy(poses)

        # initiate ensemble generation
        ensgenerator = CalibyEnsembleGenerator(
            caliby_dir=self.caliby_dir,
            python_path=self.python_path,
            pre_cmd=self.pre_cmd,
            jobstarter=jobstarter)
        
        # run ensemble generation
        ensgenerator.run(poses=poses, prefix=prefix, nstruct=nstruct, options=options, num_batches=num_batches)

        # restore input poses as primary conformers
        poses.df = poses.df[[col for col in poses.df.columns if col.startswith(prefix)]]
        poses.df.drop_duplicates(subset=[f"{prefix}_conformer_dir"], inplace=True)
        poses.df = temp_poses.df.merge(poses.df, on=f"{prefix}_temp_poses")

        return poses


def collect_scores(work_dir: str, mode: str = "seq_des", return_seq_threaded_pdbs_as_pose: bool = False, cif_to_pdb: bool = True) -> pd.DataFrame:
    """
    collect_scores Method
    =====================
    
    Collect and post-process Caliby output files into a unified DataFrame.
    Scans *work_dir* for batch sub-directories produced by a completed
    Caliby run, reads the per-batch result files, performs optional CIF-to-
    PDB conversion, writes FASTA files (for sequence-design mode), and
    returns a consolidated :class:`~pandas.DataFrame` ready for ingestion by
    :class:`~protflow.runners.RunnerOutput`.
 
    Parameters
    ----------
    work_dir : str
        Root working directory of the Caliby run.  Must contain
        ``batch_*/`` sub-directories as created by
        :meth:`_CalibyRunner.setup_batch_mode`.
    mode : {"seq_des", "ens_gen"}, optional
        Parsing mode:
 
        ``"seq_des"``
            Reads ``batch_*/seq_des_outputs.csv`` files produced by
            the sequence-design script.  Each row represents one designed
            sequence.  Default.
 
        ``"ens_gen"``
            Recursively scans ``batch_*/*/*/*.pdb`` files produced by the
            ensemble-generation script.  Each PDB (excluding the primary
            input conformer) becomes one row.
 
    return_seq_threaded_pdbs_as_pose : bool, optional
        *Sequence-design mode only.*  When ``True``, the ``location``
        column points to the (optionally converted) PDB file of the
        sequence-threaded structure rather than to a FASTA file.
        Default is ``False``.
    cif_to_pdb : bool, optional
        *Sequence-design mode only.*  When ``True`` (default), CIF files
        referenced in ``out_pdb`` are converted to PDB format using
        :func:`~protflow.utils.openbabel_tools.openbabel_fileconverter`
        and stored in ``<work_dir>/converted/``.
 
    Returns
    -------
    pandas.DataFrame
        Consolidated result table.  Guaranteed columns across both modes:
 
        ``location``
            Absolute path to the primary output file (FASTA, PDB, or CIF).
        ``description``
            Filename stem of the output file.
 
        Additional columns in ``"seq_des"`` mode (pass-through from
        Caliby's CSV):
 
        ``seq``
            Designed amino-acid sequence (one-letter code).
        ``out_pdb``
            Path to the CIF/PDB threaded structure (present when
            *cif_to_pdb* is ``False`` or *return_seq_threaded_pdbs_as_pose*
            is ``True``).
 
        Additional columns in ``"ens_gen"`` mode:
 
        ``input_description``
            Description of the parent input structure.
        ``input_path``
            Path to the primary conformer in the conformer directory.
        ``conformer_dir``
            Directory containing all conformers for the parent pose.
 
    Raises
    ------
    ValueError
        If *mode* is not one of ``"seq_des"`` or ``"ens_gen"``.
 
    Notes
    -----
    * In ``"seq_des"`` mode, FASTA files are written to
      ``<work_dir>/fasta/`` and converted PDB files to
      ``<work_dir>/converted/``.
    * In ``"ens_gen"`` mode, the ``sample_`` prefix is stripped from
      conformer filenames **in-place** (the files are renamed on disk via
      :meth:`pathlib.Path.rename`).
    * The primary input conformer (``<input_description>.pdb``) written
      by Caliby into the conformer directory is excluded from the
      ``"ens_gen"`` results to avoid conflating input structures with
      generated samples.
    * Results from all batches are concatenated with
      :func:`pandas.concat` and the index is reset.
 
    Examples
    --------
    Collect sequence-design results and convert CIF to PDB::
 
        import pandas as pd
        from protflow.runners.caliby import collect_scores
 
        scores = collect_scores(
            work_dir="/scratch/my_run",
            mode="seq_des",
            cif_to_pdb=True,
        )
        print(scores[["description", "seq", "location"]].head())
 
    Collect ensemble-generation results::
 
        scores = collect_scores(
            work_dir="/scratch/my_run",
            mode="ens_gen",
        )
        print(scores[["description", "conformer_dir"]].head())
    """
    
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

