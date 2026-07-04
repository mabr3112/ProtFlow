"""
protenix.py — ProtFlow Runner Module
======================================
 
.. module:: protflow.runners.protenix
   :synopsis: ProtFlow runner interface for Protenix, a deep-learning
              structure prediction model supporting proteins, nucleic acids,
              ligands, ions, covalent modifications, and geometric constraints.
 
This module provides :class:`ProtenixPred`, a ProtFlow
:class:`~protflow.runners.Runner` subclass that wraps Protenix's ``pred``
sub-command for multi-chain biomolecular structure prediction.  The runner
accepts input as FASTA sequences or PDB/CIF structures, constructs the
Protenix JSON input format internally, executes predictions via ProtFlow's
job-submission abstraction, and returns a scored :class:`~protflow.poses.Poses`
collection.
 
Supported input types
---------------------
FASTA (``*.fa`` / ``*.fasta``)
    Each file may contain one or more chain sequences delimited by *chain_sep*
    (default ``":"``).  One protein-chain entity is created per subsequence.
 
PDB / CIF (``*.pdb`` / ``*.cif``)
    Structures are converted to Protenix JSON format using the ``protenix json``
    sub-command before prediction.
 
Pre-built JSON
    A column in ``poses.df`` containing paths to pre-built Protenix JSON
    files can be supplied via *json_column*, bypassing internal JSON
    construction entirely.
 
Output format
-------------
Protenix writes output CIF files and per-sample JSON score files inside a
``protenix_preds/<pose_name>/seed_<N>/predictions/`` directory hierarchy.
:func:`collect_scores` traverses this hierarchy, ranks samples by
``ranking_score``, optionally converts CIF files to PDB via OpenBabel, and
returns a flat :class:`~pandas.DataFrame`.
 
Configuration
-------------
The runner reads its environment from the ProtFlow configuration file
(``~/.config/protflow/config.py`` by default).  The following keys are
relevant:
 
``PROTENIX_BIN_PATH``
    Absolute path to the Protenix binary (``protenix`` or equivalent).
 
``PROTENIX_PRE_CMD``
    Optional shell preamble executed before every Protenix command (e.g.
    a ``conda activate`` or ``module load`` statement).
 
Dependencies
------------
* :mod:`os`, :mod:`json`, :mod:`shutil`, :mod:`subprocess`,
  :mod:`random`, :mod:`pathlib`, :mod:`glob`, :mod:`logging`
  (standard library)
* `pandas <https://pandas.pydata.org/>`_
* :mod:`protflow.poses` (:class:`~protflow.poses.Poses`)
* :mod:`protflow.jobstarters` (:class:`~protflow.jobstarters.JobStarter`)
* :mod:`protflow.runners` (:class:`~protflow.runners.Runner`,
  :class:`~protflow.runners.RunnerOutput`)
* :mod:`protflow.utils.biopython_tools`
* :mod:`protflow.utils.openbabel_tools`
 
Notes
-----
The classes :class:`ProtenixMSA`, :class:`ProtenixMT`, and
:class:`ProtenixPrep` are planned for future implementation but are not
yet available in this module.
 
Examples
--------
Predict structures from FASTA files::
 
    from protflow.poses import Poses
    from protflow.jobstarters import SbatchArrayJobstarter
    from protflow.runners.protenix import ProtenixPred
 
    poses = Poses("sequences/", prefix="pred")
    jobstarter = SbatchArrayJobstarter(max_cores=4)
 
    runner = ProtenixPred()
    poses = runner.run(
        poses=poses,
        prefix="protenix",
        nstruct=3,
        seeds="random",
        return_top_n_models=1,
    )
 
Predict a protein–ligand complex::
 
    runner = ProtenixPred()
    poses = runner.run(
        poses=poses,
        prefix="protenix_ligand",
        nstruct=1,
        ligands="smiles_col",
        covalent_bonds="bond_col",
        convert_cif_to_pdb=True,
    )
"""

# general imports
import os
import logging
from glob import glob
import shutil
import json
import subprocess
from random import randint
from pathlib import Path

# dependencies
import pandas as pd

# custom
from .. import require_config, load_config_path, runners
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..poses import Poses, col_in_df, description_from_path
from ..jobstarters import JobStarter, split_list
from ..utils.biopython_tools import load_sequence_from_fasta
from ..utils.openbabel_tools import openbabel_fileconverter

# TODO: implement class ProtenixMSA(Runner), class ProtenixMT(Runner), class ProtenixPrep(Runner):



class ProtenixPred(Runner):
    """ProtFlow runner for Protenix biomolecular structure prediction.
 
    :class:`ProtenixPred` inherits from :class:`~protflow.runners.Runner` and
    wraps Protenix's ``pred`` sub-command.  It handles the full prediction
    lifecycle: JSON input construction, batch command assembly, job submission,
    output collection, and scorefile caching.
 
    Protenix supports heterogeneous molecular systems: protein chains, DNA/RNA
    sequences, small-molecule ligands, ions, covalent modifications, and
    geometric constraints can all be specified in a single run through the
    corresponding :meth:`run` parameters.
 
    Parameters
    ----------
    bin_path : str, optional
        Absolute path to the Protenix binary.  Resolved from the ProtFlow
        config key ``PROTENIX_BIN_PATH`` when omitted.
    pre_cmd : str, optional
        Shell preamble prepended to every generated command (e.g.
        ``"conda activate protenix_env &&"``).  Resolved from
        ``PROTENIX_PRE_CMD`` when omitted.
    jobstarter : JobStarter, optional
        Default :class:`~protflow.jobstarters.JobStarter` instance used when
        :meth:`run` is called without an explicit *jobstarter* argument.
 
    Attributes
    ----------
    bin_path : str
        Resolved path to the Protenix binary.
    pre_cmd : str or None
        Shell preamble, or ``None`` when not set.
    jobstarter : JobStarter or None
        Default job submission backend.
    name : str
        Runner identifier (``"protenix.py"``).
    index_layers : int
        Number of index layers added per pose (``1``).
 
    Notes
    -----
    * :meth:`__str__` currently returns ``"colabfold.py"`` — this is a
      known placeholder carried over from an earlier template and will be
      corrected in a future version.
    * When *pose_options* are provided to :meth:`run`, one command is
      generated per pose (no batching).  Without *pose_options*, poses are
      distributed evenly across ``min(len(poses), jobstarter.max_cores)``
      batch JSON files to maximise throughput.
 
    Examples
    --------
    ::
 
        from protflow.runners.protenix import ProtenixPred
 
        runner = ProtenixPred()
        poses = runner.run(poses=poses, prefix="px", nstruct=5)
    """

    def __init__(
            self,
            bin_path: str|None = None,
            pre_cmd: str|None = None,
            jobstarter: str = None
        ) -> None:

        # setup configs
        config = require_config()
        self.bin_path = bin_path or load_config_path(config, path_var="PROTENIX_BIN_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, path_var="PROTENIX_PRE_CMD", is_pre_cmd=True)

        # runner setups
        self.name = "protenix.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "colabfold.py"

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, json_column: str = None, num_copies: int = 1, msa_paired: str = None, msa_unpaired: str = None, 
                          templates: str = None, modifications: str | list | dict = None, ligands: str | list | dict = None, ions: str | list | dict = None, additional_entities: str | list | dict = None, 
                          covalent_bonds: str | list | dict = None, constraints: str | list | dict = None, options: str = None, pose_options: str = None,
            jobstarter: JobStarter = None, overwrite: bool = False, return_top_n_models: int = 1, convert_cif_to_pdb: bool = True, 
            seeds: list | int | str = "random", chain_sep: str = ":") -> Poses:
        """Run Protenix structure prediction on a collection of poses.
 
        Orchestrates the complete prediction pipeline:
 
        1. Creates the working directory and checks for a cached scorefile.
        2. Builds per-pose Protenix input JSON dictionaries (or loads them
           from *json_column*).
        3. Resolves and validates random seeds.
        4. Distributes poses into batched JSON files and assembles CLI commands.
        5. Submits commands via the jobstarter and blocks until completion.
        6. Collects, ranks, and optionally converts output structures.
        7. Saves the scorefile and returns an updated :class:`~protflow.poses.Poses`.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  Accepted pose types are FASTA
            (``*.fa``, ``*.fasta``), PDB (``*.pdb``), and CIF (``*.cif``).
            When *json_column* is set, pose files are not read directly;
            only ``poses_description`` is used from ``poses.df``.
        prefix : str
            Column prefix used to namespace all new columns added to
            ``poses.df`` and to name the working directory
            (``<poses.work_dir>/<prefix>/``).
        nstruct : int, optional
            Number of prediction structures to generate per pose.
            Implemented via distinct random seeds; each seed produces
            one independent structure.  Default is ``1``.
        json_column : str, optional
            Name of a ``poses.df`` column containing absolute paths to
            pre-built Protenix JSON files.  When provided, all other
            molecular-entity parameters (*msa_paired*, *ligands*, etc.)
            are ignored because the JSON already encodes the full input
            specification.  The ``"name"`` field inside each JSON is
            overwritten with the corresponding ``poses_description`` to
            ensure correct output naming.
        num_copies : int, optional
            Stoichiometric copy number applied to every chain in the
            input structure.  Multiplies the existing ``"count"`` field
            of each chain entity in the Protenix JSON.  Default is ``1``
            (no duplication).
        msa_paired : str, optional
            Paired MSA specification.  Accepts either:
 
            * A path to an MSA file (applied identically to every pose), **or**
            * The name of a ``poses.df`` column containing per-pose MSA paths.
 
            Limited to single-chain inputs; raises an error for multi-chain
            structures (use *json_column* in that case).
        msa_unpaired : str, optional
            Unpaired MSA specification.  Same format and restrictions as
            *msa_paired*.
        templates : str, optional
            Template specification.  Accepts a path or a ``poses.df``
            column name.  Same single-chain restriction as *msa_paired*.
        modifications : str, list, or dict, optional
            Post-translational or chemical modifications to apply to
            protein chains.  Accepts:
 
            * A ``dict`` describing one modification.
            * A ``list`` of modification dicts.
            * A ``poses.df`` column name whose values are dicts or lists.
 
            Each modification dict must conform to the Protenix
            ``modifications`` field schema.
        ligands : str, list, or dict, optional
            Small-molecule ligand(s) to include in the complex.  Accepts:
 
            * A SMILES string or SDF file path (applied to all poses).
            * A list of SMILES/SDF paths.
            * A ``poses.df`` column name containing any of the above.
            * A Protenix-format ``{"ligand": {...}}`` entity dict.
 
            Each ligand is appended to the ``"sequences"`` list in the
            Protenix JSON as a ``"ligand"`` entity.
        ions : str, list, or dict, optional
            Ion(s) to include.  Same format as *ligands*, but wrapped as
            ``"ion"`` entities using CCD codes (e.g. ``"MG"``, ``"ZN"``).
        additional_entities : str, list, or dict, optional
            Arbitrary additional molecular entities in fully specified
            Protenix entity-dict format.  Must contain at least one of the
            mandatory top-level keys: ``"proteinChain"``, ``"dnaSequence"``,
            ``"rnaSequence"``, ``"ligand"``, ``"ion"``.  Accepts a single
            dict, a list of dicts, or a ``poses.df`` column name.
        covalent_bonds : str, list, or dict, optional
            Covalent bond definitions between entities.  Accepts:
 
            * A ``dict`` describing one covalent bond.
            * A ``list`` of bond dicts.
            * A ``poses.df`` column name containing dicts or lists of dicts.
 
            Bond dicts must conform to the Protenix ``covalent_bonds``
            schema.
        constraints : str, list, or dict, optional
            Geometric constraints applied during structure prediction.
            Accepts:
 
            * A ``dict`` with a ``"constraint"`` key (unwrapped
              automatically).
            * A ``poses.df`` column name whose values are constraint dicts.
 
            Must ultimately resolve to a ``dict``; any other type raises
            a :exc:`ValueError`.
        options : str, optional
            Additional Protenix CLI flags in ``--key value`` format,
            forwarded verbatim to :meth:`write_cmd` and parsed by
            :func:`~protflow.runners.parse_generic_options`.
        pose_options : str, optional
            Per-pose additional CLI flags.  When provided, batching is
            disabled and one command per pose is generated.  Parsed by
            :func:`~protflow.runners.parse_generic_options` with
            :func:`~protflow.runners.Runner.prep_pose_options`.
        jobstarter : JobStarter, optional
            Job submission backend for this run.  Resolved via the
            standard ProtFlow fallback chain: argument →
            ``self.jobstarter`` → ``poses.default_jobstarter``.
        overwrite : bool, optional
            When ``True``, any existing scorefile and the ``input_jsons/``,
            ``protenix_preds/``, and ``output_predictions/`` sub-directories
            in the working directory are deleted before re-running.  When
            ``False`` (default), an existing scorefile causes immediate
            return of cached results.
        return_top_n_models : int, optional
            Number of top-ranked models (by ``ranking_score``, descending)
            to retain per seed per pose.  Default is ``1``.
        convert_cif_to_pdb : bool, optional
            When ``True`` (default), output CIF files are converted to PDB
            format via OpenBabel.  When ``False``, the ``location`` column
            in the returned poses points to renamed CIF files.
        seeds : list, int, or str, optional
            Random seeds controlling Protenix sampling:
 
            * ``"random"`` (default) — *nstruct* seeds are drawn uniformly
              at random from [1, 10 000].
            * ``int`` — a single seed; automatically wrapped in a list.
              Requires *nstruct* == 1.
            * ``list`` — an explicit list of integer seeds.  Length must
              equal *nstruct*.
            * Falsy value (``None``, ``0``, ``False``) — seeds
              ``[0, 1, …, nstruct-1]`` are used.
 
        chain_sep : str, optional
            Separator character used to split multi-chain sequences within
            a single FASTA file entry.  Default is ``":"``.
 
        Returns
        -------
        Poses
            Updated :class:`~protflow.poses.Poses` with new columns
            prefixed by *prefix*, including:
 
            ``<prefix>_location``
                Absolute path to the output PDB or CIF file for each
                predicted model.
            ``<prefix>_description``
                Unique identifier derived from the output filename stem.
            ``<prefix>_ranking_score``
                Protenix model ranking score (higher is better).
            ``<prefix>_seed``
                Random seed used for this prediction.
            ``<prefix>_sample``
                Sample index within the seed (``"0"`` … ``"N"``).
            ``<prefix>_input``
                Description of the parent input pose.
            All other numeric metrics written to the per-sample JSON by
            Protenix (e.g. pTM, iPTM, pLDDT-related fields).
 
        Raises
        ------
        ValueError
            If the length of an explicit *seeds* list does not equal *nstruct*.
        ValueError
            If *msa_paired*, *msa_unpaired*, or *templates* are specified
            for a multi-chain input structure (must use *json_column*
            instead).
        ValueError
            If *modifications* is not a dict, a list of dicts, or a valid
            column name.
        ValueError
            If *constraints* resolves to a non-dict value.
        ValueError
            If *covalent_bonds* is not a dict, a list of dicts, or a valid
            column name.
        ValueError
            If an entity in *ligands*, *ions*, or *additional_entities*
            does not contain a recognised Protenix entity-type key.
        ValueError
            If the input pose type is not one of ``.fa``, ``.fasta``,
            ``.pdb``, or ``.cif``.
        RuntimeError
            If the number of collected output poses is smaller than the
            number of input poses (indicating crashed prediction jobs).
 
        Notes
        -----
        * **Batching vs. per-pose commands**: when *pose_options* is
          ``None`` (the common case), poses are grouped into
          ``min(len(poses), jobstarter.max_cores)`` batch JSON files, each
          containing multiple pose specifications.  Protenix processes all
          entries in a single JSON sequentially.  When *pose_options* is
          provided, one JSON and one command is produced per pose.
        * **Seed handling**: all seeds are passed as a comma-separated
          string to Protenix's ``-s`` flag.  One output model per seed is
          generated per pose, giving ``nstruct × len(poses)`` total models
          before top-N filtering.
        * **PDB-input path**: when poses are PDB/CIF files, Protenix's
          ``protenix json`` sub-command is called via :func:`json_from_structure`
          to convert them to JSON.  Temporary directories (``temp_jsons/``
          and ``temp_pdbs/``) are created inside *work_dir* and deleted
          after conversion.
 
        Examples
        --------
        Single-seed prediction from FASTA with a ligand per pose::
 
            runner = ProtenixPred()
            poses = runner.run(
                poses=poses,
                prefix="complex",
                nstruct=1,
                ligands="ligand_smiles_col",
                covalent_bonds="bond_col",
                convert_cif_to_pdb=True,
            )
 
        Multi-seed prediction retaining the top 2 models per seed::
 
            poses = runner.run(
                poses=poses,
                prefix="multi_seed",
                nstruct=5,
                seeds=[42, 1337, 999, 7, 12345],
                return_top_n_models=2,
            )
 
        Prediction from pre-built JSON files::
 
            poses = runner.run(
                poses=poses,
                prefix="from_json",
                json_column="protenix_json_path",
                nstruct=2,
                seeds="random",
            )
        """

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")
        print("starting")
        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"protenix_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        
        if overwrite:
            if os.path.isdir(json_dir := os.path.join(work_dir, "input_json")):
                shutil.rmtree(json_dir)
            if os.path.isdir(preds_dir := os.path.join(work_dir, "protenix_preds")):
                shutil.rmtree(preds_dir)
            if os.path.isdir(pdb_dir := os.path.join(work_dir, "output_predictions")):
                shutil.rmtree(pdb_dir)

        # setup af3-specific directories:
        os.makedirs(json_dir := os.path.join(work_dir, "input_jsons"), exist_ok=True)
        os.makedirs(preds_dir := os.path.join(work_dir, "protenix_preds"), exist_ok=True)

        # create input json files
        if json_column:
            in_jsons = []
            col_in_df(poses.df, json_column)

            # import jsons
            dicts = [read_json(path) for path in poses.df[json_column].to_list()]
            
            # make sure each json has the correct pose name assigned
            for d, name in zip(dicts, poses.df["poses_description"].to_list()):
                d["name"] = name

        else:
            in_jsons = self.create_input_dicts(poses, work_dir, num_copies, msa_paired, msa_unpaired, templates, modifications, ligands, ions, additional_entities, covalent_bonds, constraints, chain_sep)

        if seeds:
            if isinstance(seeds, int):
                seeds = [seeds]
            if seeds == "random":
                seeds = [randint(1, 10000) for _ in range(nstruct)]
            if not isinstance(seeds, list) or not len(seeds) == nstruct:
                raise ValueError(f"Number of seeds must be equal to nstruct. Seeds: {seeds}, nstruct: {nstruct}")
        else:
            seeds = list(range(0, nstruct))
        seeds = [str(seed) for seed in seeds]

        if pose_options:
            # prepare pose options
            pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)
            in_jsons = [write_json(in_dict, os.path.join(json_dir, f"batch_{i}.json")) for i, in_dict in enumerate(in_jsons)]

            cmds = [self.write_cmd(in_json=in_json, output_dir=preds_dir, seeds=seeds, options=options, pose_options=pose_opt) for in_json, pose_opt in zip([in_jsons, pose_options])]
        
        else:
            # setup input-fastas in batches (to speed up prediction times.), but only if no pose_options are provided!
            num_batches = min(len(poses.df.index), jobstarter.max_cores)

            # split dicts into batches
            dicts_batches = split_list(in_jsons, n_sublists=num_batches)

            # write dicts to json
            in_jsons = [write_json(batch, os.path.join(json_dir, f"batch_{i}.json")) for i, batch in enumerate(dicts_batches)]

            cmds = [self.write_cmd(in_json=in_json, output_dir=preds_dir, seeds=seeds, options=options, pose_options=None) for in_json in in_jsons]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        logging.info(f"Starting Protenix predictions of {len(poses.df.index)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="ProtenixPred",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        logging.info("Predictions finished, starting to collect scores.")
        scores = collect_scores(work_dir=work_dir, convert_cif_to_pdb=convert_cif_to_pdb, return_top_n_models=return_top_n_models)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def write_cmd(self, in_json: str, output_dir: str, seeds: list, options: str = None, pose_options: str = None):
        """Compose a single Protenix ``pred`` shell command string.
 
        Combines :attr:`bin_path` with the mandatory ``-i``, ``-o``, and
        ``-s`` flags, then appends any additional options and boolean flags
        parsed from *options* and *pose_options*.
 
        Parameters
        ----------
        in_json : str
            Absolute path to the batch input JSON file.  Passed as
            ``-i <in_json>`` to Protenix.
        output_dir : str
            Directory where Protenix should write its ``<pose>/seed_*/``
            output hierarchy.  Passed as ``-o <output_dir>``.
        seeds : list of str
            Seed values as strings.  Joined with commas and passed as
            ``-s <seed1>,<seed2>,...``.
        options : str, optional
            Global additional CLI options in ``--key value`` or ``--flag``
            format.  Parsed by
            :func:`~protflow.runners.parse_generic_options` with ``sep="--"``.
        pose_options : str, optional
            Per-pose additional CLI options, merged with *options* during
            parsing.
 
        Returns
        -------
        str
            A complete shell command string, e.g.::
 
                /opt/protenix/protenix pred -i /run/input_jsons/batch_0.json \
                    -o /run/protenix_preds -s 42,1337 --num_workers 4
 
        Notes
        -----
        * Options are serialised as ``--key value`` pairs; boolean flags
          (no value) are serialised as ``--flag``.
        * *options* and *pose_options* are merged before serialisation by
          :func:`~protflow.runners.parse_generic_options`; pose-level values
          override global ones for the same key.
 
        Examples
        --------
        ::
 
            cmd = runner.write_cmd(
                in_json="/run/input_jsons/batch_0.json",
                output_dir="/run/protenix_preds",
                seeds=["42", "1337"],
                options="--num_workers 4",
            )
        """

        # parse options
        opts, flags = runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""
        return f"{self.bin_path} pred -i {in_json} -o {output_dir} -s {','.join(seeds)} {opts} {flags}"
    
    def create_input_dicts(self, poses: Poses, work_dir: str, num_copies: int, msa_paired: str = None, msa_unpaired: str = None, 
                            templates: str = None, modifications: str | list | dict = None, ligands: str | list | dict = None, ions: str | list | dict = None, additional_entities: str | list | dict = None, 
                            covalent_bonds: str | list | dict = None, constraints: str | list | dict = None, chain_sep: str = ":") -> list:
        """Build a list of Protenix input dictionaries, one per pose.
 
        Determines the input type of *poses* (FASTA or PDB/CIF), constructs
        a base input dict for each pose, then augments each dict with MSAs,
        templates, modifications, ligands, ions, covalent bonds, and
        constraints as specified.
 
        Parameters
        ----------
        poses : Poses
            Input pose collection.  All poses must have the same file
            extension (homogeneous collection).
        work_dir : str
            Working directory.  Used to create temporary sub-directories
            (``temp_jsons/`` and ``temp_pdbs/``) when poses are PDB/CIF
            files.  These are deleted after JSON generation.
        num_copies : int
            Stoichiometric copy number applied to every chain entity.
            Multiplied into the existing ``"count"`` field.
        msa_paired : str, optional
            Path to a paired MSA file or a ``poses.df`` column name.
            Applied only to single-chain inputs.
        msa_unpaired : str, optional
            Path to an unpaired MSA file or a ``poses.df`` column name.
            Applied only to single-chain inputs.
        templates : str, optional
            Path to a templates file or a ``poses.df`` column name.
            Applied only to single-chain inputs.
        modifications : str, list, or dict, optional
            Post-translational modifications.  See :meth:`run` for
            accepted formats.
        ligands : str, list, or dict, optional
            Ligand specifications.  See :meth:`run` for accepted formats.
        ions : str, list, or dict, optional
            Ion specifications.  See :meth:`run` for accepted formats.
        additional_entities : str, list, or dict, optional
            Fully specified additional molecular entities.  See
            :meth:`run` for accepted formats.
        covalent_bonds : str, list, or dict, optional
            Covalent bond definitions.  See :meth:`run` for accepted
            formats.
        constraints : str, list, or dict, optional
            Geometric constraint definitions.  See :meth:`run` for
            accepted formats.
        chain_sep : str, optional
            Separator used to split multi-chain FASTA entries.
            Default is ``":"``.
 
        Returns
        -------
        list of dict
            One Protenix input dict per pose.  Each dict contains at
            minimum the keys ``"name"`` (pose description) and
            ``"sequences"`` (list of entity dicts).  Additional keys
            (``"pairedMsaPath"``, ``"modifications"``, ``"covalent_bonds"``,
            ``"constraint"``, etc.) are present only when the corresponding
            parameters are supplied.
 
        Raises
        ------
        ValueError
            If the pose file extension is not one of ``.fa``, ``.fasta``,
            ``.pdb``, ``.cif``.
        ValueError
            If *msa_paired*, *msa_unpaired*, or *templates* are set for
            a multi-chain input structure.
        ValueError
            If *modifications* resolves to something other than a dict or
            a list of dicts.
        ValueError
            If *covalent_bonds* resolves to something other than a dict or
            a list of dicts.
        ValueError
            If *constraints* resolves to a non-dict value.
        ValueError
            If an entity in *ligands*, *ions*, or *additional_entities*
            does not contain a recognised Protenix entity-type key.
 
        Notes
        -----
        **FASTA path**: sequences are loaded via
        :func:`~protflow.utils.biopython_tools.load_sequence_from_fasta`
        and split by *chain_sep*; one ``"proteinChain"`` entity with
        ``"count": 1`` is created per sub-sequence.
 
        **PDB/CIF path**: all pose files are copied into a temporary
        ``temp_pdbs/`` directory, the ``protenix json`` sub-command is
        called via :func:`json_from_structure`, and the resulting JSONs
        are merged back into ``poses.df`` via a left-join on
        ``poses_description``.  The ``strip_list=True`` option of
        :func:`read_json` unwraps the outer list that ``protenix json``
        writes.
 
        **Internal helpers** (nested functions defined inside this method):
 
        ``add_msa_template_modifications(term, in_type, pose_dict, pose_row)``
            Resolves *term* to a value (optionally by looking it up as a
            column in *pose_row*), validates it, and sets the corresponding
            top-level key (``"pairedMsaPath"``, ``"unpairedMsaPath"``,
            ``"templatesPath"``, or ``"modifications"``) in *pose_dict*.
 
        ``add_additional_entities(pose_dict, pose_row, ligands, ions, additional_entities)``
            Appends ligand, ion, and arbitrary-entity dicts to the
            ``"sequences"`` list in *pose_dict*, using ``identify_entities``
            and ``check_entity`` for validation and normalisation.
 
        ``identify_entities(pose_row, entity, entity_type)``
            Recursively resolves *entity* (column name, SMILES/path string,
            list, or dict) to a validated Protenix entity dict or list of
            dicts of the given *entity_type*.
 
        ``check_entity(entity)``
            Validates that a dict or list of dicts contains at least one
            mandatory Protenix entity key (``"proteinChain"``,
            ``"dnaSequence"``, ``"rnaSequence"``, ``"ligand"``, ``"ion"``).
            Returns a list for uniform downstream handling.
 
        ``identify_bonds(pose_row, bonds)``
            Resolves *bonds* (column name, dict, or list of dicts) to a
            validated list of covalent-bond dicts.
 
        ``create_dict_from_fa(name, path, sep)``
            Reads a FASTA file, splits by *sep*, and returns a minimal
            Protenix input dict with one ``"proteinChain"`` entity per
            sub-sequence.
 
        Examples
        --------
        ::
 
            dicts = runner.create_input_dicts(
                poses=poses,
                work_dir="/scratch/protenix_run",
                num_copies=1,
                ligands="smiles_col",
                chain_sep=":",
            )
            # len(dicts) == len(poses)
            # dicts[0].keys() == {"name", "sequences"}
        """

        def add_msa_template_modifications(term: str, in_type:str, pose_dict: dict, pose_row: pd.Series) -> dict:

            if not term:
                return pose_dict
            
            # check if pose-specific msa is provided
            if term in pose_row:
                term = pose_row[term]

            # check if msa exists
            if not in_type == "modifications" and not os.path.isfile(term):
                raise ValueError(f"Could not detect msa or template at {term} for pose {pose_row['poses_description']}.")

            if in_type == "modifications":
                if isinstance(term, dict):
                    term = [term]
                if not isinstance(term, list):
                    raise ValueError("Modifications must be specified via dictionary/list of dictionaries or a poses dataframe column containing dictionaries!")
            
            # check number of chains in dict
            num_prot_chains = sum(1 for d in pose_dict["sequences"] if "proteinChain" in d)
            num_rna_chains = sum(1 for d in pose_dict["sequences"] if "rnaSequence" in d)

            if num_prot_chains + num_rna_chains > 1:
                raise ValueError("MSAs and templates cannot be specified via options if multiple sequences are present in the input structures.\n" \
                "Create jsons containing MSA/template paths manually and specify <json_col>.")

            pose_dict[in_type] = term

            return pose_dict
        
        def add_additional_entities(pose_dict: dict, pose_row: pd.Series, ligands: str | list | dict = None, ions: str | list | dict = None, 
                additional_entities: str | list | dict = None):
            
            if ligands:
                pose_dict["sequences"] = pose_dict["sequences"] +  identify_entities(pose_row, ligands, "ligand")
            if ions:
                pose_dict["sequences"] = pose_dict["sequences"] +  identify_entities(pose_row, ions, "ions")
            if additional_entities:
                additional_entities = check_entity(additional_entities)
                pose_dict["sequences"] = pose_dict["sequences"] +  additional_entities
            
            return pose_dict
        
        def identify_entities(pose_row: pd.Series, entity: str | list | dict, entity_type: str):
            if isinstance(entity, str):
                if entity in pose_row:
                    entity = pose_row[entity]
                    entity = identify_entities(pose_row, entity, entity_type)
                else:
                    entity = {entity_type: {entity_type: entity, "count": 1}}
            if isinstance(entity, list):
                entity = [identify_entities(pose_row, ent, entity_type) for ent in entity]
                check_entity(entity)
            if isinstance(entity, dict):
                entity = check_entity(entity)
            else:
                raise ValueError("Ligands and ions must be specified via dictionary, a list of dictionaries or SMILES/sdf paths/ion CCD code strings OR a poses dataframe column containing these.")

            return entity

        def check_entity(entity: dict | list):

            if isinstance(entity, dict):
                mandatory_keys = ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
                if not any(key in entity for key in mandatory_keys):
                    raise ValueError(f"Input entity type must be one of {mandatory_keys}. Affected entity: {entity}")
                return [entity]
                
            if isinstance(entity, list):
                for ent in entity:
                    check_entity(ent)
                
                return entity
            
            else:
                raise ValueError(f"Additional entities must be provided as a dict or list of dicts format, not {type(entity)}! Affected entity is {entity}")

        def identify_bonds(pose_row: pd.Series, bonds: str | list | dict):
            if isinstance(bonds, str) and bonds in pose_row:
                bonds = pose_row[bonds]
            if isinstance(bonds, list):
                if not all(isinstance(bond, dict) for bond in bonds):
                    raise ValueError(f"Restraints must be specified via dictionary. Wrong restraint: {bonds}")
            if isinstance(bonds, dict):
                bonds = [bonds]
            else:
                raise ValueError("Covalent bonds must be specified via dict, a list of dicts or a poses dataframe column containing these.")

            return bonds
        
        def create_dict_from_fa(name: str, path: str, sep:str=":"):
            seqs = str(load_sequence_from_fasta(path, return_multiple_entries=False).seq)
            protchains = [{"proteinChain": {"sequence": seq, "count": 1}} for seq in seqs.split(sep)]
            in_dict = {"name": name, "sequences": protchains}
            return in_dict

        # determine pose type to create input dictionaries
        input_type = poses.determine_pose_type(pose_col="poses")
        if input_type == [".fa"] or input_type == [".fasta"]:
            # create input dict for each pose
            poses.df["temp_protenix_in_dict_col"] = poses.df.apply(lambda row: create_dict_from_fa(name=row["poses_description"], path=row["poses"], sep=chain_sep), axis=1)
            
            # create a list
            dicts = poses.df["temp_protenix_in_dict_col"].to_list()

            # drop temp col
            poses.df.drop(["temp_protenix_in_dict_col"], axis=1, inplace=True)

        elif input_type == [".pdb"] or input_type == [".cif"]:
            # create temp folders
            os.makedirs(temp_jsons := os.path.join(work_dir, "temp_jsons"), exist_ok=True)
            os.makedirs(temp_pdbs := os.path.join(work_dir, "temp_pdbs"), exist_ok=True)

            # copy all poses into same dir
            for pose in poses.poses_list():
                shutil.copy(pose, temp_pdbs)

            # create input dict using protenix structure to json function
            json_from_structure(protenix_path=str(self.bin_path), in_dir=temp_pdbs, out_dir=temp_jsons)

            # gather files
            jsons = glob(os.path.join(temp_jsons, "*.json"))

            # merge with original df
            temp_df = pd.DataFrame({"temp_protenix_json_paths": jsons, "poses_description": [description_from_path(j) for j in jsons]})
            poses.df = poses.df.merge(temp_df, how="left", on="poses_description") # left to preserve order

            # create dicts
            dicts = [read_json(json_path, strip_list=True) for json_path in poses.df["temp_protenix_json_paths"].to_list()]
            
            # delete temp
            shutil.rmtree(temp_jsons)
            shutil.rmtree(temp_pdbs)
            poses.df.drop(["temp_protenix_json_paths"], axis=1, inplace=True)

        else:
            raise ValueError(f"Invalid input pose format: {input_type}")
        
        records = []
        for in_dict, (_, row) in zip(dicts, poses.df.iterrows()):
            in_dict["name"] = row["poses_description"]

            # update number of copies
            for seq_rec in in_dict["sequences"]:
                for chain_type in seq_rec:
                    seq_rec[chain_type]["count"] = seq_rec[chain_type]["count"] * num_copies

            # add templates, msas and modifications
            in_dict = add_msa_template_modifications(msa_paired, "pairedMsaPath", in_dict, row)
            in_dict = add_msa_template_modifications(msa_unpaired, "unpairedMsaPath", in_dict, row)
            in_dict = add_msa_template_modifications(templates, "templatesPath", in_dict, row)
            in_dict = add_msa_template_modifications(modifications, "modifications", in_dict, row)

            # add ions, ligands, additional sequences
            in_dict = add_additional_entities(in_dict, row, ligands, ions, additional_entities)

            if covalent_bonds:
                in_dict["covalent_bonds"] = identify_bonds(row, covalent_bonds)
            if constraints:
                # check for pose-specific constraint
                if isinstance(constraints, str) and constraints in row:
                    constraints = row[constraints]
                # unwrap constraints
                if isinstance(constraints, dict) and "constraint" in constraints:
                    constraints = constraints["constraint"]
                if not isinstance(constraints, dict):
                    raise ValueError("Constraints must be specified using a dict or a dataframe column containing dicts.")
                in_dict["constraint"] = constraints

            records.append(in_dict)

        return records


def json_from_structure(protenix_path: str, in_dir: str, out_dir:str, altloc:str=None, assembly_id:str=None, include_discont_poly_poly_bonds:bool=False) -> str:
    """Convert a directory of PDB/CIF structures to Protenix JSON format.
 
    Calls the ``protenix json`` sub-command to convert all structure files
    in *in_dir* into Protenix-compatible JSON input files written to
    *out_dir*.  Each output JSON encodes the full molecular system
    (chains, residues, modifications) parsed from the input structure.
 
    Parameters
    ----------
    protenix_path : str
        Absolute path to the Protenix binary.
    in_dir : str
        Directory containing input PDB or CIF structure files.  All files
        in the directory are processed; subdirectories are ignored.
    out_dir : str
        Directory where the generated JSON files will be written.  One
        JSON file is created per input structure, named after the input
        file stem.
    altloc : str, optional
        Alternate location indicator to use when multiple conformers are
        present in the input structure (e.g. ``"A"``).  Passed as
        ``--altloc <altloc>`` to Protenix.  When omitted, Protenix uses
        its default selection.
    assembly_id : str, optional
        Biological assembly ID to extract from the input structure (e.g.
        ``"1"``).  Passed as ``--assembly_id <assembly_id>``.  When
        omitted, the first assembly (or asymmetric unit) is used.
    include_discont_poly_poly_bonds : bool, optional
        When ``True``, covalent bonds between discontinuous polymer chains
        are included in the output JSON.  Passed as
        ``--include_discont_poly_poly_bonds``.  Default is ``False``.
 
    Returns
    -------
    None
        Output files are written to *out_dir*; nothing is returned.
 
    Raises
    ------
    subprocess.CalledProcessError
        Caught internally; the error message and stderr are printed to
        stdout but the exception is not re-raised.  Callers should verify
        that output JSON files exist in *out_dir* after this call.
 
    Notes
    -----
    * The command is executed via :func:`subprocess.run` with
      ``shell=True``, ``check=True``, and ``capture_output=True``.  Any
      non-zero exit code triggers the exception handler.
    * Because the exception is caught and printed rather than propagated,
      this function can silently produce zero output files if Protenix
      fails.  :meth:`ProtenixPred.create_input_dicts` detects this
      implicitly when the ``glob`` of ``*.json`` in *out_dir* returns an
      empty list.
    * This function is called internally by
      :meth:`ProtenixPred.create_input_dicts` when poses are PDB/CIF
      files.  It is exposed as a public module-level function for direct
      use in custom pipelines.
 
    Examples
    --------
    ::
 
        json_from_structure(
            protenix_path="/opt/protenix/protenix",
            in_dir="/scratch/input_pdbs",
            out_dir="/scratch/input_jsons",
            altloc="A",
        )
        # /scratch/input_jsons/<stem>.json written for each PDB in in_dir
    """

    cmd = f"{protenix_path} json -i {in_dir} -o {out_dir}"

    if altloc:
        cmd = cmd + f" --altloc {altloc}"
    if assembly_id:
        cmd = cmd + f" --assembly_id {assembly_id}"
    if include_discont_poly_poly_bonds:
        cmd = cmd + " --include_discont_poly_poly_bonds"

    try:
        # Run the command
        subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)

    except subprocess.CalledProcessError as e:
        # This triggers if the protenix command returns an error code
        print(f"Error when creating input dictionary from {in_dir}: {e.stderr}")


def collect_scores(work_dir: str, convert_cif_to_pdb: bool = True, return_top_n_models: int = 1) -> pd.DataFrame:
    """Collect, rank, convert, and consolidate Protenix prediction outputs.
 
    Traverses the ``protenix_preds/`` directory hierarchy produced by a
    completed Protenix run, reads the per-sample JSON score files, ranks
    samples within each seed by ``ranking_score``, retains the top *N*,
    optionally converts CIF output files to PDB format, copies results to a
    flat ``output_predictions/`` directory with zero-padded filenames, and
    returns a consolidated :class:`~pandas.DataFrame`.
 
    Parameters
    ----------
    work_dir : str
        Root working directory of the Protenix run.  Must contain a
        ``protenix_preds/`` sub-directory produced by Protenix.
    convert_cif_to_pdb : bool, optional
        When ``True`` (default), each selected CIF file is converted to
        PDB format via
        :func:`~protflow.utils.openbabel_tools.openbabel_fileconverter`
        and the PDB is written to ``<work_dir>/output_predictions/``.
        When ``False``, the CIF file is copied (without conversion) to
        the same directory.  In both cases the ``location`` column points
        to the file in ``output_predictions/``.
    return_top_n_models : int, optional
        Number of models to retain per seed per pose, ranked by
        ``ranking_score`` in descending order.  Default is ``1``
        (only the best model per seed is kept).
 
    Returns
    -------
    pandas.DataFrame
        One row per retained model with the following guaranteed columns:
 
        ``location`` : str
            Absolute path to the output PDB or CIF file in
            ``<work_dir>/output_predictions/``.
        ``description`` : str
            Filename stem of the output file, used as the pose identifier
            in ProtFlow.  Format: ``<input_name>_<zero_padded_counter>``.
        ``ranking_score`` : float
            Protenix model ranking score (higher is better).
        ``seed`` : int
            Random seed used to generate this model.
        ``sample`` : str
            Sample index within the seed, extracted from the per-sample
            JSON filename.
        ``input`` : str
            Name of the parent pose (the sub-directory name under
            ``protenix_preds/``).
        All other fields written to the per-sample JSON by Protenix
        (e.g. pTM, iPTM, chain-level pLDDT metrics).
 
    Notes
    -----
    **Directory structure traversal**:
    The expected hierarchy is::
 
        <work_dir>/
        └── protenix_preds/
            └── <pose_name>/          ← one per input pose
                └── seed_<N>/         ← one per seed
                    └── predictions/
                        ├── <pose_name>_sample_<K>.cif
                        └── <pose_name>_sample_<K>.json
 
    Sub-directories named ``"ERR"`` inside ``protenix_preds/`` are
    silently excluded.
 
    **File naming**: output files in ``output_predictions/`` are named
    ``<input>_<NNNN>.pdb`` (or ``.cif``), where ``<NNNN>`` is a
    zero-padded counter that increments across all seeds for the same
    pose.  This counter ensures unique filenames and the correct number
    of index layers for ProtFlow merging.
 
    **Temporary columns**: intermediate columns prefixed with ``"temp_"``
    (``temp_location``, ``temp_counter``) are created during processing
    and dropped before the DataFrame is returned.
 
    Examples
    --------
    Collect with PDB conversion (default)::
 
        scores = collect_scores("/scratch/protenix_run")
        print(scores[["description", "location", "ranking_score"]].head())
 
    Collect CIF files and keep the top 3 models per seed::
 
        scores = collect_scores(
            "/scratch/protenix_run",
            convert_cif_to_pdb=False,
            return_top_n_models=3,
        )
    """

    def cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return output

    # collect all output directories, ignore mmseqs dirs
    out_dirs = [d for d in glob(os.path.join(work_dir, "protenix_preds", "*")) if os.path.isdir(d) and not os.path.basename(d) == "ERR"]

    scores = []
    for out_dir in out_dirs:
        counter = 1
        name = Path(out_dir).name
        seeds_dirs = [d for d in glob(os.path.join(out_dir, "seed_*"))]
        for seed_dir in seeds_dirs:
            seed = int(seed_dir.split("_")[-1])
            jsons = glob(os.path.join(seed_dir, "predictions", "*.json"))
            seed_df = []
            for j in jsons:
                ser = pd.read_json(j, typ="series")
                sample = description_from_path(j).split("_")[-1] # extract sample number
                ser["sample"] = sample
                ser["temp_location"] = os.path.join(seed_dir, "predictions", f"{name}_sample_{sample}.cif")
                seed_df. append(ser)
            seed_df = pd.DataFrame(seed_df)
            seed_df.sort_values("ranking_score", ascending=False, inplace=True)
            seed_df = seed_df.head(return_top_n_models)
            seed_df["seed"] = seed
            seed_df["input"] = name
            seed_df["temp_counter"] = range(counter, counter + len(seed_df)) # assign a temp counter for later renaming
            scores.append(seed_df)
            counter += len(seed_df) # update counter
    
    scores = pd.concat(scores)
    scores.reset_index(drop=True, inplace=True)

    os.makedirs(out_dir := os.path.join(work_dir, "output_predictions"), exist_ok=True)

    # convert to pdb, otherwise rename files (without renaming, wrong number of index layers would be added)
    if convert_cif_to_pdb:
        scores["location"] = scores.apply(lambda row: cif_to_pdb(input_cif=row["temp_location"], output_format="pdb", output=os.path.abspath(os.path.join(out_dir, f"{row['input']}_{str(row['temp_counter']).zfill(4)}.pdb"))), axis=1)
    else:
        scores["location"] = scores.apply(lambda row: shutil.copy(row["temp_location"], os.path.abspath(os.path.join(out_dir, f"{row['input']}_{str(row['temp_counter']).zfill(4)}.pdb"))), axis=1)
    
    # create description col
    scores["description"] = scores.apply(lambda row: description_from_path(row["location"]), axis=1)

    # drop temp cols
    scores.drop([col for col in scores.columns if col.startswith("temp_")], axis=1, inplace=True)
    return scores

def read_json(path, strip_list=False):
    with open(path, 'r', encoding="UTF-8") as j:
        data = json.load(j)

    if strip_list:
        data = data[0]

    return data

def write_json(data, path):
    with open(path, 'w', encoding="UTF-8") as j:
        json.dump(data, j, indent=2)
    return path
