"""AtomSelection-based contact metrics for ProtFlow.

This module provides :class:`AtomSelectionContacts`, a contact-count runner that
uses explicit atom selections instead of ligand-chain semantics. It mirrors the
distance-shell calculation from :class:`protflow.metrics.ligand.LigandContacts`
while letting callers define the target and optional reference atoms with
ProtFlow :class:`~protflow.residues.AtomSelection` inputs.

The runner writes JSON worker inputs, dispatches contact calculations through a
:class:`~protflow.jobstarters.JobStarter`, and merges the collected contact
scores back into a :class:`~protflow.poses.Poses` dataframe.

Examples
--------
Count all non-hydrogen atoms within 5 A of one selected target atom:

>>> from protflow.metrics import AtomSelectionContacts
>>> contacts = AtomSelectionContacts(target_atoms=[("A", 42, "CA")])
>>> poses = contacts.run(poses=poses, prefix="atom_contacts", exclude_elements=["H"])

Use per-pose selections stored in dataframe columns:

>>> poses.df["target_atoms"] = [...]
>>> poses.df["reference_atoms"] = [...]
>>> contacts = AtomSelectionContacts(target_atoms="target_atoms", reference_atoms="reference_atoms")
>>> poses = contacts.run(poses=poses, prefix="site_contacts")
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
from typing import Any

import numpy as np
import pandas as pd

from protflow import require_config, load_config_path
from protflow.jobstarters import JobStarter, split_list
from protflow.poses import Poses
from protflow.residues import AtomSelection, AtomSelectionInput
from protflow.runners import Runner, RunnerOutput, col_in_df
from protflow.utils.biopython_tools import (
    biopython_load_structure,
    get_atoms_of_atom_selection,
)


def _setup_atom_selection_list(
    poses: Poses,
    selection: AtomSelectionInput | str | None,
    parameter_name: str,
    allow_none: bool = False,
) -> list[list[Any] | None]:
    """Expand a static or dataframe-column AtomSelection for each pose.

    Parameters
    ----------
    poses : Poses
        Input pose container whose dataframe can hold per-pose selections.
    selection : AtomSelectionInput or str or None
        Static AtomSelection-compatible value, or a column name in ``poses.df``.
        ``None`` is only accepted when ``allow_none`` is ``True``.
    parameter_name : str
        Human-readable parameter name used in validation errors.
    allow_none : bool, optional
        If ``True``, return one ``None`` entry per pose when ``selection`` is
        omitted.

    Returns
    -------
    list of list or None
        One JSON-serializable AtomSelection list per pose.

    Raises
    ------
    ValueError
        If a required selection is missing or a required per-pose value is
        ``None``.
    """
    if selection is None:
        if allow_none:
            return [None for _ in poses]
        raise ValueError(f"{parameter_name} must be specified.")

    if isinstance(selection, str):
        col_in_df(poses.df, selection)
        out = []

        # Strings identify dataframe columns containing one selection per pose.
        for idx, row_selection in enumerate(poses.df[selection].to_list()):
            if row_selection is None:
                if allow_none:
                    out.append(None)
                    continue
                raise ValueError(f"{parameter_name} column '{selection}' contains None at row {idx}.")
            out.append(AtomSelection(row_selection).to_list())
        return out

    atom_selection = AtomSelection(selection).to_list()
    return [atom_selection for _ in poses]


def _atom_is_included(atom, exclude_elements: list[str]) -> bool:
    """Return whether an atom survives element filtering.

    Parameters
    ----------
    atom : Bio.PDB.Atom.Atom
        Atom object to evaluate.
    exclude_elements : list of str
        Lowercase element symbols to exclude.

    Returns
    -------
    bool
        ``True`` when the atom should be used in the contact calculation.
    """
    return atom.element.lower() not in exclude_elements


def _calc_atomselection_contacts(
    pose: str,
    target_atoms: AtomSelectionInput,
    reference_atoms: AtomSelectionInput = None,
    min_dist: float = 0,
    max_dist: float = 5,
    exclude_elements: list[str] = None,
    normalize_by_num_atoms: bool = False,
) -> float:
    """Calculate contacts between target and reference AtomSelections.

    Parameters
    ----------
    pose : str
        Path to the structure file to score.
    target_atoms : AtomSelectionInput
        Target atoms used as the denominator when normalization is requested.
    reference_atoms : AtomSelectionInput, optional
        Reference atoms to score against. If omitted, all pose atoms except the
        target atoms are used.
    min_dist : float, optional
        Lower Angstrom bound for counted contacts.
    max_dist : float, optional
        Upper Angstrom bound for counted contacts.
    exclude_elements : list of str, optional
        Element symbols to remove from both target and reference selections.
    normalize_by_num_atoms : bool, optional
        If ``True``, divide the contact count by the number of included target
        atoms.

    Returns
    -------
    float
        Raw contact count, or normalized contacts per target atom.

    Raises
    ------
    ValueError
        If the target selection contains no atoms after element filtering.
    """
    pose = biopython_load_structure(pose)
    exclude_elements = [element.lower() for element in exclude_elements] if exclude_elements else []

    # Resolve target atoms once so the default reference can exclude them.
    target_atom_objects = get_atoms_of_atom_selection(pose, target_atoms)
    target_atom_ids = {atom.get_full_id() for atom in target_atom_objects}
    target_atom_objects = [
        atom for atom in target_atom_objects
        if _atom_is_included(atom, exclude_elements)
    ]

    # Missing reference selection means "everything except the target atoms".
    if reference_atoms is None:
        reference_atom_objects = [
            atom for atom in pose.get_atoms()
            if atom.get_full_id() not in target_atom_ids
        ]
    else:
        reference_atom_objects = get_atoms_of_atom_selection(pose, reference_atoms)
    reference_atom_objects = [
        atom for atom in reference_atom_objects
        if _atom_is_included(atom, exclude_elements)
    ]

    if len(target_atom_objects) == 0:
        raise ValueError("Target AtomSelection contains no atoms after element filtering.")
    if len(reference_atom_objects) == 0:
        return 0.0 if normalize_by_num_atoms else 0

    # Build the same all-by-all distance matrix used by LigandContacts.
    target_coords = np.array([atom.get_coord() for atom in target_atom_objects])
    reference_coords = np.array([atom.get_coord() for atom in reference_atom_objects])

    dgram = np.linalg.norm(
        reference_coords[:, np.newaxis] - target_coords[np.newaxis, :],
        axis=-1,
    )
    contacts = int(np.sum((dgram > min_dist) & (dgram < max_dist)))
    if normalize_by_num_atoms:
        return round(contacts / len(target_atom_objects), 2)
    return contacts


def _collect_scores(output_files: list[str], expected_rows: int) -> pd.DataFrame:
    """Collect worker output JSON files into one score dataframe.

    Parameters
    ----------
    output_files : list of str
        Worker JSON files written by this runner.
    expected_rows : int
        Number of input poses expected in the combined result.

    Returns
    -------
    pandas.DataFrame
        Concatenated worker scores.

    Raises
    ------
    FileNotFoundError
        If any expected worker output is missing.
    RuntimeError
        If fewer score rows are collected than input poses.
    """
    missing_outputs = [output_file for output_file in output_files if not os.path.isfile(output_file)]
    if missing_outputs:
        raise FileNotFoundError(f"AtomSelection contact output files were not created: {missing_outputs}")

    # Empty or truncated worker output usually means a subprocess failed early.
    scores = pd.concat([pd.read_json(output_file) for output_file in output_files], ignore_index=True)
    if len(scores.index) < expected_rows:
        raise RuntimeError("Some AtomSelection contact worker jobs did not produce scores.")
    return scores


class AtomSelectionContacts(Runner):
    """Count contacts between target and reference AtomSelections.

    The runner counts atom pairs whose distances are greater than ``min_dist``
    and less than ``max_dist``. ``target_atoms`` is required. ``reference_atoms``
    is optional; when omitted, every pose atom except the target atoms is used as
    the reference set. Both selections can be static AtomSelection-compatible
    values or names of ``poses.df`` columns containing per-pose selections.

    Parameters
    ----------
    target_atoms : AtomSelectionInput or str, optional
        Default target atom selection, or dataframe column containing one
        selection per pose.
    reference_atoms : AtomSelectionInput or str, optional
        Default reference atom selection, or dataframe column containing one
        selection per pose. If omitted, all non-target pose atoms are used.
    min_dist : float, optional
        Lower Angstrom bound for counted contacts.
    max_dist : float, optional
        Upper Angstrom bound for counted contacts.
    exclude_elements : list of str, optional
        Element symbols to remove from both target and reference atoms.
    jobstarter : JobStarter, optional
        Default jobstarter used when :meth:`run` is called without one.
    overwrite : bool, optional
        Default cache-overwrite behavior for :meth:`run`.
    """

    def __init__(
        self,
        target_atoms: AtomSelectionInput | str = None,
        reference_atoms: AtomSelectionInput | str = None,
        min_dist: float = 0,
        max_dist: float = 5,
        exclude_elements: list[str] = None,
        jobstarter: JobStarter = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize AtomSelection contact defaults for later :meth:`run` calls.

        Parameters
        ----------
        target_atoms : AtomSelectionInput or str, optional
            Default target atom selection or dataframe column name.
        reference_atoms : AtomSelectionInput or str, optional
            Default reference atom selection or dataframe column name.
        min_dist : float, optional
            Lower Angstrom bound for counted contacts.
        max_dist : float, optional
            Upper Angstrom bound for counted contacts.
        exclude_elements : list of str, optional
            Element symbols to remove before contact counting.
        jobstarter : JobStarter, optional
            Default jobstarter for this runner instance.
        overwrite : bool, optional
            Whether to overwrite existing cached score files by default.
        """

        # Resolve the worker interpreter from the active ProtFlow config.
        self.python = os.path.join(load_config_path(require_config(), "PROTFLOW_ENV"), "python")

        # Store defaults that can be overridden per run call.
        self.target_atoms = target_atoms
        self.reference_atoms = reference_atoms
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.exclude_elements = exclude_elements
        self.jobstarter = jobstarter
        self.overwrite = overwrite

    def __str__(self) -> str:
        """Return the short runner name.

        Returns
        -------
        str
            The literal runner name ``"AtomSelectionContacts"``.
        """
        return "AtomSelectionContacts"

    def run(
        self,
        poses: Poses,
        prefix: str,
        jobstarter: JobStarter = None,
        target_atoms: AtomSelectionInput | str = None,
        reference_atoms: AtomSelectionInput | str = None,
        min_dist: float = None,
        max_dist: float = None,
        exclude_elements: list[str] = None,
        normalize_by_num_atoms: bool = True,
        overwrite: bool = False,
    ) -> Poses:
        """Run AtomSelection contact detection and merge scores into ``poses``.

        Parameters
        ----------
        poses : Poses
            Input structures to score.
        prefix : str
            Unique run prefix used for the work directory and output columns.
        target_atoms : AtomSelectionInput or str, optional
            Target atom selection for this call. Overrides the instance default.
        reference_atoms : AtomSelectionInput or str, optional
            Reference atom selection for this call. If omitted and no instance
            default is set, all non-target pose atoms are used.
        jobstarter : JobStarter, optional
            Jobstarter for this call.
        min_dist : float, optional
            Lower Angstrom bound for counted contacts.
        max_dist : float, optional
            Upper Angstrom bound for counted contacts.
        exclude_elements : list of str, optional
            Element symbols to remove from both target and reference atoms.
        normalize_by_num_atoms : bool, optional
            If ``True``, divide contact counts by the number of included target
            atoms.
        overwrite : bool, optional
            If ``True``, rerun instead of using a cached score file.

        Returns
        -------
        Poses
            The input ``Poses`` object with contact score columns merged in.
        """

        # Set up the runner directory and choose the effective jobstarter.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter],
        )
        logging.info("Running AtomSelection contact detection in %s on %d poses.", work_dir, len(poses.df.index))

        # Runtime arguments override instance defaults.
        target_atoms = self.target_atoms if target_atoms is None else target_atoms
        reference_atoms = self.reference_atoms if reference_atoms is None else reference_atoms
        min_dist = min_dist if min_dist is not None else self.min_dist
        max_dist = max_dist if max_dist is not None else self.max_dist
        exclude_elements = exclude_elements or self.exclude_elements
        overwrite = overwrite or self.overwrite

        scorefile = os.path.join(work_dir, f"{prefix}_atomselection_contacts.{poses.storage_format}")

        # Reuse cached scores unless a rerun was requested.
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info("Found existing scorefile at %s. Returning cached AtomSelection contact scores.", scorefile)
            return RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()

        # Materialize selections and worker files before dispatching jobs.
        input_dict = self.setup_input_dict(
            poses=poses,
            target_atoms=target_atoms,
            reference_atoms=reference_atoms,
        )
        input_jsons, output_jsons = self.write_input_jsons(
            input_dict=input_dict,
            work_dir=work_dir,
            jobstarter=jobstarter,
        )
        cmds = self.write_cmds(
            input_jsons=input_jsons,
            output_jsons=output_jsons,
            min_dist=min_dist,
            max_dist=max_dist,
            exclude_elements=exclude_elements,
            normalize_by_num_atoms=normalize_by_num_atoms,
        )

        # Execute one worker command per input JSON chunk.
        jobstarter.start(
            cmds=cmds,
            jobname="atomselection_contacts",
            output_path=work_dir,
        )

        # Collect worker JSON outputs and merge them through RunnerOutput.
        scores = _collect_scores(output_files=output_jsons, expected_rows=len(poses))
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        return RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()

    def setup_input_dict(
        self,
        poses: Poses,
        target_atoms: AtomSelectionInput | str,
        reference_atoms: AtomSelectionInput | str = None,
    ) -> dict[str, dict[str, list[Any] | None]]:
        """Build per-pose worker input from target and reference selections.

        Parameters
        ----------
        poses : Poses
            Input structures to score.
        target_atoms : AtomSelectionInput or str
            Target atom selection or dataframe column name.
        reference_atoms : AtomSelectionInput or str, optional
            Reference atom selection or dataframe column name.

        Returns
        -------
        dict
            Mapping of absolute pose paths to JSON-serializable target and
            reference selections.
        """

        # Normalize static and per-pose selections to one list entry per pose.
        target_atom_lists = _setup_atom_selection_list(
            poses=poses,
            selection=target_atoms,
            parameter_name="target_atoms",
        )
        reference_atom_lists = _setup_atom_selection_list(
            poses=poses,
            selection=reference_atoms,
            parameter_name="reference_atoms",
            allow_none=True,
        )

        input_dict = {}

        # Worker JSON uses absolute paths because jobs may run from work_dir.
        for pose, target_selection, reference_selection in zip(poses, target_atom_lists, reference_atom_lists):
            pose_path = os.path.abspath(pose["poses"])
            input_dict[pose_path] = {
                "target_atoms": target_selection,
                "reference_atoms": reference_selection,
            }
        return input_dict

    def write_input_jsons(
        self,
        input_dict: dict[str, dict[str, list[Any] | None]],
        work_dir: str,
        jobstarter: JobStarter,
    ) -> tuple[list[str], list[str]]:
        """Write worker input JSON files and return input/output paths.

        Parameters
        ----------
        input_dict : dict
            Mapping produced by :meth:`setup_input_dict`.
        work_dir : str
            Runner work directory.
        jobstarter : JobStarter
            Selected jobstarter whose ``max_cores`` controls chunk count.

        Returns
        -------
        tuple of list
            Input JSON paths and their matching output JSON paths.
        """
        pose_paths = list(input_dict.keys())
        pose_sublists = split_list(pose_paths, n_sublists=jobstarter.max_cores or 1)

        input_jsons = []
        output_jsons = []

        # Each worker gets a compact pose-path subset and writes one JSON score file.
        for index, pose_sublist in enumerate(pose_sublists, start=1):
            if not pose_sublist:
                continue
            subdict = {pose_path: input_dict[pose_path] for pose_path in pose_sublist}
            input_json = os.path.join(work_dir, f"atomselection_contacts_input_{index:04}.json")
            output_json = os.path.join(work_dir, f"atomselection_contacts_output_{index:04}.json")
            with open(input_json, "w", encoding="UTF-8") as f:
                json.dump(subdict, f)
            input_jsons.append(input_json)
            output_jsons.append(output_json)
        return input_jsons, output_jsons

    def write_cmds(
        self,
        input_jsons: list[str],
        output_jsons: list[str],
        min_dist: float,
        max_dist: float,
        exclude_elements: list[str] = None,
        normalize_by_num_atoms: bool = True,
    ) -> list[str]:
        """Write shell commands for AtomSelection contact worker jobs.

        Parameters
        ----------
        input_jsons : list of str
            Worker input JSON files.
        output_jsons : list of str
            Output JSON paths paired with ``input_jsons``.
        min_dist : float
            Lower Angstrom bound for counted contacts.
        max_dist : float
            Upper Angstrom bound for counted contacts.
        exclude_elements : list of str, optional
            Element symbols to remove from both target and reference atoms.
        normalize_by_num_atoms : bool, optional
            Whether worker jobs should normalize contacts by target atom count.

        Returns
        -------
        list of str
            Shell commands ready for a :class:`~protflow.jobstarters.JobStarter`.
        """

        # Encode list-valued options into simple CLI strings for worker jobs.
        exclude_elements_str = ",".join(exclude_elements or [])
        normalize_flag = " --normalize_by_num_atoms" if normalize_by_num_atoms else ""
        return [
            (
                f"{shlex.quote(self.python)} {shlex.quote(__file__)} "
                f"--input_json {shlex.quote(input_json)} "
                f"--out {shlex.quote(output_json)} "
                f"--min_dist {float(min_dist)} "
                f"--max_dist {float(max_dist)} "
                f"--exclude_elements {shlex.quote(exclude_elements_str)}"
                f"{normalize_flag}"
            )
            for input_json, output_json in zip(input_jsons, output_jsons)
        ]


def main(args: argparse.Namespace) -> None:
    """Run AtomSelection contact calculations from a worker input JSON.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed worker CLI arguments.
    """
    with open(args.input_json, "r", encoding="UTF-8") as f:
        input_dict = json.load(f)

    exclude_elements = args.exclude_elements.split(",") if args.exclude_elements else []
    out_rows = []

    # Worker inputs are already JSON-ready AtomSelection lists.
    for pose_path, selections in input_dict.items():
        contacts = _calc_atomselection_contacts(
            pose=pose_path,
            target_atoms=selections["target_atoms"],
            reference_atoms=selections.get("reference_atoms"),
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            exclude_elements=exclude_elements,
            normalize_by_num_atoms=args.normalize_by_num_atoms,
        )
        out_rows.append(
            {
                "description": os.path.splitext(os.path.basename(pose_path))[0],
                "location": pose_path,
                "contacts": contacts,
            }
        )

    # RunnerOutput expects description and location columns in every score file.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_json(args.out)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="JSON file mapping pose paths to target/reference AtomSelections.")
    argparser.add_argument("--out", type=str, required=True, help="Output JSON path for AtomSelection contact scores.")
    argparser.add_argument("--min_dist", type=float, default=0, help="Lower Angstrom bound for contact distances.")
    argparser.add_argument("--max_dist", type=float, default=5, help="Upper Angstrom bound for contact distances.")
    argparser.add_argument("--exclude_elements", type=str, default="", help="Comma-separated element symbols to exclude from target and reference atoms.")
    argparser.add_argument("--normalize_by_num_atoms", action="store_true", help="Divide contact counts by the number of included target atoms.")

    main(argparser.parse_args())
