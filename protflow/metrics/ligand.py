"""Ligand interaction metrics for ProtFlow.

This module provides runners for simple ligand-environment geometry checks. The
metrics operate on a ligand chain and all other atoms in each pose:

- :class:`LigandClashes` counts ligand-environment atom pairs that are closer
  than a fixed distance or a scaled van der Waals cutoff.
- :class:`LigandContacts` counts ligand-environment atom pairs within a selected
  distance shell.

The runners dispatch per-pose calculations through a
:class:`~protflow.jobstarters.JobStarter`, collect JSON outputs, and merge the
resulting score columns back into :class:`~protflow.poses.Poses`.

Examples
--------
Count C-alpha contacts within 5 A of chain ``B``:

>>> from protflow.metrics.ligand import LigandContacts
>>> contacts = LigandContacts(ligand_chain="B", min_dist=0, max_dist=5, atoms=["CA"], jobstarter=jobstarter)
>>> poses = contacts.run(poses=poses, prefix="lig_contacts")

Count van der Waals clashes against chain ``B`` while ignoring ligand
hydrogens:

>>> from protflow.metrics.ligand import LigandClashes
>>> clashes = LigandClashes(ligand_chain="B", atoms=["CA"], exclude_ligand_elements=["H"], jobstarter=jobstarter)
>>> poses = clashes.run(poses=poses, prefix="lig_clashes")
"""

# import general
import os
import logging
import argparse


# import dependencies
import numpy as np
import pandas as pd

# import customs
from protflow import require_config, load_config_path
from protflow.runners import Runner, RunnerOutput
from protflow.poses import Poses
from protflow.jobstarters import JobStarter, split_list
from protflow.utils.biopython_tools import biopython_load_structure
from protflow.utils.utils import vdw_radii

class LigandClashes(Runner):
    """Count ligand-environment atom clashes for each pose.

    The runner treats all atoms in ``ligand_chain`` as ligand atoms and all atoms
    in other chains as the environment. A clash is counted when a ligand atom and
    environment atom are closer than either ``clash_distance`` or
    ``factor * (vdw_ligand + vdw_environment)``.

    Parameters
    ----------
    ligand_chain : str, optional
        Chain ID containing the ligand atoms.
    factor : float, optional
        Multiplier for van der Waals clash thresholds. Ignored when
        ``clash_distance`` is set.
    atoms : list of str, optional
        Environment atom names to include.
    clash_distance : float, optional
        Fixed Angstrom cutoff for clashes. If omitted, van der Waals radii are
        used.
    exclude_ligand_elements : list of str, optional
        Ligand element symbols to ignore, commonly ``["H"]``.
    jobstarter : JobStarter, optional
        Default jobstarter used when :meth:`run` is called without one.
    overwrite : bool, optional
        Default cache-overwrite behavior for :meth:`run`.
    """
    def __init__(self, ligand_chain: str = None, factor: float = 1, atoms: list[str] = None, clash_distance: float = None, exclude_ligand_elements: list[str] = None, jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
        """Initialize ligand clash defaults for later :meth:`run` calls.

        Parameters
        ----------
        ligand_chain : str, optional
            Chain ID containing ligand atoms.
        factor : float, optional
            Multiplier applied to van der Waals clash cutoffs.
        atoms : list of str, optional
            Environment atom names to include. If omitted, all environment atoms
            are considered.
        clash_distance : float, optional
            Fixed Angstrom cutoff for clashes.
        exclude_ligand_elements : list of str, optional
            Ligand element symbols to ignore.
        jobstarter : JobStarter, optional
            Default jobstarter for this runner instance.
        overwrite : bool, optional
            Whether to overwrite existing cached score files by default.
        """

        # Resolve the worker interpreter from the active ProtFlow config.
        self.python = os.path.join(load_config_path(require_config(), "PROTFLOW_ENV"), "python")

        # Store defaults that can be overridden per run call.
        self.set_ligand_chain(ligand_chain)
        self.set_atoms(atoms)
        self.set_factor(factor)
        self.set_exclude_ligand_elements(exclude_ligand_elements)
        self.set_jobstarter(jobstarter)
        self.set_clash_distance(clash_distance)
        self.overwrite = overwrite

    def __str__(self):
        """Return the short runner name.

        Returns
        -------
        str
            The literal runner name ``"LigandClashes"``.
        """
        return "LigandClashes"
    
    ########################## Input ################################################
    def set_ligand_chain(self, ligand_chain: str) -> None:
        """Set the default ligand chain ID.

        Parameters
        ----------
        ligand_chain : str
            Chain ID containing ligand atoms.
        """
        self.ligand_chain = ligand_chain

    def set_atoms(self, atoms:list[str]) -> None:
        """Set environment atom names used for clash calculations.

        Parameters
        ----------
        atoms : list of str or "all" or None
            Environment atom names to include. ``None`` and ``"all"`` both mean
            all non-ligand atoms.

        Raises
        ------
        TypeError
            If ``atoms`` is neither ``None``, ``"all"``, nor a list of strings.
        """
        if atoms == "all":
            self.atoms = "all"
        if not isinstance(atoms, list) or not all((isinstance(atom, str) for atom in atoms)):
            raise TypeError("Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_factor(self, factor: float) -> None:
        """Set the van der Waals clash-threshold multiplier.

        Parameters
        ----------
        factor : float
            Multiplier applied to summed atom-pair van der Waals radii.
        """
        self.factor = factor

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """Set the default jobstarter for ligand clash jobs.

        Parameters
        ----------
        jobstarter : JobStarter, optional
            Default executor for this runner instance.

        Raises
        ------
        ValueError
            If ``jobstarter`` is neither ``None`` nor a :class:`JobStarter`.
        """
        if isinstance(jobstarter, JobStarter):
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")

    def set_exclude_ligand_elements(self, exclude_ligand_elements: list[str]):
        """Set ligand element symbols to exclude from clash calculations.

        Parameters
        ----------
        exclude_ligand_elements : list of str, optional
            Element symbols ignored on the ligand side.
        """
        self.exclude_ligand_elements = exclude_ligand_elements

    def set_clash_distance(self, clash_distance: float):
        """Set a fixed Angstrom cutoff for ligand clash calculations.

        Parameters
        ----------
        clash_distance : float, optional
            Fixed clash cutoff. If omitted, van der Waals radii are used.
        """
        self.clash_distance = clash_distance

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ligand_chain: str = None, factor: float = 1, clash_distance: float = None, jobstarter: JobStarter = None, atoms: list[str] = None, exclude_ligand_elements: list[str] = None, overwrite: bool = False) -> Poses:
        """Run ligand clash detection and merge scores into ``poses``.

        Parameters
        ----------
        poses : Poses
            Input structures. Each pose must contain ``ligand_chain``.
        prefix : str
            Unique run prefix used for the work directory and output columns.
        ligand_chain : str, optional
            Chain ID containing ligand atoms. Overrides the instance default.
        factor : float, optional
            Van der Waals clash-threshold multiplier.
        clash_distance : float, optional
            Fixed Angstrom cutoff. Overrides van der Waals-based thresholds.
        jobstarter : JobStarter, optional
            Jobstarter for this call.
        atoms : list of str, optional
            Environment atom names to include.
        exclude_ligand_elements : list of str, optional
            Ligand element symbols to ignore.
        overwrite : bool, optional
            If ``True``, rerun instead of using a cached score file.

        Returns
        -------
        Poses
            The input ``Poses`` object with ligand clash columns merged in.
        """
        # Set up the runner directory and choose the effective jobstarter.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running ligand clash detection in {work_dir} on {len(poses.df.index)} poses.")

        ligand_chain = ligand_chain or self.ligand_chain
        atoms = atoms or self.atoms
        factor = factor or self.factor
        exclude_ligand_elements = exclude_ligand_elements or self.exclude_ligand_elements
        atoms_str = f"--atoms {','.join(atoms)}" if atoms else ""
        exclude_ligand_elements_str = f"--exclude_elements {','.join(exclude_ligand_elements)}" if exclude_ligand_elements else ""
        clash_distance = clash_distance or self.clash_distance
        clash_distance_str = f"--clash_distance {clash_distance}" if clash_distance else ""

        scorefile = os.path.join(work_dir, f"{prefix}_ligand_clashes.{poses.storage_format}")

        # Reuse cached clash scores unless overwrite was requested.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # Split pose paths into worker chunks while keeping command lines short.
        poses_sublists = split_list(poses.poses_list(), n_sublists=jobstarter.max_cores) if len(poses.df.index) / jobstarter.max_cores < 100 else split_list(poses.poses_list(), element_length=100)
        out_files = [os.path.join(poses.work_dir, prefix, f"out_{index}.json") for index, _ in enumerate(poses_sublists)]
        cmds = [f"{self.python} {__file__} --poses {','.join(poses_sublist)} --out {out_file} --mode clash_vdw --factor {factor} --ligand_chain {ligand_chain} {atoms_str} {exclude_ligand_elements_str} {clash_distance_str}" for out_file, poses_sublist in zip(out_files, poses_sublists)]

        # Execute one ligand-clash worker per pose chunk.
        jobstarter.start(
            cmds = cmds,
            jobname = "ligand_clash",
            output_path = work_dir
        )

        # Collect worker JSON outputs and merge them through RunnerOutput.
        scores = pd.concat([pd.read_json(output) for output in out_files]).reset_index(drop=True)
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info("Ligand clash detection completed. Returning scores.")
        return output.return_poses()

class LigandContacts(Runner):
    """Count ligand-environment atom contacts for each pose.

    The runner treats atoms in ``ligand_chain`` as ligand atoms and all atoms in
    other chains as the environment. A contact is counted when an environment
    atom and ligand atom are farther than ``min_dist`` and closer than
    ``max_dist``. This provides a simple contact-density proxy for ligand burial
    or pocket engagement.

    Parameters
    ----------
    ligand_chain : str, optional
        Chain ID containing ligand atoms.
    min_dist : float, optional
        Lower Angstrom bound for contact distances.
    max_dist : float, optional
        Upper Angstrom bound for contact distances.
    atoms : list of str, optional
        Environment atom names to include.
    exclude_elements : list of str, optional
        Element symbols to ignore in both ligand and environment atoms, commonly
        ``["H"]``.
    jobstarter : JobStarter, optional
        Default jobstarter used when :meth:`run` is called without one.
    overwrite : bool, optional
        Default cache-overwrite behavior for :meth:`run`.
    """
    def __init__(self, ligand_chain: str = None, min_dist: float = 0, max_dist: float = 5, atoms: list[str] = None, exclude_elements: list[str] = None, jobstarter: JobStarter = None, overwrite: bool = False): # pylint: disable=W0102
        """Initialize ligand contact defaults for later :meth:`run` calls.

        Parameters
        ----------
        ligand_chain : str, optional
            Chain ID containing ligand atoms.
        min_dist : float, optional
            Lower Angstrom bound for counted contacts.
        max_dist : float, optional
            Upper Angstrom bound for counted contacts.
        atoms : list of str, optional
            Environment atom names to include. If omitted, all environment atoms
            are considered.
        exclude_elements : list of str, optional
            Element symbols to remove from both ligand and environment atoms.
        jobstarter : JobStarter, optional
            Default jobstarter for this runner instance.
        overwrite : bool, optional
            Whether to overwrite existing cached score files by default.
        """

        # Resolve the worker interpreter from the active ProtFlow config.
        self.python = os.path.join(load_config_path(require_config(), "PROTFLOW_ENV"), "python")

        # Store defaults that can be overridden per run call.
        self.set_ligand_chain(ligand_chain)
        self.set_atoms(atoms)
        self.set_min_dist(min_dist)
        self.set_max_dist(max_dist)
        self.set_exclude_elements(exclude_elements)
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    def __str__(self):
        """Return the short runner name.

        Returns
        -------
        str
            The literal runner name ``"LigandContacts"``.
        """
        return "LigandContacts"

    ########################## Input ################################################
    def set_ligand_chain(self, ligand_chain: str) -> None:
        """Set the default ligand chain ID.

        Parameters
        ----------
        ligand_chain : str
            Chain ID containing ligand atoms.
        """
        self.ligand_chain = ligand_chain

    def set_atoms(self, atoms:list[str]) -> None:
        """Set environment atom names used for contact calculations.

        Parameters
        ----------
        atoms : list of str or "all" or None
            Environment atom names to include. ``None`` and ``"all"`` both mean
            all non-ligand atoms.

        Raises
        ------
        TypeError
            If ``atoms`` is neither ``None``, ``"all"``, nor a list of strings.
        """
        if atoms == "all":
            self.atoms = "all"
        if not isinstance(atoms, list) or not all((isinstance(atom, str) for atom in atoms)):
            raise TypeError("Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_min_dist(self, min_dist: float) -> None:
        """Set the lower Angstrom bound for contact distances.

        Parameters
        ----------
        min_dist : float
            Minimum distance required for a counted contact.
        """
        self.min_dist = min_dist

    def set_max_dist(self, max_dist: float) -> None:
        """Set the upper Angstrom bound for contact distances.

        Parameters
        ----------
        max_dist : float
            Maximum distance allowed for a counted contact.
        """
        self.max_dist = max_dist

    def set_jobstarter(self, jobstarter: JobStarter) -> None:
        """Set the default jobstarter for ligand contact jobs.

        Parameters
        ----------
        jobstarter : JobStarter, optional
            Default executor for this runner instance.

        Raises
        ------
        ValueError
            If ``jobstarter`` is neither ``None`` nor a :class:`JobStarter`.
        """
        if isinstance(jobstarter, JobStarter):
            self.jobstarter = jobstarter
        else:
            raise ValueError(f"Parameter :jobstarter: must be of type JobStarter. type(jobstarter= = {type(jobstarter)})")
        
    def set_exclude_elements(self, exclude_elements: list[str]):
        """Set element symbols to exclude from contact calculations.

        Parameters
        ----------
        exclude_elements : list of str, optional
            Element symbols ignored on both ligand and environment sides.
        """
        self.exclude_elements = exclude_elements

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ligand_chain: str = None, jobstarter: JobStarter = None, min_dist: float = None, max_dist: float = None, atoms: list[str] = None, exclude_elements: list[str] = None, normalize_by_num_atoms: bool = True, overwrite: bool = False) -> Poses:
        """Run ligand contact detection and merge scores into ``poses``.

        Parameters
        ----------
        poses : Poses
            Input structures. Each pose must contain ``ligand_chain``.
        prefix : str
            Unique run prefix used for the work directory and output columns.
        ligand_chain : str, optional
            Chain ID containing ligand atoms. Overrides the instance default.
        jobstarter : JobStarter, optional
            Jobstarter for this call.
        min_dist : float, optional
            Lower Angstrom bound for contact distances.
        max_dist : float, optional
            Upper Angstrom bound for contact distances.
        atoms : list of str, optional
            Environment atom names to include.
        exclude_elements : list of str, optional
            Element symbols to ignore in both ligand and environment atoms.
        normalize_by_num_atoms : bool, optional
            If ``True``, divide contact counts by the number of included ligand
            atoms.
        overwrite : bool, optional
            If ``True``, rerun instead of using a cached score file.

        Returns
        -------
        Poses
            The input ``Poses`` object with ligand contact columns merged in.
        """
        # Set up the runner directory and choose the effective jobstarter.
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running ligand contact detection in {work_dir} on {len(poses.df.index)} poses.")

        # Runtime arguments override instance defaults.
        ligand_chain = ligand_chain or self.ligand_chain
        min_dist = min_dist or self.min_dist
        max_dist = max_dist or self.max_dist
        if any(attr is None for attr in [ligand_chain, min_dist, max_dist]):
            raise ValueError(f"ligand_chain, min_dist and max_dist must be set, but are {[ligand_chain, min_dist, max_dist]}!")
        atoms = atoms or self.atoms
        exclude_elements = exclude_elements or self.exclude_elements

        atoms_str = f"--atoms {','.join(atoms)}" if atoms else ""
        exclude_elements_str = f"--exclude_elements {','.join(exclude_elements)}" if exclude_elements else ""
        normalize_by_num_atoms_str = "--normalize_by_num_atoms" if normalize_by_num_atoms else ""

        scorefile = os.path.join(work_dir, f"{prefix}_ligand_contacts.{poses.storage_format}")

        # Reuse cached contact scores unless overwrite was requested.
        overwrite = overwrite or self.overwrite
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=self.overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # Split pose paths into worker chunks while keeping command lines short.
        poses_sublists = split_list(poses.poses_list(), n_sublists=jobstarter.max_cores) if len(poses.df.index) / jobstarter.max_cores < 100 else split_list(poses.poses_list(), element_length=100)
        out_files = [os.path.join(poses.work_dir, prefix, f"out_{index}.json") for index, _ in enumerate(poses_sublists)]
        cmds = [f"{self.python} {__file__} --poses {','.join(poses_sublist)} --out {out_file} --min_dist {min_dist} --max_dist {max_dist} --mode contacts --ligand_chain {ligand_chain} {atoms_str} {exclude_elements_str} {normalize_by_num_atoms_str}" for out_file, poses_sublist in zip(out_files, poses_sublists)]

        # Execute one ligand-contact worker per pose chunk.
        jobstarter.start(
            cmds = cmds,
            jobname = "ligand_contacts",
            output_path = work_dir
        )

        # Collect worker JSON outputs and merge them through RunnerOutput.
        scores = pd.concat([pd.read_json(output) for output in out_files]).reset_index(drop=True)
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        output = RunnerOutput(
            poses = poses,
            results = scores,
            prefix = prefix,
        )
        logging.info("Ligand contact detection completed. Returning scores.")
        return output.return_poses()

def _calc_ligand_clashes_vdw(pose: str, ligand_chain: str, factor: float = 1, atoms: list[str] = None, exclude_ligand_elements: list[str] = None, clash_distance: float = None) -> int:
    """Calculate ligand-environment clashes for one structure.

    Parameters
    ----------
    pose : str
        Path to the structure file to score.
    ligand_chain : str
        Chain ID containing ligand atoms.
    factor : float, optional
        Multiplier applied to the summed van der Waals radii of each atom pair.
    atoms : list of str, optional
        Environment atom names to include. If omitted or set to ``"all"``, all
        non-ligand atoms are considered.
    exclude_ligand_elements : list of str, optional
        Ligand element symbols to ignore, commonly ``["H"]``.
    clash_distance : float, optional
        Fixed Angstrom cutoff for clashes. If omitted, van der Waals cutoffs are
        used.

    Returns
    -------
    int
        Number of ligand-environment atom pairs that satisfy the clash cutoff.

    Raises
    ------
    KeyError
        If ``ligand_chain`` is not present in the structure.
    RuntimeError
        If van der Waals mode is used and an element radius is missing.
    ValueError
        If ``atoms`` or ``exclude_ligand_elements`` have invalid types.
    """

    # Load the structure before validating chain and atom-level options.
    pose = biopython_load_structure(pose)

    if exclude_ligand_elements:
        if not isinstance(exclude_ligand_elements, list):
            raise ValueError(f"Parameter:exclude_ligand_atoms: has to be a list of str, not {type(exclude_ligand_elements)}!")
        exclude_ligand_elements = [element.lower() for element in exclude_ligand_elements]

    # Import VdW radii only once per pose calculation.
    vdw_dict = vdw_radii()

    # The ligand chain defines which atoms are separated from the environment.
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # Select environment atoms and, when needed, their VdW radii.
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
        if not clash_distance:
            pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
        if not clash_distance:
            pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
    else:
        raise ValueError("Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")

    # Select ligand atoms after applying optional ligand-side element filters.
    if exclude_ligand_elements:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
        if not clash_distance:
            ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
    else:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms()])
        if not clash_distance:
            ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms()])

    if not clash_distance:
        if np.any(np.isnan(ligand_vdw)): #pylint: disable=E0601
            raise RuntimeError("Could not find Van der Waals radii for all elements in ligand. Check protflow.utils.vdw_radii and add it, if applicable!")

    # Calculate all environment-by-ligand distances in one vectorized matrix.
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)
    if clash_distance:
        return int(np.sum((dgram < clash_distance)))

    # Compare each distance to its atom-pair-specific VdW threshold.
    distance_cutoff = pose_vdw[:, np.newaxis] + ligand_vdw[np.newaxis, :] #pylint: disable=E0601
    distance_cutoff = distance_cutoff * factor
    check = dgram - distance_cutoff
    clashes = int(np.sum((check < 0)))

    return clashes


def _calc_ligand_contacts(pose: str, ligand_chain: str, min_dist: float = 3, max_dist: float = 5, atoms: list[str] = None, exclude_elements: list[str] = None, normalize_by_num_atoms: bool = False) -> float:
    """Calculate ligand-environment contacts for one structure.

    Parameters
    ----------
    pose : str
        Path to the structure file to score.
    ligand_chain : str
        Chain ID containing ligand atoms.
    min_dist : float, optional
        Lower Angstrom bound for counted contacts.
    max_dist : float, optional
        Upper Angstrom bound for counted contacts.
    atoms : list of str, optional
        Environment atom names to include. If omitted or set to ``"all"``, all
        non-ligand atoms are considered.
    exclude_elements : list of str, optional
        Element symbols to remove from both ligand and environment atoms.
    normalize_by_num_atoms : bool, optional
        If ``True``, divide the contact count by the number of included ligand
        atoms.

    Returns
    -------
    float
        Raw contact count, or normalized contacts per ligand atom.

    Raises
    ------
    KeyError
        If ``ligand_chain`` is not present in the structure.
    ValueError
        If ``atoms`` has an invalid type.
    """

    # Load the structure before validating chain and atom-level options.
    pose = biopython_load_structure(pose)

    if exclude_elements:
        exclude_elements = [element.lower() for element in exclude_elements]

    # The ligand chain defines which atoms are separated from the environment.
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # Select environment atoms after applying optional atom-name and element filters.
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.element.lower() not in exclude_elements])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms and atom.element.lower() not in exclude_elements])
    else:
        raise ValueError("Invalid Value for parameter :atoms:. For all atoms set to one of {None, False, 'all'} or specify list of atoms e.g. ['N', 'CA', 'CO']")

    # Ligand atoms are filtered by element only; atom-name filters apply to the environment.
    ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.element.lower() not in exclude_elements])

    # Calculate all environment-by-ligand distances in one vectorized matrix.
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # Keep the same open interval semantics used by the runner wrapper.
    if normalize_by_num_atoms:
        return round(np.sum((dgram > min_dist) & (dgram < max_dist)) / len(ligand_atoms), 2)
    else:
        return np.sum((dgram > min_dist) & (dgram < max_dist))

def main(args):
    """Run ligand metric worker calculations from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed worker CLI arguments produced by the module-level argument
        parser.
    """
    input_poses = args.poses.split(",")
    if args.atoms:
        atoms = args.atoms.split(",")
    else:
        atoms = None
    if args.exclude_elements:
        exclude_elements = args.exclude_elements.split(",")
    else:
        exclude_elements = []

    # Dispatch to the selected worker mode and write RunnerOutput-compatible rows.
    if args.mode == "clash_vdw":
        clashes = [_calc_ligand_clashes_vdw(pose, args.ligand_chain, args.factor, atoms, exclude_elements, args.clash_distance) for pose in input_poses]
        out_dict = {"description": [os.path.splitext(os.path.basename(pose))[0] for pose in input_poses], "location": input_poses, "clashes": clashes}
        df = pd.DataFrame(out_dict)
        df.to_json(args.out)

    elif args.mode == "contacts":
        contacts = [_calc_ligand_contacts(pose, args.ligand_chain, args.min_dist, args.max_dist, atoms, exclude_elements, args.normalize_by_num_atoms) for pose in input_poses]
        out_dict = {"description": [os.path.splitext(os.path.basename(pose))[0] for pose in input_poses], "location": input_poses, "contacts": contacts}
        df = pd.DataFrame(out_dict)
        df.to_json(args.out)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--poses", type=str, required=True, help="Comma-separated PDB paths to score.")
    argparser.add_argument("--out", type=str, required=True, help="Output JSON path for the ligand metric scores.")
    argparser.add_argument("--factor", type=float, default=None, help="Multiplier for van der Waals clash cutoffs in clash_vdw mode.")
    argparser.add_argument("--ligand_chain", type=str, required=True, help="Chain ID containing the ligand atoms.")
    argparser.add_argument("--atoms", type=str, default=None, help="Comma-separated environment atom names to include, for example 'CA,CB'.")
    argparser.add_argument("--exclude_elements", type=str, default=None, help="Comma-separated ligand/contact element symbols to exclude, for example 'H'.")
    argparser.add_argument("--mode", type=str, required=True, help="Metric mode to run: 'clash_vdw' or 'contacts'.")
    argparser.add_argument("--min_dist", type=float, default=0, help="Lower Angstrom bound for contact distances in contacts mode.")
    argparser.add_argument("--max_dist", type=float, default=5, help="Upper Angstrom bound for contact distances in contacts mode.")
    argparser.add_argument("--clash_distance", type=float, default=None, help="Fixed Angstrom clash cutoff; overrides van der Waals cutoffs when set.")
    argparser.add_argument("--normalize_by_num_atoms", action="store_true", help="Divide contact counts by the number of included ligand atoms.")

    arguments = argparser.parse_args()
    main(arguments)
