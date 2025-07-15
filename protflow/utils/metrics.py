"""
This module provides a comprehensive suite of tools for calculating various metrics related to proteins and their sequences, specifically designed to facilitate detailed analysis and comparisons. The functionalities in this module allow users to determine mutations between protein sequences, calculate structural metrics such as the radius of gyration, and compute sequence identities, among other tasks.

Overview:
The module encompasses a range of utilities aimed at analyzing protein structures and sequences. Users can compare protein sequences to identify mutations, calculate the radius of gyration from PDB files, and assess sequence identity both pairwise and across multiple sequences. Additionally, it provides methods for evaluating protein entropy, ligand interactions, and structural consistency metrics such as self-consistency TM-score and Baker in silico success scores.

Key functionalities include:
- **Mutation Analysis**: Tools to list and count mutations between wild-type and variant sequences, and to identify mutation indices.
- **Structural Metrics**: Calculation of the radius of gyration for protein structures and evaluation of ligand clashes and contacts.
- **Sequence Analysis**: Computation of sequence identity between two sequences and across a list of sequences.
- **Entropy Calculation**: Calculation of entropy based on a given probability distribution.
- **Self-Consistency and Success Scores**: Methods to compute self-consistency TM-scores and Baker in silico success scores within dataframes.
- **Ligand Interaction Analysis**: Evaluation of ligand clashes and contacts within a protein structure.

Examples:
Here are some examples of how to use the functions provided in this module:

1. Counting mutations between two protein sequences:
    ```python
    from metrics import count_mutations
    mutation_count, mutations = count_mutations("ACDEFG", "ACDQFG")
    print(mutation_count, mutations)
    # Output: 1, ['E4Q']
    ```

2. Calculating radius of gyration from a PDB file:
    ```python
    from metrics import calc_rog_of_pdb
    rog = calc_rog_of_pdb("example.pdb")
    print(rog)
    ```

3. Finding mutation indices between two sequences:
    ```python
    from metrics import get_mutation_indeces
    indices = get_mutation_indeces("ACGTAGCT", "ACCTAGCT")
    print(indices)
    # Output: [3]
    ```

4. Calculating sequence identity between two sequences:
    ```python
    from metrics import calc_sequence_identity
    identity = calc_sequence_identity("ACDEFG", "ACDQFG")
    print(identity)
    # Output: 0.8333333333333334
    ```

5. Computing all-against-all sequence identity for a list of sequences:
    ```python
    from metrics import all_against_all_sequence_identity
    identities = all_against_all_sequence_identity(["ACDEFG", "ACDFGG", "ACDEFG"])
    print(identities)
    # Output: [0.8333333333333334, 0.8333333333333334, 1.0]
    ```

6. Calculating entropy from a probability distribution:
    ```python
    from metrics import entropy
    prob_dist = np.array([0.1, 0.2, 0.7])
    ent = entropy(prob_dist)
    print(ent)
    # Output: 1.1567796494470395
    ```

7. Calculating self-consistency TM-score in a dataframe:
    ```python
    from metrics import calc_sc_tm
    df = pd.DataFrame({"ref_col": ["A", "B"], "tm_col": [0.9, 0.85]})
    updated_df = calc_sc_tm(df, "sc_tm_score", "ref_col", "tm_col")
    print(updated_df)
    ```

These examples illustrate the primary capabilities of the module, showcasing how it can be utilized to streamline the process of analyzing protein structures and sequences.
"""
# Imports

# dependencies
import numpy as np
from Bio.PDB.Structure import Structure
import pandas as pd

# customs
from .biopython_tools import get_atoms, load_structure_from_pdbfile
from .utils import vdw_radii

def get_mutations_list(wt: str, variant:str) -> None:
    '''Not implemented.'''
    raise NotImplementedError

def count_mutations(wt: str, variant:str) -> tuple[int, list[str]]:
    """
    Compares two protein sequences and counts the number of mutations, 
    returning both the count and a detailed list of mutations.

    Each mutation is represented in the format: 
    '[original amino acid][position][mutated amino acid]'.
    
    Parameters:
    seq1 (str): The first protein sequence (e.g., wild type).
    seq2 (str): The second protein sequence (e.g., variant).

    Returns:
    tuple[int, list[str]]: A tuple where the first element is an integer 
    representing the number of mutations, and the second element is a list of 
    strings detailing each mutation.

    Raises:
    ValueError: If the input sequences are not of the same length.

    Example:
    >>> count_mutations("ACDEFG", "ACDQFG")
    (1, ['E4Q'])
    """
    # Check if the lengths of the sequences are the same
    if len(wt) != len(variant):
        raise ValueError("Sequences must be of the same length")

    mutations = []
    mutation_count = 0

    for i, (a, b) in enumerate(zip(wt, variant)):
        if a != b:
            mutation_count += 1
            mutation = f"{a}{i+1}{b}"
            mutations.append(mutation)

    return mutation_count, mutations

def get_mutation_indeces(wt: str, variant:str) -> list[int]:
    '''
    Find the indices of mutations between two sequences.
    Can be protein, or nucleic acid sequences.

    Parameters:
    - wt (str): The wild-type sequence.
    - variant (str): The variant sequence.

    Returns:
    - list[int]: A list of indices where mutations occur (1-based index).

    Raises:
    - ValueError: If the lengths of 'wt' and 'variant' sequences are not the same.

    Description:
    This function takes two sequences, 'wt' (wild-type) and 'variant', and returns a list of indices where mutations occur (i.e., where the two sequences differ). The indices are 1-based. If the lengths of 'wt' and 'variant' are not the same, a ValueError is raised.

    Example:
    >>> wt_sequence = "ACGTAGCT"
    >>> variant_sequence = "ACCTAGCT"
    >>> mutations = get_mutation_indeces(wt_sequence, variant_sequence)
    >>> print(mutations)
    [3]
    '''
    # sanity
    if len(wt) != len(variant): raise ValueError(f"wt and variant must be of same length! lengths: wt: {len(wt)} variant: {len(variant)}")

    # convert sequences into arrays
    wt_arr = np.array(list(wt))
    variant_arr = np.array(list(variant))

    # Find indices where mutations occur (1-based index)
    return list(np.where(wt_arr != variant_arr)[0] + 1)

def calc_rog_of_pdb(pdb_path: str, min_dist: float = 0, chain: str = None) -> float:
    """
    Calculate the radius of gyration of a protein from a PDB file.

    This function loads a protein structure from a PDB file and computes the
    radius of gyration for the alpha carbon atoms (Cα).

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file containing the protein structure.
    min_dist : float, optional
        Minimum distance to consider between atoms, by default 0.

    Returns
    -------
    float
        The calculated radius of gyration of the protein.

    Example
    -------
    >>> from metrics import calc_rog_of_pdb
    >>> rog = calc_rog_of_pdb("example.pdb")
    >>> print(rog)
    """
    return calc_rog(load_structure_from_pdbfile(pdb_path), min_dist=min_dist, chain = chain)

def calc_rog(pose: Structure, min_dist: float = 0, chain: str = None) -> float:
    """
    Calculate the radius of gyration of a protein's alpha carbons.

    This function computes the radius of gyration for the alpha carbon atoms (Cα)
    in a given protein structure.

    Parameters
    ----------
    pose : Bio.PDB.Structure.Structure
        A Bio.PDB.Structure.Structure object representing the protein structure.
    min_dist : float, optional
        Minimum distance to consider between atoms, by default 0.

    Returns
    -------
    float
        The calculated radius of gyration of the protein.

    Raises
    ------
    ValueError
        If the `pose` parameter is not of type Bio.PDB.Structure.Structure.

    Example
    -------
    >>> from metrics import calc_rog
    >>> from Bio.PDB import PDBParser
    >>> parser = PDBParser()
    >>> structure = parser.get_structure("example", "example.pdb")
    >>> rog = calc_rog(structure)
    >>> print(rog)
    """
    # get CA coordinates and calculate centroid
    if chain:
        ca_coords = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA" and atom.get_parent().get_parent().id == chain])
    else:
        ca_coords = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
    centroid = np.mean(ca_coords, axis=0)

    # calculate average distance of CA atoms to centroid
    dgram = np.maximum(min_dist, np.linalg.norm(ca_coords - centroid, axis=-1))

    # take root over squared sum of distances and return (rog):
    return np.sqrt(np.sum(dgram**2) / ca_coords.shape[0])

def calc_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Calculate sequence identity between two protein sequences.

    This function computes the sequence identity by comparing two protein sequences
    of the same length and determining the proportion of matching amino acids.

    Parameters
    ----------
    seq1 : str
        The first protein sequence.
    seq2 : str
        The second protein sequence.

    Returns
    -------
    float
        The sequence identity as a fraction of matching amino acids.

    Raises
    ------
    ValueError
        If the input sequences are not of the same length.

    Example
    -------
    >>> from metrics import calc_sequence_identity
    >>> identity = calc_sequence_identity("ACDEFG", "ACDQFG")
    >>> print(identity)
    # Output: 0.8333333333333334
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must be of the same length. Length of seq1: {len(seq1)}, length of seq2: {len(seq2)}")
    matching = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matching / len(seq1)

def all_against_all_sequence_identity(input_seqs: list[str]) -> list:
    """
    Calculate the maximum sequence identity for all sequences against each other.

    This function takes a list of protein sequences and computes the maximum sequence
    identity for each sequence against all others in the list.

    Parameters
    ----------
    input_seqs : list of str
        A list of protein sequences.

    Returns
    -------
    list of float
        A list of maximum sequence identities for each sequence against all others.

    Example
    -------
    >>> from metrics import all_against_all_sequence_identity
    >>> identities = all_against_all_sequence_identity(["ACDEFG", "ACDFGG", "ACDEFG"])
    >>> print(identities)
    # Output: [0.8333333333333334, 0.8333333333333334, 1.0]
    """
    # create a mapping for quick calculation with numpy
    aa_mapping = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    mapped_seqs = np.array([[aa_mapping[s] for s in seq] for seq in input_seqs])

    # create all-against-all array and calculate sequence identity
    expanded_a = mapped_seqs[:, np.newaxis]
    expanded_b = mapped_seqs[np.newaxis, :]
    similarity_matrix = np.mean(expanded_a == expanded_b, axis=2)

    # convert diagonal from 1.0 to -inf
    np.fill_diagonal(similarity_matrix, -np.inf)

    return list(np.max(similarity_matrix, axis=1))

def entropy(prob_distribution: np.ndarray, axis: int = -1) -> float:
    """
    Compute element-wise Shannon entropy H(p) = –∑ p·log₂ p along the given axis,
    safely ignoring any p == 0 terms.

    Parameters
    ----------
    prob_distribution : np.array
        An array representing the probability distribution.

    Returns
    -------
    np.ndarray
        The calculated entropies of the probability distribution.

    Example
    -------
    >>> from metrics import entropy
    >>> prob_dist = np.array([0.1, 0.2, 0.7])
    >>> ent = entropy(prob_dist)
    >>> print(ent)
    # Output: 1.1567796494470395
    """
    # sanity
    prob_distribution = np.asarray(prob_distribution)

    # Suppress log(0) warnings; log₂(0)→–inf, but we’ll zero it out next.
    with np.errstate(divide='ignore', invalid='ignore'):
        plogp = prob_distribution * np.log2(prob_distribution)

    # Wherever p was zero, force p·log₂p → 0
    plogp = np.where(prob_distribution > 0, plogp, 0.0)

    # calculate entropy element wise
    H = -np.sum(plogp, axis=axis)
    return H

def calc_sc_tm(input_df: pd.DataFrame, name: str, ref_col: str, tm_col: str) -> pd.DataFrame:
    """
    Calculate self-consistency TM-score in a dataframe.

    This function computes the self-consistency TM-score for protein structures and
    integrates the results into the input dataframe.

    Parameters
    ----------
    input_df : pd.DataFrame
        A dataframe containing protein structure data.
    name : str
        The name of the new column that should hold the self-consistency TM-score.
    ref_col : str
        The column in `input_df` pointing to the reference description or location.
    tm_col : str
        The column in `input_df` pointing to the TM-scores from TMAlign runner.

    Returns
    -------
    pd.DataFrame
        The input dataframe with the integrated self-consistency TM-score column.

    Raises
    ------
    KeyError
        If the `name` column already exists in `input_df` or if `tm_col` does not exist in `input_df`.
    ValueError
        If `ref_col` does not point to a description or location column in `input_df`.

    Example
    -------
    >>> from metrics import calc_sc_tm
    >>> df = pd.DataFrame({"ref_col": ["A", "B"], "tm_col": [0.9, 0.85]})
    >>> updated_df = calc_sc_tm(df, "sc_tm_score", "ref_col", "tm_col")
    >>> print(updated_df)
    """
    # check if name exists in poses
    if name in input_df.columns:
        raise KeyError(f"Column {name} already present in DataFrame. Choose different name!")
    if tm_col not in input_df.columns:
        raise KeyError(f"Column {tm_col} does not exist in DataFrame. Did you mean any of {[x for x in input_df.columns if tm_col.split('_')[0] in x]}")

    # get descriptions of poses and tm_col_description
    if not ref_col.endswith("description") and not ref_col.endswith("location"):
        raise ValueError(f"Parameter :ref_col: does not point to description or location column in :input_df:. ref_col: {ref_col}")

    # group df by ref_col description and get maximum tm_score in each group.
    grouped_max = input_df.groupby(ref_col)[tm_col].max().reset_index()
    grouped_max.columns = [ref_col, name]

    # merge max tm-scores into input_df[name]
    input_df = input_df.merge(grouped_max, on=ref_col, how='left')
    return input_df

def calc_ligand_clashes(pose: str|Structure, ligand_chain: str, dist: float = 3, atoms: list[str] = None, exclude_ligand_hydrogens: bool = False) -> float:
    """
    Calculate ligand clashes for a PDB file given a ligand chain.

    This method calculates the number of clashes between a specified ligand chain and the rest of the structure in a PDB file or a Bio.PDB Structure object. A clash is defined as any pair of atoms (one from the ligand, one from the rest of the structure) that are within a specified distance of each other.

    Parameters:
        pose (str | Bio.PDB.Structure.Structure): The pose representing the structure, which can be a path to a PDB file (str) or a Bio.PDB Structure object.
        ligand_chain (str): The chain identifier for the ligand within the structure.
        dist (float, optional): The distance threshold for defining a clash. Default is 3.0.
        atoms (list[str], optional): A list of atom names to consider for clash calculations. If None, all atoms are considered. If specified, only these atoms will be included in the clash calculation.

    Returns:
        float: The number of clashes found between the ligand and the rest of the structure.

    Examples:
        Here is an example of how to use the `calc_ligand_clashes` method:

        .. code-block:: python

            from Bio.PDB import PDBParser

            # Load structure from a PDB file
            parser = PDBParser()
            structure = parser.get_structure("example", "example.pdb")

            # Calculate clashes
            clashes = calc_ligand_clashes(structure, ligand_chain="A", dist=3.0, atoms=["N", "CA", "C"])
            # clashes will be a float representing the number of clashes

    Further Details:
        - **Clash Calculation:** The method calculates the Euclidean distance between all specified atoms of the ligand chain and the rest of the structure. A clash is counted if the distance is less than the specified threshold.
        - **Usage:** This function is useful for evaluating potential steric clashes in molecular docking studies or for validating the positioning of ligands in structural models.

    This method is designed to facilitate the detection of steric clashes between ligands and the surrounding structure, providing a quantitative measure of potential conflicts.
    """
    # verify inputs
    if isinstance(pose, str):
        pose = load_structure_from_pdbfile(pose)
    elif not isinstance(pose, Structure):
        raise ValueError(f"Parameter :pose: has to be of type str or Bio.PDB.Structure.Structure. type(pose) = {type(pose)}")

    # check for ligand chain
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # get atoms
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
    else:
        raise ValueError(f"Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")
    if exclude_ligand_hydrogens: 
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if not atom.element == "H"])
    else: 
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms()])

    # calculate clashes
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    return np.sum((dgram < dist))

def calc_ligand_clashes_vdw(pose: str|Structure, ligand_chain: str, factor: float = 1, atoms: list[str] = None, exclude_ligand_elements: list[str] = None) -> int:
    """
    Calculate ligand clashes for a PDB file given a ligand chain.

    This method calculates the number of clashes between a specified ligand chain and the rest of the structure in a PDB file or a Bio.PDB Structure object. A clash is defined as any pair of atoms (one from the ligand, one from the rest of the structure) that are within the sum of their Van der Waals radii multiplied by a factor.

    Parameters:
        pose (str | Bio.PDB.Structure.Structure): The pose representing the structure, which can be a path to a PDB file (str) or a Bio.PDB Structure object.
        ligand_chain (str): The chain identifier for the ligand within the structure.
        factor (float, optional): The multiplier for the VdW clash threshold for defining a clash. Lower numbers result in less stringent clash detection. Default is 1.0.
        atoms (list[str], optional): A list of atom names to consider for clash calculations. If None, all atoms are considered. If specified, only these atoms will be included in the clash calculation.
        exclude_ligand_elements (list[str], optional): A list of elements that should not be considered during clash detection (e.g. ['H']). Default is None

    Returns:
        float: The number of clashes found between the ligand and the rest of the structure.

    Examples:
        Here is an example of how to use the `calc_ligand_clashes` method:

        .. code-block:: python

            from Bio.PDB import PDBParser

            # Load structure from a PDB file
            parser = PDBParser()
            structure = parser.get_structure("example", "example.pdb")

            # Calculate clashes
            clashes = calc_ligand_clashes_vdw(structure, ligand_chain="A", factor=0.8, atoms=["N", "CA", "C"], exclude_ligand_atoms=["H"])
            # clashes will be a float representing the number of clashes

    Further Details:
        - **Clash Calculation:** The method calculates the Euclidean distance between all specified atoms of the ligand chain and the rest of the structure. A clash is detected if the distance is less than the sum of their Van der Waals radii multiplied by a set factor.
        - **Usage:** This function is useful for evaluating potential steric clashes in molecular docking studies or for validating the positioning of ligands in structural models.

    This method is designed to facilitate the detection of steric clashes between ligands and the surrounding structure, providing a quantitative measure of potential conflicts.
    """
    # verify inputs
    if isinstance(pose, str):
        pose = load_structure_from_pdbfile(pose)
    elif not isinstance(pose, Structure):
        raise ValueError(f"Parameter :pose: has to be of type str or Bio.PDB.Structure.Structure. type(pose) = {type(pose)}")

    if exclude_ligand_elements:
        if not isinstance(exclude_ligand_elements, list):
            raise ValueError(f"Parameter:exclude_ligand_atoms: has to be a list of str, not {type(exclude_ligand_elements)}!")
        exclude_ligand_elements = [element.lower() for element in exclude_ligand_elements]

    # import VdW radii
    vdw_dict = vdw_radii()

    # check for ligand chain
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # get atoms
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
        pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
        pose_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms])
    else:
        raise ValueError(f"Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")

    # get ligand atoms
    if exclude_ligand_elements:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
        ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms() if not atom.element.lower() in exclude_ligand_elements])
    else:
        ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms()])
        ligand_vdw = np.array([vdw_dict[atom.element.lower()] for atom in pose[ligand_chain].get_atoms()])

    if np.any(np.isnan(ligand_vdw)):
        raise RuntimeError("Could not find Van der Waals radii for all elements in ligand. Check protflow.utils.vdw_radii and add it, if applicable!")

    # calculate distances between all atoms of ligand and protein
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # calculate distance cutoff for each atom pair, considering VdW radii 
    distance_cutoff = pose_vdw[:, np.newaxis] + ligand_vdw[np.newaxis, :]

    # multiply distance cutoffs with set parameter
    distance_cutoff = distance_cutoff * factor

    # compare distances to distance_cutoff
    check = dgram - distance_cutoff

    # count number of clashes (where distances are below distance cutoff)
    clashes = np.sum((check < 0))

    return clashes

def calc_ligand_contacts(pose: str, ligand_chain: str, min_dist: float = 3, max_dist: float = 5, atoms: list[str] = None, excluded_elements: list[str] = None) -> float:
    """
    Calculate contacts of a ligand within a structure.

    This method calculates the number of contacts between a specified ligand chain and the rest of the structure within a specified distance range. Contacts are defined as any pair of atoms (one from the ligand, one from the rest of the structure) where the distance falls between the minimum and maximum specified distances.

    Parameters:
        pose (str | Bio.PDB.Structure.Structure): The pose representing the structure, which can be a path to a PDB file (str) or a Bio.PDB Structure object.
        ligand_chain (str): The chain identifier for the ligand within the structure.
        min_dist (float, optional): The minimum distance threshold for defining a contact. Default is 3.0.
        max_dist (float, optional): The maximum distance threshold for defining a contact. Default is 5.0.
        atoms (list[str], optional): A list of atom names to consider for contact calculations. If None, all atoms are considered. If specified, only these atoms will be included in the contact calculation.
        excluded_elements (list[str], optional): A list of element symbols to exclude from the contact calculations. Default is ["H"].

    Returns:
        float: The number of contacts normalized by the number of ligand atoms.

    Examples:
        Here is an example of how to use the `calc_ligand_contacts` method:

        .. code-block:: python

            from Bio.PDB import PDBParser

            # Load structure from a PDB file
            parser = PDBParser()
            structure = parser.get_structure("example", "example.pdb")

            # Calculate contacts
            contacts = calc_ligand_contacts(structure, ligand_chain="A", min_dist=3.0, max_dist=5.0, atoms=["N", "CA", "C"], excluded_elements=["H", "O"])
            # contacts will be a float representing the number of contacts normalized by the number of ligand atoms

    Further Details:
        - **Contact Calculation:** The method calculates the Euclidean distance between all specified atoms of the ligand chain and the rest of the structure. A contact is counted if the distance is within the specified range (min_dist to max_dist).
        - **Usage:** This function is useful for evaluating potential interactions between ligands and the surrounding structure, particularly in drug design and molecular docking studies.

    This method is designed to facilitate the detection of relevant contacts between ligands and the surrounding structure, providing a quantitative measure of potential interactions.
    """

    # verify inputs
    if isinstance(pose, str):
        pose = load_structure_from_pdbfile(pose)
    elif not isinstance(pose, Structure):
        raise ValueError(f"Parameter :pose: has to be of type str or Bio.PDB.Structure.Structure. type(pose) = {type(pose)}")

    # check for ligand chain
    pose_chains = list(chain.id for chain in pose.get_chains())
    if ligand_chain not in pose_chains:
        raise KeyError(f"Chain {ligand_chain} not found in pose. Available Chains: {pose_chains}")

    # get pose atoms
    if not atoms or atoms == "all":
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.element not in excluded_elements])
    elif isinstance(atoms, list) and all(isinstance(atom, str) for atom in atoms):
        pose_atoms = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.full_id[2] != ligand_chain and atom.id in atoms and atom.element not in excluded_elements])
    else:
        raise ValueError(f"Invalid Value for parameter :atoms:. For all atoms set to {{None, False, 'all'}} or specify list of atoms e.g. ['N', 'CA', 'CO']")
    ligand_atoms = np.array([atom.get_coord() for atom in pose[ligand_chain].get_atoms() if atom.element not in excluded_elements])

    # calculate complete dgram
    dgram = np.linalg.norm(pose_atoms[:, np.newaxis] - ligand_atoms[np.newaxis, :], axis=-1)

    # return number of contacts
    return np.sum((dgram > min_dist) & (dgram < max_dist)) / len(ligand_atoms)

def residue_contacts(pose:str, max_distance:float, target_chain:str, partner_chain:str, target_resnum: int, target_atom_names:list[str]=None, partner_atom_names:list[str]=None, min_distance:float=0, ):
    # TODO: Write proper docstrings!
    # calculates number of atoms on partner_chain that are between max_distance and min_distance from target_atom_names on target_resnum of chain target_chain.
    pose = load_structure_from_pdbfile(pose)
    target = pose[target_chain][target_resnum] #[res for res in pose[target_chain].get_residues() if res.get_segid() == target_resnum][0]
    partner = pose[partner_chain]
    if target_atom_names:
        target_coords = np.array([atom.get_coord() for atom in target.get_atoms() if atom.id in target_atom_names])
    else:
        target_coords = np.array([atom.get_coord() for atom in target.get_atoms()])

    if partner_atom_names:
        partner_coords = np.array([atom.get_coord() for atom in partner.get_atoms() if atom.id in partner_atom_names])
    else:
        partner_coords = np.array([atom.get_coord() for atom in partner.get_atoms()])

    # calculate complete dgram
    dgram = np.linalg.norm(target_coords[:, np.newaxis] - partner_coords[np.newaxis, :], axis=-1)

    # return number of contacts
    return np.sum((dgram < max_distance) & (dgram > min_distance))

def calc_interchain_contacts(pose: Structure, chains: list[str,str], contact_bounds: tuple[float,float] = (4,8), atoms: list[str] = None) -> float:
    '''Calculates contacts between chains in pose'''
    # get atoms of chains
    chain_a_atoms = np.array([atom.coord for atom in get_atoms(pose, chains=list(chains[0]), atoms=atoms)])
    chain_b_atoms = np.array([atom.coord for atom in get_atoms(pose, chains=list(chains[1]), atoms=atoms)])

    # calc dgram
    dgram = np.linalg.norm(chain_a_atoms[:, np.newaxis] - chain_b_atoms[np.newaxis, :], axis=-1)

    # return number of contacts according to dgram.
    min_dist, max_dist = contact_bounds
    return np.sum((dgram > min_dist) & (dgram < max_dist))

def calc_interchain_contacts_pdb(pdb_path: str, chains: list[str,str], contact_bounds: tuple[float,float] = (4,8), atoms: list[str] = None):
    '''Calculates interchain contacts in pose for .pdb file'''
    pose = load_structure_from_pdbfile(pdb_path)
    return calc_interchain_contacts(pose, chains, contact_bounds, atoms)
