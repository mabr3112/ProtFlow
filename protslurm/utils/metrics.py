'''
Protslurm internal module to calculate various kinds of metrics for proteins and their sequences.
'''
# Imports

# dependencies
import numpy as np
from Bio.PDB.Structure import Structure
import pandas as pd

from protslurm.utils.biopython_tools import load_structure_from_pdbfile

#import protslurm.utils.biopython_tools as bio_tools

def get_mutations_list(wt: str, variant:str) -> None:
    '''AAA'''
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

def calc_rog_of_pdb(pdb_path: str, min_dist: float = 0) -> float:
    '''Calculates radius of gyration of a protein when given a .pdb input file.'''
    return calc_rog(load_structure_from_pdbfile(pdb_path), min_dist=min_dist)

def calc_rog(pose: Structure, min_dist: float = 0) -> float:
    '''Calculates radius of gyration of a protein's alpha carbons.
     Input (pose:)   a Bio.PDB.Structure.Structure object.'''
    # get CA coordinates and calculate centroid
    ca_coords = np.array([atom.get_coord() for atom in pose.get_atoms() if atom.id == "CA"])
    centroid = np.mean(ca_coords, axis=0)

    # calculate average distance of CA atoms to centroid
    dgram = np.maximum(min_dist, np.linalg.norm(ca_coords - centroid, axis=-1))

    # take root over squared sum of distances and return (rog):
    return np.sqrt(np.sum(dgram**2) / ca_coords.shape[0])

def calc_sequence_identity(seq1: str, seq2: str) -> float:
    '''Calculates sequence identity between two sequences (seq1, seq2)'''
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must be of the same length. Length of seq1: {len(seq1)}, length of seq2: {len(seq2)}")
    matching = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matching / len(seq1)

def all_against_all_sequence_identity(input_seqs: list[str]) -> list:
    '''Given a list of sequences, the function calculates the maximum sequence identity of all sequences against each other.
    Returns a 1D List of max-sequence-identity for each sequence (len(input_seqs) == len(output_seqs))'''
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

def entropy(prob_distribution: np.array) -> float:
    '''Computes entropy when given a probability distribution as an input.'''
    # Filter out zero probabilities to avoid log(0)
    prob_distribution = prob_distribution[prob_distribution > 0]

    # Compute entropy
    H = np.sum(prob_distribution * np.log2(prob_distribution))

    return -H

def calc_sc_tm(input_df: pd.DataFrame, name: str, ref_col: str, tm_col: str) -> None:
    '''Calculates self-consistency TM-score in a dataframe.
    Calculates in-place. This means that the score is automatically added to input_df.
    
    Parameters:
        :name: (str) Name of the new column that should hold sc-TM score.
        :input_df: (pd.DataFrame) poses.df formatted pd.DataFrame
        :ref_col: (str) column pointing to the reference description or location.
        :tm_col: (str) column pointing to the column holding the tm_scores from TMAlign runner.
                        This is generally '{prefix}_TM_score_ref', where {prefix} is the prefix used for the TMAlign runner.

    Returns:
        None! New column gets integrated into the dataframe automatically. 
    '''
    # check if name exists in poses
    if name in input_df.columns:
        raise KeyError(f"Column {name} already present in DataFrame. Choose different name!")
    if tm_col not in input_df.columns:
        raise KeyError(f"Column {tm_col} does not exist in DataFrame. Did you mean any of {[x for x in input_df.columns if tm_col.split('_')[0] in x]}")

    # get descriptions of poses and tm_col_description
    if ref_col.endswith("description"):
        desc_col = ref_col
    elif ref_col.endswith("location"):
        desc_col = ref_col.replace("location", "description")
    else:
        raise ValueError(f"Parameter :ref_col: does not point to description or location column in :input_df:. ref_col: {ref_col}")

    # group df by ref_col description and get maximum tm_score in each group.
    grouped_max = input_df.groupby(desc_col)[tm_col].max().reset_index()
    grouped_max.columns = [desc_col, name]

    # merge max tm-scores into input_df[name]
    input_df = input_df.merge(grouped_max, on=desc_col, how='left')

def calc_baker_success(input_df: pd.DataFrame, af2_col: str, bb_rmsd_col: str, motif_rmsd_col: str) -> None:
    '''Implementation of Baker in silico success score. See RFDiffusion publication for info.'''
    return NotImplementedError
