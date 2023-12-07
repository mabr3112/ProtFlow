'''
Protslurm internal module to calculate various kinds of metrics for proteins and their sequences.
'''
# Imports

# dependencies
import numpy as np

#import protslurm.utils.biopython_tools as bio_tools

def get_mutations_list(wt: str, variant:str) -> None:
    '''AAA'''
    raise NotImplementedError

def count_mutations(wt: str, variant:str) -> int:
    '''Counts mutations between wt and variant and returns count (int)'''
    raise NotImplementedError

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
