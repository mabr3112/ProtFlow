"""
General Utility Functions for ProtFlow

This module provides a collection of general utility functions designed to support various
operations within the ProtFlow package. These utilities include functions for parsing data files,
calculating molecular interactions, and other common tasks needed in bioinformatics and structural
biology workflows.

Examples:
    Here is an example of how to use the `parse_fasta_to_dict` function:

    .. code-block:: python

        # Parse a FASTA file
        fasta_dict = parse_fasta_to_dict('example.fasta')
        for desc, seq in fasta_dict.items():
            print(f"{desc}: {seq}")

This module is designed to provide essential utilities for common tasks encountered in
bioinformatics and structural biology, facilitating the development of more complex workflows
within the ProtFlow package.

Authors
-------
Markus Braun, Adrian Tripp
"""
import os
import pandas as pd

def parse_fasta_to_dict(fasta_path: str, encoding:str="UTF-8") -> dict[str:str]:
    '''
    Parses a FASTA file, converting it into a dictionary mapping sequence descriptions to sequences.

    This function opens and reads a FASTA file from the given path, then parses the contents
    to create a dictionary. Each entry in the FASTA file should start with a '>' character,
    followed by the description line. The subsequent lines until the next '>' character are
    considered as the sequence associated with that description. The sequence is concatenated
    into a single string if it spans multiple lines.

    Parameters
    ----------
    fasta_path : str
        The file path to the FASTA file that needs to be parsed. The path should be a valid
        path to a file that exists and is readable. If the file cannot be found or opened,
        a `FileNotFoundError` will be raised.
    encoding : str, optional
        The character encoding of the FASTA file. This is useful for files that might have
        been created in non-UTF-8 encoding. Defaults to "UTF-8".

    Returns
    -------
    dict[str, str]
        A dictionary where the keys are the descriptions of sequences (without the '>' character),
        and the values are the sequences themselves. Sequences that span multiple lines in the
        FASTA file are concatenated into a single string.

    Examples
    --------
    Assuming we have a FASTA file `example.fasta` with the following content:

        >seq1
        AGTCAGTC
        >seq2
        GTCAACGT

    Parsing this file:

        >>> fasta_dict = parse_fasta_to_dict('example.fasta')
        >>> fasta_dict['seq1']
        'AGTCAGTC'
        >>> fasta_dict['seq2']
        'GTCAACGT'
    '''
    with open(fasta_path, 'r', encoding=encoding) as f:
        fastas = f.read()

    # split along > (separator)
    raw_fasta_list = [x.strip().split("\n") for x in fastas.split(">") if x]

    # parse into dictionary {description: sequence}
    fasta_dict = {x[0]: "".join(x[1:]) for x in raw_fasta_list if len(x) > 1}

    return fasta_dict

def sequence_dict_to_fasta(seq_dict: dict, out_path: str, combined_filename: str = None) -> None:
    '''Writes protein sequences stored into seq_dict {'description': seq, ...} to .fa files. If combined_filename is specified, all sequences will be written into one file.'''
    # make sure out_path exists
    os.makedirs(out_path, exist_ok=True)

    # if combined_filename is specified, write everything into one .fa file.
    if combined_filename:
        with open(f"{out_path}/{combined_filename}", 'w', encoding="UTF-8") as f:
            f.write("\n".join([f">{desc}\n{seq}" for desc, seq in seq_dict.items()]) + "\n")
        return

    # otherwise, write every sequence into its own .fa file, named after the 'description' (will also be put next to >)
    for description, seq in seq_dict.items():
        with open(f"{out_path}/{description}.fa", 'w', encoding="UTF-8") as f:
            f.write(f">{description}\n{seq}\n")

def vdw_radii() -> dict[str:float]:
    '''
    from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page), accessed 30.1.2023
    '''
    vdw_radii = {
        "h":1.2,
        "he":1.4,
        "li":1.82,
        "be":1.53,
        "b":1.92,
        "c":1.7,
        "n":1.55,
        "o":1.52,
        "f":1.47,
        "ne":1.54,
        "na":2.27,
        "mg":1.73,
        "al":1.84,
        "si":2.1,
        "p":1.8,
        "s":1.8,
        "cl":1.75,
        "ar":0.71,
        "k":2.75,
        "ca":2.31,
        "sc":2.11,
        "ti":None,
        "v":0, # set to zero because Rosetta uses V as virtual atoms
        "cr":None,
        "mn":None,
        "fe":2.44,
        "co":None,
        "ni":1.63,
        "cu":1.4,
        "zn":1.39,
        "ga":1.87,
        "ge":2.11,
        "as":1.85,
        "se":1.9,
        "br":1.85,
        "kr":2.02,
        "rb":3.03,
        "sr":2.49,
        "y":None,
        "zr":None,
        "nb":None,
        "mo":2.45,
        "tc":None,
        "ru":1.46,
        "rh":None,
        "pd":1.63,
        "ag":1.72,
        "cd":1.58,
        "in":1.93,
        "sn":2.17,
        "sb":2.06,
        "te":2.06,
        "i":1.98,
        "xe":2.16,
        "cs":3.43,
        "ba":2.68,
        "la":None,
        "ce":None,
        "pr":None,
        "nd":None,
        "pm":None,
        "sm":None,
        "eu":None,
        "gd":None,
        "tb":None,
        "dy":None,
        "ho":None,
        "er":None,
        "tm":None,
        "yb":None,
        "lu":None,
        "hf":None,
        "ta":None,
        "w":None,
        "re":None,
        "os":None,
        "ir":None,
        "pt":1.75,
        "au":1.66,
        "hg":1.55,
        "tl":1.96,
        "pb":2.02,
        "bi":2.07,
        "po":1.97,
        "at":2.02,
        "rn":2.2,
        "fr":3.48,
        "ra":2.83,
        "ac":None
        }
    return vdw_radii

def _mutually_exclusive(opt_a, name_a: str, opt_b, name_b: str, none_ok: bool = False):
    if opt_a and opt_b:
        raise ValueError(f"Paramters '{name_a}' and '{name_b}' are mutually exclusive. Specify either one of them, but not both.")
    if not (opt_a or opt_b or none_ok):
        raise ValueError(f"At least one of parameters {name_a} or {name_b} must be set.")

def add_group_statistics(df: pd.DataFrame, group_col: str, prefix: str, statistics: list = ('min', 'mean', 'median', 'max', 'std')) -> pd.DataFrame:
    """
    Adds group-based statistical features to the DataFrame.

    This function groups the DataFrame by the specified `group_col` and computes
    the specified statistics (default: min, mean, median, max, std) for all columns
    that start with the given `prefix`. The computed statistics are then merged
    back into the original DataFrame, with new column names indicating the
    statistic and original column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to process.
        
    group_col : str
        The name of the column to group by.
        
    prefix : str
        The prefix string to filter columns for which statistics will be computed.
        Only columns starting with this prefix will be considered.
        
    statistics : list of str, optional
        The list of statistical functions to apply. Supported statistics include
        'min', 'mean', 'median', 'max', 'std' by default. You can customize this
        list as needed, provided that the functions are supported by pandas.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the original data along with additional columns
        for each computed statistic. The new columns are named in the format
        `<original_column>_<statistic>`.
        
    Raises
    ------
    ValueError
        If the specified `group_col` does not exist in the DataFrame.
        If no columns match the specified `prefix`.
        If any of the specified `statistics` are not supported by pandas.

    Example
    -------
    ```python
    import pandas as pd

    data = {
        'group_col': ['A', 'A', 'B', 'B', 'B'],
        'start_str1': [10, 20, 30, 40, 50],
        'start_str2': [5, 15, 25, 35, 45],
        'other_col': [100, 200, 300, 400, 500]
    }
    df = pd.DataFrame(data)
    
    df_with_stats = add_group_statistics(df, group_col='group_col', prefix='start_str')
    print(df_with_stats)
    ```

    Output:
    ```
      group_col  start_str1  start_str2  other_col  start_str1_min  start_str1_mean  start_str1_median  start_str1_max  start_str1_std  start_str2_min  start_str2_mean  start_str2_median  start_str2_max  start_str2_std
    0         A          10           5        100              10              15.0               15.0              20          7.071068               5              10.0                10.0              15          7.071068
    1         A          20          15        200              10              15.0               15.0              20          7.071068               5              10.0                10.0              15          7.071068
    2         B          30          25        300              30              40.0               40.0              50         10.000000              25              35.0                35.0              45         10.000000
    3         B          40          35        400              30              40.0               40.0              50         10.000000              25              35.0                35.0              45         10.000000
    4         B          50          45        500              30              40.0               40.0              50         10.000000              25              35.0                35.0              45         10.000000
    ```

    Notes
    -----
    - The function assumes that the specified `group_col` exists in the DataFrame.
    - Columns to compute statistics on are selected based on the provided prefix.
    - The resulting DataFrame will have additional columns for each statistic applied
      to each selected column.
    - If the original DataFrame contains columns with names that could collide with
      the new statistical columns, consider renaming them before using this function
      to avoid unintended overwrites.

    """
    # Check if group_col exists in the DataFrame
    if group_col not in df.columns:
        raise ValueError(f"The group_col '{group_col}' does not exist in the DataFrame.")

    # Select columns that start with the given prefix
    cols_to_aggregate = [col for col in df.columns if col.startswith(prefix)]

    if not cols_to_aggregate:
        raise ValueError(f"No columns start with the prefix '{prefix}'.")

    # Verify that all specified statistics are supported by pandas
    supported_stats = {'min', 'mean', 'median', 'max', 'std', 'sum', 'count', 'min'}
    if not set(statistics).issubset(supported_stats):
        unsupported = set(statistics) - supported_stats
        raise ValueError(f"Unsupported statistics provided: {unsupported}. "
                         f"Supported statistics are: {supported_stats}")

    # Group by the group_col and compute the statistics
    grouped = df.groupby(group_col)[cols_to_aggregate].agg(statistics)

    # Flatten the MultiIndex columns
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    # Reset index to prepare for merging
    grouped = grouped.reset_index()

    # Merge the aggregated statistics back to the original DataFrame
    df_merged = df.merge(grouped, on=group_col, how='left', suffixes=('', '_group'))

    return df_merged
