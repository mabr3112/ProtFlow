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
        "v":None,
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
