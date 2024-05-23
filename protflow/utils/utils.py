'''Module for general utility functions of protflow.'''

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
