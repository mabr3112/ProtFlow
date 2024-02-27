'''Module for general utility functions of protslurm.'''

def parse_fasta_to_dict(fasta_path: str, encoding:str="UTF-8") -> dict[str:str]:
    '''parses fasta files.'''
    with open(fasta_path, 'r', encoding=encoding) as f:
        fastas = f.read()

    # split along > (separator)
    raw_fasta_list = [x.strip().split("\n") for x in fastas.split(">") if x]

    # parse into dictionary {description: sequence}
    fasta_dict = {x[0]: "".join(x[1:]) for x in raw_fasta_list if len(x) > 1}

    return fasta_dict