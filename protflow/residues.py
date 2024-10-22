"""
residues
========

The `residues` module is a part of the `protflow` package and is designed to handle residue selection and related operations in protein structures. This module provides functionality to parse, manipulate, and convert residue selections in various formats, making it an essential tool for bioinformatics and computational biology workflows.

The module includes the `ResidueSelection` class for representing and manipulating selections of residues, as well as various functions for parsing and converting residue selections.

Classes
-------

- `ResidueSelection`
    Represents a selection of residues with functionality for parsing, converting, and manipulating selections.

Functions
---------

- `fast_parse_selection`
    Fast parser for selections already in `ResidueSelection` format.
- `parse_selection`
    Parses a selection into `ResidueSelection` formatted selection.
- `parse_residue`
    Parses a single residue identifier into a tuple (chain, residue_index).
- `residue_selection`
    Creates a `ResidueSelection` from a selection of residues.
- `from_dict`
    Creates a `ResidueSelection` object from a dictionary specifying a motif.
- `from_contig`
    Creates a `ResidueSelection` object from a contig string.
- `reduce_to_unique`
    Reduces an input array to its unique elements while preserving order.

Example Usage
-------------

Creating and manipulating `ResidueSelection` objects:

.. code-block:: python
    
    from residues import ResidueSelection, from_dict, from_contig

    # Create a ResidueSelection from a list
    selection = ResidueSelection(["A1", "A2", "B3"])

    # Convert to string
    selection_str = selection.to_string()
    print(selection_str)
    # Output: A1, A2, B3

    # Convert to dictionary
    selection_dict = selection.to_dict()
    print(selection_dict)
    # Output: {'A': [1, 2], 'B': [3]}

    # Create a ResidueSelection from a dictionary
    selection_from_dict = from_dict({"A": [1, 2], "B": [3]})
    print(selection_from_dict.to_string())
    # Output: A1, A2, B3

    # Create a ResidueSelection from a contig string
    selection_from_contig = from_contig("A1-A3, B5")
    print(selection_from_contig.to_string())
    # Output: A1, A2, A3, B5

This module simplifies the process of handling residue selections in bioinformatics workflows, providing a consistent interface for different types of input and output formats.
"""
# imports
from collections import OrderedDict, defaultdict

class ResidueSelection:
    """
    ResidueSelection
    ================

    A class to represent selections of residues in protein structures. A selection of residues is represented 
    as a tuple with the hierarchy ((chain, residue_idx), ...).

    Parameters
    ----------
    selection : list, optional
        A list of residues in string format, e.g., ["A1", "A2", "B3"]. Default is None.
    delim : str, optional
        The delimiter used to parse the selection string. Default is ",".
    fast : bool, optional
        If True, parses the selection without any type checking. Use when `selection` is already in 
        ResidueSelection format. Default is False.

    Attributes
    ----------
    residues : tuple
        A tuple representing the parsed residues selection.

    Methods
    -------
    from_selection(selection) -> "ResidueSelection"
        Constructs a ResidueSelection instance from the provided selection.
    
    to_string(delim: str = ",", ordering: str = None) -> str
        Converts the ResidueSelection to a string.

    to_list(ordering: str = None) -> list[str]
        Converts the ResidueSelection to a list of strings.

    to_dict() -> dict
        Converts the ResidueSelection to a dictionary. Note: This destroys the ordering of specific residues 
        on the same chain in a motif.

    Examples
    --------
    >>> from residues import ResidueSelection
    >>> selection = ResidueSelection(["A1", "A2", "B3"])
    >>> print(selection.to_string())
    A1, A2, B3
    >>> print(selection.to_dict())
    {'A': [1, 2], 'B': [3]}
    """
    def __init__(self, selection: list = None, delim: str = ",", fast: bool = False, from_scorefile: bool = False):
        self.residues = parse_selection(selection, delim=delim, fast=fast, from_scorefile=from_scorefile)

    def __len__(self) -> int:
        return len(self.residues)

    def __str__(self) -> str:
        return ", ".join([f"{chain}{str(resi)}" for chain, resi in self])

    def __iter__(self):
        return iter(self.residues)

    def __add__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(self.residues + (other - self).residues, fast=True)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(tuple(res for res in self.residues if res not in set(other.residues)), fast=True)
        return NotImplemented

    ####################################### INPUT ##############################################
    def from_selection(self, selection) -> "ResidueSelection":
        """
        Constructs a ResidueSelection instance from the provided selection.

        Parameters
        ----------
        selection : list or str
            The selection of residues to be parsed.

        Returns
        -------
        ResidueSelection
            A new ResidueSelection instance.
        """
        return residue_selection(selection)

    ####################################### OUTPUT #############################################
    def to_string(self, delim: str = ",", ordering: str = None) -> str:
        """
        Converts the ResidueSelection to a string.

        Parameters
        ----------
        delim : str, optional
            The delimiter to use in the resulting string. Default is ",".
        ordering : str, optional
            Specifies the ordering of the residues in the output string. Options are "rosetta" or "pymol".
            Default is None.

        Returns
        -------
        str
            ResidueSelection object formatted as a string, separated by :delim:
            ueSelection.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_string())
        A1, A2, B3
        >>> print(selection.to_string(ordering="rosetta"))
        1A, 2A, 3B
        """
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return delim.join([str(idx) + chain for chain, idx in self])
        if ordering.lower() == "pymol":
            return delim.join([chain + str(idx) for chain, idx in self])
        return delim.join([chain + str(idx) for chain, idx in self])

    def to_list(self, ordering: str = None) -> list[str]:
        """
        Converts the ResidueSelection to a list of strings.

        Parameters
        ----------
        ordering : str, optional
            Specifies the ordering of the residues in the output list. Options are "rosetta" or "pymol".
            Default is None.

        Returns
        -------
        list of str
            The list representation of the ResidueSelection.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_list())
        ['A1', 'A2', 'B3']
        >>> print(selection.to_list(ordering="rosetta"))
        ['1A', '2A', '3B']
        """
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return [str(idx) + chain for chain, idx in self]
        if ordering.lower() == "pymol":
            return [chain + str(idx) for chain, idx in self]
        return [chain+str(idx) for chain, idx in self]

    def to_dict(self) -> dict:
        """
        Converts the ResidueSelection to a dictionary. 

        Note
        ----
        Converting to a dictionary destroys the ordering of specific residues on the same chain in a motif.

        Returns
        -------
        dict
            A dictionary representation of the ResidueSelection with chains as keys and lists of residue 
            indices as values.

        Examples
        --------
        >>> selection = ResidueSelection(["A1", "A2", "B3"])
        >>> print(selection.to_dict())
        {'A': [1, 2], 'B': [3]}
        """
        # collect list of chains and setup chains as dictionary keys
        chains = list(set([x[0] for x in self.residues]))
        out_d = {chain: [] for chain in chains}

        # aggregate all residues to the chains and return
        for (chain, res_id) in self.residues:
            out_d[chain].append(res_id)

        return out_d

    def to_rfdiffusion_contig(self) -> str:
        """
        Parses ResidueSelection object to contig string for RFdiffusion.

        Example:
            If self.residues = (("A", 1), ("A", 2), ("A", 3), ("C", 4), ("C", 6)),
            the output will be "A1-3,C4,C6".
        """
        # Collect residues per chain
        chain_residues = defaultdict(list)
        for chain, resnum in self.residues:
            chain_residues[chain].append(resnum)

        contig_parts = []

        # Process each chain separately
        for chain in sorted(chain_residues.keys()):
            # Sort residue numbers for the chain
            resnums = sorted(chain_residues[chain])

            # Find consecutive ranges
            ranges = []
            start = prev = resnums[0]
            for resnum in resnums[1:]:
                if resnum == prev + 1:
                    # Continue the consecutive range
                    prev = resnum
                else:
                    # End of the current range
                    if start == prev:
                        # Single residue
                        ranges.append(f"{chain}{start}")
                    else:
                        # Range of residues
                        ranges.append(f"{chain}{start}-{prev}")
                    # Start a new range
                    start = prev = resnum
            # Add the last range
            if start == prev:
                ranges.append(f"{chain}{start}")
            else:
                ranges.append(f"{chain}{start}-{prev}")

            # Add ranges to the contig parts
            contig_parts.extend(ranges)

        # Combine all parts into the final contig string
        contig_str = ",".join(contig_parts)
        return contig_str

def fast_parse_selection(input_selection: tuple[tuple[str, int]]) -> tuple[tuple[str, int]]:
    """
    Fast selection parser for pre-formatted selections.

    This function is a fast parser for residue selections that are already in the `ResidueSelection` format.
    It bypasses any additional type checking or parsing to improve performance when the input is guaranteed
    to be correctly formatted.

    Parameters
    ----------
    input_selection : tuple of tuple of (str, int)
        A tuple of tuples where each inner tuple represents a residue with the format (chain, residue_index).

    Returns
    -------
    tuple of tuple of (str, int)
        The input selection, unchanged.

    Examples
    --------
    >>> input_selection = (("A", 1), ("B", 2), ("C", 3))
    >>> fast_parse_selection(input_selection)
    (('A', 1), ('B', 2), ('C', 3))
    """
    return input_selection

def parse_from_scorefile(input_selection: dict) -> tuple[tuple[str, int]]:
    if isinstance(input_selection, dict) and "residues" in input_selection:
        return tuple([tuple(sele) for sele in input_selection["residues"]])
    else:
        raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. This function is meant to parse ResidueSelections that were written to file. Only dict with 'residues' as key allowed.")

def parse_selection(input_selection, delim: str = ",", fast: bool = False, from_scorefile: bool = False) -> tuple[tuple[str,int]]:
    """
    Parses a selection into ResidueSelection formatted selection.

    This function takes a selection of residues in various formats and parses it into the `ResidueSelection` 
    format, which is a tuple of tuples. Each inner tuple represents a residue with the format (chain, residue_index).

    Parameters
    ----------
    input_selection : str, list, or tuple
        The selection of residues to be parsed. This can be:
        - A string with residues separated by a delimiter.
        - A list or tuple of residue strings.
        - A list or tuple of lists/tuples, where each inner list/tuple represents a residue.
    delim : str, optional
        The delimiter used to split the input string if `input_selection` is a string. Default is ",".
    fast : bool, optional
        If True, uses `fast_parse_selection` to bypass type checking and parsing for performance reasons.
        Use when `input_selection` is already in the correct format. Default is False.
    from_scorefile : bool, optional
        If True, parses a residue selection that was read in from a scorefile (in the form {'residues': [['A', 1], ['B', 3]}).
        Default is False.

    Returns
    -------
    tuple of tuple of (str, int)
        A tuple of tuples where each inner tuple represents a residue in the format (chain, residue_index).

    Raises
    ------
    TypeError
        If `input_selection` is not a supported type (str, list, or tuple).

    Examples
    --------
    >>> parse_selection("A1, B2, C3")
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection(["A1", "B2", "C3"])
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection([["A", 1], ["B", 2], ["C", 3]])
    (('A', 1), ('B', 2), ('C', 3))

    >>> parse_selection([("A", 1), ("B", 2), ("C", 3)], fast=True)
    (('A', 1), ('B', 2), ('C', 3))
    """
    if fast and from_scorefile:
        raise RuntimeError(":fast: and :from_scorefile: are mutually exclusive!")
    if fast:
        return fast_parse_selection(input_selection)
    if from_scorefile:
        return parse_from_scorefile(input_selection)
    if isinstance(input_selection, str):
        return tuple(parse_residue(residue.strip()) for residue in input_selection.split(delim))
    if isinstance(input_selection, list) or isinstance(input_selection, tuple):
        if all(isinstance(residue, str) for residue in input_selection):
            return tuple(parse_residue(residue) for residue in input_selection)
        elif all(isinstance(residue, list) or isinstance(residue, tuple) for residue in input_selection):
            return tuple(parse_residue("".join([str(r) for r in residue])) for residue in input_selection)
    raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. Only str and list allowed.")

def parse_residue(residue_identifier: str) -> tuple[str,int]:
    """
    Parses a single residue identifier into a tuple (chain, residue_index).

    This function takes a residue identifier string and parses it into a tuple containing the chain identifier
    and the residue index. It currently only supports single-letter chain identifiers.

    Parameters
    ----------
    residue_identifier : str
        A string representing the residue identifier. The format is expected to be either "chain+residue_index" 
        or "residue_index+chain", where "chain" is a single letter and "residue_index" is an integer.

    Returns
    -------
    tuple of (str, int)
        A tuple containing the chain identifier and the residue index.

    Examples
    --------
    >>> parse_residue("A123")
    ('A', 123)

    >>> parse_residue("123A")
    ('A', 123)

    Notes
    -----
    - The function determines whether the chain identifier is at the beginning or the end of the string based 
      on whether the first character is a digit.
    - Only single-letter chain identifiers are supported.

    """
    chain_first = not residue_identifier[0].isdigit()

    # assemble residue tuple
    chain = residue_identifier[0] if chain_first else residue_identifier[-1]
    residue_index = residue_identifier[1:] if chain_first else residue_identifier[:-1]

    # Convert residue_index to int for accurate typing
    return (chain, int(residue_index))

def residue_selection(input_selection, delim: str = ",") -> ResidueSelection:
    """
    Creates a ResidueSelection from a selection of residues.

    This function takes an input selection of residues in various formats and creates a `ResidueSelection` 
    object. The selection can be provided as a string, list, or tuple.

    Parameters
    ----------
    input_selection : str, list, or tuple
        The selection of residues to be parsed. This can be:
            - A string with residues separated by a delimiter.
            - A list or tuple of residue strings.
            - A list or tuple of lists/tuples, where each inner list/tuple represents a residue.
    delim : str, optional
        The delimiter used to split the input string if `input_selection` is a string. Default is ",".

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> residue_selection("A1, B2, C3")
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>

    >>> residue_selection(["A1", "B2", "C3"])
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>

    >>> residue_selection([["A", 1], ["B", 2], ["C", 3]])
    <ResidueSelection object representing ('A', 1), ('B', 2), ('C', 3)>
    """
    return ResidueSelection(input_selection, delim=delim)

def from_dict(input_dict: dict) -> ResidueSelection:
    """
    Creates a ResidueSelection object from a dictionary.

    This function constructs a `ResidueSelection` instance from a dictionary where the keys represent 
    chain identifiers and the values are lists of residue indices. This format specifies a motif in the 
    following way: {chain: [residues], ...}.

    Parameters
    ----------
    input_dict : dict
        A dictionary specifying the motif. The keys are chain identifiers (str) and the values are lists 
        of residue indices (int).

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> input_dict = {"A": [1, 2], "B": [3, 4]}
    >>> from_dict(input_dict)
    <ResidueSelection object representing ('A', 1), ('A', 2), ('B', 3), ('B', 4)>
    """
    return ResidueSelection([f"{chain}{resi}" for chain, res_l in input_dict.items() for resi in res_l])

def from_contig(input_contig: str) -> ResidueSelection:
    """
    Creates a ResidueSelection object from a contig string.

    This function constructs a `ResidueSelection` instance from a contig string. The contig string can specify 
    ranges of residues using a hyphen (-) to denote the range, with residues separated by commas (,). For example, 
    "A1-A3, B5" specifies residues A1, A2, A3, and B5.

    Parameters
    ----------
    input_contig : str
        A contig string specifying the residues. Ranges can be denoted using hyphens, and residues are separated 
        by commas.

    Returns
    -------
    ResidueSelection
        An instance of the `ResidueSelection` class representing the parsed selection of residues.

    Examples
    --------
    >>> from_contig("A1-A3, B5")
    <ResidueSelection object representing ('A', 1), ('A', 2), ('A', 3), ('B', 5)>

    >>> from_contig("C1, C3-C5, D2")
    <ResidueSelection object representing ('C', 1), ('C', 3), ('C', 4), ('C', 5), ('D', 2)>
    """
    sel = []
    elements = [x.strip() for x in input_contig.split(",") if x]
    for element in elements:
        subsplit = element.split("-")
        if len(subsplit) > 1:
            sel += [element[0] + str(i) for i in range(int(subsplit[0][1:]), int(subsplit[-1])+1)]
        else:
            sel.append(element)
    return ResidueSelection(sel)

def reduce_to_unique(input_array: list|tuple) -> list|tuple:
    """
    Reduces an input array to its unique elements while preserving order.

    This function takes a list or tuple and returns a new list or tuple containing only the unique elements 
    from the input, with their original order preserved. The type of the returned collection matches the type 
    of the input.

    Parameters
    ----------
    input_array : list or tuple
        The input array from which to remove duplicate elements. The order of the elements is preserved.

    Returns
    -------
    list or tuple
        A new list or tuple containing only the unique elements from the input array, with the original order 
        preserved.

    Examples
    --------
    >>> reduce_to_unique([1, 2, 2, 3, 1])
    [1, 2, 3]

    >>> reduce_to_unique(("a", "b", "a", "c", "b"))
    ('a', 'b', 'c')

    Notes
    -----
    - The function uses `OrderedDict.fromkeys` to remove duplicates while preserving order.
    - The returned collection is of the same type as the input (list or tuple).
    """
    return type(input_array)(OrderedDict.fromkeys(input_array))
