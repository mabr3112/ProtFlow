"""
Module to select residues in a Poses class and add the resulting motif into Poses.df.

This module provides the functionality to select specific residues from protein structures 
represented as Poses objects. It includes various selector classes that allow for different 
criteria of residue selection, such as by chain or based on existing residue selections.

Classes:
    - ResidueSelector: Abstract base class for all residue selectors.
    - ChainSelector: Selects all residues of specified chains.
    - TrueSelector: Selects all residues of a pose.
    - NotSelector: Selects all residues except those specified by a residue selection.

Dependencies:
    - protflow.residues
    - protflow.poses
    - protflow.utils.biopython_tools

Examples:

.. code-block:: python

    # Example usage of ChainSelector
    poses = Poses()
    chain_selector = ChainSelector(poses=poses, chain='A')
    chain_selector.select(prefix='selected_chain_A')

    # Example usage of TrueSelector
    true_selector = TrueSelector(poses=poses)
    true_selector.select(prefix='all_residues')

    # Example usage of NotSelector
    residue_selection = ResidueSelection(['A1', 'A2'])
    not_selector = NotSelector(poses=poses, residue_selection=residue_selection)
    not_selector.select(prefix='not_selected')

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Author
------
Markus Braun, Adrian Tripp

Version
-------
0.1.0    
"""

# customs
from typing import Union
from itertools import product

import protflow.residues
from protflow.poses import Poses, col_in_df
from protflow.residues import ResidueSelection
from protflow.utils.biopython_tools import load_structure_from_pdbfile

class ResidueSelector:
    """
    Abstract base class for ResidueSelectors.

    All ResidueSelector classes must implement a select() method that selects residues from Poses.

    Attributes:
        poses (Poses): The Poses object containing the protein structures.
    """

    def __init__(self, poses: Poses = None):
        """
        Initializes the ResidueSelector with an optional Poses object.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.

        Examples:
            poses = Poses()
            selector = ResidueSelector(poses=poses)
        """
        self.set_poses(poses)

    def set_poses(self, poses: Poses = None) -> None:
        """
        Sets the poses for the ResidueSelector class.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.

        Raises:
            TypeError: If the poses parameter is not of type Poses.

        Examples:
            poses = Poses()
            selector.set_poses(poses=poses)
        """
        if not isinstance(poses, Poses) and poses is not None:
            raise TypeError(f"Parameter :poses: must be of type Poses. type(poses) = {type(poses)}")
        self.poses = poses

    def select(self, prefix: str) -> None:
        """
        Abstract method to select residues in poses.

        This method must be implemented by subclasses. The selected residues will be added as
        ResidueSelection objects under the column `prefix` in `Poses.df`.

        Parameters:
            prefix (str): The name of the column that will be added to Poses.df. Poses.df[prefix] holds the selected Residues as a ResidueSelection object.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            >>> class MySelector(ResidueSelector):
            ...     def select(self, prefix):
            ...         # implementation here
            ...         pass
            >>> selector = MySelector(poses=poses)
            >>> selector.select(prefix='selected_residues')
        """
        raise NotImplementedError

    def select_single(self, *args) -> ResidueSelection:
        """
        Abstract method to select residues for a single pose.

        This method must be implemented by subclasses. It returns a ResidueSelection that contains the selected residues of the pose.

        Returns:
            ResidueSelection: The selected residues of the pose.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            >>> class MySelector(ResidueSelector):
            ...     def select_single(self, *args):
            ...         # implementation here
            ...         return ResidueSelection()
            >>> selector = MySelector(poses=poses)
            >>> selection = selector.select_single()
        """
        raise NotImplementedError

class ChainSelector(ResidueSelector):
    """
    Selects all residues of a given chain in Poses.

    This class extends ResidueSelector to allow selection of residues based on specific chains
    from the protein structures contained in a Poses object.

    Attributes:
        poses (Poses): The Poses object containing the protein structures.
        chains (list[str]): A list of chain identifiers to select residues from.
        chain (str): A single chain identifier to select residues from.

    """
    def __init__(self, poses: Poses = None, chain: list = None, chains: list = None):
        """
        Initialize the ChainSelector with optional Poses object and chain(s).

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            chain (list, optional): A single chain identifier. Defaults to None.
            chains (list, optional): A list of chain identifiers. Defaults to None.

        Raises:
            ValueError: If both chain and chains are provided.

        Examples:
            >>> poses = Poses()
            >>> selector = ChainSelector(poses=poses, chain='A')
        """
        if chain and chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {chain}, chains: {chains}")
        super().__init__(poses = poses)
        self.set_chains(chains)
        self.set_chain(chain)

    def set_chains(self, chains: list[str] = None) -> None:
        """
        Sets chains for the select() method.

        Parameters:
            chains (list[str], optional): A list of chain identifiers to select residues from. Defaults to None.

        Raises:
            ValueError: If chains is not a list of single-character strings.

        Examples:
            >>> selector.set_chains(chains=['A', 'B'])
        """
        if not chains:
            self.chains = None
        elif not isinstance(chains, list):
            raise ValueError(f"Parameter :chains: must be a list containing the chains to select as single characters, e.g. chains=['A', 'C']")
        elif not all(isinstance(chain, str) or len(chain) > 1 for chain in chains):
            raise ValueError(f"Parameter :chains: must be a list containing the chains to select as single characters, e.g. chains=['A', 'C']")
        self.chains = chains
        self.chain = None

    def set_chain(self, chain: str = None) -> None:
        """
        Sets a single chain for the select() method.

        Parameters:
            chain (str, optional): A single chain identifier. Defaults to None.

        Raises:
            ValueError: If chain is not a single-character string.

        Examples:
            >>> selector.set_chain(chain='A')
        """
        if not chain:
            self.chain = None
        elif not isinstance(chain, str) or len(chain) > 1:
            raise ValueError(f"Parameter :chain: must be a string of a single character denoting the chain that should be selected. e.g. chain='B' ")
        self.chain = chain
        self.chains = None

    def select(self, prefix: str, poses: Poses = None, chain: str = None, chains: list[str] = None) -> None:
        """
        Selects all residues of a given chain for all poses in a Poses object.

        Selected residues are added as ResidueSelection objects under the column `prefix` in Poses.df.

        Parameters:
            prefix (str): The name of the column that will be added to Poses.df.
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            chain (str, optional): A single chain identifier. Defaults to None.
            chains (list[str], optional): A list of chain identifiers. Defaults to None.

        Raises:
            ValueError: If no poses are provided or set in the instance.
            ValueError: If both chain and chains are provided.
            KeyError: If specified chains are not present in the pose.

        Examples:
            >>> selector.select(prefix='selected_chain_A', chain='A')
        """
        poses.check_prefix(prefix)

        # prep inputs
        chains = self.prep_chain_input(chain, chains)
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")

        # select Residues
        poses.df[prefix] = [self.select_single(pose_path=pose, chains=chains) for pose in poses.poses_list()]

    def select_single(self, pose_path: str, chains: list[str]) -> ResidueSelection: # pylint: disable=W0221
        """
        Selects residues of a given chain of poses and returns them as a ResidueSelection object.

        Parameters:
            pose_path (str): The file path to the pose structure.
            chains (list[str]): A list of chain identifiers.

        Returns:
            ResidueSelection: The selected residues of the pose.

        Raises:
            KeyError: If specified chains are not present in the pose.

        Examples:
            >>> selection = selector.select_single(pose_path='path/to/pose.pdb', chains=['A'])
        """
        pose = load_structure_from_pdbfile(pose_path)

        # check if chain is in chains:
        pose_chains = [chain.id for chain in pose.get_chains()]
        if not all(chain in pose_chains for chain in chains):
            raise KeyError(f"Some of your specified chains {chains} are not present in the pose. chains of pose: {pose_chains}")

        # for all chain in chains collect resis
        residues = []
        for chain in chains:
            resis = pose[chain].get_residues()
            residues += resis

        # convert selected Biopython residues to ResidueSelection object and return
        return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in residues])

    def prep_chain_input(self, chain: str = None, chains: list[str] = None) -> list[str]:
        """
        Prepares chain input for chain selection.

        This method ensures that method parameters take precedence over class attributes.
        This means that ChainSelector.select(chain="A") has higher priority than ChainSelector(chain="C").

        Parameters:
            chain (str, optional): A single chain identifier. Defaults to None.
            chains (list[str], optional): A list of chain identifiers. Defaults to None.

        Returns:
            list[str]: The list of chain identifiers to use for selection.

        Raises:
            ValueError: If both chain and chains are provided.
            ValueError: If both self.chain and self.chains are set.
            ValueError: If no chain identifiers are set.

        Examples:
            >>> chains = selector.prep_chain_input(chain='A')
        """
        # error handling
        if chain and chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {chain}, chains: {chains}")
        if self.chain and self.chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {self.chain}, chains: {self.chains}")
        if all(not param for param in [chain, chains, self.chain, self.chains]):
            raise ValueError(f"Set one of parameters :chain: or :chains: to select a chain with ChainSelector!")

        # handle priorities (method parameters over class parameters)
        class_chains = self.chains if self.chains else [self.chain]
        method_chains = chains if chains else [chain]
        return method_chains or class_chains

class TrueSelector(ResidueSelector):
    """
    ResidueSelector that selects all residues of a pose.

    This class extends ResidueSelector to select all residues from each pose in a Poses object.
    It adds the selected residues as ResidueSelection objects under a specified column in Poses.df.
    """
    def __init__(self, poses: Poses = None):
        """
        Initialize the TrueSelector with an optional Poses object.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.

        Examples:
            >>> poses = Poses()
            >>> selector = TrueSelector(poses=poses)
        """
        super().__init__(poses = poses)

    def select(self, prefix: str, poses: Poses = None):
        """
        Selects all residues of a given pose for all poses in a Poses object.

        Selected residues are added as ResidueSelection objects under the column `prefix` in Poses.df.

        Parameters:
            prefix (str): The name of the column that will be added to Poses.df.
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.

        Raises:
            ValueError: If no poses are provided or set in the instance.

        Examples:
            >>> selector.select(prefix='all_residues')
        """
        # prep inputs and run
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")
        poses.check_prefix(prefix)
        poses.df[prefix] = [self.select_single(pose) for pose in poses.poses_list()]

    def select_single(self, pose_path: str) -> ResidueSelection: # pylint: disable=W0221
        """
        Selects all residues in a single pose and returns them as a ResidueSelection object.

        Parameters:
            pose_path (str): The file path to the pose structure.

        Returns:
            ResidueSelection: The selected residues of the pose.

        Examples:
            >>> selection = selector.select_single(pose_path='path/to/pose.pdb')
        """
        pose = load_structure_from_pdbfile(pose_path)
        return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in pose.get_residues()])

class NotSelector(ResidueSelector):
    """
    ResidueSelector that selects all residues except the ones specified by a residue selection.

    This class extends ResidueSelector to exclude specified residues from selection. 
    The excluded residues can be provided either as a ResidueSelection object or 
    as a contig string.
    """
    def __init__(self, poses: Poses = None, residue_selection: ResidueSelection|str = None, contig: str = None):
        """
        Initialize the NotSelector with optional Poses object and exclusion criteria.

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            residue_selection (ResidueSelection|str, optional): The residues to be excluded from selection. Can be a ResidueSelection object or a string. Defaults to None.
            contig (str, optional): A string specifying the residues to be excluded in a contig format. Defaults to None.

        Raises:
            ValueError: If both residue_selection and contig are provided.

        Examples:
            >>> poses = Poses()
            >>> residue_selection = ResidueSelection(['A1', 'A2'])
            >>> selector = NotSelector(poses=poses, residue_selection=residue_selection)
        """
        super().__init__(poses)
        if residue_selection and contig:
            raise ValueError(f"NotSelector Class cannot be initialized with both parameters :contig: or :residue_selection: set.\n Either choose a residue_selection, or give the residue selectio as a contig, but not both.")
        self.set_residue_selection(residue_selection)
        self.set_contig(contig)

    def set_residue_selection(self, residue_selection: ResidueSelection = None) -> None:
        """
        Sets the residue_selection attribute for the NotSelector class.

        Parameters:
            residue_selection (ResidueSelection, optional): The residues to be excluded from selection. Defaults to None.

        Examples:
            >>> residue_selection = ResidueSelection(['A1', 'A2'])
            >>> selector.set_residue_selection(residue_selection=residue_selection)
        """
        if not residue_selection:
            residue_selection = None
        if isinstance(residue_selection, ResidueSelection) or isinstance(residue_selection, str):
            self.residue_selection = residue_selection
            self.contig = None

    def set_contig(self, contig: str) -> None:
        """
        Sets the contig attribute for the NotSelector class.

        Parameters:
            contig (str): A string specifying the residues to be excluded in a contig format.

        Raises:
            ValueError: If contig is not a string.

        Examples:
            >>> selector.set_contig(contig='A1-7,A25-109,B45-50,C1,C3,C5')
        """
        if not contig:
            self.contig = None
        elif isinstance(contig, str):
            self.contig = contig
            self.residue_selection = None
        else:
            raise ValueError(f"Contig must be of type str. E.g.: contig='A1-7,A25-109,B45-50,C1,C3,C5. contig: {contig}")

    def prep_residue_selection(self, residue_selection: ResidueSelection|str, poses: Poses) -> list[ResidueSelection]:
        """
        Prepares the residue_selection parameter for the select() function.

        Parameters:
            residue_selection (ResidueSelection|str): The residues to be excluded from selection. Can be a ResidueSelection object or a string.
            poses (Poses): The Poses object containing the protein structures.

        Returns:
            list[ResidueSelection]: A list of ResidueSelection objects for each pose.

        Raises:
            TypeError: If the residue_selection parameter is not of a supported type.

        Examples:
            >>> residue_selection_list = selector.prep_residue_selection(residue_selection='selected_residues', poses=poses)
        """
        if isinstance(residue_selection, str):
            poses.check_prefix(residue_selection)
            return poses.df[residue_selection].to_list()
        if isinstance(residue_selection, ResidueSelection):
            return [residue_selection for _ in poses]
        raise TypeError(f"Unsupported argument type {type(residue_selection)} for NotSelector.select(). Only ResidueSelection or 'str' (column in poses.df) are allowed.")

    def select(self, prefix: str, poses: Poses = None, residue_selection: ResidueSelection|str = None, contig: str = None) -> None:
        """
        Selects all residues except the ones specified in residue_selection or by contig.

        The parameter residue_selection can be either a ResidueSelection object or a string pointing to a column in the poses.df that contains ResidueSelection objects.

        Parameters:
            prefix (str): The name of the column that will be added to Poses.df.
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            residue_selection (ResidueSelection|str, optional): The residues to be excluded from selection. Defaults to None.
            contig (str, optional): A string specifying the residues to be excluded in a contig format. Defaults to None.

        Raises:
            ValueError: If no poses are provided or set in the instance.
            ValueError: If both residue_selection and contig are provided.

        Examples:
            >>> selector.select(prefix='not_selected', residue_selection='selected_residues')
        """
        # error handling
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")
        if residue_selection and contig:
            raise ValueError(f"NotSelector Class cannot be initialized with both parameters :contig: or :residue_selection: set.\n Either choose a residue_selection, or give the residue selectio as a contig, but not both.")

        # prep inputs and run
        poses.check_prefix(prefix)
        if contig:
            residue_selection = protflow.residues.from_contig(input_contig=contig)

        # prep residue_selection
        residue_selection_list = self.prep_residue_selection(residue_selection, poses)

        # select
        poses.df[prefix] = [self.select_single(pose, res_sel) for pose, res_sel in zip(poses.poses_list(), residue_selection_list)]

    def select_single(self, pose_path: str, residue_selection: ResidueSelection) -> ResidueSelection: # pylint: disable=W0221
        """
        Selects all residues except the ones specified in residue_selection or by contig.

        Parameters:
            pose_path (str): The file path to the pose structure.
            residue_selection (ResidueSelection): The residues to be excluded from selection.

        Returns:
            ResidueSelection: The selected residues of the pose.

        Examples:
            >>> selection = selector.select_single(pose_path='path/to/pose.pdb', residue_selection=residue_selection)
        """
        pose = load_structure_from_pdbfile(pose_path)

        # load all residues form the pose
        all_res = ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in pose.get_residues()])

        # select not_selection:
        return all_res - residue_selection

class DistanceSelector(ResidueSelector):
    """
    TODO: doc string generation
    Selects all residues that have a certain distance from another residue.

    This class extends ResidueSelector to allow selection of residues based on distances to other residues
    from the protein structures contained in a Poses object.

    Attributes:
        poses (Poses): The Poses object containing the protein structures.


    """
    def __init__(self, center: Union[ResidueSelection, str, list] = None, distance: float = None, operator: str = "<=", poses: Poses = None, center_atoms: Union[list, str] = None, noncenter_atoms: Union[list, str] = None, include_center:bool=False):
        """
        Initialize the DistanceSelector with optional Poses object, center, distance, operator, center_atoms and noncenter_atoms).

        Parameters:
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            center ([ResidueSelection, str, list], optional): A single ResidueSelector, the name of a poses DataFrame column containing ResidueSelectors or a list of ResidueSelectors. Defaults to None.
            distance (float, optional): A float value indicating the distance for residue selection. Defaults to None.
            operator (str, optional): A string indicating the operator that should be used together with :distance: for residue selection. Defaults to '<='.
            center_atoms ([list, str], optional): A string containing a single atom name or a list of atom names from which distances should be calculated. Defaults to None.
            noncenter_atoms ([list, str], optional): A string containing a single atom name or a list of atom names to which distances should be calculated. Defaults to None.
            include_center (bool, optional): Include the center in the output residue selection. Defaults to False.

        Raises:
            ValueError: If both chain and chains are provided.

        Examples:
            >>> poses = Poses()
            >>> selector = ChainSelector(poses=poses, chain='A')
        """

        super().__init__(poses = poses)
        self.set_centers(center)
        self.set_distance(distance)
        self.set_operator(operator)
        self.set_center_atoms(center_atoms)
        self.set_noncenter_atoms(noncenter_atoms)
        self.set_include_center(include_center)

    def set_centers(self, center: Union[ResidueSelection, str, list] = None) -> None:
        """
        Sets centers for the select() method.

        Parameters:
            center ([ResidueSelection, str, list]): A single ResidueSelection, a list of ResidueSelections or the name of a dataframe column containing ResidueSelections.

        Raises:
            ValueError: If center is not a single ResidueSelection, a list of ResidueSelections or the name of a dataframe column containing ResidueSelections.

        Examples:
            >>> selector.set_centers(center="residue_selection_col")
        """
        if not isinstance(center, (ResidueSelection, list, str)):
            raise ValueError("Input to center must be ResidueSelection, a list of ResidueSelections or the name of poses dataframe column containing ResidueSelections!")
        self.center = center

    def set_include_center(self, include_center:bool=False) -> None:
        """
        Sets include_center for the select() method.

        Parameters:
            include_center (bool, optional): True or False. Default False.

        Raises:
            ValueError: If center is not a single ResidueSelection, a list of ResidueSelections or the name of a dataframe column containing ResidueSelections.

        Examples:
            >>> selector.set_include_center(include_center=True)
        """
        if not isinstance(include_center, bool):
            raise ValueError("Input to center must be bool!")
        self.include_center = include_center


    def set_center_atoms(self, center_atoms: Union[str, list] = None) -> None:
        """
        Sets centers for the select() method.

        Parameters:
            center_atoms ([str, list]): A single atom name or a list of atom names. Default is None.

        Raises:
            ValueError: If centcenter_atomser is not a single atom name or a list of atom names.

        Examples:
            >>> selector.set_centers(center="residue_selection_col")
        """
        if not center_atoms:
            self.center_atoms = None
        elif isinstance(center_atoms, str):
            self.center_atoms = [center_atoms]
        elif isinstance(center_atoms, list):
            self.center_atoms = center_atoms
        else:
            raise ValueError("Input to center_atoms must be a list of atom names (e.g. ['N', 'CA', 'C']) or a single atom name (e.g. 'CA')!")

    def set_noncenter_atoms(self, noncenter_atoms: Union[str, list] = None) -> None:
        """
        Sets noncenter_atoms for the select() method.

        Parameters:
            noncenter_atoms ([str, list]): A single atom name or a list of atom names. Default is None.

        Raises:
            ValueError: If noncenter_atoms is not a single atom name or a list of atom names.

        Examples:
            >>> selector.set_centers(center="residue_selection_col")
        """
        if not noncenter_atoms:
            self.noncenter_atoms = None
        elif isinstance(noncenter_atoms, str):
            self.noncenter_atoms = [noncenter_atoms]
        elif isinstance(noncenter_atoms, list):
            self.noncenter_atoms = noncenter_atoms
        else:
            raise ValueError("Input to noncenter_atoms must be a list of atom names (e.g. ['N', 'CA', 'C']) or a single atom name (e.g. 'CA')!")
        
    def set_operator(self, operator: str) -> None:
        """
        Sets the operator for the select() method.

        Parameters:
            operator (str): An operator. Must be one of '<', '>', '<=' or '>='.

        Raises:
            ValueError: If operator is not one of '<', '>', '<=' or '>='..

        Examples:
            >>> selector.set_operator(operator="<")
        """
        if not operator in ["<", ">", "<=", ">="]:
            raise ValueError(f"Operator must be '<', '>', '<=' or '>='!")
        else:
            self.operator = operator
        
    def extract_center(self, center: Union[ResidueSelection, str, list], poses: Poses) -> list:
        """
        Extracts centers from input.

        Parameters:
            center ([ResidueSelection, str, list]): A single ResidueSelection, a list of ResidueSelections or the name of a dataframe column containing ResidueSelections.
            poses (Poses): A poses object.
        Raises:
            ValueError: If center is not a single ResidueSelection, a list of ResidueSelections or the name of a dataframe column containing ResidueSelections.
            ValueError: If the length of the input ResidueSelections is different to the number of poses.

        Examples:
            >>> selector.set_centers(center="residue_selection_col")
        """
        if isinstance(center, str):
            # read in center from poses dataframe column
            col_in_df(poses.df, center)
            centers = self.poses.df[center].to_list()
        elif isinstance(center, list):
            # use poses from list
            centers = center
        elif isinstance(center, ResidueSelection):
            # use input residue selection
            centers = [center for _ in poses.poses_list()]
        if any(not isinstance(sel, ResidueSelection) for sel in centers) or not isinstance(center, (ResidueSelection, list, str)):
            raise ValueError(f"Input to center must be ResidueSelection, a list of ResidueSelections or the name of poses dataframe column containing ResidueSelections, not {type(center)}!")
        return centers

    def set_distance(self, distance: float=None) -> None:
        """
        Sets distance for the select() method.

        Parameters:
            distance (float, optional): A float value. Default None

        Raises:
            ValueError: If distance is not a single float value.

        Examples:
            >>> selector.set_distance(distance=8.4)
        """
        if not distance:
            self.distance = None
        elif isinstance(distance, (float, int)):
            self.distance = distance
        else:
            raise ValueError("Input to distance must be a float!")

    def select(self, prefix: str, poses: Poses = None, center: Union[ResidueSelection, str, list] = None, distance: float = None, operator: str = None, center_atoms: Union[str, list[str]] = None, noncenter_atoms: Union[str, list[str]] = None, include_center:bool=False) -> None:
        """
        Selects all residues with a certain distance from center for all poses in a Poses object.

        Selected residues are added as ResidueSelection objects under the column `prefix` in Poses.df.

        Parameters:
            prefix (str): The name of the column that will be added to Poses.df.
            poses (Poses, optional): The Poses object containing the protein structures. Defaults to None.
            center ([ResidueSelection, str, list], optional): A single ResidueSelector, the name of a poses DataFrame column containing ResidueSelectors or a list of ResidueSelectors. Defaults to None.
            distance (float, optional): A float value indicating the distance for residue selection. Defaults to None.
            operator (str, optional): A string indicating the operator that should be used together with :distance: for residue selection. Defaults to None.
            center_atoms ([list, str], optional): A string containing a single atom name or a list of atom names from which distances should be calculated. Defaults to None.
            noncenter_atoms ([list, str], optional): A string containing a single atom name or a list of atom names to which distances should be calculated. Defaults to None.
            include_center (bool, optional): Include the center in the output residue selection. Defaults to False.

        Raises:
            ValueError: If no poses are provided or set in the instance.
            ValueError: If no distance is provided or set in the instance.
            ValueError: If no operator is provided or set in the instance.
            ValueError: If no center is provided or set in the instance.
            ValueError: If center_atoms or noncenter_atoms is not a string or a list of strings.

        Examples:
            >>> selector.select(prefix='selected_chain_A', chain='A')
        """
        poses.check_prefix(prefix)
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")

        if not (distance := distance or self.distance): 
            raise ValueError(f"You must set a distance for your .select() method. Either with :distance: parameter of .select() or the ResidueSelector.set_distance() method to the class.")

        if not (operator := operator or self.operator):
            raise ValueError(f"You must set an operator for your .select() method. Either with :operator: parameter of .select() or the ResidueSelector.set_operator() method to the class.")

        include_center = include_center or self.include_center

        # pick class center if center is not set
        if not (center := center or self.center): 
            raise ValueError(f"You must set a center for your .select() method. Either with :center: parameter of .select() or the ResidueSelector.set_center() method to the class.")
        centers = self.extract_center(center, poses)

        if not len(centers) == len(poses.poses_list()):
            raise ValueError(f"Number of input ResidueSelections ({len(center)}) must be the same as the number of poses ({len(self.poses.poses_list())})!")

        center_atoms = center_atoms or self.center_atoms
        if isinstance(center_atoms, str):
            center_atoms = [center_atoms]
        elif isinstance(center_atoms, list):
            center_atoms = center_atoms
        elif center_atoms:
            raise ValueError("Input to center_atoms must be a list of atom names (e.g. ['N', 'CA', 'C']) or a single atom name (e.g. 'CA')!")
        
        noncenter_atoms = noncenter_atoms or self.noncenter_atoms
        if isinstance(noncenter_atoms, str):
            noncenter_atoms = [noncenter_atoms]
        elif isinstance(noncenter_atoms, list):
            noncenter_atoms = noncenter_atoms
        elif noncenter_atoms:
            raise ValueError("Input to neighbor_atoms must be a list of atom names (e.g. ['N', 'CA', 'C']) or a single atom name (e.g. 'CA')!")

        # select Residues
        poses.df[prefix] = [self.select_single(pose_path=pose, center=center, distance=distance, operator=operator, center_atoms=center_atoms, noncenter_atoms=noncenter_atoms, include_center=include_center) for pose, center in zip(poses.poses_list(), centers)]

    def select_single(self, pose_path: str, center: ResidueSelection, distance: float, operator: str, center_atoms: list = None, noncenter_atoms: list = None, include_center: bool = False) -> ResidueSelection: # pylint: disable=W0221
        """
        Selects residues of a given chain of poses and returns them as a ResidueSelection object.

        Parameters:
            pose_path (str): The file path to the pose structure.
            center (ResidueSelection): A single ResidueSelection indicating the residues from which distances should be calculated.
            distance (float): A single float value indicating the distance for residue selection.
            operator (str): A single string indicating the operator for residue selection.
            center_atoms (list, optional): A list of atom names for center residues which should be considered for distance calculation.
            noncenter_atoms (list, optional): A list of atom names for noncenter residues which should be considered for distance calculation.
            include_center (bool, optional): Include the center in the output residue selection. Defaults to False.

        Returns:
            ResidueSelection: The selected residues of the pose.

        Raises:
            KeyError: If specified center ResidueSelection is not found in the pose.

        Examples:
            >>> selection = selector.select_single(pose_path='path/to/pose.pdb', chains=['A'])
        """
        pose = load_structure_from_pdbfile(pose_path)

        # get central residues:
        center_res = []
        for chain, resnums in center.to_dict().items():
            # TODO: stupid selection because model[chain][resnum] does not always work (e.g. for ligands/heteroatoms), check back with future BioPython versions
            for resnum in resnums:
                res = [res for res in pose.get_residues() if res.id[1] == resnum and res.parent.id == chain]
                if len(res) == 0:
                    raise KeyError(f"Residue {chain}{resnum} not found in pose {pose_path}!")
                center_res.append(res[0])

        # get all noncenter residues
        noncenter_res = [res for res in pose.get_residues() if not res.id[1] == resnum or not res.parent.id == chain]

        selected_residues = self._determine_residues_in_distance(central_residues=center_res, noncentral_residues=noncenter_res, distance=distance, operator=operator, center_atoms=center_atoms, noncenter_atoms=noncenter_atoms)

        # convert selected Biopython residues to ResidueSelection object and return
        if include_center:
            return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in selected_residues]) + center
        else:
            return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in selected_residues])

    def _determine_residues_in_distance(self, central_residues: list, noncentral_residues:list, distance:float, operator:str, center_atoms:list=None, noncenter_atoms:list=None) -> list[str]:
        """
        Determines residues within distance of central residues.

        Parameters:
            central_residues (list): A single chain identifier. Defaults to None.
            noncentral_residues (list): A list of BioPython residues from which distances should be calculated to all other residues.
            distance (float): A single float value indicating the distance for residue selection.
            operator (str): A single string indicating the operator for residue selection.
            center_atoms (list, optional): A list of atom names for center residues which should be considered for distance calculation. Defaults to None.
            noncenter_atoms (list, optional): A list of atom names for noncenter residues which should be considered for distance calculation. Defaults to None.

        Returns:
            list[str]: The list of chain identifiers to use for selection.

        Raises:
            ValueError: If both chain and chains are provided.
            ValueError: If both self.chain and self.chains are set.
            ValueError: If no chain identifiers are set.

        Examples:
            >>> chains = selector.prep_chain_input(chain='A')
        """
        # get list of all center atoms
        if center_atoms:
            center_atms = [atom for central_res in central_residues for atom in central_res.get_atoms() if atom.id in center_atoms]
        else:
            center_atms = [atom for central_res in central_residues for atom in central_res.get_atoms()]

        # get list of all other atoms
        if noncenter_atoms:
            noncenter_atms = [atom for noncentral_res in noncentral_residues for atom in noncentral_res.get_atoms() if atom.id in noncenter_atoms]
        else:
            noncenter_atms = [atom for noncentral_res in noncentral_residues for atom in noncentral_res.get_atoms()]

        selected = []
        for center_atm, noncenter_atm in product(center_atms, noncenter_atms):
            if operator == "<" and center_atm - noncenter_atm < distance:
                selected.append(noncenter_atm.parent)
            elif operator == ">" and center_atm - noncenter_atm > distance:
                selected.append(noncenter_atm.parent)
            elif operator == ">=" and center_atm - noncenter_atm >= distance:
                selected.append(noncenter_atm.parent)
            elif operator == "<=" and center_atm - noncenter_atm <= distance:
                selected.append(noncenter_atm.parent)

        return list(set(selected))
        
