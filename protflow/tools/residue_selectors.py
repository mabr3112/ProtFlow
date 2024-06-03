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
import protflow.residues
from protflow.poses import Poses
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
        if not isinstance(poses, Poses):
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
        if not isinstance(chains, list):
            raise ValueError(f"Parameter :chains: must be a list containing the chains to select as single characters, e.g. chains=['A', 'C']")
        if not all(isinstance(chain, str) or len(chain) > 1 for chain in chains):
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
        if not isinstance(chain, str) or len(chain) > 1:
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
        if not isinstance(contig, str):
            raise ValueError(f"Contig must be of type str. E.g.: contig='A1-7,A25-109,B45-50,C1,C3,C5")
        self.contig = contig
        self.residue_selection = None

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
            return poses[residue_selection].to_list()
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
