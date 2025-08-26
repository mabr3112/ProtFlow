.. _load_poses:

Create Poses Tutorial
======================

Poses are the central class of the ProtFlow package. They store information like paths to design models and the corresponding scores and are the input for the different design tools.
This tutorial can also be completed interactively, by working in the Jupyter notebook found at 
`ProtFlow/examples/load_poses.ipynb <https://github.com/mabr3112/ProtFlow/blob/master/examples/load_poses.ipynb>`_. It can be opened with programs such as Visual Studio Code.
Input PDBs can be found in the same directory. Before we start, make sure the ProtFlow config file is set up properly and the ProtFlow environment is active. 

.. note::

   If you are having trouble setting up ProtFlow, please check out the
   :doc:`Quickstart guide </quickstart/index>`.

.. code-block:: python

   from protflow.poses import Poses

We can load poses either from a directory, a list of files or from a previously generated poses scorefile. We will start by loading them from a directory containing .pdb files 
by providing the path to the directory and a glob suffix, which indicates that only files ending with .pdb should be loaded. We can then look at the poses dataframe, 
which will be automatically created upon initialization of the poses. 

.. code-block:: python

   # load poses from directory
   my_poses = Poses(poses='path/to/input_pdbs/', glob_suffix='*pdb')

   # show poses dataframe
   print(my_poses.df)

The poses dataframe always contains the path to the current poses ('poses' column) and the name of the current poses ('poses_description' column). Additionally, it can 
contain various scores and other infos. The length of the dataframe always corresponds to the current number of poses. The current poses dataframe can be saved using the save 
scores attribute to the path indicated with <out_path> and the file extension <out_format>.

.. code-block:: python

   my_poses.save_scores(out_path="path/to/poses_examples", out_format="json")

We can also load poses by passing a list of pdb files instead of just loading all pdb files from a directory:

.. code-block:: python

   # load poses from list
   my_poses = Poses(poses=['path/to/structure_1.pdb', 'path/to/structure_2.pdb'])

   # show poses dataframe
   print(my_poses.df)

Alternatively, we can load the previously saved poses dataframe to create new poses. You can also directly pass a dataframe to poses instead of loading it from a file 
(but be careful, it must always contain the columns 'input_poses', 'poses' and 'poses_description'!).

.. code-block:: python
   
   # load poses from scorefile
   my_poses = Poses(poses="path/to/poses_examples.json")

   # show poses dataframe
   print(my_poses.df)

Setting up poses
----------------

In order to manipulate our poses, we have to set our working directory. This is also the directory where our poses dataframe will be automatically saved. The directory 
will be created (including subdirectories for things like filters, plots and scores) if it does not exist. Alternatively, you can also directly set the working directory when 
loading the poses using <work_dir>. If save_scores is used without attributes, the scorefile will always be saved to the working directory and named after this directory. 
We can modify the default scorefile format using the set_storage_format attribute.

.. code-block:: python
   
   # set up working directory
   my_poses.set_work_dir('load_poses_example')
   print(my_poses.work_dir)

   # define a new storage format
   my_poses.set_storage_format(storage_format="csv")
   print(my_poses.storage_format)

   # save scores to working directory
   my_poses.save_scores()

Another important thing to consider is setting a default jobstarter. This jobstarter will be used for any runner if no explicit jobstarter is provided. Jobstarters handle 
how compute jobs are distributed and will be explained in detail in the # TODO: jobstarter tutorial. 

.. code-block:: python
   
   # import the jobstarter
   from protflow.jobstarters import LocalJobStarter

   # define the jobstarter you want to use. In this case, we use the local jobstarter which runs everything on the current machine and does not use any job management applications like SLURM
   my_jobstarter = LocalJobStarter()

   my_poses.set_jobstarter(jobstarter=my_jobstarter)
   print(my_poses.default_jobstarter)

As mentioned before, all of these settings can be directly defined when setting up the poses:


LigandMPNN created structures with amino acid sequences out of our backbones. To improve our backbones, we are going to employ Rosetta Relax, a specialized Rosetta protocol that optimizes
protein structures by minimizing energies via introduction of small movements. 

.. code-block:: python

   my_poses = Poses(poses='path/to/input_pdbs/', glob_suffix='*pdb', work_dir='load_poses_example', storage_format='csv', jobstarter=my_jobstarter)

The poses are now properly set up and can be used in a design protocol.