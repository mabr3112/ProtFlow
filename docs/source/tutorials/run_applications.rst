.. _run_applications:

Run applications
=========================

In this tutorial, we are going to learn how to use protein design tools (in this case, LigandMPNN) via ProtFlow. To run different protein design applications, we first have to 
import everything we need. This tutorial can also be completed interactively, by working in the Jupyter notebook found at 
`ProtFlow/examples/runners.ipynb <https://github.com/mabr3112/ProtFlow/blob/master/examples/runners.ipynb>`_. It can be opened with programs such as Visual Studio Code.
Input PDBs can be found in the same directory. Before we start, make sure the ProtFlow config file is set up properly and the ProtFlow environment is active. 

.. note::

   If you are having trouble setting up ProtFlow, please check out the
   :doc:`Quickstart guide </quickstart/index>`.

.. code-block:: python

   from protflow.poses import Poses
   from protflow.tools.ligandmpnn import LigandMPNN
   from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter

First, we define our jobstarters. We want to test if we can run LigandMPNN on GPU or cpu using the SLURM workload manager as well as run it locally. If SLURM is not installed on your machine, 
you can skip the parts mentioning it. When defining a jobstarter, we can select on how many CPUs (or GPUs) jobs should run.

.. code-block:: python

   slurm_gpu_jobstarter = SbatchArrayJobstarter(max_cores=10, gpus=1)
   slurm_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10, gpus=False)
   local_jobstarter = LocalJobStarter(max_cores=1)

Next, we have to load our poses. We set the local_jobstarter as default jobstarter.

.. code-block:: python

   my_poses = Poses(poses='data/input_pdbs/', glob_suffix='*pdb', work_dir='runners_example', storage_format='csv', jobstarter=local_jobstarter)

To run ligandmpnn, we have to define a runner. Make sure the path to the LigandMPNN script and python path are set in protflow/config.py! 
You can set it also when creating the runner, but it is recommended to set it in the config if you want to run it again.

.. code-block:: python

   ligandmpnn = LigandMPNN()

To run ligandmpnn on our poses, we have to use the .run() function. All tools and metrics should have this function. It is mandatory to provide a unique prefix for each run. 
Each score generated will be saved to the poses dataframe in the format prefix_scorename. The output files can be found in a folder called prefix in the working_directory set for 
the input poses. The .run() function always returns poses.

.. code-block:: python
   
   ligandmpnn.run(poses=my_poses, prefix='ligmpnn_local', nseq=2, model_type='protein_mpnn')
   print(my_poses.df)

Notice how the poses dataframe has changed! It now contains all the poses generated from LigandMPNN and the corresponding scores. Since we did not provide a jobstarter when we set up 
ligandmpnn, it ran on the local machine, because it defaulted to the jobstarter we set when creating our poses. You can run the tutorial with different jobstarters, if you want.
In your working directory, there should now exist a folder called "ligmpnn_local". This is where all the output from the LigandMPNN run is stored. Typically, there is a json file containing all
scores in this directory. These are the same scores that are added to the poses DataFrame (with the set prefix for each scorename). 

To run multiple design tools in succession, just run the next tool on the same poses instance:

.. code-block:: python

   from protflow.tools.esmfold import ESMFold
   esmfold = ESMFold(jobstarter=slurm_gpu_jobstarter)
   esmfold.run(poses=my_poses, prefix='esm_pred')
   print(my_poses.df)

We now predicted the structures corresponding to the sequences we generated with LigandMPNN using ESMFold. ESMFold requires a GPU, that is why we selected the slurm_GPU_jobstarter when instantiating the runner.
If you have a GPU that can run ESMFold on your local machine, you can also create a local GPU-jobstarter and run it in this way. 