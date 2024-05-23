# Installation of ESMFold
To install ESMFold, simply following installation instructions on the facebookresearch/esm github page will not lead to a functioning installation of ESMFold. Currently (May 5th 2024), this is the easiest way to install ESMFold:

Create a new empty conda environment with python and pytorch requirements in the versions needed for ESMFold. Importantly, do not install the esmfold environment as specified in the environment.yaml file. This will also lead to a non-functioning installation.

```
conda create --name esmfold python=3.9 pytorch=1.12
```

Then install dependencies with pip as described in the repository:

```
conda activate esmfold
pip install "fair-esm[esmfold]" --no-cache-dir
```

Now ensure that the pip install in your esmfold environment did not overwrite your pytorch installation to the latest version.
The pytorch version you should have installed should be 1.12. Here is how to check:

```
conda activate esmfold
python
>>> import torch
>>> print(torch.__version__)
```

This should output version 1.12
If the version differs (a later version) then you will have to manually uninstall pytorch from the esmfold environment and reinstall with the correct version specified.

```
conda activate esmfold
conda remove pytorch
conda clean --all
conda install pytorch=1.12 cudatoolkit=11.3 -c pytorch
```

Now verify the pytorch version again as described above. This should yield pytorch version 1.12.
Then continue with the remaining dependencies:

```
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git' --no-cache-dir
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307' --no-cache-dir
```
