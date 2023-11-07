'''
Setup of ProtSLURM package.
Dependencies are:
    - pandas
    - numpy
'''

from setuptools import setup, find_packages

setup(
    name='protslurm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    # Other metadata
)
