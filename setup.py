'''
Setup of ProtSLURM package.
Dependencies are:
    - pandas
    - numpy
    - Biopython
'''

from setuptools import setup, find_packages

setup(
    name='protflow',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'Bio',
        'matplotlib',
        'pyyaml'
    ],
    # Other metadata
)
