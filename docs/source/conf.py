# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path to your module's location
sys.path.insert(0, os.path.abspath('../../protflow/'))  # Adjust the path to your module's location

# -- Project information -----------------------------------------------------
project = 'ProtFlow'
copyright = '2024, Markus Braun, Adrian Tripp'
author = 'Markus Braun'

# The full version, including alpha/beta/rc tags
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme'
]
nitpicky = False
#
# autosummary_generate = True

# prefer type hints over parsing types from docstrings
autodoc_typehints = "description"

autodoc_mock_imports = ['openbabel', 'protflow.config', 'config']
suppress_warnings = ["autodoc.mocked_object", "ref.python"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Let Sphinx resolve standard library / common libs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Add others you reference in types/links as needed:
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Extension configuration -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Convert simple/alias names (e.g., "optional") into proper typing names
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Map raw "optional" tokens to typing.Optional so intersphinx can resolve it
    "optional": "typing.Optional",
}
