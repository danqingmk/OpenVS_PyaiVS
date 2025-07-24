import os
import sys

# Add the project's script/ directory to the Python path so that autodoc can locate the modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'script')))

# Basic project information
project = "PyaiVS"

# Sphinx extension configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    'myst_parser',
]
autosummary_generate = True  # Automatically generate autosummary listings
html_theme = "sphinx_rtd_theme"  # Use the Read the Docs theme
master_doc = "index"  # Name of the master document file (e.g., index.rst)

