# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = 'AIMotionLab-Virtual'
copyright = '2024, Botond Gaal'
author = 'Botond Gaal'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.todo',      # Support for TODOs
    'sphinx.ext.autosummary',
]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'private-members': False,
    'no-value': True
}
autodoc_member_order = 'groupwise'
todo_include_todos = True
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
pygments_style = "sphinx"
pygments_dark_style = "monokai"
highlight_language = "python"  # Adjust according to your primary language
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',  # Add your custom CSS file here
]