# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


sys.path.insert(0, os.path.abspath("../../"))
sys.setrecursionlimit(1500)
MOCK_MODULES = []
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
project = "plangym"
copyright = "2018, FragileTech"
author = "Guillem Duran Ballester"

# The short X.Y version
from plangym.version import __version__


version = __version__
# The full version, including alpha/beta/rc tags
release = __version__
# -- General configuration ---------------------------------------------------
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
# The master toctree document.
master_doc = "index"
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc.typehints",
    "sphinx_book_theme",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]
suppress_warnings = ["image.nonlocal_uri"]
autodoc_typehints = "description"
# Autoapi settings
autoapi_type = "python"
autoapi_dirs = ["../../src/plangym"]
autoapi_add_toctree_entry = True
# Make use of custom templates
autoapi_template_dir = "_autoapi_templates"
exclude_patterns.append("_autoapi_templates/index.rst")

# Ignore sphinx-autoapi warnings on multiple target description
suppress_warnings.append("ref.python")

# Napoleon settings
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = ""
html_theme = "sphinx_book_theme"
# html_logo = "_static/logo-wide.svg"
# html_favicon = "_static/logo-square.svg"
html_theme_options = {
    "github_url": "https://github.com/poolsideai/fragile",
    "repository_url": "https://github.com/poolsideai/fragile",
    "repository_branch": "gh-pages",
    "home_page_in_toc": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# myst_parser options
nb_execution_mode = "off"
myst_heading_anchors = 2
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
]


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
