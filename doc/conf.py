# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
import datetime
import os

import sphinx_autosummary_accessors

import eddy_footprint

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_autosummary_accessors",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "sphinx_design",
]

extlinks = {
    "issue": ("https://github.com/arctic-carbon/eddy-footprint/issues/%s", "GH#"),
    "pull": ("https://github.com/arctic-carbon/eddy-footprint/pull/%s", "GH#"),
}

# sphinx-copybutton configurations (from https://github.com/pydata/xarray/blob/main/doc/conf.py)
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

autosummary_generate = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "eddy-footprint"
copyright = "2023, Eddy-footprint developers"
author = "Eddy-footprint developers"

copyright = f"2023-{datetime.datetime.now().year}, eddy-footprint developers"
# The short X.Y version.
version = eddy_footprint.__version__
# The full version, including alpha/beta/rc tags.
release = eddy_footprint.__version__
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# The following is from the pydata-sphinx-theme settings
# (https://github.com/pydata/pydata-sphinx-theme/blob/main/docs/conf.py)
# Define the json_url for our version switcher.
json_url = "https://eddy-footprint.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    release = eddy_footprint.__version__
    if "dev" in release or "post" in release or "rc" in release:
        version_match = "latest"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

print(f"release: {release}")

html_theme_options = dict(
    # analytics_id=''  this is configured in rtfd.io
    # canonical_url="",
    repository_url="https://github.com/arctic-carbon/eddy-footprint",
    repository_branch="main",
    navigation_with_keys=False,  # pydata/pydata-sphinx-theme#1492
    path_to_docs="doc",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
    icon_links=[],  # workaround for pydata/pydata-sphinx-theme#1220
    switcher={
        "json_url": json_url,
        "version_match": version_match,
    },
)
