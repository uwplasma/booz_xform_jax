from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from booz_xform_jax import __version__


project = "booz_xform_jax"
author = "UWPlasma contributors"
copyright = "2026, UWPlasma contributors"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
rst_prolog = """
.. |B| replace:: B
.. |m| replace:: m
.. |n| replace:: n
"""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_title = "booz_xform_jax"
html_static_path: list[str] = []
html_context = {
    "display_github": True,
    "github_user": "uwplasma",
    "github_repo": "booz_xform_jax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
