import os
import sys
from datetime import datetime

project = "Compitum"
author = "Compitum authors"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "sphinx.ext.intersphinx",
]

autosectionlabel_prefix_document = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "legacy/**"]

html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["_extra"]

# SEO / sitemap
html_baseurl = "https://paultiffany.github.io/compitum/"

# Open Graph
ogp_site_url = html_baseurl
ogp_site_name = project
ogp_image = html_baseurl + "assets/compitum-social-card.svg"

# Branding
html_logo = "_static/compitum-mark.svg"
html_favicon = "_static/compitum-mark.svg"

# Allow autodoc to find sources
sys.path.insert(0, os.path.abspath("../src"))

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
]

# Professional-grade docs: fail on unresolved cross-references
nitpicky = True

# Common intersphinx inventories to resolve external references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Ignore a small set of well-known third-party types that may not resolve
# uniformly across versions/inventories during CI builds.
nitpick_ignore = [
    ("py:class", "ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "np.ndarray"),
    ("py:class", "scipy.sparse.spmatrix"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "sklearn.neighbors.KernelDensity"),
]
\n# Suppress sitemap warning in CI (treated as error by -W)\nsuppress_warnings = ['sphinx_sitemap']\n