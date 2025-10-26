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
