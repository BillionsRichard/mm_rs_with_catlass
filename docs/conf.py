import subprocess

branch = subprocess.check_output(["/bin/bash", "-c", "git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || git rev-parse HEAD"]).strip().decode()
project = "Shmem Guidebook"
author = "xxx"
copyright = "2024"
release = "1.0.0"
html_show_sphinx = False

extensions = [
    'myst_parser',
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme= 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

breathe_projects = {"SHMEM_CPP_API": f"./{branch}/xml"}
breathe_default_project = "SHMEM_CPP_API"