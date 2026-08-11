"""Sphinx configuration for Tenax documentation."""

import logging
import warnings

from sphinx.ext import intersphinx as _intersphinx

project = "Tenax"
copyright = "2025, Tenax Contributors"
author = "Tenax Contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
]

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

# MyST (Markdown support)
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Bound each inventory fetch. The default is no timeout at all, so a server that
# accepts the connection and then stalls hangs the build until the CI job is
# killed rather than failing fast.
intersphinx_timeout = 10


# Both CI (`sphinx-build -W`) and Read the Docs (`fail_on_warning: true`) treat
# every warning as fatal. That is what we want for documentation defects and
# emphatically not what we want for a blip on docs.python.org: intersphinx
# fetches its inventories over the network on every cold build, and one
# unreachable host reds the docs build of whatever unrelated PR is in flight.
#
# `suppress_warnings` cannot express this -- it matches on a warning's `type`,
# and the unreachable-inventory message is the single warning intersphinx emits
# without one (`_load.py`; the resolve-time warnings all carry `type` of
# "intersphinx" or "ref"). That absence is therefore an exact selector, needing
# no message-text matching, and it leaves the resolve-time warnings -- which are
# genuine documentation defects -- fatal.
#
# The hook has to be the logger rather than a handler: a logger's filters run in
# `Logger.handle` before any handler is consulted, and all of Sphinx's warning
# bookkeeping, including the counter that sets the build's exit status, lives in
# handler filters.
#
# Demoted to INFO rather than dropped, so a sustained outage stays legible in the
# build log; it just no longer fails the build. Nothing else is lost: with
# `nitpicky` off (the default) a cross-reference into a missing inventory
# degrades to unlinked text without warning.
#
# To check this still holds, build against a dead proxy and require a pass:
#
#   https_proxy=http://127.0.0.1:1 http_proxy=http://127.0.0.1:1 \
#       uv run sphinx-build -W -b html docs docs/_build/offline
#
# A separate output directory means separate doctrees, so the intersphinx cache
# is cold and the fetch is genuinely attempted.
class _DemoteInventoryFetchFailure(logging.Filter):
    """Log intersphinx's unreachable-inventory warning at INFO instead."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING and not getattr(record, "type", ""):
            record.levelno = logging.INFO
            record.levelname = "INFO"
            record.msg = f"[non-fatal] {record.msg}"
            # `SphinxLoggerAdapter.warning` stamps every record `nonl=True`. The
            # warning stream ignores it, but the info stream honours it and would
            # run consecutive messages together on one line.
            record.nonl = False
        return True


# Sphinx namespaces its loggers under "sphinx.", so an extension logs to
# "sphinx." + its own module path -- hence the doubled prefix.
#
# This is the *emitting* logger, not an ancestor of it, which matters because a
# logger's filters do not apply to records propagated up from descendants. The
# whole intersphinx package shares one logger, defined in `_shared.py` and
# imported by `_load.py`; there is no `..._load` child logger. Checked against
# every Sphinx in the declared `>=7.0` range -- 7.0.0, 7.4.7, 8.0.2, 8.2.3,
# 9.0.4 and 9.1.0 all emit on this exact object.
logging.getLogger(f"sphinx.{_intersphinx.__name__}").addFilter(
    _DemoteInventoryFetchFailure()
)

# Theme
html_theme = "furo"
html_title = "Tenax"

# Source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = [
    "_build",
    "plans",
    "superpowers",
    "superpowers/**",
    # Committed benchmark-result artifacts (auto-generated tables/READMEs), not
    # narrative docs -- the toctree points at guide/benchmarks.md, never here.
    "benchmarks",
    "benchmarks/**",
]

# Suppress third-party deprecation warnings during build
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sphinx_autodoc_typehints"
)
