"""strdcheck: numerical accuracy of the Python data stack on the NIST StRD.

Top level: the dataset manifest and the LRE functions. The rest lives in submodules:
``strdcheck.testing`` and the pytest plugin, ``strdcheck.fetch``, ``strdcheck.parse``,
``strdcheck.exact``, the study under ``strdcheck.study`` (design in ``docs/DESIGN.md``).
"""

from __future__ import annotations

from strdcheck.lre import LRE, lre, lre_from_string, significant_digits
from strdcheck.manifest import DatasetInfo, get_dataset_info, list_datasets

__version__ = "0.1.0"

__all__ = [
    "LRE",
    "DatasetInfo",
    "__version__",
    "get_dataset_info",
    "list_datasets",
    "lre",
    "lre_from_string",
    "significant_digits",
]
