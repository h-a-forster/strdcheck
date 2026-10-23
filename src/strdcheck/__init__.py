"""strdcheck: numerical accuracy of the Python data stack on the NIST StRD.

The public API grows milestone by milestone (``docs/DESIGN.md`` section 8). Available now:
the dataset manifest, the checksum-verified downloader and the LRE functions.
"""

from __future__ import annotations

from strdcheck.lre import LRE, lre, lre_from_string, significant_digits
from strdcheck.manifest import DatasetInfo, get_dataset_info, list_datasets

__version__ = "0.1.0.dev0"

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
