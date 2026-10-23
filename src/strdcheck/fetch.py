"""Download the StRD files from NIST into a local cache and verify their checksums.

Only ``https://www.itl.nist.gov/div898/strd/`` is contacted. Every file is checked against
the SHA-256 in the manifest before it is moved into the cache (atomic rename), so a
partial or altered download is never used. The cache location is, in order:

1. the ``data_dir`` argument (``--strd-data-dir`` for the pytest plugin, ``--data-dir``
   for the CLI),
2. the ``STRDCHECK_DATA_DIR`` environment variable,
3. ``%LOCALAPPDATA%/strdcheck/strd`` on Windows, ``$XDG_CACHE_HOME/strdcheck/strd`` (or
   ``~/.cache/strdcheck/strd``) elsewhere.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from strdcheck.manifest import DatasetInfo, all_datasets, manifest_meta

ENV_VAR = "STRDCHECK_DATA_DIR"
ALLOWED_PREFIX = "https://www.itl.nist.gov/div898/strd/"
TIMEOUT_S = 60.0
ATTEMPTS = 3
USER_AGENT = "strdcheck (+https://github.com/h-a-forster/strdcheck)"

Opener = Callable[[str, float], bytes]


class ChecksumError(RuntimeError):
    """A downloaded or cached file does not match the manifest checksum."""


def default_data_dir() -> Path:
    """Cache directory used when neither an argument nor the environment variable is set."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "strdcheck" / "strd"


def resolve_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """The cache directory to use (see module docstring for the order)."""
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get(ENV_VAR)
    return Path(env) if env else default_data_dir()


def _urlopen(url: str, timeout: float) -> bytes:
    if not url.startswith(ALLOWED_PREFIX):
        raise ValueError(f"refusing to download from outside {ALLOWED_PREFIX}: {url}")
    # NIST's server answers 403 to the default "Python-urllib" user agent.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_cached(info: DatasetInfo, data_dir: Path) -> bool:
    """True if the file is in the cache and matches its checksum."""
    path = data_dir / info.filename
    return path.is_file() and _sha256(path.read_bytes()) == info.sha256


def fetch_one(
    info: DatasetInfo,
    data_dir: Path,
    *,
    opener: Opener = _urlopen,
    attempts: int = ATTEMPTS,
    timeout: float = TIMEOUT_S,
    backoff_s: float = 2.0,
) -> Path:
    """Ensure one dataset is cached and verified; download it if needed."""
    path = data_dir / info.filename
    if is_cached(info, data_dir):
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for i in range(attempts):
        try:
            raw = opener(info.url, timeout)
        except Exception as exc:  # network errors are retried
            last = exc
            if i + 1 < attempts:
                time.sleep(backoff_s * (i + 1))
            continue
        if _sha256(raw) != info.sha256:
            raise ChecksumError(
                f"{info.filename}: SHA-256 {_sha256(raw)} does not match manifest {info.sha256}"
            )
        fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=f".{info.name}.", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path
    raise RuntimeError(f"could not download {info.url} after {attempts} attempts") from last


def fetch(
    names: Iterable[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    *,
    opener: Opener = _urlopen,
) -> list[Path]:
    """Download (if needed) and verify the named datasets, or all of them."""
    target = resolve_data_dir(data_dir)
    wanted = {n.lower() for n in names} if names is not None else None
    infos = [d for d in all_datasets() if wanted is None or d.name.lower() in wanted]
    if wanted is not None and len(infos) != len(wanted):
        known = {d.name.lower() for d in infos}
        raise KeyError(f"unknown StRD dataset(s): {sorted(wanted - known)}")
    return [fetch_one(d, target, opener=opener) for d in infos]


def source_note() -> str:
    """Attribution line for reports."""
    meta = manifest_meta()
    return f"Data: {meta['source']} (doi:{meta['doi']}), retrieved {meta['retrieved']}."
