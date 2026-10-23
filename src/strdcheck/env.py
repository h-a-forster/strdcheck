"""Environment record (DESIGN 5.6), with local user paths scrubbed."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "duckdb",
    "statsmodels",
    "torch",
    "mpmath",
    "threadpoolctl",
    "strdcheck",
)


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor()


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def scrub(obj: Any) -> Any:
    """Replace absolute filesystem paths by their last component (and the home directory)."""
    home = str(Path.home())
    homes = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
    pat = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|usr|opt|tmp|var|private)/)[^\s'\",;]*")

    def fix(s: str) -> str:
        for h in homes:
            s = s.replace(h, "<home>")

        def repl(m: re.Match[str]) -> str:
            tail = re.split(r"[\\/]+", m.group(0).rstrip("\\/"))
            return f"<path>/{tail[-1]}" if tail and tail[-1] else "<path>"

        return pat.sub(repl, s)

    if isinstance(obj, str):
        return fix(obj)
    if isinstance(obj, dict):
        return {fix(str(k)): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def environment_record() -> dict[str, Any]:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    rec: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "packages": versions,
        "git_commit": _git_commit(),
    }
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        rec["numpy_build"] = {
            "blas": cfg.get("Build Dependencies", {}).get("blas", {}),
            "lapack": cfg.get("Build Dependencies", {}).get("lapack", {}),
            "simd": cfg.get("SIMD Extensions", {}),
        }
    except Exception as exc:  # pragma: no cover - depends on numpy build
        rec["numpy_build"] = f"unavailable: {exc}"
    try:
        from threadpoolctl import threadpool_info

        rec["threadpools"] = [
            {
                k: v
                for k, v in d.items()
                if k
                in (
                    "user_api",
                    "internal_api",
                    "version",
                    "num_threads",
                    "threading_layer",
                    "architecture",
                    "prefix",
                )
            }
            for d in threadpool_info()
        ]
    except ImportError:
        rec["threadpools"] = None
    if versions.get("torch"):
        try:
            import torch

            rec["torch"] = {
                "num_threads": torch.get_num_threads(),
                "mkl": torch.backends.mkl.is_available(),
                "openmp": torch.backends.openmp.is_available(),
            }
        except Exception as exc:  # pragma: no cover
            rec["torch"] = f"unavailable: {exc}"
    return scrub(rec)
