from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from strdcheck import fetch as fetch_mod
from strdcheck.fetch import ChecksumError, fetch_one, is_cached, resolve_data_dir
from strdcheck.manifest import DatasetInfo, get_dataset_info

PAYLOAD = b"NIST/ITL StRD\r\nfake content for tests\r\n"


def _info() -> DatasetInfo:
    return replace(get_dataset_info("Norris"), sha256=hashlib.sha256(PAYLOAD).hexdigest())


def test_fetch_one_downloads_verifies_and_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def opener(url: str, timeout: float) -> bytes:
        calls.append(url)
        return PAYLOAD

    info = _info()
    path = fetch_one(info, tmp_path, opener=opener)
    assert path.read_bytes() == PAYLOAD
    assert is_cached(info, tmp_path)
    fetch_one(info, tmp_path, opener=opener)  # cached: no second download
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.part"))


def test_fetch_one_rejects_bad_checksum(tmp_path: Path) -> None:
    with pytest.raises(ChecksumError):
        fetch_one(_info(), tmp_path, opener=lambda url, timeout: b"tampered")
    assert not (tmp_path / "Norris.dat").exists()


def test_fetch_one_retries_then_fails(tmp_path: Path) -> None:
    n = {"calls": 0}

    def flaky(url: str, timeout: float) -> bytes:
        n["calls"] += 1
        if n["calls"] < 3:
            raise OSError("connection reset")
        return PAYLOAD

    assert fetch_one(_info(), tmp_path, opener=flaky, backoff_s=0.0).exists()

    def dead(url: str, timeout: float) -> bytes:
        raise OSError("down")

    with pytest.raises(RuntimeError):
        fetch_one(_info(), tmp_path / "other", opener=dead, attempts=2, backoff_s=0.0)


def test_refuses_foreign_host() -> None:
    with pytest.raises(ValueError):
        fetch_mod._urlopen("https://example.com/Norris.dat", 1.0)


def test_resolve_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRDCHECK_DATA_DIR", str(tmp_path / "env"))
    assert resolve_data_dir() == tmp_path / "env"
    assert resolve_data_dir(tmp_path / "arg") == tmp_path / "arg"
    monkeypatch.delenv("STRDCHECK_DATA_DIR")
    assert resolve_data_dir().name == "strd"


def test_fetch_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        fetch_mod.fetch(["NotADataset"], tmp_path)


def test_urlopen_sends_user_agent(monkeypatch) -> None:
    """NIST answers 403 to the default Python-urllib user agent."""
    import io
    import urllib.request

    from strdcheck import fetch as fetch_mod

    seen = {}

    def fake(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        seen["timeout"] = timeout
        return io.BytesIO(b"data")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert fetch_mod._urlopen(fetch_mod.ALLOWED_PREFIX + "x.dat", 5.0) == b"data"
    assert seen == {"ua": fetch_mod.USER_AGENT, "timeout": 5.0}


def test_build_manifest_download_uses_strdcheck_opener(monkeypatch) -> None:
    """scripts/build_manifest.py must send the User-Agent too (NIST 403s Python-urllib)."""
    import importlib.util

    from strdcheck import fetch as fetch_mod

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []
    monkeypatch.setattr(fetch_mod, "_urlopen", lambda url, timeout: seen.append(url) or b"ok")
    assert mod._download(fetch_mod.ALLOWED_PREFIX + "x.dat") == b"ok"
    assert seen == [fetch_mod.ALLOWED_PREFIX + "x.dat"]
