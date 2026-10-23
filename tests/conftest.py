"""Shared pytest configuration: the ``slow`` marker (excluded unless ``--runslow``)."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", help="also run tests marked slow")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test (run with --runslow)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def strd_dir():
    """Directory with all 58 checksum-verified StRD files; skips when the cache is incomplete."""
    from strdcheck.fetch import is_cached, resolve_data_dir
    from strdcheck.manifest import all_datasets

    d = resolve_data_dir()
    missing = [i.name for i in all_datasets() if not is_cached(i, d)]
    if missing:
        pytest.skip(f"StRD files not cached ({len(missing)} missing); run `strdcheck fetch`")
    return d
