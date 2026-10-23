from __future__ import annotations

import strdcheck


def test_public_api() -> None:
    assert strdcheck.__version__
    for name in strdcheck.__all__:
        assert hasattr(strdcheck, name)
