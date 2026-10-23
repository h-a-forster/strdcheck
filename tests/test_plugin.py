from __future__ import annotations

import pytest


def test_strd_assert_lre_passes_and_fails(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        def test_ok(strd_assert_lre):
            worst = strd_assert_lre([1.0000001, 2.0], [1.0, 2.0], min_lre=6)
            assert 6.9 < worst < 7.1

        def test_low(strd_assert_lre):
            with pytest.raises(AssertionError, match="LRE below 6"):
                strd_assert_lre([1.001], [1.0], min_lre=6, label="mean")

        def test_nan(strd_assert_lre):
            with pytest.raises(AssertionError):
                strd_assert_lre([float("nan")], [1.0], min_lre=1)

        @pytest.mark.strd("Norris")
        def test_marker(strd_data_dir):
            assert strd_data_dir.name
        """
    )
    result = pytester.runpytest("--strict-markers")
    result.assert_outcomes(passed=4)


def test_min_lre_override(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_override(strd_assert_lre):
            strd_assert_lre([1.001], [1.0], min_lre=6)
        """
    )
    result = pytester.runpytest("--strd-min-lre", "2")
    result.assert_outcomes(passed=1)
