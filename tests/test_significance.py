"""Exact permutation test for the rank correlations.

Six yearly observations rule out any asymptotic test, so the null distribution
is enumerated in full. These tests pin the enumeration against cases whose exact
p-value is known by combinatorics, not by a reference implementation.
"""

import math
from itertools import permutations

import pandas as pd
import pytest

import analysis as an

LABEL = f"+/-{100 * an.CONFIG['reference_width']:.0f}%"


def table(driver: list[float], corrected: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"driver": driver, f"{LABEL} corr": corrected})


def test_perfect_correlation_has_the_smallest_possible_p():
    """Only the identity and its reverse reach |rho| = 1, so p = 2/n!."""
    n = 6
    t = table([float(i) for i in range(n)], [10.0 * i for i in range(n)])
    assert an.permutation_pvalue(t, "driver") == pytest.approx(2 / math.factorial(n))


def test_p_value_is_symmetric_under_reversal():
    """A two-sided test cannot distinguish rho from -rho."""
    ascending = table([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    descending = table([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert an.permutation_pvalue(ascending, "driver") == pytest.approx(
        an.permutation_pvalue(descending, "driver"))


def test_p_value_is_a_probability():
    t = table([3.0, 1.0, 4.0, 1.5, 5.0, 9.0], [2.0, 7.0, 1.0, 8.0, 2.5, 6.0])
    p = an.permutation_pvalue(t, "driver")
    assert 0.0 < p <= 1.0
    assert math.isclose(p * math.factorial(6), round(p * math.factorial(6)))  # a count over 720


def test_the_significance_threshold_at_six_observations():
    """With n = 6 the null distribution is discrete: 0.886 clears 5%, 0.829 does not.

    Pinned by enumerating the levels rather than quoting a table, because this is
    the fact the README states about its own headline correlation.
    """
    n, corrected = 6, [float(i) for i in range(6)]
    by_level = {}
    for order in permutations(range(n)):
        x = pd.Series(order).rank()
        y = pd.Series(corrected).rank()
        by_level.setdefault(round(abs(x.corr(y)), 3), []).append(order)
    for level, expected in ((0.886, True), (0.829, False)):
        order = by_level[level][0]
        p_value = an.permutation_pvalue(table([float(i) for i in order], corrected), "driver")
        assert (p_value < 0.05) is expected, (level, p_value)


def test_enumeration_refuses_samples_it_cannot_handle():
    """n! grows fast enough that the guard matters more than the speed."""
    big = table([float(i) for i in range(9)], [float(i) for i in range(9)])
    with pytest.raises(ValueError):
        an.permutation_pvalue(big, "driver")
