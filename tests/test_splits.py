"""The property the whole project rests on: group-disjoint splits really are
disjoint, in both the outer split and the inner tuning CV."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splits import inner_cv, outer_split  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    return {"y": rng.integers(0, 4, n),
            "device": np.array([f"dev{i}" for i in rng.integers(0, 12, n)]),
            "uuid": np.array([f"cap{i}" for i in rng.integers(0, 150, n)])}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("proto,key", [("device", "device"),
                                       ("capture", "uuid")])
def test_outer_split_is_group_disjoint(data, seed, proto, key):
    tr, te = outer_split(proto, data, seed)
    assert not set(data[key][tr]) & set(data[key][te])
    assert len(set(tr) | set(te)) == data["y"].size
    assert 0.1 < te.size / data["y"].size < 0.35


@pytest.mark.parametrize("proto,key", [("device", "device"),
                                       ("capture", "uuid")])
def test_inner_cv_is_group_disjoint(data, proto, key):
    tr, _ = outer_split(proto, data, 0)
    for fit, val in inner_cv(proto, data, tr, 0):
        assert not set(data[key][tr][fit]) & set(data[key][tr][val])


def test_random_split_reproducible(data):
    a = outer_split("random", data, 3)
    b = outer_split("random", data, 3)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
