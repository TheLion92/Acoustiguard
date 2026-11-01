"""Outer evaluation splits.

Three protocols, each an 80/20 hold-out redrawn for every seed:

* ``random``  -- stratified file-level split (what the literature reports).
                 Near-duplicate files of one discharge can straddle it.
* ``capture`` -- capture-disjoint: every file sharing a uuid (array channels
                 and their mean) stays on one side.
* ``device``  -- device-disjoint: every recorder is entirely train or test.

``inner_cv`` returns the matching grouped CV used for hyper-parameter search
on the training part only.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)

PROTOCOLS = ("random", "capture", "device")


def groups_for(protocol: str, data: dict) -> np.ndarray | None:
    return {"random": None, "capture": data["uuid"],
            "device": data["device"]}[protocol]


def outer_split(protocol: str, data: dict, seed: int):
    """Indices (train, test) for one seed; ~20% of files held out."""
    y, g = data["y"], groups_for(protocol, data)
    idx = np.arange(y.size)
    if g is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        return next(cv.split(idx, y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return next(cv.split(idx, y, g))


def inner_cv(protocol: str, data: dict, train: np.ndarray, seed: int):
    """List of (fit, val) index pairs *within* ``train`` for tuning."""
    y = data["y"][train]
    g = groups_for(protocol, data)
    if g is None:
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        return list(cv.split(train, y))
    return list(GroupKFold(3).split(train, y, g[train]))
