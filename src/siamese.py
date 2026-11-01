"""Proposed method (preliminary): Siamese MLP + nearest-centroid rule.

An embedding f_theta: R^d -> R^k is trained with the contrastive loss of
Hadsell, Chopra & LeCun (2006) on pairs (x_a, x_b):

    L = s * ||f(x_a) - f(x_b)||^2 + (1 - s) * max(0, m - ||f(x_a) - f(x_b)||)^2

where s = 1 if the pair shares a firearm class. Positive pairs are drawn
*across different devices* where possible, so device variation is exactly
the within-class variation the loss is asked to collapse. Classification
assigns x to the class whose training-embedding centroid is nearest.
"""

from __future__ import annotations

import time

import numpy as np


def fit_siamese(X, y, dev, seed, k=16, hidden=128, margin=1.0, epochs=40,
                pairs_per_epoch=4096, lr=1e-3):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, k))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}

    t0 = time.perf_counter()
    net.train()
    for _ in range(epochs):
        a = rng.integers(0, len(y), pairs_per_epoch)
        same = rng.random(pairs_per_epoch) < 0.5
        b = np.empty_like(a)
        for j, (i, s) in enumerate(zip(a, same)):
            if s:
                pool = by_class[y[i]]
                cross = pool[dev[pool] != dev[i]]
                b[j] = rng.choice(cross if cross.size else pool)
            else:
                b[j] = rng.integers(0, len(y))
        s_all = torch.from_numpy((y[a] == y[b]).astype(np.float32))
        for i in range(0, pairs_per_epoch, 256):
            sl = slice(i, i + 256)
            s = s_all[sl]
            dist = (net(Xt[a[sl]]) - net(Xt[b[sl]])).norm(dim=1)
            loss = (s * dist ** 2 + (1 - s) *
                    torch.clamp(margin - dist, min=0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        Z = net(Xt).numpy()
    classes = np.unique(y)
    centroids = np.stack([Z[y == c].mean(0) for c in classes])

    def predict(Xnew):
        with torch.no_grad():
            z = net(torch.from_numpy(((Xnew - mu) / sd).astype(np.float32)))
        d = ((z.numpy()[:, None, :] - centroids[None]) ** 2).sum(-1)
        return classes[d.argmin(1)]

    return predict, time.perf_counter() - t0
