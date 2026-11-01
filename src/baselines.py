"""Baselines.

B0  majority class                         -- the floor
B1a Gaussian plug-in Bayes (own numpy)     -- classical, from the course
B1b RBF-SVM                                -- classical, strongest discriminant
B1c k-NN                                   -- classical, non-parametric
B2  ImageNet-pretrained ResNet-18 fine-tuned on log-mel images -- the
    published protocol of Doungpaisan & Khunarsa (2025) with a lighter
    backbone (they use InceptionResNetV2 on CQT)

Every classical model is tuned by grid search over the *training part only*
with a grouped inner CV (see splits.inner_cv); the grids below are fixed in
advance and are the whole tuning budget.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class GaussianBayes(BaseEstimator, ClassifierMixin):
    """Plug-in Bayes rule with Gaussian class-conditionals.

    g_i(x) = -1/2 (x-mu_i)^T S_i^-1 (x-mu_i) - 1/2 log|S_i| + log P(w_i),
    S_i = Sigma_i_hat + eps * I  (ridge regularisation of the covariance).
    Decision: argmax_i g_i(x).
    """

    def __init__(self, eps: float = 1e-2):
        self.eps = eps

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        d = X.shape[1]
        self.mu_, self.prec_, self.logdet_, self.logprior_ = [], [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            S = np.cov(Xc, rowvar=False) + self.eps * np.eye(d)
            sign, logdet = np.linalg.slogdet(S)
            self.mu_.append(Xc.mean(0))
            self.prec_.append(np.linalg.inv(S))
            self.logdet_.append(logdet)
            self.logprior_.append(np.log(len(Xc) / len(X)))
        return self

    def decision_function(self, X):
        g = []
        for mu, P, ld, lp in zip(self.mu_, self.prec_, self.logdet_,
                                 self.logprior_):
            D = X - mu
            g.append(-0.5 * np.einsum("ij,jk,ik->i", D, P, D) - 0.5 * ld + lp)
        return np.stack(g, 1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), 1)]


def _pipe(model):
    return Pipeline([("scale", StandardScaler()), ("pca", PCA()),
                     ("clf", model)])


# name -> (estimator, grid). PCA dims: 54 is "no reduction".
CLASSICAL = {
    "B0 Majority": (DummyClassifier(strategy="most_frequent"), None),
    "B1a Gaussian Bayes": (_pipe(GaussianBayes()), {
        "pca__n_components": [10, 20, 30],
        "clf__eps": [1e-3, 1e-2, 1e-1, 1.0]}),
    "B1b RBF-SVM": (_pipe(SVC(kernel="rbf")), {
        "pca__n_components": [20, 54],
        "clf__C": [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.1]}),
    "B1c k-NN": (_pipe(KNeighborsClassifier()), {
        "pca__n_components": [20, 54],
        "clf__n_neighbors": [1, 3, 5, 9, 15]}),
}


def fit_classical(name, X, y, cv, seed):
    """Grid-search on the training data only; returns (model, params, secs)."""
    est, grid = CLASSICAL[name]
    t0 = time.perf_counter()
    if grid is None:
        model = est.fit(X, y)
        params = {}
    else:
        gs = GridSearchCV(est, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X, y)
        model, params = gs.best_estimator_, gs.best_params_
    return model, params, time.perf_counter() - t0


# --------------------------------------------------------------------------
# B2 -- pretrained CNN
def fit_cnn(img, y, train, val, seed, epochs=12, lr=3e-4, batch=64):
    """Fine-tune ResNet-18 (ImageNet weights) on log-mel images.

    ``val`` is a group-disjoint slice of the training part used only to pick
    the best epoch (early stopping); the test set is never touched here.
    Returns a predict(indices) function and the wall-clock seconds.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 4)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    images = torch.from_numpy(img.astype(np.float32))
    labels = torch.from_numpy(y.astype(np.int64))

    def prep(idx):
        x = images[idx].unsqueeze(1).to(dev)
        x = F.interpolate(x, size=(128, 128), mode="bilinear",
                          align_corners=False)
        return x.repeat(1, 3, 1, 1)

    def logits(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                out.append(net(prep(idx[i:i + 256])).float().cpu())
        return torch.cat(out)

    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        order = rng.permutation(train)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            loss = F.cross_entropy(net(prep(b)), labels[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = (logits(val).argmax(1) == labels[val]).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in
                          net.state_dict().items()}
    net.load_state_dict(best_state)
    secs = time.perf_counter() - t0
    return (lambda idx: logits(idx).argmax(1).numpy()), secs
