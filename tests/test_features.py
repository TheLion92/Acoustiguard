"""Core routines: the feature extractor and the own-code Gaussian Bayes."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import GaussianBayes                   # noqa: E402
from features import cepstral_vector, mel_filterbank, window  # noqa: E402


def test_cepstral_vector_shape_and_finite():
    rng = np.random.default_rng(0)
    for sr in (44100, 48000, 16000):
        seg = rng.standard_normal(int(0.05 * sr)).astype(np.float32)
        x = cepstral_vector(seg, sr)
        assert x.shape == (54,)
        assert np.all(np.isfinite(x))


def test_mel_filterbank_rows_nonnegative():
    fb = mel_filterbank(44100, 512, 40)
    assert fb.shape == (40, 257)
    assert fb.min() >= 0 and (fb.sum(1) > 0).sum() >= 35


def test_window_pads_at_edges():
    sig = np.arange(10, dtype=float)
    seg = window(sig, 1, 3, 3)
    assert seg.size == 6 and seg[0] == 0 and seg[2] == 0 and seg[3] == 1


def test_gaussian_bayes_separates_gaussians():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 1, (200, 3)) for m in (0, 4, 8)])
    y = np.repeat([0, 1, 2], 200)
    clf = GaussianBayes(eps=1e-3).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.95
