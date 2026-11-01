"""Feature extraction (own numpy implementation; no librosa).

Two views of each discharge, both anchored on the annotated onset:

* ``X``   -- a 54-d cepstral vector for the classical methods and the
             Siamese network: 13 MFCC means, 13 MFCC stds, 13 delta means,
             13 delta-delta means, zero-crossing rate and RMS, computed over
             a 50 ms window centred on the onset (Davis & Mermelstein 1980).
* ``img`` -- a 64 x 79 log-mel image over [-0.1 s, +0.4 s] around the
             onset, resampled to 22.05 kHz, for the pretrained CNN (B2).

Everything is cached to ``data/features.npz``.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample

from data import CLASSES, DATA, load_metadata, read_wav

CACHE = DATA / "features.npz"
WIN_MS = 50
N_MFCC = 13
IMG_SR = 22050
IMG_PRE, IMG_POST = 0.1, 0.4
IMG_FFT, IMG_HOP, IMG_MELS = 1024, 128, 64


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0):
    """Triangular mel filters, shape (n_mels, n_fft // 2 + 1)."""
    def h2m(h): return 2595 * np.log10(1 + h / 700)
    def m2h(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(h2m(fmin), h2m(sr / 2), n_mels + 2)
    bins = np.clip(np.floor((n_fft + 1) * m2h(mels) / sr).astype(int),
                   0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def log_mel(seg: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int):
    """Log mel energies, shape (n_mels, frames)."""
    if seg.size < n_fft:
        seg = np.pad(seg, (0, n_fft - seg.size))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(
        1 + (seg.size - n_fft) // hop)[:, None]
    frames = seg[idx] * np.hanning(n_fft)[None, :]
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 / n_fft
    fb = mel_filterbank(sr, n_fft, n_mels)
    return np.log(np.maximum(power @ fb.T, 1e-10)).T


def mfcc(seg: np.ndarray, sr: int, n_fft: int = 512) -> np.ndarray:
    """MFCCs, shape (N_MFCC, frames)."""
    lm = log_mel(seg, sr, n_fft, n_fft // 4, 40)
    return dct(lm, type=2, axis=0, norm="ortho")[:N_MFCC]


def cepstral_vector(seg: np.ndarray, sr: int) -> np.ndarray:
    """The 54-d feature vector x used by every non-CNN method."""
    m = mfcc(seg, sr)
    d1 = np.diff(m, axis=1, prepend=m[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    zc = np.mean(np.abs(np.diff(np.sign(seg)))) / 2
    rms = float(np.sqrt(np.mean(seg ** 2)))
    return np.concatenate([m.mean(1), m.std(1), d1.mean(1), d2.mean(1),
                           [zc, rms]])


def window(sig: np.ndarray, centre: int, before: int, after: int):
    """sig[centre-before : centre+after], zero-padded at the file edges."""
    lo, hi = centre - before, centre + after
    seg = sig[max(0, lo):min(sig.size, hi)]
    return np.pad(seg, (max(0, -lo), max(0, hi - sig.size)))


def extract_all(force: bool = False):
    """Return dict with X, img, y, device, uuid (cached)."""
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    recs = load_metadata()
    X, img, y, dev, grp = [], [], [], [], []
    for i, r in enumerate(recs):
        if i % 300 == 0:
            print(f"  features {i}/{len(recs)}", flush=True)
        sig, sr = read_wav(r["path"])
        c = int(r["onsets"][0] * sr) if r["onsets"] else int(np.argmax(np.abs(sig)))

        half = int(sr * WIN_MS / 2000)
        seg = window(sig, c, half, half)
        X.append(cepstral_vector(seg / (np.abs(seg).max() + 1e-9), sr))

        long = window(sig, c, int(IMG_PRE * sr), int(IMG_POST * sr))
        long = resample(long, int((IMG_PRE + IMG_POST) * IMG_SR))
        im = log_mel(long, IMG_SR, IMG_FFT, IMG_HOP, IMG_MELS)
        img.append(((im - im.mean()) / (im.std() + 1e-6)).astype(np.float16))

        y.append(CLASSES.index(r["label"]))
        dev.append(r["device"])
        grp.append(r["uuid"])

    out = {"X": np.asarray(X, np.float32), "img": np.stack(img),
           "y": np.asarray(y), "device": np.asarray(dev),
           "uuid": np.asarray(grp)}
    np.savez_compressed(CACHE, **out)
    print(f"  cached {out['X'].shape} vectors and {out['img'].shape} images "
          f"-> {CACHE}")
    return out


if __name__ == "__main__":
    from data import download
    download()
    extract_all(force=True)
