"""Corpus access: download, metadata join, and WAV reading.

The corpus is Kabealo et al. (2023), Data in Brief 48:109091 -- 2,148
labelled gunshot WAVs from 26 recording devices, CC BY 4.0, Zenodo 7004819.
Labels, device identity, capture uuid and annotated onsets all come from the
metadata CSV that ships inside the archive; nothing is inferred from paths.
"""

from __future__ import annotations

import csv
import os
import struct
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AG_DATA", ROOT / "data"))
AUDIO = DATA / "gunshots"
META = AUDIO / "gunshot-audio-all-metadata.csv"

ARCHIVE_NAME = "edge-collected-gunshot-audio.zip"
ARCHIVE_URL = ("https://zenodo.org/records/7004819/files/"
               f"{ARCHIVE_NAME}?download=1")

# CSV label -> class name, in a fixed order that defines the integer labels
CLASSES = ["pistol", "revolver", "rifle", "shotgun"]
LABEL_MAP = {"9mm": "pistol", "38cal": "revolver",
             "AR15": "rifle", "12gauge": "shotgun"}
EXPECTED = {"pistol": 669, "rifle": 597, "revolver": 503, "shotgun": 379}


def download() -> None:
    """Fetch and extract the archive unless the WAVs are already present."""
    if META.exists() and sum(1 for _ in AUDIO.rglob("*.wav")) > 2000:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / ARCHIVE_NAME
    if not dest.exists() or dest.stat().st_size < 1_000_000_000:
        print(f"downloading {ARCHIVE_URL} (1.6 GB) -> {dest}")
        urllib.request.urlretrieve(ARCHIVE_URL, dest)
    print(f"extracting -> {AUDIO}")
    AUDIO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest) as zf:
        zf.extractall(AUDIO)


def load_metadata(verbose: bool = True) -> list[dict]:
    """One record per WAV with label, device, capture uuid and onsets."""
    rows = {r["filename"]: r for r in csv.DictReader(open(META, newline=""))}
    recs = []
    for w in sorted(AUDIO.rglob("*.wav")):
        m = rows.get(w.stem)
        if m is None:
            continue
        # numpy-repr list, e.g. "[1.50206349 1.75       1.95494331]"
        raw = m["gunshot_location_in_seconds"].strip().strip("[]")
        onsets = [float(t) for t in raw.replace(",", " ").split()]
        recs.append({"path": w, "label": LABEL_MAP[m["label"]],
                     "device": m["device_name"], "uuid": m["uuid"],
                     "onsets": onsets})
    if verbose:
        cls = Counter(r["label"] for r in recs)
        ok = all(cls[k] == v for k, v in EXPECTED.items())
        print(f"metadata: {len(recs)} files, "
              f"{len(set(r['device'] for r in recs))} devices, "
              f"{len(set(r['uuid'] for r in recs))} captures; class counts "
              + ", ".join(f"{k} {cls[k]}" for k in CLASSES)
              + ("  (matches published table)" if ok else "  (MISMATCH)"))
    return recs


def _read_wav_riff(path):
    """Minimal RIFF reader for the corpus's 32-bit IEEE-float WAVs, which
    the stdlib `wave` module cannot open."""
    buf = Path(path).read_bytes()
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    fmt = sr = ch = bits = data = None
    i = 12
    while i + 8 <= len(buf):
        cid = buf[i:i + 4]
        sz = struct.unpack("<I", buf[i + 4:i + 8])[0]
        body = buf[i + 8:i + 8 + sz]
        if cid == b"fmt ":
            fmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt == 0xFFFE and len(body) >= 40:      # WAVE_FORMAT_EXTENSIBLE
                fmt = struct.unpack("<H", body[24:26])[0]
        elif cid == b"data":
            data = body
        i += 8 + sz + (sz & 1)
    dt = {(3, 32): "<f4", (3, 64): "<f8", (1, 16): "<i2",
          (1, 32): "<i4"}[(fmt, bits)]
    sig = np.frombuffer(data, dtype=dt).astype(np.float32)
    if ch and ch > 1:
        sig = sig[:sig.size - sig.size % ch].reshape(-1, ch).mean(axis=1)
    return sig, sr


def read_wav(path) -> tuple[np.ndarray, int]:
    """Mono float32 signal, peak-normalised, and its sample rate."""
    try:
        from scipy.io import wavfile
        sr, sig = wavfile.read(str(path))
        sig = sig.astype(np.float32)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
    except Exception:
        sig, sr = _read_wav_riff(path)
    return sig / (np.abs(sig).max() + 1e-9), int(sr)
