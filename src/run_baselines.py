"""Reproduce the Stage 3 baseline table.

    python src/run_baselines.py            # everything (≈ 15-25 min on an M-series Mac)
    python src/run_baselines.py --no-cnn   # classical + Siamese only (≈ 2 min)

For each protocol (random / capture-disjoint / device-disjoint) and each seed
0..S-1, one 80/20 outer split is drawn; every method sees the same split,
is tuned on the training part only, and is scored once on the test part.

Outputs in results/: runs.csv (one row per method x protocol x seed),
baselines.csv and baselines.tex (mean ± std), stats.txt (paired t-tests,
device-ID diagnostic), confusion.pdf, protocols.pdf.
"""

from __future__ import annotations

import argparse
import warnings
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from baselines import CLASSICAL, fit_classical, fit_cnn  # noqa: E402
from data import CLASSES, download                       # noqa: E402
from features import extract_all                         # noqa: E402
from siamese import fit_siamese                          # noqa: E402
from splits import PROTOCOLS, inner_cv, outer_split      # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path(__file__).resolve().parents[1] / "results"

# RQ3 loss matrix lambda(alpha_i | omega_j): rows = decision, cols = truth.
# Handguns {pistol, revolver} vs long guns {rifle, shotgun}: a within-category
# confusion costs 1, a cross-category confusion costs 4.
CATEGORY = np.array([0, 0, 1, 1])
LOSS = np.where(CATEGORY[:, None] == CATEGORY[None, :], 1.0, 4.0)
np.fill_diagonal(LOSS, 0.0)

PROPOSED = "P  Siamese + centroid"
CNN = "B2 ResNet-18 (pretrained)"


def score(y_true, y_pred):
    return {"acc": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
            "risk": float(LOSS[y_pred, y_true].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--no-siamese", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    download()
    D = extract_all()
    X, y, img, dev = D["X"], D["y"], D["img"], D["device"]
    print(f"data: X {X.shape}, images {img.shape}, "
          f"{len(set(dev))} devices, {len(set(D['uuid']))} captures")

    methods = list(CLASSICAL)
    if not args.no_cnn:
        methods.append(CNN)
    if not args.no_siamese:
        methods.append(PROPOSED)

    rows, preds = [], {}
    for proto in PROTOCOLS:
        for seed in range(args.seeds):
            tr, te = outer_split(proto, D, seed)
            cv = inner_cv(proto, D, tr, seed)
            for name in methods:
                if name in CLASSICAL:
                    model, params, secs = fit_classical(name, X[tr], y[tr],
                                                        cv, seed)
                    yp = model.predict(X[te])
                elif name == CNN:
                    fit_i, val_i = cv[0]          # group-disjoint early stop
                    predict, secs = fit_cnn(img, y, tr[fit_i], tr[val_i],
                                            seed)
                    yp, params = predict(te), {"epochs": "early-stopped"}
                else:
                    predict, secs = fit_siamese(X[tr], y[tr], dev[tr], seed)
                    yp, params = predict(X[te]), {}
                s = score(y[te], yp)
                rows.append({"method": name, "protocol": proto, "seed": seed,
                             "n_train": len(tr), "n_test": len(te),
                             **s, "train_s": secs,
                             "params": json.dumps(params, default=str)})
                preds.setdefault((name, proto), []).append((y[te], yp))
                print(f"{proto:8s} seed {seed}  {name:26s} acc {s['acc']:.3f}"
                      f"  F1 {s['f1']:.3f}  risk {s['risk']:.3f}"
                      f"  ({secs:.1f}s)", flush=True)

    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summarise(rows, methods, args.seeds)
    figures(rows, preds, methods)
    diagnostics(D, rows, methods)


def _agg(rows, name, proto, key):
    v = np.array([r[key] for r in rows
                  if r["method"] == name and r["protocol"] == proto])
    return v.mean(), v.std(ddof=1) if v.size > 1 else 0.0


def summarise(rows, methods, n_seeds):
    cols = [("random", "acc"), ("capture", "acc"), ("device", "acc"),
            ("device", "f1"), ("device", "risk"), ("device", "train_s")]
    with open(OUT / "baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{p}_{k}_{s}" for p, k in cols
                                 for s in ("mean", "std")])
        for m in methods:
            w.writerow([m] + [f"{x:.4f}" for p, k in cols
                              for x in _agg(rows, m, p, k)])

    lines = [r"\begin{tabular}{@{}lcccccr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Accuracy} & "
             r"\multicolumn{2}{c}{Device-disjoint} & Train \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Method & Random & Capture & Device & Macro-$F_1$ & "
             r"Risk$\downarrow$ & time (s) \\", r"\midrule"]
    for m in methods:
        cells = []
        for p, k in cols:
            mu, sd = _agg(rows, m, p, k)
            cells.append(f"{mu:.1f}" if k == "train_s" else
                         f"{mu:.3f}$\\pm${sd:.3f}" if k == "risk" else
                         f"{100*mu:.1f}$\\pm${100*sd:.1f}")
        label = m.replace("&", r"\&")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "baselines.tex").write_text("\n".join(lines) + "\n")
    print("\n" + (OUT / "baselines.tex").read_text())


def figures(rows, preds, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # strongest under device-disjoint, by macro-F1
    best = max((m for m in methods if not m.startswith("B0")),
               key=lambda m: _agg(rows, m, "device", "f1")[0])
    yt = np.concatenate([a for a, _ in preds[(best, "device")]])
    yp = np.concatenate([b for _, b in preds[(best, "device")]])
    cm = confusion_matrix(yt, yp, normalize="true")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > .5 else "black", fontsize=8)
    ax.set_xticks(range(4), CLASSES, rotation=30, fontsize=8)
    ax.set_yticks(range(4), CLASSES, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(f"{best.split(' ', 1)[1]} - device-disjoint", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.pdf")
    fig.savefig(OUT / "confusion.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    width = 0.8 / len(methods)
    colors = ["#9aa5b1", "#3b6fb6", "#1f4e8c", "#6b9bd2", "#c4622d", "#2f8f5b"]
    for i, m in enumerate(methods):
        mus = [100 * _agg(rows, m, p, "acc")[0] for p in PROTOCOLS]
        sds = [100 * _agg(rows, m, p, "acc")[1] for p in PROTOCOLS]
        ax.bar(np.arange(3) + i * width - 0.4 + width / 2, mus, width,
               yerr=sds, capsize=2, label=m.split(" ", 1)[1],
               color=colors[i % len(colors)])
    ax.set_xticks(range(3), ["random file split", "capture-disjoint",
                             "device-disjoint"], fontsize=8)
    ax.set_ylabel("test accuracy (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, ncol=3, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "protocols.pdf")
    fig.savefig(OUT / "protocols.png", dpi=200)
    print(f"figures: confusion matrix of '{best}', protocol comparison")


def diagnostics(D, rows, methods):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = [f"python {platform.python_version()}  {platform.platform()}  "
           f"{platform.processor() or platform.machine()}",
           f"generated {time.strftime('%Y-%m-%d %H:%M')}", ""]
    ref = "B1b RBF-SVM"
    for proto in PROTOCOLS:
        a = [r["f1"] for r in rows if r["method"] == ref
             and r["protocol"] == proto]
        for m in methods:
            if m == ref or m.startswith("B0"):
                continue
            b = [r["f1"] for r in rows if r["method"] == m
                 and r["protocol"] == proto]
            t = ttest_rel(b, a)
            try:
                wp = wilcoxon(b, a).pvalue
            except ValueError:
                wp = float("nan")
            out.append(f"{proto:8s} {m:26s} vs {ref}: dF1 "
                       f"{np.mean(b)-np.mean(a):+.3f}  paired-t p={t.pvalue:.4f}"
                       f"  wilcoxon p={wp:.4f}")
    for m in methods:
        r_ = _agg(rows, m, "random", "acc")[0]
        d_ = _agg(rows, m, "device", "acc")[0]
        out.append(f"drop random->device  {m:26s} {100*(r_-d_):+.1f} pts")

    devs, inv = np.unique(D["device"], return_inverse=True)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    acc = cross_val_score(pipe, D["X"], inv,
                          cv=StratifiedKFold(3, shuffle=True,
                                             random_state=0)).mean()
    out.append(f"\ndevice-ID diagnostic: the same 54-d features predict the "
               f"RECORDER at {acc:.3f} accuracy ({len(devs)} devices, chance "
               f"{1/len(devs):.3f})")
    (OUT / "stats.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
