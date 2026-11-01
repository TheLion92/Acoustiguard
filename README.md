# AcoustiGuard: firearm classification from gunshot audio, tested on microphones the model has never heard

COSC 750 Pattern Recognition I, Fall 2026 (Hermann Fokou Chendjou, Bowie State University).
This is the Stage 3 code: baselines, the proposed method (preliminary) and the evaluation protocol.

**Question.** Published systems classify the firearm from a gunshot recording at 90–96% accuracy.
Every one of them evaluates on a *random file split*. The public corpus used here recorded the *same*
discharges on 26 devices, so a random split puts copies of one shot on both sides of the train/test line.
How much accuracy is left when the test microphone is one the model has never heard?

## Reproduce the baseline table (one command)

```bash
make env          # python3 -m venv .venv + pinned requirements (or: conda env create -f environment.yml)
make data         # downloads the 1.6 GB corpus from Zenodo 7004819 into data/  (skip if present)
make baselines    # ≈ 15 min on an Apple M4: 6 methods x 3 splits x 5 seeds
make test         # 17 unit tests, including split disjointness
```

`make quick` runs everything except the CNN in about 2 minutes, CPU only.
If the data already exists elsewhere, use `AG_DATA=/path/to/data make baselines`.

Outputs go to `results/`:

| File | Contents |
|---|---|
| `baselines.csv`, `baselines.tex` | the table below (mean ± std over 5 seeds) |
| `runs.csv` | every run: method, split, seed, accuracy, macro-F1, risk, train time, chosen hyper-parameters |
| `stats.txt` | paired t-tests / Wilcoxon vs. the SVM, accuracy drop per method, device-ID diagnostic, platform |
| `protocols.pdf/png` | accuracy of each method under the three splits |
| `confusion.pdf/png` | confusion matrix of the strongest method under the device split |

## Results (5 seeds, mean ± std)

| Method | Random | Capture-disjoint | **Device-disjoint** | Device macro-F1 | Device risk ↓ |
|---|---|---|---|---|---|
| B0 Majority | 31.2 ± 0.0 | 31.2 ± 0.0 | 35.6 ± 0.1 | 13.1 ± 0.0 | 1.927 |
| B1a Gaussian Bayes (own) | 70.9 ± 2.4 | 47.6 ± 5.5 | 40.6 ± 3.3 | 38.9 ± 3.1 | 1.815 |
| B1b RBF-SVM | 88.5 ± 1.0 | 61.4 ± 4.1 | 48.1 ± 2.0 | 43.7 ± 1.8 | 1.576 |
| B1c k-NN | 87.5 ± 1.4 | 49.7 ± 3.7 | 38.3 ± 1.7 | 36.3 ± 1.4 | 1.772 |
| B2 ResNet-18, pretrained (published protocol) | **94.2 ± 0.9** | **69.6 ± 3.9** | **59.4 ± 5.4** | **56.1 ± 4.5** | **1.061** |
| Proposed: Siamese + centroid (preliminary) | 84.2 ± 1.3 | 60.7 ± 2.9 | 48.8 ± 5.7 | 46.1 ± 5.0 | 1.351 |

* Reproduction check: Doungpaisan & Khunarsa (2025) report 95.81% on this corpus with a random split. Our
  re-run of their protocol with a lighter backbone gives 94.2 ± 0.9%.
* Every method loses 30–49 accuracy points when the test device is unseen.
* The same 54 features identify the *recording device* at 95.9% accuracy (chance: 3.8%).

## Layout

```
src/data.py           corpus download, metadata join (device, capture uuid, onset), WAV reader
src/features.py       own numpy MFCC/log-mel: 54-d vector (50 ms at onset) and 64x79 image (0.5 s)
src/splits.py         random / capture-disjoint / device-disjoint outer splits; grouped inner CV
src/baselines.py      B0, B1a Gaussian plug-in Bayes (own), B1b SVM, B1c k-NN, B2 ResNet-18
src/siamese.py        proposed method: device-contrastive Siamese MLP + nearest centroid
src/run_baselines.py  runs the whole protocol, writes results/
tests/                feature, classifier and split-disjointness tests
```

## Protocol, in short

* For each split type and seed 0–4, there is one 80/20 outer split shared by every method.
* Hyper-parameters are chosen by a 3-fold grid search **on the training part only**. The inner folds are
  grouped the same way as the outer split. The test part is scored once.
* Metrics are accuracy, macro-F1, and expected risk under a loss matrix that charges 1 for a
  handgun↔handgun or long-gun↔long-gun confusion and 4 for a cross-category one.
* Same features for every method, except B2, which uses the log-mel image its paper prescribes.

## What is own code vs. reused

Own: the feature extractor, Gaussian Bayes, splits, Siamese network, and runner.
Reused: SVM, k-NN and grid search (scikit-learn); ResNet-18 ImageNet weights (torchvision).

## Data

Kabealo et al. (2023), *Data in Brief* 48:109091, CC BY 4.0. 2,148 WAVs, 4 firearms, 26 devices.
The data is not redistributed here; `make data` downloads it.

Seeds are fixed. Every number reproduces exactly except B2, whose MPS/GPU kernels are not
bit-deterministic and may differ in the third decimal.
