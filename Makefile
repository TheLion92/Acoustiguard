PY ?= .venv/bin/python

.PHONY: env data features baselines quick test

env:            ## create .venv and install pinned dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

data:           ## download + extract the corpus (1.6 GB, Zenodo 7004819)
	$(PY) -c "import sys; sys.path.insert(0,\"src\"); import data; data.download()"

features:       ## (re)build data/features.npz
	cd src && ../$(PY) features.py

baselines:      ## THE one command: reproduces results/baselines.{csv,tex}
	$(PY) src/run_baselines.py --seeds 5

quick:          ## classical + Siamese only, ~2 min, no GPU/MPS needed
	$(PY) src/run_baselines.py --seeds 5 --no-cnn

test:
	$(PY) -m pytest -q tests
