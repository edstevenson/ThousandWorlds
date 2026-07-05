# ThousandWorlds

<img src="imgs/MASCOT.png" align="right" width="220" alt="ThousandWorlds mascot">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/dataset-10.57967%2Fhf%2F8695-yellow.svg)](https://doi.org/10.57967/hf/8695)
[![arXiv](https://img.shields.io/badge/arXiv-2606.18338-b31b1b.svg)](https://arxiv.org/abs/2606.18338)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

The search for life beyond Earth depends on the molecular signatures it leaves behind in the atmospheres of its host planet. Correctly interpreting these signatures requires understanding the climates of potential host planets. **ThousandWorlds** is a benchmark for emulating these exoplanet climates: **1760 simulations** across 5 GCMs, 8 planet parameters, and atmospheric variables on a 32 x 64 x 10 latitude-longitude-pressure grid. It includes three nested benchmark subsets, two evaluation protocols, and eight released baseline methods.

Explore the dataset + discovered exoplanets online with the [ThousandWorlds Explorer](https://thousandworldsexplorer.com)!
Built by [Hamza Ali Shahjahan](https://github.com/hamza-ali-shahjahan)!

<br>

![ThousandWorlds dataset schematic](imgs/OVERVIEW.png)

## Quickstart

```bash
pip install -e .
```

```python
import numpy as np
import thousandworlds as tw

tw.download_dataset()
bundle = tw.load("single-complete", data_dir="dataset")

pred = np.broadcast_to(bundle.Y_train.mean(axis=0), bundle.Y_test.shape)
scores = tw.evaluate.rmse(pred, bundle.Y_test, bundle.field_mask_test, bundle.field_names)
scores["per_variable"]
```

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a short
walkthrough.

## Installation

```bash
pip install -e .              # core: data loading + evaluation
pip install -e '.[models]'    # baseline model dependencies
pip install -e '.[notebooks]' # notebook dependencies
```

## Dataset

The benchmark dataset is hosted on
[Hugging Face](https://doi.org/10.57967/hf/8695). The
repository already contains metadata and directory layout; this fills in the
large array files:

```bash
python -c "import thousandworlds as tw; tw.download_dataset()"
```

Once downloaded, [`notebooks/explore_trappist1e.ipynb`](notebooks/explore_trappist1e.ipynb)
tours an example world's climate.

## Baselines

Published baseline prediction results are distributed as separate archives:

```bash
python -c "import thousandworlds as tw; tw.download_baselines()"
```

To run baselines yourself:

```bash
python -m thousandworlds.run_model train_mean single-complete
python -m thousandworlds.run_model --config results/models/multi-partial/pca_mlp/config.json
```

The first form runs a method on a subset with default hyperparameters (override
with flags); the second reproduces a published baseline from its checked-in
`config.json`. Each run writes predictions, metrics, and the resolved config to
`results/models/<subset>/<method>/`, overwriting the checked-in results by default
(use `--out-dir` to redirect).

See [`notebooks/pca_mlp.ipynb`](notebooks/pca_mlp.ipynb) for a quick example that
trains a baseline in-notebook and compares its predictions to the targets.

## Repo Structure

```
thousandworlds/
  data.py               # download + load
  preprocessing.py      # input/output transforms, normalization
  spectral.py           # spectral coefficients <-> gridded fields
  evaluate.py           # RMSE, ACC, energy score, spread-skill ratio, etc.
  run_model.py          # CLI entry point
  make_model_tables.py  # regenerate result tables
  models/               # baseline implementations
  assets/               # precomputed SHT matrix, latitude weights

dataset/                # inputs.csv, subset CSVs, arrays after download
results/                # configs, metrics, published tables
notebooks/              # quickstart, explore_trappist1e, pca_mlp
tests/                  # test suite
```

## Citation

If you use ThousandWorlds, please cite the paper:

```bibtex
@article{thousandworlds2026,
  title = {ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets},
  author = {Stevenson, Edward T. and Mak, Mei Ting and Wolf, Eric and Sergeev, Denis E. and Hammond, Tobi and Mayne, N. J. and Cranmer, Miles},
  year = {2026},
  eprint = {2606.18338},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2606.18338}
}
```
