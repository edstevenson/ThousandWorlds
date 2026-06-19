---
pretty_name: ThousandWorlds
license: cc-by-4.0
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-regression
  - other
tags:
  - benchmark
  - datasets
  - physical-sciences
  - scientific-machine-learning
  - exoplanets
  - climate
  - astronomy
  - emulation
  - simulation
  - physics
  - pde
  - parameter-to-field-regression
  - structured-outputs
  - multi-simulator-transfer
  - spatiotemporal
configs:
  - config_name: input_planets
    data_files:
      - split: all
        path: inputs.csv
---

# ThousandWorlds

<img src="imgs/MASCOT.png" align="right" width="220" style="margin-top: -1.25rem;" alt="ThousandWorlds mascot">

ThousandWorlds is a benchmark for emulating exoplanet climates: **1760
simulations** across **5 GCMs**, **8 planet parameters**, and atmospheric
variables on a 32 x 64 x 10 latitude-longitude-pressure grid. It includes three
nested benchmark subsets, two evaluation protocols, and eight released baseline
methods.

[![Code](https://img.shields.io/badge/code-GitHub-181717.svg?logo=github)](https://github.com/edstevenson/ThousandWorlds)
[![arXiv](https://img.shields.io/badge/arXiv-2606.18338-b31b1b.svg)](https://arxiv.org/abs/2606.18338)

Inputs are 8 continuous planet parameters plus the source GCM label. Outputs
are time-averaged climate fields on a 32 x 64 latitude-longitude grid:
three-dimensional variables are stored as pressure-level channels, and
two-dimensional variables are stored as single-level fields.

![ThousandWorlds dataset schematic](imgs/OVERVIEW.png)

## Quickstart

The easiest way to use the benchmark is through the
[Python code](https://github.com/edstevenson/ThousandWorlds):

```bash
git clone https://github.com/edstevenson/ThousandWorlds.git
cd ThousandWorlds
pip install -e .
```

```python
import thousandworlds as tw

tw.download_dataset()
bundle = tw.load("single-complete", data_dir="dataset")
```

See the GitHub repository for the full quickstart, notebooks, baseline code,
evaluation utilities, and reproducing paper results.

## Files

The release includes:

- `archives/dataset.tar.gz`: the ThousandWorlds dataset.
- `archives/results-baselines-*.tar.gz`: baseline predictions for the 3
  subsets.
- `croissant.json`: Croissant metadata.
- `archives/*.sha256`: checksum sidecars.

## Dataset Contents

The dataset contains gridded fields (NumPy), input metadata (CSV), predefined
train/test splits, normalization statistics, and spherical harmonic
coefficients plus inverse-SHT weights for spectral methods.

## Subsets

The dataset is organized into three subsets of increasing complexity and
realism:

| Subset | Simulations | Fields | Description |
| --- | ---: | ---: | --- |
| `single-complete` | 256 | 48 | Smaller subset; simulations from a single GCM, complete observations only. |
| `multi-complete` | 1659 | 48 | All 5 GCMs, still with no missing fields. |
| `multi-partial` | 1760 | 53 | Full dataset; all 5 GCMs, with missing fields represented as NaNs. |

The subset split files contain:

| File | `single-complete` | `multi-complete` | `multi-partial` |
| --- | ---: | ---: | ---: |
| `train.csv` | 206 | 1538 | 1626 |
| `test.csv` | 50 | 90 | 100 |
| `test_shared_planets_only.csv` | - | 58 | 60 |
| `held_out_aux.csv` | - | 31 | 34 |

`held_out_aux.csv` is excluded from train and test to prevent train-test leakage (it contains simulations from auxiliary GCMs that correspond to identical planets present in the test set).

## Inputs

Each simulation has one row in `dataset/inputs.csv`, keyed by `simulation_id`.
The public model inputs are stellar temperature, stellar flux, radius, gravity,
rotation period, surface pressure, CO2, CH4, and `gcm_label`. The metadata also
includes `is_target_gcm`, `in_target_physical_domain`, `planet_id`, and
`source`.

| Parameter | Range |
| --- | --- |
| Radius (Earth radii) | [0.7, 1.4] |
| Surface gravity (m s^-2) | [6.0, 16.0] |
| Rotation period (days) | [0.1, 1000.0] |
| Surface pressure (bar) | [0.5, 5] |
| CO2 volume fraction (%) | [0, 100] |
| CH4 volume fraction (%) | [0, 5] |
| Incident stellar flux (W m^-2) | [500, 1500] |
| Stellar temperature (K) | [2500, 5800] |

## Outputs

Target fields include surface temperature, 3D temperature, specific humidity,
cloud fraction, east-west wind, north-south wind, absorbed shortwave radiation,
and outgoing longwave radiation. Gridded targets are provided on a 32 x 64
latitude-longitude grid, with vertical fields stored on relative pressure
levels.

| Variable | Dimensionality | Unit |
| --- | --- | --- |
| Surface temperature | 2D | K |
| Temperature | 3D | K |
| Specific humidity | 3D | dex |
| Cloud fraction | 3D | 1 |
| East-west wind | 3D | m s^-1 |
| North-south wind | 3D | m s^-1 |
| Absorbed shortwave radiation | 2D | W m^-2 |
| Outgoing longwave radiation | 2D | W m^-2 |

The gridded field archives are:

| File | Shape | Contents |
| --- | --- | --- |
| `dataset/fields/all-obs.npz` | `(1760, 53, 32, 64)` | Field archive covering all 5 GCMs with structured whole-field missingness. |
| `dataset/fields/complete-obs-only.npz` | `(1659, 48, 32, 64)` | Complete-observation field archive. |

**Spectral Coefficients:**
The spectral coefficient archives mirror those field archives with T21
spherical harmonic coefficients: `dataset/coefficients/*.npz` stores
`coefficients` with 484 coefficients per field and a `field_mask` for missing
fields. Whole-field missingness is represented as all-NaN gridded channels and
as false entries in the spectral `field_mask`.

## Evaluation

The package includes loaders and metrics for two benchmark protocols:

- **Standard**: the main test protocol, ideal for ML model comparison.
- **Shared-planets**: evaluate on planets shared across target and auxiliary
  GCMs; used to assess performance relative to inter-GCM error, i.e. how close
  a model gets to the epistemic uncertainty floor of the problem.

Released baselines include train mean, kNN, PCA ridge, PCA-MLP, Coord-MLP,
Coord-DeepONet, PPCA-ICM, and GPLFR. Baseline artifacts include predictions,
resolved configs, and metrics JSON files.

## Links

- DOI: https://doi.org/10.57967/hf/8695
- Code: https://github.com/edstevenson/ThousandWorlds
- Paper: https://arxiv.org/abs/2606.18338

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
