# Dataset layout

The Hugging Face dataset archive extracts to `dataset/`. It contains simulation
metadata, subset CSVs, gridded climate fields, spectral coefficients, and
normalization assets used by the public loaders and baselines.

## Inputs

`inputs.csv` has one row per simulation, keyed by `simulation_id`. The public
model inputs are:

- `stellar_temperature`
- `stellar_flux`
- `radius`
- `gravity`
- `rotation_period`
- `surface_pressure`
- `co2`
- `ch4`
- `gcm_label`

The file also includes metadata columns such as `is_target_gcm`, `in_target_physical_domain`, `planet_id`, and `source`.

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

## Field archives

`fields/` contains gridded climate targets:

- `all-obs.npz`: 1760 simulations, 53 fields
- `complete-obs-only.npz`: 1659 simulations, 48 fields

Both archives use:

- `simulation_id`: `(N,)` integer simulation IDs
- `field_names`: `(C,)` field/channel names
- `fields`: `(N, C, 32, 64)` float32 latitude-longitude grids

`all-obs.npz` includes surface temperature, temperature levels 0-9, specific
humidity levels 0-9, ASR, OLR, cloud fraction levels 0-9, east-west wind levels
0-9, and north-south wind levels 0-9. `complete-obs-only.npz` uses the same
order but stops 3D variables at level 8.

| Variable | Unit |
| --- | --- |
| Surface temperature | K |
| Temperature | K |
| Specific humidity | dex |
| Cloud fraction | 1 |
| East-west wind | m s^-1 |
| North-south wind | m s^-1 |
| Absorbed shortwave radiation | W m^-2 |
| Outgoing longwave radiation | W m^-2 |

Whole-field missingness is represented as all-NaN channels. Partial NaNs within a field are not part of the dataset contract.

## Spectral coefficients

`coefficients/` mirrors the two field archives in spectral space:

- `simulation_id`: `(N,)`
- `field_names`: `(C,)`
- `coefficients`: `(N, C, 484)`
- `field_mask`: `(N, C)`

`field_mask[i, j]` is false when field `j` is missing for simulation `i`; the
corresponding coefficient vector is zero-filled. Coefficients use truncation
`T=21` and the field order returned by
`thousandworlds.canonical_field_names(...)`.

## Subsets

`subsets/` contains the benchmark train/test membership CSVs. Each file has one
column, `simulation_id`, indexing rows in `inputs.csv` and entries in the field
archives.

| Subset | Simulations | Fields | Description |
| --- | ---: | ---: | --- |
| `single-complete` | 256 | 48 | UM-only complete-observation subset |
| `multi-complete` | 1659 | 48 | 5 GCMs, complete-observation subset |
| `multi-partial` | 1760 | 53 | 5 GCMs, subset with structured whole-field missingness |

| File | `single-complete` | `multi-complete` | `multi-partial` |
| --- | ---: | ---: | ---: |
| `train.csv` | 206 | 1538 | 1626 |
| `test.csv` | 50 | 90 | 100 |
| `test_shared_planets_only.csv` | - | 58 | 60 |
| `held_out_aux.csv` | - | 31 | 34 |

`held_out_aux.csv` is excluded from train and test to prevent train-test leakage (it contains simulations from auxiliary GCMs that correspond to identical planets present in the test set.)

Target GCMs are ExoCAM and UM. Auxiliary GCMs are ExoCAM pre-2022,
ExoPlaSim, and LFRic.

## Evaluation protocols

- **Standard**: evaluates on `test.csv`; available for all subsets.
- **Shared-planets**: evaluates on `test_shared_planets_only.csv`; available
  for the two multi-GCM subsets.

In Python, use `protocol="standard"` or `protocol="shared_planets"` with
`thousandworlds.load(...)`.

## Normalization assets

`norm_stats/<subset>/` contains the normalization and spectral-transform assets
used by the public preprocessing code:

- `normalize_mean.npz`
- `normalize_std.npz`
- `spectral.npz`
- `spectral.meta.json`
- `transforms.meta.json`

## Baseline results

Baseline predictions are distributed in separate Hugging Face archives:

- `results-baselines-<subset>-deterministic.tar.gz`
- `results-baselines-<subset>-gplfr.tar.gz`
- `results-baselines-<subset>-ppca_icm.tar.gz`

They extract into `results/`. Prediction files use the submission format
consumed by `thousandworlds.evaluate.score`: `predictions`, `simulation_id`,
and `field_names`.
The downloader keeps existing result docs and tables in place, and replaces
existing `predictions.npz` files only when `force=True`.
