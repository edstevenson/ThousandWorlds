# Results

This directory contains public baseline outputs for ThousandWorlds.

- `models/<subset>/<method>/`: resolved runner config, prediction archive, and metrics JSON for each baseline.
- `tables/`: paper-oriented metric tables grouped by subset, protocol, and metric.
- `scores.csv`: flat per-variable baseline scores generated from the metrics JSON files.
- `scores_5seeds.csv`: the same score format with a `seed` column. Learned
  baselines include seeds 0-4 here; these are the scores used for the paper
  results.

The checked-in configs, predictions, metrics JSON files, and rendered tables are
the seed-0 artifacts. Use `scores_5seeds.csv` for seed-averaged numbers matching the paper.

`models/*/*/predictions.npz` follows the standard ThousandWorlds submission
format: `predictions`, `simulation_id`, and `field_names`.
