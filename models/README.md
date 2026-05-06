# Model Details

This is a short map of the public baseline implementations. Resolved run configs
and metrics live under `results/models/<subset>/<method>/`; the code entry point
is `thousandworlds/run_model.py`.

All models use the benchmark transforms from `thousandworlds/preprocessing.py`
and score through `thousandworlds/evaluate.py`.

## Training Mean

`train_mean` predicts the same per-field training mean for every test case. See
`models/train_mean.py`.

## kNN

`knn` averages nearby training examples after standardizing the input
parameters. The public CV sweep uses `k = 1, 2, 3, 5, 10` and
`gcm_penalty = 0.0, 0.3, 1.0, 3.0, 10.0`; nonzero penalty values make same-GCM
neighbors closer in the multi-GCM subsets. See `models/knn.py`.

## PCA-Ridge

`pca_ridge` compresses T21 spectral coefficients with PPCA, then predicts the
latent scores with ridge regression. See `models/pca_ridge.py` and
`models/_ppca.py`.

## PCA-MLP

`pca_mlp` uses the same PPCA representation as PCA-Ridge, but maps inputs to
latent scores with a two-hidden-layer MLP. See `models/pca_mlp.py`.

## Coord-MLP

`coord_mlp` predicts one grid value at a time from planet inputs, GCM identity,
field identity, level, latitude, and longitude. See `models/coord_mlp.py`.

## Coord-DeepONet

`coord_deeponet` is the learned-basis version of Coord-MLP: a branch network
encodes the planet/GCM input and a trunk network encodes the field/grid query.
See `models/coord_deeponet.py`.

## PPCA-ICM

`ppca_icm` predicts PPCA latent scores with a Gaussian process using a
Matern-5/2 input kernel and a GCM coregionalization term. The released configs
use 64 posterior samples. See `models/ppca_icm.py`.

For deterministic metrics, `evaluate.py` uses `predictions_mean.npz` when a
probabilistic method provides one; probabilistic metrics use the ensemble in
`predictions.npz`.

## GPLFR

`gplfr` is the frozen public GPLFR recipe: MAP fitting in latent space, fixed
variable-group weights, no output coregionalization term, and posterior samples
written through the standard ThousandWorlds result path. See `models/gplfr.py`
and `models/_gplfr_core.py`.
