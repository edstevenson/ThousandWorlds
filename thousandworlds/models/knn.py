from __future__ import annotations

import numpy as np
from collections.abc import Sequence
from sklearn.neighbors import NearestNeighbors


class KNN:
    """k-Nearest Neighbors model for steady-state field prediction.

    - Input: planet constants vectors X [N, d] (already normalized via transforms)
    - Output: climate fields Y [N, C, H, W] (raw, unnormalized)
    - Distance: Euclidean in normalized input space
    - Prediction: Uniform average of k nearest neighbors' raw fields
    """

    def __init__(self, max_neighbors: int | None = None, metric: str = "euclidean", algorithm: str = "auto") -> None:
        self.max_neighbors = max_neighbors
        self.metric = metric
        self.algorithm = algorithm

        self._nbrs: NearestNeighbors | None = None
        self._X_train: np.ndarray | None = None
        self._Y_train: np.ndarray | None = None
        self._field_mask: np.ndarray | None = None
        self._field_fallback: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        k_candidates: Sequence[int] | None = None,
        field_mask: np.ndarray | None = None,
    ) -> None:
        """Fit the neighbor index on training data.

        Args:
            X_train: [N, d] float32 array of normalized planet constants
            Y_train: [N, C, H, W] float32 array of raw fields
            k_candidates: optional list to determine required max_neighbors
            field_mask: optional [N, C] boolean mask where True = field is present
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D [N,d], got shape {X_train.shape}")
        if Y_train.ndim != 4:
            raise ValueError(f"Y_train must be 4D [N,C,H,W], got shape {Y_train.shape}")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have the same first dimension (N)")

        N = X_train.shape[0]
        if N == 0:
            raise ValueError("Training set is empty")

        self._X_train = X_train.astype(np.float32)
        Y = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
        self._field_mask = None if field_mask is None else np.asarray(field_mask, dtype=bool)
        self._Y_train = Y if self._field_mask is None else np.where(self._field_mask[:, :, None, None], Y, 0.0).astype(np.float32)
        if field_mask is None:
            self._field_fallback = self._Y_train.mean(axis=0).astype(np.float32)
        else:
            mask = self._field_mask.astype(np.float32)[:, :, None, None]
            count = mask.sum(axis=0)
            summed = (self._Y_train * mask).sum(axis=0)
            self._field_fallback = np.where(count > 0, summed / count, 0.0).astype(np.float32)

        # Determine neighbors to store
        n_neighbors_needed = self.max_neighbors if self.max_neighbors is not None else (max(k_candidates) if k_candidates else 20)
        n_neighbors = int(min(max(1, n_neighbors_needed), N))

        # Fit neighbor index
        self._nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric, algorithm=self.algorithm)
        self._nbrs.fit(self._X_train)

    def kneighbors(self, X_query: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest neighbors in the training set for each query row.

        Args:
            X_query: [B,d] float32 array of normalized planet constants
            k: number of neighbors (will be clipped to [1, N])
        Returns:
            idx: [B,k] int array of neighbor indices
        """
        if self._nbrs is None or self._X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        N = self._X_train.shape[0]
        k_eff = int(min(max(1, k), N))
        X_query = X_query.astype(np.float32)
        idx = self._nbrs.kneighbors(X_query, n_neighbors=k_eff, return_distance=False)
        return idx

    def predict(self, X_query: np.ndarray, k: int | Sequence[int]) -> np.ndarray | dict[int, np.ndarray]:
        """Predict fields by averaging the k nearest neighbors' fields.

        Args:
            X_query: [B, d] float32 array of normalized planet constants
            k: number of neighbors to average
        Returns:
            preds: [B, C, H, W] float32 array (or dict if k is a sequence)
        """
        if self._Y_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        def _predict_from_neighbors(Y_neighbors: np.ndarray, neighbor_mask: np.ndarray | None) -> np.ndarray:
            # Y_neighbors: [B, k, C, H, W], neighbor_mask: [B, k, C] or None
            if neighbor_mask is None:
                return Y_neighbors.mean(axis=1).astype(np.float32)
            # Masked averaging per channel
            mask_expanded = neighbor_mask[:, :, :, None, None]  # [B, k, C, 1, 1]
            count = mask_expanded.sum(axis=1)  # [B, C, 1, 1]
            avg = (Y_neighbors * mask_expanded).sum(axis=1) / np.clip(count, a_min=1, a_max=None)
            if self._field_fallback is None:
                return avg.astype(np.float32)
            fallback = self._field_fallback[None, ...]  # [1, C, H, W]
            return np.where(count > 0, avg, fallback).astype(np.float32)

        # If multiple candidate k are provided, treat each independently
        if isinstance(k, Sequence):
            if len(k) == 0:
                raise ValueError("k sequence must be non-empty")
            max_k = int(max(k))
            idx = self.kneighbors(X_query, max_k)
            Y_neighbors_full = self._Y_train[idx]  # [B, max_k, C, H, W]
            neighbor_mask_full = self._field_mask[idx] if self._field_mask is not None else None
            results: dict[int, np.ndarray] = {}
            for k_i in k:
                k_eff = int(max(1, min(k_i, Y_neighbors_full.shape[1])))
                nm = neighbor_mask_full[:, :k_eff, :] if neighbor_mask_full is not None else None
                results[int(k_i)] = _predict_from_neighbors(Y_neighbors_full[:, :k_eff, ...], nm)
            return results

        # Single k case
        idx = self.kneighbors(X_query, int(k))
        Y_neighbors = self._Y_train[idx]
        neighbor_mask = self._field_mask[idx] if self._field_mask is not None else None
        return _predict_from_neighbors(Y_neighbors, neighbor_mask)

