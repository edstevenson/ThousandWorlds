from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import thousandworlds as tw

from ._common import average_space_grid, inverse_average_space_grid, masked_mean_grid


@dataclass
class TrainMean:
    field_names: list[str] | None = None
    stats: tw.Stats | None = None
    mean_: np.ndarray | None = None

    def fit(
        self,
        Y_train: np.ndarray,
        field_names: list[str],
        stats: tw.Stats,
        *,
        X_train: np.ndarray,
        field_mask: np.ndarray | None = None,
    ) -> TrainMean:
        Y_avg = average_space_grid(Y_train, field_names, stats, X=X_train)
        self.field_names = list(field_names)
        self.stats = stats
        self.mean_ = masked_mean_grid(Y_avg, field_mask)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.field_names is None or self.stats is None:
            raise RuntimeError("Model not fitted.")
        pred = np.broadcast_to(self.mean_, (len(X_test), *self.mean_.shape)).copy()
        return inverse_average_space_grid(pred, self.field_names, self.stats, X=X_test)
