
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


def wasserstein_gaussian(mu1: np.ndarray, sigma1: np.ndarray,
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
    sq = (mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2
    return float(np.sqrt(np.sum(sq)))


class _RollingDataset:

    def __init__(self, window: int, n_blocks: int):
        self.window = window
        self.n_blocks = n_blocks
        self.X: List[np.ndarray] = []
        self.y: List[int] = []

    def add(self, distances: np.ndarray, contribution: float):
        self.X.append(distances.copy())
        self.y.append(1 if contribution > 0 else 0)
        if len(self.X) > self.window:
            self.X.pop(0)
            self.y.pop(0)

    def ready(self, min_samples: int = 4) -> bool:
        # need both classes present and a minimal sample count for a
        # stable logistic fit
        return len(self.y) >= min_samples and len(set(self.y)) > 1

    def arrays(self):
        return np.stack(self.X, axis=0), np.array(self.y)


@dataclass
class BayesianBlockContribution:
    n_blocks: int
    window: int = 10
    prior_variance: float = 4.0
    encoder_data: _RollingDataset = field(init=False)
    decoder_data: _RollingDataset = field(init=False)
    last_encoder_contrib: Optional[np.ndarray] = None
    last_decoder_contrib: Optional[np.ndarray] = None

    def __post_init__(self):
        self.encoder_data = _RollingDataset(self.window, self.n_blocks)
        self.decoder_data = _RollingDataset(self.window, self.n_blocks)

    def update(self, encoder_distances: np.ndarray, c_e: float,
               decoder_distances: np.ndarray, c_d: float):
        self.encoder_data.add(encoder_distances, c_e)
        self.decoder_data.add(decoder_distances, c_d)

        if self.encoder_data.ready():
            self.last_encoder_contrib = self._fit_map_logistic(self.encoder_data)
        if self.decoder_data.ready():
            self.last_decoder_contrib = self._fit_map_logistic(self.decoder_data)

        return self.last_encoder_contrib, self.last_decoder_contrib

    def _fit_map_logistic(self, dataset: _RollingDataset) -> np.ndarray:
        X, y = dataset.arrays()
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])  # add intercept
        beta = np.zeros(Xb.shape[1])
        lam = 1.0 / self.prior_variance  # L2 penalty from Gaussian prior

        for _ in range(50):
            z = Xb @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            W = p * (1 - p)
            grad = Xb.T @ (y - p) - lam * beta
            # Hessian of the negative log-posterior
            H = Xb.T @ (Xb * W[:, None]) + lam * np.eye(Xb.shape[1])
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            beta = beta + step
            if np.linalg.norm(step) < 1e-6:
                break

        # drop the intercept term; per-block contribution = beta_i
        return beta[:-1]
