from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class BestPool:
    capacity: int
    items: List[np.ndarray] = field(default_factory=list)  # each: flat vector
    scores: List[float] = field(default_factory=list)       # lower = better

    def maybe_insert(self, x_flat: np.ndarray, score: float):
        self.items.append(x_flat)
        self.scores.append(score)
        if len(self.items) > self.capacity:
            # drop the worst
            worst = int(np.argmax(self.scores))
            self.items.pop(worst)
            self.scores.pop(worst)

    def sample_random(self, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self.items))
        return self.items[idx]

    def best(self) -> np.ndarray:
        idx = int(np.argmin(self.scores))
        return self.items[idx]

    def best_score(self) -> float:
        return min(self.scores) if self.scores else float("inf")

    def average(self) -> np.ndarray:
        return np.mean(np.stack(self.items, axis=0), axis=0)


def eq3_update(x_best_r: np.ndarray, x_t: np.ndarray, x_t_mu: np.ndarray,
               contribution_vec: np.ndarray, ced_vec: np.ndarray,
               crossover_prob: float, rng: np.random.Generator) -> np.ndarray:
    n = x_best_r.shape[0]
    v_cr = (rng.uniform(0, 1, size=n) < crossover_prob).astype(np.float64)

    x_next = (x_best_r
              + v_cr * contribution_vec * (x_t - x_t_mu)
              + ced_vec * (x_best_r - x_t))
    return np.clip(x_next, -1.0, 1.0)


def update_mu_sigma(mu: np.ndarray, sigma: np.ndarray,
                     winner: np.ndarray, loser: np.ndarray,
                     n_p: int) -> tuple[np.ndarray, np.ndarray]:
    mu_next = mu + (winner - loser) / n_p
    inner = (sigma ** 2 + mu ** 2 - mu_next ** 2
             + (winner ** 2 - loser ** 2) / n_p)
    inner = np.clip(inner, 1e-8, None)  # keep sigma^2 valid
    sigma_next = np.sqrt(inner)
    return mu_next, sigma_next
