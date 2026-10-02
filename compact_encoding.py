

from __future__ import annotations
import numpy as np
from scipy.special import erf, erfinv
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Search-space sizing -- EDIT THESE to match your operation pools S_N/S_R/S_U
# ---------------------------------------------------------------------------
NUM_NORMAL_OPS = 4        # N_N: number of Normal Cell ops per block
NUM_REDUCTION_OPS = 2     # N_R: number of Reduction Cell ops per block (encoder)
NUM_UPSAMPLE_OPS = 2      # N_U: number of Upsample Cell ops per block (decoder)

S_N = 5   # |S_N| size of the Normal Cell operation pool
S_R = 4   # |S_R| size of the Reduction Cell operation pool
S_U = 4   # |S_U| size of the Upsample Cell operation pool


def sample_truncated_normal(mu: np.ndarray, sigma: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.clip(sigma, 1e-6, None)  # guard against sigma -> 0

    y = rng.uniform(0.0, 1.0, size=mu.shape)

    root2sig = np.sqrt(2.0 * sigma)
    a = erf((mu + 1.0) / root2sig)
    b = erf((mu - 1.0) / root2sig)

    inner = -a - y * b + y * a
    inner = np.clip(inner, -1.0 + 1e-12, 1.0 - 1e-12)  # keep erfinv finite

    x = root2sig * erfinv(inner) + mu
    return np.clip(x, -1.0, 1.0)


def map_to_operation_range(x_c: np.ndarray, dim_index: int,
                            n_normal_dims: int) -> int:
    if dim_index < n_normal_dims:
        hi = S_N * max(dim_index, 1)
    else:
        hi = S_R  # or S_U -- swap externally for decoder Upsample Cell
    # map [-1, 1] -> [0, hi]
    val = (x_c + 1.0) / 2.0 * hi
    return int(np.clip(round(val), 0, hi))


@dataclass
class BlockGenotype:
    kind: str  # "encoder" or "decoder"
    n_normal: int
    n_parallel: int  # N_R for encoder, N_U for decoder
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma: np.ndarray = field(default_factory=lambda: np.ones(0))

    def __post_init__(self):
        n_dims = self.n_normal + self.n_parallel
        if self.mu.size == 0:
            self.mu = np.zeros(n_dims)
        if self.sigma.size == 0:
            self.sigma = np.ones(n_dims) * 0.5

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return sample_truncated_normal(self.mu, self.sigma, rng)

    def decode(self, x_c: np.ndarray) -> List[int]:
        ops = []
        for i, v in enumerate(x_c):
            ops.append(map_to_operation_range(v, i, self.n_normal))
        return ops


@dataclass
class ArchitectureGenotype:
    encoder_blocks: List[BlockGenotype]
    decoder_blocks: List[BlockGenotype]

    @classmethod
    def initialize(cls, n_blocks: int) -> "ArchitectureGenotype":
        enc = [BlockGenotype("encoder", NUM_NORMAL_OPS, NUM_REDUCTION_OPS)
               for _ in range(n_blocks)]
        dec = [BlockGenotype("decoder", NUM_NORMAL_OPS, NUM_UPSAMPLE_OPS)
               for _ in range(n_blocks)]
        return cls(enc, dec)

    def sample(self, rng: np.random.Generator):
        enc_samples = [b.sample(rng) for b in self.encoder_blocks]
        dec_samples = [b.sample(rng) for b in self.decoder_blocks]
        return enc_samples, dec_samples

    def flatten_mu(self) -> np.ndarray:
        return np.concatenate(
            [b.mu for b in self.encoder_blocks] +
            [b.mu for b in self.decoder_blocks]
        )
