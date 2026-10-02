from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def contribution_scores(f_X: float, f_X_plus_E: float, f_X_plus_D: float,
                         f_X_plus_ED: float, eps: float = 1e-12):

    denom = f_X - f_X_plus_ED
    s = sign(denom)
    if abs(denom) < eps:
        # No measurable joint change -> contributions are undefined; treat
        # as neutral (0) rather than dividing by ~0.
        return 0.0, 0.0

    c_e = s * (f_X - f_X_plus_E) / denom
    c_d = s * (f_X - f_X_plus_D) / denom
    # Eq. 2 is a ratio of differences; when the joint change (denom) is
    # small relative to the individual half-changes, the ratio can blow
    # up numerically even though it's not a ~0 denominator per se. Clip
    # to a sane range so a rare outlier doesn't dominate downstream
    # plots/logistic fits -- tune this bound to your real objective's
    # typical scale.
    c_e = float(np.clip(c_e, -3.0, 3.0))
    c_d = float(np.clip(c_d, -3.0, 3.0))
    return c_e, c_d


@dataclass
class AlternatingGameControl:

    base_M: int
    M: int = None
    active: str = "encoder"       # "encoder" or "decoder"
    iters_since_switch: int = 0
    consecutive_sign: int = 0     # + streak of same-signed contributions
    last_sign: int = 0

    def __post_init__(self):
        if self.M is None:
            self.M = self.base_M

    def step(self, c_active: float):

        s = sign(c_active)
        if s != 0 and s == self.last_sign:
            self.consecutive_sign += 1
        else:
            self.consecutive_sign = 1
        self.last_sign = s

        if self.consecutive_sign >= 2:
            if s > 0:
                self.M = min(self.M + 1, math.ceil(1.5 * self.base_M))
            elif s < 0:
                self.M = max(self.M - 1, math.floor(0.5 * self.base_M))
            self.consecutive_sign = 0  # reset streak after adjusting

        self.iters_since_switch += 1
        if self.iters_since_switch >= self.M:
            self.active = "decoder" if self.active == "encoder" else "encoder"
            self.iters_since_switch = 0

    def is_encoder_active(self) -> bool:
        return self.active == "encoder"
