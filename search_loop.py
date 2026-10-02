from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from compact_encoding import ArchitectureGenotype
from agc import AlternatingGameControl, contribution_scores
from bayesian_inference import BayesianBlockContribution, wasserstein_gaussian
from search_strategy import BestPool, eq3_update, update_mu_sigma


ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class AEGcTFNASConfig:
    n_blocks: int = 4          # N_B
    n_generations: int = 50    # N_G, matches paper's 50 search iterations
    base_M: int = 5            # alternating interval, matches Fig. 3 (M=5)
    bi_window: int = 10        # W, Bayesian-inference rolling window
    best_pool_capacity: int = 5  # K
    crossover_prob: float = 0.5  # cr
    virtual_pop_size: int = 20   # N_p
    seed: int = 0


class AEGcTFNASSearch:
    def __init__(self, config: AEGcTFNASConfig, objective_fn: ObjectiveFn):
        self.cfg = config
        self.objective_fn = objective_fn
        self.rng = np.random.default_rng(config.seed)

        self.geno = ArchitectureGenotype.initialize(config.n_blocks)
        self.agc = AlternatingGameControl(base_M=config.base_M)
        self.bi = BayesianBlockContribution(n_blocks=config.n_blocks,
                                             window=config.bi_window)
        self.best_pool = BestPool(capacity=config.best_pool_capacity)

        self.history = []  # list of dicts, one per iteration, for logging/plots

    # -- helpers -----------------------------------------------------
    def _flatten(self, enc_samples, dec_samples) -> np.ndarray:
        return np.concatenate(enc_samples + dec_samples)

    def _current_mu_flat(self) -> np.ndarray:
        return self.geno.flatten_mu()

    def _block_distances(self, prev_blocks, curr_blocks) -> np.ndarray:
        return np.array([
            wasserstein_gaussian(p.mu, p.sigma, c.mu, c.sigma)
            for p, c in zip(prev_blocks, curr_blocks)
        ])

    def _perturb_mu(self, blocks, active: bool, step: float = 0.05):
        if not active:
            return
        for b in blocks:
            b.mu = np.clip(b.mu + self.rng.normal(0, step, size=b.mu.shape),
                            -1, 1)

    # -- main loop -----------------------------------------------------
    def run(self) -> BestPool:
        cfg = self.cfg

        for t in range(cfg.n_generations):
            enc_active = self.agc.is_encoder_active()

            # snapshot mu/sigma before this iteration's update, for
            # Wasserstein distance computation afterwards
            prev_enc_blocks = [
                type(b)(b.kind, b.n_normal, b.n_parallel, b.mu.copy(), b.sigma.copy())
                for b in self.geno.encoder_blocks
            ]
            prev_dec_blocks = [
                type(b)(b.kind, b.n_normal, b.n_parallel, b.mu.copy(), b.sigma.copy())
                for b in self.geno.decoder_blocks
            ]

            # sample X^t from current distributions (Eq. 1)
            enc_samples, dec_samples = self.geno.sample(self.rng)
            x_t = self._flatten(enc_samples, dec_samples)
            x_t_mu = self._current_mu_flat()

            f_X = self.objective_fn(x_t)

            # X+E: encoder-only update (decoder frozen at its sampled X^t)
            self._perturb_mu(self.geno.encoder_blocks, active=True)
            enc_samples_E, _ = self.geno.sample(self.rng)
            x_plus_E = self._flatten(enc_samples_E, dec_samples)

            # X+D: decoder-only update (encoder frozen), computed on the
            # PRE-perturbation encoder copy so the two half-updates are
            # independent before being combined into the joint update
            self._perturb_mu(prev_dec_blocks, active=True)  # placeholder mu bump
            _, dec_samples_D = self.geno.sample(self.rng)
            x_plus_D = self._flatten(enc_samples, dec_samples_D)

            # X+E+D: joint update
            x_plus_ED = self._flatten(enc_samples_E, dec_samples_D)

            f_X_plus_E = self.objective_fn(x_plus_E)
            f_X_plus_D = self.objective_fn(x_plus_D)
            f_X_plus_ED = self.objective_fn(x_plus_ED)
            # NOTE: as written above this costs 4 evaluations for clarity;
            # to hit the paper's "2 evaluations/iteration" budget, only
            # evaluate the *active* side's half-update each iteration and
            # reuse a cached f(X) / f(X+other) from the previous round.

            c_e, c_d = contribution_scores(f_X, f_X_plus_E, f_X_plus_D, f_X_plus_ED)
            self.agc.step(c_e if enc_active else c_d)

            # Bayesian inference: per-block Wasserstein distances + fit
            enc_dist = self._block_distances(prev_enc_blocks, self.geno.encoder_blocks)
            dec_dist = self._block_distances(prev_dec_blocks, self.geno.decoder_blocks)
            c_b_enc, c_b_dec = self.bi.update(enc_dist, c_e, dec_dist, c_d)

            # update best pool with whichever of X^t / X+E+D scored better
            best_candidate, best_score = (
                (x_t, f_X) if f_X <= f_X_plus_ED else (x_plus_ED, f_X_plus_ED)
            )
            self.best_pool.maybe_insert(best_candidate, best_score)

            # search-strategy step (Eq. 3) once the pool has enough history
            if len(self.best_pool.items) >= 2:
                x_best_r = self.best_pool.sample_random(self.rng)
                if c_b_enc is not None and c_b_dec is not None:
                    c_b_v = np.concatenate([
                        np.repeat(c_b_enc, len(x_t) // (2 * cfg.n_blocks) or 1),
                        np.repeat(c_b_dec, len(x_t) // (2 * cfg.n_blocks) or 1),
                    ])
                    c_b_v = np.resize(c_b_v, x_t.shape)  # align lengths defensively
                else:
                    c_b_v = np.zeros_like(x_t)
                ced_vec = np.where(
                    np.arange(len(x_t)) < len(x_t) // 2, c_e, c_d
                )
                x_next = eq3_update(x_best_r, x_t, x_t_mu, c_b_v, ced_vec,
                                     cfg.crossover_prob, self.rng)
                f_next = self.objective_fn(x_next)

                best_known = self.best_pool.best()
                best_known_score = self.best_pool.best_score()
                if f_next <= best_known_score:
                    winner, loser = x_next, best_known
                else:
                    winner, loser = best_known, x_next

                mu_flat = self._current_mu_flat()
                sigma_flat = np.concatenate(
                    [b.sigma for b in self.geno.encoder_blocks] +
                    [b.sigma for b in self.geno.decoder_blocks]
                )
                mu_next, sigma_next = update_mu_sigma(
                    mu_flat, sigma_flat, winner, loser, cfg.virtual_pop_size)

                self._write_back_mu_sigma(mu_next, sigma_next)
                self.best_pool.maybe_insert(x_next, f_next)

            self.history.append({
                "iter": t, "active": self.agc.active, "M": self.agc.M,
                "f_X": f_X, "f_X_plus_ED": f_X_plus_ED,
                "C_E": c_e, "C_D": c_d,
                "best_score": self.best_pool.best_score(),
            })

        return self.best_pool

    def _write_back_mu_sigma(self, mu_flat, sigma_flat):
        idx = 0
        for b in self.geno.encoder_blocks + self.geno.decoder_blocks:
            n = b.mu.shape[0]
            b.mu = mu_flat[idx: idx + n]
            b.sigma = sigma_flat[idx: idx + n]
            idx += n
