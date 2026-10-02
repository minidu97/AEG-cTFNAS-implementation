import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from search_loop import AEGcTFNASSearch, AEGcTFNASConfig


def dummy_objective(x_flat: np.ndarray) -> float:
    return float(np.sum(x_flat ** 2))


def plot_history(history, save_path="search_visualization.png"):
    iters = [h["iter"] for h in history]
    c_e = [h["C_E"] for h in history]
    c_d = [h["C_D"] for h in history]
    m = [h["M"] for h in history]
    best = [h["best_score"] for h in history]
    active = [h["active"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # (a) Encoder/decoder contribution over time
    ax = axes[0]
    ax.plot(iters, c_e, label="$C_E$ (encoder)", color="#1f77b4")
    ax.plot(iters, c_d, label="$C_D$ (decoder)", color="#d62728")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Contribution")
    ax.set_title("(a) Real-time encoder/decoder contribution (Eq. 2)")
    ax.legend(loc="upper right")

    # shade which side was "active" at each iteration
    for i, a in zip(iters, active):
        ax.axvspan(i - 0.5, i + 0.5,
                    color="#1f77b4" if a == "encoder" else "#d62728",
                    alpha=0.05)

    # (b) Adaptive M schedule
    ax = axes[1]
    ax.step(iters, m, where="mid", color="#2ca02c")
    ax.set_ylabel("Alternating interval $M$")
    ax.set_title("(b) Adaptive alternating-game interval")

    # (c) Best-score convergence
    ax = axes[2]
    ax.plot(iters, best, color="#9467bd")
    ax.set_ylabel("Best objective (lower = better)")
    ax.set_xlabel("Iteration")
    ax.set_title("(c) Best-pool score convergence")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def main():
    cfg = AEGcTFNASConfig(n_blocks=4, n_generations=50, base_M=5, seed=0)
    search = AEGcTFNASSearch(cfg, dummy_objective)
    search.run()
    plot_history(search.history)


if __name__ == "__main__":
    main()
