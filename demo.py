import numpy as np
from search_loop import AEGcTFNASSearch, AEGcTFNASConfig


def dummy_objective(x_flat: np.ndarray) -> float:
    # Minimum at x = 0 vector; stand-in for a real training-free proxy.
    return float(np.sum(x_flat ** 2))


def main():
    cfg = AEGcTFNASConfig(n_blocks=4, n_generations=50, base_M=5, seed=0)
    search = AEGcTFNASSearch(cfg, dummy_objective)
    best_pool = search.run()

    print(f"Final best score: {best_pool.best_score():.4f}")
    print(f"Best pool size: {len(best_pool.items)}")
    print("\nLast 5 iterations:")
    for row in search.history[-5:]:
        print(f"  iter={row['iter']:>2} active={row['active']:<7} "
              f"M={row['M']} C_E={row['C_E']:.3f} C_D={row['C_D']:.3f} "
              f"best={row['best_score']:.4f}")


if __name__ == "__main__":
    main()
