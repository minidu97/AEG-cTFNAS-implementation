import argparse
import time
import numpy as np
import torch

from compact_encoding import ArchitectureGenotype
from unet_builder import build_unet
from proxy_metrics import compute_flops, compute_jacob_cov, compute_synflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_channels", type=int, default=16)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    rng = np.random.default_rng(0)
    geno = ArchitectureGenotype.initialize(n_blocks=4)
    enc_samples, dec_samples = geno.sample(rng)

    if args.acdc_root:
        from acdc_dataset import make_dataloaders
        train_loader, _, _ = make_dataloaders(args.acdc_root, batch_size=8, num_workers=0)
        x = next(iter(train_loader))["image"]
    else:
        x = torch.randn(8, 1, 224, 224)

    print("\n--- Timing breakdown (CPU-first pattern: matches the actual "
          "search path in proxy_metrics.make_objective_fn) ---")

    x_dev = x.to(device)

    def time_one_eval(geno_i, enc_i, dec_i, label):
        t0 = time.time()
        model = build_unet(geno_i, enc_i, dec_i, in_channels=1,
                            num_classes=4, base_channels=args.base_channels)  # stays CPU
        print(f"  [{label}] build (CPU): {time.time()-t0:.2f}s")

        t0 = time.time()
        flops = compute_flops(model, (1, 224, 224))
        print(f"  [{label}] compute_flops (CPU): {time.time()-t0:.2f}s  (={flops/1e9:.2f}G)")

        t0 = time.time()
        synflow = compute_synflow(model, (1, 224, 224))
        print(f"  [{label}] compute_synflow (CPU): {time.time()-t0:.2f}s  (={synflow:.3e})")

        t0 = time.time()
        model = model.to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"  [{label}] move to {device}: {time.time()-t0:.2f}s")

        t0 = time.time()
        jacob = compute_jacob_cov(model, x_dev)
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"  [{label}] compute_jacob_cov (on {device}): {time.time()-t0:.2f}s  (={jacob:.3f})")

    print("\nFirst call (includes any one-time CUDA/cuDNN warmup):")
    time_one_eval(geno, enc_samples, dec_samples, "1st")

    print("\nSecond call (steady-state, different architecture):")
    rng2 = np.random.default_rng(1)
    geno2 = ArchitectureGenotype.initialize(n_blocks=4)
    enc_samples2, dec_samples2 = geno2.sample(rng2)
    time_one_eval(geno2, enc_samples2, dec_samples2, "2nd")

    print("\nThird call (steady-state, different architecture again):")
    rng3 = np.random.default_rng(2)
    geno3 = ArchitectureGenotype.initialize(n_blocks=4)
    enc_samples3, dec_samples3 = geno3.sample(rng3)
    time_one_eval(geno3, enc_samples3, dec_samples3, "3rd")


if __name__ == "__main__":
    main()
