import argparse
import numpy as np
import torch

from compact_encoding import ArchitectureGenotype
from unet_builder import build_unet


def run_random_batch_test(n_trials: int = 20, n_blocks: int = 4,
                           batch_size: int = 2, img_size: int = 224):
    rng = np.random.default_rng(0)
    x = torch.randn(batch_size, 1, img_size, img_size)

    failures = []
    for trial in range(n_trials):
        geno = ArchitectureGenotype.initialize(n_blocks)
        # randomize mu a bit so different trials decode to different ops
        for b in geno.encoder_blocks + geno.decoder_blocks:
            b.mu = rng.uniform(-1, 1, size=b.mu.shape)
        enc_samples, dec_samples = geno.sample(rng)

        try:
            model = build_unet(geno, enc_samples, dec_samples,
                                in_channels=1, num_classes=4, base_channels=16)
            with torch.no_grad():
                out = model(x)
            expected = (batch_size, 4, img_size, img_size)
            if tuple(out.shape) != expected:
                failures.append((trial, f"shape mismatch: got {tuple(out.shape)}, "
                                         f"expected {expected}"))
            else:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"trial {trial:2d}: OK  output={tuple(out.shape)}  "
                      f"params={n_params/1e6:.2f}M")
        except Exception as e:
            failures.append((trial, f"{type(e).__name__}: {e}"))

    print()
    if failures:
        print(f"{len(failures)}/{n_trials} trials FAILED:")
        for trial, msg in failures:
            print(f"  trial {trial}: {msg}")
    else:
        print(f"All {n_trials} random genotypes built and forward-passed correctly.")
    return len(failures) == 0


def run_real_acdc_test(root: str, n_trials: int = 5, n_blocks: int = 4):
    from acdc_dataset import make_dataloaders

    train_loader, _, _ = make_dataloaders(root, batch_size=2, num_workers=0)
    batch = next(iter(train_loader))
    x = batch["image"]
    print(f"Loaded a real ACDC batch: {x.shape}")

    rng = np.random.default_rng(1)
    for trial in range(n_trials):
        geno = ArchitectureGenotype.initialize(n_blocks)
        for b in geno.encoder_blocks + geno.decoder_blocks:
            b.mu = rng.uniform(-1, 1, size=b.mu.shape)
        enc_samples, dec_samples = geno.sample(rng)

        model = build_unet(geno, enc_samples, dec_samples,
                            in_channels=1, num_classes=4, base_channels=16)
        with torch.no_grad():
            out = model(x)
        print(f"trial {trial}: input={tuple(x.shape)} -> output={tuple(out.shape)}")
        assert out.shape[0] == x.shape[0] and out.shape[1] == 4
        assert out.shape[2:] == x.shape[2:]
    print("Real ACDC batch: all trials passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc", type=str, default=None,
                         help="Path to ACDC_preprocessed root, to test on real data")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    ok = run_random_batch_test(n_trials=args.n_trials)

    if args.acdc:
        run_real_acdc_test(args.acdc, n_trials=5)

    if not ok:
        raise SystemExit(1)
