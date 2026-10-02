import torch
import numpy as np
from acdc_dataset import make_dataloaders
from compact_encoding import ArchitectureGenotype
from unet_builder import build_unet
from proxy_metrics import compute_jacob_cov

if __name__ == "__main__":
    train_loader, _, _ = make_dataloaders(
        "/Users/miniduwickramaarachchi/Documents/PHD Research/New Publication work/AEG-cTFNAS/ACDC_preprocessed",
        batch_size=8, num_workers=0
    )
    batch = next(iter(train_loader))["image"]  # SAME batch used for every trial

    rng = np.random.default_rng(42)
    print(f"{'trial':<6}{'jacob_cov':<14}")
    for trial in range(6):
        geno = ArchitectureGenotype.initialize(n_blocks=4)
        for b in geno.encoder_blocks + geno.decoder_blocks:
            b.mu = rng.uniform(-1, 1, size=b.mu.shape)
        enc_samples, dec_samples = geno.sample(rng)
        model = build_unet(geno, enc_samples, dec_samples, in_channels=1, num_classes=4, base_channels=16)
        score = compute_jacob_cov(model, batch)
        print(f"{trial:<6}{score:<14.4f}")
