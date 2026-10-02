import torch
from acdc_dataset import make_dataloaders
from compact_encoding import ArchitectureGenotype
from unet_builder import build_unet
from proxy_metrics import compute_proxy_scores
import numpy as np

if __name__ == "__main__":
    train_loader, _, _ = make_dataloaders(
        "/Users/miniduwickramaarachchi/Documents/PHD Research/New Publication work/AEG-cTFNAS/ACDC_preprocessed",
        batch_size=4,
        num_workers=0
    )
    batch = next(iter(train_loader))
    x = batch["image"]  # real ACDC images

    geno = ArchitectureGenotype.initialize(n_blocks=4)
    rng = np.random.default_rng(0)
    enc_samples, dec_samples = geno.sample(rng)
    model = build_unet(geno, enc_samples, dec_samples, in_channels=1, num_classes=4, base_channels=16)

    print(compute_proxy_scores(model, x))
