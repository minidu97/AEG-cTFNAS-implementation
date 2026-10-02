import torch
from compact_encoding import ArchitectureGenotype
from proxy_metrics import compute_proxy_scores
from unet_builder import build_unet
import numpy as np

geno = ArchitectureGenotype.initialize(n_blocks=4)
rng = np.random.default_rng(0)
enc_samples, dec_samples = geno.sample(rng)
model = build_unet(geno, enc_samples, dec_samples, in_channels=1, num_classes=4, base_channels=16)

x = torch.randn(4, 1, 224, 224)  # swap for a real ACDC batch when you have one
print(compute_proxy_scores(model, x))
