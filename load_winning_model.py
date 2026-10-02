from __future__ import annotations
import json
import numpy as np

from compact_encoding import ArchitectureGenotype
from unet_builder import build_unet


def load_winning_model(architecture_json_path: str, n_blocks: int = 4,
                        in_channels: int = 1, num_classes: int = 4,
                        base_channels: int = 16):
    with open(architecture_json_path, "r") as f:
        data = json.load(f)

    x_flat = np.array(data["raw_compact_vector"], dtype=np.float64)

    geno = ArchitectureGenotype.initialize(n_blocks)
    block_len = geno.encoder_blocks[0].mu.shape[0]

    enc_samples = [x_flat[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
    offset = n_blocks * block_len
    dec_samples = [x_flat[offset + i * block_len: offset + (i + 1) * block_len]
                   for i in range(n_blocks)]

    model = build_unet(geno, enc_samples, dec_samples, in_channels=in_channels,
                        num_classes=num_classes, base_channels=base_channels)
    return model
