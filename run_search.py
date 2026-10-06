from __future__ import annotations
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from compact_encoding import ArchitectureGenotype
from proxy_metrics import make_objective_fn
from search_loop import AEGcTFNASSearch, AEGcTFNASConfig
from genotype_decode import decode_encoder_block, decode_decoder_block


def pick_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_winning_genotype(x_flat: np.ndarray, geno: ArchitectureGenotype) -> dict:
    n_blocks = len(geno.encoder_blocks)
    block_len = geno.encoder_blocks[0].mu.shape[0]

    enc_samples = [x_flat[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
    offset = n_blocks * block_len
    dec_samples = [x_flat[offset + i * block_len: offset + (i + 1) * block_len]
                   for i in range(n_blocks)]

    encoder_cfgs = []
    for i, block_geno in enumerate(geno.encoder_blocks):
        cfg = decode_encoder_block(enc_samples[i], block_geno.n_normal, block_geno.n_parallel)
        encoder_cfgs.append(dataclasses.asdict(cfg))

    decoder_cfgs = []
    for i, block_geno in enumerate(geno.decoder_blocks):
        cfg = decode_decoder_block(dec_samples[i], block_geno.n_normal, block_geno.n_parallel)
        decoder_cfgs.append(dataclasses.asdict(cfg))

    return {
        "encoder_blocks": encoder_cfgs,
        "decoder_blocks": decoder_cfgs,
        "raw_compact_vector": x_flat.tolist(),  # keep the raw form too, for re-building exactly
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc_root", required=True)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--n_generations", type=int, default=50)
    parser.add_argument("--base_M", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Sample batch size used for scoring (jacob_cov "
                              "needs >1 sample; kept fixed across the whole search)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./search_results")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    from acdc_dataset import make_dataloaders
    train_loader, _, _ = make_dataloaders(args.acdc_root, batch_size=args.batch_size,
                                           num_workers=0)
    sample_batch = next(iter(train_loader))["image"]
    print(f"Sample batch for scoring: {tuple(sample_batch.shape)}")

    geno_template = ArchitectureGenotype.initialize(args.n_blocks)
    objective_fn = make_objective_fn(
        unet_kwargs=dict(in_channels=1, num_classes=4, base_channels=args.base_channels),
        sample_batch=sample_batch,
        geno=geno_template,
        device=device,
    )

    cfg = AEGcTFNASConfig(n_blocks=args.n_blocks, n_generations=args.n_generations,
                           base_M=args.base_M, seed=args.seed)
    search = AEGcTFNASSearch(cfg, objective_fn)

    print(f"Starting search: {args.n_generations} generations, "
          f"{args.n_blocks} blocks, base_channels={args.base_channels}")
    print("(each generation builds + scores up to 4 architectures -- "
          "progress prints below after each one)", flush=True)
    t0 = time.time()
    best_pool = search.run(progress_callback=lambda i, n, row: print(
        f"  iter {i+1}/{n}  active={row['active']:<7} f_X={row['f_X']:.3f}  "
        f"best={row['best_score']:.3f}  ({time.time()-t0:.1f}s elapsed)",
        flush=True))
    elapsed = time.time() - t0
    print(f"\nSearch finished in {elapsed:.1f}s "
          f"({elapsed/args.n_generations:.1f}s/iteration)")
    print(f"Final best score: {best_pool.best_score():.4f}")

    # -- save winning architecture, decoded to human-readable form --
    best_x = best_pool.best()
    decoded = decode_winning_genotype(best_x, geno_template)
    decoded["best_score"] = best_pool.best_score()
    decoded["config"] = dataclasses.asdict(cfg)
    with open(os.path.join(args.out_dir, "best_architecture.json"), "w") as f:
        json.dump(decoded, f, indent=2)
    print(f"Saved winning architecture to {args.out_dir}/best_architecture.json")

    # -- save history --
    with open(os.path.join(args.out_dir, "search_history.json"), "w") as f:
        json.dump(search.history, f, indent=2)
    print(f"Saved search history to {args.out_dir}/search_history.json")

    # -- plot --
    try:
        from visualize import plot_history
        plot_history(search.history,
                     save_path=os.path.join(args.out_dir, "search_visualization.png"))
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
