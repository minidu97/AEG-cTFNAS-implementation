from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.optim import AdamW

from load_winning_model import load_winning_model
from losses import DiceCrossEntropyLoss
from metrics import evaluate_volume


def pick_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def validate(model, val_loader, device, num_classes):
    model.eval()
    dice_scores = []
    for batch in val_loader:
        image = batch["image"].to(device)       # [1, D, H, W]
        label = batch["label"][0].numpy()        # (D, H, W)

        volume = image[0]                         # (D, H, W)
        preds = []
        for d in range(volume.shape[0]):
            slice_img = volume[d].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            logits = model(slice_img)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            preds.append(pred)
        pred_vol = np.stack(preds, axis=0)

        result = evaluate_volume(pred_vol, label, num_classes)
        dice_scores.append(result["dice_fg_mean"])

    model.train()
    return float(np.nanmean(dice_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc_root", required=True)
    parser.add_argument("--architecture_json", required=True)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_every", type=int, default=10,
                         help="Run validation every N epochs (volume-level "
                              "eval is slow; don't do it every epoch)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out_dir", default="./train_results")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    from acdc_dataset import make_dataloaders
    train_loader, val_loader, _ = make_dataloaders(
        args.acdc_root, batch_size=args.batch_size, num_workers=0)

    model = load_winning_model(args.architecture_json, n_blocks=args.n_blocks,
                                in_channels=1, num_classes=4,
                                base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded winning architecture: {n_params/1e6:.2f}M params")

    criterion = DiceCrossEntropyLoss(num_classes=4)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    history = []
    best_val_dice = -1.0
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "train_loss": mean_loss, "epoch_time_s": epoch_time}

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_dice = validate(model, val_loader, device, num_classes=4)
            row["val_dice_fg_mean"] = val_dice
            print(f"epoch {epoch:4d}  loss={mean_loss:.4f}  "
                  f"val_dice={val_dice:.4f}  ({epoch_time:.1f}s)")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  -> new best, saved to {best_ckpt_path}")
        else:
            print(f"epoch {epoch:4d}  loss={mean_loss:.4f}  ({epoch_time:.1f}s)")

        history.append(row)

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # always save a final checkpoint too, in case validation never
    # triggered (e.g. very short test runs)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
    print(f"\nTraining done. Best val Dice: {best_val_dice:.4f}")
    print(f"Saved: {args.out_dir}/best_model.pt, final_model.pt, training_history.json")


if __name__ == "__main__":
    main()
