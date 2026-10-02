from __future__ import annotations
import argparse
import json
import os

import warnings
import numpy as np
import torch

# A class simply absent from a given volume/fold produces an all-NaN
# column when averaging across volumes -- expected, not exceptional.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from load_winning_model import load_winning_model
from metrics import evaluate_volume


CLASS_NAMES = ["background", "RV", "Myo", "LV"]  # matches ACDC convention
# used in acdc_dataset.py docstring -- double check against your actual
# preprocessing if numbers look swapped (some pipelines order RV/Myo/LV
# differently)


def pick_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def run_full_evaluation(model, test_loader, device, num_classes):
    model.eval()
    per_volume_results = []

    for batch in test_loader:
        image = batch["image"].to(device)   # [1, D, H, W]
        label = batch["label"][0].numpy()    # (D, H, W)
        case_name = batch["case_name"][0]

        volume = image[0]
        preds = []
        for d in range(volume.shape[0]):
            slice_img = volume[d].unsqueeze(0).unsqueeze(0)
            logits = model(slice_img)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            preds.append(pred)
        pred_vol = np.stack(preds, axis=0)

        result = evaluate_volume(pred_vol, label, num_classes)
        result["case_name"] = case_name
        per_volume_results.append(result)

    return per_volume_results


def summarize(per_volume_results, num_classes):
    dice_matrix = np.array([r["dice_per_class"] for r in per_volume_results])  # (N, C)
    iou_matrix = np.array([r["iou_per_class"] for r in per_volume_results])
    asd_matrix = np.array([r["asd_per_class"] for r in per_volume_results])    # (N, C-1)
    hd95_matrix = np.array([r["hd95_per_class"] for r in per_volume_results])

    mean_dice_per_class = np.nanmean(dice_matrix, axis=0)
    mean_iou_per_class = np.nanmean(iou_matrix, axis=0)
    mean_asd_per_class = np.nanmean(asd_matrix, axis=0)
    mean_hd95_per_class = np.nanmean(hd95_matrix, axis=0)

    return {
        "n_volumes": len(per_volume_results),
        "dice_per_class": {CLASS_NAMES[c]: float(mean_dice_per_class[c])
                            for c in range(num_classes)},
        "dice_fg_average": float(np.nanmean(mean_dice_per_class[1:])),
        "miou_per_class": {CLASS_NAMES[c]: float(mean_iou_per_class[c])
                            for c in range(num_classes)},
        "miou_fg_average": float(np.nanmean(mean_iou_per_class[1:])),
        "asd_per_class": {CLASS_NAMES[c + 1]: float(mean_asd_per_class[c])
                           for c in range(num_classes - 1)},
        "asd_average": float(np.nanmean(mean_asd_per_class)),
        "hd95_per_class": {CLASS_NAMES[c + 1]: float(mean_hd95_per_class[c])
                            for c in range(num_classes - 1)},
        "hd95_average": float(np.nanmean(mean_hd95_per_class)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc_root", required=True)
    parser.add_argument("--architecture_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out_dir", default="./eval_results")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    from acdc_dataset import make_dataloaders
    _, _, test_loader = make_dataloaders(args.acdc_root, batch_size=1, num_workers=0)

    model = load_winning_model(args.architecture_json, n_blocks=args.n_blocks,
                                in_channels=1, num_classes=4,
                                base_channels=args.base_channels).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    per_volume = run_full_evaluation(model, test_loader, device, num_classes=4)
    summary = summarize(per_volume, num_classes=4)

    print("\n=== Table 1-style summary (ACDC test set) ===")
    print(f"{'Class':<12}{'Dice (%)':<12}")
    for name, val in summary["dice_per_class"].items():
        print(f"{name:<12}{val*100:<12.2f}")
    print(f"{'Ave (fg)':<12}{summary['dice_fg_average']*100:<12.2f}")
    print(f"\nmIoU (fg avg): {summary['miou_fg_average']*100:.2f}%")
    print(f"ASD (avg):     {summary['asd_average']:.3f}")
    print(f"HD95 (avg):    {summary['hd95_average']:.3f}")
    print(f"\n(computed over {summary['n_volumes']} test volumes)")
    print("\nNOTE: ASD/HD95 assume isotropic unit voxel spacing unless you "
          "passed real spacing -- see metrics.py docstring. Not directly "
          "comparable to the paper's mm-based numbers without that correction.")

    with open(os.path.join(args.out_dir, "per_volume_results.json"), "w") as f:
        json.dump(per_volume, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed results to {args.out_dir}/")


if __name__ == "__main__":
    main()
