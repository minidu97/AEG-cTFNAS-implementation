from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _safe_nanmean(values) -> float:
    if not values or all(np.isnan(v) for v in values):
        return float("nan")
    return float(np.nanmean(values))


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int,
                    eps: float = 1e-5) -> np.ndarray:
    scores = np.zeros(num_classes)
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        intersection = np.logical_and(p, t).sum()
        denom = p.sum() + t.sum()
        scores[c] = (2.0 * intersection + eps) / (denom + eps)
    return scores


def iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int,
                   eps: float = 1e-5) -> np.ndarray:
    scores = np.zeros(num_classes)
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        scores[c] = (intersection + eps) / (union + eps)
    return scores


def compute_surface_distances(pred: np.ndarray, target: np.ndarray,
                               spacing=None):
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan"), float("nan")

    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)

    dt_target = distance_transform_edt(~target_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    surf_dist_pred_to_target = dt_target[pred_border]
    surf_dist_target_to_pred = dt_pred[target_border]

    all_dists = np.concatenate([surf_dist_pred_to_target, surf_dist_target_to_pred])
    asd = float(all_dists.mean())
    hd95 = float(np.percentile(all_dists, 95))
    return asd, hd95


def evaluate_volume(pred_vol: np.ndarray, target_vol: np.ndarray, num_classes: int,
                     spacing=None) -> dict:
    dice = dice_per_class(pred_vol, target_vol, num_classes)
    iou = iou_per_class(pred_vol, target_vol, num_classes)

    asd_list, hd95_list = [], []
    for c in range(1, num_classes):  # skip background for surface distances
        asd, hd95 = compute_surface_distances(pred_vol == c, target_vol == c, spacing)
        asd_list.append(asd)
        hd95_list.append(hd95)

    return {
        "dice_per_class": dice.tolist(),
        "iou_per_class": iou.tolist(),
        "dice_fg_mean": float(np.nanmean(dice[1:])),
        "miou_fg_mean": float(np.nanmean(iou[1:])),
        "asd_per_class": asd_list,
        "hd95_per_class": hd95_list,
        "asd_mean": _safe_nanmean(asd_list),
        "hd95_mean": _safe_nanmean(hd95_list),
    }
