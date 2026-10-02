from __future__ import annotations
import os
import re
import random
from typing import Optional, Callable, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate


NUM_CLASSES = 4  # background + RV + Myo + LV


# ---------------------------------------------------------------------------
# Filename deduping -- handles Kaggle's "(1)" duplicate-download artifacts
# ---------------------------------------------------------------------------
_DUP_SUFFIX_RE = re.compile(r"\(\d+\)(?=\.h5$)")


def _list_h5_files_deduped(directory: str) -> List[str]:
    all_files = [f for f in os.listdir(directory) if f.endswith(".h5")]
    best_for_base = {}
    for f in all_files:
        base = _DUP_SUFFIX_RE.sub("", f)
        if base not in best_for_base or "(" not in f:
            best_for_base[base] = f
    return sorted(best_for_base.values())


# ---------------------------------------------------------------------------
# Augmentation (train only)
# ---------------------------------------------------------------------------
def _random_rot_flip(image: np.ndarray, label: np.ndarray):
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = random.randint(0, 1)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def _random_rotate(image: np.ndarray, label: np.ndarray, max_angle: int = 20):
    angle = np.random.uniform(-max_angle, max_angle)
    image = rotate(image, angle, order=1, reshape=False, mode="constant", cval=0)
    label = rotate(label, angle, order=0, reshape=False, mode="constant", cval=0)
    return image, label


class RandomGenerator:
    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = _random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = _random_rotate(image, label)

        h, w = image.shape
        if (h, w) != (self.output_size, self.output_size):
            zh = self.output_size / h
            zw = self.output_size / w
            image = zoom(image, (zh, zw), order=3)
            label = zoom(label, (zh, zw), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        return {"image": image, "label": label}


class CenterResize:
    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        h, w = image.shape[-2:]
        if (h, w) != (self.output_size, self.output_size):
            zh = self.output_size / h
            zw = self.output_size / w
            if image.ndim == 2:
                image = zoom(image, (zh, zw), order=3)
                label = zoom(label, (zh, zw), order=0)
            else:  # (D, H, W) volume, resize each slice
                image = zoom(image, (1, zh, zw), order=3)
                label = zoom(label, (1, zh, zw), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        if image.ndim == 2:
            image = image.unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        return {"image": image, "label": label}


class ACDCDataset(Dataset):
    _FOLDER = {
        "train": "ACDC_training_slices",
        "val": "ACDC_training_volumes",
        "test": "ACDC_testing_volumes",
    }

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        assert split in self._FOLDER, f"split must be one of {list(self._FOLDER)}"
        self.root = root
        self.split = split
        self.data_dir = os.path.join(root, self._FOLDER[split])
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Expected folder not found: {self.data_dir}\n"
                f"Check that `root` points at the folder containing "
                f"ACDC_training_slices / ACDC_training_volumes / ACDC_testing_volumes."
            )

        self.filenames = _list_h5_files_deduped(self.data_dir)
        if len(self.filenames) == 0:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")

        self.transform = transform if transform is not None else (
            RandomGenerator(224) if split == "train" else CenterResize(224)
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.data_dir, fname)
        with h5py.File(path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]

        sample = {"image": image, "label": label}
        sample = self.transform(sample)
        sample["case_name"] = fname.replace(".h5", "")
        return sample


def make_dataloaders(root: str, batch_size: int = 12, num_workers: int = 4,
                      include_val: bool = True):
    from torch.utils.data import DataLoader

    train_ds = ACDCDataset(root, split="train")
    test_ds = ACDCDataset(root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    # volumes vary in depth -> keep batch_size=1, evaluate per-volume
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    if include_val:
        val_ds = ACDCDataset(root, split="val")
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader

    return train_loader, test_loader
