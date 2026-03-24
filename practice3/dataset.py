import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


COVIDQU_HANDLE = "anasmohammedtahir/covidqu"


def _list_existing_dirs(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def resolve_covidqu_root(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"COVID-QU-Ex root not found: {p}")
        return str(p)

    env = os.getenv("COVIDQU_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return str(p)

    candidates = _list_existing_dirs(
        [
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "anasmohammedtahir"
            / "covidqu"
            / "versions",
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "anasmohammedtahir"
            / "covidqu",
        ]
    )
    for c in candidates:
        if c.name == "versions":
            versions = sorted([p for p in c.iterdir() if p.is_dir()], key=lambda p: p.name)
            if versions:
                return str(versions[-1])
        else:
            return str(c)

    try:
        import kagglehub

        return kagglehub.dataset_download(COVIDQU_HANDLE)
    except Exception as e:
        raise RuntimeError(
            "Could not resolve COVID-QU-Ex dataset path. "
            "Set env COVIDQU_PATH to the extracted dataset root, or download it via kagglehub."
        ) from e


def _read_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _read_mask01(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(str(path))
    return (m > 0).astype(np.float32)


def _resize01(img01: np.ndarray, target_size: tuple[int, int], is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    out = cv2.resize(img01, target_size, interpolation=interp)
    if is_mask:
        out = (out > 0.5).astype(np.float32)
    return out.astype(np.float32)


class CovidQuSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        task: str,
        split: str,
        classes: Optional[list[str]] = None,
        target_size: tuple[int, int] = (256, 256),
        include_lung_roi_for_infection: bool = True,
        transform: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        if task not in {"lung", "infection"}:
            raise ValueError("task must be 'lung' or 'infection'")
        if split not in {"Train", "Val", "Test"}:
            raise ValueError("split must be one of Train/Val/Test")

        self.root = Path(root)
        self.task = task
        self.split = split
        self.classes = classes or ["COVID-19", "Non-COVID", "Normal"]
        self.target_size = target_size
        self.include_lung_roi_for_infection = include_lung_roi_for_infection
        self.transform = transform

        if self.task == "lung":
            base = self.root / "Lung Segmentation Data" / "Lung Segmentation Data"
        else:
            base = self.root / "Infection Segmentation Data" / "Infection Segmentation Data"
        self.base = base

        samples: list[tuple[Path, Path, Optional[Path], dict]] = []
        for cls in self.classes:
            cls_dir = base / split / cls
            img_dir = cls_dir / "images"
            if not img_dir.exists():
                continue
            img_paths = sorted(img_dir.glob("*.png"))
            if self.task == "lung":
                mask_dir = cls_dir / "lung masks"
                for ip in img_paths:
                    mp = mask_dir / ip.name
                    if mp.exists():
                        samples.append((ip, mp, None, dict(split=split, cls=cls, filename=ip.name)))
            else:
                mask_dir = cls_dir / "infection masks"
                lung_dir = cls_dir / "lung masks"
                for ip in img_paths:
                    mp = mask_dir / ip.name
                    if not mp.exists():
                        continue
                    lp = (lung_dir / ip.name) if self.include_lung_roi_for_infection else None
                    if lp is not None and not lp.exists():
                        lp = None
                    samples.append((ip, mp, lp, dict(split=split, cls=cls, filename=ip.name)))

        if not samples:
            raise RuntimeError(
                f"No samples found for task={task} split={split} classes={self.classes}. "
                f"Checked under: {self.base}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, lung_path, meta = self.samples[idx]
        img01 = _read_gray01(img_path)
        mask01 = _read_mask01(mask_path)

        img01 = _resize01(img01, self.target_size, is_mask=False)
        mask01 = _resize01(mask01, self.target_size, is_mask=True)

        img_t = torch.from_numpy(img01).unsqueeze(0)
        mask_t = torch.from_numpy(mask01).unsqueeze(0)

        lung_roi_t = torch.zeros_like(mask_t)
        if self.task == "infection" and lung_path is not None:
            lung01 = _read_mask01(lung_path)
            lung01 = _resize01(lung01, self.target_size, is_mask=True)
            lung_roi_t = torch.from_numpy(lung01).unsqueeze(0)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)

        return img_t, mask_t, lung_roi_t, meta


def get_dataloaders(
    task: str,
    batch_size: int = 8,
    target_size: tuple[int, int] = (256, 256),
    num_workers: int = 0,
    covidqu_root: Optional[str] = None,
    include_lung_roi_for_infection: bool = True,
):
    root = resolve_covidqu_root(covidqu_root)
    train_ds = CovidQuSegmentationDataset(
        root=root,
        task=task,
        split="Train",
        target_size=target_size,
        include_lung_roi_for_infection=include_lung_roi_for_infection,
    )
    val_ds = CovidQuSegmentationDataset(
        root=root,
        task=task,
        split="Val",
        target_size=target_size,
        include_lung_roi_for_infection=include_lung_roi_for_infection,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl

