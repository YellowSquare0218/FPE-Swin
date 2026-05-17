# -*- coding: utf-8 -*-
"""
FPE-Swin / FSD_Swin training script.

This script implements the paper model exactly as:
    F_freq      = rFFT2(F_spatial)
    F_weighted  = F_freq * W_complex
    Delta_F     = irFFT2(F_weighted)
    F_residual  = F_spatial + Delta_F
    z_s         = GAP(F_spatial)
    z_e         = GAP(F_residual)
    z           = z_s + alpha * (z_e - z_s)

Notes:
1. Class name `FSD_Swin` is kept for compatibility with your original code.
2. Alias `FPE_Swin = FSD_Swin` is provided so the code name also matches the paper.
3. Update Data_dir and Save_dir before running.
"""

import os
import gc
import csv
import heapq
import random
import datetime
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2


warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
torch.backends.cudnn.benchmark = True


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


BRACS_7_TO_3 = {
    0: 0,  # 0_N   -> Benign
    1: 0,  # 1_PB  -> Benign
    2: 0,  # 2_UDH -> Benign
    3: 1,  # 3_FEA -> Atypia
    4: 1,  # 4_ADH -> Atypia
    5: 2,  # 5_DCIS -> Malignant
    6: 2,  # 6_IC   -> Malignant
}

BRACS_3_CLASS_NAMES = ["Benign", "Atypia", "Malignant"]
BRACS_7_CLASS_NAMES = ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

MEAN = {
    "40X": [0.8036, 0.6521, 0.7734],
    "100X": [0.7959, 0.6359, 0.7720],
    "200X": [0.7889, 0.6230, 0.7693],
    "400X": [0.7559, 0.5897, 0.7432],
    "Bracs": [0.7265, 0.5587, 0.6954],
}
STD = {
    "40X": [0.1022, 0.1447, 0.1000],
    "100X": [0.1199, 0.1724, 0.1084],
    "200X": [0.1228, 0.1752, 0.1034],
    "400X": [0.1408, 0.1995, 0.1139],
    "Bracs": [0.1947, 0.2282, 0.1714],
}

# TODO: Update these paths before running.
Data_dir = {
    "40X": r"/mnt/disk1/lyf/BreakHis/BreaKHis_40X/BreaKHis_40X_Cleaned_Strict",
    "100X": r"/mnt/disk1/lyf/BreakHis/BreaKHis_100X/BreaKHis_100X_Cleaned_Strict",
    "200X": r"/mnt/disk1/lyf/BreakHis/BreaKHis_200X/BreaKHis_200X_Cleaned_Strict",
    "400X": r"/mnt/disk1/lyf/BreakHis/BreaKHis_400X/BreaKHis_400X_Cleaned_Strict",
    "Bracs": r"/mnt/disk1/lyf/BRACS/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version",
}

Save_dir = {
    "40X": r"/mnt/disk1/lyf/BreakHis/Results/BreaKHis_40X",
    "100X": r"/mnt/disk1/lyf/BreakHis/Results/BreaKHis_100X",
    "200X": r"/mnt/disk1/lyf/BreakHis/Results/BreaKHis_200X",
    "400X": r"/mnt/disk1/lyf/BreakHis/Results/BreaKHis_400X",
    "Bracs": r"/mnt/disk1/lyf/BreakHis/Results/BRACS",
}


CONFIG = {
    "Magnification": "100X",
    "task": "eight",
    "summary_csv": r"/mnt/disk1/lyf/BreakHis/paper_Results/final_summary.csv",

    "model_name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "model_all": "FSD_Swin",
    "pretrained": True,
    "img_size": 384,
    "feature_hw": 12,

    "batch_size": 16,
    "accum_iter": 1,
    "epochs": 100,
    "seed": 180811,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "weight_decay": 0.05,

    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,

    "mixup": 0.4,
    "cutmix": 1.0,
    "label_smoothing": 0.05,

    "model_ema_decay": 0.999,
    "use_tta": False,
    "top_k": 5,
    "num_workers": 8,
}

EXPERIMENT_QUEUE = [
    {"Magnification": "40X", "task": "binary"},
    {"Magnification": "100X", "task": "binary"},
    {"Magnification": "200X", "task": "binary"},
    {"Magnification": "400X", "task": "binary"},
    {"Magnification": "40X", "task": "eight"},
    {"Magnification": "100X", "task": "eight"},
    {"Magnification": "200X", "task": "eight"},
    {"Magnification": "400X", "task": "eight"},
    {"Magnification": "Bracs", "task": "three"},
    {"Magnification": "Bracs", "task": "seven"},
]


class ResizeKeepRatioPad:
    """Resize while keeping aspect ratio and pad to a square canvas."""
    def __init__(self, size: int, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = TF.resize(img, (new_h, new_w), antialias=True)
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = TF.pad(img, [left, top, right, bottom], fill=self.fill)
        return img


class IndexWrapper(torch.utils.data.Dataset):
    """Subset wrapper with optional label mapping."""
    def __init__(self, ds, idx, transform, label_map=None):
        self.ds = ds
        self.idx = idx
        self.transform = transform
        self.label_map = label_map

    def __getitem__(self, i):
        orig_idx = self.idx[i]
        x, y = self.ds[orig_idx]
        if self.label_map is not None:
            y = self.label_map[y]
        return self.transform(x), y

    def __len__(self):
        return len(self.idx)


def build_breakhis_binary_label_map(class_to_idx):
    """Map BreaKHis subtypes to benign/malignant."""
    benign_keywords = [
        "adenosis", "fibroadenoma", "phyllodes", "tubular",
        "sob_a", "sob_f", "sob_pt", "sob_ta",
    ]
    benign_short = {"A", "F", "PT", "TA"}

    label_map = {}
    for class_name, orig_idx in class_to_idx.items():
        name_lower = class_name.lower()
        is_benign = class_name in benign_short or any(k in name_lower for k in benign_keywords)
        label_map[orig_idx] = 0 if is_benign else 1

    print("BreaKHis binary label mapping:")
    for class_name, orig_idx in class_to_idx.items():
        print(f"  {class_name:30s} -> {label_map[orig_idx]}")
    return label_map


def make_transforms(config, mean_key):
    if mean_key == "Bracs":
        train_tf = transforms.Compose([
            ResizeKeepRatioPad(config["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[mean_key], STD[mean_key]),
        ])
    else:
        train_tf = transforms.Compose([
            ResizeKeepRatioPad(config["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[mean_key], STD[mean_key]),
        ])

    eval_tf = transforms.Compose([
        ResizeKeepRatioPad(config["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[mean_key], STD[mean_key]),
    ])
    return train_tf, eval_tf


class AdaptiveFrequencyPriorBlock(nn.Module):
    """
    Adaptive frequency-prior enhancement block.

    F_freq     = rFFT2(F_spatial)
    F_weighted = F_freq * W_complex
    Delta_F    = irFFT2(F_weighted)
    F_residual = F_spatial + Delta_F
    z_s        = GAP(F_spatial)
    z_e        = GAP(F_residual)
    z          = z_s + alpha * (z_e - z_s)
    """
    def __init__(self, dim: int, h: int = 12, w: int = 12):
        super().__init__()
        self.h = h
        self.w = w
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_spatial):
        B, C, H, W = f_spatial.shape
        if H != self.h or W != self.w:
            raise ValueError(
                f"AdaptiveFrequencyPriorBlock expects feature size {self.h}x{self.w}, "
                f"but got {H}x{W}. Check CONFIG['feature_hw'] or backbone output shape."
            )

        f_freq = torch.fft.rfft2(f_spatial, dim=(-2, -1), norm="ortho")
        w_complex = torch.view_as_complex(self.complex_weight)
        f_weighted = f_freq * w_complex
        delta_f = torch.fft.irfft2(f_weighted, s=(H, W), dim=(-2, -1), norm="ortho")
        f_residual = f_spatial + delta_f
        z_s = f_spatial.mean(dim=[2, 3])
        z_e = f_residual.mean(dim=[2, 3])
        z = z_s + self.alpha * (z_e - z_s)
        return z


class FSD_Swin(nn.Module):
    """Official implementation of the manuscript's FPE-Swin model."""
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CONFIG["model_name"], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True

        self.freq_prior_module = AdaptiveFrequencyPriorBlock(
            dim=self.num_features,
            h=CONFIG["feature_hw"],
            w=CONFIG["feature_hw"],
        )

        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes),
        )

    def _to_bchw(self, feat_raw):
        if feat_raw.ndim == 4:
            if feat_raw.shape[-1] == self.num_features:
                feat_spatial = feat_raw.permute(0, 3, 1, 2).contiguous()
            elif feat_raw.shape[1] == self.num_features:
                feat_spatial = feat_raw.contiguous()
            else:
                raise ValueError(f"Unexpected 4D feature shape: {feat_raw.shape}")
        elif feat_raw.ndim == 3:
            B, L, C = feat_raw.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(f"Token length {L} cannot be reshaped to square feature map.")
            feat_spatial = feat_raw.transpose(1, 2).reshape(B, C, H, W).contiguous()
        else:
            raise ValueError(f"Unexpected feature dimension: {feat_raw.ndim}")
        return feat_spatial

    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        feat_spatial = self._to_bchw(feat_raw)
        global_vec = self.freq_prior_module(feat_spatial)
        out = self.head(global_vec)
        return out


FPE_Swin = FSD_Swin


class SpatialSwin(nn.Module):
    """Spatial-only Swin baseline."""
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CONFIG["model_name"], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes),
        )

    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        if feat_raw.ndim == 4:
            if feat_raw.shape[-1] == self.num_features:
                feat_vec = feat_raw.mean(dim=[1, 2])
            elif feat_raw.shape[1] == self.num_features:
                feat_vec = feat_raw.mean(dim=[2, 3])
            else:
                raise ValueError(f"Unexpected 4D feature shape: {feat_raw.shape}")
        elif feat_raw.ndim == 3:
            feat_vec = feat_raw.mean(dim=1)
        else:
            feat_vec = feat_raw
        out = self.head(feat_vec)
        return out


SUMMARY_FIELDS = [
    "Magnification", "Timestamp", "task", "model_name", "Model_All",
    "batch_size", "Backbone_LR", "head_lr", "Best_Val_F1", "Best_Val_Acc",
    "Test_Acc", "Test_F1_Macro", "Test_F1_Weighted", "Test_Balanced_Acc",
    "Test_MCC", "Test_Kappa", "Mixup", "CutMix", "Exp_Dir",
]


def save_summary(config, results, timestamp, exp_dir):
    csv_path = config["summary_csv"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_data = {
        "Magnification": config["Magnification"],
        "Timestamp": timestamp,
        "task": config.get("task", "N/A"),
        "model_name": config["model_name"],
        "Model_All": config["model_all"],
        "batch_size": config["batch_size"],
        "Backbone_LR": config["backbone_lr"],
        "head_lr": config["head_lr"],
        "Best_Val_F1": f"{results.get('best_val_f1', 0.0):.4f}",
        "Best_Val_Acc": f"{results.get('best_val_acc', 0.0):.2f}%",
        "Test_Acc": f"{results['test_acc']:.2f}%",
        "Test_F1_Macro": f"{results['test_f1_macro']:.4f}",
        "Test_F1_Weighted": f"{results['test_f1_weighted']:.4f}",
        "Test_Balanced_Acc": f"{results['test_bal_acc']:.4f}",
        "Test_MCC": f"{results['test_mcc']:.4f}",
        "Test_Kappa": f"{results['test_kappa']:.4f}",
        "Mixup": config.get("mixup", 0.0),
        "CutMix": config.get("cutmix", 0.0),
        "Exp_Dir": exp_dir,
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def get_parameter_groups(model, config):
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            this_wd = 0.0
        else:
            this_wd = config["weight_decay"]

        if "backbone" in name:
            backbone_params.append({"params": param, "lr": config["backbone_lr"], "weight_decay": this_wd})
        else:
            head_params.append({"params": param, "lr": config["head_lr"], "weight_decay": this_wd})

    return backbone_params + head_params


def evaluate_probs(model, loader, device, use_tta=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            views = [imgs]
            if use_tta:
                views.append(torch.flip(imgs, dims=[3]))
                views.append(torch.flip(imgs, dims=[2]))

            probs_sum = None
            for view in views:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(view)
                    probs = F.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            probs_mean = probs_sum / len(views)
            all_probs.append(probs_mean.cpu())
            all_labels.extend(labels.cpu().numpy())

    return torch.cat(all_probs), np.array(all_labels)


def evaluate_test_metrics(probs, labels, class_names):
    labels = np.array(labels)
    preds = probs.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    report_dict = classification_report(
        labels, preds, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "bal_acc": bal_acc,
        "mcc": mcc,
        "kappa": kappa,
        "report_dict": report_dict,
        "cm": cm,
        "preds": preds,
    }


def _dataloader_kwargs(config, shuffle=False, sampler=None, drop_last=False):
    kwargs = dict(
        batch_size=config["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config["num_workers"],
        drop_last=drop_last,
        pin_memory=True,
    )
    if config["num_workers"] > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return kwargs


def build_loaders(config):
    mean_key = config["Magnification"]
    train_tf, eval_tf = make_transforms(config, mean_key)

    if config["Magnification"] == "Bracs":
        data_root = Data_dir["Bracs"]
        if config["task"] == "three":
            target_tf = lambda y: BRACS_7_TO_3[y]
            class_names = BRACS_3_CLASS_NAMES
            config["num_classes"] = 3
        elif config["task"] == "seven":
            target_tf = None
            class_names = BRACS_7_CLASS_NAMES
            config["num_classes"] = 7
        else:
            raise ValueError("BRACS task must be 'three' or 'seven'.")

        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf, target_transform=target_tf)
        val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf, target_transform=target_tf)
        test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=eval_tf, target_transform=target_tf)
        assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx

        train_dl = DataLoader(train_ds, **_dataloader_kwargs(config, shuffle=True, drop_last=True))
        val_dl = DataLoader(val_ds, **_dataloader_kwargs(config, shuffle=False))
        test_dl = DataLoader(test_ds, **_dataloader_kwargs(config, shuffle=False))
        return train_dl, val_dl, test_dl, class_names

    data_root = Data_dir[config["Magnification"]]
    dataset = datasets.ImageFolder(data_root, transform=None)

    if config["task"] == "binary":
        label_map = build_breakhis_binary_label_map(dataset.class_to_idx)
        targets = [label_map[t] for t in dataset.targets]
        class_names = ["Benign", "Malignant"]
        config["num_classes"] = 2
    elif config["task"] == "eight":
        label_map = None
        targets = dataset.targets
        class_names = dataset.classes
        config["num_classes"] = 8
    else:
        raise ValueError("BreaKHis task must be 'binary' or 'eight'.")

    indices = np.arange(len(targets))
    train_idx, tmp_idx, _, tmp_y = train_test_split(
        indices, targets, test_size=0.4, stratify=targets, random_state=config["seed"]
    )
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.5, stratify=tmp_y, random_state=config["seed"]
    )

    train_targets_list = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets_list, minlength=config["num_classes"]).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets_list]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(
        IndexWrapper(dataset, train_idx, train_tf, label_map),
        **_dataloader_kwargs(config, sampler=sampler, drop_last=True),
    )
    val_dl = DataLoader(
        IndexWrapper(dataset, val_idx, eval_tf, label_map),
        **_dataloader_kwargs(config, shuffle=False),
    )
    test_dl = DataLoader(
        IndexWrapper(dataset, test_idx, eval_tf, label_map),
        **_dataloader_kwargs(config, shuffle=False),
    )
    return train_dl, val_dl, test_dl, class_names


def run_single_experiment(params):
    config = CONFIG.copy()
    config.update(params)
    seed_everything(config["seed"], deterministic=True)

    device = config["device"]
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16

    train_dl, val_dl, test_dl, class_names = build_loaders(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(Save_dir[config["Magnification"]], f"{timestamp}_{config['model_all']}_{config['task']}")
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Experiment: {config['Magnification']} | Task: {config['task']} | Model: {config['model_all']}")
    print(f"Classes: {class_names}")
    print(f"Save dir: {exp_dir}")
    print("=" * 80)

    use_mixup = config["mixup"] > 0.0 or config["cutmix"] > 0.0
    if use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=config["mixup"], cutmix_alpha=config["cutmix"],
            prob=1.0, switch_prob=0.7, mode="batch",
            label_smoothing=config["label_smoothing"], num_classes=config["num_classes"],
        )
        criterion_train = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion_train = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    criterion_val = nn.CrossEntropyLoss()

    model_class = globals()[config["model_all"]]
    model = model_class(num_classes=config["num_classes"], pretrained=config["pretrained"]).to(device)
    model = model.to(memory_format=torch.channels_last)
    model_ema = ModelEmaV2(model, decay=config["model_ema_decay"], device=device)

    optimizer = optim.AdamW(get_parameter_groups(model, config))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    top_k_checkpoints = []
    best_monitor = -float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    early_stop_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion_train(logits, y) / config["accum_iter"]

            scaler.scale(loss).backward()

            if ((step + 1) % config["accum_iter"] == 0) or ((step + 1) == len(train_dl)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                model_ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * config["accum_iter"]

        scheduler.step()

        eval_model = model_ema.module
        eval_model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []

        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = eval_model(x)
                    loss_v = criterion_val(logits, y)
                val_loss += loss_v.item()
                pred = torch.argmax(logits, dim=1)
                val_true.extend(y.cpu().numpy())
                val_pred.extend(pred.cpu().numpy())

        val_loss /= max(len(val_dl), 1)
        val_acc = accuracy_score(val_true, val_pred) * 100
        val_f1 = f1_score(val_true, val_pred, average="macro")

        print(
            f"Epoch {epoch + 1:03d}/{config['epochs']} | "
            f"TrainLoss {running_loss / max(len(train_dl), 1):.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.2f}% | ValMacroF1 {val_f1:.4f}"
        )

        state_dict = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
        save_name = f"epoch_{epoch + 1:03d}_valacc_{val_acc:.2f}_valf1_{val_f1:.4f}.pth"
        save_path = os.path.join(exp_dir, save_name)
        score_tuple = (val_f1, val_acc, -val_loss)

        if len(top_k_checkpoints) < config["top_k"]:
            heapq.heappush(top_k_checkpoints, (score_tuple, epoch, save_path))
            torch.save(state_dict, save_path)
        else:
            worst_score, _, worst_path = top_k_checkpoints[0]
            if score_tuple > worst_score:
                heapq.heappop(top_k_checkpoints)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                heapq.heappush(top_k_checkpoints, (score_tuple, epoch, save_path))
                torch.save(state_dict, save_path)

        if val_f1 > best_monitor + config["early_stop_min_delta"]:
            best_monitor = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0
            print(f"  New best Val Macro-F1: {best_monitor:.4f} at epoch {best_epoch}")
        else:
            early_stop_counter += 1
            print(f"  EarlyStop counter: {early_stop_counter}/{config['early_stop_patience']}")

        if early_stop_counter >= config["early_stop_patience"]:
            print(f"\nEarly stopping triggered. Best epoch: {best_epoch}, Best Val Macro-F1: {best_monitor:.4f}")
            break

    print("\nTraining finished. Testing Top-K checkpoints...")
    target_models = [path for _, _, path in sorted(top_k_checkpoints, reverse=True)]
    if not target_models:
        print("No checkpoints saved. Skip testing.")
        return

    for model_path in target_models:
        file_name = os.path.basename(model_path)
        print(f"\nTesting checkpoint: {file_name}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        probs, labels = evaluate_probs(model=model, loader=test_dl, device=device, use_tta=config["use_tta"])
        metrics = evaluate_test_metrics(probs, labels, class_names)
        print(
            f"Test Acc: {metrics['acc']:.2f}% | Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f} | Bal-Acc: {metrics['bal_acc']:.4f} | "
            f"MCC: {metrics['mcc']:.4f} | Kappa: {metrics['kappa']:.4f}"
        )

        report_df = pd.DataFrame(metrics["report_dict"]).T
        report_df.to_csv(os.path.join(exp_dir, file_name.replace(".pth", "_classification_report.csv")), encoding="utf-8-sig")
        cm_df = pd.DataFrame(metrics["cm"], index=class_names, columns=class_names)
        cm_df.to_csv(os.path.join(exp_dir, file_name.replace(".pth", "_confusion_matrix.csv")), encoding="utf-8-sig")

        individual_results = {
            "best_val_f1": best_monitor,
            "best_val_acc": best_val_acc,
            "test_acc": metrics["acc"],
            "test_f1_macro": metrics["macro_f1"],
            "test_f1_weighted": metrics["weighted_f1"],
            "test_bal_acc": metrics["bal_acc"],
            "test_mcc": metrics["mcc"],
            "test_kappa": metrics["kappa"],
        }
        save_summary(config, individual_results, timestamp, exp_dir)

    del model, model_ema, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nAll Top-K checkpoints tested.")


def main():
    for params in EXPERIMENT_QUEUE:
        run_single_experiment(params)


if __name__ == "__main__":
    main()
