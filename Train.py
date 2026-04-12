import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.data.mixup import Mixup 
from timm.loss import SoftTargetCrossEntropy 
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    f1_score,
    balanced_accuracy_score)
from sklearn.model_selection import train_test_split
from timm.utils import ModelEmaV2
import numpy as np
import pandas as pd
import datetime
import warnings
import csv
import gc
import torchvision.transforms.functional as TF
from timm.data import create_transform
import heapq
import sys
from PIL import Image
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None 

import random

def seed_everything(seed: int = 42, deterministic: bool = False):
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
# print([m for m in timm.list_models('*resnet*', pretrained=True)])
# sys.exit(0)


BRACS_7_TO_3 = {
    0: 0,  # 0_N   -> Benign
    1: 0,  # 1_PB  -> Benign
    2: 0,  # 2_UDH -> Benign
    3: 1,  # 3_FEA -> Atypia
    4: 1,  # 4_ADH -> Atypia
    5: 2,  # 5_DCIS -> Malignant
    6: 2   # 6_IC   -> Malignant
}
BRACS_3_CLASS_NAMES = ["Benign", "Atypia", "Malignant"]
BRACS_7_CLASS_NAMES = ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

MEAN = {'40X':[0.8036, 0.6521, 0.7734], '100X':[0.7959, 0.6359, 0.772], '200X':[0.7889, 0.623, 0.7693], '400X':[0.7559, 0.5897, 0.7432], 'Bracs':[0.7265, 0.5587, 0.6954]}
STD = {'40X':[0.1022, 0.1447, 0.1], '100X':[0.1199, 0.1724, 0.1084], '200X':[0.1228, 0.1752, 0.1034], '400X':[0.1408, 0.1995, 0.1139], 'Bracs':[0.1947, 0.2282, 0.1714]} 
Data_dir = {'40X':r'/mnt/disk1/lyf/BreakHis/BreaKHis_40X/BreaKHis_40X_Cleaned_Strict',
            '100X':r'/mnt/disk1/lyf/BreakHis/BreaKHis_100X/BreaKHis_100X_Cleaned_Strict',
            '200X':r'/mnt/disk1/lyf/BreakHis/BreaKHis_200X/BreaKHis_200X_Cleaned_Strict',
            '400X':r'/mnt/disk1/lyf/BreakHis/BreaKHis_400X/BreaKHis_400X_Cleaned_Strict',
            'Bracs':r'/mnt/disk1/lyf/BRACS/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version'}
Save_dir = {'40X':r'/mnt/disk1/lyf/BreakHis/Results/BreaKHis_40X',
            '100X':r'/mnt/disk1/lyf/BreakHis/Results/BreaKHis_100X',
            '200X':r'/mnt/disk1/lyf/BreakHis/Results/BreaKHis_200X',
            '400X':r'/mnt/disk1/lyf/BreakHis/Results/BreaKHis_400X',
            'Bracs':r'/mnt/disk1/lyf/BreakHis/Results/BRACS'}
# model_name:
# convnext_base.dinov3_lvd1689m
# vit_base_patch16_dinov3_qkvb.lvd1689m
# resnet50.tv_in1k
# resnet50.tv2_in1k
# FSD_Swin_Hybrid

CONFIG = {'Magnification': '100X',
    "summary_csv": r'/mnt/disk1/lyf/BreakHis/paper_Results/final_summary.csv',
    "model_name": 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    'model_all': 'SpatialSwin',
    "pretrained": False,
    # "batch_size": 4, 
    # "accum_iter": 8,   
    'batch_size': 16,
    'accum_iter': 1,
    "epochs": 100,   
    "seed": 180811,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "img_size": 384,
    "use_tta": True,
    'backbone_lr': 4e-5,
    'head_lr': 5e-4,   
    "weight_decay": 0.05, 
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    # "mixup": 0.4, 
    # "cutmix": 1.0, 
    # "label_smoothing": 0.1,
    "mixup": 0.4,  
    "cutmix": 1.0, 
    "label_smoothing": 0.05,  
    "model_ema_decay": 0.999 
}
EXPERIMENT_QUEUE = [
    {'Magnification': '100X', 'task': 'binary'}, 
    {'Magnification': '40X', 'task': 'binary'},
    {'Magnification': '200X', 'task': 'binary'}, 
    {'Magnification': '400X', 'task': 'binary'},
    {'Magnification': '100X', 'task': 'eight'},
    {'Magnification': '40X', 'task': 'eight'},
    {'Magnification': '200X', 'task': 'eight'}, 
    {'Magnification': '400X', 'task': 'eight'},
    {'Magnification': 'Bracs', 'task': 'three'},
    {'Magnification': 'Bracs', 'task': 'seven'}
]

class SpectralDebiasingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq_filter = SpectralGatingBlock(dim)
        self.mask_generator = nn.Sequential(          nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        freq_response = self.freq_filter(x)
        debias_mask = self.mask_generator(freq_response)
        debiased_spatial_feat = x * debias_mask
        return debiased_spatial_feat 
class FSD_Swin_Debiased(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CONFIG['model_name'], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True
        self.debiasing_module = SpectralDebiasingModule(self.num_features)
        self.head = nn.Sequential(nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes)) 
    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        if feat_raw.ndim == 3:
            B, L, C = feat_raw.shape
            H = W = int(L ** 0.5)
            feat_spatial = feat_raw.transpose(1, 2).reshape(B, C, H, W)
        elif feat_raw.ndim == 4:
            feat_spatial = feat_raw.permute(0, 3, 1, 2)
        clean_feat = self.debiasing_module(feat_spatial)   
        global_vec = clean_feat.mean(dim=[2, 3])
        out = self.head(global_vec)
        return out

    
class SpectralDebiasingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq_filter = SpectralGatingBlock(dim)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        freq_response = self.freq_filter(x)
        debias_mask = self.mask_generator(freq_response)
        debiased_spatial_feat = x * debias_mask + x 
        return debiased_spatial_feat, freq_response
class FSD_Swin_Hybrid(nn.Module):# 去偏置空间分支 + 纯频域分支的 Concat 融合
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CONFIG['model_name'], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True
        self.debiasing_module = SpectralDebiasingModule(self.num_features)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.num_features * 2, self.num_features),
            nn.LayerNorm(self.num_features),
            nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes))
    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        if feat_raw.ndim == 3:
            B, L, C = feat_raw.shape
            H = W = int(L ** 0.5)
            feat_spatial = feat_raw.transpose(1, 2).reshape(B, C, H, W)
        elif feat_raw.ndim == 4:
            feat_spatial = feat_raw.permute(0, 3, 1, 2)
        clean_spatial_feat, freq_feat = self.debiasing_module(feat_spatial)
        spatial_vec = clean_spatial_feat.mean(dim=[2, 3])
        freq_vec = freq_feat.mean(dim=[2, 3])
        concat_vec = torch.cat([spatial_vec, freq_vec], dim=1) 
        fused_vec = self.fusion_gate(concat_vec)
        out = self.head(fused_vec)
        return out
    

class DWT_2D(nn.Module):
    """ 原生 PyTorch Haar 离散小波变换 (2D) """
    def __init__(self):
        super(DWT_2D, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        LL = x1 + x2 + x3 + x4  # 低频分量 (染色背景)
        HL = -x1 - x2 + x3 + x4 # 垂直高频
        LH = -x1 + x2 - x3 + x4 # 水平高频
        HH = x1 - x2 - x3 + x4  # 对角线高频 (核心病理细节)
        return LL, HL, LH, HH
class IDWT_2D(nn.Module):
    """ 原生 PyTorch Haar 逆离散小波变换 (2D) """
    def __init__(self):
        super(IDWT_2D, self).__init__()
    def forward(self, LL, HL, LH, HH):
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = LL - HL - LH + HH
        out[:, :, 1::2, 0::2] = LL - HL + LH - HH
        out[:, :, 0::2, 1::2] = LL + HL - LH - HH
        out[:, :, 1::2, 1::2] = LL + HL + LH + HH
        return out
class WaveletDebiasingBlock(nn.Module):# 小波高频增强与低频去偏置模块
    def __init__(self, dim):
        super(WaveletDebiasingBlock, self).__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.low_freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid())
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, groups=dim), # 深度可分离卷积，极其轻量
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim * 3, kernel_size=1))
    def forward(self, x):
        LL, HL, LH, HH = self.dwt(x)
        low_mask = self.low_freq_gate(LL)
        LL_clean = LL * low_mask  # 压制无用的染色信息
        high_concat = torch.cat([HL, LH, HH], dim=1) # [B, C*3, H/2, W/2]
        high_enhanced = self.high_freq_enhance(high_concat)
        HL_e, LH_e, HH_e = torch.chunk(high_enhanced, 3, dim=1)
        x_reconstructed = self.idwt(LL_clean, HL_e, LH_e, HH_e)
        return x + x_reconstructed
class FSD_Swin(nn.Module): 
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CONFIG['model_name'], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True
        self.wavelet_module = WaveletDebiasingBlock(self.num_features)
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes))
    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        if feat_raw.ndim == 3:
            B, L, C = feat_raw.shape
            H = W = int(L ** 0.5)
            feat_spatial = feat_raw.transpose(1, 2).reshape(B, C, H, W)
        elif feat_raw.ndim == 4:
            feat_spatial = feat_raw.permute(0, 3, 1, 2)
        clean_feat = self.wavelet_module(feat_spatial)
        
        global_vec = clean_feat.mean(dim=[2, 3])
        out = self.head(global_vec)
        return out
    
        
    

class SpatialConvNeXt(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(SpatialConvNeXt, self).__init__()
        self.backbone = timm.create_model(
            CONFIG['model_name'],
            pretrained=pretrained,
            num_classes=0)
        self.num_features = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "stages.2" in name or "stages.3" in name or "norm" in name:
                param.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes))
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=[2, 3])
        out = self.head(feat)
        return out
    

class SpatialSwin(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(SpatialSwin, self).__init__()
        self.backbone = timm.create_model(CONFIG['model_name'], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layers.2" in name or "layers.3" in name or "norm" in name:
                param.requires_grad = True
        self.head = nn.Sequential(nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 2, num_classes))
    def forward(self, x):
        feat_raw = self.backbone.forward_features(x)
        if feat_raw.ndim == 4:
            feat_vec = feat_raw.mean(dim=[1, 2]) 
        elif feat_raw.ndim == 3:
            feat_vec = feat_raw.mean(dim=1)
        else:
            feat_vec = feat_raw # 万一已经是 2D 了
        out = self.head(feat_vec)
        return out
    

class SpectralGatingBlock(nn.Module):
    def __init__(self, dim, h = 12, w =12): # 需要传入特征图尺寸 h, w  
        super(SpectralGatingBlock, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight 
        x_out = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        return x + x_out


class ResizeKeepRatioPad:
    def __init__(self, size, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)   # 长边缩放到 size
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


class Wrapper(torch.utils.data.Dataset):
    def __init__(self, ds, idx, tf, label_map=None): 
        self.ds=ds; self.idx=idx; self.tf=tf
        self.label_map = label_map # 标签映射表
    def __getitem__(self, i): 
        orig=self.idx[i]
        x, y = self.ds[orig]
        if self.label_map is not None:
            y = self.label_map[y]
        return self.tf(x), y
    def __len__(self): 
        return len(self.idx)

SUMMARY_FIELDS = [
    'Magnification', 
    'Timestamp',
    'task',
    'batch_size',
    'Backbone_LR',
    'Best_Val_Acc',          
    'Test_F1_Macro',
    'Test_F1_Weighted',
    'Test_Balanced_Acc',
    'Test_MCC',
    'Test_Kappa',
    'TTA',
    'Mixup',            
    'CutMix',
    'Exp_Dir',
    'Model_Name',
    'Model_All',
    'head_lr',
    'backbone_lr',]


def save_summary(config, results, timestamp, exp_dir):
    csv_path = config['summary_csv']
    row_data = {
        'Magnification': config['Magnification'],
        'Timestamp': timestamp,
        'task': config.get('task', 'N/A'),
        'model_name': config['model_name'],
        'batch_size': config['batch_size'],
        'Backbone_LR': config['backbone_lr'],
        'Best_Val_Acc': f"{results['best_val_acc']:.2f}%",
        'Test_Acc': f"{results['test_acc']:.2f}%",
        'Test_F1_Macro': f"{results['test_f1_macro']:.4f}",
        'Test_F1_Weighted': f"{results['test_f1_weighted']:.4f}",
        'Test_Balanced_Acc': f"{results['test_bal_acc']:.4f}",
        'Test_MCC': f"{results['test_mcc']:.4f}",
        'Test_Kappa': f"{results['test_kappa']:.4f}",
        'TTA': config.get('use_tta', False),
        'Mixup': config.get('mixup', False),
        'CutMix': config.get('cutmix', False),
        'Exp_Dir': exp_dir,
        'Model_Name': config['model_name'],
        'Model_All': config['model_all'],
        'head_lr': config['head_lr'],
        'backbone_lr': config['backbone_lr'],
    }
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def get_probs_with_tta(model, loader, device):# 有频域去偏置的 TTA 版本，测试时使用原图 + 水平翻转 + 垂直翻转 三种视角，取平均概率
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(imgs)
                probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu())
            all_labels.extend(lbls.cpu().numpy())
    return torch.cat(all_probs), all_labels


def evaluate_test_metrics(probs, labels, class_names):
    labels = np.array(labels)
    preds = probs.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    report_dict = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0)
    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(len(class_names))))
    return {'acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'bal_acc': bal_acc,
        'mcc': mcc,
        'kappa': kappa,
        'report_dict': report_dict,
        'cm': cm,
        'preds': preds}


def get_parameter_groups(model, config):
    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip: 
            this_wd = 0.0
        else: 
            this_wd = config['weight_decay']
        # 💡 核心修正：让 debiasing_module 享受 backbone 的小学习率
        if "backbone" in name or "debiasing_module" in name: 
            backbone_params.append({'params': param, 'lr': config['backbone_lr'], 'weight_decay': this_wd})
        else: 
            head_params.append({'params': param, 'lr': config['head_lr'], 'weight_decay': this_wd})
    return backbone_params + head_params


def run_single_experiment(params):
    config = CONFIG.copy()
    seed_everything(config['seed'], deterministic=True)  
    config.update(params)
    use_amp = config['device'].type == 'cuda'
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if config['Magnification'] == 'Bracs':
        if config.get('task') == 'three':
            config['num_classes'] = 3
        else: config['num_classes'] = 7
    else: config['num_classes'] = 2 if config.get('task') == 'binary' else 8
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(Save_dir[config['Magnification']], f"{timestamp}")
    os.makedirs(exp_dir, exist_ok = True)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['Magnification'] == 'Bracs':
        print(f"\n🚀 启动 BRACS 7分类实验 (加载预划分数据集)...")
        # BRACS：等比例缩放后随机裁剪
        train_tf = transforms.Compose([
            ResizeKeepRatioPad(config['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN['Bracs'], STD['Bracs'])])
        val_tf = transforms.Compose([
            ResizeKeepRatioPad(config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(MEAN['Bracs'], STD['Bracs'])
        ])
        if config.get('task') == 'three':
            target_tf = lambda y: BRACS_7_TO_3[y]
            class_names = BRACS_3_CLASS_NAMES
            print("📌 当前任务: BRACS 3分类")
            print("   Benign    = 0_N + 1_PB + 2_UDH")
            print("   Atypia    = 3_FEA + 4_ADH")
            print("   Malignant = 5_DCIS + 6_IC")
        else:
            target_tf = None
            class_names = BRACS_7_CLASS_NAMES
            print("📌 当前任务: BRACS 7分类")
        train_ds = datasets.ImageFolder(
            os.path.join(Data_dir[config['Magnification']], 'train'),
            transform=train_tf,
            target_transform=target_tf)
        val_ds = datasets.ImageFolder(
            os.path.join(Data_dir[config['Magnification']], 'val'),
            transform=val_tf,
            target_transform=target_tf)
        test_ds = datasets.ImageFolder(
            os.path.join(Data_dir[config['Magnification']], 'test'),
            transform=val_tf,
            target_transform=target_tf)
        assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx

        raw_targets = train_ds.targets
        if config.get('task') == 'three':
            targets = [BRACS_7_TO_3[t] for t in raw_targets]
        else:
            targets = raw_targets
        class_counts = np.bincount(targets, minlength=config['num_classes'])
        class_weights = np.zeros_like(class_counts, dtype=np.float32)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]
        sample_weights = [class_weights[t] for t in targets]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_dl = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4)
        val_dl = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4)
        test_dl = DataLoader(
            test_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4)
    else:
        print(f"\n🚀 启动 BreaKHis 实验 | 倍率: {config['Magnification']} | 任务: {config.get('task')}")
        
        train_tf = transforms.Compose([
            ResizeKeepRatioPad(config['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[config['Magnification']], STD[config['Magnification']])])
        val_tf = transforms.Compose([
            ResizeKeepRatioPad(config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[config['Magnification']], STD[config['Magnification']])])
        dataset = datasets.ImageFolder(Data_dir[config['Magnification']], transform = None) 
        label_map = None
        if config.get('task') == 'binary':
            benign_classes = ['A', 'F', 'PT', 'TA'] 
            label_map = {}
            for class_name, orig_idx in dataset.class_to_idx.items():
                is_benign = any(kw in class_name for kw in benign_classes)
                label_map[orig_idx] = 0 if is_benign else 1
            targets = [label_map[t] for t in dataset.targets]
            class_names = ['Benign', 'Malignant']  
        else:
            targets = dataset.targets
            class_names = dataset.classes         
        indices = np.arange(len(targets))
        train_idx, tmp_idx, _, tmp_y = train_test_split(indices, targets, test_size = 0.4, stratify = targets, random_state = config['seed'])
        val_idx, test_idx = train_test_split(tmp_idx, test_size = 0.5, stratify = tmp_y, random_state = config['seed'])
        train_targets_list = [targets[i] for i in train_idx]
        class_counts = np.bincount(train_targets_list, minlength=config['num_classes'])
        class_weights = 1. / class_counts
        sample_weights = [class_weights[t] for t in train_targets_list]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True) 
        train_dl = DataLoader(Wrapper(dataset, train_idx, train_tf, label_map), batch_size = config['batch_size'], sampler = sampler, num_workers = 8, drop_last = True)
        val_dl = DataLoader(Wrapper(dataset, val_idx, val_tf, label_map), batch_size = config['batch_size'], shuffle = False, num_workers = 8)
        test_dl = DataLoader(Wrapper(dataset, test_idx, val_tf, label_map), batch_size = config['batch_size'], shuffle = False, num_workers = 8)
    use_mixup = config['mixup'] > 0.0 or config['cutmix'] > 0.0
    if use_mixup:
        mixup_fn = Mixup(mixup_alpha=config['mixup'], cutmix_alpha=config['cutmix'], 
                         prob=1.0, switch_prob=0.7, mode='batch', 
                         label_smoothing=config['label_smoothing'], num_classes=config['num_classes'])
        criterion_train = SoftTargetCrossEntropy() # Mixup 必须配合 SoftTarget
        print(f"🌟 已开启 Mixup (Alpha: {config['mixup']}) / CutMix (Alpha: {config['cutmix']})")
    else:
        mixup_fn = None
        criterion_train = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        print(f"🌟 已关闭硬增强，仅使用纯净病理图像 + 标签平滑 (Smoothing: {config['label_smoothing']})")
    model_name_str = config['model_all']
    model_class = globals()[model_name_str]  # 把字符串变成真正的类对象
    model = model_class(num_classes=config['num_classes']).to(config['device'])
    # model = FrequencySpatialSwin(num_classes=config['num_classes']).to(config['device'])
    # model = SpatialSwin(num_classes=config['num_classes']).to(config['device'])
    # model = SpatialConvNeXt(num_classes=config['num_classes']).to(config['device'])
    # model = FrequencySpatialViT(num_classes=config['num_classes']).to(config['device'])
    # model = model.to(memory_format=torch.channels_last)
    model_ema = ModelEmaV2(model, decay=config['model_ema_decay'], device=config['device'])
    params_groups = get_parameter_groups(model, config)
    optimizer = optim.AdamW(params_groups)
    criterion_val = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    top_k_checkpoints = [] 
    K = 5 # 保存 Top-5
    best_monitor = -float("inf")   # 这里监控 val_f1，越大越好
    early_stop_counter = 0
    best_epoch = 0
    # ================= 4. 开始训练循环 =================
    for epoch in range(config['epochs']):
        model.train()
        r_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x = x.to(config['device'], non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(config['device'], non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                loss = criterion_train(out, y)
                loss = loss / config['accum_iter']
            scaler.scale(loss).backward()        
            if ((i + 1) % config['accum_iter'] == 0) or ((i + 1) == len(train_dl)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                model_ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            r_loss += loss.item() * config['accum_iter']
        scheduler.step()     
        eval_target = model_ema.module
        eval_target.eval()
        v_corr, v_tot = 0, 0
        v_loss = 0.0
        val_true, val_pred = [], []
        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(config['device'], non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(config['device'], non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = eval_target(x)
                loss_v = criterion_val(out, y)
                v_loss += loss_v.item()
                _, p = torch.max(out, 1)
                v_tot += y.size(0)
                v_corr += (p == y).sum().item()
                val_true.extend(y.cpu().numpy())
                val_pred.extend(p.cpu().numpy())
        val_acc = 100 * v_corr / v_tot
        avg_val_loss = v_loss / len(val_dl)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        print(
            f"Ep {epoch+1:03d} | Train Loss: {r_loss/len(train_dl):.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc (EMA): {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        current_state = eval_target.state_dict()
        save_name = f"epoch_{epoch+1}_valacc_{val_acc:.2f}_valf1_{val_f1:.4f}.pth"
        save_path = os.path.join(exp_dir, save_name)
        score_tuple = (val_f1, val_acc, -avg_val_loss)
        if len(top_k_checkpoints) < K:
            heapq.heappush(top_k_checkpoints, (score_tuple, epoch, save_path))
            torch.save(current_state, save_path)
            print(f"  ✅ [Top-K] 新增: {save_name}")
        else:
            worst_item = top_k_checkpoints[0]
            worst_score, worst_epoch, worst_path = worst_item
            if score_tuple > worst_score:
                heapq.heappop(top_k_checkpoints)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                heapq.heappush(top_k_checkpoints, (score_tuple, epoch, save_path))
                torch.save(current_state, save_path)
                print(f"  ♻️ [Top-K] 替换: {os.path.basename(worst_path)} -> 保存 {save_name}")
        monitor_metric = val_f1
        if monitor_metric > best_monitor + config['early_stop_min_delta']:
            best_monitor = monitor_metric
            best_epoch = epoch + 1
            early_stop_counter = 0
            print(f"  🌟 [EarlyStop] 刷新最佳监控指标: Val F1 = {val_f1:.4f} (Epoch {best_epoch})")
        else:
            early_stop_counter += 1
            print(f"  ⏳ [EarlyStop] 未提升 {early_stop_counter}/{config['early_stop_patience']}")

        if early_stop_counter >= config['early_stop_patience']:
            print(f"\n🛑 Early stopping 触发：Val F1 已连续 {config['early_stop_patience']} 个 epoch 无提升。")
            print(f"🏁 最佳监控指标出现在 Epoch {best_epoch} | Best Val F1 = {best_monitor:.4f}")
            break
    print("\n" + "="*40)
    print("--> 训练结束，开始分别测试 Top-K 模型并保存结果...")
    if top_k_checkpoints:
        target_models = [path for _, _, path in sorted(top_k_checkpoints, reverse=True)]
    else:
        target_models = []

    for model_path in target_models:
        file_name = os.path.basename(model_path)
        print(f"\n🧪 正在测试模型: {file_name}")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config['device']))
        else:
            print(f"❌ 文件缺失: {model_path}")
            continue
        probs, labels = get_probs_with_tta(model, test_dl, config['device'])
        metrics = evaluate_test_metrics(probs, labels, class_names)
        print(
            f"   👉 Acc: {metrics['acc']:.2f}% | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f} | "
            f"Bal-Acc: {metrics['bal_acc']:.4f} | "
            f"MCC: {metrics['mcc']:.4f} | "
            f"Kappa: {metrics['kappa']:.4f}"
        )
        report_df = pd.DataFrame(metrics['report_dict']).T
        report_save_path = os.path.join(
            exp_dir, file_name.replace('.pth', '_classification_report.csv')
        )
        report_df.to_csv(report_save_path, encoding='utf-8-sig')
        cm_df = pd.DataFrame(metrics['cm'], index=class_names, columns=class_names)
        cm_save_path = os.path.join(
            exp_dir, file_name.replace('.pth', '_confusion_matrix.csv')
        )
        cm_df.to_csv(cm_save_path, encoding='utf-8-sig')
        try:
            val_acc_from_name = float(file_name.split('_valacc_')[1].split('_valf1_')[0])
        except:
            val_acc_from_name = 0.0
        individual_results = {
            "best_val_acc": val_acc_from_name,
            "test_acc": metrics['acc'],
            "test_f1_macro": metrics['macro_f1'],
            "test_f1_weighted": metrics['weighted_f1'],
            "test_bal_acc": metrics['bal_acc'],
            "test_mcc": metrics['mcc'],
            "test_kappa": metrics['kappa']
        }
        save_summary(config, individual_results, timestamp, exp_dir)
    del model, model_ema, optimizer    # 删除对象释放显存
    gc.collect()
    torch.cuda.empty_cache()
    print("\n🎉 所有模型单独测试完成，结果已逐行保存至 CSV。")


if __name__ == '__main__':
    
    for params in EXPERIMENT_QUEUE:
        run_single_experiment(params)