#!/usr/bin/env python3
"""
Robust PSNR / SSIM evaluator
---------------------------
用法：
    python psnr_ssim.py --pred ./output/C --gt ./output/A
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import natsort

# ---------- CLI ----------
parser = argparse.ArgumentParser(description='Compute PSNR / SSIM for de-hazing results')
parser.add_argument('--pred', required=True, help='folder with model outputs')
parser.add_argument('--gt',   required=True, help='folder with ground-truth (clear) images')
args = parser.parse_args()

pred_dir = Path(args.pred)
gt_dir   = Path(args.gt)

if not pred_dir.is_dir() or not gt_dir.is_dir():
    sys.exit('[ERROR] pred_dir 或 gt_dir 不是有效資料夾')

# ---------- Collect file list ----------
pred_files = natsort.natsorted([f for f in pred_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
gt_files   = natsort.natsorted([gt_dir / f.name for f in pred_files])   # 依 pred 檔名對應

missing = [f.name for f, g in zip(pred_files, gt_files) if not g.exists()]
if missing:
    sys.exit(f'[ERROR] 下列 GT 檔案不存在：{missing[:5]} ...')

# ---------- Compute metrics ----------
psnr_total, ssim_total = 0.0, 0.0
for p_path, g_path in zip(pred_files, gt_files):
    pred = np.asarray(Image.open(p_path).convert('RGB'), dtype=np.float32) / 255.0
    gt   = np.asarray(Image.open(g_path).convert('RGB'), dtype=np.float32) / 255.0

    psnr_total += peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_total += structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)

num = len(pred_files)
print(f'Tested {num} image pairs')
print(f'Average PSNR : {psnr_total/num:.4f} dB')
print(f'Average SSIM : {ssim_total/num:.4f}')