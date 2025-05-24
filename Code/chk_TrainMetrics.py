#!/usr/bin/env python3
# chk_TrainMetrics.py
"""
Clean & compact training-curve visualiser.

* Reads one or more training-log CSVs (epoch, psnr, ssim)
* De-duplicates duplicate epochs (mean aggregation)
* Draws raw curve + rolling-mean (window=3 by default)
* Saves high-resolution figure train_metrics.png
"""

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
LOG_FILES: List[str] = ['PSNR.txt']          # 可以放多個檔案比對
ROLLING_WINDOW: int = 3                      # 移動平均視窗大小
OUT_FIGURE: str = 'train_metrics.png'
# ----------------------------

def load_metric_file(path: str) -> pd.DataFrame:
    """Read single txt/csv, add column 'run' for legend, aggregate duplicates."""
    if not os.path.isfile(path):
        print(f'[WARN] File not found: {path}', file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(path, names=['epoch', 'psnr', 'ssim', 'learning_rate'])
    # 假如同一 epoch 被寫入多次，先 groupby 取平均
    df = df.groupby('epoch', as_index=False).mean().sort_values('epoch')
    df['run'] = os.path.splitext(os.path.basename(path))[0]  # for legend
    return df

def plot_metrics(dfs: List[pd.DataFrame]) -> None:
    """Plot PSNR & SSIM curves with rolling mean."""
    if not dfs:
        print('[ERROR] No data to plot.')
        return

    plt.figure(figsize=(12, 5))

    # ---------- PSNR ----------
    ax1 = plt.subplot(1, 2, 1)
    for df in dfs:
        label = df['run'].iloc[0]
        ax1.plot(df['epoch'], df['psnr'],
                 marker='o', linewidth=1, alpha=0.4, label=f'{label} raw')
        ax1.plot(df['epoch'],
                 df['psnr'].rolling(ROLLING_WINDOW,
                                    min_periods=1).mean(),
                 linewidth=2, label=f'{label} MA({ROLLING_WINDOW})')
    ax1.set_title('PSNR over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize='small')

    # ---------- SSIM ----------
    ax2 = plt.subplot(1, 2, 2)
    for df in dfs:
        label = df['run'].iloc[0]
        ax2.plot(df['epoch'], df['ssim'],
                 marker='o', linewidth=1, alpha=0.4, label=f'{label} raw')
        ax2.plot(df['epoch'],
                 df['ssim'].rolling(ROLLING_WINDOW,
                                    min_periods=1).mean(),
                 linewidth=2, label=f'{label} MA({ROLLING_WINDOW})')
    ax2.set_title('SSIM over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('SSIM')
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(OUT_FIGURE, dpi=300)
    print(f'[INFO] Figure saved to {OUT_FIGURE}')
    plt.show()

def main():
    dfs = [load_metric_file(p) for p in LOG_FILES]
    dfs = [df for df in dfs if not df.empty]  # drop empty ones
    plot_metrics(dfs)

if __name__ == '__main__':
    main()
