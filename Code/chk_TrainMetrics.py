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
from datetime import datetime
import logging

import pandas as pd  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error

# ---------- LOGGING & CONFIG ----------
LOG_FILES: List[str] = ['PSNR.txt']          # 可以放多個檔案比對
ROLLING_WINDOW: int = 3                      # 移動平均視窗大小
OUT_FIGURE_NAME: str = 'train_metrics.png'


def ensure_log_dir() -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(code_dir, 'Log')
    os.makedirs(base_log_dir, exist_ok=True)
    date_tag = datetime.now().strftime('%y-%m-%d')
    dated_dir = os.path.join(base_log_dir, date_tag)
    os.makedirs(dated_dir, exist_ok=True)
    return dated_dir


def setup_logger(log_dir: str, log_filename: str) -> logging.Logger:
    logger = logging.getLogger('chk_TrainMetrics')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


LOG_DIR = ensure_log_dir()
LOGGER = setup_logger(LOG_DIR, 'chk_TrainMetrics.log')


def log_and_print(message: str, level: str = 'info') -> None:
    if level == 'error':
        LOGGER.error(message)
    elif level == 'warning':
        LOGGER.warning(message)
    else:
        LOGGER.info(message)
    # print duplicated by console handler

def load_metric_file(path: str) -> pd.DataFrame:
    """Read single txt/csv, add column 'run' for legend, aggregate duplicates."""
    if not os.path.isfile(path):
        log_and_print(f'[WARN] File not found: {path}', level='warning')
        return pd.DataFrame()

    df = pd.read_csv(path, names=['epoch', 'psnr', 'ssim', 'learning_rate'])
    # 假如同一 epoch 被寫入多次，先 groupby 取平均
    df = df.groupby('epoch', as_index=False).mean().sort_values('epoch')
    df['run'] = os.path.splitext(os.path.basename(path))[0]  # for legend
    return df

def plot_metrics(dfs: List[pd.DataFrame]) -> None:
    """Plot PSNR & SSIM curves with rolling mean."""
    if not dfs:
        log_and_print('[ERROR] No data to plot.', level='error')
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
    out_path = os.path.join(LOG_DIR, OUT_FIGURE_NAME)
    plt.savefig(out_path, dpi=300)
    log_and_print(f'[INFO] Figure saved to {out_path}')
    plt.show()

def main():
    dfs = [load_metric_file(p) for p in LOG_FILES]
    dfs = [df for df in dfs if not df.empty]  # drop empty ones
    plot_metrics(dfs)

if __name__ == '__main__':
    main()
