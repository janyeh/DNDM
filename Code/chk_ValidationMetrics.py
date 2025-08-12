import ast
import os
import sys
from datetime import datetime
import logging
import matplotlib.pyplot as plt  # pylint: disable=import-error


def ensure_log_dir() -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(code_dir, 'Log')
    os.makedirs(base_log_dir, exist_ok=True)
    date_tag = datetime.now().strftime('%y-%m-%d')
    dated_dir = os.path.join(base_log_dir, date_tag)
    os.makedirs(dated_dir, exist_ok=True)
    return dated_dir


def setup_logger(log_dir: str, log_filename: str) -> logging.Logger:
    logger = logging.getLogger('chk_ValidationMetrics')
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
LOGGER = setup_logger(LOG_DIR, 'chk_ValidationMetrics.log')


def log_and_print(message: str, level: str = 'info') -> None:
    if level == 'error':
        LOGGER.error(message)
    elif level == 'warning':
        LOGGER.warning(message)
    else:
        LOGGER.info(message)
# checkpoint 4: Validation Metrics
# Should be printed in training log
log_and_print('Final Validation Results:')
with open('PSNR.txt', 'r', encoding='utf-8') as f:
    last_line = f.readlines()[-1].strip()
    epoch, psnr, ssim, learning_rate = last_line.split(',')
    log_and_print(f'PSNR: {float(psnr):.2f}')
    log_and_print(f'SSIM: {float(ssim):.4f}')
    learning_rate = ast.literal_eval(learning_rate)[0]
    log_and_print(f'Learning Rate: {float(learning_rate):.6f}')

notice ="""    
Good results should show:

1.Increasing PSNR/SSIM over epochs
2.Final PSNR > 20dB, SSIM > 0.8
3.Clear visual improvement in dehazed images
4.No missing checkpoints
5.Stable loss values in final epochs

If these metrics look poor:

1.Train for more epochs
2.Adjust learning rate
3.Modify loss weights
4.Check training data quality
5.Consider architecture changes    
"""

log_and_print(notice)

# Also save a simple image summarizing the metrics
fig = plt.figure(figsize=(6, 3))
plt.axis('off')
text = (
    'Final Validation Results\n'
    f'PSNR: {float(psnr):.2f}\n'
    f'SSIM: {float(ssim):.4f}\n'
    f'Learning Rate: {float(learning_rate):.6f}'
)
plt.text(0.01, 0.9, text, va='top', ha='left', fontsize=12)
plt.tight_layout()
out_img = os.path.join(LOG_DIR, 'validation_metrics.png')
plt.savefig(out_img, dpi=200, bbox_inches='tight')
LOGGER.info('[INFO] Figure saved to %s', out_img)
plt.show()