import glob
import os
import sys
from datetime import datetime
import logging
from PIL import Image
import matplotlib.pyplot as plt  # pylint: disable=import-error


def ensure_log_dir() -> str:
    date_tag = datetime.now().strftime('%y-%m-%d')
    log_dir = f'log{date_tag}'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(log_dir: str, log_filename: str) -> logging.Logger:
    logger = logging.getLogger('chk_CompareQualitative')
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
LOGGER = setup_logger(LOG_DIR, 'chk_CompareQualitative.log')


def log_and_print(message: str, level: str = 'info') -> None:
    if level == 'error':
        LOGGER.error(message)
    elif level == 'warning':
        LOGGER.warning(message)
    else:
        LOGGER.info(message)

# Display a few random results
inputs = sorted(glob.glob('./results/Inputs/*.png'))
outputs = sorted(glob.glob('./results/Outputs/*.png'))
targets = sorted(glob.glob('./results/Targets/*.png'))

log_and_print(f'Found {len(inputs)} inputs, {len(outputs)} outputs, {len(targets)} targets')

for i in range(min(10, len(inputs))):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Image.open(inputs[i]))
    axes[0].set_title('Input (Hazy)')
    axes[1].imshow(Image.open(outputs[i]))
    axes[1].set_title('Output (Dehazed)')
    axes[2].imshow(Image.open(targets[i]))
    axes[2].set_title('Target (Clear)')
    fig.tight_layout()
    out_img = os.path.join(LOG_DIR, f'qualitative_{i:02d}.png')
    plt.savefig(out_img, dpi=200)
    LOGGER.info('[INFO] Figure saved to %s', out_img)
    plt.show()