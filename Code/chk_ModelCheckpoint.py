import os
import glob
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
    logger = logging.getLogger('chk_ModelCheckpoint')
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
LOGGER = setup_logger(LOG_DIR, 'chk_ModelCheckpoint.log')


def log_and_print(message: str, level: str = 'info') -> None:
    if level == 'error':
        LOGGER.error(message)
    elif level == 'warning':
        LOGGER.warning(message)
    else:
        LOGGER.info(message)

def check_model_checkpoints():
    checkpoint_patterns = [
        'output/netG_content_*.pth',
        'output/netG_haze_*.pth', 
        'output/net_dehaze_*.pth',
        'output/net_G_*.pth'
    ]

    log_and_print("\nChecking Model Checkpoints:")
    for pattern in checkpoint_patterns:
        files = glob.glob(pattern)
        log_and_print(f"\nLooking for {pattern}")
        if len(files) == 0:
            log_and_print("No checkpoints found")
            continue
            
        # Sort files by epoch number
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        log_and_print(f"Found {len(files)} checkpoints")
        log_and_print(f"Latest: {files[-1]}")
        
        # Check file size
        latest_size = os.path.getsize(files[-1]) / (1024*1024) # Convert to MB
        log_and_print(f"Latest checkpoint size: {latest_size:.2f} MB")
        
        # Verify file is readable
        try:
            import torch  # pylint: disable=import-error
            _ = torch.load(files[-1])
            log_and_print("Latest checkpoint is loadable")
        except Exception as e:
            log_and_print(f"Warning: Could not load latest checkpoint: {e}", level='warning')

    # Save a simple figure summarizing the inspection
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    summary_lines = [
        'Model Checkpoints Summary:',
    ]
    for pattern in checkpoint_patterns:
        files = glob.glob(pattern)
        line = f"{pattern}: {len(files)} file(s)"
        if files:
            line += f", latest: {os.path.basename(files[-1])}"
        summary_lines.append(line)
    plt.text(0.01, 0.95, "\n".join(summary_lines), va='top', ha='left', fontsize=12)
    plt.tight_layout()
    out_img = os.path.join(LOG_DIR, 'model_checkpoints.png')
    plt.savefig(out_img, dpi=200, bbox_inches='tight')
    LOGGER.info('[INFO] Figure saved to %s', out_img)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('output'):
        log_and_print("Error: 'output' directory not found!", level='error')
    else:
        check_model_checkpoints()