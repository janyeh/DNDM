# Check PSNR.txt for training history
import pandas as pd
import matplotlib.pyplot as plt

# Load and plot training metrics
df = pd.read_csv('PSNR.txt', names=['epoch', 'psnr', 'ssim'])
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(df['epoch'], df['psnr'], 'b-')
plt.title('PSNR over epochs')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.subplot(122)
plt.plot(df['epoch'], df['ssim'], 'r-')
plt.title('SSIM over epochs')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.tight_layout()
plt.show()