# checkpoint 4: Validation Metrics
# Should be printed in training log
print('Final Validation Results:')
with open('PSNR.txt', 'r') as f:
    last_line = f.readlines()[-1].strip()
    epoch, psnr, ssim = last_line.split(',')
    print(f'PSNR: {float(psnr):.2f}')
    print(f'SSIM: {float(ssim):.4f}')

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

print(notice)