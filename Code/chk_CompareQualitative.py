import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Display a few random results
inputs = sorted(glob.glob('./results/Inputs/*.png'))
outputs = sorted(glob.glob('./results/Outputs/*.png'))
targets = sorted(glob.glob('./results/Targets/*.png'))

for i in range(min(10, len(inputs))):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Image.open(inputs[i]))
    axes[0].set_title('Input (Hazy)')
    axes[1].imshow(Image.open(outputs[i]))
    axes[1].set_title('Output (Dehazed)')
    axes[2].imshow(Image.open(targets[i]))
    axes[2].set_title('Target (Clear)')
    plt.show()