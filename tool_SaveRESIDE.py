import deeplake
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

# Load Deep Lake dataset
ds = deeplake.load("deeplake://activeloop/reside")

# Local paths for your image dataset
clear_images_path = "/home/omnisky/4t/RESIDE/OTS_BETA/clear/clear_newsize"
hazy_images_path = "/home/omnisky/4t/RESIDE/OTS_BETA/haze/hazy7"



# Save clear images
clear_images = [os.path.join(clear_images_path, f) for f in os.listdir(clear_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Save hazy images
hazy_images = [os.path.join(hazy_images_path, f) for f in os.listdir(hazy_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Validate the number of images
assert len(clear_images) == 13990, f"Expected 13,990 clear images but found {len(clear_images)}"
assert len(hazy_images) == 13990, f"Expected 13,990 hazy images but found {len(hazy_images)}"

# Add images to Deep Lake dataset
ds.create_tensor("clear_images", htype="image")
ds.create_tensor("hazy_images", htype="image")

for clear_img, hazy_img in zip(clear_images, hazy_images):
    clear_data = np.array(Image.open(clear_img))
    hazy_data = np.array(Image.open(hazy_img))
    
    # Save clear and hazy images to the dataset
    ds["clear_images"].append(clear_data)
    ds["hazy_images"].append(hazy_data)

# Commit changes to the dataset
ds.commit("Added 13,990 clear and hazy images.")

print("Dataset saved successfully!")
